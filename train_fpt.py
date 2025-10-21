import fire
from pathlib import Path
import imageio
import os
import datetime
import logging
from copy import deepcopy
from collections import OrderedDict
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from dataset import OpenXMP4VideoDataset
from fpt import FPT


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Update EMA model parameters"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            # Skip if parameter not found in EMA model (may be dynamically created projection layers)
            print(f"Warning: Parameter {name} not found in EMA model, skipping...")


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set the requires_grad flag for all parameters of model."""
    for p in model.parameters():
        p.requires_grad = flag


def init_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if available."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True
    return 0, 0, 1, False


def main(
    dataset_dir: Path = Path("sample_data"),
    checkpoint_dir: Path | None = None,
    # Dataset
    input_h: int = 256,
    input_w: int = 256,
    n_frames: int = 10,
    frame_skip: int = 1,
    subset_names: str = "bridge",
    action_dim: int = 10,
    num_workers: int = 0,
    # FPT Architecture
    latent_dim: int = 16,
    horizons: str = "1,2,4,8",
    temperature: float = 0.2,
    recon_weight: float = 0.1,
    perceptual_weight: float = 0.0,
    use_vae_init: bool = False,  # Use VAE initialization
    vae_cache_dir: str = "./models",  # VAE cache directory
    # Training
    batch_size: int = 16,
    lr: float = 5e-5,
    ema_decay: float = 0.999,
    max_train_steps: int = 10_000,
    # Early stopping
    early_stopping_patience: int = 100,
    min_delta: float = 0.001,
    # Gradient accumulation for larger effective batch size
    gradient_accumulation_steps: int = 2,
    # Logging
    validate_every: int = 500,
    log_every: int = 50,
    save_every: int = 1000,
) -> None:
    assert torch.cuda.is_available(), "CUDA device required for training"

    # Parse horizons
    horizons_list = [int(h.strip()) for h in horizons.split(",")]
    
    local_rank, rank, world_size, distributed = init_distributed()
    device = f"cuda:{local_rank}" if distributed else "cuda"
    device = torch.device(device)
    
    # Create datasets
    train_dataset = OpenXMP4VideoDataset(
        save_dir=dataset_dir,
        input_h=input_h,
        input_w=input_w,
        n_frames=n_frames,
        frame_skip=frame_skip,
        action_dim=action_dim,
        subset_names=subset_names,
        split="train",
    )
    val_dataset = OpenXMP4VideoDataset(
        save_dir=dataset_dir,
        input_h=input_h,
        input_w=input_w,
        n_frames=n_frames,
        frame_skip=frame_skip,
        action_dim=action_dim,
        subset_names=subset_names,
        split="test",
    )

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    # Create FPT model
    fpt = FPT(
        latent_dim=latent_dim,
        action_dim=action_dim,
        horizons=horizons_list,
        temperature=temperature,
        recon_weight=recon_weight,
        perceptual_weight=perceptual_weight,
        use_vae_init=use_vae_init,
        vae_cache_dir=vae_cache_dir,
    ).to(device)

    if distributed:
        fpt = torch.nn.parallel.DistributedDataParallel(fpt, device_ids=[local_rank], output_device=local_rank)
        fpt_no_ddp = fpt.module
    else:
        fpt_no_ddp = fpt

    # Exponential Moving Average
    # Special handling for VAE-initialized models
    if use_vae_init:
        # Create EMA model with same structure
        ema = FPT(
            latent_dim=latent_dim,
            action_dim=action_dim,
            horizons=horizons_list,
            temperature=temperature,
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            use_vae_init=use_vae_init,
            vae_cache_dir=vae_cache_dir,
        ).to(device)
        
        # Trigger projection layer creation (via dummy forward pass)
        with torch.no_grad():
            dummy_video = torch.randn(1, 10, 256, 256, 3).to(device)
            dummy_actions = torch.randn(1, 10, action_dim).to(device)
            _ = fpt_no_ddp(dummy_video, dummy_actions, return_losses=False)
            _ = ema(dummy_video, dummy_actions, return_losses=False)
        
        # Copy weights
        ema.load_state_dict(fpt_no_ddp.state_dict())
    else:
        ema = deepcopy(fpt_no_ddp).to(device)
    
    requires_grad(ema, False)
    update_ema(ema, fpt_no_ddp, ema_decay)

    optimizer = optim.AdamW(fpt.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))

    # Setup checkpoint directory
    if checkpoint_dir is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path("fpt_models") / run_name
        logging.info("No checkpoint_dir specified, using autogenerated directory %s", checkpoint_dir)
    else:
        checkpoint_dir = Path(checkpoint_dir)
        logging.info("Using provided checkpoint_dir %s", checkpoint_dir)

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if available
    ckpts = sorted(checkpoint_dir.glob("fpt_ckpt_*.pt"))
    train_steps = 0
    if ckpts:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
        data = torch.load(latest, map_location=device)
        fpt_no_ddp.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        if "ema" in data:
            ema.load_state_dict(data["ema"])
        else:
            update_ema(ema, fpt_no_ddp, 0.0)
        train_steps = int(data.get("step", 0))
        logging.info("Loaded checkpoint %s (step %d)", latest, train_steps)

    # Training loop
    running_loss = torch.tensor(0.0)
    num_batches = 0
    loss_history: list[torch.Tensor] = []
    nce_history: list[torch.Tensor] = []
    recon_history: list[torch.Tensor] = []
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    pbar = tqdm(total=max_train_steps, desc="FPT Training") if rank == 0 else None
    if pbar is not None:
        pbar.n = train_steps
        pbar.refresh()
    
    while train_steps < max_train_steps:
        try:
            x, actions = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, actions = next(train_iter)

        x = x.to(device)  # [B, T, H, W, C]
        actions = actions.to(device)  # [B, T, action_dim]
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Forward pass through FPT
            outputs = fpt(x, actions, return_losses=True)
            total_loss = outputs['losses']['total']
            nce_loss = outputs['losses']['nce']
            recon_loss = outputs['losses']['recon']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        update_ema(ema, fpt_no_ddp, ema_decay)

        running_loss += total_loss.detach().cpu()
        num_batches += 1

        # Logging
        if train_steps == 0 or (train_steps + 1) % log_every == 0:
            avg_loss = running_loss / num_batches
            if distributed:
                avg_loss = avg_loss.to(device)
                dist.all_reduce(avg_loss)
                avg_loss /= world_size
                avg_loss_cpu = avg_loss.detach().cpu()
            else:
                avg_loss_cpu = avg_loss.detach().cpu()

            if rank == 0:
                loss_history.append(avg_loss_cpu)
                nce_history.append(nce_loss.detach().cpu())
                recon_history.append(recon_loss.detach().cpu())
                
                if pbar is not None:
                    pbar.set_postfix({
                        "loss": avg_loss_cpu.item(),
                        "nce": nce_loss.item(),
                        "recon": recon_loss.item()
                    })
                
                # Plot losses
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot([i * log_every for i in range(len(loss_history))], 
                        [loss_tensor.cpu().numpy() for loss_tensor in loss_history])
                plt.xlabel("step")
                plt.ylabel("total loss")
                plt.title("Total Loss")
                
                plt.subplot(1, 3, 2)
                plt.plot([i * log_every for i in range(len(nce_history))], 
                        [loss_tensor.cpu().numpy() for loss_tensor in nce_history])
                plt.xlabel("step")
                plt.ylabel("NCE loss")
                plt.title("InfoNCE Loss")
                
                plt.subplot(1, 3, 3)
                plt.plot([i * log_every for i in range(len(recon_history))], 
                        [loss_tensor.cpu().numpy() for loss_tensor in recon_history])
                plt.xlabel("step")
                plt.ylabel("recon loss")
                plt.title("Reconstruction Loss")
                
                plt.tight_layout()
                plt.savefig(checkpoint_dir / "fpt_losses.png")
                plt.close()

            running_loss.zero_()
            num_batches = 0

        # Validation
        if train_steps == 0 or train_steps % validate_every == 0 and rank == 0:
            fpt.eval()
            with torch.no_grad():
                try:
                    val_x, val_actions = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_x, val_actions = next(val_iter)

                val_x = val_x.to(device)
                val_actions = val_actions.to(device)
                
                # Encode and decode for reconstruction quality
                val_latents = fpt.encode(val_x)
                val_reconstructed = fpt.decode(val_latents)
                
                # Compute reconstruction metrics
                mse = F.mse_loss(val_reconstructed, val_x)
                l1 = F.l1_loss(val_reconstructed, val_x)
                
                # Early stopping check
                current_val_loss = mse.item()
                if current_val_loss < best_loss - min_delta:
                    best_loss = current_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        "model": fpt_no_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": train_steps,
                        "val_loss": current_val_loss,
                        "config": {
                            "latent_dim": latent_dim,
                            "action_dim": action_dim,
                            "horizons": horizons_list,
                            "temperature": temperature,
                            "recon_weight": recon_weight,
                            "perceptual_weight": perceptual_weight,
                        }
                    }, checkpoint_dir / "fpt_best_model.pt")
                    print(f"New best model saved with validation loss: {current_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"Validation loss: {current_val_loss:.4f} (best: {best_loss:.4f}, patience: {patience_counter}/{early_stopping_patience})")
                
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered! Best validation loss: {best_loss:.4f}")
                    print(f"Training stopped at step {train_steps}")
                    break
                
                # Save sample reconstruction
                sample_recon = val_reconstructed[0].float().clamp(0, 1)
                sample_orig = val_x[0].float()
                
                # Create comparison image
                comparison = torch.cat([sample_orig, sample_recon], dim=1)  # Concatenate horizontally
                comparison_np = (comparison * 255).byte().cpu().numpy()
                
                step_str = f"{train_steps:09d}"
                imageio.mimsave(checkpoint_dir / f"fpt_recon_{step_str}.gif", 
                              comparison_np, fps=8)
                
                print(f"Validation - MSE: {mse.item():.4f}, L1: {l1.item():.4f}")
            
            fpt.train()

        # Save checkpoint
        if train_steps % save_every == 0 and rank == 0:
            step_str = f"{train_steps:09d}"
            torch.save(
                {
                    "model": fpt_no_ddp.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": train_steps,
                    "config": {
                        "latent_dim": latent_dim,
                        "action_dim": action_dim,
                        "horizons": horizons_list,
                        "temperature": temperature,
                        "recon_weight": recon_weight,
                        "perceptual_weight": perceptual_weight,
                    }
                },
                checkpoint_dir / f"fpt_ckpt_{step_str}.pt",
            )

        train_steps += 1
        if pbar is not None:
            pbar.update(1)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
