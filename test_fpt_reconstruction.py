"""
Test FPT reconstruction capabilities
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from fpt import FPT
from dataset import OpenXMP4VideoDataset


def test_fpt_reconstruction(
    dataset_dir: str = "sample_data",
    subset_names: str = "bridge",
    output_dir: str = "./fpt_reconstruction_test",
    num_samples: int = 5,
    fpt_checkpoint: str = None,
    latent_dim: int = 16,
    action_dim: int = 10,
):
    """
    Test FPT reconstruction capabilities
    
    Args:
        dataset_dir: Dataset directory
        subset_names: Dataset subset names
        output_dir: Output directory
        num_samples: Number of test samples
        fpt_checkpoint: FPT checkpoint path
        latent_dim: Latent dimension
        action_dim: Action dimension
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = OpenXMP4VideoDataset(
        save_dir=dataset_dir,
        input_h=256,
        input_w=256,
        n_frames=10,
        frame_skip=1,
        action_dim=action_dim,
        subset_names=subset_names,
        split="test",
    )
    
    # Create FPT model
    print("Creating FPT model...")
    fpt = FPT(
        latent_dim=latent_dim,
        action_dim=action_dim,
        horizons=[1, 2, 4, 8],
        temperature=0.1,
        recon_weight=0.1,
        perceptual_weight=0.0,
        use_vae_init=True,  # Use VAE initialization
        vae_cache_dir="./models",
    ).to(device)
    
    # Load checkpoint if available
    if fpt_checkpoint and Path(fpt_checkpoint).exists():
        print(f"Loading FPT checkpoint: {fpt_checkpoint}")
        checkpoint = torch.load(fpt_checkpoint, map_location=device)
        fpt.load_state_dict(checkpoint.get("model", checkpoint))
        print("FPT checkpoint loaded successfully")
    else:
        print("Using randomly initialized FPT model")
    
    fpt.eval()
    
    # Test reconstruction
    print("Starting reconstruction test...")
    mse_losses = []
    l1_losses = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"Processing sample {i+1}/{num_samples}")
        
        # Get data
        video, actions = dataset[i]
        video = video.unsqueeze(0).to(device)  # [1, T, H, W, C]
        actions = actions.unsqueeze(0).to(device)  # [1, T, action_dim]
        
        # FPT reconstruction
        with torch.no_grad():
            # Encode
            latents = fpt.encode(video)  # [1, T, latent_dim]
            print(f"Latent shape: {latents.shape}")
            
            # Decode
            reconstructed = fpt.decode(latents)  # [1, T, H, W, C]
            print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Calculate losses
        mse_loss = F.mse_loss(reconstructed, video)
        l1_loss = F.l1_loss(reconstructed, video)
        
        mse_losses.append(mse_loss.item())
        l1_losses.append(l1_loss.item())
        
        print(f"MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}")
        
        # Save comparison images
        save_comparison_images(
            original=video[0],  # [T, H, W, C]
            reconstructed=reconstructed[0],  # [T, H, W, C]
            output_path=output_path / f"fpt_sample_{i:03d}",
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item()
        )
        
        # Test action-conditioned prediction
        test_action_prediction(fpt, video, actions, output_path / f"fpt_sample_{i:03d}")
    
    # Calculate statistics
    avg_mse = np.mean(mse_losses)
    avg_l1 = np.mean(l1_losses)
    
    print(f"\n=== FPT Reconstruction Results ===")
    print(f"Average MSE Loss: {avg_mse:.4f}")
    print(f"Average L1 Loss: {avg_l1:.4f}")
    print(f"MSE Std: {np.std(mse_losses):.4f}")
    print(f"L1 Std: {np.std(l1_losses):.4f}")
    
    # Save statistics
    with open(output_path / "fpt_reconstruction_stats.txt", "w") as f:
        f.write(f"FPT Reconstruction Test Results\n")
        f.write(f"Dataset: {dataset_dir}/{subset_names}\n")
        f.write(f"Test samples: {num_samples}\n")
        f.write(f"Latent dim: {latent_dim}\n")
        f.write(f"Checkpoint: {fpt_checkpoint or 'Random initialization'}\n")
        f.write(f"Average MSE Loss: {avg_mse:.4f}\n")
        f.write(f"Average L1 Loss: {avg_l1:.4f}\n")
        f.write(f"MSE Std: {np.std(mse_losses):.4f}\n")
        f.write(f"L1 Std: {np.std(l1_losses):.4f}\n")
    
    return {
        "mse_losses": mse_losses,
        "l1_losses": l1_losses,
        "avg_mse": avg_mse,
        "avg_l1": avg_l1
    }


def test_action_prediction(fpt, video, actions, output_path):
    """
    Test action-conditioned prediction
    
    Args:
        fpt: FPT model
        video: Video [1, T, H, W, C]
        actions: Actions [1, T, action_dim]
        output_path: Output path
    """
    print("Testing action-conditioned prediction...")
    
    with torch.no_grad():
        # Get current frame and action
        t_current = video.shape[1] // 2
        current_frame = video[:, t_current]  # [1, H, W, C]
        current_action = actions[:, t_current]  # [1, action_dim]
        
        # Predict future representation
        predicted_latent = fpt.predict_future(current_frame, current_action)  # [1, latent_dim]
        
        # Decode predicted representation
        predicted_spatial = predicted_latent.unsqueeze(-1).unsqueeze(-1)  # [1, latent_dim, 1, 1]
        predicted_spatial = predicted_spatial.expand(-1, -1, 32, 32)  # [1, latent_dim, 32, 32]
        predicted_frame = fpt.decoder(predicted_spatial)  # [1, C, H, W]
        predicted_frame = predicted_frame.permute(0, 2, 3, 1)  # [1, H, W, C]
        
        # Save prediction results
        save_prediction_comparison(
            current_frame=current_frame[0],  # [H, W, C]
            predicted_frame=predicted_frame[0],  # [H, W, C]
            output_path=output_path
        )


def save_comparison_images(original, reconstructed, output_path, mse_loss, l1_loss):
    """
    Save comparison images
    """
    T, H, W, C = original.shape
    
    # Select middle frame for comparison
    t_mid = T // 2
    orig_frame = original[t_mid].cpu().numpy()  # [H, W, C]
    recon_frame = reconstructed[t_mid].cpu().numpy()  # [H, W, C]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(orig_frame)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(recon_frame)
    axes[1].set_title("FPT Reconstructed")
    axes[1].axis('off')
    
    # Difference image
    diff = np.abs(orig_frame - recon_frame)
    axes[2].imshow(diff)
    axes[2].set_title(f"Difference\nMSE: {mse_loss:.4f}, L1: {l1_loss:.4f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save video comparison
    try:
        save_video_comparison(original, reconstructed, f"{output_path}_video.gif")
    except Exception as e:
        print(f"Video save failed: {e}")


def save_prediction_comparison(current_frame, predicted_frame, output_path):
    """
    Save action prediction comparison
    """
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Current frame
    axes[0].imshow(current_frame.cpu().numpy())
    axes[0].set_title("Current Frame")
    axes[0].axis('off')
    
    # Predicted frame
    axes[1].imshow(predicted_frame.cpu().numpy())
    axes[1].set_title("Action-Conditioned Prediction")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_action_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_video_comparison(original, reconstructed, output_path):
    """
    Save video comparison
    """
    try:
        import imageio
        
        T, H, W, C = original.shape
        
        # Create side-by-side video
        comparison_frames = []
        for t in range(T):
            orig_frame = original[t].cpu().numpy()
            recon_frame = reconstructed[t].cpu().numpy()
            
            # Side-by-side display
            side_by_side = np.concatenate([orig_frame, recon_frame], axis=1)
            comparison_frames.append((side_by_side * 255).astype(np.uint8))
        
        # Save as GIF
        imageio.mimsave(output_path, comparison_frames, fps=4)
        print(f"Video comparison saved: {output_path}")
        
    except Exception as e:
        print(f"Video save failed: {e}")


if __name__ == "__main__":
    import fire
    fire.Fire(test_fpt_reconstruction)
