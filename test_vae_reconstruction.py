"""
Test VAE reconstruction capabilities
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from vae import VAE
from dataset import OpenXMP4VideoDataset


def test_vae_reconstruction(
    dataset_dir: str = "sample_data",
    subset_names: str = "bridge",
    output_dir: str = "./vae_reconstruction_test",
    num_samples: int = 5
):
    """
    Test VAE reconstruction capabilities
    
    Args:
        dataset_dir: Dataset directory
        subset_names: Dataset subset names
        output_dir: Output directory
        num_samples: Number of test samples
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
        action_dim=10,
        subset_names=subset_names,
        split="test",
    )
    
    # Create VAE
    print("Loading VAE...")
    vae = VAE(cache_dir="./models").to(device)
    
    # Test reconstruction
    print("Starting reconstruction test...")
    mse_losses = []
    l1_losses = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"Processing sample {i+1}/{num_samples}")
        
        # Get data
        video, actions = dataset[i]
        video = video.unsqueeze(0).to(device)  # [1, T, H, W, C]
        
        # VAE reconstruction
        with torch.no_grad():
            # Encode
            latent = vae.encode(video)  # [1, T, H/8, W/8, C]
            print(f"Latent shape: {latent.shape}")
            
            # Decode
            reconstructed = vae.decode(latent)  # [1, T, H, W, C]
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
            output_path=output_path / f"sample_{i:03d}",
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item()
        )
    
    # Calculate statistics
    avg_mse = np.mean(mse_losses)
    avg_l1 = np.mean(l1_losses)
    
    print(f"\n=== VAE Reconstruction Results ===")
    print(f"Average MSE Loss: {avg_mse:.4f}")
    print(f"Average L1 Loss: {avg_l1:.4f}")
    print(f"MSE Std: {np.std(mse_losses):.4f}")
    print(f"L1 Std: {np.std(l1_losses):.4f}")
    
    # Save statistics
    with open(output_path / "reconstruction_stats.txt", "w") as f:
        f.write(f"VAE Reconstruction Test Results\n")
        f.write(f"Dataset: {dataset_dir}/{subset_names}\n")
        f.write(f"Test samples: {num_samples}\n")
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


def save_comparison_images(original, reconstructed, output_path, mse_loss, l1_loss):
    """
    Save comparison images
    
    Args:
        original: Original images [T, H, W, C]
        reconstructed: Reconstructed images [T, H, W, C]
        output_path: Output path
        mse_loss: MSE loss
        l1_loss: L1 loss
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
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')
    
    # Difference image
    diff = np.abs(orig_frame - recon_frame)
    axes[2].imshow(diff)
    axes[2].set_title(f"Difference\nMSE: {mse_loss:.4f}, L1: {l1_loss:.4f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save video comparison (using simpler method)
    try:
        save_video_comparison(original, reconstructed, f"{output_path}_video.mp4")
    except Exception as e:
        print(f"Video save failed: {e}")
        print("Skipping video save, continuing...")


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
        
        # Save video - use more compatible format
        imageio.mimsave(output_path, comparison_frames, fps=8, codec='libx264')
        
    except Exception as e:
        print(f"Video save failed, trying GIF: {e}")
        try:
            # If MP4 fails, try saving as GIF
            gif_path = output_path.replace('.mp4', '.gif')
            imageio.mimsave(gif_path, comparison_frames, fps=4)
            print(f"Saved as GIF: {gif_path}")
        except Exception as e2:
            print(f"GIF save also failed: {e2}")
            # Save as individual image frames
            for i, frame in enumerate(comparison_frames):
                imageio.imwrite(f"{output_path.replace('.mp4', '')}_frame_{i:03d}.png", frame)
            print(f"Saved as image frames: {output_path.replace('.mp4', '')}_frame_*.png")


if __name__ == "__main__":
    import fire
    fire.Fire(test_vae_reconstruction)
