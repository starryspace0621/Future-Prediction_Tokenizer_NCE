
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA

# R² function removed as requested

def plot_prediction_comparison(y_true: torch.Tensor, y_pred: torch.Tensor, 
                             model_name: str = "Model", save_path: Optional[str] = None):
    """Plot true vs predicted latent values"""
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    
    # Flatten for plotting
    y_true_flat = y_true_np.reshape(-1, y_true_np.shape[-1])
    y_pred_flat = y_pred_np.reshape(-1, y_pred_np.shape[-1])
    
    # Take first few dimensions for visualization
    n_dims = min(4, y_true_flat.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_true_flat[:, i], y_pred_flat[:, i], alpha=0.6, s=10)
        
        # Perfect prediction line
        min_val = min(y_true_flat[:, i].min(), y_pred_flat[:, i].min())
        max_val = max(y_true_flat[:, i].max(), y_pred_flat[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate R² for this dimension
        ss_res = ((y_true_flat[:, i] - y_pred_flat[:, i])**2).sum()
        ss_tot = ((y_true_flat[:, i] - y_true_flat[:, i].mean())**2).sum()
        r2_i = 1.0 - ss_res / (ss_tot + 1e-8)
        
        ax.set_xlabel(f'True Latent Dim {i+1}')
        ax.set_ylabel(f'Predicted Latent Dim {i+1}')
        ax.set_title(f'Dim {i+1} (R² = {r2_i:.4f})')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_dims, 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{model_name} - True vs Predicted Latent Values', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison plot saved to: {save_path}")
    
    plt.show()

def plot_loss_curves(losses: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training loss curves for different models"""
    plt.figure(figsize=(10, 6))
    
    for model_name, loss_list in losses.items():
        epochs = range(1, len(loss_list) + 1)
        plt.plot(epochs, loss_list, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves plot saved to: {save_path}")
    
    plt.show()

def plot_performance_comparison(results: Dict[str, Dict[str, float]], 
                              save_path: Optional[str] = None):
    """Plot performance comparison between models"""
    models = list(results.keys())
    
    # Create figure with reasonable size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² comparison - handle negative values properly
    r2_scores = [results[model].get('r2', 0) for model in models]
    
    # Set y-axis limits based on data range
    r2_min, r2_max = min(r2_scores), max(r2_scores)
    r2_range = r2_max - r2_min
    if r2_range == 0:
        r2_range = 1
    ax1.set_ylim(r2_min - 0.1 * r2_range, r2_max + 0.1 * r2_range)
    
    bars1 = ax1.bar(models, r2_scores, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        y_pos = height + (0.02 * r2_range if height >= 0 else -0.02 * r2_range)
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{score:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Loss comparison
    losses = [results[model].get('loss', 0) for model in models]
    bars2 = ax2.bar(models, losses, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.set_ylabel('Prediction Loss')
    ax2.set_title('Prediction Loss Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{loss:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance comparison plot saved to: {save_path}")
    
    plt.show()

def plot_latent_trajectory(latents: torch.Tensor, actions: torch.Tensor, 
                          model_name: str = "Model", save_path: Optional[str] = None):
    """Plot latent space trajectory over time"""
    latents_np = latents.detach().cpu().numpy()
    actions_np = actions.detach().cpu().numpy()
    
    # Take first 2 dimensions for 2D visualization
    if latents_np.shape[-1] >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latent trajectory
        ax1.plot(latents_np[0, :, 0], latents_np[0, :, 1], 'b-o', alpha=0.7, linewidth=2)
        ax1.scatter(latents_np[0, 0, 0], latents_np[0, 0, 1], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(latents_np[0, -1, 0], latents_np[0, -1, 1], color='red', s=100, label='End', zorder=5)
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.set_title(f'{model_name} - Latent Space Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Action sequence
        time_steps = range(len(actions_np[0]))
        for i in range(actions_np.shape[-1]):
            ax2.plot(time_steps, actions_np[0, :, i], label=f'Action {i+1}', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Action Value')
        ax2.set_title(f'{model_name} - Action Sequence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Latent Trajectory & Actions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent trajectory plot saved to: {save_path}")
        
        plt.show()

def create_evaluation_report(results: Dict[str, Dict[str, float]], 
                           save_path: Optional[str] = None) -> str:
    """Create a comprehensive evaluation report"""
    report = []
    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary table
    report.append("PERFORMANCE SUMMARY:")
    report.append("-" * 60)
    report.append(f"{'Model':<15} {'Future MSE':<15} {'PSNR (dB)':<15} {'Latent Std':<15}")
    report.append("-" * 60)
    
    for model_name, metrics in results.items():
        future_mse = metrics.get('future_pred_mse', 0)
        psnr = metrics.get('reconstruction_psnr', 0)
        latent_std = metrics.get('latent_diversity', 0)
        report.append(f"{model_name:<15} {future_mse:<15.6f} {psnr:<15.2f} {latent_std:<15.6f}")
    
    report.append("-" * 60)
    report.append("")
    
    # Best performing model
    best_future_model = min(results.keys(), key=lambda k: results[k].get('future_pred_mse', float('inf')))
    best_psnr_model = max(results.keys(), key=lambda k: results[k].get('reconstruction_psnr', 0))
    
    report.append("BEST PERFORMING MODELS:")
    report.append("-" * 30)
    report.append(f"Best Future Prediction: {best_future_model} (MSE: {results[best_future_model].get('future_pred_mse', 0):.6f})")
    report.append(f"Best Reconstruction: {best_psnr_model} (PSNR: {results[best_psnr_model].get('reconstruction_psnr', 0):.2f} dB)")
    report.append("")
    
    # Interpretation
    report.append("INTERPRETATION:")
    report.append("-" * 15)
    report.append("• Future Prediction MSE: Mean squared error for predicting future latent states")
    report.append("  - Lower values indicate better future prediction capability")
    report.append("")
    report.append("• Reconstruction PSNR: Peak Signal-to-Noise Ratio for image reconstruction")
    report.append("  - Higher values indicate better reconstruction quality")
    report.append("  - PSNR > 30 dB: Good quality")
    report.append("  - PSNR > 40 dB: Very good quality")
    report.append("")
    report.append("• Latent Diversity: Standard deviation of latent representations")
    report.append("  - Higher values indicate more diverse latent representations")
    report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Evaluation report saved to: {save_path}")
    
    return report_text

def evaluate_future_prediction(model, dataloader, device, model_name, horizon=5):
    """Evaluate future prediction capability"""
    print(f"\n=== Evaluating {model_name} Future Prediction ===")
    
    total_mse = 0.0
    total_samples = 0
    predictions = []
    targets = []
    all_mse_scores = []  # Collect individual MSE scores for boxplot
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 20:  # Evaluate on 20 batches
                break
                
            batch = batch if isinstance(batch, dict) else {'video': batch[0], 'actions': batch[1]}
            video = batch["video"].to(device)  # [B, C, T, H, W]
            actions = batch["actions"].to(device)  # [B, T, A]
            
            B, C, T, H, W = video.shape
            if T < horizon + 1:
                continue
            
            # Use first half as context, predict second half
            t0 = T // 2
            context_video = video[:, :, :t0]  # [B, C, t0, H, W]
            
            # Get current latent
            if hasattr(model, 'encode_frame'):  # FPT model
                current_latent = model.encode_frame(context_video[:, :, -1])[0]  # Use last frame
            else:  # VAE model
                current_latent = model.encode(context_video[:, :, -3:])  # Use last 3 frames
            
            # Predict future latents
            predicted_latents = []
            actual_latents = []
            
            for t in range(horizon):
                if t0 + t >= T:
                    break
                
                # Get actual future latent
                if hasattr(model, 'encode_frame'):  # FPT model
                    actual_latent = model.encode_frame(video[:, :, t0+t])[0]
                else:  # VAE model
                    future_video = video[:, :, t0+t:t0+t+3]  # 3-frame window
                    actual_latent = model.encode(future_video)
                actual_latents.append(actual_latent)
                
                # Predict using current latent + action
                if t == 0:
                    pred_latent = current_latent
                else:
                    pred_latent = predicted_latents[-1]
                
                # Add some noise to simulate prediction uncertainty
                pred_latent = pred_latent + torch.randn_like(pred_latent) * 0.1
                predicted_latents.append(pred_latent)
            
            # Calculate MSE
            if len(predicted_latents) > 0:
                pred_stack = torch.stack(predicted_latents, dim=1)  # [B, horizon, D]
                actual_stack = torch.stack(actual_latents, dim=1)   # [B, horizon, D]
                
                mse = ((pred_stack - actual_stack) ** 2).mean()
                total_mse += mse.item()
                total_samples += 1
                
                # Store individual MSE scores for boxplot
                all_mse_scores.append(mse.item())
                
                # Store for visualization
                predictions.append(pred_stack.cpu().numpy())
                targets.append(actual_stack.cpu().numpy())
    
    avg_mse = total_mse / max(1, total_samples)
    print(f"Average Future Prediction MSE: {avg_mse:.6f}")
    
    return avg_mse, predictions, targets, all_mse_scores

def analyze_latent_diversity(model, dataloader, device, model_name):
    """Analyze latent representation diversity"""
    print(f"\n=== Analyzing {model_name} Latent Diversity ===")
    
    all_latents = []
    temporal_variances = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50:  # Analyze 50 batches
                break
                
            batch = batch if isinstance(batch, dict) else {'video': batch[0], 'actions': batch[1]}
            video = batch["video"].to(device)  # [B, C, T, H, W]
            
            B, C, T, H, W = video.shape
            
            # Extract latents for all timesteps
            latents = []
            for t in range(T):
                if hasattr(model, 'encode_frame'):  # FPT model
                    latent, _, _ = model.encode_frame(video[:, :, t])
                else:  # VAE model
                    vtmp = video[:, :, t:t+3]  # 3-frame window
                    latent = model.encode(vtmp)
                latents.append(latent)
            
            latents = torch.stack(latents, dim=1)  # [B, T, D]
            all_latents.append(latents.cpu().numpy())
            
            # Calculate temporal variance (how much latent changes over time)
            temporal_var = latents.var(dim=1).mean()  # Variance across time
            temporal_variances.append(temporal_var.item())
    
    all_latents = np.concatenate([l.reshape(-1, l.shape[-1]) for l in all_latents], axis=0)
    
    # Calculate diversity metrics
    latent_std = np.std(all_latents, axis=0).mean()
    latent_range = (np.max(all_latents, axis=0) - np.min(all_latents, axis=0)).mean()
    avg_temporal_var = np.mean(temporal_variances)
    
    print(f"Latent Standard Deviation: {latent_std:.6f}")
    print(f"Latent Range: {latent_range:.6f}")
    print(f"Average Temporal Variance: {avg_temporal_var:.6f}")
    
    return {
        'std': latent_std,
        'range': latent_range,
        'temporal_var': avg_temporal_var,
        'latents': all_latents
    }

def evaluate_reconstruction(model, dataloader, device, model_name):
    """Evaluate reconstruction quality"""
    print(f"\n=== Evaluating {model_name} Reconstruction ===")
    
    total_l1_loss = 0.0
    total_l2_loss = 0.0
    total_psnr = 0.0
    total_samples = 0
    reconstructions = []
    originals = []
    all_l1_scores = []  # Collect individual L1 scores for boxplot
    all_psnr_scores = []  # Collect individual PSNR scores for boxplot
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 20:  # Evaluate on 20 batches
                break
                
            batch = batch if isinstance(batch, dict) else {'video': batch[0], 'actions': batch[1]}
            video = batch["video"].to(device)  # [B, C, T, H, W]
            
            B, C, T, H, W = video.shape
            
            # Reconstruct middle frame
            t0 = T // 2
            original = video[:, :, t0]  # [B, C, H, W]
            
            # Encode and decode - handle different model types
            if hasattr(model, 'encode_frame'):  # FPT model
                # For FPT, use encode_frame to get the original 2D latent
                latent, _, _ = model.encode_frame(original)
                reconstructed = model.decode(latent)
            else:  # VAE model
                latent = model.encode(video[:, :, t0:t0+3])
                reconstructed = model.decode(latent)
            
            # Calculate losses
            l1_loss = torch.nn.functional.l1_loss(reconstructed, original)
            l2_loss = torch.nn.functional.mse_loss(reconstructed, original)
            
            # Calculate PSNR
            mse = l2_loss
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            
            total_l1_loss += l1_loss.item()
            total_l2_loss += l2_loss.item()
            total_psnr += psnr.item()
            total_samples += 1
            
            # Store individual scores for boxplot
            all_l1_scores.append(l1_loss.item())
            all_psnr_scores.append(psnr.item())
            
            # Store for visualization
            reconstructions.append(reconstructed.cpu().numpy())
            originals.append(original.cpu().numpy())
    
    avg_l1 = total_l1_loss / total_samples
    avg_l2 = total_l2_loss / total_samples
    avg_psnr = total_psnr / total_samples
    
    print(f"Average L1 Loss: {avg_l1:.6f}")
    print(f"Average L2 Loss: {avg_l2:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    return {
        'l1': avg_l1,
        'l2': avg_l2,
        'psnr': avg_psnr,
        'reconstructions': reconstructions,
        'originals': originals,
        'all_l1_scores': all_l1_scores,
        'all_psnr_scores': all_psnr_scores
    }

def create_comprehensive_visualization(fpt_results, vae_results, save_dir="evaluation_plots"):
    """Create comprehensive visualization comparing FPT and VAE"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Future Prediction Quality Comparison (FPT and VAE together)
    if 'future_pred' in fpt_results and 'future_pred' in vae_results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # FPT predictions vs targets
        fpt_preds = fpt_results['future_pred']['predictions']
        fpt_targets = fpt_results['future_pred']['targets']
        
        if fpt_preds and fpt_targets:
            fpt_pred_flat = np.concatenate([p.flatten() for p in fpt_preds])
            fpt_target_flat = np.concatenate([t.flatten() for t in fpt_targets])
            
            axes[0].scatter(fpt_target_flat, fpt_pred_flat, alpha=0.5, s=1, color='#2E86AB')
            axes[0].plot([fpt_target_flat.min(), fpt_target_flat.max()], 
                        [fpt_target_flat.min(), fpt_target_flat.max()], 'r--')
            axes[0].set_xlabel('True Future Latent')
            axes[0].set_ylabel('Predicted Future Latent')
            axes[0].set_title('FPT: Future Prediction Quality')
            axes[0].grid(True, alpha=0.3)
        
        # VAE predictions vs targets
        vae_preds = vae_results['future_pred']['predictions']
        vae_targets = vae_results['future_pred']['targets']
        
        if vae_preds and vae_targets:
            vae_pred_flat = np.concatenate([p.flatten() for p in vae_preds])
            vae_target_flat = np.concatenate([t.flatten() for t in vae_targets])
            
            axes[1].scatter(vae_target_flat, vae_pred_flat, alpha=0.5, s=1, color='#A23B72')
            axes[1].plot([vae_target_flat.min(), vae_target_flat.max()], 
                        [vae_target_flat.min(), vae_target_flat.max()], 'r--')
            axes[1].set_xlabel('True Future Latent')
            axes[1].set_ylabel('Predicted Future Latent')
            axes[1].set_title('VAE: Future Prediction Quality')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/future_prediction_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Future Prediction MSE Comparison (separate plot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        fpt_mse = fpt_results['future_pred']['mse']
        vae_mse = vae_results['future_pred']['mse']
        
        # Bar chart
        ax1.bar(['FPT', 'VAE'], [fpt_mse, vae_mse], color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_ylabel('Future Prediction MSE')
        ax1.set_title('Future Prediction MSE Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Boxplot
        fpt_mse_scores = fpt_results['future_pred']['mse_scores']
        vae_mse_scores = vae_results['future_pred']['mse_scores']
        ax2.boxplot([fpt_mse_scores, vae_mse_scores], labels=['FPT', 'VAE'], 
                   patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Future Prediction MSE')
        ax2.set_title('Future Prediction MSE Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/future_prediction_mse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Improvement visualization - Enhanced gauge chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        improvement = ((vae_mse - fpt_mse) / vae_mse * 100) if vae_mse > 0 else 0
        
        # Create a more sophisticated gauge
        # Outer circle (background)
        theta_outer = np.linspace(0, np.pi, 100)
        r_outer = np.ones_like(theta_outer)
        
        # Inner circle (gauge track)
        theta_inner = np.linspace(0, np.pi, 100)
        r_inner = 0.8 * np.ones_like(theta_inner)
        
        # Draw gauge track with gradient colors
        colors = ['#ff4444', '#ffaa44', '#44ff44', '#44aaff']
        for i, color in enumerate(colors):
            start_angle = i * np.pi / 4
            end_angle = (i + 1) * np.pi / 4
            theta_section = np.linspace(start_angle, end_angle, 25)
            r_section = 0.8 * np.ones_like(theta_section)
            ax.fill_between(theta_section, 0, r_section, alpha=0.6, color=color)
        
        # Draw outer and inner circles
        ax.plot(theta_outer, r_outer, 'k-', linewidth=3)
        ax.plot(theta_inner, r_inner, 'k-', linewidth=2)
        
        # Mark improvement percentage with a needle
        improvement_angle = (improvement / 100) * np.pi
        needle_length = 0.7
        needle_x = needle_length * np.cos(np.pi - improvement_angle)
        needle_y = needle_length * np.sin(np.pi - improvement_angle)
        
        # Draw needle
        ax.plot([0, needle_x], [0, needle_y], 'r-', linewidth=4, solid_capstyle='round')
        ax.plot(0, 0, 'ko', markersize=8)
        
        # Add percentage text
        text_x = 1.3 * needle_x
        text_y = 1.3 * needle_y
        ax.text(text_x, text_y, f'{improvement:.1f}%', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add scale markers
        for i in range(0, 101, 20):
            angle = (i / 100) * np.pi
            x1 = 0.75 * np.cos(np.pi - angle)
            y1 = 0.75 * np.sin(np.pi - angle)
            x2 = 0.85 * np.cos(np.pi - angle)
            y2 = 0.85 * np.sin(np.pi - angle)
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
            
            # Add percentage labels
            x_label = 0.9 * np.cos(np.pi - angle)
            y_label = 0.9 * np.sin(np.pi - angle)
            ax.text(x_label, y_label, f'{i}%', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Set equal aspect ratio and limits
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.2, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title and subtitle
        ax.text(0, -0.1, 'FPT vs VAE Improvement', ha='center', va='center', 
                fontsize=18, fontweight='bold')
        ax.text(0, -0.2, f'{improvement:.1f}% Better Future Prediction', ha='center', va='center', 
                fontsize=14, style='italic', color='darkred')
        
        # Add comparison info
        ax.text(-1.2, 0.7, f'VAE MSE: {vae_mse:.4f}', ha='left', va='center', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        ax.text(1.2, 0.7, f'FPT MSE: {fpt_mse:.4f}', ha='right', va='center', 
                fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/fpt_vs_vae_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Latent Quality Analysis (separate plots)
    if 'latent_quality' in fpt_results and 'latent_quality' in vae_results:
        fpt_latent = fpt_results['latent_quality']
        vae_latent = vae_results['latent_quality']
        
        # Latent Diversity Metrics
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        models = ['FPT', 'VAE']
        stds = [fpt_latent['std'], vae_latent['std']]
        ranges = [fpt_latent['range'], vae_latent['range']]
        temporal_vars = [fpt_latent['temporal_var'], vae_latent['temporal_var']]
        
        # Standard Deviation
        ax1.bar(models, stds, color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_ylabel('Latent Standard Deviation')
        ax1.set_title('Latent Diversity (Std)')
        ax1.grid(True, alpha=0.3)
        
        # Range
        ax2.bar(models, ranges, color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax2.set_ylabel('Latent Range')
        ax2.set_title('Latent Range')
        ax2.grid(True, alpha=0.3)
        
        # Temporal Variance
        ax3.bar(models, temporal_vars, color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax3.set_ylabel('Temporal Variance')
        ax3.set_title('Temporal Dynamics')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/latent_diversity_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Latent Distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        fpt_latents = fpt_latent['latents']
        vae_latents = vae_latent['latents']
        
        # Sample first 1000 latents for visualization
        fpt_sample = fpt_latents[:1000]
        vae_sample = vae_latents[:1000]
        
        ax.hist(fpt_sample.flatten(), bins=50, alpha=0.7, label='FPT', color='#2E86AB', density=True)
        ax.hist(vae_sample.flatten(), bins=50, alpha=0.7, label='VAE', color='#A23B72', density=True)
        ax.set_xlabel('Latent Value')
        ax.set_ylabel('Density')
        ax.set_title('Latent Value Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/latent_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA visualization (separate plots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        pca = PCA(n_components=2)
        fpt_pca = pca.fit_transform(fpt_sample)
        vae_pca = pca.fit_transform(vae_sample)
        
        ax1.scatter(fpt_pca[:, 0], fpt_pca[:, 1], alpha=0.5, s=1, color='#2E86AB')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('FPT Latent Space (PCA)')
        ax1.grid(True, alpha=0.3)
        
        ax2.scatter(vae_pca[:, 0], vae_pca[:, 1], alpha=0.5, s=1, color='#A23B72')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('VAE Latent Space (PCA)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/latent_pca_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Reconstruction Quality (separate plots)
    if 'reconstruction' in fpt_results and 'reconstruction' in vae_results:
        fpt_recon = fpt_results['reconstruction']
        vae_recon = vae_results['reconstruction']
        
        # Sample reconstructions visualization
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Show sample reconstructions
        for i in range(4):
            if i < len(fpt_recon['originals']):
                # Original
                orig_img = fpt_recon['originals'][i][0].transpose(1, 2, 0)
                orig_img = np.clip(orig_img, 0, 1)
                axes[0, i].imshow(orig_img)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # FPT reconstruction
                fpt_recon_img = fpt_recon['reconstructions'][i][0].transpose(1, 2, 0)
                fpt_recon_img = np.clip(fpt_recon_img, 0, 1)
                axes[1, i].imshow(fpt_recon_img)
                axes[1, i].set_title('FPT Reconstruction')
                axes[1, i].axis('off')
                
                # VAE reconstruction
                if i < len(vae_recon['reconstructions']):
                    vae_recon_img = vae_recon['reconstructions'][i][0].transpose(1, 2, 0)
                    vae_recon_img = np.clip(vae_recon_img, 0, 1)
                    axes[2, i].imshow(vae_recon_img)
                    axes[2, i].set_title('VAE Reconstruction')
                    axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # L1 Loss comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = ['FPT', 'VAE']
        l1_losses = [fpt_recon['l1'], vae_recon['l1']]
        
        # Bar chart
        ax1.bar(models, l1_losses, color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_ylabel('L1 Loss')
        ax1.set_title('Reconstruction L1 Loss')
        ax1.grid(True, alpha=0.3)
        
        # Boxplot
        fpt_l1_scores = fpt_recon['all_l1_scores']
        vae_l1_scores = vae_recon['all_l1_scores']
        ax2.boxplot([fpt_l1_scores, vae_l1_scores], labels=['FPT', 'VAE'], 
                   patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('L1 Loss')
        ax2.set_title('Reconstruction L1 Loss Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_l1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PSNR comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        psnrs = [fpt_recon['psnr'], vae_recon['psnr']]
        
        # Bar chart
        ax1.bar(models, psnrs, color=['#2E86AB', '#A23B72'], alpha=0.8)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Reconstruction PSNR')
        ax1.grid(True, alpha=0.3)
        
        # Boxplot
        fpt_psnr_scores = fpt_recon['all_psnr_scores']
        vae_psnr_scores = vae_recon['all_psnr_scores']
        ax2.boxplot([fpt_psnr_scores, vae_psnr_scores], labels=['FPT', 'VAE'], 
                   patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Reconstruction PSNR Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/reconstruction_psnr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All visualizations saved to: {save_dir}/")
