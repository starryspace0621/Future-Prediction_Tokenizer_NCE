#!/usr/bin/env python3
"""
Comprehensive evaluation script for FPT vs VAE comparison
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from utils.common import load_config, set_seed, to_device
from data.video_loader import VideoClips
from tokenizers.vae import VAE
from tokenizers.fpt_rvq import FPT_RVQ
from .metrics import (
    plot_prediction_comparison, plot_performance_comparison,
    plot_latent_trajectory, create_evaluation_report,
    evaluate_future_prediction, analyze_latent_diversity, 
    evaluate_reconstruction, create_comprehensive_visualization
)

def load_model(model_path, model_type, device):
    """Load trained model"""
    print(f"Loading {model_type} model from {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    cfg = sd.get("config", None)
    
    if model_type == "vae":
        model = VAE(z_dim=cfg["model"]["z_dim"], beta_kl=cfg["model"]["beta_kl"])
    else:
        mcfg = cfg["model"]
        model = FPT_RVQ(latent_dim=mcfg["latent_dim"],
                       rvq_levels=mcfg["rvq_levels"],
                       codebook_size=mcfg["codebook_size"],
                       commit_weight=mcfg["commit_weight"],
                       temp=mcfg["temp"],
                       horizons=tuple(mcfg["horizons"]),
                       future_pred_weight=mcfg.get("future_pred_weight", 1.0))
    
    model.load_state_dict(sd["model"], strict=False)
    model.to(device).eval()
    
    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False
    
    return model

# Probe evaluation function removed as requested

def main():
    parser = argparse.ArgumentParser(description="Comprehensive FPT vs VAE Evaluation")
    parser.add_argument("--config", type=str, default="configs/fpt_64.yaml", help="Config file path")
    parser.add_argument("--fpt-tokenizer", type=str, default="runs/fpt_model/tokenizer_latest.pt", help="FPT tokenizer checkpoint")
    parser.add_argument("--vae-tokenizer", type=str, default="runs/vae_model/tokenizer_latest.pt", help="VAE tokenizer checkpoint")
    # Probe checkpoints removed as requested
    parser.add_argument("--output-dir", type=str, default="evaluation_plots", help="Output directory for plots")
    # Skip probe parameter removed as requested
    parser.add_argument("--skip-future", action="store_true", help="Skip future prediction evaluation")
    parser.add_argument("--skip-latent", action="store_true", help="Skip latent quality analysis")
    parser.add_argument("--skip-recon", action="store_true", help="Skip reconstruction evaluation")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    
    ds = VideoClips(cfg["dataset"]["root"], split="val", 
                   clip_len=cfg["dataset"]["clip_len"], resolution=cfg["dataset"]["resolution"])
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    
    print(f"Validation dataset: {len(ds)} samples")
    
    # Load models
    fpt_tokenizer = load_model(args.fpt_tokenizer, "fpt", device)
    vae_tokenizer = load_model(args.vae_tokenizer, "vae", device)
    
    # Store results
    fpt_results = {}
    vae_results = {}
    
    # 1. Future Prediction Evaluation
    if not args.skip_future:
        print("\n" + "="*60)
        print("FUTURE PREDICTION EVALUATION")
        print("="*60)
        
        fpt_mse, fpt_preds, fpt_targets, fpt_mse_scores = evaluate_future_prediction(fpt_tokenizer, dl, device, "FPT")
        vae_mse, vae_preds, vae_targets, vae_mse_scores = evaluate_future_prediction(vae_tokenizer, dl, device, "VAE")
        
        fpt_results['future_pred'] = {
            'mse': fpt_mse,
            'predictions': fpt_preds,
            'targets': fpt_targets,
            'mse_scores': fpt_mse_scores
        }
        
        vae_results['future_pred'] = {
            'mse': vae_mse,
            'predictions': vae_preds,
            'targets': vae_targets,
            'mse_scores': vae_mse_scores
        }
        
        print(f"\nFuture Prediction Comparison:")
        print(f"FPT MSE: {fpt_mse:.6f}")
        print(f"VAE MSE: {vae_mse:.6f}")
        if vae_mse > 0:
            improvement = ((vae_mse - fpt_mse) / vae_mse * 100)
            print(f"Improvement: {improvement:.1f}%")
    
    # 2. Latent Quality Analysis
    if not args.skip_latent:
        print("\n" + "="*60)
        print("LATENT QUALITY ANALYSIS")
        print("="*60)
        
        fpt_latent_results = analyze_latent_diversity(fpt_tokenizer, dl, device, "FPT")
        vae_latent_results = analyze_latent_diversity(vae_tokenizer, dl, device, "VAE")
        
        fpt_results['latent_quality'] = fpt_latent_results
        vae_results['latent_quality'] = vae_latent_results
    
    # 3. Reconstruction Quality Evaluation
    if not args.skip_recon:
        print("\n" + "="*60)
        print("RECONSTRUCTION QUALITY EVALUATION")
        print("="*60)
        
        fpt_recon_results = evaluate_reconstruction(fpt_tokenizer, dl, device, "FPT")
        vae_recon_results = evaluate_reconstruction(vae_tokenizer, dl, device, "VAE")
        
        fpt_results['reconstruction'] = fpt_recon_results
        vae_results['reconstruction'] = vae_recon_results
        
        print(f"\nReconstruction Comparison:")
        print(f"FPT PSNR: {fpt_recon_results['psnr']:.2f} dB")
        print(f"VAE PSNR: {vae_recon_results['psnr']:.2f} dB")
    
    # Probe evaluation removed as requested
    
    # 5. Create comprehensive visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    create_comprehensive_visualization(fpt_results, vae_results, args.output_dir)
    
    # 6. Generate final report
    print("\n" + "="*60)
    print("FINAL EVALUATION REPORT")
    print("="*60)
    
    # Create summary results
    summary_results = {}
    if 'future_pred' in fpt_results:
        summary_results['FPT'] = {
            'future_pred_mse': fpt_results['future_pred']['mse'],
            'reconstruction_psnr': fpt_results.get('reconstruction', {}).get('psnr', 0),
            'latent_diversity': fpt_results.get('latent_quality', {}).get('std', 0),
            'temporal_dynamics': fpt_results.get('latent_quality', {}).get('temporal_var', 0)
        }
    
    if 'future_pred' in vae_results:
        summary_results['VAE'] = {
            'future_pred_mse': vae_results['future_pred']['mse'],
            'reconstruction_psnr': vae_results.get('reconstruction', {}).get('psnr', 0),
            'latent_diversity': vae_results.get('latent_quality', {}).get('std', 0),
            'temporal_dynamics': vae_results.get('latent_quality', {}).get('temporal_var', 0)
        }
    
    # Create and save report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    create_evaluation_report(summary_results, report_path)
    
    # Print summary
    print("\n=== SUMMARY ===")
    if 'future_pred' in fpt_results and 'future_pred' in vae_results:
        fpt_mse = fpt_results['future_pred']['mse']
        vae_mse = vae_results['future_pred']['mse']
        if vae_mse > 0:
            improvement = ((vae_mse - fpt_mse) / vae_mse * 100)
            print(f"🎯 Future Prediction Improvement: {improvement:.1f}%")
    
    if 'reconstruction' in fpt_results and 'reconstruction' in vae_results:
        fpt_psnr = fpt_results['reconstruction']['psnr']
        vae_psnr = vae_results['reconstruction']['psnr']
        print(f"📊 FPT Reconstruction PSNR: {fpt_psnr:.2f} dB")
        print(f"📊 VAE Reconstruction PSNR: {vae_psnr:.2f} dB")
    
    print(f"📁 All results saved to: {args.output_dir}/")
    print("✅ Evaluation completed!")

if __name__ == "__main__":
    main()
