#!/usr/bin/env python3
"""
Comprehensive video dataset setup script for Gymnasium environments
"""

import os
import argparse
import numpy as np
import yaml
from PIL import Image
import gymnasium as gym

def generate_trajectory(env_name, max_steps=100, output_dir="data/video", resolution=64):
    """Generate a single trajectory from a Gymnasium environment"""
    env = gym.make(env_name)
    
    # Reset with random seed for variety
    obs, _ = env.reset(seed=np.random.randint(0, 10000))
    
    frames = []
    actions = []
    
    for step in range(max_steps):
        # Simple action selection for different environments
        if "CartPole" in env_name:
            # CartPole: 30% chance to take action, encourage more dynamic trajectories
            action = 1 if np.random.random() < 0.3 else 0
        elif "MountainCar" in env_name:
            # MountainCar: random actions
            action = env.action_space.sample()
        else:
            # Default: random actions
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and save frame
        frame = env.render()
        if frame is not None:
            # Convert to PIL Image and resize
            if isinstance(frame, np.ndarray):
                img = Image.fromarray(frame)
            else:
                img = frame
            img = img.resize((resolution, resolution))
            frames.append(img)
            actions.append(action)
        
        if terminated or truncated:
            break
    
    env.close()
    
    if len(frames) > 0:
        # Create trajectory directory
        traj_id = f"traj_{len(os.listdir(output_dir)):04d}"
        traj_dir = os.path.join(output_dir, traj_id)
        os.makedirs(traj_dir, exist_ok=True)
        
        # Save frames
        frames_dir = os.path.join(traj_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame.save(os.path.join(frames_dir, f"{i:06d}.jpg"))
        
        # Save actions
        np.save(os.path.join(traj_dir, "actions.npy"), np.array(actions))
        
        print(f"Generated trajectory {traj_id} with {len(frames)} frames")
        return True
    
    return False

def setup_dataset(env_name="CartPole-v1", num_trajectories=100, 
                 max_steps=100, output_dir="data/video", resolution=64,
                 train_ratio=0.8):
    """Setup complete dataset with train/val splits"""
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Generate trajectories
    successful_trajs = 0
    total_attempts = 0
    
    while successful_trajs < num_trajectories and total_attempts < num_trajectories * 2:
        total_attempts += 1
        
        # Determine split
        if successful_trajs < int(num_trajectories * train_ratio):
            split_dir = train_dir
        else:
            split_dir = val_dir
        
        if generate_trajectory(env_name, max_steps, split_dir, resolution):
            successful_trajs += 1
    
    print(f"Generated {successful_trajs} successful trajectories out of {total_attempts} attempts")
    
    # Save dataset config
    config = {
        'env_name': env_name,
        'num_trajectories': successful_trajs,
        'max_steps': max_steps,
        'resolution': resolution,
        'clip_len': 16,  # Default clip length
        'train_ratio': train_ratio
    }
    
    config_path = os.path.join(output_dir, "dataset_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved: {config_path}")
    
    # Update existing config files
    config_files = [
        "configs/fpt_64.yaml",
        "configs/vae_64.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    cfg = yaml.safe_load(f)
                
                # Update dataset paths
                cfg['dataset']['root'] = os.path.abspath(output_dir)
                cfg['dataset']['name'] = f"Gymnasium_{env_name}"
                
                with open(config_file, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                
                print(f"Updated config: {config_file}")
            except Exception as e:
                print(f"Failed to update {config_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup Gymnasium video dataset")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                       help="Gymnasium environment name")
    parser.add_argument("--num-trajs", type=int, default=100,
                       help="Number of trajectories to generate")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per trajectory")
    parser.add_argument("--output-dir", type=str, default="data/video",
                       help="Output directory for dataset")
    parser.add_argument("--resolution", type=int, default=64,
                       help="Frame resolution")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of trajectories for training")
    
    args = parser.parse_args()
    
    print(f"Setting up dataset for {args.env}")
    print(f"Output directory: {args.output_dir}")
    
    setup_dataset(
        env_name=args.env,
        num_trajectories=args.num_trajs,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_ratio=args.train_ratio
    )
    
    print("Dataset setup complete!")

if __name__ == "__main__":
    main()
