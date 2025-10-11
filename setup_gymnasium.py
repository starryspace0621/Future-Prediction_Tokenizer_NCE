#!/usr/bin/env python3
"""
Quick setup script for Gymnasium dataset
"""

import os
import subprocess
import sys

def install_gymnasium():
    """Install gymnasium if not available"""
    try:
        import gymnasium as gym
        # Test if classic control environments work
        try:
            env = gym.make("CartPole-v1")
            env.close()
            print("✅ Gymnasium with classic control already installed")
            return True
        except Exception as e:
            print(f"📦 Installing Gymnasium with classic control support...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium[classic-control]"])
                print("✅ Gymnasium with classic control installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Gymnasium with classic control")
                return False
    except ImportError:
        print("📦 Installing Gymnasium...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium[classic-control]"])
            print("✅ Gymnasium installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Gymnasium")
            return False

def create_dataset():
    """Create Gymnasium dataset"""
    print("🎮 Creating Gymnasium dataset...")
    
    # Use CartPole as it's simple and generates good visual data
    cmd = [
        sys.executable, "data/video_setup.py",
        "--output-dir", "data/video",
        "--env", "CartPole-v1",
        "--num-trajs", "60",
        "--frames-per-traj", "64",
        "--max-size-gb", "2.0"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ Dataset created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create dataset")
        return False

def main():
    print("🚀 Quick Gymnasium Dataset Setup")
    print("=" * 40)
    
    # Install dependencies
    if not install_gymnasium():
        return
    
    # Create dataset
    if not create_dataset():
        return
    
    print("\n🎉 Setup complete!")
    print("📁 Dataset location: data/video")
    print("🚀 Ready to train:")
    print("   python train_tokenizer.py --config configs/fpt_64.yaml")
    print("   python train_tokenizer.py --config configs/vae_64.yaml")

if __name__ == "__main__":
    main()
