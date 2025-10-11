import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class VideoClips(Dataset):
    def __init__(self, root, split='train', clip_len=16, resolution=64):
        self.root = root
        self.split = split
        self.clip_len = clip_len
        self.resolution = resolution
        
        # Load dataset config
        config_path = os.path.join(root, 'dataset_config.yaml')
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'clip_len': clip_len, 'resolution': resolution}
        
        # Get trajectory directories
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        self.traj_dirs = sorted(glob.glob(os.path.join(split_dir, 'traj_*')))
        if not self.traj_dirs:
            raise FileNotFoundError(f"No trajectory directories found in {split_dir}")
        
        print(f"Found {len(self.traj_dirs)} trajectories in {split} split")
    
    def __len__(self):
        return len(self.traj_dirs)
    
    def __getitem__(self, idx):
        traj_dir = self.traj_dirs[idx]
        
        # Load actions
        actions_path = os.path.join(traj_dir, 'actions.npy')
        if not os.path.exists(actions_path):
            raise FileNotFoundError(f"Actions file not found: {actions_path}")
        
        actions = np.load(actions_path)  # [T, A]
        
        # Load frames
        frames_dir = os.path.join(traj_dir, 'frames')
        frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {frames_dir}")
        
        # Load and process frames
        frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert('RGB')
            img = img.resize((self.resolution, self.resolution))
            img_array = np.array(img).astype(np.float32) / 255.0
            # Normalize to [-1, 1]
            img_array = img_array * 2.0 - 1.0
            frames.append(img_array)
        
        frames = np.array(frames)  # [T, H, W, C]
        frames = np.transpose(frames, (0, 3, 1, 2))  # [T, C, H, W]
        
        # Ensure we have enough frames
        T = min(len(frames), len(actions))
        frames = frames[:T]
        actions = actions[:T]
        
        # Convert to torch tensors
        video = torch.from_numpy(frames).float()  # [T, C, H, W]
        actions = torch.from_numpy(actions).float()  # [T, A]
        
        return {
            'video': video,
            'actions': actions
        }

# For backward compatibility
BAIRClips = VideoClips
