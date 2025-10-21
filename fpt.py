import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import List, Dict, Any, Optional


class Encoder2D(nn.Module):
    """2D CNN encoder for video frames"""
    def __init__(self, in_channels: int = 3, latent_dim: int = 16):
        super().__init__()
        C = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, C, 4, 2, 1),  # 1/2
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 4, 2, 1),  # 1/4
            nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 4, 2, 1),  # 1/8
            nn.ReLU(inplace=True),
            nn.Conv2d(C*4, latent_dim, 3, 1, 1),  # same size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Decoder2D(nn.Module):
    """2D CNN decoder for video frames"""
    def __init__(self, out_channels: int = 3, latent_dim: int = 16):
        super().__init__()
        C = 64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, C*4, 4, 2, 1),  # 2x
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*4, C*2, 4, 2, 1),  # 4x
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*2, C, 4, 2, 1),  # 8x
            nn.ReLU(inplace=True),
            nn.Conv2d(C, out_channels, 3, 1, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z))


class InfoNCE(nn.Module):
    """InfoNCE loss for contrastive learning"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, D] - predicted representations
            key: [B, D] - target representations
        """
        # Normalize to unit vectors
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        # Compute similarity matrix
        logits = query @ key.t() / self.temperature
        
        # Positive pairs are on the diagonal
        labels = torch.arange(query.size(0), device=query.device)
        
        return F.cross_entropy(logits, labels)


class LPIPSLike(nn.Module):
    """LPIPS-like perceptual loss using VGG features"""
    def __init__(self, weight: float = 0.0):
        super().__init__()
        self.weight = weight
        try:
            from torchvision.models import vgg16, VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.enabled = weight > 0.0
        except Exception:
            self.vgg = None
            self.enabled = False
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.vgg is None or self.weight <= 0.0:
            return x.new_tensor(0.0)
        
        def prep(t):
            t = (t + 1) / 2  # [-1, 1] -> [0, 1]
            return F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        
        x = prep(x)
        y = prep(y)
        
        loss = 0.0
        feat_layers = [3, 8, 15, 22]  # VGG feature layers
        cur_x, cur_y = x, y
        
        for i, layer in enumerate(self.vgg):
            cur_x = layer(cur_x)
            cur_y = layer(cur_y)
            if i in feat_layers:
                loss = loss + F.l1_loss(cur_x, cur_y)
        
        return loss * self.weight


class VAEInitializedEncoder(nn.Module):
    """VAE-initialized encoder"""
    def __init__(self, vae, latent_dim: int):
        super().__init__()
        self.vae = vae
        self.latent_dim = latent_dim
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - input image
        Returns:
            z: [B, latent_dim, h, w] - latent representation
        """
        with torch.no_grad():
            # Use VAE encoding
            z = self.vae.encode(x).latent_dist.sample()
            z = z * self.vae.config.scaling_factor
        
        # Add projection layer if latent dims don't match
        if z.shape[1] != self.latent_dim:
            if not hasattr(self, 'projection'):
                self.projection = nn.Conv2d(z.shape[1], self.latent_dim, 1).to(z.device)
            z = self.projection(z)
        
        return z


class VAEInitializedDecoder(nn.Module):
    """VAE-initialized decoder"""
    def __init__(self, vae, latent_dim: int):
        super().__init__()
        self.vae = vae
        self.latent_dim = latent_dim
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, h, w] - latent representation
        Returns:
            x: [B, C, H, W] - reconstructed image
        """
        # Add projection layer if latent dims don't match
        if z.shape[1] != self.vae.config.latent_channels:
            if not hasattr(self, 'projection'):
                self.projection = nn.Conv2d(self.latent_dim, self.vae.config.latent_channels, 1).to(z.device)
            z = self.projection(z)
        
        with torch.no_grad():
            # Use VAE decoding
            z = z / self.vae.config.scaling_factor
            x = self.vae.decode(z, return_dict=False)[0]
            x = (x + 1) / 2  # Normalize to [0, 1]
        
        return x


class FPT(nn.Module):
    """Future-Prediction Tokenizer with InfoNCE learning"""
    
    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 10,
        horizons: List[int] = [1, 2, 4, 8],
        temperature: float = 0.07,
        recon_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        use_vae_init: bool = False,
        vae_cache_dir: str = "./models",
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizons = horizons
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.use_vae_init = use_vae_init
        
        # Encoder and decoder
        if use_vae_init:
            # Use VAE initialization
            self.encoder, self.decoder = self._create_vae_initialized_encoder_decoder(
                latent_dim, vae_cache_dir
            )
        else:
            # Use random initialization
            self.encoder = Encoder2D(3, latent_dim)
            self.decoder = Decoder2D(3, latent_dim)
        
        # Prediction head: [φ(s), a] -> φ(s')
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Loss functions
        self.infoNCE = InfoNCE(temperature=temperature)
        self.lpips = LPIPSLike(weight=perceptual_weight)
    
    def _create_vae_initialized_encoder_decoder(self, latent_dim: int, vae_cache_dir: str):
        """Create VAE-initialized encoder and decoder"""
        from diffusers.models import AutoencoderKL
        from pathlib import Path
        
        # Load pre-trained VAE
        cache_path = Path(vae_cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="vae",
            cache_dir=str(cache_path)
        )
        vae.eval()
        
        # Create encoder and decoder sharing the same VAE instance
        encoder = VAEInitializedEncoder(vae, latent_dim)
        decoder = VAEInitializedDecoder(vae, latent_dim)
        
        return encoder, decoder
    
    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single frame to latent representation"""
        # x: [B, C, H, W]
        z = self.encoder(x)  # [B, latent_dim, h, w]
        # Global average pooling to get [B, latent_dim]
        z_pooled = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return z_pooled
    
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to latent representations
        Args:
            video: [B, T, H, W, C] or [B, C, T, H, W]
        Returns:
            latents: [B, T, latent_dim]
        """
        if video.dim() == 5 and video.shape[1] == 3:  # [B, C, T, H, W]
            video = einops.rearrange(video, "b c t h w -> b t h w c")
        
        B, T, H, W, C = video.shape
        latents = []
        
        for t in range(T):
            frame = video[:, t]  # [B, H, W, C]
            frame = einops.rearrange(frame, "b h w c -> b c h w")
            z = self.encode_frame(frame)  # [B, latent_dim]
            latents.append(z)
        
        return torch.stack(latents, dim=1)  # [B, T, latent_dim]
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to video frames
        Args:
            latents: [B, T, latent_dim] or [B, latent_dim, h, w]
        Returns:
            video: [B, T, H, W, C]
        """
        if latents.dim() == 3:  # [B, T, latent_dim]
            B, T, D = latents.shape
            # Reshape to spatial format for decoder - need to match encoder output size
            # Encoder reduces by 8x, so we need to expand to 8x8 spatial
            latents = latents.unsqueeze(-1).unsqueeze(-1)  # [B, T, D, 1, 1]
            latents = latents.expand(-1, -1, -1, 32, 32)  # [B, T, D, 32, 32] - 256/8 = 32
            latents = einops.rearrange(latents, "b t d h w -> (b t) d h w")
            
            # Decode
            frames = self.decoder(latents)  # [(B*T), C, H, W]
            frames = einops.rearrange(frames, "(b t) c h w -> b t h w c", b=B, t=T)
        else:  # [B, D, h, w]
            frames = self.decoder(latents)  # [B, C, H, W]
            frames = einops.rearrange(frames, "b c h w -> b h w c")
        
        return frames
    
    def forward(
        self, 
        video: torch.Tensor, 
        actions: torch.Tensor,
        return_losses: bool = True
    ) -> Dict[str, Any]:
        """
        Forward pass with InfoNCE learning
        Args:
            video: [B, T, H, W, C] - video frames
            actions: [B, T, action_dim] - robot actions
            return_losses: whether to compute losses
        """
        B, T, H, W, C = video.shape
        
        # Encode all frames
        latents = self.encode(video)  # [B, T, latent_dim]
        
        # Reconstruct current frame (middle frame)
        t0 = T // 2
        current_latent = latents[:, t0]  # [B, latent_dim]
        
        # Decode current frame for reconstruction loss
        current_frame = video[:, t0]  # [B, H, W, C]
        current_frame_tensor = einops.rearrange(current_frame, "b h w c -> b c h w")
        
        # Decode single frame - handle spatial dimensions correctly
        current_latent_spatial = current_latent.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        current_latent_spatial = current_latent_spatial.expand(-1, -1, 32, 32)  # [B, latent_dim, 32, 32]
        reconstructed = self.decoder(current_latent_spatial)  # [B, C, H, W]
        
        if not return_losses:
            return {
                'latents': latents,
                'reconstructed': reconstructed,
                'current_latent': current_latent
            }
        
        # InfoNCE loss for temporal consistency
        nce_loss = 0.0
        for horizon in self.horizons:
            t1 = min(T - 1, t0 + horizon)
            if t1 > t0:  # Valid future frame
                # Get current and future representations
                current_repr = current_latent  # [B, latent_dim]
                future_repr = latents[:, t1]  # [B, latent_dim]
                
                # Get current action
                current_action = actions[:, t0]  # [B, action_dim]
                
                # Predict future representation
                pred_input = torch.cat([current_repr, current_action], dim=-1)
                predicted_future = self.pred_head(pred_input)  # [B, latent_dim]
                
                # InfoNCE loss: predict future representation
                nce_loss += self.infoNCE(predicted_future, future_repr)
        
        nce_loss = nce_loss / len(self.horizons)
        
        # Reconstruction loss
        recon_loss = F.l1_loss(reconstructed, current_frame_tensor)
        
        # Perceptual loss (optional)
        perceptual_loss = self.lpips(reconstructed, current_frame_tensor)
        
        # Total loss
        total_loss = (
            nce_loss + 
            self.recon_weight * recon_loss + 
            self.perceptual_weight * perceptual_loss
        )
        
        return {
            'latents': latents,
            'reconstructed': reconstructed,
            'current_latent': current_latent,
            'losses': {
                'total': total_loss,
                'nce': nce_loss,
                'recon': recon_loss,
                'perceptual': perceptual_loss
            }
        }
    
    def predict_future(
        self, 
        current_frame: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict future representation given current frame and action
        Args:
            current_frame: [B, H, W, C] or [B, C, H, W]
            action: [B, action_dim]
        Returns:
            predicted_latent: [B, latent_dim]
        """
        if current_frame.dim() == 4 and current_frame.shape[-1] == 3:  # [B, H, W, C]
            current_frame = einops.rearrange(current_frame, "b h w c -> b c h w")
        
        # Encode current frame
        current_latent = self.encode_frame(current_frame)  # [B, latent_dim]
        
        # Predict future
        pred_input = torch.cat([current_latent, action], dim=-1)
        predicted_future = self.pred_head(pred_input)
        
        return predicted_future
