import einops
import torch
from torch import nn
from diffusers.models import AutoencoderKL
from pathlib import Path


class VAE(nn.Module):
    def __init__(self, cache_dir: str = "./models"):
        super().__init__()
        # 创建本地缓存目录
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            subfolder="vae",
            cache_dir=str(cache_path)
        )
        self.vae.eval().requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1

        with torch.no_grad():
            z = self.vae.encode(x_in).latent_dist.sample()

        z = z * self.vae.config.scaling_factor
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = z.shape
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")
        z_in = z_in / self.vae.config.scaling_factor

        with torch.no_grad():
            x = self.vae.decode(z_in, return_dict=False)[0]

        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x
