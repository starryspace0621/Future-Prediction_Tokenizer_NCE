
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder2D(nn.Module):
    def __init__(self, in_channels=3, z_dim=16):
        super().__init__()
        C = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, C, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C*4, C*8, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.mu = nn.Conv2d(C*8, z_dim, 3, 1, 1)
        self.logvar = nn.Conv2d(C*8, z_dim, 3, 1, 1)
    def forward(self, x):
        h = self.conv(x)
        mu = self.mu(h); logvar = self.logvar(h)
        return mu, logvar

class Decoder2D(nn.Module):
    def __init__(self, out_channels=3, z_dim=16):
        super().__init__()
        C = 64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, C*8, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*8, C*4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*4, C*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*2, C, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, out_channels, 3, 1, 1),
        )
    def forward(self, z): return torch.tanh(self.net(z))

class VAE(nn.Module):
    def __init__(self, z_dim=16, beta_kl=0.25):
        super().__init__()
        self.enc = Encoder2D(3, z_dim); self.dec = Decoder2D(3, z_dim)
        self.beta_kl = beta_kl
    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp(); eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        B, C, T, H, W = x.shape
        x0 = x[:,:,T//2]
        mu, logvar = self.enc(x0)
        z = self.reparam(mu, logvar)
        xr = self.dec(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return xr, {"kl": kl, "z": z}
    def encode(self, x):
        B, C, T, H, W = x.shape
        x0 = x[:,:,T//2]
        mu, logvar = self.enc(x0)
        z = self.reparam(mu, logvar)
        return z
    def decode(self, z): return self.dec(z)
