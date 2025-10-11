import torch, torch.nn as nn, torch.nn.functional as F
from .losses import VQCodebook, InfoNCE, codebook_usage

class Encoder2D(nn.Module):
    def __init__(self, in_channels=3, d=16):
        super().__init__()
        C = 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, C, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C*2, C*4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C*4, d, 3, 1, 1),
        )
    def forward(self, x): return self.conv(x)

class Decoder2D(nn.Module):
    def __init__(self, out_channels=3, d=16):
        super().__init__()
        C = 64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d, C*4, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*4, C*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C*2, C, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C, out_channels, 3, 1, 1),
        )
    def forward(self, z): return torch.tanh(self.net(z))

class FPT_RVQ(nn.Module):
    def __init__(self, latent_dim=16, rvq_levels=2, codebook_size=256, commit_weight=0.25, temp=0.07, horizons=(1,2,4,8), future_pred_weight=1.0):
        super().__init__()
        self.enc = Encoder2D(3, latent_dim)
        self.dec = Decoder2D(3, latent_dim)
        self.levels = nn.ModuleList([VQCodebook(codebook_size, latent_dim, commit_weight) for _ in range(rvq_levels)])
        self.proj_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.proj_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_head = nn.Sequential(nn.Linear(latent_dim+4, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.infoNCE = InfoNCE(temp=temp)
        self.horizons = horizons
        
        # Future prediction components
        self.future_pred_weight = future_pred_weight
        self.future_predictor = nn.Sequential(
            nn.Linear(latent_dim + 4, latent_dim * 2),  # current_latent + action
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)  # predict next latent
        )
        self.future_loss_fn = nn.MSELoss()

    def encode_frame(self, x):
        feat = self.enc(x); z = feat; vq_loss = 0.0; codes_all = []
        for lvl in self.levels:
            z_q, loss_vq, codes = lvl(z)
            vq_loss = vq_loss + loss_vq
            codes_all.append(codes)
            z = z - z_q.detach() + z_q
        return z, vq_loss, codes_all

    def decode(self, z): return self.dec(z)

    def forward(self, video, actions, recon_weight=0.1):
        if actions.dim() == 2: actions = actions.unsqueeze(0).expand(video.size(0), -1, -1)
        B, C, T, H, W = video.shape
        t0 = T//2
        
        # Encode current frame
        x0 = video[:,:,t0]
        z0, vq_loss, codes_all = self.encode_frame(x0)
        xr = self.decode(z0)
        rec_loss = torch.nn.functional.l1_loss(xr, x0)
        
        # Original InfoNCE loss for temporal consistency
        nce_loss = 0.0; usage = []
        for codes in codes_all:
            u, _ = codebook_usage(codes, self.levels[0].codebook.num_embeddings)
            usage.append(u)
        for dlt in self.horizons:
            t1 = min(T-1, t0 + dlt)
            x1 = video[:,:,t1]
            z1, _, _ = self.encode_frame(x1)
            q = torch.nn.functional.adaptive_avg_pool2d(z0, 1).flatten(1)
            k = torch.nn.functional.adaptive_avg_pool2d(z1, 1).flatten(1)
            a = actions[:, t0]
            q = self.proj_q(q)
            k = self.proj_k(k).detach()
            q = self.pred_head(torch.cat([q, a], dim=-1))
            nce_loss = nce_loss + self.infoNCE(q, k)
        
        # NEW: Future prediction loss - learn p(s'|φ(s), a)
        future_loss = 0.0
        if T > 1:  # Need at least 2 frames for future prediction
            # Get current latent representation
            current_latent = torch.nn.functional.adaptive_avg_pool2d(z0, 1).flatten(1)
            current_action = actions[:, t0]  # [B, 4]
            
            # Predict next frame latent
            pred_input = torch.cat([current_latent, current_action], dim=-1)
            predicted_next_latent = self.future_predictor(pred_input)
            
            # Get actual next frame latent as target
            if t0 + 1 < T:
                x_next = video[:,:,t0+1]
                z_next, _, _ = self.encode_frame(x_next)
                target_next_latent = torch.nn.functional.adaptive_avg_pool2d(z_next, 1).flatten(1).detach()
                
                # Future prediction loss
                future_loss = self.future_loss_fn(predicted_next_latent, target_next_latent)
        
        # Combined loss with future prediction
        total_loss = nce_loss + recon_weight * rec_loss + vq_loss + self.future_pred_weight * future_loss
        
        aux = {
            'nce': float(nce_loss.item()),
            'rec': float(rec_loss.item()),
            'vq': float(vq_loss.item()),
            'future_pred': float(future_loss.item()),
            'code_usage_avg': float(sum(usage)/max(1,len(usage)))
        }
        return xr, total_loss, aux, z0

    def encode(self, video):
        B, C, T, H, W = video.shape
        t0 = T//2; x0 = video[:,:,t0]
        z0, _, _ = self.encode_frame(x0)
        return torch.nn.functional.adaptive_avg_pool2d(z0, 1).flatten(1)
