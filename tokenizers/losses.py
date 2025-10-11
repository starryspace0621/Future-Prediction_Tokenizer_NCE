
import torch, torch.nn as nn, torch.nn.functional as F

class LPIPSLike(nn.Module):
    def __init__(self, weight=0.0):
        super().__init__(); self.weight = weight
        try:
            from torchvision.models import vgg16, VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.eval()
            for p in self.vgg.parameters(): p.requires_grad = False
            self.enabled = weight > 0.0
        except Exception:
            self.vgg = None; self.enabled = False
    def forward(self, x, y):
        if not self.enabled or self.vgg is None or self.weight <= 0.0:
            return x.new_tensor(0.0)
        def prep(t):
            t = (t + 1)/2
            return F.interpolate(t, size=(224,224), mode="bilinear", align_corners=False)
        x = prep(x); y = prep(y)
        loss = 0.0; feat_layers = [3,8,15,22]
        cur_x, cur_y = x, y
        for i, layer in enumerate(self.vgg):
            cur_x = layer(cur_x); cur_y = layer(cur_y)
            if i in feat_layers: loss = loss + F.l1_loss(cur_x, cur_y)
        return loss * self.weight

class InfoNCE(nn.Module):
    def __init__(self, temp: float = 0.07):
        super().__init__(); self.temp = temp
    def forward(self, q, k):
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        logits = q @ k.t()
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits / self.temp, labels)

class VQCodebook(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commit_weight: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1.0/codebook_size, 1.0/codebook_size)
        self.commit_weight = commit_weight
    def forward(self, z_e):
        shape = z_e.shape; D = shape[1]
        z_flat = z_e.reshape(z_e.size(0), D, -1).permute(0,2,1).reshape(-1, D)
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        codes = torch.argmin(dist, dim=1)
        z_q = self.codebook(codes).view(z_e.size(0), -1, D).permute(0,2,1).reshape(shape)
        commit = self.commit_weight * ((z_e.detach() - z_q)**2).mean()
        codebk = ((z_e - z_q.detach())**2).mean()
        z_q = z_e + (z_q - z_e).detach()
        return z_q, (commit + codebk), codes.view(z_e.size(0), -1)

def codebook_usage(codes, codebook_size: int):
    with torch.no_grad():
        flat = codes.reshape(-1)
        hist = torch.bincount(flat, minlength=codebook_size).float()
        usage = (hist > 0).float().mean().item()
    return usage, hist
