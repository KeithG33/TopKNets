import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed

import torchsort

from .differentiable_topk import DifferentiableTopK


class DynamicTokenBlock(nn.Module):
    def __init__(self, dim, seq_len, k, expand=1):
        super().__init__()
        self.k  = k
        self.tau     = nn.Parameter(torch.tensor(0.0))
        self.D       = int(dim * expand)

        self.topk = DifferentiableTopK(k)

        self.tok_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.D),
            nn.GELU(),
        )
        self.ch_proj = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, self.D+1),
        )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, dim),
        )

    def forward(self, x):
        x_tok = self.tok_proj(x)  # (B, N, D)
        x_ch = self.ch_proj(x_tok.transpose(1,2)).transpose(1,2)  # (B, N, D)
        x_score, x_proj = self.proj(x_tok + x_ch).split([1, self.D], dim=-1)  # (B, N, 1), (B, N, D)
        tau = self.tau.clamp(-2.0, 5.0)
        W = self.topk(x_score.squeeze(-1), self.k, torch.sigmoid(tau))  # (B, k, N)
        Z = torch.matmul(W, x_proj) # (B, k, D)
        Y = torch.matmul(W.transpose(1,2), Z)  # (B, N, D)
        Y = self.proj_out(x_proj + Y)
        return Y
    

class TopKModel(nn.Module):
    def __init__(self, dim, k, num_classes=1000, img_size=224):
        super().__init__()
        self.k = k
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=dim)
        seq_len = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.ones(1, seq_len, dim))
        
        self.blocks = nn.ModuleList(
            [
                DynamicTokenBlock(dim, seq_len, k) for _ in range(12)
            ]
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim*4),
                    nn.GELU(),
                    nn.Linear(dim*4, dim)
                ) for _ in range(12)
            ]
        )

        pred_dim = dim * 8
        self.pred_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk, mlp in zip(self.blocks, self.mlps):
            x = x + blk(x)
            x = x + mlp(x)
        
        x = x.mean(dim=1)
        x = self.pred_head(x)
        return x
