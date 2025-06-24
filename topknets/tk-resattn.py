import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed

from .differentiable_topk import DifferentiableTopK


class DynamicTokenBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, k: int, heads: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.k = k
        self.temp = nn.Parameter(torch.tensor(0.0))
        self.D = int(dim * expand)
        self.H = heads
        self.Dh = self.D // self.H

        self.topk = DifferentiableTopK(k)

        self.ch_proj = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
        )
        self.tok_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.D),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(self.Dh),
            nn.Linear(self.Dh, self.Dh + 1),
        )
        self.proj_out = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, dim),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        x = self.tok_proj(x).view(B, N, self.H, self.Dh).transpose(1, 2)
        x = x + self.ch_proj(x.transpose(2, 3)).transpose(2, 3)

        x_score, x_val = self.proj(x).split([1, self.Dh], dim=-1)
        x_score, x_val = x_score.view(B * self.H, N), x_val.view(B * self.H, N, self.Dh)

        temp = self.temp.clamp(-2, 5.0).sigmoid()
        x_weights = self.topk(x_score, temp) # (B*H, k, N)

        x_gather = torch.matmul(x_weights, x_val) # (B*H, k, Dh)
        x_scatter = torch.matmul(x_val, x_gather.transpose(1, 2)) # (B*H, N, k)
        x_out = torch.matmul(x_scatter.softmax(dim=-1), x_gather) # (B*H, N, Dh)

        x_out = x_out.view(B, self.H, N, self.Dh).transpose(1, 2)
        x_out = self.proj_out(x_out.reshape(B, N, self.D))
        return x_out


class TopKDynamicModel(nn.Module):
    def __init__(self, dim: int, k: int, num_classes: int = 1000, img_size: int = 224, num_layers: int = 12) -> None:
        super().__init__()
        self.k = k
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=dim)
        seq_len = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.ones(1, seq_len, dim))

        self.blocks = nn.ModuleList([DynamicTokenBlock(dim, seq_len, k) for _ in range(num_layers)])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                )
                for _ in range(num_layers)
            ]
        )

        pred_dim = dim * 8
        self.pred_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, pred_dim),
            nn.GELU(),
            nn.Linear(pred_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        for blk, mlp in zip(self.blocks, self.mlps):
            x = x + blk(x)
            x = x + mlp(x)

        x = x.mean(dim=1)
        x = self.pred_head(x)
        return x
