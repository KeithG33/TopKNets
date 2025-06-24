import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import PatchEmbed

from .differentiable_topk import DifferentiableTopK


class TopKAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(dim)))
        self.topk = DifferentiableTopK(k)

        self.qkv_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3 * dim))
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.out_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [D, D, D], dim=-1)
        s = (q * k).sum(dim=-1) * self.scale

        tau = self.tau.clamp(-1.4, 5.0)
        W = self.topk(s.squeeze(-1), torch.sigmoid(tau))
        Z = torch.matmul(W, v)
        Y = torch.matmul(W.transpose(1, 2), Z)

        Y = self.out_proj(Y + v)
        return Y


class TopKAttentionModel(nn.Module):
    def __init__(self, dim: int, k: int, num_classes: int = 1000, img_size: int = 224, num_layers: int = 12) -> None:
        super().__init__()
        self.k = k
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=dim)
        seq_len = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.ones(1, seq_len, dim))

        self.blocks = nn.ModuleList([TopKAttention(dim, seq_len, k) for _ in range(num_layers)])
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
