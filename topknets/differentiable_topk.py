import torch
import torch.nn as nn
import torchsort


class DifferentiableTopK(nn.Module):
    """Differentiable approximation of top-k selection."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        positions = torch.arange(k).view(1, 1, k)
        self.register_buffer("positions", positions)

    def forward(self, attn_weights: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Return soft top-k weights.

        Args:
            attn_weights: tensor of shape (B, K)
            tau: temperature controlling sharpness
        Returns:
            tensor of shape (B, k, K) with rows that sum to one
        """
        B, K = attn_weights.size()
        ranks = torchsort.soft_rank(-attn_weights, regularization_strength=tau)
        ranks = ranks.unsqueeze(-1)

        positions = self.positions.to(attn_weights.device, attn_weights.dtype)
        positions = positions.expand(B, 1, self.k)

        soft_indicator = torch.sigmoid((positions + 0.5 - ranks) / tau) - torch.sigmoid((positions - 0.5 - ranks) / tau)
        weights = soft_indicator.transpose(1, 2)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights
