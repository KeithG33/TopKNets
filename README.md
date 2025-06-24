# TopK Networks

A collection of neural networks that take advantage of a differentiable top-K selection strategy. The common top‑k routine is provided as the `DifferentiableTopK` class which is reused across models.

```python
class DifferentiableTopK(nn.Module):
    """ Calculate differentiable top-k selection weights """

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
```

## DifferentiableTopK

`DifferentiableTopK` converts a score vector into a **soft** top‑k matrix. Given `attn_weights` of shape `(B, N)` it returns weights of shape `(B, k, D)` where each row sums to one.

The weights are computed from the differentiable ranks output by 
`torchsort.soft_rank`. For rank `r_j` at position `i` we use the smooth indicator

```
I(rank_j, i) = sigmoid((i + 0.5 - rank_j) / tau)
               - sigmoid((i - 0.5 - rank_j) / tau)
```

where `tau` controls how close the result is to a hard top‑k.

## Results

All models trained on ImageNet

| Model                | Dataset | Top-1 | Top-5 |
|----------------------|---------|-------|-------|
| TopKDynamicModel     | -       | -     | -     |
| TopKAttentionModel   | -       | -     | -     |

