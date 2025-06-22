# TopK Networks

A simple collection of neural networks that use differentiable top‑k operations. This repository stores a few model variants for experimentation. The common top‑k routine is provided as the `DifferentiableTopK` class which can be reused across models.

## Repository Layout

- `topknets/` – Python package containing the model code.
- `plots/` – directory for figures or training curves (currently empty).

## Example Usage

```python
import torch
from topknets import TopKDynamicModel, TopKAttentionModel, DifferentiableTopK

model = TopKDynamicModel(dim=128, k=96)
input = torch.randn(2, 3, 224, 224)
output = model(input)
print(output.shape)

# direct use of the soft top-k routine
scores = torch.randn(4, 32)
topk = DifferentiableTopK(k=5)
weights = topk(scores)
```

## DifferentiableTopK

`DifferentiableTopK` converts a score vector into a **soft** top‑k. Given
`attn_weights` of shape `(B, K)` it returns weights of shape `(B, k, K)` where
each row sums to one.

The weights are computed from differentiable ranks via
`torchsort.soft_rank`. For rank `r_j` at position `i` we use
the smooth indicator

```
I(rank_j, i) = sigmoid((i + 0.5 - rank_j) / tau)
               - sigmoid((i - 0.5 - rank_j) / tau)
```

where `tau` controls how close the result is to a hard top‑k.

## Results

| Model                | Dataset | Top-1 | Top-5 |
|----------------------|---------|-------|-------|
| TopKDynamicModel     | -       | -     | -     |
| TopKAttentionModel   | -       | -     | -     |

_Add your scores to the table above as you run experiments._

Plots from experiments can be placed in the `plots/` directory and referenced here.
