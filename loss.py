# from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch

from torch import Tensor
from jaxtyping import Bool, Float, Int


def cross_entropy_loss(
    logits: Float[Tensor, " batch_size seq_len vocab_size"],
    targets: Int[Tensor, " batch_size seq_len"],
) -> float:

    max_vals, _ = torch.max(logits, dim=-1)
    max_vals_expanded = max_vals.unsqueeze(-1).expand(*max_vals.shape, logits.shape[-1])

    # Derivation (subtract max_vals for numerical stability):
    # m = max(logits)
    # log(\sum_{i} exp(a_{i}-m)) = log(\sum_{i} exp(a_{i})*exp(-m)) = log(exp(-m) * \sum_{i} exp(a_{i}))
    # -m + log(\sum_{i} exp(a_{i}))
    # <=> log(\sum_{i} exp(a_{i})) = log(\sum_{i} exp(a_{i}-m)) + m
    denom = (
        torch.log(torch.sum(torch.exp(logits - max_vals_expanded), dim=-1)) + max_vals
    )

    # row_idx = torch.arange(logits.shape[0])
    # -log(p(x_{i+1}|x_{1:i})) = -log(exp(o_{i}[x_{i+1}])) + log(\sum^{vocab_size}_{a=1} exp(o_{i}[a]))
    # = -o_{i}[x_{i+1}] + log(\sum^{vocab_size}_{a=1} exp(o_{i}[a]))

    # logits.shape = [32,500,2000]
    # targets.shape = [32,500]
    # denom.shape= [32,500]
    nom = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    values = -(nom - denom)
    loss_val = torch.mean(values)

    return loss_val
