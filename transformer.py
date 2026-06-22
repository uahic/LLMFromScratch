# from __future__ import annotations

import sys
import os
import torch

from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int

sys.path.insert(0, os.path.dirname(__file__))

from cs336_basics.linear import SwiGLULinear, Linear
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.rms import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.activation_fcn import softmax


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.linear = SwiGLULinear(d_model, d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_1 = self.rms_norm_1(x)
        y = x + self.mha(norm_1)
        norm_2 = self.rms_norm_2(y)
        y_2 = y + self.linear(norm_2)
        return y + y_2


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.Sequential(
            *[
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    theta,
                    context_length,
                    dtype=dtype,
                    device=device,
                )
                for i in range(num_layers)
            ]
        )
        self.out_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.out_embedding = Linear(d_model, vocab_size, device=device, dtype=dtype)
        # input to out_embedding: [batch, seq_len, d_model]
        # output out_embedding : d_model -> vocab_size, so: [batch, seq_len, vocab_size]
        # means => a next-token prediction for each position in the input sequence is computed

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        x = self.layers(x)
        x = self.out_norm(x)
        x = self.out_embedding(x)
        return x


if __name__ == "__main__":
    pass
