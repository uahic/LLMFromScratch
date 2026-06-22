from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import math
from einops import einsum, rearrange
from torch import Tensor
from jaxtyping import Bool, Float, Int

from cs336_basics.activation_fcn import softmax
from cs336_basics.linear import init_params
from cs336_basics.embedding import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Float[Tensor, " batch_size ... seq_len d_k"],
    K: Float[Tensor, " batch_size ... seq_len d_k"],
    V: Float[Tensor, " batch_size ... seq_len d_v"],
    mask: Bool[Tensor, " seq_len seq_len"] | None = None,
) -> Float[Tensor, " batch_size ... seq_len d_v"]:
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]

    # (b,...,seq_len,d_k)x(b,...,seq_len,d_k) -> (b,...,seq_len,seq_len)
    # Flops: 2*b*...*d_k*seq_len*seq_len
    pre_softmax = einsum(
        Q,
        K,
        "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k",
    ) / math.sqrt(d_k)

    if mask is not None:
        pre_softmax = pre_softmax + mask
    scores = softmax(pre_softmax)

    # (b,...,seq_len, seq_len)x(b,...,seq_len,d_v) -> (b,...,seq_len,d_v)
    # Flops: 2*b*...*seq_len*seq_len*d_v
    scaled_values = einsum(
        scores,
        V,
        " batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v",
    )
    return scaled_values


def generate_causal_mask(
    max_seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    indices = torch.arange(
        max_seq_len,
        device=device,
    )
    i = indices.unsqueeze(1)  # (N,1)
    j = indices.unsqueeze(0)  # (1,N)
    mask = torch.where(
        j <= i,
        0.0,
        float("-inf"),
    )  # the best we can do without custom kernel, probably
    return mask


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        mask_factory: any = generate_causal_mask,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert mask_factory is not None, "You must provide a mask_factory function"
        assert d_model % num_heads == 0, "num_heads must be a divisor of d_model"
        d_k = d_v = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = torch.nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device)
        )
        self.W_k = torch.nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device)
        )
        self.W_v = torch.nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device)
        )
        self.W_o = torch.nn.Parameter(
            torch.empty(d_model, d_model, dtype=dtype, device=device)
        )
        init_params(d_model, d_model, self.W_q)
        init_params(d_model, d_model, self.W_k)
        init_params(d_model, d_model, self.W_v)
        init_params(d_model, d_model, self.W_o)

        self.embedding = None
        self.mask_factory = generate_causal_mask

        if max_seq_len is not None:
            mask = mask_factory(max_seq_len, device=device)
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

        if theta is not None and max_seq_len is not None:
            self.embedding = RotaryPositionalEmbedding(
                theta, d_k, max_seq_len, device=device
            )

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> torch.Tensor:

        # Q: Float[Tensor, " batch_size ... seq_len d_k"],
        # K: Float[Tensor, " batch_size ... seq_len d_k"],
        # V: Float[Tensor, " batch_size ... seq_len d_v"],
        seq_len = x.shape[-2]

        W_q = rearrange(
            self.W_q,
            " (num_heads d_k) d_model -> num_heads d_k d_model",
            num_heads=self.num_heads,
        )

        # (...,seq_len,d_model)x(d_model,(num_headsxd_k)=d_model) -> (... num_heads, seq_len, d_k)
        # Flops: 2*b*...*seq_len*d_model*d_model
        Q = einsum(
            x,
            W_q,
            " ... seq_len d_model, num_heads d_k d_model -> ... num_heads seq_len d_k",
        )

        W_k = rearrange(
            self.W_k,
            " (num_heads d_k) d_model -> num_heads d_k d_model",
            num_heads=self.num_heads,
        )

        # Flops: 2*b*...*seq_len*d_model*d_model
        K = einsum(
            x,
            W_k,
            " ... seq_len d_model, num_heads d_k d_model -> ... num_heads seq_len d_k",
        )

        W_v = rearrange(
            self.W_v,
            " (num_heads d_v) d_model -> num_heads d_v d_model",
            num_heads=self.num_heads,
        )

        # Flops: 2*b*...*seq_len*d_model*d_model
        V = einsum(
            x,
            W_v,
            " ... seq_len d_model, num_heads d_v d_model -> ... num_heads seq_len d_v",
        )

        # Q,K: (batch, n_heads, seq_len, d_k)
        # V  : (batch, n_heads, seq_len, d_v)

        if self.embedding is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=Q.device)
                positions = positions.unsqueeze(0).expand((self.num_heads, seq_len))

            # Flops: 3*b*...*seq_len*d_k
            Q_emb = self.embedding(Q, positions)
            # Flops: 3*b*...*seq_len*d_k
            K_emb = self.embedding(K, positions)
        else:
            Q_emb = Q
            K_emb = K

        if self.mask is not None:
            mask = self.mask[:seq_len, :seq_len]
        else:
            mask = self.mask_factory(seq_len, device=Q.device)

        # Flops: 4*b*...*seq_len*seq_len*d_k
        V_tilde = scaled_dot_product_attention(
            Q_emb, K_emb, V, mask=mask
        )  # (2,8,10,4), (batch,,seq,n_heads)

        # "Concat" through rearranging the (n_heads d_k) = d_model dimension
        V_tilde = rearrange(
            V_tilde, " ... n_heads seq_len d_k -> ... seq_len (n_heads d_k)"
        )

        # (...,seq_len,d_model)x(d_model,d_model) -> (...,seq_len,d_model)
        # Flops: 2*b*...*seq_len*d_model*d_model
        out = einsum(
            V_tilde,
            self.W_o,
            " ... seq_len d_model_in, d_model_out d_model_in -> ... seq_len d_model_out",
        )

        return out


# Total Flops:
# 6*b*...*seq_len*d_k # rot. embedding
# 4*b*...*seq_len*seq_len*d_k # scaled_dot_product_attn
# 8*b*...*seq_len*d_model*d_model # vec-mat einsum

if __name__ == "__main__":
    Q = torch.rand(2, 3, 5)
    K = torch.rand(2, 3, 5)
    V = torch.rand(2, 3, 5)

    attn_vals = scaled_dot_product_attention(Q, K, V)
    print(attn_vals)

    batch_size = 2
    n_heads = 8
    d_model = n_heads * 4
    seq_len = 10
    max_seq_len = 12
    theta = 0.5

    multi_head_attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, theta)
    x = torch.rand(batch_size, seq_len, d_model)
    out = multi_head_attn(x)
    print(out)
    print(out.shape)
