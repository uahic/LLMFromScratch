from __future__ import annotations

from jaxtyping import Bool, Float, Int

import torch
import math


def init_params(d_in, d_out, weights):
    var = 2 / (d_in + d_out)
    std = math.sqrt(var)
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3 * std, b=3 * std)


class Linear(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(d_out, d_in, dtype=dtype, device=device)
        )
        init_params(d_in, d_out, self.weights)

    def forward(
        self, x: Float[Tensor, " batch_size ... seq_len d_in"]
    ) -> Float[Tensor, " batch_size ... seq_len d_out"]:
        # Row-Major, self.weights in math notation would be W.T
        # (b,...,seq_len,d_in)x(d_in,d_out) -> Flops: b*...*seq_len*d_in*d_out
        return torch.matmul(x, self.weights.T)


class SwiGLULinear(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not d_ff:
            d_ff = (8 / 3) * d_in
            d_ff = int(max(d_ff // 64, 1)) * 64
        self.w1 = torch.nn.Parameter(
            torch.empty(d_ff, d_in, dtype=dtype, device=device)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(d_ff, d_in, dtype=dtype, device=device)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(d_out, d_ff, dtype=dtype, device=device)
        )

        self._init_params(self.w1, d_ff, d_in)
        self._init_params(self.w3, d_ff, d_in)
        self._init_params(self.w2, d_out, d_ff)

    def _init_params(self, w: torch.Tensor, d_out: int, d_in: int):
        var = 2 / (d_in + d_out)
        std = math.sqrt(var)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self, x: Float[Tensor, " batch_size ... seq_len d_in"]
    ) -> Float[Tensor, " batch_size ... seq_len d_out"]:
        # (b,...,seq_len,d_in)x(d_in,d_ff) -> (b,...,seq_len,dff); Flops: b*...*seq_len*d_in*d_ff
        w1_x = torch.matmul(x, self.w1.T)

        # Flops: b*...*seq_len*dff
        silu = torch.mul(w1_x, torch.nn.functional.sigmoid(w1_x))

        # (b,...,seq_len,d_in)x(d_in,d_ff) -> (b,...,seq_len,dff); Flops: b*...*seq_len*d_in*d_ff
        w3_x = torch.matmul(x, self.w3.T)

        # Flops: b*...*seq_len*d_in
        silu_gated = torch.mul(silu, w3_x)
        # (b,...,seq_len,d_ff)x(d_ff,d_out) -> (b,...,seq_len,d_out); Flops: b*...*seq_len*d_ff*d_out
        return torch.matmul(silu_gated, self.w2.T)


if __name__ == "__main__":
    lin = Linear(3, 5, dtype=float)

    x = torch.tensor([3, 2, 1], dtype=float)
    print(lin.forward(x))
