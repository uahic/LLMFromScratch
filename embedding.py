import torch
from jaxtyping import Bool, Float, Int


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = torch.nn.Parameter(
            torch.empty(num_embeddings, d_model, dtype=dtype, device=device)
        )
        torch.nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        rot_cos, rot_sin = self._compute_thetas(theta, d_k, max_seq_len, device)
        self.register_buffer("rot_cos", rot_cos, persistent=False)
        self.register_buffer("rot_sin", rot_sin, persistent=False)
        self.max_seq_len = max_seq_len
        self.d_k = d_k

    def _compute_thetas(
        self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None
    ):
        assert d_k % 2 == 0, "d_k must be divisible by 2"
        k = torch.arange(0, d_k // 2, device=device)  # shape: (d_k/2, )
        theta_k = torch.ones_like(k, device=device) * theta  # shape (d_k/2,)
        theta_k = 1 / torch.pow(theta_k, ((2 * k)) / d_k)  # shape (d_k/2,)

        idx = torch.arange(0, max_seq_len, device=device).unsqueeze(
            1
        )  # shape (max_seq_len, 1)
        thetas = theta_k.unsqueeze(0) * idx  # shape (max_seq_len, d_k/2)

        # Repeat (interleaved)
        thetas = (
            thetas.unsqueeze(-1)
            .expand(*thetas.shape, 2)
            .reshape(*thetas.shape[:-1], -1)
        )  # shape (max_seq_len, max_seq_len)
        rot_cos = torch.cos(thetas)
        rot_sin = torch.sin(thetas)

        return rot_cos, rot_sin

    def forward(
        self,
        x: Float[
            torch.Tensor, " batch_size ... seq_len d_k"
        ],  # ... can be n_heads for example
        token_positions: Float[torch.Tensor, " batch_size ... seq_len"],
    ) -> torch.Tensor:

        # Reorder a flat array into 2 columns (col1: even, col2: odd numbers).
        # Then flip both columns and then flatten the array again
        # This results in shuffled entries [x2,x1,x4,x3,...] without copy
        # x_permuted = x.view(-1, 2).flip(-1).clone()

        # x.shape = [32, 6, 500, 64]
        x_permuted = x.reshape(-1, 2).flip(-1).clone()
        x_permuted[:, 0] *= -1.0
        x_permuted = x_permuted.view(x.shape)

        cos_sliced = self.rot_cos[token_positions]
        sin_sliced = self.rot_sin[token_positions]

        # Flops: b*...*seq_len*d_k
        x_cos = torch.mul(x, cos_sliced)
        # Flops: b*...*seq_len*d_k
        x_sin = torch.mul(x_permuted, sin_sliced)
        # Flops: b*...*seq_len*d_k
        return x_cos + x_sin


# Total Flops: 3*b*...*seq_len*d_k


if __name__ == "__main__":
    num_embeddings = 100
    d_model = 10
    emb = Embedding(num_embeddings, d_model)
    idx = torch.tensor([0, 3, 2], dtype=int)
    print(emb(idx))

    theta = 0.5
    d_k = 6
    max_seq_len = 10
    batch = 2

    x = torch.rand(batch, max_seq_len, d_k)
    rot_emb = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    token_positions = torch.arange(0, max_seq_len)
    rot_emb(x, token_positions)
