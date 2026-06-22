import torch


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.frac = 1.0 / d_model
        self.gains = torch.nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared = torch.pow(x, 2)
        summand = self.frac * torch.sum(squared, dim=-1, keepdim=True)
        rms = torch.sqrt(summand + self.eps)
        x = torch.mul((x / rms), self.gains)
        return x.to(in_dtype)
