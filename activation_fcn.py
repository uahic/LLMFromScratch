import torch


def softmax(x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
    # for numerical stability
    max_vals, max_idx = torch.max(x, dim=-1)
    max_vals_expanded = max_vals.unsqueeze(-1).expand(*max_vals.shape, x.shape[-1])
    x_c = x - max_vals_expanded
    if temperature is None:
        x_c = torch.exp(x_c)
    else:
        x_c = torch.exp(x_c / temperature)
    x_c = x_c / torch.sum(x_c, dim=-1, keepdim=True)
    return x_c
