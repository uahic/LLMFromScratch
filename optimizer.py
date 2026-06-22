from collections.abc import Callable, Iterable
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import math

from torch import Tensor
from jaxtyping import Bool, Float, Int


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state of param p
                t = state.get("t", 0)  # Get iteration number of state or default to 0
                grad = p.grad.data
                p.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=10e-8, weight_decay=1e-3
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p]  # Get state of param p
                t = state.get("t", 1)  # Get iteration number of state or default to 0

                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data

                # Adjust lr
                lr_t = lr * (
                    math.sqrt(1 - math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                )

                # Weight decay
                p.data -= lr * weight_decay * p.data

                # First and second moment update
                state["m"] = m = beta_1 * m + (1 - beta_1) * grad
                state["v"] = v = beta_2 * v + (1 - beta_2) * grad * grad

                # Final update to theta
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                state["t"] = t + 1
        return loss


def lr_cosine_schedule(
    t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int
) -> float:
    if t < T_w:
        return (t / T_w) * alpha_max
    if t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos(((t - T_w) * math.pi) / (T_c - T_w))) * (
            alpha_max - alpha_min
        )
    return alpha_min


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
) -> torch.Tensor:
    total_norm = 0
    params = []
    for param in parameters:
        if param.grad is None:
            continue
        # l2_norm = torch.linalg.norm(param.grad.data, ord=2)
        l2_norm = torch.linalg.vector_norm(param.grad.data, ord=2)
        total_norm += torch.pow(l2_norm, 2)
        params.append(param)

    if len(params) == 0:
        return

    total_norm = torch.sqrt(total_norm)
    max_l2_norm = torch.tensor([max_l2_norm], device=params[0].device)

    if total_norm < max_l2_norm:
        return

    for param in params:
        scale_factor = max_l2_norm / (total_norm + eps)
        param.grad.data *= scale_factor


if __name__ == "__main__":

    # def toy_training(lr: float):
    #     weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    #     opt = SGD([weights], lr=1)
    #     for t in range(10):
    #         opt.zero_grad()
    #         loss = (weights**2).mean()
    #         print(loss.cpu().item())
    #         loss.backward()
    #         opt.step()
    #     print(f"For {lr}: {loss.cpu().item()}")

    # toy_training(1e-1)
    # toy_training(1e-2)
    # toy_training(1e-3)

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights])
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
