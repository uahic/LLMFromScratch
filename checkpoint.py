from __future__ import annotations

import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))

import torch
import typing

# import math
import numpy as np
import numpy.typing as npt
from pathlib import Path


def find_most_recent_checkpoint(path: str) -> str | None:
    files = sorted(
        Path(path).glob("epoch-[0-9]*.pth"),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
        # index the numbers based on the numbers alone, instead of lexicographic sorting
    )
    if len(files) == 0:
        return None
    return files[-1]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    loss=None,
):
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()
    obj = {
        "model_state": model_state,
        "optim_state": optim_state,
        "iteration": iteration,
    }
    # if loss is not None:
    #     obj |= {"loss": loss}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    print(f"Loading checkpoint... {src}")
    obj = torch.load(src)
    model.load_state_dict(obj["model_state"])
    optimizer.load_state_dict(obj["optim_state"])
    # if "loss" in obj:
    #     return obj["iteration"], loss
    return obj["iteration"]
