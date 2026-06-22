import os
import torch
import json
import dataclasses
from typing import Tuple
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    dataset_path: str = ""
    batch_size: int = 48
    context_length: int = 512
    ratio: float = 0.9


@dataclass
class OptimizerConfig:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-3


@dataclass
class ModelConfig:
    vocab_size: int = 2000
    num_layers: int = 12
    num_heads: int = 6
    d_ff: int = 1024
    d_model: int = 384  # d_model/num_layers is usually around 100
    rope_theta: float = 10000.0
    context_length: int = 512
    device: str = "cpu"


@dataclass
class TrainConfig:
    epochs: int = 10
    checkpoint_path: str = "."
    checkpoint_cadence: int = 1
    validation_cadence: int = 3
    grad_max_l2_norm: float = 1.0
    grad_clip_eps: float = 1e-6
    use_lr_cos_schedule: bool = False
    lr_max: float = 3e-4
    lr_min: float = 1e-5
    lr_schedule_T_w: int = 1024


def save_model_config(
    train_config: TrainConfig,
    opt_config: OptimizerConfig,
    dl_config: DataLoaderConfig,
    model_config: ModelConfig,
    output_path: str,
):
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        configs = {
            "train_config": dataclasses.asdict(train_config),
            "opt_config": dataclasses.asdict(opt_config),
            "dl_config": dataclasses.asdict(dl_config),
            "model_config": dataclasses.asdict(model_config),
        }
        json.dump(configs, f)


def load_model_config(file: str):

    with open(file, "r", encoding="utf-8") as f:
        configs = json.load(f)
        return (
            TrainConfig(**configs["train_config"]),
            OptimizerConfig(**configs["opt_config"]),
            DataLoaderConfig(**configs["dl_config"]),
            ModelConfig(**configs["model_config"]),
        )
