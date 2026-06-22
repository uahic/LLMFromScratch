import dataclasses

import sys
import os
import signal
import types
import json
import hashlib
import time

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.dataloader import DataLoader
from cs336_basics.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    find_most_recent_checkpoint,
)
from cs336_basics.linear import Linear, SwiGLULinear
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.config import (
    TrainConfig,
    OptimizerConfig,
    ModelConfig,
    DataLoaderConfig,
    save_model_config,
    load_model_config,
)


def setup_sigint_handler(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    shared_namespace: types.SimpleNamespace,
    output_path: str,
):
    def handler(signum, frame):
        epoch = shared_namespace.epoch
        loss = shared_namespace.loss
        ckpth = os.path.join(output_path, "checkpoints", f"epoch-{epoch}.pth")
        try:
            save_checkpoint(model, optimizer, epoch, ckpth, loss=loss)
        except Exception as e:
            print(f"[WARN] Checkpoint save failed on interrupt: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def log_gradient_norms(model: torch.nn.Module, writer: SummaryWriter, step: int):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.detach().norm(2).item()
            writer.add_scalar(f"grad_norm/{name}", norm, step)
            total_norm += norm**2
    total_norm = total_norm**0.5
    writer.add_scalar("grad_norm/total", total_norm, step)

    # Exploding gradient alert
    if total_norm > 10.0:
        print(f"[WARNING] Exploding gradient at step {step}: norm={total_norm:.2f}")
    return total_norm


# Register forward hooks
def make_hook(name: str, writer: SummaryWriter, shared_namespace: any):
    def hook(module, input, output):
        t = shared_namespace.t
        if t % 100 == 0:
            writer.add_scalar(f"activations/{name}/mean", output.mean(), t)
            writer.add_scalar(f"activations/{name}/std", output.std(), t)
            writer.add_scalar(
                f"activations/{name}/frac_dead", (output == 0).float().mean(), t
            )

    return hook


def merge_config_dicts(
    train_config: TrainConfig,
    model_config: ModelConfig,
    optim_config: OptimizerConfig,
    loader_config: DataLoaderConfig,
):
    hash_dict = {}
    hash_dict |= dataclasses.asdict(train_config)
    hash_dict |= dataclasses.asdict(model_config)
    hash_dict |= dataclasses.asdict(optim_config)
    hash_dict |= dataclasses.asdict(loader_config)
    config_str = json.dumps(hash_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return hash_dict, config_hash


def run_validation(model, validation_loader, writer, t):
    print("Start validation")
    model.eval()
    with torch.no_grad():
        valid_loss_accum = 0.0
        for batch in validation_loader:
            inputs, targets = batch
            logits = model(inputs)
            valid_loss_accum += cross_entropy_loss(logits, targets).item()
        avg_valid_loss = valid_loss_accum / len(validation_loader)
    model.train()
    print(f"Valid loss: {avg_valid_loss}")
    writer.add_scalar("val/loss", avg_valid_loss, t)
    print("End validation")
    return avg_valid_loss


def train(
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    shared_namespace: types.SimpleNamespace,
    writer: SummaryWriter,
    output_path: str,
):

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (SwiGLULinear, Linear)):
            module.register_forward_hook(make_hook(name, writer, shared_namespace))

    checkpoint_dir = os.path.join(output_path, "checkpoints")

    t = 0
    start_epoch = 0
    total_steps = len(train_loader) * config.epochs
    checkpoint_file = find_most_recent_checkpoint(checkpoint_dir)
    if checkpoint_file is not None:
        start_epoch = load_checkpoint(checkpoint_file, model, optimizer)
        t = start_epoch * len(train_loader)

    model.train()
    val_losses = []
    last_train_loss = float("nan")

    if t == 0:
        print("")
        val_losses.append(run_validation(model, validation_loader, writer, t))

    epoch_bar = tqdm(range(start_epoch, config.epochs), desc="Epochs", unit="ep", file=sys.stdout)
    for epoch in epoch_bar:
        shared_namespace.epoch = epoch
        train_loss = 0.0
        batch_bar = tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch}",
            total=len(train_loader),
            leave=False,
            file=sys.stdout,
        )
        for j, batch in batch_bar:
            optimizer.zero_grad()
            inputs, targets = batch
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"{t1-t0} seconds per pass")

            gradient_clipping(
                model.parameters(), config.grad_max_l2_norm, config.grad_clip_eps
            )
            if config.use_lr_cos_schedule:
                alpha = lr_cosine_schedule(
                    t, config.lr_max, config.lr_min, config.lr_schedule_T_w, total_steps
                )
                for i, pg in enumerate(optimizer.param_groups):
                    optimizer.param_groups[i]["lr"] = alpha
            optimizer.step()
            train_loss += loss.item()
            shared_namespace.loss = train_loss
            t += 1
            shared_namespace.t = t

            last_train_loss = loss.item()

            if t % 100 == 0:
                avg_train_loss = train_loss / (j + 1)
                batch_bar.set_postfix(
                    loss=f"{avg_train_loss:.4f}",
                    lr=f"{alpha:.2e}" if config.use_lr_cos_schedule else "",
                )
                writer.add_scalar("train/loss", avg_train_loss, t)
                writer.add_scalar(
                    "train/grad_norm", log_gradient_norms(model, writer, t), t
                )

        epoch_bar.set_postfix(loss=f"{train_loss / len(train_loader):.4f}")

        if t % 500 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad, t)
                    writer.add_histogram(f"weights/{name}", param.data, t)

        if (epoch + 1) % config.validation_cadence == 0:
            val_losses.append(run_validation(model, validation_loader, writer, t))

        if (epoch + 1) % config.checkpoint_cadence == 0:
            ckpth = os.path.join(checkpoint_dir, f"epoch-{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, ckpth, loss=train_loss)

    if not val_losses:
        val_losses.append(run_validation(model, validation_loader, writer, t))

    return last_train_loss, val_losses[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Traing Loop")
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = config_path.parent / config_path.stem
    (output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_path / "tensorboard").mkdir(parents=True, exist_ok=True)

    train_config, optim_config, dl_config, model_config = load_model_config(args.config)

    model = TransformerLM(
        model_config.vocab_size,
        model_config.context_length,
        model_config.num_layers,
        model_config.d_model,
        model_config.num_heads,
        model_config.d_ff,
        model_config.rope_theta,
        device=model_config.device,
    )

    # model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=train_config.lr_max, **vars(optim_config))

    train_loader = DataLoader(
        dl_config.dataset_path,
        dl_config.batch_size,
        dl_config.context_length,
        ratio=dl_config.ratio,
        split="train",
        device=model_config.device,
    )
    validation_loader = DataLoader(
        dl_config.dataset_path,
        dl_config.batch_size,
        dl_config.context_length,
        ratio=dl_config.ratio,
        split="validation",
        device=model_config.device,
    )

    shared_namespace = types.SimpleNamespace(epoch=0, loss=0, t=0)
    setup_sigint_handler(model, optimizer, shared_namespace, str(output_path))

    config_dict, config_hash = merge_config_dicts(
        train_config, model_config, optim_config, dl_config
    )

    log_path = Path(os.path.join(output_path, "tensorboard", f"exp_{config_hash}"))
    if log_path.exists():
        print(f"[SKIP] Run {config_hash} already exists at {log_path}")
        sys.exit(0)

    save_model_config(
        train_config, optim_config, dl_config, model_config, str(output_path)
    )

    with SummaryWriter(log_dir=str(log_path)) as writer:

        final_train_loss, final_valid_loss = train(
            train_config,
            model,
            optimizer,
            train_loader,
            validation_loader,
            shared_namespace,
            writer,
            str(output_path),
        )

        hparams = {
            k: v
            for (k, v) in config_dict.items()
            if isinstance(v, (int, float, bool, str))
        }

        metrics = {
            "final_loss": final_train_loss,
            "final_valid_loss": final_valid_loss,
        }

        writer.add_hparams(hparams, metrics)
