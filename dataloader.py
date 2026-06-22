import sys
import os
import torch
import numpy as np
import numpy.typing as npt
import typing
import tempfile

from profiler import profile


# @profile(output_file="./profile_loader.prof")
def load_data(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    rng=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = np.random.default_rng()
    num_token = x.shape[0]
    start_idx = rng.choice(num_token - context_length, size=(batch_size,), replace=True)
    seq_idx = start_idx.reshape(-1, 1) + np.arange(context_length)
    samples = torch.from_numpy(x[seq_idx]).to(device)
    next_token_targets = torch.tensor(x[seq_idx + 1], device=device)
    return samples, next_token_targets


class DataLoader:
    def __init__(
        self,
        dataset_path: str | os.PathLike,
        batch_size: int,
        context_length: int,
        split: typing.Literal["test", "train"] = "train",
        ratio: float = 0.9,
        device: torch.device | None = None,
    ):
        self.data = np.load(dataset_path, mmap_mode="r")
        if split == "test":
            self.data = self.data[-int(len(self.data) * (1.0 - ratio)) :]
        else:
            self.data = self.data[0 : int(len(self.data) * ratio)]
        assert len(self.data) > 0, "Dataset is empty!"
        self.device = device
        self.batch_size = batch_size
        self.context_length = context_length
        self.rng = np.random.default_rng()
        self.num_batches = (len(self.data) - context_length) // batch_size
        self._step = 0

    def __iter__(self):
        self._step = 0
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self._step > self.num_batches:
            raise StopIteration
        self._step += 1
        return load_data(
            self.data, self.batch_size, self.context_length, self.device, rng=self.rng
        )


if __name__ == "__main__":

    # Load as memmap
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokens.npy")
        np.save(path, np.arange(10_000, dtype=np.int32))
        # x = np.load(path, mmap_mode='r')
        # print(type(x), x.shape)
        dl = DataLoader(path, 32, 7, device="cpu")
        for batch in dl:
            print(batch)
