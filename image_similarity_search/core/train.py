import io
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from image_similarity_search.utils import LOGGER, IterableSimpleNamespace


class TrainState:
    model: nn.Module
    data: dict = {}
    hyp: IterableSimpleNamespace
    device: torch.device
    optimizer: torch.optim.Optimizer = None
    start_epoch: int = 0
    epoch: int = 0
    epochs: int = 100
    best_epoch: int = 0
    save_period: int = -1
    stop: bool = False
    save_dir: Path
    wdir: Path
    last: Path
    best: Path
    fitness: Optional[float] = None
    best_fitness: Optional[float] = None
    monitor: Optional[str] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    metrics: Dict[str, float] = {}

    def __init__(self, model, data, hyp, device, save_dir: Path):
        self.model = model
        self.data = data
        self.hyp = hyp
        self.device = device
        self.save_dir = save_dir
        wdir = save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        self.wdir = wdir
        self.last = wdir / "last.pth"
        self.best = wdir / "best.pth"
        self.epochs = hyp.epochs
        self.save_period = hyp.save_period
        self.monitor = hyp.monitor
        self.start_epoch = 0
        self.epoch = 0
        self.best_epoch = 0
        self.stop = False


def dataloader(
    dataset: Dataset,
    batch: int,
    workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch (int): The batch size for loading data.
        workers (int, optional): The number of worker threads to use for data loading. Defaults to 8.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
        collate_fn (Optional[Callable], optional): Function to merge a list of samples to form a mini-batch. Defaults to None.

    Returns:
        DataLoader: A DataLoader instance for the given dataset.
    """
    bs = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), bs if bs > 1 else 0, workers])
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def save_checkpoint(state: TrainState):
    """
    Save the current state of the model and optimizer to a checkpoint file.
    Saves to 'last.pth' and updates 'best.pth' if fitness improves.
    """
    buffer = io.BytesIO()

    ckpt = {
        "epoch": state.epoch,
        "model": deepcopy(state.model.state_dict()),
        "optimizer": state.optimizer.state_dict(),
        "args": dict(state.hyp),
        "metrics": state.metrics,
        "date": datetime.now().isoformat(),
    }
    if state.fitness is not None and state.best_fitness is not None:
        ckpt["fitness"] = state.fitness
        ckpt["best_fitness"] = state.best_fitness

    torch.save(ckpt, buffer)
    serialized_ckpt = buffer.getvalue()
    # Save the last checkpoint
    LOGGER.info("Saving checkpoint: %s ", state.last.as_posix())
    state.last.write_bytes(serialized_ckpt)
    # Update the best checkpoint if fitness improves
    if state.best_fitness is not None and state.epoch == state.best_epoch:
        LOGGER.info("Saving best checkpoint: %s", state.best.as_posix())
        state.best.write_bytes(serialized_ckpt)

    # Save periodic checkpoints
    if state.save_period > 0 and state.epoch % state.save_period == 0:
        epoch_ckpt = state.wdir / f"epoch_{state.epoch}.pt"
        LOGGER.info("Saving periodic checkpoint: %s", epoch_ckpt.as_posix())
        epoch_ckpt.write_bytes(serialized_ckpt)


def early_stopping(state: TrainState):
    """Early stopping callback."""
    # on val end
    monitor = state.hyp.monitor
    if monitor is None or state.hyp.mode not in {"min", "max"}:
        LOGGER.warning("Early stopping disabled. No monitor metric or mode found.")
        return

    if len(state.metrics) == 0:
        raise ValueError("No metrics found. Please run validation first.")

    if monitor not in state.metrics:
        raise ValueError(f"Monitor metric '{monitor}' not found in metrics.")

    fitness = state.metrics[monitor]
    state.fitness = fitness

    def improvement(x, y):
        return x <= y if state.hyp.mode == "min" else x >= y

    if state.best_fitness is None:
        state.best_fitness = fitness
    elif improvement(fitness, state.best_fitness):
        LOGGER.info("New best fitness: %f", fitness)
        state.best_fitness = fitness
        state.best_epoch = state.epoch
    elif (
        state.hyp.patience > 0
        and (state.epoch - state.best_epoch) >= state.hyp.patience
    ):
        LOGGER.info("Early stopping at epoch %d", state.epoch)
        state.stop = True
        state.best_fitness = fitness
        state.best_epoch = state.epoch
    elif (
        state.hyp.patience > 0
        and (state.epoch - state.best_epoch) >= state.hyp.patience
    ):
        LOGGER.info("Early stopping at epoch %d", state.epoch)
        state.stop = True
