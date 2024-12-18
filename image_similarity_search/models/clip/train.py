import logging as LOGGER
from pathlib import Path
from typing import Callable, Literal, Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from image_similarity_search.utils import IterableSimpleNamespace, LOGGER, colorstr
from image_similarity_search.utils.torch import (
    init_seeds,
    select_device,
    device_clear_memory,
)
from image_similarity_search.core.datatypes import FoldType, Optimizer
from image_similarity_search.core.callbacks import DEFAULT_CALLBACKS
from image_similarity_search.core.eval import evaluate
from image_similarity_search.core.train import (
    TrainState,
    dataloader,
    save_checkpoint,
    early_stopping,
)
from image_similarity_search.models.clip.data import get_dataset


def build_dataset(data: dict, mode: FoldType = "train") -> torch.utils.data.Dataset:
    """Builds the dataset based on the given data dictionary and mode."""
    return get_dataset(data["path"], data[mode])


def build_optimizer(model, hyp: IterableSimpleNamespace, name: Optimizer = "Adam"):
    """Build the optimizer for the model."""
    opt_fn = getattr(optim, name, optim.Adam)
    return opt_fn(model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)


def train_epoch(state: TrainState, train_loader) -> dict:
    """Train the model for one epoch."""
    state.model.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    train_loss = 0

    progress = tqdm.tqdm(train_loader, desc=f"Epoch {state.epoch + 1}/{state.epochs} [Training]")
    for images, texts in progress:
        images, texts = images.to(state.device, non_blocking=True), texts.to(state.device, non_blocking=True)
        state.optimizer.zero_grad()
        logits_per_image, logits_per_text = state.model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=state.device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        state.optimizer.step()
        train_loss += total_loss.item()
        progress.set_postfix(loss=total_loss.item())

    avg_loss = train_loss / len(train_loader)
    return {"train/loss": avg_loss}


def validate_epoch(state: TrainState, val_loader) -> dict:
    """Validate the model."""
    state.model.eval()
    val_loss = 0
    embeddings = []
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    progress = tqdm.tqdm(val_loader, desc=f"Epoch {state.epoch + 1}/{state.epochs} [Validation]")
    with torch.inference_mode():
        for images, texts in progress:
            images, texts = images.to(state.device, non_blocking=True), texts.to(state.device, non_blocking=True)
            logits_per_image, logits_per_text = state.model(images, texts)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=state.device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            val_loss += total_loss.item()
            progress.set_postfix(loss=total_loss.item())

            image_features = state.model.encode_image(images)
            embeddings.append(image_features.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    metrics = evaluate(np.vstack(embeddings), val_loader.dataset.data, seed=state.hyp.seed)
    metrics["val/loss"] = avg_loss
    return metrics


def train(
    model: nn.Module,
    data: dict,
    hyp: IterableSimpleNamespace,
    device: torch.device,
    save_dir: Path,
    callbacks=DEFAULT_CALLBACKS,
):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    device = select_device(device)
    save_dir = Path(save_dir) if isinstance(save_dir, str) else save_dir
    state = TrainState(model, data, hyp, device, save_dir)
    callbacks.run("on_pretrain_routine_start", state)
    state.model = state.model.to(state.device)
    state.optimizer = build_optimizer(state.model, state.hyp)

    train_set = build_dataset(state.data, mode="train")
    val_set = build_dataset(state.data, mode="val")

    dl_train = dataloader(train_set, state.hyp.batch_size, collate_fn=train_set.collate_fn)
    dl_val = dataloader(val_set, state.hyp.batch_size * 2, collate_fn=val_set.collate_fn)
    callbacks.run("on_pretrain_routine_end", state)

    callbacks.run("on_train_start", state)
    for epoch in range(state.start_epoch, state.epochs):
        state.epoch = epoch
        state.metrics = {}

        callbacks.run("on_train_epoch_start", state)
        train_metrics = train_epoch(state, dl_train)
        callbacks.run("on_train_epoch_end", state)

        callbacks.run("on_val_start", state)
        valid_metrics = validate_epoch(state, dl_val)
        state.metrics = {**train_metrics, **valid_metrics}
        callbacks.run("on_val_end", state)
        callbacks.run("on_model_save", state)
        callbacks.run("on_fit_epoch_end", state)
        device_clear_memory(state.device)
        if state.stop:
            break

    callbacks.run("on_train_end", state)
    device_clear_memory(state.device)
    callbacks.run("teardown", state)