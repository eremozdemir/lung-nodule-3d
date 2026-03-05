import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_binary_metrics


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_workers: int = 2
    seed: int = 42
    threshold: float = 0.5


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    losses = []

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=float(np.mean(losses)))

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> Dict:
    model.eval()
    all_logits = []
    all_y = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)

        y_np = y.numpy().reshape(-1)
        all_y.append(y_np)

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(int)

    return compute_binary_metrics(y_true, logits, threshold=threshold)


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(path: str, model, optimizer, epoch: int, extra: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "extra": extra,
        },
        path
    )