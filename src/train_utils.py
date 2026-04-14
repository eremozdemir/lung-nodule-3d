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
    scheduler_patience: int = 6


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
def evaluate(model, loader, device, threshold: float, criterion=None) -> Dict:
    model.eval()
    all_logits = []
    all_y = []
    total_loss = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True).float().view(-1)

        logits = model(x).detach()

        # Compute val loss if criterion is provided
        if criterion is not None:
            loss = criterion(logits, y_dev)
            total_loss += loss.item()
            n_batches += 1

        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy().reshape(-1))

    logits_np = np.concatenate(all_logits, axis=0)
    y_true    = np.concatenate(all_y, axis=0).astype(int)

    metrics = compute_binary_metrics(y_true, logits_np, threshold=threshold)

    if criterion is not None and n_batches > 0:
        metrics["val_loss"] = total_loss / n_batches

    return metrics

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    losses = []
    for x, y in tqdm(loader, desc="eval_loss", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
    return float(np.mean(losses))


def save_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def calibrate_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    Post-hoc temperature scaling calibration.

    Finds the scalar T* that minimises binary cross-entropy NLL on the
    validation set when probabilities are computed as:

        p_calibrated = sigmoid(logit / T*)

    A temperature T > 1 "softens" overconfident predictions (spreads the
    probability distribution toward 0.5); T < 1 sharpens them.

    Call this on validation-set raw logits after training is complete, before
    the threshold sweep.  The returned T* can then be applied as:

        calibrated_logits = raw_logits / T*

    Parameters
    ----------
    logits : np.ndarray  shape (N,)  raw model output (pre-sigmoid)
    y_true : np.ndarray  shape (N,)  binary labels {0, 1}

    Returns
    -------
    T* : float, optimal temperature
    """
    from scipy.optimize import minimize_scalar

    y = y_true.astype(np.float64)

    def nll(T: float) -> float:
        if T <= 0:
            return 1e9
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


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