# Author: Arun Kumar S | Roll Number: DA25M550
# Note: Missed created PRs while checking in base code. Hence adding this comment to demonstrate Github PR feature and working

"""
train.py - Full training loop for CIFAR-10 image classification.

Features:
  - Reads all hyperparameters from configs/training_config.yaml
  - Structured JSON-lines logging to stdout
  - Saves best checkpoint (by val_loss) to configurable output path
  - Early stopping with configurable patience
  - Learning-rate scheduler (CosineAnnealingLR)
  - Graceful handling of CPU-only environments (no GPU required)

Usage (local):
    python src/train.py
    python src/train.py --config configs/training_config.yaml

Usage (Docker / K8s):
    The script auto-discovers the config at /app/configs/training_config.yaml
    and falls back to configs/training_config.yaml for local runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler

# Allow running from repo root OR from inside src/
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import get_dataloaders          # noqa: E402
from model import get_model                  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(obj: dict) -> None:
    """Emit a JSON-lines log entry to stdout (flushed immediately)."""
    print(json.dumps(obj), flush=True)


def load_config(config_path: str | Path) -> dict:
    """Load and return a YAML config file as a plain dict."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def resolve_config_path(cli_path: str | None) -> Path:
    """
    Resolve the config file path with the following priority:
      1. CLI argument (--config)
      2. Environment variable CONFIG_PATH
      3. /app/configs/training_config.yaml  (Docker / K8s)
      4. configs/training_config.yaml       (local dev)
    """
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates += [
        Path("/app/configs/training_config.yaml"),
        Path("configs/training_config.yaml"),
        _SRC.parent / "configs" / "training_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find training_config.yaml. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Training / evaluation steps
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Returns:
        (avg_loss, accuracy) over the entire epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a DataLoader.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(cli_config_path: str | None = None) -> None:
    # ── Config ──────────────────────────────────────────────────────────────
    config_path = resolve_config_path(cli_config_path)
    config = load_config(config_path)
    _log({"event": "config_loaded", "path": str(config_path)})

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log({"event": "device_selected", "device": str(device)})

    # ── Model ───────────────────────────────────────────────────────────────
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log({
        "event": "model_created",
        "architecture": config["model"]["architecture"],
        "trainable_params": n_params,
    })

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        dataset=config["data"].get("dataset", "cifar10"),
    )
    _log({
        "event": "data_loaded",
        "dataset": config["data"].get("dataset", "cifar10"),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
    })

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    epochs = config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Mixed-precision scaler (GPU only; None on CPU)
    scaler: GradScaler | None = (
        GradScaler("cuda") if device.type == "cuda" else None
    )

    # ── Checkpoint dir ──────────────────────────────────────────────────────
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / config["output"]["model_name"]

    # ── Early stopping state ────────────────────────────────────────────────
    patience: int = config["training"]["early_stopping_patience"]
    best_val_loss: float = float("inf")
    patience_counter: int = 0

    # ── Training loop ───────────────────────────────────────────────────────
    _log({"event": "training_start", "epochs": epochs, "patience": patience})

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        current_lr = scheduler.get_last_lr()[0]

        _log({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": round(current_lr, 8),
            "epoch_time_s": round(elapsed, 2),
        })

        # ── Checkpoint on improvement ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": config,
                },
                best_ckpt_path,
            )
            _log({
                "event": "checkpoint_saved",
                "path": str(best_ckpt_path),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            })
        else:
            patience_counter += 1
            _log({
                "event": "no_improvement",
                "patience_counter": patience_counter,
                "patience_limit": patience,
            })
            if patience_counter >= patience:
                _log({
                    "event": "early_stopping",
                    "epoch": epoch,
                    "best_val_loss": round(best_val_loss, 4),
                })
                break

    _log({
        "event": "training_complete",
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint": str(best_ckpt_path),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 classifier."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training_config.yaml (optional; auto-discovered if omitted).",
    )
    args = parser.parse_args()
    main(cli_config_path=args.config)
