"""
dataset.py - Data loading utilities for CIFAR-10 / Fashion-MNIST.

Provides:
  - get_transforms()   : torchvision transform pipelines
  - get_dataloaders()  : train + validation DataLoader pair
  - CIFAR10_CLASSES    : human-readable class labels
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES: list[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

FASHION_MNIST_CLASSES: list[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# CIFAR-10 channel statistics (pre-computed on training set)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Fashion-MNIST channel statistics
_FMNIST_MEAN = (0.2860,)
_FMNIST_STD  = (0.3530,)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(
    train: bool = True,
    dataset: str = "cifar10",
) -> transforms.Compose:
    """
    Return a torchvision transform pipeline.

    Training augmentations:
      - Random horizontal flip
      - Random crop with padding
      - (CIFAR-10 only) Random erasing for regularisation

    Args:
        train:   If True, return augmented pipeline; else return eval pipeline.
        dataset: ``"cifar10"`` or ``"fashionmnist"``.

    Returns:
        A ``transforms.Compose`` object.
    """
    dataset = dataset.lower().replace("-", "").replace("_", "")

    if dataset == "cifar10":
        mean, std = _CIFAR10_MEAN, _CIFAR10_STD
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.1),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif dataset == "fashionmnist":
        mean, std = _FMNIST_MEAN, _FMNIST_STD
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Choose 'cifar10' or 'fashionmnist'."
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int | None = None,
    dataset: str = "cifar10",
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        data_dir:    Root directory where datasets are stored / downloaded.
        batch_size:  Mini-batch size for both loaders.
        num_workers: Number of worker processes. Defaults to
                     ``min(4, os.cpu_count())`` on Linux/Mac and ``0`` on
                     Windows to avoid multiprocessing issues inside Docker.
        dataset:     ``"cifar10"`` or ``"fashionmnist"``.
        pin_memory:  Pin host memory for faster GPU transfers.

    Returns:
        ``(train_loader, val_loader)`` tuple.
    """
    # Safe default for num_workers (Windows / Docker compatibility)
    if num_workers is None:
        if os.name == "nt":          # Windows
            num_workers = 0
        else:
            num_workers = min(4, os.cpu_count() or 1)

    dataset_lower = dataset.lower().replace("-", "").replace("_", "")
    train_tf = get_transforms(train=True,  dataset=dataset_lower)
    val_tf   = get_transforms(train=False, dataset=dataset_lower)

    if dataset_lower == "cifar10":
        train_ds = datasets.CIFAR10(
            root=data_dir, train=True,  download=True, transform=train_tf
        )
        val_ds = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_tf
        )
    elif dataset_lower == "fashionmnist":
        train_ds = datasets.FashionMNIST(
            root=data_dir, train=True,  download=True, transform=train_tf
        )
        val_ds = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=val_tf
        )
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Choose 'cifar10' or 'fashionmnist'."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def get_class_names(dataset: str = "cifar10") -> list[str]:
    """Return human-readable class labels for the given dataset."""
    d = dataset.lower().replace("-", "").replace("_", "")
    if d == "cifar10":
        return CIFAR10_CLASSES
    if d == "fashionmnist":
        return FASHION_MNIST_CLASSES
    raise ValueError(f"Unknown dataset '{dataset}'.")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading CIFAR-10 to temp dir for smoke-test …")
        tl, vl = get_dataloaders(tmp, batch_size=32, dataset="cifar10")
        imgs, labels = next(iter(tl))
        print(f"Train batch: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
        imgs, labels = next(iter(vl))
        print(f"Val   batch: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
        print("dataset.py smoke-test passed.")
