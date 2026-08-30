# Author: Arun Kumar S | Roll Number: DA25M550
# Note: Missed created PRs while checking in base code. Hence adding this comment to demonstrate Github PR feature and working

# Author: Arun Kumar S | Roll Number: DA25M550
# Note: Missed created PRs while checking in base code. Hence adding this comment to demonstrate Github PR feature and working

"""
model.py - CNN model definitions for CIFAR-10 classification.

Supports ResNet-18 (adapted for 32x32 CIFAR images) and a custom
lightweight CNN. Architecture is selected via training_config.yaml.
"""

from __future__ import annotations

import torch
import torchvision.models as tv_models
from torch import nn

# ---------------------------------------------------------------------------
# Custom lightweight CNN (fallback / fast-training option)
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    A compact CNN suitable for CIFAR-10 (32×32 RGB images).

    Architecture:
        3 × ConvBlock stacks with max-pooling → Global Average Pool → FC
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),  # 16×16
            nn.Dropout2d(0.1),
            # Block 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),  # 8×8
            nn.Dropout2d(0.1),
            # Block 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),  # 4×4
            nn.Dropout2d(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1×1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ResNet-18 adapted for CIFAR-10
# ---------------------------------------------------------------------------


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images).

    Key changes vs. the ImageNet variant:
      - First conv: 3×3, stride 1, no max-pool (preserves spatial resolution)
      - Final FC replaced to output `num_classes` logits
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()
        # Load backbone
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Replace first conv + remove max-pool for small images
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()  # type: ignore[assignment]

        # Replace classifier head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "resnet18": ResNet18CIFAR,
    "simplecnn": SimpleCNN,
}


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        architecture: One of ``"resnet18"`` or ``"simplecnn"``.
        num_classes:  Number of output classes (default 10 for CIFAR-10).
        pretrained:   Load ImageNet weights for ResNet-18 (ignored for SimpleCNN).

    Returns:
        An ``nn.Module`` ready for training or inference.

    Raises:
        ValueError: If ``architecture`` is not in the registry.
    """
    arch_lower = architecture.lower()
    if arch_lower not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[arch_lower]
    if cls is ResNet18CIFAR:
        return cls(num_classes=num_classes, pretrained=pretrained)
    return cls(num_classes=num_classes)


if __name__ == "__main__":
    # Quick smoke-test
    for arch in ("resnet18", "simplecnn"):
        m = get_model(arch)
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        print(f"{arch}: input {tuple(x.shape)} → output {tuple(out.shape)}")
        assert out.shape == (2, 10), f"Unexpected output shape for {arch}"
    print("model.py smoke-test passed.")
