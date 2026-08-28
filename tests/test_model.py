"""
tests/test_model.py
Unit and integration tests for the CIFAR-10 ML pipeline.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup: allow importing from src/ regardless of working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dataset import get_class_names, get_dataloaders, get_transforms  # noqa: E402
from model import SimpleCNN, ResNet18CIFAR, get_model                  # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def resnet18_model() -> nn.Module:
    return get_model("resnet18", num_classes=10)


@pytest.fixture(scope="session")
def simplecnn_model() -> nn.Module:
    return get_model("simplecnn", num_classes=10)


@pytest.fixture
def dummy_cifar_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small random batch that mimics CIFAR-10 (3×32×32)."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture
def dummy_png_bytes() -> bytes:
    """A tiny 32×32 RGB PNG image as raw bytes."""
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# Model tests
# ===========================================================================

class TestGetModel:
    def test_resnet18_output_shape(self, resnet18_model, dummy_cifar_batch):
        images, _ = dummy_cifar_batch
        out = resnet18_model(images)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_simplecnn_output_shape(self, simplecnn_model, dummy_cifar_batch):
        images, _ = dummy_cifar_batch
        out = simplecnn_model(images)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model("nonexistent_arch")

    def test_resnet18_is_nn_module(self, resnet18_model):
        assert isinstance(resnet18_model, nn.Module)

    def test_simplecnn_is_nn_module(self, simplecnn_model):
        assert isinstance(simplecnn_model, nn.Module)

    def test_resnet18_trainable_params(self, resnet18_model):
        n = sum(p.numel() for p in resnet18_model.parameters() if p.requires_grad)
        # ResNet-18 for CIFAR-10 should have ~11 M parameters
        assert n > 1_000_000, f"Unexpectedly few parameters: {n}"

    def test_simplecnn_trainable_params(self, simplecnn_model):
        n = sum(p.numel() for p in simplecnn_model.parameters() if p.requires_grad)
        assert n > 100_000, f"Unexpectedly few parameters: {n}"

    def test_model_eval_mode(self, resnet18_model):
        resnet18_model.eval()
        assert not resnet18_model.training

    def test_model_train_mode(self, resnet18_model):
        resnet18_model.train()
        assert resnet18_model.training
        resnet18_model.eval()  # restore

    def test_output_is_logits_not_probs(self, resnet18_model, dummy_cifar_batch):
        """Logits should not sum to 1 (that would indicate softmax was applied)."""
        images, _ = dummy_cifar_batch
        resnet18_model.eval()
        with torch.no_grad():
            out = resnet18_model(images)
        row_sums = out.sum(dim=1)
        # If softmax were applied, all sums would be ~1.0
        assert not torch.allclose(row_sums, torch.ones(4), atol=0.01), \
            "Model appears to apply softmax internally (should return raw logits)"

    def test_gradient_flows(self, dummy_cifar_batch):
        """Verify that loss.backward() produces non-zero gradients."""
        model = get_model("resnet18")
        model.train()
        images, labels = dummy_cifar_batch
        criterion = nn.CrossEntropyLoss()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients computed"
        assert any(g > 0 for g in grad_norms), "All gradients are zero"

    @pytest.mark.parametrize("num_classes", [5, 10, 100])
    def test_custom_num_classes(self, num_classes):
        model = get_model("resnet18", num_classes=num_classes)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, num_classes)


# ===========================================================================
# Dataset / transform tests
# ===========================================================================

class TestTransforms:
    def test_train_transform_returns_tensor(self):
        tf = get_transforms(train=True, dataset="cifar10")
        img = Image.new("RGB", (32, 32))
        t = tf(img)
        assert isinstance(t, torch.Tensor)

    def test_val_transform_returns_tensor(self):
        tf = get_transforms(train=False, dataset="cifar10")
        img = Image.new("RGB", (32, 32))
        t = tf(img)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 32, 32)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            get_transforms(train=True, dataset="imagenet")

    def test_fashionmnist_transform(self):
        tf = get_transforms(train=False, dataset="fashionmnist")
        img = Image.new("L", (28, 28))
        t = tf(img)
        assert t.shape == (1, 28, 28)


class TestClassNames:
    def test_cifar10_has_10_classes(self):
        names = get_class_names("cifar10")
        assert len(names) == 10

    def test_cifar10_contains_expected(self):
        names = get_class_names("cifar10")
        assert "airplane" in names
        assert "automobile" in names

    def test_fashionmnist_has_10_classes(self):
        names = get_class_names("fashionmnist")
        assert len(names) == 10

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            get_class_names("imagenet")


# ===========================================================================
# Checkpoint save / load round-trip
# ===========================================================================

class TestCheckpointRoundTrip:
    def test_save_and_load(self, resnet18_model, dummy_cifar_batch, tmp_path):
        """Save a checkpoint and reload it; verify predictions are identical."""
        ckpt_path = tmp_path / "test_ckpt.pt"
        resnet18_model.eval()

        images, _ = dummy_cifar_batch
        with torch.no_grad():
            logits_before = resnet18_model(images).clone()

        # Save
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": resnet18_model.state_dict(),
                "val_loss": 0.5,
                "val_accuracy": 0.8,
            },
            ckpt_path,
        )

        # Load into a fresh model
        fresh_model = get_model("resnet18", num_classes=10)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        fresh_model.load_state_dict(ckpt["model_state_dict"])
        fresh_model.eval()

        with torch.no_grad():
            logits_after = fresh_model(images)

        assert torch.allclose(logits_before, logits_after, atol=1e-5), \
            "Logits differ after checkpoint round-trip"


# ===========================================================================
# Serve module tests (no HTTP server required)
# ===========================================================================

class TestServeInference:
    """Test the inference logic in serve.py without starting a server."""

    def test_infer_transform_shape(self, dummy_png_bytes):
        from torchvision import transforms
        _CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        _CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ])
        img = Image.open(io.BytesIO(dummy_png_bytes)).convert("RGB")
        t = tf(img)
        assert t.shape == (3, 32, 32)

    def test_softmax_sums_to_one(self, resnet18_model, dummy_cifar_batch):
        import torch.nn.functional as F
        images, _ = dummy_cifar_batch
        resnet18_model.eval()
        with torch.no_grad():
            logits = resnet18_model(images)
            probs = F.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_top_class_in_valid_range(self, resnet18_model, dummy_cifar_batch):
        images, _ = dummy_cifar_batch
        resnet18_model.eval()
        with torch.no_grad():
            logits = resnet18_model(images)
        top_classes = logits.argmax(dim=1)
        assert all(0 <= c.item() < 10 for c in top_classes)
