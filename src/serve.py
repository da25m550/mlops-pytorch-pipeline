"""
serve.py - FastAPI inference server for the CIFAR-10 classifier.

Endpoints:
  GET  /health   → 200 {"status": "ok", "model_loaded": true}
  POST /predict  → {"class_id": int, "class_name": str, "probabilities": {...}}

Environment variables (all optional):
  CHECKPOINT_PATH  Path to the .pt checkpoint file
                   (default: /app/checkpoints/classifier_v1.pt)
  MODEL_ARCH       Architecture name (default: resnet18)
  NUM_CLASSES      Number of output classes (default: 10)
  HOST             Bind host (default: 0.0.0.0)
  PORT             Bind port (default: 8080)

Run locally:
    uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload
    # or
    python src/serve.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

# Allow running from repo root OR from inside src/
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import get_class_names  # noqa: E402
from model import get_model  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("serve")

# ---------------------------------------------------------------------------
# Global model state (populated at startup)
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "model": None,
    "device": None,
    "class_names": None,
    "checkpoint_path": None,
    "loaded_at": None,
}

# ---------------------------------------------------------------------------
# Inference transform (no augmentation)
# ---------------------------------------------------------------------------

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_INFER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ]
)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def _load_model() -> None:
    """Load the model checkpoint into _STATE. Called once at startup."""
    checkpoint_path = Path(
        os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    )
    # Fallback for local development
    if not checkpoint_path.exists():
        local_fallback = Path("checkpoints/classifier_v1.pt")
        if local_fallback.exists():
            checkpoint_path = local_fallback

    arch = os.environ.get("MODEL_ARCH", "resnet18")
    num_classes = int(os.environ.get("NUM_CLASSES", "10"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading checkpoint from %s on %s", checkpoint_path, device)

    if not checkpoint_path.exists():
        logger.warning(
            "Checkpoint not found at %s. Server will start but /predict "
            "will return 503 until a checkpoint is available.",
            checkpoint_path,
        )
        _STATE["model"] = None
    else:
        model = get_model(architecture=arch, num_classes=num_classes)
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and our wrapped checkpoint format
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _STATE["model"] = model
        _STATE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info(
            "Model loaded successfully (val_acc=%.4f, epoch=%s)",
            ckpt.get("val_accuracy", float("nan")),
            ckpt.get("epoch", "?"),
        )

    _STATE["device"] = device
    _STATE["class_names"] = get_class_names("cifar10")
    _STATE["checkpoint_path"] = str(checkpoint_path)


# ---------------------------------------------------------------------------
# FastAPI app with lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    _load_model()
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="CIFAR-10 Classifier API",
    description=(
        "Serves a PyTorch ResNet-18 / SimpleCNN trained on CIFAR-10. "
        "POST an image to /predict to get class probabilities."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health() -> JSONResponse:
    """
    Returns 200 if the server is running.
    Returns 503 if the model checkpoint has not been loaded yet.
    """
    model_loaded = _STATE["model"] is not None
    payload = {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "checkpoint_path": _STATE["checkpoint_path"],
        "loaded_at": _STATE["loaded_at"],
        "device": str(_STATE["device"]),
    }
    status_code = 200 if model_loaded else 503
    return JSONResponse(content=payload, status_code=status_code)


@app.post("/predict", summary="Classify an image")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a PNG/JPEG image and return CIFAR-10 class probabilities.

    Request:
        multipart/form-data with field ``image`` containing the image file.

    Response:
        ```json
        {
          "class_id": 3,
          "class_name": "cat",
          "confidence": 0.8731,
          "probabilities": {
            "airplane": 0.01,
            "automobile": 0.02,
            ...
          }
        }
        ```
    """
    if _STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for details.",
        )

    # ── Read & validate image ────────────────────────────────────────────
    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {exc}",
        ) from exc

    # ── Preprocess ──────────────────────────────────────────────────────
    tensor = _INFER_TRANSFORM(pil_img).unsqueeze(0).to(_STATE["device"])

    # ── Inference ───────────────────────────────────────────────────────
    with torch.no_grad():
        logits = _STATE["model"](tensor)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0]  # (num_classes,)

    class_names: list[str] = _STATE["class_names"]
    prob_list = probs.cpu().tolist()
    top_idx = int(probs.argmax().item())

    return JSONResponse(
        content={
            "class_id": top_idx,
            "class_name": class_names[top_idx],
            "confidence": round(prob_list[top_idx], 6),
            "probabilities": {
                name: round(p, 6) for name, p in zip(class_names, prob_list)
            },
        }
    )


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse({"message": "CIFAR-10 Classifier API. See /docs."})


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    logger.info("Starting server on %s:%s", host, port)
    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )
