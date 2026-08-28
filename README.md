# mlops-pytorch-pipeline

> **MLOps & Infrastructure for Machine Learning — Assignment 2**
> End-to-end PyTorch image classification pipeline: local training → Docker → Kubernetes.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Developer Workflow                           │
│                                                                     │
│  feature/* branch  ──PR──►  develop  ──PR──►  main                 │
│       │                        │                  │                 │
│       └──── GitHub Actions CI ─┘──────────────────┘                │
│              (lint → test → docker build → k8s validate)           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster (ml-training namespace)      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ConfigMap: training-config                                  │   │
│  │  (training_config.yaml mounted at /app/configs)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │  PVC: cifar10-data  │    │  PVC: model-checkpoints          │   │
│  │  (5 Gi, dataset)    │    │  (2 Gi, .pt files)               │   │
│  └─────────────────────┘    └──────────────────────────────────┘   │
│           │                              │                          │
│           ▼                              ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Job: cifar10-training                                       │   │
│  │  Image: mlops-train:v1                                       │   │
│  │  Resources: 2 CPU / 4 Gi RAM                                 │   │
│  │  ► Downloads CIFAR-10 → trains ResNet-18 → saves .pt        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                    (after Job completes)                            │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Deployment: model-serving  (2 replicas, rolling update)    │   │
│  │  Image: mlops-serve:v1                                       │   │
│  │  Resources: 500m CPU / 1 Gi RAM per pod                      │   │
│  │  Probes: liveness + readiness + startup on GET /health       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Service: model-serving  (ClusterIP, port 80 → 8080)         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  HPA: model-serving-hpa  (2–10 replicas, CPU 70%)            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── src/
│   ├── model.py                # ResNet-18 (CIFAR-adapted) + SimpleCNN
│   ├── dataset.py              # CIFAR-10 / Fashion-MNIST data loaders
│   ├── train.py                # Training loop (YAML config, JSON logging)
│   └── serve.py                # FastAPI inference server
├── configs/
│   └── training_config.yaml    # Default hyperparameters
├── docker/
│   ├── Dockerfile.train        # Multi-stage training image
│   └── Dockerfile.serve        # Slim serving image (CPU-only torch)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml       # Includes PVC definitions
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt               # Pinned training deps
│   └── serve.txt               # Pinned serving deps (CPU-only torch)
└── tests/
    └── test_model.py           # pytest unit + integration tests
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://python.org) |
| Docker Desktop | 24+ | [docker.com](https://docker.com) |
| kubectl | 1.28+ | [kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| Minikube / kind | latest | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io) |
| Git | 2.40+ | [git-scm.com](https://git-scm.com) |

---

## Part A: Git Workflow

### Branch strategy

```
main          ← production-ready, only merged via PR
  └── develop ← integration branch
        ├── feature/model-implementation
        ├── feature/docker-training
        ├── feature/docker-serving
        └── feature/k8s-deployment
```

### Setup

```bash
git clone https://github.com/<your-username>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
git checkout -b develop
git push -u origin develop

# Start a feature branch
git checkout -b feature/model-implementation
# ... make changes ...
git add .
git commit -m "feat(model): add ResNet-18 CIFAR-10 adapter and SimpleCNN"
git push -u origin feature/model-implementation
# Open a PR on GitHub: feature/model-implementation → develop
```

### Conventional Commits reference

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Build / tooling |
| `test:` | Adding / fixing tests |
| `ci:` | CI pipeline changes |
| `refactor:` | Code restructure (no behaviour change) |

---

## Part B: Local Development

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install training dependencies

```bash
pip install --upgrade pip
pip install -r requirements/train.txt
```

### 3. Run training locally

```bash
# Uses configs/training_config.yaml automatically
# Edit data_dir and checkpoint_dir to local paths first:
#   data_dir: ./data
#   checkpoint_dir: ./checkpoints

python src/train.py
# or with explicit config:
python src/train.py --config configs/training_config.yaml
```

Training emits structured JSON-lines to stdout:

```json
{"event": "config_loaded", "path": "configs/training_config.yaml"}
{"event": "device_selected", "device": "cpu"}
{"event": "model_created", "architecture": "resnet18", "trainable_params": 11173962}
{"epoch": 1, "train_loss": 1.8234, "train_accuracy": 0.3412, "val_loss": 1.6891, "val_accuracy": 0.4023, ...}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", "val_loss": 1.6891}
```

### 4. Run the inference server locally

```bash
pip install -r requirements/serve.txt
# Ensure checkpoints/classifier_v1.pt exists
CHECKPOINT_PATH=./checkpoints/classifier_v1.pt python src/serve.py
```

Test endpoints:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

### 5. Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Part C: Docker

### Build images

```bash
# Training image (multi-stage)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Serving image (CPU-only, slim)
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### Run training container

```bash
# Linux / macOS
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Windows PowerShell
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-train:v1
```

### Run serving container

```bash
# Linux / macOS
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Windows PowerShell
docker run --rm -p 8080:8080 `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-serve:v1
```

### Test prediction endpoint

```bash
# Health check
curl http://localhost:8080/health

# Prediction (replace test_image.png with any PNG/JPEG)
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

Expected response:

```json
{
  "class_id": 3,
  "class_name": "cat",
  "confidence": 0.7231,
  "probabilities": {
    "airplane": 0.012, "automobile": 0.034, "bird": 0.056,
    "cat": 0.7231, "deer": 0.021, "dog": 0.089,
    "frog": 0.011, "horse": 0.023, "ship": 0.018, "truck": 0.013
  }
}
```

---

## Part D & E: Kubernetes Deployment

### Prerequisites

```bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Point Docker CLI to Minikube's daemon (so images are available in-cluster)
eval $(minikube docker-env)          # Linux / macOS
# Windows PowerShell:
# & minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Build images inside Minikube
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### Apply manifests

```bash
# 1. Namespace
kubectl apply -f k8s/namespace.yaml

# 2. ConfigMap
kubectl apply -f k8s/configmap.yaml

# 3. Training Job (also creates PVCs)
kubectl apply -f k8s/training-job.yaml

# Monitor training
kubectl logs -f job/cifar10-training -n ml-training
kubectl get job cifar10-training -n ml-training
```

### Deploy serving layer (after training completes)

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Verify

```bash
kubectl get pods -n ml-training
kubectl get deployment model-serving -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl get hpa -n ml-training
```

### Test prediction via port-forward

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training &

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## Configuration Reference

All training hyperparameters live in `configs/training_config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model.architecture` | `resnet18` | `resnet18` or `simplecnn` |
| `model.num_classes` | `10` | Number of output classes |
| `training.epochs` | `30` | Maximum training epochs |
| `training.batch_size` | `128` | Mini-batch size |
| `training.learning_rate` | `0.001` | Initial learning rate (AdamW) |
| `training.weight_decay` | `0.0001` | L2 regularisation |
| `training.early_stopping_patience` | `5` | Epochs without improvement before stopping |
| `data.dataset` | `cifar10` | `cifar10` or `fashionmnist` |
| `data.data_dir` | `/app/data` | Dataset root directory |
| `output.checkpoint_dir` | `/app/checkpoints` | Where to save `.pt` files |
| `output.model_name` | `classifier_v1.pt` | Checkpoint filename |

---

## Environment Variables (serve container)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | `/app/checkpoints/classifier_v1.pt` | Path to model checkpoint |
| `MODEL_ARCH` | `resnet18` | Architecture name |
| `NUM_CLASSES` | `10` | Number of classes |
| `PORT` | `8080` | Server port |

---

## Reflective Write-up

### What was the most challenging part?

**Adapting ResNet-18 for CIFAR-10** was the first non-trivial challenge. The standard torchvision ResNet-18 is designed for 224×224 ImageNet images: its first convolution uses a 7×7 kernel with stride 2, followed by a 3×3 max-pool — together they reduce a 224×224 input to 56×56 before the residual blocks even begin. Applied to CIFAR-10's 32×32 images, this aggressive downsampling collapses spatial resolution to just 4×4 after the first two layers, starving the residual blocks of meaningful feature maps and capping accuracy well below what the architecture is capable of. The fix — replacing the first conv with a 3×3/stride-1 and removing the max-pool — is well-known in the literature but requires understanding *why* the original design exists.

**Multi-stage Docker builds** required careful thought about layer ordering. Placing the `pip install` step before copying application source means that rebuilding after a code change (the common case) reuses the cached dependency layer and completes in seconds rather than minutes. Getting this right for both the training image (full PyTorch) and the serving image (CPU-only wheel, no tensorboard) while keeping the final images lean took several iterations.

**Kubernetes volume semantics** were the steepest learning curve. A `ReadWriteOnce` PVC can only be mounted by pods on the same node, which means the training Job and the serving Deployment must be scheduled on the same node — or the PVC must be upgraded to `ReadWriteMany` (e.g., NFS or a cloud file store). For a single-node Minikube cluster this is transparent, but it would be a real constraint in a multi-node production cluster. The serving Deployment mounts the checkpoint PVC as `readOnly: true` to prevent accidental writes.

**Early stopping with a cosine LR scheduler** required careful ordering: the scheduler must step *after* the optimiser, and the patience counter must only increment when validation loss does not improve — not on every epoch. Getting the JSON-lines logging to flush immediately (`flush=True`) was also important so that `kubectl logs -f` shows real-time progress rather than buffered output.

---

## License

MIT — see [LICENSE](LICENSE) for details.
