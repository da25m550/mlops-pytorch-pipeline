# Steps to Run — mlops-pytorch-pipeline (Windows PC, No GPU)

All commands are for **Windows PowerShell** unless noted.
Run every command from the project root:
```
C:\Users\asreeniv\OneDrive - Qualcomm\Documents\IITM\MLOPS-Lab\Assignment-3\mlops-pytorch-pipeline
```

---

## 0. Prerequisites

Install the following if not already present:

| Tool | Download |
|------|----------|
| Python 3.11 | https://python.org/downloads |
| Git | https://git-scm.com |
| Docker Desktop | https://docker.com/products/docker-desktop |
| kubectl | https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/ |
| Minikube | https://minikube.sigs.k8s.io/docs/start/ |

---

## 1. Part A — GitHub Repository Setup

### 1a. Create the repo on GitHub
1. Go to https://github.com/new
2. Name: `mlops-pytorch-pipeline`
3. Visibility: **Public**
4. Do NOT initialise with README (we'll push our own)
5. Click **Create repository**

### 1b. Initialise Git locally
```powershell
cd "C:\Users\asreeniv\OneDrive - Qualcomm\Documents\IITM\MLOPS-Lab\Assignment-3\mlops-pytorch-pipeline"

git init
git add .
git commit -m "chore: initial project scaffold"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/mlops-pytorch-pipeline.git
git push -u origin main
```

### 1c. Create the develop branch
```powershell
git checkout -b develop
git push -u origin develop
```

### 1d. Week 1 — PR 1: Model + Dataset + Training
```powershell
git checkout develop
git checkout -b feature/model-implementation
# (files already exist — just commit them)
git add src/model.py src/dataset.py src/train.py configs/ requirements/
git commit -m "feat(model): add ResNet-18 CIFAR-10 adapter, dataset loader, and training loop"
git push -u origin feature/model-implementation
# → Open PR on GitHub: feature/model-implementation → develop
# → Merge the PR
```

### 1e. Week 1 — PR 2: Tests + CI
```powershell
git checkout develop && git pull
git checkout -b feature/tests-and-ci
git add tests/ .github/ .gitignore
git commit -m "test(ci): add pytest unit tests and GitHub Actions CI pipeline"
git push -u origin feature/tests-and-ci
# → Open PR on GitHub: feature/tests-and-ci → develop
# → Merge the PR
```

### 1f. Week 2 — PR 3: Docker
```powershell
git checkout develop && git pull
git checkout -b feature/docker-training
git add docker/ docker-compose.yml
git commit -m "feat(docker): add multi-stage training and serving Dockerfiles"
git push -u origin feature/docker-training
# → Open PR on GitHub: feature/docker-training → develop
# → Merge the PR
```

### 1g. Week 2 — PR 4: Kubernetes
```powershell
git checkout develop && git pull
git checkout -b feature/k8s-deployment
git add k8s/ src/serve.py
git commit -m "feat(k8s): add namespace, configmap, training job, serving deployment, HPA"
git push -u origin feature/k8s-deployment
# → Open PR on GitHub: feature/k8s-deployment → develop
# → Merge the PR
```

### 1h. Final merge to main
```powershell
git checkout main
git merge develop
git push origin main
```

---

## 2. Part B — Local Python Training

### 2a. Create virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2b. Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements/train.txt
```

### 2c. Run training (CPU, ~5-10 min/epoch)
```powershell
# Uses configs/training_config_local.yaml (./data and ./checkpoints paths)
python src/train.py --config configs/training_config_local.yaml
```

Expected output (JSON-lines):
```
{"event": "config_loaded", "path": "configs/training_config_local.yaml"}
{"event": "device_selected", "device": "cpu"}
{"event": "model_created", "architecture": "resnet18", "trainable_params": 11173962}
{"event": "data_loaded", "dataset": "cifar10", "train_batches": 782, "val_batches": 157}
{"event": "training_start", "epochs": 10, "patience": 3}
{"epoch": 1, "train_loss": 1.7234, "train_accuracy": 0.3812, ...}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt", ...}
...
{"event": "training_complete", "best_val_loss": 1.2341, "checkpoint": "checkpoints/classifier_v1.pt"}
```

### 2d. Run tests
```powershell
pytest tests/ -v
```

### 2e. Run inference server locally
```powershell
pip install -r requirements/serve.txt
$env:CHECKPOINT_PATH = ".\checkpoints\classifier_v1.pt"
python src/serve.py
```

In another terminal:
```powershell
# Health check
curl http://localhost:8080/health

# Prediction (create a test image first)
python -c "from PIL import Image; Image.new('RGB',(32,32),(100,150,200)).save('test_image.png')"
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## 3. Part C — Docker

### 3a. Build images
```powershell
# Training image (multi-stage, ~3-4 GB)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Serving image (CPU-only torch, ~1.5 GB)
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 3b. Run training container
```powershell
# Create directories first
New-Item -ItemType Directory -Force data, checkpoints

docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/checkpoints:/app/checkpoints" `
  mlops-train:v1 `
  --config /app/configs/training_config.yaml
```

> **Note:** The container config uses `/app/data` and `/app/checkpoints` which are
> mapped to your local `./data` and `./checkpoints` via the volume mounts.
> Training on CPU inside Docker will take ~5-10 min per epoch.

### 3c. Run serving container
```powershell
# Requires checkpoints/classifier_v1.pt to exist (run training first)
docker run --rm -p 8080:8080 `
  -v "${PWD}/checkpoints:/app/checkpoints" `
  mlops-serve:v1
```

### 3d. Test prediction endpoint
```powershell
# Health check
curl http://localhost:8080/health

# Create a test image
python -c "from PIL import Image; Image.new('RGB',(32,32),(100,150,200)).save('test_image.png')"

# Prediction
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

### 3e. Using docker-compose (alternative)
```powershell
# Train then serve automatically
docker compose up

# Or just train
docker compose run --rm train

# Or just serve (if checkpoint exists)
docker compose up serve
```

---

## 4. Parts D & E — Kubernetes

### 4a. Start Minikube
```powershell
minikube start --cpus=4 --memory=8192 --driver=docker
```

### 4b. Load Docker images into Minikube
```powershell
# Option A: Build directly inside Minikube's Docker daemon
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Option B: Load pre-built images
minikube image load mlops-train:v1
minikube image load mlops-serve:v1
```

### 4c. Apply manifests
```powershell
# Namespace
kubectl apply -f k8s/namespace.yaml

# ConfigMap
kubectl apply -f k8s/configmap.yaml

# Training Job (also creates PVCs)
kubectl apply -f k8s/training-job.yaml
```

### 4d. Monitor training
```powershell
# Watch job status
kubectl get job cifar10-training -n ml-training -w

# Stream logs
kubectl logs -f job/cifar10-training -n ml-training

# Wait for completion
kubectl wait --for=condition=complete job/cifar10-training -n ml-training --timeout=3600s
```

### 4e. Deploy serving layer
```powershell
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4f. Verify deployment
```powershell
kubectl get pods -n ml-training
kubectl get deployment model-serving -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl get hpa -n ml-training
kubectl rollout status deployment/model-serving -n ml-training
```

### 4g. Test via port-forward
```powershell
# Start port-forward in background
kubectl port-forward svc/model-serving 8080:80 -n ml-training

# In another terminal:
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## 5. Submission Checklist

- [ ] GitHub repo `mlops-pytorch-pipeline` is public
- [ ] `develop` branch exists
- [ ] At least 4 merged PRs (2 per week) with meaningful descriptions
- [ ] All code merged to `main`
- [ ] README.md has architecture diagram and setup instructions
- [ ] Docker build + run screenshots in PR description
- [ ] Kubernetes pod/deployment screenshots in final PR
- [ ] Prediction endpoint response screenshot
- [ ] 300-500 word reflective write-up (in README.md)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pip install` fails on Windows | Run PowerShell as Administrator |
| Docker build fails (network) | Check Docker Desktop is running; retry |
| Minikube OOM | Increase `--memory` or reduce `batch_size` in configmap |
| Training very slow on CPU | Reduce `epochs` to 3-5 in `training_config_local.yaml` |
| Port 8080 already in use | `netstat -ano \| findstr :8080` then kill the PID |
| `kubectl` can't find cluster | Run `minikube start` first |
| Checkpoint not found in K8s | Ensure training Job completed: `kubectl get job -n ml-training` |
