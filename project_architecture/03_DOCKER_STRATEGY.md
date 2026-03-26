# 03_DOCKER_STRATEGY.md

## Docker Implementation: 3 Dockerfiles + docker-compose

This document provides complete, copy-paste-ready Docker configurations.

**Read time:** 45-60 minutes  
**Audience:** Primarily Member 1 (infrastructure lead)  
**Next document:** 04_CI_CD_IMPLEMENTATION.md

---

## Overview: Which Dockerfile for Which Use?

| Use Case | File | When | Who |
|----------|------|------|-----|
| **Local development** | `Dockerfile.dev` | Every day (weeks 1-5) | All members |
| **GPU training** | `Dockerfile.train` | Weeks 3-4 | Members 3, 4 |
| **Production inference** | `Dockerfile.inference` | Weeks 4-5 (Enterprise) | Deployment |

---

## File 1: Dockerfile.dev

**Purpose:** Development environment with Jupyter Lab  
**Image size:** ~3.5 GB  
**Build time:** ~10 minutes

```dockerfile
# infrastructure/docker/Dockerfile.dev
FROM pytorch/pytorch:2.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt --no-cache-dir

# Install Jupyter and development tools
RUN pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipykernel==6.26.0 \
    tensorboard==2.15.1 \
    --no-cache-dir

# Create Jupyter config directory
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Expose TensorBoard port
EXPOSE 6006

# Expose development server port
EXPOSE 8000

# Start Jupyter Lab
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--allow-root", \
     "--no-browser", \
     "--NotebookApp.token=''"]
```

**How to use:**
```bash
# Build
docker build -f infrastructure/docker/Dockerfile.dev -t spatial-llava:dev .

# Run with GPU and volume mount
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -p 6006:6006 \
  spatial-llava:dev

# Open browser: http://localhost:8888
```

---

## File 2: Dockerfile.train

**Purpose:** GPU training environment  
**Image size:** ~8 GB  
**Build time:** ~20 minutes

```dockerfile
# infrastructure/docker/Dockerfile.train
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

# Install Python and system tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    curl \
    build-essential \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyTorch with CUDA support
RUN pip install --upgrade pip && \
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

# Copy and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt --no-cache-dir

# Install additional training tools
RUN pip install \
    wandb==0.16.0 \
    tensorboard==2.15.1 \
    --no-cache-dir

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/results

# TensorBoard port
EXPOSE 6006

# Default entry point (can be overridden)
ENTRYPOINT ["python"]

# Default command (can be overridden)
CMD ["pipeline/stage_3_training.py"]
```

**How to use:**
```bash
# Build
docker build -f infrastructure/docker/Dockerfile.train -t spatial-llava:train .

# Run training (coursework)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd):/workspace/code \
  spatial-llava:train \
  courses/coursework_track/train_coursework.py \
  --config courses/coursework_track/config_coursework.yaml

# Run training (enterprise)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  spatial-llava:train \
  courses/enterprise_track/train_enterprise.py \
  --config courses/enterprise_track/config_enterprise.yaml
```

---

## File 3: Dockerfile.inference

**Purpose:** Production API service  
**Image size:** ~2.5 GB  
**Build time:** ~10 minutes

```dockerfile
# infrastructure/docker/Dockerfile.inference
FROM pytorch/pytorch:2.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy lightweight dependencies
COPY requirements_inference.txt /tmp/
RUN pip install -r /tmp/requirements_inference.txt --no-cache-dir

# Copy application code
COPY core /app/core
COPY courses/enterprise_track/deployment /app/deployment

# Copy model weights (optional, can also mount as volume)
# COPY checkpoints/enterprise/best.pth /app/model.pth

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start FastAPI server
CMD ["python", "/app/deployment/fastapi_server.py", \
     "--port", "8000"]
```

**How to use:**
```bash
# Build
docker build -f infrastructure/docker/Dockerfile.inference -t spatial-llava:inference .

# Run with model weights as volume
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints/enterprise/best.pth:/app/model.pth \
  spatial-llava:inference

# Test API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -F "image=@test.jpg" \
  -F "prompt=find the person"
```

---

## docker-compose.yml: Local Development Orchestration

**Purpose:** Manage all 3 containers locally  
**Usage:** One command to start development environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Development environment (Jupyter)
  dev:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.dev
    image: spatial-llava:dev
    container_name: spatial-llava-dev
    
    volumes:
      - .:/workspace                          # Mount entire project
      - ~/.cache/huggingface:/root/.cache/huggingface  # Cache models
    
    ports:
      - "8888:8888"  # Jupyter Lab
      - "6006:6006"  # TensorBoard
    
    environment:
      - CUDA_VISIBLE_DEVICES=0               # Use GPU 0
      - JUPYTER_ENABLE_LAB=yes
    
    networks:
      - spatial-llava-net
    
    runtime: nvidia                           # Enable GPU support
    
    stdin_open: true
    tty: true


  # Training environment (GPU-intensive)
  train:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.train
    image: spatial-llava:train
    container_name: spatial-llava-train
    
    volumes:
      - .:/workspace
      - ~/.cache/huggingface:/root/.cache/huggingface
    
    environment:
      - CUDA_VISIBLE_DEVICES=0,1             # Use GPU 0, 1 if available
    
    networks:
      - spatial-llava-net
    
    runtime: nvidia
    
    # Don't start automatically, only when needed
    profiles:
      - training


  # Inference server (API)
  inference:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.inference
    image: spatial-llava:inference
    container_name: spatial-llava-api
    
    volumes:
      - ./checkpoints/enterprise/best.pth:/app/model.pth
    
    ports:
      - "8000:8000"  # FastAPI
    
    environment:
      - CUDA_VISIBLE_DEVICES=2               # Use GPU 2 (separate from dev/train)
    
    networks:
      - spatial-llava-net
    
    runtime: nvidia
    
    # Restart if crashes
    restart: unless-stopped
    
    # Only start when explicitly requested
    profiles:
      - inference
    
    depends_on:
      - dev  # Just to ensure dev is created first


networks:
  spatial-llava-net:
    driver: bridge
```

**Usage:**

```bash
# Start development environment only
docker-compose up -d dev

# Check Jupyter token
docker-compose logs -f dev | grep "token="

# Start training environment
docker-compose --profile training up -d train

# Start inference API
docker-compose --profile inference up -d inference

# Start all (dev will run, others available)
docker-compose up -d

# View logs
docker-compose logs -f dev
docker-compose logs -f train
docker-compose logs -f inference

# Stop everything
docker-compose down

# Stop only inference
docker-compose --profile inference down
```

---

## .dockerignore: Exclude Unnecessary Files

**Purpose:** Reduce Docker image size and build time

```
# .dockerignore
.git
.github
.gitignore
.dockerignore
.env
.vscode
.idea

__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
*.egg-info
build/
dist/

# Model checkpoints (too large, mount instead)
*.pth
*.pt
*.safetensors

# Data (too large, mount instead)
data/
*.pkl
*.csv

# Results (can regenerate)
results/

# OS files
.DS_Store
Thumbs.db
*.swp
*.swo

# Documentation (not needed in image)
*.md
docs/

# Test files
tests/
*.test.py
```

---

## Requirements.txt vs Requirements_Inference.txt

### Full Dependencies (requirements.txt)

```
# requirements.txt

# Core ML libraries
torch==2.0.1
torchvision==0.15.2
transformers==4.35.2
peft==0.7.1
bitsandbytes==0.41.1

# Data processing
datasets==2.14.6
scikit-learn==1.3.2

# Utilities
numpy==1.24.3
pillow==10.1.0
opencv-python==4.8.1.78
tqdm==4.66.1

# Training utilities
tensorboard==2.15.1
wandb==0.16.0

# Web frameworks
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# Development
jupyter==1.0.0
jupyterlab==4.0.9
ipykernel==6.26.0
ipython==8.17.2

# Gradio (for coursework demo)
gradio==4.17.0

# Code quality
pytest==7.4.3
pytest-cov==4.1.0
flake8==6.1.0
black==23.12.0

# Other
python-dotenv==1.0.0
pyyaml==6.0.1
```

### Lightweight Inference (requirements_inference.txt)

```
# requirements_inference.txt
# (Only what's needed for inference)

torch==2.0.1
torchvision==0.15.2
transformers==4.35.2
peft==0.7.1

# Minimal data processing
numpy==1.24.3
pillow==10.1.0

# Web server
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# Optional: for monitoring
pydantic==2.5.0
```

**Size difference:**
- requirements.txt: ~20+ packages
- requirements_inference.txt: ~8 packages
- Image size reduction: Full 3.5GB → Lightweight 2.5GB

---

## Build and Test

### Build All Images

```bash
# Build dev image
docker build -f infrastructure/docker/Dockerfile.dev \
  -t spatial-llava:dev .

# Build train image
docker build -f infrastructure/docker/Dockerfile.train \
  -t spatial-llava:train .

# Build inference image
docker build -f infrastructure/docker/Dockerfile.inference \
  -t spatial-llava:inference .

# Verify all built
docker images | grep spatial-llava
```

### Test Each Image

```bash
# Test dev (Jupyter should start)
docker run --rm --gpus all \
  -p 8888:8888 \
  spatial-llava:dev \
  jupyter --version

# Test train (model should load)
docker run --rm --gpus all \
  spatial-llava:train \
  python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test inference (FastAPI should start)
docker run --rm -p 8000:8000 \
  spatial-llava:inference \
  python -c "import fastapi; print('FastAPI ready')"
```

### Troubleshooting

```
Issue: "docker: command not found"
Solution: Install Docker

Issue: "Could not load CUDA"
Solution: 
  - Install nvidia-docker (for GPU support)
  - Add --gpus all flag

Issue: "Permission denied"
Solution:
  - Run with sudo, or
  - Add user to docker group: sudo usermod -aG docker $USER

Issue: "Image too large"
Solution:
  - Check .dockerignore is excluding large files
  - Use --squash flag to reduce layer count
  - Use multi-stage builds
```

---

## GPU Memory Management

### Memory Usage by Container

```
dev (Jupyter):
  ├─ PyTorch libraries: ~1.5 GB
  ├─ LLaVA model (when loaded): ~7.0 GB
  └─ Total available for notebooks: ~0.5 GB

train (Training):
  ├─ PyTorch libraries: ~1.5 GB
  ├─ LLaVA model: ~7.0 GB
  ├─ Batch size 8: ~0.5 GB
  └─ Optimizer state: ~0.3 GB
  ├─ Total: ~9.3 GB (exceeds 8GB RTX 2070!)

inference (Inference only):
  ├─ PyTorch libraries: ~1.5 GB
  ├─ LLaVA model: ~7.0 GB
  ├─ Inference batch: ~0.3 GB
  └─ Total: ~8.8 GB (tight fit)
```

### Solutions for RTX 2070 (8GB VRAM)

```
Option 1: Reduce batch size
├─ Batch size 8 → 4
└─ Saves ~0.3 GB

Option 2: Use fp16 (float16) training
├─ Reduces model size by half
└─ Saves ~3.5 GB
└─ Trade-off: Slightly lower precision

Option 3: Gradient checkpointing
├─ Recompute activations instead of storing
└─ Saves memory but slower

Option 4: Quantization (post-training)
├─ Convert to int8
└─ 8GB model → 2GB model
└─ But requires calibration
```

---

## Volume Mounting Strategy

```yaml
# docker-compose.yml volume mounts explained

volumes:
  - .:/workspace                    # Mount entire project
    # Effect: Can edit code locally, see changes in container
    
  - ~/.cache/huggingface:/root/.cache/huggingface
    # Effect: Don't re-download 7B model each time
    # Saves 30+ minutes of download time
    
  - ./data:/workspace/data
    # Effect: Share dataset (on disk) not in image
    # Keeps image small, data flexible
    
  - ./checkpoints:/workspace/checkpoints
    # Effect: Save trained models outside container
    # Can access trained model on host system
```

**Best practice:**
- Code in volume (edit locally, test in container)
- Data in volume (too large for Docker image)
- Models in volume (too large for Docker image)
- Libraries in image (stable, no need to remount)

---

## Docker Best Practices

### Image Naming Convention
```
spatial-llava:dev          ← For development
spatial-llava:train        ← For training
spatial-llava:inference    ← For inference
spatial-llava:latest       ← Production (Enterprise track)
spatial-llava:v1.0.0       ← Tagged release
```

### Layer Caching Optimization
```dockerfile
# ❌ Bad: Changes to code invalidate everything
FROM pytorch/pytorch:2.0
COPY . /workspace
RUN pip install -r requirements.txt

# ✅ Good: Dependencies cached separately
FROM pytorch/pytorch:2.0
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY . /workspace
```

**Why?** If code changes, Docker rebuilds from `COPY . /workspace` onward, reusing cached layers before that.

### Health Checks
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

Allows Docker to detect and restart unhealthy containers automatically.

---

## Pushing to Docker Hub (Optional)

If you want to share images or deploy from Docker Hub:

```bash
# Tag image
docker tag spatial-llava:inference \
  yourusername/spatial-llava:latest

# Login to Docker Hub
docker login

# Push
docker push yourusername/spatial-llava:latest

# On another machine, pull
docker pull yourusername/spatial-llava:latest
```

---

## Next Steps

→ Read **04_CI_CD_IMPLEMENTATION.md** for automated testing with GitHub Actions

This explains how to automatically validate Docker builds and code quality.

---

## Quick Command Reference

```bash
# Build
docker build -f infrastructure/docker/Dockerfile.dev -t spatial-llava:dev .

# Run
docker run -it --gpus all -v $(pwd):/workspace spatial-llava:dev

# With docker-compose
docker-compose up -d dev
docker-compose logs -f dev
docker-compose down

# Check image size
docker images spatial-llava

# Remove old images
docker rmi spatial-llava:dev
```
