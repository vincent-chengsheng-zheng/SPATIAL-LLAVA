# 05_IMPLEMENTATION_CHECKLIST.md

## Week-by-Week Implementation Plan

This is your execution roadmap: who does what, when, and success criteria.

**Read time:** 20-30 minutes  
**Audience:** All team members  
**Reference:** Use this document throughout execution

---

## Project Timeline

```
Total duration: 5 weeks
├─ Week 1: Foundation & Setup
├─ Week 2: Model Initialization  
├─ Week 3: Parallel Training (Coursework)
├─ Week 4: Parallel Training (Enterprise) + Evaluation
└─ Week 5: Demo & Deployment

Critical path: Stages 0, 1, 2 → (3a or 3b) → 4 → 5
Parallel work: After Stage 2, Members 3 & 4 can work independently
```

---

## WEEK 1: Foundation & Infrastructure

### Goal
Set up Docker, GitHub Actions, and data pipeline. Prepare environment for training.

---

### ✅ Member 1: Infrastructure Lead (5 days, 40 hours)

#### Day 1: Docker Setup (8 hours)

**Morning (4 hours): Create Dockerfile.dev**

```bash
# 1. Create directory structure
mkdir -p infrastructure/docker
touch infrastructure/docker/Dockerfile.dev
touch infrastructure/docker/Dockerfile.train
touch infrastructure/docker/Dockerfile.inference
touch infrastructure/docker/docker-compose.yml

# 2. Copy Dockerfile.dev code
# (Use code from 03_DOCKER_STRATEGY.md)
nano infrastructure/docker/Dockerfile.dev

# 3. Build and test
docker build -f infrastructure/docker/Dockerfile.dev -t spatial-llava:dev .
docker run -it --rm --gpus all spatial-llava:dev python --version
```

**Acceptance:**
- [ ] Dockerfile.dev exists and builds without errors
- [ ] Docker image is ~3.5 GB
- [ ] Can start container without GPU for initial test
- [ ] Imports work: `python -c "import torch; print(torch.__version__)"`

**Afternoon (4 hours): Create Dockerfile.train & Dockerfile.inference**

```bash
# Copy the other two Dockerfiles
nano infrastructure/docker/Dockerfile.train
nano infrastructure/docker/Dockerfile.train
nano infrastructure/docker/docker-compose.yml

# Build and quick test all three
docker build -f infrastructure/docker/Dockerfile.train -t spatial-llava:train .
docker build -f infrastructure/docker/Dockerfile.inference -t spatial-llava:inference .

# Verify all images exist
docker images | grep spatial-llava
```

**Acceptance:**
- [ ] All 3 Dockerfiles exist
- [ ] All 3 build successfully
- [ ] docker-compose.yml is valid YAML
- [ ] `docker-compose up -d dev` starts Jupyter

#### Day 2: docker-compose & Testing (8 hours)

**Morning (4 hours): Set up docker-compose**

```bash
# Test docker-compose configuration
docker-compose config  # Validates YAML syntax

# Start dev environment
docker-compose up -d dev

# Check Jupyter is running
docker-compose logs dev | grep "Jupyter"

# Test volume mounting
docker exec spatial-llava-dev ls /workspace | head

# Get Jupyter URL
docker-compose logs dev | grep "http://"
```

**Acceptance:**
- [ ] `docker-compose up -d dev` works
- [ ] Jupyter Lab accessible at http://localhost:8888
- [ ] Can access project files at /workspace in container
- [ ] GPU accessible in container (`nvidia-smi` works)

**Afternoon (4 hours): .dockerignore & optimization**

```bash
# Create .dockerignore
cat > .dockerignore << 'EOF'
.git
.github
__pycache__
*.pyc
*.pth
*.pkl
data/
results/
EOF

# Rebuild to test larger files don't bloat image
docker build -f infrastructure/docker/Dockerfile.dev -t spatial-llava:dev .

# Check size
docker images spatial-llava:dev --format "{{.Size}}"
```

**Acceptance:**
- [ ] .dockerignore file exists
- [ ] Docker images are reasonably sized (dev < 4GB, train < 9GB, inference < 3GB)

#### Day 3-4: GitHub Actions Setup (16 hours)

**Day 3 (8 hours): Create pr_checks.yml**

```bash
# Create workflow directory
mkdir -p .github/workflows

# Copy pr_checks.yml
# (Use code from 04_CI_CD_IMPLEMENTATION.md)
nano .github/workflows/pr_checks.yml

# Validate YAML
cat .github/workflows/pr_checks.yml | python -c "import sys, yaml; yaml.safe_load(sys.stdin)"
```

**Acceptance:**
- [ ] .github/workflows/pr_checks.yml exists
- [ ] Valid YAML syntax (no errors when parsed)
- [ ] Includes: linter, tests, Docker builds
- [ ] Triggered on pull_request to main/develop

**Day 4 (8 hours): Test CI/CD pipeline**

```bash
# Create test branch
git checkout -b test/ci-cd

# Make small test change
echo "# Test file" > test_ci.py

# Commit and push
git add test_ci.py
git commit -m "Test CI/CD"
git push origin test/ci-cd

# Create PR on GitHub
# Check: Actions should auto-run

# Monitor workflow
# GitHub → Actions → Watch workflow run

# If fails, fix and re-push
# If passes, delete test branch
```

**Acceptance:**
- [ ] pr_checks.yml runs automatically on PR
- [ ] All checks pass (lint, tests, Docker builds)
- [ ] Can see results on GitHub PR page
- [ ] Can merge once checks pass

#### Day 5: Final setup (8 hours)

```bash
# Setup requirements.txt and requirements_inference.txt
# (Provided in 03_DOCKER_STRATEGY.md)

# Test Docker builds with actual requirements
docker build -f infrastructure/docker/Dockerfile.inference -t spatial-llava:inference .

# Verify Model import works
docker run --rm spatial-llava:dev python -c "from core.model import SpatialLLaVA; print('OK')"
```

**Acceptance:**
- [ ] requirements.txt exists with all dependencies
- [ ] requirements_inference.txt exists (lightweight)
- [ ] All Dockerfiles install requirements without errors

**Member 1 Summary:**
- ✅ 3 Dockerfiles created and tested
- ✅ docker-compose.yml working
- ✅ GitHub Actions pr_checks.yml configured
- ✅ Team can: `docker-compose up -d dev` → Jupyter ready

---

### ✅ Member 2: Data Engineering (5 days, 40 hours)

#### Days 1-5: Data Preparation (Stage 1)

**Overview:**
```
Stage 1 Task: Download RefCOCO, preprocess, save as pickles
Input: Network (download from source)
Output: 3 files (refcoco_train.pkl, refcoco_val.pkl, refcoco_test.pkl)
Duration: 2-3 days of actual work + time waiting for downloads
```

**Work items:**

```python
# pipeline/stage_1_data_preparation.py (Main work)

1. Download RefCOCO (can be parallel while doing other work)
   ├─ Use: https://github.com/lichengunc/ref-coco/blob/master/data/instances_train+val.json
   ├─ Size: ~50 GB
   └─ Time: 1-2 hours (depending on internet)

2. Create data loader class
   class RefCOCODataset:
       def __init__(self, data_path, split='train'):
           # Load JSON
           # Handle coordinate normalization
       
       def __getitem__(self, idx):
           # Return: (image, text, bbox)

3. Implement preprocessing
   ├─ Image: Resize to 384x384, normalize
   ├─ Text: Tokenize with LLaVA tokenizer
   ├─ Bbox: Normalize to [0, 1], format as [x, y, w, h]

4. Create train/val/test split (80/10/10)

5. Save as pickle files for fast loading

6. Generate statistics (dataset_stats.json)
```

**Week 1 Tasks:**

```bash
# Day 1: Setup and data download
python pipeline/stage_1_data_preparation.py --download --output_dir data/

# Day 2-3: Implement preprocessing
# Write RefCOCODataset class
# Write image preprocessing
# Write text preprocessing

# Day 4-5: Testing and stats
# Load data from pickle
# Verify shapes and values
# Generate dataset_stats.json
```

**Acceptance Criteria:**

- [ ] RefCOCO dataset downloaded (50 GB)
- [ ] refcoco_train.pkl exists (~30 GB)
- [ ] refcoco_val.pkl exists (~4 GB)
- [ ] refcoco_test.pkl exists (~4 GB)
- [ ] Total samples: 120,000
- [ ] Dataset stats JSON matches expected ranges
- [ ] Can load batch from pickle: `loader = DataLoader(...); batch = next(iter(loader))`
- [ ] Batch shape: images (B, 3, 384, 384), text (B, 77), bbox (B, 4)

**Verification Commands:**

```bash
# Check files exist
ls -lh data/*.pkl

# Check can load
python -c "
import pickle
with open('data/refcoco_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f'Loaded {len(data)} samples')
    print(f'First sample keys: {data[0].keys()}')
"

# Check stats
cat data/dataset_stats.json
```

**Member 2 Summary:**
- ✅ RefCOCO dataset downloaded and preprocessed
- ✅ 3 pickle files ready for training
- ✅ Dataset statistics documented

---

### ✅ Members 3 & 4: Code Review (3 days, 24 hours total)

While Members 1-2 work:

```
Day 1: Review 01_ARCHITECTURE_OVERVIEW.md
       ├─ Understand 3 Dockerfiles
       ├─ Understand single-stage architecture
       └─ Questions? Ask in meeting

Day 2: Review 02_TRAINING_PIPELINE.md
       ├─ Read Stage 3-5 (your stages)
       ├─ Create your config files
       └─ Prepare training scripts

Day 3: Setup laptops + test Docker
       ├─ Clone repo
       ├─ Test `docker-compose up -d dev`
       ├─ Confirm Jupyter works
       └─ Read requirements, install locally
```

**Acceptance:**
- [ ] Understand the 5-stage pipeline
- [ ] Know what Stage 3a/3b will do
- [ ] Can start containers locally
- [ ] No blockers identified

---

### ✅ Team Meeting: Friday EOD (1 hour)

**Agenda:**
1. Member 1 demo: Docker setup works
2. Member 2 demo: Data loading works
3. Member 3 & 4: Questions & readiness for Week 2
4. Blockers?
5. Next week assignments

**Success:** All systems ready, no data corruption, GitHub Actions working

---

## WEEK 2: Model Initialization (Stage 2)

### Goal
Load LLaVA-7B, attach LoRA, add regression head, verify model works.

---

### ✅ Member 1: Stage 2 Implementation (2-3 days, 20 hours)

**Task: stage_2_model_initialization.py**

```python
# pipeline/stage_2_model_initialization.py

1. Load LLaVA-7B from Hugging Face
   model = AutoModel.from_pretrained("liuhaotian/llava-v1.5-7b")

2. Freeze vision encoder + LLM base

3. Add LoRA to LLM
   from peft import get_peft_model, LoraConfig
   
   config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=['q_proj', 'v_proj'],
       lora_dropout=0.05
   )
   model = get_peft_model(model, config)

4. Add [LOC] special token
   tokenizer.add_tokens(['[LOC]'])

5. Create MLP regression head
   class RegressionHead(nn.Module):
       def __init__(self, input_dim=4096):
           self.fc1 = nn.Linear(input_dim, 512)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(512, 4)
           self.sigmoid = nn.Sigmoid()
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.sigmoid(self.fc2(x))
           return x

6. Calculate parameters & memory
   print(f"Total params: 7B")
   print(f"Trainable params: 5.2M")
   print(f"GPU memory: 7.8 GB")

7. Save checkpoint
```

**Acceptance Criteria:**

- [ ] Model loads without errors
- [ ] Trainable parameters < 10M
- [ ] Forward pass works: model(image, text) → bbox
- [ ] Output shape is (batch_size, 4)
- [ ] [LOC] token in vocabulary
- [ ] Checkpoint can be saved/loaded
- [ ] GPU memory estimate < 8GB

**Verification:**

```bash
python pipeline/stage_2_model_initialization.py \
  --output_dir checkpoints/test/ \
  --test_only

# Expected output:
# ✓ Model loaded
# ✓ Total parameters: 7,000,000,000
# ✓ Trainable parameters: 5,242,880
# ✓ [LOC] token added
# ✓ Forward pass successful
# ✓ Output shape: (8, 4)
# ✓ GPU memory: 7.8 GB
```

---

### ✅ Members 3 & 4: Prepare Training Scripts (3-4 days, 20-24 hours)

While Member 1 works on Stage 2:

**Member 3 (Coursework):**

```bash
# Create config
mkdir -p courses/coursework_track
nano courses/coursework_track/config_coursework.yaml

# Copy from 02_TRAINING_PIPELINE.md:
training:
  num_epochs: 3
  batch_size: 8
  max_steps: 36000
  eval_interval: 500

# Create training script skeleton
nano courses/coursework_track/train_coursework.py

# Should contain:
# - Load config
# - Load data (use Member 2's loader)
# - Load model (use Member 1's Stage 2)
# - Training loop
# - Save checkpoints
```

**Member 4 (Enterprise):**

```bash
# Similar to Member 3, but:
training:
  num_epochs: 10
  batch_size: 16
  max_steps: 60000

# Plus: Prepare baseline model for comparison
# (will need to implement 3 baseline variants in Stage 4)
```

**Acceptance:**

- [ ] config.yaml files created
- [ ] Training script skeleton compiles
- [ ] Can load data
- [ ] Can load model
- [ ] Ready to add training loop (Stage 3)

---

## WEEK 3: Parallel Training Phase 1 (Stage 3a - Coursework)

### Goal
Train model on coursework configuration. Achieve validation IoU ≥ 0.60.

---

### ✅ Member 3: Training (Stage 3a)

**Task: Complete train_coursework.py**

```bash
# Start training
python courses/coursework_track/train_coursework.py \
  --config courses/coursework_track/config_coursework.yaml \
  --output_dir checkpoints/coursework/ \
  --data_dir data/

# Monitor progress
# Should see loss decrease, validation IoU increase
```

**Expected progress:**

```
Epoch 1/3:
├─ Step 0: Loss = 0.85, Val IoU = 0.25
├─ Step 500: Loss = 0.42, Val IoU = 0.48
├─ Step 1000: Loss = 0.38, Val IoU = 0.52
├─ Epoch complete: Loss = 0.32, Val IoU = 0.57 ✓

Epoch 2/3:
├─ Best Val IoU: 0.64 (checkpoint saved)

Epoch 3/3:
├─ Final Val IoU: 0.67 ✓
```

**Checkpoints to save:**
- [ ] best.pth (highest validation IoU)
- [ ] latest.pth (last epoch)
- [ ] Training log with metrics

**Week 3 Deliverables:**
- [ ] Training completes all 3 epochs
- [ ] Loss decreases monotonically
- [ ] Validation IoU ≥ 0.60
- [ ] Best model checkpoint saved

---

### ✅ Member 4: Parallel Work

While Member 3 is training coursework:

**Option A: Start enterprise training (if GPU pool available)**

```bash
python courses/enterprise_track/train_enterprise.py \
  --config courses/enterprise_track/config_enterprise.yaml \
  --output_dir checkpoints/enterprise/ \
  --data_dir data/
```

**Option B: Prepare evaluation & baseline scripts**

```bash
# Prepare Stage 4b tasks:
nano courses/enterprise_track/eval_enterprise.py

# Create baseline implementations:
# 1. Standard LLaVA (text coordinate output)
# 2. Regression head only (frozen LLM)
# 3. Your model (LoRA + head)
```

---

## WEEK 4: Parallel Training Phase 2 (Stage 3b) + Evaluation (Stage 4)

### Goal
Complete enterprise training. Evaluate both models. Compare baselines.

---

### ✅ Member 3: Stage 4a - Evaluation (Coursework)

```bash
python courses/coursework_track/eval_coursework.py \
  --model_path checkpoints/coursework/best.pth \
  --test_set data/refcoco_test.pkl \
  --output_dir results/coursework/

# Output: results/coursework/metrics.json
# Should contain:
# {
#   "test_iou": 0.65,
#   "test_rmse": 0.045,
#   "test_mae": 0.032
# }
```

**Acceptance:**
- [ ] Evaluation completes without error
- [ ] Test IoU ≥ 0.60 (coursework target)
- [ ] RMSE ≤ 0.050
- [ ] Metrics saved as JSON

**Then: Stage 5a - Gradio Demo**

```bash
python courses/coursework_track/demo_gradio.py \
  --model_path checkpoints/coursework/best.pth

# Should be accessible at http://localhost:7860
# Can upload image + enter prompt
# Should see bounding box on output image
```

**Acceptance:**
- [ ] Gradio demo launches
- [ ] Can upload image
- [ ] Can enter prompt
- [ ] Prediction displays with bbox drawn
- [ ] Bounding box coordinates seem reasonable

---

### ✅ Member 4: Stage 3b Completion + Stage 4b

**If not done: Complete training**
```bash
# Continue or resume training
python courses/enterprise_track/train_enterprise.py ...
```

**Stage 4b: Baseline Comparison**

```bash
python courses/enterprise_track/eval_enterprise.py \
  --model_path checkpoints/enterprise/best.pth \
  --test_set data/refcoco_test.pkl \
  --output_dir results/enterprise/ \
  --compare_baselines

# Output: results/enterprise/baseline_comparison.json
# Should show your model is 40-50% better than baseline 1
```

**Acceptance:**
- [ ] Baseline 1 evaluation complete (test IoU ≈ 0.45)
- [ ] Baseline 2 evaluation complete (test IoU ≈ 0.58)
- [ ] Your model (baseline 3) evaluated (test IoU ≥ 0.62)
- [ ] Improvement documented

---

## WEEK 5: Demo & Deployment

### Goal
Deploy demos/APIs. Complete all deliverables. Project ready for submission.

---

### ✅ Member 4: Stage 5b - FastAPI Deployment

```bash
# Create FastAPI server
# (From 02_TRAINING_PIPELINE.md Stage 5b)

nano courses/enterprise_track/deployment/fastapi_server.py

# Test locally
python courses/enterprise_track/deployment/fastapi_server.py

# In another terminal:
curl -X POST "http://localhost:8000/predict" \
  -F "image=@test.jpg" \
  -F "prompt=find the person"

# Response should include bbox, confidence, inference_time
```

**Acceptance:**
- [ ] FastAPI server starts on port 8000
- [ ] /health endpoint responds
- [ ] /predict endpoint accepts image + prompt
- [ ] Returns JSON with bbox, confidence, latency
- [ ] Inference time documented

**Optional: Docker for inference**

```bash
# Test Docker container
docker build -f infrastructure/docker/Dockerfile.inference \
  -t spatial-llava:inference .

docker run -p 8000:8000 \
  -v $(pwd)/checkpoints/enterprise/best.pth:/app/model.pth \
  spatial-llava:inference

# Should work identically to local
```

---

### ✅ All Members: Final Integration (Days 1-3)

**Monday:**
- [ ] Member 1: Verify all code integrates
- [ ] Member 2: Finalize dataset documentation
- [ ] Member 3: Polish Gradio demo
- [ ] Member 4: Polish FastAPI API

**Tuesday:**
- [ ] Run full pipeline locally (Stages 0-5)
- [ ] Verify all metrics/results
- [ ] Update README with results

**Wednesday:**
- [ ] Final testing on Docker
- [ ] GitHub Actions all green
- [ ] Submit project

---

## Dependencies & Blocking

### Critical Path

```
Stage 0 (Member 1)
  └─→ Stage 1 (Member 2)
      └─→ Stage 2 (Member 1)
          ├─→ Stage 3a (Member 3)  [Can start after Stage 2]
          └─→ Stage 3b (Member 4)  [Can start after Stage 2]
              ├─→ Stage 4a (Member 3)
              │   └─→ Stage 5a (Member 3)
              └─→ Stage 4b (Member 4)
                  └─→ Stage 5b (Member 4)

After Stage 2: Members 3 & 4 work INDEPENDENTLY (no blocking)
```

### Risk Mitigation

**If Stage 1 (data) is slow:**
- Use pre-processed reference implementation
- Can download subset (100 samples) to start training early

**If Stage 2 (model) fails:**
- Fallback: Use unmodified LLaVA for baseline

**If training (Stage 3) doesn't converge:**
- Adjust hyperparameters (learning rate, batch size)
- Use mixed precision (fp16)
- Reduce model size temporarily

---

## Success Metrics by Week

### Week 1: Foundation ✅
- [ ] Docker: All 3 images build & run
- [ ] GitHub Actions: pr_checks.yml works
- [ ] Data: 120,000 samples loaded & verified
- [ ] All team members can develop locally

### Week 2: Model Ready ✅
- [ ] Model: LLaVA + LoRA + regression head initialized
- [ ] Trainable parameters < 10M
- [ ] Forward pass works
- [ ] GPU memory < 8GB
- [ ] Training scripts ready to run

### Week 3: Coursework Training ✅
- [ ] Training starts & converges
- [ ] Validation IoU increases over epochs
- [ ] Reaches target IoU ≥ 0.60
- [ ] Checkpoints saved

### Week 4: Evaluation ✅
- [ ] Coursework: Test IoU ≥ 0.60, RMSE ≤ 0.050
- [ ] Enterprise: Test IoU ≥ 0.62
- [ ] Baseline comparison shows improvement
- [ ] Gradio demo works

### Week 5: Deployment ✅
- [ ] Coursework: Gradio demo running
- [ ] Enterprise: FastAPI server running
- [ ] Both containerized
- [ ] All metrics documented
- [ ] Project submission ready

---

## Quick Reference: Commands

```bash
# Check environment
python pipeline/stage_0_environment.py --config pipeline/pipeline_config.yaml

# Load data
python pipeline/stage_1_data_preparation.py --download

# Initialize model
python pipeline/stage_2_model_initialization.py

# Train coursework
python courses/coursework_track/train_coursework.py \
  --config courses/coursework_track/config_coursework.yaml

# Train enterprise
python courses/enterprise_track/train_enterprise.py \
  --config courses/enterprise_track/config_enterprise.yaml

# Evaluate coursework
python courses/coursework_track/eval_coursework.py \
  --model_path checkpoints/coursework/best.pth

# Evaluate enterprise
python courses/enterprise_track/eval_enterprise.py \
  --model_path checkpoints/enterprise/best.pth \
  --compare_baselines

# Demo coursework
python courses/coursework_track/demo_gradio.py \
  --model_path checkpoints/coursework/best.pth

# Deploy enterprise
python courses/enterprise_track/deployment/fastapi_server.py
```

---

## Common Issues & Solutions

### Docker Issues

**"CUDA out of memory"**
- Reduce batch size (8 → 4)
- Use fp16 training
- Reduce model size

**"Could not load CUDA"**
- Verify nvidia-docker installed
- Check CUDA 12.1 drivers
- Run on different GPU

### Training Issues

**"Loss not decreasing"**
- Check learning rate (default 1e-4 good)
- Verify data loading
- Check loss function

**"NaN loss"**
- Check input ranges
- Verify gradients
- Reduce learning rate

### GitHub Actions Issues

**"Tests timeout"**
- Increase timeout_minutes
- Optimize test suite
- Skip expensive tests on PR

**"Docker build OOM"**
- Build on machine with more RAM
- Use --squash flag
- Build locally instead

---

## Team Communication

**Daily standup:** 10 minutes each morning
- What did I do yesterday?
- What am I doing today?
- Any blockers?

**Weekly meeting:** Friday EOD
- Progress review
- Next week planning
- Issue resolution

**GitHub Issues:** For bugs/questions
- Create issue
- Assign to relevant member
- Link to PR when fixing

---

## Final Checklist

Before final submission:

- [ ] All code committed to GitHub
- [ ] All workflows passing (GitHub Actions green)
- [ ] All metrics documented
- [ ] README updated with results
- [ ] Docker images build cleanly
- [ ] Demos/APIs working
- [ ] No large files in repo (.gitignore correct)
- [ ] Documentation complete

---

## Next Steps

**Now:**
1. All team members read 00_README.md through 01_ARCHITECTURE_OVERVIEW.md
2. Schedule kickoff meeting
3. Member 1 begins Week 1 tasks

**Questions?**
- Review relevant documentation section
- Ask in standup meeting
- Create GitHub issue

Good luck! 🚀
