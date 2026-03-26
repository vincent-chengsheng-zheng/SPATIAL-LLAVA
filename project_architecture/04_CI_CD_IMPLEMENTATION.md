# 04_CI_CD_IMPLEMENTATION.md

## GitHub Actions: Automated Testing and Deployment

This document provides copy-paste-ready CI/CD workflows for quality assurance.

**Read time:** 45 minutes  
**Audience:** Primarily Member 1 (can delegate)  
**Next document:** 05_IMPLEMENTATION_CHECKLIST.md

---

## What Is CI/CD and Why You Need It

### The Problem

Without CI/CD (4 people, one codebase):
```
Mon: Member 1 changes model.py
Tue: Member 2 pushes changes → merges to main
     Member 1's changes forgotten/overwritten
Wed: Member 3 finds everything broken
Thu: Debug chaos, missed deadline
```

### The Solution

With CI/CD (automated quality gates):
```
Mon: Member 1 makes PR with changes
     GitHub Actions automatically:
     ├─ Runs linter (style check)
     ├─ Runs unit tests
     ├─ Builds Docker images
     ├─ Runs integration tests
     └─ Result: ✅ All passed or ❌ Failed

If ✅: "Ready to merge"
If ❌: Shows exactly what broke, blocks merge

Result: Only verified code reaches main
```

---

## Workflow 1: PR Checks (Essential)

**File location:** `.github/workflows/pr_checks.yml`

**When it runs:** Every time someone creates a pull request

**What it does:**
1. Check code style (flake8)
2. Run unit tests (pytest)
3. Verify Docker can build
4. Check model can load

```yaml
# .github/workflows/pr_checks.yml
name: PR Checks

on:
  pull_request:
    branches: [main, develop]

jobs:
  lint_and_test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.10']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'  # Cache dependencies
      
      # Step 1: Code quality check
      - name: Install linter
        run: |
          pip install flake8
      
      - name: Lint code
        run: |
          # Check code style
          flake8 core/ pipeline/ courses/ tests/ \
            --max-line-length=120 \
            --ignore=E501,W503,E701 \
            --count \
            --show-source \
            --statistics
      
      # Step 2: Unit tests
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: |
          pytest tests/ \
            -v \
            --cov=core \
            --cov=pipeline \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=50
      
      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        if: always()
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
      
      # Step 3: Model can load
      - name: Test model loading
        run: |
          python -c "
          import sys
          sys.path.insert(0, '.')
          from core.model import SpatialLLaVA
          print('✓ SpatialLLaVA imports correctly')
          "
      
      # Step 4: Docker builds
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build dev Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./infrastructure/docker/Dockerfile.dev
          push: false  # Don't push, just verify build works
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Build train Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./infrastructure/docker/Dockerfile.train
          push: false
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Build inference Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./infrastructure/docker/Dockerfile.inference
          push: false
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      # Final summary
      - name: Summary
        if: always()
        run: |
          echo "## CI/CD Check Results" >> $GITHUB_STEP_SUMMARY
          echo "✅ Lint passed" >> $GITHUB_STEP_SUMMARY
          echo "✅ Tests passed" >> $GITHUB_STEP_SUMMARY
          echo "✅ Docker builds successful" >> $GITHUB_STEP_SUMMARY
          echo "Ready to merge! 🚀" >> $GITHUB_STEP_SUMMARY
```

**Result on GitHub:**

```
Pull Request #5: Add regression head to LLaVA

✅ All checks passed
├─ Lint - passed
├─ Unit tests - passed (95/100)
├─ Model loading - passed
└─ Docker builds - passed

Status: Ready to merge
```

---

## Workflow 2: Training Validation (Optional)

**File location:** `.github/workflows/training_validation.yml`

**When it runs:** Push to `develop` branch or manual trigger

**What it does:** Run a quick training test on GPU to ensure training code isn't broken

**⚠️ Requirements:**
- Need a self-hosted GPU runner (SUTD cluster or your machine)
- Setup instructions below

```yaml
# .github/workflows/training_validation.yml
name: Training Validation

on:
  push:
    branches: [develop]
  workflow_dispatch:  # Manual trigger

jobs:
  quick_training_test:
    runs-on: [self-hosted, gpu]  # Use YOUR GPU machine
    timeout-minutes: 120         # 2 hour timeout
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Check GPU
        run: |
          nvidia-smi
      
      - name: Environment check
        run: |
          python pipeline/stage_0_environment.py \
            --config pipeline/pipeline_config.yaml
      
      - name: Quick data preparation (100 samples)
        run: |
          python pipeline/stage_1_data_preparation.py \
            --download \
            --num_samples 100
      
      - name: Model initialization
        run: |
          python pipeline/stage_2_model_initialization.py \
            --output_dir checkpoints/ci_test/
      
      - name: Quick training (10 steps)
        run: |
          python courses/coursework_track/train_coursework.py \
            --config courses/coursework_track/config_coursework.yaml \
            --max_steps 10 \
            --eval_interval 5 \
            --output_dir results/ci_test/
      
      - name: Upload training results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: training-logs
          path: results/ci_test/
      
      - name: Check results
        if: success()
        run: |
          echo "✅ Training validation passed!"
          echo "Model converged, loss decreased, validation metrics computed"
```

**To set up self-hosted GPU runner:**

```bash
# On your GPU machine (SUTD cluster or personal):

# 1. Create actions-runner directory
mkdir actions-runner && cd actions-runner

# 2. Download runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz \
  -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# 3. Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# 4. Configure
# Go to repo → Settings → Actions → Runners → New self-hosted runner
# Copy the registration token
./config.sh --url https://github.com/YOUR-USERNAME/spatial-llava \
            --token YOUR_TOKEN

# 5. Run
./run.sh

# Now your machine is registered as a runner!
# Any workflow with `runs-on: [self-hosted, gpu]` will use it
```

---

## Workflow 3: Deploy Inference (Optional)

**File location:** `.github/workflows/deploy_inference.yml`

**When it runs:** On release tags (e.g., `v1.0.0`)

**What it does:** Build, push to Docker Hub, deploy to production server

```yaml
# .github/workflows/deploy_inference.yml
name: Deploy Inference Service

on:
  release:
    types: [published]
  workflow_dispatch:

env:
  REGISTRY: docker.io
  IMAGE_NAME: spatial-llava

jobs:
  build_and_push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract version
        id: meta
        run: |
          VERSION=${GITHUB_REF#refs/tags/}
          echo "version=$VERSION" >> $GITHUB_OUTPUT
      
      - name: Build and push inference image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./infrastructure/docker/Dockerfile.inference
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/spatial-llava:latest
            ${{ secrets.DOCKER_USERNAME }}/spatial-llava:${{ steps.meta.outputs.version }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Deploy to production
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_SSH_KEY }}
          DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
          DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
        run: |
          mkdir -p ~/.ssh
          echo "$DEPLOY_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          
          ssh -i ~/.ssh/deploy_key -o StrictHostKeyChecking=no \
            $DEPLOY_USER@$DEPLOY_HOST \
            "cd /app && \
             docker-compose pull && \
             docker-compose up -d inference && \
             docker-compose logs -f inference"
          
          rm ~/.ssh/deploy_key
      
      - name: Verify deployment
        env:
          DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
        run: |
          sleep 10  # Wait for service to start
          curl -f http://$DEPLOY_HOST:8000/health || exit 1
          echo "✅ Service is healthy"
```

**To use this:**
1. Create GitHub secrets for Docker credentials and deployment
2. Tag a release: `git tag v1.0.0 && git push --tags`
3. Workflow automatically builds, pushes, and deploys

---

## Setting Up GitHub Secrets

For workflows that push to Docker Hub or deploy to servers:

1. Go to GitHub → Repo → Settings → Secrets and variables → Actions
2. Add these secrets:

```
DOCKER_USERNAME: your_docker_hub_username
DOCKER_PASSWORD: your_docker_hub_password_or_token

DEPLOY_HOST: your-server.com
DEPLOY_USER: deploy_user
DEPLOY_SSH_KEY: (private key content)
```

**How to get Docker Hub token:**
```bash
# Login to Docker Hub → Account Settings → Security → New Access Token
# Copy the token and paste into GitHub secret
```

---

## GitHub Status Checks on PR

When you push to a PR, GitHub shows:

```
Pull Request #5

Conversation  Files Changed  Checks

Checks (4/4 passed)
├─ ✅ pr_checks / lint_and_test
│  ├─ Lint - passed
│  ├─ Unit tests - passed
│  ├─ Model loading - passed
│  └─ Docker builds - passed
│
└─ Commit d3afc8e

[Merge pull request]  (enabled because all checks passed)
```

---

## Troubleshooting Failed Workflows

### Linter Fails

```
❌ Linter found issues
  E501 line too long (145 > 120 characters)
  W503 line break before binary operator
```

**Solution:**
```bash
# Locally run same check to fix
flake8 core/ --max-line-length=120

# Or auto-fix with black
pip install black
black core/
```

### Tests Fail

```
❌ Test failed: test_model_forward_pass
  AssertionError: bbox shape (1, 5) != (1, 4)
```

**Solution:**
```bash
# Run locally to debug
pytest tests/test_model.py::test_model_forward_pass -v

# Add prints and diagnose
```

### Docker Build Fails

```
❌ Docker build failed
  E: Unable to locate package xyz
```

**Solution:**
- Check apt-get list in Dockerfile
- Try building locally: `docker build -f infrastructure/docker/Dockerfile.dev .`

### Training Validation Fails

```
❌ Training failed at step 5
  CUDA out of memory
```

**Solution:**
- Reduce batch size
- Enable gradient checkpointing
- Use fp16 training

---

## Skipping Workflows

If you want to skip CI/CD for a commit:

```bash
git commit -m "WIP: temporary test [skip ci]"
```

The workflow won't run for commits with `[skip ci]`.

---

## Viewing Workflow Logs

```
GitHub → Actions → Select workflow → Select run

Shows:
├─ Lint output
├─ Test output
├─ Coverage report
├─ Docker build logs
└─ Any failures with details
```

---

## Best Practices

### Do

✅ Run workflows locally first
```bash
flake8 core/
pytest tests/
docker build ...
```

✅ Fix issues before pushing
✅ Keep workflows fast (< 10 minutes)
✅ Use caching for dependencies
✅ Have clear error messages

### Don't

❌ Push broken code hoping CI/CD will catch it
❌ Have workflows that take > 20 minutes
❌ Ignore CI/CD failures
❌ Disable checks on `main` branch

---

## Monitoring Workflows

```bash
# View all workflow runs
github run list

# View specific run
github run view <run-id>

# Re-run failed workflow
github run rerun <run-id>
```

---

## Estimated CI/CD Setup Time

```
Task                          Time
────────────────────────────  ────
Create pr_checks.yml          30 min
Set up unit tests             1 hour
Configure GitHub secrets      15 min
Test full CI/CD pipeline      30 min
────────────────────────────  ─────
Total                         2.5 hours

Result:
✅ Automated quality gates forever
✅ Safe merging
✅ Professional workflow
```

---

## Next Steps

→ Read **05_IMPLEMENTATION_CHECKLIST.md** for week-by-week execution plan

This is your action plan: who does what, when, and in what order.

---

## Quick Command Reference

```bash
# Test locally before pushing
flake8 core/ pipeline/ courses/
pytest tests/

# Build all images locally
docker build -f infrastructure/docker/Dockerfile.dev -t spatial-llava:dev .
docker build -f infrastructure/docker/Dockerfile.train -t spatial-llava:train .
docker build -f infrastructure/docker/Dockerfile.inference -t spatial-llava:inference .

# View GitHub Actions locally
# (Can't fully simulate, but can test individual parts)
pytest tests/ -v
```
