# 01_ARCHITECTURE_OVERVIEW.md

## Overview

This document answers 3 critical questions about your Spatial-LLaVA deployment:
1. How many Dockerfiles do we need?
2. Do we need microservices?
3. Do we need CI/CD?

**Read time:** 10-15 minutes  
**Audience:** All team members  
**Next document:** 02_TRAINING_PIPELINE.md

---

## Question 1: How Many Dockerfiles Do We Need?

### Answer: 3 Dockerfiles

| | Dev | Train | Inference |
|---|-----|-------|-----------|
| **Purpose** | Local development & debugging | GPU training (long-running) | Production API service |
| **Base image** | pytorch/pytorch:2.0 | nvidia/cuda:12.1-devel | pytorch/pytorch:2.0-runtime |
| **Size** | ~3.5 GB | ~8 GB | ~2.5 GB |
| **Tools included** | Jupyter Lab, git, vim | CUDA toolkit, gcc | FastAPI, uvicorn |
| **Users** | All team members | Members 3, 4 | Enterprise users |
| **Duration** | Interactive (hours) | Batch (hours-days) | Service (always running) |
| **GPU required?** | No (optional) | Yes, required | Optional (recommended) |

### Why 3 and Not 1?

```
❌ Single Dockerfile approach:
   └─ Would contain all tools for all environments
   └─ Unnecessarily large (10+ GB)
   └─ Slow to build and push
   └─ Mixing concerns (Jupyter + CUDA + FastAPI)

✅ Three separate Dockerfiles:
   ├─ dev: Small (3.5 GB), fast iteration
   ├─ train: Large but optimized for GPU
   └─ inference: Small, production-ready
   
   Total investment: ~1 day to write 3 Dockerfiles
   Total benefit: Faster builds, clearer responsibility
```

### Usage Context

```
Local Development (Week 1-3)
┌─────────────────────────────┐
│ docker-compose up -d        │
├─────────────────────────────┤
│ • Dockerfile.dev  ← running │ (Jupyter on port 8888)
│ • Dockerfile.train (ready)  │ (can run on demand)
│ • Dockerfile.inference      │ (for local testing)
└─────────────────────────────┘
   All on same laptop via docker-compose

Production Deployment (Week 4-5)
┌─────────────────────────────┐
│ Production Server           │
├─────────────────────────────┤
│ • Only Dockerfile.inference │ (running)
│   └─ FastAPI on port 8000   │
│                             │
│ • Dockerfile.train          │ (not running)
│   └─ Available if you need  │
│     to retrain              │
└─────────────────────────────┘
```

---

## Question 2: Do We Need Microservices Architecture?

### Answer: No (Not Now)

### What is Microservices?

Microservices = Breaking your application into multiple independent services that communicate with each other.

```
Monolithic (Current recommendation):
┌──────────────────────────────┐
│      FastAPI App             │
├──────────────────────────────┤
│ • Image preprocessing (5%)   │
│ • Model inference (80%)      │
│ • Output formatting (15%)    │
└──────────────────────────────┘
  1 Docker container, 1 process

Microservices (Not recommended for you):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Preprocess  │   │  Inference   │   │  Postprocess │
│  Service     │→→→→ Service      │→→→→ Service      │
│ (Python)     │   │ (PyTorch)    │   │ (Python)     │
└──────────────┘   └──────────────┘   └──────────────┘
   3 Docker containers, message queue, 3x complexity
```

### Decision Matrix: When to Use Each

| Factor | Monolithic (✅ Recommended) | Microservices (❌ Overkill) |
|--------|-------------|-------------|
| **Architecture complexity** | Simple | Complex |
| **Expected QPS** | < 50 | > 100 |
| **Preprocessing time** | Negligible | Long |
| **Need independent scaling?** | No | Yes |
| **Team has DevOps?** | Not required | Required |
| **Time to implement** | 1 day (FastAPI) | 2-3 weeks |
| **Risk level** | Low | High |
| **Your use case?** | ✅ Perfect fit | ❌ Overkill |

### Performance Comparison

```
Monolithic FastAPI:
Request → [preprocessing] → [inference] → [output] → Response
Latency: ~130ms
Throughput: ~500 requests/sec on single GPU

Microservices:
Request → Gateway → [preprocessing] → Queue → [inference] → Queue → [output] → Response
Latency: ~160-180ms (due to queue overhead)
Throughput: ~1000 requests/sec (but requires multiple inference replicas)

Your project:
• Model inference: ~100ms
• Preprocessing: ~5ms
• Output: ~10ms
• Total: ~130ms

Bottleneck: Model inference (can't optimize further)
Can you achieve 500 QPS with monolithic? No, because GPU max throughput ≈ 10 requests/sec
To achieve 500 QPS? Need 50 GPUs anyway, which requires orchestration (K8s)

Conclusion: Scale horizontally (more servers) not vertically (microservices)
```

### When Would You Need Microservices?

You would only upgrade if **all of these** are true:

1. ✅ Deployed to production serving real traffic
2. ✅ Hitting > 100 QPS consistently
3. ✅ Preprocessing is bottleneck (not model inference)
4. ✅ Team has dedicated DevOps engineer
5. ✅ Budget for container orchestration (K8s) and monitoring

**Probability for your project:** ~5% (very unlikely)

### What You'll Actually Use (Monolithic)

```
infrastructure/
├── docker/
│   ├── Dockerfile.dev
│   ├── Dockerfile.train
│   ├── Dockerfile.inference      ← Single inference service
│   └── docker-compose.yml
│
courses/enterprise_track/
└── deployment/
    ├── fastapi_server.py
    ├── requirements_prod.txt
    └── docker/
        └── Dockerfile.inference  ← Deployed as single container
```

**Result:** One `/predict` endpoint handling all requests

---

## Question 3: Do We Need CI/CD?

### Answer: Yes, Absolutely

### The Problem It Solves

```
Without CI/CD (4 people, no coordination):
├─ Monday: Member 1 modifies data loader
├─ Tuesday: Member 2 pushes old version → breaks Member 1's code
├─ Wednesday: Member 3 changes model interface → breaks everyone
├─ Thursday: Nobody knows what works, production breaks at deploy time
└─ Friday: Debugging, finger-pointing, missed deadline 😱

With CI/CD (automated quality gates):
├─ Member 1 makes change → GitHub Actions runs tests automatically
├─ Tests pass ✅ → Can merge to main safely
├─ Tests fail ❌ → Shows exactly what broke, blocks merge
│  └─ Feedback: "Your changes break test_model.py line 45"
│
└─ Result: Only working code reaches main branch 🎉
```

### What CI/CD Does For You

```
Every Pull Request:
┌─────────────────────────────────────┐
│ 1. Code quality check (flake8)      │
│    ├─ Line length check             │
│    ├─ Unused imports detection      │
│    └─ Name convention validation    │
│                                     │
│ 2. Unit tests (pytest)              │
│    ├─ Model can load               │
│    ├─ Data loader works            │
│    ├─ Loss calculation correct      │
│    └─ Metrics computation OK        │
│                                     │
│ 3. Docker build check               │
│    ├─ Dockerfile.dev builds         │
│    ├─ Dockerfile.train builds       │
│    └─ Dockerfile.inference builds   │
│                                     │
│ Result:                             │
│ ✅ All checks pass → Auto-merge OK │
│ ❌ Anything fails → Block merge     │
└─────────────────────────────────────┘
```

### CI/CD Tools Comparison

| Tool | Cost | Complexity | For your project |
|------|------|-----------|-----------------|
| GitHub Actions | Free (for public repos) | Low | ✅ **Recommended** |
| GitLab CI | Free | Low | ✅ Good alternative |
| Jenkins | Free (self-hosted) | High | ❌ Overkill |
| CircleCI | Paid | Medium | ❌ Unnecessary |

**Recommendation: GitHub Actions (it's free and integrated)**

### What GitHub Actions Looks Like

```yaml
# When someone makes a pull request:

.github/workflows/pr_checks.yml runs automatically
├─ Pull request detected
│  └─ "You need code review + tests passing to merge"
│
├─ Step 1: Code quality (2 minutes)
│  ├─ Run: flake8 (style check)
│  └─ Result: "Found 3 style issues, fix before merging"
│
├─ Step 2: Tests (5 minutes)
│  ├─ Run: pytest tests/
│  └─ Result: "All 10 tests passed ✅"
│
├─ Step 3: Docker builds (10 minutes)
│  ├─ Build all 3 Dockerfiles
│  └─ Result: "All containers built successfully ✅"
│
└─ Final result: 
   "✅ All checks passed - ready to merge!"
   (shown on GitHub PR page)
```

---

## Architecture Decision Summary

```
┌─────────────────────────────────────────────┐
│       Your Recommended Architecture          │
├─────────────────────────────────────────────┤
│                                             │
│  Infrastructure Layer:                      │
│  • 3 Dockerfiles (dev, train, inference)   │
│  • docker-compose for local development     │
│  • Single production container              │
│                                             │
│  Application Architecture:                  │
│  • Monolithic FastAPI app (not microservices)
│  • Single /predict endpoint                 │
│  • Deployed as one container                │
│                                             │
│  Quality Assurance:                         │
│  • GitHub Actions for CI/CD                │
│  • Automatic tests on every PR              │
│  • Block broken code from merging           │
│                                             │
│  Result:                                    │
│  ✅ Simple (can build in 1 week)           │
│  ✅ Safe (automated quality gates)          │
│  ✅ Scalable (upgrade later if needed)     │
│  ✅ Professional (real industry practices) │
│                                             │
└─────────────────────────────────────────────┘
```

---

## What This Means for Your Team

### Member 1 (Infrastructure)
```
Week 1:
├─ Write Dockerfile.dev
├─ Write Dockerfile.train
├─ Write Dockerfile.inference
├─ Create docker-compose.yml
├─ Set up GitHub Actions pr_checks.yml
└─ Total time: 2-3 days

→ Unblocks everyone else
```

### Members 2, 3, 4
```
Wait for Member 1 to complete:
├─ Infrastructure (Docker + CI/CD)
└─ Stage 1-2 (data + model initialization)

Then you can:
├─ Run docker-compose up -d dev  (instant development environment)
├─ Run docker-compose up -d train (instant training environment)
└─ Every PR automatically tested (safety net)
```

---

## Key Takeaways

✅ **3 Dockerfiles:** Different environments, optimized for each use case

❌ **No Microservices:** Monolithic FastAPI is simpler and sufficient

✅ **CI/CD Required:** GitHub Actions prevents team conflicts and catches bugs early

**Total cost:** ~5-6 days (1 week of concentrated effort)

**Total benefit:** Safe, professional deployment pipeline used in real industry

---

## Next Steps

→ Read **02_TRAINING_PIPELINE.md** to understand the 5 stages of training

This document explains **what** you're building.

---

## FAQ

**Q: Can we skip Docker and just run Python scripts?**
A: Technically yes, but:
- Everyone needs identical Python environment
- Production deployment becomes a nightmare
- No easy way to track what versions work
- Not professional for a real project

**Q: Can we use a single Dockerfile with everything in it?**
A: Technically yes, but:
- 10+ GB image (slow to build/push)
- Mixing concerns (dev tools + training tools + production tools)
- Can't optimize for each use case
- Not recommended

**Q: Do we really need GitHub Actions?**
A: If you have 4 people:
- Without CI/CD: High chance of breaking code getting merged
- With CI/CD: Automatic safety gate, very low breakage rate
- Time investment: 4-5 hours to set up once
- Recurring benefit: Every PR automatically tested forever

**Q: What if we need more than 100 QPS later?**
A: 
- Upgrade path exists (Docker makes it easier)
- Scale horizontally: Add more servers/containers
- Only upgrade to microservices if preprocessing is bottleneck
- For now: Plan for it, don't implement it

**Q: Is this "enterprise-grade"?**
A: Yes:
- Docker: Used in every major tech company
- GitHub Actions: Standard in modern development
- Monolithic + async: Handles moderate scale (100-1000s QPS)
- Best practice for your team size
