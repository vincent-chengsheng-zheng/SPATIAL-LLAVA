# Spatial-LLaVA: Complete Team Deployment Guide

## 📖 How to Read This Documentation

This guide is structured in **5 progressive documents** designed to be read in order. Each builds on the previous one.

```
START HERE ↓

00_README.md (this file)
└─ Overview of all documents + reading order
   
├─ 01_ARCHITECTURE_OVERVIEW.md
│  └─ "What are we building?" Quick answers to 3 key questions
│  └─ Team doesn't need to read everything, just this 10 min overview
│
├─ 02_TRAINING_PIPELINE.md
│  └─ "How do we train the model?" 5 stages + detailed checklist
│  └─ All team members need to understand their stage
│
├─ 03_DOCKER_STRATEGY.md
│  └─ "How do we isolate environments?" 3 Dockerfiles + compose
│  └─ For implementation (especially Member 1)
│
├─ 04_CI_CD_IMPLEMENTATION.md
│  └─ "How do we work together safely?" GitHub Actions workflows
│  └─ Reference for automation
│
└─ 05_IMPLEMENTATION_CHECKLIST.md
   └─ "What should we do first?" Week-by-week action plan
   └─ Quick reference for execution
```

---

## 🎯 Quick Document Guide

### If you have **5 minutes**: Read this section only
→ See **"Quick Answers to 3 Key Questions"** below

### If you have **30 minutes**: Read 01_ARCHITECTURE_OVERVIEW.md
→ Understand the overall strategy

### If you have **1 hour**: Read 01 + 02 + 05
→ Understand what you're building + how to execute

### If you're **implementing**: Read 02, 03, 04, 05 in order
→ Full technical details with code examples

### If you're **just joining the team**: Start with 01, skip to your section in 02
→ Understand your specific role

---

## ⚡ Quick Answers to 3 Key Questions

### Q1: How many Dockerfiles do we need?

**Answer: 3**

```
Dockerfile.dev        → Development (all team members)
Dockerfile.train      → GPU training (Members 3, 4)
Dockerfile.inference  → Production API (Enterprise deployment)
```

**Why 3 and not 1?**
- Different environments have different needs
- dev needs Jupyter, train needs CUDA, inference needs to be lightweight
- Smaller images = faster iterations

---

### Q2: Do we need microservices architecture?

**Answer: No (not now)**

```
Current: Single monolithic FastAPI service
├─ Image preprocessing
├─ Model inference
└─ Output formatting

This is sufficient because:
✅ Single model, not multiple services
✅ Inference time ≈ 130ms (fast enough)
✅ No need to scale preprocessing separately
✅ Simpler to deploy and debug

Upgrade to microservices only if:
❌ Expected >100 QPS (very unlikely)
❌ Different components need independent scaling
```

**Recommendation: Stay monolithic for now**

---

### Q3: Do we need CI/CD?

**Answer: Yes, absolutely!**

```
4 members + multiple branches = chaos without CI/CD

GitHub Actions (free):
✅ Automatic code quality checks on every PR
✅ Prevent broken code from being merged
✅ Ensure Docker images can be built
✅ Validate tests pass

Minimum setup: pr_checks.yml (takes 1 hour to set up)
```

---

## 📋 Document Structure

### **01_ARCHITECTURE_OVERVIEW.md**
**Purpose:** High-level decisions and architecture choices

**Contains:**
- Why 3 Dockerfiles (not 1, not microservices)
- Single container vs microservices comparison
- Architecture diagrams
- When to upgrade (you probably won't)

**Read time:** 10-15 minutes  
**For:** Everyone (especially decision makers)

---

### **02_TRAINING_PIPELINE.md**
**Purpose:** Complete training workflow with 5 stages

**Contains:**
- Stage 0: Environment setup
- Stage 1: Data preparation
- Stage 2: Model initialization
- Stage 3: Training loop
- Stage 4: Evaluation
- Stage 5: Deployment/Demo
- Detailed engineering directory structure
- Acceptance criteria for each stage
- Team member responsibilities + timeline

**Read time:** 30-45 minutes  
**For:** All team members (read your specific section at minimum)

**Key points:**
- Each stage has clear inputs → outputs
- Stages are mostly independent (can work in parallel after Stage 2)
- Validation criteria to know when you're done

---

### **03_DOCKER_STRATEGY.md**
**Purpose:** Detailed Docker implementation and best practices

**Contains:**
- Complete Dockerfile code for all 3 files
- docker-compose.yml configuration
- Why each layer in each Dockerfile
- Optimization techniques
- Comparison: development vs training vs inference
- Storage and caching strategies

**Read time:** 45-60 minutes  
**For:** Primarily Member 1 + Member 4 (implementers)

**Key points:**
- Copy-paste ready Dockerfile code
- Explanation of each RUN command
- Volume mounting strategies
- GPU allocation

---

### **04_CI_CD_IMPLEMENTATION.md**
**Purpose:** GitHub Actions workflows and continuous integration

**Contains:**
- pr_checks.yml (code quality + unit tests + Docker build)
- training_validation.yml (optional: verify training works)
- deploy_inference.yml (optional: auto-deploy to production)
- How to troubleshoot failed workflows
- Secret management for deployments

**Read time:** 45 minutes  
**For:** Primarily Member 1 (can delegate part of setup)

**Key points:**
- Copy-paste ready YAML configs
- When to trigger each workflow
- What to do when tests fail
- Integration with GitHub UI

---

### **05_IMPLEMENTATION_CHECKLIST.md**
**Purpose:** Quick reference, week-by-week action plan

**Contains:**
- Who does what (team matrix)
- Week-by-week milestones
- Dependency graph (what blocks what)
- Quick command reference
- Common problems and solutions
- Acceptance criteria checklist

**Read time:** 15-20 minutes (reference document)  
**For:** Everyone (bookmark this for execution phase)

**Key points:**
- Concrete tasks (not abstract concepts)
- Exact commands to run
- When each person can start/stop waiting
- Done/not-done checklist

---

## 🗺️ Reading Paths by Role

### **Member 1 (Architecture/Infrastructure Lead)**
```
1. Read 01_ARCHITECTURE_OVERVIEW (15 min)
2. Read 02_TRAINING_PIPELINE sections 0-2 (20 min)
3. Read 03_DOCKER_STRATEGY completely (60 min)
4. Read 04_CI_CD_IMPLEMENTATION completely (45 min)
5. Reference 05_IMPLEMENTATION_CHECKLIST for timeline

Focus: Docker setup + CI/CD + Environment stages
Time: Week 1 (2-3 days)
```

### **Member 2 (Data Engineering)**
```
1. Read 01_ARCHITECTURE_OVERVIEW (15 min)
2. Read 02_TRAINING_PIPELINE section 1 (15 min)
3. Skim 03_DOCKER_STRATEGY data volume section (10 min)
4. Reference 05_IMPLEMENTATION_CHECKLIST for timeline

Focus: Stage 1 data preparation + RefCOCO loading
Time: Week 1-2 (2 days)
```

### **Member 3 (Coursework Track)**
```
1. Read 01_ARCHITECTURE_OVERVIEW (15 min)
2. Read 02_TRAINING_PIPELINE sections 2, 3a, 4a, 5a (30 min)
3. Skim 03_DOCKER_STRATEGY training section (15 min)
4. Reference 05_IMPLEMENTATION_CHECKLIST for timeline

Focus: Stage 3a (coursework training) + Stage 4a (evaluation) + Stage 5a (Gradio demo)
Time: Week 3-4 (depends on Stage 1, 2 completion)
```

### **Member 4 (Enterprise Track)**
```
1. Read 01_ARCHITECTURE_OVERVIEW (15 min)
2. Read 02_TRAINING_PIPELINE sections 2, 3b, 4b, 5b (30 min)
3. Read 03_DOCKER_STRATEGY inference section completely (30 min)
4. Read 04_CI_CD_IMPLEMENTATION completely (45 min)
5. Reference 05_IMPLEMENTATION_CHECKLIST for timeline

Focus: Stage 3b (enterprise training) + Stage 4b (baseline comparison) + Stage 5b (FastAPI deployment)
Time: Week 4-5 (depends on Stage 1, 2 completion)
```

---

## 🚀 Execution Flow

### Week 1: Foundations (All members)
```
Member 1: Create Docker infrastructure + GitHub Actions
├─ Create infrastructure/docker/ directory
├─ Write 3 Dockerfiles
├─ Create docker-compose.yml
├─ Set up GitHub Actions
└─ Test: docker-compose up -d dev

Member 2: Prepare data
├─ Create data loading pipelines
├─ Download RefCOCO dataset
├─ Write preprocessing code
└─ Output: 3 pickle files (train/val/test)

Others: Code review + small fixes
```

### Week 2: Model Setup (Member 1 continues)
```
Member 1: Complete Stage 2
├─ Model initialization script
├─ LoRA configuration
├─ Regression head attachment
└─ Verify model can load and run

Others: Review + begin unit tests
```

### Week 3: Parallel Training Tracks
```
Member 3 (Coursework): Stage 3a
├─ Training script
├─ Configuration management
├─ Monitor training progress
└─ Target: Validation IoU ≥ 0.60

Member 4 (Enterprise): Stage 3b
├─ Training script (similar to 3a but longer)
├─ Configuration management
├─ Monitor training progress
└─ Target: Validation IoU ≥ 0.65
```

### Week 4: Evaluation
```
Member 3: Stage 4a + 5a
├─ Evaluate on test set
├─ Calculate metrics (IoU, RMSE)
├─ Create Gradio demo
└─ Done with coursework track

Member 4: Stage 4b
├─ Compare with 3 baselines
├─ Create comparison report
└─ Begin FastAPI implementation
```

### Week 5: Deployment
```
Member 4: Stage 5b
├─ FastAPI server implementation
├─ Docker for inference service
├─ API testing
├─ Optional: GitHub Actions deployment workflow
└─ Production ready

All: Final integration + documentation
```

---

## 📊 Document Dependencies

```
01_ARCHITECTURE_OVERVIEW
│
├─→ 02_TRAINING_PIPELINE (describes what to build)
│   ├─→ 03_DOCKER_STRATEGY (describes how to isolate)
│   ├─→ 04_CI_CD_IMPLEMENTATION (describes how to verify)
│   └─→ 05_IMPLEMENTATION_CHECKLIST (describes when to do it)
│
└─→ Quick decision making (should we use microservices? No)
```

---

## 💾 File Organization

All documents are in `/mnt/user-data/outputs/`:

```
outputs/
├── 00_README.md                           ← YOU ARE HERE
│   (Introduction + reading guide)
│
├── 01_ARCHITECTURE_OVERVIEW.md
│   (Quick decisions: 3 Dockerfiles, monolithic arch, CI/CD needed)
│
├── 02_TRAINING_PIPELINE.md
│   (5 stages: what to build, detailed steps, acceptance criteria)
│
├── 03_DOCKER_STRATEGY.md
│   (Complete Docker implementation, code examples)
│
├── 04_CI_CD_IMPLEMENTATION.md
│   (GitHub Actions workflows, automation)
│
└── 05_IMPLEMENTATION_CHECKLIST.md
    (Week-by-week tasks, quick reference, command cheatsheet)
```

---

## ⚙️ Key Concepts to Understand

### **Stage** (in Training Pipeline)
A clearly defined phase with specific inputs/outputs:
- Stage 0: Environment check
- Stage 1: Data preparation
- Stage 2: Model initialization
- Stage 3: Training
- Stage 4: Evaluation
- Stage 5: Deployment

Each stage should take 1-3 days.

### **Docker Container**
Isolated computing environment with:
- Specific Python version
- Specific libraries
- Specific entry point (what runs when you start it)

Think of it as a lightweight virtual machine.

### **docker-compose**
Tool that runs multiple containers together:
- dev container (Jupyter)
- train container (GPU training)
- inference container (API service)

Similar to "start multiple VMs from a config file"

### **GitHub Actions**
Automatic testing/deployment triggered by:
- Pull request creation
- Push to main branch
- Manual trigger
- Scheduled (e.g., daily)

Think of it as "robot checks the code every time someone submits changes"

### **CI/CD**
- **CI (Continuous Integration)**: Automatically test code
- **CD (Continuous Deployment)**: Automatically deploy code

Goal: Prevent broken code from being merged or deployed.

---

## ✅ Checklist: Are You Ready to Start?

Before diving into implementation, make sure:

- [ ] All 4 team members have read 01_ARCHITECTURE_OVERVIEW
- [ ] Each member knows their role from 02_TRAINING_PIPELINE
- [ ] Member 1 has reviewed 03_DOCKER_STRATEGY
- [ ] Member 1 has reviewed 04_CI_CD_IMPLEMENTATION
- [ ] Everyone has bookmarked 05_IMPLEMENTATION_CHECKLIST
- [ ] You've decided which branch (coursework/enterprise/both) you're implementing
- [ ] You have GitHub repo created and shared

If all checked ✅, you're ready to start implementing!

---

## 🆘 Getting Help

**Problem**: I don't understand Docker

→ Start with **03_DOCKER_STRATEGY** "Why 3 Dockerfiles?" section

**Problem**: I don't know what I should be doing this week

→ Check **05_IMPLEMENTATION_CHECKLIST** week-by-week section

**Problem**: My stage is blocked (waiting for another member)

→ Check **02_TRAINING_PIPELINE** dependency graph

**Problem**: Docker build is failing

→ Check **05_IMPLEMENTATION_CHECKLIST** troubleshooting section

**Problem**: GitHub Actions test failed

→ Check **04_CI_CD_IMPLEMENTATION** debugging section

---

## 📈 Success Metrics

By end of:

**Week 1:**
- ✅ Docker containers can start and run
- ✅ GitHub Actions pr_checks.yml is working
- ✅ Data is downloaded and preprocessed

**Week 2:**
- ✅ Model can load and do a forward pass
- ✅ Stage 0-2 fully automated

**Week 3:**
- ✅ Training converges on validation set
- ✅ At least one track (coursework or enterprise) running

**Week 4:**
- ✅ Both tracks have trained models
- ✅ Evaluation metrics documented

**Week 5:**
- ✅ At least Gradio (coursework) or API (enterprise) demo working
- ✅ GitHub repo has complete documentation

---

## 🎓 Key Learning from This Project

Beyond just "build a model", you'll learn:

1. **How to structure code** for team collaboration
2. **Docker basics** for environment isolation
3. **CI/CD fundamentals** for quality assurance
4. **Multi-stage deployment** (dev → staging → production)
5. **How to coordinate** 4 people on the same codebase

These skills transfer to real industry projects!

---

## 📞 Let's Connect

**Next Steps:**
1. All team members read 01_ARCHITECTURE_OVERVIEW (15 min)
2. Schedule 30-min team meeting to discuss architecture
3. Member 1 starts with 03_DOCKER_STRATEGY
4. Others start with 02_TRAINING_PIPELINE

**Timeline to start implementation:**
- If all team members read these docs: **This week**
- If you're still deciding: **Next week**

---

## 🏁 Summary

You have **5 well-organized documents** explaining:

1. **Why** we're making certain architectural choices
2. **What** we're building (5 stages with clear outputs)
3. **How** to implement it (Docker + CI/CD)
4. **When** each person should do their work (week-by-week)
5. **What** success looks like (acceptance criteria)

**Start with 01_ARCHITECTURE_OVERVIEW.md next** →

Good luck! 🚀

