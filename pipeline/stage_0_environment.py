"""
pipeline/stage_0_environment.py

Run this before every training session to verify the cluster environment.
Usage: python pipeline/stage_0_environment.py

Exit codes:
    0 - All checks passed, ready to proceed
    1 - One or more critical checks failed
"""

import sys
import os
import shutil
import subprocess

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS  = "  ✓"
FAIL  = "  ✗"
WARN  = "  ⚠"

critical_failures = []
warnings          = []

def check(label: str, condition: bool, fail_msg: str, critical: bool = True) -> bool:
    if condition:
        print(f"{PASS} {label}")
        return True
    else:
        marker = FAIL if critical else WARN
        print(f"{marker} {label}")
        print(f"      → {fail_msg}")
        if critical:
            critical_failures.append(label)
        else:
            warnings.append(label)
        return False


def section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")


# ── 1. Python version ─────────────────────────────────────────────────────────

section("Python")

py_version = sys.version_info
check(
    label    = f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}",
    condition= py_version >= (3, 10),
    fail_msg = "Python 3.10+ required. Current version is too old.",
)


# ── 2. Key package imports ────────────────────────────────────────────────────

section("Packages")

packages = {
    "torch":            "pip install torch",
    "transformers":     "pip install transformers==4.35.2",
    "peft":             "pip install peft==0.7.1",
    "PIL":              "pip install Pillow",
    "cv2":              "pip install opencv-python-headless",
    "gradio":           "pip install gradio==4.17.0",
    "fastapi":          "pip install fastapi==0.104.1",
    "yaml":             "pip install pyyaml",
    "tqdm":             "pip install tqdm",
}

for pkg, install_cmd in packages.items():
    try:
        __import__(pkg)
        print(f"{PASS} {pkg}")
    except ImportError:
        print(f"{FAIL} {pkg}")
        print(f"      → Not installed. Run: {install_cmd}")
        critical_failures.append(f"Missing package: {pkg}")


# ── 3. CUDA / GPU ─────────────────────────────────────────────────────────────

section("GPU / CUDA")

try:
    import torch

    cuda_available = torch.cuda.is_available()
    check(
        label    = "CUDA available",
        condition= cuda_available,
        fail_msg = "CUDA not available. Are you on a GPU node? "
                   "In JupyterHub: stop server → select 'Large GPU Server' → restart.",
    )

    if cuda_available:
        gpu       = torch.cuda.get_device_properties(0)
        vram_gb   = gpu.total_memory / 1e9
        vram_free = torch.cuda.mem_get_info(0)[0] / 1e9

        print(f"{PASS} GPU: {gpu.name}")

        check(
            label    = f"VRAM total: {vram_gb:.1f} GB",
            condition= vram_gb >= 15,
            fail_msg = f"VRAM is {vram_gb:.1f}GB, need ≥16GB. "
                        "Request a larger GPU node in JupyterHub.",
        )

        check(
            label    = f"VRAM free: {vram_free:.1f} GB",
            condition= vram_free >= 10,
            fail_msg = f"Only {vram_free:.1f}GB free. Another process may be using the GPU. "
                        "Run 'nvitop' in terminal to check.",
            critical = False,   # warning only, not a hard failure
        )

        cuda_version = torch.version.cuda
        print(f"{PASS} CUDA version: {cuda_version}")

except Exception as e:
    print(f"{FAIL} GPU check failed with error: {e}")
    critical_failures.append("GPU check error")


# ── 4. Environment variables ──────────────────────────────────────────────────

section("Environment Variables")

env_vars = {
    "HF_HOME":          "Run: echo 'export HF_HOME=~/SharedFolder/hf_cache' >> ~/.bashrc && source ~/.bashrc",
    "SPATIAL_DATA":     "Run: bash shared/scripts/setup_cluster.sh",
    "SPATIAL_CKPT":     "Run: bash shared/scripts/setup_cluster.sh",
    "SPATIAL_RESULTS":  "Run: bash shared/scripts/setup_cluster.sh",
    "SPATIAL_LOGS":     "Run: bash shared/scripts/setup_cluster.sh",
}

for var, fix in env_vars.items():
    value = os.environ.get(var)
    check(
        label    = f"{var} = {value if value else 'NOT SET'}",
        condition= value is not None,
        fail_msg = fix,
    )

# Warn if HF_HOME is not pointing to SharedFolder
hf_home = os.environ.get("HF_HOME", "")
if hf_home and "SharedFolder" not in hf_home:
    print(f"{WARN} HF_HOME is set but not pointing to SharedFolder")
    print(f"      → Current: {hf_home}")
    print(f"      → Expected: ~/SharedFolder/hf_cache")
    warnings.append("HF_HOME not pointing to SharedFolder")


# ── 5. Storage ────────────────────────────────────────────────────────────────

section("Storage")

# SharedFolder
shared = os.path.expanduser("~/SharedFolder")
check(
    label    = "~/SharedFolder exists",
    condition= os.path.isdir(shared),
    fail_msg = "SharedFolder not found. Run: bash shared/scripts/setup_cluster.sh",
)

if os.path.isdir(shared):
    # Write test
    test_file = os.path.join(shared, ".env_check_tmp")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"{PASS} ~/SharedFolder is writable")
    except Exception:
        print(f"{FAIL} ~/SharedFolder is not writable")
        critical_failures.append("SharedFolder not writable")

    # Free space
    free_gb = shutil.disk_usage(shared).free / 1e9
    check(
        label    = f"~/SharedFolder free space: {free_gb:.1f} GB",
        condition= free_gb >= 80,
        fail_msg = f"Only {free_gb:.1f}GB free. Need ≥80GB "
                   "(dataset ~50GB + model ~14GB + checkpoints). "
                   "Clear old files from SharedFolder.",
    )

    # Required subdirectories
    required_dirs = [
        "hf_cache",
        "data",
        "checkpoints/coursework",
        "checkpoints/enterprise",
        "results/coursework",
        "results/enterprise",
        "logs",
    ]
    for d in required_dirs:
        full_path = os.path.join(shared, d)
        check(
            label    = f"~/SharedFolder/{d}/ exists",
            condition= os.path.isdir(full_path),
            fail_msg = f"Run: mkdir -p ~/SharedFolder/{d}",
        )

# Home directory
home      = os.path.expanduser("~")
home_free = shutil.disk_usage(home).free / 1e9
check(
    label    = f"~/home free space: {home_free:.1f} GB",
    condition= home_free >= 10,
    fail_msg = f"Home directory nearly full ({home_free:.1f}GB free). "
               "Move large files to SharedFolder.",
    critical = False,
)


# ── 6. Network (HuggingFace access) ──────────────────────────────────────────

section("Network")

try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=5)
    print(f"{PASS} HuggingFace accessible")
except Exception:
    print(f"{WARN} HuggingFace not accessible")
    print(f"      → Model downloads will fail. Check cluster network settings.")
    warnings.append("HuggingFace not accessible")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)

if not critical_failures:
    print(" Stage 0 passed ✅  Environment is ready.")
    if warnings:
        print(f"\n Warnings ({len(warnings)}) — non-critical, but worth fixing:")
        for w in warnings:
            print(f"   ⚠ {w}")
    print("\n Next step:")
    print("   python pipeline/stage_1_data_preparation.py \\")
    print("       --output_dir ~/SharedFolder/data/")
    print("=" * 55)
    sys.exit(0)

else:
    print(f" Stage 0 FAILED ✗  Fix the following before proceeding:\n")
    for i, f in enumerate(critical_failures, 1):
        print(f"   {i}. {f}")
    print("\n Re-run after fixing:")
    print("   python pipeline/stage_0_environment.py")
    print("=" * 55)
    sys.exit(1)