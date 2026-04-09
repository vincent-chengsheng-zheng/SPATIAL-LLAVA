"""
pipeline/stage_0_environment.py

Pure environment check — run after setup_cluster.sh to verify the cluster
is ready. Checks only stable, infrastructure-level conditions.

What this checks:
    1. Python version
    2. Key packages installed at correct versions (from requirements.txt)
    3. GPU / CUDA availability and VRAM
    4. Disk space
    5. Network access to HuggingFace

What this does NOT check (belongs elsewhere):
    - Whether model classes can be imported (changes during development)
    - Whether pkl / COCO data files exist (checked in download_data.sh)
    - Whether checkpoints exist (checked in training scripts)

Usage:
    python pipeline/stage_0_environment.py

Exit codes:
    0 - All critical checks passed
    1 - One or more critical checks failed
"""

import sys
import os
import shutil

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

critical_failures = []
warnings = []


def check(label, condition, fail_msg, critical=True):
    if condition:
        print(f"{PASS} {label}")
        return True
    marker = FAIL if critical else WARN
    print(f"{marker} {label}")
    print(f"      → {fail_msg}")
    if critical:
        critical_failures.append(label)
    else:
        warnings.append(label)
    return False


def section(title):
    print(f"\n── {title} {'─' * (50 - len(title))}")


# ── 1. Python ─────────────────────────────────────────────────────────────────
section("Python")
py = sys.version_info
check(
    label=f"Python {py.major}.{py.minor}.{py.micro}",
    condition=py >= (3, 10),
    fail_msg="Python 3.10+ required.",
)

# ── 2. Packages ───────────────────────────────────────────────────────────────
section("Packages")

req_path = os.path.join(_repo_root, "requirements.txt")


def parse_requirements(path):
    reqs = {}
    if not os.path.exists(path):
        return reqs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" in line:
                name, version = line.split("==", 1)
                reqs[name.strip().lower()] = version.strip()
    return reqs


def get_installed_version(package_name):
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except Exception:
        return None


required = parse_requirements(req_path)
if not required:
    print(f"{WARN} requirements.txt not found at {req_path}")
    warnings.append("requirements.txt not found")
else:
    for pip_name, req_version in required.items():
        installed = get_installed_version(pip_name) \
                 or get_installed_version(pip_name.replace("-", "_"))
        if installed is None:
            print(f"{FAIL} {pip_name}: NOT INSTALLED (need {req_version})")
            print(f"      → Run: source shared/scripts/setup_cluster.sh")
            critical_failures.append(f"{pip_name} not installed")
        elif installed != req_version:
            print(f"{WARN} {pip_name}: {installed} (need {req_version})")
            warnings.append(f"{pip_name}: {installed} != {req_version}")
        else:
            print(f"{PASS} {pip_name}=={installed}")

# ── 3. GPU / CUDA ─────────────────────────────────────────────────────────────
section("GPU / CUDA")

try:
    import torch

    check(
        label="CUDA available",
        condition=torch.cuda.is_available(),
        fail_msg=(
            "CUDA not available. In JupyterHub: stop server → "
            "select 'Large GPU Server' → restart."
        ),
    )

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1e9
        vram_free = torch.cuda.mem_get_info(0)[0] / 1e9

        print(f"{PASS} GPU: {gpu.name}")
        print(f"{PASS} CUDA: {torch.version.cuda}")

        check(
            label=f"VRAM total: {vram_gb:.1f} GB",
            condition=vram_gb >= 15,
            fail_msg="Need ≥16GB VRAM. Request a larger GPU node.",
        )
        check(
            label=f"VRAM free: {vram_free:.1f} GB",
            condition=vram_free >= 10,
            fail_msg="Run 'nvitop' to check if another process is using the GPU.",
            critical=False,
        )

except Exception as e:
    print(f"{FAIL} GPU check error: {e}")
    critical_failures.append("GPU check error")

# ── 4. Disk ───────────────────────────────────────────────────────────────────
section("Disk")

free_gb = shutil.disk_usage(_repo_root).free / 1e9
check(
    label=f"Free space: {free_gb:.1f} GB (need ≥20 GB)",
    condition=free_gb >= 20,
    fail_msg="Need ≥20GB for COCO images + model weights.",
    critical=False,
)

# ── 5. Network ────────────────────────────────────────────────────────────────
section("Network")

try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=5)
    print(f"{PASS} HuggingFace reachable")
except Exception:
    print(f"{WARN} HuggingFace not reachable")
    print("      → Model/dataset downloads will fail.")
    warnings.append("HuggingFace not reachable")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)

if not critical_failures:
    print(" ✅ Environment ready.")
    if warnings:
        print(f"\n Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"   ⚠ {w}")
    print("\n Next: bash shared/scripts/download_data.sh")
    print("=" * 55)
    sys.exit(0)
else:
    print(" ✗ Environment NOT ready. Fix:\n")
    for i, f in enumerate(critical_failures, 1):
        print(f"   {i}. {f}")
    print("\n Run: source shared/scripts/setup_cluster.sh")
    print("=" * 55)
    sys.exit(1)
