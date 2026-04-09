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

# ── Add repo root to path ─────────────────────────────────────────────────────
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from core.paths import PATHS  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

critical_failures = []
warnings = []


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
    label=f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}",
    condition=py_version >= (3, 10),
    fail_msg="Python 3.10+ required. Current version is too old.",
)


# ── 2. Package version check against requirements.txt ────────────────────────

section("Package Versions")

req_path = str(PATHS.weights.parent / "requirements.txt")
# fallback: find requirements.txt at repo root
req_path = os.path.join(str(PATHS.weights.parent.parent), "requirements.txt")
req_path = os.path.join(_repo_root, "requirements.txt")


def parse_requirements(path: str) -> dict:
    """Parse requirements.txt into {package_name: required_version}."""
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


def get_installed_version(package_name: str) -> str:
    """Get installed version of a package."""
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
    print(f"      Checking against: {req_path}")
    for pip_name, req_version in required.items():
        installed = get_installed_version(pip_name)
        if installed is None:
            installed = get_installed_version(pip_name.replace("-", "_"))
        if installed is None:
            print(f"{FAIL} {pip_name}: NOT INSTALLED (required {req_version})")
            print("      → Run: source shared/scripts/setup_cluster.sh")
            critical_failures.append(f"{pip_name} not installed")
        elif installed != req_version:
            print(f"{WARN} {pip_name}: {installed} (required {req_version})")
            warnings.append(
                f"{pip_name} version mismatch: {installed} != {req_version}")
        else:
            print(f"{PASS} {pip_name}=={installed}")


# ── 3. Critical import check ──────────────────────────────────────────────────

section("Critical Imports")

critical_imports = [
    ("torch", None),
    ("transformers", "LlavaForConditionalGeneration"),
    ("peft", "LoraConfig"),
    ("core.model.spatial_llava", "SpatialLLaVA"),
    ("core.model.regression_head", "RegressionHead"),
    ("core.loss.spatial_loss", "spatial_loss"),
    ("core.utils.metrics", "compute_all_metrics"),
    ("core.utils.checkpoint", "save_checkpoint"),
    ("core.data.refcoco_loader", "RefCOCODataset"),
    ("core.data.preprocessing", "preprocess_image_from_pil"),
    ("courses.shared.train", "load_config"),
    ("courses.shared.eval", "generate_comparison_table"),
]

for module, attr in critical_imports:
    try:
        m = __import__(module, fromlist=[attr] if attr else [])
        if attr:
            getattr(m, attr)
        label = f"{module}" + (f".{attr}" if attr else "")
        print(f"{PASS} {label}")
    except ImportError as e:
        label = f"{module}" + (f".{attr}" if attr else "")
        print(f"{FAIL} {label}")
        print(f"      → {e}")
        critical_failures.append(f"Import failed: {label}")


# ── 4. CUDA / GPU ─────────────────────────────────────────────────────────────

section("GPU / CUDA")

try:
    import torch

    cuda_available = torch.cuda.is_available()
    check(
        label="CUDA available",
        condition=cuda_available,
        fail_msg=(
            "CUDA not available. Are you on a GPU node? "
            "In JupyterHub: stop server → select 'Large GPU Server' → restart."
        ),
    )

    if cuda_available:
        gpu = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1e9
        vram_free = torch.cuda.mem_get_info(0)[0] / 1e9

        print(f"{PASS} GPU: {gpu.name}")

        check(
            label=f"VRAM total: {vram_gb:.1f} GB",
            condition=vram_gb >= 15,
            fail_msg=(
                f"VRAM is {vram_gb:.1f}GB, need ≥16GB. "
                "Request a larger GPU node in JupyterHub."
            ),
        )

        check(
            label=f"VRAM free: {vram_free:.1f} GB",
            condition=vram_free >= 10,
            fail_msg=(
                f"Only {vram_free:.1f}GB free. Another process may be using the GPU. "
                "Run 'nvitop' in terminal to check."
            ),
            critical=False,
        )

        cuda_version = torch.version.cuda
        print(f"{PASS} CUDA version: {cuda_version}")

except Exception as e:
    print(f"{FAIL} GPU check failed with error: {e}")
    critical_failures.append("GPU check error")


# ── 5. Repo directory structure ───────────────────────────────────────────────

section("Repo Directory Structure")

print(f"      Repo root: {_repo_root}")

required_dirs = {
    PATHS.data:              "data/             (pkl files — git tracked)",
    PATHS.coco_train:        "data/coco/train2014/ (COCO images — gitignored)",
    PATHS.weights:           "weights/          (HF cache — gitignored)",
    PATHS.ckpt_main:         "checkpoints/main/ (gitignored)",
    PATHS.ckpt_ablation:     "checkpoints/ablation/ (gitignored)",
    PATHS.results_baseline:  "results/baseline/ (git tracked)",
    PATHS.results_main:      "results/main/     (git tracked)",
    PATHS.results_ablation:  "results/ablation/ (git tracked)",
    PATHS.logs:              "logs/             (git tracked)",
}

for path, label in required_dirs.items():
    check(
        label=label,
        condition=path.exists(),
        fail_msg=f"Run: source shared/scripts/setup_cluster.sh",
    )

# Check pkl files exist
for split in ["train", "val", "test"]:
    pkl = PATHS.pkl(split)
    check(
        label=f"data/refcoco_{split}.pkl",
        condition=pkl.exists() and pkl.stat().st_size > 100,
        fail_msg="Run: bash shared/scripts/download_data.sh",
    )

# Check COCO images
n_imgs = sum(1 for f in PATHS.coco_train.glob("*.jpg")) if PATHS.coco_train.exists() else 0
check(
    label=f"COCO images: {n_imgs:,} / 82,783",
    condition=n_imgs >= 82783,
    fail_msg="Run: bash shared/scripts/download_data.sh",
)

# Storage space check on repo root
free_gb = shutil.disk_usage(_repo_root).free / 1e9
check(
    label=f"Disk free space: {free_gb:.1f} GB",
    condition=free_gb >= 20,
    fail_msg=(
        f"Only {free_gb:.1f}GB free. Need ≥20GB for COCO images + weights. "
        "Check available space on this partition."
    ),
    critical=False,
)


# ── 6. Network ────────────────────────────────────────────────────────────────

section("Network")

try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=5)
    print(f"{PASS} HuggingFace accessible")
except Exception:
    print(f"{WARN} HuggingFace not accessible")
    print("      → Model downloads will fail. Check cluster network settings.")
    warnings.append("HuggingFace not accessible")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)

if not critical_failures:
    print(" Stage 0 passed ✅  Environment is ready.")
    if warnings:
        print(f"\n Warnings ({len(warnings)}) — non-critical:")
        for w in warnings:
            print(f"   ⚠ {w}")
    print("\n Next step:")
    print("   bash shared/scripts/download_data.sh")
    print("=" * 55)
    sys.exit(0)

else:
    print(" Stage 0 FAILED ✗  Fix the following before proceeding:\n")
    for i, f in enumerate(critical_failures, 1):
        print(f"   {i}. {f}")
    print("\n Fix by running:")
    print("   source shared/scripts/setup_cluster.sh")
    print("   python pipeline/stage_0_environment.py")
    print("=" * 55)
    sys.exit(1)
