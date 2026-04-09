"""
core/paths.py

Single source of truth for all project paths.
All paths are relative to PROJECT_ROOT, so the repo can be cloned anywhere.

Usage:
    from core.paths import PATHS

    data_dir   = PATHS.data          # <repo>/data/
    coco_dir   = PATHS.coco          # <repo>/data/coco/
    weights    = PATHS.weights       # <repo>/weights/
    ckpt_main  = PATHS.ckpt_main     # <repo>/checkpoints/main/
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
# This file lives at <root>/core/paths.py → parent.parent = root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Path registry ─────────────────────────────────────────────────────────────
class _Paths:
    # Data (pkl files are git-tracked; coco/ is gitignored)
    data          = PROJECT_ROOT / "data"
    coco          = PROJECT_ROOT / "data" / "coco"
    coco_train    = PROJECT_ROOT / "data" / "coco" / "train2014"

    # Model weights / HF cache (gitignored, large)
    weights       = PROJECT_ROOT / "weights"

    # Checkpoints (gitignored, large)
    checkpoints   = PROJECT_ROOT / "checkpoints"
    ckpt_main     = PROJECT_ROOT / "checkpoints" / "main"
    ckpt_ablation = PROJECT_ROOT / "checkpoints" / "ablation"

    # Results (git-tracked, small JSON + images)
    results           = PROJECT_ROOT / "results"
    results_baseline  = PROJECT_ROOT / "results" / "baseline"
    results_main      = PROJECT_ROOT / "results" / "main"
    results_ablation  = PROJECT_ROOT / "results" / "ablation"

    # Logs (git-tracked, small)
    logs = PROJECT_ROOT / "logs"

    def ensure_all(self):
        """Create all directories that should exist at runtime."""
        dirs = [
            self.data, self.coco_train,
            self.weights,
            self.ckpt_main, self.ckpt_ablation,
            self.results_baseline, self.results_main, self.results_ablation,
            self.logs,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def pkl(self, split: str) -> Path:
        """Return path to refcoco_{split}.pkl"""
        return self.data / f"refcoco_{split}.pkl"

    def __repr__(self):
        return f"<Paths root={PROJECT_ROOT}>"


PATHS = _Paths()