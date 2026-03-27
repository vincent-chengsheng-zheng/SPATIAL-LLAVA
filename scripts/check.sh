#!/bin/bash
# =============================================================
# scripts/check.sh
#
# Run this locally before every git push.
# Replicates exactly what GitHub Actions pr_checks.yml does.

# docker compose exec dev bash
# Usage (run from project root inside dev container):
#   bash scripts/check.sh
#
# Or run specific checks only:
#   bash scripts/check.sh lint
#   bash scripts/check.sh test
# =============================================================

set -e

MODE=${1:-all}   # default: run everything

PASS="✅"
FAIL="❌"

# ── Lint (flake8) ──────────────────────────────────────────────────────────────
run_lint() {
    echo ""
    echo "── Lint (flake8) ────────────────────────────────────"
    if flake8 core/ pipeline/ courses/ tests/ \
        --max-line-length=100 \
        --ignore=E501,W503,W504 \
        --count \
        --show-source \
        --statistics; then
        echo "$PASS Lint passed"
    else
        echo "$FAIL Lint failed — fix errors above before pushing"
        exit 1
    fi
}

# ── Tests (pytest) ─────────────────────────────────────────────────────────────
run_tests() {
    echo ""
    echo "── Tests (pytest) ───────────────────────────────────"
    if pytest tests/ \
        -v \
        --cov=core \
        --cov-report=term-missing \
        --cov-fail-under=50; then
        echo "$PASS Tests passed"
    else
        echo "$FAIL Tests failed — fix failures above before pushing"
        exit 1
    fi
}

# ── Run based on mode ──────────────────────────────────────────────────────────
case $MODE in
    lint) run_lint ;;
    test) run_tests ;;
    all)
        run_lint
        run_tests
        echo ""
        echo "══════════════════════════════════════════════════"
        echo "$PASS All checks passed — safe to push"
        echo "══════════════════════════════════════════════════"
        ;;
    *)
        echo "Usage: bash scripts/check.sh [lint|test|all]"
        exit 1
        ;;
esac