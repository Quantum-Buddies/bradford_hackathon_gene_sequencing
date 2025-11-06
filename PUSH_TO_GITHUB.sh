#!/bin/bash
#
# Push Autoregressive Pipeline to GitHub
# ======================================
# Commits and pushes all new files and modifications
#
# Usage:
#   bash PUSH_TO_GITHUB.sh [commit_message]
#

set -e

COMMIT_MSG="${1:-Add autoregressive next-token prediction pipeline with GPU support}"

echo "=========================================="
echo "GITHUB PUSH - AUTOREGRESSIVE PIPELINE"
echo "=========================================="
echo ""
echo "Commit message:"
echo "  $COMMIT_MSG"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "ERROR: Not a git repository"
    echo "Run: git init"
    exit 1
fi

# Check git status
echo "Current git status:"
git status --short | head -20
echo ""

# Add all new and modified files
echo "Adding files..."
git add prepare_autoregressive_data.py
git add prepare_classical_benchmarks.py
git add lambeq_encoder.py
git add train_quixer_hybrid.py
git add run_lambeq_gpu.sh
git add run_classical_prep.sh
git add LAMBEQ_GPU_GUIDE.md
git add GPU_QUICK_START.md
git add PIPELINE_VERIFICATION.md
git add CHANGES_SUMMARY.md
git add NEXT_TOKEN_PREDICTION_GUIDE.md
git add GITHUB_PUSH_GUIDE.md
git add README_AUTOREGRESSIVE.md
git add PUSH_TO_GITHUB.sh

echo "✓ Files added"
echo ""

# Show what will be committed
echo "Files to be committed:"
git diff --cached --name-only
echo ""

# Commit
echo "Committing..."
git commit -m "$COMMIT_MSG"
echo "✓ Committed"
echo ""

# Show commit info
echo "Commit details:"
git log -1 --oneline
echo ""

# Push
echo "Ready to push. Run:"
echo "  git push origin main"
echo ""
echo "Or push automatically:"
echo "  git push origin main"
echo ""

# Optional: auto-push
read -p "Push to GitHub now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing to GitHub..."
    git push origin main
    echo "✓ Pushed successfully!"
else
    echo "Skipped push. Run 'git push origin main' when ready."
fi

echo ""
echo "=========================================="
echo "✅ GitHub push complete!"
echo "=========================================="
echo ""
