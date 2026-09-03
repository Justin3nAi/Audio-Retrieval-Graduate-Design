#!/bin/bash
# GitHub上传命令脚本 (Linux/Mac)
# 此脚本包含所有必要的Git命令

echo "================================================================================"
echo "GitHub Upload Script"
echo "================================================================================"

cd "$(dirname "$0")"

echo ""
echo "[Step 1] Initializing Git repository..."
git init
if [ $? -ne 0 ]; then
    echo "ERROR: Git init failed"
    exit 1
fi

echo ""
echo "[Step 2] Adding all files..."
git add .
if [ $? -ne 0 ]; then
    echo "ERROR: Git add failed"
    exit 1
fi

echo ""
echo "[Step 3] Checking status..."
git status

echo ""
echo "[Step 4] Files to be committed:"
git diff --cached --stat

echo ""
echo "================================================================================"
echo "Ready to commit and push!"
echo "================================================================================"
echo ""
echo "Please review the files above."
echo "If everything looks correct, run these commands manually:"
echo ""
echo "  git commit -m \"Initial commit: Multi-encoder audio-text retrieval system\""
echo "  git remote add origin https://github.com/YOURUSERNAME/audio-text-retrieval.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "IMPORTANT: Replace YOURUSERNAME with your actual GitHub username!"
echo "================================================================================"
