#!/bin/bash
# Darwin v4 — Push to GitHub
#
# Usage:
#   ./push_to_github.sh <github-token> <repo-name> [private]
#
# Examples:
#   ./push_to_github.sh ghp_xxxxxxxxxxxx darwin-v4 private
#   ./push_to_github.sh ghp_xxxxxxxxxxxx darwin-v4
#
# The token needs 'repo' scope. Generate at:
#   https://github.com/settings/tokens/new?scopes=repo

set -e

TOKEN="${1:?Usage: ./push_to_github.sh <github-token> <repo-name> [private]}"
REPO="${2:?Usage: ./push_to_github.sh <github-token> <repo-name> [private]}"
VISIBILITY="${3:-private}"  # default to private

echo "══════════════════════════════════════════════"
echo "  Darwin v4 — GitHub Push"
echo "══════════════════════════════════════════════"

# Get GitHub username from token
echo "[1/4] Verifying token..."
USER=$(curl -s -H "Authorization: token $TOKEN" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin).get('login',''))")

if [ -z "$USER" ]; then
    echo "ERROR: Invalid token or API error"
    exit 1
fi

echo "  Authenticated as: $USER"

# Create repo
echo "[2/4] Creating repository $USER/$REPO ($VISIBILITY)..."
CREATE_RESPONSE=$(curl -s -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/user/repos \
    -d "{\"name\":\"$REPO\",\"private\":$([ "$VISIBILITY" = "private" ] && echo true || echo false),\"description\":\"Darwin v4 — Evolutionary Algorithmic Trading System\"}")

# Check if repo already exists
REPO_EXISTS=$(echo "$CREATE_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('exists' if 'already exists' in d.get('message','').lower() or d.get('full_name','') else '')" 2>/dev/null || echo "")

if [ -n "$REPO_EXISTS" ]; then
    echo "  Repository already exists, will push to it."
fi

# Set remote
echo "[3/4] Configuring remote..."
REMOTE_URL="https://${USER}:${TOKEN}@github.com/${USER}/${REPO}.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

# Push
echo "[4/4] Pushing to GitHub..."
git push -u origin main --force

# Clean token from remote URL
git remote set-url origin "https://github.com/${USER}/${REPO}.git"

echo ""
echo "══════════════════════════════════════════════"
echo "  ✓ Pushed successfully!"
echo "  https://github.com/$USER/$REPO"
echo "══════════════════════════════════════════════"
echo ""
echo "IMPORTANT: This repo should remain PRIVATE."
echo "It contains your trading system's full logic."
