#!/usr/bin/env bash
# Frontend code quality auto-fix script
# Run from the frontend/ directory: ./scripts/fix-quality.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$FRONTEND_DIR"

echo "=== Frontend Code Quality Auto-Fix ==="
echo ""

# Check that node_modules exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
  echo ""
fi

# Run Prettier to auto-format
echo "--- Running Prettier (auto-format) ---"
npx prettier --write .
echo "✓ Files formatted with Prettier"
echo ""

# Run ESLint with auto-fix
echo "--- Running ESLint (auto-fix) ---"
npx eslint --fix script.js && echo "✓ ESLint auto-fix applied" || echo "! Some ESLint issues require manual fixes"
echo ""

echo "=== Auto-fix complete. Run './scripts/check-quality.sh' to verify. ==="
