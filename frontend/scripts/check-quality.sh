#!/usr/bin/env bash
# Frontend code quality check script
# Run from the frontend/ directory: ./scripts/check-quality.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"

cd "$FRONTEND_DIR"

echo "=== Frontend Code Quality Checks ==="
echo ""

# Check that node_modules exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
  echo ""
fi

# Run Prettier format check
echo "--- Prettier (format check) ---"
if npx prettier --check .; then
  echo "✓ All files are formatted correctly"
else
  echo "✗ Formatting issues found. Run './scripts/fix-quality.sh' to auto-fix."
  FAILED=1
fi
echo ""

# Run ESLint
echo "--- ESLint (lint check) ---"
if npx eslint script.js; then
  echo "✓ No linting issues found"
else
  echo "✗ Linting issues found."
  FAILED=1
fi
echo ""

if [ "${FAILED}" = "1" ]; then
  echo "=== Quality checks FAILED ==="
  exit 1
else
  echo "=== All quality checks PASSED ==="
fi
