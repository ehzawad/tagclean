#!/usr/bin/env bash
# Bail on staged artifacts. Override with ALLOW_ARTIFACT_COMMIT=1.
#
# Install:
#   git config core.hooksPath tools/githooks
#   ln -sf ../../tools/precommit_artifact_check.sh tools/githooks/pre-commit
#   chmod +x tools/precommit_artifact_check.sh

set -euo pipefail

if [[ "${ALLOW_ARTIFACT_COMMIT:-0}" == "1" ]]; then
    exit 0
fi

PATTERN='^(runs/|artifacts/|.*/artifacts/)|\.parquet$|\.npy$|\.faiss$|\.idx$'
BAD="$(git diff --cached --name-only --diff-filter=AM | grep -E "$PATTERN" || true)"

if [[ -n "$BAD" ]]; then
    echo "pre-commit: refusing to commit staged artifact files:" >&2
    echo "$BAD" | sed 's/^/  /' >&2
    echo >&2
    echo "  fix: 'git restore --staged <file>' or set ALLOW_ARTIFACT_COMMIT=1 to override." >&2
    exit 1
fi
