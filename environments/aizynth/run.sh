#!/usr/bin/env bash
set -euo pipefail
AIZYNTH_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Exact versions are exported from this isolated project's uv.lock. uv run
# reuses a separate cached environment and never synchronizes the ROCm project.
exec uv run --no-project --isolated --offline --python 3.12 \
  --with-requirements "$AIZYNTH_ENV_DIR/requirements.lock.txt" "$@"
