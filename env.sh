# Mol-Metal — local-host env wiring (NVIDIA Blackwell, 2026-10-03)
#
# Source this file in your interactive shell BEFORE running anything:
#
#     source env.sh           # adds CUDA env vars, routes uv to the repo-local .venv
#     uv run python ...       # now uses /home/hugo/codes/mol-metal/.venv
#
# What this does:
#   1. UV_PROJECT_ENVIRONMENT — points uv at the repo-local CANONICAL venv
#      /home/hugo/codes/mol-metal/.venv (AGENTS.md §5.9 single-venv rule;
#      torch 2.14.1+cu130, rdkit, posebusters, meeko, vina are installed).
#      The legacy CadButEaas / kanzi environments are FORBIDDEN —
#      molmetal/scripts/check_no_stale_venv.sh fails if the repo is pointed
#      at any interpreter other than this .venv.
#   1b. MOLMETAL_DATA_DIR / MOLMETAL_TMQM_DIR — point loaders at the
#      project-local .data/molmetal/ tree (the loaders honour these since
#      the 2026-09-25 patch; see molmetal/data/cytotox.py,
#      molmetal/data/crossdocked.py, molmetal/data/tmqm.py,
#      molmetal/molmetal_lam/lam_chem/platinai_dataset.py).
#   2. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — reduces CUDA
#      allocator fragmentation for BF16 matmul (mirrors README §11.5).
#   3. MOLMETAL_CFM_AMP / MOLMETAL_CFM_AMP_DTYPE / MOLMETAL_EGNN_CHECKPOINT
#      — enable BF16 autocast + per-layer checkpoint (mol-metal defaults).
#   4. TOKENIZERS_PARALLELISM / TRANSFORMERS_VERBOSITY — quiet import hygiene
#      (also auto-applied by molmetal/__init__.py on every host).
#   5. CUDA_VISIBLE_DEVICES=0 — pin to the RTX PRO 6000 (device 1 is the
#      RTX 5090; we don't want it stealing VRAM by default).

# --- 0. Canonical-venv sanity. .venv IS canonical (AGENTS.md §5.9); there is
#     no "stale venv" concept — check_no_stale_venv.sh now fails when the
#     repo is pointed at ANY interpreter other than $REPO_ROOT/.venv.
CANONICAL_VENV=/home/hugo/codes/mol-metal/.venv
if [ ! -x "$CANONICAL_VENV/bin/python" ]; then
    echo "[mol-metal env] WARNING: canonical venv python missing at $CANONICAL_VENV/bin/python" >&2
    echo "[mol-metal env]          Run: uv sync (creates/reinstates the repo-local .venv)" >&2
fi

# --- 1. uv: use the repo-local canonical venv (AGENTS.md §5.9)
export UV_PROJECT_ENVIRONMENT="$CANONICAL_VENV"

# --- 1b. Data dir — points loaders at .data/molmetal/ (project-local)
#     instead of the default /mnt/storage/data/molmetal (not writable on
#     this host). See .data/DATA_STATUS.md for what's in there.
export MOLMETAL_DATA_DIR=/home/hugo/codes/mol-metal/.data/molmetal
export MOLMETAL_TMQM_DIR=/home/hugo/codes/mol-metal/.data/molmetal/tmQM

# --- 2. CUDA allocator hygiene (BF16 matmul + reduced fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 3. AMP / checkpoint toggles (mol-metal defaults)
export MOLMETAL_CFM_AMP=1
export MOLMETAL_CFM_AMP_DTYPE=bfloat16
export MOLMETAL_EGNN_CHECKPOINT=1

# --- 4. tokenizers / transformers hygiene
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# --- 5. pin GPUs (default to the RTX PRO 6000)
# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the "" session lock) must win; unset -> 0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

# --- Echo so we know we're wired up.
echo "[mol-metal env] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "[mol-metal env] MOLMETAL_DATA_DIR=$MOLMETAL_DATA_DIR"
echo "[mol-metal env] python (uv) =$(uv run --no-sync python -c 'import sys; print(sys.executable)' 2>/dev/null)"

# --- 6. Force VIRTUAL_ENV to match UV_PROJECT_ENVIRONMENT. uv 0.11.x
#     ignores UV_PROJECT_ENVIRONMENT for unmanaged projects unless the
#     VIRTUAL_ENV var also points at the same venv; without this, `uv run`
#     falls back to the managed Python interpreter and the chemistry
#     deps (rdkit, posebusters, torch+cu130) are absent, which surfaces as
#     `[beta_reductions] SyntheMol registration failed: No module named
#     'rdkit'` and a "122 failed" pytest outcome in
#     molmetal/tests/test_f2a_metal_ligand_exchange.py.
#     Single source of truth is still UV_PROJECT_ENVIRONMENT above; we
#     just mirror it into VIRTUAL_ENV so uv picks it up under 0.11.8.
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
