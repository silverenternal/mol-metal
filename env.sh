# Mol-Metal — local-host env wiring (NVIDIA Blackwell, 2026-09-25)
#
# Source this file in your interactive shell BEFORE running anything:
#
#     source env.sh           # adds CUDA env vars, routes uv to shared env
#     uv run python ...       # now uses /home/hugo/codes/CadButEaas/.venv-vlm
#
# What this does:
#   1. UV_PROJECT_ENVIRONMENT — points uv at the pre-existing
#      /home/hugo/codes/CadButEaas/.venv-vlm so uv sync / uv run install
#      and execute in the shared env (torch 2.14.0+cu130, triton 3.8.0,
#      rdkit, openbabel, openmm, posebusters, meeko, all chemistry libs
#      are already installed). Avoids re-downloading torch+cu130 (~3GB).
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

# --- 0. Stale-venv guard. If mol-metal/.venv exists, warn loudly so it does
#     not silently shadow the shared CadButEaas env (which we route uv to via
#     UV_PROJECT_ENVIRONMENT below). Run molmetal/scripts/check_no_stale_venv.sh
#     in CI / pre-commit to fail if the stale venv reappears.
STALE_VENV="/home/hugo/codes/mol-metal/.venv"
if [ -d "$STALE_VENV" ]; then
    echo "[mol-metal env] WARNING: stale $STALE_VENV exists." >&2
    echo "[mol-metal env]          uv would auto-activate it and shadow the shared env." >&2
    echo "[mol-metal env]          Remove it: rm -rf $STALE_VENV" >&2
fi

# --- 1. uv: install into the shared env instead of creating mol-metal/.venv
export UV_PROJECT_ENVIRONMENT=/home/hugo/codes/CadButEaas/.venv-vlm

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
export CUDA_VISIBLE_DEVICES=0

# --- Echo so we know we're wired up.
echo "[mol-metal env] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "[mol-metal env] MOLMETAL_DATA_DIR=$MOLMETAL_DATA_DIR"
echo "[mol-metal env] python (uv) =$(uv run --no-sync python -c 'import sys; print(sys.executable)' 2>/dev/null)"

# --- 6. Force VIRTUAL_ENV to match UV_PROJECT_ENVIRONMENT. uv 0.11.x
#     ignores UV_PROJECT_ENVIRONMENT for unmanaged projects unless the
#     VIRTUAL_ENV var also points at the same venv; without this, `uv run`
#     falls back to the managed Python interpreter and the chemistry
#     deps (rdkit, openbabel, torch+cu130) are absent, which surfaces as
#     `[beta_reductions] SyntheMol registration failed: No module named
#     'rdkit'` and a "122 failed" pytest outcome in
#     molmetal/tests/test_f2a_metal_ligand_exchange.py.
#     Single source of truth is still UV_PROJECT_ENVIRONMENT above; we
#     just mirror it into VIRTUAL_ENV so uv picks it up under 0.11.8.
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
