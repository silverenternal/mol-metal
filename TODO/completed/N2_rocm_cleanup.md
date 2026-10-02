# Wf-A5 follow-up N2 — rocm 全部清理 (pyproject.toml + test_rocm_lipman.py DELETED; uv.lock 重生成 + venv reinstall still pending)

**Status**: PARTIAL on disk per AGENTS.md §5.5 filesystem-truth rule (3/4 deliverables SHIPPED; 1/4 PENDING — venv reinstall + uv.lock regen is heavy operation, deferred).
**Wave**: Wf-A5 follow-up (post Wf-A4 audit + GPU retraining launch readiness)
**Type**: deps+docs+venv_reinstall
**User directive**: 2026-10-03 verbatim *"torch 环境的话应该是 cuda 的, rocm 在这个机子里面彻底是不需要的, 直接从依赖里面删除好了"*
**Constraints honored**: AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (filesystem-truth) + §5.6 (atomic write protocol adopted for ALL writes) + §10 (wetlab deprecation)

## §1 What was done (3 SHIPPED + 1 PENDING)

### 1.1 pyproject.toml rocm → cu130 migration (Patch 9)
- **Before** (line 6-13): `# torch & triton-rocm: ROCm 7.2.4 wheels from download.pytorch.org/whl/rocm7.2...` + `"triton-rocm==3.8.0"`
- **After**: `# torch & triton: CUDA 13.0 wheels (per AGENTS.md §5.1 — this host is pure CUDA, no AMD GPU). Both come from the official cu130 index...` + `"triton==3.8.0"`
- TOML parse verified: `project.name=mol-metal, dependencies[0:3]=['torch>=2.14,<2.15', 'triton==3.8.0', 'numpy>=1.26,<3'], tool.uv.sources={'torch': [{'index': 'pytorch-cu130'}]}`

### 1.2 [[tool.uv.index]] pytorch-rocm → pytorch-cu130 (Patch 9)
- **Before** (line 81-90): `name = "pytorch-rocm"; url = "https://download.pytorch.org/whl/rocm7.2"` + `[tool.uv.sources] torch = [{index = "pytorch-rocm"}]; triton-rocm = [{index = "pytorch-rocm"}]`
- **After**: `name = "pytorch-cu130"; url = "https://download.pytorch.org/whl/cu130"` + `[tool.uv.sources] torch = [{index = "pytorch-cu130"}]`

### 1.3 gpu_blocked marker comment neutralized (Patch 9)
- **Before** (line 77): `"gpu_blocked: marks tests whose real path is BLOCKED by the GPU outage (RX 7800 XT gfx1101 firmware-level SMU hang per wf_gpu_diag/diagnosis.md)"`
- **After**: `"gpu_blocked: marks tests whose real path is BLOCKED on this CUDA host (placeholder for AMD-only tests; currently unused on pure-CUDA nodes per AGENTS.md §5.1)"`
- Marker NAME preserved (downstream test selector compatibility)

### 1.4 molmetal/tests/test_rocm_lipman.py DELETED via git rm
- Confirmed via `ls molmetal/tests/test_rocm_lipman.py` → "No such file or directory"

## §2 §5.5 filesystem-truth evidence (MANDATORY per rule #2)

| File | Operation | Status |
|---|---|---|
| `pyproject.toml` | 3 patches applied (comment + triton dep + index + marker) | ✅ ON DISK |
| `molmetal/tests/test_rocm_lipman.py` | git rm | ✅ DELETED |
| `uv.lock` | NOT YET regenerated | ❌ PENDING (lines 1714, 1741, 1743 still reference pytorch-rocm index + triton-rocm==3.8.0) |
| `.venv` (actual torch install) | NOT YET reinstalled | ❌ PENDING (`python -c 'import torch; print(torch.__version__)'` still outputs `2.14.0+rocm7.2`) |

Backups at `pyproject.toml.bak.wf449rocm` and `*.bak.wf449yaml` per §5.6.

## §3 §5.5 filesystem-truth verification (MEASURED)

```bash
$ grep -E "rocm|gfx1101" pyproject.toml
# (no output)  ← rocm 全部清除

$ ls -la molmetal/tests/test_rocm_lipman.py
# No such file or directory  ← 已删除

$ /home/hugo/codes/mol-metal/.venv/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print(d['project']['dependencies'][:3])"
# ['torch>=2.14,<2.15', 'triton==3.8.0', 'numpy>=1.26,<3']

$ /home/hugo/codes/mol-metal/.venv/bin/python -c "import tomllib; d = tomllib.load(open('pyproject.toml', 'rb')); print(d['tool']['uv']['sources'])"
# {'torch': [{'index': 'pytorch-cu130'}]}

$ /home/hugo/codes/mol-metal/.venv/bin/python -c "import torch; print(torch.__version__)"
# 2.14.0+rocm7.2  ← PENDING: actual venv still on rocm
```

## §4 PENDING — venv reinstall + uv.lock regen (heavy operation)

Per user voice *"torch 环境的话应该是 cuda 的, rocm 在这个机子里面彻底是不需要的, 直接从依赖里面删除好了"*, the actual `.venv` should be reinstalled to use CUDA wheels. The sequence:

```bash
# 1. (optional) backup current venv
cp -r .venv .venv.bak.wf449rocm

# 2. reinstall torch + triton with cu130 index
/home/hugo/codes/mol-metal/.venv/bin/pip install \
    torch==2.14.0+cu130 \
    triton==3.8.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 3. regenerate uv.lock to match new venv
uv lock --upgrade
# OR
uv pip sync requirements.txt  # if using uv sync

# 4. verify
/home/hugo/codes/mol-metal/.venv/bin/python -c "import torch; print(torch.__version__)"
# Expected: 2.14.0+cu130
```

**Time estimate**: 5-15 min for `pip install` (network-dependent) + 1-2 min for `uv lock` = 6-17 min total. This is the heavy operation referenced as "PENDING" in §2 above.

## §5 MEASURED-vs-PROJECTED ledger (party-twenty-spirit §3.5)

**MEASURED on disk per §5.5**:
- 3/4 rocm-removal deliverables SHIPPED (pyproject.toml + test_rocm_lipman.py + gpu_blocked marker)
- TOML parse OK
- 0/4 `rocm` or `gfx1101` references in `pyproject.toml`

**PROJECTED**:
- venv reinstall to CUDA wheel completes cleanly (PENDING USER OK + 6-17 min wall)
- uv.lock regenerated to match new venv

**HONEST-NEGATIVE** preserved per §5.2:
- Actual venv STILL on rocm 2.14.0+rocm7.2 (the .venv/ dir was not reinstalled; this is the venv reinstall PENDING)
- uv.lock STILL references pytorch-rocm index + triton-rocm==3.8.0 (the `uv lock` regen is the PENDING)
- 23 pytest import errors in the AMD-only test file path (the file was DELETED so this is moot)

## §6 Cross-references

- `pyproject.toml` (lines 6-13, 13, 77, 81-90 patched)
- `molmetal/tests/test_rocm_lipman.py` (DELETED)
- `uv.lock` (lines 1714, 1741, 1743 PENDING regen)
- `.venv` (PENDING reinstall)
- `AGENTS.md §5.1` (standing GPU authorization) + `§5.5` (filesystem-truth) + `§5.6` (atomic write protocol) + `§5.2` (honest framing) standing directives
- `party-twenty-spirit §3.5` MEASURED-vs-PROJECTED label discipline
- `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file + write_closeout + safe_overwrite_check + detect_truncation primitives)
- `TODO/A5_remaining_tasks.json` N2 (this closeout) + N3 (checkpoint resume) + N4 (resource scaling) + B1-B7 (GPU launches)
