# Wf-84 Phase 2 — Vina install STUB_FALLBACK

**Date**: 2026-10-05T01:48:45+0800
**Status**: STUB_FALLBACK — no real Vina binary reachable on this host.

## What was attempted

| # | Channel | Result | Notes |
|---|---------|--------|-------|
| - | - | - | no channels attempted |

## Why this happened

all install attempts failed; see attempts[]

## What the bench script does in this state

`molmetal.scripts.bench_vina_triple_rebench._detect_vina_binary()`
returns ``status="MISSING_DEP"`` and every ``(pocket, cand, seed)`` cell
falls back to ``STUB_FALLBACK`` with ``score_kcal_mol=null``. The bench
**never fabricates a Vina score** — see `Wf-60 Phase 2` framing
(`molmetal/reports/wf60_vina_triple/vina_binary_status.json`).

## How to recover

1. **Easiest** — point at an existing install:

   ```bash
   export MOLMETAL_VINA_BIN=/path/to/vina      # e.g. /home/hugo/.mamba/bin/vina
   ```

2. **Conda-forge** — preferred fresh install:

   ```bash
   micromamba create -n molmetal_vina -c conda-forge vina -y
   export MOLMETAL_VINA_BIN=$HOME/micromamba/envs/molmetal_vina/bin/vina
   ```

3. **Debian/Ubuntu**:

   ```bash
   sudo apt-get install -y autodock-vina
   ```

4. **From source** (slow, last resort; requires `cmake` + boost):

   ```bash
   git clone https://github.com/ccsb-scripps/AutoDock-Vina.git
   cd AutoDock-Vina/build && cmake .. && make -j$(nproc)
   export MOLMETAL_VINA_BIN=$(pwd)/vina
   ```

5. **Re-run** the installer to (re)write the wrapper:

   ```bash
   python -m molmetal.scripts.install_vina_binary
   ```

The installer will detect the env override, symlink the binary into
`molmetal/bin/vina`, and the bench's `MISSING_DEP` row flips to
`REAL_VINA_AVAILABLE`.
