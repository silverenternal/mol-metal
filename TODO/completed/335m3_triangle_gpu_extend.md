# Wf-335-M3 — Triangle GPU wrapper extension to PlatinAI 4-cell + MetalCytoToxDB 5-metal

**Status:** SHIPPED (CPU-only wrapper; GPU launch deferred per AGENTS.md §5.1)
**Date:** 2026-09-29
**Author:** Wf-335-M3 (ultracode Triangle closure sub-flow)
**Cross-ref:** Wf-333-G2 (`TODO/completed/333g2_triangle_gpu_wrapper.md`), Wf-330-T1 (`TODO/completed/330t1_triangle_platinai.md`), Wf-330-T2 (`TODO/completed/330t2_triangle_metalcytotox.md`), Wf-331-R2 (12-d F5_RDKIT_2D CPU proxy)

---

## §1 Purpose

Wf-335-M3 extends the Wf-333-G2 Triangle 12-d GPU launch wrapper to:

1. **PlatinAI 4-cell** (per-cell breakdown, 7 metals: Pt / Pd / Ru / Rh / Ir / Os / Au)
2. **MetalCytoToxDB 5-metal** (per-metal breakdown, 5 metals: Ru / Ir / Rh / Os / Re)

The extension is **additive** — the Wf-333-G2 wrapper contract is preserved
end-to-end (`coef`, default `12d` / `hilbert_dim=64` / `launch=False` /
`mixed_precision=bf16` / `cuda_visible_devices=1` / `torch_compile=False`)
plus two NEW CLI flags:

- `--triangle-cohort {platinai_4cell, metalcytotox_5metal, both}` (default `both`)
- `--triangle-feature-mode {5d, 12d, 17d}` (default `12d`)

Per AGENTS.md §5.1, **no GPU launch** is performed — the wrapper is CPU-only
scaffolding.  Downstream Wf-333-G3+ runs the actual launch when VRAM frees up.

---

## §2 What M3 ships

### §2.1 Wrapper extension (`molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py`)

| Change | Lines | Purpose |
|---|---|---|
| `TRIANGLE_COHORT_REGISTRY` + cohort constants | ~70 LOC | Single source of truth for PlatinAI 4-cell + MetalCytoToxDB 5-metal cohort metadata (metal lists, projected R², recovery_status, wf_closeout) |
| `resolve_triangle_cohorts(cohort)` | ~15 LOC | Resolves `cohort → tuple of active cohorts` (handles `"both"`) |
| `get_cohort_metadata(cohort)` | ~10 LOC | Thin wrapper over `TRIANGLE_COHORT_REGISTRY` |
| `build_per_cohort_breakdown(cohort, feature_mode)` | ~30 LOC | Per-cohort record with cohort_kind + projected per-metal R² + PROJECTED tag |
| `build_full_per_cohort_breakdown(cohort, feature_mode)` | ~10 LOC | Builds the full `Dict[cohort, breakdown]` for `per_cohort_breakdown` |
| `TriangleGPUConfig.triangle_cohort` / `triangle_feature_mode` fields | ~10 LOC | New dataclass fields |
| `TriangleGPUSummary.per_cohort_breakdown` field | ~5 LOC | New dataclass field |
| `add_wf333g2_args` new flags | ~15 LOC | `--triangle-cohort` + `--triangle-feature-mode` |
| `parse_args` propagation | ~10 LOC | New flags → `TriangleGPUConfig` |
| `build_triangle_gpu_trainer` per-cohort population + honest-neg flags | ~25 LOC | Populates `per_cohort_breakdown`; emits per-cohort honest-negative flags |
| `write_summary` per-cohort table | ~25 LOC | MD table with cohort_kind, PROJECTED mean R², recovery_status, closeout |

### §2.2 Test file (`molmetal/tests/test_wf335m3_triangle_gpu_extend.py`)

4 NEW tests (per Wf-335-M3 task spec):

| Test | Locks |
|---|---|
| `test_platinai_4cell_per_cell_breakdown` | PlatinAI 4-cell cohort (7 metals, per_cell, projected_mean=0.594, projected_per_metal=skill §4.2 anchors, HONEST_NEGATIVE_PERSISTENT flag, closeout ref) |
| `test_metalcytotox_5metal_per_metal_breakdown` | MetalCytoToxDB 5-metal cohort (5 metals, per_metal, projected_mean=0.622, projected_per_metal=skill §4.3 anchors, HONEST_NEGATIVE_BELOW_TARGET flag, closeout ref) |
| `test_feature_modes_5d_12d_17d` | All 3 feature modes on both cohorts — feature_shape=(n, 5/12/17), CLI parsing, `resolve_triangle_cohorts`, cohort registry sanity |
| `test_gpu_wrapper_initialization_cpu_only` | Regression guard for Wf-333-G2 contract preserved end-to-end (no GPU; toy matrix is numpy float32; both cohorts in breakdown when `triangle_cohort=both`; idempotent `add_wf333g2_args`) |

Test results (CPU-only):

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf335m3_triangle_gpu_extend.py -v

test_platinai_4cell_per_cell_breakdown           PASSED
test_metalcytotox_5metal_per_metal_breakdown    PASSED
test_feature_modes_5d_12d_17d                   PASSED
test_gpu_wrapper_initialization_cpu_only        PASSED

========================= 4 passed, 1 warning in 0.31s =========================
```

Regression check (Wf-333-G2 existing tests):

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf333g2_triangle_gpu_wrapper.py -v

test_wrapper_initialization_cpu_only           PASSED
test_wrapper_cli_args_and_default_config        PASSED
test_wrapper_module_imports_clean              PASSED

========================= 3 passed, 1 warning in 0.33s =========================
```

CLI smoke (no GPU launch):

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m molmetal.scripts.wf333g2_triangle_12d_gpu_wrapper \
    --triangle-cohort both --triangle-feature-mode 12d \
    --output /tmp/wf335m3_smoke

Wf-333-G2 — Triangle 12-d GPU launch wrapper (dry-run default)
  feature_mode = 12d
  hilbert_dim  = 64
  ...
  launch       = False (False = dry-run)

Wrote /tmp/wf335m3_smoke/summary.json
Wrote /tmp/wf335m3_smoke/summary.md

Honest-negative flags: [
  'HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED',
  'HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL',
  'HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL',
]
Project tag: PROJECTED
Wall time: 0.006s

Triangle cohort breakdown (both, feature_mode=12d):
  - platinai_4cell: 7 metals (per_cell); PROJECTED mean R²=0.5940;
    recovery=HONEST_NEGATIVE_PERSISTENT
  - metalcytotox_5metal: 5 metals (per_metal); PROJECTED mean R²=0.6220;
    recovery=MEASURED=BELOW_TARGET

STATUS: SKIP (HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED — wrapper dry-run only)
```

---

## §3 CLI surface (Wf-335-M3 extensions)

| Flag | Choices | Default | Purpose |
|---|---|---|---|
| `--triangle-cohort` | `platinai_4cell` / `metalcytotox_5metal` / `both` | `both` | Cohort selector — emits per-cell / per-metal R² breakdown anchored to Wf-330-T1 + Wf-330-T2 closeouts |
| `--triangle-feature-mode` | `5d` / `12d` / `17d` | `12d` | Per-cohort feature mode (mirrors `--feature-mode` for clarity; defaults to same value) |

The pre-existing `--feature-mode` flag is preserved for backward compat — it
shares the same `("5d", "12d", "17d")` vocabulary.  When the user sets
`--triangle-feature-mode` to a different value than `--feature-mode`, the
wrapper uses `triangle_feature_mode` for the actual design matrix shape
(cohort-aware).

---

## §4 Per-cohort breakdown (Wf-335-M3)

### §4.1 PlatinAI 4-cell (per-cell, 7 metals)

Anchored to Wf-330-T1 + `mol-metal-triangle-closure` skill §4.2 PROJECTED.

| Cell | n (skill §4.2) | PROJECTED `mean_hilbert_r2` |
|---|---:|---:|
| Pt | 600 | 0.62 |
| Pd | 580 | 0.61 |
| Ru | 590 | 0.60 |
| Rh | 540 | 0.60 |
| Ir | 590 | 0.59 |
| Os | 540 | 0.58 |
| Au | 638 | 0.58 |
| **mean (unweighted)** | **4,078** | **0.594** (skill ledger) / **0.5971** (true mean; locked by Wf-330-T2 `test_7_metal_cohort_projected_mean`) |
| Cohort kind | `per_cell` | — |
| Recovery status | `HONEST_NEGATIVE_PERSISTENT` | Wf-330-T1 MEASURED: per-cell mean hilbert R² = -194.4127 (BELOW PROJECTED 0.594) |
| Closeout | `TODO/completed/330t1_triangle_platinai.md` | — |

### §4.2 MetalCytoToxDB 5-metal (per-metal, 5 metals)

Anchored to Wf-330-T2 + `mol-metal-triangle-closure` skill §4.3 PROJECTED.

| Metal | n (skill §4.3) | PROJECTED `mean_hilbert_r2` |
|---|---:|---:|
| Ru | 160K | 0.65 |
| Ir | 160K | 0.63 |
| Rh | 160K | 0.62 |
| Os | 160K | 0.61 |
| Re | 160K | 0.60 |
| **mean (unweighted)** | **800K** | **0.622** (PROJECTED) |
| Cohort kind | `per_metal` | — |
| Recovery status | `MEASURED=BELOW_TARGET` | Wf-330-T2 MEASURED: cohort mean hilbert R² = 0.4156 (BELOW PROJECTED 0.622) |
| Closeout | `TODO/completed/330t2_triangle_metalcytotox.md` | — |

### §4.3 Honesty gates per cohort

| Cohort | Honesty gate status (Wf-335-M3) |
|---|---|
| **platinai_4cell** | `HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL` — 12-d CPU proxy achieved -194.41 mean_hilbert_r2 (Wf-330-T1 MEASURED); requires TODO-93 R20-P2 GPU joint retrain. |
| **metalcytotox_5metal** | `HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL` — F5_RDKIT_2D 12-d integration is alive (non-zero feature variance); MEASURED cohort mean 0.4156 BELOW PROJECTED 0.622. |
| **wrapper (aggregate)** | `HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED` — wrapper is dry-run only; actual GPU launch deferred per AGENTS.md §5.1. |

---

## §5 Architectural decisions

1. **Additive extension (no mutation of Wf-333-G2 contract)**.  The
   `TriangleGPUConfig` defaults remain `feature_mode=12d / hilbert_dim=64 /
   launch=False / mixed_precision=bf16 / cuda_visible_devices=1 /
   torch_compile=False`.  The new fields (`triangle_cohort`, `triangle_feature_mode`)
   default to `both` / `12d` and are independent of the existing flags.

2. **Cohort registry single source of truth**.  `TRIANGLE_COHORT_REGISTRY`
   carries the per-cohort metal list, projected R² anchors, recovery_status,
   and `wf_closeout` doc reference.  This mirrors the Wf-330-T2
   `DATASET_REGISTRY` + `METALCYTO_5METAL_PROJECTED_R2` pattern in
   `bench_lambda_triangle_end_to_end.py` byte-for-byte — re-using the
   Wf-330-T2 PROJECTED R² constants as the Wf-335-M3 anchors.

3. **`triangle_feature_mode` overrides `feature_mode` when set differently**.
   The wrapper resolves the design matrix schema from
   `triangle_feature_mode` (falling back to `feature_mode` for backward
   compat).  This allows per-cohort feature ablation: e.g.
   `--triangle-cohort platinai_4cell --triangle-feature-mode 5d` vs
   `--triangle-cohort platinai_4cell --triangle-feature-mode 12d` to
   compare the Wf-49 P1 12-d RDKit lift against the legacy 5-d structural
   features on the PlatinAI 4-cell cohort.

4. **Honest-negative propagation**.  Per-cohort `recovery_status` is
   converted into a stable honest-negative flag string:
   `HONEST_NEGATIVE_PERSISTENT_PLATINAI_4CELL` (from Wf-330-T1
   `HONEST_NEGATIVE_PERSISTENT`) and
   `HONEST_NEGATIVE_BELOW_TARGET_METALCYTO_5METAL` (from Wf-330-T2
   `MEASURED=BELOW_TARGET`).  These appear in `summary.honest_negative_flags`
   and in the JSON output for downstream tooling.

5. **CPU-only by design**.  Per AGENTS.md §5.1, the wrapper does NOT
   launch the GPU.  `build_triangle_gpu_trainer` builds the toy
   design matrix with `numpy.random.default_rng` (CPU), and the Hilbert
   torch module is constructed in CPU fp32.  Downstream Wf-333-G3+ is
   responsible for the actual GPU launch (CUDA + free VRAM).

---

## §6 Files touched

| File | Change |
|---|---|
| `molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py` | +210 LOC (cohort registry, per-cohort breakdown helpers, 2 new CLI flags, `TriangleGPUConfig` + `TriangleGPUSummary` field extensions, per-cohort table in `write_summary`) |
| `molmetal/tests/test_wf335m3_triangle_gpu_extend.py` | NEW file — 4 NEW tests (PlatinAI 4-cell + MetalCytoToxDB 5-metal + feature modes + GPU wrapper CPU-only initialization) |
| `TODO/completed/335m3_triangle_gpu_extend.md` | NEW — this file |

---

## §7 Integration points (downstream consumers)

| Consumer | Path | What to do |
|---|---|---|
| **Wf-333-G3+** | TODO/pending | Re-run `build_triangle_gpu_trainer` with `config.launch=True` and `config.triangle_cohort=both` when CUDA + free VRAM available; replace per-cohort PROJECTED R² with real GPU-measured values. |
| **Wf-51 R20-P2 joint retrain (TODO-93)** | TODO/pending/93_r20_p2_hilbert_astsr_gpu.md | Consume `TriangleGPUConfig(triangle_cohort="both", triangle_feature_mode="12d")` as the wrapper launch config; emit per-cohort JSON for paper §3.6. |
| **Wf-48 R21-P4 triangle ship** | TODO | Consume `summary.json["per_cohort_breakdown"]`; per-cell R² table for PlatinAI 4-cell + per-metal R² table for MetalCytoToxDB 5-metal. |
| **paper §3.6** | (READ-only) | Cite the per-cohort PROJECTED R² ledger; mark MEASURED launches pending Wf-333-G3+. |

---

## §8 Cross-references

- `TODO/completed/333g2_triangle_gpu_wrapper.md` — Wf-333-G2 base wrapper
- `TODO/completed/330t1_triangle_platinai.md` — PlatinAI 4-cell CPU bench
- `TODO/completed/330t2_triangle_metalcytotox.md` — MetalCytoToxDB 5-metal CPU bench
- `TODO/completed/331r2_triangle_12d_cpu_scaffold.md` — Wf-331-R2 12-d CPU proxy lift (+1.024)
- `molmetal/scripts/triangle_features_12d.py` — 12-d RDKit feature schema (Wf-331-R2 ship)
- `molmetal/scripts/triangle_closure.py` — Wf-37 L8 base orchestrator
- `molmetal/scripts/bench_lambda_triangle_end_to_end.py` — Wf-330-T2 `DATASET_REGISTRY` + `METALCYTO_5METAL_PROJECTED_R2` pattern (mirrored byte-for-byte)
- `.claude/skills/mol-metal-triangle-closure/SKILL.md` §4.2 (PlatinAI 4-cell PROJECTED) + §4.3 (MetalCytoToxDB 5-metal PROJECTED)
- AGENTS.md §5.1 (no GPU launch), §5.2 (honest framing)

---

## §9 Honest framing summary

- **MEASURED anchor (Wf-330-T1)**: PlatinAI 4-cell per-cell mean
  `mean_hilbert_r2 = -194.4127` (BELOW PROJECTED 0.594).
- **MEASURED anchor (Wf-330-T2)**: MetalCytoToxDB 5-metal cohort mean
  `mean_hilbert_r2 = 0.4156` (BELOW PROJECTED 0.622).
- **PROJECTED target (R21-P4 / TODO-93)**: mean_hilbert_r² ≥ 0.594
  on PlatinAI 4-cell + ≥ 0.622 on MetalCytoToxDB 5-metal requires the
  GPU launch (Wf-333-G3+ / TODO-93 R20-P2 + Wf-49 P2 GPU ship surface).
- **HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED** applies until the launch runs.
- **MEASURED vs PROJECTED discipline**: every numeric value in this doc
  traces to either the Wf-330-T1 / Wf-330-T2 closeouts or the
  `mol-metal-triangle-closure` skill §4.2 / §4.3 PROJECTED ledger.
  No fabricated numbers.
- **No GPU launch per AGENTS.md §5.1**: the wrapper is CPU-only
  scaffolding; downstream Wf-333-G3+ runs the actual launch when
  VRAM frees up on the second RTX 5090 or RTX PRO 6000 Blackwell.

---

**END OF Wf-335-M3 SHIP RECORD** — last updated 2026-09-29.
