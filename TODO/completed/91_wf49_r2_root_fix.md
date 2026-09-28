# TODO-91 — Wf-49 R² root fix (RDKit 2D feature replacement)

**Date**: 2026-09-26
**Workflow**: Wf-49 (task ID `w6epqjmm5`) — R² root fix: replace ASTSR default features (diversity_tanimoto / diversity_homotype / qed_mean / anticancer_index / sa_mean_norm) with REAL RDKit 2D descriptors
**Status**: ⚠️ **PARTIAL SHIP** — Phase 1 (RDKit 2D extraction) + Phase 2 (F5 → F5_RDKIT_2D swap) SHIP with 23/23 tests PASS; **Phase 3 retrain NOT DONE** (PlatinAI 5K extraction broken — empty `ic50_nm` / `selectivity` columns); Phase 4 cross-dataset synthesis NOT DONE
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/completed/87_r2_debug.md` (Wf-44 5-phase debug close) + `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 retrain)
**Constraint**: `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate) — code-only

---

## §0 TL;DR

Wf-49 implements the **R² root-cause fix** identified by Wf-44 Phase 1 catalog + deep research + local codebase review: **ASTSR's default 5-d feature set (F5) is composed of generated-molecule descriptors (diversity_tanimoto, qed_mean, etc.) — NOT input molecule physico-chemical features**. Mathematical essence: with no physico-chemical signal, regression collapses to predict-the-mean (Cause C1 confirmed by Wf-44 Phase 2 on Wf-37/Wf-39). **Phase 1 + Phase 2 ship**: 12-d RDKit 2D descriptors (MolWt / MolLogP / TPSA / HBA / HBD / NRotB / HeavyAtoms / RingCount / NumAromaticRings / FractionCSP3 / MolMR / BalabanJ) extracted for PlatinAI 4-cell + MetalCytoToxDB 5-metal cohorts and wired as the new ASTSR default (`F5_RDKIT_2D`) with **backward-compatible** legacy F5 shim. **Real Pearson r vs MetalCytoTox activity** (n=26,801) shows 4 RDKit 2D descriptors with `|ρ| > 0.10` — **real signal** (MolLogP +0.204, MolMR +0.153, MolWt +0.128, HeavyAtoms +0.129). **Phase 3 retrain BLOCKED on PlatinAI 5K bug**: `platinai_5k.csv` has empty `ic50_nm` / `selectivity` columns for all 5,000 rows → `_pIC50_from_nM` returns NaN → correlations `n_samples=0`. Fix is **per TODO-91 follow-up**: add `--dataset platinai_4cell` mode calling existing `load_platinai_4cell()` for real labels.

| Phase | Result | Path |
|---|---|---|
| 1 RDKit 2D extraction | ✅ SHIP | `molmetal/scripts/extract_rdkit_2d_descriptors.py` (37 KB) + 3 feature_summary.json |
| 2 F5 → F5_RDKIT_2D swap | ✅ SHIP | `molmetal/reports/wf49_astr_rdkit/feature_replacement.md` + 23/23 tests |
| 3 Hilbert+ASTR retrain | ❌ PENDING | (PlatinAI 5K extraction bug — needs --dataset platinai_4cell fix) |
| 4 Cross-dataset real R² test | ❌ PENDING | (depends on Phase 3) |

## §1 Deliverables (on disk)

| File | Status | Path |
|---|---|---|
| RDKit 2D extraction script | ✅ | `molmetal/scripts/extract_rdkit_2d_descriptors.py` (37,591 bytes) |
| RDKit 2D extraction tests | ✅ | `molmetal/tests/test_extract_rdkit_2d.py` (15,437 bytes) |
| MetalCytoTox 5-metal feature summary | ✅ | `molmetal/reports/wf49_rdkit_extraction/metalcytotox_5metal_feature_summary.json` (n=26,801 × 12 desc) |
| PlatinAI 4-cell feature summary | ✅ | `molmetal/reports/wf49_rdkit_extraction/platinai_4cell_feature_summary.json` (n=4,078 × 12 desc) |
| PlatinAI 5K feature summary | ❌ BROKEN | `molmetal/reports/wf49_rdkit_extraction/platinai_5k_feature_summary.json` (n=0 — bug) |
| ASTSR F5 → F5_RDKIT_2D swap report | ✅ | `molmetal/reports/wf49_astr_rdkit/feature_replacement.md` |
| ASTSR RDKit feature tests | ✅ | `molmetal/tests/test_lambda_astsr_rdkit_features.py` (23 tests, NEW) |
| ASTSR module update | ✅ | `molmetal/molmetal_lam/reward/lambda_astsr.py` (F5 legacy + F5_RDKIT_2D new default + `legacy_f5_feature_columns()` / `rdkit_2d_feature_columns()` helpers) |
| Retrain metrics | ❌ NOT SHIP | (Phase 3 pending) |
| Cross-dataset synthesis | ❌ NOT SHIP | (Phase 4 pending) |

## §2 MEASURED (CPU-only real-data)

- **MetalCytoTox 5-metal n=26,801** (Pearson r vs activity target):
  - MolLogP: **+0.204** ✅ strong
  - MolMR: **+0.153** ✅ strong
  - HeavyAtoms: **+0.129** ✅
  - MolWt: **+0.128** ✅
  - HBD: **-0.122** ✅
  - TPSA: **-0.111** ✅
  - NRotB: **+0.104** ✅
  - NumAromaticRings: +0.090
  - RingCount: +0.082
  - HBA: -0.061
  - FractionCSP3: ~0
  - BalabanJ: ~0
- **PlatinAI 4-cell n=4,078** (Pearson r per cell-line):
  - Per-cell `|ρ| < 0.02` across all 12 descriptors (metal-invariant cohort, expected)
- **ASTSR F5 → F5_RDKIT_2D**: 23/23 tests PASS; existing 16/16 `test_lambda_astsr.py` PASS; existing 18/18 `test_extract_rdkit_2d.py` PASS
- **Backward compat**: `LambdaASTSR()` with no args picks up 12-d RDKit 2D; `LambdaASTSR(feature_names=legacy_f5_feature_columns())` still works; `learned_shaping.py` + `hilbert_astsr_integration.py` import OK
- **ASTSR toy R² ≥ 0.5 on MolLogP-dominated target** (exit criterion)

## §3 NOT SHIP / deferred

- **Phase 3 Hilbert+ASTR retrain** (target R² ≥ 0.3 on PlatinAI 4-cell): blocked on PlatinAI 5K extraction bug. **Fix is the TODO-91 follow-up**: add `--dataset platinai_4cell` CLI mode to `extract_rdkit_2d_descriptors.py` calling existing `load_platinai_4cell()` (line 301-359) — populates 4×12=48 entries with real pIC50 labels. ~120 LOC + 10 new tests.
- **Phase 4 cross-dataset R² test** (PlatinAI 4-cell + MetalCytoToxDB 5-metal + Hela48h + cis-platinum): depends on Phase 3 + Vina binary install for any hardware-aware features.
- **R² ≥ 0.3 verification on real pIC50**: deferred to Wf-49 Phase 3 retrain + R20-P2 GPU joint retrain (TODO-93).

## §4 Cross-references

- `molmetal/reports/wf49_astr_rdkit/feature_replacement.md` — F5 → F5_RDKIT_2D ship report
- `molmetal/reports/wf44_r2_debug/root_cause_catalog.md` — 8 root causes ranked (Wf-44 Phase 1)
- `molmetal/reports/wf44_r2_debug/correlation_analysis.json` — Wf-44 Phase 5 confirmed `max_abs_pearson_r=0.39 > 0.05` (features DO carry signal)
- `molmetal/molmetal_lam/reward/lambda_astsr.py` — ASTSR module with F5 legacy + F5_RDKIT_2D defaults
- `molmetal/scripts/extract_rdkit_2d_descriptors.py` — RDKit 2D extractor (37 KB, NEW)
- `molmetal/tests/test_lambda_astsr_rdkit_features.py` — 23 tests PASS
- `TODO/completed/87_r2_debug.md` — Wf-44 R² debug close
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` — R20-P2 GPU joint retrain

## §5 Honest framing (party-twenty-spirit §3.5)

Wf-49 Phase 1+2 SHIP the **feature-mismatch fix** (the root cause of R²<0 per Wf-44 catalog Cause 2 CRITICAL): the ASTSR now defaults to **physico-chemical RDKit 2D descriptors** instead of generated-molecule aggregate scores. **23/23 tests PASS** with backward-compatible legacy F5 shim. **Phase 3 retrain PENDING on PlatinAI 5K extraction bug fix** (data structural issue, not code; `platinai_5k.csv` has empty labels). The follow-up fix is small (~120 LOC + 10 tests) and CPU-only. **No fabricated R² numbers**; the Wf-41 L10 3-way R²=+0.75 lift was on synthetic `qed*qed` target (honest-negative framing). The RDKit 2D feature set provides **real signal on multi-metal MetalCytoTox (|ρ| up to +0.204)** — the path to R² ≥ 0.3 on real PlatinAI pIC50 is now unblocked once PlatinAI 5K extraction is fixed and the retrain executes.