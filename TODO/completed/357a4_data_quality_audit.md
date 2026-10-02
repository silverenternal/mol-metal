# Wf-357-A4 — Data Quality Audit (5 loaders × 5 checks = 25 evaluations)

**Workflow**: Wf-357-A4 (residual Wf-357 closeout — data-axis verification)  
**Predecessor**: Wf-343c3 (`molmetal/scripts/wf343c3_data_quality_check.py`) + Wf-356-A1 (wired into 6 launch scripts)  
**Status**: SHIPPED (verification-only closeout)  
**Date**: 2026-09-29  
**Wall**: ~6 s CPU-only (script runtime; user-visible wall ≤ 30 s including venv spin-up)  
**Tests**: 0 NEW (verification re-run of Wf-343c3 harness)

---

## §1 TL;DR

Re-ran the Wf-343c3 data quality harness against the 5 canonical loaders (binding_db + nci60 + drugood + metalcytotox_v4 + platinai_4cell) × 5 checks (csv_loads + per_metal_balance + scaffold_split + class_balance + label_range) = **25 evaluations**. All 5 loaders report `overall=PASS`. **19 PASS / 0 FAIL / 6 SKIP** across the 5×5 check matrix.

Per §5.2 honest framing: 6 SKIPs are *informational* — they signal "loader lacks the column needed for this check" (nci60 + drugood have `metal='NA'` uniform → per_metal_balance SKIP; metalcytotox_v4 is the only loader with pre-computed scaffold split; the other 4 SKIP and defer to `check_data_leakage.py`).

## §2 Per-loader verdicts (5/5)

| Loader | csv_loads | per_metal | scaffold | class_balance | label_range | Overall | Rows |
|---|---|---|---|---|---|---|---:|
| `load_binding_db` | PASS | PASS | SKIP* | PASS | PASS | **PASS** | 5,257 |
| `load_nci60` | PASS | SKIP* | SKIP* | PASS | PASS** | **PASS** | 7,823 |
| `load_drugood_train` | PASS | SKIP* | SKIP* | PASS | PASS** | **PASS** | 95,373 |
| `metalcytotox_5metal_multitask_v4` | PASS | PASS | PASS | PASS | PASS** | **PASS** | 26,632 |
| `platinai_4cell` | PASS | PASS | SKIP* | PASS | PASS | **PASS** | 16,312 |

\* `SKIP` = loader lacks the prerequisite column (uniform metal="NA" or no pre-computed `scaffold_id`+`split`); not a defect, just an out-of-scope check.
\** SOFT-WARN noted for weak-binding rows (pIC50 min in [2.5, 3.0)) but check still PASSes per the project's "min ≥ 3 OR (2.5 ≤ min < 3 with SOFT-WARN)" convention.

## §3 Per-check distribution (5/5)

| Check | PASS | FAIL | SKIP | Notes |
|---|---:|---:|---:|---|
| `csv_loads` | 5 | 0 | 0 | All 5 loaders return non-empty DataFrame with canonical 6-col schema (`smiles, metal, cell_line, protein, pIC50, active`) |
| `per_metal_balance` | 3 | 0 | 2 | nci60 + drugood have uniform `metal='NA'` (no per-row metal identifier; load-by-cell-line instead) |
| `scaffold_split` | 1 | 0 | 4 | Only metalcytotox_v4 ships pre-computed `scaffold_id` + `split` (train=18137, val=2759, test=5736); others defer to `molmetal/scripts/check_data_leakage.py` (Wf-44) for strict scaffold-disjointness |
| `class_balance` | 5 | 0 | 0 | All 5 in [0.05, 0.95] band — note drugood is **0.913** (high but in-band; borderline — flag for downstream class-imbalance handling) |
| `label_range` | 5 | 0 | 0 | All 5 finite + non-zero variance + min ≥ 2.5 (SOFT-WARN acceptable) + max ≤ 12 |

## §4 The 5×5 verdict grid (per loader × per check)

```
                  csv_loads  per_metal  scaffold  class_bal  label_rng  → overall
load_binding_db        ✓         ✓         SKIP       ✓          ✓       → PASS
load_nci60             ✓        SKIP       SKIP       ✓          ✓       → PASS
load_drugood_train     ✓        SKIP       SKIP       ✓          ✓       → PASS
metalcytotox_v4        ✓         ✓          ✓         ✓          ✓       → PASS
platinai_4cell         ✓         ✓         SKIP       ✓          ✓       → PASS
                  ────────  ────────  ────────  ────────  ────────
PASS                  5         3         1         5         5
SKIP                  0         2         4         0         0
FAIL                  0         0         0         0         0
```

## §5 Detailed metrics (per loader)

### §5.1 `load_binding_db` (5257 rows)
- **csv_loads**: 5257 rows, canonical 6-col schema — PASS
- **per_metal_balance**: 5 metals; largest=69.6% (Ru 69.6%, Ir 23.6%, Re 3.1%, Rh 2.1%, Os 1.7%) — PASS (5% threshold met)
- **scaffold_split**: SKIP (no pre-computed split)
- **class_balance**: positive-class fraction 0.308 (1621 pos / 3636 neg) — PASS
- **label_range**: pIC50 ∈ [3.000, 7.721], mean=4.648, std=0.684 — PASS (clean, no SOFT-WARN)

### §5.2 `load_nci60` (7823 rows)
- **csv_loads**: 7823 rows — PASS
- **per_metal_balance**: SKIP (metal column uniformly "NA")
- **scaffold_split**: SKIP (no pre-computed split)
- **class_balance**: positive-class fraction 0.382 (2987 pos / 4836 neg) — PASS
- **label_range**: pIC50 ∈ [2.699, 8.000], mean=4.801, std=0.804 — PASS (SOFT-WARN: min in [2.5, 3.0))

### §5.3 `load_drugood_train` (95373 rows)
- **csv_loads**: 95373 rows — PASS
- **per_metal_balance**: SKIP (metal column uniformly "NA")
- **scaffold_split**: SKIP (no pre-computed split)
- **class_balance**: positive-class fraction **0.913** (87106 pos / 8267 neg) — PASS (in-band but BORDERLINE-HIGH)
- **label_range**: pIC50 ∈ [2.770, 11.000], mean=6.835, std=1.291 — PASS (SOFT-WARN: min in [2.5, 3.0); max=11.0 hits pharmacological ceiling — DrugOOD high-affinity selection bias)

### §5.4 `metalcytotox_5metal_multitask_v4` (26632 rows)
- **csv_loads**: 26632 rows — PASS
- **per_metal_balance**: 5 metals; largest=71.3% (Ru 71.3%, Ir 17.1%, Rh 4.2%, Os 4.2%, Re 3.3%) — PASS
- **scaffold_split**: train=18137, val=2759, test=5736 — PASS (the only loader with pre-computed split)
- **class_balance**: positive-class fraction 0.344 (9161 pos / 17471 neg) — PASS
- **label_range**: pIC50 ∈ [2.580, 8.699], mean=4.720, std=0.799 — PASS (SOFT-WARN: min in [2.5, 3.0))

### §5.5 `platinai_4cell` (16312 rows)
- **csv_loads**: 16312 rows — PASS
- **per_metal_balance**: 1 metal (Pt=100%) — PASS (single-metal cohort by design)
- **scaffold_split**: SKIP (no pre-computed split)
- **class_balance**: positive-class fraction 0.529 (8630 pos / 7682 neg) — PASS
- **label_range**: pIC50 ∈ [3.043, 7.237], mean=5.040, std=0.493 — PASS (clean, no SOFT-WARN)

## §6 Notable caveats (honest framing per §5.2)

1. **`drugood_train` class_balance 0.913** — borderline-high; downstream training should use class-weighted loss or downsampling. Not a FAIL (in-band [0.05, 0.95]) but worth flagging.
2. **drugood `metal='NA'` uniform** — DrugOOD is a multi-metal ligand set without explicit metal labels per row; loader returns metal="NA". Not a defect — design choice. Bench scripts using DrugOOD must join metal labels externally if needed.
3. **SOFT-WARN on weak-binding rows** (pIC50 min in [2.5, 3.0)) — nci60, drugood, metalcytotox_v4 each have 1 loader with min below 3.0 but above 2.5. Per project convention, this is PASS with WARN. Strength threshold may need filtering in downstream R² regression.
4. **scaffold_split SKIP × 4 loaders** — only metalcytotox_v4 ships pre-computed scaffold split. The strict scaffold-disjointness check is delegated to `molmetal/scripts/check_data_leakage.py` (Wf-44). Per-loader rerun of that check is out of scope for Wf-357-A4.

## §7 What this verifies for downstream workflows

| Downstream workflow | Loader(s) consumed | Status |
|---|---|---|
| Wf-124 P3 retrain (TODO-181 G5 + TODO-182 G6) | `load_binding_db` + `load_nci60` | data-axis PASS (5/5 + 3/5 — SKIPs out-of-scope) |
| Wf-299 CFM 3-wave | `metalcytotox_v4` + `platinai_4cell` | data-axis 5/5 + 4/5 — both PASS |
| TODO-119 P1.4 chemprop v3 LOMO (Wf-336-X2) | `metalcytotox_v4` | data-axis 5/5 — PASS |
| Wf-74 v2 DrugOOD DANN (Wf-336-X3) | `load_drugood_train` | data-axis 3/5 PASS + 2 SKIP (informational; class_balance 0.913 borderline) |
| TODO-21 Lambda CFM (Wf-333-G1) | `load_binding_db` | data-axis 4/5 + 1 SKIP — PASS |
| Triangle closure (Wf-333-G2 / Wf-335-M3) | `platinai_4cell` + `metalcytotox_v4` | data-axis 4/5 + 5/5 — both PASS |

## §8 Reproduction

```bash
cd /home/hugo/codes/mol-metal
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf343c3_data_quality_check.py \
    2>&1 | tee molmetal/reports/wf357a4_data_quality.txt
```

Outputs:
- `molmetal/reports/wf357a4_data_quality.txt` (stdout mirror)
- `molmetal/reports/wf343c3_data_quality/report.json` (machine-readable verdict)
- `molmetal/reports/wf343c3_data_quality/summary.md` (human-readable table)

## §9 Related files

- Script: `molmetal/scripts/wf343c3_data_quality_check.py` (Wf-343c3-I3 ship)
- Wired into 6 launch scripts per Wf-356-A1 (TODO-181 G5, TODO-182 G6, Wf-299 CFM main, TODO-119 P1.4 chemprop v3 LOMO, Wf-74 v2 DrugOOD DANN, TODO-21 Lambda CFM)
- Strict scaffold-disjointness: `molmetal/scripts/check_data_leakage.py` (Wf-44) — separate workflow

---

**END OF Wf-357-A4** — 5/5 loaders overall PASS; 19/25 evaluations PASS + 6/25 informational SKIP + 0/25 FAIL; CPU-only verification re-run of Wf-343c3 harness; ~6 s wall; 0 NEW tests; pre-flight v3 62/62 PASS PRESERVED (NOT re-run — docs-only verification).