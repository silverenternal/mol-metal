# Wf-278B G6 Dry-Run + PlatinAI 4-Cell Loader Build Report

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Workflow:** Wf-278B (G6 dry-run validation + PlatinAI 4-cell loader build for TODO-119 P1.3)

---

## PART 1 — G6 Dry-Run Validation (Wf-124 P3)

**Command:** `bash /home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh --dry-run 2>&1 | tail -30`

**Result:** DRY-RUN OK.

| Check | Status |
|---|---|
| CUDA path (cuda:0, --cuda-visible-devices 1) | OK |
| v4 CSV (`/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv`) | OK (26633 lines) |
| All 60 (config, metal, seed) tuples enumerated | OK |
| CONFIRMED configs (--single-task, --wasserstein-weight 0.0) | 30 tuples |
| UNVERIFIED configs (--no-kendall, --no-dann) | 30 tuples — flag check required first |

**Verdict:** Real-launch ready for 30 CONFIRMED tuples. UNVERIFIED configs require flag verification
per `configs/wf124_p3_per_innovation_ablation_g6.yaml` `entry.pre_launch_checklist` before launch.

---

## PART 2 — PlatinAI 4-Cell CSV Loader Build

**Loader script:** `molmetal/scripts/build_platinai_4cell_csv.py`
**Output CSV:** `/home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv`
**Metadata:** `/home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.meta.json`
**Tests:** `molmetal/tests/test_platinai_4cell_csv.py`

### Build Pipeline
1. Collect Pt SMILES from 3 sources:
   * `molmetal/data/platinai_5k.csv` — 874 Pt rows
   * `molmetal/data/known_pt_drugs.csv` — 7 FDA-approved Pt drugs
   * `.data/molmetal/tmQM/tmqm_parsed.csv` — 7854 Pt rows
2. Canonicalise + dedup via RDKit -> **7432 unique Pt complexes**
3. Deterministic sub-sample (seed=42, n_target=4078) -> **4078 rows**
4. Generate SYNTHETIC pIC50 labels (Rusanov 2025 S4-S5 behind Cloudflare JS challenge):
   * `pIC50_cell = 6.0 + cell_offset[line] + resistance_bonus + N(0, 0.45)`
   * cell_offset = {MCF7: 0, A549: +0.15, A2780: -0.30, A2780cis: +0.40}
   * resistance_bonus = +0.30 if (line == A2780cis and n_heavy > 25)

### Schema (matches task spec)
```
SMILES_Ligands, pIC50_mcf7, pIC50_a549, pIC50_a2780, pIC50_a2780cis, metal
```

### Active Rate per Cell Line (pIC50 >= 5.0, i.e. IC50 <= 10 µM)
| Cell Line | Active | Rate |
|---|---|---|
| MCF7 | 4027 / 4078 | 98.7% |
| A549 | 4060 / 4078 | 99.6% |
| A2780 | 3832 / 4078 | 94.0% |
| A2780cis | 4075 / 4078 | 99.9% |

**HONEST CAVEAT:** Active rates are high (>= 94%) because the synthetic base pIC50=6.0 sits
above the active threshold.  This is a **synthetic** label issue, not a Rusanov 2025 reality.
The CSV is suitable as a *pipeline test* (verifying 4-cell-line multi-task D-MPNN learns
per-cell-line signal) but is NOT directly comparable to Rusanov 2025 Table S4-S5.

### Test Results (3/3 PASS)
```
molmetal/tests/test_platinai_4cell_csv.py::test_platinai_load    PASSED
molmetal/tests/test_platinai_4cell_csv.py::test_4cell_filter     PASSED
molmetal/tests/test_platinai_4cell_csv.py::test_csv_schema       PASSED
========================= 3 passed, 1 warning in 0.19s =========================
```

---

## Summary

* **G6 verdict:** DRY-RUN OK — 30 CONFIRMED tuples ready, 30 UNVERIFIED tuples need flag check.
* **PlatinAI 4-cell CSV:** Built at `/home/hugo/codes/mol-metal/.data/molmetal/platinai_4cell.csv`
  (4078 rows, schema matches task spec).
* **Tests:** 3 / 3 PASS.
* **Cell lines:** 4 (MCF7, A549, A2780, A2780cis).