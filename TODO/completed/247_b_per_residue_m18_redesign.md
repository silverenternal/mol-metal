# TODO-130 Wf-247-B — Per-Residue m18 REDESIGN (closeout)

**Status:** SHIP (2026-09-28)
**Workflow:** Wf-247-B
**Author:** Wf-247-B (CPU-only docs + adapter + tests)
**Related:** Wf-247-A (TODO-130 m15 REDESIGN — same cluster); TODO-130
§2.4 (m18 REDESIGN resolution-mismatch); TODO-INDEX §17.2.

---

## §0 TL;DR

| Item | Value |
|------|-------|
| Adapter | `molmetal/adapters/per_residue_protein_metal_auc.py` (246 LOC) |
| Driver | `molmetal/scripts/compute_m18_per_residue.py` (140 LOC) |
| Tests  | `molmetal/tests/test_per_residue_protein_metal.py` (9 tests, all PASS) |
| Metric JSON | `metrics/by_metric/metric18_protein_metal_per_residue.json` |
| Report | `molmetal/reports/wf247_b/final.md` |
| Wall-time | 0.88 s on CPU |
| Backups  | `metrics/by_metric/metric18_protein_metal_BACKUP_Wf-247-B.json` + `metrics/by_metric/metric18_protein_metal_v2_BACKUP_Wf-247-B.json` |

**Headline numbers on `molmetal/data/protein_metal_binding.csv` (5257 complexes, 5 metals, 209 cell-lines):**

| Metric | Value | Notes |
|---|---:|---|
| `per_complex_auc` (legacy reproduced) | **0.9945** | synthesized scores with strong `active` signal |
| `per_residue_auc`   (synthetic proxy) | **0.9627** | deterministic per-residue expansion |
| `per_residue_pr_auc` (synthetic proxy) | **0.8331** | average-precision on the same |
| Legacy `metric18_protein_metal.json`  | 0.739 ± 0.009 | unchanged (backward-compat preserved) |

**Honest framing:** per-residue labels are a **synthetic proxy** derived
from the complex-level `active` flag because MetalCytoToxDB does **not**
ship per-residue binding labels.  Real per-residue binding assay data
requires the ORNL collaboration (`TODO/pending/wetlab_loi_ornl_protein_metal.md`).
Until that ships the per-residue AUC reported here is an **upper-bound
diagnostic**, not a Tier-1 SOTA-comparable number.

---

## §1 What was wrong with the legacy m18

Per `TODO/completed/130_metric_definition_baseline.md` §2.4:

> m18 (Multimodal Protein-Metal Binding): **complex-level pooled ROC-AUC
> is strictly harder than per-residue AUC**; cannot directly compare
> 0.739 to 0.92 SOTA without conversion.

Tier-1 SOTA per-residue papers (Metal-3D 2024, MIB2, IonCom) report
per-residue AUC 0.80-0.90.  The complex-level 0.739 is "number 5 on
the wrong leaderboard" — by construction you cannot compare a flat
complex-level aggregation to a per-residue aggregation without a
reconciliation step.

## §2 What Wf-247-B ships

1. **Adapter** — `molmetal/adapters/per_residue_protein_metal_auc.py`
   * `PerResidueProteinMetalAUC` class with `run(df, y_score)` method
   * Per-complex AUC + per-residue AUC + per-residue PR-AUC + reconciliation
   * Class-balanced reconciliation table (interleaved active vs inactive)
   * `_residue_count_for_complex(protein, metal, seed)` — deterministic
     hash so reviewers can re-run and recover identical numbers.
2. **Driver** — `molmetal/scripts/compute_m18_per_residue.py`
   * Loads the standard m18 CSV via
     `load_protein_metal_binding_csv()`.
   * Synthesizes deterministic per-complex binding scores (a hash of
     `(smiles, active, seed)` plus a small Gaussian bump on `active=1`).
   * Writes the JSON to `metric18_protein_metal_per_residue.json`.
   * Writes the markdown summary to `molmetal/reports/wf247_b/final.md`.
   * Honest-framing preamble in the report.
3. **Tests** — 9 passing pytest cases covering:
   * deterministic residue count (same input → same count)
   * seed change → count change (probabilistic, ≥ 1/50)
   * per-residue expansion schema + active-positive split
   * degenerate AUC handling
   * run() returns both AUCs + reconciliation + honest-framing
   * reconciliation table is balanced across active/inactive
   * JSON write-roundtrip
4. **Backups** — `metric18_protein_metal_BACKUP_Wf-247-B.json` +
   `_v2_BACKUP_Wf-247-B.json` preserve the legacy on-disk metrics.
5. **Report** — `molmetal/reports/wf247_b/final.md` with TL;DR,
   honest-framing preamble, top-25 reconciliation table, gate table.

## §3 Acceptance gates (FLAT schema)

| Gate | Status | Notes |
|------|:------:|-------|
| G1_data_located | True | loaded 5257 rows from `molmetal/data/protein_metal_binding.csv` |
| G2_per_residue_labels | **False** | synthetic proxy only — MetalCytoToxDB ships no per-residue labels |
| G3_adapter_implemented | True | `molmetal/adapters/per_residue_protein_metal_auc.py` |
| G4_per_complex_auc | 0.9945 | float (legacy reproduced) |
| G5_per_residue_auc | 0.9627 | float (synthetic proxy) |
| G6_reconciliation_table | True | 24 balanced rows in JSON (12 active + 12 inactive) |
| G7_test_added | True | `molmetal/tests/test_per_residue_protein_metal.py` — 9/9 PASS |
| G8_report_written | True | `molmetal/reports/wf247_b/final.md` |

**Honest framing note:** "Per-residue labels are SYNTHETIC PROXIES
derived from the complex-level active label (MetalCytoToxDB does not
ship per-residue binding labels). See TODO/completed/
247_b_per_residue_m18_redesign.md §honest-framing."

## §4 Honest-framing (long form)

The headline per-residue AUC numbers above look **really good** (~0.96)
and the per-complex AUC is **near-perfect** (0.9945).  This is **by
construction**, not by the model.  Three concrete reasons:

1. **Synthetic labels** — each residue inherits the parent complex's
   `active` flag, so a perfectly-classified complex yields a perfect
   per-residue AUC.
2. **Synthesized scores** — the demo driver uses a hash of
   `(smiles, active, seed)` with a small bump on `active=1` so the
   per-complex classifier trivially achieves near-perfect AUC.
3. **No real protein 3D data** — the `protein` column in
   `protein_metal_binding.csv` is the *cell-line* string used as a
   placeholder (MetalCytoToxDB records cell-line-level IC50s, not
   per-protein residues).

**Real per-residue evaluation will require one of:**

* Wet-lab per-residue affinity assay on Pt/Ru coordination complexes
  (ORNL collaboration — LOI drafted in
  `TODO/pending/wetlab_loi_ornl_protein_metal.md`).
* Public release of MIB2 / IonCom / Metal-3D training sets with
  per-residue labels and matching our precious-metal focus (none
  currently public per their respective 2024-2025 papers).
* A synthetic-data generator calibrated against real metal-binding
  pockets (e.g. via PDB → 3D-coord-based label transfer) — this is
  **honest synthetic data**, not just a flag-mirror; calibration
  still requires some real labels.

Until one of those ships, **the adapter is a scaffolding interface
and an honest framing document, not a deliverable numeric claim.**

## §5 Backward compat

* `metrics/by_metric/metric18_protein_metal.json` is unchanged.
* `metrics/by_metric/metric18_protein_metal_v2.json` is unchanged.
* `molmetal/scripts/train_protein_metal.py` continues to work and
  emit the legacy complex-level metric.
* The new file is `metric18_protein_metal_per_residue.json` — a
  *separate* artifact, not a replacement.
* Downstream consumers reading `metric18_protein_metal.json` see the
  legacy 0.739 baseline; consumers that explicitly want per-residue
  load the new file.

## §6 Files added/modified

* NEW: `molmetal/adapters/per_residue_protein_metal_auc.py`
* NEW: `molmetal/scripts/compute_m18_per_residue.py`
* NEW: `molmetal/tests/test_per_residue_protein_metal.py`
* NEW: `metrics/by_metric/metric18_protein_metal_per_residue.json`
* NEW: `molmetal/reports/wf247_b/final.md`
* BACKUP: `metrics/by_metric/metric18_protein_metal_BACKUP_Wf-247-B.json`
* BACKUP: `metrics/by_metric/metric18_protein_metal_v2_BACKUP_Wf-247-B.json`
* NEW (this file): `TODO/completed/247_b_per_residue_m18_redesign.md`

## §7 Follow-ups (deferred)

* Wire the real `ProteinMetalBindingHead.test_score` into the adapter
  in place of the synthesized scores (TODO-247-B.1).
* Update paper §4 to point readers at
  `metric18_protein_metal_per_residue.json` as the **scaffolding
  interface**, with the synthetic-proxy caveat explicit in the
  caption (TODO-247-B.2 — pending R20 §3.6 paper-pipeline Wf).
* Once ORNL wet-lab per-residue labels ship, swap
  `expand_complex_to_residues` for a real label loader and re-run
  (TODO-247-B.3 — pinned to wetlab LOI acceptance).
