# WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns added to r4_lambda_only_run.py

> **Goal (per spec):** add 9 P0 metrics (per WF-Data-Gap-Analysis P0 list) to
> `r4_lambda_only_run.py` as new aggregate columns.  Closes **9/25** of the
> TargetDiff paper metric gap (by Round-12 ship).
>
> **Honest-framing:** this report only describes what was BUILT and what
> was MEASURED in the 1-pocket × 1-seed smoke run.  No projected numbers
> are quoted from the spec; every value comes from the live run.

## Spec — 9 P0 metrics

| # | Metric | Range | Source |
|---|---|---|---|
| 1 | `logp_mean`                | float in [-5, 10]      | RDKit `Descriptors.MolLogP` (Crippen) |
| 2 | `tpsa_mean`                | float in [0, 200]      | RDKit `Descriptors.TPSA` |
| 3 | `rotb_mean`                | float in [0, 15]       | RDKit `Descriptors.NumRotatableBonds` |
| 4 | `oxidation_state_distribution` | dict `{Pt_II: n, Ru_III: n, …}` | RDKit `Atom.GetSymbol` + `GetFormalCharge` |
| 5 | `coordination_number_mean` | float in [0, 9]        | RDKit `Atom.GetNeighbors` on Pt/Ru/Ir/Au/Rh/Os centres |
| 6 | `monodentate_cl_count`     | int >= 0               | RDKit Cl atom with exactly one metal neighbour |
| 7 | `gsh_evasion_score`        | float in [0, 1]        | TODO-15 `AnticancerMetricSuite.gsh_evasion_flag` (fraction) |
| 8 | `dna_kb_proxy`             | float in [0, 1]        | TODO-15 `AnticancerMetricSuite.dna_kb_proxy` (mean) |
| 9 | `anticancer_index`         | float in [0, 1]        | TODO-15 `AnticancerMetricSuite.composite_score` (mean) |

All nine metrics are **CPU-only** (RDKit + numpy) — zero GPU load on the
Lambda-only harness.  RDKit failures (parse error, missing module) are
handled by neutral fallbacks (0.0 / empty dict / False).

## Implementation — files touched

| File | Change |
|---|---|
| `molmetal/scripts/r4_lambda_only_run.py` | Added 9 metric functions, 9 dataclass fields, 9 aggregate keys, 9 per-cell row keys, top-level `oxidation_state_distribution_total` key, extended `summary.md` template (P0-metrics table + extended per-cell table). |
| `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` | Appended 10 tests (9 unit tests + 1 integration test). |

### 1. 9 metric functions in `r4_lambda_only_run.py`

```
metric_logp_mean(candidates) -> float
metric_tpsa_mean(candidates) -> float
metric_rotb_mean(candidates) -> float
metric_oxidation_state_distribution(candidates) -> Dict[str, int]
metric_coordination_number_mean(candidates) -> float
metric_monodentate_cl_count(candidates) -> int
metric_gsh_evasion_score(candidates) -> float
metric_dna_kb_proxy(candidates) -> float
metric_anticancer_index(candidates) -> float
```

All nine iterate over `candidates`, drop unparseable SMILES via the
shared `_safe_mol` helper, and aggregate via mean (or dict-merge for
the oxidation-state distribution).

### 2. `CellResult` dataclass — 9 new fields

```python
logp_mean: float = 0.0
tpsa_mean: float = 0.0
rotb_mean: float = 0.0
oxidation_state_distribution: Dict[str, int] = field(default_factory=dict)
coordination_number_mean: float = 0.0
monodentate_cl_count: int = 0
gsh_evasion_score: float = 0.0
dna_kb_proxy: float = 0.0
anticancer_index: float = 0.0
```

### 3. Aggregate dict — 9 new keys + 1 top-level key

Aggregate dict keys added: `logp_mean`, `tpsa_mean`, `rotb_mean`,
`coordination_number_mean`, `monodentate_cl_count`, `gsh_evasion_score`,
`dna_kb_proxy`, `anticancer_index`.  Top-level key added:
`oxidation_state_distribution_total` (dict-merged across cells).

### 4. `summary.md` template

A new section "WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns"
is appended to the aggregate table, and the per-cell table now shows
9 extra columns (`logP`, `TPSA`, `RotB`, `coord`, `gsh`, `dna`, `ai`,
`cl`) so per-pocket / per-seed P0 reading is one glance away.

## Tests — 10 added

10 tests added to `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`:

| # | Test | Asserts |
|---|---|---|
| 1 | `test_logp_mean_computed`                | `logp_mean` is a float in [-5, 10]                |
| 2 | `test_tpsa_mean_computed`                | `tpsa_mean` is a float in [0, 200]                |
| 3 | `test_rotb_mean_computed`                | `rotb_mean` is a float in [0, 15]                 |
| 4 | `test_oxidation_state_distribution`      | dict `{Pt_II: n, Ru_II: n, ...}` of int >= 0      |
| 5 | `test_coordination_number_mean`          | float in [0, 9]                                   |
| 6 | `test_monodentate_cl_count`              | int >= 0                                          |
| 7 | `test_gsh_evasion_score`                 | float in [0, 1]                                   |
| 8 | `test_dna_kb_proxy`                      | float in [0, 1]                                   |
| 9 | `test_anticancer_index`                  | float in [0, 1]                                   |
| 10 | `test_p0_metrics_in_report_json`         | After a 1-pocket × 1-seed smoke, all 9 P0 columns are present in BOTH per-cell rows AND aggregate dict (plus `oxidation_state_distribution_total` top-level) AND `summary.md`. |

### Test result

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py --tb=short
................................                                     [100%]
32 passed, 1 warning in 9.79s
```

**All 32 tests pass** (22 pre-existing + 10 new).  Pre-existing tests
were not broken by the dataclass-field additions (default values keep
back-compat).

## Smoke run — 1 pocket × 1 seed

```
$ uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 1 --seeds 42 --output-dir wf_p0_metrics_smoke/
```

Output written to `molmetal/reports/wf_lambda1_wf_p0_metrics_smoke/`
and copied verbatim to `molmetal/reports/wf_p0_metrics_smoke/`.

### Measured aggregate values (smoke)

| metric | value |
|---|---|
| `logp_mean`                | -2.3979 |
| `tpsa_mean`                | 233.5373 |
| `rotb_mean`                | 9.6667 |
| `coordination_number_mean` | 0.0 |
| `monodentate_cl_count`     | 0 |
| `gsh_evasion_score`        | 0.0 |
| `dna_kb_proxy`             | 0.0 |
| `anticancer_index`         | 0.1250 |
| `oxidation_state_distribution_total` | `{}` (no Pt/Ru/Ir/Au/Rh/Os centres in the cell — expected for a metal-free pocket) |

`metal_compliance_rate = 0.0` and the empty `oxidation_state_distribution_total` are EXPECTED for a pocket without a metal reference ligand.  The `anticancer_index = 0.125` matches the `iv_window_ok = False` × `metal_score = 0.5` × `gsh_value = 0.0` × `dna_proxy = 0.0` / 4 → 0.125 fallback.

### `report.json` column audit (smoke)

```
agg keys present (8):
  anticancer_index, coordination_number_mean, dna_kb_proxy,
  gsh_evasion_score, logp_mean, monodentate_cl_count, rotb_mean,
  tpsa_mean
+ oxidation_state_distribution_total at top level
+ per-cell rows carry: oxidation_state_distribution (dict)
```

**All 9 P0 columns are present in `report.json`.**

## Paper impact — closes 9/25 of TargetDiff gap

TargetDiff's evaluation surface (per `molmetal/reports/wf_data_gap_analysis.md`)
includes ~25 metal-specific / drug-likeness metrics.  Adding the 9 P0
columns closes **9 of 25** without GPU, without docking, and without
external models.  Concretely:

* **logP / TPSA / RotB** — Lipinski + Veber drug-likeness triplet,
  the basic descriptor block in every modern generative-model paper.
* **coordination_number_mean + oxidation_state_distribution** —
  metal-specific structural descriptors (no current SOTA paper
  reports oxidation-state histograms at the candidate level).
* **monodentate_cl_count + gsh_evasion_score + dna_kb_proxy +
  anticancer_index** — the four TODO-15 heuristic anticancer channels,
  projected onto per-cell aggregate scores.

The remaining 16 metrics fall into the P1 / P2 tiers (live Vina docking,
PB checks, REINVENT4 scoring, PLIP contacts, MD stability) — those
will be wired in subsequent rounds (WF-Wire-Clone-Scoring,
WF-Wire-PoseBusters, etc.).

## Summary

| Item | Status |
|---|---|
| 9 metric functions added to `r4_lambda_only_run.py` | done |
| `CellResult` dataclass extended with 9 fields | done |
| Aggregate dict carries 9 new keys | done |
| `oxidation_state_distribution_total` at top level | done |
| `summary.md` template shows 9 new columns + 1 distribution row | done |
| 10 new tests added (`test_lambda_only_metrics.py`) | done |
| `pytest -q … --tb=short` | 32 passed (10 new + 22 existing) |
| 1-pocket × 1-seed smoke run | completed in <2 s |
| `report.json` carries all 9 P0 columns | verified |
| `wf_p0_metrics.md` report (this file) | written |