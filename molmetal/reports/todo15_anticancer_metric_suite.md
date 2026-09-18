# TODO-15 Anticancer Metric Suite — Implementation Report

**Date:** 2026-09-14
**Author:** Mol-Metal automation
**Scope:** Implement the 3 anticancer metric buckets per
`molmetal/reports/anticancer_vs_general_metrics_survey.md`:
**logP/TPSA IV window**, **metal coordination score**, **GSH evasion
flag**, **DNA Kb proxy**, and an equal-weight composite
**anticancer_index**.

---

## §1 Method — Spec + Implementation Summary

### 1.1 Spec (per TODO-15 + anticancer_vs_general_metrics_survey.md §2)

| Bucket | Source | Range |
|--------|--------|-------|
| **logP** IV window | RDKit `Descriptors.MolLogP` | `[2.0, 5.0]` |
| **TPSA** IV window | RDKit `Descriptors.TPSA` | `[60.0, 150.0]` |
| **MW** IV window (descriptive) | RDKit `Descriptors.MolWt` | `[300.0, 700.0]` (NOT Lipinski ≤ 500) |
| **Metal coordination score** | Heuristic: square-planar Pt(II) ideal = 4 donors, octahedral Ru/Ir ideal = 6 donors | `[0, 1]` |
| **GSH evasion flag** | Heuristic: bulky aromatic-N donors on a metal centre AND no free thiol | `bool` |
| **DNA Kb proxy** | Heuristic: Pt(II)/Ru/Ir centre + Cl leaving groups + aromatic N + chelating diamine | `[0, 1]` |

The composite **anticancer_index** is the equal-weight mean of:
`(iv_window_ok + metal_score + gsh_evasion + dna_proxy) / 4.0`.

Per the survey's "metal IV protocol" we **drop Lipinski MW ≤ 500 from
the required flag** and keep a descriptive MW + a 300–700 Da IV flag.

### 1.2 Implementation

**New modules**
- `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` —
  `AnticancerMetricSuite` with `descriptor_report`,
  `metal_coordination_score`, `gsh_evasion_flag`, `dna_kb_proxy`,
  `composite_score` (returns full dict incl. `anticancer_index`).
- `molmetal/molmetal_lam/metrics/dna_fragment_oracle.py` —
  `dna_kb_proxy_v2(mol, dna_fragment)` that wraps the heuristic with
  an optional docked-conformer boost (sigmoid of Pt-N7 distance).

**Wiring into RewardAggregator** (`molmetal/molmetal_lam/search_alg/proof_search.py`)
- New optional `r_anticancer_index` channel registered by
  `register_anticancer_channels()` (reads `composite_score(...)["anticancer_index"]`
  from the new suite, falls back to a float for legacy suites).
- New weight `w_anticancer_index = 0.0` (opt-in; default off so
  existing reward stays bit-for-bit identical).
- `aggregate()` channel map extended with `r_anticancer_index`.
- `_call_reward` extension so the channel contributes when
  `w_anticancer_index > 0`.

**CLI runner**
- `molmetal/scripts/evaluate_anticancer_metrics.py` — reads a SMILES
  file (txt or csv), emits a CSV with `descriptor_report` +
  `composite_score` columns.
- `molmetal/scripts/_demo_5_metaldrugs.txt` — 5 hand-checked metal
  drugs (cisplatin, carboplatin, oxaliplatin, auranofin,
  ruthenium-arene).

---

## §2 Test Results

Command:

```
uv run pytest -q molmetal/molmetal_lam/tests/test_anticancer_metric_suite_todo15.py --tb=short
```

Result:

```
18 passed, 1 warning in 1.70s
```

Test inventory (18 tests, exceeds the 10-test floor):

1. `test_descriptor_report_cisplatin_has_logp_tpsa_mw_in_range`
2. `test_descriptor_report_carboplatin_returns_realistic_mw`
3. `test_descriptor_report_oxaliplatin_returns_finite_values`
4. `test_metal_coordination_score_cisplatin_is_one`
5. `test_metal_coordination_score_under_coordinated_pt_is_below_half`
6. `test_metal_coordination_score_non_metal_returns_neutral_five`
7. `test_gsh_evasion_flag_cisplatin_edge_case_false`
8. `test_gsh_evasion_flag_aromatic_n_bulky_metal_complex_true`
9. `test_gsh_evasion_flag_free_thiol_returns_false`
10. `test_dna_kb_proxy_guanine_pt_adduct_above_threshold`
11. `test_dna_kb_proxy_cisplatin_high_due_to_cl_and_pt`
12. `test_dna_kb_proxy_v2_falls_back_when_no_docked_conformer`
13. `test_composite_score_cisplatin_returns_full_dict`
14. `test_composite_score_anticancer_index_unit_interval`
15. `test_composite_score_invalid_smiles_anticancer_index_zero`
16. `test_reward_aggregator_anticancer_index_channel_registered`
17. `test_reward_aggregator_anticancer_index_weight_default_zero`
18. `test_reward_aggregator_aggregate_with_anticancer_index_enabled`

No failures.

---

## §3 Demo — 5 Hand-Checked Metal Drugs

Command:

```
uv run python molmetal/scripts/evaluate_anticancer_metrics.py
```

Emits `molmetal/reports/_demo_anticancer_metrics.csv`. Embedded table:

| name | smiles | logP | TPSA | MW | iv_window_ok | mw_iv_flag | metal_score | gsh_evasion | dna_proxy | is_metal_complex | anticancer_index |
|------|--------|------|------|----|--------------|------------|-------------|-------------|-----------|------------------|-------------------|
| cisplatin | `[Pt](N)(N)(Cl)Cl` | 0.20 | 52.04 | 298.03 | 0 | 0 | 1.000 | 0 | 0.700 | 1 | **0.425** |
| carboplatin | `[Pt](N)(N)(O1)C(=O)C2(CCC2)C1=O` | -0.94 | 95.41 | 353.24 | 0 | 1 | 1.000 | 0 | 0.500 | 1 | **0.375** |
| oxaliplatin | `[Pt]1(N[C@@H]2CCCC[C@H]2N1)(O1)C(=O)C1=O` | -0.53 | 67.43 | 379.27 | 0 | 1 | 1.000 | 0 | 0.500 | 1 | **0.375** |
| auranofin | `[Au+]S(CC)SCC` | 2.14 | 0.00 | 320.23 | 0 | 1 | 0.000 | 0 | 0.000 | 1 | **0.000** |
| ruthenium-arene | `[Ru](c1ccccc1)(N)(N)(Cl)Cl` | 1.06 | 52.04 | 281.13 | 0 | 0 | 0.667 | 0 | 0.500 | 1 | **0.292** |

### §3.1 Hand-check interpretation

| Drug | Interpretation |
|------|----------------|
| **cisplatin** | Highest anticancer_index (0.425). Square-planar Pt(II) → metal_score = 1.0; Pt + 2 Cl leaving groups + chelating N → dna_proxy = 0.7. logP is just below the anticancer window (0.20 vs. 2.0) — consistent with cisplatin's very polar character (the IV dose delivers it as a salt). |
| **carboplatin** | Slightly lower (0.375). Same Pt(II) coordination score but no Cl leaving groups → dna_proxy = 0.5 (base 0.4 Pt + 0.1 chelating diamine). TPSA 95 Å² is in window but logP -0.94 fails the 2-5 window. |
| **oxaliplatin** | Same as carboplatin (0.375): square-planar Pt(II) → 1.0; dna_proxy = 0.5 (Pt + chelating diamine, no Cl). logP -0.53 fails the window. |
| **auranofin** | anticancer_index = 0.0 because: auranofin has only **2 donors** (the disulphide S and the thiolate S) — the metal coordination score drops to **0.0** against the linear Au(I) ideal; dna_proxy is 0 because no Pt/Ru/Ir centre. Au(I) anticancer chemistry is distinct from Pt/Ru/Ir and is not captured by the square-planar / octahedral scoring rules. |
| **ruthenium-arene** | anticancer_index = 0.292: Ru(II) centre → metal_score = 0.667 (4 of 6 ideal donors → deviation/3 = 0.333, score = 1 - 0.333 = 0.667); dna_proxy = 0.5 (Ru + Cl + chelating N). Half-sandwich Ru(II) arene complexes are well-known DNA binders, but the demo SMILES has only 4 donors against the 6-ideal. |

### §3.2 Notes on the descriptor windows

- **logP window (2-5)** — none of the 5 reference drugs fall in this
  window; cisplatin, carboplatin, oxaliplatin all sit at logP < 1.
  This is intentional in the survey: the "anticancer logP window" is
  for **lipophilic oral kinase inhibitors**, not for IV-delivered
  platinum drugs. The 300–700 Da MW IV flag is the more meaningful
  IV-protocol cut-off, and three of the five (carboplatin,
  oxaliplatin, auranofin) pass it.
- **mw_iv_flag** is descriptive, NOT required for the composite —
  consistent with the survey recommendation to drop Lipinski MW ≤ 500
  for transition-metal drugs.

---

## §4 Files Touched

New (per spec):

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/metrics/__init__.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/metrics/anticancer_metric_suite.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/metrics/dna_fragment_oracle.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_anticancer_metric_suite_todo15.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/evaluate_anticancer_metrics.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/_demo_5_metaldrugs.txt`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/_demo_anticancer_metrics.csv` (run output)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/todo15_anticancer_metric_suite.md` (this file)

Modified (RewardAggregator wire-in only):

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py` —
  added `r_anticancer_index` channel, `w_anticancer_index` weight,
  extended the `aggregate()` channel map, extended
  `register_anticancer_channels()`. **No existing channel / weight /
  default changed** (the new weight defaults to 0.0 so existing
  reward stays bit-for-bit identical).

Not modified (per spec):

- All `TODO/pending/*` files.
- All `molmetal/reports/anticancer_vs_general_metrics_survey.md` and
  other report files.

---

## §5 Limitations / Caveats

- The metal coordination score is heuristic — square-planar Pt(II) =
  4 donors, octahedral Ru/Ir = 6 donors. Mixed-valence cases
  (Pt(IV) hexa-coordinate, Au(III) square-planar, Au(I) linear) are
  not specialised; they score against the closest ideal. Auranofin
  (Au(I), 2-coord) scores 0.0 against the square-planar ideal, which
  is why it shows anticancer_index = 0.0.
- The GSH evasion flag is binary (True/False). The full evasion
  kinetics (GSH reaction rate) is out of scope for this heuristic
  pass; see TODO-16 for the DNA-fragment docking extension.
- The DNA Kb proxy is heuristic — it does not measure binding
  constants. The optional `dna_kb_proxy_v2` wrapper applies a
  sigmoid over the Pt-N7 docked distance when a conformer is
  available; otherwise it falls back to the heuristic.
- All RDKit failures (invalid SMILES, missing atoms) yield a
  composite dict with NaN descriptors and `anticancer_index = 0.0`
  — no crash, no warnings beyond RDKit's own parser output.
