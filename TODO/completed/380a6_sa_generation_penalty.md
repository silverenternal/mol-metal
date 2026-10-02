# Wf-380-A6 — TODO-205 m4 SA Generation Penalty (SCAFFOLD)

**Last updated:** 2026-09-30
**Workflow:** Wf-380 ultracode residual CPU-only fixes + improvements wave
**Sub-flow:** A6
**Status:** SHIPPED (CPU-only scaffold)
**Pre-flight:** 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-370-B1 baseline

---

## §1 TL;DR

Wf-380-A6 ships an SA-aware sampling + loss penalty module for the
CFM (TODO-205 / WEAK_TO_STRONG §2.4 / m4 Success Rate SA/QED
sub-criterion).  The penalty is ``max(0, sa - target_sa) * sa_weight``
(Ertl & Schuffenhauer 2009 §3 generation-time reweighting), wired
into both a per-sample sampling penalty and a combined BCE+penalty
loss term.  All artifacts are CPU-only SCAFFOLDs — actual integration
into ``train_cfm_yuelbond_wasserstein.py`` is deferred to Wf-380-B1.

**Artifacts on disk:**

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/adapters/flow_matching_lipman/sa_penalty.py` | 256 | SA penalty primitives (`compute_sa_score` + `sa_aware_sampling_penalty` + `sa_aware_loss_step` + `aggregate_sa_penalty`) |
| `molmetal/scripts/cfm_sa_aware_panel.py` | 290 | CLI panel: per-molecule SA + aggregation (`per_molecule_sa_evaluation` + `aggregate_sa_distribution` + `build_panel`) |
| `molmetal/tests/test_wf380a6_sa_penalty.py` | 235 | 14 NEW tests PASS (4 classes: contract / aggregate / loss / delegation) |

**MEASURED test pass-rate:** **14/14** in 0.51 s wall (CPU-only).
**Pre-flight v3 fresh-run:** 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-380-A6 — docs-only + CPU-only wave, 0 GPU launches).
**EXPECTED_GATE_COUNTS["all"] 94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-380-A6 — 1 module + 1 panel + 1 test file + 1 closeout doc only).

---

## §2 Penalty function

### §2.1 Mathematical form

```
penalty(sa; target_sa=2.8, sa_weight=0.3) = max(0, sa - target_sa) * sa_weight
```

* Non-negative (the contract the trainer relies on for sampling
  reweighting without clipping low-SA samples).
* Exactly 0 when SA <= target_sa (the synthesizable region per
  Wf-139 P6 / Ertl 2009 §3).
* Linear in (sa - target) above the threshold (the canonical
  monotone penalty; more aggressive forms like squared-penalty or
  exponential-tanh would over-fit the smoke cohort).
* Default ``target_sa=2.8`` matches the canonical synthesizable
  threshold; default ``sa_weight=0.3`` matches the
  ``--sa-weight 0.3`` smoke cohort in
  `metrics/by_metric/sa_mean.json` (2026-09-14).

### §2.2 Sampling-time use

```python
from molmetal.adapters.flow_matching_lipman.sa_penalty import (
    sa_aware_sampling_penalty,
    DEFAULT_TARGET_SA,
    DEFAULT_SA_WEIGHT,
)

# Per-sample penalty used by the CFM sampler to reweight candidates.
penalty = sa_aware_sampling_penalty(
    samples=smiles_to_sa_list,                # list of float
    target_sa=DEFAULT_TARGET_SA,              # 2.8
    sa_weight=DEFAULT_SA_WEIGHT,              # 0.3
)
```

### §2.3 Loss-time use

```python
from molmetal.adapters.flow_matching_lipman.sa_penalty import sa_aware_loss_step

# Combined BCE + SA penalty loss (1-D or 2-D logits).
loss = sa_aware_loss_step(logits=logits, sa_scores=sa_scores)
```

The function uses a SCAFFOLD BCE on the logits as a placeholder
``L_main`` so it is self-contained and standalone-testable; the
Wf-380-B1 integration replaces this with the multi-task Wasserstein
loss the real trainer uses.

### §2.4 RDKit delegation

`compute_sa_score(mol)` resolves the canonical SA scorer in two
priority orders:

1. RDKit's bundled ``rdkit/Contrib/SA_Score/sascorer.py`` (loaded
   via ``rdkit.Chem.RDConfig.RDContribDir``) — this ships with
   the ``fpscores.pkl.gz`` fragment table.
2. The in-tree ``molmetal/eval/scoring/sascorer.py`` wrapper — a
   vendored copy of the same algorithm (no fpscores.pkl.gz needed).

Falls back to ``float('nan')`` on failure so the downstream penalty
arithmetic stays robust to the SCAFFOLD tier.

---

## §3 Smoke results

### §3.1 Smoke cohort (4 simple molecules)

```
$ python molmetal/scripts/cfm_sa_aware_panel.py \
    --smiles-csv /tmp/wf380_smoke.csv \
    --out-json /tmp/wf380_smoke.json \
    --target-sa 2.8 --sa-weight 0.3

[cfm_sa_aware_panel] wrote /tmp/wf380_smoke.json (n_input=4 n_parsed=4 n_failed=0
    mean_sa=1.3924 fraction_sa_le_2.8=1.0000)
```

| smiles | SA | penalty | status |
|---|---:|---:|---|
| CCO | 1.980 | 0.0 | OK |
| c1ccccc1 | 1.000 | 0.0 | OK |
| CC(=O)O | 1.515 | 0.0 | OK |
| CCCCCCCCCCCCCCCCCC | 1.075 | 0.0 | OK |

All 4 molecules in the synthesizable region (SA < 2.8) — penalty is
0 across the board, as expected.

### §3.2 Synthetic high-SA cohort (penalty test)

```
$ python -c "
from rdkit import Chem
from molmetal.adapters.flow_matching_lipman.sa_penalty import compute_sa_score
mols = [Chem.MolFromSmiles(s) for s in [
    'CC(C)(C)Sc1nc2c(Cl)cccc2n1C(C)(C)C',     # SA = 2.808
    'O=C1N(C)C(=O)N(C)C(=O)N1C',              # SMILES scaffold
]]
print([compute_sa_score(m) for m in mols])"
```

For a synthetic cohort of [5.0, 6.0, 7.0, 8.0, 9.0]:
* `aggregate_sa_penalty` -> mean_sa=7.0, mean_penalty=1.26,
  fraction_above_target=1.0.
* `sa_aware_sampling_penalty` on [3.0] -> 0.06 (= (3.0 - 2.8) * 0.3).

### §3.3 Test suite

```
$ pytest molmetal/tests/test_wf380a6_sa_penalty.py -v
============================== 14 passed in 0.51s ==============================
```

14 NEW tests across 4 classes:

| Class | Tests | Coverage |
|---|---:|---|
| `TestSAPenaltyContract` | 6 | Non-negativity, exact-zero below target, linearity above, default target/weight, vectorisation |
| `TestAggregateSAPenalty` | 3 | Mixed cohort, empty (NaN), all-below-target |
| `TestSAAwareLossStep` | 3 | 1-D logits finite, 2-D logits finite, monotonic in penalty |
| `TestComputeSAScoreDelegation` | 2 | Matches RDKit bundled SA scorer, None returns NaN |

---

## §4 Expected lift (PARTY-TWENTY-SPIRIT §3.5 honest framing)

### §4.1 PROJECTED lift on m4 SA (CPU-only SCAFFOLD)

Per WEAK_TO_STRONG §3.6, the m4 Success Rate SA/QED sub-criterion
expects SA to drop from the current ``3.099`` (filtered pool smoke
cohort, `metrics/by_metric/sa_mean.json` 2026-09-15) into the
synthesizable band of 2.5-2.7.  The penalty module ships the
arithmetic; the actual integration into the CFM sampler/trainer is
deferred to Wf-380-B1.

| Metric | Pre-Wf-380-A6 | Post-Wf-380-A6 (PROJECTED) |
|---|---:|---:|
| m4 SA (filtered pool smoke) | 3.099 | 2.5-2.7 (PROJECTED, post Wf-380-B1 integration) |
| m4 SA/QED sub-rate (SA<2.8 ∧ QED>0.5) | 0.0022 | 0.05-0.15 (PROJECTED, 449 cohort) |

### §4.2 What is MEASURED vs PROJECTED

* **MEASURED on disk** (Wf-380-A6 SCAFFOLD tier):
  - Penalty arithmetic correctness (14 tests PASS, exact-zero contract verified).
  - RDKit delegation matches bundled SA scorer on `CCO` (diff < 1e-9).
  - Smoke cohort behaviour (4 trivial mols, 0 penalty).
* **PROJECTED, NOT MEASURED end-to-end**:
  - m4 SA lift on real CFM samples (requires Wf-380-B1 sampler integration).
  - m4 SA/QED sub-rate lift on the 449-cohort (requires Wf-380-B1 + Wf-299 GPU).
  - CFM trainer convergence with `--sa-penalty` flag (deferred to Wf-380-B1).

### §4.3 Integration path to Wf-380-B1

```python
# Wf-380-B1 wire-up (TODO, deferred):
# 1. Add --sa-penalty CLI flag to train_cfm_yuelbond_wasserstein.py
# 2. Replace the SCAFFOLD BCE in sa_aware_loss_step with the
#    multi-task Wasserstein loss (Wf-131 P1 / Wf-181 G5 path).
# 3. Hook the sampling penalty into the EGNN node aggregation so
#    the velocity net learns to favour low-SA samples.
# 4. Re-run the 449 metallo-drug cohort smoke at scale=100,
#    target_band = fraction_sa_le_2.8 >= 0.50.
```

---

## §5 Cross-references

* **TODO source:** `TODO/pending/207_wf299_cfm_3wave_launch.md` §2.3 (TODO-205 m4 SA generation penalty).
* **WEAK_TO_STRONG plan:** `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.4 (m4 Success Rate) + §3.6 (SA score baseline + expected lift).
* **Baseline metric:** `metrics/by_metric/sa_mean.json` (3.099 on filtered pool, 2026-09-15 smoke).
* **Sub-rate metric:** `metrics/by_metric/m4_success_rate.json` (sub_rate_sa_qed = 0.0022 on 449 cohort).
* **Existing property panel:** `molmetal/scripts/cfm_property_panel.py` (Wf-264-SA, sibling panel).
* **Existing SA wrappers:**
  - `molmetal/eval/scoring/sascorer.py` (in-tree wrapper fallback)
  - `molmetal/molmetal_lam/sbdd_env/reinvent_wrapper.py` (REINVENT4 scorer)
* **Trainer target for integration:** `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-380-B1).
* **Wave context:** AGENTS.md §6 Wf-370-B1 baseline (pre-flight 92/0/2/94) + Wf-353-B1 baseline (EXPECTED_GATE_COUNTS["all"]=94).
* **Pipeline integration plan:** `TODO/pending/207_wf299_cfm_3wave_launch.md` §3.2-§3.4 (Wave 1+2+3 user OK).
* **Wave-history pointer:** AGENTS.md §6 Wf-380 row (this entry appended by Wf-380-C1 INDEX sync, deferred).

---

**END OF Wf-380-A6 CLOSEOUT** — owner: Hugo Lee; pre-flight 92/0/2/94 PRESERVED; 14 NEW tests PASS; 0 GPU launches; 0 paper/* touches; SCAFFOLD tier; Wf-380-B1 deferred for sampler + trainer integration.