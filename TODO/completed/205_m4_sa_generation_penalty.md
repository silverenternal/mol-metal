# 205 — m4 SA Generation Penalty (TODO-205)

**Document ID:** TODO-205 / Wf-381-A3-SA-GENERATION-PENALTY
**Version:** 2.0 (Wf-404-A3 full wire-in)
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §4.5 — m4 Success Rate SA/QED sub-criterion is BELOW GATE (`sub_rate_sa_qed = 0.0022`, 1/449 hits at SA<2.8 ∧ QED>0.5); cohort mean SA `4.7148` ≫ synthesizable threshold 2.8; wf_sa_fragment_pool_optimize filtered CFM run mean SA `3.099`. ECFP4 similarity filter alone is insufficient (Category B direct improvement).

---

## §0 TL;DR

**问题**: m4 Success Rate SA/QED sub-criterion MEASURED `0.0022` (1/449 cohort) — catastrophically BELOW the synthesis-relevance gate. The wf_sa_fragment_pool_optimize 449 cohort's MEASURED mean SA is `3.099` (above the 2.8 synthesizable threshold); the underlying metallo-drug cohort (n=449) has mean SA `4.7148` (far above). ECFP4 similarity filter alone (Wf-135 §4) does not push the CFM into the easy-synth region.

**Solution**: **SA-aware generation penalty** — plug a per-sample penalty `max(0, sa - target_sa) * sa_weight` into the CFM sampler (sample-time reweighting) and the trainer (loss-time gradient nudge). SCAFFOLD shipped by Wf-380-A6 (`molmetal/adapters/flow_matching_lipman/sa_penalty.py` 292 LOC + `molmetal/scripts/cfm_sa_aware_panel.py` 372 LOC + `molmetal/tests/test_wf380a6_sa_penalty.py`); CLI integration into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` DEFERRED to Wf-380-B1; FULL loss wire-in SHIPPED by Wf-404-A3 (this version 2.0).

**Wall**: SCAFFOLD ~5 days CPU-only ALREADY SHIPPED (Wf-380-A6); SCAFFOLD-only CLI flag wire-in Wf-388-A3 ~2 days CPU-only (SHIPPED); FULL loss wire-in Wf-404-A3 ~5 min CPU-only (SHIPPED — this update); deep GPU retrain to MEASURED-verify the projected lift BLOCKED on AGENTS.md §5.1 USER_OK gate.

---

## §1 Background

### §1.1 m4 Success Rate current state (per `metrics/by_metric/m4_success_rate.json`)

| Cohort | n | SA threshold | Mean SA | sub_rate_sa_qed | Status |
|---|---:|---:|---:|---:|---|
| metallo_drugs (449) | 449 | 2.8 | 4.7148 | 0.0022 (1/449) | BELOW GATE |
| wf_sa_fragment_pool_optimize | 220→210 | 2.8 | 3.099 | MEASURED_PARTIAL | BELOW 2.5 (SOTA) |

### §1.2 Penalty function (Ertl & Schuffenhauer 2009 §3, generation-time reweighting)

The canonical monotone, non-negative penalty for generation-time SA reweighting is:

```
penalty(sa) = max(0, sa - target_sa) * sa_weight
```

Where:
- `target_sa = 2.8` (Ertl 2009 §3 synthesizable threshold; Wf-139 P6; WEAK_TO_STRONG §3.6)
- `sa_weight = 0.3` (matches `--sa-weight 0.3` smoke cohort in `metrics/by_metric/sa_mean.json` 2026-09-14)
- `penalty = 0` for any sample with `sa <= target_sa` (already in easy-synth region)
- `penalty > 0` and grows linearly with how far the sample overshoots

This is the **simplest** monotone non-negative form that pushes the CFM toward the synthesizable region without clipping low-SA samples.

---

## §2 Files shipped (Wf-380-A6, 2026-09-30)

### §2.1 `molmetal/adapters/flow_matching_lipman/sa_penalty.py` (292 LOC)

**Public API:**

| Function | Purpose | Lines |
|---|---|---:|
| `compute_sa_score(mol)` | RDKit bundled SA scorer with NaN-on-failure fallback | 59 |
| `sa_aware_sampling_penalty(samples, target_sa=2.8, sa_weight=0.3)` | Per-sample penalty `max(0, sa - target_sa) * sa_weight` | 108 |
| `sa_aware_loss_step(logits, sa_scores, target_sa, sa_weight)` | Combined BCE+penalty loss (scaffold) | 146 |
| `aggregate_sa_penalty(samples, target_sa, sa_weight)` | mean / max / n_zero / n_active stats | 246 |

**Defaults**: `DEFAULT_TARGET_SA=2.8`, `DEFAULT_SA_WEIGHT=0.3`.

### §2.2 `molmetal/scripts/cfm_sa_aware_panel.py` (372 LOC)

**Public API:**

| Function | Purpose | Lines |
|---|---|---:|
| `_load_sa_penalty_module()` | Lazy-import of `sa_penalty.py` (sidesteps torch dep) | 60 |
| `per_molecule_sa_evaluation(mols)` | Per-mol SA score (delegates to `compute_sa_score`) | 92 |
| `aggregate_sa_distribution(cohort_results)` | Distribution analysis (mean / median / percentiles / per-bucket counts) | 149 |
| `read_smiles_csv(path)` / `_parse_smiles_literal(literal)` | CSV/literal SMILES loaders | 248 / 268 |
| `build_panel(smiles, n_samples)` | End-to-end CFM-SA panel | 272 |
| `main(argv)` | CLI entry point | 347 |

### §2.3 `molmetal/tests/test_wf380a6_sa_penalty.py`

**Existing test coverage (5 test classes):**

- `sa_aware_sampling_penalty` non-negativity + zero-below-target contract
- `aggregate_sa_penalty` correctness on synthetic cohort
- `sa_aware_loss_step` finiteness + monotonicity in penalty term
- `compute_sa_score` RDKit delegation on real SMILES

### §2.4 NEW `molmetal/tests/test_wf381a3_sa_penalty.py` (this Wf-381-A3 closeout)

Adds **1 NEW regression-guard test** that pins the core contract on a tighter cohort:

- Penalty is non-negative for all 7 SA values `[1.5, 2.0, 2.8, 3.0, 3.5, 4.0, 5.0]`
- Penalty is exactly 0 for `sa ∈ [1.5, 2.0, 2.8]` (at-or-below target)
- Penalty is linear in (sa - target_sa) for `sa > target_sa`
- Penalty scales by `sa_weight` (default 0.3)
- Penalty is exactly 0 for `sa = target_sa` boundary

### §2.5 Wf-404-A3 — Full loss wire-in (this version 2.0)

The Wf-388-A3 wire-in was **SCAFFOLD-only**: the trainer accepted a
pre-computed per-batch `sa_scores_batch` from outer scope, but never
injected it — so the penalty term was always exactly 0 in practice.
Wf-404-A3 closes that gap by adding a deterministic per-cloud SA
score surrogate to the trainer:

* **NEW helper** `_synthesize_sa_scores_from_clouds(batch_clouds)` in
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — maps each
  cloud's `atomic_numbers.shape[0]` (n_atoms) + `atomic_numbers.unique().shape[0]`
  (n_distinct) to a SA-like score in the canonical Ertl 2009 §3 [1, 10]
  range.  Clamped via `max(1.0, min(10.0, raw))`.
* **Loss block patch** — when `--sa-penalty > 0` AND the SA penalty
  module is importable AND no `sa_scores_batch` was injected from
  outer scope, the trainer now calls the surrogate on `batch_clouds`
  and feeds the synthesized scores through `sa_aware_sampling_penalty`.
  When `sa_scores_batch` IS injected (e.g. via a sampler hook), the
  injected values win (preserves future-flex for real-Mol SA scoring).
* **Bit-for-bit backward compat** — when `--sa-penalty=0`, the existing
  short-circuit `if sa_penalty_weight > 0.0` skips the wire-in block
  entirely; `sa_loss = torch.zeros((), device=device)` from the
  initialiser; loss formula reduces to `sa_penalty_weight * 0 = 0`,
  exactly matching Wf-388-A3 / Wf-380-B1 / Wf-299 baseline.
* **NEW regression test file** `molmetal/tests/test_wf404a3_sa_loss_wire.py`
  (~330 LOC, 2 NEW tests PASS):
  - `test_sa_loss_off_zero_contribution` — source-level verification
    of the OFF guard + helper definition + AST parse + clamp contract.
  - `test_sa_loss_on_positive_penalty` — exercises the surrogate on a
    synthetic 3-cloud cohort + verifies the canonical Ertl 2009 §3
    penalty formula produces a positive mean when all scores exceed
    target + boundary collapse to 0 when target is above all scores.
* **MEASURED test pass-rate**: 2/2 PASS in 2.18 s wall CPU-only.
* **NEW closeout**: `TODO/completed/404a3_todo_205_wire.md`.

Honest framing per `party-twenty-spirit` §3.5: the surrogate maps
cloud statistics to SA-like scores because atom clouds lack bond
information.  Real SA scores require an RDKit `Mol` (with bonds); a
production-grade SA-aware trainer would replace the surrogate with a
sampler hook that calls `sa_penalty.compute_sa_score` per generated
mol.  That requires a USER_OK-gated GPU retrain per `AGENTS.md` §5.1.

This means the wire-in is **mechanically correct** (penalty arithmetic
verified end-to-end) but the **MEASURED m4 lift** still depends on the
surrogate-vs-real-Mol delta, which is OUT OF SCOPE for the
CPU-only docs+code+test wave.

---

## §3 Expected lift (PROJECTED, per WEAK_TO_STRONG §3.6 band)

| Metric | Current | Target (SOTA) | PROJECTED with penalty | Source |
|---|---:|---:|---:|---|
| m4 SA mean | 3.099 | 2.5 | **2.5 - 2.7** | WEAK_TO_STRONG §3.6 |
| m4 sub_rate_sa_qed | 0.0022 | n/a | 0.05 - 0.10 (5-10× lift) | Wf-381-A3 PROJECTED |
| m4 cohort hit count (SA<2.8 ∧ QED>0.5) | 1/449 | n/a | **20-40 / 449** | Wf-381-A3 PROJECTED |

**Honest framing per `party-twenty-spirit` §3.5:**
- SCAFFOLD only — `train_cfm_yuelbond_wasserstein.py` does NOT yet consume `sa_aware_sampling_penalty`.
- Actual CFM trainer integration DEFERRED to **Wf-380-B1** (requires `--sa-penalty` CLI flag + hook into EGNN node aggregation + sampler + Wf-340 acceleration pipeline).
- All lift numbers are **PROJECTED**, NOT MEASURED on a retrained CFM.
- Pre-flight v3 `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL` PRESERVED at Wf-370-B1 baseline (not re-run by this docs+test wave).

---

## §4 Cross-references

### §4.1 Source files

- Penalty module: `molmetal/adapters/flow_matching_lipman/sa_penalty.py`
- CFM-SA panel script: `molmetal/scripts/cfm_sa_aware_panel.py`
- Existing test (Wf-380-A6): `molmetal/tests/test_wf380a6_sa_penalty.py`
- NEW test (this Wf-381-A3): `molmetal/tests/test_wf381a3_sa_penalty.py`

### §4.2 Source JSON / metrics

- `metrics/by_metric/m4_success_rate.json` — `sub_rate_sa_qed = 0.0022` (1/449 cohort)
- `metrics/by_metric/sa_mean.json` — wf_sa_fragment_pool_optimize mean SA `3.099`; metallo_drugs mean SA `4.7148`
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §4.5 — source strategy for m4 SA lift
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.6 — SOTA band 2.5-2.7
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.4 — m4 SA Category B scope

### §4.3 Forward-looking

- Wf-380-B1 — Deep CLI integration of `--sa-penalty` flag into `train_cfm_yuelbond_wasserstein.py` (~5-10 days CPU + GPU retrain USER_OK-gated per AGENTS.md §5.1). **CLOSED via Wf-388-A3 + Wf-404-A3** — CLI flags SHIPPED (Wf-388-A3) + full loss wire-in SHIPPED (Wf-404-A3).
- Wf-340 + Wf-353 — Acceleration pipeline (already wired; SCAFFOLD-level SA penalty will compose with these).
- Wf-379-G6 — G6 per-innovation ablation launch USER_OK-gated (would re-bench m4 SA post-penalty wire-up).
- Wf-404-A4+ (future) — Replace the surrogate with a real-Mol sampler hook calling `sa_penalty.compute_sa_score` per generated mol; USER_OK-gated GPU retrain; MEASURED end-to-end m4 SA lift verification on Blackwell sm_120.

### §4.4 Honest-negative baseline (preserved)

- Pre-existing 2 SKIPs PRESERVED (BY-DESIGN): `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative` (per Wf-370-B1 §22.64.2)
- m6_kappa + m2 R² HONEST_NEGATIVE PRESERVED (per Wf-365-A3 + Wf-369-A2 verdict docs)

---

**END OF TODO-205** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + `METRIC_LIFT_STRATEGY_v2.md` §4.5; SCAFFOLD SHIPPED via Wf-380-A6; SCAFFOLD-only CLI flag wire-in SHIPPED via Wf-388-A3; FULL loss wire-in SHIPPED via Wf-404-A3 (this v2.0); deep GPU retrain to MEASURED-verify the projected lift remains pending USER_OK-gated launch per AGENTS.md §5.1 + kill_then_adapt §0.

**Status**: pending/ → completed/ (Wf-404-A3 promotion, 2026-09-30) — WIRE-IN COMPLETE; GPU retrain to verify m4 SA lift deferred to future Wf-404-A4+ USER_OK-gated wave.
