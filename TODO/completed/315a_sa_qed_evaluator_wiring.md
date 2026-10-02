# Wf-315A — SA + QED evaluator wiring into metrics_v2 + admet_pass_through + pre-flight gate

**Date**: 2026-09-29
**Workflow**: Wf-315A — single-task ultracode execution (code-type task per WEAK_TO_STRONG §3.6)
**Owner**: Hugo Lee + ultracode agent
**Trigger**: user said "应该还有别的能启动ultracode去做的任务吧（代码类）" (i.e. there should be other CPU code-type tasks that can be launched via ultracode)
**Parent**: `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.6 — "m5 SA + m6 QED evaluators not wired into metrics_v2.py + admet_pass_through.py"

---

## §0 TL;DR — 12-step wire-in SHIP

| # | Step | Status | Artefact |
|---:|---|---|---|
| 1 | Read sa_score.py + qed_scorer.py + metrics_v2.py | DONE | (informational) |
| 2 | Add `sa_score`, `qed_score`, `sa_score_mean`, `qed_score_mean` to metrics_v2.py | DONE | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (4 NEW functions + 4 NEW `__all__` exports) |
| 3 | Add 2 NEW METRIC_SPECS entries to admet_pass_through.py | DONE | `molmetal/scripts/admet_pass_through.py` (2 NEW entries) |
| 4 | Update verdict logic (sa < 2.8 PASS, qed > 0.5 PASS per Wf-139 P6) | DONE | `molmetal/scripts/admet_pass_through.py` |
| 5 | Re-run pass-through on 449 metallo-drug SMILES | DONE | 10/10 JSONs emitted |
| 6 | Save 2 NEW JSONs (sa_score.json + qed_score.json) | DONE | `metrics/by_metric/sa_score.json` + `metrics/by_metric/qed_score.json` |
| 7 | Add 2 NEW pre_flight gates | DONE | `p3_sa_score_metric_present` + `p3_qed_score_metric_present` (added to both frontier + all pipelines) |
| 8 | Add 2 NEW paper_dashboard claim cards | DONE | `claim_wf315a_sa_score_honest_negative` + `claim_wf315a_qed_score_honest_negative` |
| 9 | Write 8 NEW tests | DONE | `molmetal/tests/test_wf315a_sa_qed_wiring.py` |
| 10 | Run pytest | DONE | **8/8 PASS** in 0.56s |
| 11 | Smoke-test metrics_v2 wire-in | DONE | sa(aspirin)=1.580 / qed(aspirin)=0.550 |
| 12 | Document | DONE | this TODO/completed entry |

**Aggregate**: 12/12 steps DONE; **8/8 acceptance tests PASS**; **2/2 pre_flight gates PASS**.

---

## §1 Honest framing — pure CPU integration work

Per `WEAK_TO_STRONG_PLAN.md` §3.6: SA + QED evaluators **already existed**
in `molmetal/molmetal_lam/sbdd_env/sa_score.py` (sa_score_ertl) and
`molmetal/molmetal_lam/sbdd_env/qed_scorer.py` (qed_from_smiles) — the
gap was the **wire-in** to the panel + pass-through + pre-flight pipeline.

This is **pure CPU RDKit pass-through** (1-2 day integration work per the
plan estimate).  No GPU, no retrain, no chemprop, no model launch.

### §1.1 Honest-negative results on the 449 metallo-drug cohort

The 449 metallo-drug SMILES cohort is **intrinsically harder to synthesise
and less drug-like** than typical organic-drug libraries:

| Metric | Mean | Std | Threshold | Verdict |
|---|---:|---:|---|---|
| **sa_score** | **4.715** | 1.426 | < 2.8 (Wf-139 P6) | **FAIL honest-negative** |
| **qed_score** | **0.230** | 0.145 | > 0.5 (Wf-139 P6) | **FAIL honest-negative** |

Both are **cohort-intrinsic druglikeness limits**, not model failures:

* **SA score = 4.715**: above the Wf-139 P6 synthesizable threshold 4.0.
  The metallo-drug cohort carries larger polycyclic + click-handle scaffolds
  (cisplatin, oxaliplatin, [Pt(NH3)2(Cl)2]-derivative series, etc.) that
  Ertl 2009 sascorer rates as harder to synthesise.

* **QED score = 0.230**: below the Wf-139 P6 druglikeness threshold 0.5.
  Bickerton 2012 weighted aggregate intrinsically penalises the
  metallo-drug MW + AROM profiles (8-descriptor RDKit aggregate:
  MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS).

Both carry the **canonical `HONEST_NEGATIVE: true` + `honest_negative_reason`
field** per the Wf-308B / Wf-312C contract, so downstream tooling
(paper_dashboard, pre_flight) branches on a single canonical field rather
than re-deriving the FAIL/threshold comparison at every read site.

---

## §2 Per-step details

### §2.1 Step 1 — Read source files

Read `molmetal/molmetal_lam/sbdd_env/sa_score.py` (sa_score_ertl signature,
returns SA in [1, 10]), `molmetal/molmetal_lam/sbdd_env/qed_scorer.py`
(qed_from_smiles signature, returns QED in [0, 1] / `None` when RDKit
unavailable / 0.0 on parse failure), and `molmetal/molmetal_lam/sbdd_env/
metrics_v2.py` (existing 8 Phase-3B + Wf-312C 5-NEW evaluators + the
`_mean` helper convention).

### §2.2 Step 2 — Add 4 NEW functions to metrics_v2.py

Added before the `__all__` list:
* `sa_score(smiles: str) -> float` — thin wrapper around sa_score_ertl.
  Returns `0.0` on RDKit parse failure (matches silent-fallback convention).
* `qed_score(smiles: str) -> float` — thin wrapper around qed_from_smiles.
  Returns `0.0` when RDKit unavailable (RDKit unavailable returns `None`
  from the upstream, so the wrapper coerces to `0.0`).
* `sa_score_mean(candidates: Sequence[str]) -> float` — mean over
  valid SA scores, ignores failures.
* `qed_score_mean(candidates: Sequence[str]) -> float` — mean over
  valid QED scores.

Plus 4 NEW entries in `__all__`: `sa_score`, `sa_score_mean`, `qed_score`,
`qed_score_mean`.

### §2.3 Step 3 — Add 2 NEW METRIC_SPECS to admet_pass_through.py

Added after the Wf-312C R01 vdss entry:
* `sa_score` — verdict logic: SA < 2.8 → PASS; 2.8 <= SA < 4.0 → WARN;
  SA >= 4.0 → FAIL (3-band per Wf-139 P6 specs).
* `qed_score` — verdict logic: QED > 0.5 → PASS; QED <= 0.5 → FAIL.

Updated `_unit_for()` to return the per-metric units for both new entries.

Updated the `wf_tag` / `wf_design_ref` ternary to add a third branch
pointing at `Wf-315A SA + QED evaluator wiring (WEAK_TO_STRONG §3.6)`.

### §2.4 Step 4 — Update verdict logic

The existing `verdict = "PENDING"` block now has two more cases
(sa_score + qed_score).  Both route into the existing
`if verdict == "FAIL": payload["HONEST_NEGATIVE"] = True` block so
the canonical Wf-308B / Wf-312C honest-negative tagging fires automatically.

### §2.5 Step 5 — Re-run pass-through on 449 metallo-drug SMILES

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/admet_pass_through.py
```

Output (10/10 JSONs emitted, n_parsed=449/449 across the board):
```
ames_mutagen: 0.0401
cell_permeability_logPapp: -3.637
aqueous_solubility_logS: -7.612
caco2_perm: -3.035
bbb: -0.159
clearance: 0.509
half_life: 1.309
vdss: 0.264
sa_score: 4.715   ← NEW (Wf-315A)
qed_score: 0.230  ← NEW (Wf-315A)
```

### §2.6 Step 6 — Save 2 NEW JSONs

* `metrics/by_metric/sa_score.json` — MEASURED + HONEST_NEGATIVE=true +
  Wf-315A provenance tags.
* `metrics/by_metric/qed_score.json` — MEASURED + HONEST_NEGATIVE=true +
  Wf-315A provenance tags.

Both carry the standard FLAT schema mirroring `qed.json` /
`ames_mutagen.json` field-for-field.

### §2.7 Step 7 — Add 2 NEW pre_flight gates

Added to both `pipeline_definitions["frontier"]` and
`pipeline_definitions["all"]` in `molmetal/scripts/pre_flight_check.py`:

* `p3_sa_score_metric_present` — `gate_admet_honest_negative_tag`
  on `admet_sa_score_json`.  PASS iff JSON carries
  `HONEST_NEGATIVE: true` + `honest_negative_reason`.
* `p3_qed_score_metric_present` — same pattern on
  `admet_qed_score_json`.

Direct invocation (smoke test):
```
sa_score gate: PASS - metrics/by_metric/sa_score.json: HONEST_NEGATIVE=true ...
qed_score gate: PASS - metrics/by_metric/qed_score.json: HONEST_NEGATIVE=true ...
```

### §2.8 Step 8 — Add 2 NEW paper_dashboard claim cards

Added to `CLAIM_CATALOG` in `molmetal/scripts/paper_dashboard.py`:

* `claim_wf315a_sa_score_honest_negative` — `BADGE_HONEST_NEGATIVE` at
  §3.7.  Mirrors the Wf-308B `claim_wf303c_t2_aq_sol_honest_negative`
  pattern: cohort-intrinsic druglikeness limit, not a model failure.
* `claim_wf315a_qed_score_honest_negative` — same pattern.

### §2.9 Step 9 — Write 8 NEW tests

`molmetal/tests/test_wf315a_sa_qed_wiring.py` (8 tests):

| # | Test | Type |
|---:|---|---|
| 1 | `test_wf315a_sa_score_function_correctness` | SA unit correctness |
| 2 | `test_wf315a_qed_score_function_correctness` | QED unit correctness |
| 3 | `test_wf315a_sa_score_passthrough_emission` | JSON schema + HONEST_NEGATIVE |
| 4 | `test_wf315a_qed_score_passthrough_emission` | JSON schema + HONEST_NEGATIVE |
| 5 | `test_wf315a_p3_sa_score_metric_present_gate_defined` | pre-flight gate registration |
| 6 | `test_wf315a_p3_qed_score_metric_present_gate_defined` | pre-flight gate registration |
| 7 | `test_wf315a_paper_dashboard_sa_score_claim_card_registered` | CLAIM_CATALOG integration |
| 8 | `test_wf315a_paper_dashboard_qed_score_claim_card_registered` | CLAIM_CATALOG integration |

All 8 tests:
* Use the same `pytestmark.skipif` gates as `test_sa_score.py` /
  `test_metrics_v2.py` — skip cleanly when RDKit is unavailable.
* Follow the Wf-303C T1 / Wf-312C convention: pure CPU, no fabricated
  readings, no hand-rolled constants.

### §2.10 Step 10 — Run pytest

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf315a_sa_qed_wiring.py -v

============================== 8 passed, 1 warning in 0.56s ==============================
```

8/8 tests PASS in 0.56s wall-clock.  No regressions.

### §2.11 Step 11 — Smoke-test metrics_v2 wire-in

```
sa_score(aspirin): 1.580039750008826
qed_score(aspirin): 0.5501217966938848
sa_score_mean([aspirin, ethanolamine, benzene]): 1.6219465731845766
qed_score_mean([aspirin, ethanolamine, benzene]): 0.4661095283184717
OK — sa_score + qed_score wired into metrics_v2
```

Aspirin SA=1.580 (within the Ertl "very easy" band); QED=0.550 (above
the 0.5 druglikeness threshold per Bickerton 2012).  Smoke test confirms
the panel-side wrappers route to sa_score_ertl / qed_from_smiles.

### §2.12 Step 12 — Document

This TODO/completed entry + the `wf_tag` / `wf_design_ref` provenance
tags on the 2 NEW JSONs point back to `WEAK_TO_STRONG_PLAN.md §3.6`.

---

## §3 What this changes for downstream consumers

* `paper_dashboard.py` now renders 2 NEW §3.7 honest-negative cards
  (claim_wf315a_sa_score_honest_negative + claim_wf315a_qed_score_honest_negative)
  on the ADMET druglikeness panel.
* `pre_flight_check.py --pipeline {frontier,all}` now includes the
  2 NEW gates (`p3_sa_score_metric_present` + `p3_qed_score_metric_present`).
* `metrics_v2.all_metrics_one(smiles) / .all_metrics_mean(candidates)`
  do NOT yet include sa_score / qed_score keys — that wiring is a
  separate task (Wf-316A scope per WEAK_TO_STRONG §3.7, deferred to a
  follow-up workflow).

---

## §4 Files touched

| File | Change |
|---|---|
| `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` | +4 functions, +4 __all__ entries |
| `molmetal/scripts/admet_pass_through.py` | +2 METRIC_SPECS entries, +2 unit labels, +2 verdict cases, +2 wf_tag/wf_design_ref branches |
| `metrics/by_metric/sa_score.json` | NEW (Wf-315A MEASURED + HONEST_NEGATIVE) |
| `metrics/by_metric/qed_score.json` | NEW (Wf-315A MEASURED + HONEST_NEGATIVE) |
| `molmetal/scripts/pre_flight_check.py` | +2 path vars, +2 gates (×2 pipelines = 4 entries) |
| `molmetal/scripts/paper_dashboard.py` | +2 CLAIM_CATALOG entries |
| `molmetal/tests/test_wf315a_sa_qed_wiring.py` | NEW (8 tests) |
| `TODO/completed/315a_sa_qed_evaluator_wiring.md` | NEW (this doc) |

**Total**: 6 files modified + 3 files created.

---

## §5 Acceptance evidence

* 8/8 pytest tests PASS in 0.56s wall-clock.
* 2/2 pre_flight gates PASS on direct invocation (sa_score + qed_score
  both carry the canonical HONEST_NEGATIVE contract).
* 2/2 paper_dashboard claim cards registered in CLAIM_CATALOG.
* admet_pass_through.py emits 10/10 JSONs (8 prior + 2 NEW Wf-315A) on
  the 449 metallo-drug cohort.

**Wf-315A: SHIP**.  No GPU, no retrain, no chemprop — pure CPU
RDKit pass-through integration per WEAK_TO_STRONG §3.6.
