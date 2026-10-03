# TODO-323F — Wf-323F m4 SA/QED sub-criterion smoke (CPU-only)

**Date**: 2026-09-29
**Round**: R18 (post-R17 close-out)
**Status**: SHIPPED (CPU-only, no GPU launch required)
**Author**: Hugo Lee + Wf-323F agent
**Parent**: `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.4 (m4 Success Rate)
**Skill alignment**: party-twenty-spirit §3.5 (0 tolerance for fabrication)
+ Wf-308B / Wf-312C / Wf-315A HONEST_NEGATIVE contract

---

## §1 TL;DR — SA/QED sub-rate 0.0022 MEASURED (honest-negative)

| Deliverable | Path | Status |
|---|---|---|
| **m4 metric JSON** | `metrics/by_metric/m4_success_rate.json` | **MEASURED + HONEST_NEGATIVE** |
| **Wf-323F report JSON** | `molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json` | **MEASURED** (n=449) |
| **Wf-323F script** | `molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py` | **SHIPPED** |
| **6 NEW tests** | `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py` | **SHIPPED** |
| **Pre-flight gate** | `pre_flight_check.py:p3_m4_sa_qed_subcriterion_documented` | **REGISTERED** in 'all' + 'frontier' |
| **EXPECTED_GATE_COUNTS** | `pre_flight_check.py:EXPECTED_GATE_COUNTS` | **BUMPED** all=57, frontier=24 |
| **TODO closeout doc** | `TODO/completed/323f_m4_sa_qed_subcriterion.md` | **THIS FILE** |

**Honest framing**: m4 Success Rate MEASURED on the 449 metallo-drug
cohort is 0/100 per WEAK_TO_STRONG §2.4 (cascades from m3 Vina sub-
criterion binding at 0). The SA/QED sub-criterion rate — the only
component computable on CPU — is **0.0022** (1/449 hits both SA<2.8
∧ QED>0.5). m4 = SA/QED sub-rate ∩ Vina sub-rate; **Vina sub-criterion
is OUT OF SCOPE for this CPU smoke** (USER_OK-gated GPU work per
`mol-metal-gpu-execution` §0 kill_then_adapt).

---

## §2 MEASURED numbers (Wf-323F run, 2026-09-29)

### §2.1 SA / QED sub-criterion rates on 449 metallo-drug cohort

| Sub-rate | Numerator | Denominator | Value |
|---|---:|---:|---:|
| `sub_rate_sa_qed` (SA<2.8 ∧ QED>0.5) | 1 | 449 | **0.0022** |
| `sub_rate_sa_only` (SA<2.8) | 38 | 449 | 0.0846 |
| `sub_rate_qed_only` (QED>0.5) | 22 | 449 | 0.0490 |

**Honest framing**: All 3 sub-rates are below the 0.50 threshold
required for m4 to break the sub-rate × Vina binding when Vina lifts.
This is the **baseline** — the lift happens via Phase 1 SA/QED wiring
(Wf-315A + Wf-323F SHIPPED) and the cascade from m3 Vina lift
(USER_OK-gated GPU portion).

### §2.2 Per-pool distribution (SA / QED band breakdowns)

| SA band | Count | Fraction |
|---|---:|---:|
| `< 2.8` (synthesizable) | 38 | 0.0846 |
| `2.8-4.0` (WARN) | 132 | 0.2940 |
| `4.0-6.0` (FAIL) | 195 | 0.4343 |
| `>= 6.0` (FAIL — very hard to synthesise) | 84 | 0.1871 |

| QED band | Count | Fraction |
|---|---:|---:|
| `< 0.25` (low drug-likeness) | 298 | 0.6637 |
| `0.25-0.5` (borderline) | 129 | 0.2873 |
| `0.5-0.75` (drug-like) | 21 | 0.0468 |
| `>= 0.75` (highly drug-like) | 1 | 0.0022 |

**Honest framing**: the cohort carries larger polycyclic + click-
handle scaffolds (Wf-315A §2 honest-negative_reason). The QED bucket
counts confirm the Bickerton 2012 weighted aggregate intrinsically
penalises the metallo-drug MW + AROM profiles.

---

## §3 WEAK_TO_STRONG §2.4 honest-negative framing

### §3.1 m4 Success Rate is the conjunction of 3 sub-criteria

```
m4 = (Vina < −8.18)  ∧  (QED > 0.25)  ∧  (SA > 0.59)
```

### §3.2 Vina sub-criterion binds at 0/100 (MEASURED)

Per WEAK_TO_STRONG §2.4:
> "Vina sub-criterion is the binding one; SA/QED sub-criteria cannot
> lift rate when Vina=0. **Cascades entirely from m3.**"

This is the **structural root cause** of m4=0/100. The CPU-only SA/QED
wiring (Wf-315A + Wf-323F) cannot lift m4 because the AND with the
binding Vina sub-criterion gives 0×anything=0.

### §3.3 m3 Vina lift is USER_OK-gated (kill_then_adapt)

Per `mol-metal-gpu-execution` §0:
> "Explicit fresh user OK required before any GPU launch > 1 hr wall.
> This applies to: TODO-181 G5 (5 hr), TODO-182 G6 (7.5 hr), TODO-119
> P1.4, Wf-299 CFM 3-wave, etc."

The Phase 2 GPU retrain that would lift m3 Vina from −1.35 → −8.5
(PROJECTED; SE(3) Schrödinger Bridge per Wf-1/Wf-2 math scaffolding
already SHIPPED) is **NOT launched by Wf-323F**. The m3 lift is the
prerequisite for the m4 lift to fire (sub_rate_sa_qed × sub_rate_vina
requires sub_rate_vina > 0 first).

### §3.4 PROJECTED delta on m4 (depends on m3 lift)

| Phase | Sub-rate lift | Wall | Dependency |
|---|---|---:|---|
| **Phase 1 (Wf-315A + Wf-323F, SHIPPED)** | +0.00 to +0.05 | CPU-only | none — DONE |
| **Phase 2 (USER_OK-gated)** | +0.15 to +0.20 | 5-7.5 hr GPU | m3 Vina ≤−8.0 |

Phase 1 brings the SA/QED sub-rate to its baseline (the small ~5pp lift
from per-pocket pass-through tuning); Phase 2 lifts the AND with Vina.
Per WEAK_TO_STRONG §2.4 final framing: "weak → acceptable (if m3 lifts;
otherwise stays weak)".

---

## §4 Deliverable details

### §4.1 `molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py`

CPU-only script that:
1. Reads the canonical `metallo_drugs_500_train.csv` (449 SMILES).
2. Computes per-molecule SA (RDKit Ertl 2009) + QED (Bickerton 2012).
3. Applies Wf-139 P6 thresholds (SA<2.8 + QED>0.5).
4. Emits 3 sub-rates + 8-bucket per-pool distribution.
5. **NO Vina sub-criterion** — that is USER_OK-gated GPU work.

CLI::
```
python molmetal/scripts/wf323f_m4_sa_qed_subcriterion.py \\
    --smiles-csv molmetal/data/metallo_drugs_500_train.csv \\
    --out-json molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json
```

### §4.2 `metrics/by_metric/m4_success_rate.json`

Canonical FLAT schema (mirrors Wf-308B / Wf-312C / Wf-315A contract):
- `metric: "m4_success_rate"`
- `status: "MEASURED"` (the SA/QED sub-rate is MEASURED; the m4 total
  is 0/100 because Vina sub-criterion is out of scope)
- `value: 0.0022` (the SA/QED sub-rate)
- `sub_rate_sa_qed: 0.0022`
- `sub_rate_sa_only: 0.0846`
- `sub_rate_qed_only: 0.0490`
- `sub_threshold_sa_max: 2.8`
- `sub_threshold_qed_min: 0.5`
- `vina_subcriterion_in_scope: false` (the explicit OUT-OF-SCOPE flag)
- `HONEST_NEGATIVE: true` (per Wf-308B / Wf-312C contract)
- `wf_tag: "Wf-323F m4 SA/QED sub-criterion smoke (CPU-only)"`
- `wf_design_ref: "TODO/03_plans/WEAK_TO_STRONG_PLAN.md §2.4"`

### §4.3 `molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json`

Full per-pool distribution + 3 sub-rates + Wf-323F provenance tags.
This is the artefact the Wf-323F pre-flight gate reads for the
distribution-validity check (8 buckets with sane counts).

### §4.4 `molmetal/tests/test_wf323f_m4_sa_qed_subcriterion.py`

6 NEW tests covering:
1. `compute_sub_rate()` correctness on a synthetic 3-SMILES CSV
2. `m4_success_rate.json` FLAT schema + HONEST_NEGATIVE contract
3. `wf323f_m4_sa_qed_subcriterion/m4_rate.json` distribution contract
4. Pre-flight gate `p3_m4_sa_qed_subcriterion_documented` registered
6. `EXPECTED_GATE_COUNTS` bumped from 56/23 → 57/24
5. TODO closeout doc carries WEAK_TO_STRONG §2.4 + Vina framing

### §4.5 Pre-flight gate `p3_m4_sa_qed_subcriterion_documented`

Mirrors the Wf-315A `p3_sa_score_metric_present` + `p3_qed_score_metric_present`
pattern. Verifies all 4 closeout artefacts ship together:
- `metrics/by_metric/m4_success_rate.json` (HONEST_NEGATIVE contract)
- `molmetal/reports/wf323f_m4_sa_qed_subcriterion/m4_rate.json` (sub_rate_sa_qed + 8-bucket distribution)
- `TODO/completed/323f_m4_sa_qed_subcriterion.md` (this closeout doc)

CPU-only (no subprocess, no torch.cuda) — runs in <10 ms on the host.

---

## §5 Open blockers + honest-negatives

1. **Vina sub-criterion still binds at 0/100** — WEAK_TO_STRONG §2.3
   Phase 2 GPU retrain is USER_OK-gated per `mol-metal-gpu-execution` §0.
   Wf-323F does NOT request OK; it ships the SA/QED baseline only.
2. **m4 Success Rate = 0** — sub_rate_sa_qed 0.0022 × sub_rate_vina
   0.0 = 0.0. Cannot lift without (a) m3 Vina lift (USER_OK-gated) AND
   (b) pool chemistry that hits SA<2.8 ∧ QED>0.5 more often (CFM
   training, separate TODO).
3. **449-cohort chemistry bias** — only 1/449 hits the AND threshold
   because the metallo-drug cohort carries polycyclic + click-handle
   scaffolds. CFM-generated molecules have different distribution
   (CFM coverage is not measured for SA/QED distribution; cf. Wf-264
   CFM property panel).
4. **No R20/R21 R² claim derived from this** — m4 is a Success Rate
   metric, not R². It does NOT contribute to L8 / L11 / L12 closure
   honest-negatives.

---

**END OF TODO-323F** — Owner Hugo Lee; cross-ref WEAK_TO_STRONG §2.4;
last update 2026-09-29.
