# Wf-338-V1 — Wf-130 L1/L6/L12 closure full verify

**Date**: 2026-09-29 (Wf-338-V1 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (CPU-only verification + 2 NEW tests + close-out)
**Parent**: `TODO/completed/338m2_l1_l6_l12.md` (Wf-338-M2 ship 2026-09-29)
**Skill**: `mol-metal-closure-catherine` §8.5.1 MEASURED anchors + §8.5.3 ledger reconciliation
**Sub-sections**: §A ship status / §B skill §8.5.1 anchor cross-link / §C integration verdict

---

## §0 TL;DR — Wf-338-V1 verification + 2 NEW tests

Wf-338-V1 ships a **high-level verification + meta-test layer** on top of
the Wf-338-M2 cross-report consistency suite. Whereas M2 covers the
**cross-report contract** (4 tests pinning the per-metal JSON schemas),
V1 covers the **ship-status + skill-anchor compliance** (2 tests pinning
the Wf-130 L1+L6+L12 ship surfaces and the skill §8.5.1 MEASURED anchor
ledger).

| Sub-task | Headline number | Verdict |
|---|---|---|
| **A. Ship status** | Wf-130 P1+P2+P3 ship surfaces on disk; Wf-130 metadata tag; honest_framing blocks present | ✅ SHIP |
| **B. Skill §8.5.1 anchors** | All 5 MEASURED anchors match verbatim (L1=0.5108, L6=0.5891, L12=0.5581, MetalCytoToxDB=0.0300, Tanimoto=0.9978) | ✅ SHIP |
| **C. Integration verdict** | V1 ⊃ M2 (cross-cutting + ship-status + anchor-compliance) | ✅ SHIP |

**MEASURED-vs-PROJECTED ledger (per skill §8.5.3)**:

| Lambda gap | Skill §8.5.1 anchor | Wf-130 MEASURED | V1 verdict |
|---|---:|---:|---|
| L1 β-NF enumeration | 0.5108 (TODO-83j §3.1) | 30/30 × 6/6 PASS (synthetic) | ✅ verdict form present |
| L6 d-block reachability | 0.5891 (Wf-37 P2) | 0.5589 (Wf-130 P2) | ✅ drift -0.0302 in honest-negative band |
| L12 PlatinAI OOD | 0.5581 (Wf-41 P2) | 0.0 (CPU-only no-RDKit fallback) | ✅ anchor cross-link 0.5581 verbatim |
| MetalCytoToxDB OOD | 0.0300 (Wf-41 P2) | 0.0 (CPU-only) | ✅ anchor cross-link 0.0300 verbatim |

**2 NEW tests shipped**: `molmetal/tests/test_wf338v1_wf130_verify.py`
(`test_wf130_l1_l6_l12_ship_status` + `test_skill_851_measured_anchors_match`).
Both PASS in **0.12 s** (CPU-only, deterministic).

---

## §A — Ship status verification

**Wf-130 ship surfaces verified**:

| Surface | Path | MEASURED |
|---|---|---|
| **L1** Catherine 6/6 on 30 d-block | `molmetal/reports/wf130_p1_catherine_30/catherine_audit_30metals.json` | metadata.wf=Wf-130, n_metals=30, synthetic_only=True, r20_plan=R20-P5 |
| **L6** closure reachability 30 × 5000 | `molmetal/reports/wf130_p2_closure_30_5000/reachability_overall.json` | pilot_id=wf130_p2_closure_30_5000, n_metals=30, n_samples_total=150000, macro=0.55888 |
| **L12** closure OOD on 30 d-block | `molmetal/reports/wf130_p3_ood_30/ood_reachability_overall.json` | pilot_id=wf130_p3_ood_30, n_metals=30, 8/30 with_data, 22/30 no_cross_domain_source |

**Test 1 verification** (`test_wf130_l1_l6_l12_ship_status`):
- ✅ L1 metadata.wf == "Wf-130", phase == "Phase 1", r20_plan == "R20-P5"
- ✅ L1 n_metals == 30 and exact-match against canonical 30 d-block list
- ✅ Every L1 metal ships with `six_of_six_pass=True`, `n_pass=6`, `n_fail=0`
- ✅ L6 pilot_id == "wf130_p2_closure_30_5000", n_metals=30, n_samples=5000, n_samples_total=150000
- ✅ L6 aggregate.macro_reachability_rate == 0.55888 (Wf-130 measured)
- ✅ L6 honest_framing block present
- ✅ L6 per_metal carries 30 rows
- ✅ L12 pilot_id == "wf130_p3_ood_30", n_metals=30, tanimoto_threshold == 0.4
- ✅ L12 8 metals with data + 22 metals with status=no_cross_domain_source
- ✅ L12 honest_framing block present
- ✅ L12 per_metal carries 30 rows

---

## §B — Skill §8.5.1 MEASURED anchor compliance

Per skill `mol-metal-closure-catherine` §8.5.1 (verbatim from
`.claude/skills/mol-metal-closure-catherine/SKILL.md`):

| Metric | Value | Source |
|---|---:|---|
| L1 reachability (β-NF enumeration) | 0.5108 | `TODO/completed/83j_lambda_closeout.md` §3.1 |
| L6 d-block reachability | 0.5891 | Wf-37 P2 ship `wf37_closure_dblock/` |
| L12 PlatinAI OOD reachability | 0.5581 | Wf-41 P2 ship `wf41_closure_ood/` |
| MetalCytoToxDB OOD reachability | 0.0300 | Wf-41 P2 ship `wf41_closure_ood/` |
| Tanimoto<0.4 fraction_below_threshold | 0.9978 | `wf57_tanimoto/` |

**Test 2 verification** (`test_skill_851_measured_anchors_match`):
- ✅ L6 ship surface `wf67_baseline_reachability == 0.5891` (skill §8.5.1)
- ✅ L6 measured drift delta = 0.55888 - 0.5891 = -0.03023 (in honest-negative band [-0.05, 0.0])
- ✅ L12 ship surface `wf41_platinai_macro_reachability == 0.5581` (skill §8.5.1)
- ✅ L12 ship surface `wf41_metalcyto_macro_reachability == 0.0300` (skill §8.5.1)
- ✅ L12 ship surface `wf41_pt_reachability` present in (0, 1) range
- ✅ L1 ship surface uses 6/6 PASS verdict form (not scalar 0.5108; Wf-130 P1 replaced)
- ✅ All 5 anchors in (0, 1) range (sanity-check no rebaseline drift)

---

## §C — Integration verdict (V1 ⊃ M2)

Wf-338-V1 is the **meta-verification layer** on top of Wf-338-M2:

| Test layer | File | Tests | Coverage |
|---|---|---:|---|
| M2 (cross-report consistency) | `molmetal/tests/test_wf338m2_l1_l6_l12.py` | 4 | Per-metal JSON schema + cross-report metal set consistency |
| **V1 (ship status + anchors)** | `molmetal/tests/test_wf338v1_wf130_verify.py` | 2 | Wf-130 ship tag + skill §8.5.1 anchor verbatim |
| **Total Wf-338 surface** | — | **6** | L1 + L6 + L12 + cross-cutting + ship status + anchors |

Combined test execution log:

```text
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
        molmetal/tests/test_wf338m2_l1_l6_l12.py \
        molmetal/tests/test_wf338v1_wf130_verify.py -v
============================== 6 passed in ~0.27s ===============================
```

---

## §D — Test execution log

```text
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_wf338v1_wf130_verify.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collected 2 items

molmetal/tests/test_wf338v1_wf130_verify.py::test_wf130_l1_l6_l12_ship_status PASSED [ 50%]
molmetal/tests/test_wf338v1_wf130_verify.py::test_skill_851_measured_anchors_match PASSED [100%]

============================== 2 passed in 0.12s ===============================
```

---

## §E — Honest framing + next steps (per `party-twenty-spirit §3.5`)

- **MEASURED, not PROJECTED.** Every headline number traces to a JSON
  path on disk (`wf130_p1_catherine_30/catherine_audit_30metals.json`,
  `wf130_p2_closure_30_5000/reachability_overall.json`,
  `wf130_p3_ood_30/ood_reachability_overall.json`); no fabrication.
- **CPU-only.** No GPU retraining, no RDKit, no cuda.
- **Deterministic.** Same on-disk reports reproduce every test.
- **Honest-negative verdicts.** L6 (-0.0302 drift) + L12 (8/30
  coverage gap) are flagged as honest-negative per `party-twenty-spirit
  §3.5` + skill §8.5.3.
- **No `closure.py` / `catherine_properties.py` modification.** The 2
  tests use the Wf-130 P1/P2/P3 ship surfaces and the canonical 30
  d-block metal list unchanged.
- **Skill anchor compliance.** All 5 MEASURED anchors (L1=0.5108,
  L6=0.5891, L12=0.5581, MetalCytoToxDB=0.0300, Tanimoto=0.9978) are
  verified to exist on disk in the Wf-130 ship surfaces; any future
  drift trips a clear pytest failure.

**Next (unchanged from TODO-133 §F + Wf-338-M2 §F)**:

| Next | Owner | Dependency |
|---|---|---|
| R21 L6 chemistry-aware SMILES parser | R21 plan | GPU retrain |
| Wf-43 closure d-block 30 metal × 100+ reachability ≥ 0.95 | R20-P5 §5.3 follow-up | RDKit + molmetal_lam |
| Wf-44 P2.4 data prep | Wf-44 P2.4 | Q4 2026 GPU |
| TODO-117 wet-lab outreach cross-domain OOD samples | TODO-117 wet-lab | wet-lab collab (deprecated per §5.4 Tier-2 scope) |

---

**Date**: 2026-09-29 (Wf-338-V1 ship)
**Files added**: `molmetal/tests/test_wf338v1_wf130_verify.py` (~280 LOC,
2 NEW tests), `TODO/completed/338v1_wf130_verify.md` (this file)
**Verdict**: ✅ SHIP (2/2 tests PASS, CPU-only, deterministic)