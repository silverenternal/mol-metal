# Wf-344-B1 — ADMET HONEST_NEGATIVE field set verification closeout

**Date:** 2026-09-29
**Workflow:** Wf-344-B1 (single-task verification closeout)
**Owner:** Hugo Lee (via subagent, party-twenty-spirit §3.5 honest framing)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED), §5.1 (kill_then_adapt
GPU directive — NO retrain launched), §5.2 (honest framing mandate)
**Cross-ref:** Wf-308B / Wf-312C R02 / Wf-325E / Wf-338-A3 finding 6-9

---

## §0 TL;DR — VERIFICATION closeout (no field-fixes needed)

| Item | Status |
|---|---|
| Read & verified 4 ADMET JSONs `HONEST_NEGATIVE` field | DONE — all 4 carry `HONEST_NEGATIVE: true` + non-empty `honest_negative_reason` + `wf_tag` + `wf_design_ref` |
| Pre-flight gate state for 4 ADMET honest-negative gates | ALL 4 PASS (per pre-flight v3 fresh-run 2026-09-29) — see §2 |
| 4 NEW regression tests written | DONE — `molmetal/tests/test_wf344b1_admet_honest_negative.py` (4 tests, all PASS) |
| Test wall | 0.11 s (CPU-only) |
| **`metrics/by_metric/*.json` files modified?** | **NO** — VERIFICATION-only closeout; all 4 JSONs already carry the canonical field set per Wf-308B / Wf-312C R02 / Wf-325E sister closeout |

**Aggregate:** 4/4 ADMET metric JSONs verified on-disk with `HONEST_NEGATIVE: true` +
companion field set; 4/4 pre-flight gates PASS; 4 NEW regression tests SHIPPED
(lock in the current correct state); ZERO GPU launches performed.

---

## §0.1 Honest framing (party-twenty-spirit §3.5)

**The Wf-344-B1 task description stated that 4 ADMET gates FAIL because the JSON
HONEST_NEGATIVE field was `None`. The reality on disk is the opposite — all 4
JSONs already carry the canonical field set as of Wf-325E sister closeout (also
dated 2026-09-29).**

**Verbatim task description (Wf-344-B1 brief):**
> "p3_admet_{aq_sol,hepatotox,logp7_4,m6_kappa}_honest_negative_tag gates FAIL
> because HONEST_NEGATIVE=None (expected True)"

**Verbatim MEASURED reality (pre-flight v3 fresh-run 2026-09-29):**
```
"name": "p3_admet_aq_sol_honest_negative_tag", "status": "PASS",
"name": "p3_admet_hepatotox_honest_negative", "status": "PASS",
"name": "p3_admet_logp7_4_honest_negative", "status": "PASS",
"name": "p3_admet_m6_kappa_honest_negative", "status": "PASS",
```

**Conclusion:** No field fix needed. The 4 NEW tests in this closeout are
**regression-protection** — they lock in the current correct state so future
metric JSON regenerations cannot silently drop the HONEST_NEGATIVE field set
without breaking a test (per Wf-308B / Wf-312C R02 contract).

**No fabrication per AGENTS.md §1 MEASURED-vs-PROJECTED discipline** — this
closeout documents the VERIFICATION finding, not a fix that was never needed.

---

## §1 Per-metric verification table

The 4 BELOW druglikeness ADMET metrics per Wf-308B / Wf-312C R02 / Wf-325E
sister closeout:

| # | Metric | Value | Std | Verdict | HONEST_NEGATIVE field | honest_negative_reason | wf_tag | JSON path |
|---:|---|---:|---:|---|---|---|---|---|
| 1 | `aqueous_solubility_logS` | -7.6122 | 3.936 | **FAIL** (HONEST_NEGATIVE) | **True** | "Mean cohort aqueous_solubility_logS = -7.6122 (target band > -4.0)... verdict FAIL; auto-attached HONEST_NEGATIVE tag per Wf-308B / Wf-312C contract..." | `Wf-303C T1 / R03 ADMET pass-through` | `metrics/by_metric/aqueous_solubility_logS.json` |
| 2 | `hepatotox_index` | 0.5477 | n/a | **FAIL** (HONEST_NEGATIVE) | **True** | "Mean cohort hepatotox_index = 0.548 > 0.5 (druglikeness Hughes 2008 threshold)... honest-negative close per Wf-312C R02 druglikeness panel." | `Wf-312C R02 honest-negative close` | `metrics/by_metric/hepatotox_index.json` |
| 3 | `logp7_4` | 6.4128 | n/a | **FAIL** (HONEST_NEGATIVE) | **True** | "Mean cohort logp7_4 = 6.413 OUT OF druglikeness 0-5 range (Lipinski ceiling) by +1.413... honest-negative close per Wf-312C R02 druglikeness panel." | `Wf-312C R02 honest-negative close` | `metrics/by_metric/logp7_4.json` |
| 4 | `m6_kappa` | 0.4719 | 0.1588 | **FAIL** (HONEST_NEGATIVE) | **True** | "Mean cohort m6_kappa = 0.472 < 0.6 (druglikeness Shape Kappa Index threshold per Wf-252 / ChemAxon PMC11387870) by -0.128 — Shape Kappa has NEGATIVE_WEAK MetalCytoToxDB validation... honest-negative close per Wf-312C R02 druglikeness panel." | `Wf-252-Apply Fix #2 (F2-M6Kappa) + Wf-312C R02 honest-negative close` | `metrics/by_metric/m6_kappa.json` |

**All 4 fields verified present and correctly typed** (see §3 test results).

---

## §2 Pre-flight gate state (per pre-flight v3 fresh-run 2026-09-29)

The 4 BELOW ADMET metric gates that emit via
`gate_admet_honest_negative_tag` helper in `molmetal/scripts/pre_flight_check.py`:

| Gate | Helper | JSON path | Status |
|---|---|---|---|
| `p3_admet_aq_sol_honest_negative_tag` | `gate_admet_honest_negative_tag` | `metrics/by_metric/aqueous_solubility_logS.json` | **PASS** |
| `p3_admet_hepatotox_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/hepatotox_index.json` | **PASS** |
| `p3_admet_logp7_4_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/logp7_4.json` | **PASS** |
| `p3_admet_m6_kappa_honest_negative` | `gate_admet_honest_negative_tag` | `metrics/by_metric/m6_kappa.json` | **PASS** |
| `p3_admet_4_below_documented` (aggregated, Wf-325E) | `gate_wf325e_admet_4_below_documented` | ALL 4 above + `TODO/completed/325e_admet_4_below_final.md` | **PASS** |

**Verbatim evidence lines from pre-flight JSON output:**
```
{"name": "p3_admet_aq_sol_honest_negative_tag", "priority": "P3", "pipeline": "admet_druglikeness", "status": "PASS"}
{"name": "p3_admet_hepatotox_honest_negative", "priority": "P3", "pipeline": "admet_druglikeness", "status": "PASS"}
{"name": "p3_admet_logp7_4_honest_negative", "priority": "P3", "pipeline": "admet_druglikeness", "status": "PASS"}
{"name": "p3_admet_m6_kappa_honest_negative", "priority": "P3", "pipeline": "admet_druglikeness", "status": "PASS"}
{"name": "p3_admet_4_below_documented", "priority": "P3", "pipeline": "admet_druglikeness", "status": "PASS"}
```

**5/5 ADMET druglikeness gates PASS** (no FAIL introduced, no SKIP).

---

## §3 NEW regression tests (4 tests, all PASS)

**File:** `molmetal/tests/test_wf344b1_admet_honest_negative.py` (NEW, ~190 LOC)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
collected 4 items

molmetal/tests/test_wf344b1_admet_honest_negative.py::TestAqueousSolubilityLogS::test_honest_negative_field_set_present PASSED [ 25%]
molmetal/tests/test_wf344b1_admet_honest_negative.py::TestHepatotoxIndex::test_honest_negative_field_set_present PASSED [ 50%]
molmetal/tests/test_wf344b1_admet_honest_negative.py::TestLogp74::test_honest_negative_field_set_present PASSED [ 75%]
molmetal/tests/test_wf344b1_admet_honest_negative.py::TestM6Kappa::test_honest_negative_field_set_present PASSED [100%]

============================== 4 passed in 0.11s ===============================
```

### §3.1 Test design

Each test inherits from a shared `_AdmetHonestNegativeMixin` that asserts:

1. **File existence on disk** — `metrics/by_metric/<metric>.json` present
   (AGENTS.md §1 MEASURED-vs-PROJECTED).
2. **JSON parseable** — `json.JSONDecodeError` not raised.
3. **Metric key matches filename** — `data["metric"] == metric_name` (defensive
   cross-check; the gate helpers use the filename path not the `metric` field).
4. **All 4 required fields present** — `HONEST_NEGATIVE`,
   `honest_negative_reason`, `wf_tag`, `wf_design_ref`.
5. **`HONEST_NEGATIVE` is exactly `True`** (boolean — NOT truthy).
6. **`honest_negative_reason` is non-empty string** (per
   `gate_admet_honest_negative_tag` helper contract in
   `molmetal/scripts/pre_flight_check.py:1288`).
7. **`wf_tag` + `wf_design_ref` are non-empty strings** (provenance per
   Wf-308B / Wf-312C R02 contract).

### §3.2 Why 4 separate tests (one per metric)

Mirrors the pre-flight gate topology — each metric has its own
`p3_admet_<metric>_honest_negative*` gate (4 single-metric gates + 1
aggregated `p3_admet_4_below_documented` gate). One-test-per-metric gives
granular regression signal: a failure pinpoints exactly which metric JSON
dropped the canonical field set.

### §3.3 What the tests deliberately do NOT do

- Do **not** modify any `metrics/by_metric/*.json` (none needed fixing).
- Do **not** modify `molmetal/scripts/pre_flight_check.py` (no gate code change
  needed; the helper already exists per Wf-308B).
- Do **not** invoke GPU (per AGENTS.md §5.1 kill_then_adapt directive).
- Do **not** touch `paper/*` (per AGENTS.md §5.3 no-paper-work directive).
- Do **not** modify `EXPECTED_GATE_COUNTS` (no new gate added; this is
  VERIFICATION-only).

---

## §4 Files touched / created

| File | Change |
|---|---|
| `molmetal/tests/test_wf344b1_admet_honest_negative.py` | NEW (190 LOC, 4 tests, regression-protection for 4 BELOW ADMET metric JSONs) |
| `TODO/completed/344b1_admet_honest_negative.md` | NEW (this doc) |

**Total**: 2 NEW files; **ZERO** modifications to existing `metrics/by_metric/*.json` or
`molmetal/scripts/pre_flight_check.py` (no fix needed; VERIFICATION-only closeout).

---

## §5 Standing-rule compliance recap

| Rule | Status | Evidence |
|---|---|---|
| AGENTS.md §1 — MEASURED-vs-PROJECTED | PASS | §1 per-metric verification table cites the actual on-disk JSON paths + verbatim field values; §2 pre-flight gate state cites verbatim JSON evidence lines |
| AGENTS.md §5.1 — kill_then_adapt GPU directive | PASS | ZERO GPU launches in Wf-344-B1; pure CPU verification; Wf-182 G6 + Wf-124 P3 G4/G5 training pipelines UNTOUCHED |
| AGENTS.md §5.2 — honest framing mandate | PASS | §0.1 explicitly contrasts the task description's "FAIL because HONEST_NEGATIVE=None" claim against the VERIFIED on-disk reality (all 4 PASS); no field-fix fabrication |
| AGENTS.md §5.3 — no paper work without authorization | PASS | ZERO touches to `paper/*`; this closeout is computational-only ship surface |
| AGENTS.md §5.4 — Tier-2 strategy is computational-only | PASS | VERIFICATION closeout is computational-only; no wet-lab planning per 2026-09-29 user directive |
| party-twenty-spirit §3.5 — honest framing | PASS | §0.1 honest framing; §1 per-metric verification; §2 verbatim pre-flight evidence |

---

## §6 Cross-references

- **Wf-308B** — Original `gate_admet_honest_negative_tag` helper introduced
  for `aqueous_solubility_logS` (`molmetal/scripts/pre_flight_check.py:1246`).
- **Wf-312C R02** — Extended the helper to 3 NEW ADMET metrics
  (`hepatotox_index` + `logp7_4` + `m6_kappa`); each metric JSON manually
  updated with the canonical field set.
- **Wf-325E** — Sister closeout (`TODO/completed/325e_admet_4_below_final.md`)
  that aggregated the 4 metrics into a single `p3_admet_4_below_documented`
  pre-flight gate; this Wf-344-B1 closeout provides regression-protection at
  the per-metric unit-test level (complementing Wf-325E's aggregated gate).
- **Wf-338-A3 finding 6-9** — The original task brief cites this finding as
  the trigger; on-disk reality (per pre-flight v3 2026-09-29 fresh-run) is
  the gates PASS (no FAIL since Wf-325E closeout).
- **Wf-308B / Wf-312C R02 contract** — Canonical field set:
  `HONEST_NEGATIVE: true` + `honest_negative_reason: <string>` + provenance
  tags (`wf_tag` + `wf_design_ref`).

---

## §7 Acceptance evidence

- **4/4 ADMET metric JSONs verified** — all carry `HONEST_NEGATIVE: true` +
  companion field set on disk.
- **4/4 pre-flight gates PASS** — verified via `pre_flight_check.py
  --pipeline all --json /tmp/preflight.json` fresh-run.
- **4/4 NEW regression tests PASS** — `pytest
  molmetal/tests/test_wf344b1_admet_honest_negative.py -v` → `4 passed in 0.11s`.
- **ZERO GPU launches** — pure CPU verification per user direction.
- **ZERO `metrics/by_metric/*.json` modifications** — VERIFICATION-only; the
  Wf-325E sister closeout already shipped the canonical field set.
- **`paper/*` UNTOUCHED** — AGENTS.md §5.3 compliance preserved.

**Wf-344-B1: SHIP (VERIFICATION)**. The 4 BELOW ADMET druglikeness metric
JSONs are confirmed to carry the Wf-308B / Wf-312C R02 canonical HONEST_NEGATIVE
field set; 4 NEW regression tests lock in the current correct state; 5/5
ADMET druglikeness pre-flight gates PASS; user can re-launch pre-flight any
time without re-deriving the FAIL/threshold comparison at every read site.