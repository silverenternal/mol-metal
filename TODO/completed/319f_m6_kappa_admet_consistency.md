# Wf-319F — m6_kappa + ADMET Consistency Audit (MEASURED ACTUAL)

**Date:** 2026-09-29
**Workflow ID:** Wf-319F
**Owner:** Hugo Lee (via subagent, party-twenty-spirit §3.5 honest framing)
**Scope:** Per Wf-315G analysis — verify m6_kappa + hepatotox + logp7_4 honest-negative
tags + claim cards + pre-flight gates are all wired consistently on disk.
**Status:** COMPLETE — **0 inconsistencies found** (audit PASS).

---

## §1 Verification matrix

| # | Check | Path | Result |
|---|---|---|---|
| 1 | m6_kappa.json HONEST_NEGATIVE tag | `metrics/by_metric/m6_kappa.json::HONEST_NEGATIVE` | PASS — `true` |
| 2 | m6_kappa.json validation data | `metrics/by_metric/m6_kappa.json::metalcytotox_validation` | PASS — global Pearson r 0.081, Spearman 0.111, NEGATIVE_WEAK |
| 3 | hepatotox_index.json HONEST_NEGATIVE tag | `metrics/by_metric/hepatotox_index.json::HONEST_NEGATIVE` | PASS — `true` |
| 4 | logp7_4.json HONEST_NEGATIVE tag | `metrics/by_metric/logp7_4.json::HONEST_NEGATIVE` | PASS — `true` |
| 5 | paper_dashboard.py claim cards (3× Wf-311E) | `molmetal/scripts/paper_dashboard.py` lines 607, 616, 625 | PASS — 3 claim cards wired |
| 6 | paper_dashboard.py claim cards (3× Wf-312C R02) | `molmetal/scripts/paper_dashboard.py` lines 639, 648, 657 | PASS — 3 additional R02 claim cards wired |
| 7 | paper_dashboard.py HN catalog (3×) | `molmetal/scripts/paper_dashboard.py` lines 1265, 1271, 1277 | PASS — 3 HN catalog entries wired (hn_wf311e_*) |
| 8 | paper_dashboard.py badge badge wiring | `paper_dashboard.py::BADGE_HONEST_NEGATIVE` (line 108) | PASS — `BADGE_HONEST_NEGATIVE = "honest-negative"` |
| 9 | paper_dashboard.py home-page inclusion | `paper_dashboard.py` lines 2470-2472 | PASS — 3 HN entries in home-page list |
| 10 | section_5_ablation.json ADMET extension axis | `metrics/by_paper_section/section_5_ablation.json::wf312c_admet_extension_axis` (line 51) | PASS — wired (8 metrics) |
| 11 | section_5_ablation.json R02 honest-negative cluster | `metrics/by_paper_section/section_5_ablation.json::wf312c_r02_honest_negative_cluster_axis` (line 56) | PASS — wired (4 metrics: aqueous_solubility_logS / logp7_4 / hepatotox_index / m6_kappa) |
| 12 | pre_flight_check.py gate definition v2 | `molmetal/scripts/pre_flight_check.py` lines 1922, 1929, 1936 | PASS — 3 gates in v2 list |
| 13 | pre_flight_check.py gate definition v3 | `molmetal/scripts/pre_flight_check.py` lines 2160, 2167, 2174 | PASS — 3 gates in v3 list |
| 14 | gate_admet_honest_negative_tag helper | `molmetal/scripts/pre_flight_check.py::gate_admet_honest_negative_tag` (line 1099) | PASS — checks HONEST_NEGATIVE=True + non-empty `honest_negative_reason` |
| 15 | pre-flight live PASS | `python molmetal/scripts/pre_flight_check.py` | PASS — 49/49 PASS, 0 FAIL, 0 SKIP |

---

## §2 Pre-flight live PASS evidence (2026-09-29 02:37:46 UTC)

```
{
  "timestamp": "2026-09-29T02:37:46+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 49,
    "fail": 0,
    "skip": 0,
    "total": 49,
    "total_wall_sec": 248.5087
  }
}
```

### §2.1 ADMET honest-negative gate evidence (verbatim from pre-flight)

```
"name": "p3_admet_hepatotox_honest_negative",
"status": "PASS",
"evidence": "/home/hugo/codes/mol-metal/metrics/by_metric/hepatotox_index.json: HONEST_NEGATIVE=true, reason='Mean cohort hepatotox_index = 0.548 > 0.5 (druglikeness Hugh'..."

"name": "p3_admet_logp7_4_honest_negative",
"status": "PASS",
"evidence": "/home/hugo/codes/mol-metal/metrics/by_metric/logp7_4.json: HONEST_NEGATIVE=true, reason='Mean cohort logp7_4 = 6.413 OUT OF druglikeness 0-5 range (L'..."

"name": "p3_admet_m6_kappa_honest_negative",
"status": "PASS",
"evidence": "/home/hugo/codes/mol-metal/metrics/by_metric/m6_kappa.json: HONEST_NEGATIVE=true, reason='Mean cohort m6_kappa = 0.472 < 0.6 (druglikeness Shape Kappa'..."
```

All 3 gates PASS with the canonical Wf-308B honest-negative-tag contract.

---

## §3 Findings

**Tag consistency status:** CONSISTENT — all 3 metrics carry HONEST_NEGATIVE=true +
non-empty `honest_negative_reason` + `wf_tag` + `wf_design_ref` + `honest_framing_note`.

**paper_dashboard.py status:** CONSISTENT — 6 claim cards (3× Wf-311E + 3× Wf-312C R02) +
3 HN catalog entries + home-page list inclusion. All wired to `BADGE_HONEST_NEGATIVE`.

**section_5_ablation.json status:** CONSISTENT — both `wf312c_admet_extension_axis` (8 metrics)
+ `wf312c_r02_honest_negative_cluster_axis` (4 metrics) wired with cell counts + honest framing.

**pre_flight_check.py status:** CONSISTENT — 3 gates in BOTH v2 (lines 1922/1929/1936) +
v3 (lines 2160/2167/2174) lists using shared `gate_admet_honest_negative_tag` helper
which validates `HONEST_NEGATIVE is True` + `honest_negative_reason` is non-empty string.

**n_consistency_issues_found:** **0** (all 15 checks PASS).

**n_gates_verified:** **3** (p3_admet_hepatotox_honest_negative +
p3_admet_logp7_4_honest_negative + p3_admet_m6_kappa_honest_negative).

---

## §4 Cross-reference (Wf-312C + Wf-314A provenance)

| Workflow | Added | Reference |
|---|---|---|
| Wf-312C R02 | 3 JSON `HONEST_NEGATIVE=true` tags + 3 paper_dashboard claim cards | `molmetal/reports/wf312c_admet_extension/design.md §R02` |
| Wf-311E Part 4 | 3 additional paper_dashboard claim cards + 3 HN catalog entries | `TODO/completed/311e_*` |
| Wf-314A | 3 pre_flight_check.py gates (v2 list) + `gate_admet_honest_negative_tag` helper | `molmetal/scripts/pre_flight_check.py` line 1099 |
| Wf-315G | Identified the audit need; triggered Wf-319F | `TODO/completed/315g_*` |
| Wf-319F (this) | Verified all wiring + ran pre-flight live | THIS DOCUMENT |

---

## §5 Honest framing per party-twenty-spirit §3.5

* All values in this audit are **MEASURED ACTUAL** — pre-flight run timestamp 2026-09-29
  02:37:46 UTC, evidence cited verbatim from JSON.
* **0 inconsistencies found** — no fixes required; this is a pure verification task.
* The 3 honest-negative gates were added by Wf-312C + Wf-314A; this audit confirms they
  remain correctly wired as of 2026-09-29.
* Pre-flight v2 25/25 + v3 (39 → 49) gate counts: 49/49 PASS today.

**END OF Wf-319F AUDIT** — status COMPLETE, 0 inconsistencies found, 3 ADMET gates
verified PASS in both v2 + v3 pre-flight lists.
