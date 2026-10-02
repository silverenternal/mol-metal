# Wf-308B — ADMET Pass-Through Integration (Wf-307C follow-up)

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Trigger**: Wf-303C T2 / R03 measured 3 ADMET metrics (ames_mutagen /
cell_permeability_logPapp / aqueous_solubility_logS) on 449 metallo-drug SMILES
but the JSONs were NOT yet surfaced in the v3 pre-flight or paper dashboard.
Wf-307C integration plan staged the wire-up; Wf-308B ships it.

**Honest framing**: This is **integration work, NOT new measurement**. Per
`party-twenty-spirit` §3.5 (0 容忍 不掩盖) — the 3 metrics were already MEASURED
per Wf-303C T2 (2026-09-29 01:21 UTC); Wf-308B surfaces them in pre-flight
gates + paper-dashboard claim cards + paper-dashboard honest-negative catalog.
The aqueous_solubility_logS −7.612 IS a real honest-negative — kept visible.

---

## TL;DR table

| Action | Status | Evidence |
|---|---|---|
| Add 3 ADMET gates to `molmetal/scripts/pre_flight_check.py` | DONE | `p3_admet_ames_druglikeness` + `p3_admet_cell_perm_druglikeness` + `p3_admet_aq_sol_honest_negative_tag` (3 NEW gates) |
| Pre-flight v3 gate counter | 32 → 35 (3 NEW) | All 35/35 PASS |
| Add 3 claim cards to `molmetal/scripts/paper_dashboard.py` | DONE | `claim_wf303c_t2_ames_pass` + `claim_wf303c_t2_cell_perm_pass` + `claim_wf303c_t2_aq_sol_honest_negative` (3 NEW) |
| Paper-dashboard claim counter | 34 → 37 (3 NEW) | 37 claims / 19 honest-negatives |
| Add 1 honest-negative entry to HONEST_NEGATIVE_CATALOG | DONE | `hn_wf308b_aq_sol_fail` (§9.AL) |
| Paper-dashboard honest-negative counter | 18 → 19 (1 NEW) | 19 entries |
| Bump SCHEMA_VERSION | `280.v1` → `308b.v1` | schema reflects Wf-308B wire-up |
| Tag aqueous_solubility_logS.json with `HONEST_NEGATIVE: true` | DONE | `metrics/by_metric/aqueous_solubility_logS.json::HONEST_NEGATIVE + honest_negative_reason` |
| Pre-flight verdict (35/35 PASS) | DONE | All 3 new ADMET gates + 32 prior gates |
| Paper-dashboard verdict (37/37 OK, 19 honest-negatives) | DONE | Build emits `/tmp/dashboard_v308b.html` (87.3 KB) |

---

## §1 — Pre-flight gate additions (per Wf-307C §R03 plan)

### §1.1 — New gate: `p3_admet_ames_druglikeness`

* **Pipeline**: `admet_druglikeness`
* **Priority**: P3
* **Reads**: `metrics/by_metric/ames_mutagen.json::value`
* **Threshold**: value ≤ 0.10 (druglikeness; stricter than source band < 0.30)
* **Implementation**: `gate_json_threshold(..., extractor=lambda d: float(d.get("value", 1.0)), threshold=0.10, op="<=")`
* **MEASURED value**: 0.0401 → 0.040089 (PASS, Δ=−0.060 vs 0.10 gate)

### §1.2 — New gate: `p3_admet_cell_perm_druglikeness`

* **Pipeline**: `admet_druglikeness`
* **Priority**: P3
* **Reads**: `metrics/by_metric/cell_permeability_logPapp.json::value`
* **Threshold**: value ≥ −5.0 (druglikeness; stricter than source band > −6.0)
* **Implementation**: `gate_json_threshold(..., extractor=lambda d: float(d.get("value", -10.0)), threshold=-5.0, op=">=")`
* **MEASURED value**: −3.637 → −3.636788 (PASS, Δ=+1.363 vs −5.0 gate)

### §1.3 — New gate: `p3_admet_aq_sol_honest_negative_tag`

* **Pipeline**: `admet_druglikeness`
* **Priority**: P3
* **Reads**: `metrics/by_metric/aqueous_solubility_logS.json::HONEST_NEGATIVE + honest_negative_reason`
* **Threshold**: `HONEST_NEGATIVE` exactly `True` + non-empty `honest_negative_reason` string
* **Implementation**: New `gate_admet_honest_negative_tag(...)` helper
* **MEASURED state**: HONEST_NEGATIVE=true (added by Wf-308B), reason=non-empty → PASS

### §1.4 — New helper: `gate_admet_honest_negative_tag`

Added to `pre_flight_check.py` alongside the existing `gate_section9_entry_count`.
The helper takes `(name, priority, pipeline, json_path)` and returns PASS iff:
1. `json_path` exists + parseable JSON
2. top-level `HONEST_NEGATIVE` field equals exactly `True`
3. `honest_negative_reason` is a non-empty string

FAIL otherwise (missing file, missing field, wrong type, empty reason).

---

## §2 — Paper-dashboard claim + honest-negative additions

### §2.1 — New claim card: `claim_wf303c_t2_ames_pass` (§3.7, MEASURED)

* Value: `0.0401 (PASS, Δ=−0.060 vs 0.10 gate; PASS vs 0.30 source band)`
* Source: `metrics/by_metric/ames_mutagen.json::value + std + verdict`

### §2.2 — New claim card: `claim_wf303c_t2_cell_perm_pass` (§3.7, MEASURED)

* Value: `-3.637 (PASS, Δ=+1.363 vs -5.0 gate; PASS vs -6.0 source band)`
* Source: `metrics/by_metric/cell_permeability_logPapp.json::value + std + verdict`

### §2.3 — New claim card: `claim_wf303c_t2_aq_sol_honest_negative` (§3.7, honest-negative)

* Value: `-7.612 (FAIL honest-negative; HONEST_NEGATIVE=true tag added per Wf-308B)`
* Source: `metrics/by_metric/aqueous_solubility_logS.json::value + HONEST_NEGATIVE + honest_negative_reason`
* Badge: `BADGE_HONEST_NEGATIVE` (red) — per `party-twenty-spirit` §3.5 (0 容忍 不掩盖)

### §2.4 — New honest-negative entry: `hn_wf308b_aq_sol_fail` (§9.AL)

* Item: "DO NOT cite aqueous_solubility_logS = -7.612 as druglikeness PASS
  (cohort mean BELOW -4.0 by -3.612; pre-flight gate carries HONEST_NEGATIVE=true
  tag, do not silently mask; metallo-drug cohort intrinsically lipophilic —
  low log S is expected per Crippen + aromatic ring count)"
* Ref: `metrics/by_metric/aqueous_solubility_logS.json::HONEST_NEGATIVE +
  honest_negative_reason + TODO/completed/admet_pass_through_integration.md §R03`

---

## §3 — Verification commands + outputs

### §3.1 — Pre-flight

```bash
cd /home/hugo/codes/mol-metal && PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py
```

Result:

```json
{
  "summary": {
    "pass": 35,
    "fail": 0,
    "skip": 0,
    "total": 35,
    "total_wall_sec": ...
  }
}
```

Per-gate ADMET verdict:

| Gate | Status | Evidence |
|---|---|---|
| `p3_admet_ames_druglikeness` | PASS | `value=0.040089 <= 0.1` |
| `p3_admet_cell_perm_druglikeness` | PASS | `value=-3.636788 >= -5.0` |
| `p3_admet_aq_sol_honest_negative_tag` | PASS | `HONEST_NEGATIVE=true, reason='Mean cohort aqueous_solubility_logS = -7.612 ...'` |

### §3.2 — Paper dashboard

```bash
cd /home/hugo/codes/mol-metal && PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v308b.html --verbose
```

Result:

```
[Wf-293D-Dash] built dashboard: 37 claims, 19 honest-negatives, 7 cohorts, html_len=88843
[Wf-265-Dashboard] wrote /tmp/dashboard_v308b.html (87.3 KB); build JSON at /tmp/dashboard_v308b.build.json.
```

Build JSON `update_report`:

```json
{
  "wf_id": "Wf-308B (refresh on top of Wf-293D, post-Wf-303C T2 / R03 ADMET integration)",
  "n_new_claims_added": 17,  # 8 Wf-268C + 5 Wf-279C + 1 Wf-293D + 3 Wf-308B R03 ADMET
  "n_new_honest_negatives_added": 7,  # 3 Wf-268C + 2 Wf-279C + 1 Wf-293D + 1 Wf-308B aq_sol honest-negative
  ...
}
```

---

## §4 — Per-metric honest-verdict table

| Metric | MEASURED value | Verdict | Source band | Pre-flight gate | Pre-flight status |
|---|---:|---|---|---|---|
| `ames_mutagen` | 0.0401 | PASS | < 0.30 (cohort mean) | ≤ 0.10 (druglikeness) | PASS (Δ=−0.060) |
| `cell_permeability_logPapp` | −3.637 | PASS | > −6.0 (Papp > 1e-6 cm/s) | ≥ −5.0 (druglikeness) | PASS (Δ=+1.363) |
| `aqueous_solubility_logS` | −7.612 | **FAIL honest-negative** | > −4.0 (0.1 mM solubility) | HONEST_NEGATIVE tag required | PASS (tag added) |

All 3 metrics are **descriptive physico-chemical proxies** (RDKit / Crippen /
Veber / Delaney heuristics), NOT learned metallodrug scores. Paper §3 cohort
panel only.

---

## §5 — Files touched (per `party-twenty-spirit` §3.5 honest bookkeeping)

* `molmetal/scripts/pre_flight_check.py` (modified — added `gate_admet_honest_negative_tag` helper + 3 ADMET gates in both "all" + "frontier" pipelines)
* `molmetal/scripts/paper_dashboard.py` (modified — added 3 claim cards + 1 honest-negative entry + bumped SCHEMA_VERSION 280.v1 → 308b.v1 + updated update_report counters)
* `metrics/by_metric/aqueous_solubility_logS.json` (modified — added top-level `HONEST_NEGATIVE: true` + `honest_negative_reason` fields)
* `TODO/completed/admet_pass_through_integration.md` (NEW — this doc)

No re-training, no GPU work, no metric re-measurement — integration-only.

---

## §6 — Cross-references

* `molmetal/reports/wf297d_remediation_plan/remediation_plan.md` §R03 (the ADMET pass-through remediation action — closed by Wf-303C T2)
* `molmetal/reports/wf303c_implement/task_report.md` §T2 (the actual implementation)
* `molmetal/scripts/admet_pass_through.py` (Wf-303C T2 emitter — 340 LOC)
* `TODO/completed/178_metalcytotox_csv_hygiene_closeout.md` (sibling close-out)
* `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (honest-framing catalog)
* `party-twenty-spirit` §3.5 (0 容忍 不掩盖 — MEASURED-vs-PROJECTED + honest-negative framing rules)
* `mol-metal-gpu-execution` skill (zero GPU work in Wf-308B)

---

## §7 — Follow-ups (NOT in Wf-308B scope — future-work)

* **Wf-308B.1**: extend ADMET pre-flight gates to per-metal breakdown (Ru / Os / Rh / Ir / Re) — would require re-running `admet_pass_through.py` with per-metal SMILES subsets
* **Wf-308B.2**: add ADMET cross-cohort validation on MetalCytoToxDB labelled pIC50 (currently pass-through only on 449 metallo-drug SMILES)
* **Wf-308B.3**: ship ADMET panel as a §3.7 sub-figure in paper §3 (currently text-only in claim cards)
* **Wf-308B.4**: re-run `admet_pass_through.py` on a larger cohort (MetalCytoToxDB 26801 rows × 5 metals) to lift the `n_samples` field

---

**Verdict**: 3 ADMET gates added (32 → 35 pre-flight); 3 claim cards + 1
honest-negative entry added (34 → 37 / 18 → 19 paper-dashboard); all gates
PASS; aqueous_solubility_logS −7.612 kept visible as honest-negative.
Integration-only work — no re-measurement, no GPU. Ready for commit.