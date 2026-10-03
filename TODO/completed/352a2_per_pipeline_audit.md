# Wf-352-A2: Per-Pipeline Gate Count Audit (post-Wf-345/346 + Wf-350)

**Date:** 2026-09-29
**Workflow:** Wf-352 (ultracode residual — user directive "覆盖所有发现的问题把残余的问题解决了")
**Sub-flow:** A2 — verify per-pipeline match (0 drift per Wf-346-B1 finding)
**Mode:** READ-only on Bash; zero new tests; CPU-only; PROJECTED 5-10 min wall.

---

## §1 Task

Per Wf-346-B1 audit: all 4 pipelines MATCH (frontier=25, chemprop_v2_v3=13, cfm=6, all=66).
This Wf-352-A2 re-verifies post-Wf-345/346 changes + post-Wf-350 expected adds.

## §2 Per-pipeline audit (fresh-run 2026-09-29)

```python
from molmetal.scripts.pre_flight_check import build_gates, EXPECTED_GATE_COUNTS
for p in EXPECTED_GATE_COUNTS:
    actual = len(build_gates(p))
    expected = EXPECTED_GATE_COUNTS.get(p)
    drift = expected - actual
```

| Pipeline       | Expected | Actual | Drift | Status |
|----------------|---------:|-------:|------:|:-------|
| all            | 71       | 71     | 0     | MATCH  |
| frontier        | 25       | 25     | 0     | MATCH  |
| chemprop_v2_v3 | 14       | 14     | 0     | MATCH  |
| cfm            | 6        | 6      | 0     | MATCH  |

**All 4 pipelines internally consistent (drift = 0).** The `p3_gate_count_consistency_audit` gate PASSES on `--pipeline all`.

## §3 Delta vs Wf-346-B1 baseline (2026-09-29)

| Pipeline       | Wf-346-B1 baseline | Wf-352-A2 current | Delta (expected) |
|----------------|-------------------:|------------------:|-----------------:|
| all            | 66                 | 71                | **+5**           |
| frontier       | 25                 | 25                | 0                |
| chemprop_v2_v3 | 13                 | 14                | **+1**           |
| cfm            | 6                  | 6                 | 0                |

**Interpretation:** EXPECTED_GATE_COUNTS was updated by Wf-350 to track NEW gates — 1 NEW gate in `chemprop_v2_v3` and 5 NEW gates in `all` (5-metal cross-coverage). All 6 NEW gates are accounted for; actual count matches new expected count exactly. Zero drift on every pipeline.

## §4 Pre-flight p3_gate_count_consistency_audit gate (post fresh-run)

`PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/pre_flight_check.py --pipeline all`

```json
{
  "name": "p3_gate_count_consistency_audit",
  "priority": "P3",
  "pipeline": "all",
  "status": "PASS",
  "evidence": "gate count consistent for pipeline='all': actual=71 == expected=71",
  "wall_sec": 0.0002
}
```

**Verdict:** p3_gate_count_consistency_audit PASSES — gate count consistent across all 4 pipelines.

## §5 Pre-flight overall verdict (independent of gate-count audit)

`pre_flight_check.py --pipeline all` totals (full log at `/tmp/wf352a2_preflight_all.log`):

- pass: 68
- fail: 3
- skip: 0
- total: 71

**3 FAIL gates (independent of gate-count audit):**

| Gate                        | Pipeline | FAIL reason                                                                                                              |
|------------------------------|----------|------------------------------------------------------------------------------------------------------------------------|
| (wf343c4 anchor STALE check) | wf343c4  | 4/11 paper-grade anchors STALE (> 24h): wf121/wf143/wf170 + 1 more                                                    |
| (wf343c4 budget overrun)     | wf343c4  | remaining_hr=-183.04h, non_deferred_cost=83.10h, effective_remaining=-73.10h                                           |
| (wf347a1 pre-retrain smoke)  | wf347a1  | wf343c2_per_wave_check.py FAIL (rc=1)                                                                                  |

**None of these 3 FAILs are caused by gate count drift** — they are pre-existing
anchor STALE / budget overrun / pre-retrain smoke FAIL issues, orthogonal to this
Wf-352-A2 verification. No follow-up required from this audit.

## §6 Verification verdict

| Check                                         | Result   |
|-----------------------------------------------|----------|
| per-pipeline drift = 0 (all 4 pipelines)     | PASS     |
| `p3_gate_count_consistency_audit` PASS        | PASS     |
| EXPECTED_GATE_COUNTS["all"] drift vs Wf-346-B1 baseline | MATCH (+5 from Wf-350 expected adds) |

**OVERALL VERDICT:** **PASS** — per-pipeline match preserved per Wf-346-B1 finding.
All 4 pipelines (all, frontier, chemprop_v2_v3, cfm) have 0 internal drift.
Wf-350 expected adds (+5 in `all`, +1 in `chemprop_v2_v3`) are properly tracked
in EXPECTED_GATE_COUNTS.

## §7 Honest framing (per party-twenty-spirit §3.5)

- **Verification only** (per Wf-346-B1 already showed 0 drift on a 3-pipeline subset).
- **Tests:** 0 NEW tests.
- **Wall:** MEASURED ~3 min CPU-only.
- **Source files:** `molmetal/scripts/pre_flight_check.py` (build_gates, EXPECTED_GATE_COUNTS),
  `molmetal/scripts/pre_flight_check.py --pipeline all` fresh-run 2026-09-29.
- **No fabricated numbers** — all values MEASURED from fresh-run audit output.

## §8 Files

- `/home/hugo/codes/mol-metal/TODO/completed/352a2_per_pipeline_audit.md` (this file)
- `/tmp/wf352a2_preflight_all.log` (pre-flight --pipeline all full log, 2026-09-29)

---

**END OF Wf-352-A2** — owner: Wf-352 verification sub-flow; all 4 pipelines MATCH,
p3_gate_count_consistency_audit PASS, no follow-up required.
