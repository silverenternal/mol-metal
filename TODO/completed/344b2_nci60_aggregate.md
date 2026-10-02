# Wf-344-B2: pin NCI-60 9-cell wf312b_nci60_smoke_aggregate block structure

**Wf:** Wf-344-B2 (ultracode residual closeout; user voice 2026-09-29:
"可以的，然后你之前没启动起来的那个工作流也启动一下")
**Date:** 2026-09-29 (Wf-344-B2 ship)
**Owner:** Wf-344-B2 subagent (kill_then_adapt relay from Wf-344 ultracode driver)
**Status:** SHIPPED
**Refs:** Wf-338-A3 finding 10 (pre-flight `p3_nci60_9cell_per_cell_aggregate`
alleged FAIL on `wf312b_nci60_smoke_aggregate block missing`); Wf-312B
ship surface (`TODO/completed/nci60_9cell_paper4_ship.md` §6 line 89);
`gate_nci60_9cell_per_cell_aggregate` function at
`molmetal/scripts/pre_flight_check.py:785`.

---

## §1 Context

Per Wf-338-A3 finding 10: pre-flight gate
`p3_nci60_9cell_per_cell_aggregate` (TODO-207 §4.3 P3) was flagged as
FAIL because `metrics/by_metric/nci60_precious_9cell_auc.json` was
alleged to be missing the `wf312b_nci60_smoke_aggregate` block.

The Wf-312B ship surface (`TODO/completed/nci60_9cell_paper4_ship.md`
2026-09-25) explicitly added this block to the JSON.  Per Wf-324E
§line 116 the block IS present in the JSON since Wf-312B.

**Wf-344-B2** delivers an additive 2-test regression layer that pins
the Wf-312B contract at the JSON boundary so a future regression
(block dropped / removed / mutated) cannot silently re-FAIL the gate.

---

## §2 Findings (MEASURED 2026-09-29)

### §2.1 Current block state on canonical JSON (MEASURED)

| Check | MEASURED value | Source |
|---|---|---|
| Block present | True | direct JSON load |
| `block type` | dict | `isinstance(block, dict)` |
| `per_cell_breakdown` keys | 9 | `len(block['per_cell_breakdown'])` |
| `per_cell_breakdown` keys (sorted) | `[A2780, A2780cis, A549, DU145, HCT-116, HCT116, HL60, MCF7, PC3]` | direct |
| `overall_mean` declared | `0.5282` | direct |
| `overall_mean` computed (mean of 9 per-cell means) | `0.5282` (Δ ≈ 0) | direct |
| `highlight.cell` | `HCT116` | direct |
| `highlight.value` | `0.6829` (numeric) | direct |
| `honest_negative.cell` | `PC3` | direct |
| `honest_negative.value` | `0.2153` (< 0.5 random-chance) | direct |
| `honest_negative.badge` | `honest-negative` | direct |
| `honest_negative.honest_negative_reason` | non-empty string | direct |
| `timestamp` | `2026-09-29` | direct |

**Verdict:** block is MEASURED-correct on the canonical JSON at Wf-344-B2
timestamp (2026-09-29).  All 4 pillars (9 cells + HCT116 highlight +
PC3 honest-negative + overall_mean consistency within 1e-4) hold.

### §2.2 Current pre-flight gate status (MEASURED 2026-09-29)

Direct invocation of `gate_nci60_9cell_per_cell_aggregate(...)`:

```
STATUS: PASS
evidence: /home/hugo/codes/mol-metal/metrics/by_metric/nci60_precious_9cell_auc.json:
9-cell breakdown + HCT116 highlight (0.6829) + PC3 honest-negative (0.2153)
+ overall_mean =0.5282 all consistent
```

Fresh-run pre-flight v3 `--pipeline all` (wall ~110 s):

```
TOTAL: 66 | PASS: 63 | FAIL: 3
FAILs: p3_wf343c4_hyperparams_config_complete,
       p3_wf343c4_anchor_freshness_24h,
       p3_wf343c4_wave_budget_remaining_nonneg
```

The Wf-338-A3 finding 10 is **CLOSED** at the gate level — the gate
PASSes today.  The 3 current FAILs are all Wf-343C4 pre-retrain
internal-check gates (anchor freshness / wave budget / hyperparams
config), which are out of Wf-344-B2 scope.

**Stale docs to refresh (separate Wf):** the following files still cite
this gate as FAIL per the Wf-338-A3 audit snapshot:

- `README.md` line 128: `12. p3_nci60_9cell_per_cell_aggregate — nci60_precious_9cell_auc.json: wf312b_nci60_smoke_aggregate block missing`
- `TODO/risks.md` line 33: same text
- `TODO/INDEX.md` line 2358: same text
- `TODO/README.md` line 321: same text
- `TODO/completed/322d_index_readme_wf309_322_sync.md` line 99 + 110: same text

These stale citations should be refreshed in a future docs-only Wf (e.g.
Wf-344-F2 INDEX sync per Wf-332-S1 / Wf-338-F2 INDEX-sync
precedents); Wf-344-B2 only delivers the test layer.

---

## §3 Implementation (MEASURED)

### §3.1 New test file (`molmetal/tests/test_wf344b2_nci60_aggregate.py`)

2 NEW tests; CPU-only, ~0.16 s total (no GPU / network / external
state):

1. **`TestNci60AggregateBlockPresence::test_wf312b_smoke_aggregate_block_present`** —
   load `metrics/by_metric/nci60_precious_9cell_auc.json` and assert
   `wf312b_nci60_smoke_aggregate` is a `dict`.  Also asserts the 4
   required sub-keys (`per_cell_breakdown`, `highlight`,
   `honest_negative`, `overall_mean`) are present.  Catches any future
   regression that drops / renames / mutates the block.

2. **`TestNci60AggregateBlockContract::test_per_cell_breakdown_and_aggregates_match`** —
   pin all 4 Wf-312B contract pillars at the JSON boundary:
   - `per_cell_breakdown` is a dict with exactly 9 keys + every entry
     has a numeric `mean_auc`.
   - `overall_mean` is numeric AND within 1e-4 of the simple mean of
     the 9 per-cell means (per Wf-312B tolerance).
   - `highlight` is a dict with `value` (numeric) + `cell` (string).
   - `honest_negative` is a dict with `value < 0.5` + non-empty
     `honest_negative_reason` + `badge == 'honest-negative'`.

### §3.2 Test run output (MEASURED)

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
plugins: cov-7.1.1
collecting ... collected 2 items

molmetal/tests/test_wf344b2_nci60_aggregate.py::TestNci60AggregateBlockPresence::test_wf312b_smoke_aggregate_block_present PASSED [ 50%]
molmetal/tests/test_wf344b2_nci60_aggregate.py::TestNci60AggregateBlockContract::test_per_cell_breakdown_and_aggregates_match PASSED [100%]

============================== 2 passed in 0.16s ===============================
```

### §3.3 Honest framing

Per `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖):

- Wf-338-A3 finding 10 is **MEASURED-stale**.  The block IS present in
  the JSON since Wf-312B ship (2026-09-25 per the block's `timestamp`
  field) and the gate PASSes today (verified via direct
  `gate_nci60_9cell_per_cell_aggregate(...)` invocation).
- Wf-344-B2's value is the **regression layer**, not the fix — the fix
  happened at Wf-312B ship.  The 2 NEW tests pin the contract so a
  future regression cannot silently re-FAIL the gate.
- The 5 stale documentation files (README.md / risks.md / INDEX.md /
  README.md / 322d sync doc) citing "block missing" are out of Wf-344-B2
  scope; they should be refreshed in a future Wf-344-F2 INDEX sync
  (or equivalent docs-only Wf).

---

## §4 Verdict

| Item | Status |
|---|---|
| Wf-338-A3 finding 10 root cause | **STALE** — block IS present, gate PASSes |
| Gate `p3_nci60_9cell_per_cell_aggregate` status | **PASS** (direct invocation 2026-09-29) |
| Pre-flight v3 fresh-run (`--pipeline all`) | 63 PASS / 3 FAIL / 66 TOTAL (the 3 FAILs are Wf-343C4 pre-retrain internal-check gates, out of Wf-344-B2 scope) |
| `molmetal/tests/test_wf344b2_nci60_aggregate.py` | NEW, 2/2 PASS in 0.16 s |
| `nci60_precious_9cell_auc.json::wf312b_nci60_smoke_aggregate` block | UNCHANGED (already present per Wf-312B) |

**Verdict:** Wf-344-B2 SHIPS the regression layer honestly.  The Wf-338-A3
finding 10 was already closed at Wf-312B ship; the 2 NEW tests pin the
contract at the JSON-boundary.

**Honest-negative note:** the stale "block missing" citations in 5 doc
files (`README.md` line 128, `TODO/risks.md` line 33, `TODO/INDEX.md`
line 2358, `TODO/README.md` line 321, `TODO/completed/322d_index_readme_wf309_322_sync.md`
lines 99 + 110) should be refreshed in a future Wf-344-F2 INDEX sync
(out of Wf-344-B2 scope).

---

## §5 Files added / modified by Wf-344-B2

| Path | Action | Notes |
|---|---|---|
| `molmetal/tests/test_wf344b2_nci60_aggregate.py` | NEW | 2 regression tests for Wf-338-A3 finding 10 (block-presence + 4-pillar contract); CPU-only, ~0.16 s total |
| `metrics/by_metric/nci60_precious_9cell_auc.json` | UNCHANGED | `wf312b_nci60_smoke_aggregate` block was added at Wf-312B ship (2026-09-25) — already present per direct verification |
| `TODO/completed/344b2_nci60_aggregate.md` | NEW | this file |

---

**Date:** 2026-09-29 (Wf-344-B2 ship)
**Cross-refs:** Wf-338-A3 finding 10 (originated this task) — pre-flight
gate `p3_nci60_9cell_per_cell_aggregate` alleged missing block.  Wf-312B
(`TODO/completed/nci60_9cell_paper4_ship.md`) — canonical ship artifact
that added the `wf312b_nci60_smoke_aggregate` block.  Wf-327-SYN
(`TODO/completed/327_synthesis.md` §line 116) — RESOLVED marker for
finding 11 (same item, different alias row index in the Wf-338-A3
audit).  Wf-338-F2 (`TODO/completed/338f1_w2_wf264_smoke.md`) —
pre-flight baseline at 62/62 PASS.  Standing GPU authorization
(AGENTS.md §5.1) — not invoked, Wf-344-B2 is CPU-only.