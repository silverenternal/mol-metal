# Wf-346-A3 — Register `gate_wf343c4_anchor_freshness_24h` in `build_gates(all)`

**Date:** 2026-09-29
**Wave:** Wf-346 ultracode residual (A3 sub-flow)
**Status:** SHIPPED (CPU-only; gate registration verification + 2 NEW tests)
**Tag discipline:** party-twenty-spirit §3.5 (MEASURED vs PROJECTED)

---

## §0 TL;DR

1. **Per Wf-343D1 finding:** the gate function
   `gate_wf343c4_anchor_freshness_24h` exists at
   `molmetal/scripts/pre_flight_check.py:3215` but was NOT wired
   into the `build_gates('all')` dispatcher — same drift as
   Wf-346-A1 but for the anchor freshness gate.
2. **Registration fix MEASURED at Wf-346-A3 timestamp:**
   `build_gates('all')` returns **66** runners
   (62 baseline + 4 NEW Wf-343C4 gates). All 4 Wf-343C4 gates
   (`p3_wf343c4_hyperparams_config_complete` +
   `p3_wf343c4_yuelbond_decoder_env` +
   `p3_wf343c4_anchor_freshness_24h` +
   `p3_wf343c4_wave_budget_remaining_nonneg`) are wired into
   the dispatcher's `if pipeline == "all"` block.
3. **`EXPECTED_GATE_COUNTS['all'] == 66`** invariant preserved
   (no count drift; matches `TODO/completed/321d_preflight_gate_audit.md`).
4. **2 NEW tests PASS** at `molmetal/tests/test_wf346a3_anchor_registered.py`:
   - `test_wf346a3_anchor_freshness_24h_in_all_pipeline` —
     direct membership check that the gate is in the dispatcher.
   - `test_wf346a3_build_gates_all_returns_at_least_63` —
     pin the "+1" delta for this specific registration (62
     baseline + at least 1 of the 4 NEW Wf-343C4 gates).
5. **Honest framing:** Same registration drift pattern as
   Wf-346-A1 (MEASURED at Wf-343D1 timestamp). Wall: PROJECTED
   10-15 min CPU-only. Tests: 2 NEW (CPU-only).
6. **Forward-looking:** The `gate_wf343c4_anchor_freshness_24h`
   gate itself currently FAILs at runtime because 4 of 11
   paper-grade anchors are stale (> 24h mtime) per Wf-345-A3
   audit. Phase 3 C1-C4 anchor refresh is queued as separate
   workflows.

---

## §1 Wf-343D1 finding context (MEASURED)

Per `TODO/completed/343d1.md` §0.4:

> The single FAIL is `p3_gate_count_consistency_audit`
> (priority P3, pipeline `all`). Drift: `actual=62 vs
> expected=66 [-DRIFT (silent gate loss)]`.

The 4 missing gates were the Wf-343C4 pre-retrain internal-check
gates (per `TODO/completed/343d_final.md` §2.3 Tier-A ladder
items 4-7 + Wf-343D3 §0.2):

| # | Gate | Purpose |
|---:|---|---|
| 1 | `p3_wf343c4_hyperparams_config_complete` | AGENTS.md §5.1 contract: configs > guessing |
| 2 | `p3_wf343c4_yuelbond_decoder_env` | Wf-272I / Wf-269C structural check |
| 3 | `p3_wf343c4_anchor_freshness_24h` | Wf-302A / Wf-304A AnchorGuard freshness (this Wf) |
| 4 | `p3_wf343c4_wave_budget_remaining_nonneg` | Wf-262 §7.1 cross-script integration |

All 4 are now registered.

---

## §2 Registration location (MEASURED at Wf-346-A3 timestamp)

The anchor freshness gate is registered in
`molmetal/scripts/pre_flight_check.py` at line **4804-4809**,
inside the `if pipeline == "all"` block:

```python
(
    "p3_wf343c4_anchor_freshness_24h",
    lambda: gate_wf343c4_anchor_freshness_24h(
        "p3_wf343c4_anchor_freshness_24h", "P3", "wf343c4",
    ),
),
```

The gate function itself is at line **3215**:
```python
def gate_wf343c4_anchor_freshness_24h(
    name: str, priority: str, pipeline: str,
) -> GateResult:
    """Anchor freshness 24h gate ..."""
    ...
```

---

## §3 Verification (MEASURED)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal python -c "
from molmetal.scripts.pre_flight_check import build_gates, EXPECTED_GATE_COUNTS
all_runners = build_gates('all')
print(f'Total gates in all pipeline: {len(all_runners)}')
print(f'EXPECTED_GATE_COUNTS[all]: {EXPECTED_GATE_COUNTS[\"all\"]}')
print(f'Match: {len(all_runners) == EXPECTED_GATE_COUNTS[\"all\"]}')
print(f'p3_wf343c4_anchor_freshness_24h in runners: '
      f'{\"p3_wf343c4_anchor_freshness_24h\" in [n for n, _ in all_runners]}')
"
```

Output (verbatim):
```
Total gates in all pipeline: 66
EXPECTED_GATE_COUNTS[all]: 66
Match: True
p3_wf343c4_anchor_freshness_24h in runners: True
```

All 4 Wf-343C4 gates confirmed PRESENT.

---

## §4 Tests (2 NEW, CPU-only, 0.24 s wall)

**File:** `molmetal/tests/test_wf346a3_anchor_registered.py`

**Run:**
```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
  molmetal/tests/test_wf346a3_anchor_registered.py -v
```

**Result:**
```
molmetal/tests/test_wf346a3_anchor_registered.py::test_wf346a3_anchor_freshness_24h_in_all_pipeline PASSED
molmetal/tests/test_wf346a3_anchor_registered.py::test_wf346a3_build_gates_all_returns_at_least_63 PASSED
========================= 2 passed, 1 warning in 0.24s =========================
```

---

## §5 Syntax check (MEASURED)

```bash
python -c "import ast; ast.parse(open('molmetal/scripts/pre_flight_check.py').read()); print('OK')"
```

Output: `OK`

---

## §6 Backward-compat verdict

- The 2 NEW tests are ADDITIVE only.
- `EXPECTED_GATE_COUNTS['all'] == 66` invariant preserved
  (matches Wf-321D audit doc).
- The 4 NEW Wf-343C4 gates are 'all'-only (NOT in 'frontier' /
  'chemprop_v2_v3' / 'cfm'); per §3.4 design contract, pre-retrain
  internal checks are not paper-grade claims.
- No GPU launches in this Wf-346-A3 (CPU-only verification + 1
  source-file check + 2 NEW tests + 1 doc).

---

## §7 Honest framing (party-twenty-spirit §3.5)

- **Registration drift MEASURED** at Wf-343D1 timestamp
  (`actual=62 vs expected=66 [-DRIFT]`).
- **Registration fix MEASURED** at Wf-346-A3 timestamp
  (`actual=66 vs expected=66 [NO DRIFT]`).
- **Tests MEASURED** 2 NEW at Wf-346-A3 timestamp
  (2 PASS / 0 FAIL / 0 SKIP / 2 TOTAL).
- **Wall PROJECTED** 10-15 min CPU-only (actual: < 1 min for
  verification + test run + doc).
- **Zero GPU launches** (CPU-only per AGENTS.md §5.1).

---

## §8 Cross-references

- `molmetal/scripts/pre_flight_check.py` line 4804-4809 — gate
  registration in `build_gates('all')`.
- `molmetal/scripts/pre_flight_check.py` line 3215 — gate function
  definition.
- `molmetal/scripts/pre_flight_check.py` line 2817-2822 —
  `EXPECTED_GATE_COUNTS` constant.
- `molmetal/tests/test_wf346a3_anchor_registered.py` — 2 NEW tests
  (this Wf).
- `TODO/completed/343d1.md` — Wf-343D1 drift detection context.
- `TODO/completed/343d_final.md` — Wf-343D final synthesis.
- `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D
  gate-count-consistency audit (the gate that catches this drift).
- `TODO/completed/345a3_anchor_freshness_audit.md` — Wf-345-A3
  anchor freshness audit (related but separate: audits the stale
  anchors that this gate detects at runtime).
