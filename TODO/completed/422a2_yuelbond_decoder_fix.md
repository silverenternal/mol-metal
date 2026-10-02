# Wf-422-A2 — YuelBond decoder env SKIP → PASS closure

**Date:** 2026-10-01
**Author:** Wf-422-A2
**Parent directive:** user verbatim 2026-09-30 *"还有什么没修的bloker吗，也启动ultracode去修"*
**Standing directives:** §5.1 GPU grant + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 + §5.5 filesystem-truth + §5.6 atomic write + §5.7 trainer-flag smoke-required

## §0 TL;DR

`p3_wf343c4_yuelbond_decoder_env` pre-flight gate **CLOSED** SKIP → **PASS** at Wf-422-A2
timestamp 2026-10-01 16:06 UTC by using the venv python interpreter
(`/home/hugo/codes/mol-metal/.venv/bin/python` which has torch 2.14.0+cu130 installed)
instead of the bare `/usr/bin/python` (which lacks torch). Pre-flight cascade lifted
from 95/18/4/117 (Wf-411-B1 baseline) to **112/2/3/117** (Wf-422-A2 timestamp) — the
remaining 3 SKIPs are all BY-DESIGN (Wf-262 wave budget -183.04h overrun + BindingDB
1.07M external dependency) per `party-twenty-spirit` §3.5 honest-framing convention.
**No source patches required.** The Wf-345-A2 env-aware SKIP contract was *correctly*
honoring a real env gap; on this host the venv closes it.

## §1 Import attempt

**Probe (with venv python):**

```bash
/home/hugo/codes/mol-metal/.venv/bin/python -c \
    "from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondDecoder; \
     print('OK')"
```

**Result:**
```
[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
OK
```

The YuelBondDecoder class is at `molmetal.molmetal_lam.lam_chem.yuelbond_decoder.YuelBondDecoder`
(242-line definition at `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py:242`) and
re-exported from `cfm_decoder_yuelbond.py` via line 74-76 `from ... import (YuelBondDecoder,
YuelBondDecoderHead, ...)`. torch 2.14.0+cu130 + CUDA available. **No ImportError.**

## §2 Fix applied

**No source-code fix required** — the env-aware SKIP contract from Wf-345-A2 was
*correctly* identifying the gap between the gate's structural probe (which needs torch)
and the bare `/usr/bin/python` (which lacks torch in `sys.path`).

**Closure path**: invoke pre-flight via the venv python that has torch installed:

```bash
/home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all
```

This is the canonical invocation per AGENTS.md §5.1 standing GPU grant + §5.7 NEW
(Wf-399-A1) trainer-flag smoke-required rule, both of which require the torch-enabled
runtime to satisfy their structural probes.

## §3 Pre-flight SKIP closure verification

**Pre-flight cascade 112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** at Wf-422-A2 timestamp
2026-10-01T16:06:56+0000 wall 200.40 s. Source JSON saved at `/tmp/wf422a2_preflight.json`.

**YuelBond gate verdict — PASS:**
```json
{
  "name": "p3_wf343c4_yuelbond_decoder_env",
  "priority": "P3",
  "pipeline": "wf343c4",
  "status": "PASS",
  "evidence": "cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder
    (per Wf-272I / Wf-269C YuelBond-conditionality structural check;
    --no-yuelbond → decode_ratio 0.000 MEASURED on real cohort)",
  "wall_sec": 0.3285
}
```

**Remaining 3 SKIPs — all BY-DESIGN:**

| Gate | Priority | Evidence | BY-DESIGN reason |
|---|---|---|---|
| `p3_wf343c4_wave_budget_remaining_nonneg` | P3 | remaining_hr=-183.04h + 8 DEFERRED waves | Wf-262 §7.1 session time budget overrun |
| `data_2_1_6_bindingdb_external_honest_negative` (×2) | P1 | BindingDB 1.07M external (~50 precious-metal entities) | AGENTS.md §5.4 Tier-2 / no wet-lab dependency |

**Remaining 2 FAILs — all pre-existing per Wf-411-B1 baseline:**

| Gate | Priority | Evidence | Status |
|---|---|---|---|
| `p2_tests_pass` | P2 | pytest FAIL rc=1: 1 failed, 701 passed, 12 skipped | Wf-358-A1 `%` escape Wf-358-B1 self-regression NOT closed |
| `p3_wf343c2_per_wave_runtime_check` | P3 | wf343c2_per_wave_check.py FAIL rc=1 Wf-347A1 wave | Wf-347A1 pre-retrain smoke cascade per Wf-358-D1 §0 honest-negative baseline |

**Pre-flight delta vs Wf-411-B1 baseline 95/18/4/117:**

| Metric | Before (Wf-411-B1) | After (Wf-422-A2) | Δ |
|---|---|---|---|
| PASS | 95 | 112 | +17 |
| FAIL | 18 | 2 | −16 |
| SKIP | 4 | 3 | −1 |
| TOTAL | 117 | 117 | 0 |

**`p3_wf343c4_yuelbond_decoder_env` is the single SKIP that closed.** The other −16
FAILs and −1 SKIP delta reflect prior wave activity (Wf-410-A2/A3/A4 + Wf-411-A1/A4 +
Wf-421 + Wf-422-A1 + Wf-422-A2 itself) — not introduced by this wave.

## §4 Forward-looking

* **Wf-422+ is the recommended follow-up**: 6+ USER_OK-gated sub-flows covering
  USER_OK-gated paper §6 v4 write + USER_OK-gated paper §3 v4 update + Wf-380
  SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-299 GPU launch
  (BLOCKED on Wf-408-A3 alias mapping) + Wf-394-A1 Lambda CFM wire-in patch
  application + Wf-246-C metal-coordination term retrain (~3 hr GPU). Total
  estimated wall ~52.9 hr GPU + 13 hr CPU + 30 min install.
* **Atomic write protocol**: B1 closeout authored via `atomic_write_file` per
  AGENTS.md §5.6; backup of any existing file in `TODO/completed/422*` created
  via `safe_overwrite_check` first (no existing file → silent mode).
* **Standing directives re-confirmed**: §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (filesystem-
  truth rule applied — file present at `TODO/completed/422a2_yuelbond_decoder_fix.md`
  before declaring SHIPPED) + §5.6 (atomic write used) + §5.7 (no trainer-flag wire-in
  in this wave; gate already covers canonical smoke-required discipline) + §10 (no
  wet-lab planning, no `paper/*` touches, Tier-2 markers preserved).
