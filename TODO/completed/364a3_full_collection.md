# Wf-364-A3 — Full ``test_wf*.py`` Collection (Wf-360-A2 expansion)

**Date:** 2026-09-29
**Owner:** Hugo Lee (via workflow harness)
**Status:** SHIPPED (CPU-only)
**Pre-flight:** v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED at
Wf-328-SYN baseline (not re-run by Wf-364-A3 — docs-only).

## §1 Background

Per Wf-360-A2: cross-Wf test integration suite collected 30/30 tests as a
**parametrized subset**. Wf-364-A3 expands that to a **dynamic collection of
every** ``test_wf*.py`` file under :mod:`molmetal.tests`, with per-file
import-time tracking + GPU-only marker skip verification.

Per Wf-364-A3 task brief: "All 227 test files → single pytest collect".
**MEASURED baseline is 283, not 227** — flagged as HONEST_NEGATIVE_FRAME
(see §3 below).

## §2 Deliverables

1. **NEW file:** `molmetal/tests/test_wf364a3_full_collection.py` (~190 LOC)
2. **NEW report:** `molmetal/reports/wf364a3/import_times.json`
3. **NEW tests:** 287 PASS / 0 FAIL / 287 TOTAL
   - 2 class methods (collection + GPU skip)
   - 285 parametrized per-file smoke tests

## §3 Honest framing (party-twenty-spirit §3.5)

### §3.1 HONEST_NEGATIVE_FRAME — 227 vs 283

| Source | Count | Status |
|---|---:|---|
| Wf-364-A3 brief | 227 | PROJECTED |
| ``find molmetal/tests -name 'test_wf*.py' \| wc -l`` (pre-Wf-364-A3) | **283** | MEASURED |
| Same command (post-Wf-364-A3) | **285** | MEASURED (incl. this file + Wf-364-A4 file added concurrently) |

**Verdict:** the Wf-364-A3 dynamic-collection surface collects the
**MEASURED 283 (→ 285)** test files, not the PROJECTED 227. This is a
~25% under-projection in the brief — collection surfaces ground truth,
not the brief's projection.

### §3.2 HONEST_NEGATIVE — pre-existing import bug exposed

`test_wf328w8_cuda_graph_production.py` has a `@dataclass` decorator on
`_MPNNConfigStub` (line 53) that fails when the module is loaded via
`importlib.util.spec_from_file_location` WITHOUT registration in
`sys.modules`. The dataclass decorator calls
`sys.modules.get(cls.__module__).__dict__` which raises
`AttributeError: 'NoneType' object has no attribute '__dict__'` when the
module isn't in `sys.modules`.

**Resolution in Wf-364-A3:** the probe now registers the synthetic module
name in `sys.modules` BEFORE `exec_module` (mirroring pytest's behaviour
for real test modules). After the fix all 285 test files import cleanly.

**MEASURED per-file import wall-times (top 5 slowest):**

| Wall (s) | File |
|---:|---|
| 0.987 | `test_wf146_resume.py` (torch import at module scope) |
| 0.132 | `test_wf325a_r2_32d_expansion.py` |
| 0.034 | `test_wf198_embed_fallback.py` |
| 0.003 | `test_wf331x4_triton_kernels.py` |
| 0.003 | `test_wf364a3_full_collection.py` (this file — minimal imports) |

### §3.3 Scope honesty

This Wf covers **collection only**, not execution. The parametrized smoke
tests verify each test file imports cleanly — they do NOT execute the
test bodies. A follow-up Wf would invoke the full CPU-only subset for
actual execution (e.g. via `pytest -m "not gpu" molmetal/tests/`).

## §4 Test results (MEASURED)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf364a3_full_collection.py -v
======================= 287 passed, 1 warning in 21.88s ========================
```

- 2 class methods PASSED
- 285 parametrized per-file import smokes PASSED
- 1 warning: hypothesis plugin Skipping collection of '.hypothesis' (unrelated)
- 0 GPU launches performed (CPU-only)

## §5 Standing directives honoured

- **§5.1 GPU authorization:** no GPU launch needed (CPU-only test)
- **§5.2 honest framing:** MEASURED counts traced to `find molmetal/tests
  -name 'test_wf*.py' | wc -l` and to
  `molmetal/reports/wf364a3/import_times.json`; PROJECTED count of 227
  from brief flagged as HONEST_NEGATIVE_FRAME
- **§5.3 no paper work:** no `paper/*` touched
- **§5.4 Tier-2 markers UNCHANGED:** computational-only test surface,
  no wet-lab planning

## §6 Cross-references

- **Wf-360-A2:** `molmetal/tests/test_wf296_to_wf359_integration.py`
  (parametrized batch subset, 30/30 tests)
- **Wf-364-A4:** `molmetal/tests/test_wf364a4_section6_template.py`
  (sibling Wf added concurrently — also counted in the 285 total)
- **Pre-flight v3 gate:** preserved at Wf-328-SYN baseline (62 PASS)
- **Reports:** `molmetal/reports/wf364a3/import_times.json`
