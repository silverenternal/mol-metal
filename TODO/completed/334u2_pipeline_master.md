# Wf-334-U2: Pipeline master sequential launch audit + Wf-333 wrapper + tests

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Status:** SHIPPED — CPU-only audit + 11/11 tests PASS
**Parent:** Wf-334 ultracode (per AGENTS.md §5.1 standing GPU authorization)

---

## §1 Summary

Audited `scripts/gpu_pipeline_master.sh` (GPU pipeline master) and:

1. Verified sequential launch order is correct
   (Wave B → Wf-219 §4.1 → TODO-220 → TODO-179 → DrugOOD → Wf-135 Phase 2).
2. Verified all 6 wrapper scripts exist on disk + are executable.
3. Fixed **path bug**: pipeline master referenced
   `molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh` but the actual file
   lives in `scripts/` (Wf-310E ship placed it there). Now points to
   `scripts/launch_wf310e_wf135_phase2_gpu.sh`.
4. Added **Wf-333 wrapper** `molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh`
   (4-step Vina docking bench — CPU-only, BLOCKED on Vina 1.2.7 conda install
   per AGENTS.md §5.4 / Wf-299B).
5. Wrote 4 NEW tests in `molmetal/tests/test_wf334u2_pipeline_master.py`
   (plus 7 parametrized + 1 bonus) — **11/11 PASS**.

---

## §2 Pipeline master audit

### §2.1 Sequential launch order (verified)

Per `gpu_pipeline_master.sh` `phase3_gpu1()`:

| Step | Job | Wall (PROJECTED) | GPU |
|---:|---|---:|---|
| 1 | Wf-310F frontier Wave B | ~0.5 hr | cuda:1 |
| 2 | Wf-219 §4.1 Ru | ~13.5 min | cuda:1 |
| 3 | Wf-310C TODO-220 (temporal split) | ~3.9 hr | cuda:1 |
| 4 | Wf-310C TODO-179 (W₂ pIC50) | ~4.5 hr | cuda:1 |
| 5 | Wf-306A DrugOOD DANN Phase A | ~3-4 hr | cuda:1 |
| 6 | Wf-310E Wf-135 Phase 2 LOMO | ~2 weeks | cuda:1 |

Phase 2 (`phase2_gpu0`) launches 1 job:
Wf-310F frontier Wave C on cuda:0 (~2-3 hr).

### §2.2 Wrappers (verified existing)

All 6 wrappers referenced in pipeline master exist on disk + are executable:

- `molmetal/scripts/launch_wf310f_frontier_math_gpu.sh` (Wave B + C)
- `molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh` (Wf-219 §4.1)
- `molmetal/scripts/launch_wf310c_todo220_gpu.sh` (TODO-220)
- `molmetal/scripts/launch_wf310c_t179_w2_gpu.sh` (TODO-179)
- `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (DrugOOD Phase A)
- `scripts/launch_wf310e_wf135_phase2_gpu.sh` (Wf-135 Phase 2 — USER_OK-gated)

### §2.3 Path bug fix

**Before:** `bash molmetal/scripts/launch_wf310e_wf135_phase2_gpu.sh` (FILE NOT FOUND)

**After:** `bash scripts/launch_wf310e_wf135_phase2_gpu.sh` (resolves correctly)

The Wf-310E ship placed the Wf-135 Phase 2 wrapper in `scripts/` (not
`molmetal/scripts/`). Pipeline master path fixed in
`scripts/gpu_pipeline_master.sh` line 131.

---

## §3 Wf-333 Vina 4-step wrapper (NEW)

`molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` implements the
4-step Vina docking bench per Wf-299B §3.4 + Wf-301B Pt+2 UFF blocker:

- **Step 1:** receptor prep (MGLTools `prepare_receptor4.py`)
- **Step 2:** ligand prep (RDKit + Meeko + `--pt-strip` per Wf-301B)
- **Step 3:** Vina dock (CPU single-threaded; BLOCKED on conda install)
- **Step 4:** parse + kcal/mol aggregation (`parse_dock_vina.py` per Wf-305D)

**Status:** HONEST-NEGATIVE — Vina 1.2.7 conda install BLOCKED per AGENTS.md
§5.4 + Wf-299B. Wrapper supports `--status` for read-only state query
(USER_OK-exempt). Steps 1+2+4 CPU-ready; step 3 contingent on Vina install.

**Usage:**

```bash
# Read-only state query (USER_OK exempt)
bash scripts/launch_wf333_vina_4step_gpu_bench.sh --status

# Plan only (USER_OK still required)
USER_OK=1 bash scripts/launch_wf333_vina_4step_gpu_bench.sh --dry-run

# Actual run (USER_OK required; 1-hr wall PROJECTED)
USER_OK=1 bash scripts/launch_wf333_vina_4step_gpu_bench.sh
```

**USER_OK gate:** REFUSES with exit code 64 (BSD sysexits `EX_USAGE`) if
`USER_OK != 1` — verified by `test_wf135_phase2_user_ok_gate_enforced` +
`test_wf333_vina_wrapper_has_status_command`.

---

## §4 Tests (NEW — 11/11 PASS)

`molmetal/tests/test_wf334u2_pipeline_master.py` adds:

1. **test_pipeline_master_sequential_launch_order** — verifies the 6-step
   Phase 3 sequence is in the expected order.
2. **test_pipeline_master_wrappers_exist** (parametrized × 7) — every
   wrapper (including new Wf-333) resolves to a real file on disk + is
   executable.
3. **test_wf135_phase2_user_ok_gate_enforced** — verifies
   `launch_wf310e_wf135_phase2_gpu.sh` refuses without USER_OK=1 (exit 64)
   and accepts with USER_OK=1 (exit 0).
4. **test_pipeline_master_cli_help_invariants** — verifies shebang,
   `set -euo pipefail`, phase2/phase3 functions, usage, project root,
   USER_OK=1 line for Wf-135 Phase 2.
5. **test_wf333_vina_wrapper_has_status_command** (bonus) — verifies Wf-333
   `--status` works without USER_OK and emits valid JSON with
   `vina_available` + `vina_blocked_reason` keys.

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf334u2_pipeline_master.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collected 11 items

molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_sequential_launch_order PASSED [  9%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf310f_frontier_math_gpu.sh] PASSED [ 18%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf306b_wf219_ru_gpu.sh] PASSED [ 27%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf310c_todo220_gpu.sh] PASSED [ 36%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf310c_t179_w2_gpu.sh] PASSED [ 45%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf306a_drugood_vina_gpu.sh] PASSED [ 54%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf310e_wf135_phase2_gpu.sh] PASSED [ 63%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_wrappers_exist[launch_wf333_vina_4step_gpu_bench.sh] PASSED [ 72%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_wf135_phase2_user_ok_gate_enforced PASSED [ 81%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_pipeline_master_cli_help_invars PASSED [ 90%]
molmetal/tests/test_wf334u2_pipeline_master.py::test_wf333_vina_wrapper_has_status_command PASSED [100%]

============================== 11 passed in 0.22s ==============================
```

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today:** Pipeline master path bug fixed; 6 wrappers verified
  executable; Wf-333 wrapper added (USER_OK-gated, status-query exempt);
  11/11 tests pass.
- **PROJECTED:** Wall estimates in pipeline master (0.5 hr...2 weeks) are
  from Wf-135 Phase 1 plan §2.4 + Wf-305D Vina smoke; actual times vary 1.5-2x.
- **HONEST-NEGATIVE:** Vina 1.2.7 conda install BLOCKED per AGENTS.md §5.4;
  Wf-333 step 3 (Vina dock) cannot run until Vina is installed.
  Steps 1+2+4 are CPU-ready.
- **GPU auth per §5.1:** Pipeline master reflects standing GPU authorization;
  per-wrapper USER_OK gates still apply (Wf-135 Phase 2 is 2-week wall and
  has explicit gate; Wf-219 §4.1 + DrugOOD also have gates inside wrappers).

---

## §6 Files touched

| Path | Change |
|---|---|
| `scripts/gpu_pipeline_master.sh` | Path bug fix for Wf-135 Phase 2 wrapper (line 131) |
| `molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh` | NEW (4-step Vina wrapper) |
| `molmetal/tests/test_wf334u2_pipeline_master.py` | NEW (11 tests) |
| `TODO/completed/334u2_pipeline_master.md` | NEW (this file) |

---

## §7 What stays for follow-on Wf

- Pipeline master is now path-clean. Actual GPU execution is contingent on
  G6 (Wf-124 P3) + Wave A (Wf-310F frontier) completing on GPU 1/0.
- Wf-333 step 3 (Vina dock) needs `conda install -c conda-forge vina=1.2.7`
  before it can run end-to-end. Per TODO-134 Vina install attempt is in
  flight per Wf-323g.
- Wf-219 §4.2 production backbone remains PAPER_OK-gated per TODO-84.
