# Wf-350-C1 — GPU/CUDA env probe gate (wraps check_blackwell_env.py)

**Ship date:** 2026-09-29
**Workflow:** Wf-350-C1 (1 closeout)
**Trigger:** Wf-343a-R3 finding #1 ("GPU/CUDA env probe should be wired into pre-flight")
**Honest framing:** Per `party-twenty-spirit` §3.5; every numeric value below traces to
the JSON artefact emitted by the probe at host run-time (MEASURED, not PROJECTED).

---

## §1 Task summary

Wire the canonical R18 Blackwell GPU/CUDA environment probe at
`molmetal/scripts/check_blackwell_env.py` into the pre-flight v2 gate
set as a NEW gate `p1_gpu_env_probe`. The gate:

1. Invokes `check_blackwell_env.py` as a subprocess (writes JSON to
   `molmetal/reports/wf350c1_gpu_env_check.json`).
2. Re-reads the JSON artefact + asserts the canonical host-runtime
   schema is valid.
3. PASS iff `torch.cuda.is_available == True` AND
   `torch.cuda.device_count >= 1` AND `gpu_names` non-empty.
4. FAIL otherwise with a precise diagnostic.

The 4 mandatory host-runtime fields (per AGENTS.md §5.1 standing GPU
grant — "显存够就放心大胆用"):

  1. `torch.cuda.is_available` (via `torch_probe.ok`)
  2. `torch.cuda.device_count` (via `torch_probe.device_count`)
  3. GPU model (via `gpu_names` + per-GPU `gpu_details[*].name`)
  4. CUDA runtime version (via `torch_probe.cuda_runtime`)

---

## §2 Per-deliverable MEASURED status

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | New gate function `gate_p1_gpu_env_probe` in `molmetal/scripts/pre_flight_check.py` | SHIPPED | New function (240 LOC including docstring) inserted after `gate_p1_runtime_energy_report` (line 3553) |
| 2 | Gate wired into `build_gates("all")` | SHIPPED | `p1_gpu_env_probe` runner registered after `p1_runtime_energy_report` in `build_gates("all")` |
| 3 | `EXPECTED_GATE_COUNTS["all"]` bump from 93 → 94 | SHIPPED | Line 3126 of `pre_flight_check.py` (with Wf-353-B1 + Wf-350-C1 attribution comment) |
| 4 | Gate-count docstring updated to mention Wf-350-C1 | SHIPPED | Line 3073 — adds 1 NEW Wf-350-C1 row in the gate provenance trail |
| 5 | 3 NEW tests in `molmetal/tests/test_wf350c1_gpu_env_probe.py` | SHIPPED | 3/3 PASS in 5.09s |
| 6 | TODO closeout doc at `TODO/completed/350c1_gpu_env_probe.md` | SHIPPED | This document |

---

## §3 Per-test MEASURED status

3 NEW tests in `molmetal/tests/test_wf350c1_gpu_env_probe.py` (5.09 s wall):

| # | Test | Status | Wall |
|---|---|---|---|
| 1 | `test_gate_wf350c1_gpu_env_probe_pass_on_host` | PASS | < 2 s |
| 2 | `test_gate_wf350c1_gpu_env_probe_json_schema_validity` (tmp_path fixture) | PASS | < 2 s |
| 3 | `test_gate_wf350c1_gpu_env_probe_registered_in_build_gates` | PASS | < 1 s |

All 3 tests assert ONE observable contract (registry membership /
PASS-on-fixture / JSON schema validity) so failures localise cleanly.

---

## §4 Pre-flight gate-count delta

| Pipeline | Pre-Wf-350-C1 | Post-Wf-350-C1 | Delta |
|---|---:|---:|---:|
| `all` | 93 | **94** | +1 |
| `frontier` | 25 | 25 | 0 |
| `chemprop_v2_v3` | 14 | 14 | 0 |
| `cfm` | 6 | 6 | 0 |

The 1 NEW gate is `all`-only per the §3.4 design contract
(pre-retrain internal check, NOT a paper-rule parity gate). The
Wf-321D `p3_gate_count_consistency_audit` in-process gate PASSes
with evidence `actual=94 == expected=94 (auto-computed via Wf-345-B3)`.

---

## §5 Test deltas + backward-compat verdict

3 NEW tests added (3/3 PASS, 0 FAIL, 0 SKIP).

Two pre-existing tests had stale hardcoded `== 71` and `== 93`
assertions that hard-froze the count at their ship-time values. Both
were relaxed to `>= floor` so the assertions stay green through later
ship waves that bump the count. The canonical exact-match invariant
is owned by `test_wf352a1_expected_gate_counts.py` (which uses
`build_gates("all")` dynamically):

- `molmetal/tests/test_wf350a1_runtime_energy.py` — `== 71` → `>= 71`
- `molmetal/tests/test_wf350a2_compute_manifest.py` — `== 71` → `>= 71`
  (chemprop_v2_v3: `== 14` → `>= 14`)

`test_wf350a3_hp_provenance.py` and `test_wf353b1_wf343c1_gates_wired.py`
already used `>= floor` assertions (>= 69 and >= 93 respectively) so no
edit was needed.

All 18 tests across the 6 affected test files PASS in 69.09 s wall.

---

## §6 Live-host MEASURED values (2026-09-29, 2-GPU Blackwell + RTX 5090)

The new gate PASSes on the dev box at `ZhaoLab` with the following
MEASURED host-runtime snapshot (from
`molmetal/reports/wf350c1_gpu_env_check.json`):

| Field | MEASURED value |
|---|---|
| `torch.cuda.is_available` | **True** |
| `torch.cuda.device_count` | **2** |
| Primary GPU model | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| CUDA runtime | 13.0 |
| `sm_120_blackwell` | True |
| `torch.__version__` | 2.14.0+cu130 |
| `gpu_count` (nvidia-smi) | 2 |
| `gpu_names` | ['NVIDIA RTX PRO 6000 Blackwell Workstation Edition', 'NVIDIA GeForce RTX 5090'] |

---

## §7 Honest framing (party-twenty-spirit §3.5)

* **MEASURED** gate function (not PROJECTED) — every PASS/FAIL
  branch traces to a live subprocess probe + JSON parse.
* **CPU-only** — the gate does NOT touch `torch.cuda` directly; it
  shells out to `check_blackwell_env.py` which itself probes nvidia-smi
  + 3 candidate venvs.
* **Backward-compat** — no pre-existing gate / runner / path was
  modified; only 2 stale hardcoded `== floor` assertions were relaxed
  to `>= floor` to accommodate the additive count bump.
* **Wall-budget** — gate runs in < 2 s on a 2-GPU host; test suite
  in 5.09 s; full pre-flight check (93 → 94 gates) unchanged.

---

## §8 Cross-references

* `molmetal/scripts/pre_flight_check.py` — `gate_p1_gpu_env_probe`
  function (line 3553 onward) + `build_gates()` registration (after
  `p1_runtime_energy_report`) + `EXPECTED_GATE_COUNTS["all"] = 94`.
* `molmetal/scripts/check_blackwell_env.py` — the canonical R18
  Blackwell env probe (186 LOC) that the gate wraps.
* `molmetal/reports/wf350c1_gpu_env_check.json` — the canonical env
  JSON artefact written by the gate at host run-time.
* `molmetal/tests/test_wf350c1_gpu_env_probe.py` — 3 NEW tests.
* `molmetal/tests/test_wf352a1_expected_gate_counts.py` — canonical
  exact-match invariant (auto-updated by the gate-count audit).
* `TODO/INDEX.md` — to be updated by `TODO/INDEX.md` final sync
  wave (Wf-332-style) — pending, not in scope of this closeout.
* `AGENTS.md` §6 — recent Wf history pointer (to be appended by the
  next AGENTS.md final-sync).

---

## §10 Lessons for next time

* The `== floor` pattern in pre-flight test files is brittle when
  the gate count keeps growing. Prefer `>= floor` assertions in
  per-workflow tests; let `test_wf352a1_expected_gate_counts.py`
  own the exact-match invariant.
* When wiring a new pre-flight gate that wraps an existing probe
  script, model the gate after `gate_p1_runtime_energy_report`
  (subprocess + JSON artefact + re-read + schema-validity check)
  rather than reimplementing the probe logic in-process. This
  keeps the gate CPU-only AND preserves the probe's own
  evolution surface.

---

**END OF Wf-350-C1 SHIP DOC** — Owner Hugo Lee;
cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.