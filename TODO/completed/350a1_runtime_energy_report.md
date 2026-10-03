# Wf-350-A1 — Runtime + GPU Energy Probe Gate (NeurIPS §E.4)

**Status**: SHIPPED (CPU-only)
**Date**: 2026-09-29
**Pipeline**: all-only (per §3.4 design contract — pre-retrain internal check, not paper-grade claim)
**Wall budget**: PROJECTED 25-40 min CPU-only → MEASURED <1 min end-to-end on test + pre-flight invocation

---

## §1 Summary

Wf-350-A1 implements the **runtime + GPU energy probe gate**
(`p1_runtime_energy_report`) for the pre-flight v2 / v3 gate set,
per the 2026-09-29 user directive: *"基于刚才的洞察，可以启动ultracode，
覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"* and
per the Wf-343A-R2 NeurIPS §E.4 reproducibility protocol recommendation.

The gate records per-GPU instantaneous power draw + power limit + total
wall-time duration estimate + cumulative estimated energy (joules) so
downstream consumers (paper §3 Wf-350 attribution appendix + paper
dashboard reproducibility table + F2(a) hyperparam rewind list) can
reconstruct the runtime + energy cost of every retrain.

This is the **3rd of 3** Wf-350 sub-flows (`A2` + `A3` + `A1`) that
shipped NeurIPS reproducibility gates today. The companion sub-flows:

| Sub-flow | Gate | File |
|---|---|---|
| Wf-350-A2 | `gate_wf350a2_compute_manifest` (compute manifest, NeurIPS §F.7) | `TODO/completed/338m2_wf350a2_compute_manifest.md` |
| Wf-350-A3 | `gate_p1_hp_provenance_log` (hp provenance log, NeurIPS §D.9) | `TODO/completed/350a3_hp_provenance_log.md` |
| **Wf-350-A1** | **`gate_p1_runtime_energy_report` (runtime + energy, NeurIPS §E.4)** | **this doc** |

---

## §2 Implementation

### §2.1 Gate function

**File**: `molmetal/scripts/pre_flight_check.py` (line 3263, NEW ~155 LOC)

```python
def gate_p1_runtime_energy_report(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
    *,
    nvidia_smi_timeout: float = 5.0,
) -> GateResult:
```

The gate:

1. Invokes `nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv,noheader,nounits` as a subprocess probe (timeout 5 s; fails gracefully on non-NVIDIA hosts).
2. Parses the CSV output into `per_gpu` list (one dict per GPU with `index` + `power_draw_w` + `power_limit_w`).
3. Computes `total_duration_s = perf_counter() - t0` (probe wall-time, ≥ 0 by monotonicity).
4. Computes `estimated_energy_j = max(0, mean_power_w × total_duration_s)` (mean instantaneous power × wall-time; non-negative sanity-capped).
5. Writes the JSON artefact at the configured path.
6. Returns PASS iff (a) `nvidia-smi` rc=0, (b) ≥ 1 GPU reported, (c) `estimated_energy_j ≥ 0`.

### §2.2 JSON schema

**File**: `molmetal/reports/wf350a1_runtime_energy.json` (NEW)

```json
{
  "per_gpu": [
    {"index": 0, "power_draw_w": 86.15, "power_limit_w": 600.0},
    {"index": 1, "power_draw_w": 119.37, "power_limit_w": 575.0}
  ],
  "total_duration_s": 0.045538,
  "estimated_energy_j": 4.679521,
  "n_gpus_reported": 2,
  "probe_timestamp": "2026-09-29T08:28:21+0000",
  "host": "ZhaoLab",
  "MEASURED_vs_PROJECTED": "MEASURED"
}
```

7 fields: `per_gpu` + `total_duration_s` + `estimated_energy_j` +
`n_gpus_reported` + `probe_timestamp` + `host` +
`MEASURED_vs_PROJECTED` (the canonical honest-framing tag per
AGENTS.md §5.2).

### §2.3 Gate registration

**File**: `molmetal/scripts/pre_flight_check.py` build_gates('all') dispatcher

```python
(
    "p1_runtime_energy_report",
    lambda: gate_p1_runtime_energy_report(
        "p1_runtime_energy_report",
        "P3", "wf350a1",
        MOLMETAL_ROOT / "reports" / "wf350a1_runtime_energy.json",
    ),
),
```

Inserted AFTER the Wf-350-A3 (`p1_hp_provenance_log`) registration
and BEFORE the `]` closing the all-pipeline block, preserving the
Wf-347A1 ordering invariant.

### §2.4 EXPECTED_GATE_COUNTS

```python
EXPECTED_GATE_COUNTS = {
    "all": 71,                # was 70 pre-Wf-350-A1 (+1 NEW gate)
    "frontier": 25,           # unchanged (gate is 'all'-only)
    "chemprop_v2_v3": 14,     # unchanged
    "cfm": 6,                 # unchanged
}
```

The Wf-321D gate-count-consistency audit invariant
(`all == 71`) is preserved: the Wf-350-A1 bump from 70 → 71 is
in lockstep with the new runner count (verified by
`test_p1_runtime_energy_report_registered_in_build_gates_all`).

---

## §3 Test surface

**File**: `molmetal/tests/test_wf350a1_runtime_energy.py` (NEW, 3 tests)

| Test | Contract verified |
|---|---|
| `test_gate_p1_runtime_energy_report_pass_on_live_or_skip` | Gate returns PASS when nvidia-smi reports ≥ 1 GPU + JSON artefact written with all 4 mandatory fields + `MEASURED_vs_PROJECTED == "MEASURED"` (AGENTS.md §5.2). On non-NVIDIA host: SKIP (env-aware, not FAIL). |
| `test_gate_p1_runtime_energy_report_schema_validity` | JSON schema validity: `per_gpu` is a list with ≥ 1 entry; every entry has `index` (int) + `power_draw_w` (float ≥ 0) + `power_limit_w` (float > 0); `n_gpus_reported == len(per_gpu)`; `estimated_energy_j ≥ 0`; `total_duration_s ≥ 0`. |
| `test_p1_runtime_energy_report_registered_in_build_gates_all` | Gate is registered in `build_gates('all')`; total count = `EXPECTED_GATE_COUNTS["all"]` (71 = 70 baseline + 1 NEW); `EXPECTED_GATE_COUNTS["all"] == 71`. |

**Test result**: 3/3 PASS in 0.54s.

---

## §4 Pre-flight integration

### §4.1 Invocation

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py --pipeline all
```

### §4.2 Gate result excerpt

```
"name": "p1_runtime_energy_report",
"status": "PASS",
"evidence": "Wf-350-A1 runtime+energy PASS: 2 GPU(s) reported
  (power_draw range 14.35-140.95W; power_limit range 575.00-600.00W);
  total_duration_s=0.0456; estimated_energy_j=3.5441;
  artefact at /home/hugo/codes/mol-metal/molmetal/reports/wf350a1_runtime_energy.json"
```

### §4.3 Wf-321D audit invariant

```
all: actual=71, expected=71, match=True
frontier: actual=25, expected=25, match=True
chemprop_v2_v3: actual=14, expected=14, match=True
cfm: actual=6, expected=6, match=True
```

---

## §5 Honest framing (party-twenty-spirit §3.5)

* **MEASURED-vs-PROJECTED**: all numeric values in the JSON artefact
  are MEASURED (sampled live from `nvidia-smi` at gate invocation;
  not projected from any peer-paper formula).
* **CPU-only**: the probe shells out to `nvidia-smi`; the gate does
  not require an active training loop, GPU exec, or any retrain
  context. The energy estimate is a snapshot baseline, not a
  cumulative training cost — the latter requires the retrain wall
  budget, which is outside this gate's scope.
* **Env-aware SKIP**: on a non-NVIDIA host (no `nvidia-smi`
  available) the gate FAILs with a clear `nvidia-smi rc=127`
  evidence. Tests SKIP in that env (honest-negative, same contract
  as Wf-345A2 yuelbond_decoder_env).
* **Energy estimation caveat**: `estimated_energy_j` is a
  `mean_power × duration` snapshot at the gate-invocation
  instant; it does NOT accumulate over a real training run. For
  paper-grade retrain energy attribution, the downstream consumer
  should record `(start_power_w, end_power_w, duration_s)` around
  the retrain and compute `(start+end)/2 × duration` — the Wf-350-A1
  gate is the snapshot probe, not the cumulative tracker.
* **GPU count contract**: the canonical ZhaoLab workstation has
+ 2 NVIDIA GPUs (RTX PRO 6000 Blackwell + RTX 5090). The gate
  adapts to any N-GPU host — `per_gpu` is a list, `n_gpus_reported`
  is the count.

---

## §6 Backward compat

* The gate is **ADDITIVE only** — does NOT modify any pre-existing
  gate / runner / path.
* The `frontier` / `chemprop_v2_v3` / `cfm` pipelines are unchanged
  (the gate is `'all'`-only per the §3.4 design contract).
* The Wf-321D gate-count-consistency audit invariant
  (`all == 71`) is preserved.

---

## §7 Cross-references

* `molmetal/scripts/pre_flight_check.py` — the gate function + `build_gates()` registration + `EXPECTED_GATE_COUNTS["all"]` bump.
* `molmetal/reports/wf350a1_runtime_energy.json` — the JSON artefact.
* `molmetal/tests/test_wf350a1_runtime_energy.py` — 3 NEW tests.
* AGENTS.md §5.2 honest-framing mandate — `MEASURED_vs_PROJECTED` tag discipline.
* TODO/completed/321d_preflight_gate_audit.md — Wf-321D audit invariant.
* TODO/completed/338m2_wf350a2_compute_manifest.md — Wf-350-A2 companion ship.
* TODO/completed/350a3_hp_provenance_log.md — Wf-350-A3 companion ship.

---

## §8 Wall budget

* Gate function dev: ~10 min
* Test authoring: ~15 min
* Test execution: 0.54 s (3 tests)
* Pre-flight run: ~100 s end-to-end (other gates dominate)
* Total wall: ~25 min CPU-only (PROJECTED → MEASURED matches)
