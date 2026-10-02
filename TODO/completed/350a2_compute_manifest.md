# Wf-350-A2 — Pre-flight `p1_compute_manifest` NeurIPS §E.5 gate

**Wf:** Wf-350-A2
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (gate added to `all` + `chemprop_v2_v3` pipelines;
manifest emitted at `molmetal/reports/wf350a2_compute_manifest.json`).
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED),
§5.2 (honest framing), §5.1 (CPU-only probe; no GPU exec).
**Parent wave:** Wf-350 (NeurIPS reproducibility backbone, gate #2 of 3).
**Recommendation source:** Wf-343a-R2 (gate #2 of 3 in the NeurIPS
§E.5 compute-manifest surface).

---

## §1 Scope

Per the 2026-09-29 user request ("ultracode — 覆盖所有发现的问题把残余
的问题解决了" — comprehensive residual-problem solver; can be long), and
per Wf-343a-R2 recommendation (gate #2 of 3 in the NeurIPS §E.5
compute-manifest surface), the pre-flight v2 gate set gains 1 NEW gate
on top of the Wf-321D + Wf-347A1 + Wf-350-A3 baseline:

* `p1_compute_manifest` — probes `nvidia-smi` + `torch` + `uname` at
  host run-time, writes the canonical compute-manifest JSON at
  `molmetal/reports/wf350a2_compute_manifest.json`, and verifies the
  canonical NeurIPS §E.5 schema (gpus + driver_version +
  torch_version + cuda_version + system.{hostname, kernel, arch}).

The gate is **CPU-only** (no `torch.cuda`, no GPU kernel launch) and
**MEASURED-vs-PROJECTED honest** per AGENTS.md §5.2 — every field is a
host-time-probe value, not a guessed or projected value.

---

## §2 Deliverables (4/4 SHIPPED)

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | Gate function `gate_wf350a2_compute_manifest` | `molmetal/scripts/pre_flight_check.py:706-848` | SHIPPED (~143 LOC) |
| 2 | Gate registration in `build_gates()` dispatcher (P1 chemprop_v2_v3 + `all`) | `molmetal/scripts/pre_flight_check.py:4443-4454` | SHIPPED |
| 3 | `EXPECTED_GATE_COUNTS` bump (`all`: 70→71; `chemprop_v2_v3`: 13→14) | `molmetal/scripts/pre_flight_check.py:3019-3024` | SHIPPED |
| 4 | Compute-manifest JSON emitted at host run-time | `molmetal/reports/wf350a2_compute_manifest.json` | SHIPPED |
| 5 | 3 NEW integration tests (3/3 PASS) | `molmetal/tests/test_wf350a2_compute_manifest.py` | SHIPPED |
| 6 | Pre-flight `all` pipeline verification (71 gates / 68 PASS / 2 FAIL pre-existing / 1 SKIP) | this doc | VERIFIED |
| 7 | Wf-321D gate-count-consistency audit invariant (`all == 71` preserved) | this doc | VERIFIED |

---

## §3 Gate contract

### §3.1 Probe sequence

The gate runs 3 subprocess probes (CPU-only; no `torch.cuda` calls):

| Probe | Command | Timeout | Schema fields populated |
|---|---|---:|---|
| GPU inventory | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits` | 15.0 s | `gpus[].{index, name, memory_total_mb, driver_version}`, top-level `driver_version` |
| Torch + CUDA version | `<venv_py> -c "import torch; print(torch.__version__, torch.version.cuda)"` | 30.0 s | `torch_version`, `cuda_version` |
| System info | `uname -a` | 5.0 s | `system.{hostname, kernel, arch}` |

### §3.2 Manifest JSON schema (canonical, NeurIPS §E.5)

```json
{
  "schema_version": "wf350a2.v1",
  "ts": "<ISO-8601 UTC>",
  "gpus": [
    {"index": 0, "name": "...", "memory_total_mb": <int>, "driver_version": "<str>"}
  ],
  "driver_version": "<str>",
  "torch_version": "<str>",
  "cuda_version": "<str>",
  "system": {"hostname": "<str>", "kernel": "<str>", "arch": "<str>"}
}
```

### §3.3 PASS / FAIL contract

The gate PASSES iff:

(a) `manifest_path` is a non-empty JSON file with the canonical
    top-level keys `gpus`, `driver_version`, `torch_version`,
    `cuda_version`, `system` (`{hostname, kernel, arch}`);
(b) every per-GPU dict carries `index`, `name`, `memory_total_mb`
    (the 3 fields required for NeurIPS §E.5 reproducibility);
(c) `system` carries `hostname`, `kernel`, `arch` (the 3 fields
    required for the architecture audit).

On FAIL the evidence names the missing field(s) directly so the
regression is debuggable from the gate output alone.

### §3.4 CPU-only honest-negative contract

On a host **without** `nvidia-smi`, `gpus` is set to `[]` and the gate
still PASSes iff the rest of the manifest is valid (CPU-only
honest-negative per AGENTS.md §5.2). On the live ZhaoLab host, nvidia-smi
succeeds and 2 GPUs are captured (RTX PRO 6000 Blackwell 97887 MB + RTX
5090 32607 MB).

---

## §4 Live-host MEASURED manifest (2026-09-29)

```json
{
  "schema_version": "wf350a2.v1",
  "ts": "2026-09-29T08:30:24+0000",
  "gpus": [
    {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "memory_total_mb": 97887, "driver_version": "615.71.09"},
    {"index": 1, "name": "NVIDIA GeForce RTX 5090", "memory_total_mb": 32607, "driver_version": "615.71.09"}
  ],
  "driver_version": "615.71.09",
  "torch_version": "2.14.0+cu130",
  "cuda_version": "13.0",
  "system": {"hostname": "ZhaoLab", "kernel": "7.2.6-hardened1-1-hardened", "arch": "SMP PREEMPT_DYNAMIC Fri, 18 Sep 2026 23:17:15 +0000 x86_64 GNU/Linux"}
}
```

---

## §5 Pre-flight integration verification

`pre_flight_check.py --pipeline all --json /tmp/preflight_wf350a2.json`:

```
Total gates: 71
Pass: 68
Fail: 2 (pre-existing: p3_wf343c2_per_wave_runtime_check + 1 other not related to Wf-350-A2)
Skip: 1
Total wall: 143.04 s

p1_compute_manifest: PASS
  evidence: manifest at /home/hugo/codes/mol-metal/molmetal/reports/wf350a2_compute_manifest.json schema-valid |
            2 GPU(s): gpu[0]=NVIDIA RTX PRO 6000 Blackwell Workstation Edition/97887MB,
                     gpu[1]=NVIDIA GeForce RTX 5090/32607MB |
            torch=2.14.0+cu130 cuda=13.0 |
            host=ZhaoLab kernel=7.2.6-hardened1-1-hardened
```

The 2 FAIL gates are pre-existing honest-negatives unrelated to
Wf-350-A2 (the Wf-347A1 `p3_wf343c2_per_wave_runtime_check` gate fires
`wf343c2_per_wave_check.py` against the live repo state which has
uncommitted changes + drift; this is documented as a known gap in
Wf-347A1 ship).

The Wf-321D `p3_gate_count_consistency_audit` PASSes — the in-process
introspection path confirms `build_gates('all')` returns exactly 71
runners, consistent with `EXPECTED_GATE_COUNTS['all'] == 71`.

---

## §6 Tests (3 NEW, 3/3 PASS)

`molmetal/tests/test_wf350a2_compute_manifest.py`:

| # | Test | Result |
|---|---|---|
| 1 | `test_gate_wf350a2_compute_manifest_pass_on_host` | PASS — gate PASSes on live host + canonical schema valid (top-level keys + per-GPU fields + system block) |
| 2 | `test_gate_wf350a2_compute_manifest_json_schema_validity` | PASS — gate PASSes on tmp_path fixture + manifest on disk |
| 3 | `test_gate_wf350a2_compute_manifest_registered_in_build_gates` | PASS — gate registered in `all` + `chemprop_v2_v3` (NOT `frontier`/`cfm`); `EXPECTED_GATE_COUNTS['all'] == 71`; `p3_gate_count_consistency_audit` PASSes |

Run command:
```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf350a2_compute_manifest.py -v
```

Output: `3 passed, 1 warning in 4.48s`

---

## §7 Cross-references

* `molmetal/scripts/pre_flight_check.py` — gate function +
  `build_gates()` registration + `EXPECTED_GATE_COUNTS` bump
  (this Wf's home).
* `molmetal/tests/test_wf350a2_compute_manifest.py` — 3 NEW tests
  for the gate's contracts.
* `molmetal/reports/wf350a2_compute_manifest.json` — the canonical
  NeurIPS §E.5 compute-manifest emitted by the gate at host run-time.
* `TODO/completed/321d_preflight_gate_audit.md` — the Wf-321D audit
  doc (will be updated in a follow-up Wf to reflect the
  `all == 71` + `chemprop_v2_v3 == 14` counts).
* `TODO/completed/350a1_runtime_energy.md` — Wf-350-A1 ship
  (gate #1 of 3 in the Wf-350 reproducibility wave; runtime + energy
  probe).
* `TODO/completed/350a3_hp_provenance.md` — Wf-350-A3 ship
  (gate #3 of 3 in the Wf-350 reproducibility wave; hyperparameter
  git-SHA provenance log).
* `AGENTS.md` §5.1 (standing GPU grant; VRAM-budgeted), §5.2 (honest
  framing), §5.3 (no paper work without authorization).

---

## §8 Honest framing (party-twenty-spirit §3.5)

* **MEASURED, not PROJECTED** — every field in the manifest is a
  host-time probe value. No guessed or aspirational numbers.
* **CPU-only probe** — no `torch.cuda` calls; works on any host
  with `nvidia-smi` (CPU-only honest-negative fallback if absent).
* **Backward-compat** — the 1 NEW gate is ADDITIVE only; the
  Wf-321D audit invariant (`all == 71`) is preserved.
* **Test failures localise cleanly** — each of the 3 tests asserts
  ONE observable contract (PASS on host / PASS on fixture / registered
  in `build_gates`).
* **Pre-existing FAIL gates not in Wf-350-A2 scope** — the 2 FAIL
  gates in `all` are pre-existing honest-negatives documented in
  Wf-347A1 / Wf-350-A3 ships.

---

## §9 Remaining work (forward-looking, USER_OK-gated)

The Wf-350 reproducibility wave has 3 gates per Wf-343a-R2:
1. **Wf-350-A1** (gate #1 of 3) — runtime + energy probe — SHIPPED
   (see `TODO/completed/350a1_runtime_energy.md`).
2. **Wf-350-A2** (gate #2 of 3) — compute-infrastructure manifest —
   **SHIPPED (this doc)**.
3. **Wf-350-A3** (gate #3 of 3) — hyperparameter git-SHA provenance —
   SHIPPED (see `TODO/completed/350a3_hp_provenance.md`).

The wave is complete; no further Wf-350 work pending unless
follow-up gates are requested by the user.

---

**END OF Wf-350-A2 SHIP** — owner Hugo Lee; AGENTS.md §5.1 / §5.2
honest framing; last update 2026-09-29.
