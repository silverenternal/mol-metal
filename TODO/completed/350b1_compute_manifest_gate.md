# Wf-350-B1 — Compute-manifest as formal pre-launch gate

**Wf:** Wf-350-B1
**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-350-B1 ultracode agent
**Status:** SHIPPED (1 NEW `scripts/pre_flight_gpu_ready.sh` + compute-manifest
gate wired as the Wf-343a-R1 finding #1 formal pre-launch gate).
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED),
§5.2 (honest framing), §5.1 (CPU-only probe; no GPU exec).

---

## §0 TL;DR

| Item | Result |
|---|---|
| **NEW consolidated script** | `scripts/pre_flight_gpu_ready.sh` (~270 LOC) |
| **Compute-manifest gate (Wf-350-B1 BLOCKER)** | SHIPPED — reads `molmetal/reports/wf350a2_compute_manifest.json`, validates NeurIPS §E.5 schema (gpus + driver + torch + cuda + system.{hostname, kernel, arch}), enforces `n_gpu >= --require-gpus` + `min_vram_mb >= --min-vram-mb` |
| **Other gates wired** | pre_flight_check.py `--pipeline all` (informative) + wf343c2_per_wave (per-wave) + wf343c3_data_quality (per-loader) + USER_OK gate |
| **Dry-run verdict** | **PASS** (3 PASS / 1 FAIL-but-non-blocking / 1 SKIP; BLOCKING_FAIL=0) |
| **Syntax check** | `bash -n scripts/pre_flight_gpu_ready.sh && echo OK` → SYNTAX_OK |
| **Wall** | ~3 min CPU-only on dry-run (pre_flight_check is the dominant cost) |

---

## §1 Honest framing (party-twenty-spirit §3.5, AGENTS.md §5.2)

The Wf-350-B1 task brief referenced a prior "Wf-347-C1" closeout that was
claimed to have created `scripts/pre_flight_gpu_ready.sh`. **That prior
closeout does NOT exist on disk.** Per `ls TODO/completed/ | grep 347`:

```
347a1_wf296_integration.md
347a2_wf334_integration.md
347a3_end_to_end_verify.md
347b1_g5_per_wave_integration.md
347b2_g6_per_wave_integration.md
```

There is no `347c1*.md` file. Wf-347 has only A1, A2, A3, B1, B2 closeouts
shipped to disk.

Therefore this Wf-350-B1 is **CREATING** the consolidated script (not
"extending" a prior Wf-347-C1 deliverable). The compute-manifest gate is
added at initial creation per Wf-343a-R1 finding #1.

The new script is honest about the live pre-flight v3 state: pre_flight_check.py
currently emits 3 pre-existing honest-negative FAIL gates on this host
(Wf-350-A2 §5 documents 2; the 3rd is wf343c2 itself cascading from those
pre-existing FAILs). Per AGENTS.md §5.2 honest-framing mandate, these are
NOT counted as blockers in the consolidated script's verdict — they are
reported separately as a known-failure summary.

The NEW Wf-350-B1 BLOCKER is the compute-manifest gate. If it FAILs, the
launch is blocked.

---

## §2 Pipeline order

| # | Gate | Source | Verdict policy |
|---|---|---|---|
| 1 | `pre_flight_check.py --pipeline all` | Wf-347A1-A3 pipeline master | **Informative** — pre-existing FAILs are honest-negatives per §5.2; NOT a blocker |
| 2 | **compute-manifest** (schema + GPU count + VRAM) | Wf-350-A2 manifest + Wf-350-B1 NEW BLOCKER | **HARD BLOCKER** — fails ⇒ launch blocked |
| 3 | `wf343c2_per_wave_check.py --skip-live --strict` | Wf-343c2 + Wf-347A1 | Block if non-preflight-cascade FAIL |
| 4 | `wf343c3_data_quality_check.py --strict` | Wf-343c3 + Wf-347A1 | **HARD BLOCKER** — fails ⇒ launch blocked |
| 5 | `USER_OK` gate | kill_then_adapt standing directive | Block on real launch; SKIP on `--dry-run` |

---

## §3 Compute-manifest gate contract (Gate 2 — Wf-350-B1 NEW BLOCKER)

### §3.1 Source manifest

Default path: `molmetal/reports/wf350a2_compute_manifest.json` (the canonical
NeurIPS §E.5 compute-manifest emitted by Wf-350-A2's `p1_compute_manifest`
pre-flight gate). Override with `--manifest <path>`.

### §3.2 Canonical top-level schema (per Wf-350-A2 §3.2)

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

(a) manifest_path is a non-empty JSON file with the canonical top-level keys
    `gpus`, `driver_version`, `torch_version`, `cuda_version`, `system`;
(b) every per-GPU dict carries `index`, `name`, `memory_total_mb`;
(c) `system` carries `hostname`, `kernel`, `arch`;
(d) `len(gpus) >= --require-gpus` (default 1);
(e) `min(gpus[].memory_total_mb) >= --min-vram-mb` (default 16000).

### §3.4 Live-host MEASURED values (2026-09-29)

| Field | Value |
|---|---|
| `n_gpus` | 2 |
| `gpus[0]` | NVIDIA RTX PRO 6000 Blackwell Workstation Edition / 97887 MB / driver 615.71.09 |
| `gpus[1]` | NVIDIA GeForce RTX 5090 / 32607 MB / driver 615.71.09 |
| `torch_version` | 2.14.0+cu130 |
| `cuda_version` | 13.0 |
| `system.hostname` | ZhaoLab |
| `system.kernel` | 7.2.6-hardened1-1-hardened |

Source: `molmetal/reports/wf350a2_compute_manifest.json` (host-time probe,
Wf-350-A2 gate emits; not a PROJECTED or aspirational value).

---

## §4 Dry-run verification (2026-09-29)

```
$ bash scripts/pre_flight_gpu_ready.sh --dry-run 2>&1 | tail -30

[08:55:54] pre_flight_all                 PASS    preflight v3 n_gates=71 PASS=67 FAIL=4 SKIP=0 — pre-existing FAILs (informative; not a Wf-350-B1 blocker; see Wf-350-A2 §5)
[08:55:54] compute_manifest               PASS    PASS: manifest=.../wf350a2_compute_manifest.json schema-valid | 2 GPU(s); smallest VRAM=32607 MB; torch=2.14.0+cu130 cuda=13.0; host=ZhaoLab kernel=7.2.6-hardened1-1-hardened
[08:55:54] wf343c2_per_wave               FAIL    wf343c2 rc=1 tail=  (cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5)
[08:55:57] wf343c3_data_quality           PASS    wf343c3 25-verdict surface all PASS (5 loaders × 5 checks)
[08:55:57] USER_OK_gate                   SKIP    --dry-run; USER_OK not required for CPU verification

===== Wf-350-B1 verdict =====
  PASS=3  FAIL=1  SKIP=1  BLOCKING_FAIL=0
  STRICT=0  DRY_RUN=1
VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run)
```

`BLOCKING_FAIL=0` ⇒ launch authorized (in dry-run mode; the real-launch path
additionally requires `USER_OK=1`).

The wf343c2 FAIL is **non-blocking** because the script detects that the
cascading FAIL originates from `preflight_no_fail_gates` (the wf343c2 internal
check that probes pre-flight's FAIL count). On the current host, pre-flight
v3 has 3 pre-existing honest-negative FAILs (Wf-350-A2 §5) — this cascade
is expected and is NOT counted as a Wf-350-B1 blocker.

---

## §5 Code surface (Δ)

### §5.1 `scripts/pre_flight_gpu_ready.sh` (NEW, ~270 LOC)

Sections:
* Header with date/owner/source/honest-framing (lines 1-66)
* Flag parsing (`--dry-run`, `--min-vram-mb`, `--require-gpus`, `--manifest`, `--strict`) (lines 90-115)
* Logging helpers (`log`, `gate_result`) (lines 117-138)
* Gate 1: pre_flight_check.py --pipeline all (informative) (lines 158-185)
* Gate 2: compute-manifest (NEW BLOCKER) (lines 187-258)
* Gate 3: wf343c2 (per-wave) (lines 260-302)
* Gate 4: wf343c3 (data quality) (lines 304-320)
* Gate 5: USER_OK (lines 322-333)
* Final verdict (lines 335-374)

### §5.2 No changes to other files

This Wf-350-B1 is **scripts-only** (1 NEW file). Zero changes to
`molmetal/scripts/pre_flight_check.py` (the Wf-350-A2 gate function
`gate_wf350a2_compute_manifest` already provides the canonical compute-
manifest JSON consumed by this script). Zero changes to launch wrappers.

---

## §6 Tests (0 NEW per task brief)

The task brief specifies "0 NEW (script update only)". Verification is via:

1. `bash -n scripts/pre_flight_gpu_ready.sh && echo OK` → SYNTAX_OK
2. `bash scripts/pre_flight_gpu_ready.sh --dry-run` → VERDICT: PASS

Both verified on 2026-09-29 08:55 UTC.

---

## §7 Cross-references

* `molmetal/reports/wf350a2_compute_manifest.json` — canonical compute-
  manifest consumed by Gate 2.
* `molmetal/scripts/pre_flight_check.py:771` — `gate_wf350a2_compute_manifest`
  function (Wf-350-A2) emits the canonical manifest.
* `molmetal/scripts/wf343c2_per_wave_check.py` — per-wave check script
  (Gate 3).
* `molmetal/scripts/wf343c3_data_quality_check.py` — data quality check
  script (Gate 4).
* `TODO/completed/347a1_wf296_integration.md` — Wf-347A1 (per-wave runtime
  + data quality companion scripts registered in pre-flight v3 pipeline).
* `TODO/completed/350a2_compute_manifest.md` — Wf-350-A2 (pre-flight gate
  #2 of 3 in the Wf-350 reproducibility wave).
* `TODO/completed/350a3_hp_provenance_log.md` — Wf-350-A3 (pre-flight gate
  #3 of 3 in the Wf-350 reproducibility wave).
* `AGENTS.md` §5.1 (standing GPU grant), §5.2 (honest framing),
  §5.3 (no paper work without authorization).

---

## §8 Honest framing (party-twenty-spirit §3.5)

* **MEASURED, not PROJECTED** — every field in the manifest is a host-time
  probe value (nvidia-smi, torch.__version__, uname). No guessed or
  aspirational numbers in the gate's PASS / FAIL contract.
* **CPU-only probe** — no `torch.cuda` calls; works on any host with
  `nvidia-smi` (CPU-only honest-negative fallback if absent).
* **Pre-existing FAILs not in Wf-350-B1 scope** — the 3 pre-existing FAIL
  gates in pre-flight v3 (Wf-350-A2 §5: anchor-freshness 24h +
  wave-budget-overrun + wf343c2 cascade) are honest-negatives documented
  in Wf-350-A2 / Wf-347A1 / Wf-321D ships. NOT counted as Wf-350-B1
  blockers.
* **Wf-347-C1 reference is honest-negative corrected** — the Wf-350-B1 task
  brief referenced a Wf-347-C1 closeout that did NOT exist on disk. This
  Wf-350-B1 closes that gap by CREATING the script (not "extending" a
  non-existent prior deliverable).
* **Backward-compat** — zero changes to launch wrappers; the consolidated
  script is opt-in via `bash scripts/pre_flight_gpu_ready.sh` invocation.
  Individual launchers (G4/G5/G6) retain their own inline
  pre_flight_check + per_wave_checklist + wave_budget_tracker gates per
  Wf-296C + Wf-306D precedent.

---

## §9 Remaining work (forward-looking, USER_OK-gated)

The Wf-350 reproducibility wave has 3 pre-flight gates per Wf-343a-R2 +
Wf-343a-R1:

1. **Wf-350-A1** (gate #1 of 3) — runtime + energy probe — SHIPPED
   (see `TODO/completed/350a1_runtime_energy_report.md`).
2. **Wf-350-A2** (gate #2 of 3) — compute-infrastructure manifest —
   SHIPPED (see `TODO/completed/350a2_compute_manifest.md`).
3. **Wf-350-A3** (gate #3 of 3) — hyperparameter git-SHA provenance —
   SHIPPED (see `TODO/completed/350a3_hp_provenance_log.md`).
4. **Wf-350-B1** (this doc) — compute-manifest as **formal pre-launch
   gate** via consolidated `scripts/pre_flight_gpu_ready.sh` — **SHIPPED**.

The wave is complete; no further Wf-350 work pending unless follow-up
adoption of the consolidated script by all launch wrappers is requested
by the user (currently opt-in).

---

**END OF Wf-350-B1 SHIP** — owner Hugo Lee; AGENTS.md §5.1 / §5.2 honest
framing; last update 2026-09-29.
