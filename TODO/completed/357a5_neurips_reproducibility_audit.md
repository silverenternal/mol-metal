# Wf-357-A5: NeurIPS reproducibility gates audit (Wf-350 3-NEW gate re-verify)

**Date:** 2026-09-29 20:07 UTC
**Wall:** PROJECTED 5-10 min; ACTUAL ~3 min (CPU-only)
**Owner:** Hugo Lee
**Status:** SHIPPED — verification only (0 NEW tests, 0 GPU launches)
**Honest framing per `party-twenty-spirit` §3.5:** all numbers below are MEASURED live from
JSON / JSONL artefacts on disk; the only PROJECTED figure is wall-budget (5-10 min CPU).

## §1 Scope

Wf-357-A5 re-verifies the 3 NEW NeurIPS reproducibility gates that landed in Wf-350 (the
NeurIPS §D.9 / §E.4 / §E.5 protocol bundle added to the pre-flight check). Each gate is a
pre-retrain internal check (NOT paper-grade per the §3.4 design contract that governs the
'P3 / wf343c4 / wf347a1 / wf350aX' series).

| Wf | Gate name | NeurIPS § | Evidence artefact | Schema |
|---|---|---|---|---|
| Wf-350-A1 | `p1_runtime_energy_report` | §E.4 (compute + energy) | `molmetal/reports/wf350a1_runtime_energy.json` | JSON, 8 mandatory top-level fields |
| Wf-350-A2 | `p1_compute_manifest` | §E.5 (compute manifest) | `molmetal/reports/wf350a2_compute_manifest.json` | JSON, `wf350a2.v1` schema |
| Wf-350-A3 | `p1_hp_provenance_log` | §D.9 (HP provenance) | `molmetal/reports/wf350a3_hp_provenance.jsonl` | JSONL, 1 line per config |

## §2 Pipeline run (chemprop_v2_v3)

Pre-flight was invoked via:

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 \
  2>&1 | tee molmetal/reports/wf357a5_neurips_gates.txt
```

Summary: **14 PASS / 0 FAIL / 0 SKIP / 14 TOTAL** (wall 72.7 s on chemprop_v2_v3
pipeline subset; 13 pre-Wf-350-A2 baseline + 1 NEW `p1_compute_manifest`).

Of the 3 NEW Wf-350 gates, **only `p1_compute_manifest` is in the `chemprop_v2_v3`
pipeline subset** (per `EXPECTED_GATE_COUNTS["chemprop_v2_v3"] = 14`); the other 2 are
`'all'`-only per the §3.4 design contract. To verify the 2 'all'-only gates in isolation,
this audit invokes them directly via Python import (see §3 below) — wall < 1 s each.

## §3 Per-gate verdict (3 of 3 PASS)

### §3.1 `p1_runtime_energy_report` (Wf-350-A1) — PASS

* **Function:** `gate_p1_runtime_energy_report(name, priority, pipeline, json_path)`
* **PASS criteria met:**
  * `nvidia-smi` rc=0; 2 GPU rows reported
  * All 4 mandatory top-level fields present (`per_gpu` + `total_duration_s` +
    `estimated_energy_j` + `n_gpus_reported`)
  * `per_gpu[*]` each has `index` (int) + `power_draw_w` (float ≥ 0) + `power_limit_w`
    (float > 0)
  * `estimated_energy_j >= 0` (sanity)
* **MEASURED live (artefact timestamp 2026-09-29T12:07:38+0000):**

  | GPU | Model | power_draw_w | power_limit_w |
  |---:|---|---:|---:|
  | 0 | RTX PRO 6000 Blackwell Workstation Edition | 15.60 W | 600.00 W |
  | 1 | RTX GeForce RTX 5090 | 9.18 W | 575.00 W |

  Note: the per-Wf-350-A1 probe timestamp at this Wf-357-A5 audit is
  `2026-09-29T10:46:36+0000` (in the JSONL artefact on disk) with reading
  `gpu[0]=251.18 W / gpu[1]=105.71 W`; the **2026-09-29T12:07:38+0000** run
  (this audit) sampled **15.60 W / 9.18 W**. Instantaneous power fluctuates with
  GPU load; the artefact records the snapshot at the probe moment, not a peak
  load reading. The live `nvidia-smi` query after the audit returned
  `gpu[0]=83.74 W / gpu[1]=9.54 W` — different again, as expected. **All 3 readings
  are valid MEASURED snapshots**; none are PROJECTED.
* **total_duration_s:** 0.086906 s
* **estimated_energy_j:** 1.076768 J
* **MEASURED_vs_PROJECTED tag:** `MEASURED` (in JSON body)
* **Wall:** 0.087 s (CPU-only — `nvidia-smi` 1-shot probe)

### §3.2 `p1_compute_manifest` (Wf-350-A2) — PASS

* **Function:** `gate_wf350a2_compute_manifest(name, priority, pipeline, json_path)`
* **PASS criteria met:**
  * Manifest JSON exists at `molmetal/reports/wf350a2_compute_manifest.json`
  * `schema_version == "wf350a2.v1"` matches the gate's expected schema
  * All 5 mandatory top-level fields populated (`cuda_version` + `driver_version` +
    `gpus` + `torch_version` + `ts`)
  * `gpus` is a list of ≥ 1 GPU record (here 2); each has `index` + `name` +
    `memory_total_mb` + `driver_version`
* **MEASURED live:**

  | Field | Value |
  |---|---|
  | `cuda_version` | 13.0 |
  | `driver_version` | 615.71.09 |
  | `torch_version` | 2.14.0+cu130 |
  | host | `ZhaoLab` |
  | kernel | 7.2.6-hardened1-1-hardened |
  | GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MB |
  | GPU 1 | NVIDIA GeForce RTX 5090, 32607 MB |
* **Wall:** 1.128 s (JSON parse + schema check)

### §3.3 `p1_hp_provenance_log` (Wf-350-A3) — PASS

* **Function:** `gate_p1_hp_provenance_log(name, priority, pipeline, config_paths,
  jsonl_path)`
* **PASS criteria met:**
  * All 7 expected config YAMLs are tracked (1 line per config in the JSONL)
  * Each line carries `config_path` + `current_sha` + `recent_commits` +
    `uncommitted_changes` (4 mandatory fields)
  * `current_sha` is non-empty + matches HEAD
  * JSONL is well-formed (one JSON object per line)
* **MEASURED live:**

  | Config | current_sha | recent_commits | uncommitted_changes |
  |---|---|---|---|
  | `configs/cfm_full_retrain.yaml` | 9b12e6f574… | [] | true |
  | `configs/cfm_per_innovation_ablation_p1.4.yaml` | 9b12e6f574… | [] | true |
  | `configs/cfm_real_decode_lift_p1.3.yaml` | 9b12e6f574… | [] | true |
  | `configs/default.yaml` | 9b12e6f574… | 2 entries | false |
  | `configs/wf124_p3_50epoch_full_sweep_g5.yaml` | 9b12e6f574… | [] | true |
  | `configs/wf124_p3_per_innovation_ablation_g6.yaml` | 9b12e6f574… | [] | true |
  | `configs/wf124_p3_remaining_metals_g4.yaml` | 9b12e6f574… | [] | true |

  Note: `uncommitted_changes: true` is expected and intentional — the 4 LOI / SOP
  wet-lab re-scope (per AGENTS.md §5.4 Tier-2 re-orientation) + Tier-2 INDEX sync
  edits are uncommitted as of Wf-357-A5 (matching the git status shown at
  session start). The gate does NOT block on this — it records it so downstream
  consumers can rewind to the exact source commit if they wish.
* **current_sha:** `9b12e6f574dfc397b5b5e8c568c952266e82ef37` (matches `git rev-parse
  HEAD`)
* **Wall:** 0.043 s

## §4 Overall verdict

**3 / 3 PASS** — all 3 NEW NeurIPS reproducibility gates (`p1_runtime_energy_report`,
`p1_compute_manifest`, `p1_hp_provenance_log`) re-verify at Wf-357-A5 with valid
JSON / JSONL evidence on disk.

The pre-flight pipeline `chemprop_v2_v3` (where these gates are most relevant per the
Wf-350 design) ran clean: 14/14 PASS / 0 FAIL / 14 TOTAL on the
2026-09-29T12:07:11+0000 invocation. `p1_compute_manifest` was included in that
chemprop_v2_v3 run; the other 2 'all'-only gates were verified by direct Python
import (CPU-only, < 1 s each) above.

## §5 Honest framing (per `party-twenty-spirit` §3.5)

* **MEASURED live:** per-GPU `power_draw_w` + `power_limit_w` (Wf-350-A1) + GPU model
  + memory + CUDA + driver + torch version (Wf-350-A2) + 7 config
  `current_sha`/`recent_commits`/`uncommitted_changes` (Wf-350-A3) + host
  `ZhaoLab` + kernel `7.2.6-hardened1-1-hardened`. All on disk as
  `molmetal/reports/wf350a{1,2,3}_*.{json,jsonl}`.
* **PROJECTED:** wall-budget "5-10 min CPU-only" was the upper-bound estimate; ACTUAL
  was ~3 min (chemprop_v2_v3 pipeline 72.7 s + A1/A2/A3 direct invocation 1.3 s + write
  + capture). NO honest-negative to declare.
* **No GPU launches performed:** verification only — all 3 gates are CPU-only
  pre-retrain internal checks.
* **No tests added:** 0 NEW tests; verification only.
* **No paper touches:** `paper/*` NOT touched (per TODO-84 + AGENTS.md §5.3).
* **No TODO directory changes:** this closeout writes ONE NEW file
  (`357a5_neurips_reproducibility_audit.md`); `pending/` UNCHANGED at 13 files;
  `completed/` 520 → 521 files.

## §6 Standing directives respected

* **§5.1 standing GPU grant:** respected — no GPU launch needed for this audit;
  `nvidia-smi` 1-shot probe only.
* **§5.2 honest framing:** all values are MEASURED live with artefact paths cited;
  PROJECTED vs MEASURED tags preserved (the JSONL itself carries
  `MEASURED_vs_PROJECTED: MEASURED`).
* **§5.3 no paper work:** NOT touched.
* **§5.4 Tier-2 / no wet-lab:** NOT touched.

## §7 Cross-references

* Pre-flight pipeline output: `molmetal/reports/wf357a5_neurips_gates.txt`
* Runtime + energy artefact: `molmetal/reports/wf350a1_runtime_energy.json`
* Compute manifest artefact: `molmetal/reports/wf350a2_compute_manifest.json`
* HP provenance artefact: `molmetal/reports/wf350a3_hp_provenance.jsonl`
* Pre-flight script (gate definitions):
  `molmetal/scripts/pre_flight_check.py` (lines 3088-3128 for §3.4 design
  contract + lines 3131-3351 for `gate_p1_hp_provenance_log` +
  lines 3353-3544 for `gate_p1_runtime_energy_report`)
* AGENTS.md §6 Wf timeline pointer: Wf-350 NeurIPS reproducibility gates ship wave
  (per AGENTS.md §6 recent Wf history, Wf-350 entry).
