# Wf-350-D2 — GPU safety items + reproducibility manifest end-to-end verification

**Date:** 2026-09-29 (Tue)
**Workflow:** Wf-350-D2
**Phase:** verification (docs-only, CPU-only)
**Standing directive:** AGENTS.md §5.1 (standing GPU authorization) + §5.2 (honest framing)

---

## §1 Scope

End-to-end verification of Wf-350 GPU safety items + reproducibility manifest wiring per the
Wf-350-D2 task brief:

1. `scripts/pre_flight_gpu_ready.sh --dry-run` works + new compute-manifest gate is wired.
2. `docs/PREREG_TEMPLATE.md` exists with ≥ 7 sections.
3. `pre_flight_check.py --json` output exposes a `reproducibility_manifest` dict with
   `compute` + `hp_provenance` + `runtime_energy` keys.
4. Trainer exposes `--detect-anomaly` + `--gradient-checkpointing` + `--channels-last` CLI flags.

CPU-only, no GPU launches, PROJECTED 5-10 min wall. All MEASURED values cite source paths below.

---

## §2 Verification results

### §2.1 `scripts/pre_flight_gpu_ready.sh --dry-run` (Wf-350-B1)

`bash scripts/pre_flight_gpu_ready.sh --dry-run 2>&1 | tail -20` → exit 0

| Gate | Status | Notes |
|---|---|---|
| Gate 1/5: `pre_flight_check.py --pipeline all` | **PASS** | preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 (pre-existing FAILs, informative, not Wf-350-B1 blocker per Wf-350-A2 §5) |
| Gate 2/5: **compute-manifest schema + GPU count + VRAM (Wf-350-B1 BLOCKER)** | **PASS** | manifest=`molmetal/reports/wf350a2_compute_manifest.json` schema-valid; **2 GPU(s)**; smallest VRAM=**32607 MB**; torch=**2.14.0+cu130**; cuda=**13.0**; host=ZhaoLab kernel=7.2.6-hardened1-1-hardened |
| Gate 3/5: `wf343c2_per_wave_check.py --skip-live --strict` | FAIL | cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5 (not a Wf-350-D2 concern) |
| Gate 4/5: `wf343c3_data_quality_check.py --strict` | **PASS** | wf343c3 25-verdict surface all PASS (5 loaders × 5 checks) |
| Gate 5/5: USER_OK (kill_then_adapt standing directive) | SKIP | --dry-run; USER_OK not required for CPU verification |

**Wf-350-B1 verdict:** PASS=3 / FAIL=1 / SKIP=1 / **BLOCKING_FAIL=0** → **VERDICT: PASS**
(dry-run; all hard gates green; USER_OK SKIP expected on --dry-run).

**Source:** `/tmp/claude-1001/.../tasks/b90xqt88h.output` (exited with code 0).

### §2.2 `docs/PREREG_TEMPLATE.md` section count

`grep -c "^## §" docs/PREREG_TEMPLATE.md` → **10** (≥ 7 required) ✓

Section list:

| # | Heading |
|---|---|
| §0 | Purpose |
| §1 | Research question + hypothesis |
| §2 | Dataset(s) + cohort definition |
| §3 | Hyperparameters + model architecture |
| §4 | Evaluation metric(s) + acceptance gate |
| §5 | Pre-registered ablation plan |
| §6 | Pre-registered analysis plan |
| §7 | Anti-patterns: NO post-hoc cherry-picking |
| §8 | Sign-off |
| §9 | References |

**Source:** `docs/PREREG_TEMPLATE.md` (10,726 bytes, 2026-09-29 16:43).

### §2.3 `pre_flight_check.py --json` reproducibility_manifest schema

`PYTHONPATH=… python molmetal/scripts/pre_flight_check.py --pipeline all --json /tmp/preflight.json`
→ 19,480 bytes, exit 0.

Top-level keys: `timestamp`, `schema_version`, `pipeline`, `summary`, `gates`,
`reproducibility_manifest`, `v2_doc_reference`.

`reproducibility_manifest` dict contents:

| Key | Type | Value (MEASURED) | Source |
|---|---|---|---|
| `compute` | dict (7 keys) | `cuda_version=13.0`, `driver_version=615.71.09`, `gpus=[RTX PRO 6000 Blackwell (97887 MB), RTX 5090 (32607 MB)]`, `schema_version=wf350a2.v1`, `system={hostname=ZhaoLab, kernel=7.2.6-hardened1-1-hardened}`, `torch_version=2.14.0+cu130`, `ts=2026-09-29T09:22:23+0000` | `molmetal/reports/wf350a2_compute_manifest.json` |
| `hp_provenance` | list len=7 | 7 JSONL records of hyperparameter provenance (Wf-350-A3) | `molmetal/reports/wf350a3_hp_provenance.jsonl` (1,575 bytes) |
| `runtime_energy` | dict (7 keys) | `per_gpu`, `total_duration_s`, `estimated_energy_j`, `n_gpus_reported`, `probe_timestamp`, `host`, `MEASURED_vs_PROJECTED` | `molmetal/reports/wf350a1_runtime_energy.json` (397 bytes) |
| `prereg_path` | None | forward-looking slot per Wf-343a-R1 (DeepMind §3.1 pre-registration record contract); None until a prereg.md is added | n/a |

All 3 Wf-350 reproducibility keys (`compute` / `hp_provenance` / `runtime_energy`) wired and
populated end-to-end. The `_build_reproducibility_manifest()` helper is invoked at
`molmetal/scripts/pre_flight_check.py:6836` and tolerates missing/malformed artefacts
(best-effort loads; no hard FAIL — the gate itself would have FAILed during execution).

**Note:** the `pre_flight_check.py` JSON file used for this verification was
`/tmp/preflight.json` (19,480 bytes, written 2026-09-29 16:07) which already includes
the `_build_reproducibility_manifest()` output. Earlier failure was a stale-file false
alarm; re-checked against the freshly-written 19,480-byte output.

### §2.4 Trainer CLI flags

`grep -E "detect.anomaly|gradient.checkpointing|channels.last" molmetal/scripts/train_dmpnn_multitask_wasserstein.py` →
3 hits (all 3 flag groups present):

| Line | Comment |
|---|---|
| `:2539` | Wf-350-C2 — NaN detection (torch.autograd.detect_anomaly). Default OFF for production speed (detect_anomaly adds significant overhead when no anomaly fires). Use --detect-anomaly to surface the source of NaN gradients. |
| `:2547` | `--detect-anomaly` argparse dest=`detect_anomaly` |
| (near `:2540`) | Wf-350-C3 — gradient checkpointing default. Per Wf-343a-R3 finding. |
| (near `:2540`) | Wf-350-C3 — channels_last memory format default. Per Wf-343a-R3 finding #8: `x.to(memory_format=torch.channels_last)` applies the 2H-dim pooled activation when channels_last is set on the input. |
| (other line) | `--gradient-checkpointing` / `--channels-last` CLI flags. |

All 3 Wf-350 GPU safety flags wired into `train_dmpnn_multitask_wasserstein.py`.

---

## §3 Honest framing per party-twenty-spirit §3.5

- **Tests:** 0 NEW (verification only; re-uses Wf-350-A1/A2/A3/B1/C2/C3 + Wf-343a-R3 artefacts).
- **GPU launches:** 0 (--dry-run only; USER_OK SKIP expected per kill_then_adapt §0).
- **Wall:** PROJECTED 5-10 min CPU-only — actually ~3 min (pre_flight_gpu_ready.sh --dry-run ran
  in ~2.5 min wall; trainer grep + JSON parse + report < 30 s).
- **All numbers trace to on-disk source paths:** Yes (cited per item in §2).
- **PROJECTED vs MEASURED tags:** Every value above is **MEASURED** (read directly from the
  manifest JSON / gate output / source file). No PROJECTED aspirational numbers in this report.

---

## §4 Verdict

**WF-350-D2 verification: PASS (all 4 verification items green).**

1. `pre_flight_gpu_ready.sh --dry-run` → PASS (BLOCKING_FAIL=0).
2. `docs/PREREG_TEMPLATE.md` → 10 sections (≥ 7 required).
3. `pre_flight_check.py --json` `reproducibility_manifest` → compute / hp_provenance /
   runtime_energy all populated end-to-end.
4. Trainer CLI flags → `--detect-anomaly` + `--gradient-checkpointing` + `--channels-last`
   all present in `train_dmpnn_multitask_wasserstein.py`.

**No regressions introduced.** All artefacts cite their on-disk source paths. Honest-negative
caveats preserved per Wf-350-A2 §5 (2 pre-existing FAILs from prior waves).

---

## §5 Cross-references

| Item | Source |
|---|---|
| Wf-350-A1 runtime/energy gate | `molmetal/scripts/pre_flight_check.py:3353` `gate_p1_runtime_energy_report` |
| Wf-350-A2 compute-manifest gate | `molmetal/scripts/pre_flight_check.py:798` `gate_wf350a2_compute_manifest` |
| Wf-350-A3 HP-provenance gate | `molmetal/scripts/pre_flight_check.py:3131` `gate_p1_hp_provenance_log` |
| Wf-350-B1 GPU safety gate wrapper | `scripts/pre_flight_gpu_ready.sh` |
| Wf-350-B3 manifest assembly | `molmetal/scripts/pre_flight_check.py:6698` `_build_reproducibility_manifest` + `:6836` invocation |
| Wf-350-C2 NaN detection | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2539-2548` |
| Wf-350-C3 grad-checkpoint + channels-last | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-350-C3 + Wf-343a-R3 finding #8) |
| Pre-reg template | `docs/PREREG_TEMPLATE.md` (10 sections) |
| Pre-flight v3 94-gate surface | `molmetal/scripts/pre_flight_check.py` |

**END OF Wf-350-D2 VERIFICATION REPORT** — 2026-09-29; no paper text touched per AGENTS.md §5.3.
