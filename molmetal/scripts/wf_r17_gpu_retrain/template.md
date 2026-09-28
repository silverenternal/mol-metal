# WF-R17-GPU-Retrain — Final Report (template, fill in after dGPU recovery)

**Workflow**: M-R17-A1 (R17 GPU retrain plan + monitoring scaffold)
**Date**: (filled by the wrapper run)
**Operator**: (filled by the operator running `r17_gpu_retrain.py`)
**Branch**: r17-ultracode-dispatch

---

## 0. TL;DR (one sentence)

(Filled: e.g. "3-seed × 10000-step YuelBond retrain achieved decode_ratio=1.000 at all 30 probes; 3 of 3 seeds reached step 10000; GPU hang recovery triggered N times.")

## 1. GPU detection

| Probe | Result |
|---|---|
| nvidia-smi -L | (filled) |
| rocm-smi -L | (filled) |
| torch.cuda.is_available() | (filled) |
| torch.cuda.device_count() | (filled) |
| Decision | NVIDIA_PATH / AMD_LEGACY / BLOCKED |

Reference: `logs/r17_retrain/gpu_probe.json` (written by the wrapper).

## 2. Configuration (R17 production)

Filled verbatim from `args.steps = 10000`:

| Knob | Value |
|---|---|
| hidden_dim | 128 |
| n_layers | 3 |
| batch_size | 4 |
| lr | 1e-4 |
| lr_warmup_steps | 200 (linear 5e-7 → 1e-4) |
| decode_smoke_every | 1000 |
| decode_smoke_samples | 16 |
| atom_vocab | 20 (post-TODO-33) |
| bond_head | learned |
| joint_train | True |

## 3. Per-seed trajectory

(Filled from `metrics/by_metric/cfm_decode_ratio_r17.json`)

| Seed | Steps reached | Wall_s | Decode probes | Max decode_ratio | First non-zero step | Resume events | Verdict |
|---|---|---|---|---|---|---|---|
| 0  |  |  |  |  |  |  |  |
| 1  |  |  |  |  |  |  |  |
| 2  |  |  |  |  |  |  |  |

## 4. Decode-ratio trajectory (combined)

(Filled from `decode_summary.all_probes`)

| Step | Seed | decode_ratio | n_decoded/N | mean_atoms | disconnected |
|---|---|---|---|---|---|
| 1000  |  |  |  |  |  |
| 2000  |  |  |  |  |  |
| 5000  |  |  |  |  |  |
| 10000 |  |  |  |  |  |

## 5. Loss trajectory (per-seed, key checkpoints)

(Filled from `seed_*/final.json`)

| seed | step 1 (cfm/atom/bond) | step 1000 | step 5000 | step 10000 |
|---|---|---|---|---|

## 6. GPU-hang recovery log

(Filled from `probe_state.json` + the wrapper's resume events)

| Hang detected at | Silent_s | Last step | Resume path | Restart rc | Wall to recovery |
|---|---|---|---|---|---|

(Spec: per `wf_r16_yuelbond_10000/final.md` the firmware SMU hang shows up as
`hsa_signal invalid`.  Recovery is operator-driven cold power cycle; this
wrapper cannot bypass that.  Resume from `--resume-from <latest.pt>` is the
operator's best effort to preserve learned weights across an unavoidable
restart.)

## 7. Verdict gate

| Gate | Threshold | Observed | Verdict |
|---|---|---|---|
| decode_ratio >= 0.05 at all probe steps | >= 0.05 | (filled) |  |
| 3-seed reproducibility | 3/3 | (filled) |  |
| 10000-step reach for each seed | reached | (filled) |  |
| No NaN/Inf cfm loss outside warmup region | no spikes > 1e4 | (filled) |  |

## 8. Artifacts

- `logs/r17_retrain/seed_<N>/final.{json,md}` — per-seed summaries
- `logs/r17_retrain/seed_<N>.log` — full child stdout per seed
- `logs/r17_retrain/checkpoints/seed_<N>/step_<k>.pt` — checkpoint archive
- `logs/r17_retrain/checkpoints/seed_<N>/latest.pt` — symlink to most recent
- `logs/r17_retrain/gpu_probe.json` — GPU detection snapshot
- `logs/r17_retrain/probe_state.json` — periodic state dump
- `metrics/by_metric/cfm_decode_ratio_r17.json` — combined decode-ratio trajectory
- `molmetal/reports/wf_r17_gpu_retrain/final.md` — this report

## 9. Comparison vs prior 1K/10K baselines

(Filled against `wf_r17_10k_retrain_done/final.md`)

## 10. Honest caveats

(Filled by the operator)

## 11. Next actions

(Filled by the operator)

---

End of report.
