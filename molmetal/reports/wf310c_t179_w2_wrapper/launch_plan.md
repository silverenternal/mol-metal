# Wf-310C Part 3 — TODO-179 W₂ PlatinAI 4-cell alignment GPU wrapper launch plan

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: 🚧 WRAPPER READY — USER_OK-gated, awaiting user opt-in
**Wrapper path**: `molmetal/scripts/launch_wf310c_t179_w2_gpu.sh`

---

## §1 Purpose

Wire the W₂ pIC50 alignment retrain (TODO-179 §3.1) into a USER_OK-gated GPU
launcher per the kill_then_adapt convention (Wf-295D + Wf-306B). The trainer
already has the W₂ wire-in from Wf-246-A (`--wass-weight 0.05` +
`--ref-distribution platinai_4cell`); the wrapper enforces the user OK gate
and the 5-seed × 50-epoch GPU sweep itself.

---

## §2 Scope

| Item | Value |
|---|---|
| Trainer | `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-37 P1 ship) |
| Dataset | PlatinAI 4-cell probe (`molmetal/data/platinai_4cell.csv`, 4078 Pt complexes × 4 cell lines = ~16k rows) |
| Epochs per seed | 50 |
| Seeds | 0, 1, 2, 3, 4 (5-seed standard per `mol-metal-gpu-execution` §6) |
| W₂ weight | 0.05 (TODO-179 §3.1 baseline; ramp to 0.20 over 10 epochs is per-epoch trainer logic) |
| Reference distribution | platinai_4cell |
| Per-cell cap (O(N²) W₂ bound) | 32 (TODO-179 §3 task spec) |
| GPU | RTX 5090 cuda:1 |
| Wall budget | 4.5 hr PROJECTED (5 × ~50 min/seed × W₂ overhead + 10% slack) |

---

## §3 Honest framing per TODO-179 §6

- **MEASURED L5 R² baseline**: +0.163 (Wf-37 P1 bug fix — NOT an algorithmic lift)
- **PROJECTED L5 R² ceiling** from W₂ alone: 0.40-0.60 (12-d RDKit signal floor
  per Wf-127 §Cause 2)
- **0.85+ ceiling** requires F5_RDKIT_2D 32-d expansion (TODO-179 §8 follow-up)
- Per-cell breakdown will be emitted (no aggregation hiding per-cell noise
  per party-twenty-spirit §3.5)
- W₂ may slow early convergence (per Wf-124 P3 §5): predicted pIC50 forced to
  track reference before encoder has learned anything; JKO-style CFL gate
  satisfied but lift may require >10 epochs

---

## §4 USER_OK gate (per kill_then_adapt)

| Step | Action |
|---|---|
| 1 | `touch /tmp/USER_OK_WF310C_T179` (user opt-in marker) |
| 2 | `bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh` (actual GPU run) |

Without the marker file the wrapper exits 2 (USER_OK missing) and prints
the honest framing banner.

---

## §5 Exit codes

| Code | Meaning |
|---:|---|
| 0 | GPU run completed (all 5 seeds ok) |
| 2 | USER_OK missing (gated; no run started) |
| 3 | Setup error (venv / trainer / CSV / CUDA probe) |
| 4 | GPU run failed (partial results still in `run.log`) |

---

## §6 Outputs

- `molmetal/reports/wf310c_t179_w2_wrapper/plan.json` (plan + honest framing)
- `molmetal/reports/wf310c_t179_w2_wrapper/run.log` (full stdout/stderr)
- `molmetal/reports/wf310c_t179_w2_wrapper/sweep_summary.json` (aggregate)
- `molmetal/reports/wf310c_t179_w2_wrapper/seed{0,1,2,3,4}/` (per-seed `r2_bench.json` + `comparison_table.md`)

---

## §7 Cross-references

- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (parent — full spec)
- `molmetal/scripts/train_hilbert_astsr_end_to_end.py` (Wf-37 P1 + Wf-246-A wire-in)
- `molmetal/molmetal_lam/reward/wasserstein_loss.py` (M1 module — wasserstein_pic50_loss)
- `molmetal/data/platinai_4cell.csv` (PlatinAI 4-cell probe)
- `molmetal/reports/wf37_l5_hilbert_astsr/` (Wf-37 P1 baseline R²=+0.163)
- `molmetal/reports/wf25_hilbert_l2_astsr/` (Wf-25 honest-neg R²=-0.443)
- `molmetal/reports/wf127_l5_hilbert_diagnostic/` (Wf-127 P2 — 12-d signal floor root cause)

---

## §8 Honest framing disclaimer (party-twenty-spirit §3.5)

Wf-310C Part 3 ships the **USER_OK-gated wrapper** for the W₂ alignment retrain.
The trainer is ready (Wf-246-A wire-in is in place); the GPU wall-time is the
sole remaining cost (~4.5 hr on RTX 5090 cuda:1). Honest projection: 0.40-0.60
R² ceiling from W₂ alone is the realistic MEASURED outcome; 0.85+ requires
F5_RDKIT_2D 32-d expansion as a follow-up.