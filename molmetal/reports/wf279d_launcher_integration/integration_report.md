# Wf-279D — Launcher × Wave-Tool Integration Report

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Question:** Do the Wf-124 P3 launchers actually invoke `wave_budget_tracker.py` / `per_wave_checklist.py` / `pre_flight_check.py` before launching?
**Reference:** Wf-262 created the three wave tools; Wf-279D verifies that launchers USE them.

---

## 1. Tool inventory (Wf-262 deliverables)

| File | Path | Size | Last-modified |
|---|---|---|---|
| `wave_budget_tracker.py` | `molmetal/scripts/wave_budget_tracker.py` | 36 KB | 2026-09-29 00:15 |
| `per_wave_checklist.py` | `molmetal/scripts/per_wave_checklist.py` | 36 KB | 2026-09-28 20:25 |
| `pre_flight_check.py` | `molmetal/scripts/pre_flight_check.py` | 32 KB | 2026-09-28 20:43 |

All three tools exist as Python source files at the path the launchers would `cd` to (`/home/hugo/codes/mol-metal` + `molmetal/scripts/...`). The path is reachable.

---

## 2. Launcher inventory

`find /home/hugo/codes/mol-metal/scripts -name 'launch_*.sh' -type f` returned exactly **3 launchers**:

| # | Launcher | Wave plan | Output dir |
|---|---|---|---|
| 1 | `scripts/launch_wf124_p3_g4.sh` | 2 waves × ≤8 pairs | `results/wf124_p3_remaining_metals_g4/` |
| 2 | `scripts/launch_wf124_p3_g5.sh` | 8 waves × ≤3 pairs | `results/wf124_p3_50epoch_full_sweep_g5/` |
| 3 | `scripts/launch_wf124_p3_g6.sh` | 8 waves × ≤8 tuples | `results/wf124_p3_per_innovation_ablation_g6/` |

---

## 3. Integration matrix (grep counts)

| Launcher | `wave_budget_tracker` | `per_wave_checklist` | `pre_flight_check` |
|---|---|---|---|
| `launch_wf124_p3_g4.sh` | 0 | 0 | 0 |
| `launch_wf124_p3_g5.sh` | 0 | 0 | 0 |
| `launch_wf124_p3_g6.sh` | 0 | 0 | 0 |
| **TOTAL** | **0 / 3** | **0 / 3** | **0 / 3** |

**Verdict:** **All three launchers SKIP every pre-launch safety tool.** They go straight from `set -e` + `mkdir -p` to invoking `train_dmpnn_multitask_wasserstein.py` via `CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} ...`.

---

## 4. Cross-check (incidental launchers)

| File | Invokes any tool? |
|---|---|
| `results/cfm_full_retrain/launch_loop.sh` | 0 — also bypasses all three |
| `references/SynFlowNet/src/synflownet/hyperopt/wandb_demo/launch_wandb_agents.sh` | 0 — third-party upstream, out of scope |

So the bypass pattern is repo-wide, not Wf-124-specific.

---

## 5. Bug summary

The Wf-262 trio (`wave_budget_tracker.py`, `per_wave_checklist.py`, `pre_flight_check.py`) was authored but **never wired into any launcher**. The Wf-124 P3 G4/G5/G6 wrappers — which collectively control **60+ GPU runs on RTX 5090 (≈10 wall-clock hours per Wf-260 G6 timing)** — have zero pre-launch guardrails:

- No `wave_budget_tracker` call → wall-time envelope can silently overrun
- No `per_wave_checklist` call → per-wave exit criteria can drift
- No `pre_flight_check` call → CSV / GPU / venv can be wrong at launch (only caught after first run fails)

The launchers DO have `--dry-run` mode (G5/G6) that does config + CSV sanity, but that's local shell, not the formal `pre_flight_check.py` tool. G4 has no `--dry-run` at all.

---

## 6. Recommendation (fix surface for Wf-279E)

1. **Insert a pre-launch block at the top of every `run_pair` / `run_tuple` function** that calls:
   ```bash
   ${PY} molmetal/scripts/pre_flight_check.py \
     --csv "${DATA_CSV}" --metals "${metal}" --seeds "${seed}" \
     --epochs "${EPOCHS}" --device "${GPU}" || { echo "PRE-FLIGHT FAIL"; exit 1; }
   ```
2. **Insert a wave-start hook** between waves that calls `per_wave_checklist.py --wave-name "${wave}"` to assert per-wave exit criteria.
3. **Insert a wall-time sentinel** at the very start of each launcher:
   ```bash
   ${PY} molmetal/scripts/wave_budget_tracker.py \
     --launcher "launch_wf124_p3_g5" --expected-hours 5 \
     --register
   ```
4. **Backward-compat gate:** keep `--dry-run` working by skipping the pre-launch block when `$1 == "--dry-run"`.
5. **Wave completion hook:** after the final `tee` of each wave, call `per_wave_checklist.py --mark-complete --wave-name "${wave}"` so the checklist's tracker has MEASURED timestamps.

Estimated LOC addition per launcher: ~15 lines (1 block + 1 hook + 1 sentinel). Total: ~45 LOC for all three launchers to be brought into compliance with Wf-262.

---

## 7. Files referenced

- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g4.sh` (no integration)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` (no integration)
- `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g6.sh` (no integration)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py` (Wf-262 deliverable, unwired)
- `/home/hugo/codes/mol-metal/molmetal/scripts/per_wave_checklist.py` (Wf-262 deliverable, unwired)
- `/home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py` (Wf-262 deliverable, unwired)