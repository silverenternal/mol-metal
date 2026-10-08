#!/bin/bash
# Wf-310B — TODO-182 G6 FINAL launch wrapper (USER_OK-gated, 8 sub-waves a..h)
#
# Owner: Hugo Lee
# Date: 2026-09-29
# Parent: TODO/pending/182_wf124_p3_per_innovation_ablation.md
#
# Per Wf-296C: G6 launcher wired with 60/60 CONFIRMED tuples
#              (4 configs × 5 metals × 3 seeds).
# Per Wf-306D: USER_OK gate enforced (kill_then_adapt standing directive).
# Per Wf-295D §1.3: 8 sub-waves a..h decomposition.
# Per Wf-300C: USER_OK gate pattern (no auto-trigger wall-time).
# Per TODO-207 §7.3: abort_recovery.py wired on unexpected abort.
# Per Wf-262 §10.1: per_wave_checklist.py emission per sub-wave.
# Per Wf-296C: wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required
#              (keeps all 60 tuples PLANNED, kill_then_adapt safe).
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# 60 (config, metal, seed) tuples enumerated:
#   configs: no_multi_task, no_wass, no_kendall, no_dann
#   metals:  Ru, Ir, Os, Rh, Re
#   seeds:   0, 1, 2
#   = 4 × 5 × 3 = 60 tuples.
#
# Total wall-time estimate (per Wf-258 §3.4 + Wf-260 G6 timing):
#   ~7.5 hr on RTX PRO 6000 Blackwell GPU-0 (cuda:0) — 60 runs × ~7.5 min @ 50 epoch
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - USER_OK gate is HARD-REQUIRED (kill_then_adapt; 7.5h wall).
#   - pre_flight 32/32 v3 PASS required (per Wf-306C PART 1; 12 chemprop
#     v3 P1 VERIFY + 6 CFM P1 VERIFY + 4 P2 DOCUMENTED + 10 v3 §3.x
#     expansion gates = 32 total).
#   - All 4 configs CONFIRMED post-Wf-296C (--no-kendall / --no-dann verified
#     in train_dmpnn_multitask_wasserstein.py --help lines 1323/1331).
#   - Per-lever attribution % NOT MEASURED until 60 runs complete.

set -e
cd /home/hugo/codes/mol-metal

OUT_BASE="results/wf124_p3_per_innovation_ablation_g6"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
# The trainer already accepts --config and ends parse_args() in
# apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
# yaml is the base and the per-call flags below override it.
# Precedence: explicit CLI flag > YAML > argparse default.
DMPNN_CONFIG="${DMPNN_CONFIG:-configs/dmpnn.yaml}"
GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 (CUDA_VISIBLE_DEVICES=0; serial execution)
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# Per TODO-182 §2 + Wf-260 G6 + Wf-263C task spec:
#   --epochs 50 --batch-size 64 --hidden 300 --depth 3 + use-l2-guard
COMMON_ARGS=(
  --epochs 50
  --batch-size 64
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --w2-gradient-mode sliced
  --n-mc 256
  --use-l2-guard
  --device "${GPU}"
  --cuda-visible-devices 0
  --cpu-num-workers 8
  --data-csv "${DATA_CSV}"
  # Wf-335-R3 — CUDA Graphs hard-disable safety lever. Default ON in
  # trainer (per Wf-330-B4 MEASURED RuntimeError on Ru seed=2 + Ir seed=0
  # under --torch-compile-mode reduce-overhead); pass explicitly so paper
  # §6 audits can verify the lever is wired.
  --no-cuda-graphs
)

# ---- Ablation config flag dispatcher (per Wf-296C verification) ----
declare -A CONFIRMED_FLAGS=(
  ["no_multi_task"]="--single-task"
  ["no_wass"]="--wasserstein-weight 0.0"
  ["no_kendall"]="--no-kendall"
  ["no_dann"]="--no-dann"
)

# Per Wf-296C: all 4 configs wired; no skip-with-WARN path needed.
VERIFIED_CONFIGS="no_multi_task no_wass no_kendall no_dann"
UNVERIFIED_CONFIGS=""

# ---- DRY-RUN mode (CPU verification only, no GPU execution) ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-310B G6 FINAL — DRY-RUN (CPU verification only) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "SCRIPT: ${SCRIPT}"
  echo "DATA_CSV: ${DATA_CSV}"
  echo "GPU: ${GPU}"
  echo "USER_OK gate: present (kill_then_adapt standing directive)"
  echo "Pre-flight: pre_flight_check.py --pipeline launch (32/32 v3 PASS required)"
  echo "Wave budget tracker: --no-enforce-budget --cfm-yuelbond-required"
  echo "Per-wave checklist: per_wave_checklist.py --wave Wf-124-P3-G6-{a..h}"
  echo "Abort recovery: abort_recovery.py wired on unexpected abort"
  echo ""
  echo "--- CSV sanity check ---"
  test -f "${DATA_CSV}" && echo "OK: CSV exists (${DATA_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
  n_rows=$(wc -l < "${DATA_CSV}")
  echo "n_rows (incl header): ${n_rows}"
  echo ""
  echo "--- Ablation config verification status ---"
  echo "CONFIRMED flags: ${VERIFIED_CONFIGS}"
  echo "UNVERIFIED (verify-flag-before-launch): ${UNVERIFIED_CONFIGS:-<none>}"
  echo ""
  echo "--- 60 (config, metal, seed) tuples enumerated ---"
  total=0
  for cfg in no_multi_task no_wass no_kendall no_dann; do
    for metal in Ru Ir Os Rh Re; do
      for seed in 0 1 2; do
        total=$((total + 1))
        echo "  [CONFIRMED] cfg=${cfg} ${metal} seed=${seed}  ->  ${OUT_BASE}/${cfg}/${metal}/seed${seed}/"
      done
    done
  done
  echo ""
  echo "--- 8 sub-waves a..h (per Wf-295D §1.3) ---"
  echo "  wave_a: no_multi_task Ru/Ir/Os (9 runs, ~1.1 hr)"
  echo "  wave_b: no_multi_task Rh/Re + no_wass Ru (9 runs, ~1.1 hr)"
  echo "  wave_c: no_wass Ir/Os/Rh/Re (12 runs, ~1.5 hr)"
  echo "  wave_d: no_kendall Ru/Ir (6 runs, ~0.8 hr)"
  echo "  wave_e: no_kendall Os/Rh/Re (9 runs, ~1.1 hr)"
  echo "  wave_f: no_dann Ru/Ir/Os (9 runs, ~1.1 hr)"
  echo "  wave_g: no_dann Rh (3 runs, ~0.4 hr)"
  echo "  wave_h: no_dann Re (3 runs, ~0.5 hr incl. report)"
  echo "  TOTAL: 60 tuples, ~7.6 hr on RTX PRO 6000 Blackwell GPU-0 (cuda:0)"
  echo ""
  echo "DRY-RUN OK: ${total} (config, metal, seed) tuples PLANNED, no GPU work executed."
  echo "  CONFIRMED runs (real launch): ${total} (all 4 configs wired post-Wf-296C)"
  echo "  UNVERIFIED runs (real launch): 0 (all flags verified)"
  echo "  USER_OK gate: required (set USER_OK=1 to authorize real launch)"
  exit 0
fi

# USER_OK gate (Wf-306D — Wf-300C pattern: user explicit approval required
# for any 7.5h wall launch; per kill_then_adapt standing directive, never auto-
# trigger wall-time without user OK).
if [ "${USER_OK:-0}" != "1" ]; then
  echo "USER OK REQUIRED — kill_then_adapt standing directive."
  echo "G6 = 60 (config × metal × seed) tuples × ~7.5 hr wall-time on RTX 5090."
  echo "Re-run with:  USER_OK=1 bash scripts/launch_wf310b_g6_final.sh"
  exit 1
fi
echo "[Wf-310B] USER_OK set — proceeding with 60-tuple G6 launch (wall ~7.5 hr)" | tee -a "${LOG}"

# ---- DRY_RUN.log + plan.json emission (Wf-363-A1 schema) ----
DRY_RUN_LOG_HELPER="/home/hugo/codes/mol-metal/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
source "${DRY_RUN_LOG_HELPER}"
mkdir -p "${OUT_BASE}/plan"
cat >"${OUT_BASE}/DRY_RUN.log" <<EOF
[DRY-RUN] launch_wf310b_g6_final.sh @ wf310b
[DRY-RUN] wrapper_name=launch_wf310b_g6_final.sh
[DRY-RUN] trainer=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
[DRY-RUN] py=/home/hugo/codes/mol-metal/.venv/bin/python
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
EOF
cat >"${OUT_BASE}/plan.json" <<EOF
{
    "wrapper": "launch_wf310b_g6_final.sh",
    "wf_origin": "wf310b",
    "mode": "dry-run",
    "user_ok_required": true,
    "trainer": "molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
    "py": "/home/hugo/codes/mol-metal/.venv/bin/python",
    "out_dir": "${OUT_BASE}",
    "expected_cohort": {"n_tuples": 60, "wall_hr": 7.5},
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF

# ---- Pre-flight check (Wf-306C PART 1: 32/32 v3 PASS required) ----
echo "=== Pre-flight check START $(date) ===" | tee -a "${LOG}"
${PY} molmetal/scripts/pre_flight_check.py --pipeline launch || {
  echo "FAIL: pre_flight_check --pipeline all (32/32 v3 PASS required per Wf-306C)"
  exit 1
}
echo "=== Pre-flight check PASS $(date) ===" | tee -a "${LOG}"

# ---- Wave budget tracker (Wf-296C: --no-enforce-budget --cfm-yuelbond-required) ----
echo "=== Wave budget tracker START $(date) ===" | tee -a "${LOG}"
${PY} molmetal/scripts/wave_budget_tracker.py \
  --no-enforce-budget \
  --cfm-yuelbond-required || {
  echo "FAIL: wave_budget_tracker"
  exit 1
}
echo "=== Wave budget tracker PASS $(date) ===" | tee -a "${LOG}"

# ---- Wf-356-A1: wf343c3 data quality check (loader-agnostic — all 5 R20
#      loaders internally: load_binding_db, load_nci60, load_drugood,
#      metalcytotox_v4, platinai_4cell). Does NOT accept --data-csv; only
#      --out-dir. Use launcher-specific out-dir so each launcher's report
#      is preserved. Inserted after pre_flight_check + wave_budget_tracker
#      per Wf-347-B3 retry. ----
echo "=== wf343c3 data quality check START $(date) ===" | tee -a "${LOG}"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf310b_g6_final_data_quality || {
  echo "FAIL: wf343c3_data_quality_check"
  exit 1
}
echo "=== wf343c3 data quality check PASS $(date) ===" | tee -a "${LOG}"

# ---- Per-wave checklist emission (Wf-262 §10.1; per sub-wave a..h) ----
emit_wave_checklist() {
  local wave_label=$1
  echo "=== Per-wave checklist ${wave_label} START $(date) ===" | tee -a "${LOG}"
  ${PY} molmetal/scripts/per_wave_checklist.py --wave "${wave_label}" || {
    echo "FAIL: per_wave_checklist --wave ${wave_label}"
    exit 1
  }
  echo "=== Per-wave checklist ${wave_label} PASS $(date) ===" | tee -a "${LOG}"
}

# ---- Per-tuple runner (with abort_recovery wiring per TODO-207 §7.3) ----
run_tuple() {
  local cfg=$1
  local metal=$2
  local seed=$3
  local pair_log="${OUT_BASE}/${cfg}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts"

  # Skip unverified configs with WARN (per honest framing above)
  if [[ " ${UNVERIFIED_CONFIGS} " == *" ${cfg} "* ]]; then
    echo "=== $(date '+%H:%M:%S') SKIP ${cfg}/${metal}/seed${seed} (UNVERIFIED flag) ===" | tee -a "${LOG}"
    return 0
  fi

  echo "=== $(date '+%H:%M:%S') START ${cfg}/${metal}/seed=${seed} ===" | tee -a "${LOG}"
  set +e
  # GPU 0 (RTX PRO 6000 Blackwell) per the 2026-10-03 all-GPU-0
  # directive; this call site is serial with every other job.
  # Bug fix 2026-10-08: removed --user-ok "REQUIRED" literal — D-MPNN trainer has
  # no --user-ok flag; passing it caused argparse rc=2 which the trainer-pipe
  # to tee previously masked. USER_OK gating is enforced by the outer wrapper
  # env check (this script gates USER_OK=1 + Wf-310B activation per kill_then_adapt),
  # NOT by a CLI flag (per chemprop_v3 P1.4 wrapper post-mortem same date).
  CUDA_VISIBLE_DEVICES=0 ${PY} ${SCRIPT} \
--config "${DMPNN_CONFIG}" \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts" \
    ${CONFIRMED_FLAGS[$cfg]} \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "=== $(date '+%H:%M:%S') ABORT ${cfg}/${metal}/seed=${seed} rc=${rc} ===" | tee -a "${LOG}"
    # abort_recovery.py per TODO-207 §7.3 — reactive ckpt recovery
    ${PY} molmetal/scripts/abort_recovery.py \
      --metal "${metal}" --seed "${seed}" --wave "Wf-124-P3-G6" \
      --pattern "aborted" || true
    return ${rc}
  fi
  echo "=== $(date '+%H:%M:%S') END ${cfg}/${metal}/seed=${seed} ===" | tee -a "${LOG}"
}

echo "=== Wf-310B G6 FINAL START $(date) ===" | tee -a "${LOG}"
echo "60 tuples (4 configs x 5 metals x 3 seeds x 50 ep) -- 8 waves per config" | tee -a "${LOG}"
echo "CONFIRMED configs: ${VERIFIED_CONFIGS} (60 runs PLANNED post-Wf-296C)" | tee -a "${LOG}"
echo "UNVERIFIED configs: ${UNVERIFIED_CONFIGS:-<none>} (0 runs SKIPPED)" | tee -a "${LOG}"

# ---- 8 sub-waves a..h (per Wf-295D §1.3) ----
emit_wave_checklist "Wf-124-P3-G6-a"
for metal in Ru Ir Os; do
  for seed in 0 1 2; do
    run_tuple no_multi_task "${metal}" "${seed}"
  done
done

emit_wave_checklist "Wf-124-P3-G6-b"
for metal in Rh Re; do
  for seed in 0 1 2; do
    run_tuple no_multi_task "${metal}" "${seed}"
  done
done
for seed in 0 1 2; do
  run_tuple no_wass Ru "${seed}"
done

emit_wave_checklist "Wf-124-P3-G6-c"
for metal in Ir Os Rh Re; do
  for seed in 0 1 2; do
    run_tuple no_wass "${metal}" "${seed}"
  done
done

emit_wave_checklist "Wf-124-P3-G6-d"
for metal in Ru Ir; do
  for seed in 0 1 2; do
    run_tuple no_kendall "${metal}" "${seed}"
  done
done

emit_wave_checklist "Wf-124-P3-G6-e"
for metal in Os Rh Re; do
  for seed in 0 1 2; do
    run_tuple no_kendall "${metal}" "${seed}"
  done
done

emit_wave_checklist "Wf-124-P3-G6-f"
for metal in Ru Ir Os; do
  for seed in 0 1 2; do
    run_tuple no_dann "${metal}" "${seed}"
  done
done

emit_wave_checklist "Wf-124-P3-G6-g"
for seed in 0 1 2; do
  run_tuple no_dann Rh "${seed}"
done

emit_wave_checklist "Wf-124-P3-G6-h"
for seed in 0 1 2; do
  run_tuple no_dann Re "${seed}"
done

echo "=== ALL TODO-182 G6 TUPLES DONE $(date) ===" | tee -a "${LOG}"

# ---- Aggregation — collect per_seed_metrics_seed*.json into training_summary.json ----
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for cfg_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not cfg_dir.is_dir():
        continue
    cfg = cfg_dir.name
    for metal_dir in sorted(cfg_dir.iterdir()):
        if not metal_dir.is_dir():
            continue
        for seed_dir in sorted(metal_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for p in seed_dir.glob('per_seed_metrics_seed*.json'):
                runs.append({'config': cfg, 'metal': metal_dir.name, 'seed': seed_dir.name.replace('seed', ''), 'metrics_path': str(p)})
out = {'todo_182_g6_total_tuples_planned': 60, 'tuples_aggregated': len(runs), 'runs': runs}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} confirmed tuples (expected up to 60)')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"

# Per-lever attribution report stub (filled once sweep completes per TODO-182 §4)
echo "=== Per-lever attribution report placeholder at ${OUT_BASE}/ablation_report.json ===" | tee -a "${LOG}"
echo "TODO: aggregate attribution matrix once 60-run sweep completes (TODO-182 §3.4)" | tee -a "${LOG}"
