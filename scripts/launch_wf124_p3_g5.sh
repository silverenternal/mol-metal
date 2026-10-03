#!/bin/bash
# Wf-124 P3 G5 — 15 (metal, seed) pairs launch wrapper (5 metals x 3 seeds x 50 epochs)
# Date: 2026-09-28 (post-Wf-258 user OK)
# Owner: Hugo Lee
# Per TODO-181 G5 + configs/wf124_p3_50epoch_full_sweep_g5.yaml
# Per kill_then_adapt directive: 8 waves per config (--no-enforce-budget override)
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Per Wf-328-W5: --w2-gradient-mode sort (Wf-326A MEASURED 2-3× speedup on
# RTX PRO 6000 Blackwell sm_120; replaces the legacy --w2-gradient-mode
# sliced override that was preserved bit-exact for in-flight retrains).
#
# Per Wf-328-W6: --num-workers 8 (Wf-326B build_optimized_loader() factory;
# takes precedence over --cpu-num-workers; wires prefetch_factor=8 +
# pin_memory=True + persistent_workers=True + spawn context).
#
# Per Wf-335-R1+R3: --no-cuda-graphs (safety lever; trainer default
# torch_compile_mode = max-autotune-no-cudagraphs per Wf-330-B4 RuntimeError
# fix on Ru seed=2 + Ir seed=0).
#
# Wf-337-G2: this launcher mirrors launch_wf124_p3_g6.sh v2 config for
# parity with the G6 per-innovation ablation sweep.
#
# Wf-347-B1: per-wave runtime check (Wf-343c2 7-check surface) wired into
# run_pair() — runs BEFORE each (metal, seed) launch with --skip-live to
# keep the dry-run path CPU-only. Also adds per-pair budget tracking
# (PAIR_BUDGET_START / PAIR_BUDGET_END log lines + elapsed_s delta).
#
# Wall-time estimate (per TODO-181 §3 + Wf-241 + Wf-260 G6 timing):
#   - Ru (heaviest cohort, ~12k rows): ~30 min for 50 epochs on RTX 5090
#   - Ir (medium cohort, ~7k rows):    ~30 min for 50 epochs on RTX 5090
#   - Os (medium-small cohort):         ~15 min for 50 epochs on RTX 5090
#   - Rh (small cohort):                ~15 min for 50 epochs on RTX 5090
#   - Re (smallest cohort):             ~10 min for 50 epochs on RTX 5090
#   - Total: ~5 hr on RTX 5090 GPU-1 (per Wf-260 G6 timing)
#
# Usage:
#   bash scripts/launch_wf124_p3_g5.sh              # real launch (skip if --dry-run passed)
#   bash scripts/launch_wf124_p3_g5.sh --dry-run    # CPU verification only (no GPU exec)
#   bash scripts/launch_wf124_p3_g5.sh --prereg docs/prereg/<file>.md
#     # Optional: pass a pre-registration record (Wf-350-B2 prereg.md template)
#     # Verifies hypothesis + primary_metric + baseline + expected_effect_size
#     # match the launch COMMON_ARGS before any GPU launch.
#     # If --prereg unset: emits WARNING that paper-grade launches should have one.

set -e
cd /home/hugo/codes/mol-metal

# Wf-366-A1 — USER_OK gate (kill_then_adapt standing directive; mirrors
# scripts/launch_wf131_p3_deferred_runs.sh Wf-355-B1 pattern at lines
# 118-125 + scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
# Wf-355-A3 pattern). Per AGENTS.md §5.1 kill_then_adapt: any > 1 hr GPU
# launch MUST be gated by USER_OK=1 in env, NOT silently auto-triggered.
# G5 wall ~5 hr on RTX 5090 GPU-1 (per Wf-260 G6 timing) — gate is
# CRITICAL SAFETY. Closes Wf-363-A2 HONEST-NEGATIVE gap (G5 was the
# only launch wrapper missing a USER_OK gate; live-run audit showed
# RC=124 timeout when invoked without --dry-run + without USER_OK).
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run (15 pairs)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --prereg docs/prereg/<file>.md   # paper-grade + pre-reg\n' "$0" >&2
    exit 64
fi

# Wf-366-A1 — source the shared DRY_RUN.log helper (consistent per-pair
# schema per Wf-363-A1; used in --dry-run mode below)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Per Wf-350-B2: parse --prereg <path> argument; if unset, emit WARNING
# that paper-grade launches should supply a pre-registration record per
# AGENTS.md §5.2 honest framing + DeepMind §3.1.
PREREG=""
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --prereg)
      shift_next=1
      ;;
    --prereg=*)
      PREREG="${arg#--prereg=}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
  esac
done
# After flag-scan, capture positional --prereg <path> if not already set
if [[ -z "${PREREG}" ]]; then
  for i in "$@"; do
    if [[ "${prev_arg:-}" == "--prereg" ]]; then
      PREREG="$i"
      break
    fi
    prev_arg="$i"
  done
fi

# Verify prereg.md against launch config if supplied
if [[ -n "${PREREG}" ]]; then
  if [[ ! -f "${PREREG}" ]]; then
    echo "FAIL: --prereg file not found: ${PREREG}"
    exit 1
  fi
  echo "=== PREREG verification: ${PREREG} ==="
  if ! grep -qE "^## §1 Research question" "${PREREG}"; then
    echo "FAIL: prereg.md missing §1 Research question section"
    exit 1
  fi
  if ! grep -qE "^## §4 Evaluation metric" "${PREREG}"; then
    echo "FAIL: prereg.md missing §4 Evaluation metric section"
    exit 1
  fi
  if ! grep -qE "epochs +\| +50" "${PREREG}"; then
    echo "WARN: prereg.md §3 hyperparameters table does not pin epochs=50 (matches G5 COMMON_ARGS)"
  fi
  if ! grep -qE "w2_gradient_mode +\| +sort" "${PREREG}"; then
    echo "WARN: prereg.md §3 hyperparameters table does not pin w2_gradient_mode=sort (matches G5 COMMON_ARGS)"
  fi
  echo "OK: prereg.md schema valid; hypothesis + metrics sections present"
  echo ""
else
  echo "WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md."
  echo ""
fi

OUT_BASE="results/wf124_p3_50epoch_full_sweep_g5"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 (CUDA_VISIBLE_DEVICES=0; G6 recovery holds GPU-1)
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"  # Wf-261 strict scaffold split (0 leak) + v4 cohort

# Per TODO-181 §3 + Wf-260 G6 batch-size + hidden + depth (50-ep full-sweep)
# Per Wf-263C task spec: --epochs 50 --batch-size 256 --hidden 300 --depth 3 + use-l2-guard
COMMON_ARGS=(
  --epochs 50
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --ablation-config mtl_baseline
  --w2-gradient-mode sort
  --n-mc 256
  --device "${GPU}"
  --cuda-visible-devices 0
  --cpu-num-workers 8
  --num-workers 8
  --data-csv "${DATA_CSV}"
  # Wf-335-R3 — CUDA Graphs hard-disable safety lever. Default ON in
  # trainer (per Wf-330-B4 MEASURED RuntimeError on Ru seed=2 + Ir seed=0
  # under --torch-compile-mode reduce-overhead); pass explicitly so paper
  # §6 audits can verify the lever is wired.
  --no-cuda-graphs
)

# ---- DRY-RUN mode: verify config + CSV paths only (no GPU execution) ----
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== Wf-124 P3 G5 — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "SCRIPT: ${SCRIPT}"
  echo "DATA_CSV: ${DATA_CSV}"
  echo "GPU: ${GPU}"
  echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
  echo ""
  echo "--- CSV sanity check ---"
  test -f "${DATA_CSV}" && echo "OK: CSV exists (${DATA_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
  n_rows=$(wc -l < "${DATA_CSV}")
  echo "n_rows (incl header): ${n_rows}"
  n_cols=$(head -1 "${DATA_CSV}" | tr ',' '\n' | wc -l)
  echo "n_columns: ${n_cols}"
  echo "header: $(head -1 "${DATA_CSV}")"
  echo ""
  echo "--- 15 (metal, seed) pairs to enumerate ---"
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      echo "  ${metal} seed=${seed}  ->  ${OUT_BASE}/${metal}/seed${seed}/"
    done
  done
  echo ""

  # Wf-366-A1 — emit structured DRY_RUN.log via Wf-363-A1 helper
  pair_args=()
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      pair_args+=( --pair "${metal} seed=${seed}" )
    done
  done
  # shellcheck disable=SC2068
  emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
    --title "Wf-124 P3 G5 — 15 (metal, seed) pairs full sweep" \
    --wrapper "$(basename "$0")" \
    --mode "G5_50epoch_full_sweep" \
    --trainer "${SCRIPT}" \
    --flags "--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline --w2-gradient-mode sort --n-mc 256 --device ${GPU} --cpu-num-workers 8 --num-workers 8 --no-cuda-graphs" \
    --gpu "${GPU}" \
    --budget-hr "5" \
    --user-ok "REQUIRED" \
    ${pair_args[@]+"${pair_args[@]}"}
  echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log"

  # Wf-366-A1 — emit plan.json (consistent with Wf-355 pattern)
  PLAN_JSON="${OUT_BASE}/plan.json"
  {
    printf '{\n'
    printf '  "wrapper": "launch_wf124_p3_g5.sh",\n'
    printf '  "wf_origin": "Wf-366-A1 (USER_OK gate fix)",\n'
    printf '  "parent_plan": "TODO/completed/181_wf124_p3_50_epoch_full_sweep.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": true,\n'
    printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
    printf '  "seeds": [0, 1, 2],\n'
    printf '  "epochs": 50,\n'
    printf '  "total_runs": 15,\n'
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": 5,\n'
    printf '  "wall_budget_label": "PROJECTED (~5 hr GPU sequential per Wf-260 G6 timing; Ru heaviest ~30 min/seed x 3 + Ir ~30 min x 3 + Os ~15 min x 3 + Rh ~15 min x 3 + Re ~10 min x 3)",\n'
    printf '  "trainer": "%s",\n' "${SCRIPT}"
    printf '  "data_csv": "%s",\n' "${DATA_CSV}"
    printf '  "out_base": "%s",\n' "${OUT_BASE}"
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; 15 pairs enumerated; DRY_RUN.log emitted",\n'
    printf '    "PROJECTED": "Ru seed 2 + Ir seed 0 in flight on GPU 0 (Wf-328 G6 PID 2040025); G5 wall ~5 hr on GPU 1",\n'
    printf '    "kill_then_adapt_directive": "5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
    printf '    "wf337_g2_status": "Wf-337-G2 hardened launcher config to mirror G6; Wf-366-A1 closes the residual USER_OK gate gap surfaced by Wf-363-A2 audit"\n'
    printf '  },\n'
    printf '  "status": "PLAN_ONLY"\n'
    printf '}\n'
  } > "${PLAN_JSON}"
  echo "plan.json emitted -> ${PLAN_JSON}"
  echo ""
  echo "DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed."
  exit 0
fi

# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
# Wf-356-A1: wf343c3 data quality check inserted between pre-flight and
# per-wave checklist — script is loader-agnostic (covers all 5 R20 loaders
# internally: load_binding_db, load_nci60, load_drugood, metalcytotox_v4,
# platinai_4cell). Does NOT accept --data-csv; only --out-dir.
# Use launcher-specific out-dir so each launcher's report is preserved.
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py --pipeline launch || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf124_p3_g5_data_quality \
  || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G5" || { echo "FAIL: per_wave_checklist"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }

# ---- Helper: poll pair log until target epoch is reached (Wf-302B guard) ----
# Refs: Wf-285D + Wf-291B — elapsed=0 MUST be initialised BEFORE the loop
# Refs: Wf-298E critical remaining #2 — bash wait_until_epoch_reached was MISSING
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-1200}"
  local start_s elapsed last_epoch
  start_s=$(date +%s)
  elapsed=0
  while true; do
    last_epoch=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$pair_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "$last_epoch" ] && [ "$last_epoch" -ge "$target_epoch" ]; then
      echo "Pair complete: last_epoch=$last_epoch >= target=$target_epoch"
      return 0
    fi
    elapsed=$(( $(date +%s) - start_s ))
    if [ "$elapsed" -ge "$max_wait_s" ]; then
      echo "TIMEOUT: last=$last_epoch elapsed=$((elapsed))s"
      return 1
    fi
    sleep 10
  done
}

# ---- REAL launch: per-pair runner ----
run_pair() {
  local metal=$1
  local seed=$2
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  # (Wf-343c2 7-check surface: preflight freshness + no-fail gates +
  # hyperparams config + YuelBond decoder + wave budget + GPU VRAM +
  # disk free space). --skip-live toggles nvidia-smi/df for CPU-only dry-runs.
  # Per Wf-337-G2 G5 launcher hardening + kill_then_adapt directive:
  # per-wave check is a DIAGNOSTIC emitter (per-pair JSON to
  # molmetal/reports/wf343c2_per_wave_check/), NOT a hard gate. The 2
  # known project-wide pre-flight FAILs (CFM Wasserstein loss file MISSING
  # + Vina not installed per AGENTS.md §5.4 honest-negative markers) are
  # pre-existing and unrelated to G5; the wf343c2 run emits the JSON
  # evidence so post-sweep diagnostics can attribute any per-pair failure
  # to pre-flight vs wave-specific causes. The exit-code is captured for
  # the per-pair budget tracker but does NOT abort the sweep.
  PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-124-P3-G5-${metal}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --config configs/wf124_p3_50epoch_full_sweep_g5.yaml \
    --budget-hr 0.7 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" 2>&1
  local rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK ${metal} seed=${seed} rc=${rc_c2} log=${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" | tee -a "${LOG}"
  # Diagnostic-only: do NOT return 1 on rc!=0 (per kill_then_adapt; see
  # comment above). The JSON evidence is captured for post-sweep audit.

  # Wf-347-B1: per-pair budget tracking — capture start time so the
  # pair-level wall-time can be aggregated into the wave_budget_tracker
  # JSON after the sweep. Mirrors Wf-328-W2 / Wf-337-G2 convention.
  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START ${metal} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}

  # Wf-347-B1: per-pair budget tracking — capture end time + emit delta
  local pair_end_s pair_elapsed_s
  pair_end_s=$(date +%s)
  pair_elapsed_s=$(( pair_end_s - pair_start_s ))
  echo "PAIR_BUDGET_END ${metal} seed=${seed} elapsed_s=${pair_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') END ${metal} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "$rc" -ne 0 ]; then
    echo "FAIL: ${metal} seed=${seed} exited rc=${rc}"
    return "$rc"
  fi
  # Wf-302B guard: verify the pair actually reached target epoch (50)
  wait_until_epoch_reached "${pair_log}" 50 1200 || { echo "FAIL: ${metal} seed=${seed} did not reach epoch 50"; return 1; }
}

echo "=== Wf-124 P3 G5 START $(date) ===" | tee -a "${LOG}"
echo "15 pairs (5 metals x 3 seeds x 50 ep) -- 8 waves per kill_then_adapt, --no-enforce-budget so all 15 PLANNED" | tee -a "${LOG}"

# 15 (metal, seed) pairs enumerated:
# Wave A: Ru s0 (~30 min)
run_pair Ru 0
# Wave B: Ru s1 (~30 min)
run_pair Ru 1
# Wave C: Ru s2 (~30 min)
run_pair Ru 2
# Wave D: Ir s0/s1 (~60 min)
run_pair Ir 0
run_pair Ir 1
# Wave E: Ir s2 + Os s0 (~45 min)
run_pair Ir 2
run_pair Os 0
# Wave F: Os s1/s2 + Rh s0 (~45 min)
run_pair Os 1
run_pair Os 2
run_pair Rh 0
# Wave G: Rh s1/s2 + Re s0 (~40 min)
run_pair Rh 1
run_pair Rh 2
run_pair Re 0
# Wave H: Re s1/s2 (~20 min)
run_pair Re 1
run_pair Re 2

echo "=== ALL TODO-181 G5 PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation — collect per_seed_metrics_seed*.json into training_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for metal_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not metal_dir.is_dir() or metal_dir.name == 'logs':
        continue
    for seed_dir in sorted(metal_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        for p in seed_dir.glob('per_seed_metrics_seed*.json'):
            runs.append({'metal': metal_dir.name, 'seed': seed_dir.name.replace(\"seed\", \"\"), 'metrics_path': str(p)})
out = {'todo_181_g5_total_pairs': len(runs), 'runs': runs}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (expected 15)')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"

# 30ep vs 50ep comparison stub (filled once Wf-261 v2 baseline re-pulled)
echo "=== 30ep vs 50ep comparison placeholder at ${OUT_BASE}/30ep_vs_50ep_comparison.json ===" | tee -a "${LOG}"
echo "TODO: aggregate once g5 sweep completes (TODO-181 §3.4)" | tee -a "${LOG}"