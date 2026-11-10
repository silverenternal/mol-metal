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

# Wf-A269 §5.11 migration (2026-11-04): OUT_BASE is overridable so that the
# dry-run audit can be exercised against a scratch tree instead of the
# tracked results/wf124_p3_per_innovation_ablation_g6/ dir (76 published
# artefacts for this wave live there).
OUT_BASE="${WF310B_OUT_BASE:-results/wf124_p3_per_innovation_ablation_g6}"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 (CUDA_VISIBLE_DEVICES=0; serial execution)
# Epoch count. The AUTHORITATIVE value is `train.epochs` in each static
# per-tuple config; this shell copy mirrors the pre-migration `--epochs 50`
# literal and is checked against the config by --expect in the fail-closed
# audit (audit_static_configs, below).
EPOCHS=50
# The CSV is ALSO declared in every static per-tuple config (top-level `csv:`
# and `data.csv:`). The config is authoritative for the trainer; the shell
# copy below exists for the pre-flight existence check and is checked against
# the config by --expect in the fail-closed audit.
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-04) — STATIC per-tuple YAML.
#
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may hand the trainer ONLY
# --config / --out-dir / --ckpt-dir / --help.
#
# WHY one config per tuple: the trainer loops over the metal cohort and the
# seed cohort INSIDE one process
# (train_dmpnn_multitask_wasserstein.py:4352 `for metal in args.metals:` /
# :4356 `for seed in args.seeds:`), while this launcher starts ONE trainer
# PROCESS per (arm, metal, seed) tuple, each with its own --out-dir.  So the
# cohort cannot move into the SHARED yaml: one yaml carrying 5 metals x 3
# seeds would turn the 60 processes into 1 and change the run.  Each tuple
# therefore needs a config that declares exactly ONE metal, ONE seed and
# exactly ONE ablation lever.
#
# WHY the file is STATIC (on disk, in git) and not written at launch time:
# §5.11's own rationale is "a knob that lives only in a shell script is a
# knob nobody audits" — a launch-time-written config puts the hyperparameters
# back into a shell variable.  A static file is reviewable BEFORE the run and
# reproducible AFTER it.
#
# The 60 files are audited BEFORE the sweep by
# scripts/_wf_yaml_static_config_c1.py, which compares the wrapper's own
# shell record (COMMON_ARGS + the arm's lever, below) against what the
# trainer will ACTUALLY read, through the trainer's OWN parser and the REAL
# loader — dest by dest, value AND type.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — the per-arm base configs + the launch-time
# derivation this replaces):
#   # 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
#   # The trainer already accepts --config and ends parse_args() in
#   # apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
#   # yaml is the base and the per-call flags below override it.
#   # Precedence: explicit CLI flag > YAML > argparse default.
#   #
#   # AGENTS.md §5.11 (2026-11-03): each ablation arm now has its OWN config file,
#   # so the wrapper selects the arm by NAME instead of by a command-line flag, and
#   # the shared configs/dmpnn.yaml (read by 8 other wrappers — deliberately NOT
#   # edited here) is no longer this wrapper's base. Every value this wrapper
#   # used to pass on the command line — the COMMON_ARGS array and the per-arm
#   # lever flag — now lives in that arm's config.
#   declare -A ARM_CONFIG=(
#     ["no_multi_task"]="${PROJECT_ROOT}/configs/wf310b_g6_final_no_multi_task.yaml"
#     ["no_wass"]="${PROJECT_ROOT}/configs/wf310b_g6_final_no_wass.yaml"
#     ["no_kendall"]="${PROJECT_ROOT}/configs/wf310b_g6_final_no_kendall.yaml"
#     ["no_dann"]="${PROJECT_ROOT}/configs/wf310b_g6_final_no_dann.yaml"
#   )
#   # Per-tuple derived config writer: the sweep dimensions (metals / seeds) are
#   # per PROCESS here — this wrapper runs 60 separate trainer processes, one per
#   # (config, metal, seed) tuple. See scripts/_wf_yaml_pair_config.py.
#   PAIR_CONFIG_HELPER="${PROJECT_ROOT}/scripts/_wf_yaml_pair_config.py"
#   DERIVED_CONFIG_DIR="${PROJECT_ROOT}/${OUT_BASE}/configs"
#
# WHY the per-arm base configs + the launch-time derivation were replaced:
# the derivation wrote each tuple's config at launch time from the arm's
# base yaml, so the tuple coordinates existed only as shell variables at
# review time.  The static cohort removes that dependency: every value the
# trainer reads is in the file named by the tuple, and that file's header
# records the pre-migration argv it reproduces.
# ---------------------------------------------------------------------------
PAIR_CONFIG_PREFIX="${PAIR_CONFIG_PREFIX:-configs/wfa269_c1_dmpnn_pair_sweep_wf310b_}"
# 4 arms x 5 metals x 3 seeds — the launcher MUST NOT silently under-launch.
PAIR_CONFIG_COUNT=60
# 5 metals x 3 seeds — the per-arm audit's own floor.
ARM_TUPLE_COUNT=15

# COMMON_ARGS is NO LONGER passed to the trainer.  It is now the wrapper's
# SHELL RECORD of the hyperparameters — the second, independent statement of
# intent that scripts/_wf_yaml_static_config_c1.py checks against the STATIC
# per-tuple YAML before the sweep (audit_static_configs, below).  The values
# the trainer actually reads live in
# configs/wfa269_c1_dmpnn_pair_sweep_wf310b_<arm>_<metal>_seed<n>.yaml.
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
)

# Per TODO-182 §2 + Wf-260 G6 + Wf-263C task spec:
#   --epochs 50 --batch-size 64 --hidden 300 --depth 3 + use-l2-guard
#
# AGENTS.md §5.11 (2026-11-03): the COMMON_ARGS array is GONE. Every flag it
# carried now lives in the arm's config (configs/wf310b_g6_final_<arm>.yaml):
#     --epochs 50              -> train.epochs
#     --batch-size 64          -> train.batch_size
#     --hidden 300             -> model.hidden
#     --depth 3                -> model.depth
#     --lr 1e-3                -> train.lr
#     --alpha 0.5              -> train.alpha
#     --w2-gradient-mode sliced-> eval.w2_gradient_mode
#     --n-mc 256               -> eval.n_mc
#     --use-l2-guard           -> eval.use_l2_guard
#     --device cuda:0          -> compute.device
#     --cuda-visible-devices 0 -> compute.cuda_visible_devices
#     --cpu-num-workers 8      -> compute.cpu_num_workers
#     --data-csv <path>        -> data.csv (dest `data_csv`)
#     --csv <path>             -> csv
#     --metals <metal>         -> data.metals        (per tuple)
#     --seeds <seed>           -> protocol.seeds     (per tuple)
# and the per-arm lever is declared in the arm's config too:
#     --single-task            -> train.single_task
#     --wasserstein-weight 0.0 -> eval.wasserstein_weight
#     --no-kendall             -> train.kendall
#     --no-dann                -> train.dann
# The superseded COMMON_ARGS array + CONFIRMED_FLAGS map are preserved
# VERBATIM in the comment block above run_tuple()'s trainer invocation.
#
# Wf-335-R3's `--no-cuda-graphs` was REMOVED here on 2026-10-27: the flag is
# not declared in train_dmpnn_multitask_wasserstein.py's argparse, so every
# tuple died rc=2 "unrecognized arguments" before an epoch ran. CUDA graphs
# only exist behind `--torch-compile`, which this launcher does not pass, so
# the lever is closed structurally. Same defect Wf-r28 / Wf-R32 fixed in the
# G5 / G6 launchers. It stays removed.

# ---- Ablation config dispatcher (per Wf-296C verification) ----
# The arm is now selected by CONFIG NAME (ARM_CONFIG above), not by a
# command-line lever flag. Wf-296C verified each lever flag against the
# trainer's --help; that verification is recorded in the arm config's header.
# The CONFIRMED_FLAGS map that used to carry those flags is preserved
# VERBATIM in the comment block above run_tuple()'s trainer invocation — it
# is a RECORD of the Wf-296C verification, not a live dispatcher.

# Per Wf-296C: all 4 configs wired; no skip-with-WARN path needed.
VERIFIED_CONFIGS="no_multi_task no_wass no_kendall no_dann"
UNVERIFIED_CONFIGS=""

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-04) — the arm's lever is no longer derived at launch
# time; it is the STATIC per-arm config's own leaf.  The table below is the
# wrapper's live SHELL RECORD of each arm's ablation — the same knob, the same
# arm, audited against the static config by scripts/_wf_yaml_static_config_c1.py
# (audit_static_configs).  The DISPATCH KEYS and the ablations they perform are
# unchanged.
declare -A ARM_LEVER_FLAG=(
  ["no_multi_task"]="--single-task"
  ["no_wass"]="--wasserstein-weight 0.0"
  ["no_kendall"]="--no-kendall"
  ["no_dann"]="--no-dann"
)

pair_config_path() {
  local cfg=$1
  local metal=$2
  local seed=$3
  printf '%s%s_%s_seed%s.yaml' "${PAIR_CONFIG_PREFIX}" "${cfg}" "${metal}" "${seed}"
}

# FAIL-CLOSED, CPU-only: compares the wrapper's own shell record (COMMON_ARGS
# + the arm's lever) against what the trainer will ACTUALLY read, through the
# trainer's OWN parser and the REAL loader — dest by dest, value AND type.
#
# The 5 auditor jobs (1 cohort + 4 arms) run CONCURRENTLY: each is a separate
# process that imports the trainer (and therefore torch) once, so running
# them in sequence costs ~21 s — longer than the 30 s timeout the repo's own
# dry-run test allows.  In parallel the wall time is the slowest single job
# (~7 s).
audit_static_configs() {
  local arm flag expect
  local audit_dir
  local -a pids=() labels=() results=()
  local i label rc=0
  audit_dir="$(mktemp -d "${TMPDIR:-/tmp}/wf310b_static_audit.XXXXXX")" || return 1

  # (1) whole cohort: every static config carries the common hyperparameters
  #     AND names the (metal, seed) tuple it carries.
  "${PY}" scripts/_wf_yaml_static_config_c1.py \
    --trainer "${SCRIPT}" \
    --config-glob "${PAIR_CONFIG_PREFIX}*.yaml" \
    --flags "${COMMON_ARGS[*]} --csv ${DATA_CSV}" \
    --expect "train.epochs=${EPOCHS}" \
    --expect "compute.device=${GPU}" \
    --expect "data.csv=${DATA_CSV}" \
    --expect "csv=${DATA_CSV}" \
    --expect-min-count "${PAIR_CONFIG_COUNT}" > "${audit_dir}/cohort.log" 2>&1 &
  pids+=("$!"); labels+=("cohort")

  # (2) per arm: the arm's lever must be the one the file actually carries.
  #     The auditor's --arm-flags matches an arm against the file name's
  #     UNDERSCORE-DELIMITED components, and every arm name here contains an
  #     underscore (no_wass, no_dann, ...) — so --arm-flags would silently
  #     match NOTHING and the lever would go UNAUDITED (a check that cannot
  #     fail is the defect, AGENTS.md §4).  The per-arm glob below carries the
  #     lever in --flags explicitly instead, so the lever's dest IS in the
  #     audited dest set, and --expect pins the lever's raw YAML path too.
  for arm in ${VERIFIED_CONFIGS}; do
    flag="${ARM_LEVER_FLAG[$arm]}"
    case "${arm}" in
      no_multi_task) expect="train.single_task=True" ;;
      no_wass)      expect="eval.wasserstein_weight=0.0" ;;
      no_kendall)   expect="train.kendall=False" ;;
      no_dann)      expect="train.dann=False" ;;
      *) echo "[static-config] FAIL: unknown arm ${arm}" >&2; return 1 ;;
    esac
    "${PY}" scripts/_wf_yaml_static_config_c1.py \
      --trainer "${SCRIPT}" \
      --config-glob "${PAIR_CONFIG_PREFIX}${arm}_*.yaml" \
      --flags "${COMMON_ARGS[*]} --csv ${DATA_CSV} ${flag}" \
      --expect "${expect}" \
      --expect "train.epochs=${EPOCHS}" \
      --expect "compute.device=${GPU}" \
      --expect "data.csv=${DATA_CSV}" \
      --expect "csv=${DATA_CSV}" \
      --expect-min-count "${ARM_TUPLE_COUNT}" > "${audit_dir}/${arm}.log" 2>&1 &
    pids+=("$!"); labels+=("${arm}")
  done

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then results+=("0"); else results+=("1"); fi
  done

  for i in "${!labels[@]}"; do
    label="${labels[$i]}"
    if [[ "${results[$i]}" == "0" ]]; then
      grep -E '^\[static-config\] PASS:' "${audit_dir}/${label}.log" || true
    else
      rc=1
      echo "[static-config] FAIL: audit job '${label}' failed" >&2
      cat "${audit_dir}/${label}.log" >&2
    fi
  done
  rm -rf "${audit_dir}"
  return "${rc}"
}

# FAIL-CLOSED abort for the static-config audit.
#
# WHY a separate function instead of an inline `audit_static_configs ||
# { ...; exit 1; }`: the --dry-run block has a source-level contract
# (Wf-366-A2, molmetal/tests/test_wf366a2_recover_g6_block_order.py) that the
# FIRST exit code literal after the `--dry-run` test must be 0 — the dry-run
# is the plan-only SUCCESS path.  Keeping the abort out of that block (as a
# function defined before it) preserves both properties at once: a failing
# audit still ABORTS the whole script with exit 1, and the dry-run block's
# own exit stays 0.
abort_if_static_config_audit_fails() {
  if ! audit_static_configs; then
    echo "FAIL: static-config audit (fail-closed)" >&2
    exit 1
  fi
}

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
  echo "CONFIRMED arms (each arm = its own STATIC per-tuple YAML, AGENTS.md §5.11): ${VERIFIED_CONFIGS}"
  for arm_cfg in ${VERIFIED_CONFIGS}; do
    echo "  ${arm_cfg} -> ${PAIR_CONFIG_PREFIX}${arm_cfg}_<metal>_seed<seed>.yaml"
  done
  echo "UNVERIFIED (verify-flag-before-launch): ${UNVERIFIED_CONFIGS:-<none>}"
  echo ""
  # Wf-A269 §5.11 (2026-11-04): the dry run must PROVE that the static
  # per-tuple configs exist and carry the hyperparameters this wrapper
  # intends.  The audit is fail-closed and CPU-only (it reads argparse + YAML
  # and never starts a trainer).
  echo "--- static per-tuple config audit (fail-closed, CPU-only) ---"
  abort_if_static_config_audit_fails
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
    "static_config_glob": "configs/wfa269_c1_dmpnn_pair_sweep_wf310b_*.yaml",
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
  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
  #
  # AGENTS.md §5.11 — YAML-only launch. The per-tuple config carries every
  # value the trainer used to receive on the command line (metals / seeds /
  # csv / data-csv / the whole COMMON_ARGS array / the arm's lever flag).
  #
  # Wf-A269 §5.11 (2026-11-04): the config is a STATIC file on disk — there
  # is NO launch-time derivation any more.  A missing config aborts the
  # tuple (fail-closed) rather than training the wrong cohort.  The
  # fail-closed audit (audit_static_configs) has already proven that every
  # one of the 60 static configs agrees with the shell record.
  local tuple_config
  tuple_config="$(pair_config_path "${cfg}" "${metal}" "${seed}")"
  if [[ ! -f "${tuple_config}" ]]; then
    echo "=== $(date '+%H:%M:%S') ABORT ${cfg}/${metal}/seed=${seed} (static tuple config missing: ${tuple_config}) ===" | tee -a "${LOG}"
    return 1
  fi
  # The config is copied into the run dir so the run's own artefacts record
  # exactly what was configured.
  install -m 0644 "${tuple_config}" "${OUT_BASE}/${cfg}/${metal}/seed${seed}/config.yaml"
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  # (2026-11-03, AGENTS.md §5.11 YAML-only migration. Every line below is the
  #  ORIGINAL line byte-for-byte (same flags, same order, same literal values,
  #  same indentation) after the "# " comment prefix, so the migration can be
  #  audited against the exact text it replaced. Every flag it carried now lives
  #  in the arm's config (configs/wf310b_g6_final_<arm>.yaml); the wrapper passes
  #  only --config, --out-dir and --ckpt-dir.
  # #
  # [SUPERSEDED 1 -- the pre-migration CONFIRMED_FLAGS map (the arm's lever), verbatim]
  # declare -A CONFIRMED_FLAGS=(
  #   ["no_multi_task"]="--single-task"
  #   ["no_wass"]="--wasserstein-weight 0.0"
  #   ["no_kendall"]="--no-kendall"
  #   ["no_dann"]="--no-dann"
  # )
  # #
  # [SUPERSEDED 2 -- the pre-migration COMMON_ARGS array, verbatim]
  # COMMON_ARGS=(
  #   --epochs 50
  #   --batch-size 64
  #   --hidden 300
  #   --depth 3
  #   --lr 1e-3
  #   --alpha 0.5
  #   --w2-gradient-mode sliced
  #   --n-mc 256
  #   --use-l2-guard
  #   --device "${GPU}"
  #   --cuda-visible-devices 0
  #   --cpu-num-workers 8
  #   --data-csv "${DATA_CSV}"
  #   # Wf-335-R3's `--no-cuda-graphs` was REMOVED here on 2026-10-27: the flag is
  #   # not declared in train_dmpnn_multitask_wasserstein.py's argparse, so every
  #   # tuple died rc=2 "unrecognized arguments" before an epoch ran. CUDA graphs
  #   # only exist behind `--torch-compile`, which this launcher does not pass, so
  #   # the lever is closed structurally. Same defect Wf-r28 / Wf-R32 fixed in the
  #   # G5 / G6 launchers.
  # )
  # #
  # [SUPERSEDED 3 -- the pre-migration trainer invocation, verbatim]
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY} ${SCRIPT} \
  # --config "${DMPNN_CONFIG}" \
  #     --metals "${metal}" \
  #     --seeds "${seed}" \
  #     --csv "${DATA_CSV}" \
  #     --data-csv "${DATA_CSV}" \
  #     --out-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}" \
  #     --ckpt-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts" \
  #     ${CONFIRMED_FLAGS[$cfg]} \
  # "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY} ${SCRIPT} \
    --config "${tuple_config}" \
    --out-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts" \
    2>&1 | tee "${pair_log}"
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

# Wf-A269 §5.11 (2026-11-04): fail-closed static-config audit BEFORE any
# tuple launches.  It compares the wrapper's own shell record (COMMON_ARGS +
# each arm's lever) against what the trainer will actually read, dest by
# dest, through the trainer's own parser and the real loader.  A drifted
# pair config aborts the sweep.
echo "=== static per-tuple config audit (fail-closed) ==="
abort_if_static_config_audit_fails
echo ""

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
