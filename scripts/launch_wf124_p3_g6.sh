#!/bin/bash
# Wf-124 P3 G6 — 60 (config, metal, seed) pairs launch wrapper
#   4 ablation configs x 5 metals x 3 seeds = 60 runs (per-innovation ablation)
# Date: 2026-09-28 (post-Wf-258 user OK)
# Owner: Hugo Lee
# Per TODO-182 G6 + configs/wf124_p3_per_innovation_ablation_g6.yaml
# Per kill_then_adapt directive: 8 waves per config (per-innovation ablation)
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Per Wf-328-W5 §Integration points: G5/G6 keep --w2-gradient-mode sliced.
# The sort default is for FUTURE launchers; these two are the in-flight
# retrains whose numbers are compared bit-exact against the Wf-131 P1
# baseline in the paper §6 reproducibility audit.  (bc29aa90 flipped this
# to `sort` in an unrelated venv-path commit; reverted 2026-10-20.)
# NOTE: the YAML also pins `w2_gradient_mode: sliced`; the explicit flag is
# kept only because Wf-334u4 asserts it is present and equal to sliced.
#
# Per Wf-328-W6: --num-workers 8 (Wf-326B build_optimized_loader() factory;
# takes precedence over --cpu-num-workers; wires prefetch_factor=8 +
# pin_memory=True + persistent_workers=True + spawn context).
#
# Wf-R32 (2026-10-07): --num-workers and --no-cuda-graphs REMOVED from
# COMMON_ARGS. Neither is declared in this trainer's argparse, so all 60
# G6 tuples aborted with rc=2 "unrecognized arguments". This is the same
# defect wf-R28 fixed in the G5 launcher (whose header named G6 as
# "still blocked on the same defect"). The Wf-328-W6 DataLoader-factory
# and Wf-335-R1+R3 cudagraphs-lever notes below are retained as history
# only — both levers are structurally satisfied (manual batch build, so
# no DataLoader workers are needed; no --torch-compile, so no CUDA
# graphs). See the COMMON_ARGS comment for the full evidence trail.
#
# Per Wf-335-R1+R3 (HISTORICAL, flag removed per Wf-R32 — see above):
# --no-cuda-graphs (safety lever; trainer default
# torch_compile_mode = max-autotune-no-cudagraphs per Wf-330-B4 RuntimeError
# fix on Ru seed=2 + Ir seed=0).
#
# Per Wf-296C: --no-kendall / --no-dann wired in CONFIRMED_FLAGS per-config
# (no_multi_task / no_wass / no_kendall / no_dann); verified at trainer
# lines 1323 / 1331 in train_dmpnn_multitask_wasserstein.py.
#
# Wf-337-G2: this launcher is the v2 reference for all Wf-328 + Wf-335
# fixes consolidated; G4 + G5 launchers mirrored to the same config.
#
# Per Wf-347-B2: per-tuple pre-launch check via wf343c2_per_wave_check.py
# (--skip-live since global pre_flight_check already covers VRAM/disk).
# Mirrors Wf-337-G1 G6 recovery script integration point at run_tuple().
#
# Wall-time estimate (per TODO-182 §3 + Wf-260 G6 timing):
#   - Per-run ~15 min on RTX 5090 (smaller subset due to ablations)
#   - Total: 60 x ~7.5 min = ~7.5 hr on RTX 5090 GPU-1
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - Configs 1 (--single-task) + 2 (--wasserstein-weight 0.0) are CONFIRMED flags
#   - Configs 3 (--no-kendall) + 4 (--no-dann) are status: verify-flag-before-launch
#     per g6 config §entry.pre_launch_checklist. If flag missing, the wrapper
#     emits a WARN and skips the affected tuples (honest-negative close).
#   - Per-lever attribution % NOT MEASURED until 60 runs complete.
#
# Usage:
#   bash scripts/launch_wf124_p3_g6.sh              # real launch
#   bash scripts/launch_wf124_p3_g6.sh --dry-run    # CPU verification only (no GPU exec)
#   bash scripts/launch_wf124_p3_g6.sh --prereg docs/prereg/<file>.md
#     # Optional: pass a pre-registration record (Wf-350-B2 prereg.md template)
#     # Verifies hypothesis + primary_metric + baseline + expected_effect_size
#     # match the launch COMMON_ARGS before any GPU launch.
#     # If --prereg unset: emits WARNING that paper-grade launches should have one.

set -e
cd /home/hugo/codes/mol-metal

# Wf-378-A2 — USER_OK gate (kill_then_adapt standing directive; canonical
# Wf-366-A1 pattern matching scripts/launch_wf124_p3_g5.sh lines 51-67 +
# scripts/recover_g6_failed_tuples.sh Wf-366-A2 lines 95-109). Per
# AGENTS.md §5.1 kill_then_adapt: any > 1 hr GPU launch MUST be gated by
# USER_OK=1 in env, NOT silently auto-triggered. G6 wall ~7.5 hr on
# RTX 5090 GPU-1 (per Wf-260 G6 timing) — gate is CRITICAL SAFETY.
# Closes Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1 HONEST-NEGATIVE finding:
# the G6 wrapper's USER_OK gate was wired AFTER the --dry-run exit block
# at lines 208-218, so `--dry-run` silently bypassed the gate. This Wf
# moves the gate BEFORE the --dry-run block (Wf-366-A2 lesson — dry-run
# must not bypass USER_OK).
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~7.5 hr GPU run (60 tuples)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --prereg docs/prereg/<file>.md   # paper-grade + pre-reg\n' "$0" >&2
    exit 64
fi

# Wf-378-A2 — source the shared DRY_RUN.log helper (consistent per-pair
# schema per Wf-363-A1; used in --dry-run mode below).
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
    --prereg=*)
      PREREG="${arg#--prereg=}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
  esac
done
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
    echo "WARN: prereg.md §3 hyperparameters table does not pin epochs=50 (matches G6 COMMON_ARGS)"
  fi
  echo "OK: prereg.md schema valid; hypothesis + metrics sections present"
  echo ""
else
  echo "WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md."
  echo ""
fi

# Wf-A269 wave-3 (2026-11-02): OUT_BASE is overridable so that the
# USER_OK-gate test emits into pytest's tmp_path.  That test used to
# emit into THIS tracked tree and then ``shutil.rmtree`` it, destroying
# the tracked published artefacts under it.  SUPERSEDED RENDERING
# PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   OUT_BASE="results/wf124_p3_per_innovation_ablation_g6"
OUT_BASE="${WF124_G6_OUT_BASE:-results/wf124_p3_per_innovation_ablation_g6}"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — COMMON_ARGS used to be the CLI array the trainer
# RECEIVED; it is now only the shell record, see below):
#   # 2026-10-07 (user directive: configs in YAML, not CLI). The trainer already
#   # accepts --config and ends parse_args() in apply_yaml_overrides
#   # (train_dmpnn_multitask_wasserstein.py:3320-3322). The yaml is the base; the
#   # per-ablation CONFIRMED_FLAGS and COMMON_ARGS below still win, which is
#   # required here — the 4 G6 arms differ ONLY by those flags, so moving them into
#   # one shared yaml would collapse the ablation design.
#   #
#   # Wf-A269 §5.11 (2026-11-03): the per-tuple override now happens INSIDE each
#   # tuple's own yaml, not on the command line: derive_pair_config writes every
#   # COMMON_ARGS value and the arm's CONFIRMED_FLAGS lever into the per-tuple
#   # config, and the trainer invocation line carries only --config / --out-dir /
#   # --ckpt-dir.  The arms stay 4 arms — each tuple gets its OWN yaml.  The
#   # VALUES are unchanged: the CLI used to win over the yaml, and the yaml now
#   # carries exactly the CLI's values.
DMPNN_CONFIG="${DMPNN_CONFIG:-configs/dmpnn.yaml}"
#
# Wf-A269 §5.11 (2026-11-04): COMMON_ARGS is NO LONGER passed to the trainer.
# It is now the wrapper's SHELL RECORD of the hyperparameters — the second,
# independent statement of intent that scripts/_wf_yaml_static_config_c1.py
# checks against the STATIC per-tuple YAML before the sweep
# (audit_static_configs, below).  The values the trainer actually reads live
# in configs/wfa269_c1_dmpnn_pair_sweep_g6_<arm>_<metal>_seed<n>.yaml.
#
# Wf-A269 (2026-11-04): ${DMPNN_CONFIG} NO LONGER HAS ANY EFFECT on this
# wrapper.  The trainer is launched with the STATIC per-tuple config named by
# (arm, metal, seed) —
# configs/wfa269_c1_dmpnn_pair_sweep_g6_<arm>_<metal>_seed<n>.yaml — so there
# is no base config left to select and configs/dmpnn.yaml is NOT read by the
# launch (it is shared by 8 wrappers).  The variable is kept DEFINED so an
# operator's exported DMPNN_CONFIG cannot trip `set -u`, and it is echoed in
# the dry-run banner so a stale override is VISIBLE rather than silently
# inert.
GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 (CUDA_VISIBLE_DEVICES=0; serial execution)
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"  # Wf-261 strict scaffold split (0 leak)

# Per TODO-182 §2 + Wf-260 G6 + Wf-263C task spec:
#   --epochs 50 --batch-size 256 --hidden 300 --depth 3 + use-l2-guard
#
# Wf-A269 §5.11 (2026-11-04): COMMON_ARGS is NO LONGER passed to the trainer.
# It is now the wrapper's SHELL RECORD of the hyperparameters — the second,
# independent statement of intent that audit_static_configs (below) checks
# against the STATIC per-tuple YAML before the sweep.  It is kept VERBATIM
# (pre-migration CLI array) because it is the dry-run display copy and it is
# pinned by molmetal/tests/test_wf337g2_launcher_v2.py +
# test_wf337g3_g6_regression.py.
COMMON_ARGS=(
  --epochs 50
  --batch-size 256
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
  # Wf-R32 (2026-10-07): `--num-workers 8` and `--no-cuda-graphs` were
  # REMOVED — neither is declared in this trainer's argparse
  # (train_dmpnn_multitask_wasserstein.py), so all 60 G6 tuples would die
  # with argparse rc=2 "unrecognized arguments" before a single epoch ran.
  # This is the identical defect wf-R28 fixed in the G5 launcher; G6 was
  # named as "still blocked on the same defect" in that closeout's header.
  # MEASURED evidence: `python -m molmetal.scripts.train_dmpnn_multitask_wasserstein
  # --config configs/dmpnn.yaml --num-workers 8` -> "error: unrecognized
  # arguments: --num-workers 8", SystemExit rc=2.
  #   - `--num-workers`: no dest exists; `--cpu-num-workers` (kept above) is
  #     the trainer's only worker flag, and main() already documents it as a
  #     structural no-op (manual batch build, no torch DataLoader).
  #   - `--no-cuda-graphs`: CUDA graphs only exist behind `--torch-compile`
  #     (action="store_true", default=False). G6 does not pass
  #     --torch-compile, so graphs are already off; the Wf-335-R3 safety
  #     lever is satisfied structurally. The only graph-related flag the
  #     trainer exposes is the enabling `--torch-compile` itself, and
  #     omitting it keeps the lever closed.
  # The Wf-328-W6 and Wf-335-R3 header notes below are retained as history
  # only — the levers are structurally satisfied.
)

# Wf-A269 §5.11 (2026-11-04): the FULL effective hyperparameter record the
# fail-closed audit compares against the static per-tuple YAML.
#
# COMMON_ARGS above is the pre-migration CLI array, kept VERBATIM.  Three
# values the old command line did NOT spell out are added here, because the
# audit must compare the EFFECTIVE value and not only the values that used
# to be typed on the command line:
#   --csv <DATA_CSV>          (the old CLI carried BOTH --csv and --data-csv)
#   --wasserstein-weight 0.05 (came from configs/dmpnn.yaml eval.wasserstein_weight)
#   --ablation-config mtl_baseline
#                             (the trainer's argparse default; configs/dmpnn.yaml
#                              carries no such key)
AUDIT_FLAGS="${COMMON_ARGS[*]} --csv ${DATA_CSV} --wasserstein-weight 0.05 --ablation-config mtl_baseline"

# ---- Ablation config flag dispatcher (per TODO-182 §3.4 + g6 yaml) ----
# Prints the per-config flag set + status line. Returns 0 if confirmed,
# returns 1 if "verify-flag-before-launch" (caller decides skip-vs-launch).
#
# Args:
#   $1 = ablation config key (no_multi_task | no_wass | no_kendall | no_dann)
# Writes to stdout: bash array string of flag args to be appended to
# CUDA_VISIBLE_DEVICES=0 ${PY} ${SCRIPT} invocation.
# Wf-A269 §5.11 (2026-11-04): these four strings are no longer handed to the
# trainer on the command line.  They are the ARM LEVER each static per-tuple
# config must carry, and audit_static_configs (below) checks them through
# --arm-flags, whose ARM token is matched against the FILE NAME's
# underscore-delimited components (scripts/_wf_yaml_static_config_c1.py:349-353).
# The G6 file stems decompose as
#   wfa269_c1_dmpnn_pair_sweep_g6_no_multi_task_Ru_seed0
#     -> ['wfa269','c1','dmpnn','pair','sweep','g6','no','multi','task','Ru','seed0']
# so the LIVE tokens are multi / wass / kendall / dann.  The literal arm names
# (no_multi_task / no_wass / ...) are NOT components and can never fire, and
# the shared component 'no' is in ALL FOUR arms — using either would be a
# DEAD GATE.  MEASURED 2026-11-04: with the live token 'wass' the audit
# FAILs when the lever is wrong (the token fired on exactly the 15 wass
# configs); with the dead token 'no_wass' the arm's lever is never applied
# and the wass configs' eval.wasserstein_weight 0.0 is compared against the
# base record's 0.05 — a FAIL for the wrong reason, i.e. the sweep aborts
# instead of running unchecked.
declare -A CONFIRMED_FLAGS=(
  ["no_multi_task"]="--single-task"
  ["no_wass"]="--wasserstein-weight 0.0"
  # Wf-296C — wire Wf-283B flags as per-config ablation levers
  # (per-config so --no-kendall / --no-dann only apply to no_kendall / no_dann
  # tuples, not blanket-disabled across all 60 runs which would collapse the
  # per-innovation ablation design). Verified via grep on
  # train_dmpnn_multitask_wasserstein.py lines 1323 / 1331.
  ["no_kendall"]="--no-kendall"
  ["no_dann"]="--no-dann"
)

# Per g6 yaml §entry.pre_launch_checklist + Wf-296C verification:
#   - --no-kendall / --no-dann NOW confirmed in --help output
#     (train_dmpnn_multitask_wasserstein.py:1323 + :1331, Wf-283B)
#   - All 4 configs wired; no skip-with-WARN path needed.
VERIFIED_CONFIGS="no_multi_task no_wass no_kendall no_dann"
UNVERIFIED_CONFIGS=""

# ---- DRY-RUN mode: verify config + CSV paths only (no GPU execution) ----
# Ordered arrays for DRY_RUN.log per-pair enumeration (Wf-378-A2 — bash
# associative arrays do not preserve order).
CONFIRMED_FLAGS_ORDER=(${G6_CONFIGS:-no_multi_task no_wass no_kendall no_dann})
METALS_ORDER=(${G6_METALS:-Ru Ir Os Rh Re})
SEEDS_ORDER=(${G6_SEEDS:-0 1 2})

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 migration (2026-11-04) — STATIC per-tuple YAML cohort.
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
# seeds would turn the 60 processes into 4 and change the run.  Each tuple
# therefore needs a config that declares exactly ONE metal and ONE seed.
#
# WHY the file is STATIC (on disk, reviewable BEFORE the run) and not written
# at launch time: the Wf-A269 c1 wave (2026-11-03) explicitly REJECTED the
# launch-time derivation — "a launch-time-written config would make the run's
# hyperparameters unreviewable before the run and unreproducible afterwards"
# and "this wave's contract forbids generating a YAML at launch time".  The
# 60 files are on disk under configs/ and each one's header records the
# pre-migration argv it reproduces.
#
# The 60 files are audited BEFORE the sweep by
# scripts/_wf_yaml_static_config_c1.py, which compares the wrapper's own shell
# record (AUDIT_FLAGS, above) against what the trainer will ACTUALLY read,
# through the trainer's OWN parser and the REAL loader — dest by dest, value
# AND type — and adds each arm's lever to the configs whose FILE NAME carries
# that arm's token.  MEASURED 2026-11-04: 60/60 PASS, 1005 flag dests
# verified, wall 3.2 s, CPU-only.
# ---------------------------------------------------------------------------
PAIR_CONFIG_PREFIX="${PAIR_CONFIG_PREFIX:-configs/wfa269_c1_dmpnn_pair_sweep_g6_}"
# 4 arms x 5 metals x 3 seeds — the launcher MUST NOT silently under-launch.
PAIR_CONFIG_COUNT=60

# The static config of one (arm, metal, seed) tuple.
pair_config_path() {
  local arm=$1
  local metal=$2
  local seed=$3
  printf '%s%s_%s_seed%s.yaml' "${PAIR_CONFIG_PREFIX}" "${arm}" "${metal}" "${seed}"
}

# The ARM -> TOKEN map the auditor needs: the auditor matches an arm against
# the FILE NAME's underscore-delimited components
# (scripts/_wf_yaml_static_config_c1.py:349-353), so each arm needs the one
# component that is unique to it.  Fail closed if the two maps disagree.
declare -A ARM_FLAG_TOKEN=(
  ["no_multi_task"]="multi"
  ["no_wass"]="wass"
  ["no_kendall"]="kendall"
  ["no_dann"]="dann"
)
# Fixed order so the audit invocation is byte-reproducible.
ARM_FLAG_ORDER=(no_multi_task no_wass no_kendall no_dann)

# Fail-closed: a missing / drifted tuple config aborts the launch.
audit_static_configs() {
  # The lever is stated ONCE: CONFIRMED_FLAGS above holds the flag string and
  # ARM_FLAG_TOKEN holds the file-name component the auditor matches it
  # against.  Fail closed if the two maps have different key sets.
  local arm
  local n_confirmed=0
  for arm in "${!CONFIRMED_FLAGS[@]}"; do n_confirmed=$((n_confirmed + 1)); done
  local n_token=0
  for arm in "${!ARM_FLAG_TOKEN[@]}"; do n_token=$((n_token + 1)); done
  local n_order=0
  for arm in "${ARM_FLAG_ORDER[@]}"; do n_order=$((n_order + 1)); done
  if [[ "${n_confirmed}" -ne "${n_token}" || "${n_confirmed}" -ne "${n_order}" ]]; then
    echo "FAIL: CONFIRMED_FLAGS has ${n_confirmed} arm(s), ARM_FLAG_TOKEN has ${n_token}, ARM_FLAG_ORDER has ${n_order}" >&2
    return 1
  fi
  for arm in "${ARM_FLAG_ORDER[@]}"; do
    if [[ -z "${CONFIRMED_FLAGS[$arm]:-}" || -z "${ARM_FLAG_TOKEN[$arm]:-}" ]]; then
      echo "FAIL: arm '${arm}' has no CONFIRMED_FLAGS lever or no ARM_FLAG_TOKEN" >&2
      return 1
    fi
  done
  local arm_flags=()
  for arm in "${ARM_FLAG_ORDER[@]}"; do
    arm_flags+=(--arm-flags "${ARM_FLAG_TOKEN[$arm]}=${CONFIRMED_FLAGS[$arm]}")
  done
  "${PY}" scripts/_wf_yaml_static_config_c1.py \
    --trainer "${SCRIPT}" \
    --config-glob "${PAIR_CONFIG_PREFIX}*.yaml" \
    --flags "${AUDIT_FLAGS}" \
    --expect-min-count "${PAIR_CONFIG_COUNT}" \
    "${arm_flags[@]}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== Wf-124 P3 G6 — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "SCRIPT: ${SCRIPT}"
  echo "DATA_CSV: ${DATA_CSV}"
  echo "GPU: ${GPU}"
  echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
  # Wf-A269 (2026-11-04): DMPNN_CONFIG has NO EFFECT on this launch (the
  # per-tuple config is selected by (arm, metal, seed)).  Echoing it makes a
  # stale operator override VISIBLE instead of silently inert.
  echo "DMPNN_CONFIG (NO EFFECT — inert override, see the note above): ${DMPNN_CONFIG}"
  echo "STATIC CONFIG PREFIX: ${PAIR_CONFIG_PREFIX} (${PAIR_CONFIG_COUNT} configs expected)"
  echo ""
  echo "--- CSV sanity check ---"
  test -f "${DATA_CSV}" && echo "OK: CSV exists (${DATA_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
  n_rows=$(wc -l < "${DATA_CSV}")
  echo "n_rows (incl header): ${n_rows}"
  echo ""
  echo "--- Ablation config verification status ---"
  echo "CONFIRMED flags: ${VERIFIED_CONFIGS}"
  echo "UNVERIFIED (verify-flag-before-launch): ${UNVERIFIED_CONFIGS}"
  echo ""
  echo "--- (config, metal, seed) tuples enumerated ---"
  total=0
  for cfg in "${CONFIRMED_FLAGS_ORDER[@]}"; do
    for metal in "${METALS_ORDER[@]}"; do
      for seed in "${SEEDS_ORDER[@]}"; do
        total=$((total + 1))
        status="CONFIRMED"
        [[ " ${UNVERIFIED_CONFIGS} " == *" ${cfg} "* ]] && status="UNVERIFIED-skip-if-real"
        echo "  [${status}] cfg=${cfg} ${metal} seed=${seed}  ->  ${OUT_BASE}/${cfg}/${metal}/seed${seed}/  [config: $(pair_config_path "${cfg}" "${metal}" "${seed}")]"
      done
    done
  done
  echo ""
  # Wf-A269 §5.11 (2026-11-04): the dry run must PROVE that the static
  # per-tuple configs exist and carry the hyperparameters this wrapper
  # intends.  The audit is fail-closed and CPU-only (it reads argparse + YAML
  # and never starts a trainer).
  echo "--- static per-tuple config audit (fail-closed, CPU-only) ---"
  audit_static_configs || { echo "FAIL: static-config audit"; exit 1; }
  echo ""

  # Wf-378-A2 — emit structured DRY_RUN.log via Wf-363-A1 helper
  pair_args=()
  for cfg in "${CONFIRMED_FLAGS_ORDER[@]}"; do
    for metal in "${METALS_ORDER[@]}"; do
      for seed in "${SEEDS_ORDER[@]}"; do
        pair_args+=( --pair "${cfg}/${metal} seed=${seed}" )
      done
    done
  done
  # shellcheck disable=SC2068
  # Wf-A269 §5.11 (2026-11-04): the --flags below is the RUN's
  # hyperparameters, but they are no longer CLI flags — they are carried
  # by the static per-tuple YAML, and audit_static_configs proves the two
  # agree.
  emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
    --title "Wf-124 P3 G6 — 60 (config, metal, seed) tuples per-innovation ablation" \
    --wrapper "$(basename "$0")" \
    --mode "G6_per_innovation_ablation" \
    --trainer "${SCRIPT}" \
    --flags "--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --w2-gradient-mode sliced --n-mc 256 --use-l2-guard --device ${GPU} --cpu-num-workers 8 ${COMMON_ARGS[*]}" \
    --gpu "${GPU}" \
    --budget-hr "7.5" \
    --user-ok "REQUIRED" \
    ${pair_args[@]+"${pair_args[@]}"}
  echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log"

  # Wf-378-A2 — emit plan.json (consistent with Wf-355 pattern)
  PLAN_JSON="${OUT_BASE}/plan.json"
  {
    printf '{\n'
    printf '  "wrapper": "launch_wf124_p3_g6.sh",\n'
    printf '  "wf_origin": "Wf-378-A2 (USER_OK gate parity closure per Wf-366-A2 lesson)",\n'
    printf '  "parent_plan": "TODO/completed/182_wf124_p3_per_innovation_ablation.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": true,\n'
    printf '  "configs": ["no_multi_task", "no_wass", "no_kendall", "no_dann"],\n'
    printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
    printf '  "seeds": [0, 1, 2],\n'
    printf '  "epochs": 50,\n'
    printf '  "total_tuples": 60,\n'
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": 7.5,\n'
    printf '  "wall_budget_label": "PROJECTED (~7.5 hr GPU sequential per Wf-260 G6 timing; per-innovation ablation 60 tuples)",\n'
    printf '  "trainer": "%s",\n' "${SCRIPT}"
    printf '  "data_csv": "%s",\n' "${DATA_CSV}"
    printf '  "out_base": "%s",\n' "${OUT_BASE}"
    printf '  "static_config_prefix": "%s",\n' "${PAIR_CONFIG_PREFIX}"
    printf '  "static_config_count": %s,\n' "${PAIR_CONFIG_COUNT}"
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; USER_OK gate enforced BEFORE --dry-run per Wf-378-A2; 60 tuples enumerated; DRY_RUN.log emitted",\n'
    printf '    "PROJECTED": "G6 wall ~7.5 hr on RTX 5090 GPU-1; per-lever attribution %% NOT MEASURED until sweep completes",\n'
    printf '    "kill_then_adapt_directive": "7.5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
    printf '    "wf378_a2_status": "Wf-378-A2 closes Wf-363-A2 + Wf-365-B2 + Wf-368-B1 §7.1 HONEST-NEGATIVE finding (G6 --dry-run was bypassing USER_OK gate at original lines 208-218; now gate fires FIRST at line ~60 per Wf-366-A2 block-order lesson)"\n'
    printf '  },\n'
    printf '  "status": "PLAN_ONLY"\n'
    printf '}\n'
  } > "${PLAN_JSON}"
  echo "plan.json emitted -> ${PLAN_JSON}"
  echo ""
  echo "DRY-RUN OK: ${total} (config, metal, seed) tuples PLANNED, no GPU work executed."
  echo "  CONFIRMED runs (real launch): ${total} (all 4 configs wired post-Wf-296C)"
  echo "  UNVERIFIED runs (real launch): 0 (all flags verified)"
  exit 0
fi

# Wf-378-A2: USER_OK gate moved BEFORE --dry-run block (see lines ~60-75)
# per Wf-366-A2 lesson. Old gate block at original lines 208-218 was
# reached only on real-launch path, so `--dry-run` silently bypassed it
# (HONEST-NEGATIVE per Wf-363-A2 audit). The new gate fires for BOTH
# --dry-run and real-launch paths, so this section is only reached on
# the real-launch path (USER_OK=1 already verified at line ~70).
echo "[Wf-378-A2] USER_OK set — proceeding with 60-tuple launch (G6 wall ~7.5 hr)" | tee -a "${LOG}"

# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
# Wf-296C: --no-enforce-budget added to wave_budget_tracker so all 60 tuples
# stay PLANNED (per kill_then_adapt directive, mirrors G5 §6 header convention).
# Wf-356-A1: wf343c3 data quality check inserted between pre-flight and
# per-wave checklist — script is loader-agnostic (covers all 5 R20 loaders
# internally: load_binding_db, load_nci60, load_drugood, metalcytotox_v4,
# platinai_4cell). Does NOT accept --data-csv; only --out-dir.
# Use launcher-specific out-dir so each launcher's report is preserved.
# 2026-10-03: declare grid + budget so the per-wave runtime gate gets REAL
# inputs instead of SKIPping. 5 metals x 3 seeds = 15 (metal, seed) pairs; the
# 4-config dimension makes 60 tuples, which the gate's (metal, seed) model does
# not express — see the PER_PAIR_BASELINE_HR note in wf343c2_per_wave_check.py.
${PY} molmetal/scripts/pre_flight_check.py \
  --pipeline launch \
  --declared-wave "Wf-124-P3-G6 60-tuple per-innovation ablation" \
  --pairs-metals "Ru,Ir,Os,Rh,Re" \
  --pairs-seeds "0,1,2" \
  --budget-hr 48 \
  || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf124_p3_g6_data_quality \
  || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G6" || { echo "FAIL: per_wave_checklist"; exit 1; }
${PY} molmetal/scripts/wave_budget_tracker.py --no-enforce-budget || { echo "FAIL: wave_budget_tracker"; exit 1; }

# Wf-A269 §5.11 (2026-11-04): fail-closed static-config audit BEFORE any
# tuple launches.  It compares the wrapper's own shell record (AUDIT_FLAGS)
# against what the trainer will actually read, dest by dest, through the
# trainer's own parser and the real loader, and adds each arm's lever to the
# configs whose FILE NAME carries that arm's token.  A drifted tuple config
# aborts the sweep.
echo "=== static per-tuple config audit (fail-closed) ==="
audit_static_configs || { echo "FAIL: static-config audit (fail-closed)"; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-04) — derive_pair_config() DELETED.  The launch-time
# derivation is exactly the mechanism the c1 contract forbids (a config
# written at launch time from a shell variable: "a launch-time-written
# config would make the run's hyperparameters unreviewable before the run and
# unreproducible afterwards").  The whole superseded region — its header
# comment and the function body — is preserved VERBATIM below (one preserved
# rendering, one marker) and it has 0 call sites (MEASURED 2026-11-04:
# grep -n derive_pair_config scripts/launch_wf124_p3_g6.sh returns only this
# comment block).  The static cohort above replaces it.
# ---------------------------------------------------------------------------
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269 wave-3, 2026-11-02 — the launch-time derivation this replaces):
# # Wf-A269 §5.11 migration (2026-11-03) — per-tuple YAML derivation.
# #
# # AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# # 训练和推理，no cli参数"): a launch wrapper may hand the trainer ONLY
# # --config / --out-dir / --ckpt-dir / --help. The four flags this wrapper
# # used to put on the trainer invocation line carry the PER-TUPLE cohort, and
# # the trainer loops over the metal cohort and the seed cohort INSIDE one
# # process — so the cohort cannot simply move into the shared yaml: one yaml
# # carrying 5 metals x 3 seeds would turn the 60 processes into 4 and change
# # the run.
# #
# # So every (ablation, metal, seed) tuple gets its OWN yaml, derived at launch
# # time from ${DMPNN_CONFIG} with EXACTLY the four values the CLI used to
# # carry:
# #   metals   <- data.metals                [metal]   (was: CLI --metals)
# #   seeds    <- data.seeds + protocol.seeds [seed]   (was: CLI --seeds)
# #   csv      <- csv (TOP-LEVEL key)        <path>   (was: CLI --csv)
# #   data_csv <- data.csv (flat form wins over the leaf form) <path>
# # The base yaml is NOT edited: configs/dmpnn.yaml is shared by 8 wrappers and
# # ${DMPNN_CONFIG} stays overridable. The per-ablation levers (CONFIRMED_FLAGS)
# # are the ablation dimension, not a per-pair cohort — they are written into
# # each tuple's own yaml, so the 4 arms stay 4 arms.
# # ---------------------------------------------------------------------------
# derive_pair_config() {
#   local metal=$1
#   local seed=$2
#   local dest=$3
#   shift 3
#   "${PY}" - "${DMPNN_CONFIG}" "${dest}" "${metal}" "${seed}" "${DATA_CSV}" "$@" <<'PYEOF'
# import os
# import pathlib
# import sys

# import yaml

# argv = sys.argv[1:]
# base_path, dest_path, metal, seed, csv_path = argv[:5]
# extra_flags = argv[5:]
# with open(base_path, "r", encoding="utf-8") as fh:
#     cfg = yaml.safe_load(fh) or {}
# if not isinstance(cfg, dict):
#     raise SystemExit("derive_pair_config: base config is not a mapping: %s" % base_path)


# def put(dotted, value):
#     parts = dotted.split(".")
#     node = cfg
#     for part in parts[:-1]:
#         nxt = node.get(part)
#         if not isinstance(nxt, dict):
#             nxt = {}
#             node[part] = nxt
#         node = nxt
#     node[parts[-1]] = value


# # Wf-A269 §5.11 (2026-11-03, user directive 2026-10-09 "no cli参数"): the
# # launcher used to hand these hyperparameters to the trainer as CLI flags
# # (COMMON_ARGS, plus the per-arm CONFIRMED_FLAGS lever).  A launch wrapper may
# # pass ONLY --config / --out-dir / --ckpt-dir / --help, so every one of them
# # is written into the per-tuple YAML instead.  The table below is the
# # flag -> (yaml path, value kind) map.  Every entry was checked against the
# # LIVE trainer parser
# # (molmetal/scripts/train_dmpnn_multitask_wasserstein.py): the yaml path
# # resolves to the argparse dest of the same name, and the value kind matches
# # that action's own type= (int / float / store_true / BooleanOptionalAction /
# # plain str).  The values are the ones the CLI carried, VERBATIM.
# FLAG_TO_YAML = {
#     "--epochs": ("train.epochs", "int"),
#     "--batch-size": ("train.batch_size", "int"),
#     "--hidden": ("model.hidden", "int"),
#     "--depth": ("model.depth", "int"),
#     "--lr": ("train.lr", "float"),
#     "--alpha": ("train.alpha", "float"),
#     "--wasserstein-weight": ("eval.wasserstein_weight", "float"),
#     "--use-l2-guard": ("eval.use_l2_guard", "on"),
#     "--ablation-config": ("eval.ablation_config", "str"),
#     "--w2-gradient-mode": ("eval.w2_gradient_mode", "str"),
#     "--n-mc": ("eval.n_mc", "int"),
#     "--device": ("compute.device", "str"),
#     "--cuda-visible-devices": ("compute.cuda_visible_devices", "str"),
#     "--cpu-num-workers": ("compute.cpu_num_workers", "int"),
#     "--data-csv": ("data.csv", "str"),
#     # The 4 G6 ablation levers (Wf-296C) — the arm dimension.
#     "--single-task": ("train.single_task", "on"),
#     "--no-kendall": ("train.kendall", "off"),
#     "--no-dann": ("eval.dann", "off"),
# }
# # Flags that carry NO value (store_true / BooleanOptionalAction spellings).
# VALUE_LESS = {"--use-l2-guard", "--single-task", "--no-kendall", "--no-dann"}


# def parse_flags(tokens):
#     out = []
#     i = 0
#     while i < len(tokens):
#         tok = tokens[i]
#         if not tok.startswith("--"):
#             raise SystemExit("derive_pair_config: unexpected token %r" % (tok,))
#         if tok in VALUE_LESS:
#             out.append((tok, None))
#             i += 1
#             continue
#         if i + 1 >= len(tokens):
#             raise SystemExit("derive_pair_config: flag %r has no value" % (tok,))
#         out.append((tok, tokens[i + 1]))
#         i += 2
#     return out


# put("data.metals", [metal])
# put("data.seeds", [int(seed)])
# put("protocol.seeds", [int(seed)])
# put("data.csv", csv_path)   # flat form wins -> dest data_csv
# put("csv", csv_path)        # top level -> dest csv (data.csv resolves to data_csv)

# for flag, raw in parse_flags(extra_flags):
#     if flag not in FLAG_TO_YAML:
#         raise SystemExit("derive_pair_config: unmapped flag %r" % (flag,))
#     path, kind = FLAG_TO_YAML[flag]
#     if kind == "int":
#         value = int(raw)
#     elif kind == "float":
#         value = float(raw)
#     elif kind == "on":
#         value = True
#     elif kind == "off":
#         value = False
#     else:
#         value = str(raw)
#     put(path, value)

# dest = pathlib.Path(dest_path)
# dest.parent.mkdir(parents=True, exist_ok=True)
# tmp = dest.with_name(dest.name + ".tmp")
# with open(tmp, "w", encoding="utf-8") as fh:
#     yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
# os.replace(tmp, dest)
# print(
#     "derive_pair_config: %s (base=%s metal=%s seed=%s flags=%s)"
#     % (dest, base_path, metal, seed, " ".join(extra_flags))
# )
# PYEOF
# }

# ---- REAL launch: per-tuple runner ----
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
  # Per Wf-347-B2: per-tuple pre-launch check (per-wave internal gate,
  # --skip-live since global pre_flight_check.py already covers VRAM + disk).
  PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-124-P3-G6/${cfg}/${metal}/seed=${seed}" \
    --pairs-metals "${metal}" --pairs-seeds "${seed}" \
    --skip-live \
    || { echo "=== $(date '+%H:%M:%S') FAIL ${cfg}/${metal}/seed=${seed} (per-wave check) ===" | tee -a "${LOG}"; return 1; }
  # GPU 0 (RTX PRO 6000 Blackwell) per the 2026-10-03 all-GPU-0
  # directive; this call site is serial with every other job.
  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
  #
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
  # (Wf-A269 wave-3, 2026-11-03 — the launch-time derivation this replaces;
  # derive_pair_config no longer exists, see the DELETED note above):
  # Wf-A269 §5.11 (2026-11-03): the per-tuple cohort and the corpus paths are
  # no longer CLI flags — they live in the per-tuple yaml derived below from
  # ${DMPNN_CONFIG} (derive_pair_config, above). The values the trainer reads
  # are unchanged.
  # Wf-A269 §5.11 (2026-11-04): the per-tuple cohort AND every hyperparameter
  # now live in a STATIC per-tuple config on disk; the trainer invocation
  # carries only the three §5.11-legal run-control flags.  The config is
  # copied into the run directory so the run's own artefacts record exactly
  # what was configured.
  local tuple_cfg
  tuple_cfg="$(pair_config_path "${cfg}" "${metal}" "${seed}")"
  if [[ ! -f "${tuple_cfg}" ]]; then
    echo "FAIL: static tuple config missing: ${tuple_cfg}" | tee -a "${LOG}"
    return 1
  fi
  install -m 0644 "${tuple_cfg}" "${OUT_BASE}/${cfg}/${metal}/seed${seed}/config.yaml"

  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
  # (Wf-A269 wave-3, 2026-11-02 — the launch-time derivation):
  #   local tuple_cfg="${OUT_BASE}/${cfg}/${metal}/seed${seed}/config.yaml"
  #   derive_pair_config "${metal}" "${seed}" "${tuple_cfg}" "${COMMON_ARGS[@]}" ${CONFIRMED_FLAGS[$cfg]}

  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY} ${SCRIPT} \
  #     --config "${DMPNN_CONFIG}" \
  #     --metals "${metal}" \
  #     --seeds "${seed}" \
  #     --csv "${DATA_CSV}" \
  #     --data-csv "${DATA_CSV}" \
  #     --out-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}" \
  #     --ckpt-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts" \
  #     ${CONFIRMED_FLAGS[$cfg]} \
  #     "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  # Wf-A269 §5.11 (2026-11-03): COMMON_ARGS and the per-arm CONFIRMED_FLAGS
  # lever are now written into ${tuple_cfg} by derive_pair_config, so the
  # trainer invocation line carries ONLY --config / --out-dir / --ckpt-dir.
  # Both arrays are kept VERBATIM — they are the single source of those
  # values, the dry-run display copy, and they are pinned by
  # molmetal/tests/test_wf337g2_launcher_v2.py + test_wf337g3_g6_regression.py.
  # The arm order is preserved: the lever is applied AFTER the common values
  # in the yaml, whereas the CLI order was the reverse (CONFIRMED_FLAGS first,
  # COMMON_ARGS last).  The two orders are equivalent HERE because the only
  # flag both could carry is --wasserstein-weight, and G6's COMMON_ARGS does
  # not carry it — so no value is contended and the resolved dest is the same
  # in both orders.
  #
  # Wf-A269 §5.11 (2026-11-04) — CORRECTION (additive + dated; the rendering
  # above is preserved VERBATIM): the launch-time derivation is GONE and the
  # static per-tuple config carries the whole effective namespace.  The arm
  # lever lives in THAT file (one lever per arm), COMMON_ARGS is the shell
  # record the audit compares against it, and the trainer invocation carries
  # ONLY --config / --out-dir / --ckpt-dir.  The trainer invocation line
  # below is UNCHANGED in shape — what changed is where the values come from.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY} ${SCRIPT} \
    --config "${tuple_cfg}" \
    --out-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${cfg}/${metal}/seed${seed}/ckpts" \
  2>&1 | tee "${pair_log}"
  echo "=== $(date '+%H:%M:%S') END ${cfg}/${metal}/seed=${seed} ===" | tee -a "${LOG}"
}

echo "=== Wf-124 P3 G6 START $(date) ===" | tee -a "${LOG}"
echo "60 tuples (4 configs x 5 metals x 3 seeds x 50 ep) -- 8 waves per config" | tee -a "${LOG}"
echo "CONFIRMED configs: ${VERIFIED_CONFIGS} (60 runs PLANNED post-Wf-296C)" | tee -a "${LOG}"
echo "UNVERIFIED configs: ${UNVERIFIED_CONFIGS:-<none>} (0 runs SKIPPED, all flags verified)" | tee -a "${LOG}"

# 60 (config, metal, seed) tuples enumerated:
# Loop scope is overridable so a reduced decisive subset can be run instead of
# the full 4x5x3 = 60 tuples (e.g. G6_CONFIGS=... G6_METALS=... G6_SEEDS=...).
for cfg in ${G6_CONFIGS:-no_multi_task no_wass no_kendall no_dann}; do
  for metal in ${G6_METALS:-Ru Ir Os Rh Re}; do
    for seed in ${G6_SEEDS:-0 1 2}; do
      run_tuple "${cfg}" "${metal}" "${seed}"
    done
  done
done

echo "=== ALL TODO-182 G6 TUPLES DONE $(date) ===" | tee -a "${LOG}"

# Aggregation — collect per_seed_metrics_seed*.json into training_summary.json
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
                runs.append({'config': cfg, 'metal': metal_dir.name, 'seed': seed_dir.name.replace(\"seed\", \"\"), 'metrics_path': str(p)})
out = {'todo_182_g6_total_tuples_planned': 60, 'tuples_aggregated': len(runs), 'runs': runs}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} confirmed tuples (expected up to 30 if --no-kendall/--no-dann skipped)')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"

# Per-lever attribution report stub (filled once sweep completes per TODO-182 §4)
echo "=== Per-lever attribution report placeholder at ${OUT_BASE}/ablation_report.json ===" | tee -a "${LOG}"
echo "TODO: aggregate attribution matrix once confirmed 30-run sweep completes (TODO-182 §3.4)" | tee -a "${LOG}"
