#!/bin/bash
# Wf-337-G1 — G6 55 PENDING + 28 FAILED tuple recovery launcher
#
# Context (MEASURED, Wf-330-B4 2026-09-29 13:23 CST):
#   - G6 launch (scripts/launch_wf124_p3_g6.sh, TODO-182) was started at
#     09:57:39 CST with stale args (--cpu-num-workers 0, no --no-kendall,
#     no --no-dann) before Wf-328-W2 + Wf-296C fixes hit the script on disk.
#   - Wf-328-W1 (torch.compile mode=reduce-overhead) triggered a
#     CUDAGraphs RuntimeError on the post-Wf-328 trainer code path
#     (Ru seed=2 + Ir seed=0 + 26 more tuples), see Wf-330-B4 §4.1.
#   - Bash does not re-read modified scripts → in-flight process keeps the
#     stale args in memory.
#
# Fixes applied (Wf-335-R1, Wf-335-R3, Wf-296C, Wf-328-W2):
#   - --cpu-num-workers 8                                  (Wf-328-W2)
#   - --no-cuda-graphs (hard-pin torch_compile_mode)        (Wf-335-R3)
#   - torch.compile mode default = max-autotune-no-cudagraphs (Wf-335-R1)
#   - --no-kendall / --no-dann per-config ablation levers   (Wf-296C)
#
# Status snapshot at 2026-09-29 15:13 CST (after scripts/recover_g6_failed_tuples.sh
# was written — see TODO/completed/337g1_g6_recovery.md):
#   - OK:    2 tuples  (no_multi_task/Ru/seed=0, seed=1)
#   - FAILED: 28 tuples (Ru/2 + Ir/0,1,2 + Os/0,1,2 + Rh/0,1,2 + Re/0,1,2 from
#              no_multi_task + all 15 from no_wass — all hit CUDAGraphs bug)
#   - RUNNING: 1 tuple  (no_kendall/Ru/seed=0, started 15:03:52)
#   - PENDING: 29 tuples (rest of no_kendall + all of no_dann)
#
# Recovery action:
#   - Skip OK tuples (status=OK in training_summary.json)
#   - Skip RUNNING tuples (training_summary.json missing but run.log mtime
#     < 30 min and status line absent)
#   - RE-LAUNCH FAILED + PENDING tuples with all Wf-328 + Wf-335 fixes wired.
#
# Wall-time estimate (PROJECTED, AGENTS.md §5.2 honest framing):
#   - 57 (FAILED + PENDING) × ~12 min/tuple = ~11.4 hr sequential on RTX 5090.
#   - Blocked on AGENTS.md §5.1 kill_then_adapt — USER_OK required.
#
# Usage:
#   bash scripts/recover_g6_failed_tuples.sh --dry-run
#     # CPU-only: enumerate failed/pending tuples, print manifest, no GPU.
#   USER_OK=1 bash scripts/recover_g6_failed_tuples.sh
#     # Real launch (post-§5.1 user authorization).
#   USER_OK=1 bash scripts/recover_g6_failed_tuples.sh --prereg docs/prereg/<file>.md
#     # Optional: pass a pre-registration record (Wf-350-B2 prereg.md template)
#     # Verifies hypothesis + primary_metric + baseline + expected_effect_size
#     # match the launch COMMON_ARGS before any GPU launch.
#     # If --prereg unset: emits WARNING that paper-grade launches should have one.
#
# Per party-twenty-spirit §3.5 honest framing:
#   - MEASURED (Wf-330-B4 §2): 2 OK + 2 FAIL + 1 in-flight + 55 PENDING at 13:23 CST.
#   - MEASURED (master.log mtime 15:03:52): 30 END markers emitted; only 2 OK.
#   - MEASURED (training_summary.json status field): 28 of 30 marked
#     status=ALL_SEEDS_FAILED with CUDAGraphs RuntimeError.
#   - PROJECTED: 11.4 hr wall for 57 tuples (BLOCKED on §5.1).
#   - PROJECTED: --cpu-num-workers 8 + --no-cuda-graphs eliminate the
#     CUDAGraphs regression (not yet bench-tested on RTX 5090 — fresh retrain
#     required to MEASURE).
#
# Per Wf-347-B2: per-tuple pre-launch check via wf343c2_per_wave_check.py
# (--skip-live since global pre_flight_check already covers VRAM/disk).
# Mirrors Wf-347-B1 G6 launcher integration at run_tuple().

set -euo pipefail
cd /home/hugo/codes/mol-metal

# Wf-363-A1: source the shared DRY_RUN.log helper (consistent per-pair schema).
# shellcheck source=./_dry_run_log_helper.sh
source "$(dirname "$0")/_dry_run_log_helper.sh"

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

# ---- Wf-366-A2 GAP-2 FIX: USER_OK gate BEFORE --dry-run block ----
# Per AGENTS.md §5.1 kill_then_adapt directive: refuse to run unless USER_OK=1
# is set in the environment. --dry-run does NOT bypass the gate (dry-run is
# still a launch path that emits DRY_RUN.log and enumerates recovery tuples).
# Reference pattern: scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
# (USER_OK gate at lines 115-122, exits 64 on refusal; --dry-run is checked
# later at line 364).
if [[ "${USER_OK:-0}" != "1" ]]; then
  echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
  echo "       USER_OK=1 bash scripts/recover_g6_failed_tuples.sh --dry-run   # plan only (still requires USER_OK=1)" >&2
  echo "       USER_OK=1 bash scripts/recover_g6_failed_tuples.sh              # actual recovery launch (~11.4 hr wall PROJECTED)" >&2
  echo "       USER_OK=1 bash scripts/recover_g6_failed_tuples.sh --prereg docs/prereg/<file>.md" >&2
  exit 64
fi
echo "[Wf-337-G1] USER_OK set — proceeding with recovery launch (~11.4 hr wall PROJECTED)"

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
    echo "WARN: prereg.md §3 hyperparameters table does not pin epochs=50 (matches recovery COMMON_ARGS)"
  fi
  echo "OK: prereg.md schema valid; hypothesis + metrics sections present"
  echo ""
else
  echo "WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md."
  echo ""
fi

# Wf-A269 §5.11 migration (2026-11-04): OUT_BASE is overridable so that the
# dry-run audit can be exercised against a scratch tree.  The tracked
# results/wf124_p3_per_innovation_ablation_g6/ dir holds 76 published
# artefacts for this wave and the dry-run must not write into it.
OUT_BASE="${WF337_G1_OUT_BASE:-results/wf124_p3_per_innovation_ablation_g6}"
LOG="${OUT_BASE}/master.log"
MANIFEST="${OUT_BASE}/recovery_manifest.tsv"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
GPU="cuda:0"
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
# (Wf-A269, 2026-11-04 — COMMON_ARGS used to be the CLI array the trainer
# RECEIVED; it is now only the shell record, see below):
#   # 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
#   # The trainer already accepts --config and ends parse_args() in
#   # apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322), so the
#   # yaml is the base and the per-call flags below override it.
#   # Precedence: explicit CLI flag > YAML > argparse default.
#   #
#   # 2026-11-03 (AGENTS.md §5.11 migration): the trainer is now invoked with
#   # --config / --out-dir / --ckpt-dir ONLY. Every hyperparameter this wrapper
#   # used to pass on the trainer command line now lives in a DEDICATED config —
#   # configs/recover_g6_failed_tuples_wf337g1.yaml — because configs/dmpnn.yaml
#   # is SHARED by 8 launchers and could not carry this wrapper's numbers.
#   #
#   # OPERATOR-OVERRIDE SEMANTICS CHANGED: DMPNN_CONFIG still selects the base
#   # config, but it can no longer be PARTIALLY overridden from the command line —
#   # §5.11 removed the command line. With DMPNN_CONFIG unset (the default, and
#   # the only way this wrapper is launched in the repo) the derived config is
#   # behaviourally identical to the pre-migration launch: MEASURED by a full
#   # argparse-namespace diff (old command line vs new --config), 0 differing
#   # dests. Setting DMPNN_CONFIG now REPLACES this wrapper's config wholesale —
#   # and a replacement that does not carry this wrapper's pinned values is
#   # REFUSED by derive_tuple_config() rather than silently run.
#   DMPNN_CONFIG="${DMPNN_CONFIG:-configs/recover_g6_failed_tuples_wf337g1.yaml}"
# ---------------------------------------------------------------------------
PAIR_CONFIG_PREFIX="${PAIR_CONFIG_PREFIX:-configs/wfa269_c1_dmpnn_pair_sweep_recover_g6_}"
# 4 arms x 5 metals x 3 seeds — the launcher MUST NOT silently under-launch.
PAIR_CONFIG_COUNT=60
# 5 metals x 3 seeds — the per-arm audit's own floor.
ARM_TUPLE_COUNT=15

# COMMON_ARGS is NO LONGER passed to the trainer.  It is now the wrapper's
# SHELL RECORD of the hyperparameters — the second, independent statement of
# intent that scripts/_wf_yaml_static_config_c1.py checks against the STATIC
# per-tuple YAML before the sweep (audit_static_configs, below).  The values
# the trainer actually reads live in
# configs/wfa269_c1_dmpnn_pair_sweep_recover_g6_<arm>_<metal>_seed<n>.yaml.
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
)

# ---- Per Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes ----
# DO NOT REMOVE the Wf-328-W2 banner comment: gates the Wf-326B revert regression.
# DO NOT REMOVE the --no-cuda-graphs flag: gates Wf-330-B4 CUDAGraphs regression.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   COMMON_ARGS=(
#     --epochs 50
#     --batch-size 256
#     --hidden 300
#     --depth 3
#     --lr 1e-3
#     --alpha 0.5
#     --w2-gradient-mode sliced
#     --n-mc 256
#     --use-l2-guard
#     --device "${GPU}"
#     --cuda-visible-devices 0
#     # Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
#     --cpu-num-workers 8
#     # REMOVED 2026-10-27: `--num-workers 8` and `--no-cuda-graphs`.
#     #
#     # Neither is declared in train_dmpnn_multitask_wasserstein.py's argparse, so
#     # every recovery tuple exited rc=2 "unrecognized arguments" before an epoch
#     # ran — the recovery path for the Wf-337-G1 G6 failures could not recover
#     # anything. MEASURED against parse_args: `--cpu-num-workers` ACCEPTED,
#     # `--num-workers` REJECTED, `--no-cuda-graphs` REJECTED.
#     #
#     # The Wf-328-W6 loader factory that `--num-workers` was meant to reach is not
#     # reachable from this trainer at all; `--cpu-num-workers` is its only worker
#     # flag. CUDA graphs only exist behind `--torch-compile`, which this launcher
#     # does not pass, so the Wf-335-R3 lever is closed structurally.
#     --data-csv "${DATA_CSV}"
#   )
#
# Per TODO-182 §3.4 + Wf-296C ablation flag dispatcher.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   declare -A CONFIRMED_FLAGS=(
#     ["no_multi_task"]="--single-task"
#     ["no_wass"]="--wasserstein-weight 0.0"
#     # Per Wf-296C: --no-kendall / --no-dann per-config ablation levers
#     ["no_kendall"]="--no-kendall"
#     ["no_dann"]="--no-dann"
#   )
#
# 2026-11-03 (AGENTS.md §5.11): the lever each arm applies is no longer a CLI
# flag — it is the YAML key the derived per-tuple config writes. The DISPATCH
# KEYS are unchanged (no_multi_task / no_wass / no_kendall / no_dann) and so
# is the ablation each arm performs; only the mechanism moved from the command
# line into the config. The lever table itself lives in derive_tuple_config()
# below, which is the single source of truth for cfg -> YAML write.
#
# Wf-A269 §5.11 (2026-11-04): the launch-time derivation is GONE.  The lever
# table below is now the wrapper's live SHELL RECORD of each arm's ablation —
# the same knob, the same arm, audited against the STATIC per-tuple config by
# scripts/_wf_yaml_static_config_c1.py (audit_static_configs).  The DISPATCH
# KEYS and the ablations they perform are unchanged.
declare -A ARM_LEVER_FLAG=(
  ["no_multi_task"]="--single-task"
  ["no_wass"]="--wasserstein-weight 0.0"
  ["no_kendall"]="--no-kendall"
  ["no_dann"]="--no-dann"
)
# Ordered CFGS — the dispatch order (bash associative arrays do not preserve
# order, so the iteration order is kept as a plain array).
CONFIRMED_FLAGS_ORDER=(no_multi_task no_wass no_kendall no_dann)

# ---- Launcher-side worker-count PIN (NOT passed on the command line) -------
# AGENTS.md §5.11 moved the worker count out of the trainer command line and
# into the YAML (compute.cpu_num_workers). The launcher keeps its own copy of
# the number because it is a Wf-326B/R2 memory-safe requirement (24 physical
# cores -> OPTIMAL_NUM_WORKERS=8) and a YAML that silently dropped it would
# put the trainer back on the manual-loop fallback path (num_workers=0, no
# prefetch, no pin_memory) with no error anywhere.
#
# Wf-A269 §5.11 (2026-11-04): derive_tuple_config() is GONE (no launch-time
# derivation).  audit_static_configs() now FAILS-CLOSED on the comparison —
# it checks the pin against the shell record (COMMON_ARGS) AND the shell
# record against the STATIC per-tuple YAML, so the YAML and this pin cannot
# drift.
CPU_NUM_WORKERS_PIN=(
  --cpu-num-workers 8
)

# ---- Static per-tuple config lookup + fail-closed audit (AGENTS.md §5.11) --
# The trainer loops over args.metals x args.seeds internally, so a
# per-(arm, metal, seed) PROCESS needs a config that declares exactly ONE
# metal, ONE seed, and the ONE ablation lever of that arm.  The 60 static
# files under ${PAIR_CONFIG_PREFIX} do exactly that, and they are ON DISK and
# IN GIT — a launch-time-written config would put the cohort back into a
# shell variable ("a knob nobody audits", §5.11's own rationale).
#
# FAIL-CLOSED: audit_static_configs() compares the wrapper's own shell record
# (COMMON_ARGS + the arm's lever) against what the trainer will ACTUALLY
# read, through the trainer's OWN parser and the REAL loader — dest by dest,
# value AND type.  A drifted config aborts the launch instead of quietly
# training something else.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — the launch-time derivation this replaces):
#   derive_tuple_config() {
#     local cfg=$1
#     local metal=$2
#     local seed=$3
#     local out=$4
#     ${PY} - "${DMPNN_CONFIG}" "${out}" "${cfg}" "${metal}" "${seed}" "${DATA_CSV}" "${CPU_NUM_WORKERS_PIN[*]}" <<'PYEOF'
#   import sys
#
#   import yaml
#
#   base, out, cfg, metal, seed, data_csv, workers_pin = sys.argv[1:8]
#   seed = int(seed)
#
#   with open(base, encoding="utf-8") as fh:
#       conf = yaml.safe_load(fh)
#
#   # The ablation lever each arm applies. The LEFT column is the CLI flag the
#   # arm used to pass, the RIGHT column is the YAML (section, key, value) it now
#   # writes — same knob, same arm, new mechanism (AGENTS.md §5.11).
#   LEVER_BY_CFG = dict(
#       no_multi_task=("--single-task", ("train", "single_task", True)),
#       no_wass=("--wasserstein-weight 0.0", ("eval", "wasserstein_weight", 0.0)),
#       no_kendall=("--no-kendall", ("train", "kendall", False)),
#       no_dann=("--no-dann", ("train", "dann", False)),
#   )
#   if cfg not in LEVER_BY_CFG:
#       raise SystemExit(f"UNKNOWN ablation cfg {cfg!r} — refusing to launch")
#   _, (section, key, value) = LEVER_BY_CFG[cfg]
#   ... (the PINNED table, the cohort narrowing, the lever write and the
#        FAIL-CLOSED assertions are all preserved in
#        molmetal/reports/wfa269_yaml_migration/recover_g6_failed_tuples.before.sh)
#   PYEOF
#   }
#
# WHY the derivation was replaced: it wrote the config at launch time from
# ${DMPNN_CONFIG} — a config SHARED by 8 wrappers — so the tuple coordinates
# and every hyperparameter existed only as shell variables at review time.
# The static cohort removes both dependencies: every value the trainer reads
# is in the file named by the tuple, and that file's header records the
# pre-migration argv it reproduces.
pair_config_path() {
  local cfg=$1
  local metal=$2
  local seed=$3
  printf '%s%s_%s_seed%s.yaml' "${PAIR_CONFIG_PREFIX}" "${cfg}" "${metal}" "${seed}"
}

audit_static_configs() {
  local arm flag expect
  local audit_dir
  local -a pids=() labels=() results=()
  local i label rc=0

  # The launcher-side worker-count PIN and the shell record must agree.
  if [[ "${COMMON_ARGS[*]}" != *"${CPU_NUM_WORKERS_PIN[*]}"* ]]; then
    echo "[static-config] FAIL: COMMON_ARGS does not carry ${CPU_NUM_WORKERS_PIN[*]}" >&2
    return 1
  fi

  audit_dir="$(mktemp -d "${TMPDIR:-/tmp}/wfa269_c1_static_audit.XXXXXX")" || return 1

  # The 5 auditor jobs (1 cohort + 4 arms) run CONCURRENTLY: each is a
  # separate process that imports the trainer (and therefore torch) once, so
  # running them in sequence costs ~21 s.  In parallel the wall time is the
  # slowest single job.  Their stdout is captured per job and summarised
  # below; a failing job's full log is dumped to stderr.
  #
  # (1) whole cohort: 60 static configs, each carrying the common
  #     hyperparameters AND naming its own (metal, seed) tuple.
  "${PY}" scripts/_wf_yaml_static_config_c1.py \
    --trainer "${SCRIPT}" \
    --config-glob "${PAIR_CONFIG_PREFIX}*.yaml" \
    --flags "${COMMON_ARGS[*]}" \
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
  for arm in "${CONFIRMED_FLAGS_ORDER[@]}"; do
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
      --flags "${COMMON_ARGS[*]} ${flag}" \
      --expect "${expect}" \
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

# Ordered versions of METALS + SEEDS for iteration order in the dry-run
# per-pair enumeration (Wf-366-A2 GAP-2 fix: --dry-run now emits a
# per-pair DRY_RUN.log). Bash associative arrays do not preserve order,
# so we keep the METALS/SEEDS list as plain arrays here too.
# (CONFIRMED_FLAGS_ORDER now lives in the static-config block above, next to
# the arm-lever map it dispatches through.)
METALS_ORDER=(Ru Ir Os Rh Re)
SEEDS_ORDER=(0 1 2)

# ---- Step 1: enumerate (cfg, metal, seed) tuples + classify ----
echo "=== Wf-337-G1 G6 RECOVERY START $(date) ===" | tee -a "${LOG}"
echo "OUT_BASE: ${OUT_BASE}" | tee -a "${LOG}"

# Python helper: scan training_summary.json status fields + run.log mtime
# Writes manifest TSV with one tuple per line (cfg<TAB>metal<TAB>seed<TAB>status).
# Status field is nested inside per-seed metrics (e.g. data["seeds"][0]["status"]);
# we recursively walk to find the leaf "status" string.
${PY} - "${OUT_BASE}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json
import os
import sys
import time
from pathlib import Path

OUT_BASE = Path(sys.argv[1])
CFGS = ["no_multi_task", "no_wass", "no_kendall", "no_dann"]
METALS = ["Ru", "Ir", "Os", "Rh", "Re"]
SEEDS = [0, 1, 2]

# Stale threshold for RUNNING detection: run.log mtime < 30 min and no summary.
STALE_RUN_MIN = 30


def _deep_find_status(obj):
    """Recursively find the deepest 'status' string in a JSON object."""
    if isinstance(obj, dict):
        if "status" in obj and isinstance(obj["status"], str):
            # Prefer the deepest match (leaf); keep recursing just in case.
            deeper = None
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    sub = _deep_find_status(v)
                    if sub is not None:
                        deeper = sub
                        break
            return deeper if deeper is not None else obj["status"]
        for v in obj.values():
            sub = _deep_find_status(v)
            if sub is not None:
                return sub
    elif isinstance(obj, list):
        for v in obj:
            sub = _deep_find_status(v)
            if sub is not None:
                return sub
    return None


manifest_lines = ["cfg\tmetal\tseed\tstatus\treason"]
counts = {"OK": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0}

now = time.time()
for cfg in CFGS:
    for metal in METALS:
        for seed in SEEDS:
            tupledir = OUT_BASE / cfg / metal / f"seed{seed}"
            summary = tupledir / "training_summary.json"
            runlog = tupledir / "run.log"
            if summary.exists():
                try:
                    data = json.loads(summary.read_text())
                    status = _deep_find_status(data) or "UNKNOWN"
                except Exception:
                    status = "CORRUPT_JSON"
                if status == "OK":
                    counts["OK"] += 1
                    manifest_lines.append(f"{cfg}\t{metal}\t{seed}\tOK\tstatus=OK in training_summary.json")
                else:
                    counts["FAILED"] += 1
                    manifest_lines.append(f"{cfg}\t{metal}\t{seed}\tFAILED\tstatus={status} in training_summary.json")
            elif runlog.exists():
                age_min = (now - runlog.stat().st_mtime) / 60.0
                if age_min < STALE_RUN_MIN:
                    counts["RUNNING"] += 1
                    manifest_lines.append(f"{cfg}\t{metal}\t{seed}\tRUNNING\trun.log mtime {age_min:.1f} min old")
                else:
                    counts["FAILED"] += 1
                    manifest_lines.append(f"{cfg}\t{metal}\t{seed}\tFAILED\trun.log mtime {age_min:.1f} min old, no summary")
            else:
                counts["PENDING"] += 1
                manifest_lines.append(f"{cfg}\t{metal}\t{seed}\tPENDING\tno run.log, no training_summary.json")

manifest_path = OUT_BASE / "recovery_manifest.tsv"
manifest_path.write_text("\n".join(manifest_lines) + "\n")
total = sum(counts.values())
print(f"RECOVERY MANIFEST: {manifest_path}")
print(f"  OK:      {counts['OK']:3d}")
print(f"  FAILED:  {counts['FAILED']:3d}")
print(f"  RUNNING: {counts['RUNNING']:3d}")
print(f"  PENDING: {counts['PENDING']:3d}")
print(f"  TOTAL:   {total:3d}  (expected 60)")
print(f"  TO_RECOVER (FAILED + PENDING): {counts['FAILED'] + counts['PENDING']}")
print()
print("Recovery launch will skip OK + RUNNING tuples and re-launch the rest with")
print(
    "Wf-328-W2 + Wf-335-R1+R3 + Wf-296C fixes wired (see the STATIC per-tuple "
    "config cohort configs/wfa269_c1_dmpnn_pair_sweep_recover_g6_*.yaml and "
    "this wrapper's banner)."
)
PYEOF

# ---- Wf-366-A2 GAP-2 FIX: --dry-run exit AFTER manifest enumeration ----
# --dry-run is checked AFTER the USER_OK gate (which is at line ~68, before
# manifest enumeration) AND AFTER the manifest is emitted, so the operator
# sees the recovery manifest + DRY_RUN.log without bypassing the safety gate.
# Reference pattern: scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
# line 364 (USER_OK=1 check at line 115, --dry-run plan emit at line 364).
if [[ "${DRY_RUN}" == "1" ]]; then
  echo ""
  echo "=== DRY-RUN OK (USER_OK=1 confirmed; plan only; no GPU work) ===" | tee -a "${LOG}"

  # Wf-363-A1: emit structured DRY_RUN.log with per-pair (cfg, metal, seed)
  # enumeration to ${OUT_BASE}/DRY_RUN.log, consistent with the 5 wrappers
  # polished in Wf-363-A1 (canonical schema: wrapper / mode / trainer /
  # pair / flags / gpu / budget / user_ok / ts / per_pair block).
  DRY_RUN_LOG="${OUT_BASE}/DRY_RUN.log"
  pair_args=()
  for cfg in "${CONFIRMED_FLAGS_ORDER[@]}"; do
    for metal in "${METALS_ORDER[@]}"; do
      for seed in 0 1 2; do
        pair_args+=( --pair "${cfg}/${metal} seed=${seed}" )
      done
    done
  done
  emit_dry_run_log "${DRY_RUN_LOG}" \
    --title "Wf-337-G1 G6 recovery (57 FAILED + PENDING tuples)" \
    --wrapper "recover_g6_failed_tuples.sh" \
    --mode "recovery_dry_run" \
    --trainer "${SCRIPT}" \
    --flags "--config <STATIC per-tuple config: ${PAIR_CONFIG_PREFIX}<cfg>_<metal>_seed<seed>.yaml> --out-dir ${OUT_BASE}/<cfg>/<metal>/seed<seed> --ckpt-dir ${OUT_BASE}/<cfg>/<metal>/seed<seed>/ckpts" \
    --gpu "${GPU}" \
    --budget-hr "11.4" \
    --user-ok "REQUIRED" \
    "${pair_args[@]}"
  echo "DRY_RUN.log emitted -> ${DRY_RUN_LOG} ($((${#pair_args[@]})) (cfg, metal, seed) pairs)" | tee -a "${LOG}"
  echo ""
  # Wf-A269 §5.11 (2026-11-04): the dry run must PROVE that the static
  # per-tuple configs exist and carry the hyperparameters this wrapper
  # intends.  The audit is fail-closed and CPU-only (it reads argparse + YAML
  # and never starts a trainer).
  echo "--- static per-tuple config audit (fail-closed, CPU-only) ---"
  abort_if_static_config_audit_fails
  echo ""

  echo "No GPU work performed. To actually launch the recovery:"
  echo "$ USER_OK=1 bash scripts/recover_g6_failed_tuples.sh"
  exit 0
fi

# ---- Step 2: launch each FAILED + PENDING tuple (skip OK + RUNNING) ----
run_tuple() {
  local cfg=$1
  local metal=$2
  local seed=$3
  local tupledir="${OUT_BASE}/${cfg}/${metal}/seed${seed}"
  local summary="${tupledir}/training_summary.json"
  local runlog="${tupledir}/run.log"

  # Skip OK (already complete)
  if [[ -f "${summary}" ]]; then
    s=$(grep -oE "\"status\": \"[A-Z_]+\"" "${summary}" 2>/dev/null | head -1 || echo "UNKNOWN")
    if [[ "${s}" == *"OK"* ]]; then
      echo "=== $(date '+%H:%M:%S') SKIP ${cfg}/${metal}/seed=${seed} (status=OK) ===" | tee -a "${LOG}"
      return 0
    fi
  fi

  # Skip RUNNING (recent run.log, no summary — leave it alone)
  if [[ ! -f "${summary}" && -f "${runlog}" ]]; then
    age_min=$(( ( $(date +%s) - $(stat -c %Y "${runlog}") ) / 60 ))
    if [[ "${age_min}" -lt 30 ]]; then
      echo "=== $(date '+%H:%M:%S') SKIP ${cfg}/${metal}/seed=${seed} (RUNNING, ${age_min} min old) ===" | tee -a "${LOG}"
      return 0
    fi
  fi

  mkdir -p "${tupledir}/ckpts"
  echo "=== $(date '+%H:%M:%S') RECOVER START ${cfg}/${metal}/seed=${seed} ===" | tee -a "${LOG}"
  # Per Wf-347-B2: per-tuple pre-launch check (per-wave internal gate,
  # --skip-live since global pre_flight_check.py already covers VRAM + disk).
  PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-337-G1-recovery/${cfg}/${metal}/seed=${seed}" \
    --pairs-metals "${metal}" --pairs-seeds "${seed}" \
    --skip-live \
    || { echo "=== $(date '+%H:%M:%S') FAIL ${cfg}/${metal}/seed=${seed} (per-wave check) ===" | tee -a "${LOG}"; return 1; }
  # GPU 0 (RTX PRO 6000 Blackwell) — pairs with --cuda-visible-devices 0
  # above; keep as-is (do not flip to the GPU-0 resolver default).
  #
  # Two mechanisms, two fixes, and neither covers the other:
  #   * `setsid --wait` detaches the trainer from this shell's session, so a
  #     recovery sweep that outlives its terminal is not SIGTERM'd mid-run
  #     (BLK-I1a). `--wait` is required: bare `setsid` may fork and return 0
  #     before the trainer starts, and `rc` below would then report success.
  #   * `set +o pipefail` for this one pipeline: DataLoader workers hold the
  #     pipe open after the trainer exits, tee takes EOF first, the trainer's
  #     next write takes SIGPIPE, and the propagated 141 aborts the sweep
  #     (BLK-I1b). PIPESTATUS[0] is only readable if errexit does not fire
  #     first, which is why the toggle is load-bearing.
  set +o pipefail
  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
  #
  # 2026-11-03 (AGENTS.md §5.11): the per-tuple config carries the cohort
  # (data.metals / data.seeds), the corpus path, the worker count and this
  # arm's ablation lever, so the trainer is invoked with --config / --out-dir
  # / --ckpt-dir only.
  #
  # Wf-A269 §5.11 (2026-11-04): the config is now a STATIC per-tuple file on
  # disk (pair_config_path), not one written at launch time.
  #
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait ${PY} ${SCRIPT} \
  #     --config "${DMPNN_CONFIG}" \
  #     --metals "${metal}" \
  #     --seeds "${seed}" \
  #     --data-csv "${DATA_CSV}" \
  #     --out-dir "${tupledir}" \
  #     --ckpt-dir "${tupledir}/ckpts" \
  #     ${CONFIRMED_FLAGS[$cfg]} \
  #     "${COMMON_ARGS[@]}" 2>&1 | tee "${tupledir}/run.log"
  # Wf-A269 §5.11 (2026-11-04): the per-tuple config is a STATIC file on
  # disk — there is NO launch-time derivation any more.  A missing config
  # aborts the tuple (fail-closed) rather than training the wrong cohort.
  local tuple_config
  tuple_config="$(pair_config_path "${cfg}" "${metal}" "${seed}")"
  if [[ ! -f "${tuple_config}" ]]; then
    echo "FAIL: static tuple config missing: ${tuple_config}" | tee -a "${LOG}"
    return 1
  fi
  # The config is copied into the run dir so the run's own artefacts record
  # exactly what was configured.
  install -m 0644 "${tuple_config}" "${tupledir}/config.yaml"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait ${PY} ${SCRIPT} \
    --config "${tuple_config}" \
    --out-dir "${tupledir}" \
    --ckpt-dir "${tupledir}/ckpts" 2>&1 | tee "${tupledir}/run.log"
  local rc=${PIPESTATUS[0]}
  set -o pipefail
  echo "=== $(date '+%H:%M:%S') RECOVER END ${cfg}/${metal}/seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  return "$rc"
}

# Wf-A269 §5.11 (2026-11-04): fail-closed static-config audit BEFORE any
# tuple launches.  It compares the wrapper's own shell record (COMMON_ARGS +
# each arm's lever) against what the trainer will actually read, dest by
# dest, through the trainer's own parser and the real loader.  A drifted
# pair config aborts the sweep.
echo "=== static per-tuple config audit (fail-closed) ==="
abort_if_static_config_audit_fails
echo ""

# Iterate in manifest order (FAILED first, then PENDING — same cfg order).
for cfg in no_multi_task no_wass no_kendall no_dann; do
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      run_tuple "${cfg}" "${metal}" "${seed}"
    done
  done
done

echo "=== ALL Wf-337-G1 G6 RECOVERY TUPLES DONE $(date) ===" | tee -a "${LOG}"
