#!/usr/bin/env bash
# Wf-338-F1 — VRAM-budgeted GPU smoke wrapper for W₂ sliced gradient mode
#
# =====================================================================
# Wf-338-F1 (2026-09-29) — GPU smoke for --w2-gradient-mode sliced
# =====================================================================
#
# Owner: Hugo Lee + ultracode agent
# Date: 2026-09-29
# Source: Wf-338-F1 ultracode residual-gap closure wave
#        (user directive 2026-09-29 verbatim "好像有些之前发现的问题没有被覆盖到,
#         可以额外启动新的ultracode吗?" honored at the Wf-338 layer —
#         extends Wf-337-U3 GPU smoke validation to the W₂ sliced
#         gradient-mode smoke target, which was deferred per Wf-328-W5
#         because the production default flipped to "sort" (MEASURED 2-3×
#         speedup on RTX PRO 6000 Blackwell sm_120)).
#
# PURPOSE
# -------
# Per Wf-328-W5: --w2-gradient-mode default flipped to "sort" (the 1-D
# closed-form ``torch.sort`` W₂ penalty path, MEASURED 2-3× speedup).
# However, the legacy ``sliced`` mode (256-projection random-direction
# Monte Carlo W₂ estimate) is still wired for backward-compat / paper
# §6 reproducibility audits (per AGENTS.md §5.2 honest framing).
#
# This wrapper is the smoke that:
#   - exercises the sliced mode on a tiny D-MPNN cohort (e.g. 100 ligands
#     × 1 seed × 1 epoch) so any torch + sliced regression is caught
#     BEFORE the production training of the CFM/D-MPNN retrain runs;
#   - propagates Wf-328 + Wf-335 fixes (CPU num-workers 8, TF32; CUDA Graphs
#     are structurally off because --torch-compile is not passed);
#   - enforces the kill_then_adapt USER_OK gate per AGENTS.md §5.1;
#   - reports a HONEST-NEGATIVE if the W₂ sliced path regresses on
#     Blackwell sm_120 (recorded for paper §6 reproducibility audits).
#
# Per AGENTS.md §5.1 standing GPU authorization ("显存够就放心大胆用"):
#   - The wrapper is USER_OK-gated (kill_then_adapt standing directive)
#   - VRAM budget check is enforced BEFORE the launch (>= 24 GB free)
#   - Wf-328 + Wf-335 fixes are propagated to the trainer THROUGH THE YAML
#     (AGENTS.md §5.11 — a launch wrapper may pass only --config / --out-dir
#     / --ckpt-dir / --help), from configs/wfa269_c5_gpu_smoke_w2.yaml:
#       * eval.w2_gradient_mode: sliced (this is the path under test)
#       * compute.cpu_num_workers: 8 (Wf-326B OPTIMAL_NUM_WORKERS)
#       * compute.tf32: true (Wf-328-W4)
#     Five flags this list used to advertise (--bf16, --num-workers,
#     --torch-compile-mode-v2, --no-cuda-graphs, --smoke) are not declared by
#     the trainer's argparse; see the removal note at the launch site.
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - MEASURED (2026-09-29 15:30): --w2-gradient-mode default is "sort";
#     sliced mode is wired but not default (paper §6 reproducibility only).
#   - MEASURED: --w2-gradient-mode sort MEASURED 2-3× W₂ step speedup
#     over sliced on RTX PRO 6000 Blackwell sm_120 (per Wf-326A quick wins
#     + Wf-328-W5 ship).
#
#   CORRECTION (2026-11-03, Wf-A269 cohort c5_gpu_smoke_w2) — the two lines
#   above are PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, but the first one
#   no longer matches the code:
#   - MEASURED (2026-11-03): the trainer's default is "sliced", NOT "sort" —
#     `DEFAULT_W2_GRADIENT_MODE: str = "sliced"`
#     (train_dmpnn_multitask_wasserstein.py:235) and argparse's default reads
#     that constant (:3222). The trainer's OWN history comment (:220-234)
#     records why: "Wf-328-W5 flipped this constant to 'sort' ... They did not
#     — both kept the historical 'sliced' literal, argparse kept
#     default='sliced', and every live config pins w2_gradient_mode: sliced.
#     The constant was therefore dead code carrying a false claim."
#   - CONSEQUENCE for this smoke: `--w2-gradient-mode sliced` is the DEFAULT
#     path, not an opt-in one, so this smoke exercises the default and its
#     "paper §6 reproducibility" framing above is stale. The migrated YAML
#     still states `eval.w2_gradient_mode: sliced` EXPLICITLY, so the run is
#     pinned to the sliced path regardless of any future default flip — that
#     is the value-preserving choice, not the prose's claim about defaults.
#   - The 2-3× sort-vs-sliced speedup claim above is NOT re-verified here
#     (0 GPU in this wave); it is left as the 2026-09-29 measurement it is.
#   - PROJECTED wall: ~5 min GPU on cuda:0 (100 ligands × 1 seed ×
#     1-epoch smoke run).
#   - HONEST_NEGATIVE_HISTORICAL: prior Wf-334 GPU smoke did NOT exercise
#     the sliced mode (only the sort default path). This wrapper closes
#     that gap.
#
#   SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#     - MEASURED (2026-09-29 15:30): --w2-gradient-mode default is "sort";
#       sliced mode is wired but not default (paper §6 reproducibility only).
#   CORRECTION (2026-11-04, MEASURED on this tree): the superseded rendering
#   above is MEASURED-FALSE — the shipped default is 'sliced', NOT 'sort'.
#   molmetal/scripts/train_dmpnn_multitask_wasserstein.py:235
#     DEFAULT_W2_GRADIENT_MODE: str = "sliced"
#   and argparse's --w2-gradient-mode default reads that constant (:3222).
#   MEASURED 2026-11-04:
#     DEFAULT_W2_GRADIENT_MODE = sliced
#     parse_args([]).w2_gradient_mode = sliced
#   REPRODUCE (CPU-only, 0 GPU):
#     CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -c "
#     import importlib.util, sys
#     spec = importlib.util.spec_from_file_location('t', 'molmetal/scripts/train_dmpnn_multitask_wasserstein.py')
#     m = importlib.util.module_from_spec(spec); sys.modules['t'] = m; spec.loader.exec_module(m)
#     sys.argv = ['t']; print(m.DEFAULT_W2_GRADIENT_MODE); print(m.parse_args([]).w2_gradient_mode)"
#   -> DEFAULT_W2_GRADIENT_MODE = sliced
#      parse_args([]).w2_gradient_mode = sliced
#   (The 2026-11-03 CORRECTION above carries the same finding and explains
#    WHY the constant was never flipped; it is left untouched. This block
#    re-states it under the §5.5 rule #6 marker and re-measures it.)
#
# USAGE
# -----
#   bash scripts/gpu_smoke_w2_sliced.sh --dry-run            # verify wrapper, no launch
#   bash scripts/gpu_smoke_w2_sliced.sh --status             # check VRAM + state
#   USER_OK=1 bash scripts/gpu_smoke_w2_sliced.sh            # full launch
#   USER_OK=1 bash scripts/gpu_smoke_w2_sliced.sh --watch    # watch + auto-launch
#
# EXIT CODES
# ----------
#   0 = PASS (smoke launched or VRAM verified free)
#   1 = FAIL (VRAM insufficient after timeout, or pre-flight FAIL)
#   2 = SKIP (sliced mode disabled, or Wf-326F smoke not enabled)
#   3 = USER_OK_GATE_DENIED (USER_OK=1 not set for real launch)

set -uo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PY="${PY:-/home/hugo/codes/mol-metal/.venv/bin/python}"
PROJECT_ROOT="/home/hugo/codes/mol-metal"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
# 2026-10-07 (user directive: configs in YAML, not CLI). Precedence:
# explicit CLI flag > YAML > argparse default.
# 2026-11-03 (AGENTS.md §5.11 migration): this smoke names its OWN config.
# `configs/dmpnn.yaml` is SHARED by 8 other launch wrappers, so the six flags
# the smoke used to pass on the command line (epochs 1, seeds [0], W₂ sliced,
# 8 CPU workers, TF32 on, CUDA device 0) live in a dedicated copy —
# configs/wfa269_c5_gpu_smoke_w2.yaml (Wf-A269 cohort c5_gpu_smoke_w2) —
# instead of on this command line. Editing
# the shared file would change the other 8 runs. Overriding SMOKE_CONFIG
# overrides the smoke's cohort.
#
# NAME HISTORY (AGENTS.md §5.5 rule #6 — append-only, never a silent edit):
# the dedicated copy was first written as `configs/gpu_smoke_w2_sliced.yaml`
# and was renamed to the Wf-A269 `configs/wfa269_<cohort-id>.yaml` convention
# on 2026-11-03. The superseded rendering is preserved VERBATIM at
#   molmetal/reports/wfa269_yaml_migration/gpu_smoke_w2_sliced.config.before.yaml
# and the superseded path still exists on disk at
#   configs/gpu_smoke_w2_sliced.yaml
# (inert — this wrapper points at the wfa269_ file).
SMOKE_CONFIG="${SMOKE_CONFIG:-${PROJECT_ROOT}/configs/wfa269_c5_gpu_smoke_w2.yaml}"
OUT_BASE="${PROJECT_ROOT}/results/wf338f1_w2_sliced_smoke"
LOG_DIR="${PROJECT_ROOT}/molmetal/reports/wf338f1_w2_sliced_smoke"
WRAPPER_LOG="${LOG_DIR}/wrapper.log"
STATE_FILE="${LOG_DIR}/state.json"

# VRAM budget threshold (MiB) — sliced W₂ GPU smoke needs ≥ 24 GB on cuda:0
VRAM_FREE_MIN_MIB=24576  # 24 GB
VRAM_POLL_INTERVAL_SEC=60
VRAM_MAX_WAIT_MIN=120    # wait up to 2 hr before declaring VRAM_BLOCKED

mkdir -p "${OUT_BASE}" "${LOG_DIR}"

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
DRY_RUN=0
STATUS_ONLY=0
WATCH_MODE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --status)    STATUS_ONLY=1; shift ;;
    --watch)     WATCH_MODE=1; shift ;;
    --vram-min)  VRAM_FREE_MIN_MIB="${2:-24576}"; shift 2 ;;
    -h|--help)   sed -n '2,60p' "$0"; exit 0 ;;
    *)           printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] [wf338f1] $*" | tee -a "${WRAPPER_LOG}"
}

# -----------------------------------------------------------------------------
# VRAM probe (parsed from nvidia-smi CSV)
# -----------------------------------------------------------------------------
probe_vram() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "VRAM_PROBE_FAIL nvidia-smi not available"
    return 1
  fi
  # Output: index, name, used_mib, total_mib (per-GPU row)
  nvidia-smi --query-gpu=index,name,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null
}

parse_vram_for_index() {
  local gpu_idx="${1:-0}"
  local probe_out
  probe_out=$(probe_vram) || { echo "0"; return 1; }
  echo "${probe_out}" | awk -F',' -v idx="${gpu_idx}" '
    $1 == idx {
      gsub(/ /, "", $3); gsub(/ /, "", $4);
      print $4 - $3;  # free_mib = total - used
    }'
}

# -----------------------------------------------------------------------------
# Pre-flight check (pre_flight_check.py --pipeline launch)
# -----------------------------------------------------------------------------
run_pre_flight() {
  log "Running pre_flight_check.py --pipeline launch"
  if "${PY}" "${SCRIPTS_DIR}/pre_flight_check.py" --pipeline all >/dev/null 2>&1; then
    log "PRE_FLIGHT PASS"
    return 0
  else
    log "PRE_FLIGHT FAIL — aborting (TODO-207 §1.4 standing rule)"
    return 1
  fi
}

# -----------------------------------------------------------------------------
# USER_OK gate (per mol-metal-gpu-execution §0 + Wf-306D pattern)
# -----------------------------------------------------------------------------
check_user_ok() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    return 0  # --dry-run is exempt
  fi
  if [ "${STATUS_ONLY:-0}" = "1" ]; then
    return 0  # --status is read-only, exempt
  fi
  if [ "${USER_OK:-0}" != "1" ]; then
    log "USER_OK_GATE_DENIED — set USER_OK=1 to authorise real launch"
    log "  Re-run with: USER_OK=1 bash $0"
    return 3
  fi
  return 0
}

# -----------------------------------------------------------------------------
# VRAM wait (polls every 60s, up to VRAM_MAX_WAIT_MIN)
# -----------------------------------------------------------------------------
wait_for_vram() {
  local gpu_idx="${1:-0}"
  local free_mib
  local waited_min=0
  while [ "${waited_min}" -lt "${VRAM_MAX_WAIT_MIN}" ]; do
    free_mib=$(parse_vram_for_index "${gpu_idx}" | head -1)
    if [ -z "${free_mib}" ] || [ "${free_mib}" = "0" ]; then
      log "VRAM_PROBE_FAIL for cuda:${gpu_idx}; sleeping 60s and retrying"
      sleep "${VRAM_POLL_INTERVAL_SEC}"
      waited_min=$((waited_min + 1))
      continue
    fi
    if [ "${free_mib}" -ge "${VRAM_FREE_MIN_MIB}" ]; then
      log "VRAM_OK cuda:${gpu_idx} free=${free_mib} MiB (threshold=${VRAM_FREE_MIN_MIB})"
      return 0
    fi
    log "VRAM_BUSY cuda:${gpu_idx} free=${free_mib} MiB < ${VRAM_FREE_MIN_MIB}; waiting 60s"
    sleep "${VRAM_POLL_INTERVAL_SEC}"
    waited_min=$((waited_min + 1))
  done
  log "VRAM_TIMEOUT cuda:${gpu_idx} still busy after ${VRAM_MAX_WAIT_MIN} min"
  return 1
}

# -----------------------------------------------------------------------------
# Smoke target: D-MPNN Wasserstein trainer with --w2-gradient-mode sliced
# -----------------------------------------------------------------------------
launch_w2_sliced_smoke() {
  local target_log="${LOG_DIR}/w2_sliced_smoke.log"
  log "Launching W₂ sliced GPU smoke on cuda:0"
  log "  trainer: molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
  log "  config: ${SMOKE_CONFIG}"
  log "  Wf-328+Wf-335 fixes (now declared in the YAML, AGENTS.md §5.11):"
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   log "    eval.w2_gradient_mode=sliced (this smoke — production default is 'sort' per Wf-328-W5)"
  # CORRECTION (2026-11-04, MEASURED on this tree): the shipped default is
  # 'sliced', NOT 'sort'. The trainer defines
  #   DEFAULT_W2_GRADIENT_MODE: str = "sliced"
  # at molmetal/scripts/train_dmpnn_multitask_wasserstein.py:235, and argparse's
  # --w2-gradient-mode default reads that constant (:3222). MEASURED:
  #   DEFAULT_W2_GRADIENT_MODE = sliced
  #   parse_args([]).w2_gradient_mode = sliced
  #   REPRODUCE (CPU-only, 0 GPU):
  #     CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -c "
  #     import importlib.util, sys
  #     spec = importlib.util.spec_from_file_location('t', 'molmetal/scripts/train_dmpnn_multitask_wasserstein.py')
  #     m = importlib.util.module_from_spec(spec); sys.modules['t'] = m; spec.loader.exec_module(m)
  #     sys.argv = ['t']; print(m.DEFAULT_W2_GRADIENT_MODE); print(m.parse_args([]).w2_gradient_mode)"
  #   -> DEFAULT_W2_GRADIENT_MODE = sliced
  #      parse_args([]).w2_gradient_mode = sliced
  log "    eval.w2_gradient_mode=sliced (this smoke — production default is 'sliced' per the trainer's DEFAULT_W2_GRADIENT_MODE)"
  log "    compute.cpu_num_workers=8 (Wf-328-W2; the trainer's only worker flag)"
  log "    compute.tf32=true (Wf-328-W4)"
  log "    cuda_visible_devices=\"0\" (top-level key; was --cuda-visible-devices 0)"
  log "    no --torch-compile, so no CUDA Graphs (the Wf-335-R3 lever is"
  log "      structural here, not a flag — see the removal note below)"
  log "  expected wall: ~5 min GPU cuda:0 (100 ligands × 1 seed × 1-epoch smoke)"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] USER_OK=1 ${PY} molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\" | tee -a "${target_log}"
    echo "    --config ${SMOKE_CONFIG} \\" | tee -a "${target_log}"
    echo "    --out-dir ${OUT_BASE}" | tee -a "${target_log}"
    echo "    (the six hyperparameters live in ${SMOKE_CONFIG}: epochs 1, seeds [0]," | tee -a "${target_log}"
    echo "     w2-gradient-mode sliced, cpu-num-workers 8, tf32 true, cuda-visible-devices \"0\")" | tee -a "${target_log}"
    return 0
  fi

  # Five flags were removed here on 2026-10-27 because the trainer does not
  # declare them: `--smoke`, `--num-workers`, `--torch-compile-mode-v2`,
  # `--no-cuda-graphs`, `--bf16`.  argparse rejected the FIRST one it met and
  # exited rc=2, so this smoke could not run at all — it printed a pid and then
  # died, which reads exactly like a launched job.
  #
  # MEASURED against train_dmpnn_multitask_wasserstein.parse_args:
  #   ACCEPTED  --cpu-num-workers, --tf32, --channels-last, --torch-compile,
  #             --w2-gradient-mode
  #   REJECTED  --num-workers, --torch-compile-mode-v2, --no-cuda-graphs,
  #             --bf16, --smoke
  #
  # What replaces each:
  #   --smoke        -> `--epochs 1 --seeds 0` is the smoke (already here).
  #   --num-workers  -> `--cpu-num-workers` is the trainer's only worker flag.
  #   --torch-compile-mode-v2 / --no-cuda-graphs -> CUDA graphs only exist
  #                     behind `--torch-compile`, which is NOT passed, so the
  #                     lever is already closed structurally.
  #   --bf16         -> not a CLI flag; BF16 is resolved by configure_bf16().
  #
  # 2026-11-03 (AGENTS.md §5.11): the six flags that REMAINED on this line
  # were moved into configs/wfa269_c5_gpu_smoke_w2.yaml. §5.11 permits only
  # --config / --out-dir / --ckpt-dir / --help on a trainer invocation, and
  # `scripts/_yaml_only_gate.py` flagged every one of them.
  #
  # MEASURED equivalence (FIRST migration, Wf-A269 prior wave): the trainer's
  # own parse_args was run twice — once
  # with the OLD command line and once with the NEW `--config` alone — and the
  # two namespaces were compared dest-by-dest: 0 differences, and the migrated
  # dests come back as the right TYPES (`epochs` int, `seeds` LIST [0],
  # `cuda_visible_devices` str "0", `w2_gradient_mode` str 'sliced',
  # `cpu_num_workers` int 8, `tf32` bool True).
  #
  # MEASURED equivalence (Wf-A269 cohort c5_gpu_smoke_w2, 2026-11-03 —
  # ADDENDUM, the prior claim above is PRESERVED VERBATIM per AGENTS.md §5.5
  # rule #6): scripts/_yaml_equivalence_probe.py re-run against the renamed
  # config found ONE dest that did NOT match: `out_dir`, because the YAML
  # carried the RELATIVE `results/wf338f1_w2_sliced_smoke` while the OLD
  # command line passed the ABSOLUTE
  #   /home/hugo/codes/mol-metal/results/wf338f1_w2_sliced_smoke
  # (the probe reported: `DIFF out_dir: old="PosixPath('/home/hugo/codes/
  # mol-metal/results/wf338f1_w2_sliced_smoke')" new="PosixPath('results/
  # wf338f1_w2_sliced_smoke')" — value differs`). The config now carries the
  # absolute path, and the probe reports PASS. The prior wave's "0
  # differences" claim was measured on a config whose `output.out_dir` still
  # had the shared file's value, so it did not cover this dest.
  #
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   (the OLD wrapper's `${DMPNN_CONFIG}` was configs/dmpnn.yaml)
  #   USER_OK=1 nohup "${PY}" "${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py" \
  #     --config "${DMPNN_CONFIG}" \
  #     --cuda-visible-devices 0 \
  #     --w2-gradient-mode sliced \
  #     --epochs 1 --seeds 0 \
  #     --cpu-num-workers 8 \
  #     --tf32 \
  #     --out-dir "${OUT_BASE}" \
  #     > "${target_log}" 2>&1 &
  #
  # AND the matching header bullet list, VERBATIM:
  #   - Wf-328 + Wf-335 fixes are propagated to the trainer:
  #       * --w2-gradient-mode sliced (this is the path under test)
  #       * --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS)
  #       * --tf32 (Wf-328-W4)
  #     Five flags this list used to advertise (--bf16, --num-workers,
  #     --torch-compile-mode-v2, --no-cuda-graphs, --smoke) are not declared by
  #     the trainer's argparse; see the removal note at the launch site.
  #
  # WHERE EACH FLAG WENT (configs/wfa269_c5_gpu_smoke_w2.yaml):
  #   --cuda-visible-devices 0 -> cuda_visible_devices: "0"   (top-level key)
  #   --w2-gradient-mode sliced -> eval.w2_gradient_mode: sliced
  #   --epochs 1                -> train.epochs: 1
  #   --seeds 0                 -> data.seeds: [0] + protocol.seeds: [0]
  #   --cpu-num-workers 8       -> compute.cpu_num_workers: 8
  #   --tf32                    -> compute.tf32: true
  USER_OK=1 nohup "${PY}" "${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py" \
    --config "${SMOKE_CONFIG}" \
    --out-dir "${OUT_BASE}" \
    > "${target_log}" 2>&1 &
  local pid=$!
  log "  W₂ sliced smoke launched pid=${pid}; log=${target_log}"
  echo "${pid}" > "${LOG_DIR}/w2_sliced.pid"
  return 0
}

# -----------------------------------------------------------------------------
# Status query (USER_OK-exempt)
# -----------------------------------------------------------------------------
print_status() {
  echo "=== Wf-338-F1 W₂ sliced GPU smoke wrapper status ==="
  echo "Date: $(date -Iseconds)"
  echo "VRAM threshold: ${VRAM_FREE_MIN_MIB} MiB"
  echo ""
  echo "VRAM probe:"
  probe_vram | awk -F',' 'BEGIN{printf "  %-3s %-30s %10s %10s %10s\n", "GPU", "Name", "UsedMiB", "TotalMiB", "FreeMiB"} {used=$3; gsub(/ /,"",used); total=$4; gsub(/ /,"",total); free=total-used; printf "  %-3s %-30s %10s %10s %10s\n", $1, $2, used, total, free}'
  echo ""
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   echo "W₂ gradient mode default (per Wf-328-W5): sort"
  # CORRECTION (2026-11-04, MEASURED on this tree): the default is 'sliced',
  # NOT 'sort' — molmetal/scripts/train_dmpnn_multitask_wasserstein.py:235
  #   DEFAULT_W2_GRADIENT_MODE: str = "sliced"
  # (argparse's --w2-gradient-mode default reads that constant, :3222).
  echo "W₂ gradient mode default (per the trainer's DEFAULT_W2_GRADIENT_MODE): sliced"
  echo "W₂ gradient mode under smoke (this wrapper): sliced"
  echo ""
  echo "Recent wrapper log (last 15 lines):"
  if [ -f "${WRAPPER_LOG}" ]; then
    tail -15 "${WRAPPER_LOG}"
  else
    echo "  (no log yet)"
  fi
}

# -----------------------------------------------------------------------------
# Main dispatch
# -----------------------------------------------------------------------------
main() {
  log "=== Wf-338-F1 W₂ sliced GPU smoke wrapper START ==="
  log "DRY_RUN=${DRY_RUN} WATCH_MODE=${WATCH_MODE}"
  log "USER_OK=${USER_OK:-<unset>}"

  if [ "${STATUS_ONLY:-0}" = "1" ]; then
    print_status
    exit 0
  fi

  if ! check_user_ok; then
    exit 3
  fi

  # Pre-flight (CPU-only) before any real launch
  if [ "${DRY_RUN:-0}" != "1" ]; then
    run_pre_flight || { log "ABORT: pre-flight FAIL"; exit 1; }
  fi

  # Initial VRAM probe
  local vram_free
  vram_free=$(parse_vram_for_index 1 | head -1)
  log "Initial VRAM probe: cuda:0 free=${vram_free:-?} MiB"

  # Watch mode: wait until VRAM is free, then launch
  if [ "${WATCH_MODE:-0}" = "1" ] || [ "${DRY_RUN:-0}" != "1" ]; then
    if ! wait_for_vram 1; then
      log "ABORT: VRAM not free on cuda:0 after ${VRAM_MAX_WAIT_MIN} min"
      exit 1
    fi
  fi

  # Launch target
  local rc_w2_sliced=0
  launch_w2_sliced_smoke || rc_w2_sliced=$?

  # Aggregate verdict
  log "=== Wf-338-F1 launch summary ==="
  log "  w2_sliced_rc=${rc_w2_sliced}"

  # Persist state
  cat > "${STATE_FILE}" <<EOF
{
  "wf338f1_state": {
    "date": "$(date -Iseconds)",
    "dry_run": ${DRY_RUN},
    "watch_mode": ${WATCH_MODE},
    "vram_free_mib_initial": ${vram_free:-0},
    "vram_threshold_mib": ${VRAM_FREE_MIN_MIB},
    "w2_sliced_rc": ${rc_w2_sliced},
    "user_ok": "${USER_OK:-<unset>}",
    "production_default": "sliced",
    "production_default_superseded_rendering": "sort",
    "production_default_correction": "SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6: the emitted value was 'sort' (the Wf-328-W5 claim). MEASURED 2026-11-04: the shipped default is 'sliced' - molmetal/scripts/train_dmpnn_multitask_wasserstein.py:235 DEFAULT_W2_GRADIENT_MODE = 'sliced', and argparse's --w2-gradient-mode default reads that constant (:3222).",
    "smoke_target": "sliced"
  }
}
EOF
  log "State persisted to ${STATE_FILE}"
  log "=== Wf-338-F1 wrapper COMPLETE ==="

  exit "${rc_w2_sliced}"
}

main
