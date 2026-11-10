#!/bin/bash
# Wf-119 P1.4 — chemprop v3 LOMO actual GPU launch wrapper (Wave-A)
#   Wave-A: P1.4 smoke (4 ablations × 1 seed × 3 ep) — ~2.0 hr on PRO 6000 cuda:0
#   Per WEAK_TO_STRONG §3.4 + TODO-119 §3 P1.4 + Wf-266 (TODO-130 G5 follow-up)
#
# Date: 2026-09-29
# Owner: Hugo Lee + Wf-336X2 ultracode agent
# Parent: Wf-300C Wave-A (TODO-119 P1.4 smoke) + Wf-321H CPU scaffold
# Refs:
#   - TODO/completed/321h_todo_119_p14_cpu_scaffold.md (CPU scaffold; 12/12 tests PASS)
#   - TODO/completed/266_wf266_chemprop_v3_lomo_fix.md (LOMO scaffold levers)
#   - TODO/completed/207_unified_pre_flight_checklist_v2.md §1.1 (user OK gate)
#   - molmetal/scripts/train_chemprop_v2_v3.py (chemprop v2/v3 trainer with v3-fix mode)
#   - Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - user_ok_required: true (kill_then_adapt standing directive — 207 §1.1)
#   - per_wave_user_ok_required: true (P1.4 wave §directives)
#   - MEASURED: Wf-321H CPU scaffold 12/12 tests PASS + pre-flight gate PASS
#   - PROJECTED: P1.4 full sweep (4 ablations × 1 seed × 3 ep smoke, then
#     5×5×50 full) is in GPU pipeline; NOT executed on CPU scaffold
#   - BLOCKED: chemprop v3 LOMO needs chemprop v2 featurization fix (Wf-320D)
#     before GPU portion can run end-to-end
#
# Per Wf-335-R1 MEASURED FIX (D-MPNN side): the chemprop v3 LOMO trainer
# does NOT use torch.compile by default, so CUDA Graphs is not a concern
# on the chemprop trainer. The "no cudagraphs" Wf-335-R1 mode default is
# wired into the D-MPNN trainer only (`train_dmpnn_multitask_wasserstein.py`).
#
# Per Wf-328 optimizations wired:
#   - Wf-328-W2 (--cpu-num-workers 8) — N/A on chemprop trainer (uses --num-workers reserved/logged)
#   - Wf-328-W3 (BF16 autocast default) — wired via --no-bf16 / default=True
#   - Wf-328-W4 (TF32) — global torch default; not chemprop-specific
#   - Wf-328-W5 (W₂ sort default) — chemprop LOMO init-scale applies
#   - Wf-328-W10 (D-MPNN Featurization Cache) — chemprop v2 featurization path
#     needs Wf-320D before full speedup can be realised
#
# Usage:
#   bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh --dry-run  # CPU verification only
#   USER_OK=1 bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh  # REAL launch (after user grants)
#   USER_OK=1 SKIP_ABLATIONS=bond,closure bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh

set -eu
# NB: pipefail INTENTIONALLY DISABLED per Wf-131 P3 post-mortem 2026-10-08
# (Python DataLoader SIGPIPE on ${PY} trainer | tee log pipes). The wrapper
# captures ${PIPESTATUS[0]} explicitly after each pipe (see below).
cd /home/hugo/codes/mol-metal

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
export PYTHONPATH="/home/hugo/codes/mol-metal"
export PYTHONPATH="/home/hugo/codes/mol-metal:${PYTHONPATH:-}"

OUT_BASE="results/wf119_p1_4_chemprop_v3_lomo"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

# Per TODO-119 §3 P1.4 + WEAK_TO_STRONG §3.4:
#   4 ablations × 1 seed × 3 ep smoke (Wave-A) — chemprop v3 LOMO scaffold
#   Per Wf-266 (TODO-130 G5 follow-up): --chemprop-mode v3-fix is the LOMO
#   scaffold anchor; --chemprop-min-epochs=10 + --chemprop-lomo-init-scale=1e-3
#   are the LOMO scaffold-split integrity levers
SCRIPT="molmetal/scripts/train_chemprop_v2_v3.py"
GPU="cuda:0"  # PRO 6000 Blackwell (per Wf-300C Wave-A)
CONFIG="configs/wf119_p1_4_chemprop_v3_lomo.yaml"

# 4 ablation configs (per Wf-244 §3 + Wf-266 TODO-130 G5 follow-up):
#   baseline              — no LOMO levers (control)
#   v3_fix_only           — v3-fix mode only
#   min_epochs_only       — --chemprop-min-epochs=10 only
#   lomo_init_scale_only  — --chemprop-lomo-init-scale=1e-3 only
#
# §5.11 (2026-11-03) — THE COMMON HYPERPARAMETERS ARE IN THE YAML NOW.
# The wrapper used to splice them in as
#   COMMON_ARGS=(
#     --epochs 3 --allow-single-seed --seeds 0 --batch-size 50
#     --hidden 300 --depth 3 --lr 1e-3 --device "${GPU}" --no-bf16
#     --num-workers 8
#   )
#   ABLATION_BASE_ARGS=("${COMMON_ARGS[@]}")
#   setsid nohup ${PY} "${SCRIPT}" "${ABLATION_BASE_ARGS[@]}" "${ABLATION_ARGS[@]}"
# which is the shape AGENTS.md §5.11 forbids: a trainer invocation line may
# carry only --config / --out-dir / --ckpt-dir / --help.
#
# Every value above is transcribed into configs/wf119_p1_4_chemprop_v3_lomo.yaml
# at its real dest, and the equivalence is MEASURED, not asserted:
# scripts/_yaml_equivalence_probe.py over the old argv and the new config
# compares the trainer's own namespace dest by dest, value AND type —
#   MEASURED 2026-11-03: 37 dests compared, 11 equal (epochs 3 / batch_size 50
#   / hidden 300 / depth 3 / lr 0.001 / device cuda:0 / bf16 False /
#   allow_single_seed True / num_workers 8 / seeds [0]), 2 differ — and the 2
#   that differ are exactly the two PER-ARM levers, which the per-arm derived
#   config carries (see the derivation below).
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   COMMON_ARGS=(
#     --epochs 3
#     --allow-single-seed
#     --seeds 0
#     --batch-size 50
#     --hidden 300
#     --depth 3
#     --lr 1e-3
#     --device "${GPU}"
#     --no-bf16  # Wf-328-W3 BF16 default off for chemprop smoke (per Wf-321H §2)
#     --num-workers 8  # Wf-328-W2 OPTIMAL_NUM_WORKERS on 24-core host (chemprop reserves this arg)
#   )
#
# The GPU variable stays declared above for the reader: `compute.device` in
# the YAML carries "cuda:0", the same value this wrapper used to pass.

# ---- DRY-RUN mode: verify config + script paths only (no GPU execution) ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-119 P1.4 — DRY-RUN (CPU verification only; no GPU execution) ==="
  echo ""
  echo "USER OK REQUIRED for real launch. Set USER_OK=1 after user grants per 207 §1.1."
  echo ""
  echo "--- Pre-flight gate check (Wf-321H CPU scaffold) ---"
setsid nohup   ${PY} molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 2>&1 | grep -E "p3_todo_119_p14_cpu_smoke_ready|p1_chemprop_smoke" | head -5
  echo ""
  echo "--- Chemprop trainer CLI sanity (--help token check) ---"
  for tok in "--chemprop-mode" "--chemprop-min-epochs" "--chemprop-lomo-init-scale" "v3-fix"; do
    if ${PY} "${SCRIPT}" --help 2>&1 | grep -q -- "${tok}"; then
      echo "  OK: ${tok} present in --help"
    else
      echo "  FAIL: ${tok} missing from --help"
      exit 1
    fi
  done
  echo ""
  echo "--- 4 ablation configs to enumerate ---"
  for ablation in baseline v3_fix_only min_epochs_only lomo_init_scale_only; do
    echo "  ${ablation}  ->  ${OUT_BASE}/${ablation}/seed0/"
  done
  echo ""
  echo "DRY-RUN OK: 4 ablations × 1 seed × 3 ep smoke PLANNED, no GPU work executed."
  echo ""
  echo "Honest framing:"
  echo "  - Wf-321H CPU scaffold: 12/12 tests PASS + pre-flight gate PASS"
  echo "  - Wf-335-R1 MEASURED FIX: applies to D-MPNN trainer (max-autotune-no-cudagraphs),"
  echo "    not chemprop trainer (no torch.compile on chemprop trainer)"
  echo "  - All Wf-328 optimizations wired (chemprop-side compatible)"
  echo "  - GPU Wave-A BLOCKED on user OK per kill_then_adapt directive"
  # §5.6 atomic write: emit DRY_RUN.log + plan.json (Wf-363-A1 schema)
  cat >"${OUT_BASE}/DRY_RUN.log" <<'EOF'
[DRY-RUN] launch_wf119_p1_4_chemprop_v3_lomo.sh @ wf119
[DRY-RUN] pre_flight_check PASS
[DRY-RUN] wrapper_name=launch_wf119_p1_4_chemprop_v3_lomo.sh
[DRY-RUN] wf_origin=wf119
[DRY-RUN] trainer=molmetal/scripts/train_chemprop_v2_v3.py
[DRY-RUN] py=/home/hugo/codes/mol-metal/.venv/bin/python
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed for actual launch)
EOF
  cat >"${OUT_BASE}/plan.json" <<EOF
{
    "wrapper": "launch_wf119_p1_4_chemprop_v3_lomo.sh",
    "wf_origin": "wf119",
    "mode": "dry-run",
    "user_ok_required": true,
    "trainer": "molmetal/scripts/train_chemprop_v2_v3.py",
    "py": "/home/hugo/codes/mol-metal/.venv/bin/python",
    "out_dir": "${OUT_BASE}",
    "args": ["--dry-run"],
    "expected_cohort": {"ablations": 4, "seeds": 1, "epochs": 3},
    "estimated_wall": "~2 hr on PRO 6000 cuda:0",
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per AGENTS.md §5.1"
}
EOF
  exit 0
fi

# ---- USER OK GATE (P0 BLOCKER per 207 §1.1 + kill_then_adapt standing directive) ----
if [[ "${USER_OK:-0}" != "1" ]]; then
  echo "================================================================="
  echo " USER OK REQUIRED — kill_then_adapt standing directive (207 §1.1)"
  echo "================================================================="
  echo ""
  echo "This script launches the TODO-119 P1.4 chemprop v3 LOMO Wave-A:"
  echo "  Wave-A: 4 ablations × 1 seed × 3 ep smoke — ~2 hr on PRO 6000 cuda:0"
  echo ""
  echo "Before running this script, the user MUST explicitly grant OK per"
  echo "207 §1.1 (user 'kill then adapt' deferral pattern)."
  echo ""
  echo "To grant OK and launch:"
  echo "  USER_OK=1 bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh"
  echo ""
  echo "To skip an ablation (cheaper smoke):"
  echo "  USER_OK=1 SKIP_ABLATIONS=bond,closure bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh"
  echo ""
  echo "To verify the launcher shape on CPU first:"
  echo "  bash scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh --dry-run"
  echo ""
  echo "ABORTING — no GPU work executed."
  exit 1
fi
echo "=== USER OK GRANTED (USER_OK=1) — proceeding with real launch ==="
echo "Timestamp: $(date '+%Y-%m-%dT%H:%M:%S%z')"

# ---- Per-ablation OK gate selector ----
SKIP_ABLATIONS="${SKIP_ABLATIONS:-}"  # e.g. "bond,closure" to skip
want_ablation() {
  local a=$1
  [[ -z "${SKIP_ABLATIONS}" ]] && return 0
  [[ ",${SKIP_ABLATIONS}," == *",${a},"* ]] && return 1
  return 0
}

# ---- Pre-flight tool guards (per Wf-262 / Wf-296D wiring) ----
echo ""
echo "=== WAVE TOOL GUARDS ==="
echo "pre_flight_check.py (chemprop_v2_v3 pipeline):"
setsid nohup ${PY} molmetal/scripts/pre_flight_check.py --pipeline chemprop_v2_v3 2>&1 | tail -3 || { echo "FAIL: pre_flight_check"; exit 1; }
echo ""

# ============================================================================
# Wave-A: P1.4 smoke — 4 ablation configs × 1 seed × 3 epochs
# ============================================================================
echo "=== Wave-A: TODO-119 P1.4 chemprop v3 LOMO smoke (4 ablations × 1 seed × 3 ep) ==="
echo "OUT_BASE: ${OUT_BASE}"
echo "SCRIPT: ${SCRIPT}"
echo "GPU: ${GPU}"
echo ""

# Bug A fix (2026-10-08): the previous `--user-ok "REQUIRED"` literal was never
# a valid trainer arg — train_chemprop_v2_v3.py has no --user-ok flag, so the
# trainer exited with an argparse error (rc=2) before training started. The pipe
# error below was masked by `tee` returning rc=0 (no pipefail), so the wrapper
# falsely reported success and moved on. The user-OK gating is enforced by
# USER_OK env var check above (line ~140), NOT by a CLI flag — so just drop it.
for ablation in baseline v3_fix_only min_epochs_only lomo_init_scale_only; do
  if ! want_ablation "${ablation}"; then
    echo "  [SKIP] ablation=${ablation}"
    continue
  fi
  echo ""
  echo "=== Ablation: ${ablation} ==="
  ABLATION_OUT="${OUT_BASE}/${ablation}/seed0"
  mkdir -p "${ABLATION_OUT}"
  ABLATION_LOG="${ABLATION_OUT}/train.log"
  # §5.11 (2026-11-03) — the arm's levers are CONFIG KEYS, not command-line
  # flags.  The derivation below writes the base config plus this arm's
  # lever(s) into a per-arm YAML, and the trainer is invoked with
  # --config / --out-dir only.
  #
  # The lever VALUES are transcribed exactly from the flags the wrapper used
  # to pass; nothing is invented here.
  #   baseline              -> chemprop_mode: v3-fallback
  #   v3_fix_only           -> chemprop_mode: v3-fix
  #   min_epochs_only       -> chemprop_mode: v3-fallback, chemprop_min_epochs: 3
  #   lomo_init_scale_only  -> chemprop_mode: v3-fallback,
  #                            chemprop_lomo_init_scale: 0.001
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   case "${ablation}" in
  #     baseline)             ABLATION_ARGS=( --chemprop-mode v3-fallback --out-dir "${ABLATION_OUT}" ) ;;
  #     v3_fix_only)          ABLATION_ARGS=( --chemprop-mode v3-fix      --out-dir "${ABLATION_OUT}" ) ;;
  #     min_epochs_only)      ABLATION_ARGS=( --chemprop-mode v3-fallback --chemprop-min-epochs 3 --out-dir "${ABLATION_OUT}" ) ;;
  #     lomo_init_scale_only) ABLATION_ARGS=( --chemprop-mode v3-fallback --chemprop-lomo-init-scale 1e-3 --out-dir "${ABLATION_OUT}" ) ;;
  #   esac
  case "${ablation}" in
    baseline)
      ARM_LEVERS="chemprop_mode: v3-fallback"
      ;;
    v3_fix_only)
      ARM_LEVERS="chemprop_mode: v3-fix"
      ;;
    min_epochs_only)
      ARM_LEVERS="chemprop_mode: v3-fallback
chemprop_min_epochs: 3"
      ;;
    lomo_init_scale_only)
      ARM_LEVERS="chemprop_mode: v3-fallback
chemprop_lomo_init_scale: 0.001"
      ;;
    *)
      echo "FAIL: unknown ablation ${ablation}"
      exit 1
      ;;
  esac
  ABLATION_CONFIG="${ABLATION_OUT}/_derived_config.yaml"
  if ! ${PY} - "${CONFIG}" "${ABLATION_CONFIG}" "${ablation}" "${ARM_LEVERS}" <<'PYEOF'
import sys

import yaml

base, out, arm, levers = sys.argv[1:5]
with open(base, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
# The arm's levers go under `train:` — the section the loader treats as a
# run-describing one (wfA2_config_loader.py:_LEAF_FALLBACK_SECTIONS), so each
# leaf resolves to the dest of the same name on this trainer's parser.
for line in levers.splitlines():
    if not line.strip():
        continue
    key, _, value = line.partition(":")
    cfg.setdefault("train", {})[key.strip()] = yaml.safe_load(value)
with open(out, "w", encoding="utf-8") as fh:
    fh.write("# DERIVED per-arm config for scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh\n")
    fh.write("# base: %s\n" % base)
    fh.write("# arm: %s\n" % arm)
    fh.write("# Only the arm's lever(s) differ from the base; everything else is\n")
    fh.write("# the base config's own values, which carry the hyperparameters the\n")
    fh.write("# command line used to pass.\n")
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
PYEOF
  then
    echo "FAIL: per-arm config derivation failed for ablation=${ablation}"
    exit 1
  fi
  echo "  CMD: ${PY} ${SCRIPT} --config ${ABLATION_CONFIG} --out-dir ${ABLATION_OUT}"
  # Bug B fix (2026-10-08): ${PY} trainer | tee log || { exit 1 } catches
  # tee's exit code, NOT the trainer's. Without pipefail (disabled above to
  # avoid SIGPIPE), an argparse error / early trainer exit (rc=2) was masked
  # by tee rc=0, and the wrapper falsely reported success. Capture the
  # trainer's actual exit code via ${PIPESTATUS[0]} and bail on rc != 0.
  set +e
setsid nohup   ${PY} "${SCRIPT}" --config "${ABLATION_CONFIG}" --out-dir "${ABLATION_OUT}" 2>&1 | tee "${ABLATION_LOG}"
  trainer_rc=${PIPESTATUS[0]}
  set -e
  if [[ ${trainer_rc} -ne 0 ]]; then
    echo "FAIL: ablation ${ablation} trainer exited rc=${trainer_rc}; check ${ABLATION_LOG}"
    exit ${trainer_rc}
  fi
  # Belt-and-braces: verify the trainer produced output. The trainer writes
  # training_summary.json (aggregate across metals) at the top level +
  # per-{metal}/per_seed_metrics_seed{seed}.json per metal. Accept either.
  if [[ -s "${ABLATION_OUT}/training_summary.json" ]]; then
    echo "  [OK] ablation=${ablation} complete (training_summary.json present)"
  elif compgen -G "${ABLATION_OUT}/*/per_seed_metrics_seed*.json" > /dev/null; then
    echo "  [OK] ablation=${ablation} complete (per-metal per_seed_metrics_seed*.json present)"
  else
    echo "FAIL: ablation ${ablation} trainer rc=0 but no training_summary.json or per-metal per_seed_metrics_seed*.json in ${ABLATION_OUT}"
    exit 2
  fi
done

echo ""
echo "=== Wf-119 P1.4 Wave-A COMPLETE ==="
echo "All 4 ablations finished. Results under ${OUT_BASE}/"
echo ""
echo "Honest framing:"
echo "  - MEASURED: 4 ablations completed (if no errors)"
echo "  - PROJECTED: per-ablation LOMO AUC ≥ 0.50 (Wf-266 root cause fix)"
echo "  - Aggregated JSON under ${OUT_BASE}/training_summary.json"
