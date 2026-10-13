#!/usr/bin/env bash
# Launch the BLK-C-L3 sampling-time guidance strength sweep.
#
# AGENTS.md 5.11: this wrapper passes --config, --ckpt and --out-dir ONLY.
# It sets up the environment, detaches, and redirects. It carries no
# hyperparameter, so no hyperparameter can live here.
#
# AGENTS.md 5.9: .venv/bin/python only. Never bare python, never another venv.
#
#   bash scripts/launch_cfm_l3_guidance_sweep.sh --dry-run   # print, launch nothing
#   bash scripts/launch_cfm_l3_guidance_sweep.sh             # run detached on GPU 0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"                      # 5.9
CFG="$REPO/configs/cfm_l3_guidance_sweep_2026_10_12.yaml"
CKPT="$REPO/molmetal/reports/cfm_patience10_seed0/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt"
OUT="$REPO/molmetal/reports/cfm_l3_guidance_sweep_seed0"
LOG_DIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
LOG="$LOG_DIR/cfm_l3_guidance_sweep.log"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

for f in "$PY" "$CFG" "$CKPT"; do
  [[ -e "$f" ]] || { echo "MISSING: $f" >&2; exit 1; }
done

# 2026-10-12 FIX — the gate has to FAIL LOUDLY. Every value below is read in a
# `$(...)` substitution, and bash discards a substitution's exit status when the
# result is interpolated into an `echo`: a KeyError prints a traceback, leaves
# the field blank, and the script still exits 0. AGENTS.md 4 defines 0=PASS, so
# a broken pre-flight was announcing itself as a green light for a multi-hour
# GPU sweep. Each read is now checked explicitly and the script exits non-zero.
if ! STRENGTHS="$("$PY" -c "
import yaml,sys
d=yaml.safe_load(open('$CFG'))
s=d['sweep']['strengths']
assert s[0]==0.0, 'the control arm 0.0 must come first or the sweep has no baseline'
assert len(s)==len(set(s)), 'duplicate strengths would spend GPU for nothing'
print(' '.join(str(x) for x in s))")"; then
  echo "FAIL: could not read sweep.strengths from $CFG" >&2
  exit 1
fi

if ! PROTOCOL="$("$PY" -c "
import yaml
d=yaml.safe_load(open('$CFG'))['eval']
print(d['n_clouds'],d['n_steps'])")"; then
  echo "FAIL: could not read eval.n_clouds / eval.n_steps from $CFG" >&2
  exit 1
fi
# Belt and braces: a substitution can succeed and still be empty.
if [[ -z "${STRENGTHS// }" || -z "${PROTOCOL// }" ]]; then
  echo "FAIL: an empty sweep grid or protocol read from $CFG" >&2
  exit 1
fi

if [[ "$DRY" == "1" ]]; then
  echo "DRY RUN -- nothing launched"
  echo "  config : $CFG"
  echo "  ckpt   : $CKPT"
  echo "  out    : $OUT"
  echo "  log    : $LOG"
  echo "  arms   : $STRENGTHS"
  echo "  n_clouds/n_steps: $PROTOCOL"
  echo "  GPU    : 0 (standing directive 只用gpu0吧)"
  exit 0
fi

mkdir -p "$OUT" "$(dirname "$LOG")"

{
  echo "=== CFM L3 guidance sweep, $(date -Is) ==="
  echo "strengths: $STRENGTHS"
  for S in $STRENGTHS; do
    ARM="strength${S}"
    echo "--- arm $ARM ---"
    # No pipefail: a `| tee` here can die of SIGPIPE when DataLoader or the
    # RDKit logger flushes after the trainer process exits (memory:
    # wrapper-pipefail-sigpipe-silent-death).
    "$PY" "$REPO/molmetal/scripts/wf_blkc_l3_guidance_probe.py" \
      --ckpt "$CKPT" \
      --config "$CFG" \
      --out "$OUT/arm_${ARM}.json" \
      --guidance-strength "$S" \
      --arm "$ARM" 2>&1 || echo "ARM $ARM FAILED rc=$?"
    echo "--- arm $ARM done ---"
  done
  echo "SWEEP DONE $(date -Is)"
} > "$LOG" 2>&1

echo "launched detached; log: $LOG"
nohup tail -f "$LOG" >/dev/null 2>&1 &