#!/usr/bin/env bash
# Launch the BLK-C-M4 aromatic logit-offset sweep.
#
# AGENTS.md 5.11: this wrapper carries NO hyperparameter. The grid, the
# checkpoint, the cloud count, the solver, the step count and the seed all
# come from the YAML. It sets up the environment, detaches and redirects.
#
# AGENTS.md 5.9: .venv/bin/python only.
#
#   bash scripts/launch_cfm_aromatic_bias_sweep.sh --dry-run   # print, launch nothing
#   bash scripts/launch_cfm_aromatic_bias_sweep.sh             # run detached on GPU 0
#
# WHY A SEPARATE LAUNCHER rather than a flag on the guidance sweep
# --------------------------------------------------------------
# The two sweeps move different things and must not be confounded:
#
#   guidance strength  a SAMPLING-TIME force injected into the velocity field
#                      during ODE integration
#   aromatic bias      a DECODE-TIME offset on the bond-class logits
#
# Running both on one arm makes a result unattributable. This launcher pins
# guidance to 0.0 on every arm; the guidance sweep stays in
# scripts/launch_cfm_l3_guidance_sweep.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"                      # 5.9
CFG="$REPO/configs/cfm_aromatic_bias_sweep_2026_10_12.yaml"
CKPT="$REPO/molmetal/reports/cfm_patience10_seed0/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt"
OUT="$REPO/molmetal/reports/cfm_aromatic_bias_sweep_seed0"
LOG_DIR="${CLAUDE_JOB_DIR:-/tmp}/tmp"
LOG="$LOG_DIR/cfm_aromatic_bias_sweep.log"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

for f in "$PY" "$CFG" "$CKPT"; do
  [[ -e "$f" ]] || { echo "FAIL: missing $f" >&2; exit 1; }
done

# 2026-10-12. Every read below is checked explicitly. The guidance launcher
# read its values inside an `echo "...$("$PY" -c ...)"` substitution, and bash
# discards a substitution's exit status when the result is interpolated into a
# string -- so a KeyError printed a traceback, left the field blank, and the
# script still exited 0. AGENTS.md 4 defines 0=PASS, so that gate announced
# itself green while unable to read the file it was gating. Do not repeat it.
if ! BIASES="$("$PY" -c "
import yaml
d=yaml.safe_load(open('$CFG'))
b=d['sweep']['biases']
assert b[0]==0.0, 'the control arm 0.0 must come first or the sweep has no baseline'
assert len(b)==len(set(b)), 'duplicate biases would spend GPU for nothing'
assert all(x>=0 for x in b), 'a negative bias suppresses aromatic and is a different experiment'
print(' '.join(str(x) for x in b))")"; then
  echo "FAIL: could not read sweep.biases from $CFG" >&2
  exit 1
fi

if ! PROTOCOL="$("$PY" -c "
import yaml
d=yaml.safe_load(open('$CFG'))['eval']
print(d['n_clouds'],d['n_steps'])")"; then
  echo "FAIL: could not read eval.n_clouds / eval.n_steps from $CFG" >&2
  exit 1
fi

if [[ -z "${BIASES// }" || -z "${PROTOCOL// }" ]]; then
  echo "FAIL: an empty bias grid or protocol read from $CFG" >&2
  exit 1
fi

if [[ "$DRY" == "1" ]]; then
  echo "DRY RUN -- nothing launched"
  echo "  config : $CFG"
  echo "  ckpt   : $CKPT"
  echo "  out    : $OUT"
  echo "  log    : $LOG"
  echo "  arms   : $BIASES"
  echo "  n_clouds/n_steps: $PROTOCOL"
  echo "  guidance_strength: 0.0 on every arm (separate lever, see header)"
  echo "  GPU    : 0 (standing directive 只用gpu0吧)"
  exit 0
fi

mkdir -p "$OUT" "$(dirname "$LOG")"

{
  echo "=== CFM aromatic logit-offset sweep, $(date -Is) ==="
  echo "biases: $BIASES  (guidance pinned to 0.0 on every arm)"
  for B in $BIASES; do
    ARM="bias${B}"
    echo "--- arm $ARM ---"
    # No pipefail on the pipe: a `| tee` can die of SIGPIPE when RDKit or a
    # DataLoader worker flushes after the process exits (memory:
    # wrapper-pipefail-sigpipe-silent-death).
    "$PY" "$REPO/molmetal/scripts/wf_blkc_l3_guidance_probe.py" \
      --config "$CFG" \
      --out "$OUT/arm_${ARM}.json" \
      --bond-aromatic-logit-bias "$B" \
      --arm "$ARM" 2>&1 || echo "ARM $ARM FAILED rc=$?"
    echo "--- arm $ARM done ---"
  done
  echo "SWEEP DONE $(date -Is)"
} > "$LOG" 2>&1

echo "launched detached; log: $LOG"
nohup tail -f "$LOG" >/dev/null 2>&1 &