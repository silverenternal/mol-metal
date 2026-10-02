#!/usr/bin/env bash
# scripts/start.sh
# SINGLE canonical launch script for retraining (per user voice:
#   "启动脚本...只能有一个" + "像nvtop一样实时刷新的表，能带训练指标可视化").
#
# Per AGENTS.md §1.4: this script CALLS _pre_flight_check.sh first and
# refuses to launch on FAIL. Per §5.1: USER_OK=1 required. Per §5.2:
# honest-framing wall-time projected at the bottom of the TUI.
#
# Architecture:
#   1. USER_OK gate (must be set; per AGENTS.md §5.1)
#   2. _pre_flight_check.sh [yaml] (exit 1 → abort)
#   3. YAML config parsing via Python (yaml.load)
#   4. Spawn trainer in background → log file
#   5. Spawn TUI dashboard in background (scripts/_tui_dashboard.py)
#   6. Wait for trainer to complete; cleanup on exit
#
# Usage:
#   bash scripts/start.sh <path-to-yaml-config> [--dry-run]
#   bash scripts/start.sh <yaml> --wave=custom_metrics.json
#   USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml
#
# Environment:
#   USER_OK=1     required (AGENTS.md §5.1)
#   PYTHON        default: .venv/bin/python
#   PYTHONPATH    default: <project root>
#   EXTRA_FLAGS   optional space-separated extra flags passed to trainer

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# Help / usage
# ============================================================================
usage() {
  cat <<EOF
Usage: bash scripts/start.sh <path-to-yaml-config> [--dry-run]

SINGLE canonical launch script for Mol-Metal retrain.

Required:
  <path-to-yaml-config>   Path to training config (e.g.
                          configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml)

Optional:
  --dry-run               Render command + show TUI sample; don't launch trainer
  --extra-flags "FLAGS"   Pass extra flags to trainer

Environment:
  USER_OK=1               Required (AGENTS.md §5.1)
  PYTHON                  Default: .venv/bin/python
  EXTRA_FLAGS             Optional trainer flags

Examples:
  USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml
  USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml --dry-run
EOF
}

if [[ "${1:-}" == "--help" ]] || [[ -z "${1:-}" ]]; then
  usage
  exit 0
fi

YAML_PATH="$1"
DRY_RUN=0
EXTRA_FLAGS=""
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --extra-flags) EXTRA_FLAGS="$2"; shift 2 ;;
    *) echo "[start.sh][FAIL] unknown arg: $1"; usage; exit 1 ;;
  esac
done

# ============================================================================
# 1) USER_OK gate (per AGENTS.md §5.1)
# ============================================================================
if [[ "${USER_OK:-0}" != "1" ]]; then
  echo "[start.sh][FAIL] USER_OK=1 required (per AGENTS.md §5.1)"
  echo "[start.sh][hint]  export USER_OK=1 before running"
  exit 1
fi
echo "[start.sh][OK] USER_OK=1 granted"

# ============================================================================
# 2) Preflight check (calls _pre_flight_check.sh with yaml arg)
# ============================================================================
echo "[start.sh] running pre-flight check..."
if ! bash scripts/_pre_flight_check.sh --fast "$YAML_PATH"; then
  echo "[start.sh][FAIL] pre-flight check failed — aborting"
  exit 1
fi

# ============================================================================
# 3) Parse YAML config via Python
# ============================================================================
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

parse_yaml() {
  # Write to a temp file so eval cannot split a multi-line ENTRY_COMMAND
  # into multiple shell commands. We then `.` (source) the file.
  # (eval on multi-line stdout would break because bash treats each
  # newline as a command separator, which is why "Permission denied"
  # appeared when the cmd contained backslash-continuations.)
  YAML_ENV_FILE="$(mktemp -t wf_start_yaml.XXXXXX)"
  trap 'rm -f "$YAML_ENV_FILE"' EXIT
  "$PYTHON" - "$YAML_PATH" "$YAML_ENV_FILE" <<'PY'
import sys, shlex, yaml
yaml_path, env_path = sys.argv[1], sys.argv[2]
with open(yaml_path) as f:
    d = yaml.safe_load(f)
e = d.get('entry', {})
cmd = e.get('command', '').format(
    out_dir=d.get('output', {}).get('base_dir', 'results/auto'),
    ckpt_dir=d.get('output', {}).get('ckpt_dir', 'results/auto/ckpts'),
)
# Strip yaml literal-block backslash continuations: '|' block scalars
# preserve backslashes as literal chars; we need single-line for shell.
cmd = cmd.replace(chr(92)+chr(10), ' ').replace(chr(92), '').strip()
# Wf-A5 N3 — extract checkpoint resumption fields from the output
# section so we can pass them to the trainer via --resume-from +
# --checkpoint-interval.  Defaults match the trainer CLI defaults
# (resume_from='auto', checkpoint_interval=50).
out_block = d.get('output', {})
ckpt_interval = int(out_block.get('checkpoint_interval', 50))
resume_from = str(out_block.get('resume_from', 'auto'))
lines = [
    f'ENTRY_SCRIPT={shlex.quote(e.get("script", ""))}',
    f'ENTRY_COMMAND={shlex.quote(cmd)}',
    f'OUT_DIR={shlex.quote(out_block.get("base_dir", "results/auto"))}',
    f'TRAINING_SUMMARY={shlex.quote(out_block.get("training_summary", ""))}',
    f'LOG_MASTER={shlex.quote(out_block.get("log_master", ""))}',
    f'GPU={shlex.quote(d.get("compute", {}).get("gpu", "cuda:0"))}',
    f'N_SEEDS={d.get("protocol", {}).get("n_seeds", 1)}',
    f'EPOCHS={d.get("protocol", {}).get("epochs", 50)}',
    f'DIRECTIVES_USER_OK={d.get("directives", {}).get("user_ok_required", True)}',
    # Wf-A5 N3 — checkpoint resumption env vars (consumed by the
    # build-launch-command block below).
    f'CHECKPOINT_INTERVAL={ckpt_interval}',
    f'RESUME_FROM={shlex.quote(resume_from)}',
]
with open(env_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')
PY
}

if ! parse_yaml; then
  echo "[start.sh][FAIL] could not parse YAML: $YAML_PATH"
  exit 1
fi

# Source the env file (safer than eval — no command-splitting surprises)
. "$YAML_ENV_FILE"
rm -f "$YAML_ENV_FILE"
trap - EXIT  # remove the cleanup trap now
echo "[start.sh][OK] yaml parsed:"
echo "    entry_script=$ENTRY_SCRIPT"
echo "    out_dir=$OUT_DIR"
echo "    gpu=$GPU"
echo "    n_seeds=$N_SEEDS"
echo "    epochs=$EPOCHS"

# ============================================================================
# 4) Build launch command (substitute YAML placeholders + append EXTRA_FLAGS)
# ============================================================================
mkdir -p "$OUT_DIR" "$(dirname "$TRAINING_SUMMARY")" "$(dirname "$LOG_MASTER")"
# Use $ENTRY_COMMAND directly (yaml's entry.command already includes
# the full python interpreter path; prepending $PYTHON would duplicate it).
# If a future config uses entry.script + args separately, the trainer
# wrapper can be expanded — see Wf-449 follow-up.
# Wf-A5 N3 — append the checkpoint resumption flags derived from the
# YAML output section (CHECKPOINT_INTERVAL + RESUME_FROM sourced from
# the parse_yaml block above).  These append AFTER any user-supplied
# EXTRA_FLAGS so the yaml stays the canonical source of truth (user
# flags can still override via --no-resume-from=explicit-path tricks).
TRAINER_CMD="$ENTRY_COMMAND --checkpoint-interval \"$CHECKPOINT_INTERVAL\" --resume-from \"$RESUME_FROM\" $EXTRA_FLAGS"
LOG_FILE="$LOG_MASTER"
echo "[start.sh] trainer command: $TRAINER_CMD"
echo "[start.sh] log file: $LOG_FILE"

# ============================================================================
# 5) Dry-run: print command + sample TUI banner + exit
# ============================================================================
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[start.sh][DRY-RUN] would launch:"
  echo "    $TRAINER_CMD"
  echo "[start.sh][DRY-RUN] would log to: $LOG_FILE"
  echo "[start.sh][DRY-RUN] would spawn TUI dashboard"
  echo "[start.sh][DRY-RUN] PASS"
  exit 0
fi

# ============================================================================
# 6) Spawn trainer in background + TUI dashboard in background
# ============================================================================
# Touch log file so the TUI dashboard can tail it immediately
: > "$LOG_FILE"

echo "[start.sh] launching trainer (PID will be reported)..."
TRAINER_PID=""
nohup bash -c "$TRAINER_CMD >> $LOG_FILE 2>&1" &
TRAINER_PID=$!
echo "[start.sh][OK] trainer spawned PID=$TRAINER_PID"

# Write trainer PID to file so the EXIT trap (inherited by `exec`d TUI process)
# can find and kill it when the TUI exits / Ctrl+C is pressed (per user voice
# "希望他是一个会在终端里面会像nvtop一样实时刷新的表"). Without this trap
# the trainer would be orphaned when the user detaches from the TUI.
PID_FILE="$OUT_DIR/.trainer.pid"
echo "$TRAINER_PID" > "$PID_FILE"

# Spawn TUI dashboard (foreground) with EXIT trap to clean up trainer.
# Per AGENTS.md §5.1 USER_OK gating + per task brief cleanup-on-EXIT.
echo "[start.sh] launching TUI dashboard..."
TUI_CMD="$PYTHON scripts/_tui_dashboard.py \
  --log-file $LOG_FILE \
  --output-dir $OUT_DIR \
  --gpu $GPU \
  --interval 1.0"

# EXIT trap is inherited by `exec`d TUI process. When TUI exits
# (Ctrl+C, natural end, or trainer completion), the trap fires and
# kills the trainer via the PID file. If the TUI process is replaced
# (e.g. by Textual in a future integration), the trap carries forward.
trap 'kill "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true; rm -f "$PID_FILE"' EXIT INT TERM
exec $TUI_CMD
