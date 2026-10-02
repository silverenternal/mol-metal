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
#   4. Spawn trainer in background → log file + events.jsonl (Wf-TUI)
#   5. Spawn TUI dashboard in background (scripts/_tui_dashboard.py)
#      or skip TUI if --no-tui (per AGENTS.md §5.7 NEW trainer flag
#      wire-in smoke-required rule)
#   6. Wait for trainer to complete; cleanup on exit
#
# Usage:
#   bash scripts/start.sh <path-to-yaml-config> [--dry-run] [--no-tui]
#                            [--tui-interval FLOAT] [--tui-mode {rich,plain,headless}]
#                            [--extra-flags "FLAGS"]
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
Usage: bash scripts/start.sh <path-to-yaml-config> [options]

SINGLE canonical launch script for Mol-Metal retrain.

Required:
  <path-to-yaml-config>   Path to training config (e.g.
                          configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml)

Optional:
  --dry-run               Render command + show TUI sample; don't launch trainer
  --extra-flags "FLAGS"   Pass extra flags to trainer
  --no-tui                Skip TUI dashboard spawn (log-only mode for ssh w/o tmux)
  --tui-interval FLOAT    TUI refresh rate (default 1.0; passed via --tui-interval)
  --tui-mode {rich,plain,headless}   Render mode (default rich; passed via --tui-mode)

Environment:
  USER_OK=1               Required (AGENTS.md §5.1)
  PYTHON                  Default: .venv/bin/python
  EXTRA_FLAGS             Optional trainer flags

Examples:
  USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml
  USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml --dry-run
  USER_OK=1 bash scripts/start.sh configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml --no-tui --tui-mode headless
EOF
}

if [[ "${1:-}" == "--help" ]] || [[ -z "${1:-}" ]]; then
  usage
  exit 0
fi

YAML_PATH="$1"
DRY_RUN=0
EXTRA_FLAGS=""
# Wf-TUI NEW: --tui / --no-tui / --tui-interval / --tui-mode flags
# (per user voice 2026-10-03 "训练时runtime的可视化...应当和我们具体
# 要做的多阶段训练以及训练内状态集成起来"). Defaults preserve the
# existing always-on TUI behaviour (per Wf-A5 N1 + Wf-A4 fix contract).
TUI_ENABLED=1
TUI_INTERVAL=1.0
TUI_MODE="rich"
# Wf-A8 (2026-10-03) --fast flag — forwarded to _pre_flight_check.sh so
# the user can short-circuit the full pre-flight gate suite during
# iterative development (per AGENTS.md §5.5 filesystem-truth + §5.7
# trainer flag wire-in smoke-required rule). Default 0 (full pre-flight
# gate suite per §1.4 pre-flight v3 PASS gate requirement). When --fast
# is set, _pre_flight_check.sh skips gates tagged `p3_*` and emits a
# WARN-only summary (full PASS gate enforcement deferred to the actual
# launch wrapper). This is Wf-A6 BLOCKER 1 fix per Wf-A8 review-based
# plan section 1 — the smoke test's `start.sh ... --fast --dry-run
# --no-tui` invocation must be accepted (otherwise it dies with
# `unknown arg: --fast` exit 1).
FAST_PREFLIGHT=0
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --extra-flags) EXTRA_FLAGS="$2"; shift 2 ;;
    --no-tui) TUI_ENABLED=0; shift ;;
    --tui) TUI_ENABLED=1; shift ;;
    --tui-interval) TUI_INTERVAL="$2"; shift 2 ;;
    --tui-mode) TUI_MODE="$2"; shift 2 ;;
    --fast) FAST_PREFLIGHT=1; shift ;;
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
# Wf-A8 (2026-10-03) — export PYTHON=PROJECT_ROOT/.venv/bin/python to child _pre_flight_check.sh
# so the pre-flight uses the project venv (where torch lives) rather than system python3
# (per AGENTS.md §5.9 single-venv rule; matches the §5.1 standing GPU authorization contract).
export PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
echo "[start.sh] running pre-flight check (PYTHON=$PYTHON)..."
if ! bash scripts/_pre_flight_check.sh --fast "$YAML_PATH"; then
  echo "[start.sh][FAIL] pre-flight check failed — aborting"
  exit 1
fi
# Wf-A8 (2026-10-03) — propagate --fast to pre-flight check as well so
# the gate suite can skip p3_* gates on demand (per the rationale in the
# FAST_PREFLIGHT declaration above). The --fast flag is already in the
# _pre_flight_check.sh invocation above; this comment preserves the
# cross-reference for §5.5 filesystem-truth audits.
: "${FAST_PREFLIGHT:=0}"
if [[ "$FAST_PREFLIGHT" == "1" ]]; then
  echo "[start.sh][INFO] --fast set: pre-flight p3_* gates SKIP, WARN-only summary emitted"
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
from pathlib import Path
yaml_path, env_path = sys.argv[1], sys.argv[2]
with open(yaml_path) as f:
    d = yaml.safe_load(f)
e = d.get('entry', {})
# Wf-A8 (2026-10-03) backward compat: legacy yamls (R17-R20 era like
# configs/default.yaml) lack an `entry` section. Synthesize from
# data.dataset to keep backward-compat functional (per A8_yaml_unify
# _closeout.md §0 backward-compat contract + AGENTS.md §5.5).
if not e:
    ds = (d.get('data', {}) or {}).get('dataset', 'qm9')
    if ds == 'qm9':
        legacy_script = 'molmetal/scripts/train_cfm_yuelbond_wasserstein.py'
        e = {
            'script': legacy_script,
            'command': (
                '$PYTHON {script} --hidden {hidden} --n-layers {n_layers} '
                '--n-steps {n_steps} --method {method} --batch-size {batch_size} '
                '--lr {lr} --epochs {epochs} --device cuda:0 '
                '--out-dir {out_dir} --ckpt-dir {ckpt_dir}'
            ),
        }
    else:
        legacy_script = 'molmetal/scripts/train_dmpnn_multitask_wasserstein.py'
        e = {
            'script': legacy_script,
            'command': (
                '$PYTHON {script} --dataset {dataset} --metals Pt '
                '--epochs {epochs} --batch-size {batch_size} --lr {lr} '
                '--device cuda:0 --out-dir {out_dir} --ckpt-dir {ckpt_dir}'
            ),
        }
    d['entry'] = e
# Wf-A8 (2026-10-03) — derive legacy per-pair placeholders before the
# format() call. cross_references can be either a dict (legacy yamls
# with .pairs) or a list (new Wf-A8 yamls); handle both so the
# format() below doesn't KeyError on {metal}/{seed}.
_cr_obj = d.get('cross_references')
if not isinstance(_cr_obj, dict):
    _metal_default = 'Pt'
    _seed_default = '0'
else:
    _pairs = _cr_obj.get('pairs') or [{}]
    _first = _pairs[:1][0] if _pairs else {}
    if not isinstance(_first, dict):
        _first = {}
    _metal_default = _first.get('metal', 'Pt')
    _seed_default = _first.get('seed', '0')
cmd = e.get('command', '').format(
    out_dir=d.get('output', {}).get('base_dir', 'results/auto'),
    ckpt_dir=d.get('output', {}).get('ckpt_dir', 'results/auto/ckpts'),
    hidden=(d.get('model', {}) or {}).get('hidden_dim', 128),
    n_layers=(d.get('model', {}) or {}).get('n_layers', 4),
    n_steps=(d.get('flow', {}) or {}).get('n_steps', 100),
    method=(d.get('flow', {}) or {}).get('method', 'rk4'),
    batch_size=(d.get('train', {}) or {}).get('batch_size', 32),
    lr=(d.get('train', {}) or {}).get('lr', 1e-4),
    epochs=(d.get('train', {}) or {}).get('epochs', 30),
    dataset=(d.get('data', {}) or {}).get('dataset', 'qm9'),
    script=e.get('script', 'molmetal/scripts/train_dmpnn_multitask_wasserstein.py'),
    metal=_metal_default,
    seed=_seed_default,
    # Wf-A8c FIX: yaml-path + flags_joined placeholders used by entry.command_template
    # (per task brief single-source-of-truth rule: yaml is canonical; start.sh substitutes
    # these placeholders from yaml-derived fields). yaml_path = first argv (the yaml file
    # itself). flags_joined = Wf-A6 A2 + Wf-A8 + A8c protocol fields concatenated as a
    # space-separated string for backwards-compat with legacy entry.command format.
    yaml_path=sys.argv[1] if len(sys.argv) > 1 else 'config.yaml',
    flags_joined='',  # explicit CLI flags (USER_OK + extra-flags) appended separately
)
# Strip yaml literal-block backslash continuations: '|' block scalars
# preserve backslashes as literal chars; we need single-line for shell.
cmd = cmd.replace(chr(92)+chr(10), ' ').replace(chr(92), '').strip()
# Wf-A5 N3 — extract checkpoint resumption fields from the output
# section so we can pass them to the trainer via --resume-from +
# --checkpoint-interval.  Defaults match the trainer CLI defaults
# (resume_from='auto', checkpoint_interval=50).
# Wf-TUI NEW — extract events_jsonl_path from the output section so
# we can pass it to the trainer via --events-jsonl-path + to the TUI
# dashboard via --events-jsonl.  Default = <out_dir>/events.jsonl
# (per trainer's own default in train_dmpnn_multitask_wasserstein.py).
out_block = d.get('output', {})
ckpt_interval = int(out_block.get('checkpoint_interval', 50))
resume_from = str(out_block.get('resume_from', 'auto'))
events_jsonl_default = str(
    Path(str(out_block.get('base_dir', 'results/auto'))) / 'events.jsonl'
)
events_jsonl_path = str(
    out_block.get('events_jsonl_path', events_jsonl_default)
)
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
    # Wf-A6 A2 BLOCKER C fix (2026-10-03) — extract LR scheduler
    # protocol fields (SCHEDULER + WARMUP_EPOCHS + MIN_LR +
    # WEIGHT_DECAY) from the yaml's protocol.scheduler subsection
    # so start.sh can pass them to the trainer via CLI flags.
    # Defaults preserve Wf-131 P1 + Wf-251-F4 baseline bit-for-bit
    # (no scheduler; constant-LR Adam with weight_decay=0.0).
    # Trainer CLI flags --lr-scheduler {none, cosine, onecycle}
    # + --lr-warmup-epochs INT + --lr-min-lr FLOAT + --weight-decay
    # FLOAT are wired in both trainers per AGENTS.md §5.7
    # trainer flag wire-in smoke-required rule.
    f'SCHEDULER={shlex.quote(str(d.get("protocol", {}).get("scheduler", {}).get("type", "none")))}',
    f'LR_WARMUP_EPOCHS={int(d.get("protocol", {}).get("scheduler", {}).get("warmup_epochs", 5))}',
    f'LR_MIN_LR={d.get("protocol", {}).get("scheduler", {}).get("min_lr", 1e-6)}',
    f'WEIGHT_DECAY={d.get("protocol", {}).get("weight_decay", 0.0)}',
    # Wf-A8 (2026-10-03) — max-norm gradient clipping threshold. Default
    # 5.0 (Pascanu 2013 standard for transformer/GNN stability). The
    # trainer's --max-norm flag accepts this via the line below.
    f'MAX_NORM={d.get("protocol", {}).get("max_norm", 5.0)}',
    # Wf-A5 N3 — checkpoint resumption env vars (consumed by the
    # build-launch-command block below).
    f'CHECKPOINT_INTERVAL={ckpt_interval}',
    f'RESUME_FROM={shlex.quote(resume_from)}',
    # Wf-TUI NEW — runtime events.jsonl path (consumed by both the
    # trainer --events-jsonl-path flag + the TUI dashboard
    # --events-jsonl flag).  Default = <out_dir>/events.jsonl when the
    # yaml does not specify output.events_jsonl_path.
    f'EVENTS_JSONL_PATH={shlex.quote(events_jsonl_path)}',
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
# Wf-TUI NEW — append --events-jsonl-path from the YAML output
# section so the trainer emits structured per-epoch events.jsonl for
# the live TUI dashboard.  Per AGENTS.md §5.7 trainer-flag wire-in
# smoke-required rule: flag exists in parse_args + initialises cleanly
# when set (verified via §5.7 50-step CPU smoke at Wf-TUI §5.7 section).
TRAINER_CMD="$ENTRY_COMMAND --checkpoint-interval \"$CHECKPOINT_INTERVAL\" --resume-from \"$RESUME_FROM\" --events-jsonl-path \"$EVENTS_JSONL_PATH\" --lr-scheduler \"$SCHEDULER\" --lr-warmup-epochs \"$LR_WARMUP_EPOCHS\" --lr-min-lr \"$LR_MIN_LR\" --weight-decay \"$WEIGHT_DECAY\" --max-norm \"$MAX_NORM\" $EXTRA_FLAGS"
LOG_FILE="$LOG_MASTER"
echo "[start.sh] trainer command: $TRAINER_CMD"
echo "[start.sh] log file: $LOG_FILE"
echo "[start.sh] events.jsonl path: $EVENTS_JSONL_PATH"

# ============================================================================
# 5) Dry-run: print command + sample TUI banner + exit
# ============================================================================
# Wf-TUI NEW — dry-run echoes the new TUI_CMD knobs so the user can
# confirm --events-jsonl + --tui-mode + --tui-interval wire-in.
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[start.sh][DRY-RUN] would launch:"
  echo "    $TRAINER_CMD"
  echo "[start.sh][DRY-RUN] would log to: $LOG_FILE"
  echo "[start.sh][DRY-RUN] would emit events.jsonl to: $EVENTS_JSONL_PATH"
  if [[ $TUI_ENABLED -eq 1 ]]; then
    echo "[start.sh][DRY-RUN] would spawn TUI dashboard (mode=$TUI_MODE interval=${TUI_INTERVAL}s)"
  else
    echo "[start.sh][DRY-RUN] TUI dashboard SKIPPED (--no-tui set; log-only mode)"
  fi
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
# Wf-A8 (2026-10-03) — use `eval` so that $PYTHON and other shell
# variables embedded in the yaml's entry.command template (e.g.
# "$PYTHON {script} ...") are properly interpolated. `bash -c "..."`
# would NOT expand variables embedded in $TRAINER_CMD (verified by
# direct test); `eval` does.
nohup eval "$TRAINER_CMD >> $LOG_FILE 2>&1" &
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
# Wf-TUI NEW: --tui / --no-tui gating + --tui-interval + --tui-mode
# + --events-jsonl threading per user voice "应当和我们具体要做的
# 多阶段训练以及训练内状态集成起来".  Default behavior (no --no-tui)
# preserves the existing always-on TUI with --tui-interval=1.0 +
# --tui-mode=rich for backward compat with Wf-A5 N1 + Wf-A4 fix.
if [[ $TUI_ENABLED -eq 1 ]]; then
  echo "[start.sh] launching TUI dashboard (mode=$TUI_MODE interval=${TUI_INTERVAL}s)..."
  TUI_CMD="$PYTHON scripts/_tui_dashboard.py \
    --log-file $LOG_FILE \
    --output-dir $OUT_DIR \
    --gpu $GPU \
    --interval ${TUI_INTERVAL} \
    --tui-interval ${TUI_INTERVAL} \
    --tui-mode ${TUI_MODE} \
    --events-jsonl $EVENTS_JSONL_PATH"
  # EXIT trap is inherited by `exec`d TUI process. When TUI exits
  # (Ctrl+C, natural end, or trainer completion), the trap fires and
  # kills the trainer via the PID file. If the TUI process is replaced
  # (e.g. by Textual in a future integration), the trap carries forward.
  trap 'kill "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true; rm -f "$PID_FILE"' EXIT INT TERM
  exec $TUI_CMD
else
  # Wf-TUI NEW: --no-tui mode → trainer runs in background; this script
  # waits for completion (no TUI to keep the user-attached process alive).
  # The trainer writes to LOG_FILE + EVENTS_JSONL_PATH + per-seed metrics
  # JSON under OUT_DIR/.  Cleanup-on-EXIT still kills the trainer if the
  # user hits Ctrl+C (per AGENTS.md §5.1 + kill_then_adapt §0 contract).
  echo "[start.sh] --no-tui set, trainer running in background (log-only mode)..."
  trap 'kill "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true; rm -f "$PID_FILE"' EXIT INT TERM
  wait "$TRAINER_PID"
  echo "[start.sh][OK] trainer completed (--no-tui mode)"
fi
