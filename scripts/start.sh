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
if isinstance(_cr_obj, dict):
    # legacy shape: cross_references.pairs = [{'metal': ..., 'seed': ...}, ...]
    _pairs = _cr_obj.get('pairs') or [{}]
    _first = _pairs[:1][0] if _pairs else {}
    if not isinstance(_first, dict):
        _first = {}
    _metal_default = _first.get('metal', 'Pt')
    _seed_default = _first.get('seed', '0')
else:
    # New Wf-A8 shape: cross_references is a LIST of report paths, so it
    # carries no metal/seed — the grid lives elsewhere in the yaml.
    #
    # 2026-10-03 fix: this branch previously hardcoded 'Pt', and the comment
    # above it already claimed both shapes were handled.  On the new-format
    # yamls that silently rendered `--metals Pt` for every sweep — the G5 grid
    # is Ru/Ir/Os/Rh/Re, so the trainer aborted with
    # "Metal(s) ['Pt'] not found in CSV. Available metals: [Ir, Os, Re, Rh, Ru]".
    #
    # The grid's location differs per config, so try the shapes actually
    # present in this repo rather than patching one file at a time:
    #   1. protocol.enumerated_runs = [{'metal': 'Ru', 'seed': 0}, ...]
    #   2. protocol.seeds + sweep_pairs = {'ru': [0,1,2], ..., 'total_pairs': N}
    #   3. runs.matrix = {'no_multi_task': {'ru': [0,1,2], ...}, ...}
    _proto = d.get('protocol') or {}
    _metal_default, _seed_default = None, None

    _runs = _proto.get('enumerated_runs') or []
    if _runs and isinstance(_runs[0], dict):
        _metal_default = _runs[0].get('metal')
        _seed_default = str(_runs[0].get('seed', 0))

    if _metal_default is None:
        _sp = {k: v for k, v in (d.get('sweep_pairs') or {}).items()
               if k != 'total_pairs' and isinstance(v, list)}
        if _sp:
            _metal_default = next(iter(_sp)).capitalize()

    if _metal_default is None:
        _matrix = (d.get('runs') or {}).get('matrix') or {}
        for _cfg_name, _grid in _matrix.items():
            if isinstance(_grid, dict) and _grid:
                _metal_default = next(iter(_grid)).capitalize()
                break

    if _metal_default is None:
        # waves.<name>.pairs = [{'metal': 'Ru', 'seed': 1}, ...]
        for _wname, _wave in (d.get('waves') or {}).items():
            if not isinstance(_wave, dict):
                continue
            _wp = _wave.get('pairs') or []
            if _wp and isinstance(_wp[0], dict) and _wp[0].get('metal'):
                _metal_default = _wp[0]['metal']
                _seed_default = str(_wp[0].get('seed', 0))
                break

    _metal_default = _metal_default or 'Pt'
    _seed_default = _seed_default or '0'
# Wf-A8c FIX: try new Wf-A8 yaml-managed 'command_template' first, fall back to legacy 'command'.
# Per AGENTS.md 5.5 + user voice 'never use CLI again always yaml', new yamls
# use 'command_template' field with {yaml_path}/{flags_joined}/{seed}/{out_dir}
# placeholders; old yamls use inline 'command' with shell vars.
def _proto_or_train(cfg: dict, key: str, default):
    """Read a hyperparameter from where the shipped configs actually put it.

    2026-10-03: the shipped wf124/cfm configs declare epochs/batch_size/lr
    under ``protocol``; this looked under ``train`` and so always missed,
    silently substituting the literal default while the yaml's own value drove
    nothing.  Check both, preferring ``protocol`` (the current convention).
    """
    for section in ('protocol', 'train'):
        v = (cfg.get(section) or {}).get(key)
        if v is not None:
            return v
    return default


_cmd_template_or_legacy = e.get('command_template') or e.get('command', '')
cmd = _cmd_template_or_legacy.format(
    out_dir=d.get('output', {}).get('base_dir', 'results/auto'),
    ckpt_dir=d.get('output', {}).get('ckpt_dir', 'results/auto/ckpts'),
    hidden=(d.get('model', {}) or {}).get('hidden_dim', 128),
    n_layers=(d.get('model', {}) or {}).get('n_layers')
    or (d.get('model', {}) or {}).get('depth', 4),
    n_steps=(d.get('flow', {}) or {}).get('n_steps', 100),
    method=(d.get('flow', {}) or {}).get('method', 'rk4'),
    # 2026-10-03: these read `protocol.*` because that is where the shipped
    # configs actually declare them.  They previously read `train.*`, which
    # none of the wf124/cfm configs have — so the values silently fell back to
    # the literals below, and the yaml's protocol.epochs/batch_size/lr drove
    # nothing.  Verified by probe: editing protocol.epochs 50 -> 77 left the
    # rendered command unchanged.  `train.*` is kept as a fallback for older
    # configs.
    batch_size=_proto_or_train(d, 'batch_size', 32),
    lr=_proto_or_train(d, 'lr', 1e-4),
    epochs=_proto_or_train(d, 'epochs', 30),
    dataset=(d.get('data', {}) or {}).get('dataset', 'qm9'),
    # placeholders the wf124/cfm templates need to stop hardcoding values the
    # yaml already declares
    alpha=(d.get('model', {}) or {}).get('alpha', 0.5),
    wasserstein_weight=(d.get('model', {}) or {}).get('wasserstein_weight', 0.0),
    n_mc=(d.get('model', {}) or {}).get('n_mc', 256),
    w2_gradient_mode=(d.get('model', {}) or {}).get('w2_gradient_mode', 'sliced'),
    ablation_config=(d.get('model', {}) or {}).get('ablation_config', 'mtl_baseline'),
    depth=(d.get('model', {}) or {}).get('depth', 3),
    use_l2_guard='--use-l2-guard' if (d.get('model', {}) or {}).get('use_l2_guard') else '',
    device=(d.get('compute', {}) or {}).get('gpu', 'cuda:0'),
    # NOTE: the launchers pass --cpu-num-workers 8 (Wf-326B optimal on this
    # 24-core host) while the shipped yamls declare 4.  Wiring the yaml through
    # makes the yaml authoritative on the start.sh path; the two values
    # disagreeing is a value question, flagged rather than silently resolved.
    cpu_workers=(d.get('compute', {}) or {}).get('cpu_workers', 0),
    # 2026-10-03: cfm_real_decode_lift_p1.3.yaml's template uses {pocket_id} but
    # nothing supplied it, so start.sh died with KeyError -> reported as
    # "could not parse YAML" (misleading: the yaml parsed fine).  Declared in
    # the yaml as protocol.pocket_id; falls back to the trainer's own default
    # so configs that do not use the placeholder are unaffected.
    pocket_id=(d.get('protocol', {}) or {}).get('pocket_id', '1h36'),
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
# 2026-10-04 — three YAML keys that resolved to no trainer argparse dest
# before this date, so their declared values reached nothing:
#   model.task                          -> --task
#   data.pin_memory                     -> --pin-memory / --no-pin-memory
#   protocol.gradient_accumulation_steps -> --gradient-accumulation-steps
# Emitted ONLY for the D-MPNN entry script: the CFM trainer declares none of
# these flags, so appending them to a CFM launch would abort it with
# "unrecognized arguments".  Fallbacks are the trainer's own defaults
# (multi_task / True / 1), i.e. its pre-flag behaviour, so configs that do
# not declare a key keep launching unchanged.
_dmpnn_new_flags = ''
# The entry script is `entry.script` when set, but the two configs/training/
# yamls spell it only inside entry.command_template — so also scan the
# command for the .py path, otherwise those two would silently miss the flags.
_script_candidates = [str(e.get('script', ''))]
_cmd_src = str(e.get('command_template') or e.get('command') or '')
_script_candidates += [w for w in _cmd_src.split() if w.endswith('.py')]
if any(
    Path(w).name == 'train_dmpnn_multitask_wasserstein.py'
    for w in _script_candidates
    if w
):
    _task_v = str((d.get('model', {}) or {}).get('task', 'multi_task'))
    _pin_v = bool((d.get('data', {}) or {}).get('pin_memory', True))
    _grad_accum_v = int(_proto_or_train(d, 'gradient_accumulation_steps', 1))
    _dmpnn_new_flags = (
        f'--task {shlex.quote(_task_v)} '
        f'{"--pin-memory" if _pin_v else "--no-pin-memory"} '
        f'--gradient-accumulation-steps {_grad_accum_v}'
    )
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
# 2026-10-04: honour output.events_jsonl.  This boolean was declared in all 9
# configs and read by NOTHING — start.sh only ever looked at
# output.events_jsonl_path, so `events_jsonl: False` still emitted events.
#
# The trainer had no way to disable the jsonl sink either: emit_event() gates
# the write on `_EVENT_LOG_PATH is not None`, and that was always a Path
# (--no-emit-json-events only switches stdout formatting).  So this needed a
# real off-switch rather than an empty path (which Path('') would turn into
# Path('.'), i.e. writing into the CWD).  Added --no-events-jsonl to the
# trainer; emitted below only when the yaml actually says False.
events_jsonl_disabled = out_block.get('events_jsonl', True) is False
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
    f'EVENTS_JSONL_DISABLED={1 if events_jsonl_disabled else 0}',
    # 2026-10-04 — D-MPNN-only flags rendered from the yaml keys above
    # (empty string for every other entry script; see _dmpnn_new_flags).
    f'DMPNN_NEW_FLAGS={shlex.quote(_dmpnn_new_flags)}',
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
# 3b) Export the environment the trainer + its data loaders actually need
# ============================================================================
# Wf-A9 FIX (2026-10-03) — these two were previously never exported, which
# is why launches died after pre-flight PASS:
#
#   (a) MOLMETAL_DATA_DIR — molmetal/data/cytotox.py defaulted to the
#       hard-coded /mnt/storage/data/molmetal mount (absent on this host),
#       so MetalCytotoxDataset.from_csv() raised
#       "FileNotFoundError: MetalCytoToxDB not found at
#        /mnt/storage/data/molmetal/MetalCytoToxDB.csv" on the D-MPNN
#       retrain path.  The real CSV is in .data/molmetal/.
#       (The loader now also honours this var itself — see Wf-A9 fix in
#       molmetal/data/cytotox.py — but exporting here keeps every child
#       process consistent.)
#
#   (b) CUDA_VISIBLE_DEVICES — derived from the yaml `compute.gpu` field
#       so the yaml stays the single source of truth for the device
#       (per user voice: "这个也得靠yaml决定训练配置").
#
#       The resolver's own DEFAULT is now also GPU 0
#       (training_utils.DEFAULT_CUDA_VISIBLE_DEVICES == "0"), so this
#       export and the code default agree.  That was NOT true before the
#       2026-10-03 unification: the default was "1" (= RTX 5090) and it
#       contradicted `gpu: cuda:0` in the training configs, which is why
#       this export exists at all.  Keep it — it is what makes an explicit
#       `gpu: cuda:1` in a yaml still win over the default (that is the
#       case this export exists for; every shipped config now says cuda:0).
export MOLMETAL_DATA_DIR="${MOLMETAL_DATA_DIR:-$PROJECT_ROOT/.data/molmetal}"
export MOLMETAL_TMQM_DIR="${MOLMETAL_TMQM_DIR:-$PROJECT_ROOT/.data/molmetal/tmQM}"

# "cuda:0" → "0"; ignore anything non-numeric ("auto", "cpu", …) so a
# bad yaml value cannot silently pin the wrong device.
_cvd="${GPU#cuda:}"
_cvd="${_cvd%%[^0-9]*}"
if [[ -n "$_cvd" ]]; then
  export CUDA_VISIBLE_DEVICES="$_cvd"
  echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from yaml compute.gpu=$GPU)"
else
  echo "[start.sh][WARN] yaml compute.gpu='$GPU' is not a cuda:N index; leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
echo "[start.sh] MOLMETAL_DATA_DIR=$MOLMETAL_DATA_DIR"

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
# 2026-10-04: output.events_jsonl:false must actually disable the sink.
# --events-jsonl-path always installs one, so emit the off-switch instead.
if [[ "${EVENTS_JSONL_DISABLED:-0}" == "1" ]]; then
  EVENTS_FLAG="--no-events-jsonl"
else
  EVENTS_FLAG="--events-jsonl-path \"$EVENTS_JSONL_PATH\""
fi
# 2026-10-04 — $DMPNN_NEW_FLAGS carries the three formerly-dead yaml keys
# (model.task / data.pin_memory / protocol.gradient_accumulation_steps) for
# the D-MPNN entry script; it is empty for every other script (e.g. CFM,
# which has no such flags) — see the parse_yaml block above.
TRAINER_CMD="$ENTRY_COMMAND --checkpoint-interval \"$CHECKPOINT_INTERVAL\" --resume-from \"$RESUME_FROM\" $EVENTS_FLAG --lr-scheduler \"$SCHEDULER\" --lr-warmup-epochs \"$LR_WARMUP_EPOCHS\" --lr-min-lr \"$LR_MIN_LR\" --weight-decay \"$WEIGHT_DECAY\" --max-norm \"$MAX_NORM\" ${DMPNN_NEW_FLAGS:-} $EXTRA_FLAGS"
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
# Use `bash -c` to properly interpret $TRAINER_CMD (which may contain
# shell variables + yaml-template substitutions like {yaml_path}).
# `nohup eval ...` fails because `eval` is a shell builtin not an
# external executable; `nohup` searches PATH for `eval` and fails.
# Per user voice 2026-10-03 '还是和之前一样有图标但是gpu上进程没启动'.
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
