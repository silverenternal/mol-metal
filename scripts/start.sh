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
# 0) Capture the ambient CUDA_VISIBLE_DEVICES mask (Wf-A165)
# ============================================================================
# Wf-A165 (2026-11-01) — an EXPLICIT ambient device mask is a caller
# decision and must WIN over the yaml's `compute.gpu`.  Before this repair
# the yaml-derived export in section 3b OVERWROTE any ambient mask, so a
# session lock of `CUDA_VISIBLE_DEVICES=""` (no GPU) was silently replaced
# by the yaml's device index and a yaml-only launch RE-ENABLED a GPU the
# operator had switched off.  The downstream resolver
# (molmetal/molmetal_lam/sbdd_env/training_utils.py ->
# apply_default_cuda_visible_devices) already honours an explicitly-empty
# env var ("env > default GPU 0"), so the launcher was the only leak.
#
# Captured HERE, at the top of the script, because everything between here
# and the mask block in section 3b could otherwise contaminate the reading:
#   * scripts/_pre_flight_check.sh is a CHILD process (`bash scripts/...`,
#     section 2) — a child cannot mutate this shell's environment;
#   * the only file this script sources is $YAML_ENV_FILE (section 3), and
#     that file carries NO CUDA_VISIBLE_DEVICES entry (verified: the
#     `lines = [...]` list built by parse_yaml has no such key);
#   * molmetal/scripts/setup_molmetal_env.sh is NOT sourced here — it is a
#     standalone env-prep script and no launcher under scripts/ sources it.
# So the value observed here is the caller's ambient value, verbatim.
#
# `_AMBIENT_CVD_SET` distinguishes "unset" from "set to the empty string".
# The empty mask is a deliberate no-GPU lock and must NOT be conflated with
# "unset" (that conflation is exactly what the `:-` idiom does).
# `${VAR+set}` is the POSIX-safe presence test and is safe under `set -u`;
# `[[ -v VAR ]]` would need bash >= 4.2.
# --- Wf-A165: ambient CUDA_VISIBLE_DEVICES capture (start of block) ---
_AMBIENT_CVD_SET=0
_AMBIENT_CVD=""
if [[ -n "${CUDA_VISIBLE_DEVICES+set}" ]]; then
  _AMBIENT_CVD_SET=1
  _AMBIENT_CVD="$CUDA_VISIBLE_DEVICES"
fi
# --- Wf-A165: ambient CUDA_VISIBLE_DEVICES capture (end of block) ---

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
    # AGENTS.md §5.11 (user directive 2026-10-09, verbatim "以后要求能且只能用
    # yaml配置启动训练和推理，no cli参数"): --extra-flags is REFUSED, not
    # deprecated. Every dest it used to carry resolves from the YAML through
    # wfA2_config_loader's leaf-fallback, verified by round-trip on 2026-10-09
    # against all 7 CFM configs. The refusal lives here, at the surface every
    # launch_*.sh wrapper funnels through, so the escape hatch cannot be
    # re-introduced by a future wrapper.
    --extra-flags)
      echo "[start.sh][FAIL] --extra-flags is refused by AGENTS.md §5.11."
      echo "[start.sh]        Training and inference launch from YAML only; every"
      echo "[start.sh]        hyperparameter must be declared in the config file."
      echo "[start.sh]        Each flag that used to be passed here has a YAML"
      echo "[start.sh]        home — e.g. --gen-head-lr-scale -> model.gen_head_lr_scale,"
      echo "[start.sh]        --entropy-reg-weight -> train.entropy_reg_weight,"
      echo "[start.sh]        --device -> compute.device."
      exit 1
      ;;
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


def _sec(name):
    """Return section ``name`` as a dict, or ``{}`` if absent / a scalar.

    2026-10-12: some legacy configs write ``data: <csv-path>`` as a scalar
    string (e.g. configs/platinai_4cell_desc_v1_rerun.yaml).  ``.get()`` on
    that string raised ``AttributeError: 'str' object has no attribute
    'get'`` and start.sh reported it as "could not parse YAML".  Treat any
    non-dict section as empty so a scalar section simply yields defaults.
    """
    v = d.get(name)
    return v if isinstance(v, dict) else {}


e = d.get('entry', {})
# Wf-A8 (2026-10-03) backward compat: legacy yamls (R17-R20 era like
# configs/default.yaml) lack an `entry` section. Synthesize from
# data.dataset to keep backward-compat functional (per A8_yaml_unify
# _closeout.md §0 backward-compat contract + AGENTS.md §5.5).
if not e:
    ds = _sec('data').get('dataset', 'qm9')
    if ds == 'qm9':
        legacy_script = 'molmetal/scripts/train_cfm_yuelbond_wasserstein.py'
        # 2026-11-03 (§5.11 repair): the synthesised command baked the
        # LITERAL trainer flag `--device cuda:0` — a hyperparameter no YAML
        # could influence (scripts/_yaml_only_gate.py flags it as
        # `start_sh_literal_trainer_flag`).  It is now the `{legacy_device}`
        # placeholder, whose value is read from the yaml's trainer-facing
        # `compute.device` key and otherwise keeps the previous literal
        # `cuda:0`  ->  the rendered command is BYTE-IDENTICAL to the
        # superseded rendering for every yaml that does not declare
        # `compute.device` (MEASURED 2026-11-03: the only entry-less yaml in
        # the repo is configs/locks/protocol_lock.yaml, which declares no
        # `compute` section at all -> `cuda:0`, unchanged).
        # CORRECTION 2026-11-03 (additive, per AGENTS.md §5.5 rule #6 — the
        # sentence above is PRESERVED VERBATIM and is superseded): there are
        # now THREE entry-less yamls under configs/ — configs/locks/protocol_lock.yaml
        # (no `compute` section at all -> `cuda:0`, unchanged) and this
        # cohort's own §5.11 fixtures configs/wfa269_c9_start_sh.yaml and
        # configs/wfa269_c9_start_sh_dmpnn.yaml (both declare
        # `compute.device: cuda:0` -> `cuda:0`, unchanged).  MEASURED
        # 2026-11-03 12:01 over the 305 *.yaml then present under configs/:
        # `compute.device` is 'cuda:0' 122, 'cpu' 33, 'auto' 15, an explicit
        # null 4, absent 78, and 53 files carry no `compute` section at all
        # (configs/ is concurrently churned by the other Wf-A269 cohorts, so
        # these counts move by a file or two).
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #   (the command key is NOT superseded - it is unchanged below; the
        #   superseded RENDERING is the command string it held)
        #   '$PYTHON {script} --hidden {hidden} --n-layers {n_layers} '
        #   '--n-steps {n_steps} --method {method} --batch-size {batch_size} '
        #   '--lr {lr} --epochs {epochs} --device cuda:0 '
        #   '--out-dir {out_dir} --ckpt-dir {ckpt_dir}'
        e = {
            'script': legacy_script,
            'command': (
                '$PYTHON {script} --hidden {hidden} --n-layers {n_layers} '
                '--n-steps {n_steps} --method {method} --batch-size {batch_size} '
                '--lr {lr} --epochs {epochs} --device {legacy_device} '
                '--out-dir {out_dir} --ckpt-dir {ckpt_dir}'
            ),
        }
    else:
        legacy_script = 'molmetal/scripts/train_dmpnn_multitask_wasserstein.py'
        # 2026-11-03 (§5.11 repair): two LITERAL trainer flags here —
        # `--metals Pt` and `--device cuda:0`.  `--metals` now renders the
        # `{legacy_metals}` placeholder, whose value is THIS FILE'S OWN
        # yaml-derived metals LIST, space-joined for the shell (the
        # derivation is in the `_legacy_metals` block below); it was
        # hardcoded 'Pt' before, which is the same defect the 2026-10-03
        # fix above already repaired for the non-legacy path (hardcoded 'Pt'
        # silently trained the wrong metal).  `--device` renders
        # `{legacy_device_dmpnn}` — the same yaml key (`compute.device`) as
        # the CFM branch above, with a DIFFERENT whitelist because 'auto' is
        # invalid for this trainer (see the `_legacy_device_*` block below).
        # Rendering delta, stated plainly: an entry-less D-MPNN yaml that
        # DECLARES a metals list or a metal grid now trains THAT cohort
        # instead of the hardcoded 'Pt'; an entry-less yaml that declares
        # neither still renders 'Pt'  ->  BYTE-IDENTICAL.  MEASURED
        # 2026-11-03: no yaml in the repo reaches this branch (the only
        # entry-less yaml, configs/locks/protocol_lock.yaml, has
        # data.dataset == 'qm9' -> the CFM branch above), so no shipped run
        # changes.
        # CORRECTION 2026-11-03 (additive, per AGENTS.md §5.5 rule #6 — the
        # sentence above is PRESERVED VERBATIM and is superseded): no SHIPPED
        # yaml reaches this branch — configs/locks/protocol_lock.yaml is the
        # only shipped entry-less yaml and its data.dataset == 'qm9' -> the
        # CFM branch above.  This cohort's OWN fixture
        # configs/wfa269_c9_start_sh_dmpnn.yaml reaches this branch on purpose
        # (it is what proves the `--metals` placeholder), so no shipped run
        # changes.
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
        #   (the command key is NOT superseded - it is unchanged below; the
        #   superseded RENDERING is the command string it held)
        #   '$PYTHON {script} --dataset {dataset} --metals Pt '
        #   '--epochs {epochs} --batch-size {batch_size} --lr {lr} '
        #   '--device cuda:0 --out-dir {out_dir} --ckpt-dir {ckpt_dir}'
        e = {
            'script': legacy_script,
            'command': (
                '$PYTHON {script} --dataset {dataset} --metals {legacy_metals} '
                '--epochs {epochs} --batch-size {batch_size} --lr {lr} '
                '--device {legacy_device_dmpnn} --out-dir {out_dir} --ckpt-dir {ckpt_dir}'
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

# 2026-11-03 (§5.11 repair): the values behind the legacy branch's
# `{legacy_device}` / `{legacy_device_dmpnn}` placeholders.  The
# trainer-facing device key is `compute.device` — `compute.gpu` is a
# start.sh-ONLY CUDA_VISIBLE_DEVICES mask source (it has NO trainer dest)
# and it is not an ordinal: the shipped yamls spell it 'cuda:0' (91),
# 'cpu' (18), the INT 0 (4) and 'cuda:1' (1), and under the very mask
# start.sh exports from that key ('cuda:1' -> CVD=1, so ordinal 0 is the
# only device the trainer can see) a `--device cuda:1` dies with
# "invalid device ordinal" at use time.  `compute.device` is the
# dest-bearing key; its shipped values are 'cuda:0' (112), 'cpu' (33),
# 'auto' (15), an explicit null (3) and absent (125) — MEASURED
# 2026-11-03 over the 288 *.yaml files under configs/ (the population
# this branch can ever see).  'auto' raises inside `torch.device()` (the
# D-MPNN trainer calls it unconditionally,
# molmetal/scripts/train_dmpnn_multitask_wasserstein.py:4198) even though
# the CFM trainer special-cases it
# (molmetal/scripts/train_cfm_yuelbond_wasserstein.py:16667), so the two
# branches get DIFFERENT whitelists: the CFM branch honours 'auto', the
# D-MPNN branch does not.  Every value outside the whitelist keeps the
# pre-repair literal 'cuda:0', so the rendered command is BYTE-IDENTICAL
# to the superseded rendering for every yaml that does not declare a
# trainer-safe `compute.device`.  A yaml that DOES declare one now has it
# reach the trainer, which is the §5.11 point.
#
# CORRECTION 2026-11-03 (additive, per AGENTS.md §5.5 rule #6 — the
# paragraph above is PRESERVED VERBATIM and is superseded): the population
# it names was MIS-STATED — it counted 288 files, while configs/ holds 306
# *.yaml (the same enumeration the bulk-render equivalence check used).
# RE-MEASURED 2026-11-03 12:01 over the 305 *.yaml then present under
# configs/: `compute.device` is 'cuda:0' 122, 'cpu' 33, 'auto' 15, an
# explicit null 4, absent 78, and 53 files carry no `compute` section at
# all (configs/ is concurrently churned by the other Wf-A269 cohorts, so
# these counts move by a file or two); the branch itself only ever sees the
# 3 entry-less yamls (configs/locks/protocol_lock.yaml + this cohort's two
# fixtures).  The `compute.gpu` counts in the paragraph above ('cuda:0' 91,
# 'cpu' 18, the INT 0 (4), 'cuda:1' (1)) were RE-MEASURED 2026-11-03 and
# REPRODUCE exactly (138 absent + 53 with no `compute` section account for
# the rest of the 305).
_legacy_device_cfm = _sec('compute').get('device')
if _legacy_device_cfm not in ('cuda', 'cuda:0', 'cpu', 'auto'):
    _legacy_device_cfm = 'cuda:0'
_legacy_device_dmpnn = _sec('compute').get('device')
if _legacy_device_dmpnn not in ('cuda', 'cuda:0', 'cpu'):
    _legacy_device_dmpnn = 'cuda:0'

# 2026-11-03 (§5.11 repair): the value behind the legacy branch's
# `{legacy_metals}` placeholder.  `--metals` is a LIST dest on the D-MPNN
# trainer (nargs="+", default ["auto"]), so the yaml must carry a LIST and
# the rendered shell command must be SPACE-JOINED: AGENTS.md §5.11 records
# the 2026-10-09 run that lost a full 4-fold x 5-seed sweep because the
# yaml said `metals: Ru Ir Os Re` and the trainer received ONE element
# named "Ru Ir Os Re" and matched zero metals.  The yaml's own list wins
# (`protocol.metals`, then `data.metals` — configs/dmpnn.yaml declares
# `data.metals` as a LIST); a yaml that declares only a sweep GRID falls
# back to the 2026-10-03 derivation above (`_metal_default`); a yaml that
# declares neither keeps the pre-repair literal 'Pt'.
_legacy_metals = None
for _metals_section in ('protocol', 'data'):
    _metals_raw = _sec(_metals_section).get('metals')
    if _metals_raw is None:
        continue
    if isinstance(_metals_raw, str):
        # a space-separated STRING is what the 2026-10-09 failure looked
        # like; split it so the shell command carries the same tokens the
        # yaml names, instead of one element with spaces inside it.
        _metals_raw = [t for t in _metals_raw.split() if t]
    if isinstance(_metals_raw, (list, tuple)) and _metals_raw:
        _legacy_metals = ' '.join(str(t) for t in _metals_raw)
        break
if _legacy_metals is None:
    _legacy_metals = str(_metal_default)

cmd = _cmd_template_or_legacy.format(
    out_dir=d.get('output', {}).get('base_dir', 'results/auto'),
    ckpt_dir=d.get('output', {}).get('ckpt_dir', 'results/auto/ckpts'),
    hidden=_sec('model').get('hidden_dim', 128),
    n_layers=_sec('model').get('n_layers')
    or _sec('model').get('depth', 4),
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
    dataset=_sec('data').get('dataset', 'qm9'),
    # placeholders the wf124/cfm templates need to stop hardcoding values the
    # yaml already declares
    alpha=_sec('model').get('alpha', 0.5),
    wasserstein_weight=_sec('model').get('wasserstein_weight', 0.0),
    n_mc=_sec('model').get('n_mc', 256),
    w2_gradient_mode=_sec('model').get('w2_gradient_mode', 'sliced'),
    ablation_config=_sec('model').get('ablation_config', 'mtl_baseline'),
    depth=_sec('model').get('depth', 3),
    use_l2_guard='--use-l2-guard' if _sec('model').get('use_l2_guard') else '',
    device=(d.get('compute', {}) or {}).get('gpu', 'cuda:0'),
    # 2026-11-03 (§5.11): the legacy branch's device + metals placeholders.
    # Kept SEPARATE from `device` above on purpose — `device` is the
    # mask-source (`compute.gpu`) semantics the three wf124
    # command_templates were built against, and changing THAT would change
    # their rendering; the legacy branch needs the trainer-dest semantics
    # (`compute.device`) instead.  See the `_legacy_device_*` derivation
    # above for why they differ.  `legacy_metals` is likewise separate from
    # `metal` above: `metal` is the single-metal sweep placeholder the
    # wf124 templates use, `legacy_metals` is the space-joined LIST the
    # legacy `--metals` flag needs (nargs="+").
    legacy_device=_legacy_device_cfm,
    legacy_device_dmpnn=_legacy_device_dmpnn,
    legacy_metals=_legacy_metals,
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
    # 2026-10-04 — placeholders for the pocket-conditioned flow-matching entry
    # (train_fm_pocket.py).  Without these, a template naming any of them dies
    # with KeyError, which start.sh reports as "could not parse YAML" -- a
    # misleading message for what is really an unsupported placeholder.
    max_atoms=_sec('data').get('max_atoms', 40),
    max_pocket_atoms=_sec('data').get('max_pocket_atoms', 200),
    multi_ph_bins=(d.get('flow', {}) or {}).get('multi_ph_bins', 32),
    multi_ph_max_radius=(d.get('flow', {}) or {}).get('multi_ph_max_radius', 5.0),
    multi_ph_max_coord=(d.get('flow', {}) or {}).get('multi_ph_max_coord', 4.0),
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

# 2026-10-04 — emit only the appended flags the ENTRY SCRIPT actually accepts.
# start.sh has always appended a fixed set (checkpoint/resume/lr-scheduler/
# warmup/min-lr/weight-decay/max-norm) after the entry command.  That works
# while every entry is one of the two trainers that grew up with those flags,
# but a third entry (train_fm_pocket.py) accepts NONE of them, so launching it
# through start.sh died on "unrecognized arguments" before doing any work.
#
# Probe the entry script's SOURCE for each flag literal rather than running
# `--help`: the flags are plain strings in every parser here (verified: all 7
# present as literals in the CFM and D-MPNN trainers, 0 of 7 in the pocket
# trainer), and a subprocess probe would pay a full torch import at launch.
# Matching on the source is also the conservative failure mode -- an
# unrecognised flag is simply not appended, never a wrong one.
_EXTRA_CANDIDATES = [
    ('--checkpoint-interval', 'CHECKPOINT_INTERVAL'),
    ('--resume-from', 'RESUME_FROM'),
    ('--lr-scheduler', 'SCHEDULER'),
    ('--lr-warmup-epochs', 'LR_WARMUP_EPOCHS'),
    ('--lr-min-lr', 'LR_MIN_LR'),
    # 2026-10-15 — was MISSING from this probe list, so the emission at
    # `LR_PCT_START=` below and the `_add_flag --lr-pct-start` call were both
    # dead: the flag was filtered out by the `EXTRA_SUPPORTED_FLAGS` gate and
    # `protocol.scheduler.pct_start` never reached the trainer (every config
    # silently ran at the trainer's 0.3 default).  Caught on the BLK-C11
    # 2026-10-15 PCGrad A/B; both arms ran at 0.3.
    ('--lr-pct-start', 'LR_PCT_START'),
    ('--weight-decay', 'WEIGHT_DECAY'),
    ('--max-norm', 'MAX_NORM'),
]
_extra_supported = ''
_extra_skipped = []
_probe_src = ''
for _w in _script_candidates:
    if not _w:
        continue
    _p = Path(_w)
    if not _p.is_file():
        _p = (Path.cwd() / _w)
    if _p.is_file():
        try:
            _probe_src = _p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            _probe_src = ''
        break
if _probe_src:
    _parts = []
    for _flag, _var in _EXTRA_CANDIDATES:
        if ('"%s"' % _flag) in _probe_src or ("'%s'" % _flag) in _probe_src:
            _parts.append(_flag)
        else:
            _extra_skipped.append(_flag)
    _extra_supported = ' '.join(_parts)
    if _extra_skipped:
        print(
            '[start.sh] entry %s does not accept: %s -- not appending them'
            % (Path(_script_candidates[0] or _cmd_src.split()[0]).name,
               ', '.join(_extra_skipped)),
            file=sys.stderr,
        )
# The events flags are appended separately (as $EVENTS_FLAG) and the pocket
# trainer has neither form, so probe them too -- otherwise the entry still
# dies on "unrecognized arguments" via the one flag the loop above missed.
_events_supported = 0
_events_path_supported = 0
_events_sink_off_supported = 0
if _probe_src:
    if (
        '"--events-jsonl-path"' in _probe_src
        or "'--events-jsonl-path'" in _probe_src
    ):
        _events_path_supported = 1
        _events_supported = 1
    if '"--no-events-jsonl"' in _probe_src:
        _events_sink_off_supported = 1
        _events_supported = 1
# 2026-11-04 (Wf-A269) — the two spellings are probed SEPARATELY, and the
# launcher now emits the EXACT spelling it transports.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5
# rule #6: the probe used to be a single OR --
#
#   _events_supported = 0
#   if _probe_src and (
#       '"--events-jsonl-path"' in _probe_src
#       or "'--events-jsonl-path'" in _probe_src
#       or '"--no-events-jsonl"' in _probe_src
#   ):
#       _events_supported = 1
#
# -- so a trainer that declared only ONE of the two forms was treated as
# supporting BOTH, and the launcher transported `--no-events-jsonl` to a
# parser that did not declare it.  MEASURED 2026-11-04: that is exactly
# what happened with molmetal/scripts/train_cfm_yuelbond_wasserstein.py,
# which declared `--events-jsonl-path` and `--emit-json-events` but NOT
# `--no-events-jsonl` -- so `output.events_jsonl: false` in a CFM config
# crashed the launch with argparse's "unrecognized arguments" before a
# single step ran.  The trainer now declares the flag (the CFM trainer
# and the D-MPNN trainer share one spelling); the two probes below stop
# the class from recurring for a THIRD trainer.
if any(
    Path(w).name == 'train_dmpnn_multitask_wasserstein.py'
    for w in _script_candidates
    if w
):
    _task_v = str(_sec('model').get('task', 'multi_task'))
    _pin_v = bool(_sec('data').get('pin_memory', True))
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
# 2026-10-24 (Wf-A108i, §6.197) — A LAUNCHER MUST NOT SUPPLY A VALUE THE CONFIG
# DID NOT DECLARE.
#
# MEASURED: the c7 full-corpus run declared `train.lr_scheduler: cosine` and
# trained at a flat 3.0e-4 for all 18 epochs.  This script read
# `protocol.scheduler.type` (absent from that yaml), defaulted it to "none", and
# passed `--lr-scheduler none` on the command line, which wins over the yaml
# under the documented precedence rule.  The same script reported `epochs=50`
# for an 18-epoch config.  Both are the §5.11 defect in the launcher: a knob
# that lives only in a shell script is a knob nobody audits.
#
# The rule now: for each flag, take `protocol.<key>` if declared, else the
# trainer's own yaml key `train.<key>` if declared, else emit NOTHING and let
# the trainer's argparse default stand.  Empty is a value here — `_add_flag`
# skips it — so "undeclared" is visible in the rendered command instead of being
# papered over by a launcher default.
_proto = d.get('protocol', {}) or {}
_train = d.get('train', {}) or {}
_out = d.get('output', {}) or {}
_sched = _proto.get('scheduler', {}) or {}


def _first(*vals):
    for v in vals:
        if v is None or v == '':
            continue
        return v
    return ''


_declared_sched = _first(_sched.get('type'), _train.get('lr_scheduler'))
_declared_warmup = _first(_sched.get('warmup_epochs'), _train.get('lr_warmup_epochs'))
_declared_min_lr = _first(_sched.get('min_lr'), _train.get('lr_min_lr'))
_declared_pct = _first(_sched.get('pct_start'), _train.get('lr_pct_start'))
_declared_wd = _first(_proto.get('weight_decay'), _train.get('weight_decay'))
_declared_max_norm = _first(_proto.get('max_norm'), _train.get('max_norm'))
_declared_epochs = _first(_proto.get('epochs'), _train.get('epochs'))
_declared_resume = _first(_out.get('resume_from'), _train.get('resume_from'))

lines = [
    f'ENTRY_SCRIPT={shlex.quote(e.get("script", ""))}',
    f'ENTRY_COMMAND={shlex.quote(cmd)}',
    f'OUT_DIR={shlex.quote(out_block.get("base_dir", "results/auto"))}',
    f'TRAINING_SUMMARY={shlex.quote(out_block.get("training_summary", ""))}',
    f'LOG_MASTER={shlex.quote(out_block.get("log_master", ""))}',
    f'GPU={shlex.quote(d.get("compute", {}).get("gpu", "cuda:0"))}',
    f'N_SEEDS={d.get("protocol", {}).get("n_seeds", 1)}',
    # 2026-10-04 — `protocol.n_seeds_min` had no consumer anywhere.  The CFM
    # trainer takes a SINGLE --seed (start.sh enumerates the sweep), so the
    # "minimum seeds for honest per-seed std" guard belongs here, at the level
    # that actually knows the sweep size -- the same semantics as chemprop's
    # --n-seeds-min (train_chemprop_v2_v3.py:1714).  Default 1 = no opinion.
    f'N_SEEDS_MIN={d.get("protocol", {}).get("n_seeds_min", 1)}',
    f'EPOCHS={_declared_epochs if _declared_epochs != "" else "(trainer default)"}',
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
    f'SCHEDULER={shlex.quote(str(_declared_sched))}',
    f'LR_WARMUP_EPOCHS={_declared_warmup}',
    f'LR_MIN_LR={_declared_min_lr}',
    # 2026-10-09 — OneCycle pct_start (warmup fraction of total steps).
    # EMITTED EMPTY when the yaml does not declare it, so the trainer keeps
    # its own None default (historical hardcoded 0.3) and the `_add_flag`
    # guard above appends nothing.
    f'LR_PCT_START={shlex.quote(str(_declared_pct))}',
    f'WEIGHT_DECAY={_declared_wd}',
    # Wf-A8 (2026-10-03) — max-norm gradient clipping threshold. Default
    # 5.0 (Pascanu 2013 standard for transformer/GNN stability). The
    # trainer's --max-norm flag accepts this via the line below.
    f'MAX_NORM={_declared_max_norm}',
    # Wf-A5 N3 — checkpoint resumption env vars (consumed by the
    # build-launch-command block below).
    f'CHECKPOINT_INTERVAL={ckpt_interval}',
    f'RESUME_FROM={shlex.quote(str(_declared_resume))}',
    # Wf-TUI NEW — runtime events.jsonl path (consumed by both the
    # trainer --events-jsonl-path flag + the TUI dashboard
    # --events-jsonl flag).  Default = <out_dir>/events.jsonl when the
    # yaml does not specify output.events_jsonl_path.
    f'EVENTS_JSONL_PATH={shlex.quote(events_jsonl_path)}',
    f'EVENTS_JSONL_DISABLED={1 if events_jsonl_disabled else 0}',
    # 2026-10-04 — D-MPNN-only flags rendered from the yaml keys above
    # (empty string for every other entry script; see _dmpnn_new_flags).
    f'DMPNN_NEW_FLAGS={shlex.quote(_dmpnn_new_flags)}',
    # 2026-10-04 — space-separated list of the flags above that the entry
    # script actually accepts (empty when the probe could not read it, in
    # which case bash falls back to the full legacy set).
    f'EXTRA_SUPPORTED_FLAGS={shlex.quote(_extra_supported)}',
    f'EXTRA_PROBE_OK={1 if _probe_src else 0}',
    f'EVENTS_SUPPORTED={_events_supported}',
    # 2026-11-04 (Wf-A269) — the two events spellings are now probed
    # separately, so the launcher can emit the EXACT one the entry
    # parser declares instead of assuming both.
    f'EVENTS_PATH_SUPPORTED={_events_path_supported}',
    f'EVENTS_SINK_OFF_SUPPORTED={_events_sink_off_supported}',
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
# 3a-bis) n_seeds_min guard (2026-10-04)
# ============================================================================
# `protocol.n_seeds_min` had no consumer before this.  It carries the same
# meaning as chemprop's --n-seeds-min: below it, a per-seed std is not an
# honest statistic, so a run that would report one must say so out loud or be
# blocked.  Checked here rather than in a trainer because the trainer sees a
# single --seed; only this script knows the sweep size.
# ALLOW_SINGLE_SEED=1 bypasses it, mirroring chemprop's --allow-single-seed --
# for smoke runs and explicit honest-negative reporting, not for results.
if [[ "${N_SEEDS_INT:-$N_SEEDS}" =~ ^[0-9]+$ ]] \
   && [[ "${N_SEEDS_MIN:-1}" =~ ^[0-9]+$ ]] \
   && (( N_SEEDS < N_SEEDS_MIN )); then
  if [[ "${ALLOW_SINGLE_SEED:-0}" == "1" ]]; then
    echo "[start.sh][WARN] n_seeds=$N_SEEDS < n_seeds_min=$N_SEEDS_MIN — proceeding because ALLOW_SINGLE_SEED=1. Any per-seed std from this run is NOT an honest estimate."
  else
    echo "[start.sh][FAIL] n_seeds=$N_SEEDS < n_seeds_min=$N_SEEDS_MIN (protocol.n_seeds_min)."
    echo "[start.sh][FAIL] A per-seed std needs at least $N_SEEDS_MIN seeds to be meaningful."
    echo "[start.sh][FAIL] Raise protocol.n_seeds, lower protocol.n_seeds_min, or set ALLOW_SINGLE_SEED=1 for a smoke / honest-negative run."
    exit 1
  fi
fi

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
#       Wf-A165 (2026-11-01): the yaml is the SSOT for the device ONLY
#       when the caller expressed no preference.  An explicitly-set
#       ambient CUDA_VISIBLE_DEVICES (including "" = the no-GPU lock)
#       now wins — see the capture at the top of this script and the mask
#       block below.  The default path (ambient unset) is byte-for-byte
#       the pre-Wf-A165 behaviour.
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

# --- Wf-A165: yaml-derived GPU mask (start of block) ---
# "cuda:0" → "0"; ignore anything non-numeric ("auto", "cpu", …) so a
# bad yaml value cannot silently pin the wrong device.
_cvd="${GPU#cuda:}"
_cvd="${_cvd%%[^0-9]*}"
# Wf-A165 (2026-11-01) — an EXPLICIT ambient mask wins over the yaml.
# The yaml remains the device SSOT (AGENTS.md §5.11) when the caller has
# expressed no preference: the ambient-UNSET path below is byte-for-byte
# the pre-Wf-A165 behaviour (same export, same stdout line).
if [[ "$_AMBIENT_CVD_SET" == "1" ]]; then
  # Ambient was explicitly set — including "" (the no-GPU lock).  Keep it
  # VERBATIM and say plainly that the yaml's device was NOT applied.  The
  # self-referencing export cannot widen the mask: an empty ambient stays
  # empty, a "1" stays "1".  `$_AMBIENT_CVD` (captured at the top of this
  # script) is still equal to `$CUDA_VISIBLE_DEVICES` here — nothing
  # between the capture and this block writes to that variable.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  echo "[start.sh] CUDA_VISIBLE_DEVICES='$_AMBIENT_CVD' (ambient mask honoured — yaml compute.gpu=$GPU NOT applied)"
  if [[ -z "$_AMBIENT_CVD" ]]; then
    echo "[start.sh][WARN] ambient CUDA_VISIBLE_DEVICES is empty (no-GPU lock) — no GPU is visible to this run."
  fi
elif [[ -n "$_cvd" ]]; then
  # No ambient preference: the yaml decides the device (§5.11).  The
  # no-colon default substitutes the yaml's index only when the variable
  # is UNSET — an explicitly-empty ambient keeps its empty value, so this
  # export can never widen a caller's mask either.
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-$_cvd}"
  echo "[start.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from yaml compute.gpu=$GPU)"
else
  echo "[start.sh][WARN] yaml compute.gpu='$GPU' is not a cuda:N index; leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
# --- Wf-A165: yaml-derived GPU mask (end of block) ---
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
  # 2026-11-04 (Wf-A269) — emit the sink off-switch ONLY when the entry
  # parser declares it.  SUPERSEDED RENDERING PRESERVED VERBATIM per
  # AGENTS.md section 5.5 rule #6: the branch used to be an
  # unconditional `EVENTS_FLAG="--no-events-jsonl"`, which transported a
  # flag to any entry that did not declare it -> argparse aborts the
  # launch with "unrecognized arguments".  The WARN keeps the residual
  # visible: a YAML that asks for the sink off, on an entry that cannot
  # express it, now says so instead of dying.
  if [[ "${EVENTS_SINK_OFF_SUPPORTED:-0}" == "1" ]]; then
    EVENTS_FLAG="--no-events-jsonl"
  else
    EVENTS_FLAG=""
    echo "[start.sh][WARN] output.events_jsonl=false but the entry parser does not declare --no-events-jsonl; the events.jsonl sink stays ON (this entry cannot switch the sink off from the YAML)" >&2
  fi
else
  if [[ "${EVENTS_PATH_SUPPORTED:-0}" == "1" ]]; then
    EVENTS_FLAG="--events-jsonl-path \"$EVENTS_JSONL_PATH\""
  else
    EVENTS_FLAG=""
  fi
fi
# 2026-10-04 — $DMPNN_NEW_FLAGS carries the three formerly-dead yaml keys
# (model.task / data.pin_memory / protocol.gradient_accumulation_steps) for
# the D-MPNN entry script; it is empty for every other script (e.g. CFM,
# which has no such flags) — see the parse_yaml block above.
# 2026-10-04 — append only what the entry script accepts.  Each entry below is
# "flag VAR"; the value is quoted at emit time so paths with spaces survive.
# EXTRA_PROBE_OK=0 (the probe could not read the script) falls back to the full
# legacy set, so nothing that worked before stops working.
APPENDED_FLAGS=""
_add_flag() {
  local _flag="$1" _var="$2"
  # 2026-10-24 (Wf-A108i, §6.197) — an EMPTY value means the config declared
  # nothing for this knob.  Emit no flag and let the trainer's argparse default
  # stand; emitting a launcher default is how c7's cosine schedule was replaced
  # by a constant LR without a single line of the config changing.
  if [[ -z "${!_var}" ]]; then
    return 0
  fi
  if [[ "${EXTRA_PROBE_OK:-0}" == "0" ]] \
     || [[ " ${EXTRA_SUPPORTED_FLAGS:-} " == *" ${_flag} "* ]]; then
    APPENDED_FLAGS="${APPENDED_FLAGS} ${_flag} \"${!_var}\""
  fi
}
_add_flag --checkpoint-interval CHECKPOINT_INTERVAL
_add_flag --resume-from RESUME_FROM
_add_flag --lr-scheduler SCHEDULER
_add_flag --lr-warmup-epochs LR_WARMUP_EPOCHS
_add_flag --lr-min-lr LR_MIN_LR
# 2026-10-09 — OneCycle warmup fraction. OPTIONAL: the trainer's
# --lr-pct-start default is None, meaning "keep the historical hardcoded
# 0.3". Only append when the yaml actually declared
# protocol.scheduler.pct_start, so every config that predates the key runs
# exactly as before.
if [[ -n "${LR_PCT_START:-}" ]]; then
  _add_flag --lr-pct-start LR_PCT_START
fi
_add_flag --weight-decay WEIGHT_DECAY
_add_flag --max-norm MAX_NORM
# Same treatment for the events sink: an entry with neither form gets no flag
# rather than an argument error.
if [[ "${EXTRA_PROBE_OK:-0}" == "1" && "${EVENTS_SUPPORTED:-0}" != "1" ]]; then
  EVENTS_FLAG=""
fi
TRAINER_CMD="$ENTRY_COMMAND$APPENDED_FLAGS $EVENTS_FLAG ${DMPNN_NEW_FLAGS:-} $EXTRA_FLAGS"
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
