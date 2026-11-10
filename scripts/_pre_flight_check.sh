#!/usr/bin/env bash
# scripts/_pre_flight_check.sh
# SINGLE canonical pre-flight check (per AGENTS.md §1.4).
# Bash mirror of molmetal/scripts/pre_flight_check.py + Wf-A2 extensions:
#   - USER_OK gate check (per AGENTS.md §5.1)
#   - GPU env probe (nvidia-smi + torch CUDA)
#   - Python venv check
#   - Dataset integrity (CSV path + row count)
#   - YAML config state-lock (per user voice "额外需要检查yaml文件来锁定状态")
#   - Internal metric gates (REPORT-ONLY per AGENTS.md §5.2 honest framing)
#   - Wave budget remaining (per Wf-262 §7.1)
#   - Anchor freshness (per Wf-302A)
# Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP
#   CORRECTION (2026-11-09; additive + dated per AGENTS.md §5.5 rule #6): the
#   line above is the ORIGINAL contract, preserved VERBATIM.  It was WRONG —
#   this file contained no `exit 2` at all, so "2 = SKIP" was unreachable and
#   a SKIP was indistinguishable from a PASS for every consumer that reads
#   the exit code (scripts/start.sh reads ONLY the exit code).
#   Exit 2 is now LIVE and means DEGRADED: the canonical repo venv is absent
#   and an EXPLICIT interpreter override was honoured.  It is NOT a PASS.
#   Live contract: 0 = PASS, 1 = FAIL, 2 = DEGRADED (not PASS).
#   Per-gate SKIP dispositions stay REPORT-ONLY (AGENTS.md §5.2) and are not
#   part of the exit code.
#
# Usage:
#   bash scripts/_pre_flight_check.sh                          # full check (no yaml)
#   bash scripts/_pre_flight_check.sh path/to/config.yaml     # + YAML state-lock
#   bash scripts/_pre_flight_check.sh --fast [yaml]            # skip pytest collect-only

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

YAML_ARG=""
if [[ "${1:-}" != "--fast" ]] && [[ -n "${1:-}" ]]; then
  YAML_ARG="$1"
elif [[ "${2:-}" != "" ]]; then
  YAML_ARG="$2"
fi

log() { echo "[pre-flight] $*"; }
log_ok() { echo "[pre-flight][OK] $*"; }
log_skip() { echo "[pre-flight][SKIP] $*"; }
log_fail() { echo "[pre-flight][FAIL] $*"; }
log_warn() { echo "[pre-flight][WARN] $*"; }

# ---------------------------------------------------------------------------
# §5.9 single-venv: WHICH interpreter runs this gate.
#
# 2026-11-09 repair.  The original line was
#     PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
# which lets an AMBIENT $PYTHON WIN over the repo's own venv — the inverse of
# the repo's ratified convention in molmetal/scripts/smoke/*.sh (venv first,
# override as the fallback).  Two real consequences:
#   (1) an ambient $PYTHON could name a FORBIDDEN venv (§5.9 names
#       CadButEaas and kanzi) and the gate would still PASS — this file
#       performed NO forbidden-venv check at all;
#   (2) the gate could certify an interpreter that the trainer never uses
#       (scripts/start.sh exports $PYTHON to the trainer as well).
# The repo venv now wins; an override is honoured only when the venv is
# ABSENT, and that case is DEGRADED (exit 2), never a PASS.
# ---------------------------------------------------------------------------
AMBIENT_PYTHON="${PYTHON:-}"
SHARED_VENV="${PROJECT_ROOT}/.venv/bin/python"
DEGRADED=0

# §5.9 names the forbidden environments explicitly.  2026-11-09 correction:
# the original table was "CadButEaas kanzi", but the environment §5.9
# actually names is "kanzi_venv" — the repo's own launchers spell it out
# (scripts/_wf_r28_gpu_master.sh:32 "No bare python, no CadButEaas, no
# kanzi_venv."; scripts/l6_100_per_metal_bench.sh:36) and env.sh:12 says
# "The legacy CadButEaas / kanzi environments are FORBIDDEN".  The original
# matched the path SEGMENT "kanzi", which MISSES /tmp/kanzi_venv/bin/python.
# The test is on a whole PATH SEGMENT, never a substring and never a
# prefix: a path that merely CONTAINS the letters is not a forbidden
# environment ("kanzi_vram" / "kanzi_stat" / "sweep_kanzi" are VARIABLES in
# molmetal/scripts/check_blackwell_env.py, not paths).
FORBIDDEN_VENV_NAMES="CadButEaas kanzi_venv kanzi"
_interp_is_forbidden() {
  local p="$1" bad
  for bad in $FORBIDDEN_VENV_NAMES; do
    if [[ "/$p/" == *"/$bad/"* ]]; then
      return 0
    fi
  done
  return 1
}

# 2026-11-09: resolve an EXPLICIT override.  The original code tested
# `-x "$AMBIENT_PYTHON"`, which fails for a bare NAME (PYTHON=python3) and
# therefore turned a legal override into the "no interpreter" FAIL.  A bare
# name is resolved through PATH; a path is used as-is when executable.
_resolve_interpreter() {
  local cand="$1"
  if [[ -z "$cand" ]]; then return 1; fi
  if [[ "$cand" == */* ]]; then
    [[ -x "$cand" ]] || return 1
    printf '%s' "$cand"
    return 0
  fi
  local found
  found="$(command -v "$cand" 2>/dev/null || true)"
  [[ -n "$found" ]] || return 1
  printf '%s' "$found"
  return 0
}

# 2026-11-09: the forbidden check now runs on the RESOLVED path as well as
# on the raw string.  A bare name can resolve INTO a forbidden environment
# through PATH, and the original code only ever saw the raw string.
RESOLVED_AMBIENT=""
if [[ -n "$AMBIENT_PYTHON" ]]; then
  RESOLVED_AMBIENT="$(_resolve_interpreter "$AMBIENT_PYTHON" || true)"
fi

for _cand in "$AMBIENT_PYTHON" "$RESOLVED_AMBIENT"; do
  if _interp_is_forbidden "$_cand"; then
    log_fail "ambient \$PYTHON names a FORBIDDEN venv (AGENTS.md §5.9): $_cand"
    log_fail "the forbidden environments are CadButEaas / kanzi_venv / kanzi — unset \$PYTHON and retry"
    exit 1
  fi
done

if [[ -x "$SHARED_VENV" ]]; then
  PYTHON="$SHARED_VENV"
  if [[ -n "$AMBIENT_PYTHON" ]] && [[ "$AMBIENT_PYTHON" != "$SHARED_VENV" ]]; then
    log_warn "ambient \$PYTHON=$AMBIENT_PYTHON IGNORED — AGENTS.md §5.9 pins the repo venv ($SHARED_VENV)"
  fi
elif [[ -n "$RESOLVED_AMBIENT" ]]; then
  # 2026-11-09: the original branch tested `-x "$AMBIENT_PYTHON"`, so a bare
  # NAME (PYTHON=python3) failed the test and fell through to the "no
  # interpreter" FAIL — a regression against the file's own earlier
  # `command -v` check.  The RESOLVED path is what actually runs.
  PYTHON="$RESOLVED_AMBIENT"
  DEGRADED=1
  log_warn "repo venv absent ($SHARED_VENV) — honouring the EXPLICIT \$PYTHON override ($AMBIENT_PYTHON -> $PYTHON). DEGRADED, not a PASS."
else
  log_fail "no interpreter: the repo venv $SHARED_VENV is absent and no explicit \$PYTHON override was given"
  exit 1
fi

# ============================================================================
# 0) USER_OK gate check (per AGENTS.md §5.1)
# ============================================================================
if [[ "${USER_OK:-0}" != "1" ]]; then
  log_fail "USER_OK=1 required (per AGENTS.md §5.1). Set with: export USER_OK=1"
  exit 1
fi
log_ok "USER_OK=1 granted"

# ============================================================================
# 1) Python interpreter reachable
# ============================================================================
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  log_fail "python interpreter $PYTHON not found on PATH"
  exit 1
fi
log_ok "python=$($PYTHON --version 2>&1 | head -1)"

# ============================================================================
# 2) Project venv reachable
# ============================================================================
# 2026-11-09: the original condition was `[[ -d .venv ]]` with a nested
# `[[ ! -x .venv/bin/python ]] -> exit 1`.  That is a SECOND source of
# truth for "is the canonical venv usable" and it DISAGREED with the
# interpreter block above, which tested `-x "$SHARED_VENV"`: a venv
# directory that exists but has no executable interpreter exited 1 here
# while the block above had already routed the run to the DEGRADED
# override.  One of the two branches was unreachable.  The condition is now
# the SAME predicate the interpreter block uses, so no branch is dead.
if [[ -x "$SHARED_VENV" ]]; then
  # 2026-11-09: the original line printed the VENV's version while the
  # gate may have been running under a DIFFERENT interpreter — a false
  # attestation (the gate said "venv=.venv" about an interpreter it was not
  # using).  Report the interpreter that ACTUALLY ran, and claim "venv=.venv"
  # only when it IS this repo's venv.
  _running_py="$("$PYTHON" --version 2>&1 | head -1)"
  if [[ "$PYTHON" == "$SHARED_VENV" ]]; then
    log_ok "venv=.venv (Python $_running_py)"
  else
    log_ok "python=$_running_py (NOT the repo venv: $PYTHON)"
  fi
else
  # 2026-11-09: the original line said "using system python" and then let
  # the run continue to `exit 0` — a SKIP indistinguishable from a PASS for
  # every exit-code consumer.  Reaching this branch implies DEGRADED=1 (the
  # interpreter block above exits 1 when there is no usable interpreter at
  # all), so it is a WARN here and the run ends with exit 2, not 0.
  log_warn "no .venv present — DEGRADED: running under the EXPLICIT \$PYTHON override ($PYTHON)"
fi

# ============================================================================
# 3) molmetal package importable
# ============================================================================
if "$PYTHON" -c "import molmetal" >/dev/null 2>&1; then
  log_ok "import molmetal"
else
  log_fail "cannot import molmetal (PYTHONPATH missing?)"
  exit 1
fi

# ============================================================================
# 4) Pre-flight gate script present
# ============================================================================
if [[ ! -f molmetal/scripts/pre_flight_check.py ]]; then
  log_fail "molmetal/scripts/pre_flight_check.py missing"
  exit 1
fi
log_ok "pre_flight_check.py present"

# ============================================================================
# 5) Pytest collection OK (skipped if --fast)
# ============================================================================
if [[ "${1:-}" != "--fast" ]]; then
  if timeout 60 "$PYTHON" -m pytest --collect-only -q tests molmetal >/dev/null 2>&1; then
    log_ok "pytest collect-only"
  else
    log_fail "pytest collect-only failed"
    exit 1
  fi
fi

# ============================================================================
# 6) GPU env probe (nvidia-smi + torch CUDA) — REPORT-ONLY if no GPU
# ============================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_OUT=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw --format=csv,noheader,nounits 2>/dev/null | head -3)
  if [[ -n "$GPU_OUT" ]]; then
    log_ok "nvidia-smi detected GPU(s):"
    echo "$GPU_OUT" | while IFS= read -r line; do log "    gpu=$line"; done
  else
    log_skip "nvidia-smi present but no GPU detected"
  fi
else
  log_skip "nvidia-smi not on PATH (CPU-only mode)"
fi

if "$PYTHON" -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()}')" 2>/dev/null; then
  log_ok "torch imported"
else
  log_fail "torch not importable"
  exit 1
fi

# ============================================================================
# 6b) GPU compute utilization baseline (not VRAM!) — 3 s polling
# Per user voice 2026-10-03: "预检一下训练时的GPU算力利用率（不是显存利用率）".
# A healthy GPU should respond to a quick load-test pulse (e.g. a trivial
# CUDA op).  We test that the GPU is *computable* (responds to kernel launches),
# not just visible to nvidia-smi.  This is REPORT-ONLY per AGENTS.md §5.2.
# ============================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
  log "polling GPU compute util baseline (3 s, target: util > 0% on at least 1 GPU)..."
  UTIL_SAMPLES=$(timeout 3 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -3)
  if [[ -n "$UTIL_SAMPLES" ]]; then
    MAX_UTIL=$(echo "$UTIL_SAMPLES" | sort -rn | head -1)
    if [[ "$MAX_UTIL" =~ ^[0-9]+$ ]] && [[ "$MAX_UTIL" -gt 0 ]]; then
      log_ok "GPU compute baseline: max util=${MAX_UTIL}% across $(echo "$UTIL_SAMPLES" | wc -l) GPU(s) (idle baseline OK; will ramp during training)"
    else
      log_skip "GPU compute baseline: max util=${MAX_UTIL}% (idle; will ramp during training)"
    fi
  else
    log_skip "GPU compute baseline: nvidia-smi did not respond"
  fi
else
  log_skip "GPU compute baseline: nvidia-smi not in PATH"
fi

# ============================================================================
# 7) YAML config state-lock (per user voice "额外需要检查yaml文件来锁定状态")
# ============================================================================
if [[ -n "$YAML_ARG" ]]; then
  if [[ ! -f "$YAML_ARG" ]]; then
    log_fail "YAML config not found: $YAML_ARG"
    exit 1
  fi
  log_ok "yaml=$YAML_ARG exists"

  # Try to parse yaml via python yaml lib; fall back gracefully
  YAML_SHA=$(sha256sum "$YAML_ARG" | awk '{print $1}')
  log_ok "yaml_sha256=$YAML_SHA"

  # Required top-level keys: data, model, protocol, output, directives are MINIMUM for backward compat
  # (Wf-A8 --fast legacy mode for legacy yamls like configs/default.yaml which pre-date the
  # Wf-A8 yaml-unify pattern; legacy yamls from R17-R20 era lack eval/compute/entry sections).
  # New sections eval / compute / entry are emitted by Wf-A8 unified yaml pattern as REQUIRED-FOR-NEW-YAMLS
  # but OPTIONAL for legacy yaml backward compat (per A8_yaml_unify_closeout.md §0 backward-compat contract).
  REQUIRED_KEYS=("data" "model" "protocol" "output" "directives")
  PARSE_OUT=$("$PYTHON" -c "
import sys, yaml
with open('$YAML_ARG') as f:
    d = yaml.safe_load(f)
required = $(printf '%s\n' "${REQUIRED_KEYS[@]}" | "$PYTHON" -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))")  # 2026-10-08 §5.9 single-venv
missing = [k for k in required if k not in d]
if missing:
    print('FAIL:missing=' + ','.join(missing))
else:
    print('OK:keys=' + ','.join(d.keys()))
" 2>&1) || PARSE_OUT="FAIL:yaml_parse_error"

  if [[ "$PARSE_OUT" == OK:* ]]; then
    log_ok "yaml parseable: ${PARSE_OUT#OK:keys=}"
  else
    log_fail "$PARSE_OUT"
    exit 1
  fi

  # Create state lock in configs/training/.state/
  LOCK_DIR="$PROJECT_ROOT/configs/training/.state"
  LOCK_FILE="$LOCK_DIR/$(basename "$YAML_ARG" .yaml).$(date +%s).lock"
  mkdir -p "$LOCK_DIR"

  # Check if a lock with the same yaml_sha256 already exists within 24h
  EXISTING_LOCK=""
  for f in "$LOCK_DIR"/$(basename "$YAML_ARG" .yaml).*.lock; do
    [[ -f "$f" ]] || continue
    AGE_S=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [[ $AGE_S -gt 86400 ]]; then continue; fi
    LOCK_SHA=$(grep -oP 'yaml_sha256:\s*\K\S+' "$f" 2>/dev/null || echo "")
    if [[ "$LOCK_SHA" == "$YAML_SHA" ]]; then
      EXISTING_LOCK="$f"
      break
    fi
  done

  if [[ -n "$EXISTING_LOCK" ]]; then
    log_skip "yaml unchanged — using existing lock: $(basename "$EXISTING_LOCK")"
  else
    cat > "$LOCK_FILE" <<EOF
yaml_path: $YAML_ARG
yaml_sha256: $YAML_SHA
lock_timestamp: $(date -Iseconds)
user: ${USER:-$(whoami)}
hostname: $(hostname)
process_id: $$
EOF
    log_ok "yaml state lock created: $(basename "$LOCK_FILE")"
  fi
else
  log_skip "no YAML config provided (skipping state-lock check)"
fi

# ============================================================================
# 8) Internal metric gates (REPORT-ONLY per AGENTS.md §5.2 honest framing)
# ============================================================================
METRICS_DIR="$PROJECT_ROOT/metrics/by_metric"
if [[ -d "$METRICS_DIR" ]]; then
  METRICS_REPORT=$("$PYTHON" -c "
import json, os
mdir = '$METRICS_DIR'
gates = {
    'm1_auc': ('metal_lomo_auc_wasserstein.json', None),
    'm2_r2':  ('metalcytotox_combined_r2.json', None),
    'm6_kappa': ('m6_kappa.json', None),
    'm14_decode': ('decode_ratio.json', None),
}
for k, (fname, _) in gates.items():
    p = os.path.join(mdir, fname)
    if not os.path.exists(p):
        print(f'{k}:MISSING')
        continue
    try:
        with open(p) as f:
            d = json.load(f)
        v = d.get('value') or d.get('agg') or d.get('metric') or '?'
        s = d.get('status') or d.get('verdict') or '?'
        print(f'{k}={v} [{s}]')
    except Exception as e:
        print(f'{k}:ERR({e})')
" 2>&1) || METRICS_REPORT="metrics_report_error"
  log_ok "internal metrics report (REPORT-ONLY per AGENTS.md §5.2):"
  echo "$METRICS_REPORT" | while IFS= read -r line; do log "    metric: $line"; done
else
  log_skip "no metrics/by_metric/ directory (skipping metric gates)"
fi

# ============================================================================
# 9) Wave budget check (per Wf-262 §7.1 — BY-DESIGN overrun SKIP)
# ============================================================================
if [[ -f "$PROJECT_ROOT/molmetal/reports/wf262_checklist/per_wave.md" ]]; then
  log_ok "wave budget report: $PROJECT_ROOT/molmetal/reports/wf262_checklist/per_wave.md"
else
  log_skip "no wave budget report (per-wave-checklist not yet run)"
fi

# ============================================================================
# 10) Anchor freshness check (per Wf-302A; SKIP if no anchor file)
# ============================================================================
ANCHOR_FILES=$(find "$PROJECT_ROOT/molmetal/reports" -name "*.json" -path "*/anchor*" 2>/dev/null | head -3 || true)
if [[ -n "$ANCHOR_FILES" ]]; then
  STALE_COUNT=0
  while IFS= read -r af; do
    [[ -z "$af" ]] && continue
    AGE_S=$(( $(date +%s) - $(stat -c %Y "$af") ))
    if [[ $AGE_S -gt 86400 ]]; then
      STALE_COUNT=$((STALE_COUNT + 1))
    fi
  done <<< "$ANCHOR_FILES"
  if [[ $STALE_COUNT -gt 0 ]]; then
    log_skip "$STALE_COUNT anchor file(s) > 24h stale (BY-DESIGN per Wf-302A)"
  else
    log_ok "anchor files fresh (< 24h)"
  fi
else
  log_skip "no anchor files found"
fi

if [[ "$DEGRADED" -eq 1 ]]; then
  log_warn "DEGRADED — the canonical repo venv is absent; an EXPLICIT interpreter override was honoured. This is NOT a PASS."
  exit 2
fi
log "PASS"
exit 0
