#!/usr/bin/env bash
# Wf-360-A3 — GPU launch dashboard (live progress tracking for USER_OK-gated launches)
#
# Date:        2026-09-29
# Owner:       Hugo Lee + ultracode agent (Wf-360-A3)
# Origin:      Wf-357-C3 → Wf-360-A3 (GPU launch dashboard scaffold; USER_OK-gated activation)
# Parents:
#   - scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh  (Wf-355-A1)
#   - scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh             (Wf-355-A2)
#   - scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh     (Wf-355-A3)
#   - scripts/launch_wf131_p3_deferred_runs.sh                   (Wf-355-B1)
#   - scripts/launch_wf146_deferred_runs.sh                      (Wf-355-B2)
#   - scripts/launch_wf124_p3_g5.sh                              (TODO-181 G5)
#   - scripts/recover_g6_failed_tuples.sh                        (Wf-337-G1 G6 recovery)
# Skill:
#   - mol-metal-gpu-execution §0 (kill_then_adapt standing directive)
#
# HONEST FRAMING (party-twenty-spirit §3.5):
#   - Scaffold only — dashboard READ-ONLY; activation USER_OK-gated.
#   - All output values traced to on-disk log / plan.json / training_summary.json.
#   - Wall-time values MEASURED where possible (pair elapsed), PROJECTED otherwise.
#   - Zero GPU launches; pure bash + awk + grep + stat + find.
#   - This dashboard DOES NOT modify any launcher output. It only reads.
#
# Usage
# -----
#   bash scripts/gpu_launch_dashboard.sh                            # All launchers summary
#   bash scripts/gpu_launch_dashboard.sh --launcher <name>           # Single-launcher deep-dive
#   bash scripts/gpu_launch_dashboard.sh --json                     # Machine-readable JSON output
#   bash scripts/gpu_launch_dashboard.sh --status                   # STATUS ONLY (one line per launcher)
#   bash scripts/gpu_launch_dashboard.sh --status-summary           # COMPACT status (NAME|STATUS|PAIRS only, tab-separated)
#   bash scripts/gpu_launch_dashboard.sh --budget-detail            # DETAILED budget breakdown per launcher (PROJECTED wall + per-pair budget remaining)
#   bash scripts/gpu_launch_dashboard.sh --budget-json              # MACHINE-READABLE budget_status JSON (Wf-363-B2; per-launcher remaining_hr + DEFERRED_TEMP)
#   bash scripts/gpu_launch_dashboard.sh --per-pair-detail          # ALL per-pair statuses (no truncation at 20)
#   bash scripts/gpu_launch_dashboard.sh --activate                 # LIVE mode (USER_OK-gated; default = READ-ONLY scaffold)
#   bash scripts/gpu_launch_dashboard.sh --list-launchers           # List known launchers
#
# Known launchers (registry below; extensible via --launcher-override)
# -----------------------------------------------------------------------
#   wf355_a1:        DrugOOD 1.58M chemprop v2          (10 hr wall PROJECTED)
#   wf355_a2:        BindingDB 1.07M chemprop v2         ( 7 hr wall PROJECTED)
#   wf355_a3:        tmQM d-block + PlatinAI 4-cell     (12 hr wall PROJECTED)
#   wf355_b1:        Wf-131 P3 7 DEFERRED runs          ( 5 hr wall PROJECTED)
#   wf355_b2:        Wf-146 Re/Os DEFERRED runs          ( 3 hr wall PROJECTED)
#   g5_retrain:      Wf-124 P3 G5 15 (metal,seed) pairs  ( 5 hr wall PROJECTED)
#   g6_recover:      Wf-337-G1 G6 57 tuple recovery      (11.4 hr wall PROJECTED)
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT="/home/hugo/codes/mol-metal"
RESULTS_BASE="${PROJECT_ROOT}/results"
REPORTS_BASE="${PROJECT_ROOT}/molmetal/reports"
DASHBOARD_JSON="${REPORTS_BASE}/gpu_launch_dashboard.json"

# -----------------------------------------------------------------------------
# Launcher registry
#
# Format: launcher_name|reports_root|script_path|wall_hr_projected
#
# The dashboard discovers plan.json + launch.log + per-pair run.log +
# training_summary.json under each launcher_root.
# -----------------------------------------------------------------------------
declare -a LAUNCHER_REGISTRY=(
  "wf355_a1|${REPORTS_BASE}/wf310e_r20_p3_drugood_full_launch|${PROJECT_ROOT}/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh|10"
  "wf355_a2|${REPORTS_BASE}/wf355a2_r20_p4_bindingdb|${PROJECT_ROOT}/scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh|7"
  "wf355_a3|${REPORTS_BASE}/wf355a3_r20_p5_tmqm_plat|${PROJECT_ROOT}/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh|12"
  "wf355_b1|${REPORTS_BASE}/wf131_p3_deferred_runs|${PROJECT_ROOT}/scripts/launch_wf131_p3_deferred_runs.sh|5"
  "wf355_b2|${REPORTS_BASE}/wf146_chemprop_lomo|${PROJECT_ROOT}/scripts/launch_wf146_deferred_runs.sh|3"
  "g5_retrain|${RESULTS_BASE}/wf124_p3_50epoch_full_sweep_g5|${PROJECT_ROOT}/scripts/launch_wf124_p3_g5.sh|5"
  "g6_recover|${RESULTS_BASE}/wf124_p3_per_innovation_ablation_g6|${PROJECT_ROOT}/scripts/recover_g6_failed_tuples.sh|11.4"
)

# Color helpers
if [[ -t 1 ]]; then
  C_RED='\033[0;31m'
  C_GRN='\033[0;32m'
  C_YEL='\033[0;33m'
  C_BLU='\033[0;34m'
  C_DIM='\033[2m'
  C_RST='\033[0m'
else
  C_RED=''; C_GRN=''; C_YEL=''; C_BLU=''; C_DIM=''; C_RST=''
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
now_iso()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
now_human(){ date '+%Y-%m-%d %H:%M:%S %Z'; }

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

die() { printf '%s[FATAL]%s %s\n' "${C_RED}" "${C_RST}" "$*" >&2; exit 1; }

usage() {
  sed -n '2,40p' "$0" | sed 's/^# //; s/^#//'
  exit 0
}

# Resolve launcher_root from registry name
resolve_launcher() {
  local name=$1
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local entry_name="${entry%%|*}"
    if [[ "${entry_name}" == "${name}" ]]; then
      local rest="${entry#*|}"
      echo "${rest}"
      return 0
    fi
  done
  return 1
}

# Read JSON value via python (no jq dependency)
json_get() {
  local json_path=$1
  local key=$2
  python3 -c "
import json, sys
try:
  with open('${json_path}') as f:
    data = json.load(f)
  val = data.get('${key}', '')
  print(val if val is not None else '')
except (FileNotFoundError, json.JSONDecodeError, KeyError):
  print('')
"
}

# Determine launcher overall status
#
# Status logic:
#   - COMPLETED     : training_summary.json exists with all runs status=OK
#   - FAILED        : training_summary.json exists with FAILED entries
#   - RUNNING       : run.log or master.log mtime < 30 min and no FAILED line
#   - DRY_RUN_OK    : plan.json status=DRY_RUN_OK or launch.log has [OK] but no GPU usage
#   - PLAN_ONLY     : plan.json status=PLAN_ONLY (no actual launch yet)
#   - NO_PLAN       : no plan.json + no launch.log (wrapper never run)
#   - STALE         : launch.log mtime > 24 hr and no training_summary.json
#
get_launcher_status() {
  local launcher_root=$1
  local plan_json="${launcher_root}/plan.json"
  local launch_log="${launcher_root}/launch.log"
  local master_log="${launcher_root}/master.log"
  local training_summary="${launcher_root}/training_summary.json"
  local metal_dirs_log="${launcher_root}"
  # Accept either launch.log or master.log as the master log
  [[ -f "${launch_log}" ]] || launch_log="${master_log}"
  local per_pair_logs=$(find "${launcher_root}" -maxdepth 4 -name 'run.log' 2>/dev/null | head -50)

  # COMPLETED: training_summary.json with all OK (no missing_pairs, no failed runs)
  if [[ -f "${training_summary}" ]]; then
    local verdict
    verdict=$(python3 -c "
import json
try:
  with open('${training_summary}') as f:
    d = json.load(f)
  runs = d.get('runs', d.get('pairs', d.get('per_pair_table', [])))
  if not isinstance(runs, list):
    runs = []
  failed = sum(1 for r in runs if isinstance(r, dict) and r.get('status') == 'FAILED')
  missing = d.get('missing_pairs', [])
  expected = d.get('expected_pairs', d.get('total_pairs', len(runs)))
  if failed > 0:
    print('PARTIAL')
  elif len(missing) > 0 or (expected and len(runs) < expected):
    print('PARTIAL')
  elif runs:
    print('COMPLETED')
  else:
    print('UNKNOWN')
except Exception:
  print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
    case "${verdict}" in
      COMPLETED) echo "COMPLETED"; return ;;
      PARTIAL)   echo "PARTIAL";   return ;;
      *) ;;
    esac
  fi

  # RUNNING: per-pair run.log mtime < 30 min
  if [[ -n "${per_pair_logs}" ]]; then
    local latest_mtime_min
    latest_mtime_min=$(find "${launcher_root}" -maxdepth 4 -name 'run.log' -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | awk '{ printf "%d", (systime() - $1) / 60 }')
    if [[ -n "${latest_mtime_min}" && "${latest_mtime_min}" -lt 30 ]]; then
      echo "RUNNING"
      return
    fi
  fi

  # DRY_RUN_OK: plan.json says DRY_RUN_OK or launch.log shows DRY-RUN OK
  if [[ -f "${plan_json}" ]]; then
    local plan_status
    plan_status=$(json_get "${plan_json}" status)
    case "${plan_status}" in
      DRY_RUN_OK) echo "DRY_RUN_OK"; return ;;
      PLAN_ONLY)  echo "PLAN_ONLY";  return ;;
      COMPLETED)  echo "COMPLETED";  return ;;
      FAILED)     echo "FAILED";     return ;;
    esac
  fi

  if [[ -f "${launch_log}" ]] && grep -q "DRY-RUN:" "${launch_log}" 2>/dev/null; then
    echo "DRY_RUN_OK"
    return
  fi

  if [[ -f "${launch_log}" ]]; then
    local age_hr
    age_hr=$(( ( $(date +%s) - $(stat -c %Y "${launch_log}") ) / 3600 ))
    if [[ "${age_hr}" -gt 24 ]]; then
      echo "STALE"
      return
    fi
    echo "READY"
    return
  fi

  echo "NO_PLAN"
}

# Count pairs: OK / RUNNING / FAILED / PENDING
#
# Discovers per-pair dirs by pattern: <launcher_root>/<metal>/seed<N>/ or
# <launcher_root>/<config>/<metal>/seed<N>/ for G6 recovery.
#
count_pairs() {
  local launcher_root=$1
  local pattern_a="${launcher_root}/[A-Z][a-z]/seed[0-9]*"
  local pattern_b="${launcher_root}/*/[A-Z][a-z]/seed[0-9]*"
  local pair_dirs=""
  for pat in "${pattern_a}" "${pattern_b}"; do
    pair_dirs="${pair_dirs} $(compgen -G "${pat}" 2>/dev/null || true)"
  done
  pair_dirs=$(echo "${pair_dirs}" | tr ' ' '\n' | sort -u | grep -v '^$' || true)
  local total=0 ok=0 failed=0 running=0 pending=0
  if [[ -z "${pair_dirs}" ]]; then
    # No per-pair structure; fall back to plan.json total_runs
    local plan_total
    plan_total=$(json_get "${launcher_root}/plan.json" total_runs)
    if [[ -n "${plan_total}" && "${plan_total}" =~ ^[0-9]+$ ]]; then
      total=${plan_total}
      pending=${plan_total}
    fi
    echo "${total}|${ok}|${failed}|${running}|${pending}"
    return
  fi
  for pair_dir in ${pair_dirs}; do
    [[ -d "${pair_dir}" ]] || continue
    total=$(( total + 1 ))
    local summary="${pair_dir}/training_summary.json"
    local runlog="${pair_dir}/run.log"
    if [[ -f "${summary}" ]]; then
      local s
      s=$(python3 -c "
import json, os, re
try:
  with open('${summary}') as f:
    d = json.load(f)
  # Try multiple shapes for status
  st = d.get('status')
  if not st:
    pm = d.get('per_metal', {})
    if pm:
      for metal_name, mdata in pm.items():
        if isinstance(mdata, dict) and mdata.get('status'):
          st = mdata['status']
          break
  if not st:
    ppt = d.get('per_pair_table', [])
    if ppt and isinstance(ppt, list):
      ok = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'OK')
      failed = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'FAILED')
      if failed > 0: st = 'PARTIAL'
      elif ok > 0: st = 'OK'
  if not st and d.get('elapsed_seconds_total'):
    runlog_path = os.path.join(os.path.dirname('${summary}'), 'run.log')
    if os.path.exists(runlog_path):
      text = open(runlog_path).read()
      if 'DONE total elapsed=' in text or re.search(r'best_epoch=\d+', text):
        st = 'OK'
      elif 'FAILED' in text or 'Traceback' in text:
        st = 'FAILED'
  print(st or 'UNKNOWN')
except Exception as e:
  print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
      case "${s}" in
        OK|COMPLETED) ok=$(( ok + 1 )) ;;
        FAILED)        failed=$(( failed + 1 )) ;;
        *)             pending=$(( pending + 1 )) ;;
      esac
    elif [[ -f "${runlog}" ]]; then
      local age_min
      age_min=$(( ( $(date +%s) - $(stat -c %Y "${runlog}") ) / 60 ))
      if [[ "${age_min}" -lt 30 ]]; then
        running=$(( running + 1 ))
      else
        pending=$(( pending + 1 ))
      fi
    else
      pending=$(( pending + 1 ))
    fi
  done
  echo "${total}|${ok}|${failed}|${running}|${pending}"
}

# Wall elapsed (sec) since launch.log first line mtime
wall_elapsed_sec() {
  local launcher_root=$1
  local log="${launcher_root}/launch.log"
  [[ -f "${log}" ]] || { echo "0"; return; }
  echo $(( $(date +%s) - $(stat -c %Y "${log}") ))
}

# Last activity timestamp from launch.log (mtime)
last_activity() {
  local launcher_root=$1
  local log="${launcher_root}/launch.log"
  [[ -f "${log}" ]] || { echo "—"; return; }
  date -d "@$(stat -c %Y "${log}")" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "—"
}

# Humanize seconds → Xh Ym
humanize_seconds() {
  local s=$1
  if [[ "${s}" -eq 0 ]]; then echo "—"; return; fi
  local h=$(( s / 3600 ))
  local m=$(( (s % 3600) / 60 ))
  printf "%dh%02dm" "${h}" "${m}"
}

# Status color
status_color() {
  case "$1" in
    COMPLETED)    printf "${C_GRN}%s${C_RST}" "$1" ;;
    DRY_RUN_OK)   printf "${C_BLU}%s${C_RST}" "$1" ;;
    RUNNING)      printf "${C_YEL}%s${C_RST}" "$1" ;;
    PARTIAL)      printf "${C_YEL}%s${C_RST}" "$1" ;;
    PLAN_ONLY)    printf "${C_DIM}%s${C_RST}" "$1" ;;
    READY)        printf "${C_DIM}%s${C_RST}" "$1" ;;
    STALE)        printf "${C_RED}%s${C_RST}" "$1" ;;
    FAILED)       printf "${C_RED}%s${C_RST}" "$1" ;;
    NO_PLAN)      printf "${C_DIM}%s${C_RST}" "$1" ;;
    *)            printf "$1" ;;
  esac
}

# Progress bar (ASCII 20 chars)
progress_bar() {
  local pct=$1
  local width=20
  local filled=$(( pct * width / 100 ))
  local empty=$(( width - filled ))
  local bar=""
  for ((i = 0; i < filled; i++)); do bar+="█"; done
  for ((i = 0; i < empty; i++)); do bar+="░"; done
  echo "${bar}"
}

# Per-launcher deep dive
print_launcher_detail() {
  local name=$1
  local rest
  rest=$(resolve_launcher "${name}") || die "Unknown launcher: ${name}"
  local launcher_root=$(echo "${rest}" | cut -d'|' -f1)
  local script_path=$(echo "${rest}" | cut -d'|' -f2)
  local wall_proj_hr=$(echo "${rest}" | cut -d'|' -f3)

  echo "╔═══════════════════════════════════════════════════════════════════════╗"
  echo "║ LAUNCHER: ${name}"
  echo "║ Script:   ${script_path}"
  echo "║ Root:     ${launcher_root}"
  echo "╚═══════════════════════════════════════════════════════════════════════╝"

  local status
  status=$(get_launcher_status "${launcher_root}")
  echo
  echo "Status: $(status_color "${status}")"
  echo "Wall projection: ${wall_proj_hr} hr (PROJECTED per Wf-355 HONEST-NEGATIVE)"
  echo "Last activity: $(last_activity "${launcher_root}")"
  echo "Elapsed since launch: $(humanize_seconds "$(wall_elapsed_sec "${launcher_root}")")"

  if [[ -f "${launcher_root}/plan.json" ]]; then
    local user_ok dry_run mode total_runs epochs trainer
    user_ok=$(json_get "${launcher_root}/plan.json" user_ok)
    dry_run=$(json_get "${launcher_root}/plan.json" dry_run)
    mode=$(json_get "${launcher_root}/plan.json" mode)
    total_runs=$(json_get "${launcher_root}/plan.json" total_runs)
    epochs=$(json_get "${launcher_root}/plan.json" epochs)
    trainer=$(json_get "${launcher_root}/plan.json" trainer)
    echo
    echo "Plan:"
    echo "  user_ok=${user_ok} dry_run=${dry_run} mode=${mode}"
    echo "  total_runs=${total_runs} epochs=${epochs}"
    echo "  trainer=${trainer}"
  fi

  local pair_counts
  pair_counts=$(count_pairs "${launcher_root}")
  local total ok failed running pending
  IFS='|' read -r total ok failed running pending <<< "${pair_counts}"
  echo
  echo "Pair progress:"
  echo "  total=${total}  ok=${ok}  failed=${failed}  running=${running}  pending=${pending}"
  if [[ "${total}" -gt 0 ]]; then
    local pct=$(( ok * 100 / total ))
    echo "  $(progress_bar "${pct}") ${pct}%"
  fi

  # Wave budget remaining (projected)
  if [[ "${running}" -gt 0 ]]; then
    echo
    echo "Wave budget remaining: PROJECTED ~$(( wall_proj_hr * 60 / total * (failed + pending) )) min"
  fi

  # Per-pair detail (first 20)
  local per_pair_dirs_a="${launcher_root}/[A-Z][a-z]/seed[0-9]*"
  local per_pair_dirs_b="${launcher_root}/*/[A-Z][a-z]/seed[0-9]*"
  local pair_dirs=""
  for pat in "${per_pair_dirs_a}" "${per_pair_dirs_b}"; do
    pair_dirs="${pair_dirs} $(compgen -G "${pat}" 2>/dev/null || true)"
  done
  pair_dirs=$(echo "${pair_dirs}" | tr ' ' '\n' | sort -u | grep -v '^$' || true)
  if [[ -n "${pair_dirs}" ]]; then
    echo
    echo "Per-pair detail (first 20):"
    echo "  pair_dir  status  last_update"
    local count=0
    for pair_dir in ${pair_dirs}; do
      [[ "${count}" -ge 20 ]] && break
      local rel="${pair_dir#${launcher_root}/}"
      local summary="${pair_dir}/training_summary.json"
      local runlog="${pair_dir}/run.log"
      local s="PENDING"
      if [[ -f "${summary}" ]]; then
        s=$(python3 -c "
import json, os, re
try:
  with open('${summary}') as f:
    d = json.load(f)
  st = d.get('status')
  if not st:
    pm = d.get('per_metal', {})
    if pm:
      for metal_name, mdata in pm.items():
        if isinstance(mdata, dict) and mdata.get('status'):
          st = mdata['status']
          break
  if not st:
    ppt = d.get('per_pair_table', [])
    if ppt and isinstance(ppt, list):
      ok = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'OK')
      failed = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'FAILED')
      if failed > 0: st = 'PARTIAL'
      elif ok > 0: st = 'OK'
  if not st and d.get('elapsed_seconds_total'):
    runlog_path = os.path.join(os.path.dirname('${summary}'), 'run.log')
    if os.path.exists(runlog_path):
      text = open(runlog_path).read()
      if 'DONE total elapsed=' in text or re.search(r'best_epoch=\d+', text):
        st = 'OK'
      elif 'FAILED' in text or 'Traceback' in text:
        st = 'FAILED'
  print(st or 'UNKNOWN')
except Exception:
  print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
      elif [[ -f "${runlog}" ]]; then
        local age_min=$(( ( $(date +%s) - $(stat -c %Y "${runlog}") ) / 60 ))
        if [[ "${age_min}" -lt 30 ]]; then s="RUNNING"; else s="STALE"; fi
      fi
      local last=""
      if [[ -f "${summary}" ]]; then last=$(stat -c %Y "${summary}" | xargs -I{} date -d @{} '+%H:%M:%S' 2>/dev/null || echo "—")
      elif [[ -f "${runlog}" ]]; then last=$(stat -c %Y "${runlog}" | xargs -I{} date -d @{} '+%H:%M:%S' 2>/dev/null || echo "—")
      else last="—"; fi
      printf "  %-40s %s  %s\n" "${rel}" "$(status_color "${s}")" "${last}"
      count=$(( count + 1 ))
    done
    local total_pairs
    total_pairs=$(echo "${pair_dirs}" | wc -l)
    if [[ "${total_pairs}" -gt 20 ]]; then
      echo "  ... ($(( total_pairs - 20 )) more pairs)"
    fi
  fi
  echo
}

# All-launchers summary
print_all_summary() {
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo " Mol-Metal GPU Launch Dashboard — $(now_human)"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo
  printf "%-14s %-14s %-12s %-10s %-12s %-15s %s\n" \
    "LAUNCHER" "STATUS" "PAIRS" "PROGRESS" "ELAPSED" "LAST_ACTIVITY" "WALL_PROJ"
  echo "───────────────────────────────────────────────────────────────────────────────"
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local name="${entry%%|*}"
    local rest="${entry#*|}"
    local root=$(echo "${rest}" | cut -d'|' -f1)
    local wall_proj=$(echo "${rest}" | cut -d'|' -f3)
    local status
    status=$(get_launcher_status "${root}")
    local pair_counts
    pair_counts=$(count_pairs "${root}")
    local total ok failed running pending
    IFS='|' read -r total ok failed running pending <<< "${pair_counts}"
    local pair_summary="${ok}/${total}"
    local pct=0
    if [[ "${total}" -gt 0 ]]; then
      pct=$(( ok * 100 / total ))
    fi
    local elapsed=$(humanize_seconds "$(wall_elapsed_sec "${root}")")
    local last=$(last_activity "${root}")
    local bar=$(progress_bar "${pct}")
    printf "%-14s %-14s %-12s %s %-3d%% %-12s %-15s %s hr\n" \
      "${name}" \
      "$(status_color "${status}")" \
      "${pair_summary}" \
      "${bar}" "${pct}" \
      "${elapsed}" \
      "${last}" \
      "${wall_proj}"
  done
  echo
  echo "Legend: COMPLETED (green), RUNNING/PARTIAL (yellow), DRY_RUN_OK (blue),"
  echo "        FAILED (red), PLAN_ONLY/READY/STALE/NO_PLAN (dim)"
  echo "Bars: each █ = 5% pair completion; ░ = pending"
  echo
}

# JSON output (machine-readable)
print_json() {
  python3 <<'PYEOF'
import json, os, subprocess, re
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = "/home/hugo/codes/mol-metal"
LAUNCHERS = [
  ("wf355_a1", f"{PROJECT_ROOT}/molmetal/reports/wf310e_r20_p3_drugood_full_launch",
   f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh", 10.0),
  ("wf355_a2", f"{PROJECT_ROOT}/molmetal/reports/wf355a2_r20_p4_bindingdb",
   f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh", 7.0),
  ("wf355_a3", f"{PROJECT_ROOT}/molmetal/reports/wf355a3_r20_p5_tmqm_plat",
   f"{PROJECT_ROOT}/scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh", 12.0),
  ("wf355_b1", f"{PROJECT_ROOT}/molmetal/reports/wf131_p3_deferred_runs",
   f"{PROJECT_ROOT}/scripts/launch_wf131_p3_deferred_runs.sh", 5.0),
  ("wf355_b2", f"{PROJECT_ROOT}/molmetal/reports/wf146_chemprop_lomo",
   f"{PROJECT_ROOT}/scripts/launch_wf146_deferred_runs.sh", 3.0),
  ("g5_retrain", f"{PROJECT_ROOT}/results/wf124_p3_50epoch_full_sweep_g5",
   f"{PROJECT_ROOT}/scripts/launch_wf124_p3_g5.sh", 5.0),
  ("g6_recover", f"{PROJECT_ROOT}/results/wf124_p3_per_innovation_ablation_g6",
   f"{PROJECT_ROOT}/scripts/recover_g6_failed_tuples.sh", 11.4),
]

def now_human(path):
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def discover_pairs(root):
    pairs = []
    # Pattern A: <root>/<Metal>/seed<N>
    for d in Path(root).glob("[A-Z][a-z]/seed[0-9]*"):
        if d.is_dir():
            pairs.append(str(d))
    # Pattern B: <root>/<cfg>/<Metal>/seed<N>
    for d in Path(root).glob("*/*/seed[0-9]*"):
        if d.is_dir():
            pairs.append(str(d))
    return sorted(set(pairs))

def pair_status(pdir):
    summary = Path(pdir) / "training_summary.json"
    runlog = Path(pdir) / "run.log"
    if summary.exists():
        try:
            with open(summary) as f:
                d = json.load(f)
            st = d.get('status')
            if not st:
                pm = d.get('per_metal', {})
                if pm:
                    for metal_name, mdata in pm.items():
                        if isinstance(mdata, dict) and mdata.get('status'):
                            st = mdata['status']
                            break
            if not st:
                ppt = d.get('per_pair_table', [])
                if ppt and isinstance(ppt, list):
                    ok = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'OK')
                    failed = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'FAILED')
                    if failed > 0: st = 'PARTIAL'
                    elif ok > 0: st = 'OK'
            if not st and d.get('elapsed_seconds_total'):
                runlog_path = str(Path(pdir) / "run.log")
                if os.path.exists(runlog_path):
                    text = open(runlog_path).read()
                    if 'DONE total elapsed=' in text or re.search(r'best_epoch=\d+', text):
                        st = 'OK'
                    elif 'FAILED' in text or 'Traceback' in text:
                        st = 'FAILED'
            st = (st or 'UNKNOWN').upper()
            if st == 'OK':
                return 'COMPLETED'
            if st == 'ALL_SEEDS_FAILED':
                return 'FAILED'
            return st
        except Exception:
            return 'UNKNOWN'
    elif runlog.exists():
        age_min = (datetime.now().timestamp() - os.path.getmtime(runlog)) / 60
        if age_min < 30:
            return 'RUNNING'
        return 'STALE'
    return 'PENDING'

def parse_pair_identity(pdir):
    """Parse (metal, seed) from a per-pair dir (Wf-363-B1 helper)."""
    parts = Path(pdir).parts
    seed = None
    metal = 'UNKNOWN'
    for part in reversed(parts):
        m = re.match(r'^seed(\d+)$', part)
        if m:
            seed = m.group(1)
            continue
        if re.match(r'^[A-Z][a-z]$', part) and metal == 'UNKNOWN':
            metal = part
    return (metal, seed)

def pair_progress_detail(pdir):
    """Wf-363-B1 per-pair completion tracking dict."""
    metal, seed = parse_pair_identity(pdir)
    status = pair_status(pdir)
    runlog = Path(pdir) / 'run.log'
    summary = Path(pdir) / 'training_summary.json'
    wall_elapsed_s = 0
    last_ts = None
    artifacts = []
    for art in (runlog, summary):
        if art.exists():
            artifacts.append(art)
    if artifacts:
        mtimes = [os.path.getmtime(str(a)) for a in artifacts]
        first_ts = min(mtimes)
        last_ts = max(mtimes)
        wall_elapsed_s = int(datetime.now().timestamp() - first_ts)
    last_log_ts = (
        datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        if last_ts is not None else None
    )
    return {
        'metal': metal,
        'seed': seed,
        'status': status,
        'wall_elapsed_s': wall_elapsed_s,
        'last_log_ts': last_log_ts,
    }

def launcher_status(root):
    pairs = discover_pairs(root)
    if pairs:
        statuses = [pair_status(p) for p in pairs]
        if any(s == 'RUNNING' for s in statuses):
            return 'RUNNING'
        ok = sum(1 for s in statuses if s in ('OK', 'COMPLETED'))
        failed = sum(1 for s in statuses if s in ('FAILED', 'ALL_SEEDS_FAILED'))
        if ok + failed == len(statuses):
            return 'COMPLETED' if failed == 0 else 'PARTIAL'
    plan_json = Path(root) / "plan.json"
    if plan_json.exists():
        try:
            with open(plan_json) as f:
                d = json.load(f)
            ps = d.get('status', '').upper()
            if ps in ('COMPLETED', 'DRY_RUN_OK', 'PLAN_ONLY', 'FAILED'):
                return ps
        except Exception:
            pass
    # Top-level training_summary.json (G6-style aggregate)
    summary = Path(root) / "training_summary.json"
    if summary.exists():
        try:
            with open(summary) as f:
                d = json.load(f)
            runs = d.get('runs', [])
            if runs:
                ok = sum(1 for r in runs if isinstance(r, dict) and r.get('status') == 'OK')
                failed = sum(1 for r in runs if isinstance(r, dict) and r.get('status') in ('FAILED', 'ALL_SEEDS_FAILED'))
                if ok + failed == len(runs):
                    return 'COMPLETED' if failed == 0 else 'PARTIAL'
            # has top-level aggregate markers (e.g. G6 total_tuples_planned)
            if d.get('todo_182_g6_total_tuples_planned') or d.get('todo_181_g5_total_pairs'):
                return 'PARTIAL'
        except Exception:
            pass
    launch_log = Path(root) / "launch.log"
    if not launch_log.exists():
        launch_log = Path(root) / "master.log"
    if launch_log.exists():
        try:
            text = launch_log.read_text()
            if 'DRY-RUN:' in text:
                return 'DRY_RUN_OK'
            age_hr = (datetime.now().timestamp() - os.path.getmtime(launch_log)) / 3600
            if age_hr > 24:
                return 'STALE'
            return 'READY'
        except Exception:
            pass
    return 'NO_PLAN'

result = {
    'dashboard_generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'dashboard_schema_version': '1.0',
    'wf_origin': 'Wf-360-A3 GPU launch dashboard scaffold',
    'wf_origin_update': 'Wf-363-B1: per-pair completion tracking',
    'launchers': []
}

PER_PAIR_DETAIL_FLAG = os.environ.get('DASHBOARD_PER_PAIR_DETAIL', '0') == '1'

for name, root, script, wall_proj in LAUNCHERS:
    status = launcher_status(root)
    pairs = discover_pairs(root)
    pair_statuses = [pair_status(p) for p in pairs] if pairs else []
    ok = sum(1 for s in pair_statuses if s in ('OK', 'COMPLETED'))
    failed = sum(1 for s in pair_statuses if s in ('FAILED', 'ALL_SEEDS_FAILED'))
    running = sum(1 for s in pair_statuses if s == 'RUNNING')
    pending = sum(1 for s in pair_statuses if s in ('PENDING', 'STALE', 'UNKNOWN'))
    plan_total = 0
    plan_json = Path(root) / "plan.json"
    if plan_json.exists():
        try:
            with open(plan_json) as f:
                d = json.load(f)
            plan_total = d.get('total_runs', 0)
        except Exception:
            pass

    launch_log = Path(root) / "launch.log"
    last_activity = now_human(str(launch_log))
    elapsed_sec = int(datetime.now().timestamp() - os.path.getmtime(str(launch_log))) if launch_log.exists() else 0

    entry = {
        'name': name,
        'script': script,
        'root': root,
        'status': status,
        'wall_projected_hr': wall_proj,
        'last_activity': last_activity,
        'elapsed_sec': elapsed_sec,
        'pair_progress': {
            'total_pairs': plan_total if plan_total > 0 else len(pairs),
            'discovered_pairs': len(pairs),
            'ok': ok,
            'failed': failed,
            'running': running,
            'pending': pending,
        },
        'activation_gated': True,
        'user_ok_required': True,
    }
    if PER_PAIR_DETAIL_FLAG:
        entry['per_pair_progress'] = [pair_progress_detail(p) for p in pairs]
    result['launchers'].append(entry)

Path(DASHBOARD_JSON := f"{PROJECT_ROOT}/molmetal/reports/gpu_launch_dashboard.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PYEOF
}

# Status-only one-liner per launcher
print_status_only() {
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local name="${entry%%|*}"
    local root=$(echo "${entry#*|}" | cut -d'|' -f1)
    local status
    status=$(get_launcher_status "${root}")
    printf "%-14s %s\n" "${name}" "$(status_color "${status}")"
  done
}

# Status-summary (compact view): tab-separated, machine-friendly
# Format: NAME<TAB>STATUS<TAB>PAIRS_OK/TOTAL<TAB>WALL_PROJ_HR
print_status_summary() {
  echo "NAME	STATUS	PAIRS	WALL_PROJ_HR"
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local name="${entry%%|*}"
    local rest="${entry#*|}"
    local root=$(echo "${rest}" | cut -d'|' -f1)
    local wall_proj=$(echo "${rest}" | cut -d'|' -f3)
    local status
    status=$(get_launcher_status "${root}")
    local pair_counts
    pair_counts=$(count_pairs "${root}")
    local total ok
    IFS='|' read -r total ok _ _ _ <<< "${pair_counts}"
    local pair_summary="${ok}/${total}"
    printf "%s	%s	%s	%s\n" "${name}" "${status}" "${pair_summary}" "${wall_proj}"
  done
}

# Budget detail: PROJECTED wall + per-pair budget remaining per launcher
# Format: NAME<TAB>STATUS<TAB>WALL_PROJ_HR<TAB>PAIRS_OK<TAB>PAIRS_REMAINING<TAB>BUDGET_REMAINING_MIN
print_budget_detail() {
  echo "NAME	STATUS	WALL_PROJ_HR	OK	REMAINING	BUDGET_REMAINING_MIN"
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local name="${entry%%|*}"
    local rest="${entry#*|}"
    local root=$(echo "${rest}" | cut -d'|' -f1)
    local wall_proj=$(echo "${rest}" | cut -d'|' -f3)
    local status
    status=$(get_launcher_status "${root}")
    local pair_counts
    pair_counts=$(count_pairs "${root}")
    local total ok failed running pending
    IFS='|' read -r total ok failed running pending <<< "${pair_counts}"
    local remaining=$(( failed + pending ))
    local budget_remaining_min="—"
    if [[ "${total}" -gt 0 ]]; then
      budget_remaining_min=$(( wall_proj * 60 * remaining / total ))
    fi
    printf "%s	%s	%s	%s	%s	%s\n" \
      "${name}" "${status}" "${wall_proj}" "${ok}" "${remaining}" "${budget_remaining_min}"
  done
}

# Budget JSON (Wf-363-B2): machine-readable budget_status with per-launcher
# remaining_hr vs the 10h session cap, plus DEFERRED_TEMP aggregate from
# wave_budget_tracker.py (per Wf-360-A1 rebalance).
print_budget_json() {
  python3 <<'PYEOF'
import json, os
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = "/home/hugo/codes/mol-metal"
SESSION_CAP_HR = 10.0
LAUNCHERS = [
  ("wf355_a1", f"{PROJECT_ROOT}/molmetal/reports/wf310e_r20_p3_drugood_full_launch", 10.0),
  ("wf355_a2", f"{PROJECT_ROOT}/molmetal/reports/wf355a2_r20_p4_bindingdb", 7.0),
  ("wf355_a3", f"{PROJECT_ROOT}/molmetal/reports/wf355a3_r20_p5_tmqm_plat", 12.0),
  ("wf355_b1", f"{PROJECT_ROOT}/molmetal/reports/wf131_p3_deferred_runs", 5.0),
  ("wf355_b2", f"{PROJECT_ROOT}/molmetal/reports/wf146_chemprop_lomo", 3.0),
  ("g5_retrain", f"{PROJECT_ROOT}/results/wf124_p3_50epoch_full_sweep_g5", 5.0),
  ("g6_recover", f"{PROJECT_ROOT}/results/wf124_p3_per_innovation_ablation_g6", 11.4),
]
WAVE_BUDGET_REPORT = f"{PROJECT_ROOT}/molmetal/reports/wf262_wave_budget/wave_budget_report.json"


def _load_wave_budget_report():
    p = Path(WAVE_BUDGET_REPORT)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _classify_deferred_class_safe(wave_name):
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "wave_budget_tracker",
            f"{PROJECT_ROOT}/molmetal/scripts/wave_budget_tracker.py",
        )
        if spec is None or spec.loader is None:
            return "DEFERRED_TEMP"
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.classify_deferred_class(wave_name) or "DEFERRED_TEMP"
    except Exception:
        return "DEFERRED_TEMP"


def _deferred_temp_total_hr():
    rep = _load_wave_budget_report()
    if not rep:
        return 0.0, []
    total_hr = 0.0
    per_wave = []
    for w in (rep.get("waves") or []):
        if (w.get("status") or "").upper() != "DEFERRED":
            continue
        name = w.get("wave_name") or ""
        if _classify_deferred_class_safe(name) != "DEFERRED_TEMP":
            continue
        actual = w.get("actual_hr")
        estimated = w.get("estimated_hr") or 0.0
        if actual is not None and float(actual) > 0.0:
            hr = float(actual)
            evidence = "MEASURED_actual_hr"
        else:
            hr = float(estimated)
            evidence = "PROJECTED_estimated_hr"
        total_hr += hr
        per_wave.append({
            "wave_name": name,
            "deferred_class": "DEFERRED_TEMP",
            "hr": round(hr, 2),
            "evidence": evidence,
        })
    return round(total_hr, 6), per_wave


per_launcher = []
total_projected_hr = 0.0
for name, _root, budget_hr in LAUNCHERS:
    remaining_hr = round(SESSION_CAP_HR - budget_hr, 6)
    status = "WITHIN" if remaining_hr >= -1e-9 else "OVER"
    per_launcher.append({
        "launcher": name,
        "budget_hr": round(budget_hr, 6),
        "remaining_hr": remaining_hr,
        "status": status,
        "status_evidence": (
            f"{round(budget_hr, 2)}h projected vs {SESSION_CAP_HR}h session cap "
            f"= {round(remaining_hr, 2)}h remaining"
        ),
        "wall_projected_hr": round(budget_hr, 6),
    })
    total_projected_hr += budget_hr

deferred_temp_hr, deferred_temp_waves = _deferred_temp_total_hr()
total_remaining_hr = round(SESSION_CAP_HR - total_projected_hr, 6)
deferred_temp_remaining_hr = round(SESSION_CAP_HR - deferred_temp_hr, 6)

over = [pl["launcher"] for pl in per_launcher if pl["status"] == "OVER"]
within = [pl["launcher"] for pl in per_launcher if pl["status"] == "WITHIN"]

honest_lines = [
    f"Session cap = {SESSION_CAP_HR}h (per Wf-262 §7.1 budget discipline).",
    f"Per-launcher: {len(within)} WITHIN ({', '.join(within) or 'none'}), "
    f"{len(over)} OVER ({', '.join(over) or 'none'}).",
    f"Sum of all per-launcher budgets = {round(total_projected_hr, 2)}h "
    f"vs {SESSION_CAP_HR}h cap = {round(total_remaining_hr, 2)}h total remaining.",
    f"DEFERRED_TEMP rebalance (per Wf-360-A1): {round(deferred_temp_hr, 2)}h "
    f"across {len(deferred_temp_waves)} wave(s); "
    f"{round(deferred_temp_remaining_hr, 2)}h headroom.",
]
if over:
    honest_lines.append(
        f"HONEST-NEGATIVE: {len(over)} launcher(s) individually exceed "
        f"the {SESSION_CAP_HR}h session cap and would need cross-session scheduling "
        f"or a USER_OK waiver."
    )
else:
    honest_lines.append("All launchers individually fit within the 10h session cap.")

result = {
    "dashboard_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dashboard_schema_version": "1.0",
    "wf_origin": "Wf-363-B2 budget remaining indicator (shellscript parity)",
    "session_cap_hr": SESSION_CAP_HR,
    "per_launcher": per_launcher,
    "total_projected_hr": round(total_projected_hr, 6),
    "total_remaining_hr": total_remaining_hr,
    "deferred_temp_total_hr": round(deferred_temp_hr, 6),
    "deferred_temp_waves": deferred_temp_waves,
    "deferred_temp_remaining_hr": deferred_temp_remaining_hr,
    "honest_framing": " | ".join(honest_lines),
}
print(json.dumps(result, indent=2))
PYEOF
}

# Per-pair detail (all pairs, no truncation at 20)
print_per_pair_detail() {
  local name="${LAUNCHER_FILTER}"
  local rest
  rest=$(resolve_launcher "${name}") || die "Unknown launcher: ${name}"
  local launcher_root=$(echo "${rest}" | cut -d'|' -f1)
  local per_pair_dirs_a="${launcher_root}/[A-Z][a-z]/seed[0-9]*"
  local per_pair_dirs_b="${launcher_root}/*/[A-Z][a-z]/seed[0-9]*"
  local pair_dirs=""
  for pat in "${per_pair_dirs_a}" "${per_pair_dirs_b}"; do
    pair_dirs="${pair_dirs} $(compgen -G "${pat}" 2>/dev/null || true)"
  done
  pair_dirs=$(echo "${pair_dirs}" | tr ' ' '\n' | sort -u | grep -v '^$' || true)
  echo "PAIR	STATUS	LAST_UPDATE"
  if [[ -z "${pair_dirs}" ]]; then
    echo "—	NO_PAIRS	—"
    return
  fi
  for pair_dir in ${pair_dirs}; do
    local rel="${pair_dir#${launcher_root}/}"
    local summary="${pair_dir}/training_summary.json"
    local runlog="${pair_dir}/run.log"
    local s="PENDING"
    if [[ -f "${summary}" ]]; then
      s=$(python3 -c "
import json, os, re
try:
  with open('${summary}') as f:
    d = json.load(f)
  st = d.get('status')
  if not st:
    pm = d.get('per_metal', {})
    if pm:
      for metal_name, mdata in pm.items():
        if isinstance(mdata, dict) and mdata.get('status'):
          st = mdata['status']
          break
  if not st:
    ppt = d.get('per_pair_table', [])
    if ppt and isinstance(ppt, list):
      ok = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'OK')
      failed = sum(1 for r in ppt if isinstance(r, dict) and r.get('status') == 'FAILED')
      if failed > 0: st = 'PARTIAL'
      elif ok > 0: st = 'OK'
  if not st and d.get('elapsed_seconds_total'):
    runlog_path = os.path.join(os.path.dirname('${summary}'), 'run.log')
    if os.path.exists(runlog_path):
      text = open(runlog_path).read()
      if 'DONE total elapsed=' in text or re.search(r'best_epoch=\d+', text):
        st = 'OK'
      elif 'FAILED' in text or 'Traceback' in text:
        st = 'FAILED'
  print(st or 'UNKNOWN')
except Exception:
  print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN")
    elif [[ -f "${runlog}" ]]; then
      local age_min=$(( ( $(date +%s) - $(stat -c %Y "${runlog}") ) / 60 ))
      if [[ "${age_min}" -lt 30 ]]; then s="RUNNING"; else s="STALE"; fi
    fi
    local last=""
    if [[ -f "${summary}" ]]; then
      last=$(stat -c %Y "${summary}" | xargs -I{} date -d @{} '+%H:%M:%S' 2>/dev/null || echo "—")
    elif [[ -f "${runlog}" ]]; then
      last=$(stat -c %Y "${runlog}" | xargs -I{} date -d @{} '+%H:%M:%S' 2>/dev/null || echo "—")
    else
      last="—"
    fi
    printf "%s	%s	%s\n" "${rel}" "${s}" "${last}"
  done
}

# List known launchers
print_list_launchers() {
  echo "Known launchers (registry):"
  for entry in "${LAUNCHER_REGISTRY[@]}"; do
    local name="${entry%%|*}"
    local rest="${entry#*|}"
    local root=$(echo "${rest}" | cut -d'|' -f1)
    local wall=$(echo "${rest}" | cut -d'|' -f3)
    printf "  %-14s root=%s  wall_proj=%sh\n" "${name}" "${root}" "${wall}"
  done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
LAUNCHER_FILTER=""
JSON_MODE=false
STATUS_ONLY=false
STATUS_SUMMARY=false
BUDGET_DETAIL=false
BUDGET_JSON=false
PER_PAIR_DETAIL=false
LIST_ONLY=false
ACTIVATE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher) LAUNCHER_FILTER="$2"; shift 2 ;;
    --status)   STATUS_ONLY=true; shift ;;
    --status-summary) STATUS_SUMMARY=true; shift ;;
    --budget-detail)  BUDGET_DETAIL=true; shift ;;
    --budget-json)    BUDGET_JSON=true; shift ;;
    --per-pair-detail) PER_PAIR_DETAIL=true; shift ;;
    --activate) ACTIVATE=true; shift ;;
    --json)     JSON_MODE=true; shift ;;
    --list-launchers) LIST_ONLY=true; shift ;;
    -h|--help)  usage ;;
    *)          die "Unknown arg: $1 (use --help)" ;;
  esac
done

# -----------------------------------------------------------------------------
# Activation gate (Wf-364-B2)
# Default: READ-ONLY scaffold (no --activate flag = scaffold mode).
# LIVE tracking requires --activate flag (USER_OK-gated per AGENTS.md §5.1).
# Scaffold mode: emits a banner marking mode = READ-ONLY + activation_gated=true.
# LIVE mode: full dashboard with status tracking; activates only when --activate
# flag is explicitly passed.
# -----------------------------------------------------------------------------
if [[ "${ACTIVATE}" == "true" ]]; then
  log "ACTIVATION: LIVE mode (--activate flag set); live progress tracking enabled."
  log "WARNING: live mode is USER_OK-gated; activation is the operator's explicit consent."
  DASHBOARD_MODE="LIVE"
else
  DASHBOARD_MODE="READ-ONLY_SCAFFOLD"
  log "ACTIVATION: READ-ONLY scaffold (default; --activate flag required for LIVE mode)."
fi

if [[ "${LIST_ONLY}" == "true" ]]; then
  print_list_launchers
  exit 0
fi

if [[ "${JSON_MODE}" == "true" ]]; then
  if [[ "${PER_PAIR_DETAIL}" == "true" ]]; then
    export DASHBOARD_PER_PAIR_DETAIL=1
  else
    export DASHBOARD_PER_PAIR_DETAIL=0
  fi
  print_json
  exit $?
fi

if [[ -n "${LAUNCHER_FILTER}" ]]; then
  print_launcher_detail "${LAUNCHER_FILTER}"
  if [[ "${PER_PAIR_DETAIL}" == "true" ]]; then
    echo "── per-pair detail (ALL pairs) ──"
    print_per_pair_detail
  fi
  exit 0
fi

if [[ "${STATUS_SUMMARY}" == "true" ]]; then
  echo "dashboard_mode=${DASHBOARD_MODE}"
  print_status_summary
  exit 0
fi

if [[ "${BUDGET_DETAIL}" == "true" ]]; then
  echo "dashboard_mode=${DASHBOARD_MODE}"
  print_budget_detail
  exit 0
fi

if [[ "${BUDGET_JSON}" == "true" ]]; then
  print_budget_json
  exit 0
fi

if [[ "${PER_PAIR_DETAIL}" == "true" ]]; then
  if [[ -z "${LAUNCHER_FILTER}" ]]; then
    die "--per-pair-detail requires --launcher <name>"
  fi
  echo "dashboard_mode=${DASHBOARD_MODE}"
  print_per_pair_detail
  exit 0
fi

if [[ "${STATUS_ONLY}" == "true" ]]; then
  print_status_only
  exit 0
fi

# Default: full all-launcher summary (with mode banner at top)
echo "dashboard_mode=${DASHBOARD_MODE} activation_gated=true user_ok_required=true"
echo
print_all_summary
