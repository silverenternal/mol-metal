#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/_dry_run_log_helper.sh — DRY-RUN log schema helper
#
# Wf-363-A1 (2026-09-29) — Polish per-pair DRY_RUN.log analysis for 5 launch
# wrappers. Provides a consistent schema for DRY_RUN.log emission so that
# downstream tooling (test_wf363a1_dry_run_log.py + a future DRY_RUN.log
# dashboard) can validate wrapper / mode / per-pair enumeration / budget /
# USER_OK gate status uniformly.
#
# Usage:
#   source scripts/_dry_run_log_helper.sh
#   emit_dry_run_log "${out_dir}/DRY_RUN.log" \
#       --title "Wf-355-A1 R1" \
#       --wrapper "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh" \
#       --mode "R1_smoke" \
#       --trainer "/abs/path/trainer.py" \
#       --subset "lbap_refined_ic50_assay" \
#       --flags "(--configs lbap_refined_ic50_assay ...)" \
#       --gpu "cpu" \
#       --budget-hr "0.5" \
#       --user-ok "OPTIONAL_smoke" \
#       --pair "Re seed=0" \
#       --pair "Os seed=4"
#
# Schema (Wf-363-A1):
#   DRY-RUN launcher for <title>
#     wrapper: <wrapper path>
#     mode:    <mode label>
#     trainer: <trainer path>
#     pair:    <pair label>           # repeated for per-pair enumeration
#     pair:    <pair label>
#     ...
#     flags:   <flags>
#     gpu:     <device>
#     budget:  <hours> hr
#     user_ok: <REQUIRED|OPTIONAL_smoke|OPEN|n/a>
#     ts:      <ISO-8601 timestamp>
#   per_pair:
#     <metal|subset> seed=<seed>   ->  <out_dir>
#     ...
#
# Returns 0 on success, 1 on schema-validation failure.
# -----------------------------------------------------------------------------

set -euo pipefail

emit_dry_run_log() {
    local out_path="$1"; shift

    local title="" wrapper="" mode="" trainer="" flags="" gpu="" budget_hr="" user_ok="n/a"
    local config="" subset="" raw_pairs=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --title)    title="${2:-}";       shift 2 ;;
            --wrapper)  wrapper="${2:-}";     shift 2 ;;
            --mode)     mode="${2:-}";        shift 2 ;;
            --trainer)  trainer="${2:-}";     shift 2 ;;
            --config)   config="${2:-}";      shift 2 ;;
            --subset)   subset="${2:-}";      shift 2 ;;
            --flags)    flags="${2:-}";       shift 2 ;;
            --gpu)      gpu="${2:-}";         shift 2 ;;
            --budget-hr) budget_hr="${2:-}";  shift 2 ;;
            --user-ok)  user_ok="${2:-n/a}";  shift 2 ;;
            --pair)
                raw_pairs+=("${2:-}")
                shift 2
                ;;
            *) printf '[FATAL] emit_dry_run_log: unknown arg %s\n' "$1" >&2; return 1 ;;
        esac
    done

    # Schema validation (Wf-363-A1)
    [ -n "${title}" ]    || { printf '[FATAL] emit_dry_run_log: --title required\n' >&2; return 1; }
    [ -n "${wrapper}" ]  || { printf '[FATAL] emit_dry_run_log: --wrapper required\n' >&2; return 1; }
    [ -n "${trainer}" ]  || { printf '[FATAL] emit_dry_run_log: --trainer required\n' >&2; return 1; }
    [ -n "${flags}" ]    || { printf '[FATAL] emit_dry_run_log: --flags required\n' >&2; return 1; }
    [ -n "${gpu}" ]      || { printf '[FATAL] emit_dry_run_log: --gpu required\n' >&2; return 1; }
    [ -n "${budget_hr}" ] || { printf '[FATAL] emit_dry_run_log: --budget-hr required\n' >&2; return 1; }

    mkdir -p "$(dirname "${out_path}")"

    {
        printf 'DRY-RUN launcher for %s\n' "${title}"
        printf '  wrapper: %s\n' "${wrapper}"
        printf '  mode:    %s\n' "${mode}"
        printf '  trainer: %s\n' "${trainer}"
        if [ -n "${config}" ]; then
            printf '  config:  %s\n' "${config}"
        fi
        if [ -n "${subset}" ]; then
            printf '  subset:  %s\n' "${subset}"
        fi
        # Per-pair enumeration (Wf-363-A1): each pair on its own `pair:` line
        for p in "${raw_pairs[@]:-}"; do
            if [ -n "${p}" ]; then
                printf '  pair:    %s\n' "${p}"
            fi
        done
        printf '  flags:   %s\n' "${flags}"
        printf '  gpu:     %s\n' "${gpu}"
        printf '  budget:  %s hr\n' "${budget_hr}"
        printf '  user_ok: %s\n' "${user_ok}"
        printf '  ts:      %s\n' "$(date -Iseconds)"
        # Summary block for downstream tooling
        if [ "${#raw_pairs[@]}" -gt 0 ]; then
            printf 'per_pair:\n'
            for p in "${raw_pairs[@]:-}"; do
                if [ -n "${p}" ]; then
                    printf '  %s\n' "${p}"
                fi
            done
        fi
    } > "${out_path}"
}

# Stand-alone entry point for ad-hoc emission
if [ "${0##*/}" = "_dry_run_log_helper.sh" ] && [ $# -gt 0 ]; then
    emit_dry_run_log "$@"
fi