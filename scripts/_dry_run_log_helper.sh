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
#       --budget-label "PROJECTED" \
#       --wall-policy "single_launch" \
#       --user-ok "OPTIONAL_smoke" \
#       --capability "OK" \
#       --pair "Re seed=0" \
#       --pair "Os seed=4"
#
#   For a wrapper whose budget covers a WHOLE SWEEP of N sequential launches,
#   declare the policy and the per-pair budget explicitly:
#       --wall-policy "whole_sweep_sequential" \
#       --per-pair-budget-hr "0.6"
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
#     budget_label: <MEASURED|PROJECTED>       # Wf-A269 wave-9
#     wall_policy: <single_launch|whole_sweep_sequential>   # Wf-A269 wave-9;
#                                            #   OPT-IN (emitted only when
#                                            #   the producer declared it)
#     per_pair_budget_hr: <hours> hr          # Wf-A269 wave-9; only when the
#                                            #   wall_policy is
#                                            #   whole_sweep_sequential
#     user_ok: <REQUIRED|OPTIONAL_smoke|OPEN|n/a>
#     capability: <OK|BLOCKED|UNVERIFIED>   # OPTIONAL (Wf-A269 wave-9)
#     ts:      <ISO-8601 timestamp>
#   per_pair:
#     <metal|subset> seed=<seed>   ->  <out_dir>
#     ...
#
# WHY wall_policy / per_pair_budget_hr exist (Wf-A269 wave-9, 2026-11-02)
# -----------------------------------------------------------------------
# The 10 hr session cap (AGENTS.md §5.1 + Wf-294B kill_then_adapt) bounds a
# single LAUNCH, not a whole sweep of sequential launches.  The producer is
# the only party that knows both the sweep total and the pair count, so the
# producer DECLARES the policy and the per-pair budget, and the test VERIFIES
# the declaration.  The test does not divide and does not trust a free-text
# label: it checks that the declared per-pair budget reconstructs the declared
# total, and that the per-pair budget is inside the cap.  A producer that
# misstates any one of the three fields fails.
#
# Returns 0 on success, 1 on schema-validation failure.
# -----------------------------------------------------------------------------

set -euo pipefail

emit_dry_run_log() {
    local out_path="$1"; shift

    local title="" wrapper="" mode="" trainer="" flags="" gpu="" budget_hr="" user_ok="n/a"
    local config="" subset="" capability="" gate_verdict="" raw_pairs=()
    # Wf-A269 wave-9 (2026-11-02): the wall-policy + per-pair budget are
    # PRODUCER declarations (see the header).  The consumer (the schema test)
    # verifies them; it never derives them.
    #
    # wall_policy is OPT-IN (default: UNSET = not emitted).  An ABSENT field
    # says "this wrapper did not declare its wall policy"; a present field is
    # a claim.  Defaulting it to "single_launch" would put an unverified claim
    # on every wrapper that never declared one, which is the same defect this
    # wave is repairing (a check that states something it has not measured).
    local budget_label="PROJECTED" wall_policy="" per_pair_budget_hr=""

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
            --budget-label) budget_label="${2:-PROJECTED}"; shift 2 ;;
            --wall-policy) wall_policy="${2:-}"; shift 2 ;;
            --per-pair-budget-hr) per_pair_budget_hr="${2:-}"; shift 2 ;;
            --user-ok)  user_ok="${2:-n/a}";  shift 2 ;;
            --capability) capability="${2:-}"; shift 2 ;;
            --gate-verdict) gate_verdict="${2:-}"; shift 2 ;;
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

    # Wf-A269 wave-9: the declared fields must be self-consistent AT EMIT TIME,
    # so a wrapper cannot publish a log whose own arithmetic is wrong.
    case "${budget_label}" in
        MEASURED|PROJECTED) ;;
        *) printf '[FATAL] emit_dry_run_log: --budget-label must be MEASURED or PROJECTED (got "%s")\n' "${budget_label}" >&2; return 1 ;;
    esac
    case "${wall_policy}" in
        "") ;;   # not declared: the field is simply not emitted
        single_launch) ;;
        whole_sweep_sequential)
            [ -n "${per_pair_budget_hr}" ] || {
                printf '[FATAL] emit_dry_run_log: --wall-policy whole_sweep_sequential requires --per-pair-budget-hr\n' >&2
                return 1
            }
            ;;
        *) printf '[FATAL] emit_dry_run_log: --wall-policy must be single_launch or whole_sweep_sequential (got "%s")\n' "${wall_policy}" >&2; return 1 ;;
    esac

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
        # Wf-A269 wave-9 (2026-11-02): a budget written BEFORE a run is a
        # PROJECTION, not a measurement.  The label makes that explicit on the
        # face of the record instead of leaving the reader to infer it.
        printf '  budget_label: %s\n' "${budget_label}"
        # Wf-A269 wave-9 (2026-11-02): PRODUCER declaration of the wall
        # policy.  The 10 hr session cap bounds a single LAUNCH; a whole
        # sweep of sequential launches is declared as such, together with the
        # per-pair budget it implies.  The consumer verifies the declaration.
        # Emitted ONLY when the producer declared it (see the default note
        # above): an absent field is "not declared", not "single launch".
        if [ -n "${wall_policy}" ]; then
            printf '  wall_policy: %s\n' "${wall_policy}"
        fi
        if [ "${wall_policy}" = "whole_sweep_sequential" ]; then
            printf '  per_pair_budget_hr: %s hr\n' "${per_pair_budget_hr}"
        fi
        printf '  user_ok: %s\n' "${user_ok}"
        # Wf-A269 wave-9 (2026-11-02): OPTIONAL field.  A launcher that runs a
        # pre-launch capability check against its trainer records the verdict
        # here, so the PLAN says whether it could actually be launched instead
        # of leaving that fact only in the wrapper's stderr.
        if [ -n "${capability}" ]; then
            printf '  capability: %s\n' "${capability}"
        fi
        # Wf-A269 wave-10 (2026-11-02): OPTIONAL field.  The MEASURED verdict
        # of the launcher's pre-launch gates (pre-flight / per-wave checklist
        # / wave budget tracker).  It is OPT-IN for the same reason
        # `capability` is: a wrapper that did not run those gates says
        # nothing about them, and a gate that was SKIPPED says UNVERIFIED —
        # never PASS.
        if [ -n "${gate_verdict}" ]; then
            printf '  gate_verdict: %s\n' "${gate_verdict}"
        fi
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
    } > "${out_path}.tmp"
    # §5.6 atomic write: same-filesystem rename(2) replaces the target in one
    # step, so a concurrent reader never observes a half-written log.
    mv -f "${out_path}.tmp" "${out_path}"
}

# Stand-alone entry point for ad-hoc emission
if [ "${0##*/}" = "_dry_run_log_helper.sh" ] && [ $# -gt 0 ]; then
    emit_dry_run_log "$@"
fi
