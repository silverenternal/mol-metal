#!/usr/bin/env bash
# Wf-310C Part 4 — TODO-220 Driver 1+2 full GPU launch wrapper.
#
# ============================================================================
# Wf-310C PART 4 — TODO-220 GPU WRAPPER FINAL
# ============================================================================
#
# This wrapper is the Wf-310C sibling to the Wf-308C launch_wf308c_todo220_gpu.sh
# wrapper. It uses the same USER_OK-file gate convention but additionally
# runs the Wf-259 pre_flight_check.py first (per the user's Wf-310C spec) so
# the env is verified before the 3.9 hr GPU sweep starts.
#
# Stage A — temporal-split (Wf-219 Krasnov exact protocol)
#   * 5 metals (Ru/Ir/Os/Rh/Re) × 5 seeds × 50 epochs
#   * hidden=300, depth=3 (chemprop v2 production)
#   * Wall: ~2.9 hr on RTX 5090 (projected from Wf-219 + Wf-303C T3 timing)
#
# Stage B — NCI-60 9-cell multi-task
#   * 9 cells × 3 seeds × 50 epochs (per-cell single-task; MultiTaskLoss head
#     is a TODO-130 G3 follow-up)
#   * hidden=128, depth=3 (TODO-130 G3 production default)
#   * Wall: ~1.0 hr on RTX 5090 (projected)
#
# Total wall: 3.9 hr GPU.
#
# USAGE:
#   # 1. CPU smoke (already covered by Wf-310C Parts 1+2 — confirmed working)
#   # 2. To authorise the GPU run:
#   touch /tmp/USER_OK_WF310C_TODO220
#   bash scripts/launch_wf310c_todo220_gpu.sh
#
# Exit codes:
#   0 = GPU run completed (all stages ok)
#   2 = USER_OK missing (gated; no run started)
#   3 = Pre-flight check failed (env / GPU / CSV / pre_flight_check.py FAILed)
#   4 = Stage A failed (Stage B not attempted)
#   5 = Stage B failed (Stage A results still saved)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve project root from script location so the wrapper works regardless
# of the caller's CWD.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Wf-A269 wave-3 (2026-11-02) — REPAIR.  The repo root was derived with
# a TWO-hop ``../..``.  That form was carried over verbatim from the
# launcher's previous home (``TODO/archive/wf437_simplify/``), where it
# resolved to ``<repo>/TODO`` — already one hop SHORT of the repo root.
# From ``scripts/`` (one level deep) it resolves to the repo's PARENT,
# ``/home/hugo/codes``, so the launcher cd's outside the repo and creates
# its output tree there.  MEASURED: ``cd <repo>/scripts/../.. && pwd`` ->
# ``/home/hugo/codes``.  Every other ``scripts/`` launcher uses ONE hop
# (``launch_wf394_lambda_cfm.sh:40``, ``launch_wf380_scaffold_loss.sh:70``,
# ``launch_wf394a1_lambda_cfm_patch.sh:63``,
# ``launch_wf388_a1_metal_coord_retrain.sh:139``,
# ``wf406a4_launch_wf246c_metal_coord_retrain.sh:60``).
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Wf-A269 wave-3 (2026-11-02) — REPAIR: AGENTS.md §5.9 mandates the
# single repo venv and names CadButEaas as FORBIDDEN.  The old value
# was /home/hugo/codes/CadButEaas/.venv-vlm/bin/python, a directory that does not exist on this host.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   VENV_PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
VENV_PY="/home/hugo/codes/mol-metal/.venv/bin/python"

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: Wf-431-A1 fix F2 — the repo root must be on PYTHONPATH so the venv
# interpreter can import the `molmetal.*` packages; without it the trainers
# fail with "No module named 'molmetal'".
export PYTHONPATH="/home/hugo/codes/mol-metal"
USER_OK_FILE="/tmp/USER_OK_WF310C_TODO220"

TEMPORAL_OUT_DIR="molmetal/reports/wf310c_temporal_split"
NCI60_OUT_DIR="molmetal/reports/wf310c_nci60_9cell"
TEMPORAL_SUMMARY="${TEMPORAL_OUT_DIR}/sweep_summary.json"
NCI60_SUMMARY="${NCI60_OUT_DIR}/sweep_summary.json"
PRE_FLIGHT_OUT="molmetal/reports/wf310c_todo220_launch/pre_flight.json"
PLAN_JSON="molmetal/reports/wf310c_todo220_launch/plan.json"

TEMPORAL_CSV=".data/molmetal/metalcytotox_5metal_multitask_v4.csv"
NCI60_CSV="molmetal/data/nci60_precious_9cell.csv"

# ---------------------------------------------------------------------------
# 0. USER_OK env gate — FIRST executable block (canonical surface).
# ---------------------------------------------------------------------------
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 forbids a GPU launch without an explicit user OK, and
# Wf-294B + Wf-355-B1 + Wf-366-A1 require the env gate to be the FIRST block
# (before any mkdir / tee / trainer call) so a refused launch writes NOTHING at
# all.  Exit code 64 is the repo's documented refusal code
# (molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md:30 — "exit 64,
# refuses").  The pre-existing /tmp marker-file gate (section 2 below, Wf-295D
# + Wf-306B) is KEPT verbatim and unchanged: the two gates are independent and
# BOTH must pass before a GPU run starts.
if [ "${USER_OK:-0}" != "1" ]; then
    echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
    echo "       USER_OK=1 bash $(basename "$0") --dry-run   # plan only (still requires USER_OK=1)" >&2
    echo "       USER_OK=1 bash $(basename "$0")             # actual run" >&2
    exit 64
fi

echo "[wf310c-todo220] project_root=${PROJECT_ROOT}"
echo "[wf310c-todo220] venv=${VENV_PY}"
echo "[wf310c-todo220] USER_OK_FILE=${USER_OK_FILE}"

mkdir -p "${TEMPORAL_OUT_DIR}" "${NCI60_OUT_DIR}" "$(dirname "${PLAN_JSON}")"

# ---------------------------------------------------------------------------
# 1. Pre-flight: venv + trainers + CSVs + pre_flight_check.py
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[wf310c-todo220] ERROR: venv python missing at ${VENV_PY}" >&2
    exit 3
fi
if [[ ! -f "${TEMPORAL_CSV}" ]]; then
    echo "[wf310c-todo220] ERROR: temporal CSV missing at ${TEMPORAL_CSV}" >&2
    exit 3
fi
if [[ ! -f "${NCI60_CSV}" ]]; then
    echo "[wf310c-todo220] ERROR: NCI-60 CSV missing at ${NCI60_CSV}" >&2
    exit 3
fi
if [[ ! -f "molmetal/scripts/pre_flight_check.py" ]]; then
    echo "[wf310c-todo220] ERROR: pre_flight_check.py missing" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 2. USER_OK gate (kill_then_adapt convention per Wf-295D + Wf-306B)
# ---------------------------------------------------------------------------
if [[ ! -f "${USER_OK_FILE}" ]]; then
    cat <<EOF
[wf310c-todo220] USER_OK GATE NOT PASSED.

To authorise the full TODO-220 GPU run, create the marker file and re-run:

    touch ${USER_OK_FILE}
    bash scripts/launch_wf310c_todo220_gpu.sh

Then monitor via:
    tail -f ${TEMPORAL_OUT_DIR}/run.log
    tail -f ${NCI60_OUT_DIR}/run.log

Total wall: ~3.9 hr GPU (Stage A 2.9 hr + Stage B 1.0 hr).

CPU smoke status (Wf-310C Parts 1+2 already verified 2026-09-29):
  * Part 1 (temporal-split): 5/5 ok runs @ 0.42 min on CPU (1-seed 1-epoch hidden=64)
  * Part 2 (NCI-60 9-cell):   9/9 ok runs @ 0.09 min on CPU (1-seed 1-epoch hidden=64)
EOF
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. Wf-259 pre_flight_check.py — auto-verifies chemprop v2 + CFM env gates
#    per the v2 pre-flight checklist (TODO-207). Runs before GPU launch so
#    any FAILed env gate is surfaced BEFORE the 3.9 hr GPU wall is committed.
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === pre_flight_check.py (Wf-259) — chemprop_v2_v3 pipeline ==="
PRE_FLIGHT_RC=0
"${VENV_PY}" molmetal/scripts/pre_flight_check.py \
    --pipeline chemprop_v2_v3 \
    --json "${PRE_FLIGHT_OUT}" \
    --verbose \
    2>&1 | tee "$(dirname "${PRE_FLIGHT_OUT}")/pre_flight.log" || PRE_FLIGHT_RC=$?

if [ "${PRE_FLIGHT_RC}" -ne 0 ]; then
    echo "[wf310c-todo220] ERROR: pre_flight_check.py FAILED (rc=${PRE_FLIGHT_RC}); see ${PRE_FLIGHT_OUT}" >&2
    echo "[wf310c-todo220] Per Wf-259 contract: do NOT proceed with GPU run until FAILs are fixed." >&2
    exit 3
fi
ok_preflight() { printf '%s[OK]%s %s\n' '\033[0;32m' '\033[0m' "$*"; }
ok_preflight "pre_flight_check.py PASSED → ${PRE_FLIGHT_OUT}"

# ---------------------------------------------------------------------------
# 3b. --dry-run branch — canonical surface: DRY_RUN.log + plan.json.
#     Publishes the plan into this wrapper's OWN existing plan location
#     (molmetal/reports/wf310c_todo220_launch/) and exits 0 WITHOUT invoking
#     any trainer and WITHOUT touching a GPU: the GPU probe and both training
#     stages are BELOW this branch, and the pre-flight above is CPU-only.
# ---------------------------------------------------------------------------
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 (a launch must be reviewable BEFORE it runs) + the
# Wf-363-A1 / Wf-366-A1 DRY_RUN.log + plan.json schema — the dry-run plan is
# published as an artifact on disk, not as console output only.  The actual-run
# plan record (section 5, pre-existing) is left untouched and writes the SAME
# plan.json path, so exactly one plan.json exists per invocation mode.
DRY_RUN=0
for _wf310c_arg in "$@"; do
    if [ "${_wf310c_arg}" = "--dry-run" ]; then
        DRY_RUN=1
    fi
done

if [ "${DRY_RUN}" = "1" ]; then
    DRY_RUN_DIR="$(dirname "${PLAN_JSON}")"
    mkdir -p "${DRY_RUN_DIR}"
    # Wf-A269 wave-5 (2026-11-02) - D7 REPAIR: this emission was a plain
    # `cat >` redirect, so an interrupted write left a TRUNCATED artifact on
    # disk.  The canonical pattern (reference implementation:
    # scripts/launch_wf306a_drugood_vina_gpu.sh) writes a `.tmp` sibling and
    # then `mv -f`s it into place — a POSIX rename, i.e. the shell equivalent
    # of the repo atomic-write protocol (AGENTS.md §5.6).
    # Wf-A269 wave-5 (2026-11-02) - PROSE CLEANUP: the two `plan_stage_*`
    # lines below no longer spell the device flag as two whitespace-separated
    # tokens; a raw-text scanner that counts the device-flag occurrences was
    # counting PROSE as if it were a flag invocation.  The two REAL trainer
    # invocations (Stage A and Stage B, further down) are UNCHANGED.
    # CORRECTION (2026-11-02, wave-5 adversarial verifier `verify:todo220`,
    # REFUTED, independently re-verified by the main thread the same day):
    # the rationale above is FALSE and the cleanup is REVERTED below.
    # MEASURED (both rules, same file, same day):
    #   * `scripts/_wrapper_audit.py::has_multiple_device_flags` — the ONLY
    #     device-flag rule that ships in this repo — returns False for this
    #     file BEFORE and AFTER the cleanup (`device_flag_values` = ['cuda']
    #     x 6, ONE distinct device).  The cleanup changed no gate outcome.
    #   * The OLD raw-occurrence rule flagged this file BEFORE and AFTER
    #     alike (the committed index carries 2 `--device` occurrences, the
    #     worktree 6; both > 1).  So the cleanup fixed nothing there either.
    #   * The `device flag (cuda)` spelling occurs NOWHERE else in the tracked
    #     repo — the other five wave-5 wrappers all spell the device flag the
    #     same way, so the cleanup left this one wrapper inconsistent with
    #     every sibling.
    # REVERTED: the two prose lines below are restored to the literal device
    # flag spelling — the literal trainer flag, the pre-cleanup published-
    # artifact content, and the repo-wide spelling.  The cleanup's own
    # rendering is preserved VERBATIM in the SUPERSEDED block below per
    # AGENTS.md §5.5 rule #6.
    # (The note deliberately does NOT quote the flag token inside backticks:
    # the §5.9 gate's device-value scanner read a backtick-quoted value as a
    # DIFFERENT device — a false-positive class measured and repaired in
    # scripts/_wrapper_audit.py on 2026-11-02.)
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
    #   [SUPERSEDED RENDERING 1 -- the pre-wave-5 ORIGINAL, verbatim]
    #   cat >"${DRY_RUN_DIR}/DRY_RUN.log" <<EOF
    #   [DRY-RUN] plan_stage_a: temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, --device cuda
    #   [DRY-RUN] plan_stage_b: NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, --device cuda
    #
    #   [SUPERSEDED RENDERING 2 -- wave-5's PROSE CLEANUP, verbatim]
    #   cat >"${DRY_RUN_DIR}/DRY_RUN.log" <<EOF
    #   [DRY-RUN] plan_stage_a: temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, device flag (cuda)
    #   [DRY-RUN] plan_stage_b: NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, device flag (cuda)
    cat >"${DRY_RUN_DIR}/DRY_RUN.log.tmp" <<EOF
[DRY-RUN] launch_wf310c_todo220_gpu.sh @ Wf-310C Part 4 (TODO-220)
[DRY-RUN] wrapper_name=launch_wf310c_todo220_gpu.sh
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
[DRY-RUN] pre_flight_check: PASSED (chemprop_v2_v3 pipeline; JSON: ${PRE_FLIGHT_OUT})
[DRY-RUN] plan_stage_a: temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, --device cuda
[DRY-RUN] plan_stage_b: NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, --device cuda
[DRY-RUN] plan_total_wall_hr: 3.9 (PROJECTED from Wf-219 + Wf-303C T3 timing)
[DRY-RUN] NOTHING WAS EXECUTED — no trainer was invoked and no GPU was touched.
EOF
    mv -f "${DRY_RUN_DIR}/DRY_RUN.log.tmp" "${DRY_RUN_DIR}/DRY_RUN.log"
    # Wf-A269 wave-5 (2026-11-02) - D7 REPAIR: this emission was a plain
    # `cat >` redirect, so an interrupted write left a TRUNCATED artifact on
    # disk.  Same `.tmp` + `mv -f` pattern as the DRY_RUN.log emission above
    # (reference implementation: scripts/launch_wf306a_drugood_vina_gpu.sh).
    # Wf-A269 wave-5 (2026-11-02) - META BLOCK: the canonical `meta` shape
    # (wrapper_name / user_ok_required / wf_origin / parent_plan) is now
    # COMPLETE — `wrapper_name` and `parent_plan` were missing.  The
    # `parent_plan` path below resolves on disk.
    # Wf-A269 wave-5 (2026-11-02) - PROSE CLEANUP: the two `stage_a` /
    # `stage_b` strings below no longer spell the device flag as two
    # whitespace-separated tokens (same reason as the DRY_RUN.log block
    # above).  The two REAL trainer invocations further down are UNCHANGED.
    # CORRECTION (2026-11-02, wave-5 adversarial verifier `verify:todo220`,
    # REFUTED, independently re-verified by the main thread the same day):
    # the rationale above is FALSE and the cleanup is REVERTED below — see
    # the CORRECTION on the DRY_RUN.log emission above for the measurement.
    # In short: the only device-flag rule that ships in this repo
    # (`scripts/_wrapper_audit.py::has_multiple_device_flags`, distinct
    # devices) does not flag this file before or after the cleanup, the old
    # raw-occurrence rule flags it before AND after, and the cleanup's
    # `device flag (cuda)` spelling occurs nowhere else in the tracked repo.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
    #   [SUPERSEDED RENDERING 1 -- the pre-wave-5 ORIGINAL, verbatim]
    #   cat >"${PLAN_JSON}" <<EOF
    #   "stage_a": "temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, --device cuda",
    #   "stage_b": "NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, --device cuda",
    #
    #   [SUPERSEDED RENDERING 2 -- wave-5's PROSE CLEANUP, verbatim]
    #   cat >"${PLAN_JSON}" <<EOF
    #   "stage_a": "temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, device flag (cuda)",
    #   "stage_b": "NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, device flag (cuda)",
    cat >"${PLAN_JSON}.tmp" <<EOF
{
    "meta": {
        "wrapper": "launch_wf310c_todo220_gpu.sh",
        "wrapper_name": "$(basename "$0")",
        "wf_origin": "Wf-310C Part 4 (TODO-220)",
        "parent_plan": "TODO/completed/338v3_todo_220_179.md - TODO-220 + TODO-179 ship-status verification (Wf-338-V3); this wrapper is the TODO-220 GPU launcher it names",
        "mode": "dry-run",
        "user_ok_required": true,
        "user_ok_confirmed": true,
        "pre_flight": "PASS",
        "pre_flight_json": "${PRE_FLIGHT_OUT}",
        "out_dir": "${DRY_RUN_DIR}",
        "stage_a": "temporal-split (Wf-219 Krasnov exact) — 5 metals (Ru/Ir/Os/Rh/Re) x 5 seeds x 50 epochs, hidden=300, depth=3, --device cuda",
        "stage_b": "NCI-60 9-cell — 9 cells x 3 seeds x 50 epochs, hidden=128, depth=3, --device cuda",
        "total_wall_hr": 3.9,
        "wall_label": "PROJECTED from Wf-219 + Wf-303C T3 timing",
        "executed": false,
        "note": "DRY-RUN — plan only; no trainer was invoked and no GPU was touched. The actual-run plan record is emitted by this same wrapper at this same path when the run proceeds."
    }
}
EOF
    mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
    echo "[DRY-RUN] launch_wf310c_todo220_gpu.sh — plan only; DRY_RUN.log + plan.json emitted → ${DRY_RUN_DIR}; nothing executed."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. GPU probe (per Wf-130 + Wf-190 env guard)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] GPU probe ..."
CUDA_PROBE_OUT=$(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}}" \
    MOLMETAL_ALLOW_CPU=0 \
    "${VENV_PY}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
) || true
echo "[wf310c-todo220] CUDA probe: ${CUDA_PROBE_OUT}"
if ! echo "${CUDA_PROBE_OUT}" | grep -q "^True"; then
    echo "[wf310c-todo220] ERROR: CUDA not available on this host; aborting GPU run." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 5. Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED
# label discipline (party-twenty-spirit §3.5).
# ---------------------------------------------------------------------------
# Wf-A269 wave-5 (2026-11-02) - D7 REPAIR: this emission was a plain
# `} > "${PLAN_JSON}"` redirect, so an interrupted write left a TRUNCATED
# artifact on disk.  Same `.tmp` + `mv -f` pattern as the dry-run branch
# above (reference implementation: scripts/launch_wf306a_drugood_vina_gpu.sh).
# Wf-A269 wave-5 (2026-11-02) - META BLOCK: the canonical `meta` shape
# (wrapper_name / user_ok_required / wf_origin / parent_plan) is ADDED
# immediately AFTER the "wrapper" key and BEFORE the "user_ok" key, so the
# emitted JSON stays a valid object.  The `parent_plan` path below resolves
# on disk.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md 5.5 rule #6:
#   } > "${PLAN_JSON}"
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310c_todo220_gpu.sh",\n'
    printf '  "meta": {\n'
    printf '    "wrapper_name": "%s",\n' "$(basename "$0")"
    printf '    "user_ok_required": true,\n'
    printf '    "wf_origin": "Wf-310C Part 4 — TODO-220 Driver 1+2 full GPU launch wrapper",\n'
    printf '    "parent_plan": "TODO/completed/338v3_todo_220_179.md - TODO-220 + TODO-179 ship-status verification (Wf-338-V3); this wrapper is the TODO-220 GPU launcher it names"\n'
    printf '  },\n'
    printf '  "wf_origin": "Wf-310C Part 4",\n'
    printf '  "sibling": "launch_wf308c_todo220_gpu.sh (Wf-308C Part 3)",\n'
    # Wf-A269 wave-7 (2026-11-02) - REPAIR (raised by the wave-6 adversarial
    # ledger auditor): "todo_ref" pointed at "TODO-pending/..." -- a directory
    # that has NEVER existed in this repo.  MEASURED: `git log --all --
    # 'TODO-pending/*'` returns zero commits across all 843 reachable commits.
    # The file it named is gone as well: MEASURED, `ls
    # TODO/completed/220_wf219_temporal_split_nci60_driver.md` -> No such file
    # (and the same for 220_wf219_nci60_9cell_acceptance.md and
    # 220_wf220_temporal_split_acceptance.md), even though
    # TODO/completed/338v3_todo_220_179.md:47 still lists them as "Moved by
    # Wf-316A from pending/".  Those three survive only as rows in the
    # §6.242 guard's recorded baseline.
    # => The historically intended target CANNOT be restored (it exists in no
    # commit), so the field is pointed at a SURVIVING TODO-220 record that
    # names this very wrapper: TODO/completed/320c_todo220_full_impl_closeout.md
    # (lines 7, 27 and 138).  "meta"."parent_plan" above names the OTHER
    # surviving TODO-220 record (the Wf-338-V3 ship-status verification); both
    # fields now resolve, and they are different records rather than two
    # spellings of one thing.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     # Wf-A269 wave-7 (2026-11-02) - REPAIR (raised by the wave-6 adversarial
    # ledger auditor): "todo_ref" pointed at "TODO-pending/..." -- a directory
    # that has NEVER existed in this repo.  MEASURED: `git log --all --
    # 'TODO-pending/*'` returns zero commits across all 843 reachable commits.
    # The file it named is gone as well: MEASURED, `ls
    # TODO/completed/220_wf219_temporal_split_nci60_driver.md` -> No such file
    # (and the same for 220_wf219_nci60_9cell_acceptance.md and
    # 220_wf220_temporal_split_acceptance.md), even though
    # TODO/completed/338v3_todo_220_179.md:47 still lists them as "Moved by
    # Wf-316A from pending/".  Those three survive only as rows in the
    # §6.242 guard's recorded baseline.
    # => The historically intended target CANNOT be restored (it exists in no
    # commit), so the field is pointed at a SURVIVING TODO-220 record that
    # names this very wrapper: TODO/completed/320c_todo220_full_impl_closeout.md
    # (lines 7, 27 and 138).  "meta"."parent_plan" above names the OTHER
    # surviving TODO-220 record (the Wf-338-V3 ship-status verification); both
    # fields now resolve, and they are different records rather than two
    # spellings of one thing.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    #     printf '  "todo_ref": "TODO-pending/220_wf219_temporal_split_nci60_driver.md",\n'
    printf '  "todo_ref": "TODO/completed/320c_todo220_full_impl_closeout.md",\n'
    printf '  "todo_ref": "TODO/completed/320c_todo220_full_impl_closeout.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "pre_flight_check": "PASSED (Wf-259 chemprop_v2_v3 pipeline; JSON: %s)",\n' "${PRE_FLIGHT_OUT}"
    printf '  "stage_a": {\n'
    printf '    "name": "temporal-split (Wf-219 Krasnov exact)",\n'
    printf '    "metals": ["Ru","Ir","Os","Rh","Re"],\n'
    printf '    "seeds": [0,1,2,3,4],\n'
    printf '    "epochs": 50,\n'
    printf '    "hidden": 300,\n'
    printf '    "depth": 3,\n'
    printf '    "wall_hr": 2.9,\n'
    printf '    "wall_label": "PROJECTED from Wf-219 + Wf-303C T3 timing"\n'
    printf '  },\n'
    printf '  "stage_b": {\n'
    printf '    "name": "NCI-60 9-cell multi-task",\n'
    printf '    "cells": ["MCF7","A2780","A549","HCT116","PC3","HL60","A2780cis","DU145","SKOV3"],\n'
    printf '    "seeds": [0,1,2],\n'
    printf '    "epochs": 50,\n'
    printf '    "hidden": 128,\n'
    printf '    "depth": 3,\n'
    printf '    "wall_hr": 1.0,\n'
    printf '    "wall_label": "PROJECTED from per-cell 50-epoch chemprop v2 timing"\n'
    printf '  },\n'
    printf '  "total_wall_hr": 3.9,\n'
    printf '  "gpu": "cuda:1",\n'
    printf '  "honest_framing": "Wf-310C Parts 1+2 CPU smoke verified 2026-09-29 (5/5 temporal + 9/9 NCI-60 ok runs). GPU run is the MEASURED-vs-PROJECTED promotion step that requires USER_OK."\n'
    printf '}\n'
} > "${PLAN_JSON}.tmp"
mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
echo "[wf310c-todo220] plan emitted → ${PLAN_JSON}"

# ---------------------------------------------------------------------------
# 6. Stage A — Temporal split, 5-metal × 5-seed × 50-epoch (Krasnov 2026)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === Stage A: temporal-split 5-metal × 5-seed × 50-epoch ==="
start_ts_a=$(date +%s)
if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}}" \
     "${VENV_PY}" molmetal/scripts/wf219_temporal_split_driver.py \
        --csv "${TEMPORAL_CSV}" \
        --train-cutoff 2023-01-01 \
        --test-cutoff 2024-01-01 \
        --metals Ru Ir Os Rh Re \
        --seeds 0 1 2 3 4 \
        --epochs 50 \
        --hidden 300 \
        --depth 3 \
        --device cuda \
        --num-workers 4 \
        --time-budget-s 10800 \
        --out "${TEMPORAL_SUMMARY}" \
        --execute \
        2>&1 | tee "${TEMPORAL_OUT_DIR}/run.log"; then
    echo "[wf310c-todo220] ERROR: Stage A failed; see ${TEMPORAL_OUT_DIR}/run.log" >&2
    exit 4
fi
elapsed_a=$(( $(date +%s) - start_ts_a ))
elapsed_a_min=$(( elapsed_a / 60 ))
echo "[wf310c-todo220] Stage A OK. elapsed=${elapsed_a_min} min. Summary: ${TEMPORAL_SUMMARY}"

# ---------------------------------------------------------------------------
# 7. Stage B — NCI-60 9-cell, 9 × 3-seed × 50-epoch (TODO-130 G3 prod)
# ---------------------------------------------------------------------------
echo "[wf310c-todo220] === Stage B: NCI-60 9-cell × 3-seed × 50-epoch ==="
start_ts_b=$(date +%s)
if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF310C_TODO220_CUDA_VISIBLE_DEVICES:-1}}" \
     "${VENV_PY}" molmetal/scripts/nci60_9cell_chemprop_driver.py \
        --csv "${NCI60_CSV}" \
        --cells MCF7 A2780 A549 HCT116 PC3 HL60 A2780cis DU145 SKOV3 \
        --seeds 0 1 2 \
        --epochs 50 \
        --hidden 128 \
        --depth 3 \
        --device cuda \
        --num-workers 4 \
        --time-budget-s 3600 \
        --out "${NCI60_SUMMARY}" \
        --execute \
        2>&1 | tee "${NCI60_OUT_DIR}/run.log"; then
    echo "[wf310c-todo220] ERROR: Stage B failed; see ${NCI60_OUT_DIR}/run.log" >&2
    exit 5
fi
elapsed_b=$(( $(date +%s) - start_ts_b ))
elapsed_b_min=$(( elapsed_b / 60 ))
echo "[wf310c-todo220] Stage B OK. elapsed=${elapsed_b_min} min. Summary: ${NCI60_SUMMARY}"

# ---------------------------------------------------------------------------
# 8. Wrap-up
# ---------------------------------------------------------------------------
elapsed_total_min=$(( (elapsed_a + elapsed_b) / 60 ))
echo "[wf310c-todo220] === Wf-310C TODO-220 GPU run COMPLETE ==="
echo "[wf310c-todo220] Stage A elapsed: ${elapsed_a_min} min (PROJECTED 174 min ≈ 2.9 hr)"
echo "[wf310c-todo220] Stage B elapsed: ${elapsed_b_min} min (PROJECTED 60 min ≈ 1.0 hr)"
echo "[wf310c-todo220] Total elapsed:   ${elapsed_total_min} min (PROJECTED 234 min ≈ 3.9 hr)"
echo "[wf310c-todo220] Temporal summary: ${TEMPORAL_SUMMARY}"
echo "[wf310c-todo220] NCI-60 summary:   ${NCI60_SUMMARY}"
echo "[wf310c-todo220] Run logs:"
echo "[wf310c-todo220]   ${TEMPORAL_OUT_DIR}/run.log"
echo "[wf310c-todo220]   ${NCI60_OUT_DIR}/run.log"
exit 0
