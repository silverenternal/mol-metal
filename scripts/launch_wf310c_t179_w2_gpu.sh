#!/usr/bin/env bash
# Wf-310C Part 3 — TODO-179 W₂ pIC50 alignment GPU retrain wrapper.
#
# ============================================================================
# Wf-310C PART 3 — TODO-179 W₂ ALIGNMENT ON PLATINAI 4-CELL PROBE
# ============================================================================
#
# This wrapper gates the W₂ pIC50 alignment retrain behind a USER_OK marker
# file so the expensive 5-seed × 50-epoch Hilbert+ASTSR+W₂ joint sweep on the
# PlatinAI 4-cell probe (4078 Pt complexes × 4 cell lines = ~16k rows) does
# NOT start without an explicit user opt-in (per kill_then_adapt convention,
# Wf-295D + Wf-306B).
#
# Trainer
# -------
#   molmetal/scripts/train_hilbert_astsr_end_to_end.py
#   --wass-weight 0.05 --ref-distribution platinai_4cell  (TODO-179 §3.1)
#
# Wf-A269 (2026-11-03) — AGENTS.md §5.11 YAML-only launch.  The two lines
# above are now the CONTENTS of a YAML config, not command-line flags: the
# trainer is invoked with `--config` ALONE, once per sweep iteration, with
# configs/wfa269_c8_hilbert_astsr_s{0..4}.yaml.  The config is the run's
# definition; this launcher keeps only the plumbing (env, gates, paths,
# logging, budget).  Equivalence of the two forms is verified dest-by-dest
# by scripts/_yaml_equivalence_probe.py — receipt:
# molmetal/reports/wfa269_yaml_migration/launch_wf310c_t179_w2_gpu.json.
#
# Scope (5 seeds × 50 epochs × PlatinAI 4-cell probe)
# ----------------------------------------------------
#   seed 0  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 1  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 2  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 3  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   seed 4  50 epochs Hilbert+ASTSR+W₂   ~50 min wall (PROJECTED)
#   total wall budget: ~4.5 hr on RTX 5090 cuda:1 (PROJECTED; 5 × 50 min
#   with 10% slack = 275 min ≈ 4.6 hr; budgeted 4.5 hr to fit a single GPU-1
#   reservation slot).
#
# Per-cell breakdown
# ------------------
# A2780 / A2780cis / MCF7 / A549 are the 4 cells in PlatinAI. Per-cell R² is
# the primary honest metric (no aggregation hiding per-cell noise, per
# party-twenty-spirit §3.5).
#
# Honest framing (per Wf-127 §Cause 2 + TODO-179 §6)
# --------------------------------------------------
# * MEASURED baseline: L5 R² = +0.163 (Wf-37 P1 bug fix; NOT algorithmic lift).
# * PROJECTED ceiling: 0.40-0.60 from W₂ alone (12-d RDKit signal floor).
# * NOT MEASURED: 0.85+ — that would require F5_RDKIT_2D 32-d expansion
#   which is a separate follow-up (TODO-179 §8).
#
# USAGE:
#   # 1. CPU smoke (already covered by Wf-246-A; this wrapper skips smoke).
#   # 2. Plan only — no trainer, no GPU (REQUIRED: USER_OK=1 in the env):
#   USER_OK=1 bash scripts/launch_wf310c_t179_w2_gpu.sh --dry-run
#   # 3. To authorise the GPU run — needs BOTH gates (REQUIRED: USER_OK=1 in
#   #    the env AND the marker file below):
#   touch /tmp/USER_OK_WF310C_T179
#   USER_OK=1 bash scripts/launch_wf310c_t179_w2_gpu.sh
#   # 4. To explicitly skip GPU: do nothing.  The launcher is fail-CLOSED —
#   #    without USER_OK=1 in the environment the FIRST executable block
#   #    refuses with exit 64 and NOTHING is executed (no trainer, no GPU
#   #    probe, no output written).
#
# Wf-A269 wave-5 (2026-11-02) - REPAIR D4 [MED].  The USAGE block above used
# to document an invocation this launcher REFUSES: the canonical USER_OK env
# gate is the FIRST executable block and exits 64 unless USER_OK=1 is in the
# environment, so the documented `bash scripts/...` (no env var) could never
# run.  The marker-file step is KEPT because that gate is still live and is
# still required for the ACTUAL run (it is a SECOND, independent gate, and it
# is reached only after the env gate and only on the non-dry-run path).
# MEASURED (this wave, re-measured 2026-11-02 by the wave-5 adversarial
# verifier): `git grep -a -n "WF310C_RUN_GPU"` over the tracked tree returns
# 2 lines and NO code site -- this comment and the SUPERSEDED rendering
# preserved below.  The env var the old step-3 instruction told the reader to
# unset is read nowhere in the repo, so that instruction was fiction; the
# truthful skip-GPU instruction is "do nothing" (the launcher is fail-CLOSED).
# CORRECTION (2026-11-02): the wave-5 text first shipped claiming the grep
# "returns THIS COMMENT LINE and nothing else".  That claim was FALSE -- it
# returns 2 lines, because the SUPERSEDED rendering names the same env var.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   # MEASURED (this wave): `git grep -a -n "WF310C_RUN_GPU"` over the tracked
#   # tree returns THIS COMMENT LINE and nothing else — the env var the old
#   # step-3 instruction told the reader to unset is read nowhere in the repo, so
#   # that instruction was fiction; the truthful skip-GPU instruction is
#   # "do nothing" (the launcher is fail-CLOSED).
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   [SUPERSEDED RENDERING 1 -- the pre-wave-3 ORIGINAL, verbatim]
#   # USAGE:
#   #   # 1. CPU smoke (already covered by Wf-246-A; this wrapper skips smoke).
#   #   # 2. To authorise the GPU run:
#   #   touch /tmp/USER_OK_WF310C_T179
#   #   bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh
#   #   # 3. To explicitly skip GPU (default behaviour):
#   #   unset WF310C_RUN_GPU   # (the default is "no" without USER_OK file)
#
#   [SUPERSEDED RENDERING 2 -- wave-3's, verbatim]
#   # USAGE:
#   #   # 1. CPU smoke (already covered by Wf-246-A; this wrapper skips smoke).
#   #   # 2. To authorise the GPU run:
#   #   touch /tmp/USER_OK_WF310C_T179
#   #   bash scripts/launch_wf310c_t179_w2_gpu.sh
#   #   # 3. To explicitly skip GPU (default behaviour):
#   #   unset WF310C_RUN_GPU   # (the default is "no" without USER_OK file)
#
# NOTE (2026-11-02, wave-5 adversarial verifier): rendering 1 names
# `molmetal/scripts/launch_wf310c_t179_w2_gpu.sh`, a path this launcher NEVER
# occupied.  MEASURED: `git log --all -- 'molmetal/scripts/launch_wf310c_t179_w2_gpu.sh'`
# returns nothing, and the launcher's only committed location is the
# TODO/archive/wf437_simplify/ directory (added in bc29aa90).
# It is preserved because AGENTS.md §5.5 rule #6 requires the superseded
# rendering to be kept; it is a DEAD path BY CONSTRUCTION, and the
# dangling-launcher-path gate allow-lists it for exactly that reason.
#
# Exit codes:
#   0 = GPU run completed (all seeds ok)
#   2 = USER_OK missing (gated; no run started)
#   3 = Setup error (venv / GPU probe / CSV missing)
#   4 = GPU run failed (partial results still in run.log)

#
# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: the canonical USER_OK env gate (Wf-294B family) refuses with exit
#         64, the repo's documented refusal code (Wf-306B wrapper_plan.md:30).
#         The exit-code table above predates it and is PRESERVED VERBATIM per
#         AGENTS.md §5.5 rule #6; the line below is ADDITIVE.
#   64 = USER_OK env gate refused (canonical Wf-294B gate; nothing executed)
#
# Also ADDITIVE (Wf-A269 wave-3): the canonical surface adds a USER_OK=1
# environment requirement and a --dry-run branch, so the USAGE block above now
# reads `USER_OK=1 bash scripts/launch_wf310c_t179_w2_gpu.sh`.
#
# Wf-A269 wave-5 (2026-11-02) - D4 REPAIR.  The wave-3 sentence above became
# TRUE only in this wave: at wave-3 the USAGE block still read
# `bash scripts/launch_wf310c_t179_w2_gpu.sh` with NO env var, so the claim
# was FALSE as written.  Wave-5 rewrote the USAGE block (additive — the old
# rendering is preserved VERBATIM above).  The wave-3 text above is left
# UNCHANGED; it is now an accurate description of the file.

set -euo pipefail

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + Wf-294B / Wf-355-B1 / Wf-366-A1 require the USER_OK
#         gate to be the FIRST executable block, so no GPU job can start
#         without an explicit user opt-in.  Exit 64 is the repo's documented
#         refusal code (molmetal/reports/wf306b_wf219_ru_wrapper/wrapper_plan.md:30).
if [ "${USER_OK:-0}" != "1" ]; then
    echo "[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch." >&2
    echo "       USER_OK=1 bash $(basename "$0") --dry-run   # plan only (still requires USER_OK=1)" >&2
    echo "       USER_OK=1 bash $(basename "$0")             # actual run" >&2
    exit 64
fi

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
# Reason: Wf-431-A1 fix F2 — the repo root must be on PYTHONPATH for the
#         `molmetal.*` packages to import under the §5.9 single-venv
#         interpreter (a bare interpreter fails with "No module named
#         'molmetal'").
export PYTHONPATH="/home/hugo/codes/mol-metal"

USER_OK_FILE="/tmp/USER_OK_WF310C_T179"

TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_hilbert_astsr_end_to_end.py"
# ---------------------------------------------------------------------------
# §5.11 STATUS: MIGRATED — the ten former command-line flags
# (--dataset / --seeds / --epochs / --n-samples / --n-folds / --wass-weight /
# --ref-distribution / --per-cell-cap / --platinai-csv / --output) now live in
# ONE YAML config per sweep iteration
# (configs/wfa269_c8_hilbert_astsr_s{0..4}.yaml) and the trainer is invoked
# with `--config` ALONE (see the sweep loop below; `config_for_seed()`).
#
# CORRECTION (2026-11-03, later the same day) — ADDITIVE, per AGENTS.md §5.5
# rule #6. The block preserved VERBATIM below said "NOT MIGRATABLE — the
# trainer has no YAML path at all". That was TRUE when it was written and is
# FALSE on disk now: the trainer's `--config` dest + `apply_yaml_overrides`
# wire-in landed in the SAME wave
# (train_hilbert_astsr_end_to_end.py:1451-1457 declares `--config`;
#  :1571-1579 applies the YAML over the parse).
# MEASURED 2026-11-03 11:53 on the working tree:
#   grep -c apply_yaml_overrides molmetal/scripts/train_hilbert_astsr_end_to_end.py
#     -> 3 (not 0)
#   grep -n '"--config"' molmetal/scripts/train_hilbert_astsr_end_to_end.py
#     -> 1 hit at :1452 (not 0 hits)
#   `main()` is at :1433 and builds its parser at :1441, so
#   `scripts/_config_binding_gate.py::_capture_parser` NOW captures the
#   namespace and audits all five configs (MEASURED: 0 violations, 0 type
#   violations).
# The migration itself was performed by the concurrent Wf-A269 agent that
# owned the trainer edit; this correction only retires the stale statement.
# The `--seeds`-is-a-COUNT observation in the preserved block is STILL TRUE
# and is still load-bearing: it is why the faithful migration needs one
# STATIC config per sweep iteration (one value of an int dest) rather than a
# list.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   §5.11 STATUS: NOT MIGRATABLE — the trainer has no YAML path at all.
#   #
#   AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
#   训练和推理，no cli参数") allows a launcher to pass only --config / --out-dir /
#   --ckpt-dir / --help to a trainer. The ten flags below
#   (--dataset / --seeds / --epochs / --n-samples / --n-folds / --wass-weight /
#   --ref-distribution / --per-cell-cap / --platinai-csv / --output) are still on
#   the command line, and they have to stay there:
#   #
#   * `train_hilbert_astsr_end_to_end.py` calls `apply_yaml_overrides` ZERO
#   * times (MEASURED: `grep -c apply_yaml_overrides` = 0). It does not even
#   * import `yaml`. Every other trainer in this repo ends its own parse with
#   * `apply_yaml_overrides(parser, args, args.config)`
#   * (train_dmpnn_multitask_wasserstein.py:3965,
#   *  train_metal_lomo_wasserstein.py:2381, train_metal_lomo_v2.py:1927,
#   *  train_cfm_yuelbond_wasserstein.py:16046) — this one does not.
#   * It has NO `--config` argparse dest (`grep -n '"--config"'` = 0 hits), so
#   * there is no dest for a YAML path to arrive through.
#   * Its parser is built INSIDE `main()` (line 1408), not at module level, so
#   * `scripts/_config_binding_gate.py::_capture_parser` reports
#   * "could not capture a parser namespace" (MEASURED) and the §5.11
#   * config-binding gate cannot audit a config for it either.
#   #
#   Migration would therefore require EDITING THE TRAINER, which is outside the
#   scope of this repair. Per AGENTS.md §5.2 the flags are reported as
#   UNMIGRATED rather than faked: moving them into a YAML the trainer cannot
#   read would make the config decorative — the exact silent-lie class the
#   §5.11 gates exist to prevent.
#   #
#   Second, unrelated observation (NOT changed here, since changing it would
#   change the run): on THIS trainer `--seeds` is a COUNT, not a seed list —
#   `--seeds`, type=int, default=5, help="Number of seeds for the benchmark"
#   (train_hilbert_astsr_end_to_end.py:1425) and the sweep is
#   `for s in range(int(args.seeds))` (:1602). The wrapper loops SEEDS=(0 1 2 3 4)
#   and passes `--seeds "${SEED}"`, so its five invocations ask for 0, 1, 2, 3 and
#   4 seeds respectively — the seed-0 invocation sweeps NOTHING and the other
#   four re-run seeds 0..3. `--epochs` defaults to 100 here while the wrapper
#   passes ${EPOCHS}=50. Both are left exactly as they were: this repair is a
#   plumbing migration, and changing them would change the run.
# ---------------------------------------------------------------------------
PLATINAI_CSV="molmetal/data/platinai_4cell.csv"
OUT_DIR="molmetal/reports/wf310c_t179_w2_wrapper"
RUN_LOG="${OUT_DIR}/run.log"
SUMMARY_JSON="${OUT_DIR}/sweep_summary.json"
PLAN_JSON="${OUT_DIR}/plan.json"

# Wf-A269 wave-5 (2026-11-02) - REPAIR D2 [HIGH]: a real run used to
# OVERWRITE a GIT-TRACKED historical record.  The trainer's per-seed
# --output pointed INSIDE the report dir that ALSO holds the tracked record
# for this wave, and the trainer writes comparison_table.md + r2_bench.json +
# run.log into whatever --output names.
# MEASURED (this wave):
#   git ls-files molmetal/reports/wf310c_t179_w2_wrapper/
#     molmetal/reports/wf310c_t179_w2_wrapper/launch_plan.md
#     molmetal/reports/wf310c_t179_w2_wrapper/seed0/comparison_table.md
#   git grep -a -n "comparison_table" molmetal/scripts/train_hilbert_astsr_end_to_end.py
#     -> 1476: help="Output directory for r2_bench.json + comparison_table.md."
#     -> 1636: table_path = out_dir / "comparison_table.md"
# The per-seed output therefore goes to a FRESH, UNTRACKED root:
#   git ls-files molmetal/reports/wf310c_t179_w2_runs/   -> (prints nothing)
#   git status --porcelain molmetal/reports/wf310c_t179_w2_wrapper/
#                                                    -> (prints nothing)
# The TRACKED record under the report dir -- launch_plan.md and
# seed0/comparison_table.md -- is write-protected BY CONSTRUCTION: no trainer
# --output names it any more.
# CORRECTION (2026-11-02, wave-5 adversarial verifier): the wave-5 text first
# shipped as "nothing in this launcher writes into it any more".  That was
# FALSE -- this launcher still writes 5 artefacts into OUT_DIR
# (preflight.json, DRY_RUN.log, plan.json, run.log, sweep_summary.json).
# MEASURED: every one of them is gitignored (.gitignore:158 `**/*.json`,
# .gitignore:33 `**/*.log`), so the TRACKED record is still safe -- but the
# stated rationale was wrong.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   # The tracked record under the report dir is now write-protected BY
#   # CONSTRUCTION: nothing in this launcher writes into it any more.
RUN_ROOT="${PROJECT_ROOT}/molmetal/reports/wf310c_t179_w2_runs"

# ---------------------------------------------------------------------------
# Wf-A269 (2026-11-03) — AGENTS.md §5.11 YAML-only launch config
# ---------------------------------------------------------------------------
# AGENTS.md §5.11 (user directive 2026-10-09): "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数" — a launch wrapper may hand the trainer only
# --config / --out-dir / --ckpt-dir / --help.  Every hyperparameter this
# launcher used to pass on the command line now lives in a YAML config under
# configs/, and the trainer receives `--config` ALONE.
#
# ONE CONFIG PER SWEEP ITERATION — why: the loop coordinate below is the seed
# INDEX (0..4), but the trainer's `--seeds` dest is a COUNT of seeds to run
# (`for s in range(int(args.seeds))`), and that dest is `type=int`, so it
# cannot hold a YAML sequence.  The faithful migration therefore needs one
# static config per iteration, each carrying that iteration's `seeds` value.
# The files are on disk (configs/wfa269_c8_hilbert_astsr_s{0..4}.yaml); NOT
# generated at launch time.
#
# The output directory moved into the config as well
# (`output.output: molmetal/reports/wf310c_t179_w2_runs/seed<N>`); the
# trainer anchors a repo-relative output path on the repo root, so it
# resolves to the SAME absolute directory the old command line named
# (${RUN_ROOT}/seed${SEED}).
config_for_seed() {
    echo "${PROJECT_ROOT}/configs/wfa269_c8_hilbert_astsr_s${1}.yaml"
}

EPOCHS=50
SEEDS=(0 1 2 3 4)
WALL_BUDGET_MIN=270   # 4.5 hr wall budget (PROJECTED from Wf-37 + Wf-246-A 1-seed timings)
GPU="cuda:1"         # RTX 5090

echo "[wf310c-t179] project_root=${PROJECT_ROOT}"
echo "[wf310c-t179] venv=${VENV_PY}"
echo "[wf310c-t179] USER_OK_FILE=${USER_OK_FILE}"

mkdir -p "${OUT_DIR}"
mkdir -p "${RUN_ROOT}"

# ---------------------------------------------------------------------------
# 1. Pre-flight: venv + trainer + CSV existence
# ---------------------------------------------------------------------------
if [[ ! -x "${VENV_PY}" ]]; then
    echo "[wf310c-t179] ERROR: venv python missing at ${VENV_PY}" >&2
    exit 3
fi
if [[ ! -f "${TRAINER}" ]]; then
    echo "[wf310c-t179] ERROR: trainer missing at ${TRAINER}" >&2
    exit 3
fi
if [[ ! -f "${PLATINAI_CSV}" ]]; then
    echo "[wf310c-t179] ERROR: PlatinAI 4-cell CSV missing at ${PLATINAI_CSV}" >&2
    echo "[wf310c-t179] Per Wf-246-A honest-negative: do not run GPU W₂ until CSV is in place." >&2
    exit 3
fi
# Wf-A269 (2026-11-03) — AGENTS.md §5.11: after the YAML-only migration the
# per-iteration config IS the run's definition (every hyperparameter lives
# there), so a missing config is a SETUP error.  It is caught here with the
# same exit code (3) as the missing venv / trainer / CSV above, so the
# launcher stays fail-CLOSED instead of failing later inside the trainer.
for SEED in "${SEEDS[@]}"; do
    CFG="$(config_for_seed "${SEED}")"
    if [[ ! -f "${CFG}" ]]; then
        echo "[wf310c-t179] ERROR: YAML config missing at ${CFG}" >&2
        echo "[wf310c-t179] Every hyperparameter of the run lives in that file (§5.11); it must exist." >&2
        exit 3
    fi
done


# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + TODO-207 require pre_flight_check.py to PASS before
#         any trainer invocation; the canonical gate sits before the
#         --dry-run branch so the emitted plan records the pre-flight verdict.
# --- pre-flight gate (AGENTS.md §4 + TODO-207) --------------------------
PYTHONPATH=/home/hugo/codes/mol-metal ${VENV_PY} molmetal/scripts/pre_flight_check.py --pipeline launch >"${OUT_DIR}/preflight.json" 2>&1 \
    || { echo "[FATAL] pre_flight_check FAIL — see ${OUT_DIR}/preflight.json" >&2; exit 3; }

# Wf-A269 wave-3 (2026-11-02) — canonical-surface hardening (AGENTS.md §5.9).
# Reason: AGENTS.md §4 + the Wf-363-A1 schema require every launcher to
#         expose a --dry-run branch that emits DRY_RUN.log + plan.json and
#         NEVER touches a GPU; the plan must precede any trainer invocation.
# --- --dry-run branch (Wf-363-A1 schema; no trainer, no GPU) ------------
MODE="actual"
for _arg in "$@"; do
    if [ "${_arg}" = "--dry-run" ]; then MODE="dry-run"; fi
done

if [ "${MODE}" = "dry-run" ]; then
    # Wf-A269 wave-5 (2026-11-02) - REPAIR D7 (class closure): the
    # DRY_RUN.log write is ATOMIC (.tmp + mv -f), the same treatment plan.json
    # already has.  Canonical reference: scripts/launch_wf306a_drugood_vina_gpu.sh.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   cat >"${OUT_DIR}/DRY_RUN.log" <<EOF
    cat >"${OUT_DIR}/DRY_RUN.log.tmp" <<EOF
[DRY-RUN] launch_wf310c_t179_w2_gpu.sh @ Wf-310C
[DRY-RUN] wrapper_name=launch_wf310c_t179_w2_gpu.sh
[DRY-RUN] wf_origin=Wf-310C
[DRY-RUN] pre_flight_check PASS (pipeline=launch; report at ${OUT_DIR}/preflight.json)
[DRY-RUN] trainer=${TRAINER}
[DRY-RUN] py=${VENV_PY}
[DRY-RUN] pythonpath=/home/hugo/codes/mol-metal
[DRY-RUN] user_ok_required: true (USER_OK=1 confirmed)
[DRY-RUN] plan_that_would_run: ${#SEEDS[@]} seeds x ${EPOCHS} epochs, --wass-weight 0.05 --ref-distribution platinai_4cell --per-cell-cap 32, GPU=${GPU}
[DRY-RUN] out_dir=${OUT_DIR}
[DRY-RUN] run_root=${RUN_ROOT}   (per-seed trainer --output; UNTRACKED — the tracked record under out_dir is never written by this launcher)
[DRY-RUN] config_per_iteration: $(config_for_seed 0) .. $(config_for_seed 4)   (one YAML config per sweep iteration; Wf-A269 §5.11 migration — every hyperparameter named above is declared there and the trainer is invoked with --config alone)
[DRY-RUN] wall_budget_min=${WALL_BUDGET_MIN} (PROJECTED, 5 seeds x 50 epochs x ~50 min/seed)
[DRY-RUN] NOTHING WAS EXECUTED: no trainer invocation, no GPU probe, no CUDA allocation by this wrapper (pre-flight gate only, CPU).
EOF
    mv -f "${OUT_DIR}/DRY_RUN.log.tmp" "${OUT_DIR}/DRY_RUN.log"
    # Wf-A269 wave-5 (2026-11-02) - REPAIR D7: the plan.json write is now
    # ATOMIC (.tmp + mv -f), the shell equivalent of the repo's atomic-write
    # protocol (AGENTS.md §5.6, POSIX os.replace).  Canonical reference:
    # scripts/launch_wf306a_drugood_vina_gpu.sh (plan.json.tmp -> mv -f).
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   cat >"${PLAN_JSON}" <<EOF
    cat >"${PLAN_JSON}.tmp" <<EOF
{
    "wrapper": "launch_wf310c_t179_w2_gpu.sh",
    "meta": {
        "wrapper_name": "$(basename "$0")",
        "user_ok_required": true,
        "wf_origin": "Wf-310C Part 3 - TODO-179 W2 pIC50 alignment GPU retrain (Hilbert+ASTSR+W2 joint loss, 5 seeds x 50 epochs, PlatinAI 4-cell probe)",
        "parent_plan": "TODO/pending/179_wf123_p3_w2_pic50_alignment.md"
    },
    "wf_origin": "Wf-310C",
    "mode": "dry-run",
    "user_ok_required": true,
    "pre_flight": "PASS",
    "trainer": "${TRAINER}",
    "config_pattern": "${PROJECT_ROOT}/configs/wfa269_c8_hilbert_astsr_s<SEED>.yaml",
    "config_note": "one YAML config per sweep iteration; the trainer is invoked with --config alone (AGENTS.md §5.11, Wf-A269 migration)",
    "py": "${VENV_PY}",
    "pythonpath": "/home/hugo/codes/mol-metal",
    "out_dir": "${OUT_DIR}",
    "run_root": "${RUN_ROOT}",
    "epochs": ${EPOCHS},
    "seeds": [$(IFS=,; echo "${SEEDS[*]}")],
    "gpu": "${GPU}",
    "wall_budget_min": ${WALL_BUDGET_MIN},
    "wass_weight": 0.05,
    "ref_distribution": "platinai_4cell",
    "per_cell_cap": 32,
    "nothing_executed": true,
    "kill_then_adapt_directive": "USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1"
}
EOF
    mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
    echo "[wf310c-t179] [DRY-RUN] plan only -- nothing executed; DRY_RUN.log + plan.json -> ${OUT_DIR}"
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. USER_OK gate (kill_then_adapt convention per Wf-295D + Wf-306B)
# ---------------------------------------------------------------------------
if [[ ! -f "${USER_OK_FILE}" ]]; then
    # Wf-A269 wave-5 (2026-11-02) - REPAIR D4 [MED]: this runtime hint had
    # the SAME defect as the USAGE block — it told the reader to invoke the
    # launcher with no env var, which the canonical USER_OK env gate (the
    # FIRST executable block) refuses with exit 64.  The hint now names the
    # env var the gate actually requires.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   [SUPERSEDED RENDERING 1 -- the pre-wave-3 ORIGINAL, verbatim]
    #     touch ${USER_OK_FILE}
    #     bash molmetal/scripts/launch_wf310c_t179_w2_gpu.sh
    #
    #   [SUPERSEDED RENDERING 2 -- wave-3's, verbatim]
    #     touch ${USER_OK_FILE}
    #     bash scripts/launch_wf310c_t179_w2_gpu.sh
    cat <<EOF
[wf310c-t179] USER_OK GATE NOT PASSED.

To authorise the W₂ alignment GPU retrain, create the marker file AND set
USER_OK=1 in the environment, then re-run (both gates are required):

    touch ${USER_OK_FILE}
    USER_OK=1 bash scripts/launch_wf310c_t179_w2_gpu.sh

Then monitor via:
    tail -f ${RUN_LOG}

Total wall: ~4.5 hr GPU (5 seeds × 50 epochs × Hilbert+ASTSR+W₂ joint loss).
Honest framing per TODO-179 §6:
  * MEASURED baseline R² = +0.163 (Wf-37 P1 bug fix; NOT algorithmic lift).
  * PROJECTED ceiling 0.40-0.60 from W₂ alone (12-d RDKit signal floor).
  * Honest per-cell breakdown will be emitted (no aggregation hiding noise).
EOF
    exit 2
fi

# ---------------------------------------------------------------------------
# 3. GPU probe (per Wf-130 + Wf-190 env guard)
# ---------------------------------------------------------------------------
echo "[wf310c-t179] GPU probe ..."
CUDA_PROBE_OUT=$(
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF310C_T179_CUDA_VISIBLE_DEVICES:-1}}" \
    MOLMETAL_ALLOW_CPU=0 \
    "${VENV_PY}" -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
) || true
echo "[wf310c-t179] CUDA probe: ${CUDA_PROBE_OUT}"
if ! echo "${CUDA_PROBE_OUT}" | grep -q "^True"; then
    echo "[wf310c-t179] ERROR: CUDA not available on this host; aborting GPU run." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# 4. Emit plan record (always, even on dry-run) — fulfils MEASURED/PROJECTED
# label discipline (party-twenty-spirit §3.5).
# ---------------------------------------------------------------------------
{
    printf '{\n'
    printf '  "wrapper": "launch_wf310c_t179_w2_gpu.sh",\n'
    # Wf-A269 wave-5 (2026-11-02) - REPAIR D6: the emitted plan.json carries
    # the CANONICAL meta block, placed immediately AFTER the "wrapper" key
    # and BEFORE the "user_ok" key so the emitted JSON stays a valid object.
    # Canonical reference: scripts/launch_wf306a_drugood_vina_gpu.sh.
    printf '  "meta": {\n'
    printf '    "wrapper_name": "%s",\n' "$(basename "$0")"
    printf '    "user_ok_required": true,\n'
    printf '    "wf_origin": "Wf-310C Part 3 - TODO-179 W2 pIC50 alignment GPU retrain (Hilbert+ASTSR+W2 joint loss, 5 seeds x 50 epochs, PlatinAI 4-cell probe)",\n'
    printf '    "parent_plan": "TODO/pending/179_wf123_p3_w2_pic50_alignment.md"\n'
    printf '  },\n'
    printf '  "wf_origin": "Wf-310C",\n'
    # Wf-A269 wave-5 (2026-11-02) - REPAIR (raised by the wave-5 adversarial
    # verifier): "todo_ref" pointed at "TODO-pending/..." -- a directory that has
    # NEVER existed in this repo.  MEASURED: `test -e TODO-pending/179_wf123_p3_w2_pic50_alignment.md`
    # fails; the real path is "TODO/pending/179_wf123_p3_w2_pic50_alignment.md",
    # which exists.  It also contradicted the "meta"."parent_plan" value that
    # D6 added to the SAME object.  Corrected to the spelling that resolves.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #     printf '  "todo_ref": "TODO-pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    printf '  "todo_ref": "TODO/pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    # Wf-A269 (2026-11-03) — AGENTS.md §5.11 YAML-only launch: the plan
    # record names the config file that DEFINES the run (one per sweep
    # iteration), so the plan cannot drift from the hyperparameters the
    # trainer actually receives.
    printf '  "config_pattern": "%s",\n' "${PROJECT_ROOT}/configs/wfa269_c8_hilbert_astsr_s<SEED>.yaml"
    printf '  "config_note": "one YAML config per sweep iteration; the trainer is invoked with --config alone (AGENTS.md 5.11, Wf-A269 migration)",\n'
    printf '  "user_ok": true,\n'
    printf '  "epochs": %d,\n' "${EPOCHS}"
    printf '  "seeds": [%s],\n' "$(IFS=,; echo "${SEEDS[*]}")"
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "wall_budget_label": "PROJECTED (5 seeds × 50 epochs × ~50 min/seed from Wf-37 1-seed timings + Wf-246-A W2 wire-in overhead; 10%% slack)",\n'
    printf '  "wass_weight": 0.05,\n'
    printf '  "ref_distribution": "platinai_4cell",\n'
    printf '  "per_cell_cap": 32,\n'
    printf '  "out_dir": "%s",\n' "${OUT_DIR}"
    printf '  "run_root": "%s",\n' "${RUN_ROOT}"
    printf '  "status": "READY_TO_RUN",\n'
    printf '  "honest_framing": "MEASURED L5 R2=+0.163 (Wf-37 P1 bug fix; NOT algorithmic lift); PROJECTED ceiling 0.40-0.60 from W2 alone (12-d RDKit signal floor per Wf-127 Cause 2)"\n'
    printf '}\n'
} > "${PLAN_JSON}.tmp"
# Wf-A269 wave-5 (2026-11-02) - REPAIR D7: atomic publish (.tmp + mv -f),
# the shell equivalent of the repo atomic-write protocol (AGENTS.md §5.6).
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   } > "${PLAN_JSON}"
mv -f "${PLAN_JSON}.tmp" "${PLAN_JSON}"
echo "[wf310c-t179] plan emitted → ${PLAN_JSON}"

# ---------------------------------------------------------------------------
# 5. Actual GPU run — sequential 5-seed sweep, ~4.5 hr total wall
# ---------------------------------------------------------------------------
echo "[wf310c-t179] === W2 alignment GPU retrain START (5 seeds × 50 epochs) ==="

# Wf-A269 wave-5 (2026-11-02) - REPAIR D2 [HIGH]: the per-seed trainer output
# is redirected to the UNTRACKED run root so a real run can never clobber the
# GIT-TRACKED historical record under the report dir.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   [the trainer's per-seed --output]
#            --output "${OUT_DIR}/seed${SEED}" \
#   [the sweep summary's per_seed_dirs field]
#    printf '  "per_seed_dirs": [%s],\n' "$(IFS=,; for s in "${SEEDS[@]}"; do printf '"%s/seed%s"' "${OUT_DIR}" "${s}"; done | sed 's/,/, /g')"
#   [the final per-seed dirs echo]
#echo "[wf310c-t179] Per-seed dirs: ${OUT_DIR}/seed{0,1,2,3,4}"
start_ts=$(date +%s)
for SEED in "${SEEDS[@]}"; do
    echo "[wf310c-t179] === seed=${SEED} Hilbert+ASTSR+W2 50 epochs on ${GPU} ==="
    # Wf-A269 (2026-11-03) — AGENTS.md §5.11 YAML-only launch: the trainer
    # receives `--config` ALONE.  Every value the superseded invocation passed
    # on the command line now lives in the config, dest-by-dest verified by
    # scripts/_yaml_equivalence_probe.py (see the migration receipt
    # molmetal/reports/wfa269_yaml_migration/launch_wf310c_t179_w2_gpu.json).
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   [the per-seed trainer invocation]
    #            --dataset platinai \
    #            --seeds "${SEED}" \
    #            --epochs "${EPOCHS}" \
    #            --n-samples 500 \
    #            --n-folds 5 \
    #            --wass-weight 0.05 \
    #            --ref-distribution platinai_4cell \
    #            --per-cell-cap 32 \
    #            --platinai-csv "${PLATINAI_CSV}" \
    #            --output "${RUN_ROOT}/seed${SEED}" \
    CONFIG="$(config_for_seed "${SEED}")"
    if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${WF310C_T179_CUDA_VISIBLE_DEVICES:-1}}" \
         "${VENV_PY}" "${TRAINER}" \
            --config "${CONFIG}" \
            >> "${RUN_LOG}" 2>&1; then
        echo "[wf310c-t179] [OK] seed=${SEED} completed"
    else
        echo "[wf310c-t179] [FATAL] seed=${SEED} failed (see ${RUN_LOG})" >&2
        exit 4
    fi
done

elapsed_sec=$(( $(date +%s) - start_ts ))
elapsed_min=$(( elapsed_sec / 60 ))
echo "[wf310c-t179] === W2 alignment GPU retrain DONE — elapsed=${elapsed_min} min (budget=${WALL_BUDGET_MIN} min) ==="

if [ "${elapsed_min}" -gt "${WALL_BUDGET_MIN}" ]; then
    echo "[wf310c-t179] [WARN] elapsed (${elapsed_min} min) exceeded PROJECTED wall budget (${WALL_BUDGET_MIN} min); flag as MEASURED-over-PROJECTED"
fi

# ---------------------------------------------------------------------------
# 6. Aggregate per-seed summaries
# ---------------------------------------------------------------------------
echo "[wf310c-t179] === Aggregating per-seed results ==="
{
    printf '{\n'
    printf '  "wf_origin": "Wf-310C",\n'
    # Wf-A269 wave-5 (2026-11-02) - REPAIR (raised by the wave-5 adversarial
    # verifier): "todo_ref" pointed at "TODO-pending/..." -- a directory that has
    # NEVER existed in this repo.  MEASURED: `test -e TODO-pending/179_wf123_p3_w2_pic50_alignment.md`
    # fails; the real path is "TODO/pending/179_wf123_p3_w2_pic50_alignment.md",
    # which exists.  It also contradicted the "meta"."parent_plan" value that
    # D6 added to the SAME object.  Corrected to the spelling that resolves.
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #     printf '  "todo_ref": "TODO-pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    printf '  "todo_ref": "TODO/pending/179_wf123_p3_w2_pic50_alignment.md",\n'
    printf '  "elapsed_min": %d,\n' "${elapsed_min}"
    printf '  "wall_budget_min": %d,\n' "${WALL_BUDGET_MIN}"
    printf '  "n_seeds_completed": %d,\n' "${#SEEDS[@]}"
    printf '  "per_seed_dirs": [%s],\n' "$(IFS=,; for s in "${SEEDS[@]}"; do printf '"%s/seed%s"' "${RUN_ROOT}" "${s}"; done | sed 's/,/, /g')"
    printf '  "honest_framing": "MEASURED baseline L5 R2=+0.163 (Wf-37 P1 bug fix; NOT algorithmic lift); PROJECTED ceiling 0.40-0.60 from W2 alone (12-d RDKit signal floor). See TODO-179 §6 for honest projection framing."\n'
    printf '}\n'
} > "${SUMMARY_JSON}.tmp"
# Wf-A269 wave-5 (2026-11-02) - REPAIR D7 (class closure): the sweep_summary
# write is ATOMIC (.tmp + mv -f), the same treatment plan.json already has.
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   } > "${SUMMARY_JSON}"
mv -f "${SUMMARY_JSON}.tmp" "${SUMMARY_JSON}"
echo "[wf310c-t179] [OK] summary → ${SUMMARY_JSON}"

echo "[wf310c-t179] === Wf-310C TODO-179 W2 GPU retrain COMPLETE ==="
echo "[wf310c-t179] Run log: ${RUN_LOG}"
echo "[wf310c-t179] Per-seed dirs: ${RUN_ROOT}/seed{0,1,2,3,4}"
echo "[wf310c-t179] Sweep summary: ${SUMMARY_JSON}"
exit 0
