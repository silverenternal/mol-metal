#!/bin/bash
# Wf-124 P3 G5 — 15 (metal, seed) pairs launch wrapper (5 metals x 3 seeds x 50 epochs)
# Date: 2026-09-28 (post-Wf-258 user OK)
# Owner: Hugo Lee
# Per TODO-181 G5 + configs/wf124_p3_50epoch_full_sweep_g5.yaml
# Per kill_then_adapt directive: 8 waves per config (--no-enforce-budget override)
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Per Wf-328-W5 §Integration points: G5/G6 keep --w2-gradient-mode sliced.
# The sort default is for FUTURE launchers; these two are the in-flight
# retrains whose numbers are compared bit-exact against the Wf-131 P1
# baseline in the paper §6 reproducibility audit.  (bc29aa90 flipped this
# to `sort` in an unrelated venv-path commit; reverted 2026-10-20.)
# NOTE: the YAML also pins `w2_gradient_mode: sliced`; the explicit flag is
# kept only because Wf-334u4 asserts it is present and equal to sliced.
#
# Wf-r28 (2026-10-07): --num-workers and --no-cuda-graphs REMOVED from
# COMMON_ARGS. Neither is declared in this trainer's argparse, so all 15
# G5 pairs aborted with rc=2 "unrecognized arguments". The Wf-328-W6
# DataLoader-factory and Wf-335-R3 cudagraphs-lever notes below are
# retained as history only — the levers are structurally satisfied
# (no --torch-compile => no cudagraphs). See the COMMON_ARGS comment
# for the full evidence trail. G6 still carries both flags and is
# blocked on the same defect.
#
# Wf-337-G2: this launcher mirrors launch_wf124_p3_g6.sh v2 config for
# parity with the G6 per-innovation ablation sweep.
#
# Wf-347-B1: per-wave runtime check (Wf-343c2 7-check surface) wired into
# run_pair() — runs BEFORE each (metal, seed) launch with --skip-live to
# keep the dry-run path CPU-only. Also adds per-pair budget tracking
# (PAIR_BUDGET_START / PAIR_BUDGET_END log lines + elapsed_s delta).
#
# Wall-time estimate (per TODO-181 §3 + Wf-241 + Wf-260 G6 timing):
#   - Ru (heaviest cohort, ~12k rows): ~30 min for 50 epochs on RTX 5090
#   - Ir (medium cohort, ~7k rows):    ~30 min for 50 epochs on RTX 5090
#   - Os (medium-small cohort):         ~15 min for 50 epochs on RTX 5090
#   - Rh (small cohort):                ~15 min for 50 epochs on RTX 5090
#   - Re (smallest cohort):             ~10 min for 50 epochs on RTX 5090
#   - Total: ~5 hr on RTX 5090 GPU-1 (per Wf-260 G6 timing)
#
# Usage:
#   bash scripts/launch_wf124_p3_g5.sh              # real launch (skip if --dry-run passed)
#   bash scripts/launch_wf124_p3_g5.sh --dry-run    # CPU verification only (no GPU exec)
#   bash scripts/launch_wf124_p3_g5.sh --prereg docs/prereg/<file>.md
#     # Optional: pass a pre-registration record (Wf-350-B2 prereg.md template)
#     # Verifies hypothesis + primary_metric + baseline + expected_effect_size
#     # match the launch COMMON_ARGS before any GPU launch.
#     # If --prereg unset: emits WARNING that paper-grade launches should have one.

set -e
cd /home/hugo/codes/mol-metal

# Wf-366-A1 — USER_OK gate (kill_then_adapt standing directive; mirrors
# scripts/launch_wf131_p3_deferred_runs.sh Wf-355-B1 pattern at lines
# 118-125 + scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
# Wf-355-A3 pattern). Per AGENTS.md §5.1 kill_then_adapt: any > 1 hr GPU
# launch MUST be gated by USER_OK=1 in env, NOT silently auto-triggered.
# G5 wall ~5 hr on RTX 5090 GPU-1 (per Wf-260 G6 timing) — gate is
# CRITICAL SAFETY. Closes Wf-363-A2 HONEST-NEGATIVE gap (G5 was the
# only launch wrapper missing a USER_OK gate; live-run audit showed
# RC=124 timeout when invoked without --dry-run + without USER_OK).
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run (15 pairs)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --prereg docs/prereg/<file>.md   # paper-grade + pre-reg\n' "$0" >&2
    exit 64
fi

# Wf-366-A1 — source the shared DRY_RUN.log helper (consistent per-pair
# schema per Wf-363-A1; used in --dry-run mode below)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN_LOG_HELPER="${REPO_ROOT}/scripts/_dry_run_log_helper.sh"
[ -f "${DRY_RUN_LOG_HELPER}" ] || { echo "FATAL: DRY_RUN log helper missing: ${DRY_RUN_LOG_HELPER}"; exit 1; }
# shellcheck source=../_dry_run_log_helper.sh
source "${DRY_RUN_LOG_HELPER}"

# Per Wf-350-B2: parse --prereg <path> argument; if unset, emit WARNING
# that paper-grade launches should supply a pre-registration record per
# AGENTS.md §5.2 honest framing + DeepMind §3.1.
PREREG=""
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --prereg)
      shift_next=1
      ;;
    --prereg=*)
      PREREG="${arg#--prereg=}"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
  esac
done
# After flag-scan, capture positional --prereg <path> if not already set
if [[ -z "${PREREG}" ]]; then
  for i in "$@"; do
    if [[ "${prev_arg:-}" == "--prereg" ]]; then
      PREREG="$i"
      break
    fi
    prev_arg="$i"
  done
fi

# Verify prereg.md against launch config if supplied
if [[ -n "${PREREG}" ]]; then
  if [[ ! -f "${PREREG}" ]]; then
    echo "FAIL: --prereg file not found: ${PREREG}"
    exit 1
  fi
  echo "=== PREREG verification: ${PREREG} ==="
  if ! grep -qE "^## §1 Research question" "${PREREG}"; then
    echo "FAIL: prereg.md missing §1 Research question section"
    exit 1
  fi
  if ! grep -qE "^## §4 Evaluation metric" "${PREREG}"; then
    echo "FAIL: prereg.md missing §4 Evaluation metric section"
    exit 1
  fi
  if ! grep -qE "epochs +\| +50" "${PREREG}"; then
    echo "WARN: prereg.md §3 hyperparameters table does not pin epochs=50 (matches G5 COMMON_ARGS)"
  fi
  if ! grep -qE "w2_gradient_mode +\| +sort" "${PREREG}"; then
    echo "WARN: prereg.md §3 hyperparameters table does not pin w2_gradient_mode=sort (matches G5 COMMON_ARGS)"
  fi
  echo "OK: prereg.md schema valid; hypothesis + metrics sections present"
  echo ""
else
  echo "WARNING: --prereg <path> not supplied. Paper-grade launches should provide a pre-registration record per AGENTS.md §5.2 + Wf-350-B2 PREREG_TEMPLATE.md."
  echo ""
fi

# Wf-A269 wave-3 (2026-11-02): OUT_BASE is overridable so that the
# USER_OK-gate test emits into pytest's tmp_path.  That test used to
# emit into THIS tracked tree and then ``shutil.rmtree`` it, destroying
# 123 tracked published artefacts (one per_seed_metrics.json /
# training_summary.json per (metal, seed) pair).  SUPERSEDED RENDERING
# PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   OUT_BASE="results/wf124_p3_50epoch_full_sweep_g5"
OUT_BASE="${WF124_G5_OUT_BASE:-results/wf124_p3_50epoch_full_sweep_g5}"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — COMMON_ARGS used to be the CLI array the trainer
# RECEIVED; it is now only the shell record, see below):
#   # 2026-10-07 (user directive: configs in YAML, not CLI). The trainer already
#   # accepts --config and ends parse_args() in apply_yaml_overrides
#   # (train_dmpnn_multitask_wasserstein.py:3320-3322), so the yaml is the base and
#   # COMMON_ARGS below override it per-pair. Anything NOT in COMMON_ARGS falls back
#   # to the yaml value rather than to the argparse default — that is the whole
#   # point of the consolidation.
#
# Wf-A269 §5.11 (2026-11-04): COMMON_ARGS is NO LONGER passed to the trainer.
# It is now the wrapper's SHELL RECORD of the hyperparameters — the second,
# independent statement of intent that scripts/_wf_yaml_static_config_c1.py
# checks against the STATIC per-tuple YAML before the sweep
# (audit_static_configs, below).  The values the trainer actually reads live
# in configs/wfa269_c1_dmpnn_pair_sweep_g5_<metal>_seed<n>.yaml.
# Wf-A269 (2026-11-04): ${DMPNN_CONFIG} NO LONGER HAS ANY EFFECT on this
# wrapper.  The trainer is launched with the STATIC per-tuple config named by
# (metal, seed) — configs/wfa269_c1_dmpnn_pair_sweep_g5_<metal>_seed<n>.yaml
# — so there is no base config left to select and configs/dmpnn.yaml is not
# read by the launch (it is shared by 8 wrappers).  The variable is kept
# DEFINED so an operator's exported DMPNN_CONFIG cannot trip `set -u`, and it
# is echoed in the dry-run banner so a stale override is VISIBLE rather than
# silently inert.
DMPNN_CONFIG="${DMPNN_CONFIG:-configs/dmpnn.yaml}"
GPU="cuda:0"  # RTX PRO 6000 Blackwell GPU-0 (CUDA_VISIBLE_DEVICES=0; G6 recovery holds GPU-1)
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"  # Wf-261 strict scaffold split (0 leak) + v4 cohort

# Per TODO-181 §3 + Wf-260 G6 batch-size + hidden + depth (50-ep full-sweep)
# Per Wf-263C task spec: --epochs 50 --batch-size 256 --hidden 300 --depth 3 + use-l2-guard
COMMON_ARGS=(
  --epochs 50
  --batch-size 256
  --hidden 300
  --depth 3
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --ablation-config mtl_baseline
  --w2-gradient-mode sliced
  --n-mc 256
  --device "${GPU}"
  --cuda-visible-devices 0
  --cpu-num-workers 8
  # Wf-328-W2: the worker count is 8 here as well as in the dry-run --flags
  # line below.  It was previously only in the log string, so the actual
  # launch ran with the argparse default.  The flag is inert in the shipped
  # trainer (it builds batches itself), but the launcher must still declare
  # what it intends — a value that only exists in a log is not a setting.
  # Wf-r28 (2026-10-07): `--num-workers 8` and `--no-cuda-graphs` were
  # REMOVED — neither is declared in this trainer's argparse
  # (train_dmpnn_multitask_wasserstein.py), so every one of the 15 G5
  # pairs died with argparse rc=2 "unrecognized arguments" before a single
  # epoch ran. This was masked while the `set -e` bug above killed the
  # sweep even earlier. Evidence: `python -m ... --config configs/dmpnn.yaml
  # --num-workers 8` and `--no-cuda-graphs` both exit rc=2.
  #   - `--num-workers`: no dest exists; `--cpu-num-workers` (kept above) is
  #     the trainer's only worker flag, and main() already documents it as a
  #     structural no-op (manual batch build, no torch DataLoader).
  #   - `--no-cuda-graphs`: CUDA graphs only exist behind
  #     `--torch-compile` (action="store_true", default=False,
  #     train_dmpnn_multitask_wasserstein.py:3286-3287). G5 does not pass
  #     --torch-compile, so graphs are already off; the Wf-335-R3 safety
  #     lever is satisfied structurally. The only graph-related flag the
  #     trainer exposes is the enabling `--torch-compile` itself, and
  #     omitting it keeps the lever closed.
  --data-csv "${DATA_CSV}"
)

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 migration (2026-11-04) — STATIC per-tuple YAML.
#
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may hand the trainer ONLY
# --config / --out-dir / --ckpt-dir / --help.
#
# WHY one config per tuple: the trainer loops over the metal cohort and the
# seed cohort INSIDE one process
# (train_dmpnn_multitask_wasserstein.py:4352 `for metal in args.metals:` /
# :4356 `for seed in args.seeds:`), while this launcher starts ONE trainer
# PROCESS per (metal, seed) pair, each with its own --out-dir.  So the cohort
# cannot move into the SHARED yaml: one yaml carrying 5 metals x 3 seeds
# would turn the 15 processes into 1 and change the run.  Each pair therefore
# needs a config that declares exactly ONE metal and ONE seed.
#
# WHY the file is STATIC (on disk, in git) and not written at launch time:
# §5.11's own rationale is "a knob that lives only in a shell script is a knob
# nobody audits" — a launch-time-written config puts the hyperparameters back
# into a shell variable.  A static file is reviewable BEFORE the run and
# reproducible AFTER it.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269 wave-3, 2026-11-02 — the launch-time derivation this replaces):
#   derive_pair_config() {
#     local metal=$1
#     local seed=$2
#     local dest=$3
#     "${PY}" - "${DMPNN_CONFIG}" "${dest}" "${metal}" "${seed}" "${DATA_CSV}" <<'PYEOF'
#   import os
#   import pathlib
#   import sys
#
#   import yaml
#
#   base_path, dest_path, metal, seed, csv_path = sys.argv[1:6]
#   with open(base_path, "r", encoding="utf-8") as fh:
#       cfg = yaml.safe_load(fh) or {}
#   if not isinstance(cfg, dict):
#       raise SystemExit("derive_pair_config: base config is not a mapping: %s" % base_path)
#
#
#   def put(dotted, value):
#       parts = dotted.split(".")
#       node = cfg
#       for part in parts[:-1]:
#           nxt = node.get(part)
#           if not isinstance(nxt, dict):
#               nxt = {}
#               node[part] = nxt
#           node = nxt
#       node[parts[-1]] = value
#
#
#   put("data.metals", [metal])
#   put("data.seeds", [int(seed)])
#   put("protocol.seeds", [int(seed)])
#   put("data.csv", csv_path)   # flat form wins -> dest data_csv
#   put("csv", csv_path)        # top level -> dest csv (data.csv resolves to data_csv)
#
#   dest = pathlib.Path(dest_path)
#   dest.parent.mkdir(parents=True, exist_ok=True)
#   tmp = dest.with_name(dest.name + ".tmp")
#   with open(tmp, "w", encoding="utf-8") as fh:
#       yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
#   os.replace(tmp, dest)
#   print("derive_pair_config: %s (base=%s metal=%s seed=%s)" % (dest, base_path, metal, seed))
#   PYEOF
#   }
#
# WHY the derivation was replaced: it wrote the config at launch time from
# ${DMPNN_CONFIG} — a config SHARED by 8 wrappers — so the tuple coordinates
# and every G5 hyperparameter existed only as shell variables at review time.
# The static cohort removes both dependencies: every value the trainer reads
# is in the file named by the tuple, and that file's header records the
# pre-migration argv it reproduces.
#
# The 15 files are audited BEFORE the sweep by
# scripts/_wf_yaml_static_config_c1.py, which compares the wrapper's own shell
# record (COMMON_ARGS, below) against what the trainer will ACTUALLY read,
# through the trainer's OWN parser and the REAL loader — dest by dest, value
# AND type.  MEASURED 2026-11-04: 15/15 PASS, 240 flag dests verified.
# ---------------------------------------------------------------------------
PAIR_CONFIG_PREFIX="${PAIR_CONFIG_PREFIX:-configs/wfa269_c1_dmpnn_pair_sweep_g5_}"
# 5 metals x 3 seeds — the launcher MUST NOT silently under-launch.
PAIR_CONFIG_COUNT=15

pair_config_path() {
  local metal=$1
  local seed=$2
  printf '%s%s_seed%s.yaml' "${PAIR_CONFIG_PREFIX}" "${metal}" "${seed}"
}

# Fail-closed: a missing / drifted pair config aborts the launch.
audit_static_configs() {
  "${PY}" scripts/_wf_yaml_static_config_c1.py \
    --trainer "${SCRIPT}" \
    --config-glob "${PAIR_CONFIG_PREFIX}*.yaml" \
    --flags "${COMMON_ARGS[*]} --csv ${DATA_CSV}" \
    --expect-min-count "${PAIR_CONFIG_COUNT}"
}

# ---- DRY-RUN mode: verify config + CSV paths only (no GPU execution) ----
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "=== Wf-124 P3 G5 — DRY-RUN (CPU verification only; USER_OK=1 confirmed) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "SCRIPT: ${SCRIPT}"
  echo "DATA_CSV: ${DATA_CSV}"
  echo "GPU: ${GPU}"
  echo "COMMON_ARGS: ${COMMON_ARGS[*]}"
  # Wf-A269 (2026-11-04): DMPNN_CONFIG has NO EFFECT on this launch (the
  # per-tuple config is selected by (metal, seed)).  Echoing it makes a
  # stale operator override VISIBLE instead of silently inert.
  echo "DMPNN_CONFIG (NO EFFECT — inert override, see the note above): ${DMPNN_CONFIG}"
  echo ""
  echo "--- CSV sanity check ---"
  test -f "${DATA_CSV}" && echo "OK: CSV exists (${DATA_CSV})" || { echo "FAIL: CSV missing"; exit 1; }
  n_rows=$(wc -l < "${DATA_CSV}")
  echo "n_rows (incl header): ${n_rows}"
  n_cols=$(head -1 "${DATA_CSV}" | tr ',' '\n' | wc -l)
  echo "n_columns: ${n_cols}"
  echo "header: $(head -1 "${DATA_CSV}")"
  echo ""
  echo "--- 15 (metal, seed) pairs to enumerate ---"
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      echo "  ${metal} seed=${seed}  ->  ${OUT_BASE}/${metal}/seed${seed}/  [config: $(pair_config_path "${metal}" "${seed}")]"
    done
  done
  echo ""

  # Wf-A269 §5.11 (2026-11-04): the dry run must PROVE that the static
  # per-tuple configs exist and carry the hyperparameters this wrapper
  # intends.  The audit is fail-closed and CPU-only (it reads argparse + YAML
  # and never starts a trainer).
  echo "--- static per-tuple config audit (fail-closed, CPU-only) ---"
  audit_static_configs || { echo "FAIL: static-config audit"; exit 1; }
  echo ""

  # Wf-A269 (2026-11-04) — the launch path now registers a provenance record
  # before every pair (see run_pair).  The dry run must PROVE that step works,
  # so it exercises it for all 15 pairs — into a SCRATCH directory, never into
  # the real run dirs: a dry run must not leave an "expected" artefact
  # behind for a run that never happened.
  echo "--- launch-provenance registration dry run (scratch dir, CPU-only) ---"
  prov_scratch="${OUT_BASE}/_dryrun_provenance"
  rm -rf "${prov_scratch}"
  prov_fail=0
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      scratch_pair="${prov_scratch}/${metal}_seed${seed}"
      "${PY}" scripts/_wf_launch_provenance.py emit \
        --trainer "${SCRIPT}" \
        --config "$(pair_config_path "${metal}" "${seed}")" \
        --out-dir "${scratch_pair}" \
        --ckpt-dir "${scratch_pair}/ckpts" >/dev/null 2>&1 \
        || { echo "  FAIL: provenance registration for ${metal} seed=${seed}"; prov_fail=1; }
    done
  done
  if [ "$prov_fail" -ne 0 ]; then
    echo "FAIL: launch-provenance registration dry run"
    exit 1
  fi
  echo "  OK: 15/15 provenance registrations succeeded (scratch: ${prov_scratch})"
  echo ""

  # Wf-366-A1 — emit structured DRY_RUN.log via Wf-363-A1 helper
  pair_args=()
  for metal in Ru Ir Os Rh Re; do
    for seed in 0 1 2; do
      pair_args+=( --pair "${metal} seed=${seed}" )
    done
  done
  # Wf-A269 §5.11 (2026-11-04): the --flags below are the RUN's
  # hyperparameters, but they are no longer CLI flags — they are carried by
  # the static per-tuple YAML, and audit_static_configs proves the two agree.
  # shellcheck disable=SC2068
  emit_dry_run_log "${OUT_BASE}/DRY_RUN.log" \
    --title "Wf-124 P3 G5 — 15 (metal, seed) pairs full sweep" \
    --wrapper "$(basename "$0")" \
    --mode "G5_50epoch_full_sweep" \
    --trainer "${SCRIPT}" \
    --flags "--epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3 --alpha 0.5 --wasserstein-weight 0.05 --use-l2-guard --ablation-config mtl_baseline --w2-gradient-mode sliced --n-mc 256 --device ${GPU} --cpu-num-workers 8" \
    --gpu "${GPU}" \
    --budget-hr "5" \
    --user-ok "REQUIRED" \
    ${pair_args[@]+"${pair_args[@]}"}
  echo "DRY_RUN.log emitted -> ${OUT_BASE}/DRY_RUN.log"

  # Wf-366-A1 — emit plan.json (consistent with Wf-355 pattern)
  PLAN_JSON="${OUT_BASE}/plan.json"
  {
    printf '{\n'
    printf '  "wrapper": "launch_wf124_p3_g5.sh",\n'
    printf '  "wf_origin": "Wf-366-A1 (USER_OK gate fix)",\n'
    printf '  "parent_plan": "TODO/completed/181_wf124_p3_50_epoch_full_sweep.md",\n'
    printf '  "user_ok": true,\n'
    printf '  "dry_run": true,\n'
    printf '  "metals": ["Ru", "Ir", "Os", "Rh", "Re"],\n'
    printf '  "seeds": [0, 1, 2],\n'
    printf '  "epochs": 50,\n'
    printf '  "total_runs": 15,\n'
    printf '  "gpu": "%s",\n' "${GPU}"
    printf '  "wall_budget_hr": 5,\n'
    printf '  "wall_budget_label": "PROJECTED (~5 hr GPU sequential per Wf-260 G6 timing; Ru heaviest ~30 min/seed x 3 + Ir ~30 min x 3 + Os ~15 min x 3 + Rh ~15 min x 3 + Re ~10 min x 3)",\n'
    printf '  "trainer": "%s",\n' "${SCRIPT}"
    printf '  "data_csv": "%s",\n' "${DATA_CSV}"
    printf '  "out_base": "%s",\n' "${OUT_BASE}"
    printf '  "honest_framing": {\n'
    printf '    "MEASURED_today": "wrapper ready; user_ok gate enforced; 15 pairs enumerated; DRY_RUN.log emitted",\n'
    printf '    "PROJECTED": "Ru seed 2 + Ir seed 0 in flight on GPU 0 (Wf-328 G6 PID 2040025); G5 wall ~5 hr on GPU 1",\n'
    printf '    "kill_then_adapt_directive": "5-hr wall budget; USER_OK gate HARD-REQUIRED per mol-metal-gpu-execution §0 + AGENTS.md §5.1",\n'
    printf '    "wf337_g2_status": "Wf-337-G2 hardened launcher config to mirror G6; Wf-366-A1 closes the residual USER_OK gate gap surfaced by Wf-363-A2 audit"\n'
    printf '  },\n'
    printf '  "status": "PLAN_ONLY"\n'
    printf '}\n'
  } > "${PLAN_JSON}"
  echo "plan.json emitted -> ${PLAN_JSON}"
  echo ""
  echo "DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed."
  exit 0
fi

# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
# Wf-356-A1: wf343c3 data quality check inserted between pre-flight and
# per-wave checklist — script is loader-agnostic (covers all 5 R20 loaders
# internally: load_binding_db, load_nci60, load_drugood, metalcytotox_v4,
# platinai_4cell). Does NOT accept --data-csv; only --out-dir.
# Use launcher-specific out-dir so each launcher's report is preserved.
# 2026-10-03: declare the wave's grid + budget so the per-wave runtime gate has
# REAL inputs. Without the declaration that gate could only SKIP (it needs a
# grid and a budget and had no way to receive them). Budget is the 48h session
# figure from the user directive; the grid is 5 metals x 3 seeds = 15 pairs.
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/pre_flight_check.py \
  --pipeline launch \
  --declared-wave "Wf-124-P3-G5 15-pair full sweep" \
  --pairs-metals "Ru,Ir,Os,Rh,Re" \
  --pairs-seeds "0,1,2" \
  --budget-hr 48 \
  || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf124_p3_g5_data_quality \
  || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G5" || { echo "FAIL: per_wave_checklist"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }

# Wf-A269 §5.11 (2026-11-04): fail-closed static-config audit BEFORE any pair
# launches.  It compares the wrapper's own shell record (COMMON_ARGS) against
# what the trainer will actually read, dest by dest, through the trainer's
# own parser and the real loader.  A drifted pair config aborts the sweep.
echo "=== static per-tuple config audit (fail-closed) ==="
audit_static_configs || { echo "FAIL: static-config audit (fail-closed)"; exit 1; }
echo ""

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 migration — the per-pair cohort lives in a STATIC yaml.
#
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may hand the trainer ONLY
# --config / --out-dir / --ckpt-dir / --help. The four flags this wrapper
# used to put on the trainer invocation line carry the PER-PAIR cohort, and
# the trainer loops over the metal cohort and the seed cohort INSIDE one
# process — so the cohort cannot simply move into the shared yaml: one yaml
# carrying 5 metals x 3 seeds would turn the 15 processes into 1 and change
# the run.
#
# So every (metal, seed) pair gets its OWN yaml — and that yaml is a STATIC
# FILE ON DISK (configs/wfa269_c1_dmpnn_pair_sweep_g5_<metal>_seed<n>.yaml),
# NOT one written at launch time.  It carries exactly the four values the
# CLI used to carry:
#   metals   <- data.metals              [metal]   (was: CLI --metals)
#   seeds    <- data.seeds + protocol.seeds [seed] (was: CLI --seeds)
#   csv      <- csv (TOP-LEVEL key)      <path>   (was: CLI --csv)
#   data_csv <- data.csv (flat form wins over the leaf form) <path>
# The base yaml configs/dmpnn.yaml is NOT edited and is NOT read by the
# launch: it is shared by 8 wrappers.
#
# Wf-A269 (2026-11-04) — derive_pair_config() DELETED.  It had been left
# behind by the launch-time-derivation -> static-file migration and had
# **0 call sites** (measured: grep -n derive_pair_config returns only the
# definition and the comments).  Leaving it was a TRAP: it is a working
# implementation of the exact mechanism the c1 contract forbids (a config
# written at launch time from a shell variable), and the next reader would
# have had no way to tell it was unreachable.
#
# The superseded function body is preserved VERBATIM in the Wf-A269 wave-3
# block further up this file (the block headed "SUPERSEDED RENDERING
# PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6", wave-3, 2026-11-02).
# It is NOT duplicated here — one preserved rendering, one marker.
# ---------------------------------------------------------------------------

# ---- Helper: poll pair log until target epoch is reached (Wf-302B guard) ----
# Refs: Wf-285D + Wf-291B — elapsed=0 MUST be initialised BEFORE the loop
# Refs: Wf-298E critical remaining #2 — bash wait_until_epoch_reached was MISSING
wait_until_epoch_reached() {
  local pair_log="$1" target_epoch="$2" max_wait_s="$3"
  max_wait_s="${max_wait_s:-1200}"
  local start_s elapsed last_epoch
  start_s=$(date +%s)
  elapsed=0
  while true; do
    last_epoch=$(grep -oE 'epoch +[0-9]+/[0-9]+' "$pair_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    if [ -n "$last_epoch" ] && [ "$last_epoch" -ge "$target_epoch" ]; then
      echo "Pair complete: last_epoch=$last_epoch >= target=$target_epoch"
      return 0
    fi
    elapsed=$(( $(date +%s) - start_s ))
    if [ "$elapsed" -ge "$max_wait_s" ]; then
      echo "TIMEOUT: last=$last_epoch elapsed=$((elapsed))s"
      return 1
    fi
    sleep 10
  done
}

# ---- REAL launch: per-pair runner ----
run_pair() {
  local metal=$1
  local seed=$2
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"

  # Wf-347-B1: per-wave runtime check BEFORE launching this pair
  # (Wf-343c2 7-check surface: preflight freshness + no-fail gates +
  # hyperparams config + YuelBond decoder + wave budget + GPU VRAM +
  # disk free space). --skip-live toggles nvidia-smi/df for CPU-only dry-runs.
  # Per Wf-337-G2 G5 launcher hardening + kill_then_adapt directive:
  # per-wave check is a DIAGNOSTIC emitter (per-pair JSON to
  # molmetal/reports/wf343c2_per_wave_check/), NOT a hard gate. The 2
  # known project-wide pre-flight FAILs (CFM Wasserstein loss file MISSING
  # + Vina not installed per AGENTS.md §5.4 honest-negative markers) are
  # pre-existing and unrelated to G5; the wf343c2 run emits the JSON
  # evidence so post-sweep diagnostics can attribute any per-pair failure
  # to pre-flight vs wave-specific causes. The exit-code is captured for
  # the per-pair budget tracker but does NOT abort the sweep.
  # Wf-r28: `set -e` is active at the top of this script. The bare
  # wf343c2 invocation below returned rc=1 (2 sub-checks FAIL) and `set -e`
  # therefore aborted the WHOLE script at this line — BEFORE the
  # `local rc_c2=$?` capture below could run, and before any pair trained.
  # That silently defeated the diagnostic-only contract documented above and
  # surfaced as TODO-181 G5 task-3 rc=1 after 3 min (Wf-r28 2026-10-07).
  # Fix: `|| rc_c2=$?` puts the command in a tested context so `set -e` does
  # not fire, and captures the real rc. Mirrors the G6 launcher's
  # `|| { ...; return 1; }` guard shape (scripts/launch_wf124_p3_g6.sh:354).
  # `local rc_c2` MUST be declared before the call under `set -u` semantics.
  local rc_c2=0
  # Wf-A269 (2026-11-04, MEASURED): the per-wave check must validate the
  # config the run ACTUALLY uses.  SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   --config configs/wf124_p3_50epoch_full_sweep_g5.yaml
  # The superseded rendering above named the SWEEP-SPEC config, whose
  # `protocol.batch_size` is 64 — but the G5 run uses 256 (the static
  # per-tuple configs carry `train.batch_size: 256`, and the pre-migration CLI
  # passed `--batch-size 256`).  MEASURED 2026-11-04 with the trainer's own
  # parse_args + the real loader (molmetal/scripts/wfA2_config_loader.py):
  #   configs/wf124_p3_50epoch_full_sweep_g5.yaml  -> batch_size == 64
  #   configs/wfa269_c1_dmpnn_pair_sweep_g5_Ru_seed0.yaml -> batch_size == 256
  # and `protocol.batch_size` in the sweep spec DOES resolve to the trainer's
  # `batch_size` dest (loader `_resolve_dest` -> dest 'batch_size', value 64),
  # so the check was validating a config whose batch_size the run does not use.
  # REPAIR: point the check at the per-tuple config for THIS (metal, seed) —
  # run_pair() knows both, and `pair_config_path` (defined above) resolves the
  # same static file the trainer is launched with a few lines below.  The
  # fail-closed existence check on that path stays below, unchanged.
  # The check reads only two things from the config
  # (wf343c2_per_wave_check.py: check_hyperparams_config_complete and
  # check_yuelbond_decoder_registered); MEASURED 2026-11-04, all 7 sub-check
  # statuses are IDENTICAL before and after the switch (the one FAIL,
  # hyperparams_config_complete, is a pre-existing schema FAIL — the sweep spec
  # is missing keys 'flow, train', the per-tuple config missing
  # 'flow, honest_framing'; REQUIRED_CONFIG_KEYS is a CFM-shaped contract).
  local pair_cfg_for_check
  pair_cfg_for_check="$(pair_config_path "${metal}" "${seed}")"
  PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c2_per_wave_check.py \
    --wave "Wf-124-P3-G5-${metal}-seed${seed}" \
    --pairs-metals "${metal}" \
    --pairs-seeds "${seed}" \
    --config "${pair_cfg_for_check}" \
    --budget-hr 0.9 \
    --min-vram-mib 8192 \
    --min-disk-gib 50 \
    --skip-live > "${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" 2>&1 || rc_c2=$?
  echo "PAIR_PER_WAVE_CHECK ${metal} seed=${seed} rc=${rc_c2} log=${OUT_BASE}/${metal}/seed${seed}/wf343c2_per_wave_check.log" | tee -a "${LOG}"
  # Diagnostic-only: do NOT return 1 on rc!=0 (per kill_then_adapt; see
  # comment above). The JSON evidence is captured for post-sweep audit.

  # Wf-347-B1: per-pair budget tracking — capture start time so the
  # pair-level wall-time can be aggregated into the wave_budget_tracker
  # JSON after the sweep. Mirrors Wf-328-W2 / Wf-337-G2 convention.
  local pair_start_s
  pair_start_s=$(date +%s)
  echo "PAIR_BUDGET_START ${metal} seed=${seed} epoch=$(date '+%s')" | tee -a "${LOG}"

  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} ===" | tee -a "${LOG}"
  # GPU 0 (RTX PRO 6000) — aligns with this script's GPU="cuda:0" header
  # and the --cuda-visible-devices 0 flag above; the old '1' predated the
  # GPU-0 resolver default.  (G6 recovery holds GPU-1.)
  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> 0.
  #
  # Wf-A269 §5.11 (2026-11-03): the per-pair cohort and the corpus paths are
  # no longer CLI flags — they live in the per-pair yaml that was, at that
  # time, derived at launch time from ${DMPNN_CONFIG}.  The values the
  # trainer reads are unchanged.
  # Wf-A269 §5.11 (2026-11-04): the per-pair cohort AND every hyperparameter
  # now live in a STATIC per-tuple config; the trainer invocation carries only
  # the three allowed run-control flags.  The config is copied into the run
  # directory so the run's own artefacts record exactly what was configured.
  local pair_cfg
  pair_cfg="$(pair_config_path "${metal}" "${seed}")"
  if [[ ! -f "${pair_cfg}" ]]; then
    echo "FAIL: static pair config missing: ${pair_cfg}" | tee -a "${LOG}"
    return 1
  fi
  install -m 0644 "${pair_cfg}" "${OUT_BASE}/${metal}/seed${seed}/config.yaml"

  # Wf-A269 (2026-11-04) — LAUNCH PROVENANCE.
  #
  # WHY: §5.11 puts every hyperparameter in a FILE, but the file records the
  # DECLARED value and nothing recorded what the trainer ACTUALLY read.  A
  # config edited after the run silently turns the declaration into a lie.
  # Hydra writes the RESOLVED config into the run dir at launch; this is the
  # same mechanism.  The wrapper registers what it is ABOUT to run
  # (resolved_config.expected.yaml + launch_provenance.json), the trainer
  # records what it ACTUALLY consumed (resolved_config.actual.yaml, written
  # by the loader when MOLMETAL_RESOLVED_CONFIG_PATH is set), and the check
  # below compares the two — in two different processes, from two different
  # inputs, so the comparison is a check and not a transcript.
  #
  # The four flags below are the §5.11 legal set (they select the RUN, not
  # WHAT to train); this is not a trainer invocation.
  local pair_out="${OUT_BASE}/${metal}/seed${seed}"
  local pair_ckpt="${pair_out}/ckpts"
  mkdir -p "${pair_out}"
  "${PY}" scripts/_wf_launch_provenance.py emit \
    --trainer "${SCRIPT}" \
    --config "${pair_cfg}" \
    --out-dir "${pair_out}" \
    --ckpt-dir "${pair_ckpt}" \
    2>&1 | tee -a "${LOG}"
  local prov_rc=${PIPESTATUS[0]}
  if [ "$prov_rc" -ne 0 ]; then
    echo "FAIL: launch-provenance registration failed for ${metal} seed=${seed} (rc=${prov_rc})" | tee -a "${LOG}"
    return 1
  fi

  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
  # (Wf-A269 wave-3, 2026-11-02 — the launch-time derivation + the CLI array):
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid nohup ${PY} ${SCRIPT} \
  #     --config "${DMPNN_CONFIG}" \
  #     --metals "${metal}" \
  #     --seeds "${seed}" \
  #     --csv "${DATA_CSV}" \
  #     --data-csv "${DATA_CSV}" \
  #     --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
  #     --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
  #     "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" \
  MOLMETAL_RESOLVED_CONFIG_PATH="${pair_out}/resolved_config.actual.yaml" \
  setsid nohup ${PY} ${SCRIPT} \
    --config "${pair_cfg}" \
    --out-dir "${pair_out}" \
    --ckpt-dir "${pair_ckpt}" \
    2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}

  # Wf-347-B1: per-pair budget tracking — capture end time + emit delta
  local pair_end_s pair_elapsed_s
  pair_end_s=$(date +%s)
  pair_elapsed_s=$(( pair_end_s - pair_start_s ))
  echo "PAIR_BUDGET_END ${metal} seed=${seed} elapsed_s=${pair_elapsed_s} rc=${rc}" | tee -a "${LOG}"

  # Wf-A269 (2026-11-04) — the provenance check.  It compares the config the
  # wrapper REGISTERED against the config the trainer CONSUMED.  A mismatch
  # means the run was not the run the wrapper intended.  --require-actual
  # makes a missing actual artefact a FAIL rather than a SKIP: a run that
  # recorded no provenance has not proven anything.
  "${PY}" scripts/_wf_launch_provenance.py verify \
    --out-dir "${pair_out}" --require-actual 2>&1 | tee -a "${LOG}"
  local prov_verify_rc=${PIPESTATUS[0]}

  echo "=== $(date '+%H:%M:%S') END ${metal} seed=${seed} (rc=${rc}) ===" | tee -a "${LOG}"
  if [ "$rc" -ne 0 ]; then
    echo "FAIL: ${metal} seed=${seed} exited rc=${rc}"
    return "$rc"
  fi
  if [ "$prov_verify_rc" -ne 0 ]; then
    echo "FAIL: launch-provenance mismatch for ${metal} seed=${seed} (rc=${prov_verify_rc})" | tee -a "${LOG}"
    return 1
  fi
  # Wf-302B guard: verify the pair actually reached target epoch (50)
  wait_until_epoch_reached "${pair_log}" 50 1200 || { echo "FAIL: ${metal} seed=${seed} did not reach epoch 50"; return 1; }
}

echo "=== Wf-124 P3 G5 START $(date) ===" | tee -a "${LOG}"
echo "15 pairs (5 metals x 3 seeds x 50 ep) -- 8 waves per kill_then_adapt, --no-enforce-budget so all 15 PLANNED" | tee -a "${LOG}"

# 15 (metal, seed) pairs enumerated:
# Wave A: Ru s0 (~30 min)
run_pair Ru 0
# Wave B: Ru s1 (~30 min)
run_pair Ru 1
# Wave C: Ru s2 (~30 min)
run_pair Ru 2
# Wave D: Ir s0/s1 (~60 min)
run_pair Ir 0
run_pair Ir 1
# Wave E: Ir s2 + Os s0 (~45 min)
run_pair Ir 2
run_pair Os 0
# Wave F: Os s1/s2 + Rh s0 (~45 min)
run_pair Os 1
run_pair Os 2
run_pair Rh 0
# Wave G: Rh s1/s2 + Re s0 (~40 min)
run_pair Rh 1
run_pair Rh 2
run_pair Re 0
# Wave H: Re s1/s2 (~20 min)
run_pair Re 1
run_pair Re 2

echo "=== ALL TODO-181 G5 PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation — collect per_seed_metrics_seed*.json into training_summary.json
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for metal_dir in sorted(Path('${OUT_BASE}').iterdir()):
    if not metal_dir.is_dir() or metal_dir.name == 'logs':
        continue
    for seed_dir in sorted(metal_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        for p in seed_dir.glob('per_seed_metrics_seed*.json'):
            runs.append({'metal': metal_dir.name, 'seed': seed_dir.name.replace(\"seed\", \"\"), 'metrics_path': str(p)})
out = {'todo_181_g5_total_pairs': len(runs), 'runs': runs}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs (expected 15)')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"

# 30ep vs 50ep comparison stub (filled once Wf-261 v2 baseline re-pulled)
echo "=== 30ep vs 50ep comparison placeholder at ${OUT_BASE}/30ep_vs_50ep_comparison.json ===" | tee -a "${LOG}"
echo "TODO: aggregate once g5 sweep completes (TODO-181 §3.4)" | tee -a "${LOG}"
