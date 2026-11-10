#!/bin/bash
# Wf-310A — Retry 2 missing G5 pairs (Ru seed=2, Ir seed=0)
# Background launcher. Each pair runs on a different GPU for parallel exec.
# Both pairs were killed mid-training by the kill_then_adapt scheduler
# during the original G5 sweep (TODO-181) on 2026-09-29 03:11:13 CST.
#
# GPU mapping:
#   - Ru seed=2 → CUDA_VISIBLE_DEVICES=1, --device cuda:0 (RTX 5090 — heavy Ru cohort)
#   - Ir seed=0 → CUDA_VISIBLE_DEVICES=0, --device cuda:0 (RTX PRO 6000 — medium Ir cohort)
#
# Wall-time estimate:
#   - Ru seed=2: ~30 min for 50 epochs on RTX 5090
#   - Ir seed=0: ~30 min for 50 epochs on RTX PRO 6000
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Usage:
#   bash scripts/retry_g5_missing_pairs.sh         # sequential (default)
#   bash scripts/retry_g5_missing_pairs.sh --parallel   # parallel on 2 GPUs
#   bash scripts/retry_g5_missing_pairs.sh --dry-run     # CPU-only verification
#     # ADDED 2026-11-04 (Wf-A269 §5.11 migration): a CPU-only dry-run that
#     # runs the fail-closed static-config audit and writes NOTHING into the
#     # tracked tree.  Before this the wrapper had NO dry-run mode, so the
#     # audit could not be exercised without a real GPU launch.

set -e
cd /home/hugo/codes/mol-metal

# Wf-A269 §5.11 migration (2026-11-04): OUT_BASE is overridable so the
# CPU-only dry-run can be exercised against a scratch tree instead of the
# tracked results/wf124_p3_50epoch_full_sweep_g5/ dir.
OUT_BASE="${WF310A_OUT_BASE:-results/wf124_p3_50epoch_full_sweep_g5}"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/mol-metal/.venv/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"

# ---------------------------------------------------------------------------
# Wf-A269 §5.11 (2026-11-04) — STATIC per-pair YAML.
#
# AGENTS.md §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动
# 训练和推理，no cli参数"): a launch wrapper may hand the trainer ONLY
# --config / --out-dir / --ckpt-dir / --help.
#
# WHY one config per pair: the trainer loops over the metal cohort and the
# seed cohort INSIDE one process
# (train_dmpnn_multitask_wasserstein.py:4352 `for metal in args.metals:` /
# :4356 `for seed in args.seeds:`), while this launcher starts ONE trainer
# PROCESS per (metal, seed) pair, each with its own --out-dir.  So the
# cohort cannot move into a SHARED yaml: one yaml carrying both metals x both
# seeds would turn the 2 processes into 1 and change the run.  Each pair
# therefore needs a config that declares exactly ONE metal and ONE seed.
#
# WHY the file is STATIC (on disk, in git) and not written at launch time:
# §5.11's own rationale is "a knob that lives only in a shell script is a
# knob nobody audits" — a launch-time-written config puts the hyperparameters
# back into a shell variable.  A static file is reviewable BEFORE the run and
# reproducible AFTER it.
#
# The 2 files are audited BEFORE the launch by
# scripts/_wf_yaml_static_config_c1.py, which compares the wrapper's own
# shell record (COMMON_ARGS) against what the trainer will ACTUALLY read,
# through the trainer's OWN parser and the REAL loader — dest by dest, value
# AND type.
#
# HOW MANY PAIRS: this wrapper starts exactly TWO processes — (Ru, seed=2)
# and (Ir, seed=0) — and there are exactly TWO static configs.  MEASURED
# 2026-11-04: `ls configs/wfa269_c1_dmpnn_pair_sweep_retry_g5_*.yaml | wc -l`
# = 2.  The audit's --expect-min-count is 2, so a cohort that shrank would
# fail closed instead of under-launching.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — the dedicated base config + the launch-time
# derivation this replaces):
#   # 2026-10-07 (user directive: "所有启动脚本都用yaml管理…不要用cli,那样不好追溯").
#   # The trainer already accepts --config and ends parse_args() in
#   # apply_yaml_overrides (train_dmpnn_multitask_wasserstein.py:3320-3322).
#   #
#   # 2026-11-03 (AGENTS.md §5.11 migration): every hyperparameter this wrapper
#   # used to pass on the trainer command line now lives in a DEDICATED config —
#   # configs/retry_g5_missing_pairs_wf310a.yaml — and the trainer is invoked with
#   # --config / --out-dir / --ckpt-dir only. configs/dmpnn.yaml is shared by 8
#   # launchers, so its values could not carry this wrapper's numbers.
#   #
#   # OPERATOR-OVERRIDE SEMANTICS CHANGED: DMPNN_CONFIG still selects the base
#   # config, but it can no longer be PARTIALLY overridden from the command line —
#   # §5.11 removed the command line. With DMPNN_CONFIG unset (the default, and
#   # the only way this wrapper is launched in the repo) the derived config is
#   # behaviourally identical to the pre-migration launch: MEASURED by a full
#   # argparse-namespace diff (old command line vs new --config), 0 differing
#   # dests. Setting DMPNN_CONFIG now REPLACES this wrapper's config wholesale.
#   DMPNN_CONFIG="${DMPNN_CONFIG:-configs/retry_g5_missing_pairs_wf310a.yaml}"
#
#   # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   # COMMON_ARGS=(
#   #   --epochs 50
#   #   --batch-size 256
#   #   --hidden 300
#   #   --depth 3
#   #   --lr 1e-3
#   #   --alpha 0.5
#   #   --wasserstein-weight 0.05
#   #   --use-l2-guard
#   #   --ablation-config mtl_baseline
#   #   --w2-gradient-mode sliced
#   #   --n-mc 256
#   #   --cpu-num-workers 8
#   #   --data-csv "${DATA_CSV}"
#   #   # Wf-335-R3's `--no-cuda-graphs` was REMOVED here on 2026-10-27: the trainer's
#   #   # argparse does not declare it, so this retry wrapper — the one path that
#   #   # exists to recover the Wf-330-B4 failures — died rc=2 before launching
#   #   # anything. Graphs only exist behind `--torch-compile`, which is not passed.
#   # )
# ---------------------------------------------------------------------------
PAIR_CONFIG_PREFIX="${PAIR_CONFIG_PREFIX:-configs/wfa269_c1_dmpnn_pair_sweep_retry_g5_}"
# 2 pairs — the launcher MUST NOT silently under-launch.
PAIR_CONFIG_COUNT=2

# COMMON_ARGS is NO LONGER passed to the trainer.  It is now the wrapper's
# SHELL RECORD of the hyperparameters — the second, independent statement of
# intent that scripts/_wf_yaml_static_config_c1.py checks against the STATIC
# per-pair YAML before the launch (audit_static_configs, below).  The values
# the trainer actually reads live in
# configs/wfa269_c1_dmpnn_pair_sweep_retry_g5_<metal>_seed<n>.yaml.
#
# NOTE on --cuda-visible-devices: this wrapper does NOT pass it, and never
# did (MEASURED 2026-11-03 from the pre-migration wrapper).  The GPU is
# selected by the CUDA_VISIBLE_DEVICES command PREFIX (an environment
# variable, not a trainer flag), so the dest keeps its argparse default ""
# and the static config carries compute.cuda_visible_devices: "".  The audit
# pins that with --expect so a future edit cannot quietly add the flag.
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
  --device cuda:0
  --cpu-num-workers 8
  --data-csv "${DATA_CSV}"
)

# ---------------------------------------------------------------------------
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (Wf-A269, 2026-11-04 — the worker-count pin and the launch-time
# derivation, both removed):
#   # ---- Launcher-side worker-count PIN (NOT passed on the command line) -------
#   # AGENTS.md §5.11 moved --cpu-num-workers out of the trainer command line and
#   # into the YAML (compute.cpu_num_workers). The launcher still keeps its own
#   # copy of the number, because the value is a Wf-326B/R2 memory-safe
#   # requirement (24 physical cores -> OPTIMAL_NUM_WORKERS=8) and a YAML that
#   # silently dropped it would put the trainer back on the manual-loop fallback
#   # path (num_workers=0, no prefetch, no pin_memory) with no error anywhere.
#   # derive_pair_config() FAILS-CLOSED on the comparison, so the YAML and this
#   # pin cannot drift apart.
#   CPU_NUM_WORKERS_PIN=(
#     --cpu-num-workers 8
#   )
#
#   # ---- Per-pair config derivation (AGENTS.md §5.11) --------------------------
#   # The trainer loops over args.metals x args.seeds internally (train_dmpnn_
#   # multitask_wasserstein.py:4352 and :4356), so a per-(metal, seed) PROCESS
#   # needs a config that declares exactly ONE metal and ONE seed. The dedicated
#   # config above describes the whole 2-pair sweep; this derives the single-pair
#   # copy the trainer is launched with.
#   # FAIL-CLOSED: the derivation asserts the narrowing BEFORE it writes, so a
#   # narrowing that silently failed aborts the launch instead of quietly
#   # training the whole cohort.
#   derive_pair_config() {
#     local metal=$1
#     local seed=$2
#     local out=$3
#     ${PY} - "${DMPNN_CONFIG}" "${out}" "${metal}" "${seed}" "${CPU_NUM_WORKERS_PIN[*]}" <<'PYEOF'
#   import sys
#   import yaml
#
#   base, out, metal, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
#   # The pin arrives as "<flag> <value>"; only the value half is compared.
#   pinned_workers = int(sys.argv[5].split()[-1])
#
#   with open(base, encoding="utf-8") as fh:
#       cfg = yaml.safe_load(fh)
#
#   cfg["data"]["metals"] = [metal]
#   cfg["data"]["seeds"] = [seed]
#   cfg["protocol"]["seeds"] = [seed]
#
#   # FAIL-CLOSED assertions: the narrowing IS the point of this file.
#   assert cfg["data"]["metals"] == [metal], cfg["data"]["metals"]
#   assert cfg["data"]["seeds"] == [seed], cfg["data"]["seeds"]
#   assert cfg["protocol"]["seeds"] == [seed], cfg["protocol"]["seeds"]
#
#   # FAIL-CLOSED: the launcher's worker-count pin must equal the YAML's
#   # compute.cpu_num_workers. They are the same knob seen from two places (§5.11
#   # moved the value into the YAML; the pin keeps the launcher's intent visible
#   # and asserts it, so a YAML edit that drops it aborts the launch).
#   assert cfg["compute"]["cpu_num_workers"] == pinned_workers, (
#       cfg["compute"]["cpu_num_workers"],
#       pinned_workers,
#   )
#
#   with open(out, "w", encoding="utf-8") as fh:
#       yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
#   PYEOF
#   }
#
# WHY the pin was removed: its fail-closed property survives the migration —
# the auditor compares the shell record's `--cpu-num-workers 8` against the
# YAML's compute.cpu_num_workers dest through the trainer's own parser, so a
# YAML edit that drops the value still aborts the launch.  Keeping the array
# would have left a live variable with ZERO readers (the trap Wf-A269 removed
# from the G5 launcher), so it goes.
# ---------------------------------------------------------------------------

pair_config_path() {
  local metal=$1
  local seed=$2
  printf '%s%s_seed%s.yaml' "${PAIR_CONFIG_PREFIX}" "${metal}" "${seed}"
}

# FAIL-CLOSED, CPU-only: compares the wrapper's own shell record (COMMON_ARGS)
# against what the trainer will ACTUALLY read, through the trainer's OWN
# parser and the REAL loader — dest by dest, value AND type.
audit_static_configs() {
  local audit_dir
  local rc=0
  audit_dir="$(mktemp -d "${TMPDIR:-/tmp}/wfa269_c1_static_audit.XXXXXX")" || return 1
  "${PY}" scripts/_wf_yaml_static_config_c1.py \
    --trainer "${SCRIPT}" \
    --config-glob "${PAIR_CONFIG_PREFIX}*.yaml" \
    --flags "${COMMON_ARGS[*]} --csv ${DATA_CSV}" \
    --expect "train.epochs=50" \
    --expect "compute.device=cuda:0" \
    --expect "compute.cuda_visible_devices=" \
    --expect "data.csv=${DATA_CSV}" \
    --expect "csv=${DATA_CSV}" \
    --expect-min-count "${PAIR_CONFIG_COUNT}" > "${audit_dir}/cohort.log" 2>&1 || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[static-config] FAIL: static-config audit" >&2
    cat "${audit_dir}/cohort.log" >&2
    rm -rf "${audit_dir}"
    return 1
  fi
  grep -E '^\[static-config\] PASS:' "${audit_dir}/cohort.log" || true
  rm -rf "${audit_dir}"
  return 0
}

# FAIL-CLOSED abort for the static-config audit.
#
# WHY a separate function instead of an inline `audit_static_configs ||
# { ...; exit 1; }`: the --dry-run block has a source-level contract
# (Wf-366-A2, molmetal/tests/test_wf366a2_recover_g6_block_order.py) that the
# FIRST exit code literal after the `--dry-run` test must be 0 — the dry-run
# is the plan-only SUCCESS path.  Keeping the abort out of that block (as a
# function defined before it) preserves both properties at once: a failing
# audit still ABORTS the whole script with exit 1, and the dry-run block's
# own exit stays 0.
abort_if_static_config_audit_fails() {
  if ! audit_static_configs; then
    echo "FAIL: static-config audit (fail-closed)" >&2
    exit 1
  fi
}

run_pair() {
  local metal=$1
  local seed=$2
  local gpu=$3
  local pair_log="${OUT_BASE}/${metal}/seed${seed}/run.log"
  mkdir -p "${OUT_BASE}/${metal}/seed${seed}/ckpts"
  echo "=== $(date '+%H:%M:%S') Wf-310A RETRY START ${metal} seed=${seed} on GPU ${gpu} ===" | tee -a "${LOG}"
  # `setsid --wait`: the retry runs in its own session so it survives the
  # invoking shell (BLK-I1a), and `$rc` below still reports the trainer's
  # status rather than setsid's. Bare `setsid` may fork and return 0.
  # Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the ambient "" no-GPU lock; an explicit ambient value must win; unset -> ${gpu}.
  #
  # 2026-11-03 (AGENTS.md §5.11): the per-pair config carries the cohort
  # (data.metals / data.seeds) and every hyperparameter, so the trainer is
  # invoked with --config / --out-dir / --ckpt-dir only.
  #
  # Wf-A269 §5.11 (2026-11-04): the config is a STATIC file on disk — there
  # is NO launch-time derivation any more.  A missing config aborts the pair
  # (fail-closed) rather than training the wrong cohort.  The fail-closed
  # audit (audit_static_configs) has already proven that both static configs
  # agree with the shell record.
  local pair_config
  pair_config="$(pair_config_path "${metal}" "${seed}")"
  if [[ ! -f "${pair_config}" ]]; then
    echo "=== $(date '+%H:%M:%S') ABORT ${metal} seed=${seed} (static pair config missing: ${pair_config}) ===" | tee -a "${LOG}"
    return 1
  fi
  # The config is copied into the run dir so the run's own artefacts record
  # exactly what was configured.
  install -m 0644 "${pair_config}" "${OUT_BASE}/${metal}/seed${seed}/config.yaml"
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu}}" setsid --wait ${PY} ${SCRIPT} \
  #     --config "${DMPNN_CONFIG}" \
  #     --metals "${metal}" \
  #     --seeds "${seed}" \
  #     --csv "${DATA_CSV}" \
  #     --data-csv "${DATA_CSV}" \
  #     --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
  #     --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" \
  #     --device cuda:0 \
  #     "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${gpu}}" setsid --wait ${PY} ${SCRIPT} \
    --config "${pair_config}" \
    --out-dir "${OUT_BASE}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${metal}/seed${seed}/ckpts" 2>&1 | tee "${pair_log}"
  local rc=${PIPESTATUS[0]}
  echo "=== $(date '+%H:%M:%S') Wf-310A RETRY END ${metal} seed=${seed} (rc=${rc}) on GPU ${gpu} ===" | tee -a "${LOG}"
  return "$rc"
}

# ---- DRY-RUN mode (CPU verification only, no GPU execution) ----
# ADDED 2026-11-04 (Wf-A269 §5.11 migration).  This wrapper had NO dry-run
# mode before, so the fail-closed static-config audit could not be exercised
# without a real GPU launch.  The dry-run writes NOTHING into the tracked
# tree: it does not run wf343c3 (whose --out-dir is a TRACKED path) and it
# does not write to ${LOG}.
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Wf-310A RETRY — DRY-RUN (CPU verification only) ==="
  echo "OUT_BASE: ${OUT_BASE}"
  echo "PY: ${PY}"
  echo "SCRIPT: ${SCRIPT}"
  echo "DATA_CSV: ${DATA_CSV}"
  echo ""
  echo "--- static per-pair config audit (fail-closed, CPU-only) ---"
  abort_if_static_config_audit_fails
  echo ""
  echo "--- 2 (metal, seed) pairs enumerated ---"
  total=0
  for pair in "Ru 2" "Ir 0"; do
    pair_metal="${pair%% *}"
    pair_seed="${pair##* }"
    pair_cfg="$(pair_config_path "${pair_metal}" "${pair_seed}")"
    total=$((total + 1))
    echo "  [STATIC] metal=${pair_metal} seed=${pair_seed} -> ${pair_cfg}"
  done
  echo ""
  echo "  sequential (default): Ru seed=2 on GPU 1, then Ir seed=0 on GPU 1"
  echo "  --parallel:           Ru seed=2 on GPU 1, Ir seed=0 on GPU 0 (2 GPUs)"
  echo ""
  echo "DRY-RUN OK: ${total} (metal, seed) pairs PLANNED, no GPU work executed."
  exit 0
fi

PARALLEL="${1:-}"

# Wf-A269 §5.11 (2026-11-04): fail-closed static-config audit BEFORE any
# work.  It compares the wrapper's own shell record (COMMON_ARGS) against
# what the trainer will actually read, dest by dest, through the trainer's
# own parser and the real loader.  A drifted pair config aborts the launch.
echo "=== static per-pair config audit (fail-closed) $(date) ===" | tee -a "${LOG}"
abort_if_static_config_audit_fails

# ---- Wf-356-A1: wf343c3 data quality check (loader-agnostic — all 5 R20
#      loaders internally: load_binding_db, load_nci60, load_drugood,
#      metalcytotox_v4, platinai_4cell). Does NOT accept --data-csv; only
#      --out-dir. Use launcher-specific out-dir so each launcher's report
#      is preserved. Inserted as the first wave-tool guard before any
#      run_pair() call per Wf-347-B3 retry. ----
echo "=== wf343c3 data quality check START $(date) ===" | tee -a "${LOG}"
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/retry_g5_missing_pairs_data_quality || {
  echo "FAIL: wf343c3_data_quality_check"
  exit 1
}
echo "=== wf343c3 data quality check PASS $(date) ===" | tee -a "${LOG}"

if [[ "${PARALLEL}" == "--parallel" ]]; then
  echo "=== Wf-310A PARALLEL retry on 2 GPUs ===" | tee -a "${LOG}"
  run_pair "Ru" "2" "1" &
  PID_RU=$!
  run_pair "Ir" "0" "0" &
  PID_IR=$!
  wait "${PID_RU}" || echo "Ru seed=2 failed (PID ${PID_RU})"
  wait "${PID_IR}" || echo "Ir seed=0 failed (PID ${PID_IR})"
  echo "=== Wf-310A PARALLEL retry DONE ===" | tee -a "${LOG}"
else
  echo "=== Wf-310A SEQUENTIAL retry ===" | tee -a "${LOG}"
  # Heavy Ru first, then Ir
  run_pair "Ru" "2" "1"
  run_pair "Ir" "0" "1"
  echo "=== Wf-310A SEQUENTIAL retry DONE ===" | tee -a "${LOG}"
fi
