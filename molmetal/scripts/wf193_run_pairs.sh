#!/bin/bash
# Wf-193 — Run 14 (metal, seed) pairs at 50 epochs, hidden=64, depth=2, CPU
# Output to molmetal/reports/wf193_wf131_p3_extended/
#
# =============================================================================
# 2026-11-03 (AGENTS.md §5.11) — the ten hyperparameters moved to YAML
# =============================================================================
# The wrapper used to pass every knob on the command line:
#
#   --metals "${metal}" --seeds "${seed}" --epochs 50 --hidden 64 --depth 2 \
#   --wasserstein-weight 0.05 --rh-oversample 500 --device cpu \
#   --batch-size 64 --lr 1e-3
#
# §5.11 (user directive 2026-10-09, "以后要求能且只能用yaml配置启动训练和推理，
# no cli参数") allows only --config / --out-dir / --ckpt-dir / --help on a
# trainer invocation, and scripts/_yaml_only_gate.py flagged all ten. They now
# live in configs/wf193_run_pairs.yaml.
#
# WHY A BASE + PER-PAIR DERIVATION (and not one config for the whole sweep):
# the trainer loops over args.metals × args.seeds INTERNALLY
# (train_metal_lomo_wasserstein.py:3112-3113). The original wrapper started
# FOURTEEN PROCESSES, one pair each. Declaring the whole cohort in one YAML
# would collapse that into ONE process training twenty folds — a DIFFERENT
# run, not a re-spelling of this one. So the base config carries the shared
# hyperparameters and this script derives one config per pair from it
# (`data.metals: [<metal>]`, `protocol.seeds: [<seed>]`), then launches the
# trainer once per pair with `--config <derived>`. Fourteen processes,
# fourteen pairs, same order, same values.
#
# MEASURED equivalence: the trainer's own _parse_args was run twice — once
# with the OLD command line and once with the NEW `--config <derived>` — and
# the two namespaces were compared dest-by-dest: 0 differences, with the right
# TYPES (`metals` LIST ['Ir'], `seeds` LIST [1], `epochs` int 50, `hidden` int
# 64, `depth` int 2, `wasserstein_weight` float 0.05, `rh_oversample` int
# 500, `device` str 'cpu', `batch_size` int 64, `lr` float 0.001).
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
#   ${PY} molmetal/scripts/train_metal_lomo_wasserstein.py \
#     --metals "${metal}" --seeds "${seed}" \
#     --epochs 50 --hidden 64 --depth 2 \
#     --wasserstein-weight 0.05 --rh-oversample 500 \
#     --out-dir "${OUT_DIR}" \
#     --device cpu \
#     --batch-size 64 --lr 1e-3 \
#     > "${LOG_DIR}/${metal}_seed${seed}.log" 2>&1
#
# AND the per-pair banner, VERBATIM:
#   echo "===== Wf-193 :: fold=${metal} seed=${seed}  epochs=50 hidden=64 depth=2 cpu ====="
#
# The banner is now rendered from the base config's own values (${WF193_BANNER})
# instead of a second copy of the hyperparameters in the shell, so the YAML
# stays the single source of truth for every number the log prints.

set -uo pipefail

PY=/home/hugo/codes/mol-metal/.venv/bin/python
OUT_DIR=/home/hugo/codes/mol-metal/molmetal/reports/wf193_wf131_p3_extended
PROJ_ROOT=/home/hugo/codes/mol-metal
LOG_DIR=${OUT_DIR}/logs

# §5.11: the base config every per-pair config is derived from.
BASE_CFG="${BASE_CFG:-${PROJ_ROOT}/configs/wf193_run_pairs.yaml}"
# Where the derived per-pair configs are written (provenance: the exact file
# each pair trained with sits next to that pair's log).
CFG_DIR="${OUT_DIR}/configs"

mkdir -p "${LOG_DIR}" "${CFG_DIR}"

cd "${PROJ_ROOT}"

# Pair list: Ir×3 + Os×3 + Rh×3 + Re×5 = 14
PAIRS=(
  "Ir 1"
  "Ir 2"
  "Ir 3"
  "Os 1"
  "Os 2"
  "Os 3"
  "Rh 1"
  "Rh 2"
  "Rh 3"
  "Re 0"
  "Re 1"
  "Re 2"
  "Re 3"
  "Re 4"
)

# -----------------------------------------------------------------------------
# Derive one config per (metal, seed) pair from the base config
# -----------------------------------------------------------------------------
# FAILS CLOSED: a derivation that half-succeeds must not fall through to a
# trainer launch with the WRONG cohort, so a non-zero exit aborts the sweep
# (this script has no `set -e`, so the exit code is checked explicitly).
#
# The first stdout line is the banner summary (the base config's own values);
# the remaining lines are the derived configs, one per pair, in PAIRS order.
derive_pair_configs() {
  "${PY}" - "${BASE_CFG}" "${CFG_DIR}" "${PAIRS[@]}" <<'PYEOF'
import pathlib
import sys

import yaml

base_path, cfg_dir = sys.argv[1], sys.argv[2]
pairs = sys.argv[3:]

cfg = yaml.safe_load(open(base_path, encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(f"base config is not a YAML mapping: {base_path}")

protocol = cfg.get("protocol") or {}
model = cfg.get("model") or {}
compute = cfg.get("compute") or {}
print(
    "SUMMARY\tepochs={} hidden={} depth={} {}".format(
        protocol.get("epochs"),
        model.get("hidden"),
        model.get("depth"),
        compute.get("device"),
    )
)

out_dir = pathlib.Path(cfg_dir)
out_dir.mkdir(parents=True, exist_ok=True)

for pair in pairs:
    metal, seed = pair.split()
    seed = int(seed)
    derived = dict(cfg)
    data = dict(derived.get("data") or {})
    data["metals"] = [metal]
    derived["data"] = data
    proto = dict(derived.get("protocol") or {})
    proto["seeds"] = [seed]
    derived["protocol"] = proto
    path = out_dir / f"{metal}_seed{seed}.yaml"
    path.write_text(
        yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"DERIVED\t{metal}\t{seed}\t{path}")
PYEOF
}

DERIVE_OUT=""
if ! DERIVE_OUT="$(derive_pair_configs)"; then
  echo "[wf193] FATAL: could not derive the per-pair configs from ${BASE_CFG}" >&2
  exit 1
fi
# First line = the banner summary; the rest = one DERIVED line per pair.
# $'...' is used for the TAB so the patterns do not depend on a grep/sed
# build that understands a backslash-t escape in a BRE.
WF193_BANNER="$(printf '%s\n' "${DERIVE_OUT}" | sed -n $'s/^SUMMARY\t//p')"
N_DERIVED="$(printf '%s\n' "${DERIVE_OUT}" | grep -c $'^DERIVED\t')"
if [ "${N_DERIVED}" -ne "${#PAIRS[@]}" ]; then
  echo "[wf193] FATAL: derived ${N_DERIVED} pair configs, expected ${#PAIRS[@]}" >&2
  exit 1
fi
printf '%s\n' "${DERIVE_OUT}" | grep $'^DERIVED' | sed $'s/^DERIVED\t/[wf193] /'
echo "[wf193] ${N_DERIVED} pair configs derived from ${BASE_CFG} -> ${CFG_DIR}"

# Start wall-time timer
WF_T0=$(date +%s)

for entry in "${PAIRS[@]}"; do
  read -r metal seed <<< "${entry}"
  T_PAIR=$(date +%s)
  echo "===== Wf-193 :: fold=${metal} seed=${seed}  ${WF193_BANNER} ====="
  ${PY} molmetal/scripts/train_metal_lomo_wasserstein.py \
    --config "${CFG_DIR}/${metal}_seed${seed}.yaml" \
    --out-dir "${OUT_DIR}" \
    > "${LOG_DIR}/${metal}_seed${seed}.log" 2>&1
  EC=$?
  T_PAIR_END=$(date +%s)
  PAIR_MIN=$(( (T_PAIR_END - T_PAIR) / 60 ))
  # 2026-11-03 (pre-existing defect, repaired): the working tree carried
  #   ELAPSED_MIN=$(( (ELAPSED_MIN - T_PAIR) / 60 ))
  # which references ELAPSED_MIN in its own initialisation. Under this
  # script's `set -u` that is an unbound-variable error, so the sweep DIED
  # after the first pair (MEASURED: `bash -c 'set -u; X=$(( (X - 1) / 60 ))'`
  # exits rc=1 with "X: unbound variable"). Restored to the form that
  # computes the cumulative minutes this log line prints.
  # This is the shell's own wall-clock arithmetic — it changes NO value that
  # reaches the trainer (verified dest-by-dest: 0 differences).
  #
  # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
  #   ELAPSED_MIN=$(( (ELAPSED_MIN - T_PAIR) / 60 ))
  ELAPSED_MIN=$(( (T_PAIR_END - WF_T0) / 60 ))
  echo "[wf193] fold=${metal} seed=${seed} ec=${EC} pair_min=${PAIR_MIN} cum_min=${ELAPSED_MIN}"
  if [ ${EC} -ne 0 ]; then
    echo "[wf193] ERROR pair failed; continuing"
  fi
done

WF_T1=$(date +%s)
WF_TOTAL_MIN=$(( (WF_T1 - WF_T0) / 60 ))
echo "[wf193] DONE total_min=${WF_TOTAL_MIN}"
