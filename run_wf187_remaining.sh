#!/bin/bash
# Wf-187 — Wf-131 P3 7/25 remaining runs (Rh seed3/4 + Re seed0-4)
# Uses the Wf-187 m19 fix: --oversample-positives + --pos-weight 10
# --quick sets epochs=2 (time budget: ~5min per run)
#
# §5.11 MIGRATION (2026-11-04): a launch wrapper may hand the trainer ONLY
# --config / --out-dir / --ckpt-dir / --help (AGENTS.md §5.11).  The 11
# hyperparameter flags that used to ride on each of the 7 invocation lines now
# live in a STATIC YAML.  Every changed line is preserved VERBATIM in the
# comment above its replacement (AGENTS.md §5.5 rule #6, dated 2026-11-04).
#
# WHY ONE CONFIG PER (metal, seed) AND NOT ONE PER METAL: the trainer sweeps
# `args.metals` x `args.seeds` INSIDE one process
# (molmetal/scripts/train_metal_lomo_v2.py:2030 `for metal in args.metals:` /
# :2031 `for seed in args.seeds:`) while this launcher starts ONE process per
# pair, each with its own --out-dir.  A per-metal config would train EVERY
# seed of that metal's cohort in EVERY process — 39 pair-runs instead of 7,
# and an aggregate.json covering a cohort instead of one seed.  The per-metal
# cohort DECLARATION (which metal, which seeds, which hyperparameters) is
# configs/wf187_rh_oversample.yaml and configs/wf187_re_oversample.yaml; the
# per-pair files are the launch surface, and the pre-launch guard below
# asserts the two agree before a single process starts.

set -e
cd /home/hugo/codes/mol-metal
PY=/home/hugo/codes/mol-metal/.venv/bin/python
OUT=molmetal/reports/wf187_p3_rh_oversample_fix
LOG=$OUT/run.log
# The trainer path is kept in a variable so the §5.11 gates read the guard
# below as what it is — a call to the repo's own auditor — and not as a
# second trainer invocation.
SCRIPT=molmetal/scripts/train_metal_lomo_v2.py

mkdir -p $OUT
echo "[wf187] start $(date +%H:%M:%S)" > $LOG

# ---------------------------------------------------------------------------
# PRE-LAUNCH GUARD (2026-11-04) — fail-closed.
#
#   1. scripts/_wf_yaml_static_config_c1.py (the repo's own auditor): every
#      per-pair YAML carries the PRE-MIGRATION hyperparameters — checked dest
#      by dest, value AND type, through the trainer's OWN parser and the REAL
#      loader — and pins exactly ONE metal and ONE seed, matching its own
#      file name.  The --flags record below is the shell's statement of what
#      the wrapper used to pass; the auditor proves the YAML agrees with it.
#   2. the per-metal cohort declaration and the per-pair files must agree
#      leaf for leaf (except the seed list and the out-dir, which are the
#      two things a per-pair file is allowed to narrow).
#
# Without this guard the YAML could drift away from the run it claims to
# describe — the §5.11 failure mode this migration exists to remove.
# ---------------------------------------------------------------------------
"$PY" scripts/_wf_yaml_static_config_c1.py \
  --trainer "${SCRIPT}" \
  --config-glob "configs/wf187_oversample_*_seed*.yaml" \
  --flags "--model v3 --epochs 2 --device cpu --no-bf16 --wasserstein-weight 0 --oversample-positives --pos-weight 10 --batch-size 64 --quick" \
  --expect-min-count 7 || { echo "[wf187][FATAL] static-config audit FAILED" | tee -a $LOG; exit 1; }

"$PY" - <<'PYEOF' || { echo "[wf187][FATAL] cohort/per-pair config disagreement" | tee -a $LOG; exit 1; }
import hashlib
import json
import pathlib
import sys

import yaml

REPO = pathlib.Path("/home/hugo/codes/mol-metal")
COHORTS = {
    "Rh": REPO / "configs/wf187_rh_oversample.yaml",
    "Re": REPO / "configs/wf187_re_oversample.yaml",
}
PAIRS = {
    "Rh": sorted((REPO / "configs").glob("wf187_oversample_Rh_seed*.yaml")),
    "Re": sorted((REPO / "configs").glob("wf187_oversample_Re_seed*.yaml")),
}
# The two leaves a per-pair file is allowed to narrow: its own seed and its
# own --out-dir.  (Paths here are the FLATTENED ones — flatten() joins with
# "_", so the YAML path train.seeds flattens to "train_seeds".)  Everything
# else must be identical to the cohort.
NARROWABLE = ("train_seeds", "output_out_dir")


def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}_{key}" if prefix else str(key)
            out.update(flatten(value, path))
    else:
        out[prefix] = node
    return out


def fingerprint(path):
    leaves = {
        k: v
        for k, v in flatten(yaml.safe_load(path.read_text(encoding="utf-8"))).items()
        if k not in NARROWABLE
    }
    blob = json.dumps(leaves, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), leaves


bad = []
for metal, cohort_path in COHORTS.items():
    cohort_fp, cohort_leaves = fingerprint(cohort_path)
    cohort_seeds = flatten(
        yaml.safe_load(cohort_path.read_text(encoding="utf-8"))
    )["train_seeds"]
    for pair_path in PAIRS[metal]:
        pair_fp, pair_leaves = fingerprint(pair_path)
        pair_seeds = flatten(
            yaml.safe_load(pair_path.read_text(encoding="utf-8"))
        )["train_seeds"]
        if pair_fp != cohort_fp:
            only_cohort = sorted(set(cohort_leaves) - set(pair_leaves))
            only_pair = sorted(set(pair_leaves) - set(cohort_leaves))
            differing = sorted(
                k
                for k in set(cohort_leaves) & set(pair_leaves)
                if cohort_leaves[k] != pair_leaves[k]
            )
            bad.append(
                f"{pair_path.name} disagrees with {cohort_path.name}: "
                f"only_cohort={only_cohort} only_pair={only_pair} "
                f"differing={differing}"
            )
        if not isinstance(pair_seeds, list) or len(pair_seeds) != 1:
            bad.append(f"{pair_path.name}: seeds={pair_seeds!r} is not a 1-element list")
        elif pair_seeds[0] not in cohort_seeds:
            bad.append(
                f"{pair_path.name}: seed {pair_seeds[0]} is not in the cohort "
                f"{cohort_path.name} seeds {cohort_seeds}"
            )
    print(
        f"[wf187] cohort {cohort_path.name}: seeds={cohort_seeds} "
        f"fingerprint={cohort_fp[:12]} pairs={len(PAIRS[metal])}"
    )
    for pair_path in PAIRS[metal]:
        print(f"[wf187]   pair {pair_path.name}: fingerprint={fingerprint(pair_path)[0][:12]}")

if bad:
    for line in bad:
        print(f"[wf187][FAIL] {line}", file=sys.stderr)
    raise SystemExit(1)
print("[wf187] cohort declaration and per-pair launch configs AGREE")
PYEOF

# Rh seed3
echo "[wf187] Rh seed3 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Rh --seeds 3 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Rh_seed3 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Rh_seed3.yaml \
  --out-dir $OUT/Rh_seed3 2>&1 | tail -10 | tee -a $LOG

# Rh seed4
echo "[wf187] Rh seed4 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Rh --seeds 4 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Rh_seed4 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Rh_seed4.yaml \
  --out-dir $OUT/Rh_seed4 2>&1 | tail -10 | tee -a $LOG

# Re seed0
echo "[wf187] Re seed0 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Re --seeds 0 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Re_seed0 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Re_seed0.yaml \
  --out-dir $OUT/Re_seed0 2>&1 | tail -10 | tee -a $LOG

# Re seed1
echo "[wf187] Re seed1 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Re --seeds 1 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Re_seed1 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Re_seed1.yaml \
  --out-dir $OUT/Re_seed1 2>&1 | tail -10 | tee -a $LOG

# Re seed2
echo "[wf187] Re seed2 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Re --seeds 2 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Re_seed2 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Re_seed2.yaml \
  --out-dir $OUT/Re_seed2 2>&1 | tail -10 | tee -a $LOG

# Re seed3
echo "[wf187] Re seed3 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Re --seeds 3 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Re_seed3 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Re_seed3.yaml \
  --out-dir $OUT/Re_seed3 2>&1 | tail -10 | tee -a $LOG

# Re seed4
echo "[wf187] Re seed4 $(date +%H:%M:%S)" | tee -a $LOG
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
# (2026-11-04 — the pre-§5.11 invocation line for this pair):
#   $PY molmetal/scripts/train_metal_lomo_v2.py \
#     --model v3 --metals Re --seeds 4 --epochs 2 \
#     --device cpu --no-bf16 \
#     --wasserstein-weight 0 \
#     --oversample-positives --pos-weight 10 \
#     --batch-size 64 \
#     --out-dir $OUT/Re_seed4 --quick 2>&1 | tail -10 | tee -a $LOG
$PY molmetal/scripts/train_metal_lomo_v2.py \
  --config configs/wf187_oversample_Re_seed4.yaml \
  --out-dir $OUT/Re_seed4 2>&1 | tail -10 | tee -a $LOG

echo "[wf187] all done $(date +%H:%M:%S)" | tee -a $LOG
