#!/bin/bash
# Wf-124 P3 G4 — 14 PENDING (metal, seed) pairs launch wrapper
# Date: 2026-09-28 (post-Wf-258 user OK)
# Owner: Hugo Lee
# Per TODO-180 G4 + configs/wf124_p3_remaining_metals_g4.yaml
# Per kill_then_adapt directive: split into 2 waves (each ≤1 hr on RTX 5090)
# Skips Ru seed=0 (already MEASURED per Wf-124 P3 §4)
#
# Wall-time estimate:
#   - Wave A: 8 pairs × ~7.7 min = ~62 min on RTX 5090
#   - Wave B: 6 pairs × ~7.7 min = ~46 min on RTX 5090
#   - Total: ~108 min ≈ 1.8 hr
#
# Per Wf-328-W2: --cpu-num-workers 8 (Wf-326B OPTIMAL_NUM_WORKERS on 24-core host)
# — replaces Wf-326B-reverted --cpu-num-workers 0 override.
#
# Per Wf-328-W5: --w2-gradient-mode sort (Wf-326A MEASURED 2-3× speedup on
# RTX PRO 6000 Blackwell sm_120; replaces the legacy --w2-gradient-mode
# sliced override that was preserved bit-exact for in-flight retrains).
#
# Per Wf-328-W6: --num-workers 8 (Wf-326B build_optimized_loader() factory;
# takes precedence over --cpu-num-workers; wires prefetch_factor=8 +
# pin_memory=True + persistent_workers=True + spawn context).
#
# Per Wf-335-R1+R3: --no-cuda-graphs (safety lever; trainer default
# torch_compile_mode = max-autotune-no-cudagraphs per Wf-330-B4 RuntimeError
# fix on Ru seed=2 + Ir seed=0).
#
# Wf-337-G2: this launcher mirrors launch_wf124_p3_g6.sh v2 config for
# parity with the G5 50-epoch full sweep + G6 per-innovation ablation.

set -e
cd /home/hugo/codes/mol-metal

OUT_BASE="results/wf124_p3_remaining_metals_g4"
LOG="${OUT_BASE}/master.log"
mkdir -p "${OUT_BASE}"

PY="/home/hugo/codes/CadButEaas/.venv-vlm/bin/python"
SCRIPT="molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
GPU="cuda:0"  # RTX 5090 GPU-1 (smaller GPU = higher % util on small models)
DATA_CSV="/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv"  # Wf-261 strict scaffold split (0 leak) + both column naming conventions (smiles/metal lowercase for split table, SMILES_Ligands/Metal/IC50_Dark_value for cytotox)

COMMON_ARGS=(
  --epochs 30
  --batch-size 256
  --lr 1e-3
  --alpha 0.5
  --wasserstein-weight 0.05
  --use-l2-guard
  --ablation-config mtl_baseline
  --w2-gradient-mode sort
  --n-mc 256
  --hidden 300
  --depth 3
  --device "${GPU}"
  --cuda-visible-devices 1
  --cpu-num-workers 8
  --num-workers 8
  --data-csv "${DATA_CSV}"
  # Wf-335-R3 — CUDA Graphs hard-disable safety lever. Default ON in
  # trainer (per Wf-330-B4 MEASURED RuntimeError on Ru seed=2 + Ir seed=0
  # under --torch-compile-mode reduce-overhead); pass explicitly so paper
  # §6 audits can verify the lever is wired.
  --no-cuda-graphs
)

run_pair() {
  local metal=$1
  local seed=$2
  local wave=$3
  local pair_log="${OUT_BASE}/${wave}/${metal}_seed${seed}.log"
  mkdir -p "${OUT_BASE}/${wave}"
  echo "=== $(date '+%H:%M:%S') START ${metal} seed=${seed} wave=${wave} ===" | tee -a "${LOG}"
  CUDA_VISIBLE_DEVICES=1 ${PY} ${SCRIPT} \
    --metals "${metal}" \
    --seeds "${seed}" \
    --csv "${DATA_CSV}" \
    --data-csv "${DATA_CSV}" \
    --out-dir "${OUT_BASE}/${wave}/${metal}/seed${seed}" \
    --ckpt-dir "${OUT_BASE}/${wave}/${metal}/seed${seed}/ckpts" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${pair_log}"
  echo "=== $(date '+%H:%M:%S') END ${metal} seed=${seed} wave=${wave} ===" | tee -a "${LOG}"
}

# Wave-tool guards (Wf-283C) — wire pre-flight + per-wave checklist + budget tracker
# Ref: Wf-279D §5 cross-cutting bug + Wf-280D recommendation #1 (~45 LOC fix)
# Wf-356-A1: wf343c3 data quality check inserted between pre-flight and
# per-wave checklist — script is loader-agnostic (covers all 5 R20 loaders
# internally: load_binding_db, load_nci60, load_drugood, metalcytotox_v4,
# platinai_4cell). Does NOT accept --data-csv; only --out-dir.
# Use launcher-specific out-dir so each launcher's report is preserved.
${PY} molmetal/scripts/pre_flight_check.py --pipeline all || { echo "FAIL: pre_flight_check"; exit 1; }
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/wf124_p3_g4_data_quality \
  || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G4" || { echo "FAIL: per_wave_checklist"; exit 1; }
${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }

echo "=== Wave A START $(date) ===" | tee -a "${LOG}"
echo "Wave A: Ru s1/s2 + Ir s0/s1/s2 + Os s0/s1/s2 (8 pairs, ~62 min)" | tee -a "${LOG}"

# Wave A: 8 pairs (skip Ru s0 already measured)
run_pair Ru 1 A
run_pair Ru 2 A
run_pair Ir 0 A
run_pair Ir 1 A
run_pair Ir 2 A
run_pair Os 0 A
run_pair Os 1 A
run_pair Os 2 A

echo "=== Wave A DONE $(date) ===" | tee -a "${LOG}"
echo "=== Wave B START $(date) ===" | tee -a "${LOG}"
echo "Wave B: Rh s0/s1/s2 + Re s0/s1/s2 (6 pairs, ~46 min)" | tee -a "${LOG}"

# Wave B: 6 pairs (Rh + Re)
run_pair Rh 0 B
run_pair Rh 1 B
run_pair Rh 2 B
run_pair Re 0 B
run_pair Re 1 B
run_pair Re 2 B

echo "=== Wave B DONE $(date) ===" | tee -a "${LOG}"
echo "=== ALL TODO-180 G4 PAIRS DONE $(date) ===" | tee -a "${LOG}"

# Aggregation
echo "=== Aggregation START $(date) ===" | tee -a "${LOG}"
${PY} -c "
import json
from pathlib import Path
runs = []
for wave_dir in Path('${OUT_BASE}').iterdir():
    if not wave_dir.is_dir():
        continue
    for metal_dir in wave_dir.iterdir():
        if not metal_dir.is_dir():
            continue
        for seed_dir in metal_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            for p in seed_dir.glob('per_seed_metrics_seed*.json'):
                runs.append({'metal': metal_dir.name, 'seed': seed_dir.name, 'metrics_path': str(p)})
out = {'todo_180_g4_total_pairs': len(runs), 'runs': runs}
Path('${OUT_BASE}/training_summary.json').write_text(json.dumps(out, indent=2))
print(f'aggregated {len(runs)} pairs')
" | tee -a "${LOG}"
echo "=== Aggregation DONE $(date) ===" | tee -a "${LOG}"