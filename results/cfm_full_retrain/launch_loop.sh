#!/bin/bash
# CFM retrain: 5 seeds × 5 pockets × 50 epochs on PRO 6000
set -e
cd /home/hugo/codes/mol-metal
POCKETS="1h36 3eml 3ny8 4aaw 4rlu"
SEEDS="0 1 2 3 4"
for pocket in $POCKETS; do
    for seed in $SEEDS; do
        echo "=== pocket=$pocket seed=$seed ==="
        CUDA_VISIBLE_DEVICES=0 /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
            molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
            --epochs 50 \
            --hidden-dim 128 \
            --n-layers 3 \
            --pocket-id $pocket \
            --seed $seed \
            --out-dir results/cfm_full_retrain/${pocket}/seed${seed} \
            >> results/cfm_full_retrain/master.log 2>&1
    done
done
echo "=== ALL DONE ==="
