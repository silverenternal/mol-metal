#!/bin/bash
# CFM retrain: 5 pockets × 5 seeds × 50 epochs.
#
# §5.11 REPAIR (Wf-A269, 2026-11-04) — the trainer invocation below now passes
# ONLY --config / --out-dir / --ckpt-dir, the three flags AGENTS.md §5.11
# allows a wrapper to carry ("flags that select WHICH run, never WHAT to
# train").  The three model flags that used to ride on the invocation line
# (--epochs / --hidden-dim / --n-layers) live in the YAML
# (protocol.epochs / model.hidden_dim / model.n_layers); the two per-run
# identity flags that used to ride on it (--pocket-id / --seed) live in a
# STATIC per-(pocket, seed) YAML, one file per cell of the sweep, because a
# value that varies per invocation cannot be passed as a flag at all.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6 — the
# wrapper as it stood at git HEAD (before this repair) read:
#
#   #!/bin/bash
#   # CFM retrain: 5 seeds × 5 pockets × 50 epochs on PRO 6000
#   set -e
#   cd /home/hugo/codes/mol-metal
#   POCKETS="1h36 3eml 3ny8 4aaw 4rlu"
#   SEEDS="0 1 2 3 4"
#   for pocket in $POCKETS; do
#       for seed in $SEEDS; do
#           echo "=== pocket=$pocket seed=$seed ==="
#           CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" /home/hugo/codes/mol-metal/.venv/bin/python \
#               molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
#               --epochs 50 \
#               --hidden-dim 128 \
#               --n-layers 3 \
#               --pocket-id $pocket \
#               --seed $seed \
#               --out-dir results/cfm_full_retrain/${pocket}/seed${seed} \
#               >> results/cfm_full_retrain/master.log 2>&1
#       done
#   done
#   echo "=== ALL DONE ==="
#
# WHY (AGENTS.md §5.11): "以后要求能且只能用yaml配置启动训练和推理，no cli参数".
# A knob that lives only in a shell script is a knob nobody audits.  The
# pre-repair wrapper was exactly that: it never passed --config, so the
# trainer ran on ARGPARSE DEFAULTS (batch_size 64, lr 3e-4, synthetic Pt-click
# clouds) while configs/cfm_full_retrain.yaml declared a different run
# (batch_size 16, lr 1e-4, tmQM).  MEASURED from the pre-repair
# results/cfm_full_retrain/master.log: "building cohorts: train=256 val=192"
# and no "[cfm-corpus] dataset=tmqm" line — i.e. the synthetic branch.  The
# repair therefore CHANGES the run's hyper-parameters and its corpus; see the
# delivery report for the exact list.  That change is the POINT of §5.11, not
# a regression, but it is not bit-identical to the pre-repair runs.
#
# The per-(pocket, seed) configs are GENERATED from configs/cfm_full_retrain.yaml
# by scripts/_wfA269_gen_cfm_full_retrain_configs.py (idempotent; re-run it
# after editing the base config).
set -e
cd /home/hugo/codes/mol-metal
# AGENTS.md §5.1: dry-run stays the default way to inspect a launch.
# `--dry-run` prints the resolved plan (one line per run) and exits without
# touching the GPU; it also exercises the missing-config check.
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument: $arg (only --dry-run is accepted)" >&2; exit 2 ;;
    esac
done
POCKETS="1h36 3eml 3ny8 4aaw 4rlu"
SEEDS="0 1 2 3 4"
for pocket in $POCKETS; do
    for seed in $SEEDS; do
        cfg="configs/cfm_full_retrain_${pocket}_seed${seed}.yaml"
        out="results/cfm_full_retrain/${pocket}/seed${seed}"
        ckpt="${out}/checkpoints"
        # §5.5 filesystem-truth: a missing config must stop the run, not
        # silently train on argparse defaults.
        if [ ! -f "$cfg" ]; then
            echo "MISSING CONFIG: $cfg (pocket=$pocket seed=$seed)" >&2
            exit 1
        fi
        # The first echo line is byte-identical to the pre-repair rendering
        # (an orchestrator parses it); the second is new provenance.
        echo "=== pocket=$pocket seed=$seed ==="
        echo "--- config=$cfg out-dir=$out ckpt-dir=$ckpt ---"
        if [ "$DRY_RUN" -eq 1 ]; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" /home/hugo/codes/mol-metal/.venv/bin/python \
            molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
            --config "$cfg" \
            --out-dir "$out" \
            --ckpt-dir "$ckpt" \
            >> results/cfm_full_retrain/master.log 2>&1
    done
done
if [ "$DRY_RUN" -eq 1 ]; then
    echo "=== DRY RUN OK: 25 run(s) planned, 0 executed ==="
else
    echo "=== ALL DONE ==="
fi
