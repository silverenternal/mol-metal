# Wf-146 end-to-end demo smoke report

## Setup
- metal: Ru (idx=0)
- seed=0, epochs=3, batch=256, n_samples=1000 (subset of metal_cytotox.pt 3.4GB)
- model: V4MultiTaskDMPNN h=128 n_layers=3 n_metals=5
- loss: UncertaintyWeightedMultiTaskLoss(num_tasks=5)
- autocast: torch.amp.autocast("cuda", dtype=bfloat16)
- pool: VinaReplayPool(storage_dir=.../vina_replay) with placeholder vina scores (no real Vina call)
- surrogate: VinaSurrogate.fit on placeholder pool
- resume: cold-start → train → save → re-init → load → epoch=3 verified

## Results
- total wall_sec=28.4
- ckpt_size_kb=8440.2
- pool_size=12
- surrogate_trained=False
- resume_epoch=3 best_val_mae=-0.0374
- 12 steps completed: load / filter / model / loss / pool / train×3 / save / pool.save+surrogate / resume / artifacts

## Honest framing
- VinaReplayPool entries use placeholder vina_score = sigmoid(model_pred); no real Vina docking was performed.
- CounterfactualReplay.mine() requires a baseline checkpoint; this demo emits an empty mined set with a note rather than fabricate counterfactuals.
- forward() is the model API; there is no forward_cached() method. SMILES go through D-MPNN encoder each step.
