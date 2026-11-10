# Wf-39 Phase 3 — Coupling Adapter IC50 bench

**Dataset**: `platinai_5k`  
**n_samples loaded**: 5 (requested 5)  
**Seeds**: [0]  
**Methods**: ['bridge', 'joint', 'bidirectional']  
**CFM ODE steps**: 5  
**Pocket size**: 8-d  
**Bridge RMSE gate (info)**: ≤ 0.05


## Aggregate (across seeds)

| Method | Bridge RMSE mean ± std | λ det | λ sep | λ norm (mean ± std) | Wall (s) mean ± std | Peak mem (MB) | Speedup |
|---|---|---|---|---|---|---|---|
| bridge | 0.0016 ± 0.0000 | 1.0000 | 0.7416 | 6.7361 ± 0.0000 | 0.000 ± 0.000 | 0.01 | 1.00x (ref) |
| joint | 0.1360 ± 0.0000 | 1.0000 | 0.7416 | 6.7361 ± 0.0000 | 0.027 ± 0.000 | 0.02 | 89.80x |
| bidirectional | 39.5707 ± 0.0000 | 1.0000 | 0.7416 | 6.7361 ± 0.0000 | 0.048 ± 0.000 | 0.03 | 157.61x |

## Per-seed bridge RMSE

| Method | seed 0 |
|---|---|
| bridge | 0.0016 |
| joint | 0.1360 |
| bidirectional | 39.5707 |

## Honest framing

* ``bridge`` measures the per-row RMSE between the PlatinAI 5K 32-d → 64-d bridge output and the ``[λ_32 || λ_32]`` L2-normalised target (the same target the pretraining used).  The R17 ship baseline is 0.0046 (final hold-out RMSE, see :file:`models/coupling_adapter_platinai5k.json`).
* ``joint`` integrates the linear-path CFM ODE for ``n_steps`` iterations from the bridge output; the RMSE is against the same target.  Numerical drift over ODE steps widens the RMSE by construction.
* ``bidirectional``'s ``bridge_rmse`` is the per-row ``round_trip_loss`` — the L2 distance in 32-d λ-feature space between the seed SMILES and the reconstructed AST.  The unit differs from the other two methods.
* Wall-time and memory are machine-dependent; the speedup column is the intended *relative* metric.
* PlatinAI 5K ``ic50_nm`` is populated as ``NaN`` — the bench reports the bridge RMSE, the upstream signal an eventual IC50 model would consume, rather than an IC50 R².


**Run timestamp (UTC):** 2026-11-09T11:59:24.535659+00:00

