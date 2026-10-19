# Delta-sign under DOI holdout: three routes, one gate (0.65)

Generated from `metrics/by_metric/*.json`; see `manifest.json` for the timestamp and source hashes.

Differencing and fusion have no DOI-clustered interval for the sign in their artifacts; direct regression does.

| Metric | Sign@|d|>=0.2091 | 95% CI | MAE(d) | Pairs | Source | Tag |
|---|---|---|---|---|---|---|
| differencing two level predictions | 0.5826 | [0.4670, 0.6805] | 0.5115 | 218 | msrs_seed_ensemble.json:/protocols/doi/ensemble/delta/0.2091/sign_accuracy; msrs_seed_ensemble.json:/protocols/doi/ensemble/delta/0.2091/bootstrap/sign_accuracy/ci95; msrs_seed_ensemble.json:/protocols/doi/ensemble/delta/0.2091/mae_delta; msrs_seed_ensemble.json:/protocols/doi/ensemble/delta/0.2091/n_pairs | MEASURED |
| fusion (MACCS tree + GNN) | 0.5505 | - | 0.5054 | 218 | msrs_feature_fusion.json:/protocols/doi/delta_sign/sign_accuracy; msrs_feature_fusion.json:/protocols/doi/delta_sign/mae_delta; msrs_feature_fusion.json:/protocols/doi/delta_sign/n_pairs | MEASURED |
| direct delta regression | 0.5183 | [0.3902, 0.6539] | 0.5559 | 218 | msrs_delta_direct.json:/gated/sign_accuracy; msrs_delta_direct.json:/gated/bootstrap/sign_accuracy/ci95; msrs_delta_direct.json:/gated/mae_delta; msrs_delta_direct.json:/gated/n_pairs | MEASURED |
