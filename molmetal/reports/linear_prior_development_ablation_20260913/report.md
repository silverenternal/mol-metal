# Linear descriptor prior: development fit and frozen off/on ablation

This is a reagent-only algorithm check, not CrossDocked or physical binding evaluation. No test reference structures or docking scores were used as training inputs or labels.

Development: 32 real CuAAC product observations; training seed 20260913. Reagents present in the evaluation standard-12 library were excluded from development.

Frozen evaluation: seeds 42, 0, 1234; 8 simulations, depth 1; the same initialization is paired off/on within each seed. LIPINSKI and PROTEASE_GENERIC gates are unchanged.

| seed | prior | PUCT calls | NFE reductions | candidates | generated | mean reward | best reward |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | off | 0 | 12 | 3 | 3 | 1.815257 | 1.868874 |
| 42 | on | 4 | 12 | 3 | 3 | 1.815257 | 1.868874 |
| 0 | off | 0 | 12 | 3 | 3 | 1.815257 | 1.868874 |
| 0 | on | 4 | 12 | 3 | 3 | 1.815257 | 1.868874 |
| 1234 | off | 0 | 12 | 1 | 1 | 1.887084 | 1.887084 |
| 1234 | on | 4 | 12 | 1 | 1 | 1.887084 | 1.887084 |

The reward is the existing SA/QED descriptor-based search objective with type/binding bonuses; it is not experimental affinity. This backend is a Ridge linear descriptor model, not PySR symbolic discovery. All frozen state comparisons were exact and the on arms invoked the actual PUCT prior dispatcher.

Paired mean-reward deltas (on − off): [0.0, 0.0, 0.0]. Zero or negative differences do not establish improvement; this small check does not support physical-quality claims.
Generated molecule overlap with development products: 0.
Total elapsed time: 10.03 seconds.
