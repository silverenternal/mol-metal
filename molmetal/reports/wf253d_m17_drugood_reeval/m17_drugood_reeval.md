# m17 DrugOOD v3 Forward-Pass Reeval (Wf-253-D / TODO-130 G5)

Closes the Wf-252 `CHECKPOINT_LOADED_BUT_FORWARD_PASS_DEFERRED` honest-negative by wiring the actual forward pass on DrugOOD scaffold ood_test split.

## Aggregate

- Forward passes completed: **8** / 8 (metal, seed) pairs
- Aggregate ROC-AUC: **0.4544260351267818 ± 0.0327**
- Wf-185 MEASURED baseline: **0.6707** ± 0.0042 (n=5 seeds, chemprop v2 retrain)

> **Verdict: BELOW_RANDOM (aggregate 0.4544)**

## Per-(metal, seed) breakdown

| metal | seed | ROC-AUC | AUPR | n_samples | wall_clock_s | eval_status |
|---|---:|---:|---:|---:|---:|---|
| Ru | 0 | 0.4574 | 0.8467 | 2000 | 9.2 | MEASURED |
| Ru | 1 | 0.4710 | 0.8679 | 2000 | 10.0 | MEASURED |
| Ru | 2 | 0.5395 | 0.8867 | 2000 | 9.9 | MEASURED |
| Ru | 3 | 0.4940 | 0.8808 | 2000 | 3.9 | MEASURED |
| Ir | 0 | 0.4266 | 0.8463 | 2000 | 1.2 | MEASURED |
| Ir | 1 | 0.4618 | 0.8716 | 2000 | 1.2 | MEASURED |
| Ir | 2 | 0.4961 | 0.8725 | 2000 | 7.4 | MEASURED |
| Ir | 3 | — | — | — | — | DEFERRED_NO_CHECKPOINT |
| Os | 0 | 0.4113 | 0.8372 | 2000 | 1.2 | MEASURED |
| Os | 1 | — | — | — | — | DEFERRED_NO_CHECKPOINT |
| Os | 2 | — | — | — | — | DEFERRED_NO_CHECKPOINT |
| Os | 3 | — | — | — | — | DEFERRED_NO_CHECKPOINT |

## Per-metal aggregate

| metal | mean ROC-AUC | std | n_seeds_measured | PROJECTED range |
|---|---:|---:|---:|---|
| Ru | 0.4905 | 0.0312 | 4 | [0.65, 0.70] |
| Ir | 0.4615 | 0.0283 | 3 | [0.55, 0.65] |
| Os | 0.4113 | 0.0000 | 1 | [0.55, 0.65] |

## Honest framing

- v3 weights were NOT re-trained on DrugOOD — no leakage.
- Per-metal head used: Ru→Ru head, Ir→Ir head, Os→Os head.
- Wf-185 is the sister chemprop v2 retrain (NOT v3 backbone); the comparison target is honest MEASURED-vs-MEASURED.
- PROJECTED AUCs from the Wf-252 design §4 are kept as a sanity-check column; v3 measured AUCs may differ.
- Per-metal coverage is uneven: Ru has 4 seeds, Ir has 3, Os has 1.
- Forward pass runs CPU-only; no GPU sync; no checkpoint re-training.