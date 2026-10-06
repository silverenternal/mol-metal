# Round 8 (W2-REAL, w=0.05, 50ep) — short synthesis
MEASURED: LOMO mean test AUC = 0.5833 +/- 0.1103 (n=25, 5 metals x 5 seeds, gate >= 0.70) -> BELOW_GATE.
Paired vs Round 7 (W2-OFF, 30ep): +0.0238, se=0.0148, t=+1.61, 15/25 wins -> NOT significant (p~0.12).
Per-fold delta (R7->R8): Re +0.0980 (0.610->0.708) | Ir +0.0243 | Rh +0.0060 | Ru -0.0048 | Os -0.0043.
HONEST-NEGATIVE: Rh LOMO 0.4106 +/- 0.0918 (below chance) remains the blocker; Re lift is 1 fold x 5 seeds, not a validated trend.
W2-REAL now reaches the gradient (Round 7 Wf-E1 measured W2 at 0.0 effective) but 2 confounds remain: epoch count 30->50 changed simultaneously.
Source: molmetal/reports/dmpnn_clean5v5_w2_real/{metrics.json,aggregate.json,train.log}; wall 53 min on cuda:0.
