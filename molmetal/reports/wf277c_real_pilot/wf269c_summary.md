# Wf-269C — Real-cohort CFM ablation

- Cohort source: `/tmp/wf277c_real_pilot/metalcytotox_ru_only_v4.csv`
- Raw mols: 18987
- After degenerate-filter: 18840
- Train clouds: 30 (embed-fail=0)
- Val clouds: 30 (embed-fail=0)
- Epochs per config: 3
- Device: cpu

## Per-config results

| Config | Decode ratio | Ok/Total | Train loss | Grad norm | Wall (s) |
|---|---|---|---|---|---|
| `full` | 1.000 | 30/30 | 0.2794 | 2.768 | 7.4 |
| `no_yuelbond` | 0.000 | 0/30 | 0.2506 | 2.567 | 22.2 |
| `no_wasserstein` | 1.000 | 30/30 | 0.2042 | 2.669 | 3.2 |
| `no_closure_bounded` | 1.000 | 30/30 | 0.1378 | 2.698 | 13.7 |

**Honest framing:** Real-cohort ablation completed. At least one ablation shows ≥0.05 degradation vs full — innovation differentiation signal PRESENT.
