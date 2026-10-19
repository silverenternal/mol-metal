# Decision curve: model vs global_best at two coverage levels

Generated 2026-10-19T11:45:22Z from `metrics/by_metric/*.json`.

Net benefit uses hit = regret <= 0.2091 and cost ratio p_t/(1-p_t).

| Metric | Mean regret | Top-1 | NB(0.5) | p (Holm) | Source | Tag |
|---|---|---|---|---|---|---|
| doi / cov=0.5 / model | 0.1115 | 0.5540 | 0.6714 | 1.0000 | msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/model/mean_regret; msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/model/top1_accuracy; msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/model/net_benefit/0.5; msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/model_vs_global_best/p_holm | MEASURED |
| doi / cov=0.5 / global_best | 0.1067 | 0.5822 | 0.6808 | - | msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/global_best/mean_regret; msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/global_best/top1_accuracy; msrs_decision_curve.json:/protocols/doi/curve[coverage=0.5]/global_best/net_benefit/0.5 | MEASURED |
| doi / cov=1.0 / model | 0.1168 | 0.5600 | 0.6094 | - | msrs_decision_curve.json:/protocols/doi/full_coverage/model/mean_regret; msrs_decision_curve.json:/protocols/doi/full_coverage/model/top1_accuracy; msrs_decision_curve.json:/protocols/doi/full_coverage_net_benefit/model/0.5 | MEASURED |
| doi / cov=1.0 / global_best | 0.1195 | 0.5765 | 0.5953 | - | msrs_decision_curve.json:/protocols/doi/full_coverage/global_best/mean_regret; msrs_decision_curve.json:/protocols/doi/full_coverage/global_best/top1_accuracy; msrs_decision_curve.json:/protocols/doi/full_coverage_net_benefit/global_best/0.5 | MEASURED |
| ligand / cov=0.5 / model | 0.1181 | 0.5587 | 0.6338 | 1.0000 | msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/model/mean_regret; msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/model/top1_accuracy; msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/model/net_benefit/0.5; msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/model_vs_global_best/p_holm | MEASURED |
| ligand / cov=0.5 / global_best | 0.1394 | 0.5728 | 0.5869 | - | msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/global_best/mean_regret; msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/global_best/top1_accuracy; msrs_decision_curve.json:/protocols/ligand/curve[coverage=0.5]/global_best/net_benefit/0.5 | MEASURED |
| ligand / cov=1.0 / model | 0.1191 | 0.5647 | 0.6047 | - | msrs_decision_curve.json:/protocols/ligand/full_coverage/model/mean_regret; msrs_decision_curve.json:/protocols/ligand/full_coverage/model/top1_accuracy; msrs_decision_curve.json:/protocols/ligand/full_coverage_net_benefit/model/0.5 | MEASURED |
| ligand / cov=1.0 / global_best | 0.1238 | 0.5671 | 0.5812 | - | msrs_decision_curve.json:/protocols/ligand/full_coverage/global_best/mean_regret; msrs_decision_curve.json:/protocols/ligand/full_coverage/global_best/top1_accuracy; msrs_decision_curve.json:/protocols/ligand/full_coverage_net_benefit/global_best/0.5 | MEASURED |
