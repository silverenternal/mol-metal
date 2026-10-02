# Wf-278 Synthesis — Flow-Up Followup Summary

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Workflow:** Wf-278E (aggregation)
**Parents:** Wf-278A (G4 agg fix), Wf-278B (G6 dry-run + PlatinAI 4-cell loader), Wf-278C (Tanimoto integration audit), Wf-278D (paper §3.5 + TODO-207 §9 update)

---

## 1. Per-Agent Summary

### Wf-278A — G4 Aggregation Path-Mismatch Fix
**Status:** SHIPPED. Created `molmetal/scripts/wf278a_g4_aggregate.py` to walk the **nested** `**/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json` tree (vs the legacy flat `wf196_aggregate.py` layout). Restored 14/14 TODO-180 G4 pairs (5 metals × 3 seeds minus 1 lost Ru seed = 14). Per-metal test_auc: Ru 0.7483, Ir 0.6702, Os 0.7025, Rh 0.7227, Re 0.7700. 5-metal aggregate test_auc = **0.7209 ± 0.0458**. No GPU retrain; aggregator only.

### Wf-278B — G6 Dry-Run + PlatinAI 4-Cell Loader Build
**Status:** PARTIAL. G6 dry-run OK — 30 CONFIRMED tuples (--single-task, --wasserstein-weight 0.0) ready for real launch; 30 UNVERIFIED tuples (--no-kendall, --no-dann) need flag verification per `configs/wf124_p3_per_innovation_ablation_g6.yaml` `entry.pre_launch_checklist`. PlatinAI 4-cell CSV built at `.data/molmetal/platinai_4cell.csv` (4078 rows, 4 cells MCF7/A549/A2780/A2780cis, 3/3 tests PASS). **HONEST-NEGATIVE:** pIC50 labels are SYNTHETIC (Rusanov 2025 S4-S5 behind Cloudflare JS challenge), with active rate 94-99.9% per cell line — pipeline test only, NOT comparable to real Rusanov Table S4-S5.

### Wf-278C — cfm_tanimoto_ood_filter Integration Audit
**Status:** PARTIAL (wired, not functional on synthetic). 3 CLI flags added (`--use-tanimoto-ood-filter`, `--ood-filter-threshold 0.4`, `--ood-filter-train-pool`) to `train_cfm_yuelbond_wasserstein.py`; import + invocation path verified. 19/19 existing pytest contracts PASS. **HONEST-NEGATIVE:** filter is no-op on synthetic Pt-click cohort (placeholder SMILES `["N.N.[Cl][Pt][Cl]"]` ⇒ self-match Tanimoto=1.0 ⇒ n_ood=0). Backward-compat documented at lines 980-985; functional mode is `--ood-filter-train-pool <real-train.smi>` (TODO-119 P1.3 follow-up).

### Wf-278D — Paper §3.5 + 207 §9 Update
**Status:** SHIPPED. Updated `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.4 + §9.7 to cite the precise 188 paper targets for the Wf-277D audit fix (188 §5.2 m17 DO NOT claim + 188 §4.5.2 Wf-266 v3 OOD fix). Reverse-cites added at §9.6.2 (m7 Tanimoto diversity → 188 §4.4) + §9.6.3 (m14 real CFM decode_ratio 0.9958 → 188 §4.1). No new §6 created; existing §9 catalog structure preserved per user backward-compat rule.

---

## 2. Aggregate Counts

| Metric | Value |
|---|---|
| **Workflows launched** | 4 (A, B, C, D) |
| **SHIPPED (full)** | 2 (A, D) |
| **PARTIAL (functional with honest gap)** | 2 (B, C) |
| **Total work completed** | G4 14 pairs restored + G6 30 CONFIRMED ready + PlatinAI 4-cell CSV + Tanimoto CLI wiring + 207 §9 update |
| **Total work failed** | 0 hard failures |
| **Total work PARTIAL / not done** | G6 30 UNVERIFIED (flag check) + PlatinAI real labels (Cloudflare blocked) + Tanimoto filter functional on synthetic (placeholder SMILES) |

---

## 3. Critical Honest-Negatives Remaining

1. **PlatinAI pIC50 labels are SYNTHETIC, not real** (Wf-278B). The Rusanov 2025 supplementary is behind a Cloudflare JS challenge that the current loader cannot bypass. All downstream 4-cell D-MPNN training on this CSV is a **pipeline test only** — no real-data validity claim is supported.
2. **CFM Tanimoto OOD filter is a no-op on synthetic Pt-click cohort** (Wf-278C). The placeholder SMILES `["N.N.[Cl][Pt][Cl]"]` ensures identical query/train ⇒ Tanimoto=1.0 ⇒ filter preserves everything. Wiring contract is intact; functional effectiveness requires `--ood-filter-train-pool <real-train.smi>` (TODO-119 P1.3).
3. **G6 30 UNVERIFIED tuples are flag-unchecked** (Wf-278B). `--no-kendall` and `--no-dann` need a `pre_launch_checklist` confirm pass before real launch; otherwise the per-innovation ablation risks silent metric regression.
4. **TODO-180 G4 missing 1 Ru seed** (Wf-278A). Ru appears with 2 seeds (1, 2) instead of 3 (0, 1, 2) — the wave-B Ru seed0 result is not in the nested layout. Aggregate is over n=14 not n=15; the gap is preserved as-is to avoid fabricating a missing seed.

---

## 4. Recommended Next Follow-Up Priorities

1. **Wf-279 (highest): PlatinAI real-label loader** — bypass Cloudflare JS challenge on Rusanov 2025 S4-S5 (browser fallback via Selenium or manual PDF harvest). Without real labels, TODO-119 P1.3 (real CFM decode lift) and TODO-98 (L12 closure OOD) cannot make paper-grade claims. Ship a `platinai_real.csv` and re-run the 4-cell pipeline.
2. **Wf-280: Wire `--ood-filter-train-pool <platinai_real.smi>` into CFM smoke** — once real labels land, the TODO-98 filter becomes functional (not just wired). Re-run `train_cfm_yuelbond_wasserstein.py --use-tanimoto-ood-filter --ood-filter-train-pool platinai_real.smi` and capture `ood_fraction` + `max_train_tanimoto` summary JSON.
3. **Wf-281: G6 flag-check then real launch** — run `configs/wf124_p3_per_innovation_ablation_g6.yaml` `entry.pre_launch_checklist` for the 30 UNVERIFIED tuples (--no-kendall, --no-dann), then launch on RTX 5090. Expect ~3-4h wall for 30×50ep at h=128.
4. **Wf-282: TODO-180 G4 Ru seed0 recovery** — re-run only Ru seed=0 to close the n=14→n=15 gap, re-aggregate via the new `wf278a_g4_aggregate.py`. Cheap (~30min).
5. **Wf-283: TODO-207 §10 cross-ref audit** — extend the Wf-277D-style cite-target audit to the remaining 207 §9.x entries (§9.1 chemprop v2 v3 honest limits, §9.6.1 SA/QED/Vina Dock, §9.Y LOMO fix). Ensure every 207 §9 line points to a precise 188 §x.y paper anchor.
6. **Wf-284: 188 paper-edit gated by TODO-84** — once wet-lab LOI lands (Wf-58 / TODO-97), apply all 207 §9.x "See also 188 §x.y" reverse-cites as forward paper edits in `paper/sections/04_results.tex` + `paper/sections/05_ablation.tex`.

---

## 5. Cross-References

- `molmetal/reports/wf278a_g4_aggregation_fix/fix_report.md`
- `molmetal/reports/wf278b_g6_platinai/loader_report.md`
- `molmetal/reports/wf278c_tanimoto_integration/integration_report.md`
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 (Wf-278D update)
- `molmetal/scripts/wf278a_g4_aggregate.py` (new aggregator)
- `molmetal/scripts/build_platinai_4cell_csv.py` (new loader)
- `.data/molmetal/platinai_4cell.csv` (synthetic 4-cell)
- `results/wf124_p3_remaining_metals_g4/training_summary.json` (restored n=14)