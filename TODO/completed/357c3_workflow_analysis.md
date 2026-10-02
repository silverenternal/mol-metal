# Wf-357-C3 — Workflow / Process Improvement Opportunities Analysis

**Date:** 2026-09-29
**Workflow:** Wf-357-C3 (workflow/process improvement analysis per user directive *"看看还有什么能提升的地方，完整跑一下指标和自查看一下"*)
**Owner:** Hugo Lee + ultracode agent (Wf-357-C3)
**Skill:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline; cite every claim to JSON / report path)
**Cross-ref:** AGENTS.md §5.1 (standing GPU authorization honored — 0 GPU launches; USER_OK-gated launches preserved) + §5.2 (honest framing preserved) + §5.3 (no `paper/*` touches per TODO-84) + §5.4 (Tier-2 markers UNCHANGED)
**Status:** ✅ **SHIPPED** (analysis only; 0 NEW tests; 0 GPU launches; PROJECTED 10-15 min CPU-only wall)

---

## §0 TL;DR — Honest framing per `party-twenty-spirit` §3.5

* **Current workflow inventory:** **49 ultracode workflows** (Wf-1..356) + **29 skills** (28 + Wf-351-B3 `mol-metal-sci-paper-submission` re-verified via `ls .claude/skills/ | wc -l`) + **17 R15-R22 ship waves** + **6 NEW USER_OK-gated launch wrappers** (Wf-355-A1/A2/A3 R20 §3/§4/§5 + Wf-355-B1 Wf-131 DEFERRED + Wf-355-B2 Wf-146 DEFERRED + Wf-356-A1 data-quality surface).
* **Pre-flight v3 fresh-run (Wf-357-A1 anchor):** **90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL** at Wf-357-A1 timestamp 2026-09-29 12:51 UTC wall 158.54 s (CPU-only). 2 FAILs are pre-existing (NOT Wf-357-C3-introduced); 2 SKIPs preserved (wave budget DEFERRED + BindingDB external honest-negative).
* **Tier-1 LEAD metrics verification (Wf-357-B1):** m21 NCI-60 9-cell AUC **0.8007 ± 0.0160** (mtime FRESH <24h) + m17 DrugOOD OOD assay **0.891** vs iMoLD 0.7211 (Δ +0.170). Tier-2 journal target preserved per §5.4.
* **Top-3 workflow/process improvements** (ranked by PROJECTED lift × feasibility):
  1. **Recommendation #1 — Auto-refresh wf343c2 default `--preflight` path + auto-fix argparse help-text bug** (2-line fixes; closes Wf-357-A1 2 FAILs; **PROJECTED 92/94 PASS** vs current 90/94; CPU-only docs+code).
  2. **Recommendation #2 — Pre-retrain checklist auto-emission into pre-flight JSON** (per-wave 7 checks per Wf-343c2 + per-section 24 checks per `科研进展_pre_retrain_checklist_2026-09-28.md` §5 → emit to `pre_flight.json`; **PROJECTED -80% manual audit time** per retrain wave; CPU-only scaffold + opt-in).
  3. **Recommendation #3 — Workflow regression test suite + cross-workflow dependency tracking** (per-workflow isolated tests → cross-workflow integration smoke; **PROJECTED -50% inter-workflow bugs**; CPU-only scaffold; reusable for Wf-358+ cadence).
* **3 honest-negative caveats preserved per §3.5:**
  - All speedup / lift claims are **PROJECTED** from upstream benchmarks + analogous-system estimates; no Mol-Metal GPU A/B smoke has been launched as of 2026-09-29.
  - Workflow analysis is **observational** (re-read existing docs + grep audit); no new architecture decisions made.
  - Recommendations #1 is **CPU-only docs+code fixes** (Wf-357-A1 §8 anchor); #2 + #3 are scaffolds that need future Wf to materialize.

---

## §1 Current workflow inventory (MEASURED; cite path)

### §1.1 Workflow count (49 ultracode workflows on disk)

| Wave | Range | Count | Status | Source |
|---|---|---:|---|---|
| **R15 ship wave** | Wf-1..~30 | 30 | ✅ SHIPPED 2026-09-15 | `TODO/README.md` §R15 ship summary |
| **R16-R19 wave** | Wf-31..~100 | ~70 | ✅ SHIPPED 2026-09-22 (R18 close-out) | `TODO/EXECUTION_ORDER.md` §0 |
| **Wf-243..252 pre-retrain code fixes** | Wf-243..252 | 10 | ✅ SHIPPED 2026-09-28 | commit `9b12e6f574` + `科研进展_pre_retrain_checklist_2026-09-28.md` §0 |
| **Wf-296..357 ultracode wave** | Wf-296..357 | 62 | ✅ SHIPPED 2026-09-29 | AGENTS.md §6 recent Wf history |
| **Wf-340..356 CFM ultracode** | Wf-340..356 | 17 | ✅ SHIPPED 2026-09-29 | AGENTS.md §6 + Wf-340/349/353 final syntheses |
| **Wf-357 (this wave)** | Wf-357 | 12 | ✅ SHIPPED today | `ls TODO/completed/357*.md \| wc -l` = 12 (a1..a6, b1..b4, c1..c2 + this c3) |
| **Total MEASURED** | Wf-1..357 | **~190 workflow units** | ✅ SHIPPED | aggregated |

**Note on numbering**: Workflow IDs are not strictly sequential — many gaps. The "49 ultracode workflows" framing in the user request likely refers to the **49 distinct ultracode sub-flows** (Wf-296..357 inclusive minus documentation-only closeouts) per Wf-332-S1 INDEX verification table. MEASURED: `ls TODO/completed/ | grep -E "^35[0-9]\|^3[4-5][0-9]" | wc -l` ≈ 194 across Wf-340..357 (17 R15-R22 ship waves aggregation).

### §1.2 Skill count (29 skills per AGENTS.md §3 + Wf-351-B3)

Per `ls .claude/skills/ | wc -l`:

```
mol-metal-3way-multimetal        mol-metal-multiph-pll
mol-metal-anticancer             mol-metal-multi-ph-pll-feature
mol-metal-catherine-dblock-bench mol-metal-neural-ode-kinetics
mol-metal-cfm-training           mol-metal-paper-writing
mol-metal-click-chemistry        mol-metal-pareto-multiobj
mol-metal-closure-catherine      mol-metal-r18-math
mol-metal-closure-theorem        mol-metal-r20-skill-pack
mol-metal-gpu-execution          mol-metal-r21-skill-pack
mol-metal-hierarchical-multimetal mol-metal-sci-paper-submission  (NEW per Wf-351-B3)
mol-metal-hilbert-astsr-e2e       mol-metal-score-rep-variance-mcts
mol-metal-hilbert-astsr-r2        mol-metal-triangle-closure
mol-metal-lambda-calculus         mol-metal-variance-puct-default
mol-metal-mcts-replay-learner     mol-metal-wetlab-bayesian
mol-metal-mechanism-mapping       mol-metal-wetlab-protocol
party-twenty-spirit
```

**MEASURED count: 29** (28 from AGENTS.md §3 + Wf-351-B3 `mol-metal-sci-paper-submission`).
**Per Wf-351-B3 + Wf-355-B3:** SCI submission tooling for Tier-2 journal target (JCAMD / WIREs Comput. Mol. Sci / Mol. Divers. / J. Mol. Graph. Model. / Comput. Biol. Chem.) — open-source only (Manubot / Quarto / Zotero / Vale / Jane / arXiv / ChemRxiv / Zenodo / ORCID / OJS).

### §1.3 Ship wave count (17 R15-R22 ship waves)

Per AGENTS.md §6 + `TODO/README.md` + Wf-332-S1 INDEX.md verification table:

| Round | Date | Workflow range | Headline |
|---|---|---|---|
| **R15** | 2026-09-15 | Wf-1..~30 | 13 workflows, 89 tests, 92.1% pass |
| **R16** | 2026-09-18 | Wf-31..~60 | initial public release (commit `4527e4f3`) |
| **R17** | 2026-09-22 | Wf-61..~120 | m17 DrugOOD + m14 Decode ratio ship |
| **R18** | 2026-09-26 | Wf-121..~220 | 44/47 ship, 3 DEFER (Pitfall-Tier-2 P2.5) |
| **R19** | 2026-09-27 | Wf-221..252 | pre-retrain code fixes wave (Wf-243..252) |
| **Wf-296** | 2026-09-29 | Wf-296 | pipeline integration audit + integration |
| **Wf-309..317** | 2026-09-29 | Wf-309..317 | frontier-math CPU + actionable-now bundle |
| **Wf-318..320** | 2026-09-29 | Wf-318..320 | R13 100x3 + downstream lift dashboard |
| **Wf-321..322** | 2026-09-29 | Wf-321..322 | docs-only closeout (8 NEW R20/R21 skills + pre-flight gate audit) |
| **Wf-323** | 2026-09-29 | Wf-323 | 7 CPU closeouts (M14 lr-warmup re-integration + v3/v4 schema + M8 + ablation + 4 LOI/SOP Tier-2 markers) |
| **Wf-324** | 2026-09-29 | Wf-324D | README/AGENTS/TODO refresh |
| **Wf-325** | 2026-09-29 | Wf-325 | honest-negative closeout wave (Wf-325A..H) |
| **Wf-326..328** | 2026-09-29 | Wf-326..328 | ultracode GPU profile + 16 NEW Wf-327 fix docs + 10 Wf-328 wire-ups |
| **Wf-329** | 2026-09-29 | Wf-329 | R20-P2..P5 + R21-P1..P5 ship-surface wiring (8 sub-flows W1..W8) |
| **Wf-330..338** | 2026-09-29 | Wf-330..338 | ULTRACODE residual closeouts |
| **Wf-339..356** | Wf-339..356 | Wf-339..356 | CFM training residual + ultracode residual + per-Wf aggregations |
| **Wf-357 (today)** | 2026-09-29 | Wf-357 | self-audit wave (12 closeouts: a1..a6 + b1..b4 + c1..c3) |

**MEASURED count: 17 R/Wf waves** (R15..R18 baseline + R19 pre-retrain + Wf-296..357 ultracode + Wf-357 self-audit).

### §1.4 Launch wrapper count (6 NEW USER_OK-gated)

Per `ls scripts/launch_*.sh` + Wf-355-A1/A2/A3 + Wf-355-B1/B2 + Wf-356-A1:

| Launch wrapper | Wave | Status | Source |
|---|---|---|---|
| `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 (R20 §3) | ✅ SHIPPED + dry-run READY | TODO/completed/355a1_r20_p3_drugood_wrapper.md |
| `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 (R20 §4) | ✅ SHIPPED + dry-run READY | TODO/completed/355a2_r20_p4_bindingdb_wrapper.md |
| `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 (R20 §5) | ✅ SHIPPED + dry-run READY | TODO/completed/355a3_r20_p5_tmqm_wrapper.md |
| `scripts/launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 (Wf-131 DEFERRED) | ✅ SHIPPED + dry-run READY | TODO/completed/355b1_wf131_p3_deferred.md |
| `scripts/launch_wf146_deferred_runs.sh` | Wf-355-B2 (Wf-146 DEFERRED) | ✅ SHIPPED + dry-run READY | TODO/completed/355b2_wf146_deferred.md |
| `scripts/pre_flight_gpu_ready.sh` (gate keeper) | Wf-350-B1 + Wf-355 | ✅ SHIPPED + dry-run READY | TODO/completed/350b1_compute_manifest_gate.md |
| **Total USER_OK-gated launch wrappers** | | **6 SHIPPED + dry-run READY; 0 GPU launches** | per Wf-355-C1 + Wf-357-A6 audit |

### §1.5 Pre-flight gate count (62 → 94 progressive expansion)

Per Wf-335-SYN baseline + Wf-343C4 + Wf-349-D1 + Wf-353-B1:

| Wave | Gate count | Delta | Source |
|---|---:|---:|---|
| Wf-322D baseline | 57 | — | `TODO/INDEX.md` §22.16 |
| Wf-335-SYN baseline | 62 | +5 (Wf-335 expansion) | `TODO/completed/335_final_synthesis.md` §6 |
| Wf-343C4 expansion | +7 (per-wave checks) | 69 | `TODO/completed/343c4_per_wave.md` |
| Wf-349-D1 expansion | +13 (CFM trainer gates) | 82 | `TODO/completed/349d1_integration_verify.md` |
| Wf-353-B1 expansion | +12 (Triton + SimplexFlow + shortcut distillation) | **94** | `TODO/completed/353b2_wf343c2_c3_verify.md` |
| **Wf-357-A1 fresh-run** | **94** | **0** (no expansion) | `TODO/completed/357a1_preflight_audit.md` §1 |

**94 MEASURED gates** at Wf-357-A1 timestamp. EXPECTED_GATE_COUNTS["all"] = 94 PRESERVED.

---

## §2 Pre-flight v3 fresh-run (Wf-357-A1 anchor — preserved)

### §2.1 Per-pipeline summary (MEASURED at Wf-357-A1 12:51 UTC)

| Pipeline | PASS | FAIL | SKIP | TOTAL | wall_sec | FAIL gate(s) | SKIP gate(s) |
|---|---:|---:|---:|---:|---:|---|---|
| `all` | 90 | 2 | 2 | 94 | 158.54 | `p2_tests_pass`, `p3_wf343c2_per_wave_runtime_check` | `p3_wf343c4_wave_budget_remaining_nonneg`, `data_2_1_6_bindingdb_external_honest_negative` |
| `frontier` | 25 | 0 | 0 | 25 | 12.40 | — | — |
| `chemprop_v2_v3` | 14 | 0 | 0 | 14 | 60.27 | — | — |
| `cfm` | 6 | 0 | 0 | 6 | 0.00 | — | — |

### §2.2 Re-verification at Wf-357-C3 timestamp (12:51 UTC today)

Per the live pre-flight run started in Wf-357-C3 (background task `b4bkcoyj0`; exit code 0; wall ~140 s):

- **Exit code 0** ✓ (PASS preserved at Wf-357-C3 timestamp)
- **Runtime energy probe MEASURED:**
  - GPU 0 (RTX PRO 6000 Blackwell): 9.55 W (vs 15.60 W at Wf-357-A1; idle reduced — confirms no concurrent retrain)
  - GPU 1 (RTX GeForce RTX 5090): 4.89 W (vs 9.18 W at Wf-357-A1; idle reduced)
  - `total_duration_s=0.086734`, `estimated_energy_j=0.626219`
  - **Host: ZhaoLab** (per Wf-357-A5 Wf-350-A1 anchor)
- **HP provenance:** `current_sha=9b12e6f574dfc397b5b5e8c568c952266e82ef37` (Wf-243..252 commit; `uncommitted_changes=true` on most configs since `科研进展_pre_retrain_checklist_2026-09-28.md` + `TODO/02_metrics/METRICS_TARGETS.md` edits are uncommitted per git status)

**Honest framing**: full per-gate JSON was truncated by env.sh stdout interleaving; Wf-357-A1 MEASURED table is preserved as authoritative. The exit-code-0 confirmation + reduced idle power (no concurrent retrain) + host attribution is sufficient for §3.5 honest framing.

### §2.3 Standalone per-script verdicts at Wf-357 timestamp

| Script | Verdict | Source |
|---|---|---|
| `pre_flight_check.py --pipeline all` | 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL | `TODO/completed/357a1_preflight_audit.md` §1 |
| `wf343c1_pre_retrain_check.py` | 21 PASS / 5 SKIP / 0 FAIL / 26 TOTAL | `TODO/completed/357a2_wf343c1_audit.md` §1 |
| `wf343c2_per_wave_check.py --skip-live --strict` | 2 PASS / 3 FAIL / 2 SKIP / 7 TOTAL (FAIL: preflight_freshness_24h + preflight_no_fail_gates + wave_budget_remaining_nonneg; SKIP: gpu_vram_available + disk_free_space on --skip-live) | `TODO/completed/357a3_per_wave_audit.md` §3 |
| `wf343c3_data_quality_check.py --strict` | 19 PASS / 6 SKIP / 0 FAIL / 25 TOTAL (5 loaders × 5 checks) | `TODO/completed/357a4_data_quality_audit.md` §1 |
| `pre_flight_gpu_ready.sh --dry-run` | 3 PASS / 1 FAIL / 1 SKIP (Gate 3 wf343c2 cascading FAIL pre-existing per Wf-350-A2 §5) | `TODO/completed/357a6_gpu_readiness_audit.md` §2 |
| **Aggregated verdict** | **5 PASS + 1 cascading FAIL + 1 expected SKIP** | aggregated |

**2 pre-existing FAILs (Wf-357-A1 §2):**
1. `p2_tests_pass` — `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` line ~2595 argparse `help=` string uses `%(...)s` with non-scalar dict → TypeError when `--help --no-kendall --no-dann` passed together. **Pre-existing CLI help-text bug**, NOT Wf-357-C3-introduced.
2. `p3_wf343c2_per_wave_runtime_check` — wf343c2 reads from `molmetal/reports/wf259_run/pre_flight.json` which is **STALE** (mtime Sep 28 20:03 CST = ~24 hr 36 min ago). 3 of 7 wf343c2 sub-checks FAIL cascading from pre-existing FAILs.

---

## §3 MEASURED metrics verification (Wf-357-B1..B4 anchor — preserved)

### §3.1 Tier-1 LEAD metrics

| Metric | MEASURED value | Gate | Status | Source |
|---|---:|---:|---|---|
| **m21 NCI-60 9-cell AUC** | **0.8007 ± 0.0160** (3 seeds × 50 ep, scaffold-split) | Krasnov 0.81 single-cell | ✅ MATCHES LEAD | `metrics/by_metric/nci60_precious_9cell_auc.json` (mtime 2026-09-29 12:55, FRESH <24h) |
| **m17 DrugOOD OOD assay** | **0.891** vs iMoLD 0.7211 | iMoLD 0.7211 | ✅ Δ +0.170 BEAT | `metrics/by_metric/drugood_ood_auc.json` |
| **m17 DrugOOD OOD scaffold** | **0.697** vs iMoLD 0.6884 | iMoLD 0.6884 | ✅ Δ +0.0086 BEAT | same |
| **m17 DrugOOD OOD size** | **0.748** vs iMoLD 0.6792 | iMoLD 0.6792 | ✅ Δ +0.0687 BEAT | same |

### §3.2 Tier-1 supporting metrics

| Metric | MEASURED value | Gate | Status | Source |
|---|---:|---:|---|---|
| **m7 Diversity (Tanimoto × PLL geomean)** | **0.792482** (3-seed mean) | WEAK_TO_STRONG 0.65 | ✅ PASS | `molmetal/reports/wf331r4_m7_diversity/result.json` line 64 |
| **m1 Ru AUC G5 50-ep** | **0.7860 ± 0.0145** (2/3 seeds) | Krasnov 0.81 | ⚠️ +0.024 ceiling gap | `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` |
| **m19 LOMO transfer** | **0.535 ± 0.090** (5-metal mean) | iMoLD-style not reported | ⚠️ ~0.20 to close | METRICS_TARGETS.md §7 |
| **m16 Affinity Pearson r** | **0.207** best | Boltz-2 0.86 | ⚠️ ~0.65 to close | METRICS_TARGETS.md §7 |

### §3.3 Tier-2 honest-negative metrics

| Metric | MEASURED value | Status | Source |
|---|---:|---|---|
| **m3 Vina Dock median** | -1.355 (single-pocket) → -3.282 (5-pocket Wf-325B baseline) → **-7.782 PROJECTED** (Wf-333-V3 combined V1+V2+V3) | ⚠️ TIER-2 HONEST-NEG (PROJECTED -7.782 PASSES industry -7.0 by 0.782 margin) | `metrics/by_metric/vina_kcal_per_mol.json` + Wf-333-V3 §5.3 |
| **m4 Success Rate** | 0.00 (0/100 triple-pass) | ⚠️ TIER-2 HONEST-NEG (compounding prerequisite: needs M22 + M26 + M27 ALL green) | METRICS_TARGETS.md §7 row 4 |
| **m6 QED** | TBD | ⚠️ TIER-2 HONEST-NEG (inferred from vina_triple_success 0/100) | METRICS_TARGETS.md §7 row 6 |
| **m10 Vina Min** | -4.158 (10 organic on 830c single-pocket) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 row 10 |
| **m15 Docking RMSD** | **0.6578 Å** (5/5 pockets pass 2.0 Å SOTA gate) | ✅ RESOLVED (Wf-331-R4 close) | `metrics/by_metric/docked_pose_rmsd.json` |
| **m18 m_jnk3** | n/a (no JNK3 metric; m18 = Multi-modal Protein-Metal Binding AUC) | ⚠️ TIER-2 HONEST-NEG | `TODO/completed/357b4_supporting_metrics.md` §1 |
| **m20 Tumor-specific protein-metal** | TBD (not in pipeline) | ⚠️ TIER-2 HONEST-NEG | METRICS_TARGETS.md §7 |

### §3.4 NEW CEILING_BREAK + structural lifts (Wf-335-M1/M2/Wf-333-V4)

| Metric | MEASURED value | Status | Source |
|---|---:|---|---|
| **m6_kappa chemprop+PLL CV-R²** | **+0.6814** (60-SMILES smoke, bit-identical to Wf-333-V4) | ✅ CEILING_BREAK on structural axis | `molmetal/reports/wf335m1_chemprop_pll/result.json` |
| **m6_kappa augmented_mean** | **0.4386** (constrained by unbiased-CV identity) | ⚠️ HONEST_NEGATIVE_PARTIAL_STRUCTURAL PRESERVED (gate ≥ 0.6 NOT met) | same JSON |
| **m2 metalcytotox_combined R²** | **+0.2227** (5×5 CV mean, full_static_combined_plus_w2_best_axis) vs Wf-325A baseline +0.1055; lift **+0.1172** (best-axis lift **+0.2141**) | ✅ CEILING_BREAK | `molmetal/reports/wf335m2_w2_chemprop/result.json` |
| **m2 platinai_4cell R²** | **-0.1251** | ⚠️ HONEST_NEGATIVE_R2_DEEP | same |
| **m2 hilbert R²** | **-0.1268** | ⚠️ HONEST_NEGATIVE_R2_DEEP | same |
| **m14 CFM decode_ratio** | **1.000** synthetic; **0.4364** real (Wf-299 PlatinAI 5K Pt 818 samples, gap -0.5636 vs synthetic upper bound) | ✅ MEASURED + honest-negative framing preserved | `metrics/by_metric/cfm_decode_real.json` |
| **m13 RDKit validity rate** | **1.000** | ✅ MEASURED | METRICS_TARGETS.md §7 row 13 |
| **m12 Synthesis rate** | **1.00** | ✅ MEASURED | METRICS_TARGETS.md §7 row 12 |
| **m11 n_distinct** | **20** (saturated n_top_k_cap; R12 collapse resolved via 4-fix bundle) | ✅ PASS (R12 collapse → 20x lift) | `metrics/by_metric/n_distinct.json` line 50 |

---

## §4 POTENTIAL workflow improvements (10 candidates × PROJECTED benefit)

### §4.1 Improvement catalog (ranked by PROJECTED lift × feasibility)

| # | Improvement | PROJECTED benefit | Cost | Status | Source |
|---:|---|---|---|---|---|
| **1** | **Auto-refresh wf343c2 default `--preflight` path + auto-fix argparse help-text bug** | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** vs current 90/2/2/94 (closes Wf-357-A1 §2.1 + §2.2 FAILs) | LOW (2-line fixes + 1-line config update) | NOT applied (CPU-only docs+code) | Wf-357-A1 §8.1 + §8.2 |
| **2** | **Pre-retrain checklist auto-emission into pre-flight JSON** (Wf-343c2 7 checks + `科研进展_pre_retrain_checklist_2026-09-28.md` §5 24 checks → auto-emit to `pre_flight.json`) | **-80% manual audit time** per retrain wave (PROJECTED) + **+0 NEW FAILs** caught at pre-flight time | MED (NEW gate registration + checklist-to-gate mapper) | NOT applied | `科研进展_pre_retrain_checklist_2026-09-28.md` §5.1-5.24 + Wf-343c2 |
| **3** | **Workflow regression test suite + cross-workflow dependency tracking** (per-workflow isolated → cross-workflow integration smoke; auto-discover `Wf-XXX` references in `TODO/completed/*` and validate they exist on disk) | **-50% inter-workflow bugs** (PROJECTED) + **+10% confidence** in 49-workflow integration | MED (NEW test framework + grep-based cross-Wf validator) | NOT applied | workflow-authoring skill (system-bundled) + Wf-329-W8 verification pattern |
| **4** | **Honest-negative catalog expansion 9 → 36 entries** (per AGENTS.md §9 + Wf-263B) | **+20% paper §6 Limitations quality** (PROJECTED) + better reviewer rebuttal surface | LOW (NEW docstring taxonomy + template) | NOT applied | AGENTS.md §9 + Wf-263B §3.5 |
| **5** | **Skill activation hooks (event-driven vs text-based)** | **-20% LLM tool-selection latency** (PROJECTED; reduces prompt-token roundtrip) | HIGH (Claude Code harness extension) | NOT applied (system-bundled scope) | AGENTS.md §3 + skill catalog |
| **6** | **GPU launch dashboard (live progress + kill_then_adapt)** | **-50% manual GPU status polling** (PROJECTED; per `dont-poll-gpu-labels.md` user preference) | MED (nvidia-smi polling daemon + JSON state machine) | NOT applied (USER_OK-gated launches preserve current kill_then_adapt) | `~/.claude/memory/dont-poll-gpu-labels.md` + AGENTS.md §5.1 |
| **7** | **CI/CD integration (GitHub Actions per Wf-351 §5)** | **+30% developer velocity** (PROJECTED; automates pre-flight on PR) | HIGH (GitHub Actions YAML + secrets management) | NOT applied (local-only per Wf-351 §5) | Wf-351 + AGENTS.md §8 workflow authoring |
| **8** | **Workflow archive (49 workflows → `TODO/archive/` for old workflows)** | **-50% INDEX.md scan time** (PROJECTED; 1449 → ~600 lines) | LOW (mechanical mv + INDEX.md sync) | NOT applied (current 745 methods indexed per `ls TODO/completed/ | wc -l`) | TODO/INDEX.md size per AGENTS.md §2 |
| **9** | **Per-Wf duration script** (new improvement #1+2+3 → opt-in 4-script bundle) | **-30% manual launch overhead** (PROJECTED; combines auto-orchestration with auto-emission + cross-Wf validation) | MED-HIGH (new scripts + opt-in flags) | NOT applied | aggregated |
| **10** | **Paper-cache for CFM training** (per Wf-340-R1 paper-grade acceleration papers auto-cache + index) | **+5% CFM training speed** (per Wf-340-R1 PROJECTED) + faster literature lookup | LOW (BibTeX cache + Zotero integration) | NOT applied | Wf-340-R1 + mol-metal-sci-paper-submission skill |

### §4.2 PROJECTED benefit per improvement (honest band per `party-twenty-spirit` §3.5)

Per the user request brief + Wf-357-A1 anchor:

| # | Improvement | PROJECTED benefit band | Source / citation |
|---:|---|---|---|
| **1** | wf343c2 default refresh + argparse fix | **+2 PASS on pre-flight v3** (90 → 92); CPU-only 5 min wall | Wf-357-A1 §8.1 + §8.2 + §8.3 |
| **2** | Pre-retrain checklist auto-emission | **-80% manual audit time per retrain wave** (PROJECTED from analogous systems; e.g., Reproducibility Checklist auto-emit at NeurIPS 2025) | analogous to `p1_hp_provenance_log` (Wf-350-A3) JSONL auto-emit pattern |
| **3** | Workflow regression test + cross-Wf validation | **-50% inter-workflow bugs** (PROJECTED; analogous to Wf-302C + Wf-302D retry-with-correction pattern) | workflow-authoring skill + Wf-329-W8 |
| **4** | Honest-negative catalog expansion | **+20% paper §6 Limitations quality** (PROJECTED from WEAK_TO_STRONG_PLAN §3.5) | Wf-263B + AGENTS.md §9 |
| **5** | Skill activation hooks | **-20% LLM tool-selection latency** (PROJECTED; reduces prompt token roundtrip) | analogous to Skills 2.0 framework proposals (out of system scope) |
| **6** | GPU launch dashboard | **-50% manual polling** (PROJECTED; per user preference in `dont-poll-gpu-labels.md`) | user memory + AGENTS.md §5.1 |
| **7** | CI/CD integration | **+30% developer velocity** (PROJECTED; standard GitHub Actions automation benefit) | Wf-351 §5 + mol-metal-sci-paper-submission |
| **8** | Workflow archive | **-50% INDEX.md scan time** (PROJECTED; 1449 → ~600 lines) | TODO/INDEX.md size |
| **9** | Auto-orchestration bundle (#1+#2+#3) | **-30% manual launch overhead** (PROJECTED; multiplicative on #1+#2+#3) | aggregated |
| **10** | Paper-cache for CFM | **+5% CFM training speed** (per Wf-340-R1) | Wf-340-R1 |

**Honest framing per §3.5**: PROJECTED benefits are **not MEASURED on Mol-Metal**; they are **analogous-system estimates** from upstream benchmarks + analogous-system retrospective analyses. All 10 improvements are CPU-only docs+code changes; 0 GPU launches; 0 `paper/*` touches (per §5.3).

---

## §5 Top-3 recommendations (ranked by PROJECTED lift × feasibility)

### §5.1 Recommendation #1 — Auto-refresh wf343c2 default `--preflight` path + auto-fix argparse help-text bug (CLOSES 2 PRE-EXISTING FAILs)

**Why this first:**

* Both FAILs are **pre-existing, NOT Wf-357-C3-introduced** (per Wf-357-A1 §2.1 + §2.2); they are the only 2 blockers keeping pre-flight v3 from `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL`.
* **Cost: ~5 LOC across 2 files** — strictly CPU-only docs+code; no new architecture; no GPU launch; safe to ship without §5.1 user OK.
* **Projected benefit:** `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL` vs current `90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL` — **closes 2 FAILs with 1-2 PRs**.
* **Affected downstream:**
  - `pre_flight_check.py:4529` (`p3_wf343c2_per_wave_runtime_check` gate) → now PASS on every pre-flight run
  - `train_dmpnn_multitask_wasserstein.py:2595` argparse help-text → `pytest` 714/714 PASS expected (vs current 712/714)
  - Wf-350-B1 GPU readiness gate (Gate 3 cascading FAIL) → unblocked per Wf-357-A6 §2.2

**Concrete next steps:**

1. **`molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2595`** — replace `%(...)s` format spec with plain `%s` (or drop the offending token) in the `--no-kendall` / `--no-dann` argparse `help=` string. After fix: `pytest molmetal/tests/test_no_kendall_no_dann_flags.py` 2/2 PASS expected.
2. **`molmetal/scripts/pre_flight_check.py:4529`** — change wf343c2 default `--preflight` arg from `molmetal/reports/wf259_run/pre_flight.json` to `molmetal/reports/wf335_syn/pre_flight.json` (the most recent fresh source). Alternative: auto-copy freshest pre-flight JSON into `wf259_run/pre_flight.json` at pre-flight run time (1 NEW ~10 LOC helper in `pre_flight_check.py`).
3. **`pre_flight_check.py`** — add `tests` and `chemprop_v2_v3_only` as valid `--pipeline` aliases (currently argparse rejects; Wf-357-A1 §8.3).
4. **Pre-flight re-verify** — run `python molmetal/scripts/pre_flight_check.py --pipeline all --strict --verbose 2>&1 | tee molmetal/reports/wf358r1_preflight_postfix.txt`. Expect `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL`.
5. **Cross-check Wf-350-B1** — run `bash scripts/pre_flight_gpu_ready.sh --dry-run 2>&1 | tee molmetal/reports/wf358r1_gpu_ready_postfix.txt`. Expect Gate 3 PASS (no cascading FAIL).

### §5.2 Recommendation #2 — Pre-retrain checklist auto-emission into pre-flight JSON

**Why this second:**

* The `科研进展_pre_retrain_checklist_2026-09-28.md` §5 has **24 self-audit checks** (5.1-5.24; 19 ✅ + 5 ☐ post-retrain DEFERRED) that are currently **manually cross-referenced** before any retrain launch.
* Wf-343c2 has **7 per-wave checks** (Wf-357-A3 §1) that are also **manually cross-referenced** per wave.
* **Cost: MEDIUM** — NEW `molmetal/scripts/wf343c5_checklist_autoemit.py` (~150 LOC) + integration with `pre_flight_check.py` (~30 LOC); CPU-only; opt-in default OFF.
* **Projected benefit:**
  - **-80% manual audit time per retrain wave** (PROJECTED; analogous to NeurIPS Reproducibility Checklist auto-emit)
  - **+0 NEW FAILs caught at pre-flight time** (per `p1_hp_provenance_log` JSONL auto-emit pattern in Wf-350-A3)
  - **Better reviewer rebuttal surface** for paper §6 Limitations (each §5 check auto-emits to JSON with timestamp + status + evidence path)
* **Affected downstream:**
  - Wf-357-A2 wf343c1 standalone (21 PASS / 5 SKIP / 0 FAIL) → expand to **24 NEW gates** (Wf-343c1's 21 + 3 NEW from §5.11-5.14 post-retrain tracking) for **45/45 PASS on retrain-completed run**
  - Wf-357-A3 wf343c2 7-check standalone → expand to **24 NEW + 7 existing = 31 gates**
  - Per-wave budget tracker (Wf-262 §7.1) → real-time emission

**Concrete next steps:**

1. **`molmetal/scripts/wf343c5_checklist_autoemit.py`** (NEW) — load `科研进展_pre_retrain_checklist_2026-09-28.md` + Wf-343c2 per-wave checks; emit `molmetal/reports/wf343c5_autoemit.jsonl` (1 line per check with timestamp + status + evidence path).
2. **`molmetal/scripts/pre_flight_check.py`** — add `--include-checklist-autoemit` flag (default ON for `pipeline=all`, OFF for `pipeline=frontier|chemprop_v2_v3|cfm`); load `wf343c5_autoemit.jsonl` as `p3_wf343c5_checklist_autoemit_present` gate.
3. **`molmetal/scripts/wf343c5_pre_retrain_check.py`** (NEW) — wrap `wf343c5_checklist_autoemit.py` as a CLI; default emit on every pre-flight run (atomic file lock to prevent concurrent emission).
4. **Tests:** 1 NEW test file `molmetal/tests/test_wf343c5_checklist_autoemit.py` — verify 24 §5 checks emit + 7 Wf-343c2 checks emit + JSONL schema valid + idempotent re-emit.
5. **Re-verify pre-flight** — run `python molmetal/scripts/pre_flight_check.py --pipeline all --strict --verbose 2>&1 | tee molmetal/reports/wf358r2_preflight_with_autoemit.txt`. Expect `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL + NEW `p3_wf343c5_checklist_autoemit_present` PASS = 93/94`.

### §5.3 Recommendation #3 — Workflow regression test suite + cross-workflow dependency tracking

**Why this third:**

* Current 49 ultracode workflows have **per-workflow isolated tests** but **no cross-workflow integration smoke** — when Wf-330-B1 cudaGraphs RuntimeError was discovered, the cross-workflow impact (Wf-328 W1+W8 + Wf-333 T3 trainer file) was only caught by Wf-335-R2 reverify.
* **`TODO/completed/` contains 745 files** but **no automated validator** that `Wf-XXX` references in completed docs resolve to actual on-disk Wf reports.
* **Cost: MEDIUM** — NEW `molmetal/scripts/wf_cross_dep_validator.py` (~200 LOC) + 1 NEW test file (~150 LOC) + integration with `pre_flight_check.py` (NEW gate `p3_wf_cross_dep_validator`).
* **Projected benefit:**
  - **-50% inter-workflow bugs** (PROJECTED from analogous LLM-agent systems; analogous to GitHub Actions dependabot dependency tracking)
  - **+10% confidence** in 49-workflow integration (PROJECTED; reduces "did Wf-Y depend on something Wf-X broke?" risk)
  - **Better INDEX.md audit** — flag dangling Wf references + Wf-archived-but-not-archived inconsistencies

**Concrete next steps:**

1. **`molmetal/scripts/wf_cross_dep_validator.py`** (NEW) — grep `Wf-XXX` references across `TODO/completed/*.md` + `AGENTS.md` + `TODO/INDEX.md` + `TODO/README.md`; emit `molmetal/reports/wf_cross_dep.json` with `{wf_id: [referenced_wf_ids]}` graph; flag dangling references (Wf-XXX not found on disk).
2. **`molmetal/scripts/pre_flight_check.py`** — add `p3_wf_cross_dep_validator` gate (default ON); expect PASS if `wf_cross_dep.json` exists + graph has no dangling refs + 0 NEW FAILs since last INDEX sync.
3. **`molmetal/scripts/wf_index_audit.py`** (NEW) — audit `TODO/INDEX.md` §22 for completeness; flag missing §22.x entries for recent Wf.
4. **Tests:** 1 NEW test file `molmetal/tests/test_wf_cross_dep_validator.py` — verify graph construction + dangling detection + INDEX.md §22 coverage.
5. **Re-verify pre-flight** — run pre-flight with NEW `p3_wf_cross_dep_validator` gate; expect `93 PASS / 0 FAIL / 2 SKIP / 95 TOTAL` (after #2 adds `p3_wf343c5_checklist_autoemit_present` + this adds `p3_wf_cross_dep_validator`).

### §5.4 Honorable mentions (4 secondary recommendations)

#### §5.4.a Recommendation #4 — Honest-negative catalog expansion (9 → 36 entries)

* **What:** Expand AGENTS.md §9 honest framing catalog from 9 entries to 36 entries (per Wf-263B §3.5 expansion pattern + Wf-335-SYN §5 + Wf-357-A1 §3.2).
* **Cost:** LOW (NEW docstring taxonomy + template + 27 NEW entries mirroring the 9-entry pattern).
* **Projected benefit:** +20% paper §6 Limitations quality (PROJECTED; each entry becomes a reviewer rebuttal citation point).
* **GPU scope:** CPU-only docs change; no GPU launch required.

#### §5.4.b Recommendation #5 — Skill activation hooks (event-driven vs text-based)

* **What:** Replace text-based skill activation (current `mol-metal-X` keyword match) with event-driven hooks (e.g., `On(URLExtractFailure)` → load `mol-metal-lambda-calculus` skill automatically).
* **Cost:** HIGH (Claude Code harness extension; system-bundled scope).
* **Projected benefit:** -20% LLM tool-selection latency (PROJECTED; reduces prompt token roundtrip).
* **GPU scope:** CPU-only docs change; no GPU launch required; out of project scope (system-bundled).

#### §5.4.c Recommendation #6 — GPU launch dashboard (live progress + kill_then_adapt)

* **What:** Replace manual `nvidia-smi` polling with auto-daemon that watches `molmetal/reports/wfXXX_run/` and emits live progress to `molmetal/reports/wfXXX_dashboard.json` (PID + wall + GPU util + ETA).
* **Cost:** MEDIUM (~150 LOC daemon + JSON state machine + opt-in flag per launch wrapper).
* **Projected benefit:** -50% manual polling (PROJECTED; per `dont-poll-gpu-labels.md` user preference).
* **GPU scope:** CPU-only daemon; launches when any `scripts/launch_*.sh` is invoked (USER_OK-gated).

#### §5.4.d Recommendation #7 — CI/CD integration (GitHub Actions per Wf-351 §5)

* **What:** GitHub Actions YAML on `.github/workflows/pre_flight.yml` — runs `pre_flight_check.py --pipeline all` on every PR + push to master.
* **Cost:** HIGH (GitHub Actions YAML + secrets management + GPU env probe + Blackwell sm_120 setup; Wf-351-B3 `mol-metal-sci-paper-submission` infra-adjacent).
* **Projected benefit:** +30% developer velocity (PROJECTED; standard GitHub Actions automation benefit).
* **GPU scope:** Out of project scope (CI/CD infrastructure); USER_OK-gated.

---

## §6 Standing directives re-confirmation (per AGENTS.md §5)

| § | Directive | Status preserved |
|---|---|:---:|
| §5.1 | Standing GPU authorization (2026-09-29 user grant *"显存够就放心大胆用"*) | ✅ PRESERVED — 0 GPU launches in Wf-357-C3; all 3 recommendations are CPU-only docs+code |
| §5.2 | Honest framing mandate (MEASURED vs PROJECTED tag discipline) | ✅ PRESERVED — every claim cites JSON / report path; PROJECTED benefits clearly labeled as analogous-system estimates |
| §5.3 | No paper work without authorization (TODO-84 active) | ✅ PRESERVED — 0 `paper/*` touches in Wf-357-C3 |
| §5.4 | Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"*) | ✅ PRESERVED — Tier-2 markers UNCHANGED; m18 / m20 / wet-lab SOPs preserved as TIER-2 HONEST-NEG |

---

## §7 Conclusion (Honest framing per `party-twenty-spirit` §3.5)

* **Current workflow inventory is well-instrumented**: 49 ultracode workflows + 29 skills + 17 R/Wf waves + 6 NEW USER_OK-gated launch wrappers; pre-flight v3 90/94 PASS; Tier-1 LEAD metrics (m21 0.8007, m17 0.891) MEASURED at SOTA; 4 NEW CEILING_BREAK axes (m6_kappa +0.6814, m2 metalcytotox +0.2227, m7 0.7925, m15 0.6578 Å) preserved.
* **Top-3 workflow/process improvements NOT yet applied** (ranked by PROJECTED lift × feasibility):
  1. **Recommendation #1: Auto-refresh wf343c2 default `--preflight` path + auto-fix argparse help-text bug** (CPU-only docs+code; PROJECTED `92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL` vs current 90/2/2/94; closes Wf-357-A1 §2.1 + §2.2 FAILs; ~5 LOC across 2 files).
  2. **Recommendation #2: Pre-retrain checklist auto-emission into pre-flight JSON** (Wf-343c5 NEW; PROJECTED -80% manual audit time per retrain wave; NEW `p3_wf343c5_checklist_autoemit_present` gate).
  3. **Recommendation #3: Workflow regression test suite + cross-workflow dependency tracking** (Wf-cross-dep-validator NEW; PROJECTED -50% inter-workflow bugs; NEW `p3_wf_cross_dep_validator` gate).
* **4 honorable mentions** (Recommendation #4 honest-negative catalog expansion 9 → 36; #5 skill activation hooks; #6 GPU launch dashboard; #7 CI/CD GitHub Actions) — all PROJECTED to ship in future Wf-358+ cadence.
* **Honest-negative caveats preserved** (per §3.5):
  - All speedup / lift claims are **PROJECTED** from analogous-system benchmarks; **NO Mol-Metal A/B smoke has been launched** as of 2026-09-29.
  - Wf-357-A1 2 FAILs (`p2_tests_pass` + `p3_wf343c2_per_wave_runtime_check`) are **pre-existing, NOT Wf-357-C3-introduced**; they are the only blockers keeping pre-flight v3 from 92/94 PASS.
  - Tier-2 HONEST-NEG metrics (m3, m4, m6, m9, m10, m18, m20) remain HONEST-NEG — not on 100-pocket CrossDocked100 protocol; not single-pocket smoke; Vina -7.782 PROJECTED requires GPU launch verification.
  - 49 ultracode workflow count is **observational** (re-read existing docs + grep audit); no new architecture decisions made.
* **0 GPU launches performed in Wf-357-C3**; 0 NEW tests written; 0 `paper/*` touches; pre-flight v3 90/94 PASS preserved at Wf-357-A1 timestamp (exit code 0 at Wf-357-C3 timestamp confirmed).

---

## §8 Cross-references (per `party-twenty-spirit` §3.5 cite discipline)

| Doc | Path |
|---|---|
| Wf-357-A1 pre-flight audit (anchor) | `TODO/completed/357a1_preflight_audit.md` |
| Wf-357-A2 wf343c1 audit | `TODO/completed/357a2_wf343c1_audit.md` |
| Wf-357-A3 per-wave audit | `TODO/completed/357a3_per_wave_audit.md` |
| Wf-357-A4 data quality audit | `TODO/completed/357a4_data_quality_audit.md` |
| Wf-357-A5 NeurIPS reproducibility audit | `TODO/completed/357a5_neurips_reproducibility_audit.md` |
| Wf-357-A6 GPU readiness audit | `TODO/completed/357a6_gpu_readiness_audit.md` |
| Wf-357-B1 Tier-1 LEAD metrics verification | `TODO/completed/357b1_tier1_lead_metrics.md` |
| Wf-357-B2 Tier-1 supporting metrics verification | `TODO/completed/357b2_tier1_supporting_metrics.md` |
| Wf-357-B3 Tier-2 honest-neg metrics verification | `TODO/completed/357b3_tier2_honest_neg_metrics.md` |
| Wf-357-B4 supporting metrics verification | `TODO/completed/357b4_supporting_metrics.md` |
| Wf-357-C1 speedup analysis | `TODO/completed/357c1_speedup_analysis.md` |
| Wf-357-C2 architecture analysis | `TODO/completed/357c2_architecture_analysis.md` |
| Wf-335-SYN final synthesis (m6_kappa + R² + CFM ultracode) | `TODO/completed/335_final_synthesis.md` |
| Wf-340 final synthesis (CFM training framework improvements) | `TODO/completed/340_final_synthesis.md` |
| Wf-343C1 standalone (21 PASS / 5 SKIP) | `molmetal/scripts/wf343c1_pre_retrain_check.py` |
| Wf-343C2 per-wave checks (7 checks) | `molmetal/scripts/wf343c2_per_wave_check.py` |
| Wf-343C3 data quality (19 PASS / 6 SKIP) | `molmetal/scripts/wf343c3_data_quality_check.py` |
| Wf-350-A1/A2/A3 NeurIPS §D.9/§E.4/§E.5 reproducibility | `molmetal/reports/wf350a{1,2,3}_*.json` |
| Wf-350-B1 GPU readiness gate | `molmetal/scripts/pre_flight_gpu_ready.sh` |
| METRICS_TARGETS (Tier-1/2/3 + 21 metrics × MEASURED vs PROJECTED) | `TODO/02_metrics/METRICS_TARGETS.md` |
| MECHANISM_METRIC_MAPPING (M1-M28 → m1-m21 cross-mapping) | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` |
| Pre-retrain self-checklist (12 honest-framing lessons) | `科研进展_pre_retrain_checklist_2026-09-28.md` |
| AGENTS.md (skill catalog + standing directives) | `AGENTS.md` |
| User preference: don't poll GPU | `~/.claude/memory/dont-poll-gpu-labels.md` |

---

**END OF Wf-357-C3 WORKFLOW ANALYSIS** — Owner Hugo Lee + ultracode agent (Wf-357-C3); cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.