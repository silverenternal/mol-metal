# WF-Decisions-Summary — 6 user-gated decisions aggregated with current evidence

**Workflow:** WF-Decisions-Summary (pure-analysis, NO code modifications)
**Date:** 2026-09-14
**Author:** WF-Decisions-Summary agent
**Output intent:** one-page user reply template with recommended (a)/(b)/(c) for each
of the 6 pending user-gated decisions documented in
`TODO/pending/19_user_decisions.md`, cross-referenced against the 4 ultracode
workflow outputs (WF-1, WF-2, WF-Extra-1, WF-Extra-2), WF-Lambda B-line
(Lambda-1, Lambda-1b, Lambda-1c, Lambda-2, Lambda-2.E), and WF-Paper-1
assemble evidence.

**Honest-framing key:**
- **MEASURED** = produced by a direct experiment + analysis on this bench.
- **PROJECTED** = extrapolated from code paths / earlier cells; not executed.
- **CITE-ONLY** = cited SOTA paper; Mol-Metal did not re-run.

---

## 0. Goal of this file

This file aggregates the 6 user-gated decisions so the user can answer
them in **one round-trip** without re-reading the underlying reports.
Each decision gets:
- a 1-paragraph **current-status** snapshot citing the evidence that
  shifts its recommended option (if any),
- the recommended option letter,
- and a 1-sentence **honest caveat** flagging what the recommendation does
  *not* establish.

The orchestrator (`w_round11_qvina_data` / `w_round12_top_journal_pilot`)
will read the user's reply and execute the chosen option.

---

## 1. D6 — REINVENT4 install path

### Current status

**Default (a) is now empirically confirmed as the working install path.**
`/mnt/storage/env-projects/reinvent4-rocm` already runs the official
REINVENT4 4.8.24 binary against the AMD GPU on gfx1101, using torch
2.14.0+rocm7.2 and torchvision 0.29.0+rocm7.2 (the upstream torch 2.12
pin was overridden and SciPy was added as an omitted runtime dep). All
576 upstream Python files are unchanged; the legacy CUDA install and the
main project env are retained. MEASURED end-to-end: `r10_cfg_real_*.md`
reports 80 RNN + 80 LSTM sampling forwards on `cuda:0` with actual
architecture `gfx1101`; CPU/GPU NLL max diff for the same input is
1.05e-5; the ChEMBL 25 prior at
`/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior` (23,226,277 bytes,
SHA256 `b6513e…732fe`) was recovered from official tag `v4.4.22` after
Zenodo timed out and is byte-identical to the upstream Git blob
`0dee328238b3d413b5a34c80e3ddab49e4e0af28`. WF-Extra-2 added a
**separate** `multiproperty` mode (`molmetal/configs/reinvent_multiproperty_amd.json`)
that drives the real REINVENT4 weighted TOML through
`reinvent4_multiproperty_jsonl_worker.py`; 10/10 SMILES succeed and
return `[0,1]` scores that are moderately distinct from the RDKit proxy
(Pearson r = 0.676, mean |Δ| = 0.107, max |Δ| = 0.32 on n-octane). The
`r_reinvent4` channel is wired into `RewardAggregator`. The previous
RDKit-only proxy remains valid as a protocol-validation channel but is
**not** a learned REINVENT4 result. Risk: the multiproperty bridge
still routes RDKit-style components (QED / SlogP / NumRings) only; a
real task-specific worker for true activity scoring remains OBS per R5
in `TODO/pending/risks.md`.

### Recommended reply

- **Recommended option: (a) — separate venv at `/mnt/storage/env-projects/reinvent4-rocm`** (already verified, lowest risk).
- **Honest caveat:** the multiproperty bridge is real REINVENT4 execution with RDKit-style built-in components; *task-specific* learned scoring (e.g. ADMET, pIC50) remains OBS until a worker that uses a real custom model is wired. Use the prior NLL channel + multiproperty channel; do not advertise "REINVENT4 → activity" yet.
- **Deadline:** 2026-09-19 (no wall-clock action; the env is already created).
- **Alternative (c) Docker** is +overhead with no benefit given (a) already works.

---

## 2. D7 — Vina → QVina swap activation

### Current status

**Default (c) both engines in the headline table is now backed by MEASURED
N=50 parity.** `molmetal/reports/round11_engine_parity_n50.md` (executed
2026-09-14) reports Pearson **r = 0.9983**, Spearman **ρ = 0.9984**,
mean paired diff (Vina − QVina) **+0.009 ± 0.018 kcal/mol (paired SE)**,
mean |diff| = 0.071 kcal/mol on N=46 paired molecules over 1h36 /
exh=8 / seed=42 / 1 CPU with matched box. This **exceeds** Alhossary
2015's published r=0.967 between Vina and QuickVina on 195 PDBbind
complexes. 4/50 mols rejected by QuickVina 2 because the vendored
`qvina02` binary (AutoDock Vina 1.1.2 from 2011) does not recognise the
modern `CG0` aromatic-aromatic atom type that `meeko` writes — a binary
vocabulary delta, not a parity failure. Byte-identity between
`qvina02` and upstream QVina/qvina was already established in
`molmetal/reports/quickvina2_binary_identity.md` (git blob
`85281985807632dc6d2e0a8564d2a3c027166f49`). Wall-clock 978 s ≈ 9.8 s /
mol / 2 engines on 1 CPU. The Round-10 R10-action-3 entry in
`TODO/pending/20_post_r10_r11_action_plan.md` upgrades D7 from
"pending parity" to "ready for user" precisely because of this number.

### Recommended reply

- **Recommended option: (c) — both engines in the headline table** with the footnote proposed in `round11_engine_parity_n50.md §Discussion`.
- **Honest caveat:** this is **single-pocket, single-seed, single-exhaustiveness** (1h36 / seed=42 / exh=8). Cross-pocket parity (≥ 5 pockets) and per-seed σ (≥ 5 seeds) remain open; do **not** advertise as a cross-pocket acceptance number. The 4 `qvina02` failures are real and should be disclosed in the paper.
- **Deadline:** now ready (Round-11 parity completed 2026-09-14); user reply unblocks Round-12 acceptance and Round-13 paper-draft headline table.
- **Alternative (a) Vina-only** if a QVina 2.1 binary becomes installable (current `qvina02` is the 2011 vendored build) and the empirical MAD vs published QVina runs exceeds the 0.6 kcal/mol ceiling.

---

## 3. test_005 (4RN0 ASP B101) cohort decision

### Current status

**Default (a) labelled modeled-atom cohort is the only honest option.**
The original 4RN0.pdb ASP B101 residue is **missing CG / OD1 / OD2
sidechain atoms** in the deposited crystal structure
(`molmetal/reports/crossdocked_first10_resolved/README.md`,
`agent_b_evidence.md §3.2 item 12`). Per `TODO/pending/17_aggregate_weak_impls_and_pending.md`,
9/10 strict-resolved receptors pass cleanly; only test_005 requires the
modeled-receptor path. Round-12 reduced-budget integration
(`molmetal/reports/r4_click_physical_test10_seed3_v2_analysis.md`) reports
that 9/30 jobs failed the *reference* PB check on the original
(unmodeled) receptor because the crystal lacks those sidechain atoms;
the separately modeled receptor completes 3 native-Vina jobs and 7
candidate docking/PB checks with **explicit labels and never merged
silently into the original-receptor result**. WF-Lambda-1b / 1c (Round-12
B-line) verified the BNF and metal-seed paths on 5 pockets × 3 seeds ×
2 arms = 30 cells without test_005 in scope (test_005 is not in the
first-10 of the Round-12 physical run; it would re-enter if Round-13
expands to 100 pockets). Dropping test_005 would lose one of ten 4RN0
matched pockets from the CrossDocked2020 target; the synthesized
receptor preserves the 10-receptor coverage target that the paper's
"metal-binding pockets" framing depends on.

### Recommended reply

- **Recommended option: (a) — keep test_005 as a separately labelled "modeled atoms" experiment** with an explicit footnote on every cohort table that the ASP B:101 CG / OD1 / OD2 atoms are reconstructed, not crystal-deposited.
- **Honest caveat:** reviewers may still query the modeled receptor; this is a real data-integrity limitation, not an engineering choice, and is documented in `crossdocked_first10_resolved/README.md`. A sensitivity sub-table with / without test_005 should be the per-pocket-table adjunct.
- **Deadline:** before Round-12 scientific-budget run (W5 / 2026-10-17).
- **Alternative (b) drop** trades scientific-honesty for power and would not change the data-integrity statement; reviewer would still see "9 pockets" and ask "why not 10?".

---

## 4. Cite-only SOTA column

### Current status

**Default (a) ship cite-only SOTA column is now the canonical pathway.**
D1 was approved 2026-09-12 (cite-only path) per `TODO/pending/decisions.md
§D1`; the cite-only column extends that decision from the DiffSBDD row
to all 9 SOTA rows (Pocket2Mol / TargetDiff / DiffSBDD / DecompDiff /
FLOWR / MolDiff / BindNet / DiffDock / RosettaFoldAA). The 9-paper
table is already drafted in
`molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` with 7
protocol-mismatch flags explicit (Flag 1–Flag 7) per
`paper/sections/02_related.tex` (`\ref{sec:related:protocol}`). WF-Paper-1
verified §2 carries all 9 citations + 7 flags
(`molmetal/reports/wf_paper1_assemble.md §2`). Honest-framing policy per
`TODO/pending/17_aggregate_weak_impls_and_pending.md`: "avoid significance
tests against cite-only published aggregates; they lack matched samples
and often use different protocols" — i.e. the cite-only column is
**context, not a one-sample significance-test null**.
WF-Extra-1 (pIC50 retrain) provides the strongest matched-baseline
analogue in this project: same HeLa48h/dark cohort, scaffold-group
splits, 3-seed mean ± std, comparing D-MPNN retrain (RMSE 0.687 ± 0.042,
Pearson r 0.198 ± 0.228) against the conditioned ridge baseline (RMSE
0.541 ± 0.089, Pearson r 0.572 ± 0.068). That is a *measured*
comparison; the cite-only column for SBDD methods is the analogue for
methods we did not re-run.

### Recommended reply

- **Recommended option: (a) — ship cite-only SOTA column** with per-row footnote pattern `*(not re-run by Mol-Metal; cited from <ref>; protocol-flag-1/2/...-incompatibilities per §2)*`.
- **Honest caveat:** cite-only rows are **context**, never significance-test samples. Predefine a multiple-comparison treatment in §4 / §5 that explicitly excludes cite-only rows from hypothesis tests.
- **Deadline:** before Round-13 paper draft (W8 / 2026-11-07).
- **Alternative (b) measured-only** would force dropping 8/9 SOTA rows and weaken the §2 positioning claim without producing any new measurement.

---

## 5. MW-range flag (Lipinski MW vs IV 300-700 Da)

### Current status

**Default (c) both flags side-by-side is the maximum-interpretability option**
and matches the `molmetal/reports/anticancer_vs_general_metrics_survey.md §6`
recommendation (logP 2-5 + TPSA 60-150 + IV MW 300-700 alongside standard
flags). The TODO-15 anticancer metric suite ships
`AnticancerMetricSuite.descriptor_report()` with raw logP / TPSA / RotB /
MW values + IV/metal-adjusted flags and is exercised by
`molmetal/reports/wf_extra1_full/final.md` (pIC50 retrain) and
`molmetal/reports/wf_lambda1c_pilot_v3/final.md` (organic-arm 0.004877
diversity_alpha, cisplatin-arm 1.0000 metal_compliance). Per Round-13
anticancer metric annotations (`TODO/pending/14_full_100pocket_paper_r13.md`
§Anticancer metric survey): "Keep Lipinski MW as a descriptive statistic
only; do not gate metal complexes at MW ≤ 500 Da. Add an MW-range flag
(300-700 Da) and document retained high-MW candidates." This is exactly
option (c). The `paper/sections/06_limitations.tex` will frame hERG /
CYP450 / PPB / NCI-60 GI50 / selectivity index / kinetic-redox as
**predictor or wet-lab requirements, not measured** — same scientific-
honesty pattern as the MW flag.

### Recommended reply

- **Recommended option: (c) — both flags side-by-side**, with explicit framing: "Lipinski MW retained as descriptive flag; 300-700 Da IV MW reported as screening window, not clinical efficacy" (parallels the "heuristics do not establish efficacy" framing already in `priors/anticancer_metric_suite.py`).
- **Honest caveat:** the 300-700 Da window is a **cytotoxic-IV literature range**, not a clinical-relevance claim. Reviewers may misread it as efficacy; the per-table footnote must be unambiguous.
- **Deadline:** before Round-13 paper draft (W8 / 2026-11-07).
- **Alternative (a) only Lipinski MW** would lose the metal-anticancer interpretability signal that the §6 of `anticancer_vs_general_metrics_survey.md` argues for.

---

## 6. Journal choice (Round-13)

### Current status

**Default = story-dependent (defer to Round-12 acceptance).** No single
story dominates today. Strongest measured evidence available at
2026-09-14:
- **§4.5 100-pocket sweep** — projected, no measurement yet.
- **§4.4 click chemistry + PB + retrosynthesis** — measured reduced-budget
  (70 products / 63 docked / 63 PB-pass on N=10 × 3 seeds;
  `r4_click_physical_test10_seed3_v2_analysis.md`).
- **§4.1 tmQM pretraining** — measured baseline (16 numbers × 4 metals ×
  4 models × 4 splits per `honest_baseline_summary.md`).
- **§4.2 pIC50 retrain** — measured negative result on HeLa48h/dark cohort
  (RMSE 0.687 ± 0.042, Pearson r 0.198 ± 0.228 vs conditioned ridge
  RMSE 0.541 ± 0.089, Pearson r 0.572 ± 0.068 per `wf_extra1_full/final.md`).
- **§3 MLC formalism + β-NF + AST** — shipped (446+ test functions
  across 51 test files per `PROJECT_STATUS.md §4.1`); WF-Paper-1
  verified §1/§2/§6/§7 = 895 LaTeX lines, 29 cross-refs
  (`wf_paper1_assemble.md §7`).
- **§4.6 100-pocket × 3-seed, cite-only SOTA column** — measured N=50 Vina
  vs QVina parity Pearson r=0.9983 (D7 unblocked per `round11_engine_parity_n50.md`).
- **REINVENT4 multiproperty NLL channel** — measured r=0.676 vs proxy
  (`wf_extra2_batch/final.md §3`); wired into `RewardAggregator.r_reinvent4`.
- **WF-Lambda-2.E homotype vs Tanimoto** — measured orthogonality
  preserved (mean_homotype_new=0.2293 vs Tanimoto=0.9243 on 10-mol / 45
  pairs; cisplatin_benzene_homotype_new=0.5000) — strongest evidence for
  the "Lambda-discovered-but-Tanimoto-invisible" framing
  (`wf_lambda2e_compare/final.md`).

Per `TODO/pending/14_full_100pocket_paper_r13.md §Top-journal submission
targets`:
- Digital Discovery (RSC, IF 8.5, Q1, OA) — if §4.5 (100-pocket sweep)
  leads; protocol-flags tolerant.
- J. Chem. Inf. Model. (ACS, IF 5.6, Q1, optional OA) — if §4.1 + §4.4
  lead (methods + pharma audience).
- Patterns (Cell Press, IF 6.0, Q1, OA) — if §3 MLC formalism + β-NF
  proof are strong.
- Briefings in Bioinformatics (Oxford, IF 7.0, Q1, optional OA) — if
  interpretability (homotype orthogonality, MCTS-VirtualLoss-TT) leads.

Today the strongest single story is **WF-Lambda-2.E** (homotype
orthogonality) + §3 formalism → favours Patterns; or **§4.4
click+PB+retro reduced-budget** → favours Digital Discovery / JCIM.
The 100-pocket sweep is **not measured yet** so Digital Discovery is a
forward bet. JCIM is the safest target (methods + pharma audience,
broad acceptance of cite-only protocol flags per
`agent_a_planning.md`).

### Recommended reply

- **Recommended option: defer to Round-12 acceptance; commit to (b) J. Chem. Inf. Model. as the safe-default Round-13 target** with a Digital Discovery re-evaluation after Round-12 numbers land. Today the safest submission path is JCIM because the §4.4 click+PB+retro + §4.1 tmQM evidence is the strongest measured layer; Patterns is a stretch if the homotype orthogonality story becomes a §3 contribution.
- **Honest caveat:** the **100-pocket × 3-seed sweep** is the strongest possible §4.5 and would shift the journal choice toward Digital Discovery; that measurement is projected, not done. If Round-12 acceptance produces a strong CFG / Pt(II)-prior end-to-end Vina delta (per `TODO/pending/20_post_r10_r11_action_plan.md §Action 1 / §Action 2`), Digital Discovery becomes viable.
- **Deadline:** Round-13 submission window (target W7 / 2026-10-31 if on the original timeline; W8 / 2026-11-07 if the revised timeline per Action 4 in `20_post_r10_r11_action_plan.md` holds).
- **Alternative (d) Briefings in Bioinformatics** is the strongest fit if the homotype-vs-Tanimoto orthogonality story (WF-Lambda-2.E) becomes the §3 lead contribution rather than a §6 framing footnote.

---

## 7. Decision status table (post-aggregation)

| ID | Decision | Default | Updated deadline | Evidence status | User reply needed? |
|---|---|---|---|---|---|
| D6 | REINVENT4 install path | (a) separate venv | 2026-09-19 | MEASURED — install + NLL RPC verified + multiproperty channel wired (r=0.676 vs proxy) | **YES — confirm (a) or pick alternative** |
| D7 | Vina/QVina swap | (c) both engines | now ready (Round-11 parity completed 2026-09-14) | MEASURED — N=50 parity r=0.9983, single-pocket caveat | **YES — pick (a) / (b) / (c)** |
| test_005 | cohort inclusion | (a) labeled modeled-atom | pre-R12-budget (W5 / 2026-10-17) | MEASURED — Round-12 reduced-budget integration ran test_005 with explicit label | **YES — pick (a) / (b)** |
| cite-SOTA | ship column | (a) ship with footnote | pre-paper-draft (W8 / 2026-11-07) | D1 already approved; 9 SOTA rows + 7 flags in §2 | **YES — pick (a) / (b)** |
| MW-range flag | dual flag | (c) both Lipinski + 300-700 Da | pre-paper-draft (W8 / 2026-11-07) | `AnticancerMetricSuite.descriptor_report()` ships both; TODO-15 annotation explicit | **YES — pick (a) / (b) / (c)** |
| journal | Round-13 pick | (b) JCIM safe-default, Digital Discovery re-evaluate post-R12 | W8 / 2026-11-07 | MEASURED §3 + §4.1 + §4.4 strongest; §4.5 + §4.6 cite-only | **YES — pick (a) / (b) / (c) / (d) or "defer"** |

---

## 8. Honest framing — MEASURED vs PROJECTED

**MEASURED on this bench (2026-09-14):**
- D6: REINVENT4 install + NLL + multiproperty bridge (10/10 SMILES, r=0.676 vs proxy, max |Δ|=0.32).
- D7: N=50 Vina-vs-QuickVina parity (Pearson r=0.9983, n=46 paired, single pocket / single seed / single exhaustiveness).
- test_005: 30-cell reduced-budget physical integration with explicit modeled-receptor label (`r4_click_physical_test10_seed3_v2_analysis.md`).
- cite-SOTA column: 9-paper table + 7 protocol-mismatch flags already drafted in §2 of `paper/sections/02_related.tex`.
- MW flag: `AnticancerMetricSuite.descriptor_report()` exposes both Lipinski and IV 300-700 Da flags; measured via `wf_extra1_full` and `wf_lambda1c_pilot_v3`.
- journal: story is §4.4 + §4.1 + §3; §4.5 / §4.6 cite-only.

**PROJECTED (not measured on this bench):**
- D7 cross-pocket parity (≥ 5 pockets) and per-seed σ (≥ 5 seeds).
- D7 `exh=8 QVina 2 ≈ exh=16 Vina 1.2.7` direct test.
- test_005 full-budget run on a Pt-pocket with real CFM data and tmQM Pt subset.
- Round-12 N=10 × 3-seed scientific-budget pilot (full budget, not reduced).
- Round-13 100-pocket × 3-seed sweep.
- CFG end-to-end decoded molecules ≥ 0.5 of finite clouds (current = 0/96 / 0/384 / 0/384 per WF-1 / WF-2.A5 / WF-2.A6).
- Pt(II) prior end-to-end Vina delta ≥ 0.5 kcal/mol on a real Pt-pocket with real CFM data (current = 0.000 kcal/mol on a Fe(HEM) pocket with synthetic random training points, per `round10_e2e_pt_cfg_vina.md`).

**OUT OF SCOPE:**
- Wet-lab activity validation; binding affinity beyond docking scores; redox/kinetic/selectivity measurements.
- D7 modern QuickVina 2.1 binary vs the current vendored `qvina02` (2011) — distinct score-function lineage and atom vocab.

---

## 9. Recommended reply template (for user)

> Reply with one letter per decision, or "Other":
>
> - **D6** (REINVENT4 install path): (a) / (b) / (c) / Other — recommended (a)
> - **D7** (Vina/QVina): (a) / (b) / (c) / Other — recommended (c)
> - **test_005** (cohort): (a) / (b) / Other — recommended (a)
> - **cite-SOTA** (column): (a) / (b) / Other — recommended (a)
> - **MW-range flag**: (a) / (b) / (c) / Other — recommended (c)
> - **journal** (Round-13): (a) Digital Discovery / (b) JCIM / (c) Patterns / (d) Briefings in Bioinformatics / "defer" — recommended (b) safe-default with re-evaluation after Round-12 acceptance
>
> Claude proceeds with the chosen options; default = recommended if no reply.

---

## 10. Cross-references

- `TODO/pending/19_user_decisions.md` — original 6-decision sheet (input).
- `TODO/pending/05_reinvent4_install.md` — REINVENT4 install + NLL + multiproperty bridge (D6).
- `TODO/pending/12_qvina_data_staging_r11.md` — Round-11 N=50 parity (D7).
- `TODO/pending/13_top_journal_pilot_r12.md` — Round-12 N=10 × 3-seed pilot spec; WF-Lambda B-line evidence (test_005 + journal pick).
- `TODO/pending/14_full_100pocket_paper_r13.md` — Round-13 100-pocket sweep + paper draft spec; journal targets in priority order (journal).
- `TODO/pending/18_activity_assay_calibration.md` — pIC50 cohort / activity calibration; NOT directly MW-range-flag but supports the IV-cytotoxic framing (MW-range flag).
- `TODO/pending/20_post_r10_r11_action_plan.md` — refined plan + Action 3 = D7 unblock; Action 4 = timeline re-stack (R12 → W5; R13 → W8).
- `TODO/pending/17_aggregate_weak_impls_and_pending.md` — audit; 9/10 strict-resolved, test_005 modeled-atom protocol (test_005).
- `molmetal/reports/round11_engine_parity_n50.md` — N=50 Vina/QVina parity Pearson r=0.9983 (D7).
- `molmetal/reports/wf_extra2_batch/final.md` — REINVENT4 multiproperty 10-SMILES r=0.676 (D6).
- `molmetal/reports/wf_extra1_full/final.md` — pIC50 retrain negative result (Pearson r=0.198 vs ridge 0.572; supports honest-framing in §6).
- `molmetal/reports/wf_paper1_assemble.md` — §1/§2/§6/§7 = 895 LaTeX lines, 29 cross-refs; 9 SOTA + 7 flags verified (cite-SOTA).
- `molmetal/reports/ultracode_audit/PROJECT_STATUS.md §8` — 6 decisions table (synthesis).
- `molmetal/configs/reinvent_prior_amd.json` — D6 prior NLL config (gfx1101, cuda:0).
- `molmetal/configs/reinvent_multiproperty_amd.json` — D6 multiproperty config (gfx1101, cuda:0, logp/ring_count/qed).
- `molmetal/reports/wf_lambda1c_pilot_v3/final.md` — BNF valence patch + metal-seed ablation; 30 cells, all 6 metrics non-zero in at least one arm (MW-range flag + journal).
- `molmetal/reports/wf_lambda2e_compare/final.md` — homotype orthogonality preserved; cisplatin_benzene = 0.5000 (journal pick evidence).
- `molmetal/reports/anticancer_vs_general_metrics_survey.md §6` — IV MW 300-700 Da + logP 2-5 + TPSA 60-150 + RotB <10 (MW-range flag).
- `molmetal/reports/r4_click_physical_test10_seed3_v2_analysis.md` — 30-cell physical integration with test_005 modeled-receptor label (test_005).

---

*Pure-analysis workflow: NO code modifications. Honest-framing mandatory.
6-decision sheet aggregated for one-round-trip user reply.*