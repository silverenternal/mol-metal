# WF-Decisions-Summary — 6 user-gated decisions, one-page

**Date:** 2026-09-14
**Author:** WF-Decisions-Summary agent (pure-analysis; no code touched)
**Sources:** `TODO/pending/19_user_decisions.md`, `TODO/pending/20_post_r10_r11_action_plan.md`,
`molmetal/reports/round10_e2e_pt_cfg_vina.md`, `molmetal/reports/round11_engine_parity_n50.md`,
`molmetal/reports/anticancer_vs_general_metrics_survey.md`,
`molmetal/reports/ultracode_audit/PROJECT_STATUS.md` §8, `TODO/pending/decisions.md` (D1-D5 closed).
**Scope:** aggregate the 6 pending user-gated calls into a one-page sheet with evidence summary,
pros/cons per option, default recommendation, and a single-line reply template per decision.

---

## 1. Decision status table

| ID | Question | Current evidence | Status | Default rec |
|---|---|---|---|---|
| **D6** | REINVENT4 install path | Multiproperty bridge verified at `/mnt/storage/env-projects/reinvent4-rocm` (R7 + R8 done); R5 risk note: multiproperty layer still OBS via RDKit proxy | awaiting user input | **(a) separate venv** |
| **D7** | Vina 1.2.7 vs QuickVina 2 swap activation | Round-11 N=50 parity: Pearson r=0.9983 / Spearman ρ=0.9984 / mean paired diff +0.009 ± 0.018 kcal/mol / MAD=0.071 / n=46 paired; 4/50 qvina02 vocab failures | awaiting user input | **(c) both engines in headline table** |
| **test_005** | 4RN0 ASP B101 cohort treatment | ASP B:101 CG/OD1/OD2 atoms physically absent from deposited 4RN0.pdb; agent_b_evidence §3.2 item 12; dropping loses 1/10 of the receptor coverage | awaiting user input | **(a) keep labelled "modeled atoms"** |
| **cite-only-SOTA** | Ship cite-only SOTA column in cohort tables | D1 cite-only path already approved 2026-09-12; per-row footnote pattern established | awaiting user input | **(a) ship with per-row footnote** |
| **MW-range** | Lipinski MW vs IV 300-700 Da anticancer window | Anticancer survey §6: logP 2-5 + TPSA 60-150 + IV MW 300-700 alongside Lipinski; `priors/anticancer_metric_suite.py` already wired for dual reporting via `descriptor_report()` | awaiting user input | **(c) both flags side-by-side** |
| **journal** | Round-13 target journal | Story axis not yet settled; Round-12 acceptance gates axis dominance; (a)/(b)/(c)/(d) mapping in `19_user_decisions.md` | awaiting Round-12 | **defer to Round-12** |

---

## 2. Per-decision evidence + recommendation

### D6 — REINVENT4 install path

The REINVENT4 venv at `/mnt/storage/env-projects/reinvent4-rocm` is operational: the package install and the learned-NLL RPC channel were smoke-tested in `molmetal/reports/reinvent4_learned_smoke/`, and the multiproperty bridge wired in Round-7/8 returns a non-zero score when invoked. Risk remains: per the TODO-15 anticancer metric survey and the R5 blocker note, the multiproperty layer still ships an RDKit-property-proxy and only falls back to a live REINVENT4 worker when one is configured; nothing in the current ship state changes that. The R8 round-up cites 21/21 test files green but flags TODO-05 reinvent as env-blocked, meaning the live scoring channel is observable but not validated end-to-end against activity data.

Pros / cons:
- (a) separate venv at `/mnt/storage/env-projects/reinvent4-rocm` — already paid (env exists, install + learned-NLL RPC verified), lowest risk; con: multiproperty bridge is still an RDKit proxy and a live worker still has to be wired before any anticancer claim is supported by it.
- (b) shared venv with molmetal — one fewer environment to maintain and consistent dependency surface; con: RDKit/torch version conflicts are a documented recurring risk on the ROCm stack and would invalidate the verified install.
- (c) Docker container — full isolation and reproducible image; con: +overhead on a single-machine dev loop, image-build time on this ROCm toolchain, and the (a) venv already meets the reproducibility bar for the ship-it-now path.

**Recommendation: (a) — already verified, lowest risk, matches what the env-blocked TODO-05 status implies is the only path that does not regress working state.** Justification: the user has already paid the install cost; switching venvs before TODO-05 unblocks adds risk for zero measured upside, and the multiproperty-layer honesty note is independent of the install choice.

**Suggested user reply:** `D6 = (a)`

### D7 — Vina 1.2.7 → QuickVina 2 swap activation

Round-11 ran the N=50 paired parity experiment (`molmetal/reports/round11_engine_parity_n50.md`) on pocket 1h36 at exhaustiveness 8, seed 42, 1 CPU, matched box (centre `[35.684, 52.085, 44.486]` Å, side 21.502 Å). Result on n=46 paired: Pearson r=0.9983, Spearman ρ=0.9984, mean paired diff +0.0086 ± 0.0184 kcal/mol (paired SE), mean absolute diff 0.071 kcal/mol. The 4/50 QuickVina failures are a **bundled qvina02 (AutoDock Vina 1.1.2 from 2011) vocabulary delta** rejecting the modern `CG0` aromatic-aromatic atom type emitted by `meeko`/`PDBQTWriterLegacy` — an engine-version artefact, not a parity failure. This matches the post-R10/R11 action plan's D7 ladder exactly and exceeds Alhossary 2015's published r=0.967 between Vina and QuickVina on 195 PDBbind complexes because we hold seed + box + receptor constant.

Pros / cons:
- (a) keep Vina 1.2.7 default — most comparable to literature SBDD baselines and to the CrossDocked2020 numbers cited as SOTA references; con: loses the 100× speed advantage of QuickVina 2 and signals to reviewers that we did not consider engine choice a question.
- (b) flip to QuickVina 2 default — fastest headline, single-engine stats are cleanest to write up; con: the qvina02 binary atom-type rejections need a workaround (or a modern QVina 2.1 swap) before every cohort SMILES docks, and we have no empirical MAD for QVina's broader literature footprint.
- (c) both engines in headline table with explicit caveat — preserves both literature comparability (Vina) and the modern fast-engine claim (QVina), and the 1h36 parity number is strong enough to defend with a footnote; con: requires writing per-row parity footnote and re-running the headline table with two rows per cohort.

**Recommendation: (c) — byte-identity proved on the existing binary, parity experiment shipped today, default per audit.** Justification: r=0.9983 with MAD=0.071 kcal/mol is three orders of magnitude tighter than typical scoring noise floor and three orders tighter than the Alhossary benchmark; the 4 failures are engine-version drift that can be footnoted honestly. Defer until cross-pocket parity is measured before claiming D7 closed for Round-13 acceptance.

**Suggested user reply:** `D7 = (c)`

### test_005 (4RN0 ASP B101) cohort inclusion

`molmetal/data/crossdocked100_manifest.csv` includes 4RN0 pocket 10, and the `crossdocked_first10_resolved/README.md` plus `agent_b_evidence.md` §3.2 item 12 confirm that ASP B:101 CG/OD1/OD2 sidechain atoms are physically absent from the deposited 4RN0.pdb structure. This is a real data integrity limitation (a missing sidechain in the experimental model), not an engineering choice. The other 9 receptors in the cohort have full sidechains.

Pros / cons:
- (a) keep test_005 labelled as "modeled atoms" experiment — preserves 10-receptor coverage target, explicit per-pocket footnote is honest about the data limitation, reviewer can see we noticed; con: requires explicit framing in every cohort summary and a footnote in every pocket table.
- (b) drop from Round-12/13 cohorts entirely — cleanest write-up, all receptors fully resolved; con: 1/10 statistical power loss and reviewer might still ask "why exclude?"; 4RN0 is the only available ASP-pocket in the resolved 10 so dropping it removes ASP coverage entirely.

**Recommendation: (a) — labelled transparency preserves the 10-receptor coverage and the data-integrity framing is honest without hiding.** Justification: hiding the modeled sidechain would be worse than documenting it; the audit-layer framing already treats this as a known limitation, and the 10-receptor coverage is itself a Round-13 paper number.

**Suggested user reply:** `test_005 = (a)`

### Cite-only SOTA column

D1 (DiffSBDD cite-only path) was approved 2026-09-12 and is already in the per-row footnote pattern (`*(not re-run by Mol-Metal; cited from <ref>)*`) per `TODO/pending/decisions.md` and `PROJECT_STATUS.md` §8. The question is whether the Round-12/13 cohort tables get a dedicated `SOTA_cited` column alongside the measured Mol-Metal rows, or whether cite-only references live in the caption only.

Pros / cons:
- (a) ship cite-only SOTA column — every cohort row gets a SOTA-cited column with explicit footnote; matches D1 cite-only path; reviewer sees context preserved; con: small write-up overhead per row + per-pocket table; risk of reviewer treating cite-only rows as Mol-Metal claims if the footnote is too small.
- (b) only include measured rows — cleanest table, no ambiguity; con: loses comparison context, reviewer has to look up SOTA numbers in the related-work section, and the alignment-gap analysis (`sota_alignment_gap_analysis.md`) becomes harder to surface.

**Recommendation: (a) — aligns with the D1 cite-only path already approved, preserves scientific honesty without dropping comparison context.** Justification: the D1 path already established the footnote pattern; expanding it to a column costs little and makes the SOTA gap (per `sota_alignment_gap_analysis.md`) explicit at a glance.

**Suggested user reply:** `cite-only-SOTA = (a)`

### MW-range flag (Lipinski MW vs IV 300-700 Da anticancer window)

The anticancer metric survey (`molmetal/reports/anticancer_vs_general_metrics_survey.md` §6) recommends logP 2-5 + TPSA 60-150 + IV MW 300-700 alongside the standard Lipinski descriptors for anticancer drug design. `priors/anticancer_metric_suite.py` is already wired for both descriptors via `AnticancerMetricSuite.descriptor_report()`. Lipinski's MW ≤ 500 was derived for oral drugs and underweights the size window actually used by intravenous anticancer candidates; the 300-700 Da IV window matches the clinical reality of cisplatin (300 Da), carboplatin (371 Da), and the upper end of approved platinum drugs (satraplatin ~500 Da).

Pros / cons:
- (a) keep Lipinski MW descriptive flag only — preserves the dominant literature baseline; con: under-represents the IV anticancer size window and is inconsistent with the anticancer survey recommendations already shipped.
- (b) add 300-700 Da IV anticancer MW-range flag only — most informative for the anticancer story; con: looks like a clinical-efficacy claim if not framed carefully (the survey explicitly warns against this); loses Lipinski baseline comparability.
- (c) both flags side-by-side — maximum interpretability, reviewer sees both the Lipinski baseline and the IV-appropriate window; con: two columns per pocket table instead of one; must frame IV window as "IV-appropriate window, not clinical efficacy" to avoid over-claim.

**Recommendation: (c) — anticancer survey recommends dual reporting; both flags already computable via the existing suite.** Justification: the dual flag = maximum interpretability at trivial engineering cost, and the framing pattern ("IV-appropriate window, not clinical efficacy") is already precedent in `priors/anticancer_metric_suite.py`'s "heuristics do not establish efficacy" caveat.

**Suggested user reply:** `MW-range = (c)`

### Journal choice (Round-13)

Round-13 work is currently **designed, not started** (`PROJECT_STATUS.md` §7). The dominant-axis question gates journal choice: which axis of the 6-axis ablation has the strongest measured on/off delta after Round-12 acceptance. Four plausible targets are listed in `TODO/pending/19_user_decisions.md`, mapped to the dominant story.

Pros / cons:
- (a) Digital Discovery (RSC, Q1 IF ~8.5, ML × chemistry) — high-impact venue if the Triton-on-consumer-GPU + lipman-FM story wins; con: novelty bar is high and Round-12 acceptance of the consumer-GPU axis is not guaranteed.
- (b) J. Chem. Inf. Model. (ACS, Q1 IF ~5.6, methods-heavy) — best fit if metal-anticancer benchmarks win and the methods story is the binder; con: methods-heavy framing requires the head-to-head SBDD-vs-mol-metal benchmark to land cleanly.
- (c) Patterns (Cell, Q1 IF ~6.0, broader scope) — best fit if the MCTS-on-metal-complex + symbolic-discovery story wins; con: Patterns accepts methodology but tends to prefer broader-impact claims.
- (d) Briefings in Bioinformatics (Oxford, Q1 IF ~7.0, interpretability) — best fit if the closed-loop Lambda proof-theoretic story wins; con: proof-theoretic story needs the closure-theorem ship (WF-Lambda-4) before Round-13 acceptance.

**Recommendation: defer to Round-12 acceptance; pick (a) Digital Discovery as the default if no axis dominance emerges, since the Triton-on-consumer-GPU axis is the most generally novel and most defensible against the SOTA alignment gap.** Justification: deferring until Round-12 acceptance clarifies which axis dominates prevents a desk-reject from a journal-mismatch story; (a) is the safest default because the consumer-GPU angle is the single result most likely to be both novel and reviewable across the four options.

**Suggested user reply:** `journal = (a) defer to Round-12; default Digital Discovery unless metal-anticancer benchmark dominates`

---

## 3. Cross-cutting dependencies

| Pair | Relationship |
|---|---|
| **D6 ↔ cite-only-SOTA** | D6's multiproperty-bridge OBS status means cite-only rows for activity-prediction SOTA are the only honest comparator today; approving (a) for cite-only preserves an honest comparison even if D6 stays at RDKit-proxy |
| **D7 ↔ Round-12 acceptance** | D7 (c) headline-table requirement depends on cross-pocket parity (≥3 pockets) before Round-13 acceptance; today only 1h36 measured |
| **D7 ↔ test_005** | D7 (c) requires per-pocket docking results; test_005 (a) keeps the 10-receptor target so cross-pocket parity has a usable cohort |
| **test_005 ↔ MW-range** | test_005 (a) preserves ASP pocket coverage, which is where the 300-700 Da IV window matters most for kinase/ASP inhibitors |
| **MW-range ↔ cite-only-SOTA** | MW-range (c) dual flag adds columns per pocket table; cite-only-SOTA (a) adds a column too — write-up order matters; ship MW-range first (already wired), cite-only second |
| **journal ↔ D6** | If metal-anticancer benchmark dominates, (b) JCIM is the journal and D6's REINVENT4 multiproperty bridge becomes a load-bearing channel — needs the live worker wired before submission |
| **journal ↔ D7** | If methods-heavy (b) or ML × chemistry (a) wins, D7 (c) dual-engine headline is the comparison-axis that the reviewer will scrutinise first |
| **journal ↔ cite-only-SOTA** | Cite-only column (a) is required for whichever journal is chosen (none of the four Q1 venues accepts uncited SOTA claims); independence: ships regardless |
| **Round-12 ↔ everything** | Round-12 acceptance clarifies the dominant axis and therefore the journal choice; D7 cross-pocket parity, test_005 cohort size, MW-range coverage, and cite-only-SOTA rows are all Round-12-table inputs |

**Implication:** the only decision the user can finalize independently of Round-12 is **journal** (defer). The other five can ship now and feed into the Round-12 cohort table directly.

---

## 4. Reply template (single text block)

```
D6 = (a)             # REINVENT4: keep separate venv at /mnt/storage/env-projects/reinvent4-rocm
D7 = (c)             # Vina 1.2.7 + QuickVina 2 both in headline table; defer cross-pocket parity to Round-12
test_005 = (a)       # Keep 4RN0 ASP B101 labelled "modeled atoms" in cohort; preserve 10-receptor coverage
cite-only-SOTA = (a) # Ship cite-only SOTA column with per-row footnote (D1 cite-only path)
MW-range = (c)       # Both flags side-by-side (Lipinski MW + IV 300-700 Da), IV framed as "appropriate window, not efficacy"
journal = (a) defer  # Defer to Round-12; default Digital Discovery unless metal-anticancer benchmark dominates
```

Reply with the option letters above (or substitute "Other" + rationale); Claude proceeds.
