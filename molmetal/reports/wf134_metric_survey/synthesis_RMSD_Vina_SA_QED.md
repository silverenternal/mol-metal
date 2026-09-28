# WF-134 — Synthesis: Tier-1 Metric Definitions for Lambda-Metal SBDD

**Date:** 2026-09-27
**Source data:** `lit_table_RMSD_Vina_SA_QED.md` + `lit_table_RMSD_Vina_SA_QED.json`
**Scope:** 1-paragraph verdict per metric + actionable changes for §3.5 / §4 honest-framing.

---

## 1. Rigid-fragment RMSD

Our `rigid_fragment_rmsd` uses a symmetry-aware MCS overlay with `GetBestRMS(symmetrizeConjugatedTerminalGroups=True)` after independent MMFF94 minimisation on both sides of the comparison. **Verdict:** This is **stricter** than the SOTA-default ligand-vs-crystal RMSD (TargetDiff / Pocket2Mol use full-ligand Kabsch without MCS, optionally with Hungarian atom matching on heavy atoms only). For our metallo-fragment tiles, the MCS convention is actually **more useful** than the SOTA convention because it isolates the small Pt(NH₃)₂-Cl₂ core that is the chemically active axis; full-ligand Kabsch would dilute that signal with periphery. **Action:** keep as-is. Document the MCS-vs-full-ligand distinction in §4.6 so reviewers understand the gap is intentional.

## 2. Ligand-vs-crystal RMSD

The SOTA standard (TargetDiff, Pocket2Mol, DiffSBDD, DecompDiff, DiffDock) reports ligand-vs-crystal RMSD after Kabsch + Hungarian atom matching on heavy atoms (no hydrogens), with success rates reported at 0.5 / 1.0 / 2.0 Å thresholds (2.0 Å = industry-standard "successful pose"). **Verdict:** We do **NOT** currently report ligand-vs-crystal RMSD because our pipeline does not dock into a reference pocket and we have no crystal reference for most of our 12-tile click-library. **Action (medium-priority):** add a `ligand_vs_crystal_rmsd` adapter post-Wf-134 that runs RDKit `Chem.rdMolAlign.GetBestRMS` against a curated set of metal-pdbqt-bound crystal ligands (1H36, 3EML, 3NY8, 4AAW, 4RLU already exist on disk per `metrics/by_metric/rigid_fragment_rmsd_*.json`). This closes the most important SOTA-metric gap.

## 3. AutoDock Vina docking score

Our `vina_kcal_per_mol` ships AutoDock Vina 1.2.7 (latest release), aligned with Eberhardt 2021 (`acs.jcim.1c00203`). **Verdict:** our Vina convention is identical to TargetDiff / Pocket2Mol / DecompDiff / DiffDock / DiffSBDD. Threshold interpretation in our docs (`-7` screening, `-8` wet-lab, `<-10` suspect) matches industry-standard review literature (vigyanllm.in 2026, drugchemsci 2024). **Critical caveat for our metallodrug pipeline:** Vina has NO metal-coordination term, so it systematically **underestimates** the binding affinity of our generated metal fragments. We already document this in `wf_boltz_affinity_fallback/` as a "model divergence" rather than "Vina error"; keep this framing. **Action:** none (already aligned).

## 4. Ertl SA score

Our `sa_score_ertl` uses the RDKit `Contrib/SA_Score/sascorer.py` reference implementation unchanged (Ertl & Schuffenhauer 2009, *J. Cheminform.*), with raw range [1, 10] (lower=better) and a `sa_score_to_unit` linear remap to [0, 1] (higher=better). **Verdict:** Our convention is **identical** to TargetDiff / Pocket2Mol / DecompDiff / MOSES / GuacaMol / DiffSBDD (all eight tier-1 papers use the same RDKit reference implementation unmodified). Our Lambda click-tile library mean SA of 2.93 is in the "easy to synthesise" regime and consistent with Pocket2Mol's published top-100 numbers. **Action:** none (already aligned). Honest-framing: SA 1-3 = easy, 4-6 = moderate, 7+ = hard (per drug-discovery convention).

## 5. Bickerton QED

Our `qed_scorer` uses RDKit `Chem.QED.qed(mol)` (Bickerton 2012 *Nature Chemistry* reference implementation unchanged), 8 desirability functions combined as a weighted geometric mean, continuous [0, 1]. **Verdict:** Our convention is **identical** to all tier-1 papers since 2014. **Critical caveat:** QED is calibrated for **oral drugs**, not metallodrugs — the metal itself is not in the ALERTS list, so Pt-coordinated molecules can score moderately high even though they have fundamentally different ADMET profiles. Our reported QED values are in [0.0, 0.6] range, below TargetDiff's 0.48 mean on CrossDocked2020, which is consistent with the MW > 500 QED penalty triggered by Pt(NH₃)₂Cl₂ + ligand fragments. **Action:** report QED as "Bickerton desirability, oral-drug-calibrated" with explicit honest-framing in §4.6, and label it as **not** a metallodrug ADMET predictor. We may want to add a `qed_metalcorrected` variant (e.g. additional desirability penalty for Pt/Pd/Au/Ir) post-Wf-134, but that's R17+ scope.

## 6. Diversity — Tanimoto / IntDiv

Our `diversity_tanimoto` (whole-molecule) uses 2048-bit Morgan-ECFP4 + mean pairwise (1 − Tanimoto), and we additionally expose `per_residue_diversity` (sub-pocket weighted Tanimoto from `per_residue_diversity.py`). **Verdict:** Our convention is **mostly aligned** with TargetDiff / Pocket2Mol / DiffSBDD (2048-bit ECFP4 + mean pairwise) but **not** identical to MOSES IntDiv₁ (1024-bit ECFP4) or IntDiv₂ (p=2 penalisation). Two IntDiv = 0.85 numbers from different papers can be **non-comparable** — this is the least-standardised of the six metrics. **Critical caveat for metallodrugs:** whole-molecule Tanimoto distances are dominated by the periphery, so our `per_residue_diversity` sub-pocket variant is **more useful** for the typical Lambda workflow. **Action:** document the 2048-vs-1024-bit and per-residue-vs-whole-molecule distinction in §4.6; prefer `per_residue_diversity` in tables; keep `diversity_tanimoto` for cross-paper comparability.

---

## Honest-framing punch-list for §3.5 / §4 (party-twenty-spirit)

1. **Rigid-fragment RMSD** — explicitly state that we use **MCS overlay** (stricter than the SOTA full-ligand Kabsch convention) and that this is intentional for metallo-fragment cores.
2. **Ligand-vs-crystal RMSD** — state that we do NOT report it (currently) because we lack crystal references for the click-tile library; flag as a gap for R17+.
3. **Vina score** — state that Vina has no metal-coordination term and therefore systematically underestimates metallodrug binding; document the model-divergence framing already in `wf_boltz_affinity_fallback/`.
4. **SA score** — state that the [1, 10] raw is the TargetDiff / Pocket2Mol convention; the [0, 1] `sa_score_to_unit` is a paper-friendly remap, not a different metric.
5. **QED** — state that QED is oral-drug-calibrated, not metallodrug-calibrated; report it as "Bickerton desirability" without claiming "drug-likeness" in the metallodrug sense.
6. **Diversity** — state the 2048-bit ECFP4 + mean pairwise (1 − Tanimoto) convention; note that IntDiv is not standardised across papers; mention per-residue sub-pocket variant.

---

## What's NOT in this survey (deferred to follow-up)

- **pIC50 / GI50** (Crippen + Hou 2007 descriptors) — covered separately in `metrics_v2.py:logp7_4` and `pic50_predictor.py`; not in this RMSD/Vina/SA/QED scope.
- **PoseBusters validity** (12 geometric checks) — `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`; per `wf_p0_metrics` / `wf_t30_p12_fg_veto`.
- **Wet-lab / experimental binding affinity** (Kd, IC₅₀) — out of scope for in-silico metrics.
- **SCScore / RDChiral retrosynthesis** — alternatives to Ertl SA; covered briefly in §4 table.

---

## File map

- `lit_table_RMSD_Vina_SA_QED.md` — full literature table (~400 lines; verbatim quotes + DOIs + URL provenance)
- `lit_table_RMSD_Vina_SA_QED.json` — structured form (10 papers per metric; ~250 lines)
- `synthesis_RMSD_Vina_SA_QED.md` — this file (1-paragraph verdicts + §3.5/§4 punch-list)