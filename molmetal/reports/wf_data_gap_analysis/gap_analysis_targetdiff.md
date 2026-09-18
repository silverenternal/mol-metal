# Data Gap Analysis — TargetDiff (Q1 benchmark) vs Mol-Metal

**Author:** WF-Data-Gap-Analysis subagent
**Date:** 2026-09-14
**Benchmark:** Guan et al., "3D Equivariant Diffusion for Target-Aware Molecule
Generation and Affinity Prediction" (TargetDiff, ICLR 2023).
**Test protocol reproduced:** 100 held-out pockets from CrossDocked2020
(filtered to RMSD < 1 Å, seq-id < 30 %), 100 ligands sampled per pocket,
AutoDock Vina Score / Min / Dock for binding affinity, QED + SA + Diversity
for drug-likeness, Jensen-Shannon divergence on bond-distance histograms for
geometric fidelity, rigid-fragment RMSD after MMFF relaxation for substructure
consistency, and ring-size distribution. 58.1 % / 57 % are the headline
"High Affinity" + "best on pocket" rates.

**Comparison framing:** TargetDiff is a **pure 3D geometric diffusion** baseline
trained end-to-end on CrossDocked2020. Mol-Metal is a **hybrid system** with
two generator arms — (i) a **Lambda-only typed proof-search** baseline and (ii)
a **CFM geometric retrain** path that is currently deferred (TODO-21). The
hybrid paradigm is fundamentally different: Lambda generates typed-reduction
proofs of valid SMILES with metal-priors as first-class constraints, while CFM
denoises 3D coordinates and atom types jointly. Side-by-side numbers are
**cite-only context**, NOT a head-to-head. The 7 protocol-mismatch flags
(M1–M7) below label every cell of the cite-only SOTA column.

---

## Per-metric gap table

Columns:
- **TargetDiff scale** = # pockets / # ligands per pocket / metric definition.
- **Mol-Metal current** = the largest directly-measured Mol-Metal run, plus its
  N pockets and seed count.
- **Gap** = the categorical delta.
- **Ship path** = concrete TODO/work item that closes (or explains deferral of)
  the gap.

| # | Metric | TargetDiff scale (pockets × seeds × mols/pocket) | Mol-Metal current (N pockets, seeds, measured?) | Gap | Ship path |
|---|---|---|---|---|---|
| 1 | **Vina Score** (kcal/mol, mean + median) | 100 × 1 × 100 = 10 000 evals, CrossDocked2020 hold-out, atom-vocab-restricted, paired pose | **Measured** at N=1 pocket × 1 seed × 50 mols (`round11_engine_parity`: 1h36 pocket, vina_mean = −7.35 kcal/mol); **NOT measured** at the 100-pocket scale | 100× scale gap; same physical engine + same pocket-family protocol (M1) | **Round-12:** N=10 × 3-seed (TODO-13). **Round-13:** N=100 × 3-seed (TODO-14). Reuse existing `molmetal/scripts/lambda_100pocket_sweep.py` harness + QVina-GPU parity-confirmed. No retrain needed — Lambda sampling path is live. |
| 2 | **Vina Min** (local-minimised score) | 100 × 1 × 100 | **NOT measured** end-to-end; MGLTools / UFF minimization step is staged but not wired into the parity harness | One-tooling gap, no retrain needed (M2) | **Round-12 B-1:** wire `meeko`+`obabel` minimize step into `round11_parity_n50.py` for 1h36 + 830c first, then scale. Estimate: 1 day. |
| 3 | **Vina Dock** (full re-dock score) | 100 × 1 × 100, exhaustiveness 8 / 32 | **Measured** as part of parity at N=1 × 50 (quickvina path); **NOT measured** at 100-pocket scale | 100× scale gap; same engine (M3) | **Round-13 only** (TODO-14). Wall-clock dominated by re-dock — QVina-GPU parity gives ~10× speedup. |
| 4 | **High Affinity** (% mols with Vina Dock < reference ligand, Avg + Med per pocket) | 100 × 1 × 100, paired per-pocket against ground-truth | **NOT measured** as a paired metric; reference-ligand Vina ground-truth is staged for 1h36 + 830c but the "fraction beats reference" comparison is not yet computed | Compute-only gap, no retrain (M4) | **Round-12 B-2:** add `high_affinity_rate` to per-pocket JSON; computed as (gen_dock < ref_dock). 0.5 day. |
| 5 | **QED** (Bickerton 2012) | 100 × 1 × 100 | **Measured** in Lambda sweep (`lambda_100pocket_sweep.py`) at proxy level; **NOT measured** on real CrossDocked decodings because Mol-Metal CFM is decoder-bound (R10 result: 0 sanitized graphs reached docking) | Scale gap closed by Lambda path; CFM path blocked by TODO-21 / WF-CFM-Retrain-Diagnose | **Round-12 A-3:** wire QED into `r4_lambda_only_run.py` aggregator. Already prototyped (`_qed` function present in `lambda_100pocket_sweep.py`). |
| 6 | **SA score** (Ertl & Schuffenhauer 2009) | 100 × 1 × 100, fragment-based | **Measured** at proxy level in Lambda sweep; SA scorer is installed (`molmetal_lam.sbdd_env.sa_score`) | Scale gap closed by Lambda path (M5) | **Round-12 A-3:** add SA to per-pocket CSV alongside QED. 0.5 day. |
| 7 | **Diversity** (mean pairwise Tanimoto, Morgan FP) | 100 × 1 × 100 | **NOT measured as Morgan-Tanimoto**; Mol-Metal reports `homotype_diversity` (typed-variable distance) and `diversity_tanimoto` (typed-variable hits / max) — these are **Lambda-native**, NOT Morgan FP. WF-Lambda-2 documents the methodological delta | Metric-definition mismatch (M6); Mol-Metal measure is defensible but not the TargetDiff axis | **Round-12 B-3:** add Morgan-ECFP4 Tanimoto as an optional 8th metric in `r4_lambda_only_run.py` for direct comparability. 1 day. |
| 8 | **Validity** (RDKit sanitization rate) | 100 × 1 × 100 | **Measured** at N=10 × 3 seeds in `r4_lambda_only_run.py` (`validity_rate` per cell); **NOT measured** at 100-pocket scale; the CFM arm has 0 sanitized graphs (decoder bug, R10 result) | 100× scale gap on Lambda arm; CFM arm blocked on TODO-21 | **Round-12 A-1:** extend `r4_lambda_only_run.py` to 10 pockets × 3 seeds; **Round-13 A-1:** to 100 × 3. No retrain. |
| 9 | **Jensen-Shannon divergence** on bond-distance histograms (C−C, C=C, C−N, C=N, C−O, C=O, C:C, C:N) | 100 × 1 × 100, distribution-vs-distribution per bond type | **NOT measured.** Mol-Metal Lambda is a typed-SMILES generator; it does not produce raw 3D coordinates by design (the proof-search lands in a β-NF, then RDKit embeds). Bond-distance histograms can be computed post-embed | Compute-only gap (M7); no retrain | **Round-12 B-4:** add RDKit UFF-embedded bond-distance JSD module + report per bond type on the 10-pocket subset. 1.5 days. Depends on R12 A-1 producing valid decodings. |
| 10 | **Rigid-fragment RMSD** (after MMFF) | 100 × 1 × 100 | **NOT measured.** MMFF optimization of generated mols → RMSD on non-rotatable fragments | No tooling gap (RDKit MMFF available) | **Round-13 B-1:** add post-MMFF RMSD module. 2 days. |
| 11 | **Ring-size distribution** (3-/4-/5-/6-/7-/8-/9-membered ring ratios) | 100 × 1 × 100 | **NOT measured.** Lambda does produce rings via typed construction; ring-size is a free observable | Compute-only gap | **Round-12 B-5:** add `ring_size_ratio` per-pocket. 0.5 day. |
| 12 | **CoM (center-of-mass) shift** vs reference ligand | 100 × 1 × 100 | **NOT measured**; needs paired reference-ligand 3D coordinate + generated-ligand 3D coordinate | Compute-only gap; depends on valid 3D embedding (RDKit ETKDG) | **Round-13 B-2:** compute CoM-shift post-RDKit-embed. 1 day. |
| 13 | **Steric clash count** (PoseCheck) | 100 × 1 × 100, ligand-pocket clash | **Measured** via `posebusters` adapter wired in Round-7 (posebusters installed via uv); **NOT measured** in any current pilot run | Tooling live, scale gap | **Round-12 B-6:** add `posebusters` pass-all metric to per-pocket JSON on the 10-pocket pilot. 1 day. |
| 14 | **Strain energy** (PoseCheck, UFF pre vs relaxed) | 100 × 1 × 100 | **NOT measured** in pilot runs; UFF is available via RDKit | Compute-only gap | **Round-13 B-3:** add strain-energy module. 2 days. |
| 15 | **Interaction fingerprint** (PLI / protein-ligand interactions, PoseCheck) | 100 × 1 × 100 | **NOT measured** in pilot runs | Tooling live, scale gap | **Round-13 B-4:** wire PoseCheck PLI fingerprints. 3 days. |
| 16 | **Lipinski** (rule-of-5 satisfied count) | 100 × 1 × 100 | **Measured** at proxy level in `lambda_100pocket_sweep.py` (`n_lipinski_pass`) | Scale gap closed by Lambda path; metric definition identical | **Round-12 A-3:** Lipinski is already in the sweep harness; promote to per-pocket CSV. 0.25 day. |
| 17 | **logP** (Crippen) | 100 × 1 × 100 | **Measured** in `r10_cfg_real_crossdocked.py` (per-decoding logp) and `train_fm_pocket.py` (per-batch) | Compute-only gap; metric identical | **Round-12 A-3:** add logP to per-pocket CSV. 0.25 day. |
| 18 | **Lipinski heavy-rule count (LPSK)** | reported in follow-ups like READ, not TargetDiff primary | **NOT measured** (different definition) | Out-of-scope (READ/IRDIFF metric, not TargetDiff) | Cite-only. **Post-Round-13**. |
| 19 | **Vina Energy per pocket** (sorted heatmap, Fig. 6) | 100 × 1 × 100, sorted bar | **NOT measured** at 100-pocket scale | Scale gap | **Round-13:** render sorted Vina-energy bar plot once full sweep completes. No new computation. |
| 20 | **Eval/Recon/Complete success** (% recon > % eval > % complete) | 100 × 1 × 100 | **Measured at decoder level** (R10 CFG: 64/96 finite raw → 0 sanitized → 0 docked). Lambda path reports `validity_rate` directly | Decoder-bound on CFM arm | **Round-12 / Round-13:** Lambda arm tracks this via `validity_rate`; CFM arm blocked on TODO-21. |
| 21 | **Lipinski pass rate / MolWt / TPSA / RotB** | not in TargetDiff primary table | **Mol-Metal-specific anticancer extensions** (TODO-15 already shipped; TPSA + RotB TODO per R13 markdown) | Out-of-scope for TargetDiff benchmark | **Round-13:** add MW-range flag 300–700 Da + TPSA 60–150 Å² + RotB<10 fraction per anticancer survey. |
| 22 | **Atom stability / Mol stability** (post-decoding RDKit stability) | reported as `atm_stable` / `mol_stable` (TargetDiff 0.94 / 0.25 example) | **NOT measured.** `mol_stable` is RDKit's per-atom valence check on generated graph | Tooling live (RDKit); compute gap | **Round-12 B-7:** add `mol_stable` / `atm_stable` to per-cell metrics. 0.5 day. |
| 23 | **CFM vs Lambda ablation** (Mol-Metal internal, not in TargetDiff) | n/a | **Measured** at N=10 × 3 seeds via `r10_cfg_real_crossdocked.py` + `r4_lambda_only_run.py` comparison; result: CFM is decoder-bound (0 docked mols), Lambda validity is 0.x for N=10×3 | n/a (Mol-Metal-only) | **Round-12 / Round-13:** report both arms with explicit "decoder-bound" framing for CFM. |
| 24 | **pIC50 (MetalCytoToxDB)** | not in TargetDiff (general SBDD, not anticancer) | **Measured** (TODO-18 retrain + REINVENT4 multiproperty bridge shipped); `evaluate_anticancer_metrics.py` reports bound-aware accuracy | Metric definition out-of-band | Cite-only SOTA context for anticancer (PlatinAI / MetalCytoToxDB survey, memory `sota_papers_r4.md`). **Round-12/13:** report in supplementary. |
| 25 | **MCF-7 / cytotox viability** (MetalCytoToxDB) | not in TargetDiff | **Measured** in TODO-18 / TODO-15 pipeline; learning predictor reports pIC50 against 8 cell lines | Out-of-band | Supplementary, not main table. |

---

## Summary

- **n_metrics in TargetDiff:** 25 (cells 1–25 above; #18 and #21 are anticancer
  / follow-up extensions not in the TargetDiff primary table — they are listed
  for honest cross-paper scope)
- **n_metrics Mol-Metal has at full scale (≥100 pockets):** **0**. No 100-pocket
  sweep has been executed in Mol-Metal. The closest is `lambda_100pocket_sweep.py`
  which is **citation-only / proxy** (it does not run real Vina on real pockets).
- **n_metrics Mol-Metal can ship by Round-12** (N=10 × 3-seed budget, 60 min wall):
  **9** — Vina Score (#1), QED (#5), SA (#6), Diversity as Morgan-ECFP4 (#7,
  after 1-day patch), Validity (#8), JSD bond-distance (#9, after R12 decodings
  exist), Ring-size distribution (#11), Lipinski (#16), logP (#17). 7 of these
  are compute-only patches; #1 requires QVina-GPU docking time on 30 evals
  (10 × 3).
- **n_metrics Mol-Metal can ship by Round-13** (N=100 × 3-seed full sweep):
  **17** — all Round-12 metrics + Vina Min (#2), Vina Dock (#3), High Affinity
  (#4), Rigid-fragment RMSD (#10), CoM shift (#12), Steric clash (#13), Strain
  energy (#14), Interaction fingerprint (#15). 14 of these are compute-only or
  PoseCheck-API calls; #15 (PLI fingerprints) is the heaviest at 3 days.
- **n_metrics deferred to post-Round-13:** **8** — #18 (LPSK heavy-rule, READ-
  specific), #21 (TPSA / RotB / MW-range anticancer extensions per TODO-15),
  #22 (mol-stable / atm-stable, depends on CFM arm recovering from decoder bug),
  #23 (CFM arm ablation, requires TODO-21 CFM geometric retrain), #24 (pIC50,
  anticancer-only and out-of-band for TargetDiff comparison), #25 (cytotox
  viability, anticancer-only). #19 (sorted-Vina-energy heatmap) is a figure
  not a metric and ships automatically with Round-13.

### Round-12 budget feasibility (N=10 × 3 seeds × 2 min/pocket-sid ≈ 60 min)
- Wall-clock budget holds for QVina-GPU docking at exhaustiveness 8.
- Compute-only metric patches (#5, #6, #7, #8, #11, #16, #17) total ~3.5 dev days.
- JSD bond-distance (#9) and ring-size (#11) ship as part of the same patch as #8.
- PoseBusters (#13) ships as part of `evaluate_generated_poses.py` extension.

### Round-13 budget feasibility (N=100 × 3 seeds × 2 min ≈ 10 h wall)
- QVina-GPU parity (Round-11 result: r=0.998) confirms the docking engine is
  ready at scale.
- PoseCheck module integration (3 days) and Vina-Dock (#3, exhaustiveness 8) are
  the long pole.

### Hard blockers (cannot ship without unblock)
- **TODO-21 (CFM geometric retrain):** affects the CFM arm only. Required for
  #22 (mol-stable), #23 (CFM-vs-Lambda ablation). Currently **deferred**; only
  ships if WF-CFM-Retrain-Diagnose (5000-step retrain) succeeds.
- **TODO-05 (REINVENT4 environment install):** affects anticancer extensions
  (#24, #25). **Env-blocked** per Round-8 summary.
- **TODO-07 (R4C sweep data + env):** the 100-pocket CrossDocked2020 staging
  is partial; current `crossdocked100_manifest.csv` covers 100 PDB entries
  but receptor preparation is not complete for all 100.

---

## Honest caveats (per 7 protocol-mismatch flags M1–M7)

- **M1 — Sampling paradigm:** TargetDiff is a pure 3D diffusion model; Mol-Metal
  Lambda is a typed proof-search that emits SMILES → RDKit embeds → docking.
  The intermediate representation differs (3D denoising vs. SMILES proof), so
  any Vina comparison is structurally indirect.
- **M2 — Vina Min toolchain:** TargetDiff uses MGLTools / UFF minimization via
  Meeko. Mol-Metal parity harness currently uses `obabel` minimization; the
  numerical delta between the two minimization paths is unmeasured (the
  Round-11 parity measured score-only paths, not minimization paths).
- **M3 — Vina Dock exhaustiveness:** TargetDiff default is 32; Round-11 parity
  used exhaustiveness 8 for speed. Scaling to exhaustiveness 32 at N=100 × 3
  requires ~4× more wall time.
- **M4 — High-Affinity reference ligand:** TargetDiff pairs generated-mol Vina
  Dock against the co-crystal reference ligand's Vina Dock. Mol-Metal's
  reference-ligand pose is staged only for 1h36 + 830c; full 100-pocket
  reference docking is not yet executed.
- **M5 — SA scorer:** TargetDiff uses Ertl & Schuffenhauer 2009 with the
  fpscores.pkl.gz pickle. Mol-Metal uses the same scorer via
  `molmetal_lam.sbdd_env.sa_score.sa_score_ertl`; numerical agreement not
  independently verified.
- **M6 — Diversity metric definition:** TargetDiff uses Morgan-ECFP4 Tanimoto
  (1 − similarity, averaged over all pairs per pocket). Mol-Metal reports
  `homotype_diversity` (typed-variable + β-reduction + click-fires) and
  `diversity_tanimoto` (typed-variable hits / max). The Mol-Metal measures are
  NOT directly comparable to Morgan-Tanimoto; Round-12 patch will add Morgan
  for direct comparability.
- **M7 — Bond-distance JSD:** TargetDiff reports JSD over 8 bond types on
  raw 3D coordinates. Mol-Metal Lambda produces SMILES, then RDKit ETKDG
  embeds them; the JSD on the post-embedded geometry reflects the **embedder's**
  bond-length priors as much as the **generator's** geometric fidelity. This
  is a structural disadvantage for Lambda in this metric — flagged honestly.

### Cite-only framing reminder

The SOTA table cells in Mol-Metal's paper (TODO-13 / TODO-14) will be **cite-only**
context, not hypothesis-test samples. Per the 7 mismatch flags above, no direct
significance test against TargetDiff / Pocket2Mol / DecompDiff / DiffSBDD /
TransDiffSBDD numbers is statistically defensible. Cite-only is the canonical
treatment per the `molmetal-state-2026-09-12` memory + the Round-4 final
decision (`molmetal_round4_final.md`).

---

## Files referenced

- /home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/README.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/scripts/evaluate_diffusion.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/round11_parity_n50.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/scripts/lambda_100pocket_sweep.py
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/r10_cfg_real_crossdocked/README.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/round11_engine_parity/parity_metrics.json
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/13_top_journal_pilot_r12.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/14_full_100pocket_paper_r13.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/15_anticancer_metric_suite_r11b.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/17_aggregate_weak_impls_and_pending.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/20_post_r10_r11_action_plan.md
- /home/hugo/codes/try_triton_on_rocm/TODO/pending/21_lambda_model_coupling.md
- /home/hugo/codes/try_triton_on_rocm/molmetal_lam/sbdd_env/sa_score.py
- /home/hugo/codes/try_triton_on_rocm/molmetal_lam/sbdd_env/vina_adapter.py

---

## Honest accounting

- **Headline:** 0/25 metrics at TargetDiff's 100-pocket scale today; 9/25 by
  Round-12 (compute-only patches, no CFM retrain); 17/25 by Round-13 (full
  sweep + PoseCheck); 25/25 only after TODO-21 CFM retrain (post-Round-13).
- **Cite-only SOTA:** the paper's main table will be TargetDiff / Pocket2Mol /
  DecompDiff / DiffSBDD / TransDiffSBDD numbers quoted from their respective
  papers with the 7 protocol-mismatch flags M1–M7 annotated.
- **Mol-Metal-native metrics** (homotype_diversity, closure-theorem depth,
  click-rule compliance, metal-prior compliance, pIC50 learning predictor)
  are the **primary** contribution of Mol-Metal — they are not in the
  TargetDiff comparison and should be reported as **separate** tables in
  the paper, not in the cite-only SOTA column.
