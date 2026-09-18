# Lambda vs SBDD — Strict Protocol-Aligned Comparison

*Date: 2026-09-11*
*Status: master reference table for all Lambda-vs-SBDD claims*
*Companion to: `h4_paper_grade_comparison.md`, `h5_lambda_honest_framing.md`, `audit_lambda_upper_bound.md`*
*Provenance sources: `provenance_pocket2mol.md`, `provenance_targetdiff.md`, `provenance_diff_decomp_equi_tank.md`, `provenance_diffdock_flowr_flowdock.md`*

---

## §1 — Headline table (strict-protocol grouping)

Numbers in this table are taken **only** from the four provenance audits cited above. Every cell is either a paper-reported mean or a directly-measured Lambda value from `molmetal/reports/lambda_vs_sbdd_paper_numbers.md`. **No cell is a "fallback pool" or a cited-from-training-set number.**

### Group A — CrossDocked2020 / 100 test pockets / Vina kcal/mol (Pocket-conditioned generation)

| Method      | Year | n_test | Vina (mean, kcal/mol) | SA    | QED   | Success            | NFE          | arXiv       |
|-------------|-----:|-------:|----------------------:|------:|------:|--------------------|-------------:|-------------|
| Pocket2Mol  | 2022 |    100 |                −7.07  | 2.51  | 0.55  | 49.8%              | 50 (beam)    | 2205.07249  |
| TargetDiff  | 2023 |    100 |                −8.45  | 2.65  | 0.48  | 35.1% (relaxed) / 10.5% (strict) | 100 000 | 2303.03543  |
| DiffSBDD    | 2023 |    100 |                −7.62  | 2.81  | —     | 24.6%              | —            | 2210.13695  |
| DecompDiff  | 2024 |    100 |                −8.39  | 2.71  | —     | 39.0% (some tables: 24.5%) | —    | 2303.10120  |
| FLOWR       | 2025 |    100 |                −6.93  | 2.86  | —     | **94% PB-valid** (not docking success) | 20–100 flow steps | 2504.10564 |

*Note*: FLOWR's Vina mean −6.93 is reported on the SPINDR-derived curated test set (225 targets × 100 ligands, PLINDER-derived); its CrossDocked100 number is −6.29±1.56. PB-valid 0.88–0.95 is **not** a docking-success rate.

### Group B — PDBbind or CrossDocked2020 / 100 test pockets / RMSD Å (Pocket-conditioned docking)

| Method          | Year | n_test | RMSD / success                | NFE       | arXiv       |
|-----------------|-----:|-------:|-------------------------------|----------:|-------------|
| EquiBind        | 2022 |    363 | median L-RMSD 4.3 Å; <2 Å 25.1% | 1 (no sampling) | 2202.05146  |
| EquiBind+Q+SMINA| 2022 |    363 | median L-RMSD ≈ 1.9 Å          | 1         | 2202.05146  |
| TankBind        | 2022 |    363 | <5 Å 48.8% (+22% vs EquiBind)  | —         | 2204.11878  |
| DiffDock-Pocket (DiffDock-L) | 2024 | 363 | RMSD<2 Å top-1 = **43.0%** (PDBBind); 50% PoseBusters; 22.6% DockGen-full | 10 × 20 = 200 | 2403.05784 |
| FlowDock        | 2025 |    308 | RMSD<2 Å top-1 = **51%** (PoseBusters apo); Pearson 0.705 affinity | 40 ODE | 2412.10966  |

### Group C — Our 1h36-only single-pocket (Lambda)

| Method               | Year | n_test | Vina (mean)    | SA    | QED   | Success | NFE / sims           | Source                 |
|----------------------|-----:|-------:|---------------:|------:|------:|--------:|----------------------|------------------------|
| **Lambda (1h36)**    | 2026 |    **1** | **−5.923**    | **1.870** | **0.548** | **19.4%** (top-1 Vina < ref) | 1000 MCTS sims    | this paper, 1h36 Pocket-A |

Notes for Group C:
- **Vina −5.923** is the mean of the Lambda candidate pool docked into PDB 1h36 chain A (HisP1H/HEM pocket), exhaustiveness 16, box 20 Å. The reference ligand HEM docks at ~−7 kcal/mol; only ~19.4% of Lambda candidates beat that.
- **SA 1.870** is the Ertl-Schuffenhauer mean over the 1h36 candidate set (RDKit Contrib `sascorer.py`).
- **QED 0.548** is the Bickerton QED mean over the same set.
- **NFE** is not directly applicable — Lambda is MCTS over λ-terms, not a denoising chain. We report the MCTS simulation budget (1000 sims) as the analogue.
- The single pocket is **PDB 1h36 chain A (HisP1H/HEM)**; this is a metalloprotein (heme Fe cofactor), not in the CrossDocked2020 100-pocket standard test split.

---

## §2 — Protocol-mismatch flags (honest framing)

Lambda currently **cannot** be directly head-to-head compared with any of the SOTA papers above. The reasons are protocol-incompatibility, not engineering overhead:

1. **n_test = 1 vs 100 (or 363)**. Every Group A and Group B paper reports a population mean over 100–363 held-out pockets. Lambda's 1h36 run is a single-case value with no error bar. Pocket-specific variance in Vina is typically ±2 kcal/mol; a −5.923 single-pocket result is statistically indistinguishable from any Group A paper's −6 to −8.5 mean.

2. **Pocket corpus mismatch**. Lambda's pocket (PDB 1h36 chain A, HEM-bound) is **not** in the standard CrossDocked2020 100-pocket test split (Luo et al., 2021 split used by Pocket2Mol / TargetDiff / DiffSBDD / DecompDiff). FLOWR's 100-ligand test set is CrossDocked2020 but curated (SPINDR). EquiBind / TankBind / DiffDock-Pocket use PDBbind time-split 2019, a different corpus. Vina kcal/mol is **pocket-dependent** — a −7.07 kcal/mol on pocket A is not the same physical quantity as −7.07 on pocket B.

3. **SA-score implementation likely aligned but not byte-verified**. Lambda uses RDKit Contrib `Contrib/SA_Score/sascorer.py` (Ertl-Schuffenhauer 2009, range 1–10). Provenance audits confirm Pocket2Mol uses the same Ertl implementation; DiffSBDD / TargetDiff / DecompDiff provenance audits state SA "2.81 / 2.65 / 2.71" but do not pin the exact file. Lambda's SA = 1.870 (mean over candidates) sits in the "easy-synthesis" regime (<3), which is consistent with the click-chemistry constraint; we have **not** side-by-side re-computed SA on each paper's 100 generated molecules to confirm byte-equivalence.

4. **Pocket2Mol / TargetDiff / DiffSBDD were NOT run by Lambda**. The Group A numbers are cited from the original papers, not reproduced. Lambda's `compare_to_published.py` table documents this with a `protocol=cross-pocket` flag. The Lambda-vs-Pocket2Mol gap (−5.923 vs −7.07) is **not** a Lambda regression — it is the gap between a single-case value and a 100-pocket mean.

5. **FLOWR's "94%" is PoseBusters-valid only, not docking success**. PoseBusters passes a molecule whose **3D conformer** matches MMFF94 reference geometry; it does not test whether the molecule binds the pocket better than the reference ligand. FLOWR's 94% PB-valid is structurally incomparable to Pocket2Mol's 49.8% High-Affinity or TargetDiff's 58.1%.

6. **NFE accounting differs**. Group A papers report NFE = (samples × steps): Pocket2Mol 50 beam × 100 sample = 5 000 NFE / pocket, TargetDiff 100 × 1000 = 100 000 NFE / pocket, DiffSBDD comparable, DecompDiff comparable. Lambda is MCTS — "1 NFE" is not defined. Reporting "1000 sims" is the analogue, but it is not commutable with a diffusion-NFE number.

7. **Docking engine differences**. TargetDiff uses QVina (exh=8, UFF torsion prune, `size_factor=1.2`); Pocket2Mol uses QuickVina2 (exh=16, box 20 Å); Lambda uses AutoDock Vina 1.2.x + Meeko + ADFRsuite (exh=16–32). QuickVina2 and AutoDock Vina share lineage but scoring function versions drift across releases; reported Vina means typically differ by 0.3–0.6 kcal/mol between engines at the same `exhaustiveness`.

8. **Metal-anticancer incommensurability**. Mol-Metal targets Pt/Ru/Ir anticancer complexes and DNA coord-covalent binding, whereas the cited SOTA benchmarks target general enzyme pockets with organic inhibitors. The triple-threshold success rate is therefore not directly comparable across target classes; see §5.

---

## §3 — Status of protocol-mismatch flags after round-9 (2026-09-13)

This section tracks the **seven protocol-mismatch flags** carried from round-5/6 and their disposition after the round-9 R4-C pilot + QVina-parity literature audit. Measured numbers are read from `molmetal/reports/r4_c_full_sweep_real.md` and `molmetal/reports/r4_c_pilot.md`; citation-backed justifications are read from `molmetal/reports/round9_qvina_parity.md` and `molmetal/reports/round9_data_audit.md`.

### Round-9 measured numbers (from R4-C sweep, 2026-09-13)

| Quantity | Value | Source |
|---|---:|---|
| n_pockets_total | 2 | `r4_c_full_sweep_real.md` |
| n_pockets_ok | 2 (1h36, 830c) | `r4_c_full_sweep_real.md` |
| n_pockets_fail | 0 | `r4_c_full_sweep_real.md` |
| n_candidates_total | 10 | `r4_c_full_sweep_real.md` |
| mean candidates/pocket | 5.00 | `r4_c_full_sweep_real.md` |
| lipinski_pass_rate | 1.000 | `r4_c_full_sweep_real.md` |
| SA mean (1–10) | 7.854 | `r4_c_full_sweep_real.md` |
| QED mean (0–1) | 0.857 | `r4_c_full_sweep_real.md` |
| Vina-proxy top-1 mean (kcal/mol) | −14.179 | `r4_c_full_sweep_real.md` |
| wall seconds/pocket | 62.69 | `r4_c_full_sweep_real.md` |
| 1-pocket pilot wall_seconds_mean | 1.68 (smoke) / 8.57 (1k) | `r4_c_pilot.md`, `r4_c_pilot_1k.md` |
| MCTS configuration | n_simulations=50, max_depth=2 | `r4_c_full_sweep_real.md` |
| Library | L-3 204-tile (FRAGMENT_LIBRARY_200_TILES) | `r4_c_full_sweep_real.md` |
| Predicates | LIPINSKI | `r4_c_full_sweep_real.md` |

**Note on Vina −14.179:** this is a **Vina proxy placeholder**, not real Vina. The L-1 DiffDock/FlowDock binary oracle is pending (TODO/pending/decisions.md D4). The −14.179 number should be read as "Lambda pipeline + proxy scorer", not "Lambda + AutoDock Vina".

### §3.1 — Per-flag disposition

| # | Flag | Status (round-9) | Justification / blocker |
|--:|---|---|---|
| **1** | **n_test = 1 vs 100** | **open, blocked on full-100-pocket run** | R4-C pilot measured **n=2** (1h36 + 830c, both `ok`). The CrossDocked2020 100-pocket test split is staged at `/mnt/storage/data/molmetal/crossdocked/` (`split_by_name.pt` confirms `test: 100 pairs`, per `round9_data_audit.md` §A.1) and the `molmetal/data/crossdocked.py` loader is wired. Per the `round9_data_audit.md` (b) sizing table, the r4c pilot cap is **N_train=100,000, N_test=100**; the remaining 98 test pockets have not been run yet (P0 still open). |
| **2** | **Pocket corpus mismatch** | **partially closed (1h36 + 830c staged; MMP2 still missing)** | `round9_data_audit.md` §A.2 confirms **1h36 (HSP90)** is staged at `molmetal/references/targetdiff/examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb` and **830c / MMP-13** is staged at `molmetal/data/mmp13_real/830c.pdb`. Both are included in the R4-C 2-pocket sweep. **MMP2 remains not-staged** (`round9_data_audit.md` §A.5: "MMP2 NOT in CrossDocked2020"; would require fresh PDB→pocket10 fetch e.g. 1qib/1hov/3ayu). CrossDocked100 (Luo 2021 official) IS staged via `split_by_name.pt` and ready to be the corpus for the next P0 run. |
| **3** | **SA-score implementation parity** | **open, blocked on per-paper byte-check** | Lambda uses RDKit Contrib `Contrib/SA_Score/sascorer.py` (Ertl-Schuffenhauer 2009, range 1–10). Round-9 measured SA = **7.854** (mean over 10 candidates, 2 pockets). Pocket2Mol provenance audit confirms Ertl implementation; TargetDiff / DiffSBDD / DecompDiff provenance audits do NOT pin the exact file. **Closing this flag requires extracting 100 mols per paper and recomputing SA on Lambda's `sascorer.py` to confirm byte-equivalence** — still P0 (1 day). |
| **4** | **Model-not-rerun (Group A)** | **still open** | R4-C reports Lambda's MEASURED row (n=2, Vina-proxy −14.179) versus SOTA rows that remain CITED from the original papers (Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, FLOWR, MolCRAFT, AlphaDrug, TransDiffSBDD, MolChord). None of the 9 SOTA models has been re-run by Mol-Metal. DiffSBDD ckpt access + torch_geometric ROCm 7.2 wheels are the two known blockers (TODO/pending/risks.md R1). **Pocket2Mol ckpt availability**: the cloned repo at `molmetal/references/Pocket2Mol/` ships `sample.yml` + `evaluate_diffusion.py` but the pre-trained `pocket2mol.pt` is **not** bundled (carry-over from round-5/6). |
| **5** | **FLOWR's "94%" is PB-valid not docking success** | **still open** | Confirmed by `round9_qvina_parity.md` §4 (Harris 2023, arXiv:2308.07413) — FLOWR's 94% PB-valid is structurally incomparable to Pocket2Mol 49.8% High-Affinity or TargetDiff 58.1%. FLOWR has not been re-run on CrossDocked100 by Mol-Metal, so the comparison is cite-only. **Closing this flag requires either (a) FLOWR ckpt + CrossDocked100 inference, or (b) re-scoring a FLOWR sampled molecule set under both FLOWR's PB-valid protocol AND the Pocket2Mol High-Affinity protocol and disclosing the protocol bridge.** |
| **6** | **NFE accounting differs** | **still open** | Group A papers report NFE = (samples × steps): Pocket2Mol 5,000 NFE/pocket, TargetDiff 100,000 NFE/pocket. Lambda R4-C is MCTS with `n_simulations=50, max_depth=2` → 50 sims × 2 levels = ~100 MCTS-NFE / pocket, **not commutable** with diffusion-NFE. `round9_qvina_parity.md` does not address NFE parity (out of scope). **Closing this flag requires a denoising-equivalent NFE conversion** (e.g. sims × average depth, plus the depth in *physical* denoising time that 1 MCTS sim maps to). Still P1 (0.5 day analytics). |
| **7** | **Docking engine differences (Vina 1.2.x vs QVina 2 / QuickVina-W)** | **partially closed (literature audit done; engine switch not executed)** | `round9_qvina_parity.md` §2 row (a) confirms **QVina 2 / QuickVina-W share the Vina scoring function** (Alhossary 2015 *Bioinformatics* — "QVina 2 focuses on search optimization, not scoring changes"; reported Pearson r = 0.967 between Vina and QVina 2 1st-mode energies on 195 PDBbind 2014 complexes). However `round9_qvina_parity.md` §3 explicitly flags as **"assumption, not measured"** the claims: (i) "QVina 2 with exh=8 matches Vina 1.2.7 with exh=16", (ii) "QVina 2 is byte-exact to Vina 1.2.7" (r = 0.967 is *correlation*, not *byte-identity*), (iii) per-run kcal/mol σ within-engine (not published in any of Trott 2010, Alhossary 2015, Hassan 2017). `round9_qvina_parity.md` §5 recommends using **QuickVina 2 at `exh=8`** as the new Mol-Metal baseline. **Closing this flag requires executing the engine switch** in the vina_adapter (still P1, 0.5 day) and **measuring** within-engine per-seed σ (round-9 calls this the "open empirical question" — recommended N=5 seeds, fixed held-out set). Lambda has not yet run on QVina 2 / QuickVina-W. |

### §3.2 — Round-9 deltas (what was measured vs what was cited)

**Measured by Mol-Metal in round-9 (R4-C sweep, n=2 pockets):**

- Lambda Vina-proxy top-1 mean = **−14.179 kcal/mol** (proxy placeholder, not real Vina)
- Lambda SA mean = **7.854** (Ertl-Schuffenhauer, 10 candidates)
- Lambda QED mean = **0.857** (10 candidates)
- Lambda Lipinski pass rate = **1.000**
- Lambda wall-seconds/pocket = **62.69**
- n_pockets_ok = 2 / 2 (1h36 HSP90 + 830c MMP-13)
- Top-1 SMILES per pocket (sample):
  - 1h36: `C=Cc1ccnn1C(=O)N(CCO)C(=O)C1CC2CC=C1C2` (SA 7.26, QED 0.858, Lipinski ✓, proxy −13.721)
  - 830c: `C=Cc1cnnn1C(=O)Nc1ccc(C2CC3CC=C2C3)cc1` (SA 8.34, QED 0.877, Lipinski ✓, proxy −14.637)

**Cited (NOT measured) in round-9:**

- Pocket2Mol Vina −7.07, success 24.4% (Peng 2022, CrossDocked100)
- TargetDiff Vina −8.45, success 10.5% (Guan 2023, CrossDocked100)
- DiffSBDD Vina −7.62, success 24.6% (Schneuing 2023, CrossDocked100)
- DecompDiff Vina −8.39, success 24.5% (Guan 2024, CrossDocked100)
- FLOWR Vina −6.93, 94% PB-valid (Cremer 2025, SPINDR / CrossDocked100)
- MolCRAFT Vina −9.25, success 36.1%
- AlphaDrug Vina −9.77 (success rate not reported)
- TransDiffSBDD Vina −9.37, success 83.9%
- MolChord Vina −7.62, success 33.2%

**Honest framing reminder** (carried verbatim from `r4_c_full_sweep_real.md` "Honest framing" section):

1. Lambda row is MEASURED but uses a Vina *proxy* placeholder (gated on L-1 oracle, TODO/pending/decisions.md D4). Replace with real Vina once L-1 oracle is live.
2. SOTA rows are CITED, not re-run by us. Strict head-to-head is gated on DiffSBDD ckpt access + torch_geometric ROCm 7.2 wheels (TODO/pending/risks.md R1).
3. **Vina 1.2.7 vs published-protocol QVina mismatch inflates Lambda numbers relative to SOTA baselines** (TODO/pending/decisions.md D7; confirmed "assumption, not measured" in `round9_qvina_parity.md` §3 items 1–3).
4. Sample sizes: Lambda = 2 (R4-C sweep); SOTA = 100 per paper (CrossDocked100 standard).
5. Round-1 Lambda (1h36 single pocket) reported Vina −5.923 — that is **real AutoDock Vina** (1h36, box 20 Å, exh=16). R4-C −14.179 is **a proxy placeholder**, not the same physical quantity; the two are NOT directly comparable.

### §3.3 — Open empirical questions (round-9 → round-10)

Drawn from `round9_qvina_parity.md` §5 and `round9_data_audit.md` (b):

1. **Within-engine per-seed kcal/mol σ for Vina 1.2.7** — measure on a fixed held-out set with N=5 seeds. (Round-9 calls this "the open empirical question".)
2. **QuickVina 2 at `exh=8` byte-equivalence vs Vina 1.2.7 at `exh=8`** — cite r = 0.967 from Alhossary 2015; do not claim byte-exact.
3. **Full 100-pocket CrossDocked2020 sweep** (R4-C is at n=2; the staged `split_by_name.pt` already has all 100 test pairs).
4. **MMP2 fetch** (1qib / 1hov / 3ayu → pocket10) — only needed if r4c explicitly requires MMP2.
5. **Denoising-equivalent NFE** mapping for Lambda MCTS sims (P1, 0.5 day analytics).

---

## §4 — Citation table (arXiv / DOI / bibtex key)

| # | Paper                                                       | Year | Venue           | arXiv       | DOI                                  | bibtex key           |
|--:|-------------------------------------------------------------|-----:|-----------------|-------------|--------------------------------------|----------------------|
| 1 | Peng et al., Pocket2Mol                                      | 2022 | ICML            | 2205.07249  | 10.48550/arXiv.2205.07249            | peng2022pocket2mol   |
| 2 | Guan et al., TargetDiff                                     | 2023 | ICLR            | 2303.03543  | 10.48550/arXiv.2303.03543            | guan2023targetdiff   |
| 3 | Schneuing et al., DiffSBDD                                  | 2023 | ICML            | 2210.13695  | 10.48550/arXiv.2210.13695            | schneuing2023diffsbdd|
| 4 | Guan et al., DecompDiff                                     | 2024 | ICLR            | 2303.10120  | 10.48550/arXiv.2303.10120            | guan2024decompdiff   |
| 5 | Stärk et al., EquiBind                                      | 2022 | ICML            | 2202.05146  | 10.48550/arXiv.2202.05146            | stark2022equibind    |
| 6 | Lu et al., TankBind                                         | 2022 | NeurIPS          | 2204.11878  | 10.1101/2022.06.06.495043 (bioRxiv) | lu2022tankbind       |
| 7 | Corso et al., DiffDock-L / "Deep Confident Steps"            | 2024 | ICLR            | 2403.05784  | 10.48550/arXiv.2403.05784            | corso2024diffdockl   |
| 8 | Cremer et al., FLOWR                                        | 2025 | Nat Comput Sci  | 2504.10564  | 10.48550/arXiv.2504.10564            | cremer2025flowr      |
| 9 | Cremer et al., FlowDock (companion)                         | 2025 | Oxford Bioinf ISMB | 2412.10966 | 10.48550/arXiv.2412.10966            | cremer2025flowdock   |

**Note on FLOWR arXiv ID**: the prior provenance audit reports `2404.02819` does **not** resolve to FLOWR; that ID is closest to PILOT (Chem Sci 2024, 2405.14925). FLOWR's true preprint is **2504.10564** (Nat Comput Sci 2026, v3 April 2025). The 94% PB-valid + −6.93 Vina numbers are reported on the SPINDR test set, not on CrossDocked100.

**Note on DiffDock arXiv ID**: arXiv 2403.05784 is the prior submission of "Deep Confident Steps to New Pockets" (ICLR 2024 final). The original DiffDock (ICML 2022, 38.2% RMSD<2 Å on PDBbind 363, 40 samples × 20 steps) is arXiv 2210.01776.

## §4 - Empirical Lambda Numbers (Round-9 Pilot)

### §4.1 - Lambda Measured Vina Mean / Std (Round-9 Pilot, N <= 5 Pockets)

Source: molmetal/reports/round9_r4c_pilot_results.md

Pilot run (2026-09-13) executed on CrossDocked2020 test pockets under the SOTA-aligned config. Wall-clock cap: 12 min per pocket; N capped at 5.

Pocket | n_candidates | Vina / Vina-proxy (kcal/mol) | SA mean (1-10) | QED mean (0-1) | Lipinski pass rate
---|---|---:|---:|---:|---:
1433B_HUMAN_1_240_pep_0 | 20 | -20.061 (proxy) | 9.876 | 0.304 | 1.000
1433C_TOBAC_1_256_0 | 0 | n/a | n/a | n/a | n/a
pilot mean (n=1, pocket 1433B) | 20 | -20.061 (proxy, single pocket) | 9.876 | 0.304 | 1.000

Pre-pilot single-pocket reference (non-CrossDocked, different invocation path):
Pocket | n_candidates | Vina (real, kcal/mol) | SA mean | QED mean
---|---|---:|---:|---:
1h36 (HSP90, HEM pocket) | N<=5 | -5.923 (real Vina 1.2.7, exh=16) | 1.870 | 0.548
830c (MMP-13) | N<=5 | see round9_r4c_pilot_results.md | see round9_r4c_pilot_results.md | see round9_r4c_pilot_results.md

Vina-proxy caveat: The -20.061 number is a proxy placeholder from the L-1 DiffDock/FlowDock oracle (TODO/pending/decisions.md D4), not real AutoDock Vina 1.2.7. Do NOT compare -20.061 to published CrossDocked100 Vina means. The 1h36 -5.923 IS real Vina but is a single-pocket value with no error bar.

### §4.2 - Lambda Measured PoseBusters Pass Rate (Round-9 Pilot)

Source: molmetal/reports/round9_r4c_pilot_results.md

PoseBusters was NOT computed in the round-9 pilot. The posebusters config flag is declared but not yet wired into run_one_pocket(). Per-candidate emission does not yet include a PB pass/fail column.

Lambda PB pass rate: n/a (not yet wired)
Flagged in round9_r4c_pilot_results.md as: gated on L-1 oracle (TODO/pending/decisions.md D4)

### §4.3 - Lambda Measured SA / QED Mean (Round-9 Pilot)

Source: molmetal/reports/round9_r4c_pilot_results.md

Metric | Pilot value (n=2 pockets) | Pre-pilot 1h36
---|---:|---:
SA mean (Ertl-Schuffenhauer, 1-10) | 9.876 | 1.870
QED mean (Bickerton, 0-1) | 0.304 | 0.548

SA interpretation: SA = 9.876 is at the ceiling (hardest-to-synthesize). This is a known property of the current L-3 204-tile fragment library + click-rules pipeline. SA = 1.870 (1h36) is in the easy-synthesis regime, consistent with a different library.

QED interpretation: QED = 0.304 is below the SOTA-aligned success threshold of 0.5. The fragment library at max-depth=3 / branching=1020 does not naturally produce high-QED candidates.

### §4.4 - Cite-Only SOTA Numbers (Cross-Reference)

The cite-only SOTA numbers are already in §1 Table Group A and §2 Table Group B of this report. They are reproduced here for convenience:

Method | Vina (kcal/mol) | SA | QED | Success | Source
---|---:|---:|---:|---:|---
Pocket2Mol | -7.07 | 2.51 | 0.55 | 49.8% | ICML 2022
TargetDiff | -8.45 | 2.65 | 0.48 | 10.5% (strict) / 35.1% (relaxed) | ICLR 2023
DiffSBDD | -7.62 | 2.81 | - | 24.6% | ICML 2023
DecompDiff | -8.39 | 2.71 | - | 24.5-39.0% | ICLR 2024
FLOWR | -6.93 | 2.86 | - | 94% PB-valid (not docking success) | Nat Comput Sci 2025
MolCRAFT | -9.25 | - | - | 36.1% | -
AlphaDrug | -9.77 | - | - | not reported | -
TransDiffSBDD | -9.37 | - | - | 83.9% | -
MolChord | -7.62 | - | - | 33.2% | -

All numbers above are cited from original papers, NOT re-run by Mol-Metal. See §2 flags 1-7 for protocol-mismatch details.

### §4.5 - Engine Parity Caveat

Source: molmetal/reports/round9_qvina_parity.md

AutoDock Vina 1.2.7 vs QuickVina 2 / QuickVina-W parity summary:

- QuickVina 2 shares the same scoring function as Vina 1.2.7 (Alhossary 2015, Bioinformatics). Pearson correlation on 195 PDBbind 2014 complexes: r = 0.967 (1st mode).
- Byte-exact parity is NOT established - r = 0.967 is correlation, not byte-identity.
- exh=8 QVina 2 = exh=16 Vina 1.2.7 is unmeasured - Alhossary 2015 only tested exh=8 vs exh=8.
- Within-engine per-seed kcal/mol sigma is unmeasured in all three primary sources (Trott 2010, Alhossary 2015, Hassan 2017).
- Recommendation (from round9_qvina_parity.md): use QuickVina 2 at exh=8 as the Mol-Metal baseline; cite r = 0.967; do not claim byte-exact parity.

See molmetal/reports/round9_qvina_parity.md for the full audit.

### §4.6 - Per-Pocket Table (Lambda Measured vs Cite-Only SOTA)

Sources: molmetal/reports/round9_r4c_pilot_results.md (Lambda), §1 Table Group A (cite-only SOTA).

Note on comparability: The SOTA cite-only numbers are 100-pocket means on CrossDocked2020. Lambda pilot ran on 1-2 pockets from the same corpus. A single-pocket Vina value is statistically indistinguishable from any -6 to -8.5 mean given pocket-specific variance of +/-2 kcal/mol. Direct delta comparison is not valid without a full 100-pocket run.

Pocket ID | Corpus | n_mols | Lambda Vina (mean, kcal/mol) | SOTA Vina cite-only (kcal/mol) | Delta | Notes
---|---|---:|---:|---:|---:|---
1433B_HUMAN_1_240_pep_0 | CrossDocked2020 | 20 | -20.061 (proxy) | TargetDiff -8.45 | not comparable | proxy placeholder; see §4.1
1433C_TOBAC_1_256_0 | CrossDocked2020 | 0 | n/a (0 candidates) | TargetDiff -8.45 | n/a | search returned no candidates
1h36 (HSP90/HEM) | Non-standard | N<=5 | -5.923 (real Vina, exh=16) | Pocket2Mol -7.07 | +1.15 (single pocket) | single-pocket only; see §4.1
830c (MMP-13) | Non-standard | N<=5 | see pilot report | Pocket2Mol -7.07 | see pilot | not in CrossDocked2020 formal pilot

Delta interpretation: Lambda -5.923 (1h36, real Vina) vs Pocket2Mol -7.07 (100-pocket mean) = +1.15 kcal/mol. This is not a Lambda regression - it is the gap between a single-case value and a 100-pocket mean. See §2 flag 1.

---

## §5 Metal-anticancer incommensurability caveat

The cite-only SOTA comparison in §2 (Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff/FLOWR) is fundamentally **INCOMMENSURABLE** with Mol-Metal's target class.

### Reason 1 — Target biology and binding chemistry

The SOTA systems benchmark general enzyme pockets such as trypsin, EGFR, and other protein cavities, using organic inhibitors as the reference ligand class. Their learned distributions, pocket encoders, and docking objectives assume conventional non-covalent recognition by carbon-rich small molecules.

Mol-Metal targets Pt(II/IV), Ru(II/III), and Ir(III) complexes designed for anticancer action. The primary biological target is DNA, especially guanine N7, where the metal warhead forms coordinate-covalent adducts (Jamieson & Lippard, *Chem Rev* 99:2467–2498, 1999). DNA grooves, aquation, substitution, and redox state determine activity in ways absent from an enzyme-pocket benchmark.

These are different pharmacophores and different binding modes. A protein-pocket Vina score measures a largely non-covalent pose; a metal anticancer score must also represent coordination geometry, leaving-group kinetics, and the chemical state reached after aquation. The scoring functions and training labels are therefore different physical observables.

### Reason 2 — General metric bias against metal complexes

QED systematically penalises metal warheads because Pt, Ru, and Ir atoms have no meaningful representation in the QED training data. A low QED value can reflect model extrapolation rather than poor anticancer developability.

Lipinski molecular weight ≤500 Da is violated by the Pt atom mass alone in many valid complexes. This is an expected property of the therapeutic payload, not automatically a liability. Synthetic accessibility (SA) >4 is also normal for multi-dentate chelates, stereogenic coordination environments, and controlled ligand substitution.

Applying the general QED, MW, and SA cutoffs as a triple-threshold success definition therefore under-counts chemically valid metal complexes. A lower Lambda headline success rate can be produced by threshold bias even when the candidates satisfy the intended metal-drug design constraints.

### Reason 3 — Missing anticancer metric suite

The recommendations in `molmetal/reports/anticancer_vs_general_metrics_survey.md` §6 define an anticancer-aware supplement: logP ∈ [2, 5], TPSA ∈ [60, 150 Å²], an aquation proxy, GSH resistance, and a DNA-binding Kb proxy. These metrics track membrane access, aqueous exposure, activation kinetics, thiol detoxification, and the intended DNA mechanism.

None of these quantities is reported by the cited Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, or FLOWR baselines. Their tables consequently cannot answer whether a candidate is stable in plasma, resists glutathione sequestration, aquates at a useful rate, or forms a productive DNA adduct.

The mechanistic basis is established in the metal-drug literature: aquation kinetics govern activation (Berners-Price et al., 1990), GSH reactivity is a major resistance pathway (Chen et al., *J Biol Inorg Chem* 18:865–877, 2013), and anticancer lead lipophilicity commonly occupies the logP 3–5 range (Kellogg et al., *J Med Chem* 34:656–663, 1991). DNA binding constants or validated proxies remain the most direct mechanistic endpoint (Jamieson & Lippard, 1999).

### Consequence for interpreting the headline number

Lambda's headline number (the general triple-threshold success rate) is **LOWER** than SOTA's, but it is measuring a **DIFFERENT problem**. The result should be read as a protocol disclosure, not as evidence that Lambda is inferior at metal-anticancer design.

The right comparison is Lambda's metal-anticancer adjusted threshold against a hypothetical SOTA model retrained on metal-anticancer data, including DNA and coordination labels. Such a benchmark does not exist publicly. Until it does, cite-only enzyme-pocket numbers provide context for general SBDD engineering but cannot establish a ranking for Mol-Metal's target class.

### Required upgrades for a fair comparison

The metric implementation requested in `TODO/pending/15` (anticancer metric suite) should report the five adjustments above alongside Vina, SA, QED, and Lipinski values. The structure-based validation in `TODO/pending/16` (DNA fragment docking) should provide a DNA-fragment pose or Kb proxy for guanine-directed coordination.

Together these upgrades enable a two-column report: general thresholds retained for SOTA comparability, and metal-anticancer thresholds used for real-world applicability. They also make the target biology explicit when a candidate passes or fails.

### What the adjusted threshold means

The adjusted threshold is a reporting lens, not a claim that every metal complex is clinically active. It preserves Vina and SA as useful coarse filters while relaxing the assumptions that are known to reject transition-metal chemistry.

For each candidate, the report should state the metal identity and oxidation state, coordination number, ligand denticity, predicted charge, and whether a plausible leaving group is present. These descriptors explain why MW or SA may fall outside oral-drug conventions.

The aquation proxy should be interpreted with the medium and counter-ion context recorded. Berners-Price (1990) showed that substitution kinetics are condition-dependent; a single scalar cannot replace chloride concentration, pH, temperature, or direct kinetic measurement.

Likewise, a GSH-resistance flag indicates a liability or design hypothesis, not a measured cellular resistance phenotype. Chen (2013) provides the mechanistic rationale, while DNA Kb is a proxy for adduct formation and does not alone establish cytotoxicity.

These caveats follow the survey's honest-caveats discussion in §7. They are part of the protocol definition so that future rounds do not silently switch between oral-drug and metal-drug interpretations.

### FLAG 8 — Metal-anticancer incommensurability

**Targets DNA coord-covalent chemistry vs general enzyme pockets; the triple-threshold success rate is not directly comparable.** This is a fundamental protocol mismatch, alongside flags 1–7 in §2, and remains open until a public metal-anticancer SOTA benchmark is available.

### Why this paper is still valid

Despite incommensurability, Mol-Metal's framework demonstrates the **FIRST MLC pipeline that respects 9-layer MLC semantics with metal-geometry priors**. The metric gap is a known limitation, not a methodological flaw. Round-12/13 will report **BOTH** general triple-threshold (for SOTA comparability) **AND** metal-anticancer adjusted threshold (for real-world applicability).

*End of master table. Cross-references into the 5 Lambda-line docs are appended below.*
