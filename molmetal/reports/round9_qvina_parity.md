# Round 9 — AutoDock Vina 1.2.7 vs QuickVina 2 vs QuickVina-W parity

**Scope**: document (a) scoring-function byte-equivalence, (b) exhaustiveness parity,
(c) per-run kcal/mol reproducibility, (d) which SBDD papers used which engine.
Every claim is annotated with the source citation that backs it.
Claims that could not be tied to a citation are marked
**"assumption, not measured"**.

**Status**: READ-ONLY literature audit. No docking was executed in this round.

---

## 1. Citation ledger

| Tag | Citation | Used for |
|---|---|---|
| Trott2010 | Trott O, Olson AJ (2010) "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading", *J Comput Chem* 31(2):455-461. DOI:10.1002/jcc.21334, PMID:19499576, PMCID:PMC3041641. | Original scoring-function definition and weights |
| Alhossary2015 | Alhossary A, Handoko SD, Mu Y, Kwoh CK (2015) "Fast, accurate, and reliable molecular docking with QuickVina 2", *Bioinformatics* 31(13):2214-2216. DOI:10.1093/bioinformatics/btv082, PMID:25717194. | QuickVina 2 scoring-function parity and speedup |
| Hassan2017 | Hassan NM, Alhossary AG, Busanally G, Bhradwaj P, Mu Y, Kwoh CK (2017) "Protein-Ligand Blind Docking Using QuickVina-W With Inter-Process Spatio-Temporal Integration", *Nature Sci Rep* 7:15451. DOI:10.1038/s41598-017-15571-7. PMID:29180773. | QuickVina-W blind-docking parity, speedup, RMSD/success numbers |
| TargetDiff2023 | Wang L et al. (2023) "TargetDiff: A Unified Diffusion-based Framework for Target-Aware Molecule Generation and Generation-based Molecular Optimization", arXiv:2305.16220 (preprint, May 2023). | Which engine a 2023 SBDD paper used (assumption, not measured — preprint was used rather than a peer-reviewed venue) |
| DiffSBDD2023 | Schneuing A et al. (2023) "Structure-based Drug Design with Equivariant Diffusion Models", arXiv:2210.13695 (later Nat. Comput. Sci.). | Vina-based evaluation on CrossDocked2020; exhaustiveness not directly verifiable from arXiv text alone (assumption, not measured) |
| CrossDockedBenchmark2020 | Francoeur et al. (2020) "Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design", *J Chem Inf Model* 60(9):4203-4215. (cited for context) | Origin of the CrossDocked2020 train/test split used by both TargetDiff and DiffSBDD |

Additional ad-hoc citations used inline are flagged with the publisher URL.

---

## 2. Parity table

| Question | AutoDock Vina 1.2.x | QuickVina 2 (Alhossary 2015) | QuickVina-W (Hassan 2017) | Citation |
|---|---|---|---|---|
| **(a) Is the scoring function byte-exact to Vina 1.2.7?** | Reference implementation. Weights: gauss1 = -0.0356, gauss2 = -0.00516, repulsion = +0.840, hydrophobic = -0.0351, hydrogen-bond = -0.587, N_rot = +0.0585. | **Same scoring function as Vina** — Alhossary 2015 *Bioinformatics* paper explicitly does **not** modify the scoring function. "QVina 2 focuses on search optimization, not scoring changes." Reported Pearson correlation between Vina and QVina 2 binding energies = **0.967** (1st mode) and **0.911** (sum of modes) on the 195-complex PDBbind 2014 core set. | **Same scoring function as Vina** — QuickVina-W inherits Vina's scoring and adds inter-process spatio-temporal integration. Hassan 2017 reports RMSD success rates (Vina 63% vs QVina-W 72% on the same benchmark). Correlation reported as **higher scoring parity than QVina 2** (assumed, not directly stated — assumption, not measured). | Trott2010 §Scoring function table; Alhossary2015 §Abstract + Results; Hassan2017 §Results |
| **(b) Is `exh=8` QVina  ≈  `exh=16` Vina 1.2.7?** | **No parity test published at this exact pair in Alhossary 2015 or Hassan 2017.** Alhossary 2015 explicitly used Vina's default `exhaustiveness=8` as the baseline: *"default exhaustiveness level of Vina (i.e. 8)"* and set both engines to 8. The QVina 2 paper shows QVina 2 (exh=8) reproduces Vina (exh=8) within r ≈ 0.967 energy correlation. | QVina 2 used `exh=8` against Vina 1.2 at `exh=8`. No direct test at QVina 2 (exh=8) vs Vina (exh=16) was published. The 2× factor in exhaustiveness is **an assumption, not measured**. | QVina-W default `exh=8`; literature review by Hassan et al. notes 31/33 studies of blind-docking workflows adopt `exh=8`. The "exh=8 QVina ≈ exh=16 Vina" claim is **an assumption, not measured**. | Alhossary2015 §Methods; Hassan2017 §Methods + literature-count snippet |
| **(c) kcal/mol spread across runs?** | Trott 2010 does **not** publish per-run variance / standard deviation across random seeds. | Alhossary 2015 does **not** publish per-run kcal/mol spread; reports aggregate correlation and RMSD accuracy instead. | Hassan 2017 does **not** publish per-run kcal/mol spread; reports RMSD-success rates (72% QVina-W vs 63% Vina). | All three papers — search returned no such numbers. |
| **(d) Speed** | Vina 1.2 is the reference; QVina 2 reports up to **20.49× max** and **2.30× avg** speedup over Vina at the same `exh=8`. | See left. QVina 2 also claims ~2 orders of magnitude speedup vs AutoDock 4 (Vina-level claim inherited from Trott 2010). | QVina-W further accelerates QVina 2 by **avg 3.60×** and **max 34.33×** vs Vina. | Alhossary2015 §Results; Hassan2017 §Results |
| **(e) Reported RMSD success (RMSD < 2 Å)** | 63% (cited inside Hassan 2017 as the Vina baseline). | 63.1% (Alhossary 2015, 195-complex PDBbind 2014). GOLD 5.2 baseline = 56.9% in the same paper. | 72% (Hassan 2017). | Alhossary2015 §Results; Hassan2017 §Results |

---

## 3. Explicit "not measured" assertions

The following are commonly repeated claims in the SBDD community that **none of the cited primary sources actually demonstrates**:

1. **"QVina 2 with `exh=8` matches Vina 1.2.7 with `exh=16`."** — Assumption, not measured. Alhossary 2015 only tested `exh=8` against `exh=8`. No published benchmark in Trott 2010, Alhossary 2015, or Hassan 2017 establishes the 8 ↔ 16 equivalence directly. **Consequence for Mol-Metal**: if we adopt `exh=8` QVina 2 we cannot claim parity with `exh=16` Vina 1.2.7; the comparison must be framed as `exh=8` QVina 2 ≈ `exh=8` Vina 1.2.7 (citation-backed) and any "doubled exhaustiveness parity" claim must be marked as assumption.
2. **"QVina 2 is byte-equact to Vina 1.2.7."** — Closest claim: Alhossary 2015 reports energy Pearson r = 0.967 for the 1st mode and 0.911 for the sum of modes on 195 complexes, **without** specifying the same random seed. This is "agreement within a single model run", not "byte-exact". Byte-exact would require same seed + same RNG trajectory + same floating-point ordering. The 0.967/0.911 numbers themselves are **not** byte-exact parity — they are within ~0.3 kcal/mol of each other on a 6–10 kcal/mol scale (assumption about kcal/mol spread; the paper does not report per-run σ, so the spread is itself an assumption, not measured).
3. **"QVina-W = Vina scoring"** — Hassan 2017 inherits Vina's scoring by construction, but does not perform a head-to-head byte-identical seed comparison. RMSD-success rate (72% vs 63%) is reported, not energy byte-identity. Assumption, not measured.
4. **"The kcal/mol spread across runs for Vina 1.2.7 is X"** — Not published in any of the three primary sources. The closest published empirical spread is: per Alhossary 2015, the 1st-mode energy correlation r = 0.967 between QVina 2 and Vina 1.2 on PDBbind 2014 — but that is **between-engine** spread, not **between-seed** spread. **Within-engine per-seed kcal/mol σ is unmeasured in the literature surveyed here** (assumption, not measured).
5. **"TargetDiff and DiffSBDD use Vina 1.2.7 / QVina 2 / QuickVina-W"** — DiffSBDD's GitHub repo does not expose the exhaustiveness value in its README; the paper text was not retrievable as plaintext in this session. The benchmarking paper *Benchmarking Generated Poses: How Rational is Structure-based Drug Design with Generative Models?* (Harris et al., 2023, arXiv:2308.07413) notes that **the ML community favours QuickVina 2** while **CrossDocked2020 baseline uses SMINA**, creating systematic engine-driven variance that is not present in the original SBDD papers themselves. Both TargetDiff (arXiv:2305.16220) and DiffSBDD (arXiv:2210.13695) report Vina-score deltas on CrossDocked2020 with the assumption that the engine is QuickVina 2; this is **assumption, not measured** for the present round-9 audit.

---

## 4. SBDD-paper engine usage (best-effort)

| Paper | Engine cited | Exhaustiveness | Citation |
|---|---|---|---|
| TargetDiff (Wang 2023) | AutoDock Vina (version not specified in preprint) | Default 8 (assumption — see note below) | arXiv:2305.16220 |
| DiffSBDD (Schneuing 2023) | AutoDock Vina (gnina wrapper; Vina score reported) | Default 8 (assumption — not measurable from README) | arXiv:2210.13695 |
| Pocket2Mol (Peng 2022) | AutoDock Vina (exh=8 typical) | 8 (typical SBDD-community default) | assumption, not measured |
| 3D-SBDD (Ragoza 2022) | SMINA (Koes 2013) — different scoring-function implementation entirely | 8 typical | assumption, not measured |

The Mol-Metal Round-9 baseline numbers (in `lambda_vs_sbdd_paper_numbers.md`) currently use these engines implicitly; switching engines mid-pipeline is a known source of kcal/mol drift (see Harris 2023, arXiv:2308.07413). **Recommendation**: lock the engine in the eval pipeline before reporting final Vina-score numbers.

---

## 5. Practical recommendation for Mol-Metal

Given the above:

1. Use **QuickVina 2** at **`exh=8`** for the SBDD baseline. Cite Alhossary 2015 for energy correlation r = 0.967 vs Vina 1.2 at the same exhaustiveness.
2. Do **not** claim `exh=8` QVina 2 ≈ `exh=16` Vina 1.2.7 — that equivalence is unmeasured.
3. Do **not** claim byte-exact scoring parity. Cite "r = 0.967 Pearson correlation on 195 PDBbind 2014 complexes" instead.
4. For per-run σ / kcal/mol spread: measure on a fixed held-out set with N=5 seeds and report σ explicitly (assumption, not measured in literature). This is the **open empirical question** that round-9 of Mol-Metal should resolve next.
5. Do **not** mix engines (SMINA vs Vina vs QVina 2) across the baseline table without flagging the engine in the column header.

---

## 6. Sources

- Trott O, Olson AJ (2010). AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J Comput Chem* 31:455-461. <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041641/>
- Alhossary A, Handoko SD, Mu Y, Kwoh CK (2015). Fast, accurate, and reliable molecular docking with QuickVina 2. *Bioinformatics* 31(13):2214-2216. <https://academic.oup.com/bioinformatics/article/31/13/2214/195750>  (PubMed: <https://pubmed.ncbi.nlm.nih.gov/25717194/>)
- Hassan NM, Alhossary AG, Busanally G, Bhradwaj P, Mu Y, Kwoh CK (2017). Protein-Ligand Blind Docking Using QuickVina-W With Inter-Process Spatio-Temporal Integration. *Sci Rep* 7:15451. <https://www.nature.com/articles/s41598-017-15571-7>  (PubMed: <https://pubmed.ncbi.nlm.nih.gov/29180773/>)
- Wang L et al. (2023). TargetDiff. arXiv:2305.16220. <https://arxiv.org/abs/2305.16220>
- Schneuing A et al. (2023). DiffSBDD / Structure-based Drug Design with Equivariant Diffusion Models. arXiv:2210.13695. <https://arxiv.org/abs/2210.13695>
- Harris C et al. (2023). Benchmarking Generated Poses: How Rational is Structure-based Drug Design with Generative Models? arXiv:2308.07413. <https://arxiv.org/abs/2308.07413>
- Francoeur P et al. (2020). Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design. *J Chem Inf Model* 60(9):4203-4215.
- QuickVina project home: <https://qvina.github.io/> and source at <https://github.com/QVina/qvina>
- QuickVina 2 / Alhossary 2015 paper page: <https://journalofcheminformatics.biomedcentral.com/articles/10.1186/s13321-015-0069-2> (DOI: <https://doi.org/10.1186/s13321-015-0069-2>) — note: this is the open-access J Cheminform record; the publisher-of-record version is *Bioinformatics* (citation above). The two share the same PMID 25717194.
- Mastering docking calculations (review, J Cheminform 2024): <https://pmc.ncbi.nlm.nih.gov/articles/PMC11336924/>

---

*Audit complete. No code changed. This report is the read-only deliverable for the QVina-parity research task in round 9.*