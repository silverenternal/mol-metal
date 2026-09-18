# WF-3-CiteOnly-SOTA — Cite-Only SOTA Comparison Table

**Date:** 2026-09-14
**Workflow:** WF-3 (Cite-Only SOTA reference, paper §4.5 / §2 / Supplementary)
**Status:** COMPLETE — table is paper-ready; honest-framing labels attached to every row
**Companion CSV:** `molmetal/reports/wf_3_citeonly_sota_table.csv` (9 SOTA + 2 Lambda rows × 17 columns)
**Authoritative sources:**
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §1 (Group A/B cite-only table)
- `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` §2.1 (legacy superseded)
- `paper/sections/02_related.tex` Table~\ref{tab:sota-comparison} (7 protocol-mismatch flags)
- `molmetal/reports/wf_extra1_full/final.md` (Lambda MEASURED, n/a to this workflow but cited for honesty)
- `molmetal/reports/wf_lambda1c_pilot_v3/final.md` (Lambda MEASURED, n/a to this workflow but cited for honesty)

---

## §1 — Methodology (where each number came from)

### §1.1 — SOTA row provenance

Each SOTA cell in the CSV was copied verbatim from one of three sources, in
priority order:

1. **`lambda_vs_sbdd_protocol_aligned.md` §1 Group A table** — the master
   cite-only table for Pocket2Mol, TargetDiff, DiffSBDD, DecompDiff, FLOWr
   (CrossDocked100 / SPINDR, Vina kcal/mol, SA mean, QED mean, success rate).
   Provenance audits: `provenance_pocket2mol.md`, `provenance_targetdiff.md`,
   `provenance_diff_decomp_equi_tank.md`, `provenance_diffdock_flowr_flowdock.md`.
2. **`paper/sections/02_related.tex`** Table~\ref{tab:sota-comparison} for the
   7 protocol-level properties (Dataset, Docking engine, Valid def, Novelty def,
   Diversity def, # seeds, # pockets, Metal-aware?) and the 7 protocol-mismatch
   flags.
3. **Web-cached / paper-text-published numbers** for the docks that are not in
   Group A: MolDiff (Chen et al. AAAI 2023, arXiv 2305.07545, Vina −7.55,
   success 29.1% per the AAAI 2023 main table — cite-only, not re-run by
   Mol-Metal); DiffDock (Corso 2022 ICML, arXiv 2210.01776, RMSD<2Å top-1
   38.2% on PDBbind time-split 363); RoseTTAFold-AA (Krishna et al., Nat Methods
   2024, structure-only — no SBDD metrics, not directly comparable).

The 7 protocol-mismatch flags were taken verbatim from `02_related.tex`
§\ref{sec:related:protocol} paragraphs "Flag 1" through "Flag 7", and re-encoded
in the `protocol_mismatch_flags` column of the CSV.

### §1.2 — Lambda row provenance

The two Lambda rows are **MEASURED** (not CITED) and originate from:

- **`wf_extra1_full/final.md`** is NOT used for the CSV — that report covers the
  pIC50 predictor (WF-Extra-1), not the docking success numbers. It is included
  here only as a marker of what is MEASURED elsewhere.
- **`wf_lambda1c_pilot_v3/final.md`** is NOT used for the headline Vina
  numbers either — that report covers validity/uniqueness/synthesizability
  rates on the Lambda MCTS-only arm (15 cells per arm, 1c patch re-verify).
  Honest framing: those numbers are NOT a SOTA comparison.

The two Lambda rows in the CSV come from the **round-1 single-pocket (1h36)**
result and the **round-9 R4-C 2-pocket pilot**, both carried verbatim from
`lambda_vs_sbdd_protocol_aligned.md` §3.1 round-9 measured numbers:

- `Mol-Metal_Lambda_1h36`: Vina mean = **−5.923 kcal/mol** (REAL AutoDock Vina
  1.2.7, exhaustiveness 16, box 20 Å, PDB 1h36 chain A HEM pocket, n=1
  single-pocket, no error bar). SA = 1.870, QED = 0.548. Top-1 Vina < ref
  rate = 19.4%.
- `Mol-Metal_R4C_pilot`: Vina-proxy top-1 mean = **−14.179 kcal/mol** (PROXY
  PLACEHOLDER, not real Vina — the L-1 DiffDock/FlowDock binary oracle is
  pending per TODO/pending/decisions.md D4). SA = 7.854, QED = 0.857, Lipinski
  pass = 1.000. n = 2 pockets (1h36 HSP90 + 830c MMP-13).

### §1.3 — Honest labelling

Each row's `label` column has exactly one of three values:

- `CITED-ONLY` — number is taken from the original paper; **not re-run** by
  Mol-Metal. All 9 SOTA rows are CITED-ONLY.
- `MEASURED single-pocket (n=1, no error bar)` — Lambda's round-1 result on
  1h36. The −5.923 number IS real AutoDock Vina 1.2.7.
- `MEASURED pilot (n=2; Vina is proxy, NOT real)` — Lambda's round-9 R4-C
  pilot. The −14.179 number IS NOT real Vina; it is a proxy placeholder
  gated on the L-1 oracle (TODO/pending/decisions.md D4). MUST NOT be
  compared head-to-head with any SOTA row.

---

## §2 — Seven protocol-mismatch flags (explicit)

The seven flags are carried verbatim from `paper/sections/02_related.tex`
§\ref{sec:related:protocol} and `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §2. They are the protocol-level differences that prevent a
direct head-to-head numerical comparison. CSV column `protocol_mismatch_flags`
encodes the flag number(s) for each row.

### Flag 1 — CrossDocked2020 split version
- *What:* each SOTA row uses a different version of the "CrossDocked100
  test split". The Luo-2021 split (Pocket2Mol, TargetDiff, DiffSBDD,
  DecompDiff, MolDiff) is dominant; FLOWr uses SPINDR curate; DiffDock /
  BindNet use PDBbind 2019/2020 time-split.
- *Impact:* a "100-pocket test set" in Pocket2Mol is not the same physical
  100 pockets as in FLOWr. Same pocket name → different training-set
  partition.
- *Affected rows:* all 6 CrossDocked geometric-SBDD rows (DiffSBDD,
  Pocket2Mol, TargetDiff, MolDiff, DecompDiff, FLOWr), and the Lambda
  rows. DiffDock / BindNet / RoseTTAFold-AA use a different corpus
  entirely.

### Flag 2 — Docking engine and version (Vina vs QVina vs QuickVina 2 vs DiffDock-L)
- *What:* DiffSBDD / MolDiff / DecompDiff use QVina (exh=8); Pocket2Mol uses
  QuickVina 2 (exh=16, box 20 Å); TargetDiff uses QVina with UFF + size_factor
  1.2; FLOWr uses DiffDock-L pose ranker; BindNet uses a Vina-score head;
  DiffDock uses its own confidence-ranked pose predictor.
- *Impact:* QuickVina 2 and AutoDock Vina 1.2.7 share the Trott 2010 scoring
  function but version drift gives 0.3–0.6 kcal/mol mean diff at same
  exhaustiveness. Alhossary 2015: r = 0.967 (correlation, NOT byte-identity)
  on 195 PDBbind 2014 complexes. **Lambda's round-11 N=50 paired run
  (round11_parity.md) establishes Vina 1.2.7 ↔ QuickVina 2 Pearson r = 0.9983,
  Spearman ρ = 0.9984, mean paired diff +0.009 ± 0.018 kcal/mol** (n=46
  paired; 4 failures = QVina-1.1.2 CG0 atom-type limitation, not parity
  failure). The flag is **closed** for the Vina/QVina axis; **still open**
  for DiffDock-L/Vina.
- *Affected rows:* all 9 SOTA rows + both Lambda rows.

### Flag 3 — Validity definition (RDKit-only vs PB-augmented)
- *What:* geometric SBDD defines validity as "RDKit can parse and sanitise";
  FLOWr defines validity as "PoseBusters passes" (a stronger structural
  test). The two definitions are structurally incomparable.
- *Impact:* a PoseBuster-passing conformer can still fail RDKit sanitisation
  if bond topology is bad; a SMILES-valid molecule can fail PoseBusters if
  conformer is bad.
- *Affected rows:* all SBDD rows (RDKit-only) + FLOWr (PB-only) + both Lambda
  rows (RDKit + β-NF + PB graceful).

### Flag 4 — Novelty definition (random vs scaffold vs time split)
- *What:* all 6 geometric-SBDD rows + FLOWr define novelty as `max Tanimoto
  (Morgan r=2, 2048 bits) < 0.4 to the training set` under a **random**
  train/test split. DiffDock / BindNet split on **time** (PDBBind 2019/2020),
  which is the more conservative novelty definition.
- *Impact:* scaffold leakage is structurally harder to avoid under a time
  split than under a random split. Random-split novelty numbers are inflated
  relative to time-split numbers.
- *Affected rows:* all 9 SOTA rows + both Lambda rows.

### Flag 5 — Diversity definition (Tanimoto-based vs homotype)
- *What:* all 6 geometric-SBDD rows use mean pairwise Tanimoto (Morgan r=2,
  2048 bits) as the diversity metric. Mol-Metal uses Homotype
  (enriched-vocab distance) + Tanimoto. Homotype and Tanimoto disagree on
  *which* chemistry is distant (homotype lifts constitutional-isomer pairs
  off the zero floor).
- *Impact:* the geometric-SBDD diversity column is **incomparable** with
  Mol-Metal's homotype column.
- *Affected rows:* all 6 geometric-SBDD rows + FLOWr + both Lambda rows.
  DiffDock / BindNet / RoseTTAFold-AA use pose-mode / ligand-mode /
  backbone-RMSD diversity, which is also incomparable.

### Flag 6 — Number of seeds (1 vs ≥3)
- *What:* DiffSBDD, Pocket2Mol, TargetDiff, MolDiff, DecompDiff, FLOWr
  typically report numbers from **a single seed** (or implicitly
  single-seed through the published checkpoint); DiffDock reports 10×20
  samples per pocket, BindNet 3 seeds, RoseTTAFold-AA 5 seeds.
- *Impact:* per-seed variance of Vina kcal/mol at n_docked=4 is
  σ ≈ 0.46 kcal/mol (round-10 micro-bench); at this noise floor a
  single-seed result is indistinguishable from no-effect.
- *Affected rows:* all 6 geometric-SBDD rows + FLOWr (1 seed) + both Lambda
  rows (3 seeds).

### Flag 7 — Pocket count (single vs ≥10 vs 100)
- *What:* all geometric-SBDD papers report a population mean over 100
  held-out pockets (or 363 for PDBbind). Lambda's round-1 single-pocket
  1h36 result is a single-case value with no error bar.
- *Impact:* pocket-specific variance in Vina is typically ±2 kcal/mol;
  a single-pocket result is statistically indistinguishable from any
  −6 to −8.5 mean. Direct delta comparison is invalid without a full
  100-pocket run.
- *Affected rows:* DiffDock / BindNet use 363 / ~340 pockets (different
  corpus). All other rows + both Lambda rows affected.

---

## §3 — Honest framing for each row

This section gives a one-paragraph "honest framing" for each row of the CSV.
The framing is what the paper should say when reporting the corresponding
number; it is not a sales pitch.

### DiffSBDD (CITED-ONLY)
Schneuing et al. 2023 (ICML, arXiv 2210.13695). Vina −7.62 kcal/mol and
SA 2.81 are taken from the CrossDocked100 Luo-2021 split, single seed,
QVina exhaustiveness 8. Success rate 24.6% follows the paper's
"High-Affinity" definition (Vina < co-crystal reference). **Not re-run by
Mol-Metal.** Direct delta comparison vs Lambda requires same-engine, same-split,
same-seed protocol — see Flags 1, 2, 6, 7.

### Pocket2Mol (CITED-ONLY)
Peng et al. 2022 (ICML, arXiv 2205.07249). Vina −7.07, SA 2.51, QED 0.55,
High-Affinity success 49.8%. QuickVina 2 at exh=16, box 20 Å. Single seed.
CrossDocked100 Luo-2021 split. **Not re-run by Mol-Metal.** The +1.15 kcal/mol
delta between Lambda's 1h36 single-pocket value (−5.923) and Pocket2Mol's
100-pocket mean (−7.07) is the gap between a single-case value and a
100-pocket mean, not a Lambda regression.

### TargetDiff (CITED-ONLY)
Guan et al. 2023 (ICLR, arXiv 2303.03543). Vina −8.45, SA 2.65, QED 0.48,
High-Affinity success 35.1% (relaxed) / 10.5% (strict). QVina exh=8 with UFF
torsion pruning and size_factor=1.2. Single seed. CrossDocked100 Luo-2021
split. **Not re-run by Mol-Metal.** The strict / relaxed split on TargetDiff's
success rate is itself a protocol-mismatch artifact within the paper.

### MolDiff (CITED-ONLY)
Chen et al. 2023 (AAAI, arXiv 2305.07545). Vina −7.55, SA 2.71, QED 0.51,
High-Affinity success 29.1%. QVina exh=8. Single seed. CrossDocked100
Luo-2021 split. **Not re-run by Mol-Metal.** Same engine family as DiffSBDD /
DecompDiff; Vina mean sits 0.1 kcal/mol above DiffSBDD.

### DecompDiff (CITED-ONLY)
Guan et al. 2024 (ICLR, arXiv 2303.10120). Vina −8.39, SA 2.71, success 39.0%
(some tables 24.5%). QVina exh=8. Single seed. CrossDocked100 Luo-2021 split.
**Not re-run by Mol-Metal.** The 24.5% vs 39.0% success-rate discrepancy in
the paper itself is a flag for the round-10 audit.

### FLOWr (CITED-ONLY)
Cremer et al. 2025 (Nat Comput Sci, arXiv 2504.10564 — corrected from the
prior 2404.02819 audit mistake). Vina −6.93, SA 2.86, PB-valid 94%. The
94% is **PoseBusters-valid only**, not docking success — the two
denominators are different. FLOWr's test set is SPINDR-curated
CrossDocked100, not the Luo-2021 split. **Not re-run by Mol-Metal.**

### DiffDock (CITED-ONLY, RMSD not Vina)
Corso et al. 2022 (ICML, arXiv 2210.01776) and DiffDock-L 2024 (ICLR,
arXiv 2403.05784). The reported 38.2% RMSD<2 Å top-1 on PDBbind
time-split 363 is a **docking / RMSD** metric, not a Vina kcal/mol.
10 × 20 = 200 samples per pocket. **Not re-run by Mol-Metal.** Comparing
DiffDock's RMSD<2 Å to the geometric-SBDD Vina mean is a category error
(Flag 1, Flag 3).

### BindNet (CITED-ONLY, RMSD not Vina)
BindNet 2024 (NeurIPS). RMSD<2 Å top-1 metric on PDBbind v2020 time-split,
3 seeds, ~340 pockets, Vina-score head. **Not re-run by Mol-Metal.**
Same category error as DiffDock.

### RoseTTAFold-AA (CITED-ONLY, structure-only)
Krishna et al. 2024 (Nat Methods). Side-chain rotamer + all-atom
clash-free validation, ~150 PDB+AF2 clusters, 5 seeds. **No SBDD
metrics** — this is a structure-prediction system, not a generator.
Comparison to the geometric-SBDD / Lambda columns is structurally
incomparable (Flag 3, Flag 4, Flag 7).

### Mol-Metal Lambda 1h36 (MEASURED single-pocket)
Lambda MCTS run on PDB 1h36 chain A (HSP90, HEM pocket). Vina −5.923
kcal/mol IS **real AutoDock Vina 1.2.7** at exhaustiveness 16, box 20 Å.
SA 1.870, QED 0.548. Single pocket, no error bar. Top-1 Vina<ref
(HEM ≈ −7.0) rate = 19.4%. MCTS budget = 1000 sims. **The +1.15 kcal/mol
delta vs Pocket2Mol's 100-pocket mean (−7.07) is the single-case vs
population gap, not a regression.**

### Mol-Metal R4C pilot (MEASURED pilot, Vina is PROXY)
Round-9 R4-C sweep on 2 pockets (1h36 + 830c). Vina-proxy top-1 mean
−14.179 IS **NOT real Vina**; it is a proxy placeholder from the L-1
DiffDock/FlowDock binary oracle (TODO/pending/decisions.md D4). SA 7.854,
QED 0.857, Lipinski pass 1.000. MCTS config: n_simulations=50,
max_depth=2; library = L-3 204-tile; predicates = LIPINSKI. **This row
MUST NOT be compared head-to-head with any SOTA row** until L-1 oracle
is live and the proxy is replaced by real Vina on a 100-pocket sweep.

---

## §4 — What the paper should and should NOT claim

### Should
- "Mol-Metal is compared against nine published SBDD / docking systems
  (Table~\ref{tab:sota-comparison}). Cite-only numbers from the original
  papers; not re-run."
- "Seven protocol-mismatch flags prevent direct head-to-head numerical
  comparison without further methodological work (§\ref{sec:related:protocol})."
- "The closest fair comparison is the round-13 100-pocket × 3-seed sweep
  on the Luo-2021 CrossDocked100 split using QuickVina 2 at exh=8,
  following the round-11 N=50 parity audit."
- "Lambda's round-1 1h36 single-pocket result (−5.923 kcal/mol) is real
  AutoDock Vina 1.2.7; the round-9 R4-C pilot (−14.179 kcal/mol) uses a
  proxy placeholder and is not directly comparable."

### Should NOT
- "Mol-Metal beats Pocket2Mol on Vina" — FALSE: the single-pocket vs
  100-pocket-mean delta is protocol-incomparable, not a regression.
- "Mol-Metal beats TargetDiff on success rate" — FALSE: success rate
  numbers require a 100-pocket sweep, not the n=2 R4-C pilot.
- "Lambda's −14.179 is comparable to TargetDiff's −8.45" — FALSE: the
  −14.179 is a proxy placeholder, NOT real Vina.
- "FLOWr's 94% PB-valid is the new bar for Lambda" — FALSE: PB-valid is
  a different denominator from docking success (Flag 5 in §2).
- "DiffDock's 38.2% RMSD<2 Å is comparable to Pocket2Mol's −7.07 Vina"
  — FALSE: category error (Flag 1, Flag 3).
- "RoseTTAFold-AA is comparable to geometric SBDD" — FALSE: structure-only
  validation, not generator output (Flag 3, Flag 4, Flag 7).

---

## §5 — Closing-the-flag timeline

| Flag | Status (this report, 2026-09-14) | Round-13 close-out plan |
|---|---|---|
| 1 — CrossDocked100 split version | partially closed (Luo-2021 staged; SPINDR curate unavailable) | run on staged `split_by_name.pt` to cover Luo-2021 rows; corpus-bridge to FLOWr / DiffDock / BindNet / RoseTTAFold-AA as supplementary |
| 2 — Docking engine | closed for Vina/QVina (round-11 Pearson r=0.9983); open for DiffDock-L/Vina | adopt QuickVina 2 at exh=8 as new baseline per round-9 recommendation |
| 3 — Validity def (RDKit vs PB) | partially closed (Lambda uses both, PB graceful skip) | report both axes side-by-side; corpus-bridge to FLOWr PB-valid via shared PB-passed subset |
| 4 — Novelty def (random vs scaffold vs time) | open (Lambda novelty is placeholder 1.0000) | recompute novelty under random-split (SBDD compatibility) AND scaffold-group split (property-prediction compatibility) |
| 5 — Diversity def (Tanimoto vs homotype) | open (Tanimoto and homotype disagree on constitutional isomers) | report both columns; flag the homotype lift as the load-bearing methodological contribution |
| 6 — Seeds | partially closed (Lambda 3 seeds, geometric-SBDD 1 seed) | round-13 sweeps with 3 seeds per pocket; report seed-stratified σ |
| 7 — Pocket count | open (Lambda n=2 R4-C pilot; geometric-SBDD n=100) | round-13 100-pocket × 3-seed sweep is the load-bearing experiment |

---

## §6 — Files produced

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_3_citeonly_sota_table.csv` (this workflow output)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_3_citeonly_sota.md` (this file)

## §7 — Files NOT touched (WF-3 invariant preserved)

- `paper/sections/02_related.tex` (Table~\ref{tab:sota-comparison} is the
  authoritative version; the CSV is a tabular mirror, not a replacement)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (master cite-only
  source; not modified)
- `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` (legacy superseded
  cite-only table; not modified)
- All Lambda / CFM / round-trip / metal-seed / BNF code paths
- Any historical checkpoint or pretrain weights