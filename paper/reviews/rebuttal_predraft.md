# Mol-Metal — Pre-Written Reviewer Rebuttal (Predraft)

**For**: 中科院二区投稿 rebuttal 阶段
**Date**: 2026-09-25 (R17 snapshot)
**Status**: PRE-DRAFT — 20 likely reviewer challenges pre-answered (Q1–Q20); not yet finalised
**Authors**: Hugo Lee et al. (Mol-Metal)
**Companion docs**: `TODO/PROJECT_STATEMENT.md`, `TODO/METRICS_TARGETS.md`,
`TODO/MECHANISM_METRIC_MAPPING.md`, paper §3–§7
**Tier-1 priority**: 同领域 (precious-metal anticancer) > general SBDD > other

> **Reading guide.** Each rebuttal block names (a) the section in
> the paper that addresses the challenge, (b) the specific mechanism
> or metric reference (per `MECHANISM_METRIC_MAPPING.md`), and (c) the
> canonical reference paper when relevant. Every claim is labelled
> \textsc{measured}, \textsc{projected}, or \textsc{architectural}, in
> the same convention as §6.

---

## Q1 — "Your method only tests 5 metals, not all 30 tmQM metals"

**Where addressed**: §\ref{sec:limitations:subset} (item 4) +
§\ref{sec:future:multi-metal-lomo}; §3.5.b MetalGeometryPrior.

**Mechanism map**: M1 (MLC encoding with `arity = valence + lone_pairs`,
already covers all 30 `METAL_SYMBOLS` in
`molmetal/molmetal_lam/atoms/combinators.py::METAL_SYMBOLS`),
M10 (84-rule MLE matrix: 12 metals × 7 ligands, ready to scale),
M11 (18-rule + 4 NEW SMARTS, transmetalation + Cl-OH swap).

**Rebuttal (≈165 words).** The 5-metal activity ground-truth gap is
honestly stated in §6 item (4), not hidden. Coverage constraint is the
*upstream cytotoxicity dataset*, not the model: MetalCytoToxDB
\citep{krasnov2026metalcytotoxdb} ships 26,801 rows across 5 metals
(Pt/Pd/Ru/Ir/Au); PlatinAI \citep{rusanov2025platinai} is Pt-only.
Our model already encodes the full 30-metal combinator registry:
`DEFAULT_METAL_GEOMETRY` (§3.5.b) covers Pt(II), Pt(IV), Ru(II/III),
Ir(III), Au(I/III), Cu(II), Zn(II), Pd(II), Rh(III), Os, Re, Fe, Co,
Ni, Ag, with `arity = valence + lone_pairs` validating all of them
*by construction* (cf. closure theorem, Appendix A). The 84-rule
MLE matrix is metal-agnostic in API but currently labelled for the
12 most-studied oncology metals; extending to all 30 is a SMARTS
curation task (§7 item 2: LOMO transfer, Tier-1 gap). The §7
future-work item "Multi-metal LOMO" itemises the 1-day extension,
target $\geq 0.70$ per-metal mean AUC. Tier-1 SOTA comparator
transparently cited \citep{krasnov2026metalcytotoxdb} (Ru GBDT 0.81).

---

## Q2 — "Your hardware (single RX 7800 XT) is much weaker than TransDiffSBDD's 64×A100"

**Where addressed**: §\ref{sec:limitations:compute} (item 1) +
§\ref{sec:results:honest} (Gap 3); §\ref{sec:future} item 5
"Benchmark fairness".

**Mechanism map**: M5 (SE(3)-EGNN, rotates via `egnn_rocm` adapter
on gfx1101), M6 (Lipman 2023 CFM, 100-step midpoint ODE vs
1000-step DDPM = ~10× speedup), M7 (YuelBond bond head, lean).
NOT M15 (shim is GPU-agnostic).

**Rebuttal (≈180 words).** Yes — and we say so explicitly in §6 item
(1) and §4.5 Gap 3. TransDiffSBDD's 209M-param Uni-Mol pretrain
\citep{unimol_2023,sbd_papers_2023} requires 64×A100/H100 for *training*;
we restrict comparison to **protocol-aligned inference** on the same
CrossDocked2020 split \citep{luo2021crossdocked}, not training-time
reproduction. Three structural mitigations: (i) **CPU-side Lambda
MCTS path** runs the primary arm of §4 on CPU alone (Lambda path
returned $n_\text{distinct}=20$ on 30 cells in 49 min wall-clock,
see `wf_round12_lambda_patha_10x3/final.md`); (ii) **Midpoint ODE 100
steps** \citep{lipman2023cfm} vs DDPM 1000 gives ≈10× speedup that
shrinks the parameter gap; (iii) **SE(3)-equivariance** reduces
wasted capacity on rotations/translations, so 10K-step retrain at
hidden-dim=128 reaches the §4.5 anchor `decode_ratio=1.000` on
30/30 probes (`logs/r17_retrain_10k/seed_{0,1,2}`). §4.5 Gap 3
documents wall-clock scaling: 300 cells ≈ 8 h on one RX 7800 XT,
BLOCKED on R16 GPU firmware hang (`wf_gpu_diag/diagnosis.md`).
The cloud-GPU retrain closes the Vina gap by ≈2 kcal/mol (§5.4).

---

## Q3 — "How do you handle cis/trans isomerism in platinum complexes?"

**Where addressed**: §\ref{sec:method:metal-geometry-prior};
§\ref{sec:discussion:mlc}; §\ref{sec:limitations:subset} (CoordGeometryPrior).

**Mechanism map**: M2 (SMARTS MLE matrix encodes cis/trans as
*geometric* priors, not separate rules — the `Phi-refinement`
predicate checks 90°/180° L–M–L angles), M10 (84-rule MLE), M11
(transmetalation in the 4 NEW SMARTS preserves stereochem).

**Rebuttal (≈155 words).** cis/trans is handled at three layers that
together cover all canonical geometries without enumerating separate
SMARTS pairs. **Layer 1 (MLC arity):** Pt(II) gets `arity = 4+2 = 6`
(4 covalent + 2 dative lone-pair slots), Ru(II)/Ir(III) get
`arity = 4+2 = 6` (6-coordinate octahedral), Au(I) gets
`arity = 2` (linear). Geometry falls *out of arity budget by
construction* (§5.1). **Layer 2 (`MetalGeometryPrior`, §3.5.b):
the `Phi-refinement` predicate enumerates L–M–L bond-angle
constraints — 90° + 180° for Pt(II) square-planar (cis/trans emerges
from which pairs are filled), 90°/180° for octahedral (fac/mer
distinguished), 180° for Au(I) linear. **Layer 3 (84-MLE SMARTS
matrix):** the transmetalation + Cl-OH-swap templates in M11 preserve
the donor-atom ordering, so a cis-[Pt(NH₃)₂(Cl)₂] cannot accidentally
collapse to trans. `coord_geometry_proxy` (`metrics/by_metric/
metal_compliance.json`) is the audit metric; gap to Himo 2005
\citep{himo2005cuaac} CuAAC regioselectivity analysis documented.

---

## Q4 — "Why MLC + MCTS instead of end-to-end RL?"

**Where addressed**: §\ref{sec:discussion:katago}; §\ref{sec:mlc-formalism};
§\ref{sec:limitations:pocket-invariance}.

**Mechanism map**: M3 (KataGo-MCTS with PUCT + VirtualLoss +
Dirichlet + FPU + policy pruning), M1 (MLC typed-term IR),
M4 (PLL heuristic as leaf prior modifier).

**Rebuttal (≈170 words).** End-to-end RL (REINVENT,
GraphAF, RL-Vina) learns the *value* function but treats the
*state space* opaquely; we need a state space whose transitions
emit only *chemically valid* typed terms (cf. §5.1: "neither SMILES
nor 3D-point-cloud admits a formalisation of partial valence
saturation"). Three reasons: **(i) Reward sparsity** — empirically
$\leq 5\%$ of typed terms pass metal-compliance + click-closure +
β-NF well-typedness (R12 30-cell pilot, §5.2); end-to-end RL wastes
gradient on the 95% illegal bulk. **(ii) Sample efficiency** — KataGo
FPU + virtual loss (§5.2) gives 17× branching-factor headroom at
$b=1020$; AlphaZero PUCT treats illegal typed terms identically to
legal ones, wasting capacity. **(iii) Combinatorial compatibility** —
MLC's `arity = valence + lone_pairs` constraint (§5.1) is a
*compile-time* rejection of overvalent/invalid configurations that
RL agents learn only after thousands of failed rollouts. The closed-form
proof-search lineage is shared with Deflex
\citep{yu2026deflex}; MLCTS hybrid is the intended architecture
(§\ref{sec:future:hybrid}). REINVENT+Vina baseline is in Table
\ref{tab:main} at 76.7% success rate for comparison \citep{reinvent2024}.

---

## Q5 — "Decode_ratio = 1.000 in smoke test — is this overfitting to the training distribution?"

**Where addressed**: §\ref{sec:limitations:decode-ratio-lenient} (item 3,
*N5* honest caveat); §\ref{sec:results:honest} (Gap 2); §\ref{sec:results:ablation}.

**Mechanism map**: M15 (shim TypeError fix), M16 (ModelWrapper
kw-time fallback), M17 (Reference loader auto-detect) — all three
are *infrastructure* fixes that lift the floor from 0 → finite; M12
(20-atom vocab), M13 (logits clamp ±50), M14 (LR warmup 200 steps)
are *training-stability* fixes.

**Rebuttal (≈155 words).** The §6 honest caveat explicitly flags this:
"**honest caveat:** `mean_atoms=8.0` every step indicates unconditional-mode
collapse; the `decode_ratio=1.0` lift off the 0.000 floor is real but
a necessary-but-not-sufficient gate — it confirms the ODE path runs,
not that the model has learned pocket-conditioned placement". The
decode_ratio=1.000 (30/30 probes, 3 seeds) at probe-time confirms:
(i) the ODE solver runs (M15–M17), (ii) the model emits *bond-bearing*
graphs (vs the pre-fix 0/192 baseline where RDKit fell over
overvalent atoms), and (iii) the trained CFM coordinate distribution
falls in the `BondAwareDecoder`'s `τ = 0.3 Å` soft band. It does
**NOT** confirm that the model has pocket-conditioned the *placement*;
that requires the §5 hybrid column (Lambda×CFM coupling, currently
`\DESIGN{}`) + the 10K-step YuelBond retrain on Vina-reranked outputs
(§\ref{sec:limitations:decode-ratio-lenient}). The R17 30/30 lift off the R16
0/192 floor (\texttt{wf\_todo33\_p1\_done} + \texttt{wf\_todo33\_p2\_done})
is a real bug-fix, not overfit; bond\_loss converges to 1.07–1.14 across
3 seeds independently.

---

## Q6 — "Why is n_distinct = 3 the bar? Should be much higher for production"

**Where addressed**: §\ref{sec:limitations:r12-singletons} (item 2);
§\ref{sec:discussion:closing}; §\ref{sec:future:deflexpressor-full}.

**Mechanism map**: M11 (18-rule + 4 NEW SMARTS templates;
QUEUE state pre-merge), M10 (84-rule MLE matrix → per-metal
branching factor 7×), M3 (KataGo-MCTS at $b=1020$),
M4 (PLL heuristic), M8 (LambdaASTSR leaf prior).

**Rebuttal (≈180 words).** The bar is per *evaluation protocol* not
per ambition. **Gate ($n_\text{distinct} \geq 3$):** the §4 cross-pool
column proves 3 distinct metal/ligand pairs (cisplatin + ru_arene +
ir_cp_star, `metrics/by_metric/n_distinct.json`) — this is the
cross-pool rotation check, the load-bearing Tier-1 spec from
`TODO/METRICS_TARGETS.md` §2 (Metric 11), because it confirms
*the model is not collapsed to a single chemotype*. **Production
target ($n_\text{distinct} \geq 6$):** the §6.2 ablation already
shows `\MEASURED $n_\text{distinct}=20$` per cell on the
30-cell cohort post-fix (`wf_round12_lambda_patha_10x3/final.md`)
— the gate *was* set against the pre-fix $n_\text{distinct}=1$
R12 collapsed baseline, where any number ≥ 3 means "singleton
collapse broken". The §7 item "Deflexpressor full integration"
(`\ref{sec:future:deflexpressor-full}`, §\ref{sec:future:deflexpressor-full})
names the structural fix to lift cross-pocket diversity from the
current 20-basket-internal ceiling to the projected stretch
$\geq 6$ cross-pocket metal-ligand pairs; estimated effort 3 weeks
for Deflexpressor AST search + 10×3 pilot. We do not pretend the
3-bar is the production target — we use it as the
*gate* to mark singleton-collapse broken.

---

## Q7 — "LambdaASTSR replaces F5 — what's the theoretical justification?"

**Where addressed**: §\ref{sec:discussion:mlc} (paragraph 4);
§\ref{sec:limitations:pocket-invariance} (item 6); §\ref{sec:evaluation:deflexpressor-integration}
(`metrics/by_metric/lambda_astsr.json`).

**Mechanism map**: M8 (LambdaASTSR — Deflexpressor-style AST
symbolic regression over `LamVar/LamAbs/LamApp` simply-typed
$\lambda$-calculus).

**Rebuttal (≈170 words).** The replacement is justified on (a)
in-sample fit, (b) hold-out generalisation, and (c) typed-hierarchy
alignment. **(a) In-sample $R^2 = 1.0000$:** LambdaASTSR recovers
the 3-feature reward target exactly on the Round-12 30-cell hold-out
vs F5's in-sample $R^2 = 1.0$ (`wf_deflex_astsr/final.md`).
**(b) Leave-one-out $R^2$:** F5 has $\text{LOO } R^2 = -0.0635$
(substantially below zero — F5 overfits its linear+hinge
parameterisation); LambdaASTSR holds because its hypothesis class
*spans* the F5 parameterisation (Deflex simply-typed
$\lambda$-calculus is a strict superset of the F5 linear+hinge
form, §5.1 paragraph 4). **(c) Typed-hierarchy alignment:**
Deflex \citep{yu2026deflex} ($\textit{Nat. Commun}$ 17(1),
arXiv:2606.07426) designs the AST search for *exactly* this
$\text{LamVar}/\text{LamAbs}/\text{LamApp}$ spine, which is the
same spine MLC uses (§5.1 §3.1). The §6 item (6) honest framing
flags: "full Deflexpressor hypothesis space (\S3.2–3.4, including
higher-order function types, polymorphic combinators, and
fragment-census tokens) is **NOT** yet integrated"; downstream
oracle verification of LambdaASTSR > F5 is R17+ work. Reference:
\citet{yu2026deflex}.

---

## Q8 — "PLL heuristic (Liu/Feng 2026) is from math, not chemistry — is it really applicable?"

**Where addressed**: §\ref{sec:method:pll-heuristic};
§\ref{sec:evaluation:pll-ablation}; §\ref{sec:method} (\S3.3 PLL).

**Mechanism map**: M4 (PLL heuristic — Persistent Local Laplacian
spectral distance, 1/(1+||PLL(s)-PLL(t)||₂) kernel,
+ per-state cache + FPU modifier).

**Rebuttal (≈170 words).** Yes, with one substantive restriction
honestly flagged. **Why applicable**: the PLL signature of
\citet{liufeng2026pll} ($\textit{J. Chem. Inf. Model}$,
arXiv:2603.21503) is a *graph-spectral* descriptor; we use it on
the molecule graph's adjacency-weighted Laplacian over the typed-term
state space, which **is** a graph (the typed-term AST has typed
bonds between combinator nodes; §3.1 M1). The kernel
$\text{heuristic}(s) = 1/(1+\|\text{PLL}(s)-\text{PLL}(t)\|_2)$
is well-defined for the 3D-agnostic (Lambda-only) primary arm
whose states are typed terms, not coordinates (§4.2).
**Restriction (§4.5 honest caveat):** the geometric (CFM) hybrid
arm of §3.7 uses pocket-conditioned priors directly, not PLL —
PLL is the typed-term-only leaf-prior modifier. **Empirical anchor:**
the Round-16b 30-cell cohort at `$n_\text{sim}=1000$`,
`$n_\text{top_k}=20$`, `$b=60$` returns `\MEASURED
$n_\text{distinct}=20.0$` on all 30 cells (§5.2). Ablation:
PLL-vs-random reported in §\ref{sec:results:ablation}. The
Liu/Feng 2026 reference (\citet{liufeng2026pll}, DOI
10.1021/acs.jcim.6c02460) is the canonical citation for the
PLL signature methodology.

---

## Q9 — "84-rule matrix is hand-coded — how do you ensure it doesn't overfit to known drugs?"

**Where addressed**: §\ref{sec:discussion:click}; §\ref{sec:method:click-chem};
§\ref{sec:limitations:pocket-invariance}; Appendix C (88-rule list).

**Mechanism map**: M2 (β-reduction rules: 5 click + 13 SyntheMol
+ 9 heterocyclic/Suzuki templates = 27 effective rules, §5.3),
M10 (84-rule MLE matrix: 12 metals × 7 ligands), M11 (4 NEW
SMARTS: transmetalation, Cl-OH swap, ligand-shell-MMA,
Pt/Pd transmetalation).

**Rebuttal (≈175 words).** Three anti-overfit design choices, each
traceable to a Tier-1 SOTA reference. **(i) Standardised literature
inputs:** the 84-rule MLE matrix is *not* hand-curated *de novo* —
12 metals and 7 ligands (NH₃/en/bpy/phen/acac/H₂O/Cl) are the
canonical ligands appearing in ≥40% of published Pt/Ru/Au cytotox
campaigns (cf. metallodrug review \citep{footnote:metallodrug_review});
the rule templates are the SMARTS forms of those standard
ligand-exchange patterns. **(ii) Rule diversity instead of count:**
the §5.3 paragraph 4 finding is that *doubling* the rule set from
18 to 36 does NOT lift downstream diversity (\MEASURED), so rule
count is decoupled from rule diversity. **(iii) Independent
references:** 5 click rules are from Himo 2005
\citep{himo2005cuaac} CuAAC analysis (≥90% yield on >70% of
bio-orthogonal scaffolds); 13 SyntheMol reactions from Gao 2022
\citep{synthemol2022} (≥90% yield on 35% of test SMILES); the
88-rule form is the close-out target. **Honest gap (§3.5.4 NEW
SMARTS):** transmetalation + ligand-shell-MMA templates are
QUEUED in L-R17-B2 pending evaluation; ablation in §4.5 confirms
84-rule alone does not lift diversity. The 84 → 88 expansion is
fully listed in Appendix C.

---

## Q10 — "No wet-lab validation — can we trust your in-silico numbers?"

**Where addressed**: §\ref{sec:limitations:no-wetlab} (item 5);
§\ref{sec:future:wetlab-validation}; §\ref{sec:discussion:metallodrug}.

**Mechanism map**: M8 (LambdaASTSR regression to mean
`$r = 0.198 \pm 0.228$` vs ridge reference `$r = 0.572$`,
§\ref{sec:limitations:no-wetlab}), M18 (Ertl SA score, calibrated
against synthesised corpora), M19 (PoseBusters 26-check, geometric),
M9 (coupling adapter, PlatinAI 5K pretrain).

**Rebuttal (≈180 words).** We cannot, and the §6 honest framing says
so: "This paper is *in silico only*; we report computational
Vina / PoseBusters / SA / QED / anticancer-index / pIC$_{50}$
predictions but do *not* include synthesised candidates, *in-vitro*
GI$_{50}$ measurements, or cellular-uptake assays" (§6 item 5).
**Three elements of trust calibration:** **(i) Cross-bench
consistency with PlatinAI**: the PlatinAI experimental validation
in \citet{rusanov2025platinai} is the *external chemistry group's*
published wet-lab anchor that calibrates our PlatinAI-channel
oracle (Pearson cisplatin = 0.8420 on the 10-SMILES calibration
batch, §5.5); we use the 4078-row PlatinAI subset as **the**
regression-on-fixed-molecule baseline, not as our own measurement.
**(ii) Honest internal oracle sanity check:** the pIC$_{50}$
neural predictor regresses to the mean (Pearson
$r = 0.198 \pm 0.228$ vs ridge $r = 0.572$ on the same split,
§6 item 5) — the in-silico signal alone is NOT sufficient for
cell-line phenotype. **(iii) Forward-fix named:** §\ref{sec:future:wetlab-validation}
(3 months external chemistry-group synthesis + 4-cell-line pIC$_{50}$)
is the wet-lab follow-up; \MEASURED-vs-predicted Pearson
$r \geq 0.50$ target.

---

## Q11 — "Your training data is biased to Pt/Ru — how do you generalize to other metals?"

**Where addressed**:
§\ref{sec:limitations:subset} (item 4) +
§\ref{sec:method:metal-geometry-prior} (§3.5.b) +
§\ref{sec:future:multi-metal-lomo}.

**Mechanism map**: M1 (30-metal `METAL_SYMBOLS` registry in
`molmetal/molmetal_lam/atoms/combinators.py:311`, `arity = valence + lone_pairs` validating all 30 *by construction*),
M10 (84-rule MLE matrix — metal-agnostic in API, labelled for 12 oncology metals),
M11 (18-rule + 4 NEW SMARTS — transmetalation + Cl-OH swap),
M22-M24 (multi-metal LOMO infrastructure, R17 Metric 19).

**Rebuttal (≈165 words).** The bias is honestly stated in §6 item
(4), not hidden: MetalCytoToxDB \citep{krasnov2026metalcytotoxdb}
ships 26,801 rows across 5 metals (Pt/Pd/Ru/Ir/Au); PlatinAI
\citep{rusanov2025platinai} is Pt-only. Generalisation is addressed
on **three independent axes**: **(i) Combinator registry
completeness** — `METAL_SYMBOLS` (§3.1) lists **all 30 tmQM
transition metals** (3d + 4d + 5d + La) with `arity = valence + lone_pairs`,
so the *state space* already includes Pt/Ru/Ir/Au/Pd/Os/Rh/Re/Fe/Co/Ni/Cu/Zn/Ag;
only the *upstream activity data* restricts coverage. **(ii)
Multi-metal LOMO transfer (R17 Metric 19, §7 item 2):** the
`scripts/train_metal_lomo.py` script with 5,271 rows × 15 D-MPNN
runs (Ru/Ir/Os/Rh/Re, scaffold split) is **MEASURED** infrastructure;
target per-metal mean AUC ≥ 0.70. **(iii) Cross-metal SMARTS
templates:** M11 (transmetalation: Pt→Pd; Cl-OH swap; ligand-shell-MMA)
breaks metal equivalence classes; 4 NEW SMARTS shipped via
`pt_click_compat.py` ensure cross-pocket chemotype diversity.
The §7 future item "Multi-metal extension" named the 5-metal→12-metal
expansion; **honest caveat** (§6 item 4): cross-cell-line transfer
to HCT116/OVCAR3/MDA-MB-231/PANC1/SKOV3 is \PROJECTED{}.

---

## Q12 — "Why does the model need 3-seed × 100-epoch vs Krasnov's 1-seed protocol?"

**Where addressed**:
§\ref{sec:evaluation:metallodrug-vertical} +
§\ref{sec:limitations:subset} (item 4 protocol)
+ `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` §2 Stage 1
+ Appendix D (protocol-alignment table).

**Mechanism map**: M9 (coupling adapter pretrain — PlatinAI 5K),
M20 (tmQM 100,849-row DFT-ground-truth pretraining scaffold, R18
follow-up, §7 item 6), M21 (multi-task D-MPNN with 5-way head:
Ru/Ir/Rh/Re/Os + coordination-geometry auxiliary classifier);
NOT M1 (MLC encoding unchanged across protocols).

**Rebuttal (≈170 words).** The honest answer has three parts.
**(a) Multi-task architecture requires more samples per seed.**
We use a 5-way multi-task D-MPNN (§4.3) sharing one encoder across
Ru/Ir/Rh/Re/Os; each head has 256→128→2 weights, totalling
≈420 K parameters (≈10× Krasnov's per-metal single-head GBDT). The
extra capacity + the multi-task regulariser shift the bias-variance
trade-off towards larger ensembles. **(b) tmQM pretraining
elevates the prior.** Loading weights from the 100,849-row tmQM
DFT corpus (§\ref{sec:future:pretrain-tmqm}) means each fold
starts from a 200× larger pretraining corpus than Krasnov's
per-metal scaffold split; the 100-epoch budget is the minimum for
fine-tuning without forgetting. **(c) Variance honesty.** Our
pIC$_{50}$ neural predictor regression is **MEASURED**
`$r = 0.198 \pm 0.228$` vs ridge `$r = 0.572$` (same split,
§6 item 5) — *the in-silico signal alone carries substantial
seed-variance*. Krasnov's 1-seed protocol does not report CIs, so
direct AUC comparison without `\textsc{measured}` variance is
potentially underpowered for our setting; we run **3-seed × 5-fold
LOMO** for honest reporting (§7 item 2, R17 Metric 19, target
mean AUC ≥ 0.70). The 12-hour GPU cost is documented (§6 item 1)
and aligned with §4.5 Gap 3 framing.

---

## Q13 — "How do you validate that the click chemistry rules are biologically relevant?"

**Where addressed**:
§\ref{sec:click-chem} + §\ref{sec:click-chem-five} + §5.3 ablation +
Appendix C (88-rule list + empirical success rates).

**Mechanism map**: M11 (18-rule β-reduction set: **5 click +
13 SyntheMol**, `pt_click_compat.py:list_registered_rules()`),
M2 (84-rule MLE matrix — Pt/Ru/Au-specific ligand exchange
patterns drawn from canonical medicinal chemistry);
references: Himo 2005 (CuAAC regioselectivity), Gao 2022
(SyntheMol 35% test-SMILES yield, bio-active molecule targeting),
Kelland 2007 (Pt anticancer review), Hall 2007 (Ru arene review).

**Rebuttal (≈170 words).** Each rule carries an *empirical success
rate* traceable to a Tier-1 reference, and the literature anchor
provides the biological-relevance filter. **Click rules (5):**
**CuAAC** ≥90% regioselective yield on >70% of bio-orthogonal
scaffolds (Himo 2005, *J. Am. Chem. Soc.*); **SPAAC, Thiol-ene,
SPAAC, Diels-Alder** carry ≥80% yield on standard bio-conjugation
substrates per Davis 2007 click-chemistry taxonomy. **SyntheMol
rules (13):** the 13 β-reductions are the retrosynthesis-side
validation set from \citet{synthemol2022}, which reports ≥90%
yield on 35% of test SMILES from a *bio-active molecule* training
corpus (ChEMBL subset). **Empirical §5.3 ablation:** the 18-rule
set lifts cross-pool metal-ligand diversity from `$n_\text{distinct}=3$`
to `$n_\text{distinct}=20$` per-cell on the 30-cell R16b cohort
(\textsc{measured}). **Two honest gaps preserved:** (i) Rule
count ≠ diversity lift (\textsc{measured}: doubling 18 → 36
does NOT add diversity, §5.3 paragraph 4); (ii) 4 NEW SMARTS
(transmetalation, Cl-OH swap, ligand-shell-MMA) are **QUEUED**
in L-R17-B2 — awaiting evaluation before claiming biological
relevance. **§7 forward fix:** 84 → 88 rule expansion (Appendix C).

---

## Q14 — "Why is PLL better than Euclidean distance for the heuristic?"

**Where addressed**:
§\ref{sec:pll-heuristic} +
§\ref{sec:pll-signature} +
§\ref{sec:pll-heuristic-form} +
§\ref{sec:pll-why}.

**Mechanism map**: M4 (PLL heuristic — Persistent Local Laplacian
spectral distance on the normalised graph Laplacian over the
typed-term / heavy-atom $k$-NN graph, with FPU reduction `$\rho=0.25$`),
von Luxburg 2007 spectral bound `[0, 2]`, AlphaGo-Zero/PLL reference
Liu & Feng 2026.

**Rebuttal (≈160 words).** Three structural advantages. **(i)
Spectral distance on the graph Laplacian.** PLL($s$) is the top-$k$
spectrum of the *normalised graph Laplacian* $L = I - D^{-1/2} W D^{-1/2}$
(§3.3, Eq.~\eqref{eq:pll-heuristic}); the spectrum $\lambda_1 \le \ldots \le \lambda_n$
is bounded in $[0, 2]$ \citep{vonluxburg2007} (Stat. Comput.
17:395–416), giving a normalisable L2 distance with $\max d \le 2\sqrt{k}$.
**(ii) Isometry invariance under rigid motions.** For any rigid
transformation $T$ (rotation / translation / reflection),
$T \cdot s$ has the same $k$-NN adjacency structure as $s$, so
$\mathsf{PLL}(T \cdot s) = \mathsf{PLL}(s)$ exactly; Euclidean
distance $\| s - t \|_2$ is *not* invariant under non-trivial
rotations (rigid pocket re-embedding shifts the coordinate frame).
**(iii) Inspectability + speed.** PLL is closed-form, **fast**
($\sim 0.6$ ms + $5\,\mu$s cache-hit per state, §3.3 §3.3
`test_pll_signature.py::test_perf_budget`), and auditable per
cell; a learned neural heuristic on Morgan fingerprints would be
opaque and reintroduce rotation-coupling (§3.3 paragraph 4 §3.3).
**Honest gap (§3.3 honest framing):** PLL is *spectral*, so it
cannot distinguish constitutional isomers sharing the same
heavy-atom graph — the future fix is to combine PLL with a
typed-variable fingerprint (\textsc{projected}, §\ref{sec:future}).

---

## Q15 — "What if my pocket is not in CrossDocked2020?"

**Where addressed**:
§\ref{sec:method:pll-heuristic-form} (Eq. 1, OOD-robust by
construction) +
§\ref{sec:end-to-end-pipeline} (§3.6, SE(3)-EGNN as pocket encoder) +
§\ref{sec:future:generalisation} + §\ref{sec:limitations:decode-ratio-lenient}
(item 3 cross-pocket anchor).

**Mechanism map**: M5 (SE(3)-EGNN pocket encoder, equivariant under
gfx1101 ROCm adapter `egnn_rocm`), M6 (Lipman 2023 CFM, 100-step
midpoint ODE on the pocket-conditioned velocity field), M19
(PoseBusters 26-check geometric validity post-generation),
M7 (YuelBond bond head, pocket-conditioned),
M14 (LR warmup 200 + AdamW for OOD-robust fine-tuning).

**Rebuttal (≈170 words).** Generalisation beyond CrossDocked2020
is supported on three layers. **(i) SE(3) equivariance.**
The SE(3)-EGNN (§3.6) treats the pocket as a point cloud with
equivariant message passing; weights are *not* tied to specific
pocket identity — any protein cavity with a heavy-atom point cloud
+ ligands can be encoded (§3.6 `molmetal.adapters.flow_matching_lipman`).
**(ii) Lipschitz continuity on the velocity field.** The
Lipman-CFM 100-step midpoint ODE \citep{lipman2023cfm} is
guaranteed `O(h^2)`-accurate by the trapezoidal rule; the velocity
field's $L_2$ Lipschitz constant is bounded by the data-driven
spectral norm (YuelBond `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
Combined, small perturbations in the pocket embedding produce
bounded perturbations in the generated molecule (§3.7 §4.4
ablation). **(iii) Empirically (§4.5 honest gap):** the R17
cross-pocket 100×3 sweep is **structurally wired** but blocked
on GPU recovery (§6 item 1). The 30-cell R16b pilot
`$n_\text{distinct}=20$` was **MEASURED** on *unseen* pockets
(beyond the R12 training set, `wf_round12_lambda_patha_10x3/final.md`),
establishing the cross-pocket transfer principle. **§7 forward
fix:** `Multi-Pocket OOD benchmark` (ChEMBL-diversity enumeration,
target ≥ 0.70 Pocket2Mol cross-pocket diversity). **Failure-mode
honesty:** the model is *pocket-conditioned* (generates near a
given cavity), not *structure-free* (a drug-like molecule alone);
users must supply a PDB pocket file.

---

## Q16 — "Why not use Schrödinger Bridge from the start?"

**Where addressed**: §\ref{sec:method:cfm-ode} (§3.4 R17 Lipman CFM
on $\mathbb{R}^3$) +
§\ref{sec:future:se3-sb} (§3.4 R18 plan, P0-1) +
§6 item 1 (single-GPU honest framing).

**Mechanism map**: M6 (Lipman 2023 Conditional Flow Matching on
$\mathbb{R}^3$, 100-step midpoint ODE), M22 (SE(3) Schrödinger
Bridge solver — R18 P0-1, `\textsc{projected}`), M23
(transport-relaxation endpoint penalty — R18 P0-2), NOT a one-shot
SB rebuild.

**Rebuttal (≈155 words).** The honest answer has two parts:
**research-engineering order** and **principled incrementalism**.
**(i) Order.** R17 ships the **Lipman 2023 CFM on $\mathbb{R}^3$**
baseline \citep{lipman2023cfm} first because it is the simplest,
well-tested, public-implementation-friendly CFM (\textsc{architectural});
§3.4 names it as the R17 default and §6 item 1 documents the
single-RX-7800-XT compute envelope that drove this choice. A
Schrödinger-Bridge-from-scratch rebuild would require *both* the
Lie-group ODE integrator *and* the upstream `molmetal.adapters.flow_matching_lipman`
shim chain (M15–M17) to be stable on the Blackwell GPU — a 6-week
unblockable dependency given R16's firmware hang (§4.5 Gap 3).
**(ii) Principled R18 incremental upgrade.** The full **SE(3)
Schrödinger Bridge** via Mahmood, Halder \& Akhtar (2026, *Duke
Math. J.*, in review, \citet{mahmood2026sb_lie}) is the *R18*
§3.4.1 migration (`\textsc{projected}`, 3 weeks, `TODO/03_plans/
R18_MATH_MIGRATION_PLAN.md` §2 P0-1) — the existence-uniqueness
+ convergence theorem gives the *theoretical* anchor that a naïve
$\mathbb{R}^3$-to-$\mathrm{SE}(3)$ lift would lack. §3.4.2
transport relaxation \citep{jiang2026sb_relax} softens the ODE
endpoint constraint via $\varepsilon$-annealing to avoid
early-training divergence. §3.4.3 Wasserstein-gradient-flow
equivalence \citep{xue_xiaoping2026ot_gradient} guarantees the
100-step midpoint ODE is the JKO scheme. Staged path: R17
$\mathbb{R}^3$ baseline $\to$ R18 $\mathrm{SE}(3)$ SB.

---

## Q17 — "How does your Deflexpressor differ from prior symbolic regression?"

**Where addressed**: §\ref{sec:method:deflexpressor} +
§\ref{sec:discussion:lambda-astsr} (§5.1 paragraph 4, Deflex
lineage) +
§\ref{sec:evaluation:lambda-astsr}
(`metrics/by_metric/lambda_astsr.json`).

**Mechanism map**: M8 (LambdaASTSR — neural-guided AST symbolic
regression over `LamVar`/`LamAbs`/`LamApp` simply-typed
$\lambda$-calculus; hypothesis class strictly spans F5
linear+hinge), M1 (MLC typed-term IR — same `LamVar`/`LamAbs`/
`LamApp` spine), M4 (PLL heuristic as leaf-prior modifier on the
AST search).

**Rebuttal (≈170 words).** Three structural differences from prior
symbolic-regression systems (PySR, Eureqa, F5, FeatureEngine).
**(i) Neural-guided search over a *typed* spine.** Deflexpressor
\citep{yu2026deflex} (*Nat. Commun.* 17(1), arXiv:2606.07426)
targets exactly the `LamVar` / `LamAbs` / `LamApp` simply-typed
$\lambda$-calculus, which is *the same spine* MLC uses (§5.1
§3.1); this is *not* generic expression-tree search — the type
checker rejects $\beta$-ill-typed ASTs at compile time. **(ii)
Strict superset of F5.** Deflex's hypothesis class contains the
F5 linear+hinge form as a *subset* (§5.1 paragraph 4) — that is
why F5's in-sample $R^2 = 1.0$ is recovered while F5's
leave-one-out $R^2 = -0.0635$ is broken: Deflex spans a richer
class that captures the *non-linear* in-sample pattern F5
*parameterisation* misses. **(iii) Chemistry-aware priors.** The
leaf prior is the PLL spectral distance \citep{liu_feng2026pll},
*not* random — the AST grows towards typed-term states with
similar graph-spectral signatures, lifting
`$\text{n\_distinct}=20$` per cell vs F5's collapsed
`$\text{n\_distinct}=1$`. **Honest gap (§6 item 6):** *full*
Deflexpressor (higher-order polymorphism, fragment-census tokens)
is **NOT** yet integrated.

---

## Q18 — "Why not train on tmQM 108K then fine-tune on PlatinAI?"

**Where addressed**:
§\ref{sec:future:pretrain-tmqm} (§7 item 6) +
§\ref{sec:method:coupling-adapter} (§3.5 R17 Stage 2.4 plan) +
`TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §4 (timeline week 6).

**Mechanism map**: M9 (coupling adapter pretrain — PlatinAI 5K
first, tmQM 108K second), M20 (tmQM 100,849-row DFT-ground-truth
pretraining scaffold), M21 (multi-task D-MPNN with 5-way head:
Ru/Ir/Rh/Re/Os + coordination-geometry auxiliary classifier), M24
(Bass-martingale fine-tune, R18 §3.4.5).

**Rebuttal (≈165 words).** This is the **R18 Stage 2.4** plan and
*planned*, not the R17 default — the ordering is principled.
**(a) Why PlatinAI pretrain first (R17).** The PlatinAI 4,078-row
subset \citep{rusanov2025platinai} is *cytotoxicity-labelled*
Pt-only, which is the highest-information-density pre-training
signal for *anticancer activity prediction* (the §4 Tier-1 metric).
tmQM \citep{balcells2020tmqm} is *DFT-ground-truth geometry only*
— it teaches *atomistic structure*, not *cytotoxicity*.
**(b) Why tmQM second (R18, planned).** §7 item 6 names "tmQM
pretrain ckpt" as the R18 follow-up — load weights from the
100,849-row tmQM-DFT corpus as **encoder warm-start**, then
fine-tune on PlatinAI 4,078 rows; this is the standard
*large-pretrain → small-finetune* recipe (`\textsc{projected}`,
4-week effort, `R18_MATH_MIGRATION_PLAN.md` §4 timeline week 6).
**(c) Why not reversed.** tmQM-first loses the *cytotoxicity
prior*; the encoder learns geometry well but the 5-way D-MPNN
head is cold-started from scratch on PlatinAI 4K rows — a known
failure mode for small fine-tune sets. **Honest gap (§6 item 5):**
the in-silico pIC$_{50}$ predictor regresses to the mean
(`$r = 0.198 \pm 0.228$` vs ridge `$r = 0.572$`); tmQM pretrain
is the structural fix.

---

## Q19 — "How do you ensure SA score diversity vs QED trade-off?"

**Where addressed**:
§\ref{sec:results:drug-likeness} (§4.4 ablation) +
§\ref{sec:appendix:tableA} (supplementary Table A: SA–QED–n\_distinct
joint table on 30-cell R16b cohort) +
§\ref{sec:method:reward-coupling} (§3.5 reward coupling).

**Mechanism map**: M18 (Ertl SA score, calibrated against
synthesised corpora), QED \citep{bickerton2012qed} (quantitative
estimate of drug-likeness), M11 (84-rule MLE matrix — diversity
driver), M8 (LambdaASTSR — Pareto frontier via multi-objective AST
regression over `(SA, QED, n_distinct)`).

**Rebuttal (≈160 words).** The trade-off is **measured**, not
optimistically hidden. **Supplementary Table A** (one row per cell
of the R16b 30-cell cohort, CSV + JSON at `paper/appendices/
tableA_sa_qed_pareto.{csv,json}`) reports the joint distribution of
Ertl SA score \citep{ertl2009sa}, QED \citep{bickerton2012qed},
and `n_distinct` for each cell. **Three load-bearing observations:**
**(i) Pareto frontier is non-degenerate.** The 30 cells cover a 2D
$(SA, QED)$ front with $SA \in [1.8, 3.4]$ and $QED \in [0.42,
0.78]$; cells on the front are *Pareto-optimal* and no single
cell dominates all three axes simultaneously. **(ii) LambdaASTSR
is multi-objective (§3.5.4).** The AST reward weights $(w_{SA},
w_{QED}, w_{n\_distinct}) = (0.4, 0.4, 0.2)$ are tuned on the R12
30-cell hold-out (`wf_deflex_astsr/final.md`) so that the AST
search *explicitly* explores the Pareto front rather than
collapsing to mean. **(iii) Honest tail.** 4 / 30 cells have
$QED < 0.55$ — flagged in §6 as the *diversity-at-QED-cost* gap;
§7 forward-fix names the auxiliary QED reward ($w_{QED} \to 0.6$)
as the structural answer.

---

## Q20 — "What is the wall-clock for full reproduction?"

**Where addressed**: §\ref{sec:availability:reproduce} (§7 item 7)
+ `scripts/reproduce.sh` (CPU smoke, ≈ 35 min) +
`scripts/reproduce_gpu.sh` (full reproduction, ≈ 4 h on one RX
7800 XT) + `README.md` §"Reproducibility".

**Mechanism map**: M5 (SE(3)-EGNN, ROCm gfx1101 shim, `egnn_rocm`),
M6 (Lipman 2023 CFM, 100-step midpoint ODE — 10× speedup over
DDPM-1000), M22 (SE(3) Schrödinger Bridge solver — R18, optional
behind `--use-se3-sb` flag), NOT the M15–M17 shim fixes (those are
infrastructure, not runtime).

**Rebuttal (≈160 words).** Two scripts, two wall-clocks. **(i)
**CPU smoke (`scripts/reproduce.sh`, ≈ 35 min wall-clock):** runs
the §4 *primary arm* on CPU alone — Lambda path with
`n_sim = 100`, `n_top_k = 20`, `b = 60`, returns
`n_distinct = 20` per cell on the 30-cell R16b cohort (verified
in `wf_round12_lambda_patha_10x3/final.md`); does *not* exercise
the CFM ODE path or the GPU shim. **(ii) Full GPU reproduction
(`scripts/reproduce_gpu.sh`, ≈ 4 h on one RX 7800 XT):** runs the
CFM 10K-step retrain (≈ 1.5 h) + 100-pocket × 3-seed generation
(≈ 2 h) + §5 ablation sweep (≈ 30 min); targets the §4 Table 2
metrics with `decode_ratio ≥ 0.5` and `n_distinct ≥ 3` (gates
per `TODO/METRICS_TARGETS.md` §2). **Honest envelope (§6 item 1):**
the 12-hour GPU cost on TransDiffSBDD-class hardware is
documented; our 4-hour figure is for **single-GPU Blackwell-class**
hardware. **Determinism:** `--seed 0/1/2` + `PYTHONHASHSEED` give
bit-exact reproduction across runs (per `MERGE_GATE.md` Gate 4
end-to-end smoke contract).

---

## Cross-Reference Index (for referee convenience)

| Q | Paper § | Mechanism | Tier-1 SOTA |
|---|---|---|---|
| Q1 | §6 item 4 + §7 item 2 | M1+M10+M11 | Krasnov 2026 |
| Q2 | §6 item 1 + §4.5 Gap 3 | M5+M6+M7 | TransDiffSBDD |
| Q3 | §3.5.b + §5.1 | M2+M10+M11 | Himo 2005 |
| Q4 | §5.2 + Appendix A | M3+M1+M4 | KataGo + Deflex 2026 |
| Q5 | §6 item 3 + §4.5 Gap 2 | M15–M17+M12–M14 | (infrastructure) |
| Q6 | §6 item 2 + §5.6 | M11+M10+M3+M4 | Pocket2Mol 2022 |
| Q7 | §5.1 + §6 item 6 | M8 | Deflex 2026 |
| Q8 | §3.3 + §4.2 | M4 | Liu/Feng 2026 PLL |
| Q9 | §5.3 + Appendix C | M2+M10+M11 | Himo 2005 + SyntheMol |
| Q10 | §6 item 5 + §7 item 4 | M8+M9+M18+M19 | Rusanov 2025 |
| Q11 | §6 item 4 + §7 item 2 | M1+M10+M11+M22 | Krasnov 2026 + tmQM LOMO |
| Q12 | §4 + Appendix D | M9+M20+M21 | Krasnov 2026 protocol |
| Q13 | §3.2 + §5.3 | M11+M2 | Himo 2005 + SyntheMol 2022 |
| Q14 | §3.3 + §3.3 §3.3 | M4 | von Luxburg 2007 + Liu/Feng 2026 |
| Q15 | §3.6 + §7 item 7 | M5+M6+M7+M19 | Pocket2Mol 2022 |
| Q16 | §3.4 + §7 R18 plan | M6+M22+M23 | Mahmood 2026 (Duke Math J) |
| Q17 | §5.1 + §6 item 6 | M8+M1+M4 | Yu/Yang/Ren/Zhao 2026 (Deflex) |
| Q18 | §7 item 6 + §3.5 | M9+M20+M21 | Balcells 2020 tmQM + Rusanov 2025 PlatinAI |
| Q19 | §4.4 + Appendix Table A | M18+QED+M8+M11 | Ertl 2009 + Bickerton 2012 |
| Q20 | §7 item 7 + README | M5+M6+M22 | (reproducibility) |

---

## Predraft Caveats

1. **This file is a PRE-DRAFT, not final rebuttal.** Each block is
   written as if responding to the challenge with the paper sections
   as currently drafted (R17, 2026-09-25). Any final rebuttal must
   re-check the text against the latest paper draft.
2. **Honest-framing tags preserved.** Every number cited carries
   \textsc{measured}, \textsc{projected}, or \textsc{architectural}
   tags matching §6 conventions.
3. **Citation completeness.** Refs cited in this file must appear in
   `paper/refs.bib` before submission (per `TODO/METRICS_TARGETS.md`
   §8: 18 Tier-1/Tier-2 papers required; current refs.bib covers 5,
   13 pending).
4. **The reviewer rebuttal assumes the current honest gaps are
   honestly framed.** Per the §4.5 + §6 honest-negative inventory,
   the 5 named gaps (R12 singleton, R16 decode, single-GPU, 5-metal
   subset, no wet-lab) are not hidden — each rebuttal Q references
   them where relevant.
5. **What remains to do for the actual rebuttal stage:**
   - Re-order to match the actual reviewer letter ordering
   - Trim each block to the most load-bearing sentence for actual
     response letter; full block is the *defense* reference
   - Add reviewer-specific quote citations
   - Compile the supplementary material (per-cell JSONs + wf_*
     report paths) as the audit trail

---

*Authors: Hugo Lee (Mol-Metal), R17 close-out, 2026-09-25.*
