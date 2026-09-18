# Phase 1 — Paper-Ablation Data Inventory

**Date:** 2026-09-15
**Workflow:** Phase 1, Step 8 (paper content integration)
**Operator:** workflow paper-ablation
**Project root:** /home/hugo/codes/try_triton_on_rocm

This document is the data inventory for the paper-ablation workflow
(§4.6 PB + §5 P0-metrics + click-rule + metal-coord). It enumerates
the MEASURED numbers from the seven input artefacts, identifies
the DESIGN cells that are candidates for promotion in §4.6 and §5,
and tabulates the honest framing caveats (search-bound vs real
measure). All numbers are reproduced verbatim from the cited
artefact; no number is fabricated or extrapolated.

The integration target is **paper/sections/04_evaluation.tex §4.6
ONLY** (not §4.1–4.5) and **paper/sections/05_ablation.tex**. The
remaining sections are owned by separate workflows:
- §4.1–§4.5 + main.tex + refs.bib → Workflow 1
- §4.6 (PB pass-rate column) → this workflow (Phase 1 inventory +
  Phase 2 integrate)
- §5 ablation (full §5.1–§5.8 + new §5.9 click-rule effect sizes) →
  this workflow

---

## 1. Artefact inventory

The seven input artefacts read by Phase 1.

| # | Artefact | Path | Status |
|---|----------|------|--------|
| 1 | WF-PB-Pass-10x3-Smoke (30-cell PB panel) | `molmetal/reports/wf_pb_pass_10x3_smoke/final.md` | READ (30 cells, all `pb_pass_rate=None`, 15 `no_candidates` + 15 `seed_only`) |
| 2 | WF-P0-Metrics (9 P0 anticancer metrics) | `molmetal/reports/wf_p0_metrics_smoke/{summary.md, report.json}` | READ (1 cell × 1 seed; 9 metrics MEASURED) |
| 3 | WF-Phase3B metrics_v2 (8 drug-likeness metrics) | `molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md` | READ (module shipped; 20 tests pass; **NOT** wired into r4_lambda_only_run.py yet) |
| 4 | WF-Phase3F click-rule effect-size study | `molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md` | READ (script shipped; 6/6 tests pass; **NO production run**, all numbers in §8 are PROJECTED) |
| 5 | WF-Phase3G metal-coord probe | `molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md` | READ (probe shipped; 13/13 tests pass; **NOT** wired into r4_lambda_only_run.py yet) |
| 6 | WF-PB-Pass-Real-Dock (1-pocket PB smoke) | `molmetal/reports/wf_pb_pass_real_dock/final.md` | READ (1 cell × 1 seed; `pb_pass_rate=1.000` on chemistry-only `mol` mode, Vina=-6.929 kcal/mol) |
| 7 | metrics/by_metric/pb_pass_rate.json | `molmetal/metrics/by_metric/pb_pass_rate.json` | **MISSING** (file does not exist on disk; metrics/ directory absent) |

**Missing file flag (honest framing).** The path
`molmetal/metrics/by_metric/pb_pass_rate.json` referenced in the
workflow brief does NOT exist in the working tree. No
`molmetal/metrics/` directory exists either (verified via
`find /home/hugo/codes/try_triton_on_rocm -type d -name metrics`).
The honest outcome is: **no JSON registry metric file is
available**; all PB numbers in this inventory are sourced
directly from the `wf_pb_pass_real_dock/final.md` and
`wf_pb_pass_10x3_smoke/final.md` reports and their per-cell JSON
artefacts in `wf_pb_pass_real_dock/r4c.json` and
`wf_pb_pass_10x3_smoke/r4c.json`. The metrics registry path in
the brief may reference a Round-15 deliverable that has not yet
been built.

---

## 2. MEASURED numbers table (cell × metric × value × n)

### 2.1 PoseBusters pass rate (artefacts 1 + 6)

#### 2.1.1 Single-pocket smoke (artefact 6, `wf_pb_pass_real_dock`)

| pocket | seed | n_cands | n_gen | n_docked | n_pb_pass | pb_pass_rate | vina_best (kcal/mol) | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| test_000 | 42 | 1 | 1 | 1 | 1 | **1.000** | -6.929 | completed (`pb_mode=mol`, chemistry-only, no protein clash) |

- Single SMILES accepted: `Cc1ccc(-c2ccc(C(=O)CS)cc1)cc1` (thiomethyl
  ketone-biaryl, ETKDGv3 + MMFF94 embedded, 25/25 chemistry+geometry
  PoseBusters `mol`-mode checks passed).
- SA mean (top1) = 1.773; QED mean (top1) = 0.640; Lipinski pass = 1.000.
- Hardware: CPU Vina (GPU dispatch incompatible with `--engine both`
  single-column); PoseBusters CPU-only; ROCm 7.2 / triton-rocm 3.8.0
  / RX 7800 XT gfx1101 wave64 declared but NOT engaged.
- **Honest framing:** n=1 is path-correctness smoke, NOT a statistical
  pass-rate. The "1.000" is one chemistry-clean molecule out of one.
  `pb_mode="mol"` validates chemistry+geometry in isolation; protein-
  aware clash check is NOT performed. The 30–50% projected pass-rate
  window in the brief is UNTESTED at scale.

#### 2.1.2 30-cell panel (artefact 1, `wf_pb_pass_10x3_smoke`)

Per-pocket × per-seed:

| pocket | seed=42 | seed=0 | seed=1234 | n_gen (all seeds) | n_cands (all seeds) | status |
|---|---|---|---|---:|---:|---|
| test_000 | None | None | None | 0 | 0 | no_candidates ×3 |
| test_001 | None | None | None | 0 | 3 | seed_only ×3 |
| test_002 | None | None | None | 0 | 0 | no_candidates ×3 |
| test_003 | None | None | None | 0 | 3 | seed_only ×3 |
| test_004 | None | None | None | 0 | 3 | seed_only ×3 |
| test_005 | None | None | None | 0 | 0 | no_candidates ×3 |
| test_006 | None | None | None | 0 | 3 | seed_only ×3 |
| test_007 | None | None | None | 0 | 0 | no_candidates ×3 |
| test_008 | None | None | None | 0 | 3 | seed_only ×3 |
| test_009 | None | None | None | 0 | 0 | no_candidates ×3 |

Per-seed aggregate (3 seeds × 10 pockets = 30 cells):

| seed | n_pockets | n_gen_total | n_cands_total | status mix |
|---|---:|---:|---:|---|
| 42   | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 0    | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |
| 1234 | 10 | 0 | 5 | no_candidates ×5, seed_only ×5 |

Aggregate:
- n_cells_total = 30
- n_pockets_ok = 0 (no cell accepted any generated molecule)
- n_generated_candidates_total = 0
- n_seed_candidates_total = 15
- **pb_pass_rate_aggregate = undefined (None)**
- physical_jobs_completed = 0; physical_n_docked = 0; physical_n_pb_pass = 0
- vina_best_kcal_mol = N/A
- wall_seconds_total = 144.3 s; wall_seconds_mean_per_cell = 4.81 s
- gap_vs_targetdiff_94 = **not_applicable** (zero denominator)

**Honest framing (search-bound vs PB-bound):** At
`n_simulations=100` with the strict
`synthesis_oracle=smarts ∧ symbolic_prior=True` gate combination
on the SOTA-aligned config, Lambda MCTS accepts **zero generated
molecules per cell**. PoseBusters therefore has nothing to evaluate
and the PB pass-rate aggregate is structurally undefined. The
pipeline is **search-bound, not PB-bound**. The 30 × None result
is an honest null, not a failure to report. The 30-cell wall-clock
cost (4.81 s/cell) sets the Round-13 100×3 production sweep budget
at ≤ ~24 min.

### 2.2 P0 anticancer metrics (artefact 2, `wf_p0_metrics_smoke`)

Source: `molmetal/reports/wf_p0_metrics_smoke/report.json`,
single cell (pocket=test_000, seed=42, n_simulations=100, n_top_k=20,
prior_enabled=True, metal_seed=None, click_rules=all-5).

| metric | value | n | honest framing |
|---|---:|---:|---|
| `validity_rate` | 1.0000 | 15/15 | MEASURED; RDKit-sanitisable |
| `uniqueness_rate` | 1.0000 | 15/15 | MEASURED; canonical SMILES dedup |
| `synthesizability_rate` | 1.0000 | 15/15 | MEASURED; β-NF + RDKit-valid |
| `metal_compliance_rate` | 0.0000 | 0/15 | MEASURED; honest zero — no `metal_seed`, no Pt/Ru/Ir centres |
| `reference_tanimoto` | 0.8400 | 15/15 | MEASURED; mean vs pocket ref |
| `diversity_tanimoto` | 0.0249 | 15/15 | MEASURED; SE(3) baseline, mean pairwise |
| `diversity_homotype` | 0.0100 | 15/15 | MEASURED; typed-variable enriched-vocab (Lambda-native) |
| `novelty` | 1.0000 | 15/15 | MEASURED; Tanimoto max vs training set |
| `logp_mean` | **-2.3979** | 15/15 | MEASURED; Wildman-Crippen octanol-water partition |
| `tpsa_mean` | **233.5373** | 15/15 | MEASURED; topological polar surface area (Å²) |
| `rotb_mean` | **9.6667** | 15/15 | MEASURED; rotatable-bond count |
| `coordination_number_mean` | **0.0000** | 15/15 | MEASURED; no metal centre — `molmetal_lam/metrics/anticancer_metric_suite.compute_coordination_number` |
| `monodentate_cl_count` | **0** | 15/15 | MEASURED; no Pt centre |
| `oxidation_state_distribution` | **(none)** | 0/15 | MEASURED (empty dict) — no metal centres to assign |
| `gsh_evasion_score` | **0.0000** | 15/15 | MEASURED; no GSH-labile leaving groups |
| `dna_kb_proxy` | **0.0000** | 15/15 | MEASURED; no DNA-binding handle |
| `anticancer_index` | **0.1250** | 15/15 | MEASURED; 2/16 ligand falls inside anticancer-target windows |
| `nfe` | 100 | 1/1 | MEASURED; n_simulations=100 budget |
| `wall_seconds` | 4.71 s | 1/1 | MEASURED; one cell, pure CPU |

**Honest framing (single-cell / single-seed).** All 9 P0 anticancer
metrics above are MEASURED on a single cell (test_000 × seed=42 × no
metal-seed × n_sim=100). The cell emits 15 distinct organic
candidates (n_distinct=15), all with reference_tanimoto=0.84
(typical Lambda "rewrite the reference" baseline). The
metal-aware cells (`coordination`, `oxidation_state`,
`monodentate_cl`, `gsh_evasion`, `dna_kb`, `anticancer_index`)
read 0 because no metal seed is set and the generated molecules
contain no Pt/Ru/Ir centres — these are NOT harness failures,
they are honest chemistry. The IV-antibiotic target windows
(`logP ∈ [2,5]`, `TPSA ∈ [60,150]`, `RotB < 10`) are not all
satisfied (`logP = -2.40`, `TPSA = 233.5` are far outside);
this is a property of the test_000 reference, not a metric
defect.

### 2.3 Drug-likeness / ADMET metrics_v2 (artefact 3)

Module `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (~480 LOC,
20/20 tests pass in 1.36 s). 8 metrics ship, each with a
`*_mean` batch helper, with `all_metrics_one(smi)` and
`all_metrics_mean(smiles_list)` aggregators.

| # | metric | formula sketch | range | lit anchor |
|---|---|---|---|---|
| 1 | `logp7_4` | `logP − 0.45·α₇.₄ + 0.30·β₇.₄` (substituted logP at pH 7.4) | unbounded | Weininger 1990 + Patrick 2009 |
| 2 | `gi50_proxy` | `4.5 + 0.30·MR − 0.015·TPSA − 0.50·(RotB/10) − 0.20·ArRings` | [0, 8] | Hou 2007 |
| 3 | `cell_permeability_logPapp` | `−4.0 + 0.33·logP − 0.013·TPSA + 0.40·HBD` | [-8, -3] | Hou 2007 / Mente 2015 |
| 4 | `herg_cardio_risk` | sum of 4 rule bonuses (MW, logP, TPSA, basic N) | [0, 1] | Veith 2009 |
| 5 | `ames_mutagen` | 1.0 if any of 6 SMARTS alerts match, else 0.0 | {0, 1} | Benigni-Richard 2005 |
| 6 | `hepatotox_index` | `0.40·(logP>3) + 0.30·(MW>500) + 0.30·(HBD≥2) + 0.20·(aniline\|hydrazine\|furan\|thiophene)` | [0, 1] | Hughes 2008 |
| 7 | `aqueous_solubility_logS` | `0.16 − 0.63·logP − 0.0062·MW + 0.066·RotB − 0.74·AromFrac` | [-12, 2] | Delaney 2004 (ESOL) |
| 8 | `plasma_protein_binding` | `sigmoid(−1.50 + 1.20·logP)` | [0, 1] | Obach 1999 |

End-to-end smoke on 6 reference SMILES (5 SMILES + cisplatin; from
`phase3b_metrics_v2.md` §4):

| SMILES | logp7_4 | gi50 | logPapp | hERG | AMES | hep | logS | PPB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CCO (ethanol) | -0.001 | 8.0 | -3.86 | 0.25 | 0.0 | 0.0 | -0.12 | 0.18 |
| c1ccccc1 (benzene) | 1.69 | 8.0 | -3.44 | 0.25 | 0.0 | 0.0 | -2.13 | 0.63 |
| Nc1ccc(cc1)N (pPDA) | 0.85 | 8.0 | -3.60 | 0.45 | **1.0** | 0.5 | -1.79 | 0.38 |
| CC(=O)Oc1ccccc1C(=O)O (aspirin) | 0.86 | 8.0 | -3.99 | 0.25 | 0.0 | 0.0 | -2.39 | 0.52 |
| CN1C=NC2=C1C(=O)N(C(=O)N2C)C (caffeine) | -1.03 | 8.0 | -5.14 | 0.25 | 0.0 | 0.0 | -1.14 | 0.06 |
| Cl[Pt](Cl)(N)N (cisplatin) | 0.20 | 8.0 | -3.81 | 0.45 | 0.0 | 0.3 | -1.81 | 0.22 |

**Honest framing (heuristic, not wet-lab calibrated).** The 8
formulas mirror published regressions/rules (Hou 2007, Veith 2009,
Delaney 2004, Obach 1999, etc.) **without** regression against any
held-out assay. They are suitable for **ranking and diversity
filtering**, not for absolute predictivity. Every claim that
touches wet-lab is SEARCHONLY/PROJECTED/DESIGN in the paper
posture; these metrics are no exception.

**Integration status:** NOT YET WIRED into `r4_lambda_only_run.py`
(per task graph contract, Phase-4 integrator owns that edit).
No batch-mean column exists in §4.6 today; the §5.8 P0 anticancer
panel (`paper/sections/05_ablation.tex`, WF-Section05-P0-Metrics)
covers only the 9 P0 metrics from artefact 2, NOT these 8.

### 2.4 Click-rule effect-size study (artefact 4)

Script `molmetal/scripts/click_rule_effect_size_study.py` (380 LOC,
6/6 tests pass in 0.17 s). Cohen's d pooled-variance formula
(Cohen 1988, eq. 2.3.5); rule-vs-others baseline (treatment = each
rule, control = empirical mean of other 4 rules).

5 × 4 panel schema (5 click rules × {diversity_tanimoto,
validity_rate, synthesizability_rate, metal_compliance_rate}):

| rule | diversity_tanimoto | validity_rate | synth_rate | metal_compliance |
|---|---|---|---|---|
| CuAAC | +0.3d (PROJECTED) | +0.2d (PROJECTED) | +0.1d (PROJECTED) | 0 (PROJECTED, default no-metal) |
| SPAAC | +0.1d | +0.1d | 0 | 0 |
| ThiolEne | -0.2d | -0.3d | -0.4d | 0 |
| Suzuki | -0.4d | -0.2d | -0.1d | 0 |
| AmideCoupling | -0.5d | -0.5d | -0.3d | 0 |

**Honest framing (no production run, all projections).** The §8
table in `phase3f_click_effect.md` is **PROJECTED** from prior
Lambda mini-pilots (WF-Round12-Lambda-Pilot, WF-Lambda-Metal-Pilot,
WF-Lambda-Only-MiniPilot), NOT measured from this script. The
smoke test (1 pocket × 1 seed per rule = 5 cells) returns
degenerate Cohen's d because treatment n=1 and baseline n=4.
A meaningful production run requires
`--pockets 5 --seeds 42 0 1234 --n-simulations 1000` (~30 s CPU
+ ~10 s metal-seed arm, 15 cells per rule, n=60 baseline). That
run is NOT in this artefact.

Lit anchors: Himo 2005 (CuAAC regio), Worrell 1984 (SPAAC),
Kolb 2001 (click canon), Suzuki 2011 (Pd cross-coupling),
Bickerton 2012 (QED), Cohen 1988 (effect size).

### 2.5 Metal coordination probe (artefact 5)

Module `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py`
(~340 LOC, 13/13 tests pass in 1.33 s). Coordinate-free
graph-theoretic probe that validates CN = canonical CN for the
metal/OS pair.

Algorithm (per molecule):
1. RDKit parse → locate metal centres in
   {Pt, Pd, Au, Ag, Ru, Ir, Rh, Os, Re, Fe, Co, Ni, Cu, Zn, Mn}
2. Infer OS (bracket `[Pt+4]` → +4, else per-element default)
3. Count CN via `GetNeighbors()` walk, H excluded
4. Classify geometry (CN=2 linear / CN=4 square_planar / CN=6
   octahedral / CN=5 square_pyramidal / …)
5. Verdict: `is_compliant = (CN == CANONICAL_CN[metal, OS]) ∧ (OS ∈ EXPECTED_OS[metal])`

`compliance_rate(smiles_list)` excludes organic-only inputs from
the denominator.

| metal | expected OS | canonical CN | geometry |
|---|---|---:|---|
| Pt | 0, 2, 4 | Pt(0)=4, Pt(II)=4, Pt(IV)=6 | square_planar / octahedral |
| Pd | 0, 2, 4 | Pd(II)=4 | square_planar |
| Au | 1, 3 | Au(I)=2, Au(III)=4 | linear / square_planar |
| Ag | 1, 2, 3 | Ag(I)=2 | linear |
| Ru | 2, 3, 4, 6, 8 | Ru(II)=6, Ru(III)=6 | octahedral |
| Ir | 1, 3, 4 | Ir(III)=6 | octahedral |
| Rh | 1, 2 | Rh(III)=6 | octahedral |
| Os | 2, 3, 4, 6, 8 | Os(IV)=6 | octahedral |
| Re | 1, 3, 4, 5, 7 | Re(V)=6 | octahedral |
| Fe | 0, 2, 3, 4 | Fe(II)=6 | octahedral |
| Co | 2, 3 | Co(II)=6 | octahedral |
| Ni | 0, 2, 3 | Ni(II)=6 | octahedral |
| Cu | 1, 2, 3 | Cu(II)=4 | square_planar |
| Zn | 2 | Zn(II)=4 | tetrahedral |
| Mn | 0, 2, 3, 4, 7 | Mn(II)=6 | octahedral |

Reference test cases (artefact 5 §5):
- `[NH3][Pt]([NH3])(Cl)Cl` (cisplatin): Pt, OS=2, CN=4, geometry=square_planar, **compliant=True**
- `[Pt+4](N)(N)(N)(N)(Cl)(Cl)` (Pt_IV bracket form): Pt, OS=4, CN=6, **compliant=True**
- `[Ru+3]` + 6 neighbours: Ru, OS=3, CN=6, **compliant=True**
- Dot-separated `N.N.Cl.Cl.[Pt]` form: CN=0 (RDKit limitation), **compliant=False** (honest limitation captured)

**Honest framing (graph-theoretic, not 3-D).** The probe does NOT
validate spatial arrangement (cis/trans, fac/mer); spatial
validation is the job of `MetalGeometryPrior` (3-D, torch-based).
Multi-component dot-separated SMILES have NO bonds between
fragments in the RDKit graph → flagged NON-COMPLIANT (correct
flagging, not silent over-claim). OS inference is heuristic; for
`[Pt]` (no charge) the probe falls back to per-element default
(Pt → +2). Bracket-tagged forms always take precedence.

**Integration status:** NOT YET WIRED into `r4_lambda_only_run.py`
(per task graph contract, Phase-4 integrator owns that edit). The
probe is CPU-only and drop-in ready (`probe_coordination(smi)`,
`probe_batch(smis)`, `compliance_rate(smis)`); 13 unit tests cover
the protocol + edge cases.

---

## 3. DESIGN cells that need promotion (Phase 2 candidates)

A "promotion" is `DESIGN → MEASURED` only when a MEASURED value
exists for the cell. The honest candidates from this inventory are:

### 3.1 §4.6 PB pass-rate column (PB cells)

**No promotion.** The 30-cell PB smoke aggregate is `None`
(undefined); the 1-pocket smoke has `pb_pass_rate=1.000` on
`n=1`. There is no MEASURED aggregate at the cell-cluster
level that can populate the §4.6 PB column for the 10×3 panel.
Per the §4.1 promotion rule (no DESIGN cell is silently
promoted), **all 30 §4.6 PB cells remain DESIGN**.

Honest artefact to add to §4.6: the existing §4.6 paragraph
"30-cell PB statistic panel" (`paper/sections/04_evaluation.tex`
lines 1481–1666) already records the 30 × None result, the 3
plausible causes, the `wall_seconds = 4.81 s/cell` scaling
estimate, and the "search-bound, not PB-bound" diagnosis. This
inventory confirms that paragraph is **already correctly
framed**; no edit is required in §4.6 for the 30-cell panel.

### 3.2 §5 ablation — P0 anticancer panel

The existing §5.8 P0 anticancer panel
(`paper/sections/05_ablation.tex`, WF-Section05-P0-Metrics,
2026-09-14) already integrates the 9 P0 metrics from artefact 2
on the single cell (test_000 × seed=42 × no metal-seed ×
n_sim=100). The §5.8 row values are:

- logp_mean = -2.40 (MEASURED)
- tpsa_mean = 233.5 (MEASURED)
- rotb_mean = 9.67 (MEASURED)
- coordination_number_mean = 0.00 (MEASURED, honest zero)
- monodentate_cl_count = 0 (MEASURED, honest zero)
- gsh_evasion_score = 0.000 (MEASURED, honest zero)
- dna_kb_proxy = 0.000 (MEASURED, honest zero)
- anticancer_index = 0.125 (MEASURED)

Source: `molmetal/reports/wf_p0_metrics_smoke` (single cell,
single seed). The §5.8 paragraph correctly tags these as a
single-cell smoke panel with 3 explicit follow-ups.

### 3.3 §5 ablation — click-rule effect-size panel (NEW §5.9 candidate)

**CANDIDATE for §5.9**, but the panel values are PROJECTED not
MEASURED (artefact 4 §8 explicitly says so). A production run
at `--pockets 5 --seeds 42 0 1234 --n-simulations 1000` is
required before any §5.9 cell can be promoted from
DESIGN/PROJECTED → MEASURED. The script is shipped and
test-verified (6/6 tests), but **NO production run was executed
in this artefact**. The §5.9 cell text in Phase 2 must therefore
be:

- Module + tests: MEASURED (shipped, tested, ready)
- 5×4 panel values: DESIGN (PROJECTED per Cohen's-d convention
  applied to the rule-vs-others baseline; replaced by MEASURED
  only after production run lands)
- Lit anchors (Himo 2005, Worrell 1984, Kolb 2001, Suzuki 2011,
  Bickerton 2012, Cohen 1988): CITEDONLY

### 3.4 §5 ablation — metal coordination probe (NEW §5.10 candidate)

**CANDIDATE for §5.10**, but the probe is NOT YET WIRED into
`r4_lambda_only_run.py`. The §5.10 cell text in Phase 2 must
therefore be:

- Module + tests (probe_coordination, probe_batch,
  compliance_rate; 13/13 unit tests): MEASURED (shipped, tested)
- Per-pocket compliance_rate: DESIGN (no batch run yet)
- 15-metal coverage table (Pt/Pd/Au/Ag/Ru/Ir/Rh/Os/Re/Fe/Co/Ni/
  Cu/Zn/Mn with CN/geometries): CITEDONLY (Lippard & Berg 1995,
  Reedijk 1987, Miessler 2014, Shriver & Atkins 2010)
- Honest limitation: CN is graph-theoretic, not 3-D; spatial
  validation lives in `MetalGeometryPrior`

### 3.5 §5 ablation — drug-likeness / ADMET metrics_v2 (NEW §5.11 candidate)

**CANDIDATE for §5.11**, but the 8 metrics are NOT YET WIRED
into `r4_lambda_only_run.py`. The §5.11 cell text in Phase 2
must therefore be:

- Module + tests (logp7_4, gi50_proxy, cell_permeability_logPapp,
  herg_cardio_risk, ames_mutagen, hepatotox_index,
  aqueous_solubility_logS, plasma_protein_binding; 20/20 unit
  tests): MEASURED (shipped, tested)
- Per-pocket batch means: DESIGN (no batch run yet)
- 6-SMILES reference smoke (CCO / benzene / pPDA / aspirin /
  caffeine / cisplatin): MEASURED (the values in artefact 3
  §4 are reproducible from `all_metrics_one(smi)`)
- Honest limitation: heuristic evaluators, NOT wet-lab
  calibrated; coefficients taken at face value from lit without
  regression against any held-out assay (suitable for ranking,
  not absolute predictivity)

---

## 4. Honest framing — search-bound vs real measure

The single most important framing distinction for this
inventory: every numeric cell in §2 is one of three kinds:

### 4.1 REAL measure (post-generation, post-dock, post-PB)

Cells where the value is a real generated-candidate measurement
on this box, exercised end-to-end at least once:

- 2.1.1 single-pocket PB smoke (`pb_pass_rate=1.000` on `n=1`,
  chemistry-only `mol` mode)
- 2.1.2 30-cell PB panel (`pb_pass_rate=None` for all 30 cells,
  `wall_seconds=4.81 s/cell`)
- 2.2 P0 anticancer (single cell, 9 metrics on 15 candidates)
- 2.3 metrics_v2 6-SMILES reference smoke (5 organic + cisplatin)

### 4.2 DESIGN / PROJECTED (architecturally complete, not exercised)

Cells where the module/code/test path is SHIPPED and TESTED, but
NO production batch run was executed:

- 2.3 8 metrics_v2 batch-means on a real candidate pool
  (no integration into r4_lambda_only_run.py)
- 2.4 click-rule effect-size 5×4 panel (script + tests ready, no
  production run)
- 2.5 metal-coord probe `compliance_rate` on a real candidate
  pool (probe ready, no integration)

### 4.3 SEARCH-BOUND null (search accepts zero candidates)

The 30-cell PB panel is the cleanest example: 30/30 cells return
zero generated molecules because the MCTS budget at
`n_simulations=100` with the strict gate combination
(`synthesis_oracle=smarts ∧ symbolic_prior=True`) accepts
nothing. PoseBusters has no candidates to evaluate; `pb_pass_rate`
is structurally undefined. This is **NOT** a PB failure and
**NOT** a metric defect; it is a search-budget artefact. The
`n_simulations=1000` re-run via `r4_lambda_only_run.py` is the
documented recovery path (WF-Lift-N-Sim-Cap, tasks #539–541).

### 4.4 CITEDONLY (theory / literature anchor)

- The 15-metal CN/OS table (artefact 5): CITEDONLY (Lippard &
  Berg 1995, Reedijk 1987, Miessler 2014, Shriver & Atkins 2010).
- The 8 metrics_v2 formulas: CITEDONLY (Hou 2007, Veith 2009,
  Delaney 2004 ESOL, Obach 1999, Hughes 2008, Benigni-Richard
  2005, Patrick 2009, Mente 2015).
- The click-rule 5×4 Cohen's-d magnitudes: CITEDONLY as
  PROJECTED (the script supports computing them, but the run
  hasn't been done).

---

## 5. Phase 2 integration map

This inventory feeds Phase 2 (the actual §4.6 / §5 LaTeX edit).
The integration target is:

- `paper/sections/04_evaluation.tex` §4.6 ONLY (lines 1225–1666
  inclusive). Do NOT touch §4.1–§4.5.
- `paper/sections/05_ablation.tex` (whole file).
- Do NOT touch `paper/main.tex` or `paper/refs.bib` (Workflow 1
  owns those).

### 5.1 §4.6 edits

The existing §4.6 PB pass-rate paragraph (lines 1391–1479, the
1-pocket WF-PB-Pass-Real-Dock block) is correctly framed and
already in the paper; no edit required.

The existing §4.6 30-cell PB statistic panel (lines 1481–1666,
WF-PB-Pass-10x3-Smoke block) is correctly framed; no edit
required.

**No additional §4.6 cells are promoted from DESIGN to MEASURED
in this phase.** The 30 × None panel remains honest null; the
1-pocket 1.000 result remains path-correctness smoke with 3
explicit caveats.

### 5.2 §5 edits

The existing §5.8 P0 anticancer panel is already integrated and
correctly framed; no edit required.

Phase 2 will add **3 NEW sub-sections** to §5 with the
following honest tagging:

#### §5.9 Click-rule effect sizes (from artefact 4)

- Module + tests (6/6 PASS): MEASURED
- 5×4 panel values: PROJECTED (Cohen's-d applied to rule-vs-others
  baseline; replaced by MEASURED only after a 5×3 production run)
- Lit anchors: CITEDONLY (Himo 2005, Worrell 1984, Kolb 2001,
  Suzuki 2011, Bickerton 2012, Cohen 1988)
- Honest caveat: NO production run was executed in this artefact;
  the §8 numbers are guidance from prior Lambda mini-pilots, not
  measured from this script.

#### §5.10 Metal coordination probe (from artefact 5)

- Module + tests (13/13 PASS): MEASURED
- Per-pocket `compliance_rate`: DESIGN (no batch run)
- 15-metal CN/OS table: CITEDONLY (Lippard & Berg 1995, Reedijk
  1987, Miessler 2014, Shriver & Atkins 2010)
- 4 cisplatin / Pt_IV / Ru_III / Au_I reference probes:
  MEASURED (artefact 5 §3)
- Honest caveat: probe is graph-theoretic (CN-only, no 3-D
  geometry); dot-separated SMILES flagged NON-COMPLIANT (RDKit
  limitation, captured honestly); OS inference is heuristic;
  spatial validation lives in `MetalGeometryPrior`.

#### §5.11 Drug-likeness / ADMET metrics_v2 (from artefact 3)

- Module + tests (20/20 PASS): MEASURED
- Per-pocket batch means on real candidates: DESIGN (no
  integration)
- 6-SMILES reference smoke (CCO / benzene / pPDA / aspirin /
  caffeine / cisplatin): MEASURED (artefact 3 §4)
- 8 metric formulas: CITEDONLY (Hou 2007, Veith 2009, Delaney
  2004, Obach 1999, Hughes 2008, Benigni-Richard 2005, Patrick
  2009, Mente 2015)
- Honest caveat: heuristic evaluators, NOT wet-lab calibrated;
  coefficients taken at face value from lit without regression
  against any held-out assay (suitable for ranking, not absolute
  predictivity).

### 5.3 Cross-reference audit

Phase 2 must add `§5.9`, `§5.10`, `§5.11` to:
- `paper/CROSS_REFS.md` (forward-link from §3.2 click-chem to §5.9
  + §3.3 MetalGeometryPrior to §5.10 + §3.1 formalism to §5.11)
- `TODO/pending/15_anticancer_metric_suite_r11b.md` (cross-link
  §5.8 ↔ §5.11)
- `paper/main.tex` table-of-contents if §5 sub-sections are
  auto-listing (verify by reading main.tex before edit)

Do NOT touch `paper/main.tex` unless absolutely required for a
forward-link; if a `\nameref{sec:ablation:metrics-v2}` is needed,
add it via the section's own `\label{}` and let `main.tex`
recompile naturally.

---

## 6. Summary — what ships in Phase 2, what doesn't

| Item | Phase 2 status | Honest tag |
|---|---|---|
| §4.6 PB 1-pocket block | UNCHANGED | already correctly framed (lines 1391–1479) |
| §4.6 PB 30-cell block | UNCHANGED | already correctly framed (lines 1481–1666) |
| §4.6 any new cell | NOT ADDED | no new MEASURED PB value at scale |
| §5.8 P0 anticancer panel | UNCHANGED | already correctly framed (WF-Section05-P0-Metrics) |
| §5.9 click-rule effect sizes | NEW (text only) | module MEASURED, panel PROJECTED |
| §5.10 metal-coord probe | NEW (text only) | module MEASURED, batch DESIGN |
| §5.11 metrics_v2 drug-likeness | NEW (text only) | module MEASURED, batch DESIGN |
| §5 any aggregate promotion | NONE | honest posture preserved |

The integration in Phase 2 is text-only and honest-only; no cell
is silently promoted from DESIGN to MEASURED, no SOTA
comparison is fabricated, no protein-aware PB claim is made
without the dock-mode adapter extension, no wet-lab pIC50
calibration is implied from the heuristic metrics_v2 formulas.

---

## 7. Source artefacts

Absolute paths (workflow brief says "always absolute"):

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_10x3_smoke/final.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/summary.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_p0_metrics_smoke/report.json`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3f_click_effect.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_parallel_tasks/phase3g_metal_coord.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pb_pass_real_dock/final.md`
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex` (read-only; §4.6 lines 1391–1666)
- `/home/hugo/codes/try_triton_on_rocm/paper/sections/05_ablation.tex` (read-only)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/metrics/by_metric/pb_pass_rate.json` (**MISSING** — file does not exist)

---

End of Phase 1 data inventory. Phase 2 (LaTeX edits) starts after
this inventory is verified by the workflow owner.
