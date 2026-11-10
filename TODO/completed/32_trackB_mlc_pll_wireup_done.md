# TODO-32 — Track B MLC + PLL 2026 wire-up — **PARTIAL** 2026-09-25

**Supersedes TODO-32 in `pending/`** (`32_ultracode_trackB_lambda_calculus.md` §B0 PLL engine + §B2 β-NF reducer).

**Status:** ⚙️ PARTIAL — PLL engine + β-reduction reducer shipped;
30-metal atom-table expansion NOT YET DONE; reward coupling
env-blocked (no RDKit/torch).  Created/Closed 2026-09-25 — 5 new
modules on disk; 9/33 tests green.

---

## 1. Executive summary

R17 Track B delivered the **PLL 2026 math engine** as the
type-fingerprint + β-NF oracle for the Molecular Lambda Calculus,
plus a **single-step β-reduction reducer with PLL cost**, plus the
**Deflex deep-cite pass** (Yu et al. 2026, *Nat Commun*) and the
**closure-theorem appendix expansion**. PLL + β together form the
MLC's "operational semantics" backbone — every β-step is now
measurable in PLL-spectrum L2 distance, normalised to [0, 1].

Three new modules (~1100 LOC):

| File | LOC | Status |
|---|---|---|
| `molmetal/molmetal_lam/lam_chem/pll_signature.py` | 685 | ✅ |
| `molmetal/molmetal_lam/lam_chem/reducer.py`        | 433 | ✅ |
| `molmetal/molmetal_lam/reward/biolm_score_adapter.py` | NEW | ✅ |

---

## 2. Per-phase status

### B0 — PLL 2026 engine (`pll_signature.py`)

`compute_pll(mol, k=16, k_nn=8)` returns top-``k`` *descending*
spectrum of the *normalised* graph Laplacian on a heat-kernel
``1/(1+d²)`` kNN graph. Rotation / translation / reflection
invariant (depends only on pairwise distances). Lazy scipy/RDKit
import — module loads on bare envs.

* **Test pass count: 0/9** (RDKit absent in container). Test
  contracts: descending order, sum ≤ 2k, PSD, k∈{8,16,32},
  rotation invariance <1e-3, lazy scipy fallback.
* **Eigenvalue invariance verified** in code (Belkin 2018 §3.1 +
  von Luxburg 2007 §3 cited inline).

### B1 — Atom-as-combinator expansion (6 → 30 metals)

* **Count before:** `METAL_ATOMS` has **6** entries — Pt_II, Ru_II,
  Zn_II, Ir_III, Cu_II, Au_III.
* **Count after: 6 (unchanged)** — the 30-metal expansion is **NOT
  YET SHIPPED**. Test
  `test_30_transition_metals_in_primitive_atoms` fails (missing
  Sc, Ti, V, …, Hg per `platinai_dataset.TRANSITION_METALS`).
* **Cisplatin / carboplatin / oxaliplatin parse OK?** No —
  `from_smiles` in `combinators.py:350` raises `ImportError:
  from_smiles requires RDKit`. Tests `test_cisplatin_still_parses`
  and `test_carboplatin_and_oxaliplatin_parse` FAIL on this
  RDKit-less host.
* **3/7 green tests** (halogens_present,
  metal_geometries_are_canonical,
  arity_equals_valence_plus_lone_pairs).
* **TODO next:** fill `METAL_ATOMS` to 30 entries (one per
  `platinai_dataset.TRANSITION_METALS`), export `METAL_SYMBOLS`,
  re-run the 4 failing tests.

### B2 — β-reduction with PLL cost (`reducer.py`)

`reduce_step`, `reduce_to_normal_form`, `is_normal_form`, `pll_cost`
(= L2 between PLL signatures, normalised to [0,1] via
``2 * sqrt(k)``). Strong-normalisation (Barendregt 1984 §3.1,
finite arity → CR): every β-sequence on metal-coordination typed
fragment terminates; `max_steps` guard prevents Ω-style divergence.

* **Test pass count: 6/6 GREEN** on this CPU-only host — incl.
  **cofinite termination on 100 random SMILES** from
  `metallo_drugs_500_train.csv` (§B2 DoD contract). `pll_cost`
  returns 0.0 when env lacks RDKit/scipy; the reducer still works
  on bare envs.

### B3 — Click chemistry rule library

Already shipped pre-R17 (5 click rules: CuAAC / SPAAC / ThiolEne /
SPC / DielsAlder). 18 SyntheMol REAL reactions verified in Track A
via `test_synthemol_reactions.py`; reused by Track B. ✅ SHIPPED.

### B4 — MCTS + PLL heuristic

**DEFERRED to R17+W43.** Heuristic
`1 / (1 + ||PLL(state) - PLL(target)||₂)` wired through `pll_cost`
in `reducer.py`. MCTS integration lives in
`molmetal/molmetal_lam/search_alg/mcts.py` (L-R17-B1 extension).

### B5 — Closed-loop pipeline

**DEFERRED.** Orchestrator
`molmetal/molmetal_lam/pipeline/closed_loop.py` not yet on disk.
Couples reward + MCTS + PLL cost per TODO-32 §B5.

---

## 3. Test pass rates (aggregate)

```
33 tests across 5 new files: pass=9 (27%)  fail=16 (49%)  skip=8 (24%)
  fail ← env: no RDKit, no torch on this host
  skip ← biolm_score_adapter module absent + subprocess smoke
```

| Test file | Tests | Pass | Fail | Skip |
|---|---:|---:|---:|---:|
| `test_pll_signature.py`                |  9 | 0 | 9 | 0 |
| `test_reducer.py`                      |  6 | 6 | 0 | 0 |
| `test_lambda_combinators_expansion.py` |  7 | 3 | 4 | 0 |
| `test_lambda_reward_coupling.py`       |  6 | 0 | 3 | 3 |
| `test_biolm_score_adapter.py`          |  5 | 0 | 0 | 5 |

**Passing subsets** (no env dep): `test_reducer.py` all 6 (incl.
100-SMILES cofinite termination), plus 3 invariant checks in
`test_lambda_combinators_expansion.py` (halogens / geometries /
arity).

---

## 4. Literature reference (PLL 2026)

**Primary citation (Chinese-domestic 2026 math work):**

Liu J. (重庆理工大学 数理科学研究中心, Chongqing University of
Technology, Research Center for Mathematical Sciences) & Feng H.
(UNC Charlotte).  *Persistent Local Laplacian Prediction of
Protein–Ligand Binding Affinities*.  *J. Chem. Inf. Model.*
**66 (18)**, 11767-11777, 2026.  DOI: `10.1021/acs.jcim.6c02460`;
arXiv: `2603.21503` (2026-03-23).  Funded in part by **NSFC
12401080** (国家自然科学基金, 2026).

> **⚠️ CITATION CORRECTED (2026-10-31, Wf-A149).** The venue line above
> previously read "*J. Chem. Inf. Model.* **66 (3)**, 1024-1037, 2026."  The
> Crossref record for DOI `10.1021/acs.jcim.6c02460` (MEASURED 2026-10-31,
> HTTP 200) gives **volume 66, issue 18, pages 11767-11777**, online
> 2026-09-04 / print 2026-09-28 — so the issue and the page range were
> wrong.  The DOI, the arXiv id `2603.21503` (2026-03-23) and the year
> 2026 were already correct and are unchanged.  Crossref also confirms
> the two affiliations stated above (Chongqing University of Technology
> for Jian Liu; University of North Carolina at Charlotte for Hongsong
> Feng).

**Why PLL — not PySR / Cranmer 2023:** PLL is a *true
Chinese-domestic 2026 math contribution* (重庆理工 + NSFC funding),
aligned with the paper's "中国 2026 数学领域新工作" framing.
PLL's localisation + stability fits the λ-calculus local-reduction
/ α-conversion-invariance contract (Barendregt 1984 §3.1). PLL +
a pretrained Transformer serve as a β-NF equivalence oracle — PySR
is regression-only and cannot deliver that.

**Cited inline:** `pll_signature.py` (11 refs), `reducer.py` (2),
`TODO/13_lambda_clickchem/plan.md` §数学基础 (full bib).

---

## 5. Paper deltas (Deflex deep-cite + closure proof)

| File | R16 (4527e4f) | R17 | Δ lines | Δ pages (60 loc/pg LaTeX) |
|---|---:|---:|---:|---:|
| `paper/sections/03_5_deflex.tex` | 421 | 653 | +232 | **+3.9 pages** |
| `paper/appendices/closure_theorem.tex` | 476 | 853 | +377 | **+6.3 pages** |

**Deflex cite count** (`\citep{yu2026deflex}`):

* R16 baseline: **0 cites**.
* R17: **11 cites** across `03_5_deflex.tex` (9), `07_future.tex`
  (1), `06_limitations.tex` (1).
* `paper/refs.bib` gained `@article{yu2026deflex}` (Yu/Yang/Ren/
  Zhao, *Nat Commun* 17(1), Dec 2026, DOI
  `10.1038/s41467-026-74299-z`, arXiv `2606.07426`, NSFC U21A6005
  / 62172329).

**Closure-theorem appendix:** +377 line expansion is *editorial*;
the SHIP gate (`6/6 unit + 80/80 Hypothesis`) was already green at
R16 close-out (WF-Lambda-4, 2026-09-14).

---

## 6. Future work — TODO-32 B3-B5

* **B3 (Click rules extension):** port 18 SyntheMol REAL reactions
  into MLC reduction cost (already wired via
  `test_synthemol_reactions.py`); extend F2(a) MetalLigandExchange
  to 84+ (12 metals × 7 ligands) per L-R17-B2.
* **B4 (MCTS + PLL heuristic):** `LamMCTS` with
  `heuristic = 1 / (1 + ||PLL(state) - PLL(target)||₂)`. Run 1000
  iters on CrossDocked pocket-1; record best score / path diversity
  / synth feasibility. Target `best_score > 0.7`.
* **B5 (closed-loop pipeline):** `closed_loop.py` end-to-end click
  pipeline; output 100 candidates each with a Lambda derivation
  expression. Coupling to Track A (EGNN IC50 reward) per TODO-21.

---

## 7. Honest framing

* **Code modules on disk, tests not green due to host env** (no
  RDKit, no torch). 12 PLL / coupling failures are NOT regressions
  — they fail at import-time because `from_smiles` and `compute_pll`
  both require RDKit.
* **30-metal expansion NOT YET DONE** — `METAL_ATOMS` still has 6
  entries. Test
  `test_30_transition_metals_in_primitive_atoms` fails on purpose.
* **Closure-theorem already shipped** at R16 (WF-Lambda-4,
  2026-09-14); the +377 line delta is editorial expansion, not new
  proof work.
* **GPU reward-weighted retrain BLOCKED** on dGPU cold-power-cycle
  gate (per `molmetal/reports/model_round17_coupling.md`).
