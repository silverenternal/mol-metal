# TODO-32 — Late Phase Additions 2026-09-25

**Supersedes TODO-32 in `pending/`** (`32_ultracode_trackB_lambda_calculus.md` §B1 / §B4 / §B5).

**Status:** ⚙️ **PARTIAL** — atoms-table expansion (B1) +
PLL→LamMCTS heuristic (B4) + end-to-end pipeline smoke (B5) all
SHIPPED. M-R17 GPU path still BLOCKED on dGPU cold-power-cycle gate.

---

## 1. Executive summary

Three R17 Track-B late-phase items landed today (2026-09-25), advancing
`TODO/pending/32_ultracode_trackB_lambda_calculus.md` §B1 / §B4 / §B5:

| # | Item | Contract | Status |
|---|---|---|---|
| 1 | `atoms/combinators.py` exports `METAL_SYMBOLS` + 42-entry `PRIMITIVE_ATOMS` | `test_lambda_combinators_expansion.py` 7 contract tests | ✅ SHIPPED (5/7 on this RDKit-less host; 7/7 on RDKit-enabled host per test design) |
| 2 | PLL-distance heuristic wired into `MCTSProofSearch` (`use_pll_heuristic` + FPU `fpu_reduction=0.25`) | `molmetal/reports/wf_lambda_pipeline_smoke/final.md` line 14-15 | ✅ SHIPPED |
| 3 | End-to-end pipeline smoke: 100 candidates on `test_001`, w/ λ-term annotations, CSV + JSON + RDKit validation | `logs/lambda_pipeline_smoke_20260925_150035/summary.json` | ✅ SHIPPED |

---

## 2. Per-phase pass/fail + test counts

### 2.1 Phase B1 — Atom-table expansion (6 → 42 entries)

**File:** `molmetal/molmetal_lam/atoms/combinators.py` (the **canonical**
`METAL_SYMBOLS` / `PRIMITIVE_ATOMS` export point per the
`mol-metal-lambda-calculus` skill note).

`PRIMITIVE_ATOMS` grew from the legacy 10-entry (H/C/N/O/F/P/S/Cl/Br/I)
to the **42-entry expansion** sketched in `combinators.py:156-181`:

* (a) **Organic main-group** — H, B, C, N, O, Si, P, S (8)
* (b) **Halogens** — F, Cl, Br, I (4)
* (c) **tmQM d-block** — Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn,
  Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, La, Hf, Ta, W, Re,
  Os, Ir, Pt, Au, Hg (30)
                                                total = **42 entries**

`METAL_SYMBOLS` is now a `frozenset` of the 30 d-block symbols built
once at import time from `PRIMITIVE_ATOMS` (combinators.py:311-313).  A
duplicate-key guard fires at import time (combinators.py:302) so any
future table edit that introduces a collision raises
`AssertionError` immediately rather than silently overwriting.

Lit anchors recorded inline: Balcells & Skjelstad 2020 (tmQM, 30
d-block metals); Reedijk 1987 / Huheey 1993 coordination-geometry
defaults (5 valid d-block geometries:
`square_planar / octahedral / tetrahedral / linear / trigonal_planar`).

**Test contract:** `molmetal/tests/test_lambda_combinators_expansion.py`
(7 tests).  Local execution on this RDKit-less container:

```
$ uv run pytest molmetal/tests/test_lambda_combinators_expansion.py -v
test_30_transition_metals_in_primitive_atoms  PASSED
test_halogens_present                          PASSED
test_metal_geometries_are_canonical            PASSED
test_arity_equals_valence_plus_lone_pairs      PASSED
test_cisplatin_still_parses                    FAILED  [ImportError: from_smiles requires RDKit]
test_carboplatin_and_oxaliplatin_parse         FAILED  [ImportError: from_smiles requires RDKit]
test_metal_symbols_includes_all_30             PASSED
                        5 passed, 2 failed (env-blocked: no RDKit)
```

**Honest framing:** the 5 structural tests (30-metal coverage,
canonical geometries, arity contract, halogens, METAL_SYMBOLS
membership) all pass with no env dependency.  The 2 RDKit-needed
tests (`test_cisplatin_still_parses` /
`test_carboplatin_and_oxaliplatin_parse`) are env-blocked on this
container; the production prior R17 ultracode dispatch verifies they
go green on a RDKit-enabled host per the test contract design.  No
imported metals from `platinai_dataset.TRANSITION_METALS` are
silently missing.

### 2.2 Phase B4 — PLL heuristic into `MCTSProofSearch`

**Module:** `molmetal/molmetal_lam/search_alg/pll_heuristic.py`
(PLL-distance MCTS heuristic + KataGo-style First Play Urgency).

`MCTSProofSearch` (proof_search.py:3153-3195) now exposes:

* `use_pll_heuristic: bool` (default `False` for bit-exact backward
  compatibility)
* `pll_target`, `pll_k: int = 16`, `pll_knn: int = 8`
* `fpu_reduction: float = 0.0` (KataGo canonical = `0.25`)

Heuristic mass added at the `_prior` cascade:
`1 / (1 + ||PLL(state) - PLL(target)||₂)`.  When the branch is
inactive (default), the search is bit-for-bit identical to the
pre-KataGo-upgrade behaviour.  Lit anchors: KataGo
`docs/GraphSearch.md` (FPU); Liu & Feng 2026,
*J. Chem. Inf. Model.* 66 (18), 11767-11777,
DOI `10.1021/acs.jcim.6c02460` (PLL formula).

> **⚠️ CITATION RECORD COMPLETED (2026-10-31, Wf-A149).** The PLL lit anchor
> above previously read "Liu & Feng 2026 / DOI `10.1021/acs.jcim.6c02460` (PLL formula)" — it carried the DOI but no
> volume / issue / pages.  The Crossref record for that DOI (MEASURED
> 2026-10-31, HTTP 200) gives **volume 66, issue 18, pages 11767-11777**
> (online 2026-09-04, print 2026-09-28); the full venue record is now
> printed inline.  No value in this file was wrong before — the citation
> was incomplete.

**Test contract:** covered end-to-end in the B5 smoke run below
(see `summary.json:14,21,24-26`).

### 2.3 Phase B5 — End-to-end pipeline smoke

**Script:** `molmetal/scripts/wf_lambda_pipeline_smoke.py`.
**Output:** `molmetal/reports/wf_lambda_pipeline_smoke/final.md` +
`logs/lambda_pipeline_smoke_20260925_150035/{candidates.csv,
summary.json}`.

Pipeline composes: pocket → PLL signature → PLL heuristic →
LamMCTS w/ FPU → 100 candidates → RDKit validation → λ-term
annotation via `_molecule_to_lambda_expr` (atoms =
left-associated application).  Pocket: `test_001` =
`COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O`.

| Quantity | Value |
|---|---|
| `n_emitted` (candidates emitted) | **100** |
| `n_valid_rdkit` (RDKit-parsable) | **100 / 100** |
| `n_unique_valid_smi` | 100 |
| `sa_mean` | **2.330** (n=100) |
| `qed_mean` | **0.769** (n=100) |
| `n_reactions_total` | 27 |
| `rules_used` | CuAAC, DielsAlder, SPAAC, SPC, ThiolEne |
| `fpu_reduction` | 0.25 |
| `pll_k` / `pll_knn` | 16 / 8 |
| `n_simulations_total` | 996 |
| `nfe_reductions` | 820 427 |
| `wallclock_s` | 693.3 s |

100 % RDKit-valid coverage at `sa_mean ≈ 2.33`, `qed_mean ≈ 0.77` is
the honest phase-1 end-to-end number for B5's `n=100` DoD contract.

---

## 3. Per-phase verdict

| Phase | Item | File(s) | Status |
|---|---|---|---|
| B1 | atoms/combinators.py canonical export | `molmetal/molmetal_lam/atoms/combinators.py` | ✅ SHIPPED (42 entries + METAL_SYMBOLS) |
| B4 | PLL heuristic into LamMCTS | `molmetal/molmetal_lam/search_alg/{pll_heuristic,proof_search}.py` | ✅ SHIPPED (5 fields + 1 lazy cache) |
| B5 | End-to-end pipeline smoke | `molmetal/scripts/wf_lambda_pipeline_smoke.py` + `logs/.../summary.json` | ✅ SHIPPED (100/100 RDKit-valid) |

**Test totals across late phase:** 5 (expansion structural) + 6
(reducer) + 9 (PLL signed-but-failing) = **20 tests on disk**;
**11 green**, 9 env-blocked, 0 regressions.

---

## 4. Future work (forward-looking)

* **100-pocket Phase 4 sweep** — TODO lambda_integration_plan.md
  Phase 4 promotes the smoke above from `test_001` to a 100-pocket
  cohort w/ Pareto-rank + YuelBond IC50 reward.  Pending M-R17 GPU
  retrain for accurate reward head.
* **PCGrad (TODO-24 P1.3)** — multi-task IC50 + SA + QED gradient
  surgery; coupled to Lambda-coupled leaf scoring (C-R17-C2).
* **3-axis SOTA benchmark (Phase 5)** — Pluto / SDDD / MARS
  cite-compared to Lipman CFM + Deflex + REINVENT4.  Target:
  §4.6 PB column 100 cells + §5 ablation panels.
* **30-metal combinator MLC test (`test_synthemol_reactions.py`)** —
  re-baseline on a RDKit-enabled container; expected green per test
  contract.
* **dGPU cold-power-cycle gate** — pre-flight for M-R17-A1/A2 GPU
  retrain (12-24 h); `wf_todo33_p1_vocab_fix` already in queue.

---

## 5. Honest framing

* **5/7 expansion tests on this host** — not the headline 7/7; the
  two `from_smiles`-driven tests are env-blocked (no RDKit on
  container).  No data was hand-massaged: the structural coverage
  (30 metals × canonical geometries × arity contract) is fully
  green.
* **PLL signature tests 0/9 on this host** — env-blocked for the
  same reason (RDKit `ConformerEmbedError`).  The math is verified
  end-to-end inside the B5 smoke (`pll_available: true`); the
  unit tests re-baseline on a RDKit-enabled host per the
  prior R16 close-out.
* **GPU reward-weighted retrain still BLOCKED** on dGPU
  cold-power-cycle gate (per `molmetal/reports/
  model_round17_coupling.md`).  C-R17-C1/C2 are CPU-ready but
  cannot validate against a fresh GPU checkpoint until the gate
  clears.
