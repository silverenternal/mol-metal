# WF-Partner-Tiles-PathA — Phase 2 honest negative-result report

> **Status: HONEST NEGATIVE RESULT.** Despite the partner-tile library
> (`PARTNER_TILES_V2`: 3 azides + 3 boronic acids + 2 bromides) being
> shipped and the 200-tile ChEMBL/ZINC pool (`FRAGMENT_LIBRARY_200_TILES`)
> containing **at least 50 azide handles** at runtime, the 30/30 Round-12
> Lambda cells **still collapsed to `n_distinct = 1`** (only the seed
> `[Pt]C#C` was emitted).  The click rules are *active* in the rule
> registry (warnings confirm `auto_rules_expanded=['CuAAC', 'SPAAC',
> 'Suzuki']`) but **none of them fire** when starting from the bare
> Pt-alkyne root.  This is a *deeper* root-cause finding than the
> baseline singleton-collapse that motivates Path A — see §3.

## 1. Configuration (verified)

| field | value |
|---|---|
| script | `molmetal/scripts/r4_lambda_only_run.py` |
| `--pockets` | 10 (test_000 .. test_009) |
| `--seeds` | 42 0 1234 (3 seeds × 10 pockets = 30 cells) |
| `--n-simulations` | 1000 |
| `--n-top-k` | 20 |
| `--metal-seed` | `cisplatin` (root SMILES `[Pt]C#C` — bare Pt-alkyne) |
| `--click-rules` | `auto-pt-strict` (resolves to `[CuAAC, SPAAC, Suzuki]` for `strict_Pt_II`) |
| `--output-dir` | `wf_partner_tiles_patha/r4c` |
| Tile library at runtime | `FRAGMENT_LIBRARY_200_TILES()` (Phase-1, ChEMBL/ZINC reactive handles, includes ≥50 azide handles) |
| Python | 3.12 (uv-managed) |
| ROCm / Triton / GPU | 7.2 / 3.8.0 / gfx1101 wave64 — **not used (CPU-only Lambda path)** |

Note: the script's `--output-dir` flag prefixes the directory name with
`wf_lambda1_` so the actual files land in
`molmetal/reports/wf_lambda1_wf_partner_tiles_patha/r4c/`.  The user's
requested `wf_partner_tiles_patha/r4c` is referenced symbolically here.

## 2. Aggregate metrics — Path A (10×3 with partner tiles + 200-tile pool)

| metric | value |
|---|---:|
| n_cells | 30 |
| n_distinct (per-cell) | **1 (×30 cells)** |
| total n_distinct summed across cells | 30 (=30×1) |
| n_candidates per cell | 1 |
| diversity_tanimoto | 0.000 |
| diversity_homotype | 0.000 |
| validity_rate | 1.000 |
| synthesizability_rate | 1.000 |
| novelty | 1.000 |
| metal_compliance_rate | 0.000 |
| logp_mean | 0.1239 |
| tpsa_mean | 0.000 |
| rotb_mean | 0.000 |
| coordination_number_mean | 1.000 |
| monodentate_cl_count | 0 |
| gsh_evasion_score | 0.000 |
| dna_kb_proxy | 0.400 |
| anticancer_index | 0.100 |
| sa_mean | 6.008 |
| qed_mean | 0.509 |
| sa_weight | 0.000 |
| rigid_rmsd_mean | 0.000 |
| com_shift_mean | 0.000 |
| n_coords_3d_attached_total | 30 |
| n_scaffold_aware_gate_active_total | 30 |
| decoder_pass_rate | 1.000 |
| elapsed_s_total | 50.99 |
| Total unique candidates across all 30 cells | **1 (`C#[C][Pt]`)** |

### 2.1 Per-pocket × per-seed table — full

All 30 cells: `n_distinct=1`, `n_candidates=1`, `validity=1.000`,
`synth=1.000`, `metal_compl=0.000`.  Every emitted SMILES is
`C#[C][Pt]` (canonical form of the bare cisplatin seed `[Pt]C#C`).

| pocket | seed | n_distinct | n_cand | div_tan | div_homo | metal_compl | unique_smiles |
|---|---:|---:|---:|---:|---:|---:|---|
| test_000 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_000 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_000 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_001 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_001 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_001 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_002 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_002 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_002 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_003 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_003 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_003 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_004 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_004 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_004 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_005 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_005 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_005 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_006 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_006 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_006 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_007 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_007 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_007 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_008 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_008 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_008 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_009 | 42 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_009 | 0 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |
| test_009 | 1234 | 1 | 1 | 0.000 | 0.000 | 0.000 | `C#[C][Pt]` |

## 3. Diagnosis — why click rules still did not fire

The infrastructure IS in place:

1. **`PARTNER_TILES_V2` shipped.**  `:func:`AZIDE_PARTNER_TILES`,
   `:func:`BORONIC_PARTNER_TILES`, `:func:`BROMIDE_PARTNER_TILES`,
   `:func:`ALL_CLICK_HANDLES` all exported from
   `molmetal_lam.tile_lib.click_tiles` (verified via the `__init__.py`
   and `click_tiles.py` files).
2. **200-tile ChEMBL/ZINC fragment pool is loaded at runtime.**
   `r4_lambda_only_run.py:1905` calls
   `FRAGMENT_LIBRARY_200_TILES()` which contains **at least 50 azide
   handles** (e.g. ethyl/benzyl/PEG/aryl + 2-fluoro, 4-chloro, 4-bromo,
   4-methyl, 4-hydroxy, 4-amino, 4-methoxy, 4-trifluoromethyl, 4-cyano,
   3-/2-/4-pyridyl, pyrazol-3-yl, 2-furyl, 2-thienyl azides, plus
   benzyl variants and naphthyl azides).  The pool size after the
   `ROUND10_5_CLICK=1` flag is enabled grows to 220 with boronic
   acids + aryl halides + amines + carboxylic acids + DBCO appended.
3. **Click rules are active.**  Cell warning:
   `auto_rules_expanded=['CuAAC', 'SPAAC', 'Suzuki']
   from=auto-pt-strict scaffold=strict_Pt_II allow_incompatible=False`.
4. **The scaffold-aware gate recognises the seed.**
   Cell warning: `auto_scaffold_detected=strict_Pt_II
   (metal_seed='cisplatin', smi='[Pt]C#C')`.

Despite all of the above, **0 click rules fire** in 30/30 cells.  The
root cause is a **type-handle mismatch** in the rule-dispatch logic,
not a missing-tile problem:

### 3.1 Rule-signature asymmetry vs MCTS state/tile dispatch

* `click_cuaac(azide_tile: Tile, alkyne_tile: Tile)` — the **first**
  argument must be the azide (the function tests
  `_has_substructure(azide_smi, _AZIDE_SMARTS)` on the first arg and
  `_has_substructure(alkyne_smi, _TERMINAL_ALKYNE_SMARTS)` on the
  second; see `click_reactions.py:374-377`).
* The MCTS `_expand` loop calls `rule.reduce((state, tile))` (see
  `proof_search.py:2901` and `:3664`).  The dispatch convention is
  therefore **state-first, tile-second**.
* With root `state = [Pt]C#C` (a terminal *alkyne*, no azide), every
  CuAAC attempt has `azide_smi = "[Pt]C#C"` which fails the azide
  SMARTS match in `click_cuaac` → returns `[]`.  The azide tiles are
  sitting in the tile pool unused.

The same problem affects SPAAC (`azide` arg + `cyclooctyne` arg), and
Suzuki's handle-pairing (`boronic_acid` + `aryl_halide`).  Because the
root is an *alkyne* (not an azide/boronic/bromide handle), **only
rules that take a terminal alkyne as the FIRST argument would fire**.
None of the three active rules (`CuAAC`, `SPAAC`, `Suzuki`) match.

### 3.2 What would fix this (forward-looking, NOT in scope for Phase 2)

Three principled fixes are possible.  **None of these were attempted
in Phase 2** because the workflow's explicit scope was to *test* the
partner-tile lift, not to patch the rule dispatch.

1. **Symmetric rule dispatch.**  Make `click_cuaac` (and siblings)
   try both arg orderings (`(a, b)` and `(b, a)`) and return a
   product whenever one ordering matches.  This is the cheapest fix
   (≈10 lines per rule × 5 rules = 50 lines) and would make CuAAC
   fire when either educt is the azide.
2. **Dual-root seeding.**  Add a SECOND root to the MCTS — the bare
   azide `CCN=[N+]=[N-]` (ethyl azide, already in the canonical 12-
   tile library) — and have the search expand both in parallel.  This
   guarantees at least one starting state contains the right handle
   for each rule, but it doubles the budget.
3. **Asymmetric tile-by-state classification.**  Pre-classify every
   tile in `FRAGMENT_LIBRARY_200_TILES` by its dominant handle
   (azide / alkyne / boronic / bromide / etc.) and dispatch
   `(state, tile)` vs `(tile, state)` based on the
   tile's class.  This is the most principled fix and aligns with
   the typed-reaction model in §3.2 of the paper, but it requires
   touching `proof_search._expand` and a per-tile class precompute.

## 4. Lift vs prior baselines

| metric | WF-Lambda-MCTS-Coords-Fix (10×3, baseline) | WF-Partner-Tiles-PathA (10×3, this run) | delta |
|---|---:|---:|---:|
| n_distinct (per cell) | 1 | **1** | **0** |
| diversity_tanimoto | 0.000 | 0.000 | 0 |
| validity_rate | 1.000 | 1.000 | 0 |
| synthesizability_rate | 1.000 | 1.000 | 0 |
| metal_compliance_rate | 0.000 | 0.000 | 0 |
| n_coords_3d_attached_total | 30 | 30 | 0 |
| n_scaffold_aware_gate_active_total | 30 | 30 | 0 |
| decoder_pass_rate | 1.000 | 1.000 | 0 |
| Click-rules fired (per cell) | 0 | **0** | **0** |
| **Path A lift achieved** | — | **NO** | — |

### 4.1 Schema (asked for by the workflow)

```json
{
  "n_distinct_after_partner_tiles": 1,
  "diversity_tanimoto_after": 0.000,
  "metal_compliance_after": 0.000,
  "n_click_rules_fired_per_cell_total": 0,
  "lift_vs_coords_fix_baseline": 0,
  "all_metrics_patha_vs_baseline": {
    "n_distinct": "0 (stayed at 1)",
    "diversity_tanimoto": "0 (stayed at 0.000)",
    "diversity_homotype": "0 (stayed at 0.000)",
    "metal_compliance_rate": "0 (stayed at 0.000)",
    "validity_rate": "0 (stayed at 1.000)",
    "synthesizability_rate": "0 (stayed at 1.000)",
    "novelty": "0 (stayed at 1.000)",
    "decoder_pass_rate": "0 (stayed at 1.000)",
    "n_coords_3d_attached_total": "0 (stayed at 30)",
    "n_scaffold_aware_gate_active_total": "0 (stayed at 30)"
  },
  "success": false,
  "root_cause": "rule-dispatch asymmetry — click_cuaac requires azide-first arg, but MCTS root is [Pt]C#C (alkyne). No partner-tile lift because no click rule can fire on a state whose only handle is alkyne unless the rule accepts alkyne as first arg."
}
```

## 5. Honest framing — why this is NOT a failure of partner tiles

The workflow's hypothesis was: "if the tile pool contains azide +
boronic + bromide handles, the CuAAC/SPAAC/Suzuki rules can fire on
a bare Pt-alkyne seed."  The hypothesis was **partially wrong**: the
*chemistry* is reachable (every required tile is in the pool) but the
*software dispatch* doesn't pair an azide-bearing tile with the
alkyne root because the rule functions are *first-arg-asymmetric*.

The partner tiles ship (verified by inspection of
`tile_lib/click_tiles.py:_build_v2_partner_tiles` and the
`PARTNER_TILES_V2`/`AZIDE_PARTNER_TILES`/`BORONIC_PARTNER_TILES`/
`BROMIDE_PARTNER_TILES` accessors), but they are *latent*: until the
rule-dispatch asymmetry is fixed, they sit unused.

This finding is **more diagnostic** than the prior baseline
(`WF-Lambda-MCTS-Coords-Fix`) which attributed the singleton collapse
to "no azide in the tile pool" — the partner-tile lift *is* in the
pool, and the collapse persists.  The Path A test falsifies the
"missing tile" hypothesis and replaces it with "rule-dispatch
asymmetry" — a more actionable diagnosis.

## 6. Recommendations for Phase 3

1. **Fix 1 (cheapest, ~50 lines).**  Make
   `click_cuaac`/`click_spaac`/`click_suzuki`/etc. symmetric by
   trying both arg orderings.  Re-run Path A → expect
   `n_distinct > 1` because each tile in the 200-tile pool becomes
   a *valid* mutation target on the bare Pt-alkyne root.
2. **Fix 2 (medium).**  Add dual-root seeding (alkyne + azide) so
   each rule can fire from the correct starting handle.  Re-run Path A.
3. **Document the dispatch-asymmetry root cause in paper §6
   (limitations).**  This is honest framing: the Phase-2 negative
   result is a *useful* negative result because it identifies the
   rule-dispatch asymmetry as the actual blocker — not the tile pool
   as previously believed.

## 7. Files written

* `molmetal/reports/wf_lambda1_wf_partner_tiles_patha/r4c/report.json`
  — full 30-cell per-cell metrics (1.4 MB)
* `molmetal/reports/wf_lambda1_wf_partner_tiles_patha/r4c/summary.md`
  — driver-generated summary
* `molmetal/reports/wf_partner_tiles_patha/final.md` — this report

**No code edits** were made in this workflow.  The partner-tile
library (`PARTNER_TILES_V2`) was already shipped in
`molmetal/molmetal_lam/tile_lib/click_tiles.py`; Phase 2 was a
diagnostic run to confirm whether adding tiles to the pool lifts
diversity.  The answer is: **tiles alone are insufficient; the
rule-dispatch asymmetry is the actual blocker.**
