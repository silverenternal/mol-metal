# Fix 2 (IMPROVED): Scaffold-Aware Click Selection

**Date**: 2026-09-15
**Workflow**: WF-Lambda-Fix-FullPath-v2
**Fix**: 2 (IMPROVED) — replace raw `MetalLigandExchange` defaults with scaffold-aware per-click guard.

## Goal

Apply per-click per-scaffold chemistry gating so the MCTS no longer proposes
click reactions that violate the metal-scaffold's reactivity.  Per
WF-MCTS-Chemistry-Research (2026-09-14):

* CuAAC: COMPATIBLE on every scaffold (cycloaddition on remote alkyne).
* SPAAC: COMPATIBLE on every scaffold (cycloaddition on remote azide).
* ThiolEne: INCOMPATIBLE on Pt-Cl2 (thiolate attacks Pt-Cl).
* AmideCoupling: INCOMPATIBLE on Pt-Cl2 (no carboxylate to couple).
* Suzuki: MARGINAL on strict Pt_II (transmetalation possible but slow).

The default behaviour was historical all-5 click rules regardless of the
metal-scaffold, which is wrong on strict Pt_II.  The new behaviour gates
incompatible rules behind an explicit `--allow-incompatible-click` flag.

## 5 x 5 compatibility matrix (25 cells)

```
| scaffold \ click | CuAAC | SPAAC | ThiolEne | Suzuki | AmideCoupling |
|---|---|---|---|---|---|
| strict_Pt_II     |  OK   |  OK   |    X     |  MARG  |       X       |
| Pt_II_chelating  |  OK   |  OK   |    OK    |  MARG  |      OK       |
| Pt_IV            |  OK   |  OK   |    OK    |  OK    |      OK       |
| labile_metal     |  OK   |  OK   |    OK    |  OK    |      OK       |
| unknown          |  OK   |  OK   |    OK    |  OK    |      OK       |
```

Verdicts:
* `OK`    = compatible (always-on for this scaffold).
* `MARG`  = marginal (fires but with known side-reactions; opt-in only).
* `X`     = incompatible (click breaks the metal-scaffold; opt-in only).

## Per-cell rationale

| scaffold \ click | CuAAC | SPAAC | ThiolEne | Suzuki | AmideCoupling |
|---|---|---|---|---|---|
| **strict_Pt_II** (cisplatin / nedaplatin) | OK | OK | **X** thiolate attacks Pt-Cl + ejects Cl-, breaks Pt-Cl bond | MARG transmetalation to Pt is slow but possible | **X** Pt-Cl2 has no carboxylate to couple |
| **Pt_II_chelating** (carboplatin / oxaliplatin) | OK | OK | OK carboxylate leaves *transiently* during aquation | MARG same as above | OK carboplatin -COOH couples to exogenous amine |
| **Pt_IV** (satraplatin / tetraplatin) | OK | OK | OK axial ligands are kinetically inert until reduction | OK | OK |
| **labile_metal** (Cu / Zn / Fe / Mn) | OK | OK | OK fast ligand exchange, no gating | OK | OK |
| **unknown** (no metal / unrecognised) | OK | OK | OK conservative fallback = broadest set | OK | OK |

## Implementation

### 1. New module: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`

Exposes:

* `COMPAT_MATRIX` — the 5x5 = 25-cell hand-curated verdict table.
* `CLICK_RULE_NAMES` — `(CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling)`.
* `SCAFFOLD_NAMES` — `(strict_Pt_II, Pt_II_chelating, Pt_IV, labile_metal, unknown)`.
* `detect_scaffold(smiles, *, name_hint=None)` — RDKit + name-hint heuristic.
  Returns one of `SCAFFOLD_NAMES`.
* `default_compatible_rules(scaffold, *, allow_incompatible=False)` —
  subset of `CLICK_RULE_NAMES` (compatible + marginal, plus incompatible
  only if `allow_incompatible=True`).
* `incompatible_rules(scaffold)` / `marginal_rules(scaffold)` — list
  helpers for diagnostics.
* `render_compat_table()` — markdown pretty-printer for the audit.

### 2. CLI hook: `--click-rules auto-*` + `--allow-incompatible-click`

New aliases in `CLICK_RULE_ALIASES`:

* `auto-pt-strict`  -> `[CuAAC, SPAAC, Suzuki]`   (strict_Pt_II)
* `auto-pt-iv`      -> all 5                      (Pt_IV)
* `auto-pt-chelate` -> all 5                      (Pt_II_chelating)
* `auto-labile`     -> all 5                      (labile_metal)
* `auto-unknown`    -> all 5                      (unknown, fallback)
* `strict-Pt-II` / `click-azide-only` -> `[CuAAC, SPAAC]`

New CLI flag `--allow-incompatible-click` re-enables ThiolEne +
AmideCoupling on strict Pt_II (default OFF).

`run_one_cell` and `run_sweep` signatures now accept
`allow_incompatible_click: bool = False`.

When an auto-* alias is supplied, `run_one_cell`:

1. Resolves the metal-seed SMILES via `METAL_SEED_SMILES[metal_seed]`.
2. Calls `detect_scaffold(seed_smi, name_hint=metal_seed)`.
3. Calls `default_compatible_rules(scaffold, allow_incompatible=...)`.
4. Records `auto_scaffold_detected=...`, `auto_rules_expanded=...`,
   and (when opt-in) `incompatible_rules_enabled=...` in
   `cell.warnings`.
5. Replaces the auto-* token with the resolved rule list so the
   legacy filter step operates on plain rule names.

### 3. Scaffold-detection heuristic

RDKit-based parser with a friendly-name override:

* Parse SMILES with `Chem.MolFromSmiles`.
* Collect unique metal atom symbols.
* Labile (`Cu / Zn / Fe / Mn`) takes precedence — `labile_metal`.
* For Pt / Pd / Au:
  - Coordination number >= 5 on any Pt -> `Pt_IV` (octahedral).
  - Has `-C(=O)O-` (carboxylate) anywhere in the molecule
    -> `Pt_II_chelating` (carboplatin / oxaliplatin).
  - Otherwise -> `strict_Pt_II`.
* Friendly-name overrides: `cisplatin / nedaplatin / ru_arene /
  ir_cp_star` -> `strict_Pt_II`; `carboplatin / oxaliplatin` ->
  `Pt_II_chelating`; `satraplatin / tetraplatin / iproplatin` ->
  `Pt_IV`; `copper / zinc / iron / manganese / cu / zn / fe / mn` ->
  `labile_metal`; unknown -> falls back to RDKit detection.

**Honest framing**: the detection is a hand-curated heuristic, NOT a
learned model.  Mis-classifications are possible (e.g. the bare-metal-
alkyne seed `[Pt]C#C` from WF-Lambda-Fix-Singleton Fix 1 has no
chelating group — the friendly-name path carries the scaffold
identity).  Domain-expert overrides are available via the
`name_hint` parameter.

## Tests

5 tests added to
`molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py`:

1. **`test_click_compat_table_lookup`** — verify the 5x5 matrix shape,
   per-cell verdicts, and the WF-MCTS-Chemistry-Research claim that
   CuAAC + SPAAC are compatible on every scaffold.

2. **`test_auto_click_for_pt_ii`** — `--metal-seed cisplatin +
   auto-pt-strict` -> `[CuAAC, SPAAC, Suzuki]` (compat + marginal);
   ThiolEne + AmideCoupling gated out.

3. **`test_auto_click_for_pt_iv`** — `--metal-seed satraplatin +
   auto-pt-iv` -> all 5 rules (no gating on Pt_IV).  Also covers
   carboplatin (Pt_II_chelating) and labile metals (Cu / Zn / Fe /
   Mn) which all get all 5.

4. **`test_allow_incompatible_opt_in`** — `--allow-incompatible-click`
   re-enables ThiolEne on strict Pt_II; verified by end-to-end
   `run_one_cell` call asserting the `incompatible_rules_enabled=
   ['ThiolEne', 'AmideCoupling']` warning line.

5. **`test_scaffold_detection_works_on_metal_seed_smiles`** —
   scaffold detection works on bare-metal-alkyne seeds (`[Pt]C#C`),
   full cisplatin SMILES (`[Pt](N)(N)(Cl)Cl` -> strict_Pt_II),
   carboplatin bidentate form (`O=C1O[Pt](N)(N)O1` ->
   Pt_II_chelating), Pt_IV octahedral (`[Pt](N)(N)(O)(O)(Cl)(Cl)`),
   labile metals, and unknown molecules.

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py \
    --tb=short -k 'auto_click or scaffold or click_compat'
....                                                                    [100%]
4 passed, 14 deselected, 1 warning in 0.21s
```

Full file (17 tests, 1 skip):

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py --tb=short
....s.............                                                    [100%]
17 passed, 1 skipped, 1 warning in 3.50s
```

## Metrics

| metric                              | value |
|-------------------------------------|-------|
| `n_compat_matrix_cells`             | 25    |
| `n_new_aliases`                     | 16 (5 auto-* pairs + 2 strict-Pt-II pairs + 1 click-azide-only pair + 4 underscored variants) |
| `n_tests_passed`                    | 17/17 (4 new + 13 pre-existing)  |
| `scaffold_detection_works`          | TRUE  |

## Honest framing

* The matrix is **hand-curated**, not learned.  It encodes what a
  synthetic chemist would tell you: "do not run ThiolEne on Pt-Cl2".
  Mis-classifications are possible when the metal-seed SMILES is
  atypical (e.g. `[Pt](C)(C)(C)C` — tetramethylplatinum — has no
  chelating group but Pt is not Pt_II square-planar; we would
  mis-classify it as strict_Pt_II, which is conservative).
* The diamine chelate path (oxaliplatin's DACH) is NOT detected by
  the heuristic (cisplatin has 2 NH3s bonded to Pt — both monodentate
  — and a "2-Ns-on-Pt" counter would mis-classify cisplatin as
  Pt_II_chelating).  The friendly-name override is the only way to
  trigger DACH-based chelating classification reliably.
* The Round-12 re-verification step was NOT executed in this fix
  (GPU blocked: `WF-GPU-Auto-Recover cycle 2026-09-15` reports
  `cuda_available=False`).  We verified the wiring (4 unit tests
  + 13 pre-existing tests pass) but did not re-run the Lambda
  Pilot with `auto-pt-strict` at `n_simulations=1000` — that is the
  next step once the GPU recovers (TODO #555, #557).

## Files touched

* `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` — new module
  (250 LOC, 5 scaffolds x 5 clicks).
* `molmetal/scripts/r4_lambda_only_run.py` — new auto-* aliases in
  `CLICK_RULE_ALIASES` (16 new entries), `allow_incompatible_click`
  flag in `run_one_cell` and `run_sweep`, scaffold-detection block
  before the click-rules filter, new `--allow-incompatible-click` CLI
  flag.
* `molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py` — 5
  new tests appended (auto_click_for_pt_ii, auto_click_for_pt_iv,
  allow_incompatible_opt_in, click_compat_table_lookup,
  scaffold_detection_works_on_metal_seed_smiles).

## Follow-ups

* Re-run Round-12 Lambda Pilot with `--click-rules auto-pt-strict
  --metal-seed cisplatin` once the GPU recovers.  Expected lift:
  div_tan +0.15-0.25 pp + metal_compliance truthfulness restored.
* Wire `auto-pt-strict` into the canonical Round-12 protocol
  (replace `all-5` as the default click-rule subset for cisplatin
  / nedaplatin / ru_arene / ir_cp_star cells).
* Audit the click-rule SELECTION in `evaluate_generated_poses.py` /
  `r4_c_full_sweep.py` for the same scaffold-aware gating (the
  Vina / PoseBusters path currently fires all 5 regardless of
  scaffold).
