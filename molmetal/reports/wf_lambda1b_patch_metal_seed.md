# WF-Lambda-1b Patch 2 — `--metal-seed` CLI flag

**Date:** 2026-09-14
**Author:** WF-Lambda-1b Patch 2
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Status:** SHIPPED — exit code 0, 1 cell, 1 Pt-bearing candidate emitted, `metal_compliance_rate = 1.0` (MEASURED).

---

## Honest-framing

- MEASURED numbers in this report come from running
  `molmetal/scripts/r4_lambda_only_run.py` with `--metal-seed cisplatin`
  on the local gfx1101 box on 2026-09-14.
- PROJECTED numbers are quoted only when explicitly noted as such.
- This patch fixes the second of the two structural defects from the
  WF-Lambda-1 verify report: **the absence of a metal-seeded root**.
  After this patch, `metal_compliance_rate` is non-zero in the
  re-run because every candidate inherits a Pt/Ru/Ir centre from the
  metal-containing root scaffold.

---

## Goal

Add an optional metal-containing root scaffold to the Lambda-only
harness so the search starts inside the Pt/Ru/Ir coordination
subspace.  Before this patch every search began from an organic
reference ligand (or the `Cl[Pt]Cl` fallback), so the metal prior
could only ever fire on candidates the random policy happened to
push into a metal-bearing product.  After this patch the user can
seed MCTS from a canonical cisplatin / Ru-arene / Ir-Cp* molecule
and watch `metal_compliance_rate` climb.

---

## CLI diff

### `molmetal/scripts/r4_lambda_only_run.py`

Added module-level constant + CLI argument + plumbing into
`run_one_cell` and `run_sweep`.

```diff
+ # Metal-seed SMILES (canonical, RDKit-sanitizable).
+ METAL_SEED_SMILES: Dict[str, str] = {
+     "cisplatin":  "[NH3][Pt]([NH3])(Cl)Cl",
+     "ru_arene":   "[Ru](c1ccccc1)(c1ccccc1)(Cl)(Cl)(N)N",
+     "ir_cp_star": "[Ir](C1C(C)=C(C)C(C)=C1C)(Cl)(N)N",
+ }

  def run_one_cell(
      *,
      pocket_id: str,
      reference_smiles: Optional[str],
      seed: int,
      n_simulations: int,
      n_top_k: int,
      prior_enabled: bool = True,
      max_depth: int = 3,
      click_rules: Optional[Sequence[str]] = None,
+     metal_seed: Optional[str] = None,
  ) -> CellResult:

  # Build the root state.  Order of preference:
  #   1. --metal-seed
  #   2. reference_smiles
  #   3. fallback Cl[Pt]Cl
+ if metal_seed is not None:
+     seed_smi = METAL_SEED_SMILES.get(metal_seed)
+     if seed_smi is None:
+         cell.warnings.append(...)
+     else:
+         cell.warnings.append(f"metal_seed_active={metal_seed} smi={seed_smi}")
+         try:
+             root = MoleculeClosedTerm.from_smiles(seed_smi, embed_3d=False)
+         except Exception as exc:
+             cell.warnings.append(f"metal_seed_from_smiles_failed: {exc}")

  def run_sweep(..., metal_seed: Optional[str] = None):
      ...
      cell = run_one_cell(..., metal_seed=metal_seed)

+ parser.add_argument(
+     "--metal-seed",
+     nargs=None,
+     default=None,
+     choices=[None, "cisplatin", "ru_arene", "ir_cp_star"],
+     help="Optional metal-containing root scaffold for MCTS ...",
+ )
```

### `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`

Added 6 tests under a "WF-Lambda-1b Patch 2" header:

* `test_metal_seed_smiles_table_cisplatin_has_pt`
* `test_metal_seed_smiles_table_ru_arene_has_ru`
* `test_metal_seed_smiles_table_ir_cp_star_has_ir`
* `test_metal_seed_smiles_table_keys`
* `test_metal_seed_cisplatin_runs` (1 pocket × 1 seed smoke, asserts
  `metal_seed_active=cisplatin` warning appears)
* `test_metal_seed_ru_arene_runs` (same shape, ru_arene)

---

## `--metal-seed` choices

| key          | canonical SMILES                              | heavy atoms | prior coord. target |
|--------------|-----------------------------------------------|-------------|---------------------|
| cisplatin    | `[NH3][Pt]([NH3])(Cl)Cl`                       | 5           | Pt=4 (square-planar)|
| ru_arene     | `[Ru](c1ccccc1)(c1ccccc1)(Cl)(Cl)(N)N`          | 17          | Ru=6 (octahedral)   |
| ir_cp_star   | `[Ir](C1C(C)=C(C)C(C)=C1C)(Cl)(N)N`            | 14          | Ir=6 (octahedral)   |

All three canonical SMILES are RDKit-sanitizable; verified by the
three `test_metal_seed_smiles_table_*_has_<metal>` unit tests.

---

## Smoke run

**Command** (verbatim from the spec):

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pockets 1 --seeds 42 --n-simulations 50 --n-top-k 5 \
    --metal-seed cisplatin \
    --output-dir /tmp/wf_lambda1b_smoke --quiet
```

**Result** (MEASURED, 2026-09-14, gfx1101):

* **Exit code:** `0`
* **n_cells:** `1`
* **n_candidates:** `1`
* **n_candidates_with_Pt:** `1` (candidate = `[NH2][Pt]([NH2])([Cl])[Cl]`)
* **aggregate:**
  * `validity_rate`         = `1.0000`
  * `uniqueness_rate`       = `1.0000`
  * `diversity_alpha`       = `0.0000`
  * `novelty`               = `1.0000`
  * `synthesizability_rate` = `0.0000`
  * `metal_compliance_rate` = `1.0000`  ← was 0.0 before the patch
  * `reference_tanimoto`    = `0.0141`
* **Cell warnings:** `["metal_seed_active=cisplatin smi=[NH3][Pt]([NH3])(Cl)Cl"]`
* **elapsed_s_total:** `~3.86 s`

Output paths:
* JSON: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_/tmp/wf_lambda1b_smoke/report.json`
* Markdown: same dir `summary.md`
* (NOTE: `--output-dir /tmp/...` is concatenated as `wf_lambda1_/tmp/...` because the harness always prefixes with `molmetal/reports/`. This is the same behaviour as the existing `--output-dir` flag and is documented in the script's `--help`.)

---

## Test results

```
uv run python -m pytest molmetal/molmetal_lam/tests/test_lambda_only_metrics.py -v
======================== 17 passed, 1 warning in 5.08s =========================
```

All 17 tests green, including:

* 8 original Lambda-only baseline tests (untouched)
* 3 Patch 1 round-trip tests (`test_round_trip_cisplatin`,
  `test_round_trip_ru_arene`, `test_round_trip_30_atom_smiles`)
* 6 Patch 2 metal-seed tests

---

## Discussion — what `metal_compliance_rate = 1.0` actually means

With `--metal-seed cisplatin` the root state is
`[NH3][Pt]([NH3])(Cl)Cl`.  Every β-reduction step applied by MCTS
inherits the Pt centre (and at least 4 of the 5 coordination
bonds).  Therefore:

1. **Every** leaf the search returns contains Pt.
2. **Every** leaf therefore scores `1.0` on
   `metal_geometry_prior_bonus` (Pt with coord=4 satisfies the
   square-planar target).
3. The cell-level `metal_compliance_rate` is the mean over candidates
   and equals `1.0`.

This is exactly what the patch was supposed to deliver: a non-zero
`metal_compliance_rate` baseline from which a future ablation can
deliberately weaken (e.g. by setting `--metal-seed None` to fall back
to the organic reference).

The flip side — `synthesizability_rate = 0.0` on the same cell —
reflects the fact that the patched-from_smiles gate still classifies
cisplatin as *not* in strict beta-NF under the lone-pair counting
rule.  Patch 1 relaxed the rejection to a non-fatal warning; the
synthesizability_score channel will become non-zero once the
metal-arity accounting (the W-3 of the Round-3 algorithmic axes) is
fully wired in.

---

## Files touched

* `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`
  — added `METAL_SEED_SMILES`, `metal_seed=` kwarg on `run_one_cell`
  + `run_sweep`, root-state plumbing, CLI flag, summary rendering,
  and `payload["config"]` recording.
* `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`
  — 6 new tests under "WF-Lambda-1b Patch 2".
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1b_patch_metal_seed.md`
  — this report.
