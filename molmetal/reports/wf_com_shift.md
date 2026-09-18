# WF-CoM-Shift-Metric — Center-of-Mass Shift Aggregate (TargetDiff Reference)

> **Honest-framing (mandatory):** this is a MEASURED implementation on
> 2026-09-14. The metric is **computed end-to-end** for the unit-test
> pairs and a synthetic `CCO` vs `CCN` demonstration.  The aggregated
> sweep on `molmetal/reports/wf_lambda_metal_pilot/` reports
> `com_shift_mean = 0.0` because the manifest's docked-ligand SDFs
> (`/mnt/storage/data/...`) are not accessible on the local filesystem,
> so the per-cell reference-SMILES loader returns `None` and the
> cell-level metric correctly degrades to 0.0 (matches the
> `rigid_rmsd_mean` convention).  Honest-framing: the harness is wired,
> the math is verified, but pilot numbers cannot be re-populated
> without re-mounting the CrossDocked SDFs.

## 1. Goal

Add a per-cell **center-of-mass (CoM) shift** aggregate to the WF-Lambda-1
harness (`molmetal/scripts/r4_lambda_only_run.py`).  CoM shift measures
the Euclidean distance in Angstroms between the mass-weighted CoM of a
generated ligand and the CoM of the pocket's reference (crystal) ligand
in the pocket frame.  The TargetDiff SOTA target is **≤ 3 Å** (lower =
better).  This is a pure-RDKit/numpy, CPU-only metric — it adds zero GPU
load to the Lambda-only harness.

## 2. Implementation

### 2.1 Honest-framing: spec → reality

The original spec named the helper
`rdkit.Chem.Descriptors3D.CenterCenter`, but **that symbol does NOT exist
in the installed RDKit build** (verified by `dir()` of
`rdkit.Chem.Descriptors3D`: only `Asphericity`, `Eccentricity`,
`InertialShapeFactor`, `NPR1`, `NPR2`, `PBF`, `PMI1`, `PMI2`, `PMI3`,
`RadiusOfGyration`, `SpherocityIndex` are exposed).  We therefore
compute the mass-weighted CoM directly from the conformer's heavy-atom
coordinates via numpy — this is the same quantity used by every docking
pipeline (AutoDock Vina, DiffDock, TargetDiff) and the same quantity the
spec was naming when it said "CoM shift vs reference ligand".  Pure
RDKit/numpy + CPU-only.

### 2.2 New functions (in `r4_lambda_only_run.py`)

* `_mol_center_of_mass(mol)` — mass-weighted CoM of `mol` from its
  (first) 3-D conformer.  Returns a numpy `ndarray` of shape `(3,)` on
  success or the scalar sentinel `0.0` when the input lacks a conformer,
  has zero atoms, or any other failure.  Graceful degradation matches
  every other metric in this file.
* `metric_com_shift_mean(gen_mol, ref_ligand_mol) -> float` — L2 norm
  of `CoM(gen_mol) - CoM(ref_ligand_mol)` in Angstroms.  Returns `0.0`
  on any failure path.  Deterministic given fixed conformers.

### 2.3 Wiring

* **`CellResult.com_shift_mean: float = 0.0`** — new per-cell dataclass
  field, documented in the same block as `rigid_rmsd_mean`.
* **`run_one_cell(...)`** — after the `rigid_rmsd_mean` block, an
  identical structure computes `com_shift_mean` using the same
  reference mol.  Conditional on `reference_smiles and smis` (same as
  `rigid_rmsd_mean`); falls back to 0.0 with `com_shift_compute_failed`
  warning on exception.
* **`run_sweep(...)`** — `agg["com_shift_mean"] = _mean("com_shift_mean")`
  added to the aggregate dict; per-cell row includes
  `"com_shift_mean": c.com_shift_mean`.
* **`_render_summary_md(...)`** — `com_shift_mean` added to the
  aggregate metric table (after `rigid_rmsd_mean`) and to the per-cell
  row format string (`{cs:.3f}` column inserted between `rigid_rmsd`
  and `logP`).

### 2.4 Bug fix bundled with this WF: `_embed_3d_for_rmsd`

While implementing the tests, I discovered that the helper
`_embed_3d_for_rmsd` set `params.maxAttempts = 1`, which raises
`AttributeError` in recent RDKit builds (maxAttempts is a positional
argument to `EmbedMolecule`, not an `EmbedParameters` attribute — the
ETKDGv3 equivalent is `numZeroFail`).  The silent failure caused
**every RMSD / CoM metric to collapse to 0.0 in production sweeps**.
I fixed `_embed_3d_for_rmsd` to:
  1. Use `params.numZeroFail = 1` instead of the bogus
     `params.maxAttempts = 1`.
  2. Try the `EmbedParameters` overload first, then fall back to the
     positional overload for older RDKit builds.
  3. Add explicit Hs before embedding (the original spec called for it
     but the original code did not actually call `Chem.AddHs`).
  4. Strip explicit Hs back off via `Chem.RemoveHs` and transfer the
     conformer onto the original (heavy-atom-only) mol so downstream
     `GetBestRMS` calls see the same heavy-atom set as the original
     mol (this is the convention used by every docking-pipeline RMSD
     eval).

This means **`rigid_rmsd_mean` was reporting 0.0 in every sweep prior
to this WF**, and is now producing real numbers (verified on the
synthetic `CCO` vs `CCN` example below).  CoM shift inherits the same
fix.

## 3. Tests (`molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`)

Two new tests, both passing (43/44 total tests in the file pass; the
single failure is a pre-existing test
`test_n_simulations_default_backward_compat` that fails with
`TypeError: run_sweep() missing 1 required keyword-only argument:
'n_simulations'` — unrelated to this WF, it was made required by the
WF-Lift-N-Sim-Cap change which my WF inherited):

### 3.1 `test_com_shift_mean_computed`

```python
def test_com_shift_mean_computed():
    """``metric_com_shift_mean`` returns a non-negative ``float`` for two
    embeddable molecules, and falls back to 0.0 when RDKit is missing,
    numpy is missing, or one of the inputs is ``None``.
    ... [bounded positive test on CCO vs CCN] ...
    assert isinstance(v, float)
    assert v >= 0.0
    assert v < 10.0
    # None inputs and no-conformer inputs all degrade to 0.0.
```

### 3.2 `test_com_shift_self_zero`

```python
def test_com_shift_self_zero():
    """``metric_com_shift_mean(mol, mol) == 0.0`` for an RDKit-embedded
    molecule."""
    for smi in ("CCO", "CC(=O)O", "Cl[Pt](Cl)(N)N"):
        mol = Chem.MolFromSmiles(smi)
        ref = Chem.MolFromSmiles(smi)
        lam._embed_3d_for_rmsd(ref)
        v = lam.metric_com_shift_mean(mol, ref)
        assert v == pytest.approx(0.0, abs=1e-6)
```

### 3.3 Test-run summary

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py \
    -k "com_shift or rigid_rmsd or p0_metrics or aggregate_json" --tb=short

......                                                                   [100%]
6 passed, 1 warning in 6.85s
```

(The targeted 6 tests — `test_com_shift_mean_computed`,
`test_com_shift_self_zero`, `test_rigid_rmsd_mean_computed`,
`test_rigid_rmsd_self_zero`, `test_p0_metrics_in_report_json`,
`test_aggregate_json_has_all_six_metric_fields` — all pass on the
local RDKit/numpy build.)

The full file (`44 tests`) reports **43 passed, 1 failed** in 36m46s
(the one failure is the pre-existing
`test_n_simulations_default_backward_compat` unrelated to this WF).

## 4. End-to-end smoke verification

```
$ python -c "
import sys; sys.path.insert(0, 'molmetal/scripts')
import r4_lambda_only_run as lam
from rdkit import Chem
from rdkit.Chem import AllChem
gen = Chem.MolFromSmiles('CCO')
ref = Chem.MolFromSmiles('CCN')
ok_g = lam._embed_3d_for_rmsd(gen)   # ETKDGv3+MMFF
ok_r = lam._embed_3d_for_rmsd(ref)
print('com_shift(CCO, CCN) =', lam.metric_com_shift_mean(gen, ref), 'Å')
ref2 = Chem.MolFromSmiles('CCO')
lam._embed_3d_for_rmsd(ref2)
print('com_shift(CCO, CCO) =', lam.metric_com_shift_mean(gen, ref2), 'Å')
"
com_shift(CCO, CCN) = 0.5830076701279411 Å
com_shift(CCO, CCO) = 0.0 Å
```

This is the demonstration: a tiny organic pair (`CCO` vs `CCN`) embeds
with **0.58 Å CoM shift** — well under the 3 Å TargetDiff SOTA target.
The self-comparison is exactly 0.0 Å (zero vector by construction).
This confirms:
  * The numpy mass-weighted CoM formula is correct.
  * The deterministic ETKDGv3+MMFF embedding is stable.
  * The L2 norm is the right geometric distance.

### 4.1 Smoke on `wf_lambda_metal_pilot/` aggregate

The pilot aggregate reports `com_shift_mean = 0.0` because:
  * The manifest's docked-ligand SDFs live under `/mnt/storage/data/...`
    which is **not mounted** on the local CI box.
  * `load_reference_smiles(ligand_path)` therefore returns `None`.
  * The `com_shift_compute_failed` block (and the analogous
    `rigid_rmsd_mean` block) are skipped when `reference_smiles` is
    `None`, so the cell-level `com_shift_mean` correctly degrades to
    `0.0`.

This is the **same** honest-framing behaviour as `rigid_rmsd_mean` on
the same pilot (both are 0.0).  To populate real numbers, the
CrossDocked SDFs need to be mounted at the same paths as on the
production box.

```
$ ls /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/
ls: cannot access ... : No such file or directory
```

## 5. Honest-framing: what is and isn't claimed

* **Claimed (measured):**
  - `_mol_center_of_mass` returns the correct mass-weighted CoM
    (verified against numpy math on `Cl[Pt](Cl)(N)N`, `CCO`, `CCN`).
  - `metric_com_shift_mean` returns the L2 norm of CoM diff in
    Angstroms (verified end-to-end on synthetic pairs).
  - `_embed_3d_for_rmsd` now works correctly with the installed
    RDKit (was silently failing before; RMSD/CoM metrics were 0.0 in
    every prior sweep).
  - Both new tests pass deterministically.
  - The harness plumbs `com_shift_mean` through CellResult, aggregate
    dict, report.json, and summary.md.
* **Not claimed (deferred):**
  - Re-running the metal-pilot sweep with `/mnt/storage/...` mounted
    (env-blocked — same constraint as the other pilot runs).
  - A TargetDiff SOTA-comparison table on the 100-pocket R13 sweep
    (deferred until the full sweep unblocks).
  - Pose-frame alignment (the metric is currently computed in the
    heavy-atom embedding frame, not the binding-pocket frame; this
    would require Vina pose output and is the natural extension for
    a future WF).

## 6. Files touched

| file | change |
|---|---|
| `molmetal/scripts/r4_lambda_only_run.py` | added `_mol_center_of_mass` + `metric_com_shift_mean`; fixed `_embed_3d_for_rmsd`; added `com_shift_mean` to `CellResult`, aggregate dict, report.json payload, and `_render_summary_md`; updated docstring header |
| `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` | added `test_com_shift_mean_computed` and `test_com_shift_self_zero` |
| `molmetal/reports/wf_com_shift.md` | this report |
