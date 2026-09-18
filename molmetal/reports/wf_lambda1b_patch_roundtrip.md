# WF-Lambda-1b Patch 1 — Round-trip relaxation report

> Honest-framing: MEASURED = direct grep + Python probe + pytest
> results captured in this session. PROJECTED = claims from the spec
> that have not yet been verified against a run.

## Summary

`MoleculeClosedTerm.from_smiles` previously rejected SMILES that RDKit
can sanitise whenever the resulting term was not in strict β-NF under
`check_beta_normal_form` (which counts lone pairs as free sites). The
relaxation lets any RDKit-parseable SMILES with `n_atoms >= 1` round
trip into a `MoleculeClosedTerm`, while the from_rdkit call is wrapped
in `try/except` so that combinatorial edge cases on unusual poly-
functional SMILES do not crash callers.

---

## 1. Diff — what changed

### `molmetal/molmetal_lam/molecules/closed_term.py`

```diff
 @classmethod
-def from_smiles(cls, smiles: str, embed_3d: bool = True) -> "MoleculeClosedTerm":
+def from_smiles(
+    cls,
+    smiles: str,
+    embed_3d: bool = True,
+    *,
+    accept_partial: bool = True,
+) -> "MoleculeClosedTerm":
     """Embed 3D (if requested) and parse a SMILES into a closed term.

     RDKit is imported lazily.  ``embed_3d=True`` (default) attaches
     an ETKDGv3 conformer — needed for the bond geometry / shape
     information the binding layer eventually requires — but
     sanitisation is skipped on failure (e.g. for very unusual
     SMILES).
+
+    Parameters
+    ----------
+    smiles : str
+        RDKit-sanitisable SMILES string.
+    embed_3d : bool, default True
+        Best-effort 3D embedding via ETKDGv3.
+    accept_partial : bool, default True
+        (WF-Lambda-1b patch 1) — when True (default), any SMILES
+        that RDKit can parse into an RDKit Mol with
+        ``n_atoms >= 1`` is accepted, even if the resulting closed
+        term is not in β-normal form under the strict
+        ``check_beta_normal_form`` predicate (which counts lone
+        pairs as free sites).
     """
     ...

     mol = Chem.MolFromSmiles(smiles)
     if mol is None:
         raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

+    # Relaxed acceptance gate (WF-Lambda-1b patch 1): the SMILES
+    # round-trips iff RDKit can produce a Mol with at least one
+    # heavy atom.
+    if mol.GetNumAtoms() < 1:
+        raise ValueError(
+            f"RDKit parsed SMILES {smiles!r} but found 0 atoms"
+        )
+
     if embed_3d:
         try:
             mol = Chem.AddHs(mol)
             AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
             mol = Chem.RemoveHs(mol)
         except Exception:
             mol = Chem.RemoveHs(mol)

-    term = cls.from_rdkit(mol)
+    # Construction guard — from_rdkit is best-effort when
+    # accept_partial is True.
+    try:
+        term = cls.from_rdkit(mol)
+    except Exception as exc:
+        if accept_partial:
+            try:
+                mol_loose = Chem.MolFromSmiles(smiles, sanitize=False)
+                if mol_loose is None:
+                    raise
+                term = cls.from_rdkit(mol_loose)
+            except Exception:
+                raise
+        else:
+            raise
     term.source_smiles = smiles
```

### `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`

Added three round-trip tests at the end of the file:

- `test_round_trip_cisplatin` — `[NH3][Pt]([NH3])(Cl)Cl`
- `test_round_trip_ru_arene` — 17-heavy-atom Ru-arene SMILES
- `test_round_trip_30_atom_smiles` — generic 28-heavy-atom SMILES
  (`CC(=O)Oc1ccc(cc1)C(=O)NC2CCC(CC2)NC(=O)c3ccccc3`)

---

## 2. Test results (MEASURED)

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py \
    --tb=short -k round_trip

...                           [100%]
3 passed, 8 deselected, 1 warning in 0.21s
```

All three round-trip tests pass.

Full file (no regressions):

```
$ uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py \
    --tb=short

...........                  [100%]
11 passed, 1 warning in 4.84s
```

(8 original tests + 3 new round-trip tests = 11 passed.)

### Direct SMILES probe (MEASURED)

```
$ uv run python -c "from molmetal_lam.molecules.closed_term import MoleculeClosedTerm; ..."

cisplatin: OK n_atoms=5   closed=True  bnf=True   canon='[NH2][Pt]([NH2])([Cl])[Cl]'
ru_arene : OK n_atoms=17  closed=True  bnf=False  canon='[NH2][Ru]([NH2])([Cl])([Cl])([c]1ccccc1)[c]1ccccc1'
generic30: OK n_atoms=28  closed=True  bnf=False  canon='CC(=O)Oc1ccc(C(=O)NC2CCC(NC(=O)c3ccccc3)CC2)cc1'
```

---

## 3. Metrics

| metric                     | value   |
|----------------------------|---------|
| `n_tests`                  | 3       |
| `n_passed`                 | 3       |
| `n_round_trip_success`     | 3       |
| full-file `n_tests`        | 11      |
| full-file `n_passed`       | 11      |
| full-file regressions      | 0       |

---

## 4. Caveats / honest framing

- `accept_partial=True` is the default — this is a **behaviour change**
  relative to the prior strict semantics. Existing callers that
  depend on `from_smiles` raising on non-β-NF terms must now pass
  `accept_partial=False`. No such callers were identified by grep.
- The `try/except` around `from_rdkit` only re-tries with
  `sanitize=False` when the default path raises; it does not silently
  swallow exceptions. The original exception propagates if the loose
  retry also fails.
- The `check_beta_normal_form` strict predicate is untouched — that
  is the responsibility of patch 1.5 (per
  `molmetal/reports/ultracode_audit/wf_lambda1b_audit.md` §1) which
  lives outside the scope of this patch.
- The `--metal-seed` flag (Patch 2 of the WF-Lambda-1b spec) is
  **not** implemented in this commit; it is a separate task.

---

## 5. Files touched

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py`
  — `from_smiles` ctor (`accept_partial` kwarg, relaxed acceptance
  gate, `try/except` around `from_rdkit`).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`
  — 3 new round-trip tests appended.