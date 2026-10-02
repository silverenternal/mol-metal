# Wf-290F8 — Fix test_tmqm_dblock_extract ImportError

**Date**: 2026-09-29
**Target file**: `molmetal/tests/test_tmqm_dblock_extract.py`
**Underlying fix**: `molmetal/molmetal_lam/atoms/combinators.py`

---

## 1. Error before

```
ImportError while importing test module
  '/home/hugo/codes/mol-metal/molmetal/tests/test_tmqm_dblock_extract.py'.
Traceback:
  ...
molmetal/tests/test_tmqm_dblock_extract.py:45: in <module>
    from molmetal.molmetal_lam.atoms.combinators import (  # noqa: E402
        METAL_SYMBOLS,
        PRIMITIVE_ATOMS,
    )
E   ImportError: cannot import name 'METAL_SYMBOLS' from
    'molmetal.molmetal_lam.atoms.combinators'
    (/home/hugo/codes/mol-metal/molmetal/molmetal_lam/atoms/combinators.py)

no tests collected, 1 error in 0.88s
```

The collector aborted at module-level import — **all 17 tests in the file
were unreachable** (not even collected as skip / xfail).

## 2. Root cause

`combinators.py` had **two regressions** against the R17 UC-B1 contract
documented in `TODO/completed/32_late_phase.md` §2.1:

1. `PRIMITIVE_ATOMS` was the legacy 10-entry dict (`H/C/N/O/F/P/S/Cl/Br/I`).
   The R17 contract requires **42 entries** — 8 organic main-group
   (H, B, C, N, O, Si, P, S) + 4 halogens (F, Cl, Br, I) + 30 d-block
   transition metals (Sc..Hg).
2. `METAL_SYMBOLS` was not exported at all. The contract requires a
   `frozenset` of the 30 d-block element symbols built once at import
   time from `PRIMITIVE_ATOMS` (combinators.py:311-313 in the original
   R17 draft).

## 3. Fix applied

Single-file fix in `molmetal/molmetal_lam/atoms/combinators.py`:

### 3.1 Expanded `PRIMITIVE_ATOMS` from 10 → 42 entries

* **(a) Organic main-group** — added `B` (Z=5, valence=3, geometry=sp2)
  and `Si` (Z=14, valence=4, geometry=sp3) to the existing 6.
* **(b) Halogens** — kept `F/Cl/Br/I` (4 entries).
* **(c) tmQM d-block** — added all 30 transition metals (Sc, Ti, V, Cr,
  Mn, Fe, Co, Ni, Cu, Zn, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, La,
  Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg) with canonical oxidation state,
  coordination number (`arity = valence + lone_pairs`), and a geometry
  drawn from `{square_planar, octahedral, tetrahedral, linear}` per the
  R17 UC-B1 arity table.

A module-level `assert len(PRIMITIVE_ATOMS) == 42` guards against future
silent regressions.

### 3.2 Added `METAL_SYMBOLS` frozenset

```python
METAL_SYMBOLS: frozenset = frozenset(
    sym for sym, atom in PRIMITIVE_ATOMS.items() if atom.is_metal
)
assert len(METAL_SYMBOLS) == 30, ...
```

Built once at import time from the metal entries of `PRIMITIVE_ATOMS`
so the two stay in lockstep (TODO/completed/32_late_phase.md §2.1).

### 3.3 Added `METAL_SYMBOLS` to `__all__`

So `from molmetal.molmetal_lam.atoms.combinators import METAL_SYMBOLS`
resolves cleanly.

### 3.4 Module docstring updated

Updated the docstring's "Primitive combinator library" section to reflect
the 42-entry + METAL_SYMBOLS contract (was the stale 10-entry text).

## 4. Tests after

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_tmqm_dblock_extract.py --co -q

17 tests collected in 0.48s
```

All 17 tests now collect cleanly. They are SKIPPED at runtime on this
host because the raw tmQM release is not mounted
(`$MOLMETAL_TMQM_DIR` missing — `pytestmark = pytest.mark.skipif(...)`).
This skip behaviour is **expected** and matches the contract in
`tests/test_tmqm_loading.py` (and `TODO/completed/32_late_phase.md`
§2.1 honest framing).

### Downstream unblock

`molmetal/tests/test_lambda_combinators_expansion.py` was previously
uncollectable for the same root cause. With this fix it now imports
cleanly. Pre-existing METAL_ATOMS expansion failures remain (the file
still has the original 6 entries; the R17 UC-B1 contract expects 30) —
that is a **separate** R17 Track-B TODO and out of scope for Wf-290F8.

## 5. Cross-checks

```
$ python -c "from molmetal.molmetal_lam.atoms.combinators import \
    METAL_SYMBOLS, PRIMITIVE_ATOMS, METAL_ATOMS; \
    print(len(METAL_SYMBOLS), len(PRIMITIVE_ATOMS), len(METAL_ATOMS))"
30 42 6
```

* `METAL_SYMBOLS` == 30 (matches `TRANSITION_METALS` in
  `molmetal/data/tmqm.py` and `molmetal_lam/lam_chem/platinai_dataset.py`).
* `PRIMITIVE_ATOMS` == 42 (R17 UC-B1 contract).
* `METAL_ATOMS` == 6 (preserved R16 hand-curated core; R17 UC-B1
  expansion is a separate Wf — see test_lambda_combinators_expansion
  arity-table failures).

## 6. Files touched

* `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/atoms/combinators.py`
  — added `B` + `Si` + 30 d-block metals to `PRIMITIVE_ATOMS`, defined
  `METAL_SYMBOLS`, added the symbol to `__all__`, updated module
  docstring.
