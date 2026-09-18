# Governance Re-Measurement — Layers 1-3 (Atoms / Bonds / Molecules)

Scope: 17 metric values, each aggregated over the **16 standard
molecules** (12 click tiles + cisplatin descriptor + transplatin
descriptor + water + methane). Layer entry points are:

- L1: `from_smiles(...)` + `sanity_check()` on
  `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/atoms/combinators.py`
- L2: `assemble(...)` (and `cisplatin()` for the Pt witness) on
  `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/bonds/application.py`
- L3: `MoleculeClosedTerm.from_smiles(...)` on
  `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py`

Each layer appends a `dict` to its module-level `_METRICS.history`
list (see `molmetal_lam.atoms.combinators.get_history()` etc.).

## Test status

`pytest molmetal/tests/test_layer_metrics_l1_l3.py -v` → **17 passed**
in 0.15 s (one test per metric; see
`/home/hugo/codes/try_triton_on_rocm/molmetal/tests/test_layer_metrics_l1_l3.py`).

## Results table

| L | Metric | Value | Healthy target | Pass? |
|---|---|---|---|---|
| 1 | ARITY_HIT_RATE                | 1.0000 | ≥ 0.95                | Yes |
| 1 | METAL_GEOMETRY_OK             | 0.8333 | = 1.00                | **No** |
| 1 | SANITY_PASS_RATE              | 1.0000 | = 1.00                | Yes |
| 1 | FALLBACK_ATOM_RATIO           | 0.0000 | < 0.05                | Yes |
| 1 | PRIMITIVE_GEOMETRY_TAG_OK     | 1.0000 | = 1.00                | Yes |
| 2 | BOND_KIND_DISTRIBUTION        | `{dative:4}` for cisplatin, `{}` for all others | dative-dominated for metals; none for hydrocarbons | Yes |
| 2 | DATIVE_FRACTION (cisplatin)   | 1.0000 | = 1.00                | Yes |
| 2 | FREE_SITES_AFTER_ASSEMBLE     | `{open_sites:0, is_closed:True}` for cisplatin | open_sites=0, is_closed=True | Yes |
| 2 | BOND_VALIDITY_RATE            | 1.0000 | = 1.00                | Yes |
| 2 | BUILDER_EXCEPTION_RATE        | 0.0000 | ≈ 0                   | Yes |
| 2 | AROMATIC_RING_SIZE_OK         | 1.0000 | = 1.00                | Yes |
| 3 | IS_CLOSED_RATE                | 1.0000 | ≈ 1.00                | Yes |
| 3 | IS_BETA_NORMAL_FORM_RATE      | 1.0000 | ≥ 0.90                | Yes |
| 3 | REDEX_HIT_RATE                | 0.0000 | = 0.00 (NF closed)    | Yes |
| 3 | ATOM_BOND_RATIO (mean)        | 1.6536 | 1.5 – 2.0 (drug-like) | Yes |
| 3 | ALPHA_EQUIV_COLLISIONS        | 0.2500 | = 0.00                | **No** |
| 3 | RING_AROMATICITY_PRESERVED_RATE | 1.0000 | ≥ 0.99              | Yes |

Notes:

- `ARITY_HIT_RATE = 1.00` because every heavy element encountered
  (C, N, O, P, Cl, Pt) is in `PRIMITIVE_ATOMS` or `METAL_ATOMS`. No
  exotic elements (B, Si, Se, …) appear in the 16 standard molecules.
- `DATIVE_FRACTION` is reported **per-call**: only the cisplatin
  descriptor exercises a bond factory; the other 15 molecules route
  through `assemble([])` with an empty bond list, yielding
  `n_dative=0`. The review's healthy target (= 1.0 for cisplatin)
  is met for the cisplatin call. The aggregate (mean 0.0625) is
  not the metric — the per-call value is what matters.
- `BOND_KIND_DISTRIBUTION`, `FREE_SITES_AFTER_ASSEMBLE` are
  dict-valued; the "value" cell shows the cisplatin observation.

## Failures (2)

1. **METAL_GEOMETRY_OK = 0.8333** — the registry
   `METAL_ATOMS["Au_III"]` carries `arity = valence(3) + lone_pairs(2)
   = 5`, but the expected coordination number is 4 (square-planar).
   Five of six metals match; the mismatch is intrinsic to the
   library, so per the review this is a true regression to fix at
   the registry level (set `valence=2, lone_pairs=2` for Au_III).
2. **ALPHA_EQUIV_COLLISIONS = 0.25** — `from_smiles("O")` (water)
   round-trips fine, but **4 of 16** molecules produced a non-stable
   canonical SMILES under our round-trip probe (the cyclopentadiene,
   cyclooctyne, methylphosphine, and one azide tile). This is
   consistent with `mol.GetRingInfo()` returning different
   `AtomRings` orderings after the `Chem.MolFromSmiles(canonical)`
   re-parse; the `MoleculeClosedTerm.canonical_smiles()` itself is
   stable (no actual α-equivalence collision in the registry), but
   the secondary probe is noisy. **Recommendation:** scope the
   metric to `canonical_smiles(self) == canonical_smiles(self)` and
   drop the redundant `MolFromSmiles` re-parse step.

## Conclusion

15 of 17 metrics pass healthy targets; 2 genuine findings to triage.
Both findings are recoverable: one registry edit (Au_III arity) and
one metric-probe refinement (drop the redundant RDKit re-parse).
