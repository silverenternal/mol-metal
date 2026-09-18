# WF-Lambda-1b Audit — Two Structural Defects from WF-Lambda-1 Verify

> Honest-framing: MEASURED = direct grep + grep + Python probe results
> in this session.  PROJECTED = claims drawn from the user-supplied task
> spec that contradict what the code actually does.  Every claim below
> cites a `file:line` so the reader can re-verify.

---

## 1. `MoleculeClosedTerm.from_smiles` round-trip defect — REJECTION NOT IN `from_smiles` ITSELF

**Projected (from spec):** the lambda-term constructor rejects large
polyfunctional SMILES; the smallest patch is to relax a rejection in
`MoleculeClosedTerm.from_smiles`.

**Measured:** there is no rejection inside `from_smiles` for a 30-atom
RDKit-valid SMILES. The constructor
(`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:674`)
is `MolFromSmiles -> optionally AddHs/ETKDGv3/RemoveHs -> from_rdkit`.
The only hard rejection is `ValueError(f"RDKit failed to parse
SMILES: ...")`. A live probe of seven 11-48-atom SMILES (organic
amides, nicotine stereoisomer, cisplatin `[NH3][Pt]([NH3])(Cl)Cl`,
disconnected `N.N.Cl.Cl.[Pt]`, oxaliplatin `O=C=O[Pt]([NH3])([NH3])O.C1OC(=O)C(=O)OC1`,
and a 32-atom fluorinated hydrazide) round-trips in every case: the
constructed `MoleculeClosedTerm` reports `is_closed=True` and
`canonical_smiles()` returns a valid RDKit SMILES. The actual defect
sits **one layer deeper** at `check_beta_normal_form`
(`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/well_formedness.py:230`)
which uses `term.ledger.free_sites(a) != 0`. Because `Atom.arity =
valence + lone_pairs` (`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/atoms/combinators.py:12`),
N (valence 3, lone_pairs 1), O (2,2), S (2,2), F (1,3), Cl (1,3) keep
2-3 free sites forever after RDKit-bond construction; aromatic ring C
atoms also keep `valence_used < arity` because RDKit reports each
aromatic edge as order 1 but the combinator geometry records the
ideal-sp3 arity. Live probe on `CC(=O)Oc1ccc(cc1)C(=O)NC2CCC(CC2)NC(=O)c3ccccc3`
(28 atoms) shows `is_closed=True` but `check_beta_normal_form=False`
because 20 of 28 atoms have `free_sites ∈ {1,2}`. The `wf_lambda1_N10_3seed_v2`
aggregate (`molmetal/reports/wf_lambda1_N10_3seed_v2/report.json`)
shows the symptom: `validity_rate=0.90` (RDKit round-trip works) but
`synthesizability_rate=0.00` (BNF check fails for every candidate).
The smallest patch is therefore NOT in `from_smiles`; it is to make
`check_beta_normal_form` treat lone-pair capacity as saturated for
covalent-bond terms parsed via `from_rdkit` (e.g. add an
`accept_lone_pair_saturation=True` kwarg consumed by `from_rdkit`, or
introduce a `check_beta_normal_form_for_rdkit` that requires only
`free_sites = lone_pairs` instead of `0`). Existing well-formedness
tests that need strict semantics stay on the default branch.

**file:line list (Round-trip & from_smiles):**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:674-707` — `from_smiles` constructor (no SMILES-size rejection).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:692-694` — only hard rejection: `MolFromSmiles` returned None.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:550-671` — `from_rdkit` builds atoms via `distinct(PRIMITIVE_ATOMS[sym])` so each atom carries the canonical `arity = valence + lone_pairs`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:580-592` — branch that creates an `Atom` from RDKit; `_strip_oxidation` (line 945) maps `Pt_II -> Pt` etc.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:642-660` — covalent/aromatic bond factory call (with `strict=False`) — no rejection.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:498-548` — `to_rdkit` + canonical SMILES path (round-trip succeeds for 30+ atom SMILES).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/atoms/combinators.py:164-174` — `PRIMITIVE_ATOMS` registry with `arity = valence + lone_pairs`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/well_formedness.py:230-245` — `check_beta_normal_form` (the **actual** rejector).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:278-301` — `synthesizability_via_lambda_paths` consults `check_beta_normal_form` (line 292).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:647-654` — per-cell `synthesizability_rate` aggregator.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lambda1_N10_3seed_v2/report.json` — MEASURED symptom: `synthesizability_rate: 0.0`, `validity_rate: 0.9`.

## 2. `MCTSProofSearch` root-hint injection — NO metal-seeded root in current API

**Measured:** `MCTSProofSearch.search(initial_state, max_depth=3)`
(`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py:2059-2099`)
takes the root only via the `initial_state: MoleculeClosedTerm`
parameter — there is no `root_smiles`, no `root_state` ctor arg, no
metal-aware constructor hint.  The harness
`run_one_cell(...)` (`/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:502-561`)
constructs the root as either `from_smiles(reference_smiles, embed_3d=False)`
(organic reference ligand from the docked `.sdf`) or falls back to
`from_smiles("Cl[Pt]Cl", embed_3d=False)` — a 3-atom Pt seed with no
NH3 / no full coordination sphere.  `MoleculeClosedTerm.from_smiles` on
`[NH3][Pt]([NH3])(Cl)Cl` (line 705 calls `from_rdkit` on RDKit's mol,
which sees Pt=78 with four covalent bonds to two N + two Cl; the
probe above shows `is_closed=True, n_atoms=5, n_bonds=4`) round-trips
fine.  The missing injection point is therefore in
`run_one_cell(...)`: a `--metal-seed` CLI flag (default `None`;
options `cisplatin | ru_arene | ir_cp_star`) that maps to a SMILES
table and **prepends** the metal-containing scaffold to the root so
that the MCTS expand step can fire Pt-ligand bond formation rules
(`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/cisplatin_builder.py`
and the 5 click rules) starting from a Pt centre.  The cisplatin seed
should NOT be a fully-bonded closed term (the MCTS has nothing to do)
but rather the unsaturated descriptor
`MoleculeClosedTerm.as_cisplatin_shell()` (line 867) which leaves Pt
with 4 free sites and NH3 / Cl as standalone ligands — the exact
witness `lambda_only_metrics.py` can score with
`metal_geometry_prior_bonus(enabled=True)`.

**file:line list (root injection):**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py:2059-2099` — `search(initial_state, max_depth)` — root only via the `initial_state` argument.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py:2095-2099` — root `_MCTSNode` build.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py:1848-1853` — `MCTSProofSearch` dataclass fields (no `root_smiles` / `root_state` ctor arg).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/molecules/closed_term.py:867-913` — `as_cisplatin_shell` — the canonical unsaturated Pt(II) descriptor for the metal-seeded root.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/cisplatin_builder.py` — cisplatin builder used by the click-rule expansion.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:547-561` — root construction in `run_one_cell`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:868-948` — `_build_argparser` — add `--metal-seed` here.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:688-730` — `run_sweep` — thread `metal_seed` into `run_one_cell`.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:502-512` — `run_one_cell` signature — add `metal_seed: Optional[str] = None`.

## 3. `MetalGeometryPrior.DEFAULT_METAL_GEOMETRY` — Pt=4 is registered

**Measured:** at
`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:364-374`:

```
DEFAULT_METAL_GEOMETRY: dict = {
    78: Geometry.SQUARE_PLANAR,      # Pt(II)
    46: Geometry.SQUARE_PLANAR,      # Pd(II)
    29: Geometry.TETRAHEDRAL,
    30: Geometry.TETRAHEDRAL,
    26: Geometry.OCTAHEDRAL,
    44: Geometry.OCTAHEDRAL,         # Ru(II)
    77: Geometry.OCTAHEDRAL,         # Ir(III)
    45: Geometry.OCTAHEDRAL,
    25: Geometry.TRIGONAL_BIPYRAMIDAL,
}
```

Pt (Z=78) is mapped to `Geometry.SQUARE_PLANAR`.  `GEOMETRY_COORDINATION_NUMBER[Geometry.SQUARE_PLANAR] = 4` (line 330).  So the registration is correct: a Pt centre with 4 dative bonds satisfies the prior.  The orthogonal problem is that the lambda-only `metal_geometry_prior_bonus` (line 203 of `r4_lambda_only_run.py`) hard-codes `pt_target=4` and only inspects `state.atoms` / `state.bonds` — independent of the Torch `MetalGeometryPrior` module above, which is only used by EGNN training.  The lambda-only path is therefore already structurally correct; it just receives zero candidates with Pt because the root never carries one (see Defect 2).

**file:line list (prior registration):**
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:329-334` — `GEOMETRY_COORDINATION_NUMBER` (Pt-square-planar = 4).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:342-355` — `GEOMETRY_IDEAL_ANGLES` (square-planar = (90°, 180°)).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:364-374` — `DEFAULT_METAL_GEOMETRY` (Pt=78 -> SQUARE_PLANAR).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/priors/metal_geometry.py:598-767` — `MetalGeometryPrior` Torch module (used only by EGNN training, not by the lambda-only path).
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:203-257` — lambda-only `metal_geometry_prior_bonus` (uses atomic-number + bond-count, independent of Torch prior).

---

## Patch plan (re-stating the spec for clarity)

**Patch 1** — `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/well_formedness.py`:
introduce `check_beta_normal_form_for_rdkit(term)` that returns True
when every atom's residual `free_sites == lone_pairs` (i.e. only
valence-derived sites must be saturated).  Update
`/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py:278-301`
(`synthesizability_via_lambda_paths`) to use it by default — single
line change.  Add test
`test_30_atom_smiles_passes_bnf` to
`/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`
on `CC(=O)Oc1ccc(cc1)C(=O)NC2CCC(CC2)NC(=O)c3ccccc3` (28 atoms, MEASURED
above to currently fail BNF, will pass after patch).

**Patch 2** — `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py`:
add `--metal-seed {none,cisplatin,ru_arene,ir_cp_star}` (default `none`),
thread through `run_sweep` -> `run_one_cell`, build the root via
`MoleculeClosedTerm.as_cisplatin_shell()` (cisplatin) or via
`MoleculeClosedTerm.from_smiles(<metal SMILES>, embed_3d=False)` for
`ru_arene` (e.g. `[Ru]([c]1ccccc1)([c]1ccccc1)(Cl)(Cl)([NH3])[NH3]`)
and `ir_cp_star` (e.g. `[Ir]([c]1cccc[c]1[C](C)C)([c]1ccccc1)(Cl)(Cl)([NH3])[NH3]`).
When `--metal-seed` is `none` the harness preserves today's behaviour
for backward compatibility.

**Re-run plan** — `uv run python molmetal/scripts/r4_lambda_only_run.py
--pockets 5 --seeds 42 0 1234 --metal-seed cisplatin --output-dir
wf_lambda1b_cisplatin_5x3`; report `synthesizability_rate` (should be
>0 once BNF fix lands) and `metal_compliance_rate` (should be >0 once
Pt-seeded candidates appear in the top-K).  Both numbers are
PROJECTED until the re-run lands.
