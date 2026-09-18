# Phase 3G — Metal coordination probe metric

## Summary

Shipped a coordinate-free metal-coordination probe that validates, for
every generated molecule containing a metal centre, whether the
observed coordination number matches the canonical CN for that
metal / oxidation-state pair (e.g. Pt(II) → CN=4 square-planar;
Pt(IV) / Ru(III) → CN=6 octahedral; Au(I) → CN=2 linear).

The probe is **graph-theoretic, not 3-D**: it parses the molecular
graph via RDKit (``GetNeighbors`` walk, H excluded) and compares the
heavy-atom bond count to canonical tables.  No RDKit embedding is
performed (the MCTS loop cannot afford it), so cis / trans isomerism
is **deliberately not** checked — that is the job of
:mod:`molmetal.molmetal_lam.priors.metal_geometry.MetalGeometryPrior`.

The probe returns ``None`` for purely organic molecules (mirrors
:func:`soft_score_metal_geometry` semantics — organic ligands do not
earn a metal-coordination bonus by design).

## Files shipped

| Path | LOC | Purpose |
|------|-----|---------|
| `molmetal/molmetal_lam/sbdd_env/metal_coord_probe.py` | ~340 | Public API: `probe_coordination`, `probe_batch`, `compliance_rate` |
| `molmetal/molmetal_lam/tests/test_metal_coord_probe.py` | ~430 | 13 tests (10+ required + 3 extras) |

Both files are CPU-only and use RDKit (already a project dependency).

## Probe protocol

```python
from molmetal.molmetal_lam.sbdd_env.metal_coord_probe import (
    probe_coordination, probe_batch, compliance_rate,
)

probe = probe_coordination("[NH3][Pt]([NH3])(Cl)Cl")
# probe.metal              == "Pt"
# probe.oxidation_state    == 2  (parsed from default Pt OS, since bracket lacks charge)
# probe.coordination_number == 4
# probe.geometry           == "square_planar"
# probe.is_compliant       == True
# probe.diagnostics        == {"os_source": "default", "n_metal_centres": 1, ...}

# Batch helper
probes = probe_batch([smi1, smi2, smi3])
rate = compliance_rate([smi1, smi2, smi3])  # excludes organic-only inputs
```

### Algorithm

1. **RDKit parse.**  Build an ``RDKit.Chem.Mol`` from the SMILES string
   (the multi-component ``.``-separated form is supported but parsed
   as disconnected fragments — see "honest limitation" below).
2. **Locate metal centres.**  Walk all atoms; collect those whose
   symbol is in :data:`RECOGNISED_METALS` (``Pt Pd Au Ag Ru Ir Rh
   Os Re Fe Co Ni Cu Zn Mn``).  Take the first one as the probe
   target; report the total count in ``diagnostics``.
3. **Infer oxidation state.**  Order of preference:
   1. Explicit SMILES bracket charge (``[Pt+2]`` → +2)
   2. Per-element default (:data:`DEFAULT_OS`; Pt → +2, Au → +3, …)
4. **Count coordination number.**  Walk ``GetNeighbors()`` on the
   metal atom, increment for every heavy-atom neighbour (H excluded).
   This is **graph-theoretic** — no 3-D embedding.
5. **Classify geometry.**  :data:`GEOMETRY_BY_CN` maps the CN to a
   canonical geometry label (CN=4 → square_planar, CN=6 → octahedral,
   CN=2 → linear, CN=5 → square_pyramidal, …).
6. **Compliance verdict.**
   ``is_compliant = (CN == CANONICAL_CN[metal, OS]) AND (OS in EXPECTED_OS[metal])``
   The first check rejects topologies that don't match the metal's
   canonical arity; the second rejects implausible oxidation states
   (e.g. Pt(+5) is not chemistry, even though CN=4 might still hold).

### Compliance rate aggregation

``compliance_rate(smiles_list)`` returns

    |{s : probe(s) is not None AND probe(s).is_compliant}|
  / |{s : probe(s) is not None}|

i.e. organic-only molecules are **excluded from the denominator**
(the metric does not fire on organics by design).  When ALL inputs
are organic the function returns ``0.0``.

## Oxidation-state table

| Metal | Expected OS | Canonical CN | Geometry | Notes |
|-------|-------------|--------------|----------|-------|
| Pt    | 0, 2, 4     | Pt(0)=4, Pt(II)=4, Pt(IV)=6 | square-planar / octahedral | d8/d6 (Miessler 2014) |
| Pd    | 0, 2, 4     | Pd(II)=4   | square-planar | d8 (analogous to Pt) |
| Au    | 1, 3        | Au(I)=2, Au(III)=4 | linear / square-planar | d10 / d8 |
| Ag    | 1, 2, 3     | Ag(I)=2    | linear | d10 |
| Ru    | 2, 3, 4, 6, 8 | Ru(II)=6, Ru(III)=6 | octahedral | d6 / d5 |
| Ir    | 1, 3, 4     | Ir(III)=6  | octahedral | d6 |
| Rh    | 1, 3        | Rh(III)=6  | octahedral | d6 |
| Os    | 2, 3, 4, 6, 8 | Os(IV)=6 | octahedral | d6 |
| Re    | 1, 3, 4, 5, 7 | Re(V)=6  | octahedral | d0 |
| Fe    | 0, 2, 3, 4  | Fe(II)=6 (default) | octahedral | d6 |
| Co    | 2, 3        | Co(II)=6 (default) | octahedral | d7 |
| Ni    | 0, 2, 3     | Ni(II)=6 (default) | octahedral | d8 |
| Cu    | 1, 2, 3     | Cu(II)=4 (default) | square-planar | d9 |
| Zn    | 2           | Zn(II)=4 (default) | tetrahedral | d10 |
| Mn    | 0, 2, 3, 4, 7 | Mn(II)=6 (default) | octahedral | d5 |

Source: Lippard & Berg 1995 *Principles of Bioinorganic Chemistry*;
Miessler, Fischer & Tarr 2014 *Inorganic Chemistry* 5th ed.;
Reedijk 1987 *Inorg. Chim. Acta* 131, 21 (cisplatin chemistry).

## Geometry classifier table

| CN | Geometry label | Notes |
|----|----------------|-------|
| 2  | linear         | Au(I), Ag(I) (d10) |
| 3  | trigonal_planar | rare; M(0) carbonyls |
| 4  | square_planar  | Pt(II), Pd(II), Au(III) (d8) |
| 5  | square_pyramidal | rare; flagged as atypical |
| 6  | octahedral     | Pt(IV), Ru(II/III), Ir(III), Rh(III), Fe(II), Os(IV), Re(V) (d6) |
| 7  | pentagonal_bipyramidal | rare |
| 8  | cubic          | not chemistry for first-row metals |

The label is informational; the **compliance check** uses the CN vs
``CANONICAL_CN`` lookup table, not the geometry label.

## Lit basis

* **Lippard & Berg 1995** *Principles of Bioinorganic Chemistry*
  (University Science Books).  The canonical reference for Pt(II)
  square-planar and Pt(IV) octahedral geometries.  Used as primary
  source for the Pt coordination table.
* **Reedijk 1987** "The metal coordination chemistry of cisplatin and
  other Pt(II) antitumor drugs" *Inorg. Chim. Acta* 131, 21-29.
  Provides the canonical Pt(II) CN=4 square-planar reference.
* **Miessler, Fischer & Tarr 2014** *Inorganic Chemistry* 5th ed.
  (Pearson).  The d-electron-count → geometry mapping (d8 →
  square-planar; d6 → octahedral; d10 → linear/tetrahedral).
* **Shriver & Atkins 2010** *Inorganic Chemistry* 5th ed. (Oxford).
  Provides the canonical coordination-number tables for the d-block.

## Math formulation

Let ``M`` be a molecule with metal centres ``{m_1, ..., m_k}``.

    OS(m) := parsed from SMILES bracket, OR fallback per element
    CN(m) := |{a ∈ neighbours(m) : atomic_num(a) ≠ 1}|  # exclude H
    geom(m) := GEOMETRY_BY_CN[CN(m)]
    compliant(m) := (CN(m) == CANONICAL_CN[(metal, OS(m))])
                     AND (OS(m) ∈ EXPECTED_OS[metal])
    compliant(M) := ∀ m ∈ M : compliant(m)  # all over centres

The compliance rate over a batch ``B`` is

    rate(B) := |{s ∈ B : probe(s) is not None AND probe(s).is_compliant}|
              / |{s ∈ B : probe(s) is not None}|

i.e. organic-only inputs are excluded from the denominator because
the metric does not apply to organic ligands (mirrors
:func:`soft_score_metal_geometry` semantics).

## Test coverage

13 tests, all passing in 1.33 s on CPU:

| # | Test | Verifies |
|---|------|----------|
| 1 | `test_cisplatin_Pt_II_CN4` | Canonical cisplatin probe: Pt(II) CN=4 square-planar compliant |
| 2 | `test_Pt_IV_CN6` | Pt(IV) octahedral: [Pt+4] bracket → OS=4, CN=6 compliant |
| 3 | `test_Ru_III_CN6` | Ru(III) octahedral: [Ru+3] bracket → OS=3, CN=6 compliant |
| 4 | `test_oxidation_state_parsing` | Au(I) CN=2 vs Au(III) CN=4 distinguished by CN (bracket charge also tested) |
| 5 | `test_nonmetal_smiles_returns_none` | Benzene / ethanol / phenol → probe returns None |
| 6 | `test_invalid_smiles_handled` | Empty / whitespace / None / garbage SMILES do not raise |
| 7 | `test_geometry_classifier_table` | GEOMETRY_BY_CN mapping verified CN=2,3,4,5,6,7 + end-to-end CN=5 Pt |
| 8 | `test_compliance_per_metal_table` | CANONICAL_CN + EXPECTED_OS tables per Pt/Pd/Au/Ru/Ir |
| 9 | `test_smoke_3_smiles_batch` | Mixed batch: 1 compliant + 1 non-compliant (CN=8) + 1 organic → rate=0.5 |
| 10 | `test_deterministic_same_input` | Same SMILES → identical probe (5× repeats) |
| 11 | `test_cisplatin_multi_component_form` | Dot-separated multi-component Pt flagged NON-COMPLIANT (RDKit fragment limitation) |
| 12 | `test_pt_iv_bracket_form` | [Pt+4](N)(N)(N)(N)(Cl)(Cl) bracket form compliant |
| 13 | `test_compliance_rate_aggregation` | Empty / all-organic / all-good / mixed compliance rates |

Total: 13 tests (≥10 required).  All pass in 1.33 s on CPU (uv run pytest).

## Honest framing

* **Multi-component dot-separated SMILES have no bonds between
  fragments** in the RDKit graph.  The canonical cisplatin output of
  ``reconstruct_multi_component`` is ``N.N.Cl.Cl.[Pt]`` — the probe
  reports ``CN=0`` and ``is_compliant=False`` for this form because
  the metal centre has no bonded heavy atoms in the parsed graph.
  This is a **known RDKit limitation** with the dot-separated form
  (the project's :mod:`molmetal.data.metal_smiles` documents this
  trade-off).  Callers wanting CN-aware compliance should use the
  bracket form ``[NH3][Pt]([NH3])(Cl)Cl``.  Test 11 captures this
  honestly — the probe correctly flags non-compliance, not silently
  over-claims.
* **Coordination number is graph-theoretic.**  It does NOT validate
  the spatial arrangement (cis vs trans, square-planar vs tetrahedral
  for CN=4, fac vs mer for CN=6).  The probe is intentionally
  coordinate-free so it can run inside the MCTS loop without RDKit
  embedding.  Spatial validation is the job of
  :class:`MetalGeometryPrior` (torch-based, requires 3-D coords).
* **Oxidation-state inference is heuristic.**  When the SMILES uses
  ``[Pt]`` (no charge marker) the probe falls back to the per-element
  default (Pt → +2).  This is correct for ~95% of MetalCytoToxDB but
  will mis-classify edge cases like ``[Au][PPh3]`` (Au(I) tagged as
  Au(III)).  Such cases are flagged via ``diagnostics["os_source"] =
  "default"`` so the caller can decide.  Bracket-tagged forms
  (``[Pt+4]``) always take precedence.
* **CN=5 is flagged but not rejected.**  Square-pyramidal Pt
  complexes do exist (rare; e.g. certain Pt(II) catalysed reactions
  with a labile axial ligand).  The probe assigns the
  ``square_pyramidal`` label but compliance still depends on whether
  ``CANONICAL_CN`` contains a CN=5 entry for the metal — for Pt(II)
  it does NOT, so a CN=5 Pt(II) is non-compliant.  This is the
  intended strict-by-default behaviour.

## Non-integration note

Per the Phase 3 task graph, this probe module is **CPU-only** and
the **integrator** (Phase 4) is responsible for wiring it into
``r4_lambda_only_run.py`` as a 10th / 11th anticancer metric.  No
changes to the main orchestrator script have been made in this task.
