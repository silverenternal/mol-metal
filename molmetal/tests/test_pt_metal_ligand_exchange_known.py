"""Known-Pt regression tests for MetalLigandExchange + AquaExchange SMARTS.

Background
----------
TODO-30 Rank-3 (P1.4 — Known-Pt regression tests).  This file is pure
INSURANCE — every test should pass on the existing SMARTS library as
shipped in ``molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py``.
The intent is to lock down the canonical Pt(II) / Pt(IV) coordination
chemistry rules so that future edits to the SMARTS library do not
silently break the well-known reactions:

    cisplatin + NH3         -> [Pt(NH3)3Cl]+  + Cl-   (mono-ammine)
    cisplatin + 2 NH3       -> [Pt(NH3)4]2+    + 2 Cl- (diammine)
    carboplatin + NH3       -> ring-opened product + free cyclobutane-1,1-dicarboxylate
    cisplatin + H2O         -> [Pt(NH3)2Cl(OH2)]+ + Cl-  (first aquation)
    aqua-Pt-Cl + H2O        -> [Pt(NH3)2(OH2)2]2+ + Cl-   (second aquation)

Plus a Bemis-Murcko scaffold comparison of cisplatin vs satraplatin that
reports the pairwise Tanimoto of their Morgan fingerprints.

Honest framing
--------------
* The tests do NOT fire the SMARTS end-to-end (RDKit cannot sanitise
  most Pt-containing products; see the ``_L4_COUNTERS`` instrumentation
  + the explicit honest-framing note in
  ``pt_metal_ligand_exchange.py:1181-1190``).
* Instead we exercise the public ``available_smarts()`` accessor
  + ``_rdkit_reaction_template`` machinery, verify the SMARTS syntax
  is parseable, and assert the canonical SMARTS contain the canonical
  donor/acceptor atom-maps for the relevant reaction.
* The scaffold novelty test uses RDKit's ``MurckoScaffold`` directly
  on the canonical SMILES from ``molmetal/data/known_pt_drugs.csv``
  (cisplatin + satraplatin are both acyclic Pt complexes, so their
  Bemis-Murcko scaffolds are both the empty string; we therefore use
  Morgan-fingerprint Tanimoto as the comparison metric, which is
  meaningful here).

Lit anchors
-----------
* Lippard 1995 — Pt(II) square-planar coordination chemistry.
* Reedijk 1987 — Pt(II) aquation kinetics (first / second aquation).
* Comba-Hambley 2009 — d8 analogue extension to Pd / Au.
"""

from __future__ import annotations

from typing import List

import pytest

# All tests in this file are CPU-only (RDKit + atom-map inspection only).


# ---------------------------------------------------------------------------
# Canonical Pt drug SMILES — sourced from
# molmetal/data/known_pt_drugs.csv.  Kept here as Python literals so the
# tests are self-contained (no CSV dependency at import time).
# ---------------------------------------------------------------------------
CISPLATIN_SMILES = "N.N.[Cl][Pt][Cl]"
CARBOPLATIN_SMILES = "N.N.[Pt]12(OC(=O)CCC1)OC(=O)CCC2"
OXALIPLATIN_SMILES = (
    "N.N.O[Pt]([O-])([NH3])([NH3])OC(=O)[C@@H](N)CCC(=O)O"
)
SATRAPLATIN_SMILES = (
    "CC1=C(C(=O)O[Pt]([NH3])([NH3])(OC(=O)c2ccccc2)"
    "C(C)(C)C)C(C)(C)C=C1"
)
NEDAPLATIN_SMILES = "N.N.OCC(=O)O[Pt]([NH3])([NH3])OCC(=O)O"


# ---------------------------------------------------------------------------
# Helpers — SMARTS parsing + reaction-template instantiation
# ---------------------------------------------------------------------------
def _load_metal_ligand_patterns():
    """Load the 5 MetalLigandExchange SMARTS patterns as (name, smarts) tuples."""
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        METAL_LIGAND_EXCHANGE_SMARTS,
    )
    return [(n, s) for (n, s, _d) in METAL_LIGAND_EXCHANGE_SMARTS]


def _load_aqua_patterns():
    """Load the 2 AquaExchange SMARTS patterns as (name, smarts) tuples."""
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        AQUA_EXCHANGE_SMARTS,
    )
    return [(n, s) for (n, s, _d) in AQUA_EXCHANGE_SMARTS]


def _reaction_template(smarts: str):
    """Return an RDKit reaction template from a SMARTS reaction string.

    We do NOT call RunReactants — the SMARTS may target Pt_II which
    RDKit cannot sanitise.  This helper only verifies that RDKit can
    PARSE the SMARTS into a reaction template, which is a syntactic
    check (not a chemistry check).
    """
    from rdkit import Chem
    # ReactionFromSmarts lives in rdkit.Chem.AllChem (not Chem).
    try:
        from rdkit.Chem import AllChem
        rxn = AllChem.ReactionFromSmarts(smarts)
    except ImportError:
        # Fallback for older RDKit versions where AllChem may not have
        # ReactionFromSmarts — use the rdChemDraws.Reaction module as a
        # last resort.
        rxn = Chem.ReactionFromSmarts(smarts)  # type: ignore[attr-defined]
    assert rxn is not None, f"RDKit failed to parse SMARTS: {smarts}"
    return rxn


# ---------------------------------------------------------------------------
# Tests — cisplatin + NH3 ammine substitutions (canonical activation step)
# ---------------------------------------------------------------------------
def test_cisplatin_mono_ammine():
    """Cisplatin + NH3 -> mono-ammine cation (one Cl replaced by NH3).

    Locks down the canonical Pt_Cl_NH3 SMARTS pattern:

        [Pt:1]([Cl:2])[*:3].[NH3:4] >> [Pt:1]([NH3:4])[*:3].[Cl:2]

    Per Lippard 1995 §4.2: the rate-limiting step of cisplatin
    activation is associative substitution of one chloride by water
    (aquation).  Subsequent replacement by an ammine ligand (here the
    second NH3 of the cis-diammine coordination sphere) follows the
    same SMARTS shape but with [NH3] as the donor instead of [OH2].
    """
    patterns = dict(_load_metal_ligand_patterns())
    # Sanity: the canonical Pt_Cl_NH3 pattern must be present.
    assert "Pt_Cl_NH3" in patterns, (
        f"canonical Pt_Cl_NH3 pattern missing; got {list(patterns)}"
    )
    smarts = patterns["Pt_Cl_NH3"]
    # Atom-map invariants:
    #   :1 = Pt centre (kept in product)
    #   :2 = Cl leaving group (released as separate term)
    #   :3 = other Pt ligand (kept on Pt, spectator)
    #   :4 = incoming NH3 donor (kept in product on Pt)
    assert "[Pt:1]" in smarts, "missing Pt atom-map (:1)"
    assert "[Cl:2]" in smarts, "missing Cl leaving atom-map (:2)"
    assert "[NH3:4]" in smarts, "missing NH3 donor atom-map (:4)"
    assert ">>" in smarts, "missing reaction separator"
    # Product must keep the Pt-N bond (Pt + NH3 with their atom-maps
    # carried over) and release Cl.
    product_side = smarts.split(">>", 1)[1]
    assert "[Pt:1]" in product_side, "Pt atom-map lost in product"
    assert "[Cl:2]" in product_side, "Cl leaving group not in product term"
    assert "[NH3:4]" in product_side, "NH3 donor not in product"
    # Syntactic check — RDKit can parse the SMARTS into a reaction.
    rxn = _reaction_template(smarts)
    assert rxn.GetNumReactantTemplates() == 2, "should be 2 reactant templates"
    assert rxn.GetNumProductTemplates() == 2, "should be 2 product templates"


def test_cisplatin_di_ammine():
    """Cisplatin + 2 NH3 -> [Pt(NH3)4]2+ (diammine cation, no Cl).

    The di-ammine end-state is reached by firing the canonical
    Pt_Cl_NH3 pattern TWICE — once for each Cl.  This test pins the
    invariant that the SAME pattern shape applies to each Cl leaving
    step (no per-Cl variant pattern is required).  We verify this by
    counting the Cl occurrences on each side of the reaction arrow
    and the NH3 occurrences on each side:

      Reactant side:  [Pt:1]([Cl:2])[*:3].[NH3:4]
        - ONE  [Cl:2] (bound Cl, the leaving group)
        - ONE  [NH3:4] (the incoming donor, separate reactant term)

      Product side:   [Pt:1]([NH3:4])[*:3].[Cl:2]
        - ONE  [Cl:2] (free Cl- leaving group, separate product term)
        - ONE  [NH3:4] (NH3 now bonded to Pt)

    One fire per Cl, two fires for the diammine end-state — the
    pattern is reusable and contains no per-Cl variant.
    """
    patterns = dict(_load_metal_ligand_patterns())
    smarts = patterns["Pt_Cl_NH3"]
    reactant_side, product_side = smarts.split(">>", 1)
    # The pattern matches a SINGLE Pt-Cl bond (one leaving group per
    # fire).  This is the MLC-layer design choice: the MCTS calls the
    # rule twice for the di-ammine end-state, not once with a "2 Cl"
    # pattern.  Verify ONE [Cl:2] on the reactant side (the bound Cl).
    assert reactant_side.count("[Cl:2]") == 1, (
        f"Pt_Cl_NH3 reactant side should have ONE Pt-Cl per fire; "
        f"got {reactant_side.count('[Cl:2]')}"
    )
    # The incoming NH3 donor appears ONCE on the reactant side (as a
    # separate reactant term, denoted by the '.' separator).
    assert reactant_side.count("[NH3:4]") == 1, (
        f"Pt_Cl_NH3 reactant side should have ONE NH3 donor; "
        f"got {reactant_side.count('[NH3:4]')}"
    )
    # The product side must carry the NH3 atom-map over to Pt (NH3
    # now bonded) AND release the Cl as a separate product term.
    assert product_side.count("[NH3:4]") == 1, (
        f"Pt_Cl_NH3 product side should have ONE NH3 bonded to Pt; "
        f"got {product_side.count('[NH3:4]')}"
    )
    assert product_side.count("[Cl:2]") == 1, (
        f"Pt_Cl_NH3 product side should release ONE Cl- leaving group; "
        f"got {product_side.count('[Cl:2]')}"
    )
    # The pattern is symmetric in its interpretation: firing it on
    # [Pt(NH3)2Cl]+ (post-first-fire) must still match (the spectator
    # Pt ligand [*:3] can be anything including an NH3 group).  We
    # verify this by checking that the [Pt]([Cl])[*] wildcard accepts
    # an arbitrary ligand at position [*:3].
    assert "[*:3]" in smarts, "Pt pattern missing spectator ligand wildcard"
    # RDKit parseability of the symmetric pattern.
    rxn = _reaction_template(smarts)
    assert rxn.GetNumReactantTemplates() == 2


# ---------------------------------------------------------------------------
# Carboplatin — cyclobutane-1,1-dicarboxylate (CBDCA) ring opening
# ---------------------------------------------------------------------------
def test_carboplatin_cbdca_ring_opening():
    """Carboplatin + NH3 -> ring-opened monodentate + free carboxylate.

    Carboplatin differs from cisplatin by the CBDCA chelate: a
    cyclobutane-1,1-dicarboxylate bidentate ligand that opens
    transiently during aquation (the carboxylate arm dissociates, then
    water enters).  The canonical MetalLigandExchange SMARTS targets
    Pt-Cl substitution, NOT Pt-O (carboxylate) substitution — so this
    test is intentionally NARROWER:

      We assert that the carboplatin SMILES parses with a Pt centre
      bonded to at least one carboxylate oxygen AND that the carboplatin
      Pt atom has coordination number >= 4 (typical CBDCA chelate + 2
      NH3 = CN 4).  This pins the substrate geometry on which a
      future F2(a)-style ring-opening rule would operate.

    Honest framing: the current 5-pattern SMARTS library does NOT
    cover Pt-O(carboxylate) substitution explicitly.  This test is a
    SPEC-level invariant for any future CBDCA ring-opening rule, not
    a check that the existing rule fires on carboplatin.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(CARBOPLATIN_SMILES)
    assert mol is not None, f"failed to parse carboplatin SMILES: {CARBOPLATIN_SMILES}"
    # Pt atom index + degree.
    pt_idx = None
    pt_degree = 0
    pt_bonded_O_count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Pt":
            pt_idx = atom.GetIdx()
            pt_degree = atom.GetDegree()
            for neighbour in atom.GetNeighbors():
                if neighbour.GetSymbol() == "O":
                    pt_bonded_O_count += 1
            break
    assert pt_idx is not None, "carboplatin must contain Pt"
    # CN == 4 is the canonical Pt_II target.  The CBDCA chelate
    # provides 2 O donors + 2 NH3 = CN 4.
    assert pt_degree == 4, (
        f"carboplatin Pt should have CN=4 (CBDCA + 2 NH3); got {pt_degree}"
    )
    # CBDCA provides at least 1 (and at most 2) O donors directly on Pt.
    assert pt_bonded_O_count >= 1, (
        f"carboplatin Pt should have >=1 Pt-O bond (CBDCA); got {pt_bonded_O_count}"
    )
    # The carboplatin SMILES contains the cyclobutane-1,1-dicarboxylate
    # motif (C1-C-C-C1 with two -C(=O)O- branches on the quaternary C).
    # We assert that two carboxylate carbons are present (one per
    # -COO- arm).
    n_carboxylate_C = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        oxy_neighbours = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if len(oxy_neighbours) >= 2:
            n_carboxylate_C += 1
    assert n_carboxylate_C >= 2, (
        f"carboplatin must have >=2 carboxylate carbons (CBDCA); "
        f"got {n_carboxylate_C}"
    )


# ---------------------------------------------------------------------------
# Aquation kinetics — first + second aquation step (Reedijk 1987)
# ---------------------------------------------------------------------------
def test_pt_aqua_first_aquation():
    """Cisplatin + H2O -> [Pt(NH3)2Cl(OH2)]+ + Cl- (first aquation).

    The Pt_Cl_H2O_first pattern is the canonical first-aquation SMARTS:

        [Pt:1]([Cl:2])[*:3].[OH2:4] >> [Pt:1]([OH:4])[*:3].[Cl:2]

    Per Reedijk 1987: first aquation pKa ~ 6.5, t_1/2 ~ 2 h at 37 °C
    in 4 mM Cl-; the entering water loses one proton to become an
    [OH] hydroxyl in the product term (the [OH] is charge-0 because
    Pt_II is divalent — full formal-charge bookkeeping is left to the
    caller, see the honest-framing note in
    ``pt_metal_ligand_exchange.py:1184-1190``).
    """
    patterns = dict(_load_aqua_patterns())
    assert "Pt_Cl_H2O_first" in patterns, (
        f"canonical Pt_Cl_H2O_first pattern missing; got {list(patterns)}"
    )
    smarts = patterns["Pt_Cl_H2O_first"]
    # Atom-map invariants:
    #   :1 = Pt centre
    #   :2 = Cl leaving group
    #   :3 = spectator ligand
    #   :4 = water donor (becomes [OH] hydroxyl after proton loss)
    assert "[Pt:1]" in smarts, "missing Pt atom-map (:1)"
    assert "[Cl:2]" in smarts, "missing Cl leaving atom-map (:2)"
    assert "[OH2:4]" in smarts, "missing water donor atom-map (:4)"
    assert ">>" in smarts, "missing reaction separator"
    # Product side: water must lose one proton → [OH].
    product_side = smarts.split(">>", 1)[1]
    assert "[OH:4]" in product_side, (
        "product should show [OH:4] hydroxyl (water lost 1 proton)"
    )
    assert "[OH2:4]" not in product_side, (
        "product should NOT show [OH2:4] (proton already lost)"
    )
    # AquaContext must propagate pKa1 ~ 6.5.
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        get_aqua_context,
    )
    ctx = get_aqua_context("Pt_Cl_H2O_first")
    assert abs(ctx.pka1 - 6.5) < 1e-6, (
        f"first aquation pKa1 must be 6.5 per Reedijk 1987; got {ctx.pka1}"
    )
    assert ctx.ionic_strength_M == pytest.approx(0.10, abs=1e-6)
    assert ctx.temperature_K == pytest.approx(310.0, abs=1e-6)
    # RDKit parseability.
    rxn = _reaction_template(smarts)
    assert rxn.GetNumReactantTemplates() == 2


def test_pt_aqua_second_aquation():
    """Aqua-Pt-Cl + H2O -> [Pt(NH3)2(OH2)2]2+ + Cl- (second aquation).

    The Pt_OHCl_H2O_second pattern handles the second aquation step
    (the diaqua complex that binds DNA-N7-guanine, the actual cytotoxic
    event):

        [Pt:1]([OH:2])([Cl:3])[*:4].[OH2:5] >>
        [Pt:1]([OH:2])([OH:5])[*:4].[Cl:3]

    The pattern requires BOTH a hydroxyl and a chloride still bound to
    Pt (post-first-aquation state) — this is what distinguishes it
    from the first-aquation pattern.
    """
    patterns = dict(_load_aqua_patterns())
    assert "Pt_OHCl_H2O_second" in patterns, (
        f"second-aquation pattern missing; got {list(patterns)}"
    )
    smarts = patterns["Pt_OHCl_H2O_second"]
    # The second-aquation pattern MUST distinguish itself from the
    # first by requiring a Pt-OH bond (not just Pt-Cl).  We assert
    # this distinguishing feature is present.
    assert "[OH:2]" in smarts, (
        "second-aquation pattern must require a Pt-OH bond (post first aquation)"
    )
    assert "[Cl:3]" in smarts, (
        "second-aquation pattern must require a Pt-Cl bond "
        "(only one Cl should remain to be displaced)"
    )
    assert "[OH2:5]" in smarts, "missing water donor atom-map (:5)"
    assert ">>" in smarts, "missing reaction separator"
    # Product side: the second aquation converts Pt-OH(Cl) to
    # Pt-(OH)2 — both ligands become hydroxyl (the original OH
    # spectator + the incoming H2O → OH after proton loss).
    product_side = smarts.split(">>", 1)[1]
    assert "[OH:2]" in product_side, "spectator OH must remain in product"
    assert "[OH:5]" in product_side, "incoming water must become [OH:5]"
    assert "[Cl:3]" in product_side, "leaving Cl must appear in product term"
    # AquaContext for the second aquation has pKa1 > first-aquation
    # pKa (per Reedijk 1987 the second Cl is more labile).
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        get_aqua_context,
    )
    ctx_first = get_aqua_context("Pt_Cl_H2O_first")
    ctx_second = get_aqua_context("Pt_OHCl_H2O_second")
    assert ctx_second.pka1 > ctx_first.pka1, (
        f"second aquation pKa should be > first; got "
        f"{ctx_second.pka1} <= {ctx_first.pka1}"
    )
    # Notes must mention the diaqua complex (the species that actually
    # binds DNA).
    assert "diaqua" in ctx_second.notes or "DNA" in ctx_second.notes, (
        f"second-aquation notes should mention diaqua/DNA binding; "
        f"got: {ctx_second.notes!r}"
    )
    # RDKit parseability.
    rxn = _reaction_template(smarts)
    assert rxn.GetNumReactantTemplates() == 2
    # The product template (Mol object) must show TWO hydroxyls
    # (the original + new).  We use Chem.MolToSmarts to serialise the
    # product template to a SMARTS string for substring counting.
    # NOTE: RDKit's MolToSmarts converts the SMARTS atom-spec ``[OH]``
    # to ``[O&H1]`` (an oxygen with exactly one hydrogen) — this is
    # the canonical expanded form of the hydroxyl shorthand.  We
    # therefore count ``O&H1`` occurrences (the atom-spec fragment) in
    # the serialised SMARTS.
    from rdkit import Chem
    product_template = rxn.GetProductTemplate(0)
    product_smarts = Chem.MolToSmarts(product_template)
    # Count the O&H1 hydroxyl atoms (one per hydroxyl).
    oh_count = product_smarts.count("O&H1")
    assert oh_count >= 2, (
        f"product template should show 2 OH groups; got {oh_count} in {product_smarts!r}"
    )


# ---------------------------------------------------------------------------
# Scaffold novelty — cisplatin vs satraplain via Bemis-Murcko + Morgan FP
# ---------------------------------------------------------------------------
def test_scaffold_novelty_cisplatin_satraplatin():
    """Bemis-Murcko scaffold + Morgan FP Tanimoto of cisplatin vs satraplatin.

    Honest framing
    --------------
    Cisplatin (N.N.[Cl][Pt][Cl]) is an *acyclic* Pt complex with zero
    Bemis-Murcko rings -> the canonical scaffold_smiles returns the
    empty string.

    Satraplatin (CC1=C(C(=O)O[Pt]...)C(C)(C)C=C1) DOES contain aromatic
    rings (the cyclohexadiene + benzoate arms), so its Bemis-Murcko
    scaffold is non-empty.  This is the EXPECTED outcome — satraplatin
    is structurally a far more elaborate scaffold than cisplatin.

    We therefore report:

      1. Cisplatin scaffold is empty (acyclic Pt complex).
      2. Satraplatin scaffold is NON-empty (contains aromatic rings).
      3. The two scaffolds are different strings.
      4. The Morgan-fingerprint Tanimoto similarity is well below 0.4
         (cisplatin and satraplatin are chemically distinct).

    The empty-scaffold agreement on cisplatin is the basis for the
    ``scaffold_in_known_pt_drugs`` function's "Pt-required" disambiguation
    in :mod:`molmetal_lam.lam_chem.data_diversity` (acyclic alkanes
    would also have empty scaffolds but lack Pt).
    """
    from molmetal.validation.training_novelty import scaffold_smiles

    # Sanity: parse the SMILES via RDKit first (the Bemis-Murcko
    # helper does this internally; we want to fail fast with a clear
    # message if the parse breaks).
    from rdkit import Chem
    for name, smi in [
        ("cisplatin", CISPLATIN_SMILES),
        ("satraplatin", SATRAPLATIN_SMILES),
    ]:
        m = Chem.MolFromSmiles(smi)
        assert m is not None, f"failed to parse {name} SMILES: {smi}"

    s_cis = scaffold_smiles(CISPLATIN_SMILES)
    s_sat = scaffold_smiles(SATRAPLATIN_SMILES)
    # Cisplatin is acyclic -> empty Bemis-Murcko scaffold.
    assert s_cis == "", (
        f"cisplatin should have empty Bemis-Murcko scaffold "
        f"(acyclic Pt complex); got {s_cis!r}"
    )
    # Satraplatin contains aromatic rings -> non-empty Bemis-Murcko
    # scaffold.  We pin the *non-empty* property (the exact scaffold
    # SMILES depends on RDKit's canonicalisation, which can change
    # between versions, so we do NOT pin the exact string).
    assert s_sat != "", (
        f"satraplatin should have non-empty Bemis-Murcko scaffold "
        f"(contains aromatic rings); got {s_sat!r}"
    )
    # Cisplatin and satraplatin have distinct scaffolds.
    assert s_cis != s_sat, (
        f"cisplatin and satraplatin must have distinct scaffolds; "
        f"got cisplatin={s_cis!r} satraplatin={s_sat!r}"
    )

    # Morgan-fingerprint Tanimoto similarity (radius=2, 2048 bits) —
    # the meaningful "scaffold novelty" proxy for these structurally
    # distinct Pt complexes.
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    fp_cis = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(CISPLATIN_SMILES))
    fp_sat = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(SATRAPLATIN_SMILES))
    inter = int(((fp_cis > 0) & (fp_sat > 0)).sum())
    union = int(((fp_cis > 0) | (fp_sat > 0)).sum())
    assert union > 0, "union of bitsets must be non-empty"
    tanimoto = inter / union
    # Cisplatin (2 NH3, 2 Cl) and satraplatin (cyclohexadiene ligand +
    # benzoate + tert-butyl acetate) are chemically distinct.  We
    # expect Tanimoto < 0.4 — well below the 0.4 MODERATE-risk
    # threshold used in ``metric_max_sim_known_pt_drugs`` (so the
    # generated molecule would NOT trip the MODERATE-risk flag).
    assert tanimoto < 0.4, (
        f"cisplatin vs satraplatin Tanimoto should be < 0.4 "
        f"(structurally distinct); got {tanimoto:.3f}"
    )

    # And for completeness: cisplatin vs itself must be Tanimoto == 1.0.
    inter_id = int(((fp_cis > 0) & (fp_cis > 0)).sum())
    union_id = int(((fp_cis > 0) | (fp_cis > 0)).sum())
    tanimoto_id = inter_id / union_id if union_id > 0 else 0.0
    assert tanimoto_id == pytest.approx(1.0, abs=1e-6), (
        f"self-similarity must be 1.0; got {tanimoto_id}"
    )


# ---------------------------------------------------------------------------
# Bonus regression tests — pattern library integrity
# ---------------------------------------------------------------------------
def test_pattern_library_length_invariants():
    """5 MetalLigandExchange + 2 AquaExchange patterns — the F2(a) cardinality."""
    from molmetal.molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
        METAL_LIGAND_EXCHANGE_SMARTS,
        AQUA_EXCHANGE_SMARTS,
    )
    assert len(METAL_LIGAND_EXCHANGE_SMARTS) == 5, (
        "MetalLigandExchange SMARTS library must be 5 patterns (F2(a) cardinality)"
    )
    assert len(AQUA_EXCHANGE_SMARTS) == 2, (
        "AquaExchange SMARTS library must be 2 patterns (F2(a) cardinality)"
    )
    # Pattern-name uniqueness within each library (no duplicates).
    metal_names = [n for (n, _s, _d) in METAL_LIGAND_EXCHANGE_SMARTS]
    aqua_names = [n for (n, _s, _d) in AQUA_EXCHANGE_SMARTS]
    assert len(set(metal_names)) == 5, (
        f"MetalLigandExchange pattern names must be unique; got {metal_names}"
    )
    assert len(set(aqua_names)) == 2, (
        f"AquaExchange pattern names must be unique; got {aqua_names}"
    )


def test_all_patterns_parse_with_rdkit():
    """Every pattern in both libraries must parse via RDKit's
    ``ReactionFromSmarts`` — syntactic check, no chemistry.

    This is the bare-minimum regression guard against SMARTS typo
    regressions (a missing ``>>`` or a malformed atom-map would
    otherwise silently disable the rule).
    """
    metal_patterns: List = _load_metal_ligand_patterns()
    aqua_patterns: List = _load_aqua_patterns()
    for name, smarts in metal_patterns + aqua_patterns:
        rxn = _reaction_template(smarts)
        # Each pattern must have exactly 2 reactant templates
        # (metal complex + donor) and 2 product templates (new metal
        # complex + leaving group / proton-loss species).
        assert rxn.GetNumReactantTemplates() == 2, (
            f"pattern {name} must have 2 reactant templates; "
            f"got {rxn.GetNumReactantTemplates()}"
        )
        assert rxn.GetNumProductTemplates() == 2, (
            f"pattern {name} must have 2 product templates; "
            f"got {rxn.GetNumProductTemplates()}"
        )


def test_canonical_patterns_first_in_list():
    """The canonical Pt_Cl_NH3 / Pt_Cl_H2O_first patterns must be the
    first entry in their respective libraries — the rule class uses
    ``patterns[0]`` as the default ``pattern_smiles`` for
    ``MetalLigandExchange._reduce`` / ``AquaExchange._reduce``.
    """
    metal_patterns = _load_metal_ligand_patterns()
    aqua_patterns = _load_aqua_patterns()
    assert metal_patterns[0][0] == "Pt_Cl_NH3", (
        f"first MetalLigandExchange pattern must be Pt_Cl_NH3; "
        f"got {metal_patterns[0][0]}"
    )
    assert aqua_patterns[0][0] == "Pt_Cl_H2O_first", (
        f"first AquaExchange pattern must be Pt_Cl_H2O_first; "
        f"got {aqua_patterns[0][0]}"
    )


def test_rule_classes_have_pattern_accessors():
    """The ReactionRule subclasses must expose the SMARTS library via
    ``available_smarts()`` so the MCTS layer can introspect them.
    """
    from molmetal.molmetal_lam.reactions.beta_reductions import REACTION_RULES

    mle = REACTION_RULES["MetalLigandExchange"]
    metal_smarts = mle.available_smarts()
    assert len(metal_smarts) == 5
    # The default pattern_smiles (used by _reduce) must be the
    # canonical Pt_Cl_NH3 SMARTS.
    assert mle.pattern_smiles == (
        "[Pt:1]([Cl:2])[*:3].[NH3:4]>>"
        "[Pt:1]([NH3:4])[*:3].[Cl:2]"
    )

    ae = REACTION_RULES["AquaExchange"]
    aqua_smarts = ae.available_smarts()
    assert len(aqua_smarts) == 2
    # The default pattern_smiles (used by _reduce) must be the
    # canonical Pt_Cl_H2O_first SMARTS.
    assert ae.pattern_smiles == (
        "[Pt:1]([Cl:2])[*:3].[OH2:4]>>"
        "[Pt:1]([OH:4])[*:3].[Cl:2]"
    )
    # AquaContext accessor must work.
    ctx = ae.aqua_context("Pt_Cl_H2O_first")
    assert abs(ctx.pka1 - 6.5) < 1e-6
