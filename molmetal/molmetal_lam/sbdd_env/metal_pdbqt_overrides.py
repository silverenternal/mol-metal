"""Meeko atom-type overrides for d-block Pt-group metals.

================================================================
What this is
================================================================
meeko 0.8.0 ships ``ad4_types.json`` covering only Mg/Ca/Mn/Fe/Zn, so when
a ligand contains a Pt-group metal (Pt/Ru/Ir/Os/Rh/Pd/Au/Re) the
``AtomTyper._type_atoms`` pass leaves ``molsetup.atom_type is None`` for
those atoms and ``PDBQTWriterLegacy.write_string`` rejects the molecule
in ``_is_molsetup_ok`` (writer.py:481-499).

This module provides a small set of SMARTS rules that the caller hands to
``MoleculePreparation(add_atom_types=...)``. Each rule maps a single
element to a generic AutoDock atom type — by convention ``"M"`` for
``M``(etal) and ``"M1"`` for ``M1`` (oxidation-state / coordination-tag
variant) — so Vina accepts the PDBQT (it logs a warning for unknown
atom types and falls back to generic VDW).

We deliberately keep this CPU-only and do NOT modify the meeko source.
Instead, we feed the rules via the public ``add_atom_types`` parameter
which is appended under a unique ``"add_atom_types"`` group key (see
``meeko/preparation.py:get_atom_params`` lines 476-486).

Honest framing
--------------
The atype ``"M"`` / ``"M1"`` is *not* a chemically calibrated parameter —
it is a graceful-degradation marker that lets Vina emit a PDBQT and
compute *something*. Downstream consumers MUST treat the resulting
kcal/mol as ``status="MEASURED_GENERIC_ATOM"``, not a full-fidelity Vina
score for Pt-group metallodrugs. For higher fidelity, callers should
fall back to ``boltz_affinity`` (D-MPNN pIC50, MEASURED, SMILES-only) —
that path lives in ``molmetal/scoring/score_any.py`` and is wired up in
Wf-121 P1.3.

Chemotype-tuned VDW (Wf-143 P3)
--------------------------------
Wf-143 Stage 2 aggregated per-metal (r, ε) VDW parameters from
literature mining (Morris 2009 AD4 for Mg/Ca/Mn/Fe/Zn, Rappe 1992 UFF
+ Bondi 1964 VDW radii for the 8 d-block Pt-group metals). The data
lives in :data:`DBLOCK_VDW_TABLE_PATH`. This module now exposes
:func:`get_metal_vdw_table` to load it and lets callers switch the
SMARTS rules from generic ``"M"`` to chemotype-specific atype strings
(e.g., ``"Pt"``, ``"Ru"``) via :func:`add_metal_atom_types` with
``chemotype_vdw=True``.

Reference
---------
- meeko PR #283 / issue tracking 0.8.0 coverage of d-block Pt-group metals
- TODO-110 (parent plan) §2 Solution Path D
- Wf-121 Phase 1 — SMARTS rules definition
- Wf-143 Stage 2 — aggregated d-block VDW literature mining
- Wf-143 Stage 3 — meko atype injection (this module)
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------
# Exception types
# -----------------------------------------------------------------


class MetalPDBQTUnavailable(RuntimeError):
    """Raised when a metallodrug SMILES cannot be converted to PDBQT.

    meeko 0.8.0 ships ``ad4_types.json`` covering only Mg/Ca/Mn/Fe/Zn.
    Even with our Pt-group SMARTS overrides injected via
    ``add_atom_types``, some pathological metallodrugs (e.g. exotic
    oxidation states, unsupported coordination geometries, or SMILES
    that fail to embed) may still cause ``PDBQTWriterLegacy.write_string``
    to reject the molecule. We surface that failure as a typed exception
    so callers can distinguish "we tried, but meeko could not produce a
    PDBQT" from generic Vina runtime errors.

    Notes
    -----
    This inherits from :class:`RuntimeError` so existing
    ``except RuntimeError`` handlers still catch it; the dedicated class
    exists so test assertions can pin the type and so callers can
    implement a metal-specific fallback (e.g. ``boltz_affinity``).
    """


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

#: The 8 d-block Pt-group metals we extend (TODO-110 §1).
#: Atomic numbers: Pt=78, Ru=44, Ir=77, Os=76, Rh=45, Pd=46, Au=79, Re=75.
PT_GROUP_ELEMENTS = ("Pt", "Ru", "Ir", "Os", "Rh", "Pd", "Au", "Re")

#: Generic AutoDock atype used for all Pt-group metals when we don't
#: have a calibrated MM/GB-SA or VDW parameter set.
#: ``M`` (single letter) follows the same convention meeko uses for the
#: existing metals (``Mg``, ``Ca``, ``Mn``, ``Fe``, ``Zn``); when we
#: later add a coordination-tagged variant, we map it to ``M1``.
GENERIC_ATYPE = "M"
COORDINATION_ATYPE = "M1"

#: Vina fallback atype when chemotype-specific atype is rejected by
#: the parser. ``Mg`` is on Vina's accepted atype list (alongside
#: Mn/Zn/Ca/Fe); it's the closest calibrated VDW atype Vina knows
#: natively (Morris 2009 AD4 metalloprotein table). Downstream
#: consumers MUST treat kcal/mol driven by ``Mg`` as
#: ``status="MEASURED_GENERIC_ATOM"``.
VINA_FALLBACK_ATYPE = "Mg"

# -----------------------------------------------------------------
# Element coverage — MEASURED 2026-10-12 against Vina 1.2.7
# -----------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# ``PT_GROUP_ELEMENTS`` covers 8 metals. It does NOT cover the rest of
# the tmQM periodic table, so every ligand carrying boron, silicon,
# selenium, arsenic, titanium, ... sailed through
# :func:`bench_cisplatin_real_vina._rewrite_metal_atype_for_vina`
# untouched and died at ``set_ligand_from_file`` with
# ``PDBQT parsing error: Atom type B is not a valid AutoDock type``.
# Boron is in 6.22% of tmQM complexes, so this is not a corner case.
#
# HOW IT WAS MEASURED (not guessed)
# ---------------------------------
# MEASURED by *asking Vina*, not by reading a table. For each of the
# 44 element symbols that occur in ``.data/molmetal/tmQM/tmQM_X1.xyz.gz``
# we took a real meeko-emitted cisplatin ligand PDBQT
# (``molmetal/reports/wf121_cisplatin_real_vina/830c/cisplatin.pdbqt``),
# substituted the symbol into the atom-name field ``[12:16]`` and the
# AutoDock atype field ``[77:79]``, and called
# ``Vina(sf_name="vina").set_ligand_from_file()``. A parse error means
# Vina has no row for that symbol. Script + raw verdict:
# ``molmetal/scripts/vina_element_atype_probe.py``; verdict persisted to
# ``metrics/by_metric/vina_127_element_atypes.json``.
#
# 44 symbols probed: 21 ACCEPTED, 23 REJECTED. Corpus element counts
# live in ``metrics/by_metric/tmqm_element_inventory.json``.
#
# NOTE this list is strictly larger than :data:`PT_GROUP_ELEMENTS`.
# ``Ni``/``Cu``/``Co``/``Cd``/``Hg``/``W``/``Se``/``Fe``/``Mn``/``Zn``
# ARE accepted natively and must therefore NEVER be rewritten — the
# old code would have clobbered them with ``Mg`` if the match were
# naively widened to "every metal".

#: Elements Vina 1.2.7's PDBQT parser accepts natively as an atom type.
#: Substituting any of these is a REGRESSION, not a fix.
VINA_ACCEPTED_ELEMENTS = frozenset(
    {
        "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
        "Si", "Se",
        # metals Vina has a real row for
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg", "W",
    }
)

#: Elements present in tmQM that Vina 1.2.7 REJECTS. These are the
#: only elements that may ever be rewritten. Every one of them, if
#: left alone, kills the whole pocket with a parse error.
VINA_REJECTED_ELEMENTS = frozenset(
    {
        # metalloid / p-block
        "B", "As",
        # d-block + f-block beyond Vina's table
        "Ti", "V", "Cr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
        "Rh", "Pd", "Ag", "La", "Hf", "Ta", "Re", "Os", "Ir",
        "Pt", "Au", "Sc",
    }
)

#: Non-metal rejected elements and their SURROGATE. These are NOT
#: metals, so sending them to ``Mg`` would be chemically incoherent —
#: ``Mg`` is a divalent metal ion and carries a metal VDW/charge
#: assumption that a boron or arsenic atom does not have.
#:
#: Justification is by COVALENT RADIUS proximity among Vina-accepted
#: non-metals (RDKit periodic table, MEASURED):
#:
#:   * ``B``  Rcov 0.84 A. Nearest accepted non-metal is ``C``
#:     (Rcov 0.76 A, |d| 0.08). Boron is a small covalent p-block
#:     atom; carbon is its nearest neighbour in both size and
#:     bonding. Cost of being wrong: a B gets carbon's aromatic/
#:     hydrophobic typing, so a borole ring scores as if it were a
#:     carbocycle. That is a real but bounded error, and it is FAR
#:     smaller than ``B -> Mg``.
#:   * ``As`` Rcov 1.19 A. Nearest accepted non-metal is ``P``
#:     (Rcov 1.07 A, |d| 0.12) — same group 15, and P is a
#:     Vina-accepted atype. ``As`` in tmQM is almost always an
#:     arsenate/arsinothiol auxiliary group, i.e. P-like.
#:     Cost of being wrong: an As is scored with P's VDW radius.
#:
#: What a WRONG rewrite would cost, concretely: the atype column
#: drives BOTH the VDW term AND the hydrogen-bond donor/acceptor
#: typing. Getting it wrong does not merely shift the score — it
#: silently changes the molecule Vina believes it is scoring, and the
#: resulting kcal/mol is reported as a real measurement. That is why
#: the rule is an explicit allowlist keyed on a MEASURED rejection
#: set, and never a broad "anything that is not C/H/N/O" heuristic.
VINA_NONMETAL_SURROGATES = {
    "B": "C",
    "As": "P",
}

#: Path to the aggregated per-metal VDW literature-mining table
#: (Wf-143 Stage 2). When this file is missing or unreadable we fall
#: back to :data:`GENERIC_ATYPE` for all metals.
DBLOCK_VDW_TABLE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "references"
    / "soft_params"
    / "dblock_vdw_table.json"
)

#: Probe SMILES — one canonical coordination complex per metal.
#: Each probe contains exactly ONE atom of the named element so that
#: a SMARTS match count of 1 == correct typing for the whole ligand.
#: Sources:
#:   - Pt:  cisplatin  ([Pt](N)(N)(Cl)Cl)  -- Pt(II) square planar
#:   - Ru:  ruthenium(II) pentaammine analog
#:   - Ir:  iridium(III) methyl  -- typical Cp*Ir fragment
#:   - Os:  osmium tetroxide fragment  -- Os(VIII)
#:   - Rh:  rhodium(III) methyl  -- Wilkinson-class
#:   - Pd:  palladium(II) phosphine  -- Pd(II) square planar
#:   - Au:  auranofin precursor gold(I) phosphine  -- Au(I) linear
#:   - Re:  rhenium heptoxide fragment  -- Re(VII)
PROBE_SMILES: Dict[str, str] = {
    "Pt": "[Pt](N)(N)(Cl)Cl",
    "Ru": "[Ru](N)(N)(N)(N)(N)Cl",
    "Ir": "[Ir](C)(C)(C)C(C)(C)C",  # Ir(III) hexamethyl simplified
    "Os": "[Os](=O)(=O)(=O)N",
    "Rh": "[Rh](C)(C)(C)C(C)(C)C",  # Rh(III) hexamethyl simplified
    "Pd": "[Pd](P)(P)(C)C",
    "Au": "[Au]P",
    "Re": "[Re](=O)(=O)(=O)C",
}


# -----------------------------------------------------------------
# SMARTS rules
# -----------------------------------------------------------------

#: Per-element SMARTS rule for the Pt-group metals.
#:
#: Format matches meeko ``ad4_types.json`` rows -- one dict per rule
#: with at minimum ``"smarts"`` and ``"atype"`` keys. We use atomic-
#: number SMARTS (``[Pt]``) rather than element-symbol (``Pt``) so the
#: rule matches the atom by Z, regardless of how RDKit canonicalises
#: the SMILES. ``[Pt]`` is the canonical form meeko uses for the
#: supported metals (see ``ad4_types.json``: ``"[#12]"`` for Mg, etc.).
#:
#: Each rule also carries an ``"IDX": [0]`` to be explicit about which
#: atom in the SMARTS pattern gets the atype assignment. For a single-
#: atom SMARTS this is trivially 0 but writing it out documents intent.
#:
#: ``"comment"`` is preserved by meeko for diagnostics (see
#: ``AtomTyper._type_atoms`` loop in ``meeko/atomtyper.py:43-93``).
#:
#: When ``chemotype_vdw=True`` (Wf-143 P3 default), the ``atype`` field
#: is overridden to the chemotype-specific element symbol (e.g.
#: ``"Pt"``, ``"Ru"``) by :func:`get_metal_atom_type_rules`.
METAL_ATOM_TYPE_RULES: Dict[str, Dict[str, object]] = {
    "Pt": {
        "smarts": "[Pt]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Ru": {
        "smarts": "[Ru]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Ir": {
        "smarts": "[Ir]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Os": {
        "smarts": "[Os]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Rh": {
        "smarts": "[Rh]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Pd": {
        "smarts": "[Pd]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Au": {
        "smarts": "[Au]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
    "Re": {
        "smarts": "[Re]",
        "atype": GENERIC_ATYPE,
        "IDX": [0],
        "comment": "Pt-group d-block metal — generic VDW (TODO-110, Wf-121 P1.1)",
    },
}


# -----------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------


def get_metal_vdw_table(
    elements: Optional[List[str]] = None,
    table_path: Optional[pathlib.Path] = None,
) -> Dict[str, Tuple[float, float]]:
    """Load per-metal (r, ε) from the Wf-143 Stage 2 aggregated table.

    Returns a mapping ``{element: (r_angstrom, epsilon_kcal_mol)}`` for
    each of the 8 d-block Pt-group metals listed in
    :data:`PT_GROUP_ELEMENTS`. The values come from
    ``molmetal/references/soft_params/dblock_vdw_table.json`` which was
    produced by Wf-143 Stage 2 (literature mining aggregation).

    Parameters
    ----------
    elements : list of str, optional
        Subset of :data:`PT_GROUP_ELEMENTS` to include. When ``None``
        (default) every supported metal is included.
    table_path : pathlib.Path, optional
        Override path to the aggregated table (handy for tests). When
        ``None`` (default) the module-level
        :data:`DBLOCK_VDW_TABLE_PATH` is used.

    Returns
    -------
    dict
        ``{element: (r_angstrom, epsilon_kcal_mol)}``. ``r`` is in
        ångström (Bondi 1964 + UFF derivative convention); ``ε`` is in
        kcal/mol (UFF D1 for neutral atoms).

    Raises
    ------
    FileNotFoundError
        If the aggregated table cannot be located.
    ValueError
        If any element in ``elements`` is not in
        :data:`PT_GROUP_ELEMENTS`, or if the table is missing a
        requested element.

    Honest framing
    --------------
    The aggregated values have ``+/-0.05-0.10 A`` uncertainty on ``r``
    and ``+/-0.020-0.030 kcal/mol`` on ``ε`` (per the table's
    ``_meta.sanity_ranges`` and per-metal ``uncertainty_range`` rows).
    They are NOT experimental measurements — they are literature
    mining aggregations of Bondi 1964 + Rappe 1992 UFF, with oxidation-
    state-specific retuning deferred to the R20 GPU wet-lab
    calibration (TODO-128 Phase 2).
    """
    if elements is None:
        elements = list(PT_GROUP_ELEMENTS)

    unknown = sorted(set(elements) - set(PT_GROUP_ELEMENTS))
    if unknown:
        raise ValueError(
            "unsupported metals requested in get_metal_vdw_table: %s "
            "(supported: %s)" % (unknown, list(PT_GROUP_ELEMENTS))
        )

    path = pathlib.Path(table_path) if table_path is not None else DBLOCK_VDW_TABLE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"d-block VDW aggregated table not found at {path}; "
            "this should be present after Wf-143 Stage 2 ship. Falling "
            "back to chemotype_vdw=False would be safer."
        )

    with path.open("r", encoding="utf-8") as fh:
        table = json.load(fh)

    out: Dict[str, Tuple[float, float]] = {}
    for elem in elements:
        if elem not in table:
            raise ValueError(
                f"element {elem!r} missing from d-block VDW table at {path}"
            )
        record = table[elem]
        r = float(record["r_angstrom"])
        eps = float(record["epsilon_kcal_mol"])
        out[elem] = (r, eps)
    return out


def get_metal_atom_type_rules(
    elements: Optional[List[str]] = None,
    chemotype_vdw: bool = True,
) -> List[Dict[str, object]]:
    """Return a list of meeko ``add_atom_types`` rule rows.

    Parameters
    ----------
    elements : list of str, optional
        Subset of :data:`PT_GROUP_ELEMENTS` to include. When ``None``
        (default) every supported metal is included.
    chemotype_vdw : bool, default ``True``
        When ``True`` (Wf-143 P3 default), the ``atype`` field of
        each rule is the chemotype-specific element symbol (e.g.
        ``"Pt"``, ``"Ru"``) rather than the generic ``"M"``. This
        lets us match PDBQT atom-type to per-metal (r, ε) parameters
        downstream. When the chemotype atype is rejected by Vina
        1.2.7 the caller (e.g. ``bench_cisplatin_real_vina.
        _rewrite_metal_atype_for_vina``) falls back to
        :data:`VINA_FALLBACK_ATYPE` (``"Mg"``).

        When ``False``, every rule carries the generic
        :data:`GENERIC_ATYPE` (``"M"``) — the Wf-121 P1 behaviour.

    Returns
    -------
    list of dict
        Each dict has the meeko ``add_atom_types`` row shape:
        ``{"smarts": str, "atype": str, "IDX": [int], "comment": str}``.
        This list can be passed directly to
        ``MoleculePreparation(add_atom_types=...)``.

    Raises
    ------
    ValueError
        If any element in ``elements`` is not in :data:`PT_GROUP_ELEMENTS`.
    """
    if elements is None:
        elements = list(PT_GROUP_ELEMENTS)

    unknown = sorted(set(elements) - set(PT_GROUP_ELEMENTS))
    if unknown:
        raise ValueError(
            "unsupported metals requested in add_metal_atom_types: %s "
            "(supported: %s)" % (unknown, list(PT_GROUP_ELEMENTS))
        )

    rules: List[Dict[str, object]] = []
    for elem in elements:
        rule = dict(METAL_ATOM_TYPE_RULES[elem])
        if chemotype_vdw:
            # Chemotype-specific atype. Meeko / Vina both accept the
            # element symbol verbatim when ``len(atype) == 2`` matches
            # the meeko column width and the symbol is recognised (Vina
            # *will* log a warning if the symbol isn't in its AD4
            # atom-type table — but it doesn't reject it the way it
            # rejects ``M``). If Vina does reject, the rewrite helper
            # downgrades to ``Mg``.
            rule["atype"] = elem
        rules.append(rule)
    return rules


def add_metal_atom_types(preparator, chemotype_vdw: bool = True) -> None:
    """Attach Pt-group metal SMARTS rules to an existing preparator.

    Parameters
    ----------
    preparator : meeko.MoleculePreparation
        An instantiated ``MoleculePreparation``. The function does NOT
        re-call ``prepare`` — it only mutates ``preparator.atom_params``
        in place by injecting a fresh ``"add_atom_types"`` group. The
        preparator's subsequent ``prepare(mol)`` call will pick up the
        new rules via ``AtomTyper._type_atoms``.
    chemotype_vdw : bool, default ``True``
        Forwarded to :func:`get_metal_atom_type_rules`. When ``True``
        the injected rules use chemotype-specific atype (``Pt``,
        ``Ru``, ...) rather than the generic ``"M"``.

    Notes
    -----
    meeko de-duplicates ``"add_atom_types"`` keys (``preparation.py:476-486``).
    If the preparator already has ``add_atom_types`` we fall back to the
    ``"add_atom_types_0"`` slot so we never collide. We do not strip any
    caller-supplied rules — Pt-group rules simply add coverage.

    Honest framing
    --------------
    This mutates an opaque object on the meeko public surface. If a
    future meeko release renames ``atom_params`` we will break loudly at
    the ``AttributeError`` here, which is preferable to silently doing
    nothing.

    With ``chemotype_vdw=True``, the emitted PDBQT will carry atype
    strings like ``Pt`` / ``Ru`` / ``Au`` that Vina 1.2.7 may or may
    not recognise. Downstream callers (e.g. ``_rewrite_metal_atype_for_vina``
    in :mod:`molmetal.scripts.bench_cisplatin_real_vina`) must therefore
    probe Vina's parse path and fall back to :data:`VINA_FALLBACK_ATYPE`
    (``"Mg"``) on rejection. With ``chemotype_vdw=False`` the original
    Wf-121 P1 ``M → Mg`` rewrite is the only path.
    """
    rules = get_metal_atom_type_rules(chemotype_vdw=chemotype_vdw)

    # Determine a unique group key (mirrors meeko.get_atom_params).
    base_key = "add_atom_types"
    key = base_key
    if hasattr(preparator, "atom_params") and preparator.atom_params is not None:
        for attempt in range(100):
            if key not in preparator.atom_params:
                break
            key = f"add_atom_types_{attempt}"
        else:
            raise RuntimeError(
                "could not find a unique key for add_metal_atom_types "
                "(tried add_atom_types .. add_atom_types_99)"
            )
    preparator.atom_params[key] = rules


def verify_metal_smarts() -> Dict[str, Dict[str, int]]:
    """Verify each Pt-group SMARTS rule matches its probe SMILES.

    Returns
    -------
    dict
        Mapping ``{element: {"probe_smiles": str, "expected": int,
        "matched": int, "ok": bool}}``. Used by the rule-verification
        report and by ``test_metal_pdbqt_overrides.py``.

    Notes
    -----
    Pure-Python validation that does not require meeko — handy for
    unit-testing the SMARTS coverage before the heavier
    ``MoleculePreparation`` round-trip in P1.2.
    """
    try:
        from rdkit import Chem  # type: ignore
    except ImportError as exc:  # pragma: no cover — RDKit is a hard dep
        raise RuntimeError("rdkit is required to verify metal SMARTS rules") from exc

    results: Dict[str, Dict[str, int]] = {}
    for elem in PT_GROUP_ELEMENTS:
        probe = PROBE_SMILES[elem]
        mol = Chem.MolFromSmiles(probe)
        if mol is None:
            results[elem] = {
                "probe_smiles": probe,
                "expected": 1,
                "matched": 0,
                "ok": False,
            }
            continue
        n_metal_atoms = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == elem)
        smarts = METAL_ATOM_TYPE_RULES[elem]["smarts"]
        pattern = Chem.MolFromSmarts(smarts)
        n_matches = len(mol.GetSubstructMatches(pattern)) if pattern is not None else 0
        results[elem] = {
            "probe_smiles": probe,
            "expected": n_metal_atoms,
            "matched": n_matches,
            "ok": n_matches == n_metal_atoms and n_metal_atoms >= 1,
        }
    return results


__all__ = [
    "PT_GROUP_ELEMENTS",
    "VINA_ACCEPTED_ELEMENTS",
    "VINA_REJECTED_ELEMENTS",
    "VINA_NONMETAL_SURROGATES",
    "GENERIC_ATYPE",
    "COORDINATION_ATYPE",
    "VINA_FALLBACK_ATYPE",
    "DBLOCK_VDW_TABLE_PATH",
    "PROBE_SMILES",
    "METAL_ATOM_TYPE_RULES",
    "get_metal_vdw_table",
    "get_metal_atom_type_rules",
    "add_metal_atom_types",
    "verify_metal_smarts",
    "MetalPDBQTUnavailable",
]
