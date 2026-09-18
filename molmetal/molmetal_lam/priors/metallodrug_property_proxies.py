"""Metallodrug-specific property proxies (TODO-25 §4.6 / §5.7 follow-up).

This module houses **screening proxies** for metallodrug-specific
properties that the canonical drug-discovery metrics do not cover
directly.  Every function here:

  * operates CPU-only on SMILES strings (RDKit-only)
  * returns a value in ``[0, 1]`` — these are PROXIES, not measured
    quantities
  * returns exactly ``0.0`` for non-metallodrug molecules or invalid
    SMILES (graceful fallback for tests / empty pools)
  * cites the chemistry / bioinorganic literature it leans on so
    readers can decide which proxy is fit-for-purpose

The four proxies shipped here cover the canonical Pt(II)/Pd(II)
therapeutic liabilities surfaced by Reedijk (1996), Lippard &
Berg (1995), and Sadler / Barnham / al-Hossary-style medicinal
inorganic chemistry references:

  1. :func:`reduction_potential_proxy`  — heuristic Pt(II)/Pt(IV)
     redox liability (Shriver & Atkins Table 17.7 reference values).
  2. :func:`trans_effect_proxy`         — count of high-trans-effect
     ligands in the inner sphere (Appleton 1997 Coord. Chem. Rev.).
  3. :func:`lfse_proxy`                 — simplified ligand-field
     stabilization energy for the d8 square-planar Pt(II) case
     (Δ_oct and Δ_tet / cis/trans mapping per Miessler 2014).
  4. :func:`pt_dna_crosslink_proxy`     — Pt-DNA covalent crosslink
     propensity: counts labile Pt-Cl / Pt-O bonds scaled by logP
     (Wang 2003 / Cohen 2007) — non-Pt molecules return ``0.0``.

Honest framing
--------------
These are screening proxies.  They are NOT a substitute for
wet-lab measurement (Cyclic Voltammetry, Kb assays, etc.).
The values are clipped to ``[0, 1]`` for use in the MCTS reward
aggregator without dominating the dominant Vina / SA / QED reward.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None


__all__ = [
    "reduction_potential_proxy",
    "trans_effect_proxy",
    "lfse_proxy",
    "pt_dna_crosslink_proxy",
    "compute_metallodrug_proxies",
]


# ---------------------------------------------------------------------------
# Ligand-field strength heuristic (spectrochemical series, qualitative)
# ---------------------------------------------------------------------------
# Reference: Shriver & Atkins *Inorganic Chemistry* Table 17.7
# (5th ed.).  We anchor the relative ordering and clip the absolute
# values to [0, 1] for use in a screening reward.
#
# Order (weak → strong): I- < Br- < Cl- < F- < OH- < H2O < NH3 < en
#                        < bpy < NO2- < PR3 < CN- < CO (approximate)
_LIGAND_FIELD_STRENGTH: Dict[str, float] = {
    "I": 0.20, "Br": 0.25, "Cl": 0.30, "F": 0.50,
    "O": 0.55, "S": 0.50, "N": 0.65, "P": 0.85,
    "CN": 0.95, "CO": 0.95, "C2H4": 0.80,
}

# Trans-effect donor strengths (Pt(II) series) per Appleton et al.,
# *Coord. Chem. Rev.* 166 (1997) 313–359.  These ARE NOT the same as
# ligand-field strength — the trans effect is a kinetic (substitution)
# effect whereas ligand-field strength is a thermodynamic (splitting)
# effect.  We expose both for transparency.
_TRANS_EFFECT_STRENGTH: Dict[str, float] = {
    "CN": 1.00, "CO": 0.95, "NO2": 0.85,
    "PR3": 0.90, "C2H4": 0.85,
    "Cl": 0.40, "Br": 0.40, "I": 0.30,
    "NH3": 0.30, "H2O": 0.25, "OH": 0.30,
}

# Counts as "high-trans-effect" if strength >= this threshold.
_HIGH_TRANS_THRESHOLD: float = 0.80


def _mol(smiles):
    """Parse ``smiles`` into an RDKit mol with safe fallback.

    Mirrors the convention in
    :class:`molmetal_lam.priors.anticancer_metric_suite.AnticancerMetricSuite`:
    invalid / wildcard / non-string input returns ``None`` so callers
    can fall back to a neutral ``0.5`` proxy value.
    """
    if Chem is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            mol = None
    return mol


def _detect_metal(mol, metal: str = "Pt") -> bool:
    """Return True iff ``mol`` contains at least one atom of ``metal``.

    ``metal`` defaults to "Pt" which is the canonical target of these
    proxies; callers may pass "Pd" / "Au" etc. for related d8
    analogues (same square-planar geometry, similar liabilities).
    """
    if mol is None:
        return False
    target = metal.strip().title()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == target:
            return True
    return False


def _inner_sphere_ligands(mol, metal: str = "Pt") -> Iterable[Tuple[int, str]]:
    """Yield ``(idx, ligand_key)`` for each inner-sphere donor of the metal.

    Heuristic — we walk the metal's direct neighbours and bucket each
    by the dominant donor element / motif:
      * Cl/Br/I  -> halogen
      * O        -> "O"
      * S        -> "S"
      * N        -> "N" (ammine / amine / heterocycle)
      * P        -> "P"  (phosphine)
      * C with C=C neighbour -> "C2H4" (olefin)
      * C with C#C neighbour -> "C#" (alkyne / CO surrogate)
      * N with N#N or N=N   -> "N3"  (azide — keep distinct from ammine)
      * N with C#N triple   -> "CN"  (cyanide)
      * otherwise C         -> "C"
    """
    if mol is None:
        return []
    target = metal.strip().title()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != target:
            continue
        for nbr in atom.GetNeighbors():
            sym = nbr.GetSymbol()
            idx = nbr.GetIdx()
            # Aromatic N is treated as ammine N unless explicitly an azide.
            if sym == "N":
                # Azide N-N=N detection (terminal NN=N pattern)
                has_n_double_n = any(
                    b.GetBondTypeAsDouble() >= 2 and b.GetOtherAtom(nbr).GetSymbol() == "N"
                    for b in nbr.GetBonds()
                )
                if has_n_double_n:
                    yield (idx, "N3")
                    continue
                # Cyanide: N triple-bonded to C (e.g. CN- ligand — note
                # RDKit may attach Pt via the C atom, so we check both
                # directions below when this N is the inner-sphere
                # donor; if Pt is attached via C and the C carries a
                # triple bond to N we classify as "CN" via the C-side
                # inspection in the sym == "C" branch.)
                yield (idx, "N")
                continue
            if sym in ("Cl", "Br", "I"):
                yield (idx, sym)
                continue
            if sym == "O":
                yield (idx, "O")
                continue
            if sym == "S":
                yield (idx, "S")
                continue
            if sym == "P":
                yield (idx, "P")
                continue
            if sym == "C":
                # Inspect C neighbours for olefin / alkyne / CO / CN motifs
                olefin = any(
                    b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(nbr).GetSymbol() == "C"
                    for b in nbr.GetBonds()
                )
                alkyne = any(
                    b.GetBondTypeAsDouble() == 3 and b.GetOtherAtom(nbr).GetSymbol() == "C"
                    for b in nbr.GetBonds()
                )
                carbonyl = any(
                    b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(nbr).GetSymbol() == "O"
                    for b in nbr.GetBonds()
                )
                # Cyanide: C triple-bonded to N (CN- ligand, often
                # Pt-bonded via the C atom).  Check this first so
                # cyanide is classified correctly.
                cyanide = any(
                    b.GetBondTypeAsDouble() == 3 and b.GetOtherAtom(nbr).GetSymbol() == "N"
                    for b in nbr.GetBonds()
                )
                if cyanide:
                    yield (idx, "CN")
                elif carbonyl:
                    yield (idx, "CO")
                elif alkyne:
                    yield (idx, "C#")
                elif olefin:
                    yield (idx, "C2H4")
                else:
                    yield (idx, "C")
                continue
            yield (idx, sym)


def _labile_pt_bonds(mol) -> int:
    """Count labile Pt-X bonds (X = Cl or O of aqua / hydroxide).

    These are the bonds that hydrolyse fastest in the intracellular
    milieu and are the kinetic gateway to Pt-DNA crosslink formation
    (Wang 2003 *J. Biol. Inorg. Chem.* 8; Cohen 2007 *Chem. Biol.*).
    """
    if mol is None:
        return 0
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "Pt":
            continue
        for nbr in atom.GetNeighbors():
            sym = nbr.GetSymbol()
            if sym == "Cl":
                count += 1
            elif sym == "O":
                # Aqua vs alkoxide: very rough — count any Pt-O as labile.
                count += 1
    return count


def reduction_potential_proxy(smiles: str, metal: str = "Pt") -> float:
    """Heuristic Pt(II)/Pt(IV) reduction-potential proxy in ``[0, 1]``.

    The proxy uses the **mean ligand-field strength** of the inner-
    sphere donors (Shriver & Atkins Table 17.7).  Strong-field
    ligands (CN-, CO, PR3) raise the d-orbital splitting and
    therefore shift the M(II) ↔ M(IV) redox couple positive
    (more reducing-on-paper).  For non-Pt / non-recognised metals,
    the function returns ``0.5`` (the neutral default used by
    :class:`AnticancerMetricSuite`).

    Lit anchor:
      * Shriver & Atkins *Inorganic Chemistry* Table 17.7 (spectrochemical series).
      * Reedijk 1996 *Platinum Metals Rev.* 40 (aquation + redox of Pt drugs).
    """
    if Chem is None:
        return 0.5
    mol = _mol(smiles)
    if mol is None or not _detect_metal(mol, metal):
        return 0.0 if not _detect_metal(mol, metal) and mol is not None else 0.0
    # Need a real Pt in the mol for a non-trivial proxy.
    ligands = list(_inner_sphere_ligands(mol, metal))
    if not ligands:
        return 0.5
    strengths = [_LIGAND_FIELD_STRENGTH.get(key, 0.5) for _, key in ligands]
    mean_strength = sum(strengths) / len(strengths)
    # Map mean_strength in [0.20, 0.95] to a [0, 1] score.
    lo, hi = 0.20, 0.95
    score = (mean_strength - lo) / (hi - lo)
    return float(max(0.0, min(1.0, score)))


def trans_effect_proxy(smiles: str, metal: str = "Pt") -> float:
    """Count of high-trans-effect ligands in the inner sphere.

    Pt(II) specific by default (the kinetic series is canonical for
    square-planar d8 centres).  Returns the **count** of inner-sphere
    donors whose trans-effect strength >= ``_HIGH_TRANS_THRESHOLD``,
    divided by the canonical 4-coordinate limit and clipped to
    ``[0, 1]``.  Returns ``0.0`` when the molecule has no ``metal``
    centre (matches the "no-Pt mols return 0.0" requirement).

    Lit anchor:
      * Appleton, T. G. et al. *Coord. Chem. Rev.* 166 (1997) 313-359.
    """
    if Chem is None:
        return 0.0
    mol = _mol(smiles)
    if mol is None or not _detect_metal(mol, metal):
        return 0.0
    ligands = list(_inner_sphere_ligands(mol, metal))
    if not ligands:
        return 0.0
    # Map motif -> strength, with `key` being "CN" for CN- (Appleton notation)
    high_count = 0
    for _, key in ligands:
        # Map our key to Appleton's label
        if key == "CN":
            s = _TRANS_EFFECT_STRENGTH["CN"]
        elif key == "C#":
            # Alkyne is a moderate trans donor — treat it as ~moderate
            s = 0.55
        elif key in ("Cl", "Br", "I"):
            s = _TRANS_EFFECT_STRENGTH[key]
        elif key == "C2H4":
            s = _TRANS_EFFECT_STRENGTH["C2H4"]
        elif key == "P":
            s = _TRANS_EFFECT_STRENGTH["PR3"]
        elif key == "CO":
            s = _TRANS_EFFECT_STRENGTH["CO"]
        else:
            s = 0.0
        if s >= _HIGH_TRANS_THRESHOLD:
            high_count += 1
    # Divide by 4 (canonical CN for square-planar Pt(II))
    return float(max(0.0, min(1.0, high_count / 4.0)))


def lfse_proxy(
    smiles: str,
    metal: str = "Pt",
    d_electron_count: int = 8,
) -> float:
    """Ligand-field stabilization energy proxy in ``[0, 1]``.

    Simplified for Pt(II) d8 square-planar geometry.  Returns:

      * ``0.0`` when d0 / d10 (no t2g electrons) or no metal centre
      * ``(n_t2g * Δ_oct_proxy) / LFSE_REFERENCE`` otherwise

    where Δ_oct_proxy is the mean ligand-field strength of the
    inner-sphere donors (Shriver & Atkins) and n_t2g is the number
    of t2g electrons in the d8 configuration (n_t2g = 6 for d8,
    capped at the absolute maximum possible).

    Lit anchor:
      * Miessler, Fischer & Tarr *Inorganic Chemistry* (2014, 5th ed.) §10
        (LFSE for octahedral / square-planar d8 cases).
    """
    if Chem is None:
        return 0.0
    if d_electron_count <= 0 or d_electron_count >= 10:
        return 0.0
    mol = _mol(smiles)
    if mol is None or not _detect_metal(mol, metal):
        return 0.0
    ligands = list(_inner_sphere_ligands(mol, metal))
    if not ligands:
        return 0.0
    # n_t2g = electrons below the eg gap; for d8 octahedral it's 6.
    # For d8 square-planar the dx2-y2 is pushed higher (12B + something)
    # but we keep a simple proxy: n_t2g = min(d_electron_count, 6)
    # (the surplus above 6 sits in eg in octahedral, but for square-planar
    # it's mostly in the dz2/dx2-y2 orbitals).  We cap at 6 so the score
    # remains bounded.
    n_t2g = min(int(d_electron_count), 6)
    if n_t2g <= 0:
        return 0.0
    # Mean Δ_oct proxy from the ligands (range ~0.2 .. 1.0 in our table).
    mean_field = sum(_LIGAND_FIELD_STRENGTH.get(k, 0.5) for _, k in ligands) / len(ligands)
    # Score: 0.4 * n_t2g * mean_field  (clipped at 1.0)
    raw = 0.4 * n_t2g * mean_field
    return float(max(0.0, min(1.0, raw)))


def pt_dna_crosslink_proxy(smiles: str) -> float:
    """Estimate Pt-DNA covalent crosslink propensity in ``[0, 1]``.

    Heuristic: count labile Pt-Cl / Pt-O bonds (Wang 2003;
    Cohen 2007), then scale by logP (membrane-permeability proxy).
    Returns ``0.0`` for any non-Pt molecule.  Score is clipped to
    ``[0, 1]``.

    Lit anchor:
      * Wang, D. & Lippard, S. J. *Nat. Rev. Drug Discov.* 4 (2005) 307-320.
      * Cohen, S. M. *New J. Chem.* 31 (2007) 1329-1337.

    Honest caveat: logP is a coarse membrane permeability proxy
    (cf. Egan 2002 *J. Pharm. Sci.* 91, 1842).  The labile-bond
    count ignores hydrolysis rate constants and only counts the
    number of substitution-labile sites — sufficient for relative
    ranking in a screening reward.
    """
    if Chem is None or Descriptors is None:
        return 0.0
    mol = _mol(smiles)
    if mol is None or not _detect_metal(mol, "Pt"):
        return 0.0
    n_labile = _labile_pt_bonds(mol)
    if n_labile == 0:
        return 0.0
    try:
        log_p = float(Descriptors.MolLogP(mol))
    except Exception:
        return 0.0
    # Membrane-permeability scaling: peak near logP ~ 0 (per Egan 2002
    # Wertz model), falling off on either side.  Clamp to [0, 1] so
    # the function always produces a bounded score.
    abs_log_p = abs(log_p)
    # 1.0 at logP ~ 0; ~0.3 at |logP| ~ 3; ~0.1 at |logP| ~ 5
    membrane = max(0.0, 1.0 - 0.25 * abs_log_p)
    # Crosslink count: 2 labile bonds is the canonical "good" Pt(II)
    # motif (e.g. cisplatin).  Saturate at >= 2 with a gentle tail.
    if n_labile <= 2:
        bond_term = 0.5 * n_labile
    else:
        bond_term = 1.0 - 1.0 / (n_labile + 1)
    score = bond_term * membrane
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------
def compute_metallodrug_proxies(smiles: str, metal: str = "Pt") -> Dict[str, float]:
    """Run all four metallodrug proxies on ``smiles``.

    Returns a dict with four keys:

      * ``reduction_potential``  — :func:`reduction_potential_proxy`
      * ``trans_effect``         — :func:`trans_effect_proxy`
      * ``lfse``                 — :func:`lfse_proxy`
      * ``pt_dna_crosslink``     — :func:`pt_dna_crosslink_proxy`

    Each value is in ``[0, 1]``.  Non-metallodrug molecules return
    ``0.0`` for the last three and ``0.0`` for ``reduction_potential``
    when the metal is not present.
    """
    return {
        "reduction_potential": reduction_potential_proxy(smiles, metal=metal),
        "trans_effect": trans_effect_proxy(smiles, metal=metal),
        "lfse": lfse_proxy(smiles, metal=metal),
        "pt_dna_crosslink": pt_dna_crosslink_proxy(smiles),
    }