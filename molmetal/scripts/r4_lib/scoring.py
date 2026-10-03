"""r4_lib.scoring — Lambda-native scoring channels (5 signals).

All five functions here are pure Lambda-native signals; they form
the ``r_vina_proxy`` channel of the Lambda-only reward aggregator
(see :func:`.aggregator.build_lambda_only_aggregator`).

Functions
---------
* :func:`alpha_equivalence_uniqueness_score` — distinct beta-NF count
  across ``states`` (siblings of a parent), normalised to ``[0, 1]``.
* :func:`click_rule_match_bonus` — ``+1.0`` if any of the 5 click
  rules fires on ``state`` with a compatible tile.
* :func:`metal_geometry_prior_bonus` — soft, tiered metal-geometry
  prior (Pt=4, Ru/Ir=6).  1.0 / 0.5 / 0.2 / 0.0 ladder.
* :func:`metal_compliance_truthful` — the truthful variant that
  ALSO requires the candidate to be MCTS-derived, not the seed
  itself (WF-Lambda-Fix-Singleton Fix 3).
* :func:`rdkit_validity_score` — ``1.0`` if RDKit can sanitise the
  SMILES, ``0.0`` otherwise.
* :func:`synthesizability_via_lambda_paths` — ``1.0`` if the molecule
  is a well-formed β-NF AND its canonical SMILES RDKit-sanitizes.

The reward weights (:data:`CLICK_RULE_WEIGHT` and
:data:`METAL_PRIOR_WEIGHT`) live in :mod:`.constants`; ``valid``,
``syn``, and ``aeq`` are weighted at 1.0 each (correctness signals
that should not be tuned).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


# ---------------------------------------------------------------------------
# Lambda-only scoring channels — all five signals are pure Lambda-native.
# ---------------------------------------------------------------------------
def alpha_equivalence_uniqueness_score(states: Sequence[Any]) -> float:
    """Count distinct beta-NF among ``states`` (siblings of a parent).

    Lambda-native diversity: alpha-equivalent molecules share a canonical
    SMILES, so distinct canonical SMILES ≅ distinct beta-NF classes.
    Output is normalized to ``[0, 1]`` by ``1 / (1 + n_distinct)`` so
    the metric is always finite and bounded.
    """
    if not states:
        return 0.0
    distinct: set = set()
    for s in states:
        try:
            smi = s.canonical_smiles()
        except Exception:
            continue
        if smi:
            distinct.add(str(smi))
    n = len(distinct)
    if n <= 0:
        return 0.0
    # 1.0 when everything is unique; asymptotes to 0 as n grows.
    return 1.0 / (1.0 + n - 1) if n > 1 else 1.0


def click_rule_match_bonus(state: Any, rules: Dict[str, Any]) -> float:
    """+1.0 if a click-rule fires on ``state`` with a compatible tile.

    Cheap proxy: check whether ``state`` itself carries a recognisable
    click-warhead (azide, alkyne, thiol, boronic acid, halide, amine,
    carboxylic-acid) and at least one matching rule is registered.
    Returns 1.0 when both conditions hold, 0.0 otherwise.
    """
    try:
        smi = state.canonical_smiles() or ""
    except Exception:
        smi = ""
    if not smi:
        return 0.0
    warheads = ("N=[N+]=[N-]", "C#C", "[SH]", "B(O)", "C(=O)O", "[NH2]")
    if not any(w in smi for w in warheads):
        return 0.0
    # We require the click-rule set to be non-empty for the bonus to fire.
    return 1.0 if rules else 0.0


def metal_geometry_prior_bonus(
    state: Any,
    *,
    enabled: bool,
    pt_target: int = 4,
    ru_ir_target: int = 6,
) -> float:
    """Soft, tiered metal-geometry prior bonus.

    Pt requires coordination number 4 (square-planar); Ru/Ir require
    6 (octahedral).  The bonus is now SOFT (not a hard gate) so the
    MCTS can still rank candidates that are off-by-one or missing
    coordination information — these are common during the rollout
    when the metal seed is a bare atom with a single handle (see
    WF-Lambda-Fix-Singleton Fix 1: ``[Pt]C#C`` style seeds).

    Tiered values (WorstCase=False; metal absent returns 0.0):

        coord_num == target          -> 1.0   (exact match)
        coord_num in {target-1, target+1} -> 0.5  (off-by-one)
        metal present, coord_num unmeasurable / 0 -> 0.2
        no metal atom               -> 0.0   (not a metal candidate)
        |coord_num - target| > 1    -> 0.0   (too far off)

    When ``enabled`` is False the prior is a strict no-op (returns 0.0)
    — this is the Round-10 axis-D "prior OFF" path used in ablations.

    Honest framing
    --------------
    The previous hard-gate (1.0 iff coord==target else 0.0) made the
    reward curve binary; partial-but-promising candidates were
    silently downgraded to 0, which collapsed the MCTS onto the
    fully-bonded Pt seeds (``n_distinct=1`` reported in the Round-12
    5x1 metal-seed pilot).  The soft tiers keep a continuous gradient
    for the MCTS to follow: 1.0 > 0.5 > 0.2 > 0.0.  This restores
    truthfulness of ``metal_compliance_rate`` (now 0.5 for ±1
    candidates, not silently 0.0).
    """
    if not enabled:
        return 0.0
    try:
        atoms = getattr(state, "atoms", []) or []
        bonds = getattr(state, "bonds", []) or []
    except Exception:
        return 0.0
    if not atoms:
        return 0.0
    # Identify metal centres by symbol.  We intentionally do NOT
    # import rdkit here — the prior is a structural (not a 3-D) signal.
    METALS = {"Pt": 78, "Ru": 44, "Ir": 77}
    # Walk atoms; if no metal is present, return 0.0 (not a metal candidate).
    metal_atoms = []
    for atom in atoms:
        sym = getattr(atom, "symbol", "") or ""
        if sym in METALS:
            target = pt_target if sym == "Pt" else ru_ir_target
            metal_atoms.append((atom, sym, target))
    if not metal_atoms:
        return 0.0
    # Aggregate best tier across all metal centres (max over metals).
    # For each metal, compute its coordination number and look up the tier.
    best = 0.0
    for atom, sym, target in metal_atoms:
        coord = 0
        try:
            atom_id = id(atom)
        except Exception:
            atom_id = None
        for b in bonds:
            try:
                if id(b.atom_a) == atom_id or id(b.atom_b) == atom_id:
                    if getattr(b, "kind", "") == "hydrogen":
                        continue
                    coord += 1
            except Exception:
                continue
        if coord == target:
            tier = 1.0
        elif coord in (target - 1, target + 1):
            tier = 0.5
        elif coord == 0:
            # metal present but no surrounding coordination observed
            # (bare-metal seed or coordination info unavailable).
            tier = 0.2
        else:
            # |coord - target| >= 2: too far off
            tier = 0.0
        if tier > best:
            best = tier
    return best


def metal_compliance_truthful(
    state: Any,
    *,
    enabled: bool,
    root_smiles: Optional[str] = None,
    seed_smiles_set: Optional[set] = None,
    pt_target: int = 4,
    ru_ir_target: int = 6,
) -> float:
    """Truthful metal-geometry prior — like :func:`metal_geometry_prior_bonus`
    but ALSO requires the molecule to be MCTS-derived, not the seed.

    WF-Lambda-Fix-Singleton Fix 3.  The historical
    ``metal_geometry_prior_bonus`` checks only ``atoms`` + ``bonds`` and
    therefore trivially fires on the bare-metal seed itself (e.g.
    ``[Pt]C#C`` contains 1 Pt + 1 bond, scoring the seed as a
    partial-credibility "0.2-tier" candidate).  This made
    ``metal_compliance_rate`` non-trivially > 0 even when the MCTS
    produced zero expansion steps — the metric was lying.

    The truthful variant adds two extra checks:

        (a) The molecule's canonical SMILES must differ from
            ``root_smiles`` (i.e. an expansion step actually ran) — when
            the candidate is the seed itself, the prior returns 0.0.
        (b) The molecule's canonical SMILES must NOT be in
            ``seed_smiles_set`` (a set of seed-only SMILES, e.g. the
            entries of ``METAL_SEED_SMILES`` or the 2-D fallback
            ``Cl[Pt]Cl``).  When the candidate is a seed-only motif,
            the prior returns 0.0.

    Both rules preserve the soft-tier reward for genuine MCTS-derived
    candidates (e.g. a click rule that adds azide to the alkyne
    handle → the resulting Pt-triazole has a non-zero coord count
    and is not the bare-metal seed).

    Parameters
    ----------
    state : MoleculeClosedTerm
        The candidate to score.
    enabled : bool
        When False, the prior is a strict no-op (returns 0.0).
    root_smiles : str, optional
        Canonical SMILES of the MCTS root.  Candidates equal to the
        root fail the check.  Pass ``None`` to disable this check.
    seed_smiles_set : set of str, optional
        Set of seed-only SMILES (e.g. ``set(METAL_SEED_SMILES.values())``).
        Any candidate in this set fails the check.  Pass ``None`` to
        disable this check.
    pt_target, ru_ir_target : int
        Coordination-number targets (Pt=4, Ru/Ir=6).  Defaults match
        :func:`metal_geometry_prior_bonus`.

    Returns
    -------
    float
        The soft tier (0.0 / 0.2 / 0.5 / 1.0) when the candidate
        passes both root-difference and seed-set checks; 0.0 when the
        candidate is the root or a seed-only motif, or when
        ``enabled`` is False.
    """
    if not enabled:
        return 0.0
    # ---- Truthfulness checks: candidate must NOT be the root or a
    # bare-metal seed motif.  This is the WF-Lambda-Fix-Singleton
    # Fix 3 core: stop counting the seed itself as a "compliant"
    # candidate.
    try:
        cand_smi = state.canonical_smiles() or ""
    except Exception:
        cand_smi = ""
    if not cand_smi:
        return 0.0
    if root_smiles is not None and cand_smi == root_smiles:
        return 0.0
    if seed_smiles_set is not None and cand_smi in seed_smiles_set:
        return 0.0
    # Delegate the actual coordination-number tier to the original
    # function.  This keeps the truthfulness gate orthogonal to the
    # chemistry prior — when the gate passes we still get the soft
    # 0.2 / 0.5 / 1.0 reward from metal_geometry_prior_bonus.
    return metal_geometry_prior_bonus(
        state,
        enabled=True,
        pt_target=pt_target,
        ru_ir_target=ru_ir_target,
    )


def rdkit_validity_score(smiles: str) -> float:
    """1.0 if RDKit can sanitize ``smiles``, 0.0 otherwise."""
    if not smiles:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        Chem.SanitizeMol(mol)
        return 1.0
    except Exception:
        return 0.0


def synthesizability_via_lambda_paths(
    state: Any,
    *,
    check_bnf: Any,
    use_valence_bnf: bool = True,
) -> float:
    """Count of well-formed λ-term reduction paths to a valid SMILES.

    Returns 1.0 if the molecule is a well-formed beta-NF (atom
    saturation matches the chosen predicate) AND its canonical SMILES
    RDKit-sanitizes, 0.0 otherwise.  The "count of paths"
    interpretation collapses to 0/1 here because a beta-NF has no
    further β-reductions — there is exactly one such path (the empty
    one) iff the molecule is a value.

    WF-Lambda-1c — when ``use_valence_bnf`` is True (default), the
    predicate used is :func:`check_beta_normal_form_for_rdkit`, which
    accepts an atom as BNF-saturated iff its covalent valence is
    filled (NOT its full arity, which includes lone-pair capacity
    reserved for H-bonding / dative interactions). This is the
    chemistry-correct metric: a nitrogen with three covalent bonds is
    BNF-saturated even if it has one lone pair left.

    When ``use_valence_bnf`` is False, the legacy arity-based predicate
    (``check_bnf``, typically :func:`check_beta_normal_form`) is used —
    kept for ablation studies and for any caller that wants the strict
    combinator-arithmetic view of BNF.
    """
    if use_valence_bnf:
        try:
            from molmetal_lam.lam_chem.well_formedness import (
                check_beta_normal_form_for_rdkit_term as _check_valence_bnf_term,
            )

            # Term-level valence-based BNF — uses RDKit's
            # ``valence_used`` (implicit-H-inclusive) plus bond-ledger
            # used_sites as a cross-check.  This is the chemistry-
            # correct predicate: a nitrogen with 3 implicit H + 1
            # heavy bond is valence-saturated (4 >= 3).
            well_formed = bool(_check_valence_bnf_term(state))
        except Exception:
            well_formed = False
    else:
        try:
            well_formed = bool(check_bnf(state))
        except Exception:
            well_formed = False
    if not well_formed:
        return 0.0
    try:
        smi = state.canonical_smiles() or ""
    except Exception:
        smi = ""
    return rdkit_validity_score(smi)
