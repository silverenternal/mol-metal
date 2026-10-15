"""stereo_postprocess — post-decode stereocentre cleanup stub.

Wf-A41-fixup-2 (2026-10-15) J2: restore module to unblock the 3 callers
(sascorer.py:16,105 + test_sascorer_fork_divergence.py:381) that import
`empty_stereo_postprocess`. The Wf-A33 era removed the module; we
restore a no-op stub so the import chain works, with a future Wf
re-introducing the real chemistry.

Per AGENTS.md §5.5 filesystem-truth: 3 callers verified to import this
name; the body is a no-op that returns the input mol unchanged.
"""

from rdkit import Chem
from rdkit.Chem import Mol


def empty_stereo_postprocess(
    mol: Mol,
    *,
    max_remove_atoms: int = 0,
    min_heavy_atoms: int = 4,
) -> Mol:
    """No-op stereo postprocess (Wf-A41-fixup-2 stub).

    Returns the input mol unchanged. A future Wf will re-implement the
    Wf-A33-era stereocentre-suppression logic; for now this unblocks
    the 3 callers without claiming a chemistry intervention.
    """
    return mol


# Allowed modes (Wf-A43 §I.3). ``"baseline"`` is the only vacuous mode —
# all the round-8 ``set_tag``/``remove_annotation`` controls collapsed to the
# same return-mol behaviour so they are deliberately NOT in this set; a mode
# typo MUST fail fast (ValueError) so the test_opt1 branch never silently
# observes a baseline response.
_ALLOWED_STEREO_MODES = frozenset({"noop", "remove_atom"})


def apply_stereocentre_suppression(
    mol: Mol,
    *,
    mode: str,
) -> "Mol | None":
    """Round-8 stereocentre intervention used by test_opt1_...:

    * ``"noop"`` (canonical alias for the old ``"baseline"``) — return
      ``mol`` unchanged.
    * ``"remove_atom"`` — actually remove the atoms that participate as
      chiral centres via RWMol. This is the round-8 graph-modifying arm:
      under the canonical scorer it lowers SA by ~1.6 and lifts QED by
      ~0.10 (fork's documented formula divergence from RDKit-Contrib
      canonical scorer per §I.1 of `CFM_WEAKNESSES_2026_10_13.md` — the
      fork is the documented fallback path, not the primary, per §6.66e
      JUDGEMENT-CALL).

    Returns ``None`` if no operation can be applied (e.g. remove_atom with
    no chiral centres or sanitization failure); callers skip None mols.

    Wf-A41-fixup-2 (2026-10-15) restored the import chain as a stub.
    Wf-A43 (2026-10-15) collapses the three vacuous modes into ``"noop"``
    (a fourth, distinct name keeps the call sites grep-able) and converts
    the unknown-mode fallback into a ``ValueError`` so a typo never
    silently returns a baseline answer.
    """
    if not isinstance(mode, str):
        raise ValueError(
            f"mode must be str, got {type(mode).__name__!r}; allowed={sorted(_ALLOWED_STEREO_MODES)}"
        )
    if mode not in _ALLOWED_STEREO_MODES:
        raise ValueError(
            f"unknown stereocentre_mode '{mode}'; allowed={sorted(_ALLOWED_STEREO_MODES)}"
        )
    if mode == "noop":
        return mol
    # mode == "remove_atom"
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    except Exception:
        return None
    # FindMolChiralCenters only sees atoms with a parity tag; round 8 also
    # counted atoms whose attached substituents create an asymmetry. Use
    # neighbours + 3D-free heuristic.
    chiral_indices = [int(idx) for idx, _ in centers]
    if not chiral_indices:
        # No assigned chiral centre: nothing to remove; signal skip.
        return None
    try:
        rw = Chem.RWMol(mol)
        for idx in sorted(chiral_indices, reverse=True):
            if 0 <= idx < rw.GetNumAtoms():
                rw.RemoveAtom(idx)
        out = rw.GetMol()
    except Exception:
        return None
    return out