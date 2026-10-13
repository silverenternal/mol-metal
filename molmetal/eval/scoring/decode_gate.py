"""Per-atom decoded-validity gate for CFM-generated SMILES.

Track E (Wf-60231cff-03e-15, m14-admet-paper-align) — addresses the
m14 LEAD pathology that ``RDKit-strict`` alone (parse SMILES +
``Chem.SanitizeMol``) silently accepts multi-fragment survivors such
as ``C.C.C.C`` (4 disconnected CH4 fragments, but RDKit parses it
fine and the molecule sanitises).  The CFG-level bond-graph
connectivity (JTVAE-style, ``ConnectivityAwareDecoder``) catches the
predictor-side failure but the assembly-side failure lives here.

Three gates, applied in order:

1. ``Chem.MolFromSmiles(smi)`` — parse.
2. ``Chem.SanitizeMol(mol)`` — valence / aromaticity / implicit-H check.
3. :func:`_largest_fragment_frac` — ``max(GetMolFrags(...).GetNumAtoms()) / N``
   with a threshold ``largest_frac_min`` (default 0.80, the standard
   MOSES / GuacaMol acceptance floor).
4. Minimum atoms per molecule ``min_atoms`` (default 2; rejects
   ``[Pt]``, ``[He]``, etc., i.e. single-atom "molecules").

Defaults preserve the legacy RDKit-strict-only behaviour:
``largest_frac_min=0.0`` and ``min_atoms=0`` means "no per-atom filter,
RDKit-strict only" — bit-exact backward compat with the previous
``_decode_with_rdkit_strict`` callers.

Lit anchors
-----------
- Polykovskiy 2020 *MOSES* (arXiv:1811.01457) — validity = sanitised
  AND single connected fragment AND at least one heavy atom.
- Brown 2019 *GuacaMol* (J. Cheminform. 11:49) — validity =
  sanitised AND passes RDKit ``MolVS`` basic checks.
- Jin 2018 *JTVAE* (ICML, arXiv:1802.04364) — "valid molecule has
  exactly one connected component over its covalent-bond graph."

Honest framing
--------------
This module is a CPU-only post-check.  It does NOT change the
velocity field or the bond head.  It only re-classifies accepted
samples.  When most surviving samples are multi-fragment (e.g. a
generator that learnt to emit lone-atom rejects), the per-atom
decode_ratio will drop sharply (from 0.99...  RDKit-only to a
0.4-0.6 per-atom-aligned floor).  This is the correct behaviour
for honest validation: the failure was always there; it now shows.

All public functions return ``DecomposedResult`` so callers can
differentiate a connectivity reject from a parse reject from a
min-atoms reject without re-running the gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from rdkit import Chem  # noqa: F401
    from rdkit.Chem import Descriptors, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    """Decomposed verdict of :func:`gate_smiles`.

    Attributes
    ----------
    ok : bool
        ``True`` iff the molecule passed all three gates.
    reason : Optional[str]
        ``None`` when ``ok=True``.  Otherwise a short human-readable
        reason string.  Recognised prefixes:
          - ``"parse_failed"`` — Chem.MolFromSmiles returned None.
          - ``"sanitize_failed"`` — Chem.SanitizeMol raised.
          - ``"empty_mol"`` — 0 heavy atoms after sanitise.
          - ``"min_atoms"`` — n_heavy_atoms < ``min_atoms``.
          - ``"multi_fragment"`` — largest_frac < ``largest_frac_min``.
          - ``"rdkit_unavailable"`` — RDKit import failed.
    largest_frac : float
        ``max(GetMolFrags.GetNumAtoms()) / max(n_heavy_atoms, 1)``;
        ``0.0`` if a prior gate failed before this could be computed.
        Range [0.0, 1.0]; 1.0 = single fragment.
    n_heavy_atoms : int
        ``mol.GetNumAtoms()`` (counting ``*`` & isotope / aromatic
        C/N/O/...; explicit H counts).  ``0`` if a prior gate failed.
    sanitized : bool
        ``True`` iff RDKit ``SanitizeMol`` ran without exception.
        Diagnostic only — does NOT gate ``ok`` alone.
    """

    ok: bool
    reason: Optional[str] = None
    largest_frac: float = 0.0
    n_heavy_atoms: int = 0
    sanitized: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _largest_fragment_frac(mol) -> Tuple[float, int]:
    """Compute (largest_frac, n_heavy_atoms) on a sanitised RDKit Mol.

    Returns
    -------
    (largest_frac, n_heavy_atoms)
        ``largest_frac = max(f.GetNumAtoms() for f in GetMolFrags) / max(N, 1)``.
        For ``n_heavy_atoms == 0`` returns ``(0.0, 0)`` (degenerate).
        For ``n_heavy_atoms == 1`` returns ``(1.0, 1)`` (single atom is
        its own largest fragment).
    """
    if mol is None:
        return 0.0, 0
    n = int(mol.GetNumAtoms())
    if n == 0:
        return 0.0, 0
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        # Sanitise stripped all fragments (rare); fall back to ``N == 1``.
        return (1.0 if n == 1 else 0.0), n
    largest = max((int(f.GetNumAtoms()) for f in frags), default=0)
    return float(largest) / float(max(n, 1)), int(n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def gate_smiles(
    smi: Optional[str],
    *,
    largest_frac_min: float = 0.0,
    min_atoms: int = 0,
) -> GateResult:
    """Apply the per-atom validity gate to a single SMILES string.

    Parameters
    ----------
    smi : str or None
        Candidate SMILES.  ``None`` / empty / unparseable → ``ok=False``
        with ``reason="parse_failed"``.
    largest_frac_min : float, default 0.0
        If ``> 0``, reject molecules whose largest connected fragment
        has fewer than ``largest_frac_min * N`` atoms.  ``0.0``
        preserves the legacy RDKit-strict-only behaviour (no per-atom
        fragment gate).  Standard MOSES / GuacaMol floor: ``0.80``.
    min_atoms : int, default 0
        Reject molecules with fewer than ``min_atoms`` heavy atoms.
        ``0`` preserves legacy behaviour (no min-atoms gate).  Standard
        MOSES floor: ``2``.

    Returns
    -------
    :class:`GateResult`
        Decomposed verdict.  See class docstring.

    Notes
    -----
    Setting both ``largest_frac_min=0.0`` and ``min_atoms=0`` is
    bit-exact equivalent to the pre-Track-E "RDKit-strict only"
    gate (``Chem.MolFromSmiles → SanitizeMol → return True``).
    """
    if not _RDKIT_OK:
        return GateResult(
            ok=False,
            reason="rdkit_unavailable",
            largest_frac=0.0,
            n_heavy_atoms=0,
            sanitized=False,
        )
    if smi is None or not isinstance(smi, str) or not smi.strip():
        return GateResult(ok=False, reason="parse_failed")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return GateResult(ok=False, reason="parse_failed")
    # Sanitise first; capture failure reason.
    sanitized = False
    try:
        Chem.SanitizeMol(mol)
        sanitized = True
    except Exception:
        return GateResult(
            ok=False,
            reason="sanitize_failed",
            largest_frac=0.0,
            n_heavy_atoms=int(mol.GetNumAtoms()),
            sanitized=False,
        )
    # Per-atom gates.
    largest_frac, n_atoms = _largest_fragment_frac(mol)
    if n_atoms == 0:
        return GateResult(
            ok=False,
            reason="empty_mol",
            largest_frac=0.0,
            n_heavy_atoms=0,
            sanitized=True,
        )
    if min_atoms > 0 and n_atoms < min_atoms:
        return GateResult(
            ok=False,
            reason="min_atoms",
            largest_frac=largest_frac,
            n_heavy_atoms=n_atoms,
            sanitized=True,
        )
    if largest_frac_min > 0 and largest_frac < largest_frac_min:
        return GateResult(
            ok=False,
            reason="multi_fragment",
            largest_frac=largest_frac,
            n_heavy_atoms=n_atoms,
            sanitized=True,
        )
    return GateResult(
        ok=True,
        reason=None,
        largest_frac=largest_frac,
        n_heavy_atoms=n_atoms,
        sanitized=True,
    )


def gate_ratio(
    smiles: list,
    *,
    largest_frac_min: float = 0.0,
    min_atoms: int = 0,
) -> dict:
    """Aggregate :func:`gate_smiles` over a cohort.

    Parameters
    ----------
    smiles : list[str or None]
        Candidate SMILES (may contain ``None`` / ``""``).
    largest_frac_min, min_atoms
        Forwarded to :func:`gate_smiles`.

    Returns
    -------
    dict
        Schema::

            {
              "n_input": int,
              "n_ok": int,                 # ok=True
              "n_total": int,              # len(smiles)
              "per_atom_validity": float, # n_ok / n_total (0.0 on n_total=0)
              "mean_largest_frac": float, # over ok=True subset
              "mean_n_atoms": float,      # over ok=True subset
              "multi_frag_rejects": int,  # reason == "multi_fragment"
              "min_atoms_rejects": int,   # reason == "min_atoms"
              "parse_failures": int,      # reason == "parse_failed"
              "sanitize_failures": int,   # reason == "sanitize_failed"
              "largest_frac_min": float,
              "min_atoms": int,
            }

    NOTE (Wf-A30-4 D4, 2026-10-14, doc-only): this function is called
    TWICE in :func:`molmetal.eval.scoring.paper_table.build_paper_table` —
    first with the project-strict params (largest_frac_min, min_atoms from
    the caller's defaults — historically 0.80 / 2), then with
    ``largest_frac_min=0.0, min_atoms=0`` as the RDKit-strict proxy.  When
    the first call uses NON-ZERO strict params, the second call is a
    STRICT SUPERSET of the first: dropping both thresholds can only widen
    the ok=True population.  The ``rdkit_strict_proxy`` field is therefore
    identical to ``per_atom_validity`` once strict params are non-zero,
    and the pair is emitted as separate fields for backward-compat with
    the ``paper_table.md`` JSON contract (m14 row schema).  This is by
    DESIGN — do NOT silence the duplicate call as a "no-op": the field
    is part of the on-disk contract and downstream readers index into it
    by name.  See HONEST_NEGATIVE_CATALOG.md §6.52f.
    """
    n_total = len(smiles)
    if n_total == 0:
        return {
            "n_input": 0,
            "n_ok": 0,
            "n_total": 0,
            "per_atom_validity": 0.0,
            "mean_largest_frac": 0.0,
            "mean_n_atoms": 0.0,
            "multi_frag_rejects": 0,
            "min_atoms_rejects": 0,
            "parse_failures": 0,
            "sanitize_failures": 0,
            "largest_frac_min": float(largest_frac_min),
            "min_atoms": int(min_atoms),
        }
    n_ok = 0
    multi = 0
    min_a = 0
    parse_f = 0
    san_f = 0
    lfs: list = []
    nas: list = []
    for smi in smiles:
        r = gate_smiles(smi, largest_frac_min=largest_frac_min, min_atoms=min_atoms)
        if r.ok:
            n_ok += 1
            lfs.append(r.largest_frac)
            nas.append(r.n_heavy_atoms)
        elif r.reason == "multi_fragment":
            multi += 1
        elif r.reason == "min_atoms":
            min_a += 1
        elif r.reason == "parse_failed":
            parse_f += 1
        elif r.reason == "sanitize_failed":
            san_f += 1
    return {
        "n_input": n_total,
        "n_ok": int(n_ok),
        "n_total": int(n_total),
        "per_atom_validity": float(n_ok) / float(n_total),
        "mean_largest_frac": float(sum(lfs) / len(lfs)) if lfs else 0.0,
        "mean_n_atoms": float(sum(nas) / len(nas)) if nas else 0.0,
        "multi_frag_rejects": int(multi),
        "min_atoms_rejects": int(min_a),
        "parse_failures": int(parse_f),
        "sanitize_failures": int(san_f),
        "largest_frac_min": float(largest_frac_min),
        "min_atoms": int(min_atoms),
    }


AVAILABLE: bool = _RDKIT_OK

__all__ = [
    "GateResult",
    "gate_smiles",
    "gate_ratio",
    "_largest_fragment_frac",
    "AVAILABLE",
]


# ---------------------------------------------------------------------------
# Smoke (per §5.7)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """Self-test: returns 0 on PASS, 1 on FAIL.

    Tests:
      1. ``C.C.C.C`` (4 disconnected CH4 fragments) → ok=False with
         ``reason="multi_fragment"`` at ``largest_frac_min=0.80``.
      2. Same molecule → ok=True at ``largest_frac_min=0.0`` (legacy
         RDKit-strict backward compat).
      3. ``CCO`` (ethanol) → ok=True at the standard floor.
      4. ``[He]`` → ok=False with ``reason="min_atoms"`` at ``min_atoms=2``.
      5. ``nonsense_smiles`` → ok=False with ``reason="parse_failed"``.
      6. ``gate_ratio`` aggregation arithmetic.
    """
    if not _RDKIT_OK:
        print("[decode_gate] SKIP — RDKit unavailable")
        return 2
    # Test 1
    r1 = gate_smiles("C.C.C.C", largest_frac_min=0.80, min_atoms=2)
    assert r1.ok is False and r1.reason == "multi_fragment", r1
    assert r1.sanitized is True, r1  # RDKit does sanitise C.C.C.C
    # Test 2: backward compat
    r2 = gate_smiles("C.C.C.C", largest_frac_min=0.0, min_atoms=0)
    assert r2.ok is True, r2
    # Test 3: standard molecule
    r3 = gate_smiles("CCO", largest_frac_min=0.80, min_atoms=2)
    assert r3.ok is True, r3
    # Test 4: single-atom reject
    r4 = gate_smiles("[He]", largest_frac_min=0.0, min_atoms=2)
    assert r4.ok is False and r4.reason == "min_atoms", r4
    # Test 5: unparseable
    r5 = gate_smiles("nonsense_smiles")
    assert r5.ok is False and r5.reason == "parse_failed", r5
    # Test 6: gate_ratio
    gr = gate_ratio(["CCO", "C.C.C.C", "nonsense", "[He]", "CC"],
                    largest_frac_min=0.80, min_atoms=2)
    assert gr["n_total"] == 5, gr
    # CCO ok, CC ok, others not
    assert gr["n_ok"] == 2, gr
    assert abs(gr["per_atom_validity"] - 0.4) < 1e-9, gr
    assert gr["multi_frag_rejects"] == 1, gr
    assert gr["min_atoms_rejects"] == 1, gr
    assert gr["parse_failures"] == 1, gr
    # Test 7: empty cohort
    gr0 = gate_ratio([], largest_frac_min=0.80)
    assert gr0["n_total"] == 0 and gr0["per_atom_validity"] == 0.0, gr0
    print(f"[decode_gate] SMOKE PASS  7/7")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_smoke())
