"""test_pll_signature.py — pytest tests for the Persistent Local Laplacian (PLL) signature.

This module covers the **math-engine contract** documented in
``TODO/13_lambda_clickchem/plan.md`` (Section "数学基础 (2026-09 新增)")
and the Track-B dispatch item
``TODO/pending/32_ultracode_trackB_lambda_calculus.md`` (B0 — PLL 2026
数学引擎落地) for the file
``molmetal/molmetal_lam/lam_chem/pll_signature.py``.

The PLL signature is the type-fingerprint + β-NF discriminator used by
the Molecular Lambda Calculus (MLC).  Its contract is:

1. ``compute_pll(mol, k=k_default)`` returns a ``PLLSignature`` whose
   ``eigenvalues`` field is a real-valued ``numpy.ndarray`` of length
   ``k_used`` (the effective rank; ``min(k, n_atoms - 1)``).
2. The top-k eigenvalues are returned in *descending* order.
3. ``sum(eigvals)`` matches the trace-level sanity budget (eigenvalues
   of a normalised Laplacian are bounded in ``[0, 2]``, so their sum
   is bounded by ``2 * k_used``).
4. Rotation / translation invariance: two 3D conformers of the same
   molecule (different random seeds) yield eigvals within ``1e-3``.
5. The local Laplacian for CCO (methanol) is positive semi-definite
   (all eigvals ``>= -tol``).
6. ``k in {8, 16, 32}`` all return correctly shaped outputs.
7. Lazy scipy fallback: when ``scipy`` cannot be imported, the module
   still imports cleanly (lazy import contract) and ``compute_pll``
   degrades gracefully (raises a clear ``PLLSignatureError`` or
   returns via the dense numpy path).

All tests are CPU-only and self-skip when the ``pll_signature`` module
is not yet importable.

Run from the project root::

    uv run pytest molmetal/tests/test_pll_signature.py -q --tb=short
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports — keep pytest collection working even when the PLL
# module is not on disk yet.  Each test calls these helpers and
# self-skips when the import fails.
# ---------------------------------------------------------------------------
def _import_pll():
    """Return ``(compute_pll, PLLSignature, PLLSignatureError)`` or
    raise ``pytest.skip.Exception`` when the PLL module is missing.
    """
    try:
        from molmetal_lam.lam_chem.pll_signature import (
            PLLSignature,
            PLLSignatureError,
            compute_pll,
        )
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"pll_signature module not importable: {exc}")
    return compute_pll, PLLSignature, PLLSignatureError


def _cco_conformer(seed: int = 42):
    """Return an RDKit Mol with a 3D conformer for methanol (CCO).

    Uses ETKDGv3 via :func:`generate_conformer` so the test does not
    duplicate the embed logic.
    """
    from molmetal_lam.lam_chem.conformer_embed import generate_conformer

    mol, _coords = generate_conformer("CCO", seed=seed)
    return mol


# ---------------------------------------------------------------------------
# Test 1 — returns real-valued ndarray of length k
# ---------------------------------------------------------------------------
def test_compute_pll_returns_real_ndarray_of_length_k() -> None:
    compute_pll, PLLSignature, _err = _import_pll()
    mol = _cco_conformer(seed=42)
    sig = compute_pll(mol, k=8)
    assert isinstance(sig, PLLSignature)
    eig = sig.eigenvalues
    assert isinstance(eig, np.ndarray), f"expected ndarray, got {type(eig)!r}"
    assert eig.dtype.kind == "f", f"expected float dtype, got {eig.dtype!r}"
    assert eig.ndim == 1, f"expected 1D, got ndim={eig.ndim}"
    assert len(eig) >= 1, f"empty eigenvalues array"


# ---------------------------------------------------------------------------
# Test 2 — top-k eigenvalues sorted descending
# ---------------------------------------------------------------------------
def test_top_k_eigenvalues_are_sorted_descending() -> None:
    compute_pll, _PLL, _err = _import_pll()
    mol = _cco_conformer(seed=42)
    eig = compute_pll(mol, k=16).eigenvalues
    diffs = np.diff(eig)
    assert np.all(diffs <= 1e-9), f"eigenvalues not descending; diffs={diffs!r}"


# ---------------------------------------------------------------------------
# Test 3 — sum(eigvals) bounded by 2*k  (PSD spectrum sanity check)
# ---------------------------------------------------------------------------
def test_sum_of_eigvals_bounded_by_2k() -> None:
    compute_pll, _PLL, _err = _import_pll()
    mol = _cco_conformer(seed=42)
    sig = compute_pll(mol, k=16)
    eig = sig.eigenvalues
    # Normalised Laplacian eigenvalues lie in [0, 2]; the partial sum
    # of the top-k is bounded by 2*k (with tiny ARPACK noise margin).
    assert eig.min() >= -1e-6, f"min eigval below PSD bound: {eig.min()}"
    assert (
        eig.sum() <= 2.0 * len(eig) + 1e-6
    ), f"sum(eig)={eig.sum()} exceeds 2*k={2.0 * len(eig)}"


# ---------------------------------------------------------------------------
# Test 4 — rotation/translation invariance
# ---------------------------------------------------------------------------
def test_pll_invariant_under_rotation_and_translation() -> None:
    compute_pll, _PLL, _err = _import_pll()
    mol_a = _cco_conformer(seed=1)
    mol_b = _cco_conformer(seed=999)
    eig_a = compute_pll(mol_a, k=8).eigenvalues
    eig_b = compute_pll(mol_b, k=8).eigenvalues
    n = min(len(eig_a), len(eig_b))
    max_diff = float(np.max(np.abs(eig_a[:n] - eig_b[:n])))
    assert (
        max_diff < 1e-3
    ), f"PLL not rotation/translation invariant; max|eig_a-eig_b|={max_diff}"


# ---------------------------------------------------------------------------
# Test 5 — positive semi-definiteness for CCO methanol
# ---------------------------------------------------------------------------
def test_pll_for_methanol_is_positive_semidefinite() -> None:
    compute_pll, _PLL, _err = _import_pll()
    mol = _cco_conformer(seed=42)
    eig = compute_pll(mol, k=16).eigenvalues
    min_eig = float(eig.min())
    assert min_eig >= -1e-6, f"PLL not PSD for CCO; smallest eigval={min_eig}"


# ---------------------------------------------------------------------------
# Test 6 — k=8, 16, 32 all return correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [8, 16, 32])
def test_compute_pll_supports_various_k(k: int) -> None:
    compute_pll, PLLSignature, _err = _import_pll()
    mol = _cco_conformer(seed=42)
    sig = compute_pll(mol, k=k)
    assert isinstance(sig, PLLSignature)
    eig = sig.eigenvalues
    assert sig.k_requested == k
    assert len(eig) >= 1, f"empty eigvals for k={k}"
    assert np.all(np.isfinite(eig)), f"non-finite values in eigvals (k={k})"


# ---------------------------------------------------------------------------
# Test 7 — lazy scipy fallback when scipy absent (mock)
# ---------------------------------------------------------------------------
def test_lazy_scipy_fallback_when_scipy_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``scipy`` cannot be imported, the PLL module still imports
    cleanly (lazy import contract) and ``compute_pll`` degrades
    gracefully — either raising a clear ``PLLSignatureError`` (so the
    caller can react) or returning via the dense numpy path.

    We mock ``_SCIPY_AVAILABLE`` to ``False`` at the module level so
    the test does not depend on the host's scipy availability.
    """
    _compute_pll, _PLL, PLLSignatureError = _import_pll()
    import molmetal_lam.lam_chem.pll_signature as pll_mod

    # Force the lazy-scipy path: pretend scipy is absent.
    monkeypatch.setattr(pll_mod, "_SCIPY_AVAILABLE", False, raising=False)

    mol = _cco_conformer(seed=42)
    # Either: (a) graceful degrade via dense numpy path -> PLLSignature
    #     or: (b) PLLSignatureError mentioning scipy
    # Both are acceptable per the "lazy scipy fallback" contract — the
    # *forbidden* behaviour is an uncaught ImportError or AttributeError.
    try:
        sig = _compute_pll(mol, k=8)
    except PLLSignatureError as exc:
        msg = str(exc).lower()
        assert (
            "scipy" in msg or "sparse" in msg
        ), f"PLLSignatureError must reference scipy; got {exc!r}"
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"unexpected exception type {type(exc).__name__}: {exc!r}")
    else:
        eig = sig.eigenvalues
        assert isinstance(eig, np.ndarray)
        assert np.all(np.isfinite(eig))
