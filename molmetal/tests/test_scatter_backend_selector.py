"""Unit tests for the RDNA3-aware scatter backend selector.

Per ``molmetal/reports/triton_integration_r0.md``:

  Triton ``aggregate_vectors`` is 3.0-3.5x slower than
  ``torch.index_add_`` on gfx1101 RDNA3 at every shape the Mol-Metal
  stack exercises.  ``models._scatter`` must therefore default to the
  torch path on RDNA3 and only fall back to Triton when the shape is
  large enough that the gap narrows.

These tests pin the selector's decision matrix so future changes cannot
silently regress the dispatch.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _enable_triton_for_selector(monkeypatch):
    """Force the selector to assume ``aggregate_vectors`` is importable.

    ``_pick_backend`` is the policy layer — it must work even when the
    Triton kernel isn't loaded (the integration report measures both
    backends).  Clear the override and turn the kernel on.
    """
    from models import _scatter

    monkeypatch.setattr(_scatter, "_TRITON_AGGREGATE", object())
    monkeypatch.setattr(_scatter, "_BACKEND_OVERRIDE", None)
    _scatter._cached_pick_backend.cache_clear()
    yield
    _scatter._cached_pick_backend.cache_clear()


def _pick(device_cap, n_atoms, rbf_dim):
    from models._scatter import _cached_pick_backend

    return _cached_pick_backend(device_cap, int(n_atoms), int(rbf_dim))


def test_rdna3_small_shape_uses_torch():
    """(512 atoms, 32-dim RBF) is well below the threshold → torch wins."""
    assert _pick((11, 0), 512, 32) == "torch"


def test_rdna3_large_shape_uses_triton():
    """(8192 atoms, 2048-dim RBF) is the report's shape → triton wins."""
    assert _pick((11, 0), 8192, 2048) == "triton"


def test_cdna_small_shape_uses_triton():
    """CDNA defaults to Triton unless the shape is tiny."""
    assert _pick((9, 0), 512, 32) == "triton"


def test_cdna_micro_shape_uses_torch():
    """Below 1024 elements the kernel-launch cost beats Triton."""
    assert _pick((9, 0), 8, 8) == "torch"


def test_rdna3_boundary_just_below_threshold():
    """Just below the (n_atoms * rbf_dim >= 1_000_000) gate → torch."""
    # 999 * 1001 = 999_999 (just under 1e6).
    assert _pick((11, 0), 999, 1001) == "torch"


def test_rdna3_boundary_just_above_threshold():
    """At the threshold (1_000_000) → triton."""
    # 1000 * 1000 = 1_000_000.
    assert _pick((11, 0), 1000, 1000) == "triton"


def test_override_takes_precedence():
    """``set_scatter_backend('triton')`` beats the policy decision."""
    from models import _scatter

    _scatter.set_scatter_backend("triton")
    try:
        assert _pick((11, 0), 16, 16) == "triton"
    finally:
        _scatter.set_scatter_backend(None)


def test_override_clears_back_to_policy():
    """``set_scatter_backend(None)`` restores auto-dispatch."""
    from models import _scatter

    _scatter.set_scatter_backend("torch")
    _scatter.set_scatter_backend(None)
    assert _pick((11, 0), 512, 32) == "torch"


def test_pick_is_cached(monkeypatch):
    """Repeated calls must hit the lru_cache (no repeated policy eval).

    We assert this by patching the underlying ``_pick_backend`` with a
    counter and calling the memoised ``_cached_pick_backend`` twice
    with the same args.
    """
    from models import _scatter

    call_count = {"n": 0}
    original = _scatter._pick_backend

    def spy(device_cap, n_atoms, rbf_dim):
        call_count["n"] += 1
        return original(device_cap, n_atoms, rbf_dim)

    monkeypatch.setattr(_scatter, "_pick_backend", spy)
    _scatter._cached_pick_backend.cache_clear()
    _scatter._cached_pick_backend((11, 0), 512, 32)
    _scatter._cached_pick_backend((11, 0), 512, 32)
    _scatter._cached_pick_backend((11, 0), 512, 32)
    assert call_count["n"] == 1


def test_scatter_backend_string_reflects_gpu():
    """The module-level ``scatter_backend`` constant should be a string."""
    from models import _scatter

    assert isinstance(_scatter.scatter_backend, str)
    assert _scatter.scatter_backend in ("torch", "triton")
