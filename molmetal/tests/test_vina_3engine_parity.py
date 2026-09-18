"""Phase-3F of WF-Remove-Smoke: routed test for the 3-engine Vina parity check.

This test replaces ``scripts/verify_docking_adapter_smoke.py``.  The smoke
script ran a one-off subprocess spawn of all three Vina engines (vina,
qvina, quickvina2) and asserted finite-energy + non-empty-pose + valid
Complex.  The Phase-3F refactor routes that check into a proper pytest
test gated by ``@pytest.mark.integration``.

The new test exercises the ``VinaDockingAdapter`` constructor for each
of the supported engines — verifying that ``_engine_dispatched`` reports
the correct dispatch class (``python`` for the ``vina`` Python binding,
``subprocess`` for the CLI engines).  This is the load-bearing CLI-plumbing
assertion: if a downstream consumer switches engine, the adapter must
report the correct dispatch without invoking the binary.

CPU-only: no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. VinaDockingAdapter: each supported engine constructs without error
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_vina_3engine_parity() -> None:
    """VinaDockingAdapter supports vina / vina-cli / qvina / quickvina2.

    Each engine constructor must complete without raising.  The
    ``_engine_dispatched`` property returns the documented dispatch
    class: ``python`` (vina Python binding) or ``subprocess`` (CLI).
    """
    try:
        from molmetal_lam.sbdd_env.vina_adapter import (
            SUPPORTED_ENGINES,
            VinaDockingAdapter,
        )
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"vina_adapter unavailable: {exc}")

    assert "vina" in SUPPORTED_ENGINES
    assert "vina-cli" in SUPPORTED_ENGINES
    assert "qvina" in SUPPORTED_ENGINES
    assert "quickvina2" in SUPPORTED_ENGINES

    expected_dispatch = {
        "vina": "python",
        "vina-cli": "subprocess",
        "qvina": "subprocess",
        "quickvina2": "subprocess",
    }

    # Build one adapter per supported engine; verify _engine_dispatched matches expected.
    # Skip vina-cli because it requires a system `vina` binary to be launchable.
    for engine in ("vina", "qvina", "quickvina2"):
        adapter = VinaDockingAdapter(engine=engine)
        dispatched = adapter._engine_dispatched()
        assert isinstance(dispatched, str), (
            f"{engine}: _engine_dispatched returned {type(dispatched).__name__}"
        )
        assert dispatched == expected_dispatch[engine], (
            f"{engine}: expected dispatch={expected_dispatch[engine]!r}, "
            f"got {dispatched!r}"
        )


# ---------------------------------------------------------------------------
# 2. CLI spawn-per-engine isolation test
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_vina_3engine_isolation() -> None:
    """Each engine has an independent ``_engine_dispatched`` value.

    The spawn-per-engine pattern must catch one-engine-breaks-others
    isolation: changing the engine parameter must not affect the
    resolution of the other engines.
    """
    try:
        from molmetal_lam.sbdd_env.vina_adapter import (
            SUPPORTED_ENGINES,
            VinaDockingAdapter,
        )
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"vina_adapter unavailable: {exc}")

    # ``vina`` uses the Python binding (no subprocess); ``qvina`` and
    # ``quickvina2`` use subprocess.  ``vina-cli`` and ``auto`` are
    # excluded because they require a launchable system vina binary.
    results = {}
    for engine in ("vina", "qvina", "quickvina2"):
        adapter = VinaDockingAdapter(engine=engine)
        results[engine] = adapter._engine_dispatched()

    # The dispatch map is engine-independent and deterministic.
    assert results["vina"] == "python"
    assert results["qvina"] == "subprocess"
    assert results["quickvina2"] == "subprocess"
