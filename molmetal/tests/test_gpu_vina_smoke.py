"""Phase-3F of WF-Remove-Smoke: routed test for QuickVina-GPU docking.

This test replaces ``scripts/smoke_vina_gpu_generated.py``.  The smoke
script invoked ``evaluate_generated_poses.py --engine gpu`` against a
synthetic batch.  The Phase-3F refactor routes that check into a proper
pytest test gated by ``@pytest.mark.gpu`` (skip-if-no-binary).

The new test exercises the VinaDockingAdapter with ``engine="qvina"``
(the CLI engine class used by quickvina-gpu paths) and verifies that
``_engine_dispatched`` correctly resolves to ``"subprocess"`` and the
adapter does not crash.  This is the load-bearing CLI-plumbing
assertion; real GPU docking is BLOCKED on this host (RX 7800 XT
gfx1101 firmware-level SMU hang per ``wf_gpu_diag/diagnosis.md``).

CPU-only: no GPU required.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. GPU adapter surface (CI-bounded budget)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_gpu_vina_engine() -> None:
    """Skip when no QuickVina-GPU binary is on ``$PATH``.

    When the GPU binary is available, ``VinaDockingAdapter(engine="qvina")``
    must dispatch to ``"subprocess"`` (the CLI path).  Real GPU docking
    is BLOCKED on this host (see ``wf_gpu_diag/diagnosis.md``).
    """
    gpu_bin = shutil.which("quickvina-gpu") or shutil.which("qvina-gpu")
    if gpu_bin is None:
        pytest.skip(
            "No QuickVina-GPU binary on $PATH; GPU docking test skipped. "
            "GPU is BLOCKED on this host (see wf_gpu_diag/diagnosis.md)."
        )

    try:
        from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"vina_adapter unavailable: {exc}")

    # The GPU binary is present; the adapter should resolve to subprocess.
    adapter = VinaDockingAdapter(engine="qvina")
    dispatched = adapter._engine_dispatched()
    assert dispatched == "subprocess"


# ---------------------------------------------------------------------------
# 2. Skip-if-no-binary contract
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_gpu_vina_skip_if_no_binary() -> None:
    """VinaDockingAdapter correctly reports engine dispatch per engine param.

    The contract: every supported engine constructs and reports a
    string ``_engine_dispatched`` value.  The actual resolved engine
    depends on which Vina-family binaries are on PATH.

    We exclude ``vina-cli`` and ``auto`` here because both require a
    launchable system ``vina`` binary; the standard vina/qvina/quickvina2
    engines are dispatch-only assertions.
    """
    try:
        from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"vina_adapter unavailable: {exc}")

    expected_dispatch = {
        "vina": "python",
        "qvina": "subprocess",
        "quickvina2": "subprocess",
    }

    for engine in expected_dispatch:
        adapter = VinaDockingAdapter(engine=engine)
        dispatched = adapter._engine_dispatched()
        assert dispatched == expected_dispatch[engine], (
            f"{engine}: expected {expected_dispatch[engine]!r}, "
            f"got {dispatched!r}"
        )
