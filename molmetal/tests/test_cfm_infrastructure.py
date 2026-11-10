"""WF-257-Fix-Infra — CFM infrastructure unit tests.

Three infrastructure blockers prevented CFM training from running end-to-end:

  Blocker 1: flow_matching source missing
    Loader expected ``{ref_repo_path}/flow_matching/__init__.py`` but the
    canonical clone path (parent + inner ``flow_matching/`` subdir) was
    not always discoverable from the caller's CWD.

  Blocker 2: _reference NAMESPACE confusion
    The ``_reference/__init__.py`` redirected ``__path__`` to the
    upstream Facebook clone, but the local shim submodules
    (``path/``, ``solver/``, ``utils/``) live in the LOCAL directory.
    Python then failed to resolve
    ``molmetal.adapters.flow_matching_lipman._reference.path`` with
    ``ModuleNotFoundError`` because the upstream Facebook clone does
    not contain a ``path/`` subdirectory.

  Blocker 3: ref_repo_path not passed by CFM script
    ``metallo_drug_smoke_retrain.py`` did not pass ``ref_repo_path``
    to ``LipmanFlowMatchingAdapter``, relying on the default
    ``'molmetal/references/flow_matching'``.  This made it impossible
    to override the upstream path from the CLI in environments where
    the canonical default doesn't resolve.

Each test below targets one blocker with a minimal, isolated check that
does not depend on GPU, real training data, or the Facebook upstream
network.  All three tests pass on CPU with the smoke-test configuration
of the adapter (hidden_dim=8, n_layers=1).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_flow_matching_modules() -> None:
    """Drop every cached ``molmetal.adapters.flow_matching_lipman.*`` entry.

    The conftest pre-loads some of these into ``sys.modules`` for tests
    without a Facebook clone; we want each test to start from a clean
    slate so we exercise the production code paths.
    """
    for name in list(sys.modules.keys()):
        if (
            name == "flow_matching"
            or name.startswith("flow_matching.")
            or name.startswith("molmetal.adapters.flow_matching_lipman")
        ):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# Blocker 1 — flow_matching source discoverable
# ---------------------------------------------------------------------------


def test_blocker_1_configure_reference_resolves_default_clone(tmp_path):
    """``configure_reference()`` finds the upstream ``flow_matching/__init__.py``.

    The loader expects ``{ref_repo_path}/flow_matching/__init__.py`` to
    exist.  The canonical mol-metal default at
    ``<project>/molmetal/references/flow_matching`` carries the inner
    ``flow_matching/__init__.py`` of the Facebook clone and so resolves
    the loader's ``root / "__init__.py"`` check.
    """
    _reset_flow_matching_modules()
    from molmetal.adapters.flow_matching_lipman import _reference_loader

    root = _reference_loader.configure_reference("molmetal/references/flow_matching")
    assert root.is_dir(), f"expected root dir, got {root}"
    assert (root / "__init__.py").is_file(), (
        f"expected {root}/__init__.py to exist — Facebook clone is "
        f"missing or in the wrong place"
    )
    # Sanity: the clone must carry the standard Facebook upstream modules.
    # Wf-A94 (2026-10-19) — they live one level DOWN (root/flow_matching/),
    # not at the root; ``configure_reference`` only guarantees
    # ``root/__init__.py`` (a symlink into the inner package).  The old
    # assertions looked at the outer root and failed on a complete clone.
    inner = root / "flow_matching"
    assert inner.is_dir(), f"expected {inner} to exist"
    assert (
        inner / "interpolation.py"
    ).is_file(), "expected flow_matching/interpolation.py — clone is incomplete"
    assert (
        inner / "sampler.py"
    ).is_file(), "expected flow_matching/sampler.py — clone is incomplete"


# ---------------------------------------------------------------------------
# Blocker 2 — _reference NAMESPACE confusion (local shim resolution)
# ---------------------------------------------------------------------------


def test_blocker_2_reference_path_module_importable_without_upstream_clone():
    """``_reference.path`` is importable via the LOCAL shim, not the upstream.

    The pre-fix code redirected ``_reference.__path__`` to the upstream
    Facebook clone, which broke the local ``path/``, ``solver/``,
    ``utils/`` shims because the upstream doesn't carry those
    subdirectories.  The fix removes the ``__path__`` redirection so
    Python uses the local shim directory; ``AffineProbPath`` is
    constructed via the local shim regardless of whether the upstream
    clone is reachable.
    """
    _reset_flow_matching_modules()
    # Importing ``_reference.path`` triggers the local shim, which in
    # turn calls ``configure_reference`` + imports from the upstream
    # alias (or the in-shim fallback if the upstream isn't loaded).
    from molmetal.adapters.flow_matching_lipman._reference.path import AffineProbPath
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver
    from molmetal.adapters.flow_matching_lipman._reference.utils import ModelWrapper

    # All three shim classes must be importable and instantiable.
    path = AffineProbPath()
    assert path.scheduler is None  # CondOTScheduler assigned in adapter.setup()
    solver = ODESolver()
    assert solver.n_steps >= 1
    wrapper = ModelWrapper(lambda x, t: torch.zeros_like(x))
    assert wrapper(torch.zeros(2, 3), torch.tensor(0.5)).shape == (2, 3)


def test_blocker_2_adapter_setup_resolves_through_local_shims():
    """Full ``LipmanFlowMatchingAdapter.setup()`` end-to-end without errors.

    This is the gate test for Blocker 2: a full adapter setup must
    resolve ``_reference.path`` (AffineProbPath), ``_reference.solver``
    (ODESolver), and ``_reference.utils`` (ModelWrapper) via the local
    shim, not the upstream Facebook clone.
    """
    _reset_flow_matching_modules()
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    adapter = LipmanFlowMatchingAdapter(hidden_dim=8, n_layers=1)
    # suppress the production-scale warning for the 8-d toy size
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter.setup("cpu")
    # All three legacy Lipman 2023 entry points must be wired up.
    assert adapter.path is not None
    assert adapter.scheduler is not None
    assert adapter._ODESolver is not None
    assert adapter._ModelWrapper is not None


# ---------------------------------------------------------------------------
# Blocker 3 — ref_repo_path CLI flag on the smoke retrain script
# ---------------------------------------------------------------------------


def test_blocker_3_ref_repo_path_cli_flag_passed_to_adapter():
    """``--ref-repo-path`` reaches ``LipmanFlowMatchingAdapter`` as ``ref_repo_path``.

    The retrain script now accepts a ``--ref-repo-path`` CLI flag and
    forwards it to the adapter constructor.  We exercise this by
    importing the script's ``main`` and inspecting the constructed
    adapter (via a stub).
    """
    _reset_flow_matching_modules()
    # Run the script in a subprocess with ``--ref-repo-path`` set; the
    # script's ``[metallo_smoke] ref_repo_path`` print confirms the flag
    # is forwarded.  We use a tiny config (steps=1, n-train=8,
    # hidden_dim=8) to keep wall time under the smoke-test budget.
    import subprocess

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    script = PROJECT_ROOT / "molmetal" / "scripts" / "metallo_drug_smoke_retrain.py"
    if not script.is_file():
        pytest.skip(f"missing retrain script: {script}")
    # Wf-A122 FIX 6 follow-up: this test's own docstring says it "does not
    # depend on GPU", but the smoke script resolves
    # ``device = "cuda" if torch.cuda.is_available() else "cpu"`` with no
    # override, so on a GPU host the subprocess allocates VRAM.
    # ``CUDA_VISIBLE_DEVICES=""`` + ``MOLMETAL_SMOKE_DEVICE=cpu`` make the
    # CPU contract explicit.  Same class as test_smoke_flag.py (already fixed)
    # and molmetal/tests/test_integration.py::test_005_end_to_end_smoke.
    _env = dict(os.environ)
    _env["CUDA_VISIBLE_DEVICES"] = ""
    _env["MOLMETAL_SMOKE_DEVICE"] = "cpu"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            "1",
            "--n-train",
            "8",
            "--batch-size",
            "4",
            "--hidden-dim",
            "8",
            "--n-layers",
            "1",
            "--decode-smoke-every",
            "1000",
            "--out-dir",
            "/tmp/wf257_infra_blocker3_smoke",
            "--ref-repo-path",
            str(PROJECT_ROOT / "molmetal" / "references" / "flow_matching"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_env,
    )
    # Look for the marker line in stdout — proves the flag was
    # forwarded to ``LipmanFlowMatchingAdapter``.
    assert "ref_repo_path =" in proc.stdout, (
        f"--ref-repo-path not forwarded to LipmanFlowMatchingAdapter; "
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    # And the script must have completed without crashing.
    assert proc.returncode == 0, (
        f"retrain script exited {proc.returncode}; " f"stderr:\n{proc.stderr}"
    )


def test_blocker_3_ref_repo_path_default_resolves_to_canonical():
    """Without ``--ref-repo-path``, the adapter uses the canonical default."""
    _reset_flow_matching_modules()
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    # Default behaviour (no ref_repo_path argument) — must still find
    # the upstream clone via the loader's DEFAULT_REFERENCE.
    adapter = LipmanFlowMatchingAdapter(hidden_dim=8, n_layers=1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter.setup("cpu")
    assert adapter._ref_repo == "molmetal/references/flow_matching"
    assert adapter.path is not None


# ---------------------------------------------------------------------------
# Backward-compat gate
# ---------------------------------------------------------------------------


def test_backward_compat_existing_caller_default_construction():
    """``LipmanFlowMatchingAdapter()`` with no kwargs constructs successfully.

    This is the existing call pattern used by
    ``molmetal/scripts/r10_cfg_real_crossdocked.py`` and many test
    scripts.  After the fix it must still construct + setup end-to-end
    without any new required arguments.
    """
    _reset_flow_matching_modules()
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    adapter = LipmanFlowMatchingAdapter()  # no kwargs — legacy call style
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adapter.setup("cpu")
    # The adapter's full machinery must be initialised.
    assert adapter.velocity_field is not None
    assert adapter.path is not None
    assert adapter.scheduler is not None
    assert adapter.optimizer is not None
    # Default constructor signature must still expose ref_repo_path as
    # the first positional arg (back-compat with ``LipmanFlowMatchingAdapter(
    # ref_repo_path=...)`` calls).
    import inspect

    sig = inspect.signature(LipmanFlowMatchingAdapter.__init__)
    assert "ref_repo_path" in sig.parameters
    assert (
        sig.parameters["ref_repo_path"].default == "molmetal/references/flow_matching"
    )
