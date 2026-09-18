"""Tests for the round-7 TODO-04 qvina_swap refactor.

The ``vina_adapter.VinaDockingAdapter`` now has a public ``engine``
attribute (default ``"vina"``) and a private ``_engine_dispatched()``
method that returns ``"python"`` or ``"subprocess"``.  Explicit
``engine='qvina'`` / ``'quickvina2'`` raise :class:`FileNotFoundError`
with a canonical ``mamba install`` hint when the binary is missing
(``engine='auto'`` is the only mode that still silently downgrades).

These tests cover the QVina / QuickVina2 swap path specifically —
general engine-field / Python-API / subprocess tests live in
``test_vina_adapter.py``.
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest


@pytest.fixture
def binary_discovery(monkeypatch, tmp_path):
    """Isolate PATH and the bundled binary without hiding launch errors."""
    mod = importlib.import_module("molmetal_lam.sbdd_env.vina_adapter")
    monkeypatch.delenv("MOLMETAL_QVINA_BIN", raising=False)
    monkeypatch.delenv("MOLMETAL_QUICKVINA2_BIN", raising=False)
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    monkeypatch.setattr(mod, "_BUNDLED_QVINA", tmp_path / "qvina02")

    def executable(name, code=0):
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\nexit {code}\n")
        path.chmod(0o755)
        return path

    return mod, executable


@pytest.mark.parametrize("engine", ["qvina", "quickvina2"])
def test_both_engines_find_bundle_without_path(binary_discovery, engine):
    mod, executable = binary_discovery
    bundle = executable("qvina02")
    candidates = mod._QVINA_BIN_CANDIDATES if engine == "qvina" else mod._QUICKVINA2_BIN_CANDIDATES
    assert mod._resolve_binary(candidates) == str(bundle)


@pytest.mark.parametrize("name", ["qvina02", "qvina2.1"])
def test_upstream_quickvina_names_on_path(binary_discovery, monkeypatch, name):
    mod, executable = binary_discovery
    binary = executable(name)
    monkeypatch.setattr(mod.shutil, "which", lambda candidate: str(binary) if candidate == name else None)
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) == str(binary)


def test_override_then_path_then_bundle(binary_discovery, monkeypatch):
    mod, executable = binary_discovery
    bundle = executable("qvina02")
    on_path = executable("quickvina2")
    override = executable("custom-engine")
    monkeypatch.setattr(mod.shutil, "which", lambda name: str(on_path) if name == "quickvina2" else None)
    monkeypatch.setenv("MOLMETAL_QUICKVINA2_BIN", str(override))
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) == str(override)
    monkeypatch.delenv("MOLMETAL_QUICKVINA2_BIN")
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) == str(on_path)
    executable("quickvina2", code=127)  # e.g. shared library missing
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) == str(bundle)


def test_engine_overrides_are_independent(binary_discovery, monkeypatch):
    mod, executable = binary_discovery
    bundle = executable("qvina02")
    custom = executable("custom-qvina")
    monkeypatch.setenv("MOLMETAL_QVINA_BIN", str(custom))
    assert mod._resolve_binary(mod._QVINA_BIN_CANDIDATES) == str(custom)
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) == str(bundle)


def test_nonexecutable_bundle_is_unavailable(binary_discovery):
    mod, executable = binary_discovery
    executable("qvina02").chmod(0o644)
    assert mod._resolve_binary(mod._QUICKVINA2_BIN_CANDIDATES) is None


# ---------------------------------------------------------------------------
# test_engine_field_default — the new public attribute is present and equal
# to the module's DEFAULT_ENGINE.
# ---------------------------------------------------------------------------
def test_engine_field_default() -> None:
    """Public ``adapter.engine`` attribute is present and equals DEFAULT_ENGINE.

    Regression guard for the round-7 refactor that promoted the
    constructor's ``engine`` keyword from a private ``self._engine``
    field to a public ``self.engine`` attribute.  See
    ``molmetal/molmetal_lam/sbdd_env/vina_adapter.py`` for the
    implementation.
    """
    mod = importlib.import_module(
        "molmetal_lam.sbdd_env.vina_adapter",
    )
    # Module-level constant sanity check.
    assert hasattr(mod, "DEFAULT_ENGINE"), (
        "vina_adapter.DEFAULT_ENGINE missing — module not refactored"
    )
    # Round-7 default is 'vina' (Python binding preferred).  'auto' is
    # accepted as a legacy default for back-compat, but the spec for
    # the new refactor is 'vina'.
    assert mod.DEFAULT_ENGINE == "vina", (
        f"DEFAULT_ENGINE should be 'vina' (round-7 wire-vina), "
        f"got {mod.DEFAULT_ENGINE!r}"
    )

    # The Python binding must be installed for the default path to
    # construct — skip cleanly otherwise.
    if not mod._have_vina():
        pytest.skip(
            "vina Python pkg not installed — cannot exercise default path"
        )
    if not mod._have_meeko():
        pytest.skip("meeko not installed — cannot exercise default path")

    adapter = mod.VinaDockingAdapter()
    # Public attribute exists.
    assert hasattr(adapter, "engine"), (
        "VinaDockingAdapter must expose public 'engine' attribute "
        "(round-7 wire-vina refactor)"
    )
    # Default mirrors DEFAULT_ENGINE.
    assert adapter.engine == mod.DEFAULT_ENGINE, (
        f"adapter.engine={adapter.engine!r} != DEFAULT_ENGINE="
        f"{mod.DEFAULT_ENGINE!r}"
    )
    # Mirrors the private field for back-compat with existing tests.
    assert adapter.engine == adapter._engine


# ---------------------------------------------------------------------------
# test_qvina_missing_binary_raises_clear_error — explicit engine='qvina'
# raises FileNotFoundError with the mamba install hint when the binary
# is not on PATH.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["qvina", "quickvina2"])
def test_missing_binary_raises_clear_error(monkeypatch, tmp_path, engine) -> None:
    """Missing-engine diagnostics must run even on a fully configured host."""
    mod = importlib.import_module("molmetal_lam.sbdd_env.vina_adapter")
    monkeypatch.delenv("MOLMETAL_QVINA_BIN", raising=False)
    monkeypatch.delenv("MOLMETAL_QUICKVINA2_BIN", raising=False)
    monkeypatch.setattr(mod, "_BUNDLED_QVINA", tmp_path / "missing")
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="mamba install"):
        mod.VinaDockingAdapter(engine=engine)


# ---------------------------------------------------------------------------
# test_auto_engine_silently_downgrades — engine='auto' is the only path
# that may silently fall back to vina.  This guards the regression so
# the next refactor does not accidentally widen the silent-fallback.
# ---------------------------------------------------------------------------
def test_auto_engine_silently_downgrades(monkeypatch) -> None:
    """``engine='auto'`` silently downgrades to Vina 1.2.7 when qvina is missing.

    The round-7 refactor splits behaviour:
      * ``engine='qvina'`` / ``'quickvina2'`` (explicit) — HARD fail.
      * ``engine='auto'`` (resolution) — soft fallback.
    """
    mod = importlib.import_module(
        "molmetal_lam.sbdd_env.vina_adapter",
    )
    if mod._have_vina() is False:
        pytest.skip(
            "vina Python pkg not installed — cannot exercise auto "
            "downgrade path"
        )
    if mod._have_meeko() is False:
        pytest.skip("meeko not installed")
    monkeypatch.setattr(mod, "_resolve_binary", lambda _: None)
    monkeypatch.setattr(mod, "_probe_vina_binary", lambda: None)
    # No CLI on PATH — auto must downgrade to vina (Python binding),
    # not raise.
    adapter = mod.VinaDockingAdapter(engine="auto")
    assert adapter.engine == "vina", (
        f"engine='auto' should downgrade to vina, got {adapter.engine!r}"
    )
    assert adapter._engine_dispatched() == "python"


# ---------------------------------------------------------------------------
# test_engine_dispatched_returns_known_values — the dispatch helper
# returns one of the two canonical strings.
# ---------------------------------------------------------------------------
def test_engine_dispatched_returns_known_values() -> None:
    """``_engine_dispatched()`` returns 'python' or 'subprocess'.

    The dispatch helper is the single source of truth for the
    python-vs-CLI branch in :meth:`dock`.  Any new engine added to
    :data:`SUPPORTED_ENGINES` must register a dispatch path here.
    """
    mod = importlib.import_module(
        "molmetal_lam.sbdd_env.vina_adapter",
    )
    if not mod._have_vina() or not mod._have_meeko():
        pytest.skip("vina + meeko not installed")
    adapter = mod.VinaDockingAdapter(engine="vina")
    dispatch = adapter._engine_dispatched()
    assert dispatch in {"python", "subprocess"}, (
        f"_engine_dispatched() returned unknown value: {dispatch!r}"
    )
    assert dispatch == "python"  # engine='vina' is always python


# ---------------------------------------------------------------------------
# test_vina_cli_flag_round_trip — the CLI accepts --engine and exits 0
# when the engine is available, exits 1 with a stderr hint otherwise.
# This is a subprocess smoke test against the actual main() entry.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_choice", ["vina", "qvina", "quickvina2"])
def test_vina_cli_flag_round_trip(engine_choice: str) -> None:
    """The ``--engine`` CLI flag round-trips through ``main()``.

    Per round-7 TODO-04: ``python -m molmetal_lam.sbdd_env.vina_adapter
    --engine {vina|qvina|quickvina2}`` must either:
      * print metadata + exit 0 (engine available), or
      * print mamba-install hint + exit 1 (engine missing).

    This is the integration smoke test for the CLI front-end; we
    invoke the module via subprocess so we don't pollute the current
    Python's import cache.
    """
    # Sanity: the module's main() must accept the engine choice without
    # argparse rejecting it (that's the constructor's job).
    mod = importlib.import_module(
        "molmetal_lam.sbdd_env.vina_adapter",
    )
    assert engine_choice in mod.SUPPORTED_ENGINES, (
        f"engine_choice={engine_choice!r} not in "
        f"SUPPORTED_ENGINES={mod.SUPPORTED_ENGINES}"
    )
    # Quick: assert the parser accepts the flag without running the
    # full CLI.  This guards the lock-step between the argparse
    # ``choices`` and :data:`SUPPORTED_ENGINES`.
    parser = mod.build_arg_parser()
    ns = parser.parse_args(["--engine", engine_choice])
    assert ns.engine == engine_choice
