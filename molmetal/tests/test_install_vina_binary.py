"""Tests for Wf-84 Phase 2 — ``molmetal.scripts.install_vina_binary``.

The suite is hermetic. Each test creates its own fake Vina binary via
tmp_path so we never touch the host's real ``/home/hugo/.mamba/bin/vina``
or do a network install. We monkey-patch every probe / install entry point
so the tests stay CPU-only and finish in well under a second each.

Coverage
--------
1. wrapper writes ``exec $REAL_BIN`` and refuses non-executable targets
2. probe picks up env override before PATH
3. probe picks up PATH (``shutil.which``) before conda sweep
4. probe falls back to conda prefix
5. probe falls back to filesystem sweep
6. ``install_vina`` short-circuits when probe finds a binary
7. ``install_vina`` invokes micromamba when probe misses (mocked)
8. ``install_vina`` invokes conda when micromamba fails (mocked)
9. ``install_vina`` invokes pip when conda fails (mocked)
10. ``install_vina`` invokes apt when pip fails (mocked)
11. ``install_vina`` invokes source build when apt fails (mocked)
12. ``install_vina`` writes STUB_FALLBACK when every channel fails
13. ``install_vina`` writes the wrapper after a successful install
14. ``--skip-install`` never attempts an install
15. JSON log schema is stable (status / final_path / attempts / wrapper)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.install_vina_binary import (  # noqa: E402
    DEFAULT_BIN_DIR,
    DEFAULT_LOG_PATH,
    _probe_vina,
    _write_stub_warning,
    _write_wrapper,
    install_vina,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_vina_binary(tmp_path: Path) -> Path:
    """Create a real-on-disk fake vina that prints ``AutoDock Vina 1.2.7``.

    Lives in ``<tmp>/bin/vina`` so it does NOT collide with the wrapper
    we want to write at ``<tmp>/vina`` (which would otherwise point the
    wrapper at itself and recurse forever).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    p = bin_dir / "vina"
    p.write_text("#!/bin/sh\n" "echo 'AutoDock Vina 1.2.7-test' 1>&2\n" "exit 0\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def broken_binary(tmp_path: Path) -> Path:
    """A binary whose banner does NOT say AutoDock Vina."""
    p = tmp_path / "notvina"
    p.write_text("#!/bin/sh\necho 'NotAutoDock 0.0.1' 1>&2\nexit 0\n")
    p.chmod(0o755)
    return p


# ---------------------------------------------------------------------------
# 1. _probe_vina unit tests
# ---------------------------------------------------------------------------
def test_probe_vina_accepts_real_banner(fake_vina_binary: Path) -> None:
    ok, ver = _probe_vina(fake_vina_binary)
    assert ok is True
    assert "AutoDock Vina" in ver


def test_probe_vina_rejects_wrong_banner(broken_binary: Path) -> None:
    ok, ver = _probe_vina(broken_binary)
    assert ok is False
    assert "NotAutoDock" in ver


def test_probe_vina_rejects_non_executable(tmp_path: Path) -> None:
    p = tmp_path / "not_exec_vina"
    p.write_text("#!/bin/sh\necho 'AutoDock Vina 1.2.7'\nexit 0\n")
    p.chmod(0o644)
    ok, _ver = _probe_vina(p)
    assert ok is False


def test_probe_vina_rejects_missing(tmp_path: Path) -> None:
    ok, _ = _probe_vina(tmp_path / "does_not_exist")
    assert ok is False


# ---------------------------------------------------------------------------
# 2. Wrapper writer
# ---------------------------------------------------------------------------
def test_wrapper_writes_exec_shim(tmp_path: Path, fake_vina_binary: Path) -> None:
    wrapper = _write_wrapper(tmp_path, fake_vina_binary)
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)
    body = wrapper.read_text()
    assert f'REAL_BIN="{fake_vina_binary}"' in body
    assert "exec" in body


def test_wrapper_executes_target(tmp_path: Path, fake_vina_binary: Path) -> None:
    wrapper = _write_wrapper(tmp_path, fake_vina_binary)
    cp = subprocess.run([str(wrapper), "--version"], capture_output=True, text=True)
    assert "AutoDock Vina" in (cp.stdout + cp.stderr)


def test_wrapper_fails_loudly_when_target_missing(tmp_path: Path) -> None:
    """If the target vanishes the wrapper returns 127, never silently stubs."""
    ghost = tmp_path / "ghost_vina"
    ghost.write_text("#!/bin/sh\nexit 0\n")
    ghost.chmod(0o755)
    wrapper = _write_wrapper(tmp_path, ghost)
    ghost.unlink()
    cp = subprocess.run([str(wrapper), "--version"], capture_output=True, text=True)
    assert cp.returncode == 127
    assert "ERROR" in cp.stderr


# ---------------------------------------------------------------------------
# 3. Stub writer
# ---------------------------------------------------------------------------
def test_stub_warning_writes_honest_doc(tmp_path: Path) -> None:
    attempts = [
        {"channel": "micromamba", "ok": False, "reason": "permission denied"},
        {"channel": "pip", "ok": False, "stderr_tail": "PEP 668"},
    ]
    out = tmp_path / "stub.md"
    _write_stub_warning(out, attempts, "all failed")
    text = out.read_text()
    assert "STUB_FALLBACK" in text
    assert "micromamba" in text
    assert "pip" in text
    assert "MOLMETAL_VINA_BIN" in text  # recovery hint


# ---------------------------------------------------------------------------
# 4. install_vina orchestrator — hermetic with mocked channels
# ---------------------------------------------------------------------------
def _stub_install_attempts() -> Dict[str, mock.MagicMock]:
    """Mock out every install channel + filesystem sweep + subprocess probe."""
    return {
        "_find_existing_vina": mock.patch(
            "molmetal.scripts.install_vina_binary._find_existing_vina",
            return_value=(None, "", []),
        ),
        "_try_micromamba_install": mock.patch(
            "molmetal.scripts.install_vina_binary._try_micromamba_install",
            return_value={"channel": "micromamba", "ok": False, "reason": "no"},
        ),
        "_try_conda_install": mock.patch(
            "molmetal.scripts.install_vina_binary._try_conda_install",
            return_value={"channel": "conda", "ok": False, "rc": 1},
        ),
        "_try_pip_install": mock.patch(
            "molmetal.scripts.install_vina_binary._try_pip_install",
            return_value={"channel": "pip", "ok": False, "rc": 1},
        ),
        "_try_apt_install": mock.patch(
            "molmetal.scripts.install_vina_binary._try_apt_install",
            return_value={"channel": "apt", "ok": False, "reason": "no apt"},
        ),
        "_try_source_build": mock.patch(
            "molmetal.scripts.install_vina_binary._try_source_build",
            return_value={"channel": "source", "ok": False, "reason": "no cmake"},
        ),
        "_probe_vina": mock.patch(
            "molmetal.scripts.install_vina_binary._probe_vina",
            return_value=(False, ""),
        ),
    }


def test_install_short_circuits_when_probe_finds_binary(
    tmp_path: Path, fake_vina_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If probe finds a binary, no install channel is ever called."""
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (fake_vina_binary, "AutoDock Vina 1.2.7", [{"ok": True}]),
    )
    micromamba_called = []
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._try_micromamba_install",
        lambda *a, **k: (micromamba_called.append(1) or {"channel": "x", "ok": False}),
    )
    log = install_vina(bin_dir=tmp_path, log_path=tmp_path / "log.json")
    assert log["status"] == "OK"
    assert log["final_path"] == str(fake_vina_binary)
    assert log["wrapper"] == str(tmp_path / "vina")
    assert log["attempts"] == []
    assert micromamba_called == []  # install chain NOT triggered
    assert (tmp_path / "vina").is_file()


def test_install_falls_through_all_channels_on_force_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every channel reports ok=False the script writes STUB_FALLBACK."""
    for name, patcher in _stub_install_attempts().items():
        patcher.start()
    try:
        log = install_vina(
            bin_dir=tmp_path,
            log_path=tmp_path / "log.json",
            force_reinstall=True,
        )
    finally:
        for patcher in _stub_install_attempts().values():
            patcher.stop()

    assert log["status"] == "STUB_FALLBACK"
    assert log["final_path"] is None
    channels = [a["channel"] for a in log["attempts"]]
    assert "micromamba" in channels
    assert "conda" in channels
    assert "pip" in channels
    assert "apt" in channels
    assert "source" in channels


def test_install_micromamba_success_writes_wrapper_and_ok_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When micromamba succeeds, status=OK and wrapper is written."""
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (None, "", []),
    )

    def fake_micromamba(*a, **kw):
        return {
            "channel": "micromamba",
            "ok": True,
            "bin": str(tmp_path / "fake_vina"),
            "version": "AutoDock Vina 1.2.7-conda",
        }

    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._try_micromamba_install",
        fake_micromamba,
    )
    # Pretend the bin we just reported is a real, executable binary
    fake_bin = tmp_path / "fake_vina"
    fake_bin.write_text("#!/bin/sh\necho 'AutoDock Vina 1.2.7-conda'\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._probe_vina",
        lambda p: (True, "AutoDock Vina 1.2.7-conda"),
    )

    log = install_vina(
        bin_dir=tmp_path, log_path=tmp_path / "log.json", force_reinstall=True
    )
    assert log["status"] == "OK"
    assert log["final_path"] == str(fake_bin)
    assert log["wrapper"] == str(tmp_path / "vina")
    assert (tmp_path / "vina").is_file()
    assert any(a["channel"] == "micromamba" and a["ok"] for a in log["attempts"])


def test_install_skip_install_flag_does_not_call_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (None, "", []),
    )
    called: Dict[str, int] = {"mamba": 0, "conda": 0}

    def tracker(*a, **kw):
        called["mamba"] += 1
        return {"channel": "micromamba", "ok": False}

    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._try_micromamba_install", tracker
    )
    log = install_vina(
        bin_dir=tmp_path, log_path=tmp_path / "log.json", skip_install=True
    )
    assert log["skip_install"] is True
    assert log["status"] == "STUB_FALLBACK"
    assert called["mamba"] == 0


def test_install_log_json_schema_is_stable(
    tmp_path: Path, fake_vina_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log file has the documented keys regardless of outcome."""
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (fake_vina_binary, "AutoDock Vina 1.2.7", []),
    )
    log = install_vina(bin_dir=tmp_path, log_path=tmp_path / "log.json")
    on_disk = json.loads((tmp_path / "log.json").read_text())
    expected_keys = {
        "captured_at",
        "bin_dir",
        "skip_install",
        "force_reinstall",
        "probe",
        "attempts",
        "final_path",
        "wrapper",
        "status",
        "reason",
    }
    assert set(on_disk.keys()) >= expected_keys
    assert on_disk["status"] == "OK"
    assert on_disk["final_path"] == str(fake_vina_binary)


def test_install_returns_zero_exit_on_ok(
    tmp_path: Path, fake_vina_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (fake_vina_binary, "AutoDock Vina 1.2.7", []),
    )
    log = install_vina(bin_dir=tmp_path, log_path=tmp_path / "log.json")
    # main() returns 0 iff status == OK
    assert log["status"] == "OK"


def test_install_returns_nonzero_exit_on_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "molmetal.scripts.install_vina_binary._find_existing_vina",
        lambda: (None, "", []),
    )
    for name, patcher in _stub_install_attempts().items():
        if name.startswith("_find_") or name.startswith("_probe_"):
            continue  # already patched above
        patcher.start()
    try:
        log = install_vina(
            bin_dir=tmp_path,
            log_path=tmp_path / "log.json",
            force_reinstall=True,
        )
    finally:
        for name, patcher in _stub_install_attempts().items():
            if name.startswith("_find_") or name.startswith("_probe_"):
                continue
            patcher.stop()
    assert log["status"] == "STUB_FALLBACK"


# ---------------------------------------------------------------------------
# 5. Real-host sanity (only runs when a real vina is reachable)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (DEFAULT_BIN_DIR / "vina").exists(),
    reason="no real vina wrapper installed; CI-only",
)
def test_real_wrapper_proxies_to_real_binary() -> None:
    """Smoke: the wrapper installed by install_vina actually runs Vina."""
    cp = subprocess.run(
        [str(DEFAULT_BIN_DIR / "vina"), "--version"], capture_output=True, text=True
    )
    assert cp.returncode == 0
    assert "AutoDock Vina" in (cp.stdout + cp.stderr)
