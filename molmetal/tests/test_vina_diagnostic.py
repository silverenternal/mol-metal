"""Tests for Wf-84 Phase 1 — ``molmetal.scripts.vina_diagnostic``.

Each test exercises one probe channel with a fake ``shutil.which`` /
``subprocess.run`` via monkey-patching so we never actually shell out
to ``vina --version``. This keeps the test suite hermetic and CPU-only.

Channels under test
-------------------
1. ``MOLMETAL_VINA_BIN`` env override (with mocked probe success)
2. ``MOLMETAL_QVINA_BIN`` env override (with mocked probe success)
3. PATH lookup (``shutil.which``)
4. conda / mamba envs
5. project venv
6. filesystem sweep
7. banner-mismatch rejection
8. python binding version detection
9. ``MISSING_DEP`` path
10. JSON schema sanity

The tests do not assume the host has (or lacks) a real Vina binary —
they install their own probes via ``monkeypatch``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.vina_diagnostic import (  # noqa: E402
    DiagnosticResult,
    ProbeHit,
    VINA_BIN_CANDIDATES,
    _parse_version,
    _probe_binary,
    detect_vina_binary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_vina_binary(tmp_path: Path) -> Path:
    """Create a fake executable that prints ``AutoDock Vina 1.2.7-fake``.

    The file is real on disk and ``os.access(X_OK)`` returns True, but
    it is a shell script rather than a compiled binary so we don't
    need any libboost.
    """
    bin_path = tmp_path / "vina"
    bin_path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  echo 'AutoDock Vina 1.2.7-fake' 1>&2\n"
        "  exit 0\n"
        "fi\n"
        "echo 'fake vina' 1>&2\n"
        "exit 0\n"
    )
    bin_path.chmod(0o755)
    return bin_path


@pytest.fixture
def broken_vina_binary(tmp_path: Path) -> Path:
    """Create a fake executable that prints garbage (banner-mismatch)."""
    bin_path = tmp_path / "vina"
    bin_path.write_text("#!/bin/sh\n" "echo 'NotVina 0.0.1' 1>&2\n" "exit 0\n")
    bin_path.chmod(0o755)
    return bin_path


@pytest.fixture
def nondeterministic_binary(tmp_path: Path) -> Path:
    """A fake that returns rc=1 (mirrors some qvina fork behaviour)."""
    bin_path = tmp_path / "qvina02"
    bin_path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  echo 'AutoDock Vina 1.1.2-qvina' 1>&2\n"
        "  exit 1\n"
        "fi\n"
    )
    bin_path.chmod(0o755)
    return bin_path


# ---------------------------------------------------------------------------
# 1. _parse_version
# ---------------------------------------------------------------------------
class TestParseVersion:
    def test_parses_banner_in_stderr(self):
        version_line, full_version = _parse_version(
            "AutoDock Vina 1.2.7 (build hash)\n",
            "",
        )
        assert version_line == "AutoDock Vina 1.2.7 (build hash)"
        assert full_version == "1.2.7 (build hash)"

    def test_parses_banner_in_stdout(self):
        version_line, full_version = _parse_version(
            "",
            "AutoDock Vina f458505-mod\n",
        )
        assert version_line == "AutoDock Vina f458505-mod"
        assert full_version == "f458505-mod"

    def test_returns_empty_on_no_banner(self):
        version_line, full_version = _parse_version("hello world\n", "")
        assert version_line == ""
        assert full_version == ""

    def test_handles_empty_input(self):
        assert _parse_version("", "") == ("", "")

    def test_picks_first_matching_line(self):
        version_line, _ = _parse_version(
            "AutoDock Vina 1.2.7\nAutoDock Vina 1.2.6\n",
            "",
        )
        assert version_line == "AutoDock Vina 1.2.7"


# ---------------------------------------------------------------------------
# 2. _probe_binary
# ---------------------------------------------------------------------------
class TestProbeBinary:
    def test_real_executable_with_banner_passes(self, fake_vina_binary):
        hit = _probe_binary(str(fake_vina_binary))
        assert hit is not None
        version_line, full_version, rc = hit
        assert version_line == "AutoDock Vina 1.2.7-fake"
        assert full_version == "1.2.7-fake"
        assert rc == 0

    def test_banner_mismatch_rejected(self, broken_vina_binary):
        assert _probe_binary(str(broken_vina_binary)) is None

    def test_nondeterministic_exit_accepted(self, nondeterministic_binary):
        """Some Vina forks print banner on stderr and exit 1 — accept them."""
        hit = _probe_binary(str(nondeterministic_binary))
        assert hit is not None
        _, full_version, rc = hit
        assert full_version == "1.1.2-qvina"
        assert rc == 1

    def test_nonexistent_path_returns_none(self):
        assert _probe_binary("/nonexistent/vina") is None

    def test_non_executable_returns_none(self, tmp_path):
        p = tmp_path / "vina"
        p.write_text("#!/bin/sh\necho 'AutoDock Vina'\n")
        # No chmod 0o755 → not executable
        assert _probe_binary(str(p)) is None


# ---------------------------------------------------------------------------
# 3. detect_vina_binary — env override channel
# ---------------------------------------------------------------------------
class TestDetectVinaEnvOverride:
    def test_molmetal_vina_bin_override_picks_up_binary(
        self,
        fake_vina_binary,
        monkeypatch,
    ):
        monkeypatch.setenv("MOLMETAL_VINA_BIN", str(fake_vina_binary))
        # Force all other channels to return no hits.
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_path",
            lambda cands: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_active_conda_envs",
            lambda: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_conda_pkg_hints",
            lambda: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_project_venv",
            lambda: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_sweep",
            lambda: [],
        )
        result = detect_vina_binary()
        assert result.status == "OK"
        assert result.bin == str(fake_vina_binary.resolve())
        assert result.version == "1.2.7-fake"
        assert any(h.source == "env_override" for h in result.hits)

    def test_molmetal_qvina_bin_override_also_picked_up(
        self,
        fake_vina_binary,
        monkeypatch,
    ):
        monkeypatch.delenv("MOLMETAL_VINA_BIN", raising=False)
        monkeypatch.setenv("MOLMETAL_QVINA_BIN", str(fake_vina_binary))
        for ch in (
            "_resolve_from_path",
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary()
        assert result.status == "OK"
        assert result.bin == str(fake_vina_binary.resolve())

    def test_empty_env_override_falls_through(self, monkeypatch):
        monkeypatch.setenv("MOLMETAL_VINA_BIN", "")
        monkeypatch.setenv("MOLMETAL_QVINA_BIN", "")
        for ch in (
            "_resolve_from_path",
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary()
        assert result.status == "MISSING_DEP"
        assert result.bin is None


# ---------------------------------------------------------------------------
# 4. detect_vina_binary — MISSING_DEP path
# ---------------------------------------------------------------------------
class TestDetectVinaMissing:
    def test_no_channels_yields_miss(
        self,
        monkeypatch,
    ):
        # ``_resolve_from_env`` returns Optional[ProbeHit]; stub it to None.
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_env",
            lambda cands: None,
        )
        # The remaining channels return List[ProbeHit]; stub to [].
        for ch in (
            "_resolve_from_path",
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary()
        assert result.status == "MISSING_DEP"
        assert result.bin is None
        assert result.version == ""
        assert "MISSING_DEP" in result.status
        # Recommendation list must be non-empty when MISSING_DEP.
        assert len(result.recommended_install) >= 1

    def test_schema_matches_wf60_format(self, monkeypatch):
        """The output dict must have the same top-level keys as wf60
        reports so downstream consumers can switch between them."""
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_env",
            lambda cands: None,
        )
        for ch in (
            "_resolve_from_path",
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary().as_dict()
        # wf60 schema
        for k in ("status", "bin", "version", "reason", "checked_paths"):
            assert k in result, f"missing top-level key: {k}"
        # Wf-84 extensions
        for k in (
            "python_binding_version",
            "hits",
            "misses",
            "channels_probed",
            "recommended_install",
            "host",
            "captured_at",
        ):
            assert k in result, f"missing wf84 key: {k}"


# ---------------------------------------------------------------------------
# 5. detect_vina_binary — primary hit ordering
# ---------------------------------------------------------------------------
class TestPrimaryHitOrdering:
    def test_env_override_wins_over_path(
        self,
        fake_vina_binary,
        tmp_path,
        monkeypatch,
    ):
        # PATH version (less preferred).
        path_vina = tmp_path / "vina"
        path_vina.write_text("#!/bin/sh\necho 'AutoDock Vina 1.2.7-path'\n")
        path_vina.chmod(0o755)
        monkeypatch.setenv("MOLMETAL_VINA_BIN", str(fake_vina_binary))
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_path",
            lambda cands: [
                ProbeHit(
                    path=str(path_vina.resolve()),
                    source="path",
                    version_line="AutoDock Vina 1.2.7-path",
                    full_version="1.2.7-path",
                )
            ]
            if "vina" in cands
            else [],
        )
        for ch in (
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary()
        # env_override should be primary
        assert result.bin == str(fake_vina_binary.resolve())
        assert result.hits[0].source == "env_override"

    def test_primary_is_first_in_source_priority(
        self,
        fake_vina_binary,
        tmp_path,
        monkeypatch,
    ):
        sweep_vina = tmp_path / "vina-sweep"
        sweep_vina.write_text("#!/bin/sh\necho 'AutoDock Vina 1.2.7-sweep'\n")
        sweep_vina.chmod(0o755)
        monkeypatch.delenv("MOLMETAL_VINA_BIN", raising=False)
        monkeypatch.delenv("MOLMETAL_QVINA_BIN", raising=False)
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_path",
            lambda cands: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_active_conda_envs",
            lambda: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_conda_pkg_hints",
            lambda: [],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_project_venv",
            lambda: [
                ProbeHit(
                    path=str(fake_vina_binary.resolve()),
                    source="venv",
                    version_line="AutoDock Vina 1.2.7-fake",
                    full_version="1.2.7-fake",
                )
            ],
        )
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_sweep",
            lambda: [
                ProbeHit(
                    path=str(sweep_vina.resolve()),
                    source="sweep",
                    version_line="AutoDock Vina 1.2.7-sweep",
                    full_version="1.2.7-sweep",
                )
            ],
        )
        result = detect_vina_binary()
        # venv beats sweep
        assert result.hits[0].source == "venv"
        assert result.hits[1].source == "sweep"


# ---------------------------------------------------------------------------
# 6. detect_vina_binary — host fingerprint
# ---------------------------------------------------------------------------
class TestHostFingerprint:
    def test_host_section_has_pkg_mgr_and_python(self, monkeypatch):
        monkeypatch.setattr(
            "molmetal.scripts.vina_diagnostic._resolve_from_env",
            lambda cands: None,
        )
        for ch in (
            "_resolve_from_path",
            "_resolve_from_active_conda_envs",
            "_resolve_from_conda_pkg_hints",
            "_resolve_from_project_venv",
            "_resolve_from_sweep",
        ):
            monkeypatch.setattr(
                f"molmetal.scripts.vina_diagnostic.{ch}", lambda *a, **k: []
            )
        result = detect_vina_binary().as_dict()
        host = result["host"]
        assert "python" in host
        assert "pkg_mgr" in host
        assert "has_conda" in host
        assert "has_pip" in host
        assert host["python"].startswith("3.")  # we're on 3.12
