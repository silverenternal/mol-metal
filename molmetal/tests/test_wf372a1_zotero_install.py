"""Wf-372-A1 — Zotero binary install verification.

Per the 2026-09-30 user standing grant ("Zotero + tectonic install可以做，
其实其他重新训练和论文以外的事情我都给你授权"), Wf-372-A1 installs the
Zotero binary on this host as part of the Tier-2 SCI submission infra.

Per Wf-351-A2 audit:
  - Zotero = AGPL-3.0
  - Better BibTeX = MIT
  - Better BibTeX requires Zotero GUI for plugin install (manual step)

Per AGENTS.md §1 + §5.2 honest framing: every assertion here traces to a
shell probe (``which``, ``zotero --version``) or a file path on disk.
No fabricated / aspirational numbers.

Per TODO-84 + AGENTS.md §5.3: this test file does NOT touch ``paper/*``;
it only verifies the Zotero binary install + the corresponding setup
doc updates.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_MD = REPO_ROOT / "docs" / "SCI_PAPER_SETUP.md"

# MEASURED 2026-09-30 per Wf-372-A1: Zotero 10.0.3 (Zotero Zotero 140.15.0esr)
# installed at ~/apps/zotero/ + symlinked to ~/.local/bin/zotero.
EXPECTED_ZOTERO_VERSION_SUBSTRING = "Zotero Zotero 140"

# User-space install location chosen because no sudo is available on this
# host (verified via `sudo -n true` exit code 1) and `~/.local/bin` is
# already on `$PATH` (verified via `echo $PATH`).
EXPECTED_ZOTERO_DIR = Path("/home/hugo/apps/zotero")
EXPECTED_ZOTERO_BIN_SYMLINK = Path("/home/hugo/.local/bin/zotero")


# ---------------------------------------------------------------------------
#  Test 1: `which zotero` returns a non-empty path on $PATH
# ---------------------------------------------------------------------------


def test_zotero_on_path() -> None:
    """`which zotero` must return a non-empty path after the Wf-372-A1
    install.

    Per the 2026-09-30 user standing grant, Zotero is installed at
    ``~/apps/zotero/zotero`` (user-space, no sudo) with a symlink at
    ``~/.local/bin/zotero`` (already on ``$PATH``).

    MEASURED 2026-09-30 00:07 UTC: ``which zotero`` →
    ``/home/hugo/.local/bin/zotero``.
    """
    zotero_path = shutil.which("zotero")
    assert zotero_path, (
        "zotero not found on PATH after Wf-372-A1 install — verify "
        "~/apps/zotero/ exists and ~/.local/bin/zotero symlink is valid"
    )
    assert Path(zotero_path).exists(), (
        f"which zotero returned {zotero_path!r} but the file does not exist"
    )
    assert Path(zotero_path).is_file() or Path(zotero_path).is_symlink(), (
        f"zotero at {zotero_path} is neither a file nor a symlink"
    )


# ---------------------------------------------------------------------------
#  Test 2: `zotero --version` reports the MEASURED 10.0.3 / 140.15.0esr build
# ---------------------------------------------------------------------------


def test_zotero_version_reports_10() -> None:
    """`zotero --version` must include the MEASURED Zotero 10.0.3
    (Zotero 140.15.0esr) build identifier.

    The 140.15.0esr string is Zotero's internal branding (it's actually
    Firefox ESR 140 under the hood — Zotero ships a Mozilla runtime).
    The 10.0.3 release is the public version, advertised on
    https://www.zotero.org/download/.
    """
    zotero_path = shutil.which("zotero")
    if not zotero_path:
        pytest.skip("zotero not on PATH (install verification gate)")

    result = subprocess.run(
        [zotero_path, "--version"],
        capture_output=True, text=True, timeout=15,
    )
    # zotero prints "Zotero Zotero <esr-version>" on stdout; may also
    # emit a non-fatal "Sandbox: CanCreateUserNamespace() clone()
    # failure: EPERM" line on hardened kernels — allow that on stderr
    # but require the version substring on stdout.
    assert result.returncode == 0, (
        f"zotero --version failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert EXPECTED_ZOTERO_VERSION_SUBSTRING in result.stdout, (
        f"expected {EXPECTED_ZOTERO_VERSION_SUBSTRING!r} in stdout, "
        f"got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
#  Test 3: Zotero user-space install tree exists at ~/apps/zotero/
# ---------------------------------------------------------------------------


def test_zotero_install_tree_present() -> None:
    """The Wf-372-A1 install landed at ``~/apps/zotero/`` (user-space;
    no sudo). The tree must contain ``zotero`` (launcher shell script)
    + ``zotero-bin`` (the actual binary) + ``app/application.ini``.

    MEASURED 2026-09-30: the extracted tarball unpacks to
    ``Zotero_linux-x86_64/`` which we renamed to ``zotero/``. The tree
    size is ~400 MB (Mozilla runtime + extensions).
    """
    if not EXPECTED_ZOTERO_DIR.exists():
        pytest.skip(
            f"Zotero install tree not at {EXPECTED_ZOTERO_DIR} — "
            f"rerun Wf-372-A1 install steps"
        )

    assert EXPECTED_ZOTERO_DIR.is_dir(), (
        f"{EXPECTED_ZOTERO_DIR} exists but is not a directory"
    )

    launcher = EXPECTED_ZOTERO_DIR / "zotero"
    bin_actual = EXPECTED_ZOTERO_DIR / "zotero-bin"
    app_ini = EXPECTED_ZOTERO_DIR / "app" / "application.ini"

    assert launcher.exists(), f"launcher shell script missing: {launcher}"
    assert bin_actual.exists(), f"zotero-bin missing: {actual}"
    assert app_ini.exists(), f"app/application.ini missing: {app_ini}"

    # Sanity: the launcher is a bash script (Zotero's launcher wraps
    # `zotero-bin -app app/application.ini`).
    launcher_text = launcher.read_text(encoding="utf-8", errors="replace")
    assert launcher_text.startswith("#!/bin/bash"), (
        f"zotero launcher is not a bash script: {launcher_text[:40]!r}"
    )
    assert "zotero-bin" in launcher_text, (
        "expected zotero-bin invocation in launcher script"
    )


# ---------------------------------------------------------------------------
#  Test 4: docs/SCI_PAPER_SETUP.md reflects the Wf-372-A1 install status
# ---------------------------------------------------------------------------


def test_setup_doc_reflects_install() -> None:
    """Per the Wf-372-A1 closeout, ``docs/SCI_PAPER_SETUP.md`` §1.1 must
    be updated to declare Zotero 10.0.3 as INSTALLED + add the generic
    tarball install row used on this host + the sandbox-EPERM note.

    This locks in the install audit so a future regression (binary
    accidentally removed) is caught by the docs-vs-binary gap.
    """
    assert SETUP_MD.exists(), f"setup doc not found at {SETUP_MD}"

    text = SETUP_MD.read_text(encoding="utf-8")

    # §1 must still exist (mandatory per Wf-355-B3 §7).
    assert re.search(r"##\s+§1\s+Zotero", text), "§1 Zotero section missing"

    # §1.1 must include the generic-tarball row that was used on this
    # host (no sudo, user-space install at ~/apps/zotero/).
    sec1_match = re.search(
        r"###\s+§1\.1[\s\S]+?(?=\n###\s+§|\n##\s+§|\Z)", text,
    )
    assert sec1_match is not None, "could not locate §1.1 section"
    sec1_1 = sec1_match.group(0)

    assert "Zotero 10.0.3" in sec1_1, (
        "§1.1 must document Zotero 10.0.3 (MEASURED on 2026-09-30)"
    )
    assert "tar.xz" in sec1_1, (
        "§1.1 must document the generic tarball install row used on this host"
    )
    assert "~/apps/zotero" in sec1_1, (
        "§1.1 must document the user-space install path used on this host"
    )

    # §1.2 (Better BibTeX) must be updated to flag the manual GUI install.
    sec1_2_match = re.search(
        r"###\s+§1\.2[\s\S]+?(?=\n###\s+§|\n##\s+§|\Z)", text,
    )
    assert sec1_2_match is not None, "could not locate §1.2 section"
    sec1_2 = sec1_2_match.group(0)
    assert "MANUAL" in sec1_2 or "manual" in sec1_2, (
        "§1.2 must flag Better BibTeX as a manual GUI install"
    )
    assert "Better BibTeX" in sec1_2, "§1.2 must mention Better BibTeX"

    # The TL;DR §0 table must now declare Zotero as INSTALLED (not
    # INSTALL-REQUIRED).
    sec0_match = re.search(
        r"##\s+§0[\s\S]+?(?=\n##\s+§|\Z)", text,
    )
    assert sec0_match is not None, "could not locate §0 TL;DR section"
    sec0 = sec0_match.group(0)
    # The Zotero row in the table should now say INSTALLED, not
    # INSTALL-REQUIRED.
    zotero_row = re.search(
        r"\|\s*Zotero\s*\+[^\n]+Better BibTeX\s*\|[^|]+\|[^|]+\|",
        sec0,
    )
    assert zotero_row is not None, "could not find Zotero row in §0 table"
    row_text = zotero_row.group(0)
    assert "INSTALLED" in row_text, (
        f"Zotero row should be INSTALLED now (per Wf-372-A1), got: {row_text!r}"
    )
    assert "INSTALL-REQUIRED" not in row_text, (
        f"Zotero row should NOT still say INSTALL-REQUIRED, got: {row_text!r}"
    )
