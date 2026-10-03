"""Wf-372-A2 — Tectonic binary install verification.

Per the 2026-09-30 user standing grant ("Zotero + tectonic install可以做，
其实其他重新训练和论文以外的事情我都给你授权"), Wf-372-A2 installs the
tectonic binary on this host as part of the Tier-2 SCI submission infra
(docs/SCI_PAPER_SETUP.md §2).

Per Wf-351-A3 audit:
  - tectonic = MIT (modern Rust-based LaTeX engine)
  - Tectonic self-bundles TeX resources on first compile (network
    fetch from tectonic bundle server). Hermetic, reproducible across
    hosts.

Per AGENTS.md §1 + §5.2 honest framing: every assertion here traces to a
shell probe (``which``, ``tectonic --version``) or a file path on disk.
No fabricated / aspirational numbers.

Per TODO-84 + AGENTS.md §5.3: this test file does NOT touch ``paper/*``;
it only verifies the tectonic binary install + the corresponding setup
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

# MEASURED 2026-09-30 per Wf-372-A2: tectonic 0.17.0 installed at
# ~/.local/bin/tectonic (user-space install; downloaded from the
# official tectonic-typesetting/tectonic GitHub release 0.17.0 linux
# musl x86_64 binary tarball — the drop-sh.fullyjustified.net installer
# is unreachable from this host).
EXPECTED_TECTONIC_VERSION = "0.17.0"
EXPECTED_TECTONIC_BIN = Path("/home/hugo/.local/bin/tectonic")


# ---------------------------------------------------------------------------
#  Test 1: `which tectonic` returns a non-empty path on $PATH
# ---------------------------------------------------------------------------


def test_tectonic_on_path() -> None:
    """`which tectonic` must return a non-empty path after the Wf-372-A2
    install.

    Per the 2026-09-30 user standing grant, tectonic 0.17.0 is installed
    at ``~/.local/bin/tectonic`` (user-space, no sudo required).
    ``~/.local/bin`` is already on ``$PATH`` (verified via ``echo $PATH``
    — see ``env`` output: PATH=/home/linuxbrew/.linuxbrew/bin:...:
    /home/hugo/.local/bin:...).

    MEASURED 2026-09-30 00:07 UTC: ``which tectonic`` →
    ``/home/hugo/.local/bin/tectonic``.
    """
    tectonic_path = shutil.which("tectonic")
    assert tectonic_path, (
        "tectonic not found on PATH after Wf-372-A2 install — verify "
        "~/.local/bin/tectonic exists and is executable"
    )
    assert Path(tectonic_path).exists(), (
        f"which tectonic returned {tectonic_path!r} but the file does not exist"
    )
    # Tectonic binary is a static musl executable — must be a regular
    # file, not a directory or symlink-to-nothing.
    assert Path(tectonic_path).is_file(), (
        f"tectonic at {tectonic_path} is not a regular file"
    )


# ---------------------------------------------------------------------------
#  Test 2: `tectonic --version` reports the MEASURED 0.17.0 semver
# ---------------------------------------------------------------------------


def test_tectonic_version_reports_017() -> None:
    """``tectonic --version`` must include the MEASURED 0.17.0 semver.

    Tectonic prints ``Tectonic 0.17.0`` on stdout (no trailing
    copyright/extra). The semver is per the official
    ``tectonic-typesetting/tectonic`` GitHub release 2026-07-27.
    """
    tectonic_path = shutil.which("tectonic")
    if not tectonic_path:
        pytest.skip("tectonic not on PATH (install verification gate)")

    result = subprocess.run(
        [tectonic_path, "--version"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"tectonic --version failed: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # tectonic prints "Tectonic 0.17.0" — match the semver substring.
    assert EXPECTED_TECTONIC_VERSION in result.stdout, (
        f"expected tectonic semver {EXPECTED_TECTONIC_VERSION!r} in "
        f"stdout, got: {result.stdout!r}"
    )
    # Also assert the "Tectonic" branding to disambiguate from any
    # other tool that happens to print "0.17.0".
    assert "Tectonic" in result.stdout, (
        f"expected 'Tectonic' branding in stdout, got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
#  Test 3: docs/SCI_PAPER_SETUP.md §2 reflects the Wf-372-A2 install
# ---------------------------------------------------------------------------


def test_setup_doc_reflects_tectonic_install() -> None:
    """Per the Wf-372-A2 closeout, ``docs/SCI_PAPER_SETUP.md`` §2 must
    be updated to declare tectonic 0.17.0 as INSTALLED + add the
    GitHub-releases-tarball fallback row used on this host (since the
    drop-sh.fullyjustified.net installer is unreachable from this
    network) + the MEASURED ``tectonic --version`` line.

    This locks in the install audit so a future regression (binary
    accidentally removed) is caught by the docs-vs-binary gap.
    """
    assert SETUP_MD.exists(), f"setup doc not found at {SETUP_MD}"

    text = SETUP_MD.read_text(encoding="utf-8")

    # §2 must still exist (per Wf-351-A3 §2 of the original audit).
    assert re.search(r"##\s+§2\s+tectonic", text), (
        "§2 tectonic section missing from SCI_PAPER_SETUP.md"
    )

    # §2.1 must include the GitHub-releases-tarball fallback row used
    # on this host (drop-sh.fullyjustified.net was unreachable).
    sec2_match = re.search(
        r"##\s+§2\s+tectonic[\s\S]+?(?=\n##\s+§|\Z)", text,
    )
    assert sec2_match is not None, "could not locate §2 section"
    sec2 = sec2_match.group(0)

    assert EXPECTED_TECTONIC_VERSION in sec2, (
        f"§2 must reference tectonic {EXPECTED_TECTONIC_VERSION} "
        f"(MEASURED on 2026-09-30)"
    )
    assert "github.com/tectonic-typesetting" in sec2 or (
        "GitHub releases" in sec2 or "releases/download" in sec2
    ), (
        "§2 must document the GitHub-releases-tarball fallback row "
        "used on this host"
    )
    assert "~/.local/bin/tectonic" in sec2 or "local/bin/tectonic" in sec2, (
        "§2 must document the user-space install path used on this host"
    )

    # §2.4 (verify) must include the MEASURED `tectonic --version` line.
    sec2_4_match = re.search(
        r"###\s+§2\.4[\s\S]+?(?=\n###\s+§|\n##\s+§|\Z)", text,
    )
    assert sec2_4_match is not None, "could not locate §2.4 verify section"
    sec2_4 = sec2_4_match.group(0)
    assert "Tectonic 0.17.0" in sec2_4 or "tectonic 0.17.0" in sec2_4 or (
        "MEASURED" in sec2_4
    ), (
        "§2.4 must include the MEASURED `tectonic --version` line"
    )

    # The TL;DR §0 table must now declare tectonic as INSTALLED (not
    # INSTALL-REQUIRED).
    sec0_match = re.search(
        r"##\s+§0[\s\S]+?(?=\n##\s+§|\Z)", text,
    )
    assert sec0_match is not None, "could not locate §0 TL;DR section"
    sec0 = sec0_match.group(0)
    tectonic_row = re.search(
        r"\|\s*tectonic\s*\([^)]*\)\s*\|[^|]+\|[^|]+\|",
        sec0,
    )
    # Fallback regex if the parenthetical is missing
    if tectonic_row is None:
        tectonic_row = re.search(
            r"\|\s*tectonic\s*\|[^|]+\|[^|]+\|", sec0,
        )
    assert tectonic_row is not None, (
        "could not find tectonic row in §0 table"
    )
    row_text = tectonic_row.group(0)
    assert "INSTALLED" in row_text, (
        f"tectonic row should be INSTALLED now (per Wf-372-A2), "
        f"got: {row_text!r}"
    )
    assert "INSTALL-REQUIRED" not in row_text, (
        f"tectonic row should NOT still say INSTALL-REQUIRED, "
        f"got: {row_text!r}"
    )
