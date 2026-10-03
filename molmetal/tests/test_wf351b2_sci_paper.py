"""Wf-351-B2 — unit tests for the SCI paper submission toolchain configuration.

Covers the contract documented in
``.tools/sci-paper/README.md`` + ``TODO/completed/351b2_zotero_setup.md``:

  1. pybtex-compile.py CLI round-trip:
     ``--bib`` -> ``--out`` produces a non-empty .bbl; ``--version`` reports
     pybtex version; missing bib file returns exit 1.
  2. references.bib entry schema — at least 6 entries, all with required
     fields (author, title, year), no duplicate citation keys.

Both tests are CPU-only (no GPU dependency); they invoke the wrapper as a
subprocess against the in-repo fixtures under ``.tools/sci-paper/refs/``.

Per AGENTS.md §1 + §5.2 honest framing: the wrapper uses upstream
``pybtex>=0.24`` (MIT). Project test depends on pybtex being installed in
the local venv (``.venv/bin/python``) — skip the test if pybtex is missing
to keep CI green on hosts without the optional dep.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / ".tools" / "sci-paper" / "bin" / "pybtex-compile.py"
BIB_FILE = REPO_ROOT / ".tools" / "sci-paper" / "refs" / "references.bib"
BTX_CONFIG = REPO_ROOT / ".tools" / "sci-paper" / "refs" / "better-bibtex-config.json"


def _python_with_pybtex() -> str | None:
    """Return path to a Python interpreter that has ``pybtex`` installed.

    Tries, in order: ``.venv/bin/python`` (project venv), ``sys.executable``
    (current interpreter).  Returns ``None`` if no usable interpreter is
    found, so the test can skip rather than fail when pybtex is absent.
    """
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python3",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            try:
                r = subprocess.run(
                    [str(c), "-c", "import pybtex; print(pybtex.__version__)"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    return str(c)
            except (OSError, subprocess.TimeoutExpired):
                continue
    return None


# ---------------------------------------------------------------------------
#  Test 1: pybtex-compile.py CLI works end-to-end
# ---------------------------------------------------------------------------


def test_pybtex_compile_cli_roundtrip(tmp_path: Path) -> None:
    """pybtex-compile.py --bib ... --out ... writes a non-empty .bbl."""
    python = _python_with_pybtex()
    if python is None:
        pytest.skip("pybtex not installed in any reachable interpreter")

    assert WRAPPER.exists(), f"wrapper not found: {WRAPPER}"
    assert BIB_FILE.exists(), f"bib fixture not found: {BIB_FILE}"

    out_bbl = tmp_path / "main.bbl"

    # 1a. happy path: real bib → real bbl
    result = subprocess.run(
        [python, str(WRAPPER), "--bib", str(BIB_FILE), "--out", str(out_bbl)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"wrapper exit {result.returncode}: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert out_bbl.exists(), "wrapper did not produce .bbl"
    bbl_text = out_bbl.read_text(encoding="utf-8")
    assert len(bbl_text) > 100, f".bbl too small ({len(bbl_text)} bytes)"
    assert "@article" in bbl_text or "@misc" in bbl_text, (
        "expected .bbl to contain at least one BibTeX entry"
    )

    # 1b. --version reports pybtex version
    ver_result = subprocess.run(
        [python, str(WRAPPER), "--version"],
        capture_output=True, text=True, timeout=10,
    )
    assert ver_result.returncode == 0
    assert "pybtex-compile" in ver_result.stdout
    assert re.search(r"pybtex \d+\.\d+", ver_result.stdout), (
        f"expected pybtex version in stdout, got: {ver_result.stdout!r}"
    )

    # 1c. missing bib file → exit 1 with stderr error
    err_result = subprocess.run(
        [python, str(WRAPPER), "--bib", "/nonexistent.bib", "--out", str(out_bbl)],
        capture_output=True, text=True, timeout=10,
    )
    assert err_result.returncode == 1, (
        f"expected exit 1 for missing bib, got {err_result.returncode}"
    )
    assert "not found" in err_result.stderr.lower(), (
        f"expected 'not found' in stderr, got: {err_result.stderr!r}"
    )


# ---------------------------------------------------------------------------
#  Test 2: references.bib has expected entry format
# ---------------------------------------------------------------------------


_BIB_ENTRY_RE = re.compile(
    r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE
)


def test_references_bib_format() -> None:
    """references.bib is a valid BibTeX file with ≥6 entries + JSON config sane."""
    assert BIB_FILE.exists(), f"bib fixture not found: {BIB_FILE}"

    bib_text = BIB_FILE.read_text(encoding="utf-8")
    entries = _BIB_ENTRY_RE.findall(bib_text)
    assert len(entries) >= 6, (
        f"expected ≥6 BibTeX entries, found {len(entries)}: {entries!r}"
    )

    # All keys unique (no duplicate citation keys)
    assert len(entries) == len(set(entries)), (
        f"duplicate citation keys in bib: "
        f"{[k for k in entries if entries.count(k) > 1]}"
    )

    # Every entry must have at least one author, title, and year field.
    # Quick heuristic: split into per-entry blocks and check each.
    entry_blocks = re.findall(
        r"@\w+\s*\{[^@]+?\n\}", bib_text, re.DOTALL,
    )
    assert len(entry_blocks) >= 6
    for block in entry_blocks:
        assert "title" in block.lower(), f"missing title in entry: {block[:80]}"
        assert "year" in block.lower(), f"missing year in entry: {block[:80]}"
        assert "author" in block.lower(), f"missing author in entry: {block[:80]}"

    # Wf-351-B2 SHIP-MARKER entry must be present (used by closeout audit)
    assert "wf351b2_ship2026" in bib_text, (
        "expected wf351b2_ship2026 ship-marker entry in references.bib"
    )

    # Better BibTeX config JSON must be valid + citeFormat=bibtex
    assert BTX_CONFIG.exists(), f"Better BibTeX config not found: {BTX_CONFIG}"
    config = json.loads(BTX_CONFIG.read_text(encoding="utf-8"))
    assert config["citeFormat"] == "bibtex", (
        f"expected citeFormat='bibtex', got {config['citeFormat']!r}"
    )
    assert config["autoExport"] == "idle", (
        f"expected autoExport='idle', got {config['autoExport']!r}"
    )
    assert len(config["extraExtensions"]) >= 1, (
        "expected ≥1 extraExtensions entry pointing to references.bib"
    )
