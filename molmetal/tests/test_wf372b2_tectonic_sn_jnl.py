"""Wf-372-B2: tectonic + sn-jnl end-to-end compile test.

Per Wf-372-A2: tectonic installed. Per Wf-355-B1: sn-jnl template at
.tools/latex/sn-jnl.cls. This test verifies that tectonic can compile
a minimal sn-jnl document end-to-end and produce a valid PDF.

Patches applied to sn-jnl.cls (backup at .tools/latex/sn-jnl.cls.bak.wf372b2):
  1. \\allowdisplaybreaks wrapped in \\providecommand{...}{} (mathtools/amsmath
     dependency — original template assumes mathtools loaded).
  2. \\headerps@out provided as no-op (breakurl/hyperref firstpage shipout
     hook on newer LaTeX kernel expects this macro but sn-jnl.cls only sets
     up \\ps@headings pagestyle).

Test runs the compile in a tmp dir, asserts PDF produced, verifies it is a
real PDF (magic bytes %PDF-).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LATEX_DIR = REPO_ROOT / ".tools" / "latex"
SN_JNL_CLS = LATEX_DIR / "sn-jnl.cls"


def _have_tectonic() -> bool:
    return shutil.which("tectonic") is not None


@pytest.mark.skipif(not _have_tectonic(), reason="tectonic not installed")
def test_tectonic_sn_jnl_compile_produces_valid_pdf():
    """Compile a minimal sn-jnl document with tectonic and verify PDF output."""
    assert SN_JNL_CLS.is_file(), f"sn-jnl.cls missing at {SN_JNL_CLS}"

    tex_source = (
        r"\documentclass[springer]{sn-jnl}" "\n"
        r"\begin{document}" "\n"
        r"Test article." "\n"
        r"\end{document}" "\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tex_file = tmp_path / "wf372b2_smoke.tex"
        tex_file.write_text(tex_source, encoding="utf-8")
        # Copy sn-jnl.cls into tmp so tectonic finds it via kpsewhich
        shutil.copy(SN_JNL_CLS, tmp_path / "sn-jnl.cls")

        proc = subprocess.run(
            ["tectonic", "--keep-logs", str(tex_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        pdf_path = tmp_path / "wf372b2_smoke.pdf"
        assert pdf_path.is_file(), (
            f"tectonic did not produce PDF. exit={proc.returncode}\n"
            f"stdout={proc.stdout[-1000:]}\n"
            f"stderr={proc.stderr[-1000:]}"
        )
        # Verify PDF magic bytes
        with pdf_path.open("rb") as f:
            magic = f.read(5)
        assert magic[:4] == b"%PDF", f"not a valid PDF, magic={magic!r}"
        size_kb = pdf_path.stat().st_size / 1024
        assert size_kb > 1.0, f"PDF unexpectedly small: {size_kb:.2f} KiB"
