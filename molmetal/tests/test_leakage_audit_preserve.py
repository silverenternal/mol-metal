"""The leakage audit must not delete sections it does not own.

Wf-A53 (2026-10-19) — ``wf_a51_leakage_audit.py`` writes
``docs/DATA_LEAKAGE_AUDIT.md`` wholesale.  §8 (MSRS strong-model splits) is
maintained outside that script, so a re-run deleted 54 lines of it.  A
regeneration is not allowed to destroy another section.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "wf_a51_leakage_audit_under_test",
        PROJECT_ROOT / "molmetal" / "scripts" / "wf_a51_leakage_audit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


def test_preserves_section_8(audit, tmp_path):
    doc = tmp_path / "audit.md"
    doc.write_text(
        "# Audit\n\n## 7. Verdict\n\nours\n\n---\n\n## 8. MSRS\n\nnot ours\n",
        encoding="utf-8",
    )
    tail = audit._preserved_tail(doc)
    assert tail.startswith("## 8. MSRS")
    assert "not ours" in tail


def test_returns_empty_when_no_foreign_section(audit, tmp_path):
    doc = tmp_path / "audit.md"
    doc.write_text("# Audit\n\n## 7. Verdict\n\nours\n", encoding="utf-8")
    assert audit._preserved_tail(doc) == ""


def test_missing_file_is_not_an_error(audit, tmp_path):
    assert audit._preserved_tail(tmp_path / "absent.md") == ""
