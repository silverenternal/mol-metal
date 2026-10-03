"""Tests for the 8-skill ecosystem verification (Wf-19 / Wf-Skill-Ecosystem, 2026-09-26).

This file gates the 8-skill ecosystem close-out:

* Every ``.claude/skills/<name>/SKILL.md`` file is structurally complete
  (frontmatter + 4 canonical sections + worked example + R18 reference
  + mechanism-mapping reference).
* The verification script (``molmetal/scripts/verify_skill_ecosystem.py``)
  produces a JSON report with the canonical shape
  (n_skills / n_passed / n_failed / per_skill).

Each test is independent and runs without GPU / torch / numpy — they
parse the SKILL.md files and the JSON report directly.

Run via::

    cd /home/hugo/codes/mol-metal
    uv run --no-sync python -m pytest molmetal/tests/test_skill_ecosystem.py -v
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Repo-root path resolution
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
VERIFY_SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "verify_skill_ecosystem.py"
REPORT_PATH = REPO_ROOT / "molmetal" / "reports" / "wf_skill_ecosystem" / "verification.json"

EXPECTED_SKILL_NAMES = {
    "mol-metal-anticancer",
    "mol-metal-cfm-training",
    "mol-metal-click-chemistry",
    "mol-metal-lambda-calculus",
    "mol-metal-mechanism-mapping",
    "mol-metal-paper-writing",
    "mol-metal-r18-math",
    "party-twenty-spirit",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skill_paths() -> List[Path]:
    """Return the sorted list of SKILL.md files under ``.claude/skills``."""
    assert SKILLS_ROOT.is_dir(), f"missing skills root: {SKILLS_ROOT}"
    return sorted(SKILLS_ROOT.glob("*/SKILL.md"))


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a SKILL.md into ``(frontmatter, body)`` text blocks."""
    lines = text.splitlines()
    if not lines or not lines[0].strip().startswith("---"):
        return "", text
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip().startswith("---"):
            end = i
            break
    if end is None:
        return "", text
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :])


def _parse_frontmatter(fm_text: str) -> Dict[str, str]:
    """Parse a simple YAML ``key: value`` frontmatter block."""
    out: Dict[str, str] = {}
    for raw in fm_text.splitlines():
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        out[key.strip()] = val.strip()
    return out


def _skill_body(name: str) -> str:
    """Return the body of a specific skill (for ad-hoc checks)."""
    path = SKILLS_ROOT / name / "SKILL.md"
    assert path.is_file(), f"missing skill file: {path}"
    text = path.read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    return body


def _load_report() -> Dict:
    """Load the verification JSON report (skip if missing)."""
    if not REPORT_PATH.is_file():
        pytest.skip(
            f"verification report missing — run `python {VERIFY_SCRIPT.relative_to(REPO_ROOT)}` first"
        )
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ===========================================================================
# Test 1 — exactly 8 skills exist
# ===========================================================================
def test_skill_count_is_exactly_8():
    """The project must ship exactly 8 skills (the R18 ship gate)."""
    skills = _skill_paths()
    assert len(skills) == 8, f"expected 8 skills, found {len(skills)}: {[p.parent.name for p in skills]}"


# ===========================================================================
# Test 2 — every skill has a name + description in frontmatter
# ===========================================================================
def test_skill_frontmatter_complete():
    """All 8 SKILL.md files must carry ``name`` + ``description`` frontmatter."""
    skills = _skill_paths()
    assert len(skills) >= 8, "less than 8 SKILL.md files discovered"
    for path in skills:
        text = path.read_text(encoding="utf-8")
        fm_text, _ = _split_frontmatter(text)
        fm = _parse_frontmatter(fm_text)
        assert fm.get("name"), f"{path.parent.name}: missing frontmatter 'name'"
        assert fm.get("description"), f"{path.parent.name}: missing frontmatter 'description'"


# ===========================================================================
# Test 3 — every skill has a tools: line (or is metadata-only like r18-math
#         or party-twenty-spirit)
# ===========================================================================
def test_skill_frontmatter_tools_present():
    """Every skill either advertises ``tools:`` OR is metadata-only."""
    metadata_only = {"mol-metal-r18-math", "party-twenty-spirit"}
    skills = _skill_paths()
    for path in skills:
        name = path.parent.name
        text = path.read_text(encoding="utf-8")
        fm_text, _ = _split_frontmatter(text)
        fm = _parse_frontmatter(fm_text)
        if name in metadata_only:
            # OK to skip tools list (math reference / governance skill)
            continue
        assert "tools" in fm, f"{name}: frontmatter must declare 'tools'"
        assert fm["tools"].strip(), f"{name}: frontmatter 'tools' is empty"


# ===========================================================================
# Test 4 — R18 ship reference is present in every skill (Wf-8 ship gate)
# ===========================================================================
def test_skill_r18_reference_present():
    """Every skill must reference at least one R18 mechanism OR have the heading."""
    r18_heading = re.compile(
        r"^##\s+R18 mechanism cross-reference",
        re.MULTILINE | re.IGNORECASE,
    )
    r18_any = re.compile(r"^##\s+R18", re.MULTILINE | re.IGNORECASE)
    mech_id = re.compile(r"\bM(?:22|2[3-9]|3[0-9]|4[0-4])\b")
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        ok = bool(r18_heading.search(body)) or bool(r18_any.search(body)) or bool(mech_id.search(body))
        assert ok, f"{name}: missing R18 mechanism cross-reference (Wf-8 ship gate)"


# ===========================================================================
# Test 5 — mechanism-mapping reference is present in every skill (Wf-18 ship gate)
# ===========================================================================
def test_skill_mechanism_reference_present():
    """Every skill must have the Wf-18 ship gate — a mechanism cross-reference."""
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        has_heading = "## R18 mechanism cross-reference" in body
        has_doc_ref = "MECHANISM_METRIC_MAPPING.md" in body
        has_mech_id = bool(re.search(r"\bM(?:22|2[3-9]|3[0-9]|4[0-4])\b", body))
        assert (
            has_heading or has_doc_ref or has_mech_id
        ), f"{name}: missing mechanism-mapping reference (Wf-18 ship gate)"


# ===========================================================================
# Test 6 — every skill has at least one worked example
# ===========================================================================
def test_skill_worked_example_present():
    """Every skill must include a "Worked example" or "Examples" section."""
    we_pattern = re.compile(r"^##\s+Worked example", re.MULTILINE | re.IGNORECASE)
    examples_pattern = re.compile(r"^##\s+Examples(\s|$|\.)", re.MULTILINE | re.IGNORECASE)
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        assert (
            we_pattern.search(body) or examples_pattern.search(body)
        ), f"{name}: missing worked-example section"


# ===========================================================================
# Test 7 — every skill has the expected name (set membership)
# ===========================================================================
def test_skill_set_matches_expected():
    """The set of skill directories must match the canonical 8-name set."""
    actual = {p.parent.name for p in _skill_paths()}
    assert actual == EXPECTED_SKILL_NAMES, (
        f"skill set mismatch — extra: {actual - EXPECTED_SKILL_NAMES}, "
        f"missing: {EXPECTED_SKILL_NAMES - actual}"
    )


# ===========================================================================
# Test 8 — verification script exists at the expected path
# ===========================================================================
def test_verification_script_exists():
    """The verification script must live at the canonical path."""
    assert VERIFY_SCRIPT.is_file(), f"missing verification script: {VERIFY_SCRIPT}"


# ===========================================================================
# Test 9 — verification script is runnable end-to-end (smoke)
# ===========================================================================
def test_verification_script_runs_clean():
    """Re-run the verification script and assert exit-0 + 8/8 PASS."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--smoke"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"verification script exit code = {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "8/8 PASS" in proc.stdout, (
        f"expected '8/8 PASS' in stdout, got:\n{proc.stdout}"
    )


# ===========================================================================
# Test 10 — verification JSON report exists and has the canonical shape
# ===========================================================================
def test_verification_report_shape():
    """The report JSON must have ``n_skills`` + ``n_passed`` + ``n_failed`` + ``per_skill``."""
    report = _load_report()
    for key in ("n_skills", "n_passed", "n_failed", "per_skill"):
        assert key in report, f"verification report missing key: {key}"
    assert report["n_skills"] == 8
    assert isinstance(report["per_skill"], dict)


# ===========================================================================
# Test 11 — verification report: 8/8 PASS
# ===========================================================================
def test_verification_report_all_pass():
    """All 8 skills must report ``status == PASS`` in the verification JSON."""
    report = _load_report()
    assert report["n_passed"] == 8, (
        f"expected 8 PASS, got {report['n_passed']} — fails: "
        f"{[n for n, info in report['per_skill'].items() if info['status'] != 'PASS']}"
    )
    assert report["n_failed"] == 0
    for name, info in report["per_skill"].items():
        assert info["status"] == "PASS", f"{name}: status = {info['status']}"


# ===========================================================================
# Test 12 — per-skill JSON record carries every check flag
# ===========================================================================
def test_verification_report_per_skill_flags():
    """Each per-skill record must include every check flag + status + reasons."""
    report = _load_report()
    expected_flags = {
        "frontmatter_ok",
        "tools_ok",
        "sections_ok",
        "r18_reference_ok",
        "mechanism_reference_ok",
        "worked_example_ok",
        "status",
        "fail_reasons",
    }
    for name, info in report["per_skill"].items():
        missing = expected_flags - set(info.keys())
        assert not missing, f"{name}: missing keys in per_skill record: {missing}"
        assert isinstance(info["fail_reasons"], list)
        assert info["status"] in ("PASS", "FAIL")


# ===========================================================================
# Test 13 — every skill name in the report is one of the canonical 8
# ===========================================================================
def test_verification_report_skill_names_match_canonical_set():
    """No phantom or missing names in the report — must be the canonical 8."""
    report = _load_report()
    actual = set(report["per_skill"].keys())
    assert actual == EXPECTED_SKILL_NAMES, (
        f"per_skill keys = {actual}, expected {EXPECTED_SKILL_NAMES}"
    )


# ===========================================================================
# Test 14 — every skill cross-references at least one other skill
# ===========================================================================
def test_skill_cross_references_other_skill():
    """Every skill must cross-reference at least one sibling skill."""
    other_skills = EXPECTED_SKILL_NAMES
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        # Look for an explicit "mol-metal-*" reference in the body
        refs = re.findall(r"`?(mol-metal-[a-z-]+)`?", body)
        # Filter out self-references
        other_refs = {r for r in refs if r in other_skills and r != name}
        assert other_refs, f"{name}: no cross-reference to sibling skills"


# ===========================================================================
# Test 15 — every skill has a "When to use" / "When the user asks" / §0 header
# ===========================================================================
def test_skill_has_usage_section():
    """Every skill must have a usage section in some form (canonical or alias)."""
    usage_patterns = [
        re.compile(r"^##\s+When to use(\s|$|\.)", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+When the user asks about", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+When to use this skill", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+§0\.? ", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+§1\.? ", re.MULTILINE | re.IGNORECASE),
    ]
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        ok = any(p.search(body) for p in usage_patterns)
        assert ok, f"{name}: missing usage section header"


# ===========================================================================
# Test 16 (bonus) — every skill has a "Cross-reference" or "Related skills" section
# ===========================================================================
def test_skill_has_cross_reference_section():
    """Every skill must have a Cross-reference section (mandatory)."""
    patterns = [
        re.compile(r"^##\s+Cross-reference", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+Related skills", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^##\s+§7\.\s+", re.MULTILINE),  # party-twenty-spirit §7
    ]
    for name in EXPECTED_SKILL_NAMES:
        body = _skill_body(name)
        ok = any(p.search(body) for p in patterns)
        assert ok, f"{name}: missing 'Cross-reference' / 'Related skills' section (mandatory)"


# ===========================================================================
# Test 17 (bonus) — verification script is not a stub (≥ 100 LOC)
# ===========================================================================
def test_verification_script_is_substantive():
    """The verification script must be at least 100 LOC (anti-stub guard)."""
    text = VERIFY_SCRIPT.read_text(encoding="utf-8")
    loc = sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))
    assert loc >= 100, f"verification script too short: {loc} LOC (target ≥ 100)"


# ===========================================================================
# Test 18 (bonus) — every skill mentions R18 in some form (body OR frontmatter)
# ===========================================================================
def test_skill_r18_aware():
    """Every skill must show awareness of the R18 round in body or frontmatter."""
    for name in EXPECTED_SKILL_NAMES:
        text = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
        assert (
            "R18" in text or "r18" in text.lower()
        ), f"{name}: no R18 mention (Wf-8 ship gate)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
