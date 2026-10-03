"""Wf-329-W8 wiring verification for 2 NEW R20/R21 skill stubs.

Per AGENTS.md §3.3 + Wf-321B R20+R21 skill ecosystem audit, this file gates
the Wf-329-W8 ship surface. The 2 NEW skill candidates are::

    R20-P5  mol-metal-closure-catherine  (TODO-96)  — wire target (Wf-329-W8)
    R21     mol-metal-triangle-closure   (TODO-93 L8) — already WIRED per Wf-327-F10

Status snapshot at Wf-329-W8 ship (2026-09-29):

* ``mol-metal-closure-catherine`` — UPGRADED from partial (92 LOC, 6 sections,
  no YAML frontmatter) → FULL (227 LOC, 8 sections, YAML frontmatter added)
  in Wf-329-W8. Was NOT covered by Wf-327-F9 (different from closure-theorem)
  or Wf-327-F10 (R21 sweep).
* ``mol-metal-triangle-closure`` — ALREADY WIRED per Wf-327-F10 (574 LOC, 8
  sections, YAML frontmatter present). Wf-329-W8 only VERIFIED status + adds
  drift guards.

The 4 invariants verified here:

1. SKILL.md exists and parses YAML frontmatter + has ``name:`` field
2. SKILL.md has all 8 canonical sections (§0-§8)
3. Cross-references in §5/§8 resolve to real files on disk
4. Activation keywords + code anchors exist (skill discovery + grounding)

Each test is independent and runs without GPU / torch — they only parse
SKILL.md files and probe the filesystem.

Run via::

    cd /home/hugo/codes/mol-metal
    PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf329w8_skill_wiring.py -v
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Repo-root path resolution
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"

# The 2 NEW skill candidates per Wf-329-W8 task spec + AGENTS.md §3.3.
TARGET_SKILLS: Dict[str, Tuple[str, str]] = {
    # name: (R-phase, expected LOC floor)
    "mol-metal-closure-catherine": ("R20-P5 (TODO-96)", 100),  # Wf-329-W8 wire target
    "mol-metal-triangle-closure":  ("R21 (TODO-93 L8)", 400),  # Wf-327-F10 WIRED
}

# Code anchors that MUST resolve on disk for each skill to be considered WIRED.
CODE_ANCHORS: Dict[str, List[str]] = {
    "mol-metal-closure-catherine": [
        "molmetal/scripts/bench_catherine_6_props_25metals.py",
        "molmetal/scripts/bench_closure_ood.py",
        "molmetal/math_skills/catherine_properties.py",
        "molmetal/molmetal_lam/lam_chem/closure.py",
    ],
    "mol-metal-triangle-closure": [
        "molmetal/scripts/triangle_closure.py",
        "molmetal/scripts/bench_lambda_triangle_end_to_end.py",
        "molmetal/molmetal_lam/reward/hilbert_primal_dual.py",
        "molmetal/molmetal_lam/reward/lambda_astsr.py",
        "molmetal/molmetal_lam/search_alg/variance_mcts_integration.py",
    ],
}

# Activation keywords (skill discovery via description frontmatter match).
ACTIVATION_KEYWORDS: Dict[str, List[str]] = {
    "mol-metal-closure-catherine": [
        "closure theorem",
        "Catherine",
        "P1-P6",
        "Tanimoto",
        "R20-P5",
    ],
    "mol-metal-triangle-closure": [
        "triangle closure",
        "ASTSR",
        "Hilbert",
        "MCTS",
        "R21-P4",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skill_path(name: str) -> Path:
    """Return the SKILL.md path for a given skill name."""
    return SKILLS_ROOT / name / "SKILL.md"


def _split_frontmatter(text: str) -> Tuple[str, str]:
    """Split a SKILL.md into (frontmatter, body) text blocks."""
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
    front = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    return front, body


def _extract_yaml_field(frontmatter: str, field: str) -> str:
    """Extract a single YAML scalar field from frontmatter (best-effort, no PyYAML)."""
    # Match `field: "..."` or `field: value` patterns on a single line
    for line in frontmatter.splitlines():
        m = re.match(rf'\s*{re.escape(field)}\s*:\s*"?(.*?)"?\s*$', line)
        if m:
            return m.group(1).strip()
    return ""


def _backticked_paths(text: str) -> List[str]:
    """Extract backticked file paths from a SKILL.md body."""
    return re.findall(r"`([\w./\-]+\.(?:py|md|sh|json|tex|csv))`", text)


# Candidate roots for cross-ref resolution (per Wf-327-F10 design notes).
# Order matters — most specific first.
_CANDIDATE_ROOTS: Tuple[str, ...] = (
    "molmetal/scripts",
    "molmetal/molmetal_lam/reward",
    "molmetal/molmetal_lam/search_alg",
    "molmetal/molmetal_lam/lam_chem",
    "molmetal/molmetal_lam",
    "molmetal/math_skills",
    "molmetal/reports",
    "molmetal/data",
    "molmetal/tests",
    "TODO/03_plans",
    "TODO/pending",
    "TODO/completed",
    "TODO",
    "paper",
)


def _path_resolves(rel: str, skill_name: str = "") -> bool:
    """Check if a relative path resolves against any candidate root."""
    candidate = REPO_ROOT / rel
    if candidate.exists():
        return True
    for root in _CANDIDATE_ROOTS:
        c = REPO_ROOT / root / rel
        if c.exists():
            return True
    # Try relative to skill directory (for references.md etc.)
    if skill_name:
        skill_dir_candidate = SKILLS_ROOT / skill_name / rel
        if skill_dir_candidate.exists():
            return True
    return False


# ---------------------------------------------------------------------------
# Test 1: SKILL.md exists + has YAML frontmatter + name field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_md_exists_with_frontmatter(skill_name: str) -> None:
    """SKILL.md exists, has YAML frontmatter, and the name field matches."""
    path = _skill_path(skill_name)
    assert path.exists(), f"SKILL.md missing for {skill_name}: {path}"
    text = path.read_text(encoding="utf-8")
    frontmatter, _body = _split_frontmatter(text)
    assert frontmatter, f"{skill_name} SKILL.md lacks YAML frontmatter"
    name_field = _extract_yaml_field(frontmatter, "name")
    assert name_field == skill_name, (
        f"{skill_name} frontmatter name mismatch: got {name_field!r}"
    )
    desc_field = _extract_yaml_field(frontmatter, "description")
    assert len(desc_field) > 50, (
        f"{skill_name} description too short ({len(desc_field)} chars); "
        "skill discovery needs a substantive description"
    )
    tools_field = _extract_yaml_field(frontmatter, "tools")
    assert tools_field, (
        f"{skill_name} frontmatter missing 'tools:' field (per Wf-327 WIRED contract)"
    )


# ---------------------------------------------------------------------------
# Test 2: SKILL.md has 8 canonical sections (§0-§8)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_md_has_eight_sections(skill_name: str) -> None:
    """SKILL.md body has all 8 canonical sections (§1..§8 or §0..§7)."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    _frontmatter, body = _split_frontmatter(text)
    section_pattern = re.compile(r"^## §(\d+)\s+", re.MULTILINE)
    sections_found = sorted({int(m.group(1)) for m in section_pattern.finditer(body)})
    # Accept either §0..§7 (8 sections) or §1..§8 (8 sections)
    has_0_to_7 = set(range(0, 8)).issubset(set(sections_found))
    has_1_to_8 = set(range(1, 9)).issubset(set(sections_found))
    assert has_0_to_7 or has_1_to_8, (
        f"{skill_name} needs 8 canonical sections (§0..§7 OR §1..§8); "
        f"found: §{sections_found}"
    )
    # §8 cross-references ship surface
    has_section_8 = bool(re.search(r"^## §8\s+", body, re.MULTILINE))
    assert has_section_8, (
        f"{skill_name} missing §8 (Cross-references) — Wf-329-W8 wire contract"
    )


# ---------------------------------------------------------------------------
# Test 3: Cross-references resolve to real files on disk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_cross_references_resolve(skill_name: str) -> None:
    """All backticked paths in SKILL.md resolve to real files (or marked STUB/PROJECTED)."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    _frontmatter, body = _split_frontmatter(text)
    paths = _backticked_paths(body)
    unresolved: List[str] = []
    for rel in paths:
        # Skip non-path tokens (URLs, function names with .py extensions etc.)
        if rel.startswith("http") or " " in rel:
            continue
        if _path_resolves(rel, skill_name):
            continue
        # Check surrounding context for STUB/PROJECTED/NOT YET CREATED markers
        # (line that contains the backtick reference)
        in_stub_context = False
        for line in body.splitlines():
            if f"`{rel}`" in line:
                markers = ("STUB", "PROJECTED", "NOT YET CREATED", "Wait for")
                if any(m in line for m in markers):
                    in_stub_context = True
                    break
        if in_stub_context:
            continue
        unresolved.append(rel)
    assert not unresolved, (
        f"{skill_name} unresolved cross-references: {unresolved[:5]}"
        + (f" ... (+{len(unresolved) - 5} more)" if len(unresolved) > 5 else "")
    )


# ---------------------------------------------------------------------------
# Test 4: Activation keywords + code anchors present
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_activation_and_anchors(skill_name: str) -> None:
    """Activation keywords are present in description; code anchors resolve."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    frontmatter, _body = _split_frontmatter(text)
    desc = _extract_yaml_field(frontmatter, "description")
    # Activation keywords (skill discovery via description match)
    for kw in ACTIVATION_KEYWORDS[skill_name]:
        assert kw.lower() in desc.lower(), (
            f"{skill_name} description missing activation keyword {kw!r}"
        )
    # Code anchors must resolve on disk
    for anchor_rel in CODE_ANCHORS[skill_name]:
        anchor_path = REPO_ROOT / anchor_rel
        assert anchor_path.exists(), (
            f"{skill_name} code anchor missing: {anchor_rel}"
        )


# ---------------------------------------------------------------------------
# Test 5: LOC threshold (per Wf-327 stub contract — < 50 LOC = STUB)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_md_above_stub_threshold(skill_name: str) -> None:
    """SKILL.md is above the Wf-327 stub threshold (≥ 50 LOC)."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    loc = len(text.splitlines())
    floor = TARGET_SKILLS[skill_name][1]
    assert loc >= 50, (
        f"{skill_name} SKILL.md is STUB ({loc} LOC, < 50 threshold)"
    )
    assert loc >= floor, (
        f"{skill_name} SKILL.md below expected LOC floor {floor}: got {loc}"
    )


# ---------------------------------------------------------------------------
# Test 6: Wf-329-W8 audit gate — both skills WIRED, AGENTS.md updated
# ---------------------------------------------------------------------------
def test_wf329w8_audit_both_wired() -> None:
    """Wf-329-W8 audit gate: both target skills show WIRED markers."""
    agents_md = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for skill_name, (phase, _floor) in TARGET_SKILLS.items():
        # Look for the skill name + WIRED marker in §3.3
        assert skill_name in agents_md, (
            f"{skill_name} not found in AGENTS.md §3.3 catalog"
        )
        # WIRED marker can be from any Wf — Wf-327-F9 / F10 / Wf-329-W8
        # Pattern: `mol-metal-xxx | R20-P5 | TBD (**WIRED per Wf-NNN | ...)` or `(**WIRED`
        pattern = (
            rf"\`{re.escape(skill_name)}\`.*?\*\*WIRED"
        )
        assert re.search(pattern, agents_md, re.DOTALL), (
            f"{skill_name} lacks WIRED marker in AGENTS.md §3.3 "
            f"(audit gate failed)"
        )


# ---------------------------------------------------------------------------
# Test 7: MEASURED-vs-PROJECTED ledger present (per AGENTS.md §5.2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(TARGET_SKILLS.keys()))
def test_skill_has_measured_vs_projected_ledger(skill_name: str) -> None:
    """SKILL.md carries MEASURED-vs-PROJECTED ledger per party-twenty-spirit §3.5."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    has_measured = bool(re.search(r"MEASURED", text))
    has_projected = bool(re.search(r"PROJECTED", text))
    assert has_measured and has_projected, (
        f"{skill_name} SKILL.md lacks MEASURED-vs-PROJECTED ledger "
        f"(has MEASURED={has_measured}, has PROJECTED={has_projected})"
    )


# ---------------------------------------------------------------------------
# Test 8: Wf-327-F10 cross-ref acknowledged for triangle-closure
# ---------------------------------------------------------------------------
def test_triangle_closure_wf327f10_acknowledged() -> None:
    """triangle-closure acknowledges Wf-327-F10 WIRED status."""
    text = _skill_path("mol-metal-triangle-closure").read_text(encoding="utf-8")
    # Either Wf-327-F10 or Wf-329-W8 marker
    assert "Wf-327-F10" in text or "Wf-329-W8" in text, (
        "mol-metal-triangle-closure SKILL.md lacks Wf-327-F10/Wf-329-W8 "
        "WIRED acknowledgement"
    )


# ---------------------------------------------------------------------------
# Test 9: closure-catherine Wf-329-W8 wire marker present
# ---------------------------------------------------------------------------
def test_closure_catherine_wf329w8_wire_marker() -> None:
    """closure-catherine SKILL.md carries Wf-329-W8 WIRED marker."""
    text = _skill_path("mol-metal-closure-catherine").read_text(encoding="utf-8")
    assert "Wf-329-W8" in text, (
        "mol-metal-closure-catherine SKILL.md lacks Wf-329-W8 WIRED marker"
    )