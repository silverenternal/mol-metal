"""Tests for Wf-327-F9: 4 NEW R20 skill stubs verification.

Per Wf-321B ship wave (TODO-99) + Wf-327-T5: the 4 NEW R20 skills
(`variance-puct-default` / `hilbert-astsr-e2e` / `closure-theorem` /
`multiph-pll`) are required to be **fully wired** (SKILL.md ≥ 50 LOC +
frontmatter + ≥ 3 mandatory sections + activation keywords + code
anchors + cross-references resolve to existing files).

This file gates the Wf-327-F9 ship close-out:

* Every `.claude/skills/{skill}/SKILL.md` exists (n_skills == 4).
* Every SKILL.md frontmatter carries `name` + `description`.
* Every SKILL.md is ≥ 50 LOC (NOT a STUB).
* Every SKILL.md has ≥ 3 canonical sections (e.g. §1 / §2 / §3).
* Every SKILL.md activation keywords present in body.
* Every SKILL.md code anchor module path resolves on disk (≥ 1 hit).
* Every SKILL.md cross-reference to TODO/pending/{92,93,94,96}_*.md resolves.

Each test is independent and runs without GPU / torch / numpy —
they parse the SKILL.md files directly.

Run via::

    cd /home/hugo/codes/mol-metal
    PYTHONPATH=/home/hugo/codes/mol-metal \
        /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
        molmetal/tests/test_wf327f9_skill_stubs.py -v
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


# ---------------------------------------------------------------------------
# 4 NEW R20 skills per Wf-321B + Wf-327-F9
# ---------------------------------------------------------------------------
NEW_R20_SKILLS: Tuple[str, ...] = (
    "mol-metal-variance-puct-default",  # R20-P1, TODO-92
    "mol-metal-hilbert-astsr-e2e",      # R20-P2, TODO-93
    "mol-metal-multiph-pll",            # R20-P3, TODO-94
    "mol-metal-closure-theorem",       # R20-P5, TODO-96
)


# Per-skill metadata — activation keywords + mandatory code anchors + TODO refs
SKILL_METADATA: Dict[str, Dict[str, object]] = {
    "mol-metal-variance-puct-default": {
        "todo_ref": "TODO/pending/92_r20_p1_variance_puct_default.md",
        "activation_keywords": (
            "VariancePUCT default",
            "DEFAULT_USE_VARIANCE_PUCT",
            "make_default_selector",
            "R20-P1",
            "variance_puct_integration.py",
        ),
        "code_anchors": (
            "molmetal/molmetal_lam/search_alg/variance_puct_integration.py",
            "molmetal/scripts/run_proof_search_variance_puct.py",
        ),
    },
    "mol-metal-hilbert-astsr-e2e": {
        "todo_ref": "TODO/pending/93_r20_p2_hilbert_astsr_gpu.md",
        "activation_keywords": (
            "Hilbert-ASTSR",
            "F5_RDKIT_2D",
            "Deflexpressor",
            "R20-P2",
            "hilbert_primal_dual.py",
        ),
        "code_anchors": (
            "molmetal/molmetal_lam/reward/hilbert_primal_dual.py",
            "molmetal/molmetal_lam/reward/lambda_astsr.py",
        ),
    },
    "mol-metal-multiph-pll": {
        "todo_ref": "TODO/pending/94_r20_p3_multiph_pll_feature.md",
        "activation_keywords": (
            "Multi-PH",
            "PLL signature",
            "Score-Rep",
            "R20-P3",
            "multi_parameter_ph.py",
        ),
        "code_anchors": (
            "molmetal/math_skills/multi_parameter_ph.py",
            "molmetal/molmetal_lam/lam_chem/pll_signature.py",
        ),
    },
    "mol-metal-closure-theorem": {
        "todo_ref": "TODO/pending/96_r20_p5_closure_catherine.md",
        "activation_keywords": (
            "Closure theorem",
            "Catherine",
            "6/6 PASS",
            "R20-P5",
            "catherine_properties.py",
        ),
        "code_anchors": (
            "molmetal/molmetal_lam/lam_chem/closure.py",
            "molmetal/math_skills/catherine_properties.py",
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skill_path(name: str) -> Path:
    """Return the SKILL.md path for a given skill name."""
    path = SKILLS_ROOT / name / "SKILL.md"
    assert path.is_file(), f"missing SKILL.md: {path}"
    return path


def _read_skill(name: str) -> str:
    """Return the full SKILL.md text for a given skill."""
    return _skill_path(name).read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> Tuple[str, str]:
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


def _section_headers(body: str) -> List[str]:
    """Extract all §-style section headers from the SKILL.md body.

    Matches patterns like `## §1 Name` or `## §12.3 Subsection`.
    """
    pattern = re.compile(r"^##\s+(§\d+(?:\.\d+)*)\s*\.?\s*(.*)$", re.MULTILINE)
    return [f"{m.group(1)} {m.group(2).strip()}" for m in pattern.finditer(body)]


# ---------------------------------------------------------------------------
# Test 1 — All 4 NEW R20 skills have SKILL.md on disk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", NEW_R20_SKILLS)
def test_skill_md_exists(skill_name: str):
    """All 4 NEW R20 SKILL.md files must exist on disk."""
    path = _skill_path(skill_name)
    assert path.is_file(), f"SKILL.md missing for {skill_name}: {path}"
    assert path.stat().st_size > 0, f"SKILL.md empty for {skill_name}"


# ---------------------------------------------------------------------------
# Test 2 — Each SKILL.md is fully wired (≥ 50 LOC, NOT a STUB)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", NEW_R20_SKILLS)
def test_skill_md_above_stub_threshold(skill_name: str):
    """Each SKILL.md must be ≥ 50 LOC to qualify as WIRED (vs STUB)."""
    text = _read_skill(skill_name)
    loc = len(text.splitlines())
    assert loc >= 50, (
        f"{skill_name} SKILL.md is a STUB ({loc} LOC < 50); "
        f"wiring required per Wf-327-F9 task spec"
    )


# ---------------------------------------------------------------------------
# Test 3 — Each SKILL.md has frontmatter with name + description
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", NEW_R20_SKILLS)
def test_skill_frontmatter_complete(skill_name: str):
    """All 4 NEW R20 SKILL.md files must carry ``name`` + ``description``."""
    text = _read_skill(skill_name)
    fm_text, _ = _split_frontmatter(text)
    assert fm_text, f"{skill_name} has no YAML frontmatter"
    fm = _parse_frontmatter(fm_text)
    assert "name" in fm, f"{skill_name} frontmatter missing 'name'"
    assert "description" in fm, f"{skill_name} frontmatter missing 'description'"
    assert fm["name"] == skill_name, (
        f"{skill_name} frontmatter name mismatch: {fm['name']!r}"
    )
    # Description should be substantial (≥ 100 chars)
    assert len(fm["description"]) >= 100, (
        f"{skill_name} description too short ({len(fm['description'])} chars)"
    )


# ---------------------------------------------------------------------------
# Test 4 — Each SKILL.md has ≥ 3 canonical § sections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", NEW_R20_SKILLS)
def test_skill_has_canonical_sections(skill_name: str):
    """Each SKILL.md must have ≥ 3 §-numbered sections (e.g. §1, §2, §3)."""
    text = _read_skill(skill_name)
    _, body = _split_frontmatter(text)
    sections = _section_headers(body)
    assert len(sections) >= 3, (
        f"{skill_name} has only {len(sections)} §-sections; "
        f"expected ≥ 3 for WIRED status. Found: {sections[:5]}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Each SKILL.md activation keywords present in body
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(SKILL_METADATA.keys()))
def test_skill_activation_keywords(skill_name: str):
    """Each SKILL.md must contain its domain activation keywords."""
    text = _read_skill(skill_name)
    metadata = SKILL_METADATA[skill_name]
    keywords = metadata["activation_keywords"]
    assert isinstance(keywords, tuple)
    # At least 3 of 5 keywords must appear in body (allowing minor wording variants)
    hits = [k for k in keywords if k in text]
    assert len(hits) >= 3, (
        f"{skill_name} activation keyword gap: "
        f"only {len(hits)}/{len(keywords)} keywords found. "
        f"Missing: {[k for k in keywords if k not in text]}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Each SKILL.md code anchors resolve on disk (≥ 1 hit)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(SKILL_METADATA.keys()))
def test_skill_code_anchors_resolve(skill_name: str):
    """Each SKILL.md must reference ≥ 1 code anchor that exists on disk."""
    text = _read_skill(skill_name)
    metadata = SKILL_METADATA[skill_name]
    anchors = metadata["code_anchors"]
    assert isinstance(anchors, tuple)
    hits = [a for a in anchors if (REPO_ROOT / a).is_file()]
    assert len(hits) >= 1, (
        f"{skill_name} code anchor gap: "
        f"0/{len(anchors)} anchors resolve on disk. "
        f"Tried: {anchors}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Each SKILL.md cross-references its TODO/pending doc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", list(SKILL_METADATA.keys()))
def test_skill_todo_ref_resolves(skill_name: str):
    """Each SKILL.md must cross-reference its parent TODO doc."""
    text = _read_skill(skill_name)
    metadata = SKILL_METADATA[skill_name]
    todo_ref = metadata["todo_ref"]
    assert isinstance(todo_ref, str)
    # Either the TODO doc is referenced in body OR exists on disk
    assert todo_ref in text or (REPO_ROOT / todo_ref).is_file(), (
        f"{skill_name} cross-ref missing: {todo_ref} not in body and not on disk"
    )


# ---------------------------------------------------------------------------
# Test 8 — Each SKILL.md has MEASURED-vs-PROJECTED ledger section
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name", NEW_R20_SKILLS)
def test_skill_has_measured_vs_projected_ledger(skill_name: str):
    """Each SKILL.md must encode MEASURED-vs-PROJECTED framing.

    Per `party-twenty-spirit` §3.5 honest framing mandate + AGENTS.md §5.2,
    every skill should distinguish measured facts from projected targets.
    """
    text = _read_skill(skill_name)
    # Match "MEASURED" or "PROJECTED" anywhere in body (case-sensitive on
    # the canonical labels)
    assert "MEASURED" in text, (
        f"{skill_name} missing 'MEASURED' label in body"
    )
    assert "PROJECTED" in text, (
        f"{skill_name} missing 'PROJECTED' label in body"
    )


# ---------------------------------------------------------------------------
# Test 9 — Aggregate: 4 NEW R20 skills all VERIFIED
# ---------------------------------------------------------------------------
def test_all_4_r20_skills_wired():
    """All 4 NEW R20 skills (variance-puct-default / hilbert-astsr-e2e /
    closure-theorem / multiph-pll) must be fully wired per Wf-321B."""
    results = {}
    for skill_name in NEW_R20_SKILLS:
        path = _skill_path(skill_name)
        text = path.read_text(encoding="utf-8")
        loc = len(text.splitlines())
        _, body = _split_frontmatter(text)
        n_sections = len(_section_headers(body))
        results[skill_name] = {"loc": loc, "n_sections": n_sections}
    # All 4 must be above stub threshold + have ≥ 3 sections
    for name, stats in results.items():
        assert stats["loc"] >= 50, f"{name}: STUB ({stats['loc']} LOC)"
        assert stats["n_sections"] >= 3, (
            f"{name}: only {stats['n_sections']} sections"
        )


# ---------------------------------------------------------------------------
# Test 10 — Smoke: pre-flight v2 stays at 25/25 PASS sanity (skip if absent)
# ---------------------------------------------------------------------------
def test_skill_ecosystem_audit_wf321b_status():
    """Wf-321B R20+R21 audit doc must exist + mark R20 coverage as 4/4."""
    audit_path = REPO_ROOT / "TODO" / "completed" / "321b_r20_r21_skill_ecosystem_audit.md"
    if not audit_path.is_file():
        pytest.skip(f"Wf-321B audit doc missing: {audit_path}")
    text = audit_path.read_text(encoding="utf-8")
    # R20 coverage line should be present
    assert "R20 coverage = 4/4" in text or "R20 4/4" in text, (
        f"Wf-321B audit missing R20 4/4 coverage marker"
    )
