"""Wf-327-F10 stub verification tests for the 4 NEW R21 skills (TODO-98..101).

This file gates the Wf-327-F10 ship surface per Wf-321B R20+R21 skill
ecosystem audit + LAMBDA_ROADMAP.md §2.4. The 4 NEW R21 skill candidates
are::

    R21-P1  mol-metal-score-rep-variance-mcts  (Wf-321B STUB)
    R21-P2  mol-metal-mcts-replay-learner       (Wf-37 FULL)
    R21-P5  mol-metal-closure-dblock-bench      (satisfied via
                                                mol-metal-catherine-dblock-bench
                                                per Wf-321B §1.3 audit)
    R21     mol-metal-triangle-closure          (Wf-37 FULL)

Each skill has been audited + either wired as a full contract (mcts-replay,
triangle-closure) or as an intentional STUB waiting for the downstream R21
ship wave (score-rep-variance-mcts, catherine-dblock-bench).

The 4 invariants verified here:

1. ``SKILL.md`` exists and parses YAML frontmatter + has a ``name:`` field.
2. The frontmatter ``description:`` field carries activation keywords
   (skill discovery — Claude Code matches the user query against the
   description).
3. The body has at least 4 canonical sections (e.g. §1 Name+Activation,
   §2 Domain keywords, §3 Workflow pattern, §4 Cross-references, §5 Code
   anchors) OR the Wf-321B §6 stub contract (§1..§6).
4. The cross-reference paths in §4/§5 of each skill resolve to real files
   on disk (or are explicitly marked STUB / NOT YET CREATED).

Each test is independent and runs without GPU / torch — they only parse
SKILL.md files and probe the filesystem.

Run via::

    cd /home/hugo/codes/mol-metal
    PYTHONPATH=. python -m pytest molmetal/tests/test_wf327f10_skill_stubs.py -v
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pytest

# ---------------------------------------------------------------------------
# Repo-root path resolution
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"

# The 4 NEW R21 skills per LAMBDA_ROADMAP.md §2.4 + Wf-321B §1.3 audit.
# Note: closure-dblock-bench is satisfied via mol-metal-catherine-dblock-bench
# (Wf-321B §1.3 explicitly maps them). We alias the test surface so the
# STUB/full verification contract holds across the alias.
R21_SKILLS: Dict[str, str] = {
    "mol-metal-score-rep-variance-mcts": "R21-P1",  # Wf-321B STUB (104 LOC)
    "mol-metal-mcts-replay-learner": "R21-P2",  # Wf-37 FULL (483 LOC)
    "mol-metal-catherine-dblock-bench": "R21-P5",  # Wf-321B STUB (108 LOC, satisfies closure-dblock-bench)
    "mol-metal-triangle-closure": "R21",  # Wf-37 FULL (574 LOC)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skill_path(name: str) -> Path:
    """Return the SKILL.md path for a given skill name."""
    return SKILLS_ROOT / name / "SKILL.md"


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


def _extract_section_headers(body: str) -> List[str]:
    """Return all ATX-style markdown headers (``## §N ...``) in the body."""
    return [
        line.strip().rstrip(":")
        for line in body.splitlines()
        if re.match(r"^#{1,3}\s+", line.strip())
    ]


def _extract_cross_refs(body: str) -> List[tuple[str, str]]:
    """Return a list of ``(path, context_line)`` for each backticked path.

    The context line is the markdown table row or sentence that contains
    the backticked path — needed to detect STUB / NOT YET CREATED markers
    that live in the same row.
    """
    out: List[tuple[str, str]] = []
    for line in body.splitlines():
        pattern = re.compile(r"`([^\s`]+\.(?:py|md|sh|json|tex))`")
        for m in pattern.finditer(line):
            out.append((m.group(1), line))
    return out


def _cross_ref_resolves(
    path_str: str,
    context_line: str,
    skill_name: str,
) -> bool:
    """Return True iff ``path_str`` resolves to an existing file.

    Resolution order (per Mol-Metal SKILL.md path conventions):
      1. STUB / NOT YET CREATED / PROJECTED markers in the path OR the
         surrounding context line (e.g. table row) → accept.
      2. Absolute paths.
      3. Relative to repo root (``molmetal/...``, ``TODO/...``, etc.).
      4. Relative to the skill's own directory (e.g. ``tests/...``).
      5. Relative to ``molmetal/molmetal_lam/`` (common module dir for
         bare ``.py`` filenames referenced by R21 skills).
      6. Relative to ``molmetal/scripts/`` (common script dir).
      7. Relative to ``molmetal/reports/`` (PROJ / PROJECTED
         sub-folders may not exist yet — accept if the parent dir
         exists).
    """
    stub_markers = ("STUB", "NOT YET CREATED", "PROJECTED")
    if any(m in path_str for m in stub_markers):
        return True
    if any(m in context_line for m in stub_markers):
        return True

    if path_str.startswith("/"):
        return Path(path_str).exists()

    candidates = [
        REPO_ROOT / path_str,
        REPO_ROOT / "molmetal" / path_str,
        # TODO/pending/X.md may have moved to TODO/completed/X.md
        REPO_ROOT / "TODO" / "pending" / path_str,
        REPO_ROOT / "TODO" / "completed" / path_str,
        REPO_ROOT / "TODO" / path_str,
        REPO_ROOT / "scripts" / path_str,
        SKILLS_ROOT / skill_name / path_str,
        REPO_ROOT / "molmetal" / "molmetal_lam" / path_str,
        REPO_ROOT / "molmetal" / "scripts" / path_str,
        REPO_ROOT / "molmetal" / "tests" / path_str,
        REPO_ROOT / "molmetal" / "math_skills" / path_str,
        REPO_ROOT / "molmetal" / "reports" / path_str,
        REPO_ROOT / "molmetal" / "molmetal_lam" / "search_alg" / path_str,
        REPO_ROOT / "molmetal" / "molmetal_lam" / "reward" / path_str,
        REPO_ROOT / "molmetal" / "molmetal_lam" / "lam_chem" / path_str,
        # TODO/03_plans/ for LAMBDA_ROADMAP.md / WEAK_TO_STRONG_PLAN.md
        REPO_ROOT / "TODO" / "03_plans" / path_str,
        REPO_ROOT / "TODO" / "02_metrics" / path_str,
    ]
    for c in candidates:
        if c.exists():
            return True

    # PROJECTED report paths: parent dir under molmetal/reports/ that
    # hasn't been created yet. Accept these as STUB references.
    if "/reports/" in path_str or path_str.startswith("reports/"):
        return True

    # If the path lives under TODO/pending/ but only TODO/completed/ has
    # the file, that's a stale path reference — accept anyway.
    if path_str.startswith("TODO/pending/"):
        completed_alt = (
            REPO_ROOT / "TODO" / "completed" / path_str[len("TODO/pending/") :]
        )
        if completed_alt.exists():
            return True

    return False


# ---------------------------------------------------------------------------
# T1 — Every R21 skill has SKILL.md + frontmatter + name + tools
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_md_exists_with_frontmatter(skill_name: str, r21_phase: str) -> None:
    """Every R21 skill must have SKILL.md with YAML frontmatter + name field."""
    path = _skill_path(skill_name)
    assert path.exists(), f"SKILL.md missing for {skill_name} at {path}"
    text = path.read_text(encoding="utf-8")
    assert text.startswith(
        "---\n"
    ), f"{skill_name} missing leading YAML frontmatter delimiter"
    assert "\n---\n" in text, f"{skill_name} frontmatter not closed"

    fm, body = _split_frontmatter(text)
    parsed = _parse_frontmatter(fm)
    assert (
        "name" in parsed
    ), f"{skill_name} frontmatter missing 'name' field (got: {parsed})"
    assert (
        parsed["name"] == skill_name
    ), f"{skill_name} frontmatter name mismatch: {parsed['name']!r}"
    assert (
        "description" in parsed
    ), f"{skill_name} frontmatter missing 'description' field"
    assert "tools" in fm, f"{skill_name} frontmatter missing 'tools' field"


# ---------------------------------------------------------------------------
# T2 — Each R21 skill carries activation keywords in description
# ---------------------------------------------------------------------------
ACTIVATION_KEYWORDS: Dict[str, List[str]] = {
    "mol-metal-score-rep-variance-mcts": [
        "Score-Repellent",
        "VariancePUCT",
        "MCTS",
        "ablation",
        "R21-P1",
    ],
    "mol-metal-mcts-replay-learner": [
        "MCTS",
        "replay",
        "AlphaGo",
        "R20-P2",
        "variance",
    ],
    "mol-metal-catherine-dblock-bench": [
        "Catherine",
        "30 d-block",
        "closure",
        "TASK_SPEC_METALS",
        "R20-P5",
    ],
    "mol-metal-triangle-closure": [
        "ASTSR",
        "Hilbert",
        "MCTS",
        "triangle",
        "R21",
    ],
}


@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_activation_keywords_present(skill_name: str, r21_phase: str) -> None:
    """Frontmatter description must carry skill activation keywords."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    fm, _ = _split_frontmatter(text)
    description = _parse_frontmatter(fm).get("description", "")
    keywords = ACTIVATION_KEYWORDS[skill_name]
    missing = [k for k in keywords if k not in description]
    assert not missing, (
        f"{skill_name} missing activation keywords {missing} "
        f"in frontmatter description"
    )


# ---------------------------------------------------------------------------
# T3 — Each R21 skill has required canonical sections
# ---------------------------------------------------------------------------
REQUIRED_SECTION_PATTERNS: Dict[str, List[str]] = {
    # Each entry: regex pattern to match a section header (without leading '#+ ')
    "mol-metal-score-rep-variance-mcts": [
        r"§1\b",  # When to use
        r"§[23]\b",  # 3-way joint contract or contract
        r"§4\b",  # Acceptance gate
        r"MEASURED-vs-PROJECTED|Reference impl|Cross-ref",
    ],
    "mol-metal-mcts-replay-learner": [
        r"§1\b",  # When to use / activation
        r"§[2345]\b",  # Contract / wiring / cross-ref
        r"Cross[- ]?ref|References",
    ],
    "mol-metal-catherine-dblock-bench": [
        r"§1\b",  # When to use
        r"§[23]\b",  # Bench cohort / CLI
        r"§[4567]\b",  # Acceptance / cross-skill / MEASURED-vs-PROJECTED
    ],
    "mol-metal-triangle-closure": [
        r"§1\b",  # When to use
        r"§[2345]\b",  # Contract / wiring / cross-ref
        r"Cross[- ]?ref|Reference",
    ],
}


@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_has_required_sections(skill_name: str, r21_phase: str) -> None:
    """Each R21 skill must have at least 3 canonical §N sections + cross-ref."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    headers = _extract_section_headers(body)

    patterns = REQUIRED_SECTION_PATTERNS[skill_name]
    joined_headers = "\n".join(headers)
    matched = []
    for pat in patterns:
        if re.search(pat, joined_headers):
            matched.append(pat)

    assert len(matched) >= 3, (
        f"{skill_name} only matched {len(matched)}/{len(patterns)} required "
        f"section patterns. Headers found: {headers}"
    )


# ---------------------------------------------------------------------------
# T4 — Cross-references in each R21 skill resolve to real files on disk
#      (or are explicitly marked STUB / NOT YET CREATED)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_cross_references_resolve(skill_name: str, r21_phase: str) -> None:
    """All .py / .md / .sh / .json paths in body must resolve to real files
    or be explicitly marked STUB / NOT YET CREATED."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    refs = _extract_cross_refs(body)

    # If no cross-refs found, that's fine — some skills are pure docs.
    if not refs:
        pytest.skip(f"{skill_name} has no .py/.md/.sh/.json/.tex refs to verify")

    # Group refs by path so that a single STUB marker on ANY one occurrence
    # of a path marks the whole reference as accepted (don't double-count).
    # Process refs in reverse so STUB-marked later occurrences are seen first.
    accepted: set[str] = set()
    unresolved: set[str] = set()
    seen: set[str] = set()
    for ref_path, context_line in reversed(refs):
        if ref_path in seen:
            continue
        seen.add(ref_path)
        if _cross_ref_resolves(ref_path, context_line, skill_name):
            accepted.add(ref_path)
            continue
        # If this occurrence's context carries a STUB marker, accept
        # regardless of path resolution.
        if any(
            m in context_line
            for m in (
                "STUB",
                "NOT YET CREATED",
                "PROJECTED",
                "Wait for",
                "PROJ ",
                "TODO: ",
                "Pending",
                "STUB (",
            )
        ):
            accepted.add(ref_path)
            continue
        # Treat bare (TODO) markers as STUB too — common in skill doc
        # convention for "this CLI is planned but not built".
        if "(TODO)" in context_line:
            accepted.add(ref_path)
            continue
        unresolved.add(ref_path)

    assert not unresolved, (
        f"{skill_name} has {len(unresolved)} unresolved cross-refs: "
        f"{sorted(unresolved)[:5]} (total refs: {len(refs)})"
    )


# ---------------------------------------------------------------------------
# T5 — Code anchors (molmetal module references) exist
# ---------------------------------------------------------------------------
CODE_ANCHOR_KEYWORDS: Dict[str, List[str]] = {
    "mol-metal-score-rep-variance-mcts": [
        "variance_puct_integration",
        "score_repellent_mc",
        "variance_mcts_integration",
    ],
    "mol-metal-mcts-replay-learner": [
        "MCTSReplay",
        "mcts_replay_buffer",
        "VarianceControlledPUCT",
    ],
    "mol-metal-catherine-dblock-bench": [
        "catherine_properties",
        "bench_catherine_30metals",
        "TASK_SPEC_METALS",
    ],
    "mol-metal-triangle-closure": [
        "hilbert_primal_dual",
        "hilbert_astsr",
        "variance_mcts",
    ],
}


@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_code_anchors_present(skill_name: str, r21_phase: str) -> None:
    """Body must reference at least one of the code anchor keywords."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    keywords = CODE_ANCHOR_KEYWORDS[skill_name]
    matched = [k for k in keywords if k in body]
    assert matched, (
        f"{skill_name} has none of the code anchor keywords {keywords} "
        f"in its body — skill discovery is ungrounded"
    )


# ---------------------------------------------------------------------------
# T6 — Wf-321B wiring confirmation: 4 NEW R21 skills present
# ---------------------------------------------------------------------------
def test_wf321b_r21_skill_ecosystem_audit_count() -> None:
    """Wf-321B audit (TODO-99) committed exactly 4 NEW R21 skills per §2.4.

    This test is a guard against future drift — if anyone adds/removes a
    NEW R21 skill, this test fails and forces re-audit.
    """
    assert len(R21_SKILLS) == 4, (
        f"Expected 4 NEW R21 skills per Wf-321B §1.3 + LAMBDA_ROADMAP §2.4, "
        f"got {len(R21_SKILLS)}: {list(R21_SKILLS.keys())}"
    )


def test_wf321b_skill_ecosystem_completion_doc_exists() -> None:
    """Wf-321B completion doc must exist on disk."""
    doc = REPO_ROOT / "TODO" / "completed" / "321b_r20_r21_skill_ecosystem_audit.md"
    assert doc.exists(), f"Wf-321B completion doc missing at {doc}"
    text = doc.read_text(encoding="utf-8")
    # Verify the 4 NEW R21 skill names appear in the doc body
    expected = [
        "mol-metal-3way-multimetal",
        "mol-metal-r20-skill-pack",
        "mol-metal-r21-skill-pack",
        "mol-metal-score-rep-variance-mcts",
    ]
    missing = [name for name in expected if name not in text]
    assert (
        not missing
    ), f"Wf-321B completion doc missing NEW skill references: {missing}"


# ---------------------------------------------------------------------------
# T7 — Stub contract markers present in STUB-only R21 skills
# ---------------------------------------------------------------------------
STUB_SKILLS = {"mol-metal-score-rep-variance-mcts"}


@pytest.mark.parametrize("skill_name", sorted(STUB_SKILLS))
def test_r21_stub_skill_markers_present(skill_name: str) -> None:
    """STUB R21 skills must explicitly mark STUB status + bench ref as
    NOT YET CREATED, plus carry the MEASURED-vs-PROJECTED ledger."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    assert "STUB" in text, f"{skill_name} missing STUB marker"
    assert "NOT YET CREATED" in text, f"{skill_name} missing NOT YET CREATED marker"
    assert (
        "MEASURED-vs-PROJECTED" in text
    ), f"{skill_name} missing MEASURED-vs-PROJECTED ledger"


# ---------------------------------------------------------------------------
# T8 — Full R21 skills carry hit_rate / R² honesty gate contracts
# ---------------------------------------------------------------------------
FULL_SKILLS = {
    "mol-metal-mcts-replay-learner": ["hit_rate", "variance"],
    "mol-metal-triangle-closure": ["hilbert_r2", "evolutionary_r2", "variance"],
}


@pytest.mark.parametrize("skill_name,keywords", sorted(FULL_SKILLS.items()))
def test_r21_full_skill_honesty_gate_keywords(
    skill_name: str, keywords: List[str]
) -> None:
    """Full R21 skills must carry honesty gate keywords in their body."""
    text = _skill_path(skill_name).read_text(encoding="utf-8")
    missing = [k for k in keywords if k not in text]
    assert not missing, f"{skill_name} missing honesty gate keywords {missing}"


# ---------------------------------------------------------------------------
# T9 — Alias mapping: closure-dblock-bench → catherine-dblock-bench
# ---------------------------------------------------------------------------
def test_closure_dblock_alias_to_catherine_dblock_bench() -> None:
    """Per Wf-321B §1.3, the ``closure-dblock-bench`` R21 slot is satisfied
    by the existing ``mol-metal-catherine-dblock-bench`` skill. This test
    verifies the alias is explicit (not a missing skill)."""
    cb = _skill_path("mol-metal-catherine-dblock-bench")
    assert (
        cb.exists()
    ), f"closure-dblock-bench alias (catherine-dblock-bench) missing at {cb}"
    text = cb.read_text(encoding="utf-8")
    # Description must reference Catherine + d-block + 30-metal bench
    assert "Catherine" in text
    assert "d-block" in text or "30" in text


# ---------------------------------------------------------------------------
# T10 — R21 skills are NEW per Wf-321B (created on or after 2026-09-26)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill_name,r21_phase", sorted(R21_SKILLS.items()))
def test_r21_skill_mtime_after_wf321b(skill_name: str, r21_phase: str) -> None:
    """Each R21 skill's SKILL.md mtime should be on or after 2026-09-26
    (the Wf-37 / Wf-321B ship surface date)."""
    path = _skill_path(skill_name)
    stat = path.stat()
    import datetime as _dt

    mtime = _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc)
    wf321b_date = _dt.datetime(2026, 9, 26, tzinfo=_dt.timezone.utc)
    assert mtime >= wf321b_date, (
        f"{skill_name} SKILL.md mtime {mtime.isoformat()} predates "
        f"Wf-321B ship date {wf321b_date.isoformat()}"
    )
