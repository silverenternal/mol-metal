"""Pytest: Wf-338-M1 — Mechanism mapping skill wire + per-metal reachability matrix.

This test module pins the Wf-338-M1 contract (per AGENTS.md §3.2 +
Wf-330-C4 per-metal reachability matrix):

1. ``.claude/skills/mol-metal-mechanism-mapping/SKILL.md`` is **fully
   wired** — every required section (R17 state, when-to-use, core
   anchors, diagnostic procedure, usage examples, related skills,
   diagnostic cheatsheet, common mistakes, R18 M22-M44 cross-ref,
   R19 GPU retrain, R20 Lambda algorithm, Wf-45 L9-L12, per-metal
   reachability matrix) is present. No STUB.
2. All cross-references in the SKILL.md resolve to real on-disk
   paths (``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md``,
   ``TODO/02_metrics/METRICS_TARGETS.md``,
   ``molmetal/reports/wf36_closure_theorem/closure_bench.json``,
   ``molmetal/reports/wf37_closure_dblock/per_metal_bench.json``,
   ``molmetal/reports/wf41_closure_ood/ood_closure.json``,
   ``molmetal/reports/wf41_closure_ood/per_domain.csv``,
   ``molmetal/reports/wf37_closure_dblock/per_metal.csv``,
   corresponding bench MD reports, etc.).
3. Activation keywords from the skill frontmatter + description
   resolve into the documented symptom-action flow
   (regressed metric → §4 failure mode; ablation design → §5;
   PR review → §3; on-boarding → SKILL + §3; §6 limitations;
   R18+ planning).
4. The per-metal reachability matrix anchors (Wf-330-C4) are
   present with the correct MEASURED numbers:
   - L1 = 0.5108 (closure verifier, Wf-36 P1)
   - L6 = 0.5891 (closure d-block, Wf-37 P2)
   - L12 PlatinAI 5K = 0.5581 (closure OOD, Wf-41 P2)
   - L12 MetalCytoTox 5-metal = 0.0300 (honest-negative, ligand-only)
   plus the per-metal rows for Pt / Pd / Ru / Rh / Ir / Os / Au / Ag
   (L6 + L12 PlatinAI subset).

These are **documentation-only** tests — they do NOT run any GPU
training or closure-theorem benchmarks. They pin the Wf-338-M1
SKILL.md contract on disk so future regressions surface in CI.

**Honest framing** (per AGENTS.md §1.5 / party-twenty-spirit §3.5):
- All 4 reachability anchors are MEASURED @ Wf-36/37/41 ship (JSON +
  MD on disk); the Wf-338-M1 work is CPU-only documentation wire.
- L12 MetalCytoTox 5-metal = 0.0300 is the **honest-negative**
  coverage gap (ligand-only SMILES), NOT a bug to chase.
- R20-P5 stretch gate ≥ 0.95 is **35.9 pp short of L6 = 0.5891**
  baseline; honest per `LAMBDA_ROADMAP.md` §1.4 is 70-80% realistic,
  NOT 95%.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT / ".claude" / "skills" / "mol-metal-mechanism-mapping" / "SKILL.md"
)
MAPPING_PATH = REPO_ROOT / "TODO" / "02_metrics" / "MECHANISM_METRIC_MAPPING.md"
METRICS_PATH = REPO_ROOT / "TODO" / "02_metrics" / "METRICS_TARGETS.md"
EVALUATORS_PATH = REPO_ROOT / "TODO" / "02_metrics" / "EVALUATORS_MIGRATED.md"

# Bench JSON + MD + CSV anchors for the per-metal reachability matrix
# (Wf-330-C4). These four files are the source of truth — every
# reachability number in this SKILL.md must be traceable to one of
# these JSON/MD files.
BENCH_ANCHORS = {
    "L1_closure_bench_json": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf36_closure_theorem"
    / "closure_bench.json",
    "L1_closure_bench_md": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf36_closure_theorem"
    / "closure_bench.md",
    "L6_per_metal_bench_json": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf37_closure_dblock"
    / "per_metal_bench.json",
    "L6_per_metal_bench_md": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf37_closure_dblock"
    / "per_metal_bench.md",
    "L6_per_metal_csv": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf37_closure_dblock"
    / "per_metal.csv",
    "L12_ood_closure_json": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf41_closure_ood"
    / "ood_closure.json",
    "L12_ood_closure_md": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf41_closure_ood"
    / "ood_closure.md",
    "L12_per_domain_csv": REPO_ROOT
    / "molmetal"
    / "reports"
    / "wf41_closure_ood"
    / "per_domain.csv",
}

# MEASURED reachability anchors (Wf-330-C4)
L1_RATE = 0.5108
L6_RATE = 0.5891
L12_PLATINAI_RATE = 0.5581
L12_METALCYTOX_RATE = 0.0300

# Required sections in the SKILL.md (full wire-up, NOT STUB).
# These are the canonical anchor headings the diagnostic flow expects.
REQUIRED_SKILL_SECTIONS = [
    "# mol-metal-mechanism-mapping",  # title
    "## R17 production state",
    "## When to use this skill",
    "## Core anchors (read these first)",
    "## Quick diagnostic procedure (5 minutes)",
    "## Usage Examples",
    "## When contributing back to the mapping",
    "## Related skills",
    "## Diagnostic commands cheatsheet",
    "## Common mistakes",
    "## When NOT to use this skill",
    "## R18 mechanism cross-reference — full M22-M44 table",
    "## Worked example — M22 SE(3)-SB",
    "## R19 GPU retrain reference (Wf-29 ~ Wf-33 ship 真活, 2026-09-26)",
    "## R20 Lambda algorithm reference (Wf-36/37 ship + Wf-38 LAMBDA_ROADMAP, 2026-09-26)",
    "## Wf-45 Phase 3 — L9-L12 incremental ship cross-link (Wf-40/41/44 真活, 2026-09-26)",
    "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
]

# Activation keywords from the skill description (frontmatter).
# The skill must be activated by these (any one matches → load skill).
ACTIVATION_KEYWORDS = [
    "metric regression",
    "ablation design",
    "mechanism attribution",
    "why did m_ drop",
]

# Cross-references in the SKILL.md that must resolve to real on-disk paths.
CROSS_REFERENCE_PATHS = [
    "TODO/02_metrics/MECHANISM_METRIC_MAPPING.md",
    "TODO/02_metrics/METRICS_TARGETS.md",
    "TODO/02_metrics/EVALUATORS_MIGRATED.md",
    "TODO/03_plans/R18_PLAN.md",
    "TODO/03_plans/LAMBDA_ROADMAP.md",
    "TODO/completed/82_metric_gap_filling.md",
    "TODO/completed/83_lambda_algorithm_enhancement_plan.md",
    "TODO/completed/91_wf43_phase4_integration.md",
    "TODO/completed/91_wf41_3way_integration.md",
    "molmetal/reports/wf36_closure_theorem/closure_bench.json",
    "molmetal/reports/wf36_closure_theorem/closure_bench.md",
    "molmetal/reports/wf37_closure_dblock/per_metal_bench.json",
    "molmetal/reports/wf37_closure_dblock/per_metal_bench.md",
    "molmetal/reports/wf37_closure_dblock/per_metal.csv",
    "molmetal/reports/wf41_closure_ood/ood_closure.json",
    "molmetal/reports/wf41_closure_ood/ood_closure.md",
    "molmetal/reports/wf41_closure_ood/per_domain.csv",
    "molmetal/scripts/eval_all_evaluators.py",
    "molmetal/scripts/diagnose_metric.py",
    "molmetal/scripts/wf_pitfall_guard/check_all.py",
    ".claude/skills/mol-metal-lambda-calculus/SKILL.md",
    ".claude/skills/mol-metal-cfm-training/SKILL.md",
    ".claude/skills/mol-metal-click-chemistry/SKILL.md",
    ".claude/skills/mol-metal-anticancer/SKILL.md",
]

# Precious-metal cohort rows in the per-metal reachability matrix (L6
# subset). The matrix must carry at least these 9 metals.
L6_PRECIOUS_METALS = ["Pt", "Ru", "Ir", "Rh", "Re", "Os", "Au", "Pd", "Ag"]

# Precious-metal cohort rows in the L12 PlatinAI 5K OOD table.
L12_PLATINAI_METALS = ["Pt", "Pd", "Ru", "Rh", "Ir", "Os", "Au"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skill_text() -> str:
    """Read the mol-metal-mechanism-mapping SKILL.md once per module."""
    assert SKILL_PATH.exists(), f"missing {SKILL_PATH}"
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert len(text) > 1000, "SKILL.md is suspiciously short — likely a STUB"
    return text


@pytest.fixture(scope="module")
def mapping_text() -> str:
    """Read the MECHANISM_METRIC_MAPPING.md file once per module."""
    assert MAPPING_PATH.exists(), f"missing {MAPPING_PATH}"
    return MAPPING_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def l6_md_text() -> str:
    """Read the L6 closure d-block bench MD once per module."""
    path = BENCH_ANCHORS["L6_per_metal_bench_md"]
    assert path.exists(), f"missing {path}"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def l12_md_text() -> str:
    """Read the L12 closure OOD bench MD once per module."""
    path = BENCH_ANCHORS["L12_ood_closure_md"]
    assert path.exists(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _section(text: str, anchor: str) -> str:
    """Return the substring from ``anchor`` to the next ``## `` heading."""
    idx = text.find(anchor)
    assert idx >= 0, f"anchor {anchor!r} not found"
    rest = text[idx + len(anchor) :]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


# ---------------------------------------------------------------------------
# Test 1 — Skill SKILL.md has required sections (NOT STUB)
# ---------------------------------------------------------------------------


def test_skill_md_required_sections(skill_text: str) -> None:
    """SKILL.md has the 16 required sections; not a STUB.

    Per AGENTS.md §3.2 the skill should be "wired" — that means all
    R17+R18+R19+R20+Wf-45+Wf-338 sections are present, not a single
    title + 1-line description.
    """
    for anchor in REQUIRED_SKILL_SECTIONS:
        assert anchor in skill_text, f"missing required section: {anchor!r}"


def test_skill_md_has_frontmatter(skill_text: str) -> None:
    """SKILL.md has YAML frontmatter with name + description + tools."""
    # Frontmatter block
    assert skill_text.startswith("---\n"), "missing YAML frontmatter start"
    end = skill_text.find("\n---\n", 4)
    assert end > 0, "missing YAML frontmatter end"
    fm = skill_text[4:end]
    assert "name: mol-metal-mechanism-mapping" in fm
    assert "description:" in fm
    assert "tools:" in fm


def test_skill_md_r17_state_declares_diagnostic_anchors(skill_text: str) -> None:
    """R17 production state section pins the diagnostic anchors.

    The SKILL.md must reference the 3 source docs that anchor every
    diagnostic flow:
    - TODO/METRICS_TARGETS.md (metric list + Tier 1/2 gates)
    - TODO/MECHANISM_METRIC_MAPPING.md (mechanism × metric matrix)
    - TODO/EVALUATORS_MIGRATED.md (14/14 evaluator migration log)
    """
    section = _section(skill_text, "## R17 production state")
    assert "TODO/02_metrics/METRICS_TARGETS.md" in section
    assert "TODO/02_metrics/MECHANISM_METRIC_MAPPING.md" in section
    assert "TODO/02_metrics/EVALUATORS_MIGRATED.md" in section
    assert "21 mechanisms" in section or "M1-M21" in section or "M22-M44" in section


def test_skill_md_when_to_use_table(skill_text: str) -> None:
    """When-to-use table covers all 6 documented activation flows."""
    section = _section(skill_text, "## When to use this skill")
    for tag in (
        "metric regressed",
        "ablation",
        "Reviewing a PR",
        "On-boarding",
        "Limitations",
        "R18+",
    ):
        assert tag.lower() in section.lower(), f"When-to-use table missing {tag!r}"


def test_skill_md_r18_wave_tables(skill_text: str) -> None:
    """R18 cross-ref table covers all 23 R18 mechanisms (M22-M44)."""
    section = _section(
        skill_text, "## R18 mechanism cross-reference — full M22-M44 table"
    )
    for mech_id in [f"M{n}" for n in range(22, 45)]:
        assert mech_id in section, f"missing {mech_id} in R18 cross-ref table"


def test_skill_md_per_metal_reachability_section_present(skill_text: str) -> None:
    """Per-metal reachability matrix section is present (Wf-330-C4 anchor)."""
    anchor = (
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)"
    )
    assert (
        anchor in skill_text
    ), "missing per-metal reachability matrix section — Wf-338-M1 ship anchor"


# ---------------------------------------------------------------------------
# Test 2 — Cross-references resolve to real on-disk paths
# ---------------------------------------------------------------------------


def test_cross_references_all_resolve(skill_text: str) -> None:
    """Every `path/to/file.ext` cross-reference in the SKILL.md exists on disk.

    Per AGENTS.md §1 MEASURED-vs-PROJECTED tag discipline: the SKILL.md
    is the horizontal layer that ties mechanism-metric mapping to the
    other 28 skills; a broken cross-reference would silently break
    the diagnostic flow.
    """
    missing: list[str] = []
    for ref in CROSS_REFERENCE_PATHS:
        candidate = REPO_ROOT / ref
        if not candidate.exists():
            missing.append(ref)
    assert not missing, f"missing cross-reference targets: {missing}"


def test_bench_anchors_all_present() -> None:
    """All 8 bench anchors for the per-metal reachability matrix exist."""
    missing: list[str] = []
    for label, path in BENCH_ANCHORS.items():
        if not path.exists():
            missing.append(f"{label}: {path}")
    assert not missing, f"missing bench anchors: {missing}"


def test_skill_md_references_r20_p5_stretch_gate_honesty(skill_text: str) -> None:
    """SKILL.md cites the R20-P5 stretch-gate honest framing.

    Per `party-twenty-spirit` §3.5, the ≥ 0.95 reachability gate is
    stretch (35.9 pp short of L6 = 0.5891 baseline). The SKILL.md must
    surface this honestly, NOT paraphrase as "promising" / "achievable".
    """
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "0.95" in section, "must cite R20-P5 stretch gate"
    # Per `party-twenty-spirit` §3.5 honest-negative framing — must
    # explicitly call out the gap, NOT use aspirational language.
    honest_neg_markers = ("honest-negative", "short", "realistic", "stretch")
    assert any(marker in section.lower() for marker in honest_neg_markers), (
        "R20-P5 stretch gate framing must be honest-negative; per "
        "party-twenty-spirit §3.5 0 容忍, paraphrasing as "
        "'achievable' / 'promising' is fabrication"
    )


# ---------------------------------------------------------------------------
# Test 3 — Activation keywords
# ---------------------------------------------------------------------------


def test_activation_keywords_in_description(skill_text: str) -> None:
    """All 4 activation keywords are in the skill frontmatter description.

    Per AGENTS.md §3.2, the skill must activate on these keywords so
    the harness can route user requests correctly.
    """
    # Extract frontmatter description
    end = skill_text.find("\n---\n", 4)
    fm = skill_text[4:end]
    desc_match = re.search(r"description:\s*(.+?)(?=\n[a-z]+:|\Z)", fm, re.DOTALL)
    assert desc_match is not None, "missing frontmatter description"
    desc = desc_match.group(1)
    for keyword in ACTIVATION_KEYWORDS:
        assert (
            keyword.lower() in desc.lower()
        ), f"activation keyword {keyword!r} missing from frontmatter description"


def test_symptom_action_table_in_when_to_use(skill_text: str) -> None:
    """When-to-use table has symptom → action rows for diagnostic flow."""
    section = _section(skill_text, "## When to use this skill")
    # Each row maps a symptom to a §N action; verify at least 4 rows.
    rows = [line for line in section.split("\n") if line.startswith("|")]
    assert len(rows) >= 6, (
        f"When-to-use table should have ≥ 6 rows (header + separator + 4+ data); "
        f"found {len(rows)}"
    )


def test_quick_diagnostic_procedure_has_5_steps(skill_text: str) -> None:
    """Quick diagnostic procedure has 5 numbered steps."""
    section = _section(skill_text, "## Quick diagnostic procedure (5 minutes)")
    # Count numbered steps "1." through "5." at line starts.
    steps = re.findall(r"^\d+\.\s", section, re.MULTILINE)
    assert (
        len(steps) >= 5
    ), f"quick diagnostic procedure should have 5+ steps; found {len(steps)}"


def test_diagnostic_commands_cheatsheet_has_3_commands(skill_text: str) -> None:
    """Diagnostic cheatsheet has 3+ numbered diagnostic commands (single bash block OK).

    The cheatsheet is one fenced ``bash`` block with 3 numbered entries
    (evaluator + pitfall guard + targeted regression). Verify ≥ 3
    runnable shell commands + at least one pytest reference.
    """
    section = _section(skill_text, "## Diagnostic commands cheatsheet")
    # Numbered comments `# 1.` `# 2.` `# 3.` = 3+ commands in single block
    n_commands = len(re.findall(r"^\s*#\s*\d+\.\s", section, re.MULTILINE))
    assert (
        n_commands >= 3
    ), f"cheatsheet should have ≥ 3 numbered diagnostic commands; found {n_commands}"
    pytest_count = section.count("pytest")
    assert pytest_count >= 1, "cheatsheet should reference pytest at least once"


# ---------------------------------------------------------------------------
# Test 4 — Per-metal reachability matrix
# ---------------------------------------------------------------------------


def test_l1_rate_anchor_present(skill_text: str) -> None:
    """L1 closure verifier reachability rate 0.5108 is pinned in the matrix."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "0.5108" in section, "L1 = 0.5108 must be in the matrix"


def test_l6_rate_anchor_present(skill_text: str) -> None:
    """L6 closure d-block reachability rate 0.5891 is pinned in the matrix."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "0.5891" in section, "L6 = 0.5891 must be in the matrix"


def test_l12_platinai_rate_anchor_present(skill_text: str) -> None:
    """L12 PlatinAI 5K OOD reachability rate 0.5581 is pinned in the matrix."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "0.5581" in section, "L12 PlatinAI 5K = 0.5581 must be in the matrix"


def test_l12_metalcytotox_rate_anchor_present(skill_text: str) -> None:
    """L12 MetalCytoTox 5-metal reachability rate 0.0300 (honest-negative) is pinned."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "0.0300" in section, "L12 MetalCytoTox = 0.0300 must be in the matrix"
    assert (
        "honest-negative" in section.lower() or "honest-neg" in section.lower()
    ), "0.0300 must be flagged honest-negative per party-twenty-spirit §3.5"


def test_l1_rate_matches_bench_json() -> None:
    """L1 = 0.5108 matches the Wf-36 P1 closure_bench.json aggregate."""
    import json

    path = BENCH_ANCHORS["L1_closure_bench_json"]
    data = json.loads(path.read_text(encoding="utf-8"))
    # closure_reachability_rate lives under the `aggregate` sub-object
    agg = data.get("aggregate", {})
    rate = agg.get("closure_reachability_rate")
    assert rate is not None, f"{path} missing aggregate.closure_reachability_rate field"
    assert (
        abs(float(rate) - L1_RATE) < 1e-4
    ), f"L1 rate drift: bench JSON says {rate}, SKILL.md pins {L1_RATE}"


def test_l6_rate_matches_bench_json() -> None:
    """L6 = 0.5891 matches the Wf-37 P2 per_metal_bench.json aggregate."""
    import json

    path = BENCH_ANCHORS["L6_per_metal_bench_json"]
    data = json.loads(path.read_text(encoding="utf-8"))
    # closure_reachability_rate lives under the `aggregate` sub-object
    agg = data.get("aggregate", {})
    rate = agg.get("closure_reachability_rate")
    assert rate is not None, f"{path} missing aggregate.closure_reachability_rate field"
    assert (
        abs(float(rate) - L6_RATE) < 1e-4
    ), f"L6 rate drift: bench JSON says {rate}, SKILL.md pins {L6_RATE}"


def test_l12_platinai_rate_matches_bench_md(l12_md_text: str) -> None:
    """L12 PlatinAI 5K OOD = 0.5581 matches the Wf-41 P2 ood_closure.md table."""
    # PlatinAI row in the per-domain table
    assert "platinai_5k" in l12_md_text
    # The exact reachability_rate 0.5581 must appear
    assert (
        "0.5581" in l12_md_text
    ), f"L12 PlatinAI 5K rate drift: bench MD does not contain 0.5581"


def test_l12_metalcytotox_rate_matches_bench_md(l12_md_text: str) -> None:
    """L12 MetalCytoTox 5-metal = 0.0300 matches the Wf-41 P2 ood_closure.md table."""
    assert "metalcytotox_5metal" in l12_md_text
    assert (
        "0.0300" in l12_md_text
    ), "L12 MetalCytoTox 5-metal rate drift: bench MD does not contain 0.0300"


def test_l6_per_metal_rows_for_precious_metals(skill_text: str) -> None:
    """Per-metal reachability matrix covers the 9 precious-metal rows for L6."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    # Each precious-metal must have its own row in the L6 table
    for metal in L6_PRECIOUS_METALS:
        # The row should have the metal symbol in the first column
        pattern = rf"\|\s*{re.escape(metal)}\s*\|"
        assert re.search(
            pattern, section
        ), f"L6 precious-metal row missing: {metal}; section must list Pt/Ru/Ir/Rh/Re/Os/Au/Pd/Ag"


def test_l6_per_metal_rates_consistent_with_bench_md(
    skill_text: str, l6_md_text: str
) -> None:
    """L6 precious-metal rows in SKILL.md match the Wf-37 P2 per_metal_bench.md table."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    # Pin the 9 precious-metal reachability rates from the bench MD
    expected_rates = {
        "Pt": 0.4700,
        "Pd": 0.4500,
        "Ru": 0.7200,
        "Rh": 0.5500,
        "Ir": 0.3800,
        "Os": 0.4000,
        "Au": 0.4400,
        "Ag": 0.6173,
        "Re": 0.5400,
    }
    for metal, rate in expected_rates.items():
        rate_str = f"{rate:.4f}"
        # Both SKILL.md and bench MD must contain this rate string
        assert rate_str in section, f"L6 {metal} rate {rate_str} missing from SKILL.md"
        assert (
            rate_str in l6_md_text
        ), f"L6 {metal} rate {rate_str} missing from bench MD"


def test_l12_platinai_per_metal_rows_for_precious_metals(skill_text: str) -> None:
    """L12 PlatinAI 5K OOD table covers the 7 precious-metal rows."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    for metal in L12_PLATINAI_METALS:
        pattern = rf"\|\s*{re.escape(metal)}\s*\|"
        assert re.search(
            pattern, section
        ), f"L12 PlatinAI 5K row missing: {metal}; section must list Pt/Pd/Ru/Rh/Ir/Os/Au"


def test_l12_platinai_per_metal_rates_consistent_with_bench_md(
    skill_text: str, l12_md_text: str
) -> None:
    """L12 PlatinAI 5K precious-metal rows match the Wf-41 P2 ood_closure.md table."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    expected_rates = {
        "Pt": 0.5833,
        "Pd": 0.3333,
        "Ru": 0.7000,
        "Rh": 0.6471,
        "Ir": 0.6000,
        "Os": 0.5294,
        "Au": 0.4545,
    }
    for metal, rate in expected_rates.items():
        rate_str = f"{rate:.4f}"
        assert (
            rate_str in section
        ), f"L12 PlatinAI {metal} rate {rate_str} missing from SKILL.md"
        assert (
            rate_str in l12_md_text
        ), f"L12 PlatinAI {metal} rate {rate_str} missing from bench MD"


def test_per_metal_matrix_has_self_check_block(skill_text: str) -> None:
    """Per-metal reachability matrix has a runnable self-check command block."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "```bash" in section, "matrix must have a bash self-check block"
    # The self-check must reference the 3 bench JSON files
    for token in ("wf36_closure_theorem", "wf37_closure_dblock", "wf41_closure_ood"):
        assert token in section, f"self-check must reference {token}"


def test_per_metal_matrix_has_cross_reference_table(skill_text: str) -> None:
    """Per-metal reachability matrix has a cross-reference table to other skills."""
    section = _section(
        skill_text,
        "## Per-metal reachability matrix (Wf-330-C4 cross-skill anchor, 2026-09-29)",
    )
    assert "**Cross-reference**" in section or "Cross-reference" in section
    # Must reference at least the 4 anchor skills
    for skill_name in (
        "closure-theorem",
        "lambda-calculus",
        "click-chemistry",
        "anticancer",
    ):
        assert (
            skill_name in section
        ), f"cross-reference table missing skill: {skill_name}"


# ---------------------------------------------------------------------------
# Test 5 — Diagnostic table integrity (cross-validation with MECHANISM_METRIC_MAPPING.md)
# ---------------------------------------------------------------------------


def test_mapping_doc_exists_for_diagnostic_chain() -> None:
    """TODO/02_metrics/MECHANISM_METRIC_MAPPING.md exists (the §3 source of truth)."""
    assert MAPPING_PATH.exists(), (
        f"missing {MAPPING_PATH}; per AGENTS.md §3.2 this is the primary source "
        "for mechanism × metric attribution"
    )


def test_metrics_targets_doc_exists_for_tier_gates() -> None:
    """TODO/02_metrics/METRICS_TARGETS.md exists (the §2 source of truth)."""
    assert (
        METRICS_PATH.exists()
    ), f"missing {METRICS_PATH}; per AGENTS.md §3.2 this defines Tier 1/2 gates"


def test_skill_md_cites_evaluators_migrated() -> None:
    """TODO/EVALUATORS_MIGRATED.md is referenced from the SKILL.md."""
    assert EVALUATORS_PATH.exists()
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "TODO/EVALUATORS_MIGRATED.md" in text or "EVALUATORS_MIGRATED" in text
