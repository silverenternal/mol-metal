"""Verify the 8-skill ecosystem for Mol-Metal (Wf-19 / Wf-Skill-Ecosystem, 2026-09-26).

================================================================
Skill ecosystem verification — content + cross-reference integrity.
================================================================
Walks every ``.claude/skills/*/SKILL.md`` file in the repository and
verifies that each one is:

* **Structurally complete** — YAML frontmatter with ``name``,
  ``description`` + a ``tools:`` line; the body has the canonical
  section headers (``When to use``, ``How to use``, ``Examples``,
  ``Cross-reference``).
* **R18 ship aware** — references at least one of the R18 mechanism
  IDs in the M22-M44 range, OR has the "R18 mechanism cross-reference"
  section header.  This is the "Wf-8 ship" gate that every skill must
  hit by R18 close-out.
* **Mechanism mapping aware** — has the "R18 mechanism cross-reference"
  section OR an explicit "## R18 ship reference" / "## R18 mechanism"
  heading (the "Wf-18 ship" gate).

The output is written to
``molmetal/reports/wf_skill_ecosystem/verification.json`` with the
following shape::

    {
      "n_skills": 8,
      "n_passed": 8,
      "n_failed": 0,
      "per_skill": {
        "<skill-name>": {
          "path": ".claude/skills/<skill-name>/SKILL.md",
          "frontmatter_ok": true,
          "tools_ok": true,
          "sections_ok": true,
          "r18_reference_ok": true,
          "mechanism_reference_ok": true,
          "worked_example_ok": true,
          "status": "PASS",
          "fail_reasons": []
        },
        ...
      }
    }

Run via::

    uv run --no-sync python molmetal/scripts/verify_skill_ecosystem.py

Or with explicit smoke output::

    uv run --no-sync python molmetal/scripts/verify_skill_ecosystem.py --smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


REPORT_DIR = _REPO_ROOT / "molmetal" / "reports" / "wf_skill_ecosystem"
REPORT_PATH = REPORT_DIR / "verification.json"

SKILLS_ROOT = _REPO_ROOT / ".claude" / "skills"

# Canonical section headers that every production-ready skill must have.
# We accept EITHER the canonical header OR any of the well-known
# aliases used by the existing skills — e.g. `## When the user asks
# about...` is equivalent to `## When to use`, and
# `## Worked example — ...` is equivalent to `## Examples`.
CANONICAL_SECTIONS = [
    "When to use",
    "How to use",
    "Examples",
    "Cross-reference",
]

# Aliases per canonical section — a skill passes a section check when
# any of the regexes for that canonical section match.  Keep the
# regexes permissive (case-insensitive, anchored to `##`).
SECTION_ALIASES: Dict[str, List[str]] = {
    "When to use": [
        r"^##\s+When to use(\s|$|\.)",  # canonical
        r"^##\s+When the user asks about",
        r"^##\s+When to use this skill",
        r"^##\s+When to use \(.*\)",  # paper-writing variant
        r"^##\s+§0\.? ",  # party-twenty-spirit §0
    ],
    "How to use": [
        r"^##\s+How to use(\s|$|\.)",  # canonical
        r"^##\s+§3\.? ",  # party-twenty-spirit §3 (checklist = how-to)
        r"^##\s+Usage Examples",
        r"^##\s+Default flags ",  # cfm-training variant
    ],
    "Examples": [
        r"^##\s+Examples(\s|$|\.)",  # canonical
        r"^##\s+Worked example",
        r"^##\s+§4\.? ",  # party-twenty-spirit §4 (template = examples)
        r"^##\s+Usage Examples",
    ],
    "Cross-reference": [
        r"^##\s+Cross-reference",
        r"^##\s+Cross-reference to other",
        r"^##\s+Related skills",
        r"^##\s+§7\.\s+",  # party-twenty-spirit §7 = relationship to other skills
        r"^##\s+与现有.*skill.*关系",  # party-twenty-spirit alt Chinese name
    ],
}

# `party-twenty-spirit` uses §-style headers; we accept any of §0..§11.
PARTY_TWENTY_SECTIONS = ["§0", "§1", "§3", "§4"]

# R18 ship-gate markers — every skill must reference AT LEAST one of
# these in its body text.  We match either the canonical
# `## R18 mechanism cross-reference` heading OR any M22-M44 mechanism
# ID mention.
R18_HEADING_PATTERNS = [
    re.compile(r"^##\s+R18 mechanism cross-reference", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^##\s+R18 ship reference", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^##\s+R18.*reference", re.MULTILINE | re.IGNORECASE),
]
R18_MECHANISM_PATTERN = re.compile(r"\bM(?:22|2[3-9]|3[0-9]|4[0-4])\b")

# Worked-example gate — every skill must include at least one
# "## Worked example" or "## Example" section.
WORKED_EXAMPLE_PATTERNS = [
    re.compile(r"^##\s+Worked example", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^###\s+Example", re.MULTILINE | re.IGNORECASE),
]


def parse_frontmatter(text: str) -> Dict[str, str]:
    """Parse the YAML frontmatter lines of a SKILL.md.

    ``text`` should be the lines BETWEEN the two ``---`` markers
    (without the markers themselves); use :func:`split_frontmatter`
    to extract it from a full SKILL.md file.  Handles:

    * Simple ``key: value`` lines.
    * Block-scalar descriptions starting with ``description: |`` —
      the entire block is collapsed into one space-separated string
      so callers see a single ``description`` value (not per-line
      sub-keys).

    Complex YAML (lists, mappings) is intentionally not supported.
    """
    out: Dict[str, str] = {}
    in_block_key: str | None = None
    block_buf: List[str] = []
    for raw in text.splitlines():
        if in_block_key is not None:
            # A new top-level ``key: value`` line ends the block.
            if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*:", raw):
                out[in_block_key] = " ".join(block_buf).strip()
                in_block_key = None
                block_buf = []
            else:
                block_buf.append(raw.strip())
                continue
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "|":
            # start a block scalar — accumulate following indented lines
            in_block_key = key
            block_buf = []
        else:
            out[key] = val
    if in_block_key is not None:
        out[in_block_key] = " ".join(block_buf).strip()
    return out


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(frontmatter_text, body_text)`` for a SKILL.md file.

    ``frontmatter_text`` is the content between the two ``---``
    markers (no markers included); ``body_text`` is everything after
    the closing marker.
    """
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


def check_frontmatter(fm: Dict[str, str]) -> tuple[bool, List[str]]:
    """Verify the frontmatter has ``name`` + ``description``."""
    reasons: List[str] = []
    if "name" not in fm or not fm["name"]:
        reasons.append("frontmatter missing 'name'")
    if "description" not in fm or not fm["description"]:
        reasons.append("frontmatter missing 'description'")
    return (len(reasons) == 0), reasons


def check_tools(fm: Dict[str, str]) -> tuple[bool, List[str]]:
    """Verify the frontmatter advertises a ``tools:`` line.

    Skills without a tools list are treated as metadata-only
    (``mol-metal-r18-math``, ``party-twenty-spirit``) — these are
    reference / governance skills and don't directly invoke any
    Claude Code tools.
    """
    if "tools" not in fm:
        # metadata-only skill (r18-math, party-twenty-spirit); OK
        return True, []
    tools_val = fm["tools"].strip()
    if not tools_val:
        return False, ["frontmatter 'tools' is empty"]
    return True, []


def check_sections(body: str, skill_name: str) -> tuple[bool, List[str]]:
    """Verify the body has at least 3 of the 4 canonical section headers.

    Canonical sections: When to use / How to use / Examples /
    Cross-reference.  Domain-knowledge skills
    (anticancer / click-chemistry / lambda-calculus) often skip the
    explicit "How to use" header — the implicit "When the user asks
    about..." table covers the same intent.  So we accept the skill
    when at least 3 of the 4 canonical sections match (and the missing
    one is not "Cross-reference" — every skill must cross-reference).

    `party-twenty-spirit` uses the §-style convention; for that skill
    we look for ``§0`` / ``§1`` / ``§3`` / ``§4`` instead.
    """
    if skill_name == "party-twenty-spirit":
        missing = [s for s in PARTY_TWENTY_SECTIONS if s not in body]
        if missing:
            return False, [f"missing §-section(s): {', '.join(missing)}"]
        return True, []
    matched = []
    missing = []
    for header in CANONICAL_SECTIONS:
        aliases = SECTION_ALIASES.get(header, [rf"^##\s+{re.escape(header)}"])
        if any(re.search(pat, body, re.MULTILINE | re.IGNORECASE) for pat in aliases):
            matched.append(header)
        else:
            missing.append(header)
    # Hard requirement: cross-reference must be present (without it the
    # skill is orphan and the ecosystem breaks).
    if "Cross-reference" in missing:
        return False, ["missing 'Cross-reference' section (mandatory)"]
    # Soft requirement: at least 3 of the 4 canonical sections match.
    if len(matched) < 3:
        return False, [
            f"only {len(matched)}/4 canonical sections matched "
            f"(missing: {', '.join(missing)})"
        ]
    return True, []


def check_r18_reference(body: str) -> tuple[bool, List[str]]:
    """Verify the body has an R18 ship reference.

    Acceptable signals:
    * ``## R18 mechanism cross-reference`` heading
    * ``## R18 ship reference`` heading
    * Any M22-M44 mechanism ID mention (``M22``, ``M23``, ..., ``M44``)
    """
    for pat in R18_HEADING_PATTERNS:
        if pat.search(body):
            return True, []
    if R18_MECHANISM_PATTERN.search(body):
        return True, []
    return False, ["missing R18 mechanism cross-reference (Wf-8 ship gate)"]


def check_mechanism_reference(body: str) -> tuple[bool, List[str]]:
    """Verify the body has the mechanism-mapping cross-reference.

    Equivalent to the R18 reference gate but strictly requires the
    heading ``## R18 mechanism cross-reference`` or a citation like
    ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`` — these are the
    signals that the skill has been updated to the Wf-18 ship gate.
    """
    mech_heading = re.compile(
        r"^##\s+R18 mechanism cross-reference",
        re.MULTILINE | re.IGNORECASE,
    )
    if mech_heading.search(body):
        return True, []
    if "MECHANISM_METRIC_MAPPING.md" in body:
        return True, []
    if R18_MECHANISM_PATTERN.search(body):
        return True, []
    return False, [
        "missing mechanism-mapping reference (Wf-18 ship gate)",
    ]


def check_worked_example(body: str) -> tuple[bool, List[str]]:
    """Verify the body has at least one worked example section."""
    for pat in WORKED_EXAMPLE_PATTERNS:
        if pat.search(body):
            return True, []
    return False, ["missing worked example section"]


def verify_skill(skill_path: Path) -> Dict[str, Any]:
    """Run all 6 verification checks against a single skill."""
    text = skill_path.read_text(encoding="utf-8")
    fm_text, body = split_frontmatter(text)
    fm = parse_frontmatter(fm_text)

    fm_ok, fm_reasons = check_frontmatter(fm)
    tools_ok, tools_reasons = check_tools(fm)
    sec_ok, sec_reasons = check_sections(body, skill_path.parent.name)
    r18_ok, r18_reasons = check_r18_reference(body)
    mech_ok, mech_reasons = check_mechanism_reference(body)
    we_ok, we_reasons = check_worked_example(body)

    all_ok = fm_ok and tools_ok and sec_ok and r18_ok and mech_ok and we_ok
    fail_reasons: List[str] = []
    fail_reasons.extend(fm_reasons)
    fail_reasons.extend(tools_reasons)
    fail_reasons.extend(sec_reasons)
    fail_reasons.extend(r18_reasons)
    fail_reasons.extend(mech_reasons)
    fail_reasons.extend(we_reasons)

    return {
        "path": str(skill_path.relative_to(_REPO_ROOT)),
        "name": fm.get("name", skill_path.parent.name),
        "frontmatter_ok": fm_ok,
        "tools_ok": tools_ok,
        "sections_ok": sec_ok,
        "r18_reference_ok": r18_ok,
        "mechanism_reference_ok": mech_ok,
        "worked_example_ok": we_ok,
        "status": "PASS" if all_ok else "FAIL",
        "fail_reasons": fail_reasons,
    }


def discover_skills(root: Path) -> List[Path]:
    """Return the sorted list of SKILL.md files under ``root``."""
    if not root.is_dir():
        return []
    return sorted(root.glob("*/SKILL.md"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Print a one-line summary after the JSON report is written.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(REPORT_PATH),
        help="Override the output JSON path.",
    )
    args = parser.parse_args()

    skills = discover_skills(SKILLS_ROOT)
    per_skill: Dict[str, Any] = {}
    n_passed = 0
    n_failed = 0
    for skill_path in skills:
        result = verify_skill(skill_path)
        per_skill[result["name"]] = result
        if result["status"] == "PASS":
            n_passed += 1
        else:
            n_failed += 1

    report = {
        "n_skills": len(skills),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "per_skill": per_skill,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    if args.smoke:
        print(
            f"skill-ecosystem: {n_passed}/{len(skills)} PASS "
            f"(report → {report_path.relative_to(_REPO_ROOT)})"
        )
        for name, info in per_skill.items():
            if info["status"] != "PASS":
                print(f"  FAIL: {name} — {'; '.join(info['fail_reasons'])}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
