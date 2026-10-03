"""F3 — full pipeline demo guard.

This test verifies the ``wf_full_pipeline.sh`` end-to-end demo script
delivered by the R17 IMPROVEMENT_BACKLOG F3 item.  It does NOT execute
the pipeline (that is the responsibility of CI / a manual smoke run) —
it only proves that the wiring on this host is correct:

* The script exists at the canonical path and is executable.
* It defines all 6 expected sections.
* Each section invokes a Python entry point that actually exists.
* Each entry point's required CLI flag is present in the script.

Run with::

    cd /home/hugo/codes/mol-metal
    source env.sh
    pytest -q molmetal/tests/test_full_pipeline_demo.py

Honest framing
==============

This is a static-only guard.  If you want to verify the demo actually
*runs end-to-end* on this host, execute the script directly::

    bash molmetal/scripts/wf_full_pipeline.sh 2>&1 | tee /tmp/full_pipeline_test.log
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Canonical locations
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_full_pipeline.sh"

# Section index -> (display label, python entry point, must-contain flag).
# Adding a new section requires updating BOTH the script and this table.
EXPECTED_SECTIONS = [
    (
        "1/6",
        "Eval all evaluators",
        "molmetal/scripts/eval_all_evaluators.py",
        "eval_all_evaluators.py",
    ),
    (
        "2/6",
        "Run pitfall guards",
        "molmetal/scripts/wf_pitfall_guard/check_all.py",
        "wf_pitfall_guard/check_all.py",
    ),
    (
        "3/6",
        "Train 200-step CFM smoke",
        "molmetal/scripts/metallo_drug_smoke_retrain.py",
        "metallo_drug_smoke_retrain.py",
    ),
    (
        "4/6",
        "10-pocket Lambda pilot",
        "molmetal/scripts/r4_lambda_only_run.py",
        "r4_lambda_only_run.py",
    ),
    (
        "5/6",
        "Aggregate metrics",
        "molmetal/scripts/eval_all_evaluators.py",
        "eval_all_evaluators.py",
    ),
    (
        "6/6",
        "Tier-1 summary (diagnose m14)",
        "molmetal/scripts/diagnose_metric.py",
        "diagnose_metric.py",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_script() -> str:
    assert SCRIPT.exists(), f"wf_full_pipeline.sh not found at {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_wf_full_pipeline_script_exists():
    """Script must live at the canonical R17 path."""
    assert SCRIPT.exists(), f"missing: {SCRIPT}"


def test_wf_full_pipeline_script_is_executable():
    """Script must be chmod +x so it can be invoked directly."""
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT} is not user-executable"
    assert mode & stat.S_IXGRP, f"{SCRIPT} is not group-executable"


def test_wf_full_pipeline_script_has_shebang():
    """Top line must be a bash shebang so `bash script.sh` and `./script.sh` both work."""
    first_line = SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), f"unexpected shebang: {first_line!r}"
    assert "bash" in first_line, f"expected bash shebang, got {first_line!r}"


@pytest.mark.parametrize(
    "section_idx,label,entry_point,needle",
    EXPECTED_SECTIONS,
    ids=[s[0] for s in EXPECTED_SECTIONS],
)
def test_wf_full_pipeline_has_section(section_idx, label, entry_point, needle):
    """Each of the 6 sections must be present and reference a real script."""
    text = _read_script()
    # Section header pattern, e.g. "[1/6] Eval all evaluators ..."
    pattern = rf"\[{re.escape(section_idx)}\]"
    assert re.search(
        pattern, text
    ), f"section header {section_idx!r} not found in wf_full_pipeline.sh"
    assert needle in text, (
        f"section {section_idx} ({label}) does not invoke {entry_point} "
        f"(missing needle {needle!r} in wf_full_pipeline.sh)"
    )
    # And the Python entry point must actually exist on disk.
    ep = REPO_ROOT / entry_point
    assert ep.exists(), f"section {section_idx} references missing file: {ep}"


def test_wf_full_pipeline_has_exactly_six_sections():
    """We pin the section count to 6 so removing/adding a section forces an update."""
    text = _read_script()
    # The script uses "[N/6]" headers (with the literal /6 suffix).
    section_headers = re.findall(r"\[(\d+)/6\]", text)
    # The summary line at the end also reuses "[N/6]" so de-duplicate.
    section_headers = sorted(set(section_headers))
    assert section_headers == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ], f"expected 6 sections [1..6], got {section_headers}"


def test_wf_full_pipeline_critical_flags_present():
    """Pin the load-bearing CLI flags so a silent arg rename fails CI."""
    text = _read_script()
    # [3/6] metallo_drug_smoke_retrain flags
    for flag in ("--steps 200", "--batch-size 4", "--hidden-dim 64", "--n-layers 2"):
        assert flag in text, f"smoke retrain missing flag {flag!r}"
    # [4/6] r4_lambda_only_run flags
    for flag in (
        "--metal-seed cisplatin",
        "--pockets 10",
        "--seeds 0",
        "--n-simulations 100",
        "--click-rules auto-pt-strict",
    ):
        assert flag in text, f"lambda pilot missing flag {flag!r}"
    # [6/6] diagnose metric flag
    assert "--metric m14" in text, "diagnose missing --metric m14"


def test_wf_full_pipeline_sources_env_sh():
    """If env.sh is present the script must source it (matches project convention)."""
    text = _read_script()
    assert (
        "source" in text and "env.sh" in text
    ), "wf_full_pipeline.sh does not source env.sh — uv/venv routing may break"


def test_wf_full_pipeline_runs_within_cpu_budget():
    """Smoke check: no individual section may request more than 10 minutes
    (the demo must complete in <10 min on CPU).  This guards against
    someone raising a per-section budget that breaks the F3 SLA.

    The script declares per-section budgets as ``SEC_BUDGET_<NAME>=NNN``
    variables at the top; we parse those instead of the function-call
    arguments (which are multi-line and use shell variable references
    that are awkward to match with a single regex).
    """
    text = _read_script()
    matches = re.findall(r"SEC_BUDGET_[A-Z_]+\s*=\s*(\d+)", text)
    assert matches, "no SEC_BUDGET_* variables defined"
    for budget in (int(m) for m in matches):
        assert budget <= 600, (
            f"per-section budget {budget}s exceeds 10 min — would break the "
            f"F3 <10 min SLA"
        )
    # Sanity: 6 sections => 6 budget variables.
    assert (
        len(matches) == 6
    ), f"expected 6 SEC_BUDGET_* variables (one per section), got {len(matches)}"
