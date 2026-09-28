"""Pytest: Wf-35 Phase 4 — ``.github/workflows/ci.yml`` contract.

This module pins the CI workflow contract for the Wf-35 Phase 4
master-verify integration.  It guards against accidental drift in:

* The ``r18-train-scripts-master-verify`` job's presence and shape.
* The runner + timeout contract (CPU-only ``ubuntu-latest``,
  15-min cap).
* The script invocation path (``molmetal/scripts/master_verify_r18.py``).
* The YAML structure (parses without errors, top-level ``jobs`` key).

All tests are CPU-only and finish in <1 s.  They do NOT run the CI
workflow itself — they only lint the on-disk YAML.

CLI::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_ci_config.py -q --tb=short
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — pyyaml is in pyproject.toml deps
    yaml = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CI_YML = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
MASTER_VERIFY_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "master_verify_r18.py"
)

JOB_NAME = "r18-train-scripts-master-verify"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ci_workflow() -> Dict[str, Any]:
    """Load + parse ``ci.yml`` exactly once for the test module.

    Skips if the file is missing or PyYAML is unavailable so the
    test does not hard-fail on hosts without optional deps.
    """
    if not CI_YML.exists():
        pytest.skip(f"ci.yml not present at {CI_YML}")
    if yaml is None:
        pytest.skip("PyYAML not installed")
    loaded = yaml.safe_load(CI_YML.read_text())
    assert isinstance(loaded, dict), "ci.yml top-level must be a mapping"
    return loaded


@pytest.fixture(scope="module")
def r18_job(ci_workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the Wf-35 Phase 4 master-verify job out of the workflow."""
    jobs = ci_workflow["jobs"]
    assert JOB_NAME in jobs, (
        f"missing {JOB_NAME!r} job; present jobs: {sorted(jobs)}"
    )
    return jobs[JOB_NAME]


# ---------------------------------------------------------------------------
# 1. File existence + parse-ability
# ---------------------------------------------------------------------------
def test_ci_yml_exists() -> None:
    """The CI workflow file must exist at the canonical path."""
    assert CI_YML.exists(), f"missing {CI_YML}"


def test_ci_yml_parses_as_yaml(ci_workflow: Dict[str, Any]) -> None:
    """``ci.yml`` is well-formed YAML with a top-level ``jobs`` key."""
    assert "jobs" in ci_workflow, "ci.yml must declare a top-level jobs: key"
    assert isinstance(ci_workflow["jobs"], dict), (
        "ci.yml 'jobs' must be a mapping"
    )


# ---------------------------------------------------------------------------
# 2. Job shape — the Wf-35 Phase 4 master-verify contract
# ---------------------------------------------------------------------------
def test_ci_yml_has_r18_master_verify_job(r18_job: Dict[str, Any]) -> None:
    """The Wf-35 Phase 4 master-verify job is wired in.

    The job name is the contract surface: Phase 4 follow-ups that
    rename or remove it must update this test (and the Wf-35 plan).
    """
    # The fixture already asserts presence; this test makes the
    # contract explicit in the test report.
    assert r18_job.get("name", "").startswith("[4b]"), (
        f"expected name to start with '[4b]', got {r18_job.get('name')!r}"
    )


def test_ci_yml_r18_job_uses_ubuntu_latest_cpu(
    r18_job: Dict[str, Any],
) -> None:
    """The R18 master-verify job is CPU-only on ``ubuntu-latest``.

    GitHub-hosted ``ubuntu-latest`` is the documented target — this
    keeps the Wf-35 refactor gate free of self-hosted-runner
    dependencies.
    """
    assert r18_job["runs-on"] == "ubuntu-latest", (
        f"expected 'ubuntu-latest', got {r18_job['runs-on']!r}"
    )


def test_ci_yml_r18_job_has_fifteen_minute_timeout(
    r18_job: Dict[str, Any],
) -> None:
    """The R18 master-verify job is capped at 15 minutes.

    15 min covers torch+rdkit install + 6 ship-scripts' argparse
    checks on a fresh runner.  Drift above this likely indicates
    an unintended smoke test slipped in.
    """
    assert r18_job.get("timeout-minutes") == 15, (
        f"expected timeout-minutes=15, got {r18_job.get('timeout-minutes')!r}"
    )


# ---------------------------------------------------------------------------
# 3. Step wiring — install + invoke the Phase 3 script
# ---------------------------------------------------------------------------
def test_ci_yml_r18_job_invokes_master_verify_script(
    r18_job: Dict[str, Any],
) -> None:
    """The R18 master-verify job calls master_verify_r18.py.

    The Phase 3 script (``molmetal/scripts/master_verify_r18.py``) is
    the input to this Phase 4 integration.  Drift in the path must
    be flagged here so the Wf-35 plan + Phase 3 docstring stay in
    sync.
    """
    joined = "\n".join(
        str(s.get("run", "")) for s in r18_job["steps"] if isinstance(s, dict)
    )
    assert "master_verify_r18.py" in joined, (
        "expected 'master_verify_r18.py' in run steps; "
        f"runs were:\n{joined}"
    )


def test_ci_yml_r18_job_uses_uv_setup_action(
    r18_job: Dict[str, Any],
) -> None:
    """The R18 master-verify job uses ``astral-sh/setup-uv``.

    The Wf-35 install path is uv-based; switching to a different
    package manager without re-running the Phase 2 contract tests
    is a regression.
    """
    uses = [s.get("uses", "") for s in r18_job["steps"]]
    assert any("setup-uv" in u for u in uses), (
        f"expected astral-sh/setup-uv in steps; got uses={uses}"
    )


def test_master_verify_script_exists() -> None:
    """The Phase 3 master-verify script (input to Phase 4) is on disk."""
    assert MASTER_VERIFY_SCRIPT.exists(), (
        f"missing {MASTER_VERIFY_SCRIPT}"
    )


# ---------------------------------------------------------------------------
# 4. ci-summary dependency wiring
# ---------------------------------------------------------------------------
def test_ci_summary_includes_r18_master_verify_job(
    ci_workflow: Dict[str, Any],
) -> None:
    """``ci-summary`` lists the new job in ``needs:`` so the badge
    reflects Phase 4 regression signal.

    Without this hookup the master-verify job would still gate the
    workflow (GitHub Actions fails the run on any job failure), but
    the human-readable summary would silently drop the result.
    """
    summary = ci_workflow["jobs"]["ci-summary"]
    needs = summary.get("needs", [])
    assert JOB_NAME in needs, (
        f"expected {JOB_NAME!r} in ci-summary.needs; got {needs}"
    )