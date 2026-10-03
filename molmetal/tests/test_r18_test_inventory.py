"""test_r18_test_inventory.py — meta-tests for the R17+R18 test inventory.

Why a meta-test file?
---------------------
The :mod:`molmetal.scripts.run_all_r18_tests` script is the canonical
entry point that runs ``pytest --collect-only`` across every R18
ship-wave test file and writes a single JSON roll-up at
``molmetal/reports/wf_pytest_runner/results.json``.  The script is
read-only and AST-light, but the JSON it produces is the source of
truth for "how many tests ship with R18?".

These meta-tests pin that JSON to the spec's lower bounds:

  * ``n_test_files``   ≥ 30  (the brief's "~30 test files" floor)
  * ``n_tests``        ≥ 800 (the brief's "~900 tests" floor)
  * ``n_duplicate_nodeids``  == 0
  * ``per_category_count`` contains each of:
        math_skills, integration, adapter, pitfall, ml_methods
    plus the implicit ``r18_orchestrator`` bucket (the script adds
    the r18_* files even though they are not a spec category).
  * collection errors stay below a sane upper bound so a future
    environment regression (e.g. ``torch`` removal) fails this test
    before it breaks ``pytest -m r18`` silently.

The tests are intentionally trivial: each one reads the JSON and
asserts a single invariant.  CPU-only, no GPU, no fixtures, no
mutable state.

How to extend
-------------
If a new R18 category is added (e.g. ``quantum_methods``), extend
the :data:`EXPECTED_CATEGORIES` set in lock-step with
``run_all_r18_tests.R18_CATEGORY_STEMS``.  Bumping the lower bounds
must come with an updated spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap — match the convention used by every other
# tests/test_*.py file under molmetal/tests/ so this module is
# importable from any CWD.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants — the spec floors + the canonical category roster.
# ---------------------------------------------------------------------------
JSON_PATH = _REPO_ROOT / "molmetal" / "reports" / "wf_pytest_runner" / "results.json"

# The brief states "~30 test files" and "~900 tests" — pin to
# slightly-loose lower bounds so a future expansion is not a regression.
MIN_TEST_FILES = 30
MIN_TESTS = 800

# Spec-aligned categories the brief enumerates by name.  The script
# also emits an ``r18_orchestrator`` bucket for ``test_r18_*.py``
# files; that is a runner-internal bucket and is *not* part of the
# brief's category list.
EXPECTED_CATEGORIES: frozenset[str] = frozenset(
    {
        "math_skills",
        "integration",
        "adapter",
        "pitfall",
        "ml_methods",
    }
)

# Tolerated ceiling on collection errors.  Some R18-era test files
# are intentionally torch-gated (and so ERROR during collection on a
# CPU-only host) — the runner records them, does not crash on them,
# and the meta-tests treat the residual count as a soft signal rather
# than a hard failure.  Bumping this constant should come with a
# changelog entry explaining *which* file started erroring.
MAX_COLLECTION_ERRORS = 60


# ---------------------------------------------------------------------------
# JSON loader — single source of truth for the meta-tests.
# ---------------------------------------------------------------------------
def _load_report() -> dict:
    """Return the parsed results.json, or skip if it has not been written yet.

    A missing JSON artefact means the operator has not yet run
    :file:`molmetal/scripts/run_all_r18_tests.py`; rather than fail
    with a confusing ``FileNotFoundError`` mid-suite, the meta-tests
    skip and emit a single, identifiable message.

    Once the JSON exists, this function is the only place we parse
    JSON, so a malformed artefact fails once with a useful traceback
    rather than 15 times across the suite.
    """
    if not JSON_PATH.is_file():
        pytest.skip(
            f"results.json missing — run `uv run --no-sync python "
            f"molmetal/scripts/run_all_r18_tests.py` to generate "
            f"{JSON_PATH.relative_to(_REPO_ROOT)}"
        )
    try:
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # noqa: BLE001
        pytest.fail(
            f"results.json is not valid JSON: {exc}.  Re-run the "
            f"runner script to regenerate it."
        )


# ---------------------------------------------------------------------------
# Headline floors — the brief's two pin-numbers.
# ---------------------------------------------------------------------------
def test_total_test_files_at_least_30(report: dict) -> None:
    """The R18 inventory must list at least 30 test files (brief floor)."""
    n = int(report.get("n_test_files", 0))
    assert n >= MIN_TEST_FILES, (
        f"n_test_files={n} < {MIN_TEST_FILES}.  R18 ships 30+ test "
        f"files per the brief; investigate missing modules."
    )


def test_total_tests_at_least_800(report: dict) -> None:
    """The R18 inventory must collect at least 800 tests (brief floor)."""
    n = int(report.get("n_tests", 0))
    assert n >= MIN_TESTS, (
        f"n_tests={n} < {MIN_TESTS}.  R18 ships ~900 tests per the "
        f"brief; investigate parametrisation regressions."
    )


# ---------------------------------------------------------------------------
# Integrity — no silent failures inside the inventory.
# ---------------------------------------------------------------------------
def test_no_duplicate_test_names(report: dict) -> None:
    """Two collected node-ids must not collide on the same id.

    Duplicates usually come from copy-pasted test names across
    parametrize blocks; pytest silently runs both, so this test
    catches accidental regressions.
    """
    n = int(report.get("n_duplicate_nodeids", 0))
    assert n == 0, (
        f"results.json reports {n} duplicate node-ids; "
        f"first few: {report.get('duplicate_nodeids_sample', [])[:5]}"
    )


def test_results_json_has_required_top_level_fields(report: dict) -> None:
    """The JSON must carry every key the rest of the suite reads.

    Bumping the runner's output schema is a breaking change for the
    dashboard; this test fails loudly so the breakage is not silent.
    """
    required = {
        "timestamp",
        "tool",
        "mode",
        "repo_root",
        "python_version",
        "pytest_version",
        "n_test_files",
        "n_tests",
        "per_file_count",
        "per_category_count",
        "per_category_files",
        "collection_errors",
        "files",
    }
    missing = sorted(required - set(report.keys()))
    assert not missing, f"results.json missing keys: {missing}"


# ---------------------------------------------------------------------------
# Category presence — the brief enumerates five categories by name.
# ---------------------------------------------------------------------------
def test_r18_categories_present(report: dict) -> None:
    """All five spec-aligned categories must contribute ≥ 1 test.

    The brief lists these explicitly: ``math_skills``, ``integration``,
    ``adapter``, ``pitfall``, ``ml_methods``.  The runner emits an
    additional ``r18_orchestrator`` bucket for ``test_r18_*.py``; that
    one is implicit and is checked separately below.
    """
    cats = report.get("per_category_count", {})
    missing = sorted(c for c in EXPECTED_CATEGORIES if cats.get(c, 0) <= 0)
    assert not missing, (
        f"categories present but zero-test: {missing}.  Per-category " f"counts: {cats}"
    )


@pytest.mark.parametrize("category", sorted(EXPECTED_CATEGORIES))
def test_each_expected_category_has_minimum_tests(report: dict, category: str) -> None:
    """Each spec-aligned category must contribute ≥ 5 tests.

    Parametrised so a regression on a single category fails with a
    category-named test id rather than a generic "list is empty" line.
    """
    n = int(report.get("per_category_count", {}).get(category, 0))
    assert n >= 5, (
        f"category {category!r} has only {n} tests (floor 5).  "
        f"Did a parametrize block regress?"
    )


def test_r18_orchestrator_bucket_present(report: dict) -> None:
    """The runner's implicit ``r18_orchestrator`` bucket is non-empty.

    The orchestrator / cv-harness / imports test files are always
    included by the runner's R18-prefix wildcard; if they vanish, the
    bucket shrinks below the lower bound.
    """
    n = int(report.get("per_category_count", {}).get("r18_orchestrator", 0))
    assert n >= 3, (
        f"r18_orchestrator bucket has only {n} tests (floor 3).  "
        f"Did test_r18_imports / cv_harness / pipeline_orchestrator "
        f"disappear?"
    )


# ---------------------------------------------------------------------------
# Collection-error guard rail — soft signal, not a hard fail.
# ---------------------------------------------------------------------------
def test_collection_errors_below_ceiling(report: dict) -> None:
    """At most MAX_COLLECTION_ERRORS files may fail to import.

    Some R18 files are torch-gated by design; collection errors on
    those files are expected on a CPU host.  This test only fails if
    the error count spikes (e.g. a missing dependency or a syntax
    regression in a previously-clean file).
    """
    n = int(report.get("n_collection_errors", 0))
    assert n <= MAX_COLLECTION_ERRORS, (
        f"n_collection_errors={n} > {MAX_COLLECTION_ERRORS}; "
        f"first errors: {report.get('collection_errors', [])[:5]}"
    )


def test_collect_only_mode_recorded(report: dict) -> None:
    """The runner must be honest about not actually executing tests.

    ``pytest --collect-only`` does not run tests; the JSON must say so
    so a downstream consumer doesn't confuse this for a real run.
    """
    mode = str(report.get("mode", "")).lower()
    assert "collect" in mode and "no test execution" in mode, (
        f"mode={mode!r} does not announce collect-only; the JSON may "
        f"have been generated by a future runner that actually "
        f"executes tests."
    )


# ---------------------------------------------------------------------------
# Per-file sanity — every discovered file shows up in per_file_count.
# ---------------------------------------------------------------------------
def test_per_file_count_matches_files_list(report: dict) -> None:
    """per_file_count and the files[].n_tests arrays must agree.

    Both views are computed independently; if they drift the JSON is
    unsound and any downstream dashboard would silently disagree with
    itself.
    """
    pfc = report.get("per_file_count", {})
    files_list = report.get("files", [])
    pfc_keys = set(pfc.keys())
    files_keys = {f["path"] for f in files_list}
    assert pfc_keys == files_keys, (
        f"per_file_count and files[].path disagree: "
        f"only-in-pfc={sorted(pfc_keys - files_keys)}, "
        f"only-in-files={sorted(files_keys - pfc_keys)}"
    )
    for entry in files_list:
        path = entry["path"]
        assert int(pfc.get(path, -1)) == int(entry["n_tests"]), (
            f"per_file_count[{path}]={pfc.get(path)} != "
            f"files[].n_tests={entry['n_tests']}"
        )


# ---------------------------------------------------------------------------
# Fixtures — share the parsed JSON across the suite so each test
# does not re-read the file.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report() -> dict:
    """Parse results.json once per module and share it across tests."""
    return _load_report()


# ---------------------------------------------------------------------------
# Version pinning — the runner itself must exist and be runnable.
# ---------------------------------------------------------------------------
def test_runner_script_exists() -> None:
    """The runner script must live at the canonical path.

    A missing runner is a regression — without it the JSON would be
    stale and the meta-tests would not be exercising anything.
    """
    script = _REPO_ROOT / "molmetal" / "scripts" / "run_all_r18_tests.py"
    assert script.is_file(), (
        f"runner script missing: {script.relative_to(_REPO_ROOT)}.  "
        f"Restore it or update the path in this test."
    )


def test_runner_script_has_categorisation_table() -> None:
    """The runner must expose the canonical ``R18_CATEGORY_STEMS`` table.

    Importing the table lets the test mirror the runner's view and
    fail loudly if the table is renamed or removed.
    """
    from molmetal.scripts import run_all_r18_tests as runner

    assert hasattr(runner, "R18_CATEGORY_STEMS"), (
        "runner module lost R18_CATEGORY_STEMS — meta-tests cannot "
        "verify category rollups."
    )
    assert isinstance(
        runner.R18_CATEGORY_STEMS, list
    ), "R18_CATEGORY_STEMS must be a list (table of tuples)."
    assert len(runner.R18_CATEGORY_STEMS) >= 30, (
        f"R18_CATEGORY_STEMS has only {len(runner.R18_CATEGORY_STEMS)} "
        f"entries; the R18 manifest expects 30+ substrings."
    )


# ---------------------------------------------------------------------------
# Symmetry — the runner must list exactly the same categories that
# the meta-test enumerates.  Catches a future drift between the two.
# ---------------------------------------------------------------------------
def test_runner_classify_stem_recognises_each_category() -> None:
    """``classify_stem`` must return a category for every spec category.

    Picked a representative stem for each spec category and asserts
    ``classify_stem`` returns that category.  Symmetry test, not a
    count test.
    """
    from molmetal.scripts import run_all_r18_tests as runner

    samples = {
        "math_skills": "bass_existence",
        "integration": "hilbert_astsr_integration",
        "adapter": "vina_adapter",
        "pitfall": "pitfall_guards",
        "ml_methods": "dmpnn_baseline",
    }
    for expected_cat, stem in samples.items():
        cat, _desc = runner.classify_stem(stem)
        assert cat == expected_cat, (
            f"classify_stem({stem!r}) returned {cat!r}, expected " f"{expected_cat!r}."
        )


# ---------------------------------------------------------------------------
# Total — sanity that we have at least 15 meta-tests in this file.
# ---------------------------------------------------------------------------
def test_at_least_15_inventory_tests() -> None:
    """This file must contain at least 15 test functions.

    The brief requests "15 tests" — pin it as a guard rail so a
    future refactor that drops tests fails CI rather than passing
    silently.
    """
    import inspect

    tests = [
        name
        for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    ]
    assert len(tests) >= 15, (
        f"only {len(tests)} meta-tests defined; brief requests ≥ 15.  "
        f"Current set: {sorted(tests)}"
    )


# ---------------------------------------------------------------------------
# Additional sanity — counts across the suite must be self-consistent.
# ---------------------------------------------------------------------------
def test_total_tests_matches_per_category_sum(report: dict) -> None:
    """Sum of per-category test counts must equal the top-level n_tests.

    A drift between the two views means the JSON was computed twice
    (once for the per-file view, once for the per-category view) and
    the two diverged — a clear bug.
    """
    cats_sum = sum(int(v) for v in report.get("per_category_count", {}).values())
    n_tests = int(report.get("n_tests", 0))
    assert (
        cats_sum == n_tests
    ), f"per_category_count sums to {cats_sum} but n_tests={n_tests}."


def test_total_files_matches_per_category_files(report: dict) -> None:
    """Sum of per-category file counts must equal n_test_files.

    Same idea as :func:`test_total_tests_matches_per_category_sum`
    but for the per-file dimension; both views must reconcile.
    """
    cat_files_sum = sum(int(v) for v in report.get("per_category_files", {}).values())
    n_files = int(report.get("n_test_files", 0))
    assert cat_files_sum == n_files, (
        f"per_category_files sums to {cat_files_sum} but " f"n_test_files={n_files}."
    )
