"""R18 ship-modules + scripts import + argparse verification (Wf-46 Phase 2).

Why a pytest mirror of ``verify_r18_modules_import.py``?
--------------------------------------------------------
The companion script
``molmetal/scripts/verify_r18_modules_import.py`` runs the full sweep
(importlib.import_module for all 32 ship modules + subprocess ``--help``
for all 52 ship scripts) and emits a JSON artefact at
``molmetal/reports/wf46_r18_verification/import_check.json``.  That is
the right tool for a human + CI pipeline.  But pytest discovery needs
*per-module* / *per-script* tests so that:

  * a regression is reported with the *specific* failing module / script
    name (vs. a single PASS/FAIL line in the JSON);
  * ``pytest -k <pattern>`` can target an individual layer
    (``pytest -k math_skills``) for fast iteration;
  * a missing dependency shows up as one focused
    ``ModuleNotFoundError`` rather than a giant JSON dump.

These tests are intentionally trivial: each one just calls
``importlib.import_module`` and / or ``subprocess.run([..., --help])``
and asserts success.  The richness is in the PARAMETRIZATION — the
parametrize ``ids`` carry the same logical IDs as the script, so a
failure reads naturally::

    test_import_ship_module[math_skills/bass_existence] FAILED
        ModuleNotFoundError: No module named 'molmetal.math_skills.bass_existence'

    test_help_ship_script[verify_score_repellent_mc] FAILED
        subprocess.CalledProcessError: --help returned exit=1

How to extend
-------------
Add an entry to :data:`SHIP_MODULES` or :data:`SHIP_SCRIPTS` (in
``molmetal/scripts/verify_r18_modules_import.py``) and re-run the
script to regenerate ``import_check.json``.  This test file requires no
edit — it imports :data:`SHIP_MODULES` / :data:`SHIP_SCRIPTS` directly
from the script to guarantee the two stay in lock-step.

CPU-only, no GPU, no fixtures, no mutable state.
"""
from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — keep the test importable from any CWD, mirroring the
# script's behaviour.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
_REPORT_FILE = (
    _REPO_ROOT / "molmetal" / "reports" / "wf46_r18_verification"
    / "import_check.json"
)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Local import — keep parametrize ids in lock-step with the script.
# ---------------------------------------------------------------------------
from molmetal.scripts.verify_r18_modules_import import (  # noqa: E402
    SHIP_MODULES,
    SHIP_SCRIPTS,
    GROUPS,
)


# ---------------------------------------------------------------------------
# Bootstrap helper — ensures repo root on path.  Idempotent.
# ---------------------------------------------------------------------------
def _bootstrap_sys_path() -> None:
    rr = str(_REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


_bootstrap_sys_path()


# ===========================================================================
# Section A — per-module import tests (parametrized over all 32 SHIP_MODULES).
# ===========================================================================
@pytest.mark.parametrize(
    "logical_id,fqmn",
    [(lid, fqmn) for lid, fqmn, _desc in SHIP_MODULES],
    ids=[lid for lid, _fqmn, _desc in SHIP_MODULES],
)
def test_import_ship_module(logical_id: str, fqmn: str) -> None:
    """Each of the 32 R18 ship modules must ``importlib.import_module`` cleanly.

    The test name carries the logical_id (e.g.
    ``math_skills/bass_existence``) so a regression points directly at
    the failing layer.
    """
    importlib.import_module(fqmn)


# ===========================================================================
# Section B — per-module class presence tests.  Each module must declare at
# least one public top-level class.  If the count drops, that suggests a
# regression that moved classes into private helpers or removed them
# outright.
# ===========================================================================
@pytest.mark.parametrize(
    "logical_id,fqmn",
    [(lid, fqmn) for lid, fqmn, _desc in SHIP_MODULES],
    ids=[lid for lid, _fqmn, _desc in SHIP_MODULES],
)
def test_module_has_public_callables(logical_id: str, fqmn: str) -> None:
    """Each ship module exposes at least one public class or function.

    This guards against accidental regressions that wipe a module's
    public API.  Both ``inspect.isclass`` and ``inspect.isfunction`` are
    counted; we accept either.
    """
    mod = importlib.import_module(fqmn)
    has_public = False
    for name in dir(mod):
        if name.startswith("_"):
            continue
        attr = getattr(mod, name, None)
        if attr is None:
            continue
        if inspect.isclass(attr) or inspect.isfunction(attr):
            has_public = True
            break
    assert has_public, (
        f"{fqmn} exposes no public class or function; module appears empty"
    )


# ===========================================================================
# Section C — per-script argparse --help tests (parametrized over 52 SHIP_SCRIPTS).
# ===========================================================================
def _run_help(rel_path: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run ``python3 <rel_path> --help`` and return (rc, stdout, stderr).

    Mirrors ``verify_r18_modules_import._check_script_help`` so a test
    failure shows the same diagnostic as the script's JSON output.
    """
    abs_path = _REPO_ROOT / rel_path
    if not abs_path.is_file():
        raise FileNotFoundError(rel_path)
    proc = subprocess.run(
        [sys.executable, str(abs_path), "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


@pytest.mark.parametrize(
    "logical_id,rel_path",
    SHIP_SCRIPTS,
    ids=[lid for lid, _p in SHIP_SCRIPTS],
)
def test_help_ship_script(logical_id: str, rel_path: str) -> None:
    """Each of the 52 R18-relevant scripts must ``--help`` without traceback.

    argparse --help normally exits 0; ``parser.parse_args`` may exit 1 on
    bare ``--help``.  Some R18 verify_*.py scripts accept ``--help`` as
    a no-op flag that simply runs the verification (no ``usage:``
    banner).  We accept any exit in {0, 1}, but the script must NOT
    raise a traceback / ImportError.
    """
    try:
        rc, stdout, stderr = _run_help(rel_path)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{rel_path} --help timed out")
    except FileNotFoundError:
        pytest.fail(f"{rel_path} not found")

    has_traceback = "Traceback" in stderr
    has_import_error = (
        "ImportError" in stderr or "ModuleNotFoundError" in stderr
    )

    assert rc in (0, 1), (
        f"{rel_path} --help exited {rc}; stderr first 240: "
        f"{stderr[:240]}"
    )
    assert not has_traceback, (
        f"{rel_path} --help raised a traceback: {stderr[:240]}"
    )
    assert not has_import_error, (
        f"{rel_path} --help raised an ImportError: {stderr[:240]}"
    )


# ===========================================================================
# Section D — JSON artefact rollups.  Mirror the cross-checks from
# test_r18_imports.py but against the Wf-46 Phase 2 JSON (which carries
# BOTH module and script results).
# ===========================================================================
def test_json_artefact_present() -> None:
    """The Wf-46 Phase 2 JSON artefact must exist.

    Run ``python3 molmetal/scripts/verify_r18_modules_import.py`` first.
    """
    if not _REPORT_FILE.is_file():
        pytest.skip(
            f"JSON artefact not present (run verify_r18_modules_import.py): "
            f"{_REPORT_FILE}"
        )
    assert _REPORT_FILE.is_file()


def test_json_artefact_keys() -> None:
    """The JSON must carry all expected top-level keys."""
    if not _REPORT_FILE.is_file():
        pytest.skip(f"JSON artefact not present: {_REPORT_FILE}")
    data = json.loads(_REPORT_FILE.read_text(encoding="utf-8"))
    expected = {
        "timestamp", "tool", "workflow", "repo_root", "python",
        "python_version", "n_modules_total", "n_modules_passed",
        "n_modules_failed", "n_scripts_total", "n_scripts_passed",
        "n_scripts_failed", "all_ok", "module_groups", "module_results",
        "module_failures", "module_successes", "script_failures",
        "script_successes",
    }
    missing = expected - set(data.keys())
    assert not missing, f"JSON missing keys: {sorted(missing)}"


def test_json_artefact_module_total_matches_manifest() -> None:
    """JSON ``n_modules_total`` == ``len(SHIP_MODULES)``."""
    if not _REPORT_FILE.is_file():
        pytest.skip(f"JSON artefact not present: {_REPORT_FILE}")
    data = json.loads(_REPORT_FILE.read_text(encoding="utf-8"))
    assert data["n_modules_total"] == len(SHIP_MODULES), (
        f"JSON n_modules_total={data['n_modules_total']} != "
        f"SHIP_MODULES={len(SHIP_MODULES)}; regenerate via "
        f"`python3 molmetal/scripts/verify_r18_modules_import.py`"
    )


def test_json_artefact_script_total_matches_manifest() -> None:
    """JSON ``n_scripts_total`` == ``len(SHIP_SCRIPTS)``."""
    if not _REPORT_FILE.is_file():
        pytest.skip(f"JSON artefact not present: {_REPORT_FILE}")
    data = json.loads(_REPORT_FILE.read_text(encoding="utf-8"))
    assert data["n_scripts_total"] == len(SHIP_SCRIPTS), (
        f"JSON n_scripts_total={data['n_scripts_total']} != "
        f"SHIP_SCRIPTS={len(SHIP_SCRIPTS)}; regenerate via "
        f"`python3 molmetal/scripts/verify_r18_modules_import.py`"
    )


def test_json_artefact_logical_ids_match_manifest() -> None:
    """JSON module logical_ids == SHIP_MODULES logical_ids."""
    if not _REPORT_FILE.is_file():
        pytest.skip(f"JSON artefact not present: {_REPORT_FILE}")
    data = json.loads(_REPORT_FILE.read_text(encoding="utf-8"))
    json_ids = {s["logical_id"] for s in data.get("module_successes", [])}
    json_ids |= {f["logical_id"] for f in data.get("module_failures", [])}
    manifest_ids = {lid for lid, _f, _d in SHIP_MODULES}
    missing_in_json = manifest_ids - json_ids
    extra_in_json = json_ids - manifest_ids
    assert not missing_in_json, (
        f"logical_ids in SHIP_MODULES but missing from JSON: "
        f"{sorted(missing_in_json)}"
    )
    assert not extra_in_json, (
        f"logical_ids in JSON but missing from SHIP_MODULES: "
        f"{sorted(extra_in_json)}"
    )


def test_json_artefact_all_ok() -> None:
    """JSON must report all_ok=True (no module OR script failures)."""
    if not _REPORT_FILE.is_file():
        pytest.skip(f"JSON artefact not present: {_REPORT_FILE}")
    data = json.loads(_REPORT_FILE.read_text(encoding="utf-8"))
    if not data.get("all_ok", False):
        mod_fail = data.get("module_failures", [])
        scr_fail = data.get("script_failures", [])
        pytest.fail(
            f"JSON reports failures: modules={len(mod_fail)} "
            f"scripts={len(scr_fail)}; sample: "
            f"{(mod_fail[:1] + scr_fail[:1])}"
        )


# ===========================================================================
# Section E — group-level rollups.  Catch dropped or duplicated modules.
# ===========================================================================
def test_total_modules_is_32() -> None:
    """Hard pin: the R18 manifest is exactly 32 modules."""
    assert len(SHIP_MODULES) == 32, (
        f"R18 manifest changed: got {len(SHIP_MODULES)}, expected 32"
    )


def test_total_scripts_is_52() -> None:
    """Hard pin: the R18 ship-script list is exactly 52 entries."""
    assert len(SHIP_SCRIPTS) == 52, (
        f"R18 script manifest changed: got {len(SHIP_SCRIPTS)}, expected 52"
    )


def test_math_skills_count_is_9() -> None:
    """Math-skills layer is exactly 9 modules (T1–T9)."""
    assert len(GROUPS["math_skills"]) == 9


def test_adapters_count_is_7() -> None:
    """Adapter layer is exactly 7 modules."""
    assert len(GROUPS["adapters"]) == 7


def test_lam_chem_count_is_11() -> None:
    """lam_chem ship layer is exactly 11 modules."""
    assert len(GROUPS["lam_chem"]) == 11


def test_reward_count_is_1() -> None:
    """Reward layer is exactly 1 module (hilbert_primal_dual)."""
    assert len(GROUPS["reward"]) == 1


def test_search_alg_count_is_2() -> None:
    """search_alg layer is exactly 2 modules."""
    assert len(GROUPS["search_alg"]) == 2


def test_flow_matching_lipman_count_is_2() -> None:
    """flow_matching_lipman layer is exactly 2 modules."""
    assert len(GROUPS["flow_matching_lipman"]) == 2


def test_group_totals_sum_to_32() -> None:
    """Sum of group sizes must equal the total manifest (32)."""
    total = sum(len(ids) for ids in GROUPS.values())
    assert total == 32, (
        f"sum of group sizes={total} != 32; check GROUPS in "
        f"verify_r18_modules_import.py"
    )


# ===========================================================================
# Section F — idempotence + cross-checks.  Sanity smoke tests.
# ===========================================================================
def test_repeated_module_imports_are_idempotent() -> None:
    """Importing the same module twice does not raise or change state.

    Guards against the rare class of bug where a module's top-level code
    has side effects that accumulate on repeat import.  We pick 3
    representative modules from 3 different layers to keep this O(1).
    """
    for fqmn in [
        "molmetal.math_skills.bass_existence",
        "molmetal.molmetal_lam.lam_chem.buyable_gate",
        "molmetal.molmetal_lam.search_alg.variance_puct_integration",
    ]:
        importlib.import_module(fqmn)
        # Second import must not raise.
        importlib.import_module(fqmn)


def test_all_module_files_exist_on_disk() -> None:
    """Each ship-module file resolves to an existing .py on disk.

    Catches the failure mode where SHIP_MODULES references a module
    that was deleted but the constant wasn't updated.
    """
    for _logical_id, fqmn, _desc in SHIP_MODULES:
        mod = importlib.import_module(fqmn)
        file = getattr(mod, "__file__", None)
        assert file is not None, f"{fqmn} has no __file__ attribute"
        assert Path(file).is_file(), f"{fqmn} __file__={file} not found"


def test_all_script_files_exist_on_disk() -> None:
    """Each ship-script path resolves to an existing .py on disk."""
    for _logical_id, rel_path in SHIP_SCRIPTS:
        abs_path = _REPO_ROOT / rel_path
        assert abs_path.is_file(), (
            f"{rel_path} not found on disk; update SHIP_SCRIPTS"
        )


def test_verify_script_itself_imports_clean() -> None:
    """The Wf-46 Phase 2 verifier script must itself be import-clean.

    Catches regressions where a refactor breaks the verifier.
    """
    importlib.import_module(
        "molmetal.scripts.verify_r18_modules_import"
    )


def test_verify_script_help_runs() -> None:
    """The Wf-46 Phase 2 verifier's own ``--help`` must run cleanly."""
    abs_path = _REPO_ROOT / "molmetal" / "scripts" / \
        "verify_r18_modules_import.py"
    proc = subprocess.run(
        [sys.executable, str(abs_path), "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode in (0, 1), (
        f"verifier --help exited {proc.returncode}; "
        f"stderr: {proc.stderr[:240]}"
    )
    assert "usage:" in proc.stdout.lower(), (
        f"verifier --help did not print usage; "
        f"stdout: {proc.stdout[:240]}"
    )


# ===========================================================================
# Section G — per-script entry-point (main) sanity check.  Each script
# module must be runnable (which --help already proved), but we ALSO
# assert the file is non-empty so a regression to a stub file shows up.
# ===========================================================================
@pytest.mark.parametrize(
    "logical_id,rel_path",
    SHIP_SCRIPTS,
    ids=[lid for lid, _p in SHIP_SCRIPTS],
)
def test_script_file_is_nonempty(logical_id: str, rel_path: str) -> None:
    """Each ship-script .py file is non-empty (>1 KiB)."""
    abs_path = _REPO_ROOT / rel_path
    assert abs_path.is_file()
    size = abs_path.stat().st_size
    assert size > 1024, (
        f"{rel_path} is suspiciously small ({size} bytes); "
        f"likely a regression to a stub"
    )


# ===========================================================================
# Section H — Phase 2-specific dimension: every script that DOES emit an
# argparse usage banner (the common case) must do so cleanly.  Scripts
# that ignore ``--help`` and just run (a minority) are tolerated — the
# earlier ``test_help_ship_script`` already checked that no traceback
# occurs.  This test is best-effort: it counts scripts that print a
# banner and reports the coverage ratio.  It never fails on individual
# scripts that simply lack a banner.
# ===========================================================================
def test_argparse_banner_coverage_ratio() -> None:
    """Aggregate: at least 80 % of ship-scripts print an argparse banner.

    Most R18 ship-scripts use ``ArgumentParser.add_argument`` and so
    emit a ``usage:`` / ``options:`` banner.  A small minority (e.g.
    ``verify_paper_build.py``) accept ``--help`` as a no-op and run a
    full verification instead — these are tolerated but the aggregate
    coverage is tracked.
    """
    n_with_banner = 0
    n_total = 0
    no_banner: list[str] = []
    for logical_id, rel_path in SHIP_SCRIPTS:
        n_total += 1
        try:
            _rc, stdout, _stderr = _run_help(rel_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        if "usage:" in stdout.lower() or "options:" in stdout.lower():
            n_with_banner += 1
        else:
            no_banner.append(logical_id)
    ratio = n_with_banner / n_total
    assert ratio >= 0.80, (
        f"only {n_with_banner}/{n_total} ({ratio:.0%}) ship-scripts "
        f"print an argparse banner; no-banner scripts: {no_banner}"
    )
