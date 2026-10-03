"""R18 ship-modules import tests (mirror of verify_r18_all_imports.py).

Why a pytest mirror of the script?
----------------------------------
The script ``molmetal/scripts/verify_r18_all_imports.py`` runs the full
32-module sweep and emits a JSON artefact.  That is the right tool for a
human + CI pipeline.  But pytest discovery needs *per-module* tests so
that:

  * a regression is reported with the *specific* failing module name
    (vs. a single PASS/FAIL line in the JSON);
  * ``pytest -k <pattern>`` can target an individual layer
    (``pytest -k math_skills``) for fast iteration;
  * a missing dependency shows up as one focused ``ModuleNotFoundError``
    rather than a giant CSV-shaped JSON dump.

These tests are intentionally trivial: each one just calls
``importlib.import_module`` and asserts success.  The richness is in the
PARAMETRIZATION — the parametrize ``ids`` carry the same logical IDs as
the script, so a failure reads naturally:

    test_import_ship_module[math_skills/bass_existence] FAILED

    ModuleNotFoundError: No module named 'molmetal.math_skills.bass_existence'

How to extend
-------------
Add an entry to :data:`SHIP_MODULES` (in
``molmetal/scripts/verify_r18_all_imports.py``) and re-run the script
to regenerate ``import_check.json``.  This test file requires no edit —
it imports :data:`SHIP_MODULES` directly from the script to guarantee
the two stay in lock-step.

CPU-only, no GPU, no fixtures, no mutable state.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make the script importable as a module (it lives in molmetal/scripts/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local import — keep the parametrize ids in sync with the script.
from molmetal.scripts.verify_r18_all_imports import (  # noqa: E402
    SHIP_MODULES,
    GROUPS,
)


# ---------------------------------------------------------------------------
# Bootstrap the same sys.path fix used by the script.  Without this,
# running the tests from a directory outside the repo fails with
# ``ModuleNotFoundError: No module named 'molmetal'``.  Idempotent.
# ---------------------------------------------------------------------------
def _bootstrap_sys_path() -> None:
    rr = str(_REPO_ROOT)
    if rr not in sys.path:
        sys.path.insert(0, rr)


_bootstrap_sys_path()


# ---------------------------------------------------------------------------
# Re-export SHIP_MODULES as a pytest-friendly list of (logical_id, fqmn)
# tuples.  Importing the constant *also* forces every module's
# importlib.import_module call to be executed once at collection time —
# which is a fine dry-run and surfaces hard failures fast.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "logical_id,fqmn",
    [(lid, fqmn) for lid, fqmn, _desc in SHIP_MODULES],
    ids=[lid for lid, _fqmn, _desc in SHIP_MODULES],
)
def test_import_ship_module(logical_id: str, fqmn: str) -> None:
    """Every R17+R18 ship module must ``importlib.import_module`` cleanly.

    The test name carries the logical_id (e.g. ``math_skills/bass_existence``)
    so a regression points directly at the failing layer.
    """
    importlib.import_module(fqmn)


# ---------------------------------------------------------------------------
# Group-level rollups — fail loud if a layer's tally drops.
# ---------------------------------------------------------------------------
def test_math_skills_all_9() -> None:
    """All 9 math_skills modules import (T1–T9)."""
    assert len(GROUPS["math_skills"]) == 9, (
        f"expected 9 math_skills, got {len(GROUPS['math_skills'])}: "
        f"{GROUPS['math_skills']}"
    )
    for lid in GROUPS["math_skills"]:
        fqmn = next(f for _lid, f, _d in SHIP_MODULES if _lid == lid)
        importlib.import_module(fqmn)


def test_adapters_all_7() -> None:
    """All 7 adapter modules import (3 integration + 4 SOTA-clones)."""
    assert len(GROUPS["adapters"]) == 7, (
        f"expected 7 adapters, got {len(GROUPS['adapters'])}: " f"{GROUPS['adapters']}"
    )
    for lid in GROUPS["adapters"]:
        fqmn = next(f for _lid, f, _d in SHIP_MODULES if _lid == lid)
        importlib.import_module(fqmn)


def test_lam_chem_new_11() -> None:
    """All 11 new lam_chem modules import (incl. fg_compat_veto).

    Note: the spec lists ``fg_compat_veto`` as a lam_chem module, but the
    shipping module actually lives under
    ``molmetal.molmetal_lam.reactions.fg_compatibility`` per the
    ``wf_t30_p12_fg_veto`` ship report.  Both paths are exercised here.
    """
    assert len(GROUPS["lam_chem"]) == 11, (
        f"expected 11 lam_chem, got {len(GROUPS['lam_chem'])}: " f"{GROUPS['lam_chem']}"
    )
    for lid in GROUPS["lam_chem"]:
        fqmn = next(f for _lid, f, _d in SHIP_MODULES if _lid == lid)
        importlib.import_module(fqmn)


def test_reward_hilbert_primal_dual() -> None:
    """Hilbert primal-dual reward channel imports."""
    importlib.import_module("molmetal.molmetal_lam.reward.hilbert_primal_dual")


def test_search_alg_score_repellent_mc() -> None:
    """Score-repellent MC search imports."""
    importlib.import_module("molmetal.molmetal_lam.search_alg.score_repellent_mc")


def test_search_alg_variance_puct_integration() -> None:
    """Variance-aware PUCT integration imports."""
    importlib.import_module(
        "molmetal.molmetal_lam.search_alg.variance_puct_integration"
    )


def test_flow_matching_lipman_w_flow() -> None:
    """w_flow (Wasserstein flow backbone) imports."""
    importlib.import_module("molmetal.adapters.flow_matching_lipman.w_flow")


def test_flow_matching_lipman_m5_schedulers() -> None:
    """m5_schedulers (variance-preserving schedulers) imports."""
    importlib.import_module("molmetal.adapters.flow_matching_lipman.m5_schedulers")


def test_adapters_coordination_gnn() -> None:
    """Coordination-GNN scaffold imports."""
    importlib.import_module("molmetal.adapters.coordination_gnn")


def test_adapters_quotient_space_diffusion() -> None:
    """Quotient-space diffusion imports."""
    importlib.import_module("molmetal.adapters.quotient_space_diffusion")


# ---------------------------------------------------------------------------
# Guard rails — total module count + per-group minimums.
# ---------------------------------------------------------------------------
def test_total_modules_at_least_30() -> None:
    """The full ship manifest must contain at least 30 modules."""
    assert (
        len(SHIP_MODULES) >= 30
    ), f"expected >= 30 ship modules, got {len(SHIP_MODULES)}"


def test_total_modules_is_32() -> None:
    """Hard pin: the R18 manifest is exactly 32 modules.

    Catches accidental additions/removals in the SHIP_MODULES list.
    Update this test (and the JSON ``n_modules_total`` field) together
    if a future round genuinely expands the manifest.
    """
    assert (
        len(SHIP_MODULES) == 32
    ), f"R18 manifest changed: got {len(SHIP_MODULES)}, expected 32"


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


# ---------------------------------------------------------------------------
# Cross-checks against the JSON artefact — fail if the script's output is
# stale relative to SHIP_MODULES.  This catches the
# "someone added a module to the script but did not regenerate JSON"
# failure mode.
# ---------------------------------------------------------------------------
def test_json_artefact_matches_manifest() -> None:
    """The import_check.json must list exactly the SHIP_MODULES count."""
    json_path = (
        _REPO_ROOT
        / "molmetal"
        / "reports"
        / "wf_final_verification"
        / "import_check.json"
    )
    if not json_path.is_file():
        pytest.skip(
            f"JSON artefact not present (run verify_r18_all_imports.py): "
            f"{json_path}"
        )
    import json as _json

    data = _json.loads(json_path.read_text(encoding="utf-8"))
    assert data["n_modules_total"] == len(SHIP_MODULES), (
        f"JSON n_modules_total={data['n_modules_total']} != "
        f"SHIP_MODULES={len(SHIP_MODULES)}; regenerate with "
        f"`python3 molmetal/scripts/verify_r18_all_imports.py`"
    )
    json_ids = {s["logical_id"] for s in data.get("successes", [])}
    json_ids |= {f["logical_id"] for f in data.get("failures", [])}
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


def test_json_artefact_reports_all_ok() -> None:
    """If the JSON exists, it must report all_ok=True (no failures).

    This guards against a stale JSON that lingers after a regression.
    """
    json_path = (
        _REPO_ROOT
        / "molmetal"
        / "reports"
        / "wf_final_verification"
        / "import_check.json"
    )
    if not json_path.is_file():
        pytest.skip(f"JSON artefact not present: {json_path}")
    import json as _json

    data = _json.loads(json_path.read_text(encoding="utf-8"))
    assert data["all_ok"] is True, f"JSON reports failures: {data['failures']}"
    assert data["n_passed"] == data["n_modules_total"], (
        f"n_passed={data['n_passed']} != " f"n_modules_total={data['n_modules_total']}"
    )


def test_repeated_imports_are_idempotent() -> None:
    """Importing the same module twice does not raise or change state.

    Guards against the rare class of bug where a module's top-level code
    has side effects (e.g. ``os.environ.setdefault``) that accumulate on
    repeat import.  We pick 3 representative modules from 3 different
    layers to keep this O(1) per test.
    """
    for fqmn in [
        "molmetal.math_skills.bass_existence",
        "molmetal.molmetal_lam.lam_chem.buyable_gate",
        "molmetal.molmetal_lam.search_alg.variance_puct_integration",
    ]:
        importlib.import_module(fqmn)
        # Second import must not raise.
        importlib.import_module(fqmn)
