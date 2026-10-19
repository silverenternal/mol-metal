"""Wf-345B1 — env-aware SKIP contract for the pre-flight scaffold_zero_leak gate.

================================================================
Per the 2026-09-29 user ultracode directive ("覆盖所有发现的问题把残余
的问题解决了，一定要全") + Wf-345B1: the pre-flight gate
``p0_scaffold_zero_leak`` at
``molmetal/scripts/pre_flight_check.py:561`` raised ``rc=1`` on the
host Python (per wx63q8fwe ultracode review 2026-09-29 15:51 UTC).
The verifier subprocess imports ``pandas`` at module top-level; on a
host Python without pandas the structural probe fails with
``ModuleNotFoundError: No module named 'pandas'``. The CSV itself is
intact — the FAIL is purely an env gap, not a regression.

The fix (Wf-345B1):
  * Capture ``stderr`` from the verifier subprocess (was previously
    discarded via ``_``).
  * Detect ``ModuleNotFoundError`` / ``ImportError`` for ``pandas`` in
    stderr → emit SKIP (not FAIL) with the missing dep named + the
    expected pandas-enabled runtime per AGENTS.md §5.1 standing GPU
    grant.
  * Keep the original FAIL contract for genuine data-integrity
    violations (CSV missing / >0 leaks / unexpected subprocess errors),
    but include a stderr snippet in the FAIL evidence for operator
    verifiability.

Tests (2 NEW)
-------------

1. **test_gate_scaffold_zero_leak_pass_on_valid_csv** — PASS branch
   regression test. Writes a tmp_path CSV with disjoint
   train/val/test scaffold sets (Bemis-Murcko style), invokes the
   gate directly, asserts ``status == "PASS"`` and
   ``"0 scaffold leak"`` in evidence.

2. **test_gate_scaffold_zero_leak_skip_when_pandas_missing** — SKIP
   branch contract test. Monkeypatches the gate's ``_run`` helper to
   return a synthesised ``ModuleNotFoundError: No module named
   'pandas'`` stderr (simulating the host Python env gap), asserts
   ``status == "SKIP"`` (NOT FAIL) with ``"pandas"`` named in the
   evidence + a pointer to the expected pandas-enabled runtime per
   AGENTS.md §5.1.

Why monkeypatch (and not a real subprocess with hidden pandas):
  The host Python on this dev box has pandas installed (via the
  project venv). Trying to launch a real subprocess that hides
  pandas requires either ``sys.path`` mutation in the subprocess
  (brittle + conda/venv-specific) or a sidecar interpreter without
  pandas (none available on this host). The monkeypatch approach
  synthesises the exact stderr shape that ``import pandas`` raises
  on a no-pandas host, so the gate's SKIP branch can be exercised
  deterministically without any env gymnastics.

Backward-compat verdict
-----------------------
The fix is a strict superset of the original Wf-261 contract:
  * On a valid CSV + pandas-enabled runtime → the gate emits the
    same PASS / FAIL / evidence as before (``0 scaffold leak`` on
    PASS; leak counts on FAIL).
  * On a missing-pandas host → the gate now emits SKIP instead of
    FAIL with a misleading "rc=1" message. The SKIP evidence names
    ``pandas`` as the missing dep + the expected pandas-enabled
    runtime so the operator can resolve the env gap.
  * On a genuine data-integrity violation (CSV missing / >0 leaks)
    → the gate still FAILs, but now includes a stderr snippet in
    the evidence (was previously lost).

No existing PASS / FAIL behavior is altered for the canonical
happy-path run; the only change is FAIL → SKIP on the no-pandas
host, which is the honest-negative framing per AGENTS.md §5.2 +
party-twenty-spirit §3.5.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (PASS on a valid CSV,
  SKIP on a simulated env gap). Tests do NOT fabricate any metric
  values — every assertion traces to the on-disk stub state or the
  monkeypatched return value.
* The SKIP-vs-FAIL contract is documented in the test docstrings +
  the gate docstring; the on-disk evidence string names the missing
  dep + the expected runtime so the operator can resolve the env
  gap without re-reading the code.
* This gate has NEVER been able to PASS on the no-pandas host (env
  gap is MEASURED at the wx63q8fwe review timestamp). The
  env-aware SKIP contract is the honest-negative close for that gap.

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py:561`` —
  ``gate_scaffold_zero_leak`` (env-aware + stderr-capturing).
* ``molmetal/scripts/pre_flight_check.py:5151-5155`` — gate
  registration in ``build_gates("all")`` (unchanged).
* ``TODO/completed/261_v2_p0_scaffold_split.md`` — Wf-261 ship
  surface (the original v2 scaffold-split that this gate verifies).
* ``TODO/completed/345b1_scaffold_zero_leak_fix.md`` — this
  Wf-345B1 ship surface.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Tuple

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly.

    NOTE: the project conftest pre-loads real torch at pytest
    session start (cf. ``molmetal/tests/conftest.py``). This function
    expects real torch to already be in ``sys.modules`` when it is
    called (otherwise ``molmetal/__init__.py``'s
    ``_apply_rocm_gfx1101_env`` will raise on
    ``torch.cuda.is_available()``).
    """
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_under_test_wf345b1",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_under_test_wf345b1"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-scope fixture: pre_flight_check loaded once with the host env.

    Uses the conftest-pre-loaded real torch (so the project's
    ``molmetal/__init__.py`` bootstrap works). The gate is invoked on
    tmp_path CSVs (NOT the canonical v4 CSV) so the PASS test is
    self-contained.
    """
    return _load_preflight_module()


# ---------------------------------------------------------------------------
# CSV stub helpers — write a tmp_path CSV that the gate's verifier can
# consume via the real subprocess path.
# ---------------------------------------------------------------------------


def _write_disjoint_scaffold_csv(path: Path) -> Path:
    """Write a tmp_path CSV with disjoint train/val/test scaffold sets.

    Columns match the canonical v4 CSV:
        smiles, metal, scaffold_id, split, activity

    Train uses scaffolds {A, B, C}, val uses {D, E}, test uses {F, G}.
    No scaffold overlap across splits → gate emits PASS.
    """
    rows = []
    splits_scaffolds = {
        "train": ["A", "B", "C"],
        "val": ["D", "E"],
        "test": ["F", "G"],
    }
    for split, scaffolds in splits_scaffolds.items():
        for sc in scaffolds:
            rows.append(f"CC{sc},Pt,{sc},{split},1.0")
    body = "smiles,metal,scaffold_id,split,activity\n" + "\n".join(rows) + "\n"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1 — PASS branch regression test (on a disjoint-scaffold CSV)
# ---------------------------------------------------------------------------


def test_gate_scaffold_zero_leak_pass_on_valid_csv(preflight, tmp_path) -> None:
    """Wf-345B1 — gate PASSes on a CSV with disjoint train/val/test scaffolds.

    Pre-condition: writes a tmp_path CSV with 3 disjoint scaffold sets
    across the train/val/test splits (the Wf-261 v2 contract).
    Subprocess imports pandas (already in the .venv runtime) and
    emits ``0 scaffold leak`` to stdout.

    Contract: the gate emits ``status == "PASS"`` with
    ``"0 scaffold leak"`` in evidence. This is the Wf-261 PASS
    contract preserved verbatim by the Wf-345B1 fix (the fix only
    adds an env-aware SKIP branch; the PASS branch is untouched).
    """
    csv_path = _write_disjoint_scaffold_csv(tmp_path / "scaffold_disjoint.csv")
    result = preflight.gate_scaffold_zero_leak(
        "wf345b1_sc_pass",
        "P0",
        "data",
        [(csv_path, "scaffold_disjoint")],
    )
    assert result.status == "PASS", (
        f"disjoint-scaffold CSV should PASS; got {result.status}: " f"{result.evidence}"
    )
    assert "0 scaffold leak" in result.evidence, (
        f"PASS evidence must contain '0 scaffold leak' for "
        f"operator verifiability; got: {result.evidence}"
    )
    assert result.wall_sec >= 0.0


# ---------------------------------------------------------------------------
# 2 — SKIP branch contract test (synthesised pandas-missing env gap)
# ---------------------------------------------------------------------------


def test_gate_scaffold_zero_leak_skip_when_pandas_missing(
    preflight,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wf-345B1 — gate SKIPs (NOT FAIL) when pandas import fails.

    Pre-condition: writes a valid disjoint-scaffold CSV (the same
    fixture as test 1, so the gate's ``csv_path.is_file()`` pre-check
    passes), then monkeypatches the gate's ``_run`` helper to return
    a synthesised ``ModuleNotFoundError: No module named 'pandas'``
    stderr (simulating the host Python env gap that the wx63q8fwe
    review observed).

    Contract: the gate emits ``status == "SKIP"`` (NOT FAIL) with
    ``"pandas"`` named as the missing dep + a pointer to the expected
    pandas-enabled runtime per AGENTS.md §5.1 (kanzi_venv or
    .venv). The Wf-345B1 explicit honest-negative contract is:
    env gap MUST emit SKIP, never FAIL — the gate cannot PASS but is
    also not regressing.

    Why monkeypatch (and not a real subprocess with hidden pandas):
    The host Python on this dev box has pandas installed. Trying to
    launch a real subprocess that hides pandas requires brittle
    ``sys.path`` mutation in the subprocess + conda/venv-specific
    gymnastics. The monkeypatch approach synthesises the exact
    stderr shape that ``import pandas`` raises on a no-pandas host,
    so the gate's SKIP branch can be exercised deterministically.
    """
    csv_path = _write_disjoint_scaffold_csv(tmp_path / "scaffold_disjoint.csv")

    # Synthesised stderr shape — matches the exact exception text
    # that ``import pandas`` raises on a no-pandas host.
    synth_stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "    import pandas as pd\n"
        "    ^^^^^^^^^^^^^^^\n"
        "ModuleNotFoundError: No module named 'pandas'\n"
    )

    def fake_run(
        cmd: Any,
        *,
        timeout: Any = None,
        cwd: Any = None,
    ) -> Tuple[int, str, str]:
        return (1, "", synth_stderr)

    # Monkeypatch the gate's _run helper. The gate references
    # ``_run`` as a module-level name, so patching
    # ``preflight._run`` replaces it for the gate's call.
    monkeypatch.setattr(preflight, "_run", fake_run)

    result = preflight.gate_scaffold_zero_leak(
        "wf345b1_sc_skip",
        "P0",
        "data",
        [(csv_path, "scaffold_disjoint")],
    )

    assert result.status == "SKIP", (
        f"synthesised pandas-missing env gap should SKIP; got "
        f"{result.status}: {result.evidence}"
    )
    assert result.status != "FAIL", (
        "Wf-345B1 explicit: env gap MUST emit SKIP, never FAIL — "
        "the gate cannot PASS but is also not regressing. Original "
        "FAIL-with-just-'rc=1' contract was the wx63q8fwe regression."
    )
    assert "pandas" in result.evidence, (
        f"SKIP evidence must name 'pandas' as the missing dep so the "
        f"operator can resolve the env gap; got: {result.evidence}"
    )
    # The expected runtime per AGENTS.md §5.1 standing GPU grant +
    # §5.9 single-venv rule: the repo-local canonical .venv is THE
    # pandas-enabled runtime.
    assert "/home/hugo/codes/mol-metal/.venv/bin/python" in result.evidence, (
        f"SKIP evidence must name the expected pandas-enabled "
        f"runtime per AGENTS.md §5.1; got: {result.evidence}"
    )
    # The pre-flight MISSING-file branch should NOT fire (csv_path
    # exists + is_file()), so the FAIL-with-missing-CSV contract is
    # NOT what we're testing here. The SKIP must come from the
    # env-gap branch.
    assert "missing" not in result.evidence.lower() or "v4" in result.evidence, (
        f"SKIP should be from env-gap branch (NOT missing-CSV "
        f"branch); got: {result.evidence}"
    )
    assert result.wall_sec >= 0.0
