"""Wf-308A — global audit: every smoke test must FAIL-on-missing, never skip.

==============================================================================
Wf-308A / TODO-178b — smoke-test FAIL-on-missing global audit.

Per Wf-307B (TODO-178b hygiene) + party-twenty-spirit §3.5: silent
``pytest.skip`` is NOT acceptable for ship-cohort tests. If the data a
smoke test depends on is missing, the test must raise AssertionError /
``pytest.fail`` so the absence surfaces as a ship blocker rather than a
hidden hygiene gap.

Wf-303C reference: ``molmetal/tests/test_wasserstein_closure_integration.py``
introduced the FAIL-on-missing pattern with the
``_metalcytotox_preflight()`` helper + meta-test at line 763+
(``test_metalcytotox_csv_integration_smoke_is_active_not_skipped``).
Wf-307B extended it to the smoke / named-scripts cohort. Wf-308A audits
*every* test file in ``molmetal/tests/`` whose name contains ``smoke`` to
verify the FAIL-on-missing contract is upheld project-wide.

This is itself a meta-test that walks the test tree, classifies each
``pytest.skip`` call in the smoke files as LEGITIMATE (hardware /
optional-dep / optional-probe) or ILLEGITIMATE (ship-data missing), and
fails the suite when it finds an illegitimate skip-on-missing.

Honest framing note (party-twenty-spirit §3.5): this meta-test does NOT
itself run the smoke tests — it statically audits their skip calls.
A test that uses ``pytest.fail`` instead of ``pytest.skip`` for missing
data trivially passes this audit because the audit looks for the
``skip`` token, not the ``fail`` token. The audit therefore enforces a
negative contract (no illegitimate skip) rather than a positive
contract (must use fail). The positive contract is enforced by the
smoke tests themselves, which we read and report verbatim.

Tolerance: we accept that the audit is intrinsically heuristic. A
maintainer adding a new smoke test with an illegitimate skip-on-missing
will trip this meta-test and be forced to choose between FAIL-on-missing
or a deliberate LEGITIMATE classification with a comment. This is the
intended feedback loop.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# Heuristics for legitimate skip. A pytest.skip() is legitimate when
# the reason text contains one of these substrings OR the missing
# dependency is clearly hardware / optional-probe / optional-dep.
LEGITIMATE_SKIP_TOKENS = (
    # GPU/CUDA availability — host hardware probe
    "no cuda",
    "no gpu",
    "gpu not available",
    "gpu is unavailable",
    "gpu unavailable",
    "rocm",
    "no rocm",
    "no quickvina",
    "quickvina-gpu binary",
    "qvina-gpu binary",
    "no quickvina-gpu",
    # Optional probes — explicitly marked optional in the docstring
    "probe optional",
    # Missing optional dependency (NOT ship data)
    "rdkit not installed",
    "rdkit not available",
    "sklearn unavailable",
    "module not importable",
    "module unavailable",
    "not importable",
    "wheel not installed",
    "not present at",
    "sascorer not available",
    # Empty-branch checkout toleration — the entire suite should still run
    "smoke scripts directory not present",
    "smoke script not present",
    "smoke retrain script not present",
    "smoke script",
    "all.sh not present",
    # Test runner limitations — non-ship-data
    "pytest is not on path",
    # Data-CSV integrity loss of a *single* sample (not the whole cohort)
    "embed failed",
    "mmff94s unavailable for",
    "mmff94s relax failed for",
    "multi-reactant canonical inputs not curated",
    "canonical reactants did not match",
    "sklearn unavailable; chemprop v3 smoke cannot run",
    "single class; chemprop v3 smoke cannot",
    "rdkit returns none for empty smiles",
    # CLI feature not yet added (acknowledged gap, not ship-data)
    "--lambda-reward not yet added",
    "non-fatal in ci",
    # Data file presence for *new* training runs that the test itself
    # may have skipped
    "csv not found",  # tolerated when test is a CLI smoke that needs
                       # pre-existing data; we still flag the bulk
                       # skip patterns below as illegitimate.
)

# Patterns that are *always* illegitimate: the test silently skips
# when SHIP DATA is missing. A legitimate skip-on-missing-data is
# only acceptable when the missing file is explicitly documented as
# OPTIONAL in the test docstring.
ILLEGITIMATE_SKIP_TOKENS = (
    "platinai csv not found",
    "platinai 4-cell csv not found",
    "platinai csv",
    "metalcytotox csv",
    "metalcytotoxdb",
    "metalcytotox_5metal",
    "post-fix dir missing",
    "missing per-seed metrics for",
    "no per-seed data",
    "missing synthesis.md",
    "wf-30 phase-3 output missing",
    "phase-3 output missing",
    "phase-3 outputs are missing",
    "wf30 phase-3 outputs",
    "canonical report missing",
    "missing csv",
)


def _is_smoke_file(p: Path) -> bool:
    """Smoke tests are identified by filename containing 'smoke'."""
    return "smoke" in p.stem.lower()


def _read_skip_calls(path: Path) -> List[Tuple[int, str, str]]:
    """Parse ``path`` and return [(lineno, fn_name, skip_reason_text)].

    ``fn_name`` is the enclosing function / method name (best-effort).
    ``skip_reason_text`` is the string passed to ``pytest.skip(...)``.
    Skips where the reason can't be resolved (e.g. f-string variable) are
    returned with ``skip_reason_text='<unresolved>'`` so the audit still
    surfaces them.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out: List[Tuple[int, str, str]] = []

    def _name(node: ast.AST) -> str:
        # Walk up to find enclosing FunctionDef / AsyncFunctionDef
        # (caller passes parent via closure — we approximate by name attr)
        return getattr(node, "name", "<module>")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match ``pytest.skip("...")`` calls.  Skip calls on the
        # ``pytest`` module are the canonical silent-skip path.
        func = node.func
        is_pytest_skip = False
        if isinstance(func, ast.Attribute) and func.attr == "skip":
            if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                is_pytest_skip = True
        if not is_pytest_skip:
            continue
        # Resolve the skip reason string.
        reason = "<unresolved>"
        if node.args and isinstance(node.args[0], ast.Constant):
            reason = str(node.args[0].value)
        else:
            # Joined string or f-string: best-effort literal scan.
            for arg in node.args:
                if isinstance(arg, ast.JoinedStr):
                    parts = []
                    for v in arg.values:
                        if isinstance(v, ast.Constant):
                            parts.append(str(v.value))
                    reason = " ".join(parts) or "<fstring>"
                    break
        # Locate enclosing function name by walking up.
        # ast.walk doesn't preserve parents, so we re-walk and use the
        # closest preceding FunctionDef with end_lineno > node.lineno.
        fn_name = "<module>"
        for parent in ast.walk(tree):
            if (
                isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                and parent.lineno <= node.lineno
                and (parent.end_lineno or parent.lineno) >= node.lineno
                and (
                    fn_name == "<module>"
                    or (parent.lineno > _dummy_max_lineno(fn_name, tree))
                )
            ):
                fn_name = parent.name
        out.append((node.lineno, fn_name, reason))
    return out


def _dummy_max_lineno(name: str, tree: ast.AST) -> int:
    """Return max lineno of any function named ``name`` in ``tree``."""
    best = -1
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            best = max(best, node.lineno)
    return best


def _classify_skip(reason: str) -> str:
    """Return 'LEGITIMATE', 'ILLEGITIMATE', or 'INDETERMINATE'."""
    r = reason.lower()
    if any(tok in r for tok in ILLEGITIMATE_SKIP_TOKENS):
        return "ILLEGITIMATE"
    if any(tok in r for tok in LEGITIMATE_SKIP_TOKENS):
        return "LEGITIMATE"
    return "INDETERMINATE"


def _walk_smoke_files() -> List[Path]:
    """Find all *smoke* test files (not _smoke_ fixtures / dirs)."""
    out: List[Path] = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        if not p.is_file():
            continue
        if _is_smoke_file(p):
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Meta-test 1 — every smoke test file must declare ≥ 1 test (sanity)
# ---------------------------------------------------------------------------
def test_every_smoke_file_has_at_least_one_test():
    """Smoke files are not allowed to be empty placeholders."""
    empty: List[str] = []
    for p in _walk_smoke_files():
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            empty.append(f"{p.name}: SyntaxError {exc}")
            continue
        tests = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        ]
        if not tests:
            empty.append(f"{p.name}: no test_* functions")
    assert not empty, (
        f"{len(empty)} smoke files are empty placeholders: {empty}"
    )


# ---------------------------------------------------------------------------
# Meta-test 2 — collect the audit (informational; does not fail)
# ---------------------------------------------------------------------------
def test_smoke_skip_audit_collects_inventory(tmp_path: Path):
    """Walk every *smoke* test file and dump the skip inventory.

    The audit sidecar lands at ``<tmp>/smoke_skip_audit.json`` so a
    reviewer can inspect the per-file skip classification. This test
    ALWAYS passes — the actual fail-on-missing contract is enforced by
    the next meta-test (#3).
    """
    audit: dict = {
        "n_smoke_files": 0,
        "n_skip_calls": 0,
        "n_legitimate": 0,
        "n_illegitimate": 0,
        "n_indeterminate": 0,
        "files": {},
        "illegitimate_examples": [],
    }
    for p in _walk_smoke_files():
        audit["n_smoke_files"] += 1
        skips = _read_skip_calls(p)
        file_rec: dict = {
            "path": str(p.relative_to(TESTS_DIR.parent)),
            "n_skips": len(skips),
            "skips": [],
        }
        for lineno, fn_name, reason in skips:
            cls = _classify_skip(reason)
            audit["n_skip_calls"] += 1
            if cls == "LEGITIMATE":
                audit["n_legitimate"] += 1
            elif cls == "ILLEGITIMATE":
                audit["n_illegitimate"] += 1
                if len(audit["illegitimate_examples"]) < 10:
                    audit["illegitimate_examples"].append({
                        "file": p.name, "line": lineno,
                        "fn": fn_name, "reason": reason,
                    })
            else:
                audit["n_indeterminate"] += 1
            file_rec["skips"].append({
                "line": lineno, "fn": fn_name, "reason": reason,
                "class": cls,
            })
        audit["files"][p.name] = file_rec

    out = tmp_path / "smoke_skip_audit.json"
    import json
    out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    # Inventory test always passes — its job is to produce the sidecar.
    assert out.exists()


# ---------------------------------------------------------------------------
# Meta-test 3 — enforce the FAIL-on-missing contract
# ---------------------------------------------------------------------------
def test_no_smoke_test_silently_skips_on_missing_ship_data():
    """Every ``pytest.skip`` in a smoke file must be LEGITIMATE.

    ILLEGITIMATE skip-on-missing-data is forbidden per party-twenty-spirit
    §3.5: a missing ship-data artifact must FAIL the suite (raise
    AssertionError via ``pytest.fail`` or `assert`), not silently skip.

    This is the load-bearing assertion: if a maintainer re-introduces a
    silent skip on a missing PlatinAI / MetalCytoTox / WF-30 / Wf-122
    post-fix / canonical-report file, this meta-test will fail and force
    the maintainer to either:

      (a) Replace the skip with a hard ``pytest.fail`` + AssertionError
          (the preferred fix; cf. ``_metalcytotox_preflight`` in
          ``test_wasserstein_closure_integration.py``), OR
      (b) Move the data dependency under the LEGITIMATE tokens list
          with a justifying rationale in the test docstring.

    Tolerated: indeterminate skips are surfaced but not failed because
    the heuristic is intrinsically imprecise — a maintainer who is
    uncertain about a skip can leave it indeterminate and a human
    reviewer resolves it during code review.
    """
    illegitimate: List[dict] = []
    for p in _walk_smoke_files():
        for lineno, fn_name, reason in _read_skip_calls(p):
            cls = _classify_skip(reason)
            if cls == "ILLEGITIMATE":
                illegitimate.append({
                    "file": p.name,
                    "line": lineno,
                    "fn": fn_name,
                    "reason": reason,
                })

    assert not illegitimate, (
        f"{len(illegitimate)} illegitimate pytest.skip(s) found in smoke "
        "tests. Per party-twenty-spirit §3.5 silent skip on missing ship-"
        "data is forbidden. Fix each by replacing with pytest.fail +\n"
        "AssertionError, OR reclassify as LEGITIMATE with a justifying "
        "docstring. Examples:\n"
        + "\n".join(
            f"  - {x['file']}:{x['line']} ({x['fn']}): {x['reason'][:80]}"
            for x in illegitimate[:10]
        )
        + (f"\n  ... and {len(illegitimate) - 10} more" if len(illegitimate) > 10 else "")
    )


# ---------------------------------------------------------------------------
# Meta-test 4 — the FAIL-on-missing reference module is still in place
# ---------------------------------------------------------------------------
def test_metalcytotox_preflight_still_raises_on_missing():
    """Defence-in-depth: the Wf-303C reference must remain FAIL-on-missing.

    We import ``_metalcytotox_preflight`` from the wasserstein-closure
    integration test and confirm it raises FileNotFoundError /
    AssertionError when pointed at a non-existent CSV. This is the
    canonical pattern that Wf-308A audits every other smoke test against.
    """
    import importlib.util

    ref_path = TESTS_DIR / "test_wasserstein_closure_integration.py"
    spec = importlib.util.spec_from_file_location(
        "_ref_wass_closure", ref_path,
    )
    assert spec is not None and spec.loader is not None, (
        f"could not load {ref_path} as a module spec"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Confirm the helper exists and is callable.
    assert hasattr(mod, "_metalcytotox_preflight"), (
        f"reference module {ref_path} no longer exposes "
        "_metalcytotox_preflight — Wf-303C hygiene has been undone."
    )
    preflight = mod._metalcytotox_preflight

    # When the CSV is absent, the helper must raise (NOT silently skip).
    # We monkey-patch the helper's path-resolution to return None so the
    # preflight sees a missing file even on hosts where the CSV is
    # present (which it is on this checkout).
    original = mod._metalcytotox_csv_path
    mod._metalcytotox_csv_path = lambda: None  # type: ignore[assignment]
    raised = False
    raised_exc_type: str = "<none>"
    try:
        preflight()
    except (FileNotFoundError, AssertionError) as exc:
        raised = True
        raised_exc_type = type(exc).__name__
    except Exception as exc:
        raised = True
        raised_exc_type = type(exc).__name__
        raise AssertionError(
            f"_metalcytotox_preflight raised unexpected "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        mod._metalcytotox_csv_path = original  # type: ignore[assignment]
    assert raised, (
        "_metalcytotox_preflight silently returned instead of raising; "
        "the Wf-303C FAIL-on-missing contract has been undone."
    )
    assert raised_exc_type in ("FileNotFoundError", "AssertionError"), (
        f"_metalcytotox_preflight raised {raised_exc_type}; expected "
        "FileNotFoundError or AssertionError per the Wf-303C contract."
    )


# ---------------------------------------------------------------------------
# Meta-test 5 — the new audit test itself is importable & well-formed
# ---------------------------------------------------------------------------
def test_this_meta_test_is_self_consistent():
    """The audit meta-test must classify its OWN skips correctly.

    This file declares no pytest.skip calls (the audit enumerates skips
    statically rather than running them), so the n_skip_calls count for
    THIS file should be zero. If a future maintainer adds a
    pytest.skip to this meta-test (e.g. to tolerate a specific test
    file), the LEGITIMATE_TOKENS list above must classify it
    correctly, otherwise the audit will false-positive on its own
    file.
    """
    skips = _read_skip_calls(TESTS_DIR / "test_smoke_fail_on_missing_audit.py")
    assert skips == [], (
        f"this meta-test contains {len(skips)} pytest.skip calls; "
        "the audit is supposed to inspect other files, not skip itself. "
        f"Found: {skips}"
    )
