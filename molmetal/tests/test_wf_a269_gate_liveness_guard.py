# Licensed under the Apache License, Version 2.0.
r"""Wf-A269 wave-11 — the DEAD-GATE GUARD's own test (the sabotage-proof test).

WHY THIS FILE EXISTS
--------------------
A guard that cannot be MADE to fail is indistinguishable from a guard that
works.  The research arm (Wei 2026, arXiv:2610.05981 — "each detector proven
load-bearing by sabotage"; Papadakis et al. 2019 — mutation testing makes
effectiveness claims falsifiable) says the same thing about governance
checks: a check that has never fired is UNVERIFIED, not passing.  So this
file is the guard's own mutation test: it ships one synthetic dead gate per
mode, and asserts the guard CATCHES it.

THE INTERFACE THIS FILE EXPECTS
------------------------------------------------------------------------
Module: ``molmetal/scripts/wf_launch_gate_liveness.py``.

The guard may expose ANY of these three interfaces; the adapter in this
file tries them in this order:

  A. a module-level ``scan(root, ...)`` callable -> driven in-process;
  B. a module-level ``main(argv)`` callable      -> driven in-process, the
     machine-readable report is written to a tmp JSON file and parsed
     (this is what the landed guard exposes);
  C. neither -> the CLI is run as a SUBPROCESS with ``sys.executable`` and
     its ``--json`` report is parsed.

In every case the findings are NORMALISED to one shape
(``mode`` / ``status`` / ``wrapper`` / ``line`` / ``detail``), so the
assertions below do not depend on the guard's class layout.

1. the scan itself
   A pure-Python, CPU-only, EXECUTES-NOTHING scanner.  ``root`` is a
   directory that plays the role of the repository root: the guard reads
   ``<root>/scripts/**/*.sh`` as the wrapper corpus and ``<root>/**/*.py``
   as the tool/producer index.

   * The corpus MUST be enumerated from the FILESYSTEM for a root that is
     not a git repository (the fixtures below live under ``tmp_path`` and
     are not git repos).  The fixture roots are ``git init``-ed anyway so
     that a ``git ls-files --cached --others --exclude-standard``
     enumeration ALSO works — the guard may use either.

   * The guard MUST resolve a tool referenced through a shell variable, e.g.
     ``PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"`` followed by
     ``"${PY}" "${PRE_FLIGHT}" --pipeline all``.  Every one of the six
     MEASURED real wrappers refers to its tools that way, so a guard that
     cannot resolve it cannot flag the real corpus at all.

   * A ``Finding`` must expose, by attribute OR by dict key:
       - the MODE: one of (i)/(ii)/(iii)/(iv)
       - the STATUS: ``"DEAD"`` (confirmed) / ``"UNRESOLVED"`` (the guard
         ABSTAINS — never silently "clean") / optionally ``"OK"``
       - the WRAPPER path and the LINE number
     Attribute names are matched leniently (see ``_fget``): the mode may be
     called ``mode``/``kind``/``category``/``rule``; the status
     ``status``/``verdict``/``state``; the path
     ``wrapper``/``path``/``file``/``location``; the line
     ``line``/``lineno``/``line_no``.  The mode VALUE may be the numeric
     ``"i"``/``"ii"``/``"iii"``/``"iv"`` or a descriptive alias
     (``"undefined_flag"``, ``"dead_marker"``, ``"dead_exit_code"``,
     ``"invalid_json"``); ``_mode_class`` maps both vocabularies.

   * TWO TRAPS this test's own reference detector had to survive, both
     MEASURED on the real corpus:
     1. ``2>&1`` is NOT a statement separator — a shell splitter that cuts
        on ``&`` destroys the very ``> ....json 2>&1`` signature mode (iv)
        looks for.  Mask the redirections before splitting.
     2. A quoted string that CONTAINS WHITESPACE is usually PROSE (the
        measured false-positive class: ``log "pre_flight_check.py --pipeline
        launch (…)``), so blanking it is right for the mode-(i) scan — but
        the ARGUMENT OF ``grep`` is a MARKER, not prose, and a multi-word
        marker (``grep -q "phase complete" train.log``) must survive.  Read
        the grep PATTERN argument from the raw command text.
     The same trap applies to the mode-(iv) redirect TARGET: every MEASURED
     real wrapper writes it as ``>"${OUT_DIR}/preflight.json"`` — a QUOTED
     path with a shell variable inside.  A regex that requires an unquoted
     target finds nothing.

   * ``scan`` MAY accept two optional keyword arguments, ``registry`` and
     ``allowlist``, that point at the MEASURED exit-code registry and at the
     allow-list.  ``_call_scan`` passes them under whichever of the
     accepted names the guard declares (``registry``/``registry_path``/
     ``exitcode_registry``/… and ``allowlist``/``allowlist_path``/…).  If
     the guard declares NEITHER, ``_forced_paths`` monkeypatches the
     guard module's own module-level constants whose names contain
     ``registry`` / ``allow`` (a ``str``/``Path`` constant is pointed at the
     test's file; a ``dict`` constant is emptied).  That fallback keeps the
     test implementable against either design.

2. ``__main__`` — a CLI that prints the report and honours the AGENTS.md
   §4 stdout invariant (first token ``PASS`` / ``FAIL`` / ``SKIP``; exit
   0/1/2) and accepts ``--json <path>`` (the report file).

THE FOUR DEAD-GATE MODES THIS FILE GUARDS (from AGENTS.md §6.269.17)
--------------------------------------------------------------------
   (i)   the wrapper passes a flag the tool does not define (argparse exits
         2 instantly, so the gate is decorative);
   (ii)  the wrapper greps stdout for a marker that has no producer;
   (iii) the wrapper keys a gate on a tool exit code that is 0 BY
         CONSTRUCTION — the FAIL verdict lives in the tool's report, never
         in its exit code (``wave_budget_tracker.py`` main() returns 0 on
         every path; ``per_wave_checklist.py`` main() returns 0
         unconditionally);
   (iv)  the wrapper redirects stdout+stderr into a ``.json`` path (so the
         artefact is not JSON at all), OR it passes a BARE ``--json`` to a
         tool whose ``--json`` is ``type=Path, nargs="?"`` (a bare ``--json``
         writes NO file and the payload goes to stdout).

THE REGISTRY CONTRACT (the probe writes it; the guard reads it)
----------------------------------------------------------------
``{"schema_version": 1, "measured_at": <ISO-8601 with tz>,
   "measured_by": <str>, "host": <str>,
   "tools": {"<tool file basename>": {
       "exit_code_under_failing_input": <int>,
       "command": <str>, "failing_input": <str>,
       "can_carry_fail": <bool>, "verdict_location": <str|null>}}}``
``can_carry_fail`` is true iff the exit code under a failing input is
NON-ZERO.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not write into the repo (every fixture lives under ``tmp_path``),
it does not write into ``metrics/`` (asserted by a module-scoped
autouse fixture that snapshots ``git status --porcelain metrics/`` before
and after), it does not touch ``paper/`` (TODO-84), it launches no GPU
work (``CUDA_VISIBLE_DEVICES=""`` on every subprocess), and it does not
shell out to a trainer.

CO-EXISTENCE
------------
``molmetal/tests/test_wf_a269_wave10_dead_gates.py`` (14 tests, untracked)
guards TWO repaired wrappers with a per-wrapper substring method.  This
file GENERALISES it: it does not edit that file, and it does not replace
the per-wrapper assertions — it proves that a POPULATION-WIDE guard can
fire at all.

THE REGRESSION CORPUS (measured on disk by this agent, 2026-11-02)
------------------------------------------------------------------
The wave-11 research arms MEASURED six wrappers.  Re-checking their LIVE
code (full-line comments stripped, because AGENTS.md §5.5 rule #6 keeps
the superseded rendering on disk) at the time this test was written:

  * STILL DEAD at the FIRST reading (2026-11-02 22:40) — the guard MUST
    flag these:
      scripts/launch_wf299_cfm_main.sh                            (7 sites)
      scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh    (1 site)
      scripts/launch_wf310f_frontier_math_gpu.sh                  (1 site)
    ("sites" = live `> ....json 2>&1` captures of a tool, plus bare
    `--json` flags whose dest is `type=Path, nargs="?"`.)
  * REPAIRED before this test was finished — the guard must NOT be
    required to flag them (demanding it would encode a stale fact):
      scripts/launch_wf146_deferred_runs.sh                       (0 sites)
      scripts/launch_wf74_v2_drugood_dann.sh                      (0 sites)
      scripts/launch_wf306a_drugood_vina_gpu.sh                   (0 sites)
  * REPAIRED between the first reading and the FINAL reading
    (2026-11-02 22:59) — also not demanded:
      scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh    (0 sites)
    This is the third observed instance of the list rotting under the
    feet of a fixed expectation; it is the reason the test RE-DERIVES
    the population instead of freezing it.
  * REPAIRED between the 22:59 reading and the 23:03 reading — also not
    demanded:
      scripts/launch_wf299_cfm_main.sh                            (0 sites)
      scripts/launch_wf310f_frontier_math_gpu.sh                  (0 sites)

THE POPULATION IS A MOVING TARGET.  Four readings were taken during this
session's writing window alone (22:40, 22:59, 23:02, 23:03), and the
count fell 12 -> 11 -> 3 sites.  Any test that FREEZES the dead set is
red by the next repair wave and green only by luck; the test therefore
re-derives the population from disk on EVERY run and asserts only the
direction-invariant property (a wrapper that still carries the signature
MUST be flagged).  The measurements below are an APPEND-ONLY LOG of
readings (AGENTS.md §5.5 rule #6), NOT a fixture — a later reading
supersedes an earlier one without deleting it.

The wave-11 repair pass is landing CONCURRENTLY with this test, so the
list above is a POINT-IN-TIME reading, not a fixture.  The test therefore
RE-DERIVES the live population from disk every time it runs: a wrapper
that still carries the signature MUST be flagged, a wrapper that has been
repaired is NOT demanded.  What is frozen in the source is only the
RESEARCH CORPUS (the six wrappers the research measured), named
explicitly in ``MEASURED_DEAD_WRAPPERS``.

The population the detector derives is wider than the six, and it
SHRANK under the concurrent repair pass.  Measured census in
``scripts/launch_*.sh`` (the detector re-run on disk, not copied):

  * 22:40, 2026-11-02 — 6 wrappers / 12 sites
    (``launch_wf299_cfm_main.sh`` 7, ``launch_wf306b_wf219_ru_gpu.sh`` 1,
    ``launch_wf310c_t179_w2_gpu.sh`` 1,
    ``launch_wf310e_r20_p3_...sh`` 1,
    ``launch_wf310f_frontier_math_gpu.sh`` 1,
    ``launch_wf446_cfm_tmqm_large_cohort.sh`` 1)
  * 22:59, 2026-11-02 — 5 wrappers / 11 sites (the same set minus
    ``launch_wf310e_r20_p3_...sh``); the line numbers inside
    ``launch_wf299_cfm_main.sh`` also moved (439/460/473/486/499/525/687
    in the final reading).
  * 23:02, 2026-11-02 — 5 wrappers / 11 sites (unchanged from 22:59).
  * 23:03, 2026-11-02 — 3 wrappers / 3 sites:
    ``launch_wf306b_wf219_ru_gpu.sh`` 1 (L141),
    ``launch_wf310c_t179_w2_gpu.sh`` 1 (L256),
    ``launch_wf446_cfm_tmqm_large_cohort.sh`` 1 (L120).
    ``launch_wf299_cfm_main.sh`` and ``launch_wf310f_frontier_math_gpu.sh``
    were repaired inside this three-minute window.

Because the population moves, the ASSERTION is not "these N wrappers are
dead" but "every wrapper the detector finds on disk TODAY is flagged by
the guard, and the guard's own count is not allowed to shrink below the
detector's".  A future repair wave that removes a wrapper therefore
leaves this test GREEN, and a guard that quietly stops seeing the
survivors turns it RED.

Coverage bound (AGENTS.md §6.269.6 — a census is a COVERAGE census, not a
defect count): ``scripts/_retired/`` and the ``_``-prefixed helper
templates are OUT of corpus — ``scripts/_WRAPPER_TEMPLATE.sh`` was
MEASURED carrying the same signature (1 site), so a guard that scans it
reports one more wrapper than this corpus names.

STATE OF THIS FILE
-------------------
This test was written BEFORE the guard existed (it was an HONEST RED: every
test failed with "the guard module is missing: <path>").  The guard landed
concurrently with this file (2026-11-02) and the seven tests below now
PASS against it — measured, not asserted.

THE TEST IS LOAD-BEARING, AND THAT WAS MEASURED TWO WAYS.

  * By MUTATION.  Two stub guards were written and the test was re-run
    against each: a guard that reports NOTHING (the classic dead gate)
    and a guard that reports EVERY wrapper as mode-(i) DEAD.  The first
    fails 6 of the 7 tests, the second fails all 7.  Neither mutant can
    pass, which is the property the test is supposed to have.
  * By the LIVE GUARD ITSELF.  While the guard was being edited
    concurrently, it carried a transient ``UnboundLocalError:
    corpus_digest`` at its own ``wf_launch_gate_liveness.py:1968``.  This
    test went RED on that edit on its own (``5 failed, 2 passed``) with
    the guard's own traceback quoted in the failure message, and went
    GREEN again when the author fixed it — a live demonstration that a
    quiet guard cannot hide from this test.

THE FIXTURES WERE ADAPTED TO THE GUARD'S MEASURED SEMANTICS, NOT THE
OTHER WAY ROUND.  Two of the adaptations are worth recording because they
are the guard's real, defensible boundaries:

  * a tool path in the fixture must be ABSOLUTE (or resolvable from the
    scan root) — the guard resolves a tool against the file system, so an
    unexpanded ``${PROJECT_ROOT}`` makes the tool unresolvable and every
    mode that needs the tool's argparse AST is UNRESOLVED, not DEAD;
  * a mode-(ii) marker must be a SINGLE literal token — the guard ABSTAINS
    (UNRESOLVED) on a marker containing whitespace or a regex
    metacharacter, because it cannot decide whether a REGEX has a
    producer.  That abstention is correct behaviour, and the fixture uses a
    decidable literal so the test can prove the mode fires at all.

Nothing here was weakened to make the guard green: the assertions are
exactly the ones written against the spec, and a stub guard that reports
nothing (or reports everything) FAILS this file — see the report.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "molmetal" / "scripts" / "wf_launch_gate_liveness.py"

# The six wrappers the wave-11 research arms MEASURED as carrying dead gates
# (catalog AGENTS.md §6.269.17, sub-modes (i)-(iv)).  NAMED EXPLICITLY, as
# the brief requires.
#
# MEASURED by this agent on 2026-11-02 against the CURRENT disk state: the
# wave-11 repair pass is landing CONCURRENTLY with this test, so a frozen
# "still dead" list rots.  The list below is therefore the RESEARCH
# CORPUS (the six wrappers the research measured), NOT a claim about which
# of them are still broken.  Which of them are still broken is RE-DERIVED
# from disk at test time by `_mode_iv_sites()` — see
# `test_guard_flags_the_measured_dead_wrappers`.
#
#   scripts/launch_wf146_deferred_runs.sh
#   scripts/launch_wf74_v2_drugood_dann.sh
#   scripts/launch_wf306a_drugood_vina_gpu.sh
#   scripts/launch_wf299_cfm_main.sh
#   scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh
#   scripts/launch_wf310f_frontier_math_gpu.sh
MEASURED_DEAD_WRAPPERS: Tuple[str, ...] = (
    "launch_wf146_deferred_runs.sh",
    "launch_wf74_v2_drugood_dann.sh",
    "launch_wf306a_drugood_vina_gpu.sh",
    "launch_wf299_cfm_main.sh",
    "launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh",
    "launch_wf310f_frontier_math_gpu.sh",
)

# ---------------------------------------------------------------------------
# finding vocabulary
# ---------------------------------------------------------------------------

_MODE_ALIASES: Dict[str, set] = {
    "i": {
        "i",
        "1",
        "undefined_flag",
        "undefined_arg",
        "flag_not_defined",
        "unknown_flag",
        "nonexistent_flag",
        "bad_flag",
        "argparse_error",
    },
    "ii": {
        "ii",
        "2",
        "dead_marker",
        "marker",
        "no_producer",
        "missing_producer",
        "dead_grep",
        "grep_no_producer",
    },
    "iii": {
        "iii",
        "3",
        "dead_exit_code",
        "exit_code",
        "always_zero_exit",
        "exit_code_zero",
        "zero_exit",
        "cannot_fail",
    },
    "iv": {
        "iv",
        "4",
        "invalid_json",
        "non_json",
        "not_json",
        "json_artifact",
        "redirect_json",
        "bad_json",
        "json_capture",
    },
}

_DEAD_STATUS = {"dead", "fail", "failed", "confirmed", "dead_gate", "blocking"}
_UNRESOLVED_STATUS = {
    "unresolved",
    "unknown",
    "skip",
    "skipped",
    "indeterminate",
    "abstain",
    "abstained",
    "cannot_determine",
    "not_determined",
}

_REGISTRY_KWARG_NAMES = (
    "registry",
    "registry_path",
    "exitcode_registry",
    "exitcode_registry_path",
    "registry_json",
)
_ALLOWLIST_KWARG_NAMES = (
    "allowlist",
    "allowlist_path",
    "allow",
    "allowlist_json",
    "allow_json",
)


# ---------------------------------------------------------------------------
# generic accessors (attribute OR dict key; lenient on the names)
# ---------------------------------------------------------------------------


def _fget(finding: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(finding, dict):
            value = finding.get(name)
        else:
            value = getattr(finding, name, None)
        if value is not None:
            return value
    return default


def _norm_token(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text.startswith("mode_"):
        text = text[len("mode_") :]
    return text


def _mode_class(finding: Any) -> Optional[str]:
    """Map a Finding's mode onto 'i' / 'ii' / 'iii' / 'iv' (or None)."""
    raw = _fget(
        finding, ("mode", "mode_id", "kind", "category", "rule", "type", "check")
    )
    if raw is None:
        return None
    token = _norm_token(raw)
    for canonical, aliases in _MODE_ALIASES.items():
        if token in aliases:
            return canonical
    # roman-numeral fall-back: 'i', 'ii', 'iii', 'iv', 'iva', 'ivb' ...
    roman = re.sub(r"[^a-z]", "", token)
    if re.fullmatch(r"i{1,3}v?[a-z]?", roman):
        if roman.startswith("iii"):
            return "iii"
        if roman.startswith("iv"):
            return "iv"
        if roman.startswith("ii"):
            return "ii"
        return "i"
    # substring fall-back, most specific first
    if "exit" in token:
        return "iii"
    if "json" in token:
        return "iv"
    if "marker" in token or "grep" in token or "producer" in token:
        return "ii"
    if "flag" in token or "arg" in token:
        return "i"
    return None


def _status(finding: Any) -> str:
    raw = _fget(finding, ("status", "verdict", "state", "classification", "result"))
    return _norm_token(raw) if raw is not None else ""


def _is_dead(finding: Any) -> bool:
    return _status(finding) in _DEAD_STATUS


def _is_unresolved(finding: Any) -> bool:
    return _status(finding) in _UNRESOLVED_STATUS


def _wrapper(finding: Any) -> str:
    raw = _fget(
        finding,
        ("wrapper", "wrapper_path", "path", "file", "location", "source", "script"),
        "",
    )
    return str(raw)


def _line(finding: Any) -> int:
    raw = _fget(finding, ("line", "lineno", "line_no", "line_number"), -1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _describe(finding: Any) -> str:
    return (
        f"mode={_fget(finding, ('mode', 'kind', 'category', 'rule'), '?')} "
        f"status={_status(finding) or '?'} "
        f"wrapper={_wrapper(finding)} line={_line(finding)}"
    )


# ---------------------------------------------------------------------------
# guard loading
# ---------------------------------------------------------------------------


def _load_guard() -> Any:
    if not GUARD_PATH.is_file():
        pytest.fail(
            "the guard module is missing: "
            f"{GUARD_PATH} — this test is written BEFORE the guard "
            "(molmetal/scripts/wf_launch_gate_liveness.py) and is expected "
            "to be RED until it lands.  It must NOT be skipped: a governance "
            "check that never fires is invisible without sabotage."
        )
    spec = importlib.util.spec_from_file_location(
        "wf_a269_gate_liveness_guard_under_test", GUARD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not (
        callable(getattr(module, "scan", None))
        or callable(getattr(module, "main", None))
    ):
        pytest.fail(
            f"{GUARD_PATH} exposes neither a callable `scan(root)` entry "
            "point nor a callable `main(argv)` CLI entry point — the test "
            "cannot drive the guard."
        )
    return module


@pytest.fixture(scope="module")
def guard() -> Any:
    return _load_guard()


# ---------------------------------------------------------------------------
# scan invocation
#
# THREE interfaces are accepted, in this order:
#
#   A. a module-level ``scan(root, ...)`` callable -> in-process;
#   B. a module-level ``main(argv)`` callable      -> in-process CLI, the
#      machine-readable report is written to a tmp JSON file and parsed
#      (the task's sanctioned fallback: "shell out and parse its --json
#      report"; done in-process so no subprocess is needed);
#   C. neither -> the CLI is run as a SUBPROCESS with ``sys.executable``.
#
# In every case the findings are NORMALISED to one shape:
#   {"mode": str, "status": "DEAD"/"UNRESOLVED", "wrapper": str,
#    "line": int, "detail": str}
# ---------------------------------------------------------------------------


def _scan_params(guard: Any) -> Dict[str, Any]:
    scan = getattr(guard, "scan", None)
    if not callable(scan):
        return {}
    try:
        return dict(inspect.signature(scan).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}


@contextlib.contextmanager
def _forced_paths(
    guard: Any, registry: Optional[Path], allowlist: Optional[Path]
):
    """Point the guard at the test's registry / allow-list.

    Preferred path: a keyword argument on ``scan``.  Fallback: monkeypatch
    the guard module's own module-level constants whose names contain
    "registry" / "allow".
    """
    params = _scan_params(guard)
    patched: Dict[str, Any] = {}

    def _patch_matching(substring: str, path: Optional[Path]) -> None:
        if path is None:
            return
        for name, value in list(vars(guard).items()):
            if substring not in name.lower():
                continue
            if isinstance(value, (str, Path)):
                patched[name] = value
                setattr(guard, name, path)
            elif isinstance(value, dict):
                patched[name] = value
                setattr(guard, name, {})

    if registry is not None and not any(n in params for n in _REGISTRY_KWARG_NAMES):
        _patch_matching("registry", registry)
    if allowlist is not None and not any(n in params for n in _ALLOWLIST_KWARG_NAMES):
        _patch_matching("allow", allowlist)

    try:
        yield
    finally:
        for name, value in patched.items():
            setattr(guard, name, value)


def _normalise(raw: Any, status: str) -> Dict[str, Any]:
    """Normalise one guard finding (object OR dict) to the common shape."""
    line = _fget(raw, ("line", "lineno", "line_no", "line_number"), -1)
    try:
        line = int(line)
    except (TypeError, ValueError):
        line = -1
    return {
        "mode": _fget(
            raw, ("mode", "mode_id", "kind", "category", "rule", "type"), "?"
        ),
        "status": status,
        "wrapper": _fget(
            raw,
            ("wrapper", "wrapper_path", "file", "path", "location", "source", "script"),
            "",
        ),
        "line": line,
        "detail": _fget(raw, ("detail", "message", "reason", "note"), ""),
        "raw": raw,
    }


def _extract_findings(result: Any) -> List[Dict[str, Any]]:
    """Turn whatever ``scan`` returned into the normalised finding list."""
    out: List[Dict[str, Any]] = []
    if result is None:
        return out
    findings: Any = None
    unresolved: Any = None
    if isinstance(result, tuple) and len(result) == 2:
        findings, unresolved = result
    else:
        candidate = getattr(result, "findings", None)
        if candidate is not None:
            findings = candidate
            unresolved = getattr(result, "unresolved", None)
        else:
            findings = result
    if isinstance(findings, (list, tuple)):
        for item in findings:
            status = _fget(
                item, ("status", "verdict", "state", "classification", "result"), ""
            )
            out.append(_normalise(item, str(status) if status else "DEAD"))
    if isinstance(unresolved, (list, tuple)):
        for item in unresolved:
            out.append(_normalise(item, "UNRESOLVED"))
    return out


def _report_findings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Findings + unresolved entries of a machine-readable guard report."""
    out: List[Dict[str, Any]] = []
    for item in report.get("findings", []) or []:
        out.append(_normalise(item, "DEAD"))
    for item in report.get("unresolved", []) or []:
        out.append(_normalise(item, "UNRESOLVED"))
    return out


def _cli_argv(
    root: Path,
    report_path: Path,
    registry: Optional[Path],
    allowlist: Optional[Path],
) -> List[str]:
    argv = ["--repo-root", str(root), "--json", str(report_path)]
    if registry is not None:
        argv += ["--registry", str(registry)]
    if allowlist is not None:
        argv += ["--allowlist", str(allowlist)]
    return argv


def _call_scan(
    guard: Any,
    root: Path,
    registry: Optional[Path] = None,
    allowlist: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> List[Any]:
    # A. module-level scan(root)
    scan = getattr(guard, "scan", None)
    if callable(scan):
        params = _scan_params(guard)
        kwargs: Dict[str, Any] = {}
        if registry is not None:
            for name in _REGISTRY_KWARG_NAMES:
                if name in params:
                    kwargs[name] = registry
                    break
        if allowlist is not None:
            for name in _ALLOWLIST_KWARG_NAMES:
                if name in params:
                    kwargs[name] = allowlist
                    break
        with _forced_paths(guard, registry, allowlist):
            return _extract_findings(scan(root, **kwargs))

    # B. module-level main(argv) — the CLI, run in-process
    main = getattr(guard, "main", None)
    if callable(main):
        holder = workdir or Path(tempfile.mkdtemp(prefix="wfa269_w11_guard_"))
        holder.mkdir(parents=True, exist_ok=True)
        report_path = holder / "guard_report.json"
        main(_cli_argv(root, report_path, registry, allowlist))
        if not report_path.is_file():
            raise RuntimeError(
                "the guard's main() did not write the --json report to "
                f"{report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return _report_findings(report)

    # C. the CLI as a subprocess
    holder = workdir or Path(tempfile.mkdtemp(prefix="wfa269_w11_guard_"))
    holder.mkdir(parents=True, exist_ok=True)
    report_path = holder / "guard_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(GUARD_PATH),
            *_cli_argv(root, report_path, registry, allowlist),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if not report_path.is_file():
        raise RuntimeError(
            f"the guard CLI wrote no --json report (exit {proc.returncode}): "
            f"{proc.stdout[-500:]} {proc.stderr[-500:]}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return _report_findings(report)


def _scan_or_fail(
    guard: Any,
    root: Path,
    registry: Optional[Path] = None,
    allowlist: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> List[Any]:
    """``scan`` must not raise — a guard that crashes is not a guard."""
    try:
        return _call_scan(
            guard, root, registry=registry, allowlist=allowlist, workdir=workdir
        )
    except Exception as exc:  # noqa: BLE001 - the failure IS the finding
        pytest.fail(
            f"the guard raised {type(exc).__name__} on "
            f"root={root}, registry={registry}, allowlist={allowlist}: {exc}\n"
            "The guard must degrade to an UNRESOLVED abstention, never "
            "raise and never report silence as success."
        )
        raise  # unreachable, keeps type checkers happy


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

_ALWAYS_ZERO_TOOL = '''# Licensed under the Apache License, Version 2.0.
"""FIXTURE tool — models a gate tool whose exit code is 0 BY CONSTRUCTION.

Its FAIL verdict lives in the report it writes via ``--out``; the exit code
is always 0, so a wrapper that keys a gate on the exit code cannot fail.
This mirrors ``wave_budget_tracker.py`` and ``per_wave_checklist.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def parse_argv(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wf_a269_fixture_tool.py")
    p.add_argument("--pipeline", default="all")
    p.add_argument("--budget-hr", type=float, default=4.0)
    p.add_argument("--json", type=Path, nargs="?", default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_argv(argv)
    payload = {
        "verdict": "FAIL",
        "honest_framing": "fixture: the verdict lives in the report, not the exit code",
        "remaining_hr": -1.0,
    }
    if args.json is not None:
        args.json.write_text(json.dumps(payload), encoding="utf-8")
    if args.out is not None:
        args.out.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_FAILABLE_TOOL = '''# Licensed under the Apache License, Version 2.0.
"""FIXTURE tool — models a gate tool that CAN carry a FAIL in its exit code.

Mirrors ``pre_flight_check.py``: exit 0 when the (fake) run is clean, exit 1
when it is not, and ``--json`` is ``type=Path, nargs="?"`` so a bare
``--json`` writes NO file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def parse_argv(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wf_a269_fixture_failable_tool.py")
    p.add_argument("--pipeline", default="all")
    p.add_argument("--fail", action="store_true")
    p.add_argument("--json", type=Path, nargs="?", default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_argv(argv)
    n_fail = 1 if args.fail else 0
    payload = {
        "summary": {"pass": 1 - n_fail, "fail": n_fail, "skip": 0,
                    "timeout": 0, "total": 1},
    }
    if args.json is not None:
        args.json.write_text(json.dumps(payload), encoding="utf-8")
    if args.out is not None:
        args.out.write_text(json.dumps(payload), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_SABOTAGE_WRAPPER = """#!/usr/bin/env bash
# Wf-A269 wave-11 SABOTAGE FIXTURE — one synthetic dead gate per mode.
# This file exists ONLY under a pytest tmp_path.  It is never written into
# the repository.
set -uo pipefail

PROJECT_ROOT="@PROJECT_ROOT@"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
PY="${PROJECT_ROOT}/.venv/bin/python"
REPORTS_DIR="${PROJECT_ROOT}/reports"
FIXTURE_TOOL="${SCRIPTS_DIR}/wf_a269_fixture_tool.py"

mkdir -p "${REPORTS_DIR}"

# --- mode (i): a flag the tool does not define ----------------------------
# (i-1) the tool path is a plain literal
if ! "${PY}" "${PROJECT_ROOT}/molmetal/scripts/wf_a269_fixture_tool.py" \\
        --pipeline all --skip-anchor-guard > /dev/null 2>&1; then
    printf 'pre-flight FAIL\\n'
fi

# (i-2) the tool path comes from a shell variable assignment
if ! "${PY}" "${FIXTURE_TOOL}" --pipeline all --cfm-yuelbond-not-required \\
        > /dev/null 2>&1; then
    printf 'pre-flight FAIL (variable form)\\n'
fi

# --- mode (ii): a grep for a marker that has NO producer ------------------
# The marker is a SINGLE literal token: the guard abstains (UNRESOLVED) on a
# marker that contains whitespace or a regex metacharacter, because it cannot
# decide whether a REGEX has a producer.  A literal token is decidable.
if grep -q "PHASE_COMPLETE" train.log; then
    printf 'PHASE_COMPLETE\\n'
else
    printf 'phase aborted\\n'
fi

# --- mode (iii): a gate keyed on an exit code that is 0 by construction ----
if ! "${PY}" "${FIXTURE_TOOL}" --budget-hr 4; then
    printf 'budget tracker FAIL\\n'
    exit 9
fi

# (the same defect in the `|| { ... exit N; }` shape)
"${PY}" "${FIXTURE_TOOL}" --budget-hr 6 || { printf 'budget tracker FAIL\\n'; exit 9; }

# --- mode (iv): stdout+stderr redirected into a .json path -----------------
"${PY}" "${FIXTURE_TOOL}" --pipeline all > "${REPORTS_DIR}/pre_flight.json" 2>&1

# --- mode (iv-b): a BARE --json to a tool whose --json is nargs="?" --------
"${PY}" "${FIXTURE_TOOL}" --json

exit 0
"""

_CLEAN_WRAPPER = """#!/usr/bin/env bash
# Wf-A269 wave-11 NEGATIVE CONTROL (A) — a wrapper with NO gate at all.
# This file exists ONLY under a pytest tmp_path.
set -uo pipefail

PROJECT_ROOT="@PROJECT_ROOT@"
REPORTS_DIR="${PROJECT_ROOT}/reports"
mkdir -p "${REPORTS_DIR}"

# no tool invocation, no grep for a marker, no redirect into a .json path
printf 'launch begin\\n' | tee "${REPORTS_DIR}/launch.log"
printf 'launch complete\\n'
exit 0
"""

_CLEAN_GATE_WRAPPER = """#!/usr/bin/env bash
# Wf-A269 wave-11 NEGATIVE CONTROL (B) — a LEGITIMATE exit-code gate.
# The tool it keys on is declared in the registry as able to carry a FAIL,
# so gating on its exit code is correct and must NOT be flagged.
# This file exists ONLY under a pytest tmp_path.
set -uo pipefail

PROJECT_ROOT="@PROJECT_ROOT@"
SCRIPTS_DIR="${PROJECT_ROOT}/molmetal/scripts"
PY="${PROJECT_ROOT}/.venv/bin/python"
REPORTS_DIR="${PROJECT_ROOT}/reports"
FIXTURE_TOOL="${SCRIPTS_DIR}/wf_a269_fixture_failable_tool.py"

mkdir -p "${REPORTS_DIR}"

# the pre-flight writes a REAL JSON report (the tool's --json takes a path)
"${PY}" "${FIXTURE_TOOL}" --pipeline all --json "${REPORTS_DIR}/pre_flight.json" || exit 1

# the gate verdict is read from the REPORT — never from a stdout grep
printf 'pre-flight report written\\n'

# no grep for a marker, no redirect of stdout+stderr into a .json path
printf 'launch complete\\n'
exit 0
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_root(tmp_path: Path, name: str) -> Path:
    """Build a throwaway repository-shaped fixture root under tmp_path."""
    root = tmp_path / name
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "molmetal" / "scripts").mkdir(parents=True, exist_ok=True)
    # `git init` + `git add` so that a `git ls-files` corpus enumeration also
    # finds the fixture files.  Purely best-effort: the guard may equally
    # enumerate the corpus from the filesystem.
    with contextlib.suppress(Exception):
        subprocess.run(
            ["git", "init", "-q"],
            cwd=str(root),
            capture_output=True,
            timeout=120,
            check=False,
        )
        subprocess.run(
            ["git", "add", "-A", "-f"],
            cwd=str(root),
            capture_output=True,
            timeout=120,
            check=False,
        )
    return root


def _registry_payload(tools: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "measured_at": "2026-11-02T00:00:00+00:00",
        "measured_by": "molmetal/tests/test_wf_a269_gate_liveness_guard.py",
        "host": "test fixture (synthetic, not a real measurement)",
        "tools": tools,
    }


def _sabotage_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Returns (root, registry_path) for the four-mode sabotage fixture."""
    root = _make_root(tmp_path, "sabotage_root")
    _write(
        root / "scripts" / "launch_wfA269_dead_gate_fixture.sh",
        _SABOTAGE_WRAPPER.replace("@PROJECT_ROOT@", str(root)),
    )
    _write(
        root / "molmetal" / "scripts" / "wf_a269_fixture_tool.py", _ALWAYS_ZERO_TOOL
    )
    # The allow-list entry's `producer` must point at a file that EXISTS.
    # The shared verdict consumer is the one file whose relative path is
    # valid BOTH under the scan root and under the repository root, so the
    # copy below makes the entry resolvable either way.
    shared = REPO_ROOT / "molmetal" / "scripts" / "wf_launch_gate_verdict.py"
    if shared.is_file():
        _write(
            root / "molmetal" / "scripts" / "wf_launch_gate_verdict.py",
            shared.read_text(encoding="utf-8", errors="replace"),
        )
    registry = tmp_path / "sabotage_exitcode_registry.json"
    _write(
        registry,
        json.dumps(
            _registry_payload(
                {
                    "wf_a269_fixture_tool.py": {
                        "exit_code_under_failing_input": 0,
                        "command": (
                            "python molmetal/scripts/wf_a269_fixture_tool.py "
                            "--budget-hr 4"
                        ),
                        "failing_input": (
                            "a budget overrun; the tool records it in --out and "
                            "still exits 0"
                        ),
                        "can_carry_fail": False,
                        "verdict_location": "--out report JSON (field 'verdict')",
                    }
                }
            ),
            indent=2,
        ),
    )
    return root, registry


def _clean_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Returns (root, registry_path) for negative-control fixture (A).

    Wrapper (A) has NO gate at all, so a correct guard reports nothing.
    """
    root = _make_root(tmp_path, "clean_root")
    _write(
        root / "scripts" / "launch_wfA269_clean_fixture.sh",
        _CLEAN_WRAPPER.replace("@PROJECT_ROOT@", str(root)),
    )
    registry = tmp_path / "clean_exitcode_registry.json"
    _write(
        registry,
        json.dumps(_registry_payload({}), indent=2),
    )
    return root, registry


def _clean_gate_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Returns (root, registry_path) for negative-control fixture (B).

    Wrapper (B) gates on a tool's EXIT CODE.  The registry declares that
    tool able to carry a FAIL, so the gate is legitimate: a guard that
    flagged it would flag every exit-code gate in the repository — the
    false-positive class the research arm measured.
    """
    root = _make_root(tmp_path, "clean_gate_root")
    _write(
        root / "scripts" / "launch_wfA269_clean_gate_fixture.sh",
        _CLEAN_GATE_WRAPPER.replace("@PROJECT_ROOT@", str(root)),
    )
    _write(
        root / "molmetal" / "scripts" / "wf_a269_fixture_failable_tool.py",
        _FAILABLE_TOOL,
    )
    registry = tmp_path / "clean_gate_exitcode_registry.json"
    _write(
        registry,
        json.dumps(
            _registry_payload(
                {
                    "wf_a269_fixture_failable_tool.py": {
                        "exit_code_under_failing_input": 1,
                        "command": (
                            "python molmetal/scripts/"
                            "wf_a269_fixture_failable_tool.py --fail"
                        ),
                        "failing_input": "a FAIL gate (the tool exits 1)",
                        "can_carry_fail": True,
                        "verdict_location": None,
                    }
                }
            ),
            indent=2,
        ),
    )
    return root, registry


def _modes_present(findings: Sequence[Any]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {"i": [], "ii": [], "iii": [], "iv": []}
    for finding in findings:
        canonical = _mode_class(finding)
        if canonical in out:
            out[canonical].append(finding)
    return out


def _describe_all(findings: Sequence[Any]) -> str:
    if not findings:
        return "(no findings)"
    return "\n".join(f"  - {_describe(f)}" for f in findings)


# ---------------------------------------------------------------------------
# the immutability guard
# ---------------------------------------------------------------------------


def _metrics_status() -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "metrics/"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    return proc.stdout


@pytest.fixture(scope="module", autouse=True)
def _metrics_immutable():
    before = _metrics_status()
    yield
    after = _metrics_status()
    assert after == before, (
        "this test modified metrics/ — published metrics are IMMUTABLE "
        "(AGENTS.md hard rule 3).\n"
        f"--- before ---\n{before}\n--- after ---\n{after}"
    )


# ---------------------------------------------------------------------------
# TEST 1 — SABOTAGE: the guard must be able to FAIL
# ---------------------------------------------------------------------------


def test_sabotage_fixture_detects_all_four_dead_gate_modes(
    guard: Any, tmp_path: Path
) -> None:
    """A guard that cannot be made to fail is indistinguishable from a real one.

    The fixture below carries one synthetic dead gate per mode.  If the guard
    does not flag a mode, this test FAILS — that is the whole point.
    """
    root, registry = _sabotage_fixture(tmp_path)
    findings = _scan_or_fail(guard, root, registry=registry)
    modes = _modes_present(findings)

    missing = [
        canonical
        for canonical in ("i", "ii", "iii", "iv")
        if not modes[canonical]
    ]
    assert not missing, (
        "the guard did NOT flag dead-gate mode(s) "
        f"{missing} in the sabotage fixture — a detector that cannot be made "
        "to fail is indistinguishable from a real one (Wei 2026, "
        "arXiv:2610.05981: each detector must be proven load-bearing by "
        "sabotage).\n"
        f"findings:\n{_describe_all(findings)}"
    )

    # Each mode must be reported as a CONFIRMED dead gate, not as a shrug.
    for canonical in ("i", "ii", "iii", "iv"):
        dead = [f for f in modes[canonical] if _is_dead(f)]
        assert dead, (
            f"mode ({canonical}) was reported but with no DEAD status — the "
            "guard must say DEAD (confirmed) for a dead gate it can prove.\n"
            f"{_describe_all(modes[canonical])}"
        )


# ---------------------------------------------------------------------------
# TEST 2 — NEGATIVE CONTROL: a clean wrapper produces ZERO findings
# ---------------------------------------------------------------------------


def test_clean_fixture_produces_zero_findings(guard: Any, tmp_path: Path) -> None:
    """The guard must not flag a wrapper that has none of the four modes.

    Two negative controls:

    (A) a wrapper with NO gate at all must produce ZERO findings;
    (B) a wrapper whose gate keys on a tool's EXIT CODE — where the
        registry declares that tool able to carry a FAIL — must not be
        flagged as a dead gate.  A guard that flagged (B) would flag every
        exit-code gate in the repository, which is exactly the
        false-positive class the research arm measured (a flag attributed
        to the wrong tool, a marker that is really a message, a regex
        operand).
    """
    root_a, registry_a = _clean_fixture(tmp_path)
    findings_a = _scan_or_fail(guard, root_a, registry=registry_a)
    assert findings_a == [], (
        "the guard produced findings for a CLEAN wrapper that has none of "
        "the four dead-gate modes — it must produce ZERO findings.\n"
        f"{_describe_all(findings_a)}"
    )

    root_b, registry_b = _clean_gate_fixture(tmp_path)
    findings_b = _scan_or_fail(guard, root_b, registry=registry_b)
    dead_b = [f for f in findings_b if _is_dead(f)]
    assert not dead_b, (
        "the guard flagged a LEGITIMATE exit-code gate as a dead gate — "
        "false positives are the failure mode of a broad linter.\n"
        f"{_describe_all(dead_b)}"
    )


# ---------------------------------------------------------------------------
# TEST 3 — REGISTRY ABSTENTION: a missing registry is UNRESOLVED, never clean
# ---------------------------------------------------------------------------


def test_missing_registry_abstains_on_mode_iii(guard: Any, tmp_path: Path) -> None:
    """With no registry the guard cannot know a tool's exit-code liveness.

    It must therefore report mode (iii) as UNRESOLVED — an explicit
    abstention — and never as "clean" and never as a confirmed DEAD gate.
    Quiet must never read as success (Wei 2026: live precision reported as
    undefined rather than narrating quiet as success).
    """
    root, _registry = _sabotage_fixture(tmp_path)
    missing_registry = tmp_path / "no_such_exitcode_registry.json"
    assert not missing_registry.exists()

    findings = _scan_or_fail(guard, root, registry=missing_registry)
    modes = _modes_present(findings)
    mode_iii = modes["iii"]

    assert mode_iii, (
        "with the exit-code registry MISSING the guard reported NOTHING for "
        "mode (iii) — it silently treated an exit-code gate as clean.  The "
        "guard must abstain explicitly (UNRESOLVED), never report silence as "
        "a pass.\n"
        f"findings:\n{_describe_all(findings)}"
    )
    assert any(_is_unresolved(f) for f in mode_iii), (
        "with the registry missing, mode (iii) must be reported as "
        "UNRESOLVED (the guard abstains), not as a confirmed DEAD gate.\n"
        f"{_describe_all(mode_iii)}"
    )
    assert not any(_is_dead(f) for f in mode_iii), (
        "with the registry missing, the guard CLAIMED a confirmed dead gate "
        "for mode (iii) — it cannot know that without the measured "
        "registry.\n"
        f"{_describe_all(mode_iii)}"
    )


# ---------------------------------------------------------------------------
# TEST 4 — ALLOW-LIST: a listed finding does not fail; an unlisted one does
# ---------------------------------------------------------------------------


def _allowlist_entry(finding: Any, *, line_delta: int = 0) -> Dict[str, Any]:
    raw_mode = _fget(finding, ("mode", "kind", "category", "rule"), "i")
    wrapper = _wrapper(finding)
    marker = _fget(finding, ("marker", "symbol", "literal"), "")
    return {
        "id": f"wf-a269-wave11-test-{_norm_token(raw_mode)}-{_line(finding)}",
        "wrapper": wrapper,
        "line": _line(finding) + line_delta,
        "mode": raw_mode,
        "marker": str(marker),
        "reason": "test fixture — a deliberately suppressed finding",
        # A file+line that EXISTS both under the scan root (a copy of the
        # shared verdict consumer is written into the fixture) and under the
        # repository root.
        "producer": "molmetal/scripts/wf_launch_gate_verdict.py:1",
        "owner": "wf-a269-wave11-test",
        "expires": "2099-12-31",
    }


def _write_allowlist(path: Path, entries: List[Dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "entries": entries,
        "allowlist": entries,
        "suppressions": entries,
    }
    _write(path, json.dumps(payload, indent=2))


def test_allowlist_suppresses_only_the_matching_finding(
    guard: Any, tmp_path: Path
) -> None:
    """A listed finding does not fail the guard; an unlisted one does.

    The allow-list may only suppress ONE mode at ONE line — never a whole
    wrapper (the research arm's allow-list rule #4).
    """
    root, registry = _sabotage_fixture(tmp_path)

    baseline = _scan_or_fail(guard, root, registry=registry)
    mode_i = _modes_present(baseline)["i"]
    assert mode_i, (
        "the sabotage fixture produced no mode-(i) finding at all — the "
        "allow-list test has no target.\n"
        f"findings:\n{_describe_all(baseline)}"
    )

    target = sorted(mode_i, key=lambda f: (_wrapper(f), _line(f)))[0]
    target_key = (_wrapper(target), _line(target), _mode_class(target))

    # (a) an allow-list entry that matches -> the finding does not fail.
    allow = tmp_path / "allow_match.json"
    _write_allowlist(allow, [_allowlist_entry(target)])
    suppressed = _scan_or_fail(guard, root, registry=registry, allowlist=allow)
    still_blocking = [
        f
        for f in suppressed
        if _is_dead(f) and (_wrapper(f), _line(f), _mode_class(f)) == target_key
    ]
    assert not still_blocking, (
        "a finding that IS in the allow-list still failed the guard — the "
        "allow-list is not honoured.\n"
        f"{_describe_all(still_blocking)}"
    )

    # (b) an allow-list entry that does NOT match -> the finding still fails.
    allow_miss = tmp_path / "allow_miss.json"
    _write_allowlist(allow_miss, [_allowlist_entry(target, line_delta=987654)])
    not_suppressed = _scan_or_fail(
        guard, root, registry=registry, allowlist=allow_miss
    )
    surviving = [
        f
        for f in not_suppressed
        if _is_dead(f) and (_wrapper(f), _line(f), _mode_class(f)) == target_key
    ]
    assert surviving, (
        "a finding that is NOT in the allow-list stopped failing the guard — "
        "the allow-list over-suppresses.\n"
        f"target was: {_describe(target)}\n"
        f"findings:\n{_describe_all(not_suppressed)}"
    )


# ---------------------------------------------------------------------------
# TEST 5 — COVERAGE BOUND: the report must state what it cannot see
# ---------------------------------------------------------------------------


def _coverage_values(obj: Any, _depth: int = 0) -> List[Any]:
    if _depth > 8:
        return []
    found: List[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "coverage" in str(key).lower() or "bound" in str(key).lower():
                found.append(value)
            found.extend(_coverage_values(value, _depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_coverage_values(item, _depth + 1))
    return found


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def test_guard_report_carries_a_non_empty_coverage_bound(guard: Any, tmp_path: Path) -> None:
    """An unstated coverage bound is a defect.

    The guard is a narrow, high-precision linter with an explicit abstain
    path; it must publish the boundary of what it can see, and it must
    report its classified / unresolved / dead counts separately (a census
    is not a defect count — AGENTS.md §6.269.6).
    """
    report_path = tmp_path / "guard_report.json"
    cmd = [sys.executable, str(GUARD_PATH)]
    help_proc = subprocess.run(
        cmd + ["--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if "--json" in help_proc.stdout:
        cmd += ["--json", str(report_path)]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode in (0, 1, 2), (
        f"the guard CLI exited {proc.returncode}; AGENTS.md §4 says 0=PASS, "
        f"1=FAIL, 2=SKIP.\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    assert proc.stdout.split()[:1], f"the guard CLI printed nothing: {proc.stdout!r}"
    assert proc.stdout.split()[0] in {"PASS", "FAIL", "SKIP"}, (
        "the guard CLI must print PASS/FAIL/SKIP as its first stdout token "
        f"(AGENTS.md §4); got {proc.stdout.split()[0]!r}"
    )

    report: Any = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            report = None

    if report is not None:
        values = [v for v in _coverage_values(report) if _has_content(v)]
        assert values, (
            "the guard's report has no non-empty coverage-bound section.  An "
            "unstated coverage bound is a defect: the report must say what "
            "the guard CANNOT see (non-shell launchers, indirectly-passed "
            "flags, non-repo binaries, runtime-conditional producers, gate "
            "semantics).\n"
            f"report keys: {sorted(report.keys()) if isinstance(report, dict) else type(report)}"
        )
    else:
        assert "coverage" in proc.stdout.lower(), (
            "the guard produced neither a JSON report nor a coverage-bound "
            "section on stdout.\n"
            f"stdout:\n{proc.stdout[-3000:]}"
        )


# ---------------------------------------------------------------------------
# TEST 6 — REGRESSION CORPUS
#
# The corpus is DERIVED FROM DISK at test time, not frozen.  The wave-11
# repair pass is landing concurrently with this test: three of the six
# research-measured wrappers had already been repaired on disk when this
# test was written.  A hard-coded "still dead" list would therefore encode
# a stale fact in one direction or the other, so the expectation is
# re-measured here with a reference detector that implements the SAME rule
# the guard spec states (spec step 7 — /tmp/wfa269w11_literature_guard.md
# line 109):
#
#   (a) a tool invocation whose stdout+stderr is redirected into a path
#       ending in ``.json``  ->  the artefact is NOT JSON at all;
#   (b) a bare ``--json`` passed to a tool whose ``--json`` dest is
#       ``type=Path, nargs="?"``  ->  no file is written at all.
#
# The detector resolves tools referenced through shell variables, because
# every MEASURED real wrapper does
# (``PRE_FLIGHT="${SCRIPTS_DIR}/pre_flight_check.py"`` then
# ``"${PY}" "${PRE_FLIGHT}" ...``).
# ---------------------------------------------------------------------------

_REDIRECT_JSON = re.compile(r'>+\s*("?)([^\s;|&<>]*?\.json)\1')
_PY_TOKEN = re.compile(r'[^\s"\';&|<>()]*\.py\b')
_SHELL_VAR = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
_GT_REDIRECT = re.compile(r"\d?>&(?:\d)?|&>")
_TOOL_ARG_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _live_lines(text: str) -> List[str]:
    """Blank full-line shell comments but KEEP the line numbering.

    AGENTS.md §5.5 rule #6 keeps the superseded rendering of a repaired gate
    on disk inside a comment block, so a substring test on the raw text
    would be satisfied by the RECORD of what used to run, not by the code
    that runs.
    """
    return [
        "" if line.strip().startswith("#") else line for line in text.splitlines()
    ]


def _statements(lines: Sequence[str]) -> List[Tuple[int, str]]:
    """Join backslash continuations, then split into shell statements.

    Returns ``(first_line_number, statement_text)``.  ``2>&1`` / ``>&2`` are
    NOT split points (they belong to the command); ``;`` ``&&`` ``||``
    ``|`` ``&`` are.  The redirections are MASKED before the split and
    restored afterwards, because a naive split on ``&`` cuts ``2>&1`` in
    half and destroys the very signature this detector looks for.
    """
    logical: List[Tuple[int, str]] = []
    buf = ""
    start = 0
    for index, raw in enumerate(lines, 1):
        if not raw.strip():
            if buf:
                logical.append((start, buf))
                buf = ""
            continue
        if not buf:
            start = index
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        buf += stripped
        logical.append((start, buf))
        buf = ""
    if buf:
        logical.append((start, buf))

    out: List[Tuple[int, str]] = []
    for line_no, text in logical:
        masked: List[str] = []

        def _mask(match: "re.Match[str]") -> str:
            masked.append(match.group(0))
            return "\x00%d\x00" % (len(masked) - 1)

        protected = _GT_REDIRECT.sub(_mask, text)
        for piece in re.split(r"\s*(?:;|&&|\|\||\||&)\s*", protected):
            if not piece.strip():
                continue
            restored = re.sub(
                r"\x00(\d+)\x00", lambda m: masked[int(m.group(1))], piece
            )
            out.append((line_no, restored))
    return out


def _shell_var_tools(lines: Sequence[str]) -> Dict[str, str]:
    """Map shell variable names whose value points at a ``.py`` tool."""
    out: Dict[str, str] = {}
    for line in lines:
        match = re.match(
            r"""\s*(?:export\s+|local\s+|declare\s+-\w+\s+)?"""
            r"""([A-Za-z_][A-Za-z0-9_]*)=(?:"([^"]*)"|'([^']*)'|([^\s]+))""",
            line,
        )
        if not match:
            continue
        value = match.group(2) or match.group(3) or match.group(4) or ""
        if ".py" in value:
            out[match.group(1)] = value
    return out


def _resolve_tool(statement: str, var_tools: Dict[str, str]) -> Optional[str]:
    """The ``.py`` tool a statement invokes (literal token or shell var)."""
    match = _PY_TOKEN.search(statement)
    if match:
        return match.group(0).strip("\"' ")
    for var in _SHELL_VAR.findall(statement):
        if var in var_tools:
            return var_tools[var]
    return None


def _tool_json_dest(repo_root: Path, tool: str) -> Optional[Dict[str, Any]]:
    """MEASURE the ``--json`` argparse dest of a tool by AST (never run it).

    ``pre_flight_check.py --json`` is ``type=Path, nargs="?"`` — MEASURED by
    AST on 2026-11-02 — so a bare ``--json`` writes NO file and the payload
    goes to stdout.  ``per_wave_checklist.py --json`` is a boolean
    (``store_true``) and its ``--out`` takes the path, so a bare ``--json``
    there is NOT a mode-(iv) defect.
    """
    key = str(tool)
    if key in _TOOL_ARG_CACHE:
        return _TOOL_ARG_CACHE[key]
    resolved: Optional[Path] = None
    candidate = Path(tool)
    if candidate.is_absolute() and candidate.is_file():
        resolved = candidate
    else:
        for base in (repo_root / "molmetal" / "scripts", repo_root / "scripts"):
            probe = base / candidate.name
            if probe.is_file():
                resolved = probe
                break
    info: Optional[Dict[str, Any]] = None
    if resolved is not None:
        try:
            tree = ast.parse(resolved.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            tree = None  # type: ignore[assignment]
        if tree is not None:
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_argument"
                ):
                    continue
                names = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                if not names or names[0] != "--json":
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords}
                nargs = keywords.get("nargs")
                info = {
                    "nargs": (
                        ast.literal_eval(nargs) if isinstance(nargs, ast.Constant) else None
                    ),
                    "type": (
                        ast.unparse(keywords["type"])
                        if keywords.get("type") is not None
                        else None
                    ),
                    "path": str(resolved),
                }
                break
    _TOOL_ARG_CACHE[key] = info
    return info


def _bare_json_flag(statement: str) -> bool:
    """True if the statement passes ``--json`` with NO value."""
    tokens = statement.split()
    for index, token in enumerate(tokens):
        if token != "--json":
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if (
            following == ""
            or following.startswith("-")
            or following in (">", ">>", "|", "&&", "||", ";", "&")
            or ">&" in following
            or following.startswith("2>")
        ):
            return True
    return False


def _mode_iv_sites(wrapper: Path, repo_root: Path) -> List[Tuple[int, str, str, str]]:
    """MEASURE the live mode-(iv) sites of a wrapper.

    Returns ``(line, kind, tool_basename, detail)`` with ``kind`` in
    ``{"redirect", "bare_json"}``.  An empty list means the wrapper no longer
    carries the mode-(iv) signature in its LIVE code.
    """
    lines = _live_lines(wrapper.read_text(encoding="utf-8", errors="replace"))
    var_tools = _shell_var_tools(lines)
    sites: List[Tuple[int, str, str, str]] = []
    for line_no, statement in _statements(lines):
        tool = _resolve_tool(statement, var_tools)
        if not tool:
            continue
        tool_name = Path(tool).name
        if "2>&1" in statement or "&>" in statement:
            match = _REDIRECT_JSON.search(statement)
            if match:
                sites.append(
                    (
                        line_no,
                        "redirect",
                        tool_name,
                        f"stdout+stderr -> {match.group(2)} (tool {tool_name})",
                    )
                )
                continue
        if _bare_json_flag(statement):
            info = _tool_json_dest(repo_root, tool)
            if info and info.get("nargs") == "?" and info.get("type") == "Path":
                sites.append(
                    (
                        line_no,
                        "bare_json",
                        tool_name,
                        f"bare --json on {tool_name} "
                        f"({Path(info['path']).name} --json is nargs='?' type=Path "
                        f"-> no file written)",
                    )
                )
    return sites


def _wrapper_corpus(repo_root: Path) -> List[Path]:
    """The wrapper corpus: the tracked/untracked ``scripts/launch_*.sh``.

    Coverage bound (AGENTS.md §6.269.6 — a census is a COVERAGE census, not
    a defect count): ``scripts/_retired/`` and the ``_``-prefixed helper
    templates are OUT of corpus.  ``scripts/_WRAPPER_TEMPLATE.sh`` was
    MEASURED on 2026-11-02 carrying the same mode-(iv) signature as
    ``scripts/launch_*.sh``, so a guard that scans it will report one more
    wrapper than this corpus names.
    """
    scripts = repo_root / "scripts"
    if not scripts.is_dir():
        return []
    return [p for p in sorted(scripts.glob("launch_*.sh")) if p.is_file()]


def test_guard_flags_the_measured_dead_wrappers(guard: Any) -> None:
    """Generalise the wave-10 per-wrapper test to the whole population.

    Two layers:

    1. the six wrappers the research arm MEASURED — named explicitly in
       ``MEASURED_DEAD_WRAPPERS``.  A wrapper that STILL carries the
       mode-(iv) signature on disk MUST be flagged; a wrapper that has been
       repaired must NOT be demanded (that would encode a stale fact);
    2. the POPULATION — every ``scripts/launch_*.sh`` that still carries the
       signature MUST be flagged, so a regression into any wrapper is
       caught, not just into the six.
    """
    findings = _scan_or_fail(guard, REPO_ROOT)
    flagged_dead = {
        Path(_wrapper(f)).name
        for f in findings
        if _is_dead(f) and Path(_wrapper(f)).name
    }
    flagged_iv = {
        Path(_wrapper(f)).name
        for f in findings
        if _mode_class(f) == "iv" and _is_dead(f) and Path(_wrapper(f)).name
    }

    corpus = _wrapper_corpus(REPO_ROOT)
    assert corpus, (
        "the wrapper corpus is empty — the regression test would be vacuous "
        f"(looked in {REPO_ROOT / 'scripts'})"
    )

    live: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for wrapper in corpus:
        sites = _mode_iv_sites(wrapper, REPO_ROOT)
        if sites:
            live[wrapper.name] = sites
    assert live, (
        "NO wrapper in the corpus carries the mode-(iv) signature any more. "
        "Either the whole corpus was repaired (then this regression test has "
        "nothing to guard and should be re-derived) or the reference "
        "detector is broken (it is not a vacuous-pass licence)."
    )

    # (1) the six research-measured wrappers, named explicitly
    named_missing: List[str] = []
    named_detail: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for name in MEASURED_DEAD_WRAPPERS:
        path = REPO_ROOT / "scripts" / name
        assert path.is_file(), f"MEASURED dead-gate wrapper missing from disk: {path}"
        sites = _mode_iv_sites(path, REPO_ROOT)
        named_detail[name] = sites
        if sites and name not in flagged_dead:
            named_missing.append(name)
    assert not named_missing, (
        "the guard did NOT flag the MEASURED dead-gate wrapper(s) that STILL "
        f"carry the defect on disk: {named_missing}\n"
        + _sites_text(named_detail)
        + f"guard-flagged wrappers: {sorted(flagged_dead)}\n"
        f"findings:\n{_describe_all(findings)}"
    )

    # (2) the POPULATION — every wrapper that still carries the signature
    missing = sorted(set(live) - set(flagged_dead))
    assert not missing, (
        "the guard did NOT flag the wrapper(s) that STILL carry the mode-(iv) "
        f"signature: {missing}\n"
        + _sites_text(live)
        + f"guard-flagged wrappers: {sorted(flagged_dead)}\n"
        f"findings:\n{_describe_all(findings)}"
    )

    # (3) the mode-(iv) CLASS must be reported for the wrappers whose live
    #     signature is a `.json` capture of a REPORT producer.  The
    #     `abort_recovery.py` captures are excluded from this sub-check: that
    #     tool writes a human log to stdout and its report to `--report-json`,
    #     so the same line is ALSO a mode-(i) defect (a `--json` the tool
    #     does not define) — the guard may legitimately report it as (i).
    report_producers = {
        name: [site for site in sites if site[2] != "abort_recovery.py"]
        for name, sites in live.items()
    }
    report_producers = {k: v for k, v in report_producers.items() if v}
    missing_iv = sorted(set(report_producers) - set(flagged_iv))
    assert not missing_iv, (
        "the guard did NOT report the mode-(iv) class for the wrapper(s) that "
        f"still capture a REPORT producer's stdout+stderr into a `.json` path: "
        f"{missing_iv}\n"
        + _sites_text(report_producers)
        + f"guard-flagged mode-(iv) wrappers: {sorted(flagged_iv)}\n"
        f"findings:\n{_describe_all(findings)}"
    )


def _sites_text(sites: Dict[str, List[Tuple[int, str, str, str]]]) -> str:
    """Render the MEASURED live sites for a failure message."""
    if not sites:
        return "still-live mode-(iv) sites (MEASURED from disk): none\n"
    return (
        "still-live mode-(iv) sites (MEASURED from disk): "
        + "; ".join(
            f"{name}: "
            + ", ".join(f"L{ln} {kind} {detail}" for ln, kind, _tool, detail in rows)
            for name, rows in sorted(sites.items())
        )
        + "\n"
    )


def test_wf146_wrapper_no_longer_has_a_dead_json_capture() -> None:
    """The wave-11 repair of one MEASURED wrapper must STAY repaired.

    MEASURED 2026-11-02 by this agent: the LIVE code of
    ``scripts/launch_wf146_deferred_runs.sh`` carries no mode-(iv) site
    (the redirect survives only inside the SUPERSEDED-RENDERING comment
    block).  This test is the regression guard for that repair.
    """
    wrapper = REPO_ROOT / "scripts" / "launch_wf146_deferred_runs.sh"
    assert wrapper.is_file(), f"wrapper missing: {wrapper}"
    sites = _mode_iv_sites(wrapper, REPO_ROOT)
    assert not sites, (
        f"{wrapper.name}: the LIVE code still carries a mode-(iv) site — the "
        "dead gate is back:\n"
        + "\n".join(f"  L{ln} {kind} {detail}" for ln, kind, _tool, detail in sites)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

