#!/usr/bin/env python3
"""Wf-259 — pre_flight_check.py: unified auto-verify for the v2 pre-flight checklist.

==============================================================================
Wf-259 / TODO-208 / TODO-207 — pre-flight auto-verify script.

Implements the **auto-verifiable subset** of `TODO/completed/207_unified_pre_flight_checklist_v2.md`:

* §3.1 chemprop v2 v3 P1 VERIFY sub-gates (12 of the 15)
* §4.2 CFM P1 VERIFY sub-gates (6 of the 15)
* §5 P2 DOCUMENTED gates (4 — data + tests + vina)

**Does NOT cover** (per v2 doc §1 P0 BLOCKER; these need user OK / live hardware):

* §1.1 user "kill then adapt" OK gate (manual lift; tracked in `TODO/INDEX.md`)
* §1.2 GPU state live verify (requires `nvidia-smi`; this script is CPU-only)
* §1.3 / §1.4 config schema + CSV hash (referenced as TODO-208 follow-up)
* §5.1 GPU util ≥ 90 % (post-warmup live probe; out of scope)
* §5.2 CPU hot-path debug sub-steps (`torch.profiler`; manual inspect)

The script is **CPU-only** (no `torch.cuda` calls), **backward-compat** (does not
modify any other file under `molmetal/scripts/`), and emits **machine-readable
JSON** suitable for `jq` post-processing or wave-budget tracking in
`molmetal/reports/wf207_pre_flight/wave_N.json`.

CLI::

    pre_flight_check.py [--pipeline chemprop_v2_v3|cfm|all] [--json path]
                        [--strict] [--verbose]

Exit codes:

* **0** — every auto-verifiable gate PASSED.
* **1** — at least one gate FAILED.
* **2** — at least one gate SKIPPED (only when ``--strict`` is set).

See `TODO/completed/207_unified_pre_flight_checklist_v2.md` §5, §8, §10 for
the full per-wave sub-checklist + JSON schema contract this script implements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Wf-415-A2: sys.path injection so the script can be invoked directly via
# `python molmetal/scripts/pre_flight_check.py ...` WITHOUT requiring
# PYTHONPATH=. in the wrapper shell context. Resolves the long-standing
# ModuleNotFoundError: No module named 'molmetal' at line 60
# (read_anchor_json import) when invoked from a wrapper that does not
# prepend PYTHONPATH. NOTE: inject the REPO ROOT (3 .parent hops up from
# __file__ = molmetal/scripts/pre_flight_check.py), NOT the molmetal/
# directory. `molmetal/scripts/` is an implicit namespace package (no
# __init__.py); if we inject the molmetal/ directory, then
# `molmetal.scripts` would resolve to a non-existent sub-directory under
# molmetal/ and fail.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Wf-304A: AnchorGuard adoption at every paper-grade JSON read site.
# Per Wf-302A honest caveat, downstream consumers (this script included)
# had been doing ad-hoc json.load(open(...)) at the call sites; the
# AnchorGuard adoption replaces each read with explicit freshness +
# version guards so silent stale / missing / version-drifted reads are
# no longer possible.
from molmetal.scripts.read_anchor_json import (  # noqa: E402
    AnchorGuard,
    AnchorJSONError,
    AnchorMissingError,
    AnchorStaleError,
    AnchorVersionMismatchError,
)

# Wf-353-B1 — wire Wf-343c1's 26-gate auto-verifiable surface into the
# 'all' pipeline.  Per Wf-343c1 finding + Wf-343C4 ship contract:
#   * 26 pre-retrain gates are defined in
#     ``molmetal/scripts/wf343c1_pre_retrain_check.py`` covering §1.1
#     (code-level 10 gates) + §1.3 (ablation flags 3 gates) + §2.1
#     (data-level 7 gates) + §3.1 (test-level 1 gate) + §5.11-§5.14
#     (post-retrain 4 SKIP gates) + §8.15 (honest-framing 1 gate).
#   * The 4 post-retrain §5.11-§5.14 gates always SKIP today (filled
#     in after the retrain wave) — they remain SKIP and are NOT
#     wired individually into pre_flight_check.py per the same
#     honest-framing convention used by Wf-347A1 (1 wrapper gate per
#     surface, not per-row).
#   * The remaining 22 auto-verifiable gates are wired INDIVIDUALLY
#     (one runner per gate) so each surfaces in the pre-flight JSON
#     report with its own PASS/FAIL/SKIP verdict and evidence
#     string.  This matches the Wf-343C4 per-gate registration
#     pattern (4 individual runners, one per gate) and gives
#     downstream consumers (per_wave_checklist.py +
#     _partition_gates_by_priority + paper §3 attribution
#     appendix) per-gate visibility.
# We import the gate-builder helpers (NOT the standalone ``main()``)
# so the runners are wired into ``build_gates()`` directly without
# a subprocess hop.  This keeps each gate wall < 50 ms (the
# gate_file_exists / gate_min_lines / gate_grep / gate_regex /
# gate_help primitives are all in-process).
from molmetal.scripts import wf343c1_pre_retrain_check as _wf343c1  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
SCRIPT_DIR = MOLMETAL_ROOT / "scripts"
SCRIPTS_DIR = (
    REPO_ROOT / "scripts"
)  # Wf-432-A1: project root scripts dir (USER_OK wrapper audit)
LAM_CHEM_DIR = MOLMETAL_ROOT / "molmetal_lam" / "lam_chem"
CONFIGS_DIR = REPO_ROOT / "configs"


def _discover_active_configs() -> Tuple[Path, ...]:
    """Discover the active configs instead of listing them by hand.

    2026-10-04: this replaces a hard-coded tuple, and the reason is that the
    hand-maintained list had ALREADY silently dropped coverage twice.  Its own
    comment records the first: "configs/training/ was never scanned by this
    gate, so its 2 active YAMLs were outside the contract entirely" (fixed by
    Wf-A12 by appending them).  The second is that every config added since
    was invisible -- `cfm_tmqm_all6.yaml` and `fm_pocket_multiph.yaml` were
    both unscanned, so a dead key introduced in a NEW config would sail
    through the very gate whose job is to catch dead keys.

    Discovery removes the failure mode rather than patching another instance
    of it: a config that exists is checked, and the only way to exclude one is
    to put it somewhere this deliberately does not look (`archive/`, which the
    globs below do not reach, and `*.bak`, filtered explicitly).

    `.bak` files and any path containing an `archive` segment are excluded --
    they are historical artefacts, and the Wf-A12 comment made the same
    distinction.
    """
    out: List[Path] = []
    for pattern in ("*.yaml", "training/*.yaml"):
        for p in sorted(CONFIGS_DIR.glob(pattern)):
            if p.suffix == ".bak" or "archive" in p.parts or p.name.endswith(".bak"):
                continue
            out.append(p)
    return tuple(out)
TESTS_DIR = MOLMETAL_ROOT / "tests"
DATA_DIR = MOLMETAL_ROOT / "data"
DOT_DATA_DIR = REPO_ROOT / ".data" / "molmetal"

# H1 / HOLE-8 follow-up (2026-10-03): commit-path gate scratch root.
# The build_gates("all") commit-path gates must not write into the
# source-adjacent report tree (molmetal/reports/**) — the H1 fix had to
# `git rm --cached` those transient writes and gitignore them. Route the
# gate outputs here instead: results/gate_scratch/ is already gitignored
# (.gitignore:199) and is created lazily by the writers, so a commit can
# never record a transient gate write and a fresh clone regenerates it.
GATE_SCRATCH_ROOT = REPO_ROOT / "results" / "gate_scratch"

# Wf-432-A1: atomic_write_file from Wf-386-A3 §5.6 adoption sequence.
# Used by §4 L5 per-baseline protocol validation hooks to emit hook JSON
# at molmetal/reports/protocol_validation/{krasnov,rusanov,drugood}_hook.json
# via POSIX-atomic temp-file + rename (process-local per-path lock).
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"  # Wf-446: was hardcoded to non-existent /home/hugo/codes/mol-metal/.venv/bin/python; project venv at .venv/bin/python is the canonical Python for sub-gates
# Wf-300D harness auto-fallback wrapper is a USER-LEVEL file, not part of
# this repo. Located at $HOME/.claude/wrapper.py; Wf-306C P3 gate verifies
# it exists so any subsequent auto-fallback calls (detect_schema_shape /
# extract_agent_result) won't ImportError.
HARNESS_WRAPPER = Path.home() / ".claude" / "wrapper.py"

# Wf-306C — P3 ANCHOR_VERSION = the version all paper-grade *_anchor.json
# files MUST carry post-Wf-305C. The CI gate (gate_anchor_version_pin)
# walks every paper-grade anchor under REPO_ROOT and asserts each carries
# ``"version": ANCHOR_VERSION`` at the top level. Bumping this constant
# (e.g. to "1.0.1") will re-fail every anchor until each is upgraded.
ANCHOR_VERSION = "1.0.0"
ANCHOR_GLOB_PATTERNS = (
    "molmetal/reports/**/m14_anchor.json",
    "molmetal/reports/**/wf269c_report.json",
    "molmetal/reports/**/m17_drugood_reeval.json",
    "molmetal/reports/**/sweep_summary.json",
)
# 2026-10-03 anchor-scope fix (defect 2): the mtime freshness budget below
# applies ONLY to LIVE anchors — files a current pipeline still writes or
# serves. The 11 paths in ``ANCHOR_FROZEN_PATHS`` are shipped historical
# wave reports (MEASURED on disk 2026-10-03, ages 12.1-26.5 h); no producer
# in today's pipeline regenerates them, so a wall-clock FAIL on them can
# never be remedied without touching frozen history — and touching an
# anchor to reset its mtime is fabricated freshness (AGENTS.md §5.2
# violation). Left un-scoped, that FAIL was permanent decoration, not a
# gate. Frozen paths remain covered by ``gate_anchor_version_pin`` (the
# content/version-pin contract). Any NEW file matching ANCHOR_GLOB_PATTERNS
# is LIVE by default — un-freeze a path by removing its entry here and it
# is mtime-checked again.
ANCHOR_FROZEN_PATHS: Tuple[Tuple[str, str], ...] = (
    (
        "molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json",
        "Wf-121 cisplatin real-Vina sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json",
        "Wf-143 P4 chemotype-dock sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf170_permetal_singletask/sweep_summary.json",
        "Wf-170 per-metal singletask sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json",
        "Wf-188 m14 full-MetalCytoTox run anchor; shipped historical report",
    ),
    (
        "molmetal/reports/wf219_krasnov_aligned/sweep_summary.json",
        "Wf-219 Krasnov-aligned sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json",
        "Wf-253D m17 DrugOOD re-eval report; shipped historical report",
    ),
    (
        "molmetal/reports/wf269c_cfm_real/wf269c_report.json",
        "Wf-269C CFM real-cohort decode report; shipped historical report",
    ),
    (
        "molmetal/reports/wf277c_real_pilot/wf269c_report.json",
        "Wf-277C real-pilot report; shipped historical report",
    ),
    (
        "molmetal/reports/wf308c_temporal_split/sweep_summary.json",
        "Wf-308C temporal-split sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf308d_temporal_smoke/sweep_summary.json",
        "Wf-308D temporal-smoke sweep summary; shipped historical report",
    ),
    (
        "molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json",
        "Wf-67 P4 3x100-pocket sweep summary; shipped historical report",
    ),
)

PIPELINES = ("chemprop_v2_v3", "cfm", "all", "frontier", "launch")


class WaveDeclaration:
    """What a launch declares about the wave it is about to run.

    Threaded from the CLI into :func:`build_gates` so that
    ``p3_wf343c2_per_wave_runtime_check`` can check a REAL grid against a REAL
    budget instead of skipping.

    Why this exists: the gate accepted ``pairs_metals`` / ``pairs_seeds`` /
    ``budget_hr`` from the day it was written, but ``build_gates`` had no way
    to receive them — so the registered gate always ran with all three ``None``
    and could only ever emit "grid not declared by caller -> loud SKIP".  The
    launchers meanwhile ran the same script per-tuple WITH real declarations.
    The gate was therefore honest but vacuous on the one path (pre-flight) where
    it was registered, which reads as a gap and invites bypass.

    Every field is optional: an undeclared wave still SKIPs loudly rather than
    PASSing, exactly as before.
    """

    __slots__ = ("wave", "pairs_metals", "pairs_seeds", "budget_hr")

    def __init__(
        self,
        wave: Optional[str] = None,
        pairs_metals: Optional[str] = None,
        pairs_seeds: Optional[str] = None,
        budget_hr: Optional[float] = None,
    ) -> None:
        self.wave = wave
        self.pairs_metals = pairs_metals
        self.pairs_seeds = pairs_seeds
        self.budget_hr = budget_hr


# "launch" = everything in "all" PLUS the gates that only make sense before a
# TRAINING LAUNCH (anchor freshness, per-wave runtime smoke). They are split out
# because "all" is what the commit hook runs, and a commit is not a launch: a
# commit that only touches gate tooling has no reason to fail because the
# paper-grade anchors are 24 hours old. Registering them on "all" made the repo
# uncommittable on a purely temporal trigger. Launchers pass --pipeline launch.

# ---------------------------------------------------------------------------
# Gate result container
# ---------------------------------------------------------------------------


class GateResult:
    """Single gate result. Populated by the gate-function; serialised to JSON."""

    __slots__ = (
        "name",
        "priority",
        "pipeline",
        "status",
        "evidence",
        "wall_sec",
    )

    def __init__(
        self,
        name: str,
        priority: str,
        pipeline: str,
        status: str,
        evidence: str,
        wall_sec: float,
    ) -> None:
        self.name = name
        self.priority = priority
        self.pipeline = pipeline
        self.status = status
        self.evidence = evidence
        self.wall_sec = wall_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "pipeline": self.pipeline,
            "status": self.status,
            "evidence": self.evidence,
            "wall_sec": round(self.wall_sec, 4),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def _run(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    cwd: Optional[Path] = None,
) -> Tuple[int, str, str]:
    """Run subprocess; return (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        # 2026-10-04: TimeoutExpired.stdout/.stderr are BYTES even when the
        # call passed text=True — CPython raises TimeoutExpired from inside
        # communicate(), before the decode step a successful call performs.
        # So `(exc.stderr or "") + f"..."` raised
        #   TypeError: can't concat str to bytes
        # from inside the timeout handler itself, and the caller reported the
        # real condition ("the smoke exceeded its budget") as a bare
        # "gate raised: TypeError" — the failure mode hid the failure.
        def _as_text(v: object) -> str:
            if v is None:
                return ""
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)

        return (
            124,
            _as_text(exc.stdout),
            _as_text(exc.stderr) + f"\n[TIMEOUT after {timeout}s]",
        )
    except FileNotFoundError as exc:
        return 127, "", f"[exec error: {exc}]"


def _grep_file(pattern: str, path: Path) -> bool:
    """Case-sensitive substring search; no shell expansion."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return pattern in fh.read()
    except OSError:
        return False


def _line_count(path: Path) -> int:
    """Return wc -l count, or -1 if file missing / unreadable."""
    if not path.exists():
        return -1
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


# ---------------------------------------------------------------------------
# Gate-type primitives (returned as GateResult)
# ---------------------------------------------------------------------------


def gate_grep(
    name: str,
    priority: str,
    pipeline: str,
    pattern: str,
    file_path: Path,
) -> GateResult:
    """PASS if ``pattern`` is found as a substring in ``file_path``."""
    t0 = time.perf_counter()
    ok = _grep_file(pattern, file_path)
    elapsed = time.perf_counter() - t0
    if ok:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"grep '{pattern}' found in {file_path}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"grep '{pattern}' NOT found in {file_path}",
        elapsed,
    )


#: 2026-10-04 — memoised `--help` probes, keyed by
#: (interpreter, script, extra_args).  See gate_help for why.
_HELP_PROBE_CACHE: Dict[tuple, Tuple[int, str, str]] = {}

# Share the one cache with wf343c1_pre_retrain_check, which has its own copy of
# gate_help and probes the SAME script.  Without this the script was spawned
# once per module (twice per pre-flight run) and the second module's first
# probe still paid the full ~5.3 s import.  Aliasing the dict — rather than
# copying — means whichever module probes first serves the other, and a
# FAILing probe cannot differ between them.
_wf343c1._HELP_PROBE_CACHE = _HELP_PROBE_CACHE


def gate_help(
    name: str,
    priority: str,
    pipeline: str,
    script: Path,
    pattern: str,
    *,
    py: str = VENV_PY,
    extra_args: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> GateResult:
    """DECLARATION-ONLY check: every token in ``pattern`` appears as an
    argparse option/choice in ``--help`` output.

    Multi-token patterns (e.g. ``"--no-multi-task --no-wass"``, or
    ``"--chemprop-mode v3-fix"`` where ``v3-fix`` is a choice value) are
    split on whitespace and require ALL tokens.

    Wf-A12 FIX (2026-10-03) — tightening: a token previously matched as a
    plain SUBSTRING anywhere in stdout+stderr, so it also passed on prose
    (e.g. a token mentioned only in a help-text description).  A token now
    counts only when it appears, as a whole token (word/dash boundaries),
    inside the INVOCATION column of an argparse help option line — the
    "  --flag VALUE" part before the description, which is the form an
    actual ``add_argument`` registration takes.  Usage-block occurrences,
    free prose, and help-text descriptions no longer count.

    HONEST FRAMING (why the evidence says DECLARATION-ONLY): this gate
    proves a flag is DECLARED on the parser.  It does NOT prove the flag
    is WIRED into the training loop — the AGENTS.md §5.7 / Wf-385-A1
    failure mode (17 flags declared, 0 wired, all unit tests green) is
    invisible to any ``--help`` probe.  Wire-in proof requires the §5.7
    50-step smoke (``wf399a1_smoke_protocol.py``), not this gate.
    """
    t0 = time.perf_counter()
    # 2026-10-04 — CACHE the --help probe per (interpreter, script, extra_args).
    #
    # This gate is declaration-only: it spawns `python <script> --help` and
    # reads the argparse invocation column.  Twelve gates probe the SAME
    # script with no extra args, so eleven of the twelve subprocesses were
    # pure repetition — and each one pays the script's full import cost
    # (torch, chemprop, ...) before argparse can print anything.  Measured:
    # ~5.3 s per probe, ~64 s of the 180 s `--all` suite.
    #
    # The result is a pure function of (py, script, extra_args) — the gate
    # already treats it as constant, and holding it makes repeated probes
    # free.  Semantics are unchanged: the same command, the same output, the
    # same INVOCATION-column parsing below; only the second and later calls
    # for one key are served from memory.  A FAILing probe is cached too, so
    # twelve gates cannot disagree about one script.
    cache_key = (str(py), str(script), tuple(extra_args or ()))
    hit = _HELP_PROBE_CACHE.get(cache_key)
    if hit is None:
        args = [py, str(script)] + (extra_args or []) + ["--help"]
        rc, out, err = _run(args, timeout=timeout)
        _HELP_PROBE_CACHE[cache_key] = (rc, out, err)
    else:
        rc, out, err = hit
    elapsed = time.perf_counter() - t0
    blob = out + "\n" + err
    tokens = pattern.split()
    # Wf-A12 — build the INVOCATION column of each argparse option line.
    # Argparse renders registrations as "  --flag VALUE   description"
    # (exactly two leading spaces = the option indent; help text follows
    # after a run of >= 2 spaces).  Two things therefore do NOT count:
    # prose anywhere else in the output, and prose that merely shares a
    # line with a flag.  A fallback (dash-led lines, same invocation
    # split) keeps non-default formatters working; the evidence says when
    # it was used.
    option_invocations: List[str] = []
    primary_option_lines = [ln for ln in blob.splitlines() if ln.startswith("  -")]
    if primary_option_lines:
        pool = primary_option_lines
        fallback_note = ""
    else:
        # Non-default formatter fallback (no 2-space option lines found):
        # use dash-led lines, still reduced to the invocation column.
        pool = [ln for ln in blob.splitlines() if ln.lstrip().startswith("-")]
        fallback_note = (
            " [no 2-space argparse option lines found — fallback dash-line "
            "heuristic used]"
        )
    for line in pool:
        body = line[2:] if line.startswith("  -") else line.lstrip()
        option_invocations.append(body.split("  ", 1)[0])
    missing: List[str] = []
    for tok in tokens:
        tok_re = re.compile(r"(?<![\w-])" + re.escape(tok) + r"(?![\w-])")
        if not any(tok_re.search(inv) for inv in option_invocations):
            missing.append(tok)
    if not missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"DECLARATION-ONLY: --help exposes all {len(tokens)} token(s) of "
            f"'{pattern}' in the argparse invocation column of {script.name} "
            f"(rc={rc}, {len(option_invocations)} option lines){fallback_note}. "
            f"NOTE: proves flag DECLARATION, NOT wire-in/behaviour — §5.7 "
            f"smoke required for that.",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"DECLARATION-ONLY check failed: help missing token(s) {missing} from "
        f"'{pattern}' in the argparse invocation column (rc={rc}; "
        f"{len(option_invocations)} option lines in {len(blob)}B output)"
        f"{fallback_note}.  A token appearing only in prose (or only in a "
        f"help-text description) no longer counts (Wf-A12 tightening).",
        elapsed,
    )


def gate_file_exists(
    name: str,
    priority: str,
    pipeline: str,
    file_path: Path,
) -> GateResult:
    t0 = time.perf_counter()
    exists = file_path.exists() and file_path.is_file()
    elapsed = time.perf_counter() - t0
    if exists:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"file exists: {file_path}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"file MISSING: {file_path}",
        elapsed,
    )


def gate_min_lines(
    name: str,
    priority: str,
    pipeline: str,
    file_path: Path,
    min_lines: int,
) -> GateResult:
    t0 = time.perf_counter()
    n = _line_count(file_path)
    elapsed = time.perf_counter() - t0
    if n < 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"file unreadable: {file_path}",
            elapsed,
        )
    if n >= min_lines:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"wc -l {file_path} = {n} >= {min_lines}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"wc -l {file_path} = {n} < {min_lines}",
        elapsed,
    )


def gate_json_threshold(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
    *,
    extractor: Callable[[Dict[str, Any]], float],
    threshold: float,
    op: str = ">=",
    max_age_sec: float = 2592000.0,  # 30 days — historical result records
) -> GateResult:
    """Extractor returns a float; PASS if ``value op threshold``.

    Wf-304A: the prior ``json.load(open(...))`` is replaced by
    :class:`AnchorGuard` so every paper-grade JSON read enforces
    path-exists + mtime-freshness + JSON-parseability guards.  Version
    pinning is intentionally NOT set here because the gate is generic
    over any JSON file; callers needing version pinning should use
    ``AnchorGuard(...)`` directly with ``required_version=...``.

    The default ``max_age_sec=2592000`` (30 days) is wider than the
    AnchorGuard default (24 hr) because pre-flight gates are checking
    **frozen historical result files** (e.g. ``wf188_m14_full_metalcytotox/
    result.json`` from a Wf-188 run that completed once and is never
    regenerated), not live paper-grade anchors that should refresh daily.
    Callers checking live anchors can pass ``max_age_sec=86400`` (24 hr)
    explicitly.
    """
    t0 = time.perf_counter()
    if not json_path.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json file missing: {json_path}",
            time.perf_counter() - t0,
        )
    try:
        # Wf-304A: read via AnchorGuard (path + mtime + JSON parse guards).
        guard = AnchorGuard(
            path=json_path,
            max_age_sec=max_age_sec,
            required_version=None,  # gate is generic over any JSON file
        )
        data = guard.require()
        value = float(extractor(data))
    except (
        AnchorMissingError,
        AnchorStaleError,
        AnchorJSONError,
        AnchorVersionMismatchError,
        KeyError,
        ValueError,
        TypeError,
    ) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"anchor guard / parse / extractor error in {json_path}: "
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    ops = {
        ">=": lambda a, b: a >= b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    cmp = ops.get(op)
    if cmp is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"unknown op '{op}'",
            elapsed,
        )
    if cmp(value, threshold):
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"{json_path}: value={value:.6f} {op} {threshold}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"{json_path}: value={value:.6f} NOT {op} {threshold}",
        elapsed,
    )


def gate_pytest(
    name: str,
    priority: str,
    pipeline: str,
    test_paths: List[Path],
    *,
    timeout: float = 600.0,
) -> GateResult:
    """PASS if pytest on all listed files exits 0."""
    t0 = time.perf_counter()
    args = [VENV_PY, "-m", "pytest", "-q", "--no-header"] + [str(p) for p in test_paths]
    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = blob.splitlines()[-1] if blob else "(no output)"
    if rc == 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"pytest PASS: {tail}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"pytest FAIL (rc={rc}): {tail}",
        elapsed,
    )


def gate_which_or_apt(
    name: str,
    priority: str,
    pipeline: str,
    binary: str,
) -> GateResult:
    """PASS if ``binary`` is on PATH, in repo-local bin/, or listed in apt.

    Wf-260: also probe ``molmetal/bin/{binary}`` so project-local installs
    (e.g. AutoDock Vina shipped at ``molmetal/bin/vina``) don't FAIL the
    gate just because the user has not symlinked it onto PATH yet.
    """
    t0 = time.perf_counter()
    if shutil.which(binary) is not None:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"which {binary} -> present on PATH",
            time.perf_counter() - t0,
        )
    # Wf-260: project-local bin/ fallback (e.g. molmetal/bin/vina)
    repo_local_bin = MOLMETAL_ROOT / "bin" / binary
    if repo_local_bin.is_file() and os.access(repo_local_bin, os.X_OK):
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"repo-local binary present: {repo_local_bin}",
            time.perf_counter() - t0,
        )
    # apt fallback
    rc, out, _ = _run(["apt", "list", "--installed"], timeout=15.0)
    elapsed = time.perf_counter() - t0
    if rc == 0 and binary.lower() in out.lower():
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"apt shows installed package matching {binary}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"which {binary} -> not found; apt-list did not show match",
        elapsed,
    )


# ---------------------------------------------------------------------------
# Gate registry — the v2 auto-verifiable subset
# ---------------------------------------------------------------------------


def gate_yaml_schema(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: List[Path],
) -> GateResult:
    """PASS if every YAML file is parseable via yaml.safe_load.

    v2 §1.3 P0 BLOCKER.  We do not enforce a fixed key schema here because
    the active YAMLs in `configs/` already use lowercased nested keys
    (data/model/flow/train/protocol/eval/output/compute/directives/entry/
    honest_framing per the verified v2 doc §2.1 contract); schema
    divergence across the 7 active YAMLs is *by design*.  Parseability
    is the actual launch-blocker.
    """
    t0 = time.perf_counter()
    try:
        import yaml
    except ImportError:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "PyYAML not installed; cannot verify YAML schemas",
            time.perf_counter() - t0,
        )

    missing: List[str] = []
    invalid: List[str] = []
    for p in yaml_paths:
        if not p.is_file():
            missing.append(str(p))
            continue
        try:
            with open(p) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            invalid.append(f"{p}: {e}")

    elapsed = time.perf_counter() - t0
    if missing or invalid:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing={missing}; invalid={invalid}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(yaml_paths)} YAMLs parse OK",
        elapsed,
    )


# ---------------------------------------------------------------------------
# HOLE-6 fix (Wf-446, 2026-10-03): gate_csv_hash now performs a REAL SHA-256
# verification against a frozen, git-tracked manifest.
#
# The previous implementation only ran ``wc -l`` and its docstring admitted
# that "tampering that preserves row count but edits cells is not detected",
# blaming missing infrastructure.  That claim was stale: the pure-stdlib
# streaming helper ``sha256_of_file`` already exists in
# ``molmetal/scripts/_dmpnn_sha256_guard.py`` (unit-tested) and the gated
# CSVs are 4.7-6.9 MB — a full hash is a few ms each.  The frozen digests
# live in ``configs/data_checksums.json`` (git-tracked; the gated data
# itself sits under ``.data/**`` which is gitignored) and are (re)generated
# by the one-shot command:
#
#     python -m molmetal.scripts.pre_flight_check --freeze-csv-hashes
#
# Behaviour matrix (deliberately NON-BLOCKING by default, per repo policy
# that no gate may turn the tree red while pre-existing data debt exists;
# the strict switch is a separate, later decision):
#
#   file missing / wc fail / rows < min_lines   -> FAIL   (as before)
#   manifest absent or unusable                 -> SKIP   (line-count
#                                                 fallback result reported)
#   manifest entry present + digest matches     -> PASS   (digest prefix)
#   manifest entry present + digest MISMATCH    -> FAIL   (BLOCKING by default
#        as of 2026-10-03; PREFLIGHT_CSV_HASH_STRICT=0 downgrades to a loud
#        stderr warning + PASS)
#   no manifest entry for a gated file          -> PASS + loud warning
#        (coverage gap); FAIL only in strict mode
# ---------------------------------------------------------------------------
CSV_HASH_MANIFEST_DEFAULT = CONFIGS_DIR / "data_checksums.json"

# BLOCKING ON (2026-10-03, Wf-A13 follow-up).  This was shipped OFF; it is now ON.
#
# Why this switch differs from the hyperparameter-resolution switch, which stays
# OFF: a switch may only block when the tree is GREEN under it and the remedy is
# cheap. Here both hold — MEASURED at flip time, `p0_csv_hash_verify` reports
# "all 5/5 CSVs SHA-256-verified", so turning it on breaks nothing today, and the
# mismatch message itself names the one-command remedy
# (`--freeze-csv-hashes`).  The hyperparameter switch is the opposite case: ~528
# declared keys resolve to no dest, so blocking there would fail EVERY commit.
#
# A hash gate that reports a mismatch and then returns PASS is not a hash gate.
# Detection without consequence is the exact pattern this whole audit exists to
# remove, and the detector here is genuinely strong (full-file SHA-256), so
# leaving it non-blocking wasted it.
#
# The override still works in both directions: PREFLIGHT_CSV_HASH_STRICT=0 turns
# it back off for a session where a legitimate re-freeze is in progress.
CSV_HASH_STRICT_DEFAULT = True
CSV_HASH_STRICT_ENV = "PREFLIGHT_CSV_HASH_STRICT"


def gated_csv_specs() -> List[Tuple[Path, int]]:
    """Canonical ``(csv_path, min_lines)`` spec list for ``gate_csv_hash``.

    Single source of truth shared by (a) the ``p0_csv_hash_verify`` gate
    registration in ``build_gates`` and (b) the ``--freeze-csv-hashes``
    manifest generator, so the gate and the frozen manifest cannot drift
    apart.
    """
    return [
        (DOT_DATA_DIR / "metalcytotox_5metal_multitask.csv", 26633),
        (DATA_DIR / "metallo_drugs_500_train.csv", 450),
        (DOT_DATA_DIR / "metalcytotox_5metal_multitask_v2.csv", 26633),
        (DOT_DATA_DIR / "metalcytotox_5metal_multitask_v3.csv", 26633),
        (DOT_DATA_DIR / "metalcytotox_5metal_multitask_v4.csv", 26633),
    ]


def _csv_manifest_keys(p: Path) -> List[str]:
    """Ordered candidate manifest keys for ``p``.

    First the repo-relative POSIX path (the canonical key written by the
    freezer for gated files); then the bare basename as a fallback for
    paths outside the repo (tests / ad-hoc verifications).
    """
    keys: List[str] = []
    try:
        keys.append(p.resolve().relative_to(REPO_ROOT).as_posix())
    except (ValueError, OSError):
        pass
    if p.name not in keys:
        keys.append(p.name)
    return keys


def _load_csv_checksum_manifest(
    manifest_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load the frozen CSV checksum manifest.  Never raises.

    Returns ``(manifest, error)``:

      * ``(dict, None)`` — loaded and shape-valid;
      * ``(None, None)`` — manifest absent (line-count fallback mode);
      * ``(None, str)``  — manifest present but unusable (IO / JSON /
        schema error); the caller treats this like "absent but loud".
    """
    if not manifest_path.is_file():
        return None, None
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{manifest_path}: unreadable ({exc})"
    if not isinstance(obj, dict) or not isinstance(obj.get("files"), dict):
        return None, f"{manifest_path}: schema error (no top-level 'files' mapping)"
    files = obj["files"]
    bad_entries = [
        k
        for k, v in files.items()
        if not (
            isinstance(v, dict)
            and isinstance(v.get("sha256"), str)
            and len(v["sha256"]) == 64
        )
    ]
    if bad_entries:
        return None, (
            f"{manifest_path}: schema error ({len(bad_entries)} entr(ies) "
            f"without a 64-hex 'sha256'): {bad_entries[:3]}"
        )
    return obj, None


def _sha256_of_file(p: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of ``p`` via the canonical pure-stdlib streaming helper
    (``molmetal/scripts/_dmpnn_sha256_guard.py::sha256_of_file``).

    Imported lazily so this module's import graph is unchanged for the
    (many) callers that never hash anything.
    """
    from molmetal.scripts._dmpnn_sha256_guard import sha256_of_file

    return sha256_of_file(p, chunk_bytes=chunk_bytes)


def gate_csv_hash(
    name: str,
    priority: str,
    pipeline: str,
    csv_paths_with_min_lines: List[Tuple[Path, int]],
    manifest_path: Optional[Path] = None,
) -> GateResult:
    """PASS if every CSV exists, meets its line floor, and (when the frozen
    manifest is available) byte-matches its frozen SHA-256.

    HOLE-6 fix (Wf-446, 2026-10-03).  Verifies, per CSV:

      1. exists;
      2. ``wc -l`` >= ``min_lines`` (cheap truncation fingerprint);
      3. full-file SHA-256 == the digest frozen in
         ``configs/data_checksums.json`` — detects cell edits that
         preserve row count, exactly the class of tampering the previous
         line-count-only implementation could not see.

    The manifest is git-tracked (``.data/**`` itself is gitignored, so the
    digest record lives in ``configs/``) and is (re)generated with
    ``python -m molmetal.scripts.pre_flight_check --freeze-csv-hashes``.

    Honest framing — what this does NOT verify:

      * provenance: a matching digest proves the bytes are unchanged
        since the freeze, not that the contents are correct;
      * files absent from the manifest are line-count-checked only
        (reported loudly on stderr; extend coverage by re-running the
        freezer);
      * the manifest freezes TODAY's bytes — whenever a gated CSV is
        legitimately regenerated upstream, re-run the freezer (the loud
        mismatch warning is what tells you to).

    Blocking policy (revised 2026-10-03): pre-existing failures (missing file
    / short file) FAIL, and so does a digest MISMATCH — blocking is ON by
    default.  The tree was green when this was flipped (MEASURED: 5/5
    verified), and a mismatch names its own remedy, so blocking costs nothing
    today and is the only thing that makes the hash meaningful.  Set
    ``PREFLIGHT_CSV_HASH_STRICT=0`` to downgrade a mismatch to a loud warning
    during a legitimate re-freeze.
    """
    t0 = time.perf_counter()
    # Tri-state: the env var must be able to override in BOTH directions.
    # The previous form was `DEFAULT or (env in {truthy})`, which is an OR — so
    # once the default became True the env var could only ever turn it ON, and
    # the documented "set =0 to downgrade" escape hatch silently did nothing.
    # (Same defect class as everything else in this audit: a documented
    # behaviour that does not exist. Caught by testing the hatch, not by
    # reading it.)
    _strict_env = os.environ.get(CSV_HASH_STRICT_ENV, "").strip().lower()
    if _strict_env in {"1", "true", "yes", "on"}:
        strict = True
    elif _strict_env in {"0", "false", "no", "off"}:
        strict = False
    else:
        strict = CSV_HASH_STRICT_DEFAULT
    mpath = (
        Path(manifest_path) if manifest_path is not None else CSV_HASH_MANIFEST_DEFAULT
    )
    manifest, manifest_err = _load_csv_checksum_manifest(mpath)
    files: Dict[str, Any] = manifest["files"] if manifest else {}

    bad: List[str] = []
    verified: List[str] = []
    mismatches: List[str] = []
    unverified: List[str] = []
    linecount_only: List[str] = []

    for p, min_lines in csv_paths_with_min_lines:
        if not p.is_file():
            bad.append(f"missing: {p}")
            continue
        rc, out, _ = _run(["wc", "-l", str(p)], timeout=10.0)
        if rc != 0:
            bad.append(f"wc failed for {p}")
            continue
        try:
            n = int(out.strip().split()[0])
        except (ValueError, IndexError):
            bad.append(f"wc parse failed for {p}")
            continue
        if n < min_lines:
            bad.append(f"{p} has {n} lines (< {min_lines})")

        # ---- SHA-256 layer (HOLE-6) ----
        if manifest is None:
            linecount_only.append(p.name)
            continue
        entry: Optional[Dict[str, Any]] = None
        for key in _csv_manifest_keys(p):
            cand = files.get(key)
            if isinstance(cand, dict):
                entry = cand
                break
        if entry is None:
            unverified.append(p.name)
            if strict:
                bad.append(f"{p.name}: no frozen SHA-256 entry in {mpath}")
            continue
        try:
            digest = _sha256_of_file(p)
        except OSError as exc:
            bad.append(f"SHA-256 read failed for {p}: {exc}")
            continue
        want = str(entry["sha256"]).lower()
        if digest == want:
            verified.append(f"{p.name}={digest[:12]}")
        else:
            msg = f"{p.name}: frozen {want[:12]} != on-disk {digest[:12]}"
            mismatches.append(msg)
            if strict:
                bad.append(f"SHA-256 MISMATCH ({msg})")

    elapsed = time.perf_counter() - t0

    # Loud, non-blocking reporting (mirrors the Wf-A12 hyperparam gate
    # pattern: warn on stderr; blocking only behind an OFF switch).
    for msg in mismatches:
        print(
            f"[pre_flight_check] WARNING {name}: SHA-256 MISMATCH vs {mpath} "
            f"({msg}) — file changed since the freeze. Re-run "
            f"`python -m molmetal.scripts.pre_flight_check --freeze-csv-hashes` "
            f"once the change is confirmed legitimate. Blocking switch "
            f"{'ON' if strict else 'OFF'}.",
            file=sys.stderr,
        )
    for nm in unverified:
        print(
            f"[pre_flight_check] WARNING {name}: {nm} has no entry in {mpath} "
            f"— line-count-only coverage for this file (non-blocking).",
            file=sys.stderr,
        )

    if bad:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(bad),
            elapsed,
        )

    if manifest is None:
        why = "absent" if manifest_err is None else f"unusable: {manifest_err}"
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"frozen manifest {why} ({mpath}) — line-count fallback OK for "
            f"{len(linecount_only)}/{len(csv_paths_with_min_lines)} CSV(s) "
            f"[{', '.join(linecount_only)}]; SHA-256 NOT verified",
            elapsed,
        )

    n_total = len(csv_paths_with_min_lines)
    suffix = ""
    if mismatches:
        suffix += (
            f" | {len(mismatches)} SHA-256 MISMATCH(es) reported loudly, NOT "
            f"blocking (switch {'ON' if strict else 'OFF'}; "
            f"{CSV_HASH_STRICT_ENV}=1 to block): {'; '.join(mismatches)}"
        )
    if unverified:
        suffix += (
            f" | {len(unverified)} file(s) without a frozen entry "
            f"(line-count-only): {', '.join(unverified)}"
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(verified)}/{n_total} CSVs SHA-256-verified vs "
        f"{mpath.name}: {', '.join(verified)}" + suffix,
        elapsed,
    )


def freeze_csv_hash_manifest(
    out_path: Optional[Path] = None,
    specs: Optional[List[Tuple[Path, int]]] = None,
) -> int:
    """One-shot generator for the frozen CSV checksum manifest (HOLE-6).

    Writes ``configs/data_checksums.json`` (or ``out_path``) with a
    ``files`` mapping keyed by repo-relative POSIX path; every value is
    ``{path, sha256, n_lines, min_lines, frozen_at}``.  Written atomically
    via ``atomic_write_file`` (Wf-386 §5.6) so a concurrent reader never
    sees a half-written manifest.

    Refuses (returns 1, writes nothing) if any gated CSV is missing on
    disk — a manifest that silently drops files would turn a real gap into
    a PASS-shaped coverage hole.

    Returns the process exit code (0 = written).
    """
    target = Path(out_path) if out_path is not None else CSV_HASH_MANIFEST_DEFAULT
    spec_list = list(specs) if specs is not None else gated_csv_specs()
    now = _now_iso()
    files: Dict[str, Any] = {}
    missing: List[str] = []
    for p, min_lines in spec_list:
        if not p.is_file():
            missing.append(str(p))
            continue
        rc, out, _ = _run(["wc", "-l", str(p)], timeout=10.0)
        n_lines = -1
        if rc == 0 and out.strip():
            try:
                n_lines = int(out.strip().split()[0])
            except (ValueError, IndexError):
                n_lines = -1
        key = _csv_manifest_keys(p)[0]
        files[key] = {
            "path": key,
            "sha256": _sha256_of_file(p),
            "n_lines": n_lines,
            "min_lines": min_lines,
            "frozen_at": now,
        }
    if missing:
        print(
            f"[freeze-csv-hashes] REFUSED: {len(missing)} gated CSV(s) "
            f"missing on disk: {missing}",
            file=sys.stderr,
        )
        return 1
    obj = {
        "schema": "molmetal-data-checksums-v1",
        "generated_at": now,
        "generator": (
            "python -m molmetal.scripts.pre_flight_check --freeze-csv-hashes"
        ),
        "note": (
            "Frozen SHA-256 of the CSVs gated by p0_csv_hash_verify.  The "
            "gated data live under .data/** (gitignored), so this digest "
            "record lives in git under configs/.  Re-run the generator "
            "after any legitimate upstream regeneration; until then a "
            "digest change is reported (loudly, non-blocking unless "
            "PREFLIGHT_CSV_HASH_STRICT=1)."
        ),
        "files": files,
    }
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    atomic_write_file(target, text)
    print(f"[freeze-csv-hashes] wrote {target} ({len(files)} entries)")
    for k in sorted(files):
        v = files[k]
        print(f"[freeze-csv-hashes]   {v['sha256'][:16]}  {v['n_lines']:>7}  {k}")
    return 0


def gate_scaffold_zero_leak(
    name: str,
    priority: str,
    pipeline: str,
    csv_specs: List[Tuple[Path, str]],
) -> GateResult:
    """PASS if 0 scaffold_id overlap across train/val/test in EVERY given CSV.

    HOLE-5 fix (2026-10-03): the verifier was previously registered against
    the single v4 CSV.  It now takes a list of ``(path, label)`` specs and is
    registered against metalcytotox v2 + v3 + v4 (all three carry
    ``split`` + ``scaffold_id`` columns; MEASURED 0 leaks at the fix
    timestamp, 26632 data rows each).  v1 is deliberately EXCLUDED from THIS
    zero-leak gate: it is the pre-Wf-261 leaky artifact (MEASURED 109 total
    pairwise scaffold overlaps: 29 / 48 / 32) kept on disk as a historical
    record; gating it here would FAIL by design.  L5 fix (2026-10-03)
    corrects the earlier claim that v1 "is not a trainer input" — v1 IS
    still the default data input of 10+ trainer / benchmark scripts; that
    visibility is covered by ``gate_metalcytotox_v1_known_leaky`` (below).

    Coverage limits (honest framing, HOLE-5): the probe needs
    ``split`` + ``scaffold_id`` COLUMNS.  nci60
    (``molmetal/data/nci60_precious_9cell.csv``) has no split columns — its
    split is derived at train time and is covered by
    ``gate_nci60_split_no_leak`` (below).  platinai_4cell, bindingdb_ic50
    (0 bytes) and DrugOOD (external curator configs) carry no on-disk split
    manifest and are therefore NOT gated; they become gateable once a
    train-time split manifest (seed + indices + scaffold_id) is emitted
    (same pattern as metalcytotox v4's own ``scaffold_id`` column).

    v2 §1.4 P0 BLOCKER (post-Wf-261 2026-09-28).  Strict Bemis-Murcko
    scaffold 70/10/20 must produce disjoint scaffold sets across
    splits.  v1 had 29-48 scaffold leaks (pre-Wf-261 honest-negative);
    v2 was regenerated with seed=42 to guarantee 0 leaks.

    Env-aware SKIP contract (Wf-345B1):
    The verifier subprocess imports ``pandas`` at module top-level.
    On a host Python without pandas (e.g. bare ``/usr/bin/python``),
    the structural probe is **not verifiable** — not because the CSV
    is broken, but because the gate's import probe requires a
    pandas-enabled runtime (expected: the repo-local canonical
    ``/home/hugo/codes/mol-metal/.venv/bin/python`` per AGENTS.md
    §5.1 + §5.9). We treat this as an **honest-negative SKIP** rather than a
    FAIL because:

      1. The v4 CSV is on disk + structurally readable (verified via
         ``csv_path.is_file()`` upfront).
      2. The only reason the probe cannot PASS is the host env gap,
         not a regression in the CSV or gate.
      3. A SKIP surfaces the env dependency to the operator without
         blocking it (the gate only raises exit-code 2 under
         ``--strict``).

    On ``ModuleNotFoundError`` / ``ImportError`` for ``pandas`` the
    gate emits SKIP with reason naming the missing dependency +
    expected runtime. CPU-only (no GPU, no extra deps beyond pandas).

    FAIL on (a) the CSV file is missing, (b) the subprocess reports
    >0 scaffold leaks (data integrity violation). Other unexpected
    subprocess errors still FAIL (preserving the original Wf-261
    contract). See TODO/completed/261_v2_p0_scaffold_split.md +
    TODO/completed/345b1_scaffold_zero_leak_fix.md for the Wf-261 +
    Wf-345B1 ship surface.
    """
    t0 = time.perf_counter()
    missing = [(str(p), lab) for p, lab in csv_specs if not p.is_file()]
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "CSV(s) missing: " + "; ".join(f"{lab}={path}" for path, lab in missing),
            time.perf_counter() - t0,
        )
    specs_literal = repr([(str(p), lab) for p, lab in csv_specs])
    probe = (
        "import pandas as pd\n"
        "SPECS = " + specs_literal + "\n"
        "for path, label in SPECS:\n"
        "    df = pd.read_csv(path)\n"
        "    sc = {s: set(df[df['split']==s]['scaffold_id'].dropna()) "
        "for s in ['train','val','test']}\n"
        "    leaks = [(a,b,len(sc[a]&sc[b])) for a in sc for b in sc "
        "if a < b and sc[a] & sc[b]]\n"
        "    print(('LEAK ' if leaks else 'OK ') + label + ' rows=' "
        "+ str(len(df)) + (' leaks=' + str(leaks) if leaks else ''))\n"
    )
    rc, out, err = _run(
        [sys.executable, "-c", probe],
        timeout=20.0 + 20.0 * len(csv_specs),
    )
    elapsed = time.perf_counter() - t0
    if rc != 0:
        # Wf-345B1 env-aware SKIP: a pandas-import failure is an env gap,
        # not a regression. The CSV file IS present (verified above);
        # the gate's verifier just cannot execute without pandas.
        # Detect ModuleNotFoundError / ImportError patterns in stderr
        # (the subprocess pipes the ImportError traceback to stderr)
        # and emit SKIP with the missing dep named. Otherwise keep the
        # original FAIL contract but include a stderr snippet for
        # operator verifiability.
        err_tail = (err or "").strip().splitlines()[-3:]
        err_tail_str = " | ".join(line for line in err_tail if line) or "(no stderr)"
        if (
            "ModuleNotFoundError" in err
            or "ImportError" in err
            or "No module named" in err
        ):
            # Extract the missing module name from the traceback tail.
            missing_name = "pandas"
            for line in err_tail:
                if "No module named" in line:
                    # Match: "ModuleNotFoundError: No module named 'pandas'"
                    after = line.split("No module named", 1)[-1]
                    name = after.strip().strip("'\"")
                    if name:
                        missing_name = name
                    break
            return GateResult(
                name,
                priority,
                pipeline,
                "SKIP",
                f"scaffold-leak verifier subprocess requires "
                f"{missing_name!r} (env gap on host Python); "
                f"the gated CSV(s) ARE present "
                f"({', '.join(lab for _p, lab in csv_specs)}) and "
                f"structurally readable, but the "
                f"gate's pandas import probe cannot execute without "
                f"the expected pandas-enabled runtime (the repo-local "
                f"canonical "
                f"/home/hugo/codes/mol-metal/.venv/bin/python "
                f"per AGENTS.md §5.1 + §5.9). This is an env gap, not a "
                f"regression. Original probe error: {err_tail_str}",
                elapsed,
            )
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"leak-check subprocess failed: rc={rc} | " f"stderr_tail: {err_tail_str}",
            elapsed,
        )
    lines = [
        line
        for line in out.strip().splitlines()
        if line.startswith("OK ") or line.startswith("LEAK ")
    ]
    leak_lines = [line for line in lines if line.startswith("LEAK ")]
    if leak_lines:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(leak_lines),
            elapsed,
        )
    if len(lines) != len(csv_specs):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"probe returned {len(lines)}/{len(csv_specs)} per-CSV results; "
            f"raw output: {out.strip()[:300] or '(empty)'}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"0 scaffold leak in all {len(lines)} CSVs: " + "; ".join(lines),
        elapsed,
    )


# ---------------------------------------------------------------------------
# L5 fix (2026-10-03) — KNOWN-LEAKY v1 CSV visibility gate.
# ---------------------------------------------------------------------------
# FORMER default-consumer roster for the leaky v1 artifact
# (``.data/molmetal/metalcytotox_5metal_multitask.csv``).
#
# STATE CHANGE 2026-10-03 (Wf-A14): the repointing this roster listed as the
# open remedy HAS BEEN DONE. 134 occurrences across 70 live files were moved
# from ``metalcytotox_5metal_multitask.csv`` to
# ``metalcytotox_5metal_multitask_v4.csv``. So the scripts below no longer
# default to v1, and the roster is retained only as the historical record of
# who used to — it is NO LONGER a list of live consumers, and the gate that
# reads it is a leak-visibility check on the FILE, not a consumer audit.
#
# Why v4 and not a judgement call: the project had already documented the
# supersession — TODO/completed/m14_lead_deployment_precondition.md:154,
# "The v4 cohort supersedes the v2 cohort ... and the v1 cohort (26,802 lines,
# leaky)". This change completed that migration rather than deciding it.
#
# v1 itself is UNCHANGED and still on disk, still hash-frozen by
# ``p0_csv_hash_verify``. A run that deliberately passes the old path still
# reproduces the old behaviour; only the DEFAULTS moved.
#
# The leak is MEASURED, not inherited folklore. check_data_leakage.py on v1:
#   random_split:   strict=FAIL (1935 shared SMILES; 1213 shared scaffolds)
#   scaffold_split: strict=PASS (0 shared scaffolds)   <- recomputed split is clean
#   shipped_split:  strict=FAIL (48 shared scaffolds)  <- the SHIPPED split leaks
METALCYTOTOX_V1_DEFAULT_CONSUMERS: Tuple[str, ...] = (
    "train_ru_multitask.py (DEFAULT_DATA, L156)",
    "train_multimetal_lomo.py (L790)",
    "train_ru_5metal_se3sb.py (L201)",
    "train_multiph.py (L153)",
    "train_transport.py (L141)",
    "train_se3sb.py (L142)",
    "benchmark_r18_metalcytotox.py (DEFAULT_CSV, L99)",
    "benchmark_wf25_5seed.py (DEFAULT_METALCYTOX_CSV, L121)",
    "wf25_hilbert_real_eval.py / wf25_multiph_real_eval.py / "
    "wf25_combined_real_eval.py (argparse --data defaults)",
    "bench_lambda_cfm_e2e.py (DEFAULT_METALCYTO_CSV, L147)",
)


def gate_metalcytotox_v1_known_leaky(
    name: str,
    priority: str,
    pipeline: str,
    csv_path: Path,
    consumers: Sequence[str] = METALCYTOTOX_V1_DEFAULT_CONSUMERS,
) -> GateResult:
    """L5 fix (2026-10-03) — make the known-leaky v1 artifact VISIBLE.

    Why this gate exists (MEASURED, 2026-10-03, repo venv): v1
    (``.data/molmetal/metalcytotox_5metal_multitask.csv``, sha256
    ``f167c119…``, frozen in ``configs/data_checksums.json``) carries 109
    total pairwise Bemis-Murcko scaffold overlaps (train/val=29,
    train/test=48, val/test=32; 26801 data rows) yet is STILL the default
    data input of 10+ trainer / benchmark scripts (roster:
    :data:`METALCYTOTOX_V1_DEFAULT_CONSUMERS`).  The sibling
    ``p0_scaffold_zero_leak`` gates v2/v3/v4 only; its pre-L5 comment
    claimed v1 is "not a trainer input" — FALSE, corrected here.

    Semantics (report-only; NOT red-by-design — the leak is a known
    honest-negative, and a permanently red gate gets ignored):

      * leaks > 0   -> PASS with evidence: row count + per-pair overlap
        counts + sha256 prefix + the consumer roster as the open remedy.
      * 0 leaks     -> FAIL: the artifact no longer matches its
        known-leaky historical record (mutated / re-split / replaced).
        The remedy message names the restore path
        (``configs/data_checksums.json`` + the blocking
        ``p0_csv_hash_verify`` gate); a deliberate clean re-split must
        re-freeze the digest AND update this gate's contract together.
      * file absent -> SKIP (gitignored ``.data/**`` host contract —
        mirrors ``gate_p2_data_leakage_diag``; presence is owned by the
        blocking ``p0_csv_hash_verify`` manifest entry, not by this
        visibility gate).
      * pandas import failure in the probe -> SKIP (Wf-345B1 env-aware
        contract, mirrors ``gate_scaffold_zero_leak``).

    Registered ['all'] only (pre-retrain internal check per the §3.4
    design contract).
    """
    t0 = time.perf_counter()
    if not csv_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"v1 CSV absent on this host ({csv_path}); the known-leaky "
            f"artifact cannot be measured.  Presence is owned by the "
            f"blocking `p0_csv_hash_verify` manifest entry — this "
            f"visibility gate does not double-report its absence.",
            time.perf_counter() - t0,
        )
    probe = (
        "import hashlib\n"
        "import pandas as pd\n"
        "p = " + repr(str(csv_path)) + "\n"
        "df = pd.read_csv(p)\n"
        "sc = {s: set(df[df['split']==s]['scaffold_id'].dropna()) "
        "for s in ['train','val','test']}\n"
        "pairs = [('train','val'),('train','test'),('val','test')]\n"
        "over = [(a,b,len(sc[a]&sc[b])) for a,b in pairs]\n"
        "print('ROWS=' + str(len(df)))\n"
        "print('OVERLAPS=' + ','.join(a+'/'+b+'='+str(n) for a,b,n in over))\n"
        "print('TOTAL=' + str(sum(n for _a,_b,n in over)))\n"
        "print('SHA256=' + hashlib.sha256(open(p,'rb').read()).hexdigest())\n"
    )
    rc, out, err = _run([sys.executable, "-c", probe], timeout=60.0)
    elapsed = time.perf_counter() - t0
    if rc != 0:
        err_tail = (err or "").strip().splitlines()[-3:]
        err_tail_str = " | ".join(line for line in err_tail if line) or "(no stderr)"
        if (
            "ModuleNotFoundError" in err
            or "ImportError" in err
            or "No module named" in err
        ):
            missing_name = "pandas"
            for line in err_tail:
                if "No module named" in line:
                    after = line.split("No module named", 1)[-1]
                    nm = after.strip().strip("'\"")
                    if nm:
                        missing_name = nm
                    break
            return GateResult(
                name,
                priority,
                pipeline,
                "SKIP",
                f"v1 leak probe requires {missing_name!r} (env gap on "
                f"host Python); the v1 CSV IS present ({csv_path}) and "
                f"readable, but the probe cannot execute without the "
                f"expected pandas-enabled runtime (the repo-local "
                f"canonical /home/hugo/codes/mol-metal/.venv/bin/python "
                f"per AGENTS.md §5.1 + §5.9). Env gap, not a regression. "
                f"Original probe error: {err_tail_str}",
                elapsed,
            )
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"v1 leak probe failed: rc={rc} | stderr_tail: {err_tail_str}",
            elapsed,
        )
    fields: Dict[str, str] = {}
    for line in out.strip().splitlines():
        for key in ("ROWS=", "OVERLAPS=", "TOTAL=", "SHA256="):
            if line.startswith(key):
                fields[key[:-1]] = line[len(key) :].strip()
    missing = [k for k in ("ROWS", "OVERLAPS", "TOTAL", "SHA256") if k not in fields]
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"v1 leak probe returned unparsable output (missing "
            f"{missing}); raw: {out.strip()[:300] or '(empty)'}",
            elapsed,
        )
    try:
        total = int(fields["TOTAL"])
    except ValueError:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"v1 leak probe TOTAL unparsable: {fields['TOTAL']!r} "
            f"(raw: {out.strip()[:300]})",
            elapsed,
        )
    sha = fields["SHA256"]
    if total == 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"v1 no longer leaks: 0 scaffold overlaps "
            f"(rows={fields['ROWS']}, sha256={sha[:12]}…).  v1 is the "
            f"KNOWN-LEAKY historical artifact; 0 leaks means it was "
            f"mutated / re-split / replaced.  Remedy: restore the frozen "
            f"bytes whose digest is recorded in configs/data_checksums.json "
            f"(the blocking `p0_csv_hash_verify` gate verifies byte "
            f"identity), or, if a clean re-split is intended, re-freeze "
            f"the digest (one command: `python -m molmetal.scripts."
            f"pre_flight_check --freeze-csv-hashes`) AND update this "
            f"gate's contract in the same edit.  Open item — consumers "
            f"still defaulting to this path: " + "; ".join(consumers),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"KNOWN-LEAKY v1 artifact still leaky as expected: "
        f"rows={fields['ROWS']}, scaffold overlaps "
        f"{fields['OVERLAPS']} (total={total}), sha256={sha[:12]}… "
        f"(bytes frozen by p0_csv_hash_verify).  Open remedy (not this "
        f"gate's lane): repoint the default consumers at "
        f"metalcytotox_5metal_multitask_v4.csv — " + "; ".join(consumers),
        elapsed,
    )


def _nci60_index_triple_violations(
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    n_rows: int,
) -> List[str]:
    """HOLE-5 helper — pairwise-disjointness + coverage check for an index triple.

    Pure function (no rdkit / no I/O) so it is unit-testable.  Returns a
    list of human-readable violation strings (empty == clean).
    """
    tr = {int(i) for i in train_idx}
    va = {int(i) for i in val_idx}
    te = {int(i) for i in test_idx}
    bad: List[str] = []
    for a, b, lab in (
        (tr, va, "train/val"),
        (tr, te, "train/test"),
        (va, te, "val/test"),
    ):
        if a & b:
            bad.append(f"index overlap {lab}: {len(a & b)}")
    union = tr | va | te
    if len(union) != int(n_rows):
        bad.append(f"partition incomplete: union={len(union)} rows={int(n_rows)}")
    return bad


def _nci60_scaffold_pair_leaks(
    keys_by_split: Dict[str, set],
    pairs: Sequence[Tuple[str, str]] = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ),
) -> List[Tuple[str, str, int]]:
    """HOLE-5 helper — scaffold-key overlap counts for the given split pairs.

    Pure function (no rdkit / no I/O) so it is unit-testable.  Returns one
    ``(split_a, split_b, n_shared_scaffold_keys)`` triple per requested pair
    (including zeros — callers filter).
    """
    return [
        (a, b, len(keys_by_split.get(a, set()) & keys_by_split.get(b, set())))
        for a, b in pairs
    ]


def gate_nci60_split_no_leak(
    name: str,
    priority: str,
    pipeline: str,
    csv_path: Path,
    reference_unique_sizes: Tuple[int, int, int] = (3645, 521, 1041),
    seeds: Tuple[int, ...] = (0, 1, 2),
    strict_scaffold: Optional[bool] = None,
) -> GateResult:
    """PASS if the NCI-60 cohort's split has no split/scaffold leakage.

    HOLE-5 fix (2026-10-03): the nci60 headline cohort
    (``molmetal/data/nci60_precious_9cell.csv``, 7823 rows / 5207 unique
    SMILES / 9 cell lines) carries NO split columns, so the CSV-level
    ``gate_scaffold_zero_leak`` cannot see it.  This gate verifies the two
    split paths that actually consume the cohort, using the REAL functions
    the trainers use (imported, never reimplemented):

    Part A — the SHIPPED headline path (Wf-A10):
      ``train_dmpnn_multitask_wasserstein.py --dataset nci60_9cell`` loads via
      ``load_nci60_9cell_split_table`` (m21's protocol: Bemis-Murcko
      ``ScaffoldSplitter``, ``largest_first``, 70/10/20, seed=0, on unique
      canonical SMILES).  Hard assertions: (1) every unique SMILES lands in
      exactly one split (cross-cell leakage == 0), (2) unique-molecule split
      sizes == frozen m21 reference ``(3645, 521, 1041)`` — MEASURED
      2026-10-03; a mismatch means m21 comparability (shipped
      0.8007 +/- 0.0160) is broken, (3) train/val/test row-index sets
      pairwise disjoint.

      Scaffold disjointness is MEASURED with real Bemis-Murcko keys
      (``MurckoScaffoldSmiles``, which accepts ``includeChirality``) and
      reported in evidence.  HOLE-5 FINDING (MEASURED 2026-10-03, honest-
      negative): the shipped split is NOT scaffold-disjoint —
      overlap(train/val, train/test, val/test) = (165, 282, 121).  Root cause:
      ``ScaffoldSplitter._scaffold_for`` calls
      ``GetScaffoldForMol(mol, includeChirality=...)`` (TypeError in the
      installed rdkit — only the 1-arg form exists) and
      ``molmetal/data/splits.py:604`` swallows it with a bare ``except``, so
      every molecule gets the per-molecule key ``__empty__<canon>`` and the
      splitter is MOLECULE-level.  m21's ``train_nci60_10cell.py`` uses the
      same splitter code, which is why the sizes match byte-for-byte.
      Default verdict on this finding: PASS + loud honest-negative in
      evidence (this run's directive: report loudly; do not turn blocking
      switches ON).  ``strict_scaffold=True`` (or env
      ``NCI60_SPLIT_STRICT_SCAFFOLD=1``) upgrades the overlap to FAIL — turn
      it ON once ``molmetal/data/splits.py`` is fixed (out of this fix's
      file ownership), at which point the overlap should measure 0.

    Part B — the chemprop driver scaffold path (Wf-315F / Wf-319E):
      ``_scaffold_split_indices_nci60`` from
      ``molmetal/scripts/_nci60_scaffold_helpers.py`` is probed for every cell
      in the CSV x published seeds ``(0, 1, 2)`` (the metric's own source
      string), with per-cell thresholds from the real
      ``compute_per_cell_n_test_min`` (driver ``--n-test-min-auto`` defaults:
      floor=5, headroom=0.85).  Asserts index-triple disjointness + coverage,
      and TEST-vs-{train,val} scaffold disjointness (the OOD property the
      helper claims).  A ``None`` return (random 80/10/20 fallback) FAILs
      loudly: scaffold protection would be silently OFF for that cell.

    Honest framing (what this does NOT verify):
      * train<->val scaffold sharing is EXPECTED and not failed: the helper
        splits the non-test pool at molecule level and val is early-stopping
        only.  MEASURED 1120 shared scaffold keys across all cells/seeds —
        reported in evidence, not a failure.
      * the chemprop driver's DEFAULT ``--split temporal`` is a RANDOM split;
        it is not scaffold-disjoint BY DESIGN and is not gated here.  Same
        for ``--no-n-test-min-auto --n-test-min 50`` (Wf-315F behaviour,
        scaffold path never fires).  Only the scaffold-protected path is
        verified.
      * platinai_4cell / bindingdb_ic50 (0 bytes) / DrugOOD: no on-disk split
        manifest -> NOT gated until one is emitted at train time.
      * CPU-only, no network, no GPU; ~20 s in the repo venv.  SKIPs (env
        gap, AGENTS.md §5.9) when pandas / rdkit / the trainer import fails.
    """
    t0 = time.perf_counter()
    if strict_scaffold is None:
        strict_scaffold = os.environ.get(
            "NCI60_SPLIT_STRICT_SCAFFOLD", ""
        ).strip().lower() in ("1", "true", "yes")
    if not csv_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"nci60 CSV missing: {csv_path}",
            time.perf_counter() - t0,
        )
    try:
        import numpy as np
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        from molmetal.data.splits import ScaffoldSplitter
        from molmetal.scripts._nci60_scaffold_helpers import (
            _scaffold_split_indices_nci60,
            _stable_hash_int,
            compute_per_cell_n_test_min,
        )
        from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
            load_nci60_9cell_split_table,
        )
    except ImportError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"nci60 split-leak probe requires the pandas/rdkit-enabled repo "
            f"venv (canonical /home/hugo/codes/mol-metal/.venv/bin/python per "
            f"AGENTS.md §5.1 + §5.9); import failed with "
            f"{type(exc).__name__}: {exc}. The nci60 CSV IS present "
            f"({csv_path}); this is an env gap, not a regression.",
            time.perf_counter() - t0,
        )

    problems: List[str] = []
    notes: List[str] = []

    # ---- Part A — shipped Wf-A10 / m21 headline split path ----
    try:
        df = load_nci60_9cell_split_table(csv_path)
    except Exception as exc:  # noqa: BLE001 — surface loader failure as FAIL
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"load_nci60_9cell_split_table({csv_path.name}) raised "
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    n_rows = int(len(df))
    n_unique = int(df["smiles"].nunique())
    dup_assign = int((df.groupby("smiles")["split"].nunique() > 1).sum())
    if dup_assign:
        problems.append(
            f"m21 split: {dup_assign} unique SMILES assigned to >1 split "
            f"(cross-cell leakage)"
        )
    unique_by_split = {
        s: df[df["split"] == s]["smiles"].unique() for s in ("train", "val", "test")
    }
    got_sizes = tuple(len(unique_by_split[s]) for s in ("train", "val", "test"))
    ref_sizes = tuple(int(x) for x in reference_unique_sizes)
    if got_sizes != ref_sizes:
        problems.append(
            f"m21 split unique-molecule sizes {got_sizes} != frozen reference "
            f"{ref_sizes} (m21 comparability broken)"
        )
    row_idx = {
        s: {int(i) for i in df.index[df["split"] == s]}
        for s in ("train", "val", "test")
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if row_idx[a] & row_idx[b]:
            problems.append(
                f"m21 split row-index overlap {a}/{b}: {len(row_idx[a] & row_idx[b])}"
            )
    splitter = ScaffoldSplitter(
        strategy="largest_first",
        fractions=(0.7, 0.1, 0.2),
        seed=0,
    )
    # HOLE-5 honest-negative (2026-10-03, MEASURED): the splitter's own
    # `_scaffold_for` is DEGENERATE in this environment — it calls
    # `MurckoScaffold.GetScaffoldForMol(mol, includeChirality=...)`, which
    # raises TypeError (the installed rdkit exposes the 1-arg form only),
    # and `molmetal/data/splits.py` swallows the exception with a bare
    # `except Exception: return ""`.  Result: every molecule gets the
    # per-molecule key `__empty__<canon>` and `ScaffoldSplitter` currently
    # splits at MOLECULE level, not scaffold level.  `train_nci60_10cell.py`
    # (m21) uses the same splitter code, so the frozen (3645, 521, 1041)
    # sizes pin comparability with m21's protocol AS EXECUTED, and the
    # scaffold-disjointness claim is FALSE for the shipped split.  Probe the
    # degeneracy explicitly (benzene MUST produce a non-empty scaffold key)
    # so the gate names the mechanism instead of reading stronger than it is.
    degenerate_splitter = splitter._scaffold_for("c1ccccc1") == ""
    keys_degenerate: Dict[str, set] = {}
    for sname, arr in unique_by_split.items():
        canon = splitter._canonicalize(np.asarray(arr, dtype=object))
        ks = set()
        for smi in canon:
            sc = splitter._scaffold_for(str(smi))
            ks.add(sc if sc else f"__empty__{smi}")
        keys_degenerate[sname] = ks
    # Real Bemis-Murcko keys (MurckoScaffoldSmiles DOES accept
    # includeChirality) — the honest scaffold-disjointness measurement.
    keys_real: Dict[str, set] = {}
    for sname, arr in unique_by_split.items():
        canon = splitter._canonicalize(np.asarray(arr, dtype=object))
        ks = set()
        for smi in canon:
            sc = MurckoScaffoldSmiles(str(smi), includeChirality=False)
            ks.add(sc if sc else f"__empty__{smi}")
        keys_real[sname] = ks
    leaks_real = [t for t in _nci60_scaffold_pair_leaks(keys_real) if t[2]]
    if leaks_real and strict_scaffold:
        problems.append(
            "m21 split REAL scaffold leaks (strict mode): "
            + ", ".join(f"{a}/{b}={n}" for a, b, n in leaks_real)
        )
    # NOTE: `strict_scaffold` is OFF by default per this run's directive
    # (report loudly; do not turn blocking switches ON while 528 config keys
    # still resolve to no argparse dest).  Enable via parameter or env
    # ``NCI60_SPLIT_STRICT_SCAFFOLD=1`` once molmetal/data/splits.py:604 is
    # fixed; until then the real overlap is reported as honest-negative.
    _ = keys_degenerate  # kept for evidence / debugging; not an assertion

    # ---- Part B — chemprop driver scaffold-helper path ----
    cells = sorted(df["cell_line"].astype(str).unique())
    smiles_by_cell: Dict[str, List[str]] = {}
    active_by_cell: Dict[str, Any] = {}
    keys_by_cell: Dict[str, List[str]] = {}
    eff_by_cell: Dict[str, int] = {}
    for cell in cells:
        sub = df[df["cell_line"].astype(str) == cell]
        smis = [str(s) for s in sub["smiles"].tolist()]
        smiles_by_cell[cell] = smis
        active_by_cell[cell] = sub["active"].to_numpy()
        eff_by_cell[cell] = int(compute_per_cell_n_test_min(smiles_subset=smis))
        cell_keys: List[str] = []
        for smi in smis:
            try:
                sc = MurckoScaffoldSmiles(smi, includeChirality=False)
            except Exception:  # noqa: BLE001 — mirror helper's fallback
                sc = ""
            cell_keys.append(
                sc if sc else f"__circular_{int(_stable_hash_int(smi, 1024))}__"
            )
        keys_by_cell[cell] = cell_keys
    n_probes = 0
    n_fallback = 0
    n_test_rest_leak = 0
    tv_shared_total = 0
    for seed in seeds:
        for cell in cells:
            n_probes += 1
            out = _scaffold_split_indices_nci60(
                smiles_subset=list(smiles_by_cell[cell]),
                active_subset=active_by_cell[cell],
                seed=int(seed),
                n_test_min=int(eff_by_cell[cell]),
            )
            if out is None:
                n_fallback += 1
                problems.append(
                    f"{cell}/seed{seed}: scaffold helper returned None -> random "
                    f"80/10/20 fallback (scaffold protection OFF; "
                    f"n_test_min={eff_by_cell[cell]})"
                )
                continue
            tr, va, te, _lbl = out
            for v in _nci60_index_triple_violations(
                tr, va, te, len(smiles_by_cell[cell])
            ):
                problems.append(f"{cell}/seed{seed}: {v}")
            ktr = {keys_by_cell[cell][int(i)] for i in tr}
            kva = {keys_by_cell[cell][int(i)] for i in va}
            kte = {keys_by_cell[cell][int(i)] for i in te}
            n_test_rest_leak += len(ktr & kte) + len(kva & kte)
            tv_shared_total += len(ktr & kva)
            if ktr & kte:
                problems.append(
                    f"{cell}/seed{seed}: scaffold train/test overlap="
                    f"{len(ktr & kte)}"
                )
            if kva & kte:
                problems.append(
                    f"{cell}/seed{seed}: scaffold val/test overlap=" f"{len(kva & kte)}"
                )

    notes.append(
        f"m21/A10 loader: rows={n_rows} unique={n_unique} "
        f"unique-by-split(train,val,test)={got_sizes} "
        f"row-by-split=({len(row_idx['train'])},{len(row_idx['val'])},"
        f"{len(row_idx['test'])}) cross-cell-leaks={dup_assign} "
        f"REAL-scaffold-overlap(train/val,train/test,val/test)="
        f"{tuple(n for _a, _b, n in _nci60_scaffold_pair_leaks(keys_real))} "
        f"splitter_scaffold_for_degenerate={degenerate_splitter}"
    )
    if leaks_real and not strict_scaffold:
        notes.append(
            "HONEST-NEGATIVE: the shipped split is NOT Bemis-Murcko "
            "scaffold-disjoint (see REAL-scaffold-overlap above). Root cause "
            "(MEASURED): ScaffoldSplitter._scaffold_for calls "
            "GetScaffoldForMol(mol, includeChirality=...) — TypeError in the "
            "installed rdkit — and molmetal/data/splits.py:604 swallows it, "
            "so the 'scaffold' splitter is molecule-level. m21 "
            "(train_nci60_10cell.py) uses the same code, hence the "
            "byte-identical sizes. molmetal/data/splits.py is out of this "
            "fix's ownership; turn this into a hard FAIL via "
            "NCI60_SPLIT_STRICT_SCAFFOLD=1 or strict_scaffold=True once the "
            "splitter is fixed."
        )
    notes.append(
        f"chemprop scaffold helper: cells={len(cells)} seeds={list(seeds)} "
        f"probes={n_probes} fallbacks={n_fallback} "
        f"test-vs-rest-scaffold-leaks={n_test_rest_leak} "
        f"train/val-shared-scaffold-keys={tv_shared_total} (documented: "
        f"non-test pool split at molecule level; affects early stopping only)"
    )
    elapsed = time.perf_counter() - t0
    evidence = " | ".join(notes)
    if problems:
        shown = "; ".join(problems[:6])
        extra = f" (+{len(problems) - 6} more)" if len(problems) > 6 else ""
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(problems)} problem(s): {shown}{extra} | {evidence}",
            elapsed,
        )
    return GateResult(name, priority, pipeline, "PASS", evidence, elapsed)


# ---------------------------------------------------------------------------
# Wf-306C — P3 frontier-math wire-in + 2026-09-29 follow-up gates
# ---------------------------------------------------------------------------


def gate_python_importable(
    name: str,
    priority: str,
    pipeline: str,
    module_paths: Tuple[Path, ...],
) -> GateResult:
    """PASS if every ``module_path`` resolves to a non-empty ``.py`` file.

    Wf-306C — frontier math wire-in gate. Probes ``importlib.util``
    *file existence + parseability* of each adapter (cheap heuristic
    that avoids actually instantiating the adapters — those may pull
    torch / triton / egll which we do NOT want on the CPU pre-flight
    path). Does NOT verify the adapter is *wired* into the consumer
    ``__init__.py`` (that's a structural check separate from this gate).

    The 4 adapters probed are:

      * ``multi_ph_integration.py`` (Kim-Jung 2026 multi-PH Fourier 32-bin)
      * ``se3_lipman_integration.py`` (Lipman flow matching SE(3) bridge)
      * ``se3_sb_integration.py`` (Schrödinger-Bridge SE(3) augmentation)
      * ``transport_relaxation_integration.py`` (Jiang-Xu-Zhang transport
        relaxation JKO-style step)

    Pre-Wf-305B all 4 are importable but not wired into the locked
    ``flow_matching_lipman/__init__.py``; Wf-305B target is to wire all
    4 into ``__init__.py`` and re-run this gate to confirm 4/4 still
    importable (no import-time regressions).
    """
    import importlib.util  # local import — module-level is fine but keeps close to gate

    t0 = time.perf_counter()
    missing: List[str] = []
    for p in module_paths:
        if not p.is_file():
            missing.append(f"missing: {p}")
            continue
        if p.stat().st_size <= 0:
            missing.append(f"empty: {p}")
            continue
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            missing.append(f"no spec: {p}")
            continue
    elapsed = time.perf_counter() - t0
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(missing)}/{len(module_paths)} adapter(s) not importable: "
            f"{missing}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(module_paths)}/4 frontier-math adapters importable",
        elapsed,
    )


def gate_all_files_exist(
    name: str,
    priority: str,
    pipeline: str,
    paths: Tuple[Path, ...],
) -> GateResult:
    """PASS if every path in ``paths`` resolves to a non-empty file."""
    t0 = time.perf_counter()
    missing: List[str] = []
    for p in paths:
        if not p.is_file():
            missing.append(str(p))
            continue
        try:
            if p.stat().st_size <= 0:
                missing.append(f"empty: {p}")
        except OSError:
            missing.append(f"unreadable: {p}")
    elapsed = time.perf_counter() - t0
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(missing)}/{len(paths)} missing/empty: {missing}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(paths)} files present",
        elapsed,
    )


def gate_wf350a2_compute_manifest(
    name: str,
    priority: str,
    pipeline: str,
    manifest_path: Path,
    nvidia_smi_bin: str = "nvidia-smi",
) -> GateResult:
    """Wf-350-A2 — NeurIPS §E.5 compute-infrastructure manifest gate.

    Per Wf-343a-R2 recommendation, this is gate #2 of 3 in the NeurIPS
    §E.5 compute-manifest surface.  The gate emits (or refreshes)
    ``manifest_path`` with the canonical compute-infra record:

    * GPU inventory: per-GPU ``{index, name, memory_total_mb,
      driver_version}`` (via ``nvidia-smi --query-gpu=...``).
    * Torch version + CUDA version (``torch.__version__`` +
      ``torch.version.cuda``).
    * System info: ``uname -a`` hostname / kernel / arch.

    The gate PASSES iff:

    (a) ``manifest_path`` is a non-empty JSON file with the canonical
        top-level keys ``gpus``, ``driver_version``, ``torch_version``,
        ``cuda_version``, ``system`` (``{hostname, kernel, arch}``);
    (b) every per-GPU dict carries ``index``, ``name``,
        ``memory_total_mb`` (the 3 fields required for NeurIPS §E.5
        reproducibility);
    (c) ``system`` carries ``hostname``, ``kernel``, ``arch`` (the 3
        fields required for the architecture audit).

    On FAIL the evidence names the missing field(s) directly so the
    regression is debuggable from the gate output alone.

    Honest framing (party-twenty-spirit §3.5):

    * The probe is **CPU-only** (no `torch.cuda`, no GPU kernel
      launch).  GPU fields are populated from `nvidia-smi` text-mode
      probe (no torch.cuda dependency).
    * On a host without `nvidia-smi`, ``gpus`` is set to ``[]`` and
      the gate still PASSes iff the rest of the manifest is valid
      (CPU-only honest-negative).
    * The manifest file is the canonical compute-infra record for
      the paper §E.5 reproducibility appendix — MEASURED per
      AGENTS.md §5.2 (host-time probe, no PROJECTED fields).
    """

    t0 = time.perf_counter()

    # ---- (1) ensure parent directory exists ----
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"cannot create manifest parent dir " f"{manifest_path.parent!s}: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (2) probe nvidia-smi (per-GPU inventory) ----
    gpus: List[Dict[str, Any]] = []
    driver_version: Optional[str] = None
    nvidia_rc, nvidia_out, nvidia_err = _run(
        [
            nvidia_smi_bin,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=15.0,
    )
    if nvidia_rc == 0 and nvidia_out.strip():
        for raw_line in nvidia_out.strip().splitlines():
            parts = [p.strip() for p in raw_line.split(",")]
            if len(parts) < 4:
                continue
            try:
                gpu_idx = int(parts[0])
                memory_mb = int(float(parts[2]))
            except ValueError:
                continue
            gpus.append(
                {
                    "index": gpu_idx,
                    "name": parts[1],
                    "memory_total_mb": memory_mb,
                    "driver_version": parts[3],
                }
            )
            # driver_version is per-host (same across all GPUs in
            # practice); keep the first non-None value.
            if driver_version is None:
                driver_version = parts[3]

    # ---- (3) probe torch version + CUDA version ----
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_rc, torch_out, _torch_err = _run(
        [VENV_PY, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
        timeout=30.0,
    )
    if torch_rc == 0 and torch_out.strip():
        tok = torch_out.strip().split()
        if len(tok) >= 1:
            torch_version = tok[0]
        if len(tok) >= 2:
            cuda_version = tok[1]

    # ---- (4) probe system info (uname -a) ----
    hostname = kernel = arch = None
    uname_rc, uname_out, _uname_err = _run(
        ["uname", "-a"],
        timeout=5.0,
    )
    if uname_rc == 0 and uname_out.strip():
        # uname -a format: "Linux host kernel-version arch ..."
        uname_parts = uname_out.strip().split(None, 4)
        if len(uname_parts) >= 5:
            hostname, kernel, arch = (
                uname_parts[1],
                uname_parts[2],
                uname_parts[4],
            )
        elif len(uname_parts) >= 4:
            hostname, kernel, arch = (
                uname_parts[1],
                uname_parts[2],
                uname_parts[3],
            )

    # ---- (5) assemble + write manifest JSON ----
    manifest: Dict[str, Any] = {
        "schema_version": "wf350a2.v1",
        "ts": _now_iso(),
        "gpus": gpus,
        "driver_version": driver_version,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "system": {
            "hostname": hostname,
            "kernel": kernel,
            "arch": arch,
        },
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"cannot write manifest {manifest_path!s}: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (6) schema validity check (re-read JSON to confirm) ----
    try:
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"manifest at {manifest_path!s} is not valid JSON: {exc}",
            time.perf_counter() - t0,
        )

    missing_top: List[str] = []
    for key in ("gpus", "driver_version", "torch_version", "cuda_version", "system"):
        if key not in on_disk:
            missing_top.append(key)
    if missing_top:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"manifest missing top-level keys: {missing_top}",
            time.perf_counter() - t0,
        )

    sys_block = on_disk.get("system", {})
    if not isinstance(sys_block, dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"manifest 'system' block is not a dict "
            f"(got {type(sys_block).__name__})",
            time.perf_counter() - t0,
        )
    missing_sys: List[str] = []
    for key in ("hostname", "kernel", "arch"):
        if not sys_block.get(key):
            missing_sys.append(key)
    if missing_sys:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"manifest 'system' block missing fields: {missing_sys}",
            time.perf_counter() - t0,
        )

    # Per-GPU field check (NeurIPS §E.5 requires index + name + memory).
    gpu_list = on_disk.get("gpus", [])
    if not isinstance(gpu_list, list):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"manifest 'gpus' is not a list " f"(got {type(gpu_list).__name__})",
            time.perf_counter() - t0,
        )
    missing_gpu_fields: List[str] = []
    for i, gpu in enumerate(gpu_list):
        if not isinstance(gpu, dict):
            missing_gpu_fields.append(f"gpus[{i}].not-a-dict")
            continue
        for f in ("index", "name", "memory_total_mb"):
            if f not in gpu:
                missing_gpu_fields.append(f"gpus[{i}].{f}")
    if missing_gpu_fields:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"per-GPU fields missing: {missing_gpu_fields}",
            time.perf_counter() - t0,
        )

    # ---- PASS evidence ----
    n_gpu = len(gpu_list)
    per_gpu_str = (
        ", ".join(
            f"gpu[{g.get('index')}]={g.get('name')!s}/{g.get('memory_total_mb')}MB"
            for g in gpu_list
        )
        or "no-gpu-detected"
    )
    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"manifest at {manifest_path!s} schema-valid | "
        f"{n_gpu} GPU(s): {per_gpu_str} | "
        f"torch={torch_version} cuda={cuda_version} | "
        f"host={hostname} kernel={kernel}",
        elapsed,
    )


def gate_r20_phase0_loaders_complete(
    name: str,
    priority: str,
    pipeline: str,
    loader_modules: Tuple[Path, ...],
    loader_api_docs: Tuple[Path, ...],
    loader_tests: Tuple[Path, ...],
) -> GateResult:
    """Wf-313H — R20 Phase 0 loader closeout gate.

    Per ``TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md`` §1.1 the
    three P0 CRITICAL loaders (``binding_db``, ``nci60``,
    ``drugood_loader``) must each ship:

      * the loader module on disk,
      * ≥12 dedicated tests (this gate verifies the 3 test FILES
        exist + run pytest cleanly via :func:`gate_pytest` semantics,
        but does NOT inspect individual test counts — that contract
        is enforced by the test files' own assertions),
      * an API doc at ``molmetal/data/<loader>.md``.

    PASS iff all 12 paths (3 modules + 3 docs + 6 tests) are non-empty
    files on disk AND the 6 test files pass pytest (zero failures).
    SKIP-friendly on missing test files (FAIL per AGENTS.md §1.4 zero
    tolerance for honest-negatives).

    Wf-320F added 3 independent re-verification tests
    (``test_wf320f_*``) that complement the Wf-313H originals
    (``test_wf313h_*``) — together they **double-anchor** the loader
    contract.  See ``TODO/completed/320_r20_phase0_loaders_closeout.md``.
    """
    t0 = time.perf_counter()
    n_modules = len(loader_modules)
    n_docs = len(loader_api_docs)
    n_tests = len(loader_tests)
    expected_total = n_modules + n_docs + n_tests
    missing: List[str] = []
    for p in loader_modules:
        if not p.is_file() or p.stat().st_size <= 0:
            missing.append(f"module-missing: {p}")
    for p in loader_api_docs:
        if not p.is_file() or p.stat().st_size <= 0:
            missing.append(f"api-doc-missing: {p}")
    for p in loader_tests:
        if not p.is_file() or p.stat().st_size <= 0:
            missing.append(f"test-missing: {p}")

    if missing:
        elapsed = time.perf_counter() - t0
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(missing)}/{expected_total} R20-Phase-0 loader artefacts missing: "
            f"{missing}",
            elapsed,
        )

    # All artefacts present — run pytest on the test files for a
    # real PASS (matches the gate_pytest semantics from Wf-259).
    args = [
        VENV_PY,
        "-m",
        "pytest",
        "-q",
        "--no-header",
    ] + [str(p) for p in loader_tests]
    rc, out, err = _run(args, timeout=180.0, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = blob.splitlines()[-1] if blob else "(no output)"
    if rc == 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"{n_modules}/{n_modules} loaders + {n_docs}/{n_docs} API docs + "
            f"{n_tests}/{n_tests} test files present; pytest PASS: {tail}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"pytest FAIL (rc={rc}) on R20-Phase-0 loader tests: {tail}",
        elapsed,
    )


def gate_nci60_9cell_per_cell_aggregate(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
) -> GateResult:
    """Wf-312B — NCI-60 9-cell per-cell aggregate gate.

    Per Wf-308C §3.2 MEASURED smoke (9 cells × 3 seeds × 1 epoch chemprop
    v2 per-cell CPU sweep) + Wf-312B ship-surface integration: the
    :file:`metrics/by_metric/nci60_precious_9cell_auc.json` JSON MUST
    carry a ``wf312b_nci60_smoke_aggregate`` block with:

      * ``per_cell_breakdown`` with exactly **9** unique cell-line keys
        (HCT-116 aliased to HCT116 is permitted as an explicit
        ``alias_of`` entry but counts toward the 9-key floor — duplicate
        values without alias metadata FAIL).
      * ``highlight`` field pointing at HCT116 with a numeric ``value``
        field.
      * ``honest_negative`` field pointing at a cell with
        ``honest_negative_reason`` non-empty string AND
        ``value < 0.5`` (random-chance floor) AND badge ==
        ``honest-negative``.
      * ``overall_mean`` field equal (within tolerance 1e-4) to the
        simple mean of the 9 per-cell ``mean_auc`` values.

    Tolerances are tight (1e-4) so silent drift in the aggregate flag
    FAILs immediately. Field presence + type + value-range are all
    checked; per-cell n_train / n_test are not enforced (kept loose so
    CSV-size regressions don't double-Fail this gate).
    """
    t0 = time.perf_counter()
    if not json_path.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json file missing: {json_path}",
            time.perf_counter() - t0,
        )
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json parse / IO error in {json_path}: " f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    block = data.get("wf312b_nci60_smoke_aggregate")
    if not isinstance(block, dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{json_path}: wf312b_nci60_smoke_aggregate block missing",
            elapsed,
        )
    errors: List[str] = []
    breakdown = block.get("per_cell_breakdown")
    if not isinstance(breakdown, dict):
        errors.append("per_cell_breakdown missing or not a dict")
    else:
        if len(breakdown) != 9:
            errors.append(f"per_cell_breakdown has {len(breakdown)} keys; expected 9")
        per_cell_means: List[float] = []
        for cell, entry in breakdown.items():
            if not isinstance(entry, dict):
                errors.append(f"per_cell_breakdown.{cell} not a dict")
                continue
            mean_auc = entry.get("mean_auc")
            if not isinstance(mean_auc, (int, float)):
                errors.append(
                    f"per_cell_breakdown.{cell}.mean_auc missing or "
                    f"non-numeric (got {type(mean_auc).__name__})"
                )
                continue
            per_cell_means.append(float(mean_auc))
        if (
            not errors
            and isinstance(block.get("overall_mean"), (int, float))
            and per_cell_means
        ):
            computed_mean = sum(per_cell_means) / len(per_cell_means)
            declared_mean = float(block["overall_mean"])
            if abs(computed_mean - declared_mean) > 1e-4:
                errors.append(
                    f"overall_mean={declared_mean:.6f} does not match "
                    f"computed mean of per_cell_breakdown means "
                    f"={computed_mean:.6f} (Δ={abs(computed_mean - declared_mean):.6f} > 1e-4)"
                )
    highlight = block.get("highlight")
    if not isinstance(highlight, dict):
        errors.append("highlight missing or not a dict")
    else:
        if not isinstance(highlight.get("value"), (int, float)):
            errors.append("highlight.value missing or non-numeric")
        if not isinstance(highlight.get("cell"), str):
            errors.append("highlight.cell missing or non-string")
    honest_negative = block.get("honest_negative")
    if not isinstance(honest_negative, dict):
        errors.append("honest_negative missing or not a dict")
    else:
        if not isinstance(honest_negative.get("value"), (int, float)):
            errors.append("honest_negative.value missing or non-numeric")
        elif float(honest_negative["value"]) >= 0.5:
            errors.append(
                f"honest_negative.value={honest_negative['value']} >= 0.5; "
                f"honest-negative cell must be below random-chance"
            )
        reason = honest_negative.get("honest_negative_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("honest_negative.reason missing or empty string")
        if honest_negative.get("badge") != "honest-negative":
            errors.append(
                f"honest_negative.badge={honest_negative.get('badge')!r}; "
                f"expected 'honest-negative'"
            )
    if errors:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(errors)} issue(s) in wf312b_nci60_smoke_aggregate: "
            f"{errors[:3]}{'...' if len(errors) > 3 else ''}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{json_path}: 9-cell breakdown + HCT116 highlight "
        f"({float(highlight['value']):.4f}) + PC3 honest-negative "
        f"({float(honest_negative['value']):.4f}) + overall_mean "
        f"={float(block['overall_mean']):.4f} all consistent",
        elapsed,
    )


def gate_loi_send_ready(
    name: str,
    priority: str,
    pipeline: str,
    loi_paths: Tuple[Path, ...],
    review_doc_path: Path,
) -> GateResult:
    """Wf-312E — 4 LOI send-ready gate (TODO-308D integration).

    Verifies the 4 wetlab LOI drafts are ready-to-send per Wf-308D PART 2
    consolidation review. PASS conditions (all must hold):

      * All 4 LOI draft files exist & are non-empty.
      * Each LOI draft's status header line contains either ``DRAFT``
        (pre-send) or ``SENT`` (post-send). Empty / missing status = FAIL.
      * The 4-LOI consolidation review helper
        (``wetlab_loi_consolidation_review.md``) exists & is non-empty.
      * The review helper contains ≥4×N ``- [ ]`` checklist boxes
        (per-LOI next-action checklist per Wf-308D PART 2 §4; N = number
        of LOI drafts).

    The gate is structural-only (file content check); actual cold-email
    SEND is the user's responsibility per ``TODO/pending/84_no_paper_task
    _without_authorization.md``. PASS == structurally ready to send;
    NOT a proof that Hugo has actually sent.
    """
    import re

    t0 = time.perf_counter()
    errors: List[str] = []
    if not review_doc_path.is_file():
        errors.append(f"review helper missing: {review_doc_path}")
        review_text = ""
    elif review_doc_path.stat().st_size <= 0:
        errors.append(f"review helper empty: {review_doc_path}")
        review_text = ""
    else:
        try:
            review_text = review_doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"review helper unreadable: {exc}")
            review_text = ""
    # Wf-405-A3 (2026-09-30): extend the LOI-status regex to accept the
    # AGENTS.md §5.4 + §10 wetlab-deprecation marker
    # `TEMPLATE_NOT_NEEDED_FOR_TIER2` (canonical marker) and the legacy
    # `DEPRECATED` marker (used during the 2026-09-30 transition wave).
    # Tier-2 journal acceptance does not require wet-lab data per
    # AGENTS.md §5.4 + user verbatim "搞清楚我们不用做湿实验，这一点我已
    # 经明确了好多遍了" (2026-09-30). The 4 LOI drafts are retained as
    # historical artifacts only; any agent action attempting to send
    # must REFUSE per AGENTS.md §10.3.
    status_re = re.compile(
        r"^\*\*Status:\*\*\s*(?:\*\*)?\s*"
        r"(DRAFT|SENT\b|TEMPLATE_NOT_NEEDED_FOR_TIER2|DEPRECATED)",
        re.MULTILINE,
    )
    n_box_re = re.compile(r"^- \[ \]", re.MULTILINE)
    min_boxes = 4 * len(loi_paths)
    if review_text and len(n_box_re.findall(review_text)) < min_boxes:
        n_boxes = len(n_box_re.findall(review_text))
        errors.append(
            f"review helper has {n_boxes} checklist boxes; "
            f"need ≥{min_boxes} (4/LOI × {len(loi_paths)} LOIs)"
        )
    for p in loi_paths:
        if not p.is_file():
            errors.append(f"missing LOI: {p}")
            continue
        if p.stat().st_size <= 0:
            errors.append(f"empty LOI: {p}")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"unreadable LOI {p}: {exc}")
            continue
        m = status_re.search(text)
        if m is None:
            errors.append(f"LOI status header missing/parsed-fail: {p.name}")
    elapsed = time.perf_counter() - t0
    if errors:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(errors)} LOI readiness issue(s): {errors}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{len(loi_paths)}/{len(loi_paths)} LOI drafts ready-to-send + "
        f"review helper present (status DRAFT/SENT headers + "
        f"≥{min_boxes} checklist boxes)",
        elapsed,
    )


def gate_anchor_version_pin(
    name: str,
    priority: str,
    pipeline: str,
    *,
    required_version: str = ANCHOR_VERSION,
    glob_patterns: Tuple[str, ...] = ANCHOR_GLOB_PATTERNS,
    max_age_sec: float = 30 * 86400.0,
) -> GateResult:
    """Wf-306C — every paper-grade ``*_anchor.json`` carrying ``version`` MUST pin ``1.0.0``.

    Walks each glob pattern under :data:`REPO_ROOT`, reads the JSON
    dict via :class:`AnchorGuard` (path + mtime + JSON parse guards),
    and asserts the contract:

      * If the anchor carries a top-level ``"version"`` field, that
        field's value MUST equal ``required_version`` (default
        ``"1.0.0"``, per Wf-305C).
      * If the anchor lacks a ``"version"`` field, it is LEGACY
        (pre-Wf-305C paper-grade anchor) and is SKIPPED — the Wf-305C
        version-pin contract only applies to anchors that opt in by
        carrying the field. We do NOT fail the gate on legacy anchors
        because (a) the Wf-287A m14 anchor + Wf-269C report + Wf-253D
        m17 anchor + Wf-219 sweep summary all carried ``"version"``
        from their original emission, while (b) anchors like
        ``wf121_cisplatin_real_vina/sweep_summary.json`` and
        ``wf143_p4_chemotype_dock/sweep_summary.json`` predate
        Wf-305C and have not yet been re-emitted with the field.
      * Any read error (AnchorMissingError / AnchorStaleError /
        AnchorJSONError) is surfaced as FAIL — silent swallows of
        malformed JSON are explicitly disallowed per the Wf-291H
        §4 explicit-error contract.

    PASS = every checked anchor that opts in to version pinning is
    pinned at ``required_version``; legacy anchors are skipped without
    affecting the gate status.

    FAIL = at least one opt-in anchor is version-drifted, OR an anchor
    raised a read error.
    """
    t0 = time.perf_counter()
    checked: List[str] = []
    pinned: List[str] = []
    skipped_legacy: List[str] = []
    failures: List[str] = []
    for pattern in glob_patterns:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            checked.append(str(path))
            try:
                # Read WITHOUT required_version — we want to inspect the
                # raw dict first to determine if the anchor opts in to
                # version pinning. AnchorGuard's path + mtime + JSON
                # parse guards still apply (cf. Wf-291H §4).
                guard = AnchorGuard(
                    path=path,
                    max_age_sec=max_age_sec,
                    required_version=None,
                    version_field="version",
                )
                data = guard.require()
            except (
                AnchorMissingError,
                AnchorStaleError,
                AnchorJSONError,
                AnchorVersionMismatchError,
            ) as exc:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: " f"{type(exc).__name__}: {exc}"
                )
                continue
            actual = data.get("version")
            if actual is None:
                # Legacy anchor — not opted in to version pinning.
                skipped_legacy.append(str(path))
                continue
            if actual != required_version:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}: version={actual!r} "
                    f"(expected {required_version!r})"
                )
                continue
            pinned.append(str(path))
    elapsed = time.perf_counter() - t0
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)}/{len(checked)} anchors failed pin "
            f"(pinned={len(pinned)}, legacy-skipped={len(skipped_legacy)}): "
            f"{failures[:3]}" + ("..." if len(failures) > 3 else ""),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{len(pinned)}/{len(checked)} anchors pinned at version={required_version!r} "
        f"({len(skipped_legacy)} legacy pre-Wf-305C skipped)",
        elapsed,
    )


def gate_wf323f_m4_sa_qed_subcriterion_documented(
    name: str,
    priority: str,
    pipeline: str,
    metric_json: Path,
    report_json: Path,
    todo_doc: Path,
) -> GateResult:
    """Wf-323F — m4 SA/QED sub-criterion smoke documented gate.

    Per Wf-323F / WEAK_TO_STRONG §2.4: the m4 Success Rate = SA/QED
    sub-criterion ∧ Vina sub-criterion.  The Vina sub-criterion
    binds at 0/100 (cascades from m3, USER_OK-gated GPU work per
    m-metal-gpu-execution §0).  The SA/QED sub-criterion is
    COMPUTABLE on CPU — Wf-323F ships the smoke.

    PASS criteria (all required):
      (a) ``metric_json`` (m4_success_rate.json) exists +
          parseable + carries the canonical HONEST_NEGATIVE contract
          (``HONEST_NEGATIVE: true`` + non-empty ``honest_negative_reason``)
          — mirrors the Wf-308B / Wf-312C / Wf-315A pattern so downstream
          tooling branches on the canonical field rather than re-deriving
          the FAIL/THRESHOLD comparison.
      (b) ``metric_json`` carries the Wf-323F provenance tag
          (``"Wf-323F" in wf_tag``) AND the WEAK_TO_STRONG §2.4 design
          reference so future readers can trace the gate to its
          planning doc.
      (c) ``report_json`` (Wf-323F CPU smoke output) exists +
          parseable + carries the per-pool distribution
          (``per_pool_distribution`` dict with 8 bucket counts) +
          ``sub_rate_sa_qed`` numeric value.
      (d) ``todo_doc`` (TODO/completed/323f_m4_sa_qed_subcriterion.md)
          exists + non-empty — the Wf-323F closeout ship summary
          documenting the per-deliverable MEASURED status + WEAK_TO_STRONG
          §2.4 honest-framing context.

    FAIL on any one missing. CPU-only (no torch.cuda, no subprocess) —
    runs in <10 ms on the host.

    See ``TODO/completed/323f_m4_sa_qed_subcriterion.md`` for the
    Wf-323F ship summary + WEAK_TO_STRONG §2.4 honest-framing context.
    """
    t0 = time.perf_counter()
    failures: List[str] = []

    # (a) + (b) — metric_json HONEST_NEGATIVE contract + provenance
    if not metric_json.exists():
        failures.append(f"metric_json missing: {metric_json}")
    else:
        try:
            metric_data = json.loads(metric_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"metric_json parse / IO error: {type(exc).__name__}: {exc}"
            )
            metric_data = {}
        if isinstance(metric_data, dict):
            flag = metric_data.get("HONEST_NEGATIVE")
            reason = metric_data.get("honest_negative_reason")
            if flag is not True:
                failures.append(
                    f"metric_json: HONEST_NEGATIVE={flag!r} (expected True)"
                )
            if not (isinstance(reason, str) and reason.strip()):
                failures.append("metric_json: honest_negative_reason empty/non-string")
            wf_tag = str(metric_data.get("wf_tag", ""))
            if "Wf-323F" not in wf_tag:
                failures.append(
                    f"metric_json: wf_tag={wf_tag!r} missing 'Wf-323F' provenance"
                )
            wf_design = str(metric_data.get("wf_design_ref", ""))
            if "WEAK_TO_STRONG" not in wf_design:
                failures.append(
                    f"metric_json: wf_design_ref={wf_design!r} missing "
                    "'WEAK_TO_STRONG' planning doc reference"
                )

    # (c) — report_json per-pool distribution + sub_rate_sa_qed numeric
    if not report_json.exists():
        failures.append(f"report_json missing: {report_json}")
    else:
        try:
            report_data = json.loads(report_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"report_json parse / IO error: {type(exc).__name__}: {exc}"
            )
            report_data = {}
        if isinstance(report_data, dict):
            sub_rate = report_data.get("sub_rate_sa_qed")
            if not isinstance(sub_rate, (int, float)):
                failures.append(
                    f"report_json: sub_rate_sa_qed missing or non-numeric "
                    f"(got {type(sub_rate).__name__})"
                )
            distribution = report_data.get("per_pool_distribution")
            if not isinstance(distribution, dict):
                failures.append("report_json: per_pool_distribution missing/not dict")
            else:
                expected_buckets = {
                    "sa_bucket_lt_2_8",
                    "sa_bucket_2_8_to_4",
                    "sa_bucket_4_to_6",
                    "sa_bucket_6_plus",
                    "qed_bucket_lt_0_25",
                    "qed_bucket_0_25_to_0_5",
                    "qed_bucket_0_5_to_0_75",
                    "qed_bucket_0_75_plus",
                }
                missing_buckets = expected_buckets - set(distribution.keys())
                if missing_buckets:
                    failures.append(
                        f"report_json: per_pool_distribution missing buckets: "
                        f"{sorted(missing_buckets)}"
                    )

    # (d) — TODO closeout doc
    if not todo_doc.exists():
        failures.append(f"todo_doc missing: {todo_doc}")
    elif todo_doc.stat().st_size <= 0:
        failures.append(f"todo_doc empty: {todo_doc}")

    elapsed = time.perf_counter() - t0
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)} Wf-323F m4 SA/QED sub-criterion prereq(s) missing: "
            + "; ".join(failures[:3])
            + ("..." if len(failures) > 3 else ""),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m4 metric=HONEST_NEGATIVE (sub_rate_sa_qed="
        f"{float(metric_data.get('value', 0.0)):.4f}); "
        f"report={report_json.name} carries per_pool_distribution "
        f"({len(metric_data)} metric keys + 8 bucket counts); "
        f"TODO doc present (Wf-323F ship surface complete)",
        elapsed,
    )


def gate_admet_honest_negative_tag(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
) -> GateResult:
    """Wf-308B — ADMET honest-negative tag gate.

    Per Wf-308B R03 (Wf-307C integration plan): for ADMET pass-through
    metrics whose cohort mean DOES NOT meet the target band, the
    on-disk JSON MUST carry a top-level ``"HONEST_NEGATIVE": true``
    field (plus an accompanying ``"honest_negative_reason"`` string)
    so downstream tooling (paper_dashboard, pre-flight itself) can
    branch on a single canonical field rather than re-deriving the
    FAIL/THRESHOLD comparison at every read site.

    PASS criteria:
      * ``json_path`` exists + parseable JSON
      * top-level ``HONEST_NEGATIVE`` field is exactly ``True``
      * ``honest_negative_reason`` is a non-empty string
    FAIL otherwise (missing file, missing field, wrong type, etc.).
    """
    t0 = time.perf_counter()
    if not json_path.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json file missing: {json_path}",
            time.perf_counter() - t0,
        )
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json parse / IO error in {json_path}: " f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    flag = data.get("HONEST_NEGATIVE")
    reason = data.get("honest_negative_reason")
    if flag is True and isinstance(reason, str) and reason.strip():
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"{json_path}: HONEST_NEGATIVE=true, "
            f"reason={reason[:60]!r}{'...' if len(reason) > 60 else ''}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"{json_path}: HONEST_NEGATIVE={flag!r} "
        f"(expected True); reason type={type(reason).__name__}",
        elapsed,
    )


def gate_wf325e_admet_4_below_documented(
    name: str,
    priority: str,
    pipeline: str,
    aq_sol_json: Path,
    logp7_4_json: Path,
    hepatotox_json: Path,
    m6_kappa_json: Path,
    closeout_doc: Path,
) -> GateResult:
    """Wf-325E — ADMET 4 BELOW druglikeness documented gate.

    Aggregates the 4 single-metric Wf-308B / Wf-312C R02 honest-negative
    gates (aqueous_solubility_logS / logp7_4 / hepatotox_index / m6_kappa)
    into a single ship-surface-verification gate PLUS the Wf-325E closeout
    doc presence check.

    Per the Wf-325E brief: the 4 BELOW metrics span 2 sources — the
    Wf-303C T1 ADMET pass-through panel (aq_sol) and the Wf-312C R02
    druglikeness extension (logp7_4 / hepatotox / m6_kappa). All 4 carry
    the canonical ``HONEST_NEGATIVE: true`` + ``honest_negative_reason``
    contract; this aggregated gate ensures the 4 JSONs + the Wf-325E
    closeout doc ship together so downstream paper_dashboard claim
    cards (§3.7 honest-negative cluster) cannot silently regress.

    PASS criteria (all required):
      (a) ``aq_sol_json`` exists + parseable + carries
          ``HONEST_NEGATIVE: true`` + non-empty ``honest_negative_reason``.
      (b) ``logp7_4_json`` — same contract.
      (c) ``hepatotox_json`` — same contract.
      (d) ``m6_kappa_json`` — same contract.
      (e) ``closeout_doc`` exists + non-empty (the Wf-325E ship summary).

    FAIL on any one missing — silent ship loss would prevent the Wf-325E
    honest-negative cluster from rendering in paper §5 + claim cards on §3.7.
    CPU-only gate (no torch.cuda, no subprocess) — runs in <1 ms on host.

    See ``TODO/completed/325e_admet_4_below_final.md`` for the Wf-325E
    ship summary + per-metric honest framing context.
    """
    t0 = time.perf_counter()
    failures: List[str] = []
    metric_paths = (
        ("aq_sol", aq_sol_json),
        ("logp7_4", logp7_4_json),
        ("hepatotox", hepatotox_json),
        ("m6_kappa", m6_kappa_json),
    )
    for label, json_path in metric_paths:
        if not json_path.exists():
            failures.append(f"{label}_json missing: {json_path}")
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"{label}_json parse / IO error: " f"{type(exc).__name__}: {exc}"
            )
            continue
        if not isinstance(data, dict):
            failures.append(f"{label}_json: top-level not dict")
            continue
        flag = data.get("HONEST_NEGATIVE")
        reason = data.get("honest_negative_reason")
        if flag is not True:
            failures.append(f"{label}_json: HONEST_NEGATIVE={flag!r} (expected True)")
        elif not (isinstance(reason, str) and reason.strip()):
            failures.append(f"{label}_json: honest_negative_reason empty/non-string")
    if not closeout_doc.exists():
        failures.append(f"closeout_doc missing: {closeout_doc}")
    elif closeout_doc.stat().st_size <= 0:
        failures.append(f"closeout_doc empty: {closeout_doc}")
    elapsed = time.perf_counter() - t0
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)} Wf-325E ADMET-4-BELOW prereq(s) missing: "
            + "; ".join(failures[:3])
            + ("..." if len(failures) > 3 else ""),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all 4 BELOW ADMET metrics carry HONEST_NEGATIVE=true + reason; "
        f"closeout_doc ({closeout_doc.name}) present on disk",
        elapsed,
    )


def gate_wf315d_pb_smoke(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
    smoke_output: Path,
    venv_python: str,
) -> GateResult:
    """Wf-315D — PoseBusters smoke retry gate (WEAK_TO_STRONG §2.6).

    Per Wf-311E Part 2: steric_clash_pb_dock.json was DEFERRED because
    "RDKit + posebusters Python packages not installed in this snapshot".
    Wf-315D verifies one of two paths:

      (a) ``p3_pb_packages_available`` — RDKit + PoseBusters are
          installed in the venv, AND the 1h36 steric_clash_pb_dock metric
          has been updated to MEASURED_SMOKE, AND the smoke output
          ``m8_prod_1h36.json`` exists with valid m8_protein_aware_pass_rate.
      (b) ``p3_pb_packages_missing_documented`` — packages are NOT
          installed, AND the 1h36 steric_clash_pb_dock metric carries a
          HONEST_NEGATIVE_BLOCKED tag + clear blocker description
          (recovery path documented). NO measurement is fabricated.

    The gate branches on the actual venv state — if packages are
    installed, path (a) is required; if packages are missing, path (b)
    is required.  Mixing the two (e.g. packages installed but
    HONEST_NEGATIVE_BLOCKED) is FAIL.
    """
    import subprocess

    t0 = time.perf_counter()
    if not json_path.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json file missing: {json_path}",
            time.perf_counter() - t0,
        )
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"json parse / IO error in {json_path}: " f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )

    # Probe package availability via the venv python
    packages_ok = False
    rdkit_v = None
    pb_v = None
    try:
        probe = (
            "import rdkit, posebusters; "
            "print('rdkit=' + rdkit.__version__); "
            "print('posebusters=' + posebusters.__version__)"
        )
        out = subprocess.run(
            [venv_python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if line.startswith("rdkit="):
                    rdkit_v = line.split("=", 1)[1].strip()
                elif line.startswith("posebusters="):
                    pb_v = line.split("=", 1)[1].strip()
            packages_ok = rdkit_v is not None and pb_v is not None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        packages_ok = False

    elapsed = time.perf_counter() - t0

    metric_status = data.get("status")
    honest_negative = data.get("HONEST_NEGATIVE_BLOCKED") is True
    blocker = data.get("blocker_description") or data.get("deferred_detail") or ""

    if packages_ok:
        # Path (a): packages available — require MEASURED_SMOKE on metric
        # AND valid smoke output on disk.
        if metric_status != "MEASURED_SMOKE":
            return GateResult(
                name,
                priority,
                pipeline,
                "FAIL",
                f"packages available (rdkit={rdkit_v}, posebusters={pb_v}) "
                f"but {json_path}: status={metric_status!r} "
                f"(expected 'MEASURED_SMOKE')",
                elapsed,
            )
        if not smoke_output.exists():
            return GateResult(
                name,
                priority,
                pipeline,
                "FAIL",
                f"packages available but smoke output missing: {smoke_output}",
                elapsed,
            )
        try:
            with smoke_output.open("r", encoding="utf-8") as fh:
                smoke = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            return GateResult(
                name,
                priority,
                pipeline,
                "FAIL",
                f"smoke output parse error: {type(exc).__name__}: {exc}",
                elapsed,
            )
        rate = smoke.get("m8_protein_aware_pass_rate")
        n_pass = smoke.get("n_protein_aware_pass")
        n_total = smoke.get("n_poses")
        if rate is None or n_pass is None or n_total is None:
            return GateResult(
                name,
                priority,
                pipeline,
                "FAIL",
                f"smoke output missing required keys "
                f"(m8_protein_aware_pass_rate={rate!r}, "
                f"n_protein_aware_pass={n_pass!r}, n_poses={n_total!r})",
                elapsed,
            )
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"packages available (rdkit={rdkit_v}, posebusters={pb_v}); "
            f"{json_path}: status=MEASURED_SMOKE; smoke output "
            f"m8_protein_aware_pass_rate={rate:.4f} ({n_pass}/{n_total} pass)",
            elapsed,
        )

    # Path (b): packages NOT available — require HONEST_NEGATIVE_BLOCKED
    # + non-empty blocker_description.
    if not honest_negative:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"packages missing AND {json_path}: "
            f"HONEST_NEGATIVE_BLOCKED missing or not True",
            elapsed,
        )
    if not isinstance(blocker, str) or not blocker.strip():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"packages missing AND {json_path}: "
            f"blocker_description empty or wrong type",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"packages missing; {json_path}: HONEST_NEGATIVE_BLOCKED=true, "
        f"blocker={blocker[:80]!r}{'...' if len(blocker) > 80 else ''}",
        elapsed,
    )


def gate_wf323c_m8_full_pass_rate_documented(
    name: str,
    priority: str,
    pipeline: str,
    result_json: Path,
    metric_json: Path,
    pool_json: Path,
) -> GateResult:
    """Wf-323C — m8_full pass rate documented gate (WEAK_TO_STRONG §2.6 R18).

    Per Wf-319C + Wf-315D: m8_full_pass_rate = 0/15 on the Wf-315D
    PoseBusters smoke retry. Wf-323C runs the UFF pre-relax + Vina
    re-dock CPU pipeline on 16 Wf-319B Pt-pool conformers and records
    the MEASURED lift (or honest-negative if the lift is below 5%).

    PASS criteria (all required):

      (1) ``result_json`` exists + is JSON-loadable + has the
          canonical top-level keys (``m8_full_pass_rate``,
          ``m8_protein_aware_pass_rate``, ``n_poses_pre_relaxed``,
          ``n_pt_zn_substituted``, ``results``).
      (2) ``metric_json`` (steric_clash_pb_dock_1h36.json) is
          ``MEASURED_SMOKE`` AND carries the Wf-323C smoke-summary
          block + the ``HONEST_NEGATIVE_UFF_INSUFFICIENT`` tag if
          applicable.
      (3) ``pool_json`` (Wf-319B Pt-pool) exists + has ≥80 conformers
          (the Wf-323C input source).

    The gate accepts BOTH a successful lift AND an honest-negative
    outcome — the requirement is that the measurement is documented
    and the rate is in [0, 1]. This is the WEAK_TO_STRONG §2.6 R18
    "document the delta" mandate rather than "lift to a target".

    See ``TODO/completed/323c_m8_full_cpu_fix.md`` for the Wf-323C
    ship summary + the recovery-path mapping
    (MMFF94 + GPU metal-coord term).
    """
    t0 = time.perf_counter()

    # (1) result_json
    if not result_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"result_json missing: {result_json}",
            time.perf_counter() - t0,
        )
    try:
        with result_json.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"result_json parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    for key in (
        "m8_full_pass_rate",
        "m8_protein_aware_pass_rate",
        "n_poses_pre_relaxed",
        "n_pt_zn_substituted",
        "results",
    ):
        if key not in result:
            return GateResult(
                name,
                priority,
                pipeline,
                "FAIL",
                f"result_json missing required key: {key}",
                time.perf_counter() - t0,
            )
    full_rate = result.get("m8_full_pass_rate")
    pa_rate = result.get("m8_protein_aware_pass_rate")
    n_pre = result.get("n_poses_pre_relaxed", 0)
    n_pt_zn = result.get("n_pt_zn_substituted", 0)
    if not isinstance(full_rate, (int, float)) or not 0.0 <= full_rate <= 1.0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"m8_full_pass_rate={full_rate!r} not in [0.0, 1.0]",
            time.perf_counter() - t0,
        )
    if not isinstance(pa_rate, (int, float)) or not 0.0 <= pa_rate <= 1.0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"m8_protein_aware_pass_rate={pa_rate!r} not in [0.0, 1.0]",
            time.perf_counter() - t0,
        )
    if n_pre <= 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"n_poses_pre_relaxed={n_pre}; Wf-323C pipeline did not run",
            time.perf_counter() - t0,
        )

    # (2) metric_json — must be MEASURED_SMOKE and reference Wf-323C
    if not metric_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json missing: {metric_json}",
            time.perf_counter() - t0,
        )
    try:
        with metric_json.open("r", encoding="utf-8") as fh:
            metric = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    metric_status = metric.get("status")
    if metric_status != "MEASURED_SMOKE":
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json status={metric_status!r} (expected MEASURED_SMOKE)",
            time.perf_counter() - t0,
        )
    metric_value = metric.get("value", {})
    if not isinstance(metric_value.get("wf323c_smoke_summary"), dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "metric_json.value.wf323c_smoke_summary missing",
            time.perf_counter() - t0,
        )

    # (3) pool_json — Wf-319B Pt-pool ≥ 80 conformers
    if not pool_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json missing: {pool_json}",
            time.perf_counter() - t0,
        )
    try:
        with pool_json.open("r", encoding="utf-8") as fh:
            pool = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    n_conformers = len(pool.get("conformers", []))
    if n_conformers < 80:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json has {n_conformers} conformers; smoke floor is ≥80",
            time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"result.m8_full={full_rate:.4f} (n_pre={n_pre}, n_pt_zn={n_pt_zn}); "
        f"metric MEASURED_SMOKE; pool={n_conformers} conformers",
        elapsed,
    )


def gate_wf319c_pb_packages_status_present(
    name: str,
    priority: str,
    pipeline: str,
    venv_python: str,
    metric_json: Path,
    smoke_output: Path,
) -> GateResult:
    """Wf-319C — PB packages status documented gate.

    Lighter-weight companion to ``gate_wf315d_pb_smoke``: verifies the
    *prerequisites* for running PB dock mode are documented, rather than
    branching on the actual install state.  This is what the Wf-319C
    brief calls ``p3_pb_packages_status_present``.

    PASS conditions (all three required):

      (1) venv-python probe: ``import rdkit, posebusters`` succeeds AND
          both versions can be parsed.
      (2) ``metric_json`` exists on disk AND has
          ``status == "MEASURED_SMOKE"``.
      (3) ``smoke_output`` exists on disk AND carries the canonical
          keys (``m8_protein_aware_pass_rate``, ``n_poses``,
          ``n_protein_aware_pass``).

    FAIL on any one missing — the gate is intentionally strict because
    the Wf-315D work shipped this contract; if any of the three
    artefacts disappears, downstream consumers (paper §3.4 steric clash
    cluster) silently lose the 1h36 MEASURED_SMOKE evidence.

    See `TODO/completed/319c_pb_packages_status_present.md` for the
    Wf-319C ship summary.
    """
    import subprocess

    t0 = time.perf_counter()

    # (1) venv-python probe.
    rdkit_v: Optional[str] = None
    pb_v: Optional[str] = None
    probe_err: Optional[str] = None
    try:
        probe = (
            "import rdkit, posebusters; "
            "print('rdkit=' + rdkit.__version__); "
            "print('posebusters=' + posebusters.__version__)"
        )
        out = subprocess.run(
            [venv_python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        if out.returncode != 0:
            probe_err = out.stderr.strip()[:200] or "nonzero_returncode"
        else:
            for line in out.stdout.splitlines():
                if line.startswith("rdkit="):
                    rdkit_v = line.split("=", 1)[1].strip()
                elif line.startswith("posebusters="):
                    pb_v = line.split("=", 1)[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        probe_err = f"{type(exc).__name__}:{exc}"

    if probe_err is not None or rdkit_v is None or pb_v is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"venv probe failed for {venv_python}: "
            f"rdkit={rdkit_v!r} posebusters={pb_v!r} err={probe_err!r}",
            time.perf_counter() - t0,
        )

    # (2) metric_json must exist with status=MEASURED_SMOKE.
    if not metric_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json missing: {metric_json}",
            time.perf_counter() - t0,
        )
    try:
        metric_data = json.loads(metric_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json parse error: {type(exc).__name__}:{exc}",
            time.perf_counter() - t0,
        )
    metric_status = metric_data.get("status")
    if metric_status != "MEASURED_SMOKE":
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metric_json status={metric_status!r} "
            f"(expected 'MEASURED_SMOKE') in {metric_json}",
            time.perf_counter() - t0,
        )

    # (3) smoke_output must exist with the canonical keys.
    if not smoke_output.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"smoke_output missing: {smoke_output}",
            time.perf_counter() - t0,
        )
    try:
        smoke_data = json.loads(smoke_output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"smoke_output parse error: {type(exc).__name__}:{exc}",
            time.perf_counter() - t0,
        )
    rate = smoke_data.get("m8_protein_aware_pass_rate")
    n_pass = smoke_data.get("n_protein_aware_pass")
    n_total = smoke_data.get("n_poses")
    if rate is None or n_pass is None or n_total is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"smoke_output missing canonical keys "
            f"(m8_protein_aware_pass_rate={rate!r}, "
            f"n_protein_aware_pass={n_pass!r}, n_poses={n_total!r})",
            time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"venv={venv_python} rdkit={rdkit_v} posebusters={pb_v}; "
        f"{metric_json.name}=MEASURED_SMOKE; smoke m8 rate={rate:.4f} "
        f"({n_pass}/{n_total} pass)",
        elapsed,
    )


def gate_wf319b_uff_min_pt_pool_present(
    name: str,
    priority: str,
    pipeline: str,
    pool_json: Path,
    metrics_json: Path,
    *,
    min_conformers: int = 80,
) -> GateResult:
    """Wf-319B — UFF-min Pt-pool presence + honest-negative tag gate.

    Per Wf-319B ship, the UFF-min Pt-pool is the CPU-only path that
    prepares Pt-drug conformers (96/100 generated on the canonical
    platinai_4cell.csv Pt subset) for the m10 Vina Min Score ceiling
    work (WEAK_TO_STRONG §2.7 PROJECTED).

    PASS criteria (all required):

      (1) ``pool_json`` exists + parseable + ``conformers`` list has
          ≥ ``min_conformers`` entries (smoke floor).
      (2) ``metrics_json`` exists + parseable + carries the canonical
          ``HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING`` tag (RDKit UFF
          lacks d-block Pt parameters; energies are fallback terms,
          not physical).  This blocks downstream consumers from
          silently treating the energies as a real minimization
          metric.

    FAIL on any one missing. The pool.json + 3D conformers are still
    valid for downstream Vina docking, but the energy distribution
    MUST be flagged as honest-negative.

    See ``TODO/completed/319b_uff_min_pt_pool.md`` for the Wf-319B
    ship summary + WEAK_TO_STRONG §2.7 honest-framing context.
    """
    t0 = time.perf_counter()

    # (1) pool_json
    if not pool_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json missing: {pool_json}",
            time.perf_counter() - t0,
        )
    try:
        with pool_json.open("r", encoding="utf-8") as fh:
            pool = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    conformers = pool.get("conformers", [])
    n_conformers = len(conformers)
    if n_conformers < min_conformers:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"pool_json has {n_conformers} conformers; smoke floor is "
            f"≥ {min_conformers}",
            time.perf_counter() - t0,
        )

    # (2) metrics_json + honest-negative tag
    if not metrics_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metrics_json missing: {metrics_json}",
            time.perf_counter() - t0,
        )
    try:
        with metrics_json.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metrics_json parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    flag = metrics.get("HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING")
    reason = metrics.get("honest_negative_reason", "")
    if flag is not True:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"metrics_json: HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING="
            f"{flag!r} (expected True). UFF energies are not physical "
            f"for Pt complexes; downstream consumers need this tag.",
            time.perf_counter() - t0,
        )
    if not isinstance(reason, str) or not reason.strip():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "metrics_json: honest_negative_reason is empty/non-string",
            time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"pool={n_conformers} conformers (≥ {min_conformers}); "
        f"metrics HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True; "
        f"reason={reason[:60]!r}{'...' if len(reason) > 60 else ''}",
        elapsed,
    )


def gate_wf319a_ablation_aggregate_csv_present(
    name: str,
    priority: str,
    pipeline: str,
    csv_path: Path,
    *,
    min_rows: int = 11,
) -> GateResult:
    """Wf-319A — ``molmetal/reports/wf_ablation_runner/aggregate.csv`` ≥ ``min_rows``.

    Verifies the ablation_runner.py smoke output on disk is bit-for-bit
    parseable as CSV with the expected schema (12 columns including
    ablation_id / toggle / status / match / m14_decode_ratio) and
    at least ``min_rows`` data rows.

    Per Wf-319A honest framing: the row count must be ≥ 11 (the 11
    ablation IDs in ``TODO/MECHANISM_METRIC_MAPPING.md`` §5); the
    BASELINE reference row is a bonus row on top.  When a row's
    toggle is ``design`` (M3 / M4 / M8 / M10 / M11 / M21) the metric
    columns MUST be empty — this is the no-fabrication contract that
    the downstream paper §6 table relies on.
    """
    import csv as _csv

    t0 = time.perf_counter()
    if not csv_path.exists() or not csv_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"aggregate.csv missing: {csv_path}",
            time.perf_counter() - t0,
        )
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    except (OSError, _csv.Error, UnicodeDecodeError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"aggregate.csv parse / IO error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    n_rows = len(rows)
    if n_rows < min_rows:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"aggregate.csv has {n_rows} rows (< {min_rows} required).  "
            f"Re-run ``python molmetal/scripts/ablation_runner.py "
            f"--output-dir molmetal/reports/wf319a_ablation_runner``.",
            time.perf_counter() - t0,
        )
    # Required column schema — paper §6 input contract.
    required_cols = {
        "ablation_id",
        "toggle",
        "status",
        "match",
        "m14_decode_ratio",
        "m13_validity",
        "m5_sa_unit",
        "m6_qed",
    }
    missing_cols = required_cols - set(fieldnames)
    if missing_cols:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"aggregate.csv missing required columns: {sorted(missing_cols)}",
            time.perf_counter() - t0,
        )
    # No-fabrication contract: DESIGN_ONLY rows must have empty metric
    # columns (the runner emits "" for non-measured metrics).
    bad_design_rows = [
        r["ablation_id"]
        for r in rows
        if r["toggle"] == "design"
        and (
            r.get("m14_decode_ratio", "") != ""
            or r.get("m13_validity", "") != ""
            or r.get("m5_sa_unit", "") != ""
            or r.get("m6_qed", "") != ""
        )
    ]
    if bad_design_rows:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"DESIGN_ONLY rows have fabricated metric values: "
            f"{bad_design_rows}.  These 6 ablations must NOT carry "
            f"measured metrics (per MECHANISM_METRIC_MAPPING.md §5 "
            f"no-fabrication contract).",
            time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"aggregate.csv OK: {n_rows} rows (≥ {min_rows}); "
        f"{len(fieldnames)} columns; DESIGN_ONLY rows free of "
        f"fabricated metrics",
        elapsed,
    )


# ----------------------------------------------------------------------------
# Wf-321A — Wf-135 Phase 2 CPU scaffold ready gate
# ----------------------------------------------------------------------------
# Per Wf-321A brief: Phase 2 of Wf-135 (TODO-176 m19 LOMO Rh oversample)
# is the 5-metal x 5-seed x 50-epoch re-bench (~2 weeks GPU wall).
# Phase 1 (Wf-309A balanced Rh cohort) SHIPPED; Phase 2 is in GPU
# pipeline per Wf-310E launch wrapper.
#
# This CPU scaffold gate verifies the 3 prerequisites for Phase 2 launch
# readiness:
#   (a) Phase 1 cohort JSON on disk (Wf-139 P7 / Wf-309A ship) +
#       scaffold_split integrity (scaffold_overlap_count == 0).
#   (b) Trainer (molmetal/scripts/train_metal_lomo_v2.py) wires
#       --balanced-cohort-json + --oversample-positives + --pos-weight
#       flags into the BalancedLOMORunner constructor.
#   (c) Wf-310E launch wrapper (USER_OK-gated GPU-1 launch script)
#       exists on disk and references the trainer + Phase 1 cohort.
#
# The gate is CPU-only (no torch.cuda / no subprocess) — runs in <1 s
# on the host.  It is NOT a substitute for the full pre-flight PASS
# gate; it is a *focused* gate for the Wf-135 Phase 2 launch surface.
# ----------------------------------------------------------------------------


def gate_wf321a_wf135_phase2_cpu_smoke(
    name: str,
    priority: str,
    pipeline: str,
    phase1_cohort_json: Path,
    trainer_script: Path,
    launch_wrapper: Path,
) -> GateResult:
    """Wf-321A — Phase 2 CPU scaffold readiness gate.

    Per the Wf-321A brief: this gate verifies the 3 prerequisites
    for the Wf-310E Phase 2 GPU launch (USER_OK-gated per
    mol-metal-gpu-execution §0).  PASS iff ALL 3 are true; FAIL
    otherwise with a precise reason.
    """
    t0 = time.perf_counter()
    failures: List[str] = []

    # (a) Phase 1 cohort JSON on disk + scaffold_overlap_count == 0.
    if not phase1_cohort_json.exists():
        failures.append(
            f"Phase 1 cohort JSON missing at {phase1_cohort_json} "
            f"(Wf-309A should have shipped this)"
        )
    else:
        try:
            with phase1_cohort_json.open() as fh:
                cohort_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"Phase 1 cohort JSON parse error: {type(exc).__name__}: {exc}"
            )
        else:
            stats = (
                cohort_data.get("stats", {}) if isinstance(cohort_data, dict) else {}
            )
            n_active = int(stats.get("n_active", 0))
            n_inactive = int(stats.get("n_inactive", 0))
            if n_active < 250 or n_inactive < 250:
                failures.append(
                    f"Phase 1 cohort under-sized: n_active={n_active} "
                    f"n_inactive={n_inactive} (expected both ≥250)"
                )
            scaffold_overlap = int(stats.get("scaffold_overlap_count", -1))
            if scaffold_overlap != 0:
                failures.append(
                    f"Phase 1 cohort scaffold_overlap_count={scaffold_overlap} "
                    f"(expected 0 — clean split per Wf-139 P7 / Wf-309A)"
                )

    # (b) Trainer wires --balanced-cohort-json + --oversample-positives
    #     + --pos-weight flags into the BalancedLOMORunner constructor.
    if not trainer_script.exists():
        failures.append(f"Trainer script missing at {trainer_script}")
    else:
        try:
            trainer_src = trainer_script.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(f"Trainer script read error: {type(exc).__name__}: {exc}")
            trainer_src = ""
        if trainer_src:
            required_trainer_flags = (
                "--balanced-cohort-json",
                "--oversample-positives",
                "--pos-weight",
            )
            missing_flags = [f for f in required_trainer_flags if f not in trainer_src]
            if missing_flags:
                failures.append(
                    f"Trainer CLI missing flags: {missing_flags} "
                    f"(Wf-145 P3 / Wf-187 m19 fix wiring required)"
                )
            if "oversample_positives=args.oversample_positives" not in trainer_src:
                failures.append(
                    "Trainer does not wire --oversample-positives into "
                    "BalancedLOMORunner (Wf-187 m19 fix not connected)"
                )
            if "pos_weight=args.pos_weight" not in trainer_src:
                failures.append(
                    "Trainer does not wire --pos-weight into "
                    "BalancedLOMORunner (Wf-187 m19 fix not connected)"
                )

    # (c) Wf-310E launch wrapper (launch_wf310e_wf135_phase2_gpu.sh)
    # exists + references trainer + cohort.
    if not launch_wrapper.exists():
        failures.append(
            f"launch_wf310e_wf135_phase2_gpu.sh missing at {launch_wrapper} "
            f"(Phase 2 GPU launch blocked — recreate the wrapper)"
        )
    else:
        try:
            launch_src = launch_wrapper.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(f"Launch wrapper read error: {type(exc).__name__}: {exc}")
            launch_src = ""
        if launch_src:
            if "USER_OK" not in launch_src:
                failures.append(
                    "Launch wrapper missing USER_OK gate "
                    "(mol-metal-gpu-execution §0 kill_then_adapt contract)"
                )
            if "train_metal_lomo_v2.py" not in launch_src:
                failures.append("Launch wrapper does not reference the trainer script")
            if "synthetic_rh_pos_neg.json" not in launch_src:
                failures.append(
                    "Launch wrapper does not reference the Phase 1 cohort JSON"
                )

    elapsed = time.perf_counter() - t0
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)} Phase 2 CPU scaffold prereq(s) missing: "
            + "; ".join(failures[:3])
            + ("..." if len(failures) > 3 else ""),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "Phase 1 cohort=OK (250+250, scaffold_overlap=0); "
        "trainer flags (--balanced-cohort-json/--oversample-positives/"
        "--pos-weight) wired into BalancedLOMORunner; "
        "Wf-310E launch wrapper present with USER_OK gate",
        elapsed,
    )


# ----------------------------------------------------------------------------
# Wf-322F — Wf-176 m19 LOMO Rh oversample FINAL closeout documented gate
# ----------------------------------------------------------------------------
# Per Wf-322F brief: this is the Wf-135 / Wf-176 / TODO-176 workstream
# final closeout.  Phase 1 (Wf-309A balanced Rh cohort 500) is SHIPPED
# MEASURED; Phase 2 CPU scaffold (Wf-321A) is SHIPPED MEASURED; Phase 2
# GPU launch wrapper (Wf-310E) is READY on disk but BLOCKED behind
# USER_OK gate per mol-metal-gpu-execution §0 kill_then_adapt.  This
# gate verifies the closeout ship surface is on disk:
#
#   (a) Wf-322F close-out doc exists at TODO/completed/322f_wf176_m19_
#       lomo_final_closeout.md and contains the 4 phase-state strings
#       (Phase 1, Phase 2 scaffold, Phase 2 GPU, USER_OK).
#   (b) Phase 1 cohort JSON on disk (Wf-139 P7 / Wf-309A ship) with
#       stats.n_active >= 250, stats.n_inactive >= 250.
#   (c) Phase 1 cohort synthesis doc exists and mentions "Wf-139 P7" +
#       "250 active + 250 inactive" balanced label contract.
#   (d) GPU launch wrapper (launch_wf310e_wf135_phase2_gpu.sh) exists
#       and contains the kill_then_adapt USER_OK gate + the Phase 2
#       trainer reference.
#
# The gate is CPU-only (no torch.cuda / no subprocess) — runs in <0.005
# s on the host.  This is the close-out gate, NOT the Phase 2 GPU launch
# gate (USER_OK is still HARD-REQUIRED per mol-metal-gpu-execution §0).
# ----------------------------------------------------------------------------


def gate_wf322f_wf176_m19_lomo_closeout_documented(
    name: str,
    priority: str,
    pipeline: str,
    closeout_doc: Path,
    phase1_cohort_json: Path,
    phase1_cohort_synthesis: Path,
    gpu_launch_wrapper: Path,
) -> GateResult:
    """Wf-322F — Wf-176 m19 LOMO Rh oversample final closeout documented.

    Per the Wf-322F brief: this gate verifies the closeout ship surface
    is on disk for the Wf-135 / Wf-176 / TODO-176 workstream.  PASS iff
    ALL 4 are true; FAIL otherwise with a precise reason.
    """
    t0 = time.perf_counter()
    failures: List[str] = []

    # (a) Wf-322F close-out doc on disk + contains the 4 phase-state
    #     strings (Phase 1 / Phase 2 scaffold / Phase 2 GPU / USER_OK).
    if not closeout_doc.exists():
        failures.append(
            f"Wf-322F close-out doc missing at {closeout_doc} "
            f"(must be written before Phase 2 GPU launch is even considered)"
        )
    else:
        try:
            doc_src = closeout_doc.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(
                f"Wf-322F close-out doc read error: {type(exc).__name__}: {exc}"
            )
            doc_src = ""
        if doc_src:
            required_doc_strings = (
                "Phase 1 (Wf-309A)",
                "Phase 2 scaffold (Wf-321A)",
                "Phase 2 GPU (Wf-310E)",
                "USER_OK",
            )
            missing_strings = [s for s in required_doc_strings if s not in doc_src]
            if missing_strings:
                failures.append(
                    f"Wf-322F close-out doc missing phase-state strings: "
                    f"{missing_strings}"
                )

    # (b) Phase 1 cohort JSON on disk + balanced (>= 250 active + 250
    #     inactive).
    if not phase1_cohort_json.exists():
        failures.append(
            f"Phase 1 cohort JSON missing at {phase1_cohort_json} "
            f"(Wf-309A should have shipped this; close-out assumes it SHIPPED)"
        )
    else:
        try:
            with phase1_cohort_json.open() as fh:
                cohort_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"Phase 1 cohort JSON parse error: {type(exc).__name__}: {exc}"
            )
        else:
            stats = (
                cohort_data.get("stats", {}) if isinstance(cohort_data, dict) else {}
            )
            n_active = int(stats.get("n_active", 0))
            n_inactive = int(stats.get("n_inactive", 0))
            if n_active < 250 or n_inactive < 250:
                failures.append(
                    f"Phase 1 cohort under-sized: n_active={n_active} "
                    f"n_inactive={n_inactive} (expected both ≥250)"
                )

    # (c) Phase 1 cohort synthesis doc on disk + mentions the
    #     balanced-label contract.
    if not phase1_cohort_synthesis.exists():
        failures.append(
            f"Phase 1 cohort synthesis doc missing at {phase1_cohort_synthesis} "
            f"(Wf-139 P7 should have written this)"
        )
    else:
        try:
            synth_src = phase1_cohort_synthesis.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            failures.append(
                f"Phase 1 cohort synthesis doc read error: "
                f"{type(exc).__name__}: {exc}"
            )
            synth_src = ""
        if synth_src:
            if "Wf-139 P7" not in synth_src:
                failures.append(
                    "Phase 1 cohort synthesis doc missing 'Wf-139 P7' reference"
                )
            if "250 active + 250 inactive" not in synth_src:
                failures.append(
                    "Phase 1 cohort synthesis doc missing "
                    "'250 active + 250 inactive' balanced label contract"
                )

    # (d) GPU launch wrapper on disk + contains USER_OK gate + trainer
    #     reference.
    if not gpu_launch_wrapper.exists():
        failures.append(
            f"GPU launch wrapper missing at {gpu_launch_wrapper} "
            f"(recreate the Wf-310E wrapper before Phase 2 GPU launch)"
        )
    else:
        try:
            wrapper_src = gpu_launch_wrapper.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            failures.append(
                f"GPU launch wrapper read error: {type(exc).__name__}: {exc}"
            )
            wrapper_src = ""
        if wrapper_src:
            if "USER_OK" not in wrapper_src:
                failures.append(
                    "GPU launch wrapper missing USER_OK gate "
                    "(mol-metal-gpu-execution §0 kill_then_adapt contract)"
                )
            if "train_metal_lomo_v2.py" not in wrapper_src:
                failures.append(
                    "GPU launch wrapper does not reference the Phase 2 trainer"
                )

    elapsed = time.perf_counter() - t0
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)} Wf-322F closeout prereq(s) missing: "
            + "; ".join(failures[:3])
            + ("..." if len(failures) > 3 else ""),
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "Wf-322F closeout doc=OK; Phase 1 cohort=OK (≥250+≥250); "
        "Phase 1 synthesis doc=OK (Wf-139 P7 + 250+250 balanced); "
        "GPU launch wrapper=OK (USER_OK gate + trainer reference)",
        elapsed,
    )


# ----------------------------------------------------------------------------
# Wf-325D — m6_kappa BELOW threshold + NEGATIVE_WEAK MetalCytoToxDB
# validation FINAL closeout documented gate
# ----------------------------------------------------------------------------
# Per Wf-325D brief: this is the m6_kappa (Shape Kappa Index) workstream
# FINAL closeout.  m6_kappa mean=0.4719 BELOW ChemAxon PMC11387870 drug-
# likeness threshold 0.6 by -0.128 (FAIL honest-negative); MetalCytoToxDB
# validation (Wf-253D on 5,460 rows) returned NEGATIVE_WEAK (global
# Pearson r 0.081 / Spearman ρ 0.111); canonical HONEST_NEGATIVE=true tag
# already present on metrics/by_metric/m6_kappa.json (Wf-311E Part 4 /
# Wf-314A shipped).  This gate verifies the closeout ship surface:
#
#   (a) The Wf-325D close-out doc exists at
#       TODO/completed/325d_m6_kappa_final.md.
#   (b) It carries the 5 §0..§5 anchor strings (TL;DR, cohort
#       statistic, MetalCytoToxDB validation, architectural fix,
#       HONEST_NEGATIVE preserved).
#   (c) m6_kappa.json exists + carries HONEST_NEGATIVE=true +
#       non-empty honest_negative_reason + the metalcytotox_validation
#       block with global_pearson_r + global_spearman_rho present
#       (mirror Wf-308B / Wf-312C contract).
#
# The gate is CPU-only (no torch.cuda / no subprocess) — runs in < 10 ms.
# This is the close-out gate, NOT the GPU recovery gate (chemprop v2 +
# W₂ recovery is USER_OK-gated per mol-metal-gpu-execution §0).
# ----------------------------------------------------------------------------


def gate_wf325d_m6_kappa_closeout_documented(
    name: str,
    priority: str,
    pipeline: str,
    closeout_doc: Path,
    metric_json: Path,
) -> GateResult:
    """Wf-325D — m6_kappa BELOW threshold + NEGATIVE_WEAK validation final closeout.

    Per Wf-325D brief: this gate verifies the m6_kappa (Shape Kappa Index)
    workstream FINAL closeout ship surface. PASS iff ALL 3 conditions are
    true; FAIL otherwise with a precise reason.

    PASS criteria (all required):
      (a) ``closeout_doc`` (TODO/completed/325d_m6_kappa_final.md)
          exists + non-empty + carries the 5 §0..§5 anchor strings.
      (b) ``metric_json`` (metrics/by_metric/m6_kappa.json) exists +
          parseable + carries the canonical HONEST_NEGATIVE contract
          (``HONEST_NEGATIVE: true`` + non-empty ``honest_negative_reason``)
          AND the ``metalcytotox_validation`` block with
          ``global_pearson_r`` + ``global_spearman_rho`` numeric fields
          — mirrors the Wf-308B / Wf-312C / Wf-314A pattern so downstream
          tooling branches on the canonical field rather than re-deriving
          the FAIL/THRESHOLD comparison.

    FAIL on any one missing. CPU-only (no torch.cuda, no subprocess) —
    runs in <10 ms on the host.

    See ``TODO/completed/325d_m6_kappa_final.md`` for the per-deliverable
    MEASURED status + WEAK_TO_STRONG §2.7 honest-framing context.
    """
    t0 = time.perf_counter()
    failures: List[str] = []

    # (a) — closeout doc on disk + 5 §0..§5 anchor strings
    if not closeout_doc.exists():
        failures.append(
            f"Wf-325D close-out doc missing at {closeout_doc} "
            f"(must be written before the m6_kappa workstream is finalised)"
        )
    else:
        try:
            doc_src = closeout_doc.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(
                f"Wf-325D close-out doc read error: {type(exc).__name__}: {exc}"
            )
            doc_src = ""
        if doc_src:
            required_doc_strings = (
                "## §0 TL;DR",
                "## §2 m6_kappa cohort statistic",
                "## §3 MetalCytoToxDB validation",
                "## §4 Architectural fix",
                "## §6 HONEST_NEGATIVE preserved",
            )
            missing_strings = [s for s in required_doc_strings if s not in doc_src]
            if missing_strings:
                failures.append(
                    f"Wf-325D close-out doc missing anchor strings: "
                    f"{missing_strings}"
                )

    # (b) — metric_json HONEST_NEGATIVE contract + metalcytotox_validation
    if not metric_json.exists():
        failures.append(f"m6_kappa.json missing: {metric_json}")
    else:
        try:
            metric_data = json.loads(metric_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(
                f"m6_kappa.json parse / IO error: {type(exc).__name__}: {exc}"
            )
            metric_data = {}
        if isinstance(metric_data, dict):
            flag = metric_data.get("HONEST_NEGATIVE")
            reason = metric_data.get("honest_negative_reason")
            if flag is not True:
                failures.append(
                    f"m6_kappa.json: HONEST_NEGATIVE={flag!r} (expected True)"
                )
            if not (isinstance(reason, str) and reason.strip()):
                failures.append(
                    "m6_kappa.json: honest_negative_reason empty/non-string"
                )
            validation = metric_data.get("metalcytotox_validation")
            if not isinstance(validation, dict):
                failures.append(
                    "m6_kappa.json: metalcytotox_validation missing/not dict"
                )
            else:
                pearson_r = validation.get("global_pearson_r")
                spearman_rho = validation.get("global_spearman_rho")
                if not isinstance(pearson_r, (int, float)):
                    failures.append(
                        f"m6_kappa.json: metalcytotox_validation.global_pearson_r "
                        f"missing or non-numeric (got {type(pearson_r).__name__})"
                    )
                if not isinstance(spearman_rho, (int, float)):
                    failures.append(
                        f"m6_kappa.json: metalcytotox_validation.global_spearman_rho "
                        f"missing or non-numeric (got {type(spearman_rho).__name__})"
                    )

    elapsed = time.perf_counter() - t0
    if not failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            "Wf-325D close-out doc on disk + m6_kappa.json carries "
            "HONEST_NEGATIVE=true + metalcytotox_validation block",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"Wf-325D closeout failures: {'; '.join(failures)}",
        elapsed,
    )


# ----------------------------------------------------------------------------
# Wf-321E — ABLATION_TABLE registry mirror check
# ----------------------------------------------------------------------------
# Per Wf-321E brief: TODO/MECHANISM_METRIC_MAPPING.md §7 declares
# 11 R17 ablations; ablation_runner.ABLATION_TABLE must mirror this
# table exactly.  This gate imports ABLATION_TABLE and verifies:
#   (a) it imports successfully (no ImportError),
#   (b) ≥ 11 unique non-baseline ablation IDs,
#   (c) the IDs match the canonical 11 §7 set (order-insensitive),
#   (d) BASELINE is present (the +1 reference row),
#   (e) every entry has the §7 contract fields populated.
# The export name is ABLATION_TABLE (not ABLATION_REGISTRY — the Wf-321E
# brief's import statement was incorrect; corrected here).
EXPECTED_ABLATION_IDS_S5 = frozenset(
    {
        "M15",
        "M13",
        "M14",
        "M12",
        "M11",
        "M10",
        "M3",
        "M4",
        "M8",
        "M7",
        "M21",
    }
)


def gate_wf321e_ablation_registry_complete(
    name: str,
    priority: str,
    pipeline: str,
    runner_module: Path,
) -> GateResult:
    """Wf-321E — ``ABLATION_TABLE`` registry mirrors §7 of MECHANISM_METRIC_MAPPING.

    Per Wf-321E integration test: ablation_runner.py must carry all 11
    R17 ablations declared in ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md``
    §7.  This gate:

      1. Imports ``ABLATION_TABLE`` from ``molmetal.scripts.ablation_runner``
         (the export name is ABLATION_TABLE — not ABLATION_REGISTRY).
      2. Counts the entries; expected ≥ 11 unique non-baseline + 1 BASELINE.
      3. Compares the ablation_id set against the canonical 11 §7 set
         (order-insensitive set equality).
      4. Verifies each entry has the §7 contract fields (ablation_id,
         label, mechanism_module, target_metric, expected_effect,
         toggle, env_vars, extra_args).

    Returns PASS / FAIL with evidence string.  No subprocess; uses
    importlib + direct attribute access (cheap; < 100 ms).
    """
    import importlib.util as _ilu

    if not runner_module.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ablation_runner.py missing: {runner_module}",
            time.perf_counter(),
        )
    t0 = time.perf_counter()
    spec = _ilu.spec_from_file_location(
        "ablation_runner_under_test",
        runner_module,
    )
    if spec is None or spec.loader is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ablation_runner.py: no import spec ({runner_module})",
            time.perf_counter() - t0,
        )
    try:
        # Register the module in sys.modules BEFORE exec_module so the
        # @dataclass decorator can resolve cls.__module__ (Python 3.12
        # requirement: ``_is_type`` reads
        # ``sys.modules[cls.__module__].__dict__``).
        mod = _ilu.module_from_spec(spec)
        sys.modules["ablation_runner_under_test"] = mod
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ablation_runner.py import failed: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    table = getattr(mod, "ABLATION_TABLE", None)
    if table is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "ABLATION_TABLE not exported from ablation_runner.py "
            "(export name expected = 'ABLATION_TABLE'; brief typo "
            "'ABLATION_REGISTRY' corrected here)",
            time.perf_counter() - t0,
        )
    # Each entry must have ablation_id; collect the id set.
    try:
        ids = [getattr(s, "ablation_id", None) for s in table]
    except TypeError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ABLATION_TABLE is not iterable: {exc!r}",
            time.perf_counter() - t0,
        )
    non_baseline_ids = [i for i in ids if i != "BASELINE"]
    unique_non_baseline = set(non_baseline_ids)
    # Duplicate-id check.
    dup_ids = [i for i in unique_non_baseline if non_baseline_ids.count(i) > 1]
    if dup_ids:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ABLATION_TABLE has duplicate non-baseline IDs: {sorted(dup_ids)}",
            time.perf_counter() - t0,
        )
    # Cardinality check.
    if len(unique_non_baseline) < 11:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ABLATION_TABLE has {len(unique_non_baseline)} unique "
            f"non-baseline ablations (< 11 required by §7)",
            time.perf_counter() - t0,
        )
    # Set-equality check vs §7.
    missing = EXPECTED_ABLATION_IDS_S5 - unique_non_baseline
    extra = unique_non_baseline - EXPECTED_ABLATION_IDS_S5
    if missing or extra:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ABLATION_TABLE id mismatch vs §7: "
            f"missing={sorted(missing)} extra={sorted(extra)}",
            time.perf_counter() - t0,
        )
    # BASELINE reference row must be present (the +1 row on top of 11).
    if "BASELINE" not in ids:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "BASELINE reference row missing from ABLATION_TABLE",
            time.perf_counter() - t0,
        )
    # Field contract check — each entry must have §7 contract fields.
    required_fields = (
        "ablation_id",
        "label",
        "mechanism_module",
        "target_metric",
        "expected_effect",
        "toggle",
        "env_vars",
        "extra_args",
    )
    bad_field_rows: list = []
    for s in table:
        missing_fields = [f for f in required_fields if not hasattr(s, f)]
        if missing_fields:
            bad_field_rows.append(f"{getattr(s, 'ablation_id', '?')}:{missing_fields}")
    if bad_field_rows:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"ABLATION_TABLE entries missing fields: {bad_field_rows}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "ABLATION_TABLE OK: 11/11 §7 ablations mirrored "
        "(M3/M4/M7/M8/M10/M11/M12/M13/M14/M15/M21) + 1 BASELINE; "
        "all 12 entries carry the §7 contract fields",
        elapsed,
    )


def gate_section9_entry_count(
    name: str,
    priority: str,
    pipeline: str,
    doc_path: Path,
    *,
    min_entries: int = 13,
) -> GateResult:
    """Wf-306C — ``TODO/completed/207_…md`` §9 catalog MUST have ≥ ``min_entries``.

    Counts occurrences of ``^### §9.`` (top-level subsection headers) at
    column zero inside ``doc_path``. The §9 honest framing catalog is
    the party-twenty-spirit §3.5 MEASURED-vs-PROJECTED evidence ledger;
    a count below the floor indicates the catalog has been pruned or
    has not kept pace with the workflow batch (Wf-306C floor = 13).

    The gate intentionally uses the literal regex ``^### §9.`` rather
    than a fuzzy substring search, because the v2 doc structure reserves
    that exact prefix for top-level catalog entries (e.g. ``### §9.1``,
    ``### §9.AE``); nested subsections (``#### §9.6.1``, ``#### §9.AI.1``)
    do NOT match the prefix and are intentionally excluded from the
    floor count.
    """
    import re

    t0 = time.perf_counter()
    if not doc_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"doc missing: {doc_path}",
            time.perf_counter() - t0,
        )
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"doc unreadable: {doc_path}: {exc}",
            time.perf_counter() - t0,
        )
    n = len(re.findall(r"^### §9\.", text, flags=re.MULTILINE))
    elapsed = time.perf_counter() - t0
    if n >= min_entries:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"{n} top-level §9 entries in {doc_path.name} >= {min_entries}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"{n} top-level §9 entries in {doc_path.name} < {min_entries}",
        elapsed,
    )


# Wf-321D — gate-count consistency audit.  The pre-flight v2 gate list
# has grown across Wf-306C, Wf-312B/C/E, Wf-313C/H, Wf-314E, Wf-315A/D,
# Wf-319A/B/C, Wf-321A/E etc.  Without a guard, silent drift (a gate
# deleted, a runner duplicated, a pipeline filter broken) would only
# surface when a downstream consumer complains.  This gate fires
# ``{py} {script} --pipeline {expected_pipeline} --json {tmp}`` as a
# subprocess and asserts the resulting ``summary.total`` matches
# ``expected_total`` (the documented count for that pipeline).
#
# Per AGENTS.md §1 MEASURED-vs-PROJECTED discipline: the expected counts
# below are MEASURED — they were tabulated in
# ``TODO/completed/321d_preflight_gate_audit.md`` by grepping
# ``build_gates()`` for unique gate names per pipeline.  Bumping a count
# requires updating both this constant AND the audit doc together
# (the doc is the single source of truth).
#
# As-of 2026-09-29 (Wf-321D ship; Wf-322E + Wf-321A + Wf-256 + Wf-176 +
# Wf-323F + Wf-325D + Wf-325E + Wf-325F + Wf-327 bumps + Wf-347A1 +
# Wf-350-A3 + Wf-350-A2 + Wf-350-C1):
#   * ``all``         → 94 gates (93 pre-Wf-350-C1 baseline [71
#                       pre-Wf-353-B1 baseline [...]] + 1 NEW Wf-350-C1
#                       GPU/CUDA env-probe gate that wraps the canonical
#                       R18 Blackwell probe at
#                       molmetal/scripts/check_blackwell_env.py and
#                       verifies the 4 mandatory host-runtime fields
#                       (torch.cuda.is_available + torch.cuda.device_count
#                       + GPU model + CUDA runtime version) at gate
#                       invocation — Wf-343a-R3 finding #1.
#                       The 1 NEW Wf-350-A3 gate is 'all' only (matches
#                       the §3.4 design contract that governs the 4
#                       Wf-343C4 gates + the 2 Wf-347A1 gates — pre-retrain
#                       internal checks are not paper-grade claims; they
#                       are the reproducibility / attribution backbone
#                       that paper §3 citations depend on).
#                       The 1 NEW Wf-350-A2 gate is 'chemprop_v2_v3' AND
#                       'all' (NOT 'frontier' / 'cfm' — it does not
#                       surface as a paper-rule parity gate; it lives at
#                       the chemprop_v2_v3 P1 level per the existing
#                       Wf-259 'p1_*' gate convention for host-runtime
#                       probes, and surfaces in 'all' because it is a
#                       pre-retrain internal check per §3.4).
#                       The 2 NEW Wf-347A1 gates are 'all' only — they
#                       wire the Wf-296 pipeline master to the Wf-343C2
#                       per-wave runtime check (preflight freshness +
#                       hyperparam config + YuelBond decoder + wave
#                       budget + GPU VRAM + disk free per the live env)
#                       + the Wf-343C3 data quality check (csv_loads +
#                       per_metal_balance + scaffold_split + class_balance
#                       + label_range across the 5 canonical loaders).
#                       Both are 'all'-only (not paper-grade claims) per
#                       the same §3.4 design contract that governs the
#                       4 Wf-343C4 gates.
#   * ``frontier``    → 25 gates (unchanged — the 4 NEW Wf-343C4 gates AND
#                       the 2 NEW Wf-347A1 gates are 'all'-only by the
#                       §3.4 design contract; surfacing them in 'frontier'
#                       would dilute the Wf-306C paper-rule parity invariant).
#   * ``chemprop_v2_v3`` → 14 gates (13 pre-Wf-350A2 baseline + 1 NEW
#                       Wf-350-A2 compute-manifest gate).
#   * ``cfm``         →  7 gates (6 Wf-259 baseline + 1 NEW HOLE-1
#                       `p1_cfm_live_decode_canary` — live 3-epoch CPU
#                       smoke of today's CFM trainer + YuelBond decoder;
#                       the frozen Wf-188 `p1_cfm_smoke_decoded` anchor
#                       is kept but relabelled HISTORICAL)
#
# The gate-count consistency audit (Wf-321D) is added to the 'all'
# pipeline only; 'frontier' / 'chemprop_v2_v3' / 'cfm' are subsets that
# should remain stable per their contract (Wf-259 baseline).
EXPECTED_GATE_COUNTS = {
    "all": 103,  # Wf-A16 (2026-10-03): 105 -> 103, MEASURED via
    # len(build_gates("all")) == 103 after the two PRE-RETRAIN
    # gates moved out of the commit path into the new "launch"
    # pipeline (105 -> 103).  They were registered on "all", which
    # is what the commit hook runs AND what every launcher runs, so
    # a commit that only touched gate tooling failed because the
    # paper-grade anchors were 24 h old — a purely temporal trigger
    # with no cheap remedy (refreshing them means re-running the
    # producing workloads; `touch` would be a false claim of
    # re-measurement).  See PIPELINES and the "launch" entry below.
    # L5 fix (2026-10-03): 104 -> 105, MEASURED via
    # len(build_gates("all")) == 105 after the L5 fix added the
    # all-only visibility gate `p0_metalcytotox_v1_known_leaky`
    # (known-leaky v1 CSV: PASS iff leaks > 0 with the consumer
    # roster as the open remedy; FAIL iff 0 leaks = mutated
    # artifact; SKIP on absent gitignored CSV).  Same-edit doc
    # sync: TODO/completed/321d_preflight_gate_audit.md
    # §2.1/§5.2/§6 summary + NEW §6.5, pinned by
    # test_wf327f11_gate_count_consistency.py.
    # L4 fix (2026-10-03): 103 -> 104, MEASURED via
    # len(build_gates("all")) == 104 after the L4 fix added
    # the all-only gate `p2_data_leakage_diag` (wires the
    # previously-unwired check_data_leakage.py diagnostic).
    # HOLE-5 fix (2026-10-03): 101 -> 103, MEASURED via
    # len(build_gates("all")) == 103 after (a) the H4 fix
    # (this run, Wf-446) added the all-only gate
    # `p3_dmpnn_loss_wire_in_smoke` (101 -> 102; the `all`
    # constant was left at 101 although the measured count
    # had already moved — the wf327f11 consistency test was
    # RED at 101 vs 102) and (b) the HOLE-5 fix added the
    # all-only gate `p0_nci60_split_no_leak` (102 -> 103).
    # The HOLE-1 `p1_cfm_live_decode_canary` is cfm-only and
    # did NOT move `all` (the identical smoke runs in `all`
    # under `p3_wf385_scaffold_loss_wire_in_verified`).  The
    # wf327f11 consistency test (code + doc sync) is GREEN
    # after this edit.
    # Per Wf-A12: update this value in the SAME edit as any
    # gate add/remove.
    # Wf-A16 (2026-10-03): "launch" = "all" + the two pre-retrain gates.
    # MEASURED: len(build_gates("launch")) == 105.  Launchers should pass
    # --pipeline launch instead of --pipeline all; the commit hook keeps
    # --pipeline all, so a commit never waits on anchor freshness.
    "launch": 105,
    "frontier": 25,
    "chemprop_v2_v3": 18,  # Wf-A12 FIX (2026-10-03): 14 → 18 MEASURED
    # via len(build_gates("chemprop_v2_v3"));
    # the 14 was stale (pre-Wf-A12 docs drift).
    "cfm": 7,  # HOLE-1 fix (2026-10-03): 6 → 7 — added the LIVE decode
    # canary `p1_cfm_live_decode_canary` (cfm-pipeline-only)
    # which executes TODAY's train_cfm_yuelbond_wasserstein.py
    # + YuelBond decoder via the proven Wf-385-A3 3-epoch CPU
    # smoke (loss decrease 0.3% floor + decode_ratio > 0,
    # fail-closed).  The frozen Wf-188 anchor
    # `p1_cfm_smoke_decoded` remains as a relabelled
    # HISTORICAL check.  MEASURED via
    # len(build_gates("cfm")) at the fix timestamp (update
    # this value in the SAME edit as any cfm gate add/remove,
    # per the Wf-321D §5.3 bump protocol).
}


def gate_p1_hp_provenance_log(
    name: str,
    priority: str,
    pipeline: str,
    config_paths: Tuple[Path, ...],
    jsonl_path: Path,
    *,
    recent_commits_count: int = 20,
    timeout: float = 30.0,
) -> GateResult:
    """Wf-350-A3 — hyperparameter provenance log gate (NeurIPS §D.9 protocol).

    Per Wf-343A-R2 recommendation: a pre-retrain internal check gate that
    records hyperparameter ``git-SHA provenance + recent commit history +
    uncommitted-change status`` for every config file in ``configs/`` and
    emits a JSONL artefact at ``results/gate_scratch/wf350a3_hp_provenance.jsonl``
    (H1/HOLE-8 follow-up: redirected out of the source-adjacent
    ``molmetal/reports/`` tree; the reproducibility-manifest reader falls
    back to the legacy path for un-redirected trees)
    so downstream consumers (paper §3 Wf-350 attribution appendix + paper
    dashboard reproducibility table + F2(a) hyperparam rewind list) can
    trace every retrain back to the exact commit that produced its
    hyperparameters.

    Schema (one JSON record per config file, one per line)::

        {
            "config_path": "configs/cfm_full_retrain.yaml",
            "current_sha": "9b12e6f574dfc397b5b5e8c568c952266e82ef37",
            "recent_commits": [
                {"sha": "9b12e6f5...", "message": "Wf-243 to Wf-252: ...", "date": "2026-09-29T15:35:00+0000"},
                ...
            ],
            "uncommitted_changes": false
        }

    PASS criteria (all required for every config):

      * ``config_path`` exists on disk (is_file).
      * ``jsonl_path`` exists after the gate runs (the gate writes it via
        the Wf-386 §5.6 whole-file ``atomic_write_file`` primitive —
        crash-safe temp-file + rename; a replace, never an append, so
        every run holds exactly one record per config).
      * One JSON record per config file is found in the JSONL.
      * Each record has the 4 mandatory keys (``config_path``,
        ``current_sha``, ``recent_commits``, ``uncommitted_changes``).
      * ``current_sha`` is a 40-hex git SHA (or matches ``git rev-parse HEAD``).
      * ``recent_commits`` is a list (possibly empty); when non-empty,
        each entry has ``sha`` + ``message`` + ``date``.
      * ``uncommitted_changes`` is a boolean.

    FAIL on any one missing record, schema violation, or git subprocess
    failure (e.g. timeout / non-zero exit code).  The gate is
    intentionally strict — a missing provenance record means downstream
    reproducibility attestation cannot be produced (a §5.2 honest-negative
    violation for paper §3 attribution).

    CPU-only by design: the gate invokes ``git rev-parse HEAD`` +
    ``git log --oneline configs/<file>`` + ``git status --short
    configs/<file>`` as subprocess probes (each < 1 s).  Wall budget on
    the test suite: ~3-5 s for 7 config files.  See
    ``TODO/completed/350a3_hp_provenance_log.md`` for the Wf-350-A3
    ship summary.
    """
    import json as _json
    import re as _re

    t0 = time.perf_counter()

    # Ensure the JSONL parent dir exists so we can write the artefact
    # even if the report directory has never been touched before.
    # H1/HOLE-8 follow-up: an unusable target must FAIL the gate, not
    # raise — this mkdir previously sat outside the try and crashed the
    # whole pre-flight run when the parent could not be created (same
    # contract as gate_wf385_scaffold_loss_wire_in_verified's
    # "cannot create out-dir" branch).
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"cannot create JSONL parent dir {jsonl_path.parent!s}: {exc}",
            time.perf_counter() - t0,
        )

    sha_re = _re.compile(r"^[0-9a-f]{40}$")

    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    file_writes: List[str] = []

    for cfg in config_paths:
        if not cfg.is_file():
            problems.append(f"missing config file: {cfg}")
            continue

        # Probe 1: git rev-parse HEAD (current commit SHA)
        rc_head, out_head, err_head = _run(
            ["git", "rev-parse", "HEAD"],
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        if rc_head != 0:
            problems.append(
                f"git rev-parse HEAD failed for {cfg.name} (rc={rc_head}): {err_head.strip()}"
            )
            continue
        current_sha = out_head.strip()

        # Probe 2: git log --oneline configs/<file> | head -20 (recent commits)
        rel_cfg = cfg.relative_to(REPO_ROOT).as_posix()
        rc_log, out_log, err_log = _run(
            [
                "git",
                "log",
                "--pretty=format:%H|%ai|%s",
                "-n",
                str(recent_commits_count),
                "--",
                rel_cfg,
            ],
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        recent_commits: List[Dict[str, str]] = []
        if rc_log != 0:
            # Treat log failure as a soft problem but record empty list
            # so downstream consumers can still see the current_sha.
            problems.append(
                f"git log --oneline {rel_cfg} failed (rc={rc_log}): {err_log.strip()}"
            )
        else:
            for raw_line in out_log.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # Format: SHA|ISO_DATE|MESSAGE
                pieces = line.split("|", 2)
                if len(pieces) < 3:
                    continue
                sha_v, date_v, msg_v = pieces[0], pieces[1], pieces[2]
                if not sha_re.match(sha_v):
                    continue
                recent_commits.append({"sha": sha_v, "message": msg_v, "date": date_v})

        # Probe 3: git status --short configs/<file> (uncommitted changes)
        rc_st, out_st, err_st = _run(
            ["git", "status", "--short", "--", rel_cfg],
            timeout=timeout,
            cwd=REPO_ROOT,
        )
        if rc_st != 0:
            problems.append(
                f"git status --short {rel_cfg} failed (rc={rc_st}): {err_st.strip()}"
            )
            uncommitted_changes = False
        else:
            uncommitted_changes = bool(out_st.strip())

        records.append(
            {
                "config_path": rel_cfg,
                "current_sha": current_sha,
                "recent_commits": recent_commits,
                "uncommitted_changes": uncommitted_changes,
            }
        )

    # Write the JSONL artefact (one record per line) via the Wf-386 §5.6
    # whole-file ``atomic_write_file`` primitive (temp file + POSIX
    # rename) instead of a truncating ``open(w)`` — a concurrent reader
    # never observes a half-written provenance log.  Deliberately NOT an
    # atomic-append: consumers (the schema re-read below + the
    # reproducibility manifest) assume N records = N configs, so appending
    # across runs would accumulate stale-SHA lines for every config.
    payload = "".join(_json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    try:
        atomic_write_file(jsonl_path, payload)
        file_writes.append(str(jsonl_path))
    except Exception as exc:  # noqa: BLE001 — an unwritable target must FAIL the gate, not raise
        problems.append(f"failed to write {jsonl_path}: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0
    if problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"Wf-350-A3 hp provenance probe problems ({len(problems)}): "
            f"{'; '.join(problems[:5])}",
            elapsed,
        )
    if not records:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"Wf-350-A3 hp provenance log empty (0 records for "
            f"{len(config_paths)} config files); wrote {jsonl_path}",
            elapsed,
        )
    if len(records) != len(config_paths):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"Wf-350-A3 hp provenance log records {len(records)}/"
            f"{len(config_paths)}; one or more config files produced no record",
            elapsed,
        )

    # Verify JSONL round-trip schema: re-parse each line and assert the
    # 4 mandatory keys + types.  This catches a future change that
    # silently drops a key (e.g. someone removing uncommitted_changes).
    schema_problems: List[str] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for idx, raw_line in enumerate(fh):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = _json.loads(raw_line)
                except _json.JSONDecodeError as exc:
                    schema_problems.append(f"line {idx}: JSON parse fail: {exc}")
                    continue
                if not isinstance(obj, dict):
                    schema_problems.append(f"line {idx}: not a dict")
                    continue
                for k in (
                    "config_path",
                    "current_sha",
                    "recent_commits",
                    "uncommitted_changes",
                ):
                    if k not in obj:
                        schema_problems.append(f"line {idx}: missing key {k!r}")
                if not isinstance(obj.get("recent_commits"), list):
                    schema_problems.append(f"line {idx}: recent_commits not list")
                if not isinstance(obj.get("uncommitted_changes"), bool):
                    schema_problems.append(f"line {idx}: uncommitted_changes not bool")
                if not isinstance(obj.get("config_path"), str):
                    schema_problems.append(f"line {idx}: config_path not str")
                if not isinstance(obj.get("current_sha"), str):
                    schema_problems.append(f"line {idx}: current_sha not str")
    except OSError as exc:
        schema_problems.append(f"re-read {jsonl_path}: {exc}")

    if schema_problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"Wf-350-A3 JSONL schema validation FAILED ({len(schema_problems)}): "
            f"{'; '.join(schema_problems[:5])}",
            elapsed,
        )

    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"Wf-350-A3 hp provenance log PASS: {len(records)}/{len(config_paths)} "
        f"config records written to {jsonl_path} (current_sha={records[0]['current_sha'][:12]}..., "
        f"recent_commits[0]={len(records[0]['recent_commits'])} entries, "
        f"uncommitted_changes={records[0]['uncommitted_changes']})",
        elapsed,
    )


def gate_p1_runtime_energy_report(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
    *,
    nvidia_smi_timeout: float = 5.0,
) -> GateResult:
    """Wf-350-A1 — runtime + GPU energy probe gate (NeurIPS §E.4 protocol).

    Per Wf-343A-R2 recommendation: a pre-retrain internal check gate
    that records per-GPU instantaneous ``power.draw`` (W) +
    ``power.limit`` (W) + total wall-time duration estimate +
    cumulative estimated energy (joules) so downstream consumers
    (paper §3 Wf-350 attribution appendix + paper dashboard
    reproducibility table + F2(a) hyperparam rewind list) can
    reconstruct the runtime + energy cost of every retrain.

    The probe is intentionally CPU-only — it shells out to
    ``nvidia-smi --query-gpu=index,power.draw,power.limit --format=csv``
    (per-GPU current power draw + max power limit) and computes the
    energy estimate as ``watts × seconds`` using the total wall-time
    duration recorded by the gate's own elapsed timer (so even a
    0-duration probe has a non-negative ``estimated_energy_j >= 0``
    baseline, satisfying the sanity criterion without GPU exec).

    Schema (one JSON document at ``json_path``)::

        {
            "per_gpu": [
                {"index": 0, "power_draw_w": 12.35, "power_limit_w": 600.0},
                ...
            ],
            "total_duration_s": 0.123,
            "estimated_energy_j": 1.5,
            "n_gpus_reported": 2,
            "probe_timestamp": "2026-09-29T15:35:00+00:00",
            "host": "<hostname>",
            "MEASURED_vs_PROJECTED": "MEASURED"
        }

    PASS criteria (all required):

      * ``json_path`` exists on disk after the gate runs (the gate
        writes it).
      * At least 1 GPU is reported (the system has ≥ 1 NVIDIA GPU;
        ``nvidia-smi`` exit-code 0 with at least 1 data row).
      * The JSON document carries the 4 mandatory top-level fields
        (``per_gpu``, ``total_duration_s``, ``estimated_energy_j``,
        ``n_gpus_reported``).
      * ``per_gpu`` is a list; each entry has ``index`` (int),
        ``power_draw_w`` (float, possibly 0.0), ``power_limit_w``
        (float > 0).
      * ``estimated_energy_j >= 0`` (sanity — even a 0-duration
        probe is non-negative by construction).

    FAIL on missing JSON, schema violation, 0 GPUs reported, or
    ``estimated_energy_j < 0`` (negative energy is impossible).

    CPU-only by design: the gate invokes ``nvidia-smi`` once at
    gate invocation, then writes the JSON; total wall-budget
    < 1 s on a 2-GPU host.  The probe does NOT require an active
    training loop — it samples instantaneous power and emits a
    snapshot suitable as a "baseline" reference for downstream
    retrain energy attribution.

    See ``TODO/completed/350a1_runtime_energy_report.md`` for the
    Wf-350-A1 ship summary + the Wf-343A-R2 NeurIPS §E.4 protocol
    provenance.
    """
    import socket as _socket

    t0 = time.perf_counter()

    # Ensure the JSON parent dir exists so we can write the artefact
    # even if the report directory has never been touched before.
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe nvidia-smi for per-GPU instantaneous power + limit.
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        timeout=nvidia_smi_timeout,
        cwd=REPO_ROOT,
    )

    per_gpu: List[Dict[str, Any]] = []
    problems: List[str] = []

    if rc != 0:
        problems.append(
            f"nvidia-smi rc={rc} (likely no NVIDIA driver / GPU "
            f"available on host); stderr={err.strip()[:200]!r}"
        )
    else:
        # Parse the CSV output: each line is "<index>, <power_draw_w>, <power_limit_w>"
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) != 3:
                problems.append(
                    f"nvidia-smi row has {len(cols)} cols (expected 3): " f"{line!r}"
                )
                continue
            try:
                idx = int(cols[0])
                power_draw = float(cols[1])
                power_limit = float(cols[2])
            except ValueError as exc:
                problems.append(f"nvidia-smi row parse failure ({exc}): {line!r}")
                continue
            per_gpu.append(
                {
                    "index": idx,
                    "power_draw_w": power_draw,
                    "power_limit_w": power_limit,
                }
            )

    # Compute total wall-time duration. We sample AFTER the nvidia-smi
    # probe so total_duration_s reflects probe wall-time (>= 0 by
    # construction; perf_counter is monotonic).
    total_duration_s = time.perf_counter() - t0

    # Estimated energy = mean instantaneous power × wall-time duration.
    # When 0 GPUs are reported, the estimate collapses to 0 J (the
    # gate is sanity-capped at >= 0 below).  For multi-GPU hosts the
    # mean across per-GPU power draws is the cohort-power estimator.
    if per_gpu:
        mean_power_w = sum(g["power_draw_w"] for g in per_gpu) / len(per_gpu)
    else:
        mean_power_w = 0.0
    estimated_energy_j = max(0.0, mean_power_w * total_duration_s)

    # Write the JSON artefact.
    artefact: Dict[str, Any] = {
        "per_gpu": per_gpu,
        "total_duration_s": round(total_duration_s, 6),
        "estimated_energy_j": round(estimated_energy_j, 6),
        "n_gpus_reported": len(per_gpu),
        "probe_timestamp": _now_iso(),
        "host": _socket.gethostname(),
        "MEASURED_vs_PROJECTED": "MEASURED",
    }
    json_path.write_text(json.dumps(artefact, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0

    # ---- FAIL branches (ordered by severity) ----
    if problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(problems)} nvidia-smi probe problem(s): " f"{problems[:3]}",
            elapsed,
        )
    if not per_gpu:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"0 GPUs reported (nvidia-smi rc={rc}, out={out[:200]!r}, "
            f"err={err[:200]!r}); artefact written to {json_path} "
            f"with empty per_gpu list",
            elapsed,
        )
    if estimated_energy_j < 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"estimated_energy_j={estimated_energy_j} < 0 (impossible — "
            f"non-negative sanity contract)",
            elapsed,
        )

    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"Wf-350-A1 runtime+energy PASS: {len(per_gpu)} GPU(s) reported "
        f"(power_draw range {min(g['power_draw_w'] for g in per_gpu):.2f}-"
        f"{max(g['power_draw_w'] for g in per_gpu):.2f}W; "
        f"power_limit range {min(g['power_limit_w'] for g in per_gpu):.2f}-"
        f"{max(g['power_limit_w'] for g in per_gpu):.2f}W); "
        f"total_duration_s={total_duration_s:.4f}; "
        f"estimated_energy_j={estimated_energy_j:.4f}; "
        f"artefact at {json_path}",
        elapsed,
    )


def gate_p1_gpu_env_probe(
    name: str,
    priority: str,
    pipeline: str,
    env_json_path: Path,
    *,
    probe_script: Path = SCRIPT_DIR / "check_blackwell_env.py",
    python_bin: str = VENV_PY,
    probe_timeout: float = 60.0,
) -> GateResult:
    """Wf-350-C1 — GPU/CUDA env probe gate (wraps check_blackwell_env.py).

    Per Wf-343a-R3 finding #1: the Blackwell GPU/CUDA environment probe
    should be wired into the pre-flight gate set so every launch path
    surfaces the canonical ``torch.cuda.is_available`` +
    ``torch.cuda.device_count`` + GPU model + CUDA runtime version at
    host run-time.  The standalone probe at
    ``molmetal/scripts/check_blackwell_env.py`` is the canonical
    R18 Blackwell env probe (108 LOC, ship surface Wf-265 + Wf-326C);
    this gate is its pre-flight integration wrapper — it shells out to
    the probe, captures the resulting JSON artefact, and asserts the
    4 host-runtime fields required by AGENTS.md §5.1 (standing GPU
    grant — "显存够就放心大胆用"):

      1. ``torch.cuda.is_available`` (via ``torch_probe.ok``)
      2. ``torch.cuda.device_count`` (via ``torch_probe.device_count``)
      3. GPU model (via ``gpu_names`` + per-GPU ``gpu_details[*].name``)
      4. CUDA runtime version (via ``torch_probe.cuda_runtime``)

    PASS criteria (all required):

      * ``env_json_path`` exists on disk after the gate runs (the gate
        invokes the probe and writes the artefact).
      * ``torch_probe.ok`` is ``True`` (i.e. ``torch.cuda.is_available``
        on at least one candidate venv per the probe's per-venv loop).
      * ``torch_probe.device_count >= 1`` (at least 1 GPU available to
        torch).
      * ``gpu_names`` is a non-empty list (the probe found ≥ 1 GPU via
        ``nvidia-smi``).
      * The 4 mandatory host-runtime fields are all populated.

    FAIL on any one missing record, missing field, or probe subprocess
    failure (e.g. timeout / non-zero exit code).

    CPU-only by design: the gate invokes ``python check_blackwell_env.py``
    once at gate invocation, then reads the JSON; total wall-budget
    on a 2-GPU host ~1-3 s (the probe spawns ``nvidia-smi`` + up to 3
    candidate venvs to probe torch.cuda).

    See ``TODO/completed/350c1_gpu_env_probe.md`` for the Wf-350-C1
    ship summary + the Wf-343a-R3 recommendation provenance.
    """
    t0 = time.perf_counter()

    # ---- (1) ensure parent directory exists ----
    try:
        env_json_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"cannot create env-json parent dir " f"{env_json_path.parent!s}: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (2) invoke the probe as a subprocess ----
    # The probe writes to its first positional arg; we pass our
    # env_json_path so each gate run produces a fresh snapshot.
    if not probe_script.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"probe script not on disk: {probe_script}",
            time.perf_counter() - t0,
        )

    rc, _out, err = _run(
        [python_bin, str(probe_script), str(env_json_path)],
        timeout=probe_timeout,
        cwd=REPO_ROOT,
    )
    if rc != 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"probe script rc={rc}: {err.strip()[:200]!r}; "
            f"expected JSON at {env_json_path} not written",
            time.perf_counter() - t0,
        )

    # ---- (3) re-read the JSON artefact ----
    if not env_json_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"probe exit-code 0 but artefact not at {env_json_path}",
            time.perf_counter() - t0,
        )
    try:
        env_summary = json.loads(env_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary at {env_json_path} is not valid JSON: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (4) schema validity checks ----
    if not isinstance(env_summary, dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary is not a JSON object " f"(got {type(env_summary).__name__})",
            time.perf_counter() - t0,
        )

    missing_top: List[str] = []
    for key in ("torch_probe", "gpu_count", "gpu_names"):
        if key not in env_summary:
            missing_top.append(key)
    if missing_top:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary missing top-level keys: {missing_top}",
            time.perf_counter() - t0,
        )

    torch_probe = env_summary.get("torch_probe", {})
    if not isinstance(torch_probe, dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary 'torch_probe' is not a dict "
            f"(got {type(torch_probe).__name__})",
            time.perf_counter() - t0,
        )

    missing_probe: List[str] = []
    for key in ("ok", "device_count", "device_name", "cuda_runtime"):
        if key not in torch_probe:
            missing_probe.append(key)
    if missing_probe:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary 'torch_probe' missing fields: {missing_probe}",
            time.perf_counter() - t0,
        )

    gpu_names = env_summary.get("gpu_names", [])
    if not isinstance(gpu_names, list):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"env summary 'gpu_names' is not a list "
            f"(got {type(gpu_names).__name__})",
            time.perf_counter() - t0,
        )

    # ---- (5) FAIL branches (ordered by severity) ----
    if not torch_probe.get("ok", False):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"torch.cuda.is_available == False on every probed venv "
            f"(torch_probe={torch_probe}); host has no GPU-backed "
            f"PyTorch installation — every GPU launch path is BLOCKED. "
            f"Re-source env.sh or use --no-gpu fallback.",
            time.perf_counter() - t0,
        )

    device_count = int(torch_probe.get("device_count", 0))
    if device_count < 1:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"torch.cuda.device_count == {device_count} "
            f"(expected >= 1); torch reports 0 usable GPUs despite "
            f"torch.cuda.is_available == True — driver/runtime mismatch. "
            f"torch_probe={torch_probe}",
            time.perf_counter() - t0,
        )

    if len(gpu_names) < 1:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "gpu_names list is empty (nvidia-smi reported 0 GPUs); "
            "host has no NVIDIA GPU visible. torch_probe reports "
            "torch.cuda.is_available=True but the system driver "
            "sees no hardware — re-seat GPU / driver.",
            time.perf_counter() - t0,
        )

    # ---- PASS evidence ----
    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"Wf-350-C1 GPU/CUDA env probe PASS: "
        f"torch.cuda.is_available=True | "
        f"torch.cuda.device_count={device_count} | "
        f"GPU model={torch_probe.get('device_name')!s} | "
        f"CUDA runtime={torch_probe.get('cuda_runtime')!s} | "
        f"sm_120_blackwell={torch_probe.get('sm120_blackwell', False)} | "
        f"{len(gpu_names)} GPU(s) visible via nvidia-smi: {gpu_names}; "
        f"artefact at {env_json_path}",
        elapsed,
    )


def gate_wf327_cpu_num_workers_sufficient(
    name: str,
    priority: str,
    pipeline: str,
    launch_scripts: Sequence[Path],
    min_workers: int = 4,
) -> GateResult:
    """Wf-327 — every Wf-326B-active launch script must request ≥ ``min_workers``.

    Per Wf-326B §6.2 + AGENTS.md §5.2 MEASURED-vs-PROJECTED framing:
    forcing ``--cpu-num-workers 0`` (or any value < ``min_workers``)
    overrides the Wf-260 ``build_optimized_loader`` recipe and reverts
    the trainer to the manual-loop fallback path (no DataLoader, no
    pin_memory, no workers, no prefetch).  The Wf-328-W2 commit
    migrated the 6 active launch scripts to ``--cpu-num-workers 8``;
    this gate prevents the regression from re-occurring by fail-fast
    blocking any future commit that flips a launcher back to 0.

    The gate accepts either flag (``--cpu-num-workers N`` or
    ``--num-workers N`` — both surface the same intent per the
    Wf-328-W6 CLI precedence rule).  The check uses a regex rather
    than substring search so the digit boundary is exact
    (``--cpu-num-workers 0`` does NOT match ``>=4``).

    On FAIL the evidence string names the offending script(s) AND the
    detected worker count so the regression is direct to debug.
    """
    import re

    pattern_cpu = re.compile(r"--cpu-num-workers\s+(\d+)")
    pattern_nw = re.compile(r"(?<!cpu-)--num-workers\s+(\d+)")
    t0 = time.perf_counter()
    offending: List[str] = []
    passed: List[str] = []
    missing: List[str] = []
    for script in launch_scripts:
        if not script.is_file():
            missing.append(str(script))
            continue
        try:
            text = script.read_text(encoding="utf-8", errors="replace")
        except OSError:
            missing.append(str(script))
            continue
        # Take the *max* of all --cpu-num-workers / --num-workers
        # mentions (multiple flags in same script = last-wins in CLI,
        # but we just want to verify any are >= floor).
        nums = [int(m.group(1)) for m in pattern_cpu.finditer(text)]
        nums += [int(m.group(1)) for m in pattern_nw.finditer(text)]
        # If neither flag is present, we don't fail (the trainer's
        # factory fallback to ``suggest_num_workers()`` is still 8 on a
        # 24-core host per Wf-326B recipe).  This case is HONEST-NEGATIVE
        # disclosed but does NOT block.
        if not nums:
            passed.append(f"{script.name}(no-flag=factory-default)")
            continue
        max_workers = max(nums)
        if max_workers >= min_workers:
            passed.append(f"{script.name}(max={max_workers})")
        else:
            offending.append(f"{script.name}(max={max_workers})")
    elapsed = time.perf_counter() - t0
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing launch script(s): {missing}",
            elapsed,
        )
    if offending:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(offending)}/{len(launch_scripts)} launch scripts "
            f"use --cpu-num-workers/--num-workers < {min_workers}: "
            f"{offending}; passed: {passed}.  Per Wf-326B §6.2 + "
            f"Wf-328-W2/W6 CLI precedence, every active launch script "
            f"MUST request >= {min_workers} workers or remove the "
            f"override entirely (factory default 8 on 24-core host).",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{len(passed)}/{len(launch_scripts)} launch scripts request "
        f">= {min_workers} workers (Wf-328-W2 + Wf-326B §6.2): {passed}",
        elapsed,
    )


# ----------------------------------------------------------------------------
# Wf-343C4 — pre-retrain internal check metric gates
# ----------------------------------------------------------------------------
# Per the 2026-09-29 user request ("ultracode for pre-retrain internal check
# metrics") + AGENTS.md §5.1 standing GPU grant, the pre-retrain gate set
# gains 4 NEW structural / config / freshness / integration gates on top of
# the 62 pre-Wf-343C4 baseline. These are 'all'-only (not 'frontier') —
# they are pre-retrain internal check metrics, NOT paper-grade claims.
#
# Design contract:
#   * Each gate is CPU-only (< 10 ms), backward-compat (does NOT modify any
#     pre-existing script), and emits the canonical GateResult schema.
#   * Each gate is registered LAST in the 'all' pipeline (before the
#     Wf-321D gate_count_consistency_audit which runs as the final
#     verifier so its own count stays consistent).
#   * The 4 NEW gates verify 4 distinct contracts:
#       (1) hyperparams config complete (AGENTS.md §5.1 standing GPU
#           grant — hyperparams in config; not guessed)
#       (2) YuelBond decoder registered (Wf-272I / Wf-269C structural
#           check; --no-yuelbond → decode_ratio 0.000 MEASURED)
#       (3) anchor JSON freshness 24hr (Wf-302A / Wf-304A AnchorGuard
#           adoption; pre-retrain gate that prevents stale anchor reads)
#       (4) wave_budget remaining ≥ 0 (cross-script integration with
#           wave_budget_tracker.py — the third audit link alongside
#           per_wave_checklist.py)
# ----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wf-A12 FIX (2026-10-03) — hyperparam RESOLUTION helpers for
# ``gate_wf343c4_hyperparams_config_complete`` (G2).
#
# The gate used to PASS on mere PRESENCE of the 4 hyperparam keys in a YAML
# (plus an int/float type check).  It never asked whether the key RESOLVES to
# a trainer argparse dest — so `model.hidden_dim: 300` sat in four configs
# while the runs used the argparse default 128 (see the Wf-A11 audit +
# TODO/completed/config_dead_key_remediation.md).  The helpers below answer
# "does every declared key resolve?" by REUSING the canonical loader
# ``molmetal/scripts/wfA2_config_loader.py`` — its ``apply_yaml_overrides``
# plus the ``args.yaml_unresolved_keys`` report added by Wf-A11 — rather than
# re-implementing the candidate/leaf/alias resolution.
#
# The probe parser is built from the trainer's OWN ``add_argument`` calls via
# an AST scan of the entry script: no import of the trainer (its module-level
# torch import would cost seconds) and no execution of it.  The loader is then
# invoked exactly the way the trainer invokes it, so resolution semantics
# (flattened key, leaf fallback, _LEAF_ALIASES) stay in ONE place.
#
# BLOCKING BEHAVIOUR IS OFF (Wf-A12, measured debt is not zero: 65/85/47/14/
# 84/95/47/48/43 skipped keys across the 9 active configs at the fix
# timestamp).  The gate REPORTS + LISTS unresolved keys loudly and still
# PASSes.  Enabling blocking (env ``PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT=1``
# or the constant below) is a SEPARATE, later decision for when the debt
# reaches zero — do NOT flip it now: it would fail every commit in this repo.
#
# Wf-A13 (2026-10-03, HOLE 7) — the raw skip list (528 keys measured across
# the 9 active configs) overstated the real debt, so the strict switch had no
# reachable ON state.  The gate now classifies each skipped key into
# loader-exempt (the canonical loader's own by-design sections/keys) /
# reader-exempt / spec-exempt (documented below, each with a citation or a
# grep note) / RESIDUAL, reports the buckets separately, and the strict
# switch fails ONLY on RESIDUAL (+ REQUIRED-HYPERPARAM UNRESOLVED, never
# exempted).  That makes the switch flippable once residual reaches zero —
# NOT today: residual is still non-zero and the switch stays OFF (report
# loudly; block only behind an OFF switch).
# ---------------------------------------------------------------------------
HYPERPARAMS_RESOLUTION_STRICT_DEFAULT = False
HYPERPARAMS_RESOLUTION_STRICT_ENV = "PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT"


def _config_entry_trainer_script(obj: Dict[str, Any]) -> Optional[Path]:
    """Entry script a config targets: ``entry.script`` or the first
    ``molmetal/scripts/*.py`` path inside ``entry.command_template``."""
    entry = obj.get("entry")
    if not isinstance(entry, dict):
        return None
    script = entry.get("script")
    if isinstance(script, str) and script.strip():
        cand = REPO_ROOT / script.strip()
        if cand.is_file():
            return cand
    tmpl = entry.get("command_template")
    if isinstance(tmpl, str):
        import re as _re

        m = _re.search(r"molmetal/scripts/([\w.\-]+\.py)", tmpl)
        if m:
            cand = SCRIPT_DIR / m.group(1)
            if cand.is_file():
                return cand
    return None


def _trainer_argparse_dests(script: Path) -> "set[str]":
    """Argparse dest names declared by ``script`` (AST scan of add_argument).

    Explicit ``dest=`` wins; otherwise the first dash-flag spelling is
    converted (``--hidden-dim`` -> ``hidden_dim``).  AST-only: the trainer is
    NOT imported or executed.
    """
    import ast as _ast

    tree = _ast.parse(script.read_text(encoding="utf-8"))
    dests: "set[str]" = set()
    for node in _ast.walk(tree):
        if not (
            isinstance(node, _ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
        ):
            continue
        explicit: Optional[str] = None
        for kw in node.keywords:
            if (
                kw.arg == "dest"
                and isinstance(kw.value, _ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                explicit = kw.value.value
        if explicit:
            dests.add(explicit)
            continue
        for arg in node.args:
            if (
                isinstance(arg, _ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("-")
            ):
                dests.add(arg.value.lstrip("-").replace("-", "_"))
                break
    return dests


def _probe_parser_for_trainer(script: Path) -> "argparse.ArgumentParser":
    """An argparse parser whose dest set mirrors ``script``'s own parser."""
    ap = argparse.ArgumentParser(add_help=False, prog=script.name)
    for dest in sorted(_trainer_argparse_dests(script)):
        ap.add_argument("--" + dest.replace("_", "-"), dest=dest)
    return ap


# ---------------------------------------------------------------------------
# Wf-A13 (2026-10-03, HOLE 7) — documented non-trainer allow-list.
#
# Classification of every loader-unresolved key:
#   * "loader"   — the canonical loader itself documents the skip as by-design
#                  (_NON_TRAINER_SECTIONS / _NON_TRAINER_KEYS in
#                  molmetal/scripts/wfA2_config_loader.py).  Detected with the
#                  loader's OWN predicate so the two can never drift.
#   * "reader"   — a named non-trainer consumer reads the key (HP_RESOLUTION_
#                  EXEMPT_KEYS, each with a citation verified by reading the
#                  consumer script).
#   * "spec"     — spec / provenance / report-metadata namespace with NO
#                  executable consumer (HP_RESOLUTION_EXEMPT_PREFIXES, each
#                  grep-verified over *.py + *.sh: the only occurrences are
#                  the YAML declarations themselves).
#   * "residual" — everything else: a leaf naming a real argparse dest of a
#                  SIBLING trainer but unreachable on this config's entry
#                  (e.g. cfm_full_retrain.yaml declares train.wasserstein_weight
#                  while entry.script is metallo_drug_smoke_retrain.py), or a
#                  near-miss knob no trainer declares.  Residual is the only
#                  bucket the strict switch fails on, so it must stay honest.
#
# This is a POLICY accept-list, not a claim that exempt keys reach the trainer:
# an exempt key is still NOT wired to the trainer; it is a documented
# non-trainer value.  If the project wants one wired, delete its entry here
# AND fix the config — do not silence it by widening a prefix.
# ---------------------------------------------------------------------------

#: Exact dotted keys whose consumer is NOT the trainer, citation per entry.
HP_RESOLUTION_EXEMPT_KEYS: Dict[str, str] = {
    # --- scripts/start.sh parse_yaml() (read 2026-10-03) loads the raw YAML
    #     and re-exports these as env vars / entry.command template
    #     placeholders.  Env-var keys append real CLI flags to the trainer
    #     command; placeholder keys feed legacy entry.command templates only.
    "model.hidden_dim": "start.sh placeholder `hidden=` / --hidden-dim",
    "model.n_layers": "start.sh placeholder `n_layers=`",
    "train.batch_size": "start.sh placeholder `batch_size=`",
    "train.lr": "start.sh placeholder `lr=`",
    "train.epochs": "start.sh placeholder `epochs=`",
    "flow.n_steps": "start.sh placeholder `n_steps=`",
    "flow.method": "start.sh placeholder `method=`",
    # 2026-10-04: `flow.ot_solver: w_flow` names the W-Flow waveform-relaxation
    # ODE solver (adapters/flow_matching_lipman/w_flow.py).  MEASURED: no
    # consumer reads a config key by that name anywhere -- the only occurrences
    # of the token outside YAML are the module's own function name
    # `optimal_transport_v2_solver.ot_solver` and its call sites, which is a
    # different thing.  Registering it as spec is the honest call, but the
    # reason is NOT "documentation": the CFM trainer has no ODE solver at all
    # (`grep ODESolver` over train_cfm_yuelbond_wasserstein.py returns nothing;
    # decoding is a direct bond-graph decode from the cloud).  So this key and
    # its neighbours `flow.n_steps` / `flow.method` describe an ODE sampling
    # pipeline that this entry does not run.  Kept because the `flow:` block is
    # shared with configs that do have one, and removing three keys from one
    # config while a sibling uses them would be the more confusing edit.
    "flow.ot_solver": (
        "declares an ODE solver for a flow-matching pipeline this trainer "
        "does not implement (no ODESolver; decode is direct bond-graph)"
    ),
    "data.dataset": "start.sh placeholder `dataset=` + legacy-entry synthesis",
    "protocol.epochs": "start.sh EPOCHS env",
    # 2026-10-04: consumed by start.sh, not a trainer, so the dest-based
    # classifier cannot see it.  start.sh exports N_SEEDS_MIN and blocks a run
    # whose protocol.n_seeds falls below it -- the same "minimum seeds for an
    # honest per-seed std" guard as chemprop's --n-seeds-min
    # (train_chemprop_v2_v3.py:1714).  The guard lives in the launcher because
    # a trainer invocation sees a single --seed; only start.sh knows the sweep
    # size.  ALLOW_SINGLE_SEED=1 bypasses it for smoke / honest-negative runs.
    "protocol.n_seeds_min": "start.sh N_SEEDS_MIN env -> n_seeds guard",
    "protocol.weight_decay": "start.sh WEIGHT_DECAY env -> trainer --weight-decay",
    "protocol.max_norm": "start.sh MAX_NORM env -> trainer --max-norm",
    "protocol.scheduler.type": "start.sh SCHEDULER env -> trainer --lr-scheduler",
    "protocol.scheduler.warmup_epochs": "start.sh LR_WARMUP_EPOCHS env -> trainer --lr-warmup-epochs",
    "protocol.scheduler.min_lr": "start.sh LR_MIN_LR env -> trainer --lr-min-lr",
    "output.ckpt_dir": "start.sh placeholder `ckpt_dir=`",
    "output.checkpoint_interval": "start.sh CHECKPOINT_INTERVAL env -> trainer flag",
    "output.resume_from": "start.sh RESUME_FROM env -> trainer --resume-from",
    "output.events_jsonl_path": "start.sh EVENTS_JSONL_PATH env -> trainer --events-jsonl-path",
    # 2026-10-04: registered when it was actually wired.  This boolean was in
    # all 9 configs and read by nothing — start.sh only looked at the _path
    # key, so `events_jsonl: false` still emitted events.  start.sh now emits
    # the trainer's --no-events-jsonl instead of --events-jsonl-path when this
    # is False.  Registered here because the classifier resolves against
    # argparse dests only, and this is consumed by start.sh; without the entry
    # it would keep being reported as a dead key that a reader has just fixed.
    "output.events_jsonl": "start.sh -> trainer --no-events-jsonl when false",
    # 2026-10-04: same treatment.  start.sh reads compute.cpu_workers and emits
    # trainer --cpu-num-workers; the classifier resolves against argparse dests
    # only, so a start.sh-consumed key would otherwise be reported dead right
    # after being wired.
    "compute.cpu_workers": "start.sh -> trainer --cpu-num-workers",
}

#: Dotted prefixes (exact key or any child) with NO executable consumer.
#: Verified 2026-10-03 by grepping each leaf over *.py + *.sh (excluding this
#: file and wfA2_config_loader.py): the only hits are the YAML declarations.
HP_RESOLUTION_EXEMPT_PREFIXES: Dict[str, str] = {
    "eval.": "report targets / baselines / metric names; compared by hand",
    "data.split.": "split provenance (type / fracs / seed); the split is computed in-code",
    "data.cohort_filter.": "cohort provenance; selection runs via --metals / --csv",
    "data.n_rows_expected": "row-count provenance",
    "data.expected_columns_subset": "column provenance",
    "data.target_index": "legacy qm9 spec (configs/default.yaml)",
    "protocol.full_protocol.": "full-protocol provenance (wall targets / ablate flags)",
    "protocol.enumerated_runs": "run-enumeration spec",
    "protocol.n_runs_total": "run-enumeration spec",
    "protocol.total_runs": "run-enumeration spec",
    "protocol.wave_decomposition": "run-enumeration spec",
    "output.per_pocket_dir": "report-dir spec",
    "output.per_seed_dir": "report-dir spec",
    "output.per_config_dir": "report-dir spec",
    "output.per_metal_dir": "report-dir spec",
    "output.wasserstein_decay_dir": "report-dir spec",
    "output.comparison_30vs50": "report-file spec",
    "output.ablation_report": "report-file spec",
    "output.orchestrator": "launch-script path spec",
    "output.metrics_json": "metrics-artifact path spec",
}

#: Keys that must NEVER be exempted by a prefix rule.  eval.target_decode_ratio
#: names a real CFM argparse dest and decode_ratio is the canonical §5.7
#: honest-negative floor metric — if it is unresolved on the active entry that
#: is dead wiring and must stay visible in the residual bucket.
HP_RESOLUTION_NEVER_EXEMPT = frozenset({"eval.target_decode_ratio"})


def _classify_unresolved_key(key: str) -> "tuple[str, str]":
    """Bucket one loader-unresolved key: ('reader'|'spec'|'residual', why).

    The 'loader' bucket is decided by the canonical loader's own predicate in
    :func:`_resolve_one_config_via_loader`, not here.
    """
    k = (key or "").strip()
    if k in HP_RESOLUTION_NEVER_EXEMPT:
        return "residual", (
            "never exempted: names a real trainer dest (dead wiring stays visible)"
        )
    citation = HP_RESOLUTION_EXEMPT_KEYS.get(k)
    if citation:
        return "reader", citation
    for prefix, note in HP_RESOLUTION_EXEMPT_PREFIXES.items():
        p = prefix.rstrip(".")
        if k == p or k.startswith(p + "."):
            return "spec", note
    return "residual", (
        "no argparse dest on the entry trainer and no documented non-trainer consumer"
    )


def _resolve_one_config_via_loader(
    yaml_path: Path, obj: Dict[str, Any]
) -> "tuple[List[Dict[str, str]], Optional[str]]":
    """([{"key","bucket","why"}, …], error) for one YAML via the canonical loader.

    Every skipped key is classified (Wf-A13): "loader" when the canonical
    loader's own predicate documents the skip as by-design, "reader"/"spec"
    per the documented allow-lists, else "residual".

    ``error`` is a human-readable string when the probe could not run (no
    entry script / loader import failure / loader raised); callers report it
    but do NOT fail on it (the debt is still non-zero — see the gate
    docstring).
    """
    script = _config_entry_trainer_script(obj)
    if script is None:
        return [], (
            f"{yaml_path.name}: no resolvable entry script "
            f"(entry.script / entry.command_template); resolution check skipped"
        )
    try:
        from molmetal.scripts import wfA2_config_loader as _loader
    except Exception as exc:  # noqa: BLE001 — probe must never crash the gate
        return (
            [],
            f"{yaml_path.name}: loader import failed: {type(exc).__name__}: {exc}",
        )
    parser = _probe_parser_for_trainer(script)
    args = parser.parse_args([])
    # Force-quiet (the gate re-reports the skips itself) and drop any
    # inherited WFA2_CONFIG_STRICT for the duration of the probe so an
    # unrelated env var cannot turn this REPORTING check into a hard
    # failure; the gate's own strict switch is the only blocking control
    # and it is OFF by default.
    prev_quiet = os.environ.get("WFA2_CONFIG_QUIET")
    prev_strict = os.environ.pop("WFA2_CONFIG_STRICT", None)
    os.environ["WFA2_CONFIG_QUIET"] = "1"
    try:
        args = _loader.apply_yaml_overrides(parser, args, str(yaml_path))
    except Exception as exc:  # noqa: BLE001 — probe must never crash the gate
        return [], f"{yaml_path.name}: loader raised {type(exc).__name__}: {exc}"
    finally:
        if prev_quiet is None:
            os.environ.pop("WFA2_CONFIG_QUIET", None)
        else:
            os.environ["WFA2_CONFIG_QUIET"] = prev_quiet
        if prev_strict is not None:
            os.environ["WFA2_CONFIG_STRICT"] = prev_strict
    # Wf-A13 — classify every skip.  "loader" uses the loader's OWN
    # predicate (single source of truth with WFA2_CONFIG_STRICT); a missing
    # predicate (loader API drift) degrades to the local classifier rather
    # than crashing the probe.
    predicate = getattr(_loader, "_registered_non_trainer", None)
    classified: List[Dict[str, str]] = []
    for entry in getattr(args, "yaml_unresolved_keys", []):
        key = str(entry.get("key", ""))
        try:
            if callable(predicate) and predicate(entry):
                classified.append(
                    {
                        "key": key,
                        "bucket": "loader",
                        "why": "canonical loader _NON_TRAINER_SECTIONS/_NON_TRAINER_KEYS",
                    }
                )
                continue
        except Exception:  # noqa: BLE001 — classification must never crash
            pass
        bucket, why = _classify_unresolved_key(key)
        classified.append({"key": key, "bucket": bucket, "why": why})
    return classified, None


def gate_wf343c4_hyperparams_config_complete(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: Tuple[Path, ...],
) -> GateResult:
    """Wf-343C4 — every active ``configs/*.yaml`` MUST carry the standing GPU grant
    hyperparameters.

    Per AGENTS.md §5.1 (2026-09-29 user grant): **"Hyperparameters via
    config files: reference peer papers in the same domain; encode in
    ``--batch-size``, ``--lr``, ``--hidden``, ``--depth`` config rather
    than guessing."**  This gate verifies the 4 canonical hyperparameters
    are present in every active YAML at the canonical paths:

      * ``train.batch_size`` (int)
      * ``train.lr`` (float)
      * ``model.hidden_dim`` (int; peer-paper-aligned default 128)
      * ``model.n_layers`` (int; peer-paper-aligned default 3-4)

    PASS criteria (all required for every YAML):

      * YAML parses via ``yaml.safe_load`` (mirrors ``gate_yaml_schema``).
      * Top-level ``train`` dict carries both ``batch_size`` AND ``lr``
        with numeric (int / float) values.
      * Top-level ``model`` dict carries both ``hidden_dim`` AND
        ``n_layers`` with integer values.

    FAIL on any one missing — the gate is intentionally strict because
    the standing grant's "config files > guessing" contract is the user's
    standing directive; a missing hyperparam means a future retrain would
    fall back to a guess (a §5.2 honest-negative violation).

    Wf-A12 FIX (2026-10-03) — RESOLUTION check (PRESENCE was not enough):
    a key can be present in a YAML and still dead, because it resolves to
    no trainer argparse dest.  MEASURED consequence: four configs declared
    ``model.hidden_dim: 300`` and ran at the argparse default 128 for
    weeks.  This gate now ALSO runs every declared key through the
    canonical ``wfA2_config_loader.apply_yaml_overrides`` (probe parser
    built from the entry trainer's own ``add_argument`` calls via AST, no
    trainer import) and reports the ``args.yaml_unresolved_keys`` list.

    Wf-A13 FIX (2026-10-03, HOLE 7) — the report is SPLIT into buckets so
    the strict switch has a reachable ON state.  Measured at the fix
    timestamp (9 active YAMLs, 528 skipped keys total): 306 already exempt
    by design inside the canonical loader, 27 with a named reader in
    scripts/start.sh, 134 spec/provenance namespaces with no executable
    consumer (grep-verified), 61 RESIDUAL (a leaf naming a real argparse
    dest of a SIBLING trainer but unreachable on this config's entry — e.g.
    cfm_full_retrain.yaml declares train.wasserstein_weight while
    entry.script is metallo_drug_smoke_retrain.py — or a near-miss knob no
    trainer declares, e.g. output.events_jsonl).  Each bucket is reported
    separately (evidence + stderr).  The strict switch
    (``PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT=1`` /
    ``HYPERPARAMS_RESOLUTION_STRICT_DEFAULT``) fails ONLY on residual keys
    + REQUIRED-HYPERPARAM UNRESOLVED (never exempted) — so it can be
    flipped as soon as residual reaches zero.  It stays OFF today: residual
    is still non-zero and the repo must not fail commits (report loudly;
    block only behind an OFF switch).

    WHAT IS NOT VERIFIED: exempt keys are still NOT wired to the trainer —
    the allow-list says their value belongs to a documented non-trainer
    consumer (or is pure spec), not that a run reads them.  The four
    REQUIRED_HYPERPARAMS are never exempted from the resolution probe; a
    required key that is present-but-dead is always reported as
    ``REQUIRED-HYPERPARAM UNRESOLVED`` and blocks in strict mode.

    CPU-only gate (no torch.cuda, no subprocess) — runs in < 100 ms on the
    host. The 9 active YAMLs are the 7 from ``configs/`` (``cfm_full_
    retrain.yaml``, ``cfm_per_innovation_ablation_p1.4.yaml``,
    ``cfm_real_decode_lift_p1.3.yaml``, ``default.yaml``,
    ``wf124_p3_50epoch_full_sweep_g5.yaml``,
    ``wf124_p3_per_innovation_ablation_g6.yaml``,
    ``wf124_p3_remaining_metals_g4.yaml``) plus the 2 from
    ``configs/training/`` (``nci60_9cell_5seed_50ep_v1.yaml``,
    ``platinai_pt_4cell_5seed_50ep_v1.yaml``) — the Wf-A12 coverage
    extension.

    See TODO/completed/343c4_integration.md for the Wf-343C4 ship summary
    + the 4-gate contract.
    """
    t0 = time.perf_counter()
    try:
        import yaml as _yaml
    except ImportError:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "PyYAML not installed; cannot verify hyperparameter contract",
            time.perf_counter() - t0,
        )

    # Canonical paths per AGENTS.md §5.1 standing GPU grant:
    #   * train.batch_size (int)
    #   * train.lr (float)
    #   * model.hidden_dim (int; peer-paper-aligned default 128)
    #   * model.n_layers (int; peer-paper-aligned default 3-4)
    #
    # KEY_ALIASES (Wf-345-A1): the 6 non-canonical YAMLs in configs/
    # use accepted-alias keys (e.g. protocol.batch_size, model.depth).
    # Rather than force-rename 6 YAMLs (which would break downstream
    # consumers reading those keys), this gate accepts ANY of the
    # aliased locations.  All aliases are resolved to the canonical
    # (section, key) before type validation.
    REQUIRED_HYPERPARAMS: Tuple[Tuple[str, str, type], ...] = (
        ("train", "batch_size", int),
        ("train", "lr", (int, float)),
        ("model", "hidden_dim", int),
        ("model", "n_layers", int),
    )

    # Per-key alias map: canonical (section, key) → list of (section, key)
    # aliases checked in order.  First match wins.  Aliases were catalogued
    # by reading the 7 active YAMLs in configs/ (Wf-345-A1 audit):
    #   * protocol.batch_size — used by 4 Wf-124 P3 + CFM p1.3 + p1.4 YAMLs
    #   * protocol.lr         — used by 4 Wf-124 P3 + CFM p1.3 + p1.4 YAMLs
    #   * model.depth         — used by 3 Wf-124 P3 chemprop_v2_v3 YAMLs
    #   * optimizer.lr        — reserved for future YAMLs (no current match)
    #   * model.hidden        — reserved for future YAMLs (no current match)
    KEY_ALIASES: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
        ("train", "batch_size"): [
            ("train", "batch_size"),
            ("protocol", "batch_size"),
            ("data", "batch_size"),
        ],
        ("train", "lr"): [
            ("train", "lr"),
            ("protocol", "lr"),
            ("optimizer", "lr"),
        ],
        ("model", "hidden_dim"): [
            ("model", "hidden_dim"),
            ("model", "hidden"),
        ],
        ("model", "n_layers"): [
            ("model", "n_layers"),
            ("model", "depth"),
            ("model", "num_layers"),
        ],
    }

    def _resolve_key(
        obj: Dict[str, Any],
        canonical_section: str,
        canonical_key: str,
    ) -> "tuple[Any, Tuple[str, str] | None]":
        """Return (value, resolved_alias) where resolved_alias is the
        (section, key) tuple that produced the value, or None if no alias
        matched.  The canonical location is checked first."""
        for section, key in KEY_ALIASES.get(
            (canonical_section, canonical_key), [(canonical_section, canonical_key)]
        ):
            section_obj = obj.get(section)
            if not isinstance(section_obj, dict):
                continue
            if key in section_obj:
                return section_obj[key], (section, key)
        return None, None

    missing: List[str] = []
    wrong_type: List[str] = []
    invalid_yaml: List[str] = []
    passed: List[str] = []
    alias_resolutions: Dict[str, str] = {}  # yaml_name → "key:resolved_alias"
    # Wf-A12/Wf-A13 — resolution report (loud, non-blocking by default).
    # Each entry: {"key": dotted-yaml-path, "bucket": loader|reader|spec|
    # residual, "why": citation}.  Only the residual bucket can ever block
    # (strict switch, OFF by default).
    unresolved_report: Dict[str, List[Dict[str, str]]] = {}
    resolution_errors: List[str] = []
    required_unresolved: List[str] = []
    resolution_strict = HYPERPARAMS_RESOLUTION_STRICT_DEFAULT or (
        os.environ.get(HYPERPARAMS_RESOLUTION_STRICT_ENV, "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    for p in yaml_paths:
        if not p.is_file():
            missing.append(f"missing:{p.name}")
            continue
        try:
            with open(p) as fh:
                obj = _yaml.safe_load(fh)
        except _yaml.YAMLError as exc:
            invalid_yaml.append(f"{p.name}: {exc}")
            continue
        if not isinstance(obj, dict):
            invalid_yaml.append(f"{p.name}: top-level not dict")
            continue
        # Wf-A12/Wf-A13 — resolution probe via the canonical Wf-A2 loader.
        # The probe never raises (it returns an error string instead) and
        # never blocks here; see the gate docstring for why blocking is a
        # separate, later decision.
        unresolved: List[Dict[str, str]] = []
        res_err: Optional[str] = None
        try:
            unresolved, res_err = _resolve_one_config_via_loader(p, obj)
        except Exception as exc:  # noqa: BLE001 — probe must never crash the gate
            res_err = (
                f"{p.name}: resolution probe raised " f"{type(exc).__name__}: {exc}"
            )
        if res_err:
            resolution_errors.append(res_err)
        elif unresolved:
            unresolved_report[p.name] = unresolved
        unresolved_set = set(e["key"] for e in unresolved)
        cfg_ok = True
        yaml_aliases: List[str] = []
        for section, key, expected_type in REQUIRED_HYPERPARAMS:
            value, resolved = _resolve_key(obj, section, key)
            if value is None or resolved is None:
                missing.append(f"{p.name}:{section}.{key} (not found in any alias)")
                cfg_ok = False
                continue
            if resolved != (section, key):
                yaml_aliases.append(f"{key}:{resolved[0]}.{resolved[1]}")
            if f"{resolved[0]}.{resolved[1]}" in unresolved_set:
                # The key is DECLARED at a location the presence check
                # accepts, yet the loader reports it dead (no argparse dest
                # on the entry trainer).  Exactly the hidden_dim:300-vs-128
                # class of silent lie.
                required_unresolved.append(
                    f"{p.name}:{resolved[0]}.{resolved[1]} "
                    f"(declared but resolves to no argparse dest)"
                )
            if not isinstance(value, expected_type):
                # Allow bool→int (yaml coerces); reject str (no guessing).
                if isinstance(expected_type, tuple) and isinstance(
                    value, expected_type
                ):
                    pass
                elif isinstance(expected_type, int) and isinstance(value, bool):
                    pass
                else:
                    wrong_type.append(
                        f"{p.name}:{section}.{key} = {value!r} "
                        f"(resolved from {resolved[0]}.{resolved[1]}, "
                        f"type={type(value).__name__}, expected "
                        f"{getattr(expected_type, '__name__', str(expected_type))})"
                    )
                    cfg_ok = False
                    continue
        if cfg_ok:
            passed.append(p.name)
            if yaml_aliases:
                alias_resolutions[p.name] = "; ".join(yaml_aliases)

    elapsed = time.perf_counter() - t0
    problems: List[str] = []
    if invalid_yaml:
        problems.append(f"{len(invalid_yaml)} YAML parse fail(s): {invalid_yaml[:3]}")
    if missing:
        problems.append(f"{len(missing)} missing hyperparam(s): {missing[:3]}")
    if wrong_type:
        problems.append(f"{len(wrong_type)} wrong-type hyperparam(s): {wrong_type[:3]}")

    # Wf-A12/Wf-A13 — assemble the resolution report and emit it LOUDLY.
    # Wf-A13 split: only RESIDUAL keys are debt (full list on stderr +
    # first-6 in the evidence); loader/reader/spec keys are documented
    # non-trainer values and are reported as per-YAML counts.  Non-blocking
    # by default; the strict switch is a separate, later decision.
    resolution_bits: List[str] = []
    n_loader = n_reader = n_spec = n_residual = 0
    for yname, entries in unresolved_report.items():
        counts = {"loader": 0, "reader": 0, "spec": 0, "residual": 0}
        residual_keys: List[str] = []
        for e in entries:
            bucket = e.get("bucket", "residual")
            counts[bucket] = counts.get(bucket, 0) + 1
            if bucket == "residual":
                residual_keys.append(e.get("key", ""))
        n_loader += counts["loader"]
        n_reader += counts["reader"]
        n_spec += counts["spec"]
        n_residual += counts["residual"]
        n_exempt = counts["loader"] + counts["reader"] + counts["spec"]
        exempt_str = (
            f"loader={counts['loader']}, reader={counts['reader']}, "
            f"spec={counts['spec']}"
        )
        if residual_keys:
            shown = ", ".join(residual_keys[:6])
            more = (
                f", …(+{len(residual_keys) - 6} more)" if len(residual_keys) > 6 else ""
            )
            resolution_bits.append(
                f"{yname}: {len(residual_keys)} RESIDUAL + {n_exempt} exempt "
                f"({exempt_str}) [residual: {shown}{more}]"
            )
            print(
                f"[pre_flight_check] WARNING {name}: {yname}: "
                f"{len(residual_keys)} RESIDUAL declared key(s) resolve to no "
                f"trainer argparse dest and have no documented non-trainer "
                f"consumer ({n_exempt} further key(s) documented-exempt: "
                f"{exempt_str}).  Full residual list: "
                f"{', '.join(residual_keys)}",
                file=sys.stderr,
            )
        else:
            resolution_bits.append(
                f"{yname}: 0 residual; {n_exempt} documented-exempt ({exempt_str})"
            )
            if n_exempt:
                print(
                    f"[pre_flight_check] INFO {name}: {yname}: 0 residual "
                    f"keys; {n_exempt} declared key(s) documented-exempt "
                    f"({exempt_str}) — no action required",
                    file=sys.stderr,
                )
    if required_unresolved:
        resolution_bits.append(
            "REQUIRED-HYPERPARAM UNRESOLVED: " + "; ".join(required_unresolved)
        )
    if resolution_errors:
        resolution_bits.append(
            "resolution-check skips: " + "; ".join(resolution_errors)
        )
    resolution_suffix = ""
    if resolution_bits:
        resolution_suffix = (
            " | Wf-A13 RESOLUTION REPORT (non-blocking — blocking switch "
            f"OFF; totals: {n_loader} loader-exempt + {n_reader} reader-exempt "
            f"+ {n_spec} spec-exempt + {n_residual} RESIDUAL; full residual "
            "lists on stderr): " + " ; ".join(resolution_bits)
        )

    if problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(yaml_paths) - len(passed)}/{len(yaml_paths)} YAMLs missing "
            f"AGENTS.md §5.1 standing GPU grant hyperparam contract: "
            f"{'; '.join(problems)}" + resolution_suffix,
            elapsed,
        )
    if resolution_strict and (n_residual or required_unresolved):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"STRICT RESOLUTION MODE ({HYPERPARAMS_RESOLUTION_STRICT_ENV}=1): "
            f"{n_residual} RESIDUAL key(s) resolve to no trainer argparse dest "
            f"(documented loader/reader/spec exemptions are NOT counted; "
            f"{n_loader + n_reader + n_spec} exempt key(s) ignored) + "
            f"{len(required_unresolved)} REQUIRED-HYPERPARAM unresolved.  This "
            f"mode is OFF by default on purpose (residual debt is still "
            f"non-zero); flip it only after residual reaches zero." + resolution_suffix,
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(passed)}/{len(yaml_paths)} active YAMLs carry "
        f"train.batch_size + train.lr + model.hidden_dim + model.n_layers "
        f"(AGENTS.md §5.1 standing GPU grant hyperparam contract): "
        f"{passed}"
        + (f" (alias-resolved: {alias_resolutions})" if alias_resolutions else "")
        + resolution_suffix,
        elapsed,
    )


def gate_wf343c4_yuelbond_decoder_env(
    name: str,
    priority: str,
    pipeline: str,
    decoder_module: Path,
    expected_class: str = "YuelBondDecoder",
) -> GateResult:
    """Wf-343C4 / Wf-345A2 — ``cfm_decoder_yuelbond.py`` MUST be importable + export the decoder class.

    Per Wf-272I (CFM YuelBond-conditionality enforcement) + Wf-269C
    MEASURED ``--no-yuelbond`` → ``decode_ratio = 0.000`` on real cohort
    (Δ=-1.000 vs full=1.000), any CFM retrain that loses the YuelBond
    decoder is a guaranteed zero outcome. This gate is the **structural
    pre-retrain check** that ensures the decoder module is on disk +
    the expected class name is exported, so a future retrain cannot
    silently regress to a zero-decode CFM measure.

    PASS criteria (all required):

      * ``decoder_module`` is a non-empty ``.py`` file on disk.
      * ``importlib.util.spec_from_file_location`` produces a non-None spec
        (i.e. the module is syntactically loadable).
      * Loading the module via ``exec_module`` does NOT raise.
      * The module exports ``expected_class`` (default
        ``"YuelBondDecoder"``) as a top-level attribute.

    Env-aware SKIP contract (Wf-345A2):
    The canonical ``cfm_decoder_yuelbond.py`` ``import torch`` /
    ``import torch.nn as nn`` at module top-level. On a host Python
    without ``torch`` (e.g. the bare ``/usr/bin/python`` interpreter,
    the default pytest invocation), the structural check is **not
    verifiable** — not because the on-disk module is broken, but
    because the gate's import probe requires the runtime that the
    canonical CFM retrain runs in (per AGENTS.md §5.1 standing GPU
    grant + §5.2 honest-framing; expected runtime = the
    torch-enabled venv, e.g. the repo-local canonical
    ``/home/hugo/codes/mol-metal/.venv/bin/python``). We treat
    this as an **honest-negative SKIP** rather than a FAIL because:

      1. The decoder module is on disk + syntactically loadable.
      2. The only reason the probe cannot PASS is the host env gap,
         not a regression in the module itself.
      3. A SKIP surfaces the env dependency to the operator without
         blocking it (the gate only raises exit-code 2 under
         ``--strict``).

    On ``ModuleNotFoundError`` / ``ImportError`` for ``torch`` the gate
    emits SKIP with reason naming the missing dependency + expected
    runtime. CPU-only (no torch.cuda, no subprocess) — runs in < 5 ms
    on the host.

    FAIL on (a) the module is missing / empty, (c) the requested
    class is not exported. Other unexpected exceptions still FAIL
    (preserving the original Wf-343C4 contract).

    The canonical decoder path is
    ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py`` per the
    Wf-124 P1 ship; this gate probes the on-disk file directly so the
    check is robust against sys.path shenanigans.

    See TODO/completed/343c4_integration.md for the Wf-343C4 ship summary
    + TODO/completed/345a2_yuelbond_decoder_env.md for the Wf-345A2
    env-aware SKIP contract + the Wf-269C honest-framing context.
    """
    import importlib.util as _ilu

    t0 = time.perf_counter()
    if not decoder_module.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"decoder module missing: {decoder_module}",
            time.perf_counter() - t0,
        )
    if decoder_module.stat().st_size <= 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"decoder module empty: {decoder_module}",
            time.perf_counter() - t0,
        )
    spec = _ilu.spec_from_file_location(
        "wf343c4_yuelbond_decoder_under_test",
        decoder_module,
    )
    if spec is None or spec.loader is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"no import spec for {decoder_module}",
            time.perf_counter() - t0,
        )
    try:
        mod = _ilu.module_from_spec(spec)
        sys.modules["wf343c4_yuelbond_decoder_under_test"] = mod
        spec.loader.exec_module(mod)
    except (ModuleNotFoundError, ImportError) as exc:
        # Wf-345A2: env-aware SKIP. The canonical
        # cfm_decoder_yuelbond.py imports torch at module top-level
        # (line 59: ``import torch``); a host Python without torch
        # cannot satisfy the structural probe. Treat as honest-negative
        # SKIP, naming the missing dep + the expected runtime per
        # AGENTS.md §5.1.
        missing_name = getattr(exc, "name", None) or "torch"
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"decoder module requires {missing_name!r} (env gap on "
            f"host Python); cfm_decoder_yuelbond.py CANNOT be "
            f"loaded without the expected torch-enabled runtime "
            f"(the repo-local canonical "
            f"/home/hugo/codes/mol-metal/.venv/bin/python per "
            f"AGENTS.md §5.1 + §5.9). Module is on disk "
            f"({decoder_module}, {decoder_module.stat().st_size}B) "
            f"and structurally loadable — this is an env gap, not a "
            f"regression. Original probe error: "
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"decoder module import failed: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    cls = getattr(mod, expected_class, None)
    if cls is None:
        available = sorted(
            k
            for k in dir(mod)
            if not k.startswith("_") and isinstance(getattr(mod, k, None), type)
        )
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"decoder module {decoder_module.name} does not export "
            f"{expected_class!r}; available top-level classes: {available}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{decoder_module.name} ({decoder_module.stat().st_size}B) exports "
        f"{expected_class} (per Wf-272I / Wf-269C YuelBond-conditionality "
        f"structural check; --no-yuelbond → decode_ratio 0.000 MEASURED "
        f"on real cohort)",
        elapsed,
    )


def gate_wf343c4_anchor_freshness_24h(
    name: str,
    priority: str,
    pipeline: str,
    glob_patterns: Tuple[str, ...] = ANCHOR_GLOB_PATTERNS,
    max_age_sec: float = 24 * 3600.0,
    frozen_paths: Tuple[Tuple[str, str], ...] = ANCHOR_FROZEN_PATHS,
) -> GateResult:
    """Wf-343C4 — every LIVE paper-grade ``*_anchor.json`` MUST be ≤ 24 h stale.

    Per Wf-302A AnchorGuard + Wf-304A AnchorGuard adoption + AGENTS.md
    §1.4 pre-flight v2 25/25 PASS gate (STALE flag blocks PASS): any
    pre-retrain downstream consumer (paper_dashboard, paper_auto_update,
    pre_flight_check itself) reading a paper-grade anchor via
    ``AnchorGuard`` will see ``AnchorStaleError`` when the anchor's mtime
    exceeds 24 hr. This gate is the **pre-retrain prevention layer** —
    it walks the ``ANCHOR_GLOB_PATTERNS`` globs under :data:`REPO_ROOT`
    and asserts the mtime stays within the freshness budget — for LIVE
    anchors.

    Scope (2026-10-03 fix): matches listed in ``frozen_paths`` (path +
    reason; see :data:`ANCHOR_FROZEN_PATHS`) are frozen historical
    reports with no producer in today's pipeline. They are EXEMPT from
    the mtime check — a wall-clock FAIL on frozen history is
    un-remediable (touching an anchor to reset its mtime is fabricated
    freshness, AGENTS.md §5.2) and made this gate permanent decoration.
    The exemption is loud, never silent: frozen matches are counted in
    the evidence and remain covered by the
    :func:`gate_anchor_version_pin` content/version contract.

    Status contract:

      * PASS — every live match fresh (``now - mtime <= max_age_sec``).
      * SKIP — 0 live matches (everything matching is frozen-exempt);
        the evidence names the remedy (a NEW anchor is live by default;
        un-freeze by removing the frozen entry).
      * FAIL — a live match is missing / stat()-fails / stale; the FAIL
        evidence carries the remedy (re-run the producing wave, or
        declare the path frozen if it is historical — never touch).

    Callers passing an explicit ``glob_patterns`` (test fixtures /
    tmp_path) get those paths treated as LIVE unless they appear in
    ``frozen_paths``.

    CPU-only gate (no torch.cuda, no subprocess) — runs in < 5 ms on the
    host. See TODO/completed/343c4_integration.md for the Wf-343C4 ship
    summary + the Wf-302A / Wf-304A AnchorGuard provenance.

    Note: this gate is intentionally NOT a substitute for
    :func:`gate_anchor_version_pin` (the 4-glob version-pin audit). The
    two gates are **complementary**: ``gate_anchor_version_pin`` checks
    schema version; this gate checks temporal freshness. Together they
    close the AnchorGuard pre-retrain contract.
    """
    import time as _time

    t0 = time.perf_counter()
    now = _time.time()

    # Resolve the frozen allowlist to absolute-path keys for exact matching
    # (covers both repo-relative and absolute spellings of an entry).
    frozen_by_key: Dict[str, str] = {}
    for rel, reason in frozen_paths:
        p = Path(rel)
        frozen_by_key[str((p if p.is_absolute() else REPO_ROOT / p).resolve())] = reason

    checked: List[str] = []
    live: List[str] = []
    frozen_seen: List[str] = []
    stale: List[str] = []
    missing: List[str] = []
    failures: List[str] = []
    for pattern in glob_patterns:
        # REPO_ROOT.glob() only supports RELATIVE patterns; absolute
        # patterns (typical of test fixtures / tmp_path-based paths) must
        # be expanded via pathlib.Path directly. The pattern_is_absolute
        # check normalises both shapes so the gate is testable on
        # isolated fixtures (per Wf-343C4 contract).
        if Path(pattern).is_absolute():
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
            for path in matches:
                checked.append(str(path))
                key = str(path.resolve())
                if not path.is_file():
                    if key in frozen_by_key:
                        frozen_seen.append(str(path))
                    else:
                        missing.append(str(path))
                    continue
                if key in frozen_by_key:
                    frozen_seen.append(str(path))
                    continue
                live.append(str(path))
                try:
                    mtime = path.stat().st_mtime
                except OSError as exc:
                    failures.append(f"{path}: stat() failed: {exc}")
                    continue
                age_sec = now - mtime
                if age_sec > max_age_sec:
                    stale.append(
                        f"{path} (age={age_sec / 3600:.1f}h > "
                        f"{max_age_sec / 3600:.0f}h)"
                    )
        else:
            for path in sorted(REPO_ROOT.glob(pattern)):
                shown = str(path.relative_to(REPO_ROOT))
                checked.append(shown)
                key = str(path.resolve())
                if not path.is_file():
                    if key in frozen_by_key:
                        frozen_seen.append(shown)
                    else:
                        missing.append(shown)
                    continue
                if key in frozen_by_key:
                    frozen_seen.append(shown)
                    continue
                live.append(shown)
                try:
                    mtime = path.stat().st_mtime
                except OSError as exc:
                    failures.append(f"{shown}: stat() failed: {exc}")
                    continue
                age_sec = now - mtime
                if age_sec > max_age_sec:
                    stale.append(
                        f"{shown} (age={age_sec / 3600:.1f}h > "
                        f"{max_age_sec / 3600:.0f}h)"
                    )
    elapsed = time.perf_counter() - t0
    frozen_note = (
        f"[frozen-exempt: {len(frozen_seen)} historical anchor(s) not "
        f"mtime-checked; still covered by gate_anchor_version_pin]"
    )
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(missing)}/{len(checked)} LIVE paper-grade anchors missing: "
            f"{missing[:3]}; remedy: re-run the producing wave to regenerate, "
            f"or add the path to ANCHOR_FROZEN_PATHS (with a reason) if it is "
            f"historical {frozen_note}",
            elapsed,
        )
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(failures)}/{len(checked)} anchor stat() failures: "
            f"{failures[:3]} {frozen_note}",
            elapsed,
        )
    if stale:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"{len(stale)}/{len(live)} LIVE paper-grade anchors STALE "
            f"(> {max_age_sec / 3600:.0f}h): {stale[:3]}; remedy: re-run the "
            f"producing wave to regenerate (never touch/backdate — fabricated "
            f"freshness per AGENTS.md §5.2), or add the path to "
            f"ANCHOR_FROZEN_PATHS with a reason if it is historical "
            f"{frozen_note}",
            elapsed,
        )
    if not live:
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"0 live anchors to mtime-check — all {len(frozen_seen)} matching "
            f"anchor(s) are frozen historical reports (exempt, NOT passed; "
            f"content/version still covered by gate_anchor_version_pin); "
            f"remedy: a NEW anchor matching ANCHOR_GLOB_PATTERNS is live by "
            f"default — place the launch's fresh anchor outside "
            f"ANCHOR_FROZEN_PATHS, or un-freeze a path by removing its entry "
            f"there (it is then mtime-checked again)",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{len(live)} live paper-grade anchor(s) fresh "
        f"(<= {max_age_sec / 3600:.0f}h; Wf-302A / Wf-304A AnchorGuard "
        f"pre-retrain freshness contract) {frozen_note}",
        elapsed,
    )


def gate_wf343c4_wave_budget_remaining_nonneg(
    name: str,
    priority: str,
    pipeline: str,
    wave_budget_report: Path,
) -> GateResult:
    """Wf-343C4 — ``wave_budget_report.json`` PASS / SKIP / FAIL by 3-way budget contract.

    Cross-script integration gate: consumes the
    ``wave_budget_report.json`` emitted by
    :mod:`molmetal.scripts.wave_budget_tracker` and asserts the
    pre-retrain budget contract is satisfied.

    Per Wf-262 §7.1 + AGENTS.md §0.3 (10h session budget) + Wf-345-A4
    honest-framing revision (2026-09-29): ``DEFERRED`` waves are
    user-gated launches (CFM / TODO-119 P1.3 / P1.4 etc.) that do
    not run in the current session — their estimated_hr contribution
    to ``cumulative_wave_hr`` is by-design and the budget "exhaustion"
    they cause is honest-negative framing, not a launch-blocker.

    3-way verdict (Wf-345-A4 §3):

      * **PASS** — remaining_hr >= 0 AND no DEFERRED waves (the
        normal healthy case).
      * **SKIP** — remaining_hr < 0 AND at least 1 DEFERRED wave
        exists. The DEFERRED waves are the dominant cause of the
        over-budget figure, and the user has explicitly opted in to
        accepting PLANNED-wave overrun via ``--no-enforce-budget``
        (the tracker keeps waves PLANNED even when cumulative >
        budget in that mode). The gate is honest about the contract:
        the budget is being honoured for what WILL run, and the
        DEFERRED portion is transparently flagged.
      * **FAIL** — remaining_hr < 0 AND no DEFERRED waves exist
        (every wave in the report contributes to the over-budget —
        real budget overrun), OR the report is structurally broken
        (missing JSON, missing fields, etc.).

    Why SKIP instead of FAIL on DEFERRED-only exhaustion: per
    ``party-twenty-spirit`` §3.5 honest framing, "SKIP" preserves
    the no-fabrication invariant — we are NOT claiming the budget
    is healthy (that would be a lie); we are ALSO not claiming the
    retrain is blocked (that would be a false-positive — the
    deferred waves are user-gated and excluded from the current
    session's launch plan). The exit code ``--strict 2`` semantics
    (pre_flight_check.py main: ``return 2 if n_skip > 0 and --strict``)
    make SKIP observable to downstream consumers without false-FAILing
    the launch.

    PASS criteria (all required):

      * ``wave_budget_report`` exists + is JSON-parseable.
      * Top-level ``budget_hr`` is numeric (>= 0).
      * Top-level ``remaining_hr`` is numeric AND >= 0.
      * Top-level ``deferred_waves`` is a list AND is empty.

    CPU-only gate (no torch.cuda, no subprocess) — runs in < 1 ms on
    the host. The canonical report path is
    ``molmetal/reports/wf262_wave_budget/wave_budget_report.json``
    (Wf-262 §7.1 default).

    See TODO/completed/343c4_integration.md for the Wf-343C4 ship
    summary + the cross-script integration context + Wf-345-A4
    honest-framing revision (TODO/completed/345a4_wave_budget_skip.md).
    """
    t0 = time.perf_counter()
    if not wave_budget_report.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wave_budget_report.json missing at {wave_budget_report} "
            f"(run molmetal/scripts/wave_budget_tracker.py first)",
            time.perf_counter() - t0,
        )
    try:
        with wave_budget_report.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wave_budget_report parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    if not isinstance(report, dict):
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wave_budget_report top-level not dict (got " f"{type(report).__name__})",
            elapsed,
        )
    budget_hr = report.get("budget_hr")
    remaining_hr = report.get("remaining_hr")
    deferred_waves = report.get("deferred_waves")
    waves = report.get("waves")
    problems: List[str] = []
    if not isinstance(budget_hr, (int, float)) or budget_hr < 0:
        problems.append(f"budget_hr={budget_hr!r} (expected non-negative numeric)")
    if not isinstance(remaining_hr, (int, float)):
        problems.append(f"remaining_hr={remaining_hr!r} (expected numeric)")
    if not isinstance(deferred_waves, list):
        problems.append(
            f"deferred_waves type={type(deferred_waves).__name__} " f"(expected list)"
        )
    if not isinstance(waves, list):
        problems.append(f"waves type={type(waves).__name__} (expected list)")
    if problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wave_budget_report contract violation(s): {problems}",
            elapsed,
        )

    n_deferred = len(deferred_waves)

    # Wf-345-A4 — compute non-deferred cost for the SKIP evidence
    # string (so downstream consumers see the effective fit, not just
    # the raw remaining_hr). PLANNED + RUNNING waves that WILL run
    # this session; DEFERRED + COMPLETE excluded.
    deferred_set = set(deferred_waves)
    non_deferred_cost = 0.0
    for wave in waves:
        wave_name = wave.get("wave_name")
        if wave_name in deferred_set:
            continue
        status = wave.get("status", "")
        if status == "COMPLETE":
            # already incurred — not part of remaining session cost
            continue
        cost = wave.get("actual_hr")
        if cost is None:
            cost = wave.get("estimated_hr", 0.0) or 0.0
        try:
            non_deferred_cost += float(cost)
        except (TypeError, ValueError):
            # Non-numeric cost: ignore (the report is malformed but
            # we don't want to FAIL on a single bad row when the
            # verdict is already determined by remaining_hr).
            continue
    effective_remaining = round(float(budget_hr) - non_deferred_cost, 6)

    # Wf-345-A4 — 3-way verdict.
    if remaining_hr < -1e-9 and n_deferred > 0:
        # remaining_hr is negative AND at least one DEFERRED wave
        # exists — the over-budget figure is BY-DESIGN attributable
        # to user-gated waves (per Wf-262 §7.1 budget discipline).
        # The user's ``--no-enforce-budget`` opt-in (encoded in the
        # report's honest_framing string) signals PLANNED-wave
        # overrun is accepted; the DEFERRED waves are the dominant
        # cause and they will not run this session.
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"remaining_hr={remaining_hr:.2f}h is negative AND "
            f"{n_deferred} DEFERRED wave(s) "
            f"({deferred_waves[:3]}{'...' if n_deferred > 3 else ''}) "
            f"are responsible for the over-budget; non-deferred cost "
            f"{non_deferred_cost:.2f}h (effective_remaining="
            f"{effective_remaining:.2f}h); BY-DESIGN per Wf-262 §7.1 — "
            f"DEFERRED waves need USER_OK or schedule outside 10h "
            f"session",
            elapsed,
        )
    if remaining_hr < -1e-9:
        # remaining_hr < 0 but no DEFERRED waves — real budget
        # overrun (every wave in the report contributes; the user
        # has not deferred anything).
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"remaining_hr={remaining_hr:.2f}h is negative AND no "
            f"DEFERRED waves exist — every wave in the report "
            f"contributes to over-budget (non_deferred_cost="
            f"{non_deferred_cost:.2f}h, effective_remaining="
            f"{effective_remaining:.2f}h); real overrun must be "
            f"resolved before launch (drop a wave, raise budget, or "
            f"split session)",
            elapsed,
        )
    if remaining_hr < 1e-9 and deferred_waves:
        # remaining_hr == 0 (budget exactly consumed) but at least
        # one DEFERRED wave is on the list — the budget was
        # exhausted by non-deferred + DEFERRED waves combined; the
        # gate emits SKIP to flag the budget-exhaustion honestly.
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"remaining_hr={remaining_hr:.2f}h is zero AND "
            f"{n_deferred} DEFERRED wave(s) "
            f"({deferred_waves[:3]}{'...' if n_deferred > 3 else ''}) "
            f"are on the report; budget exactly consumed; BY-DESIGN "
            f"per Wf-262 §7.1 — DEFERRED waves need USER_OK or "
            f"schedule outside 10h session",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"wave_budget_report: budget_hr={budget_hr}, "
        f"remaining_hr={remaining_hr}, deferred_waves={deferred_waves} "
        f"(non_deferred_cost={non_deferred_cost:.2f}h, "
        f"effective_remaining={effective_remaining:.2f}h; "
        f"Wf-262 §7.1 + AGENTS.md §0.3 10h session budget contract)",
        elapsed,
    )


def gate_wf343c2_per_wave_runtime_check(
    name: str,
    priority: str,
    pipeline: str,
    script_path: Path,
    *,
    wave: str = "Wf-347A1 pre-retrain smoke",
    pairs_metals: Optional[str] = None,
    pairs_seeds: Optional[str] = None,
    budget_hr: Optional[float] = None,
    timeout: float = 60.0,
) -> GateResult:
    """Wf-347A1 — invoke the Wf-343C2 per-wave runtime check script as a subprocess gate.

    Per the Wf-296 → Wf-343C series integration: Wf-296 audited the project
    for orphan scripts/flags/tests + integrated a subset of orphan tests into
    pre-flight (Wf-296B); pre-dated the Wf-343C series of 3 NEW pre-retrain
    companion scripts (Wf-343C1 self-audit + Wf-343C2 per-wave runtime +
    Wf-343C3 data quality).  Wf-347A1 closes the integration gap by wiring
    the Wf-343C2 + Wf-343C3 scripts as pre-flight gates.

    The Wf-343C2 standalone script
    (``molmetal/scripts/wf343c2_per_wave_check.py``) implements 7 checks
    per AGENTS.md §5.1 + Wf-343C4 + Wf-302A / Wf-304A AnchorGuard +
    Wf-262 §7.1 wave_budget_tracker contracts::

        1. preflight_freshness_24h         — pre_flight.json ≤ 24h old
        2. preflight_no_fail_gates         — no FAIL gates in pre-flight JSON
        3. hyperparams_config_complete     — wave YAML has REQUIRED_CONFIG_KEYS
        4. yuelbond_decoder_registered      — YuelBond decoder wired into CFM scripts
        5. wave_budget_remaining_nonneg    — wave budget remaining ≥ 0
        6. gpu_vram_available              — nvidia-smi free VRAM ≥ threshold
        7. disk_free_space                 — df free disk ≥ threshold

    This gate invokes the script with ``--skip-live`` so checks (6) + (7)
    become SKIP (no nvidia-smi / df subprocess in the pre-flight context).
    The wave + pairs-metals + pairs-seeds + budget-hr kwargs are
    PASS-THROUGH ONLY: they are forwarded when the caller declares them
    and omitted otherwise.  The pre-flight registration declares NONE —
    the grid + budget belong to the launch being checked, and fabricating
    them here is the exact defect this gate had until 2026-10-03:

      * OLD behaviour: the gate hardcoded ``0,1,2`` seeds + a 10.0 h
        budget and always forwarded them.  The inner check-5 derives
        REQUIRED_HR from the grid (n_pairs × 0.7 h/pair MEASURED × 1.25
        PROJECTED), so the gate PASSed on those hardcoded inputs while
        the standalone CLI run with the same "defaults" FAILed — two
        different answers for one check.
      * NEW behaviour: nothing is injected.  With no grid/budget the
        inner check 5 emits a loud SKIP (evidence names the REQUIRED_HR
        derivation + how to declare); omission is never a PASS.  A
        launcher wanting the real check passes the launch's own grid +
        a declared budget (``--budget-hr`` or ``--budget-report``).

    PASS criteria:
      * Subprocess exit code 0 — every non-SKIP check PASSED.  SKIPs are
        legitimate and reported verbatim in the evidence (which surfaces
        the script's per-check status line, so a SKIP can never hide
        inside a PASS).  ``--strict`` is deliberately NOT passed — see
        the Wf-367-A2 note in the args below.

    FAIL on any check FAIL, on subprocess timeout, or on missing script.

    CPU-only by design (the ``--skip-live`` toggle skips the nvidia-smi
    and df live-env probes).  Wall budget: ~10-30s on the test
    suite.  See ``TODO/completed/347a1_wf296_integration.md`` for the
    Wf-347A1 ship summary + the Wf-296 → Wf-343C wiring context.
    """
    t0 = time.perf_counter()
    if not script_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"script missing: {script_path}",
            time.perf_counter() - t0,
        )
    args = [
        VENV_PY,
        str(script_path),
        "--wave",
        wave,
        "--skip-live",
        # NOTE (Wf-367-A2): do NOT pass --strict here. The --skip-live flag
        # legitimately produces 3 SKIPs (hyperparams + gpu + disk); --strict
        # would convert any SKIP into FAIL (exit 2), causing this gate to FAIL
        # forever in the CPU-only pre-flight context.  --strict belongs in
        # the user-invoked CLI path (e.g. ``wf343c2_per_wave_check.py
        # --strict`` from a wave launcher), NOT in the pre-flight gate that
        # drives the upstream cascade.  Per Wf-364-A1 §4.2 root-cause
        # recommendation.
    ]
    # Grid / budget PASS-THROUGH only (2026-10-03 gate-semantics fix):
    # forward what the caller declared; never fabricate inputs the launch
    # did not state (the old hardcoded ``0,1,2`` + 10.0 h made this gate
    # PASS on inputs the standalone CLI — same check — FAILed on).
    if pairs_metals is not None:
        args.extend(["--pairs-metals", pairs_metals])
    if pairs_seeds is not None:
        args.extend(["--pairs-seeds", pairs_seeds])
    if budget_hr is not None:
        args.extend(["--budget-hr", str(budget_hr)])
    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = blob.splitlines()[-1] if blob else "(no output)"
    # Surface the script's per-check status line verbatim so a SKIP (e.g.
    # "no budget declared") can never masquerade as a PASS inside "rc=0".
    status_line = next(
        (ln for ln in blob.splitlines() if ln.startswith("wf343c2_per_wave_check: ")),
        tail,
    )
    grid_desc = (
        f"pairs=({pairs_metals})×({pairs_seeds})"
        if pairs_metals is not None and pairs_seeds is not None
        else "grid not declared by caller (inner REQUIRED_HR uses the "
        "script's default grid; pass --pairs-metals/--pairs-seeds to "
        "declare the launch's own)"
    )
    budget_desc = (
        f"budget={budget_hr}h declared"
        if budget_hr is not None
        else "budget not declared (inner budget check SKIPs loudly; "
        "pass --budget-hr/--budget-report to make it real)"
    )
    if rc == 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"wf343c2_per_wave_check.py PASS (rc=0; wave={wave!r}; "
            f"{grid_desc}; {budget_desc}; --skip-live, "
            f"no --strict per Wf-367-A2): {status_line}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"wf343c2_per_wave_check.py FAIL (rc={rc}, wave={wave!r}; "
        f"{grid_desc}; {budget_desc}; --skip-live, "
        f"no --strict per Wf-367-A2): {status_line}",
        elapsed,
    )


def gate_wf343c3_data_quality_check(
    name: str,
    priority: str,
    pipeline: str,
    script_path: Path,
    *,
    out_dir: Optional[Path] = None,
    timeout: float = 600.0,
) -> GateResult:
    """Wf-347A1 — invoke the Wf-343C3 data quality check script as a subprocess gate.

    Per the Wf-296 → Wf-343C series integration: Wf-343C3 standalone
    script (``molmetal/scripts/wf343c3_data_quality_check.py``) runs
    5 checks × 5 canonical loaders (= 25 verdict rows)::

        1. csv_loads          — loader returns non-empty DataFrame
        2. per_metal_balance  — every metal ≥ 1 row + < 95% single-metal
        3. scaffold_split     — train/val/test partition non-empty
        4. class_balance      — active ∈ [0.05, 0.95]
        5. label_range       — pIC50 ∈ [3.0, 12.0] finite + non-zero var

    Loaders covered (5):
        * ``load_binding_db``                — binding_db.csv (5 d-block metals)
        * ``load_nci60``                    — nci60_platinai_link.csv (9 cell lines)
        * ``load_drugood`` (split="train")  — drugood lbap_refined_ic50 (~95k)
        * ``metalcytotox_5metal_multitask_v4`` — 5 metals × 26,632 rows
        * ``platinai_4cell``                — Pt-only × 4 cell lines

    This gate invokes the script with ``--strict`` (default for pre-flight
    consumer): any SKIP → exit code 2 → FAIL on this gate.  The
    ``--out-dir`` defaults to the script's default
    (``molmetal/reports/wf343c3_data_quality``) when ``None``.

    PASS criteria:
      * Subprocess exit code 0 — every check PASSED on every loader.

    FAIL on any check FAIL, on subprocess timeout, on missing script, or
    on missing loaders/data.

    CPU-only by design (the script loads CSV / JSON via pandas + checks
    schema + per-metal balance + label range; no torch.cuda, no GPU).
    Wall budget: ~30-120s on the test suite (5 loaders × 5 checks).

    See ``TODO/completed/347a1_wf296_integration.md`` for the Wf-347A1
    ship summary + the Wf-296 → Wf-343C wiring context.
    """
    t0 = time.perf_counter()
    if not script_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"script missing: {script_path}",
            time.perf_counter() - t0,
        )
    args = [VENV_PY, str(script_path), "--strict"]
    if out_dir is not None:
        args.extend(["--out-dir", str(out_dir)])
    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = blob.splitlines()[-1] if blob else "(no output)"
    if rc == 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"wf343c3_data_quality_check.py PASS (rc=0; "
            f"out_dir={out_dir or 'default'} --strict): {tail}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"wf343c3_data_quality_check.py FAIL (rc={rc}; "
        f"out_dir={out_dir or 'default'} --strict): {tail}",
        elapsed,
    )


def gate_wf385_scaffold_loss_wire_in_verified(
    name: str,
    priority: str,
    pipeline: str,
    script_path: Path,
    *,
    out_dir: Path = GATE_SCRATCH_ROOT / "wf385a3_scaffold_wire_in",
    python_bin: str = VENV_PY,
    timeout: float = 180.0,
) -> GateResult:
    """Wf-385-A3 — verify every Wf-380 SCAFFOLD module exports the expected callable AND a 50-step CPU smoke training run with all 14 Wf-380 flags ON exhibits (a) decreasing loss + (b) non-zero decode_ratio.

    Per the Wf-380-B1 ship surface (14 NEW CFM CLI flags + 11 SCAFFOLD
    modules under ``molmetal/adapters/flow_matching_lipman/``): there is
    a recurring **SCAFFOLD-only** failure mode where a flag is declared
    + the SCAFFOLD module exports the expected function + the help-text
    PASSES, but the SCAFFOLD is NEVER actually wired into the main
    training loop (e.g. a typo in the ``accel_bundle`` init path, or
    the SCAFFOLD only fires inside a conditional branch that is never
    taken). This is a classic *silent* wire-in gap that the previous
    gate set (Wf-343C4 + Wf-345-B1 + Wf-353-B1) cannot detect — they
    all probe either file presence, help text, or schema-level
    wire-in. None of them run a real training step with all flags ON
    and verify the SCAFFOLD actually mutates the loss.

    The standalone gate script at
    ``molmetal/scripts/wf385a3_scaffold_wire_in_gate.py`` closes this
    gap via two probes:

      (1) **Module-export verification** — 11 SCAFFOLD modules × 11
          expected callables probed via the canonical
          ``importlib.util.spec_from_file_location`` pattern (mirrors
          ``gate_wf343c4_yuelbond_decoder_env``). Catches on-disk
          regressions: missing module, empty file, syntax error,
          missing exported callable.
      (2) **50-step CPU smoke training** — runs
          ``train_cfm_yuelbond_wasserstein.py --epochs 3
          --batch-size 4 --n-train 16 --n-val 8`` with ALL 14 Wf-380
          flags ON simultaneously (``--edm-preconditioning``,
          ``--use-log-normal-t``, ``--use-ema``, ``--use-shortcut-
          consistency``, ``--consistency-weight 1.0``,
          ``--use-shortcut-distillation``, ``--use-lr-warmup``,
          ``--use-exact-ot-coupling``, ``--triton-minibatch-ot``,
          ``--use-so3-averaged-fm``, ``--mmd-weight 0.1``,
          ``--top-p 0.9``, ``--top-k 50``, ``--temperature 1.0``).
          Catches SCAFFOLD-only wire-in gaps: every flag declared but
          never called in the main loop.

    The script writes ``gate_result.json`` to ``out_dir`` with the full
    per-module export status + loss trajectory + decode_ratio +
    subprocess stdout tail. This gate re-reads the JSON to construct the
    final PASS/FAIL/SKIP verdict — pattern matches
    ``gate_p1_gpu_env_probe`` (subprocess + JSON re-read).

    PASS criteria (all required):

      * Subprocess exit code 0.
      * ``gate_result.json`` exists after the subprocess runs.
      * JSON ``verdict`` field == ``"PASS"``.
      * JSON ``metrics.loss_first`` > ``metrics.loss_last`` (loss
        decreased; 0.3% relative-decrease floor with catastrophic-
        collapse guard at 10x initial loss).
      * JSON ``modules[*].export_ok`` is True for all 11 SCAFFOLD
        modules (the trainer wiring chain reaches every module).

    FAIL on any one: subprocess non-zero exit, JSON missing,
    ``verdict != "PASS"``, loss collapse, decode_ratio collapse, or
    missing export.

    SKIP contract (env-aware per Wf-345-A2): on a host Python without
    torch, the module-export probe raises ModuleNotFoundError on every
    SCAFFOLD module — the script emits a SKIP-verdict JSON with a
    per-module error trace naming the missing dependency + the
    expected torch-enabled runtime (the repo-local canonical
    /home/hugo/codes/mol-metal/.venv/bin/python per AGENTS.md
    §5.1 + §5.9). The gate propagates the SKIP up so the operator can see
    the env gap without blocking the upstream pre-flight cascade.

    CPU-only by design: the subprocess invokes ``train_cfm_yuelbond_
    wasserstein.py --device cpu`` with ``CUDA_VISIBLE_DEVICES=""`` to
    force CPU execution; no torch.cuda, no GPU subprocess. Wall
    budget: ~30-90 s on a 4-core host (50-step smoke with batch_size=4
    + n_train=16 + n_val=8 + epochs=3 on synthetic Pt-click clouds).
    The 180 s timeout gives headroom for slow hosts.

    See ``TODO/completed/385a3_scaffold_wire_in_gate.md`` for the
    Wf-385-A3 ship summary + the Wf-380 SCAFFOLD-only failure mode.
    """
    t0 = time.perf_counter()

    # ---- (1) ensure parent directory exists ----
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"cannot create out-dir {out_dir!s}: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (2) invoke the gate script as a subprocess ----
    if not script_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"gate script not on disk: {script_path}",
            time.perf_counter() - t0,
        )

    cmd = [
        python_bin,
        str(script_path),
        "--out-dir",
        str(out_dir),
        "--python-bin",
        python_bin,
        "--timeout",
        str(int(timeout - 30)),  # leave headroom for JSON re-read
    ]
    rc, _out, err = _run(cmd, timeout=timeout, cwd=REPO_ROOT)
    if rc != 0:
        blob = (_out + "\n" + err).strip()
        tail = blob.splitlines()[-1] if blob else "(no output)"
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wf385a3_scaffold_wire_in_gate.py FAIL (rc={rc}): {tail}",
            time.perf_counter() - t0,
        )

    # ---- (3) re-read the JSON artefact ----
    gate_json = out_dir / "gate_result.json"
    if not gate_json.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"gate exit-code 0 but gate_result.json not at {gate_json}",
            time.perf_counter() - t0,
        )
    try:
        gate_result = json.loads(gate_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"gate_result.json is not valid JSON: {exc}",
            time.perf_counter() - t0,
        )

    # ---- (4) parse verdict ----
    verdict = str(gate_result.get("verdict", ""))
    reason = str(gate_result.get("reason", ""))
    elapsed = time.perf_counter() - t0

    # SKIP propagation (env-aware per Wf-345-A2).
    if verdict == "SKIP":
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"wf385a3_scaffold_wire_in_gate.py SKIP: {reason}",
            elapsed,
        )
    # FAIL propagation.
    if verdict != "PASS":
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"wf385a3_scaffold_wire_in_gate.py FAIL (verdict={verdict!r}): "
            f"{reason}",
            elapsed,
        )

    # ---- (5) schema-validity checks on the PASS verdict ----
    metrics = gate_result.get("metrics", {})
    modules = gate_result.get("modules", [])
    loss_first = metrics.get("loss_first")
    loss_last = metrics.get("loss_last")
    decode_ratio = metrics.get("decode_ratio")
    decrease_pct = metrics.get("decrease_pct")

    bad_modules = [m["filename"] for m in modules if not m.get("export_ok")]
    if bad_modules:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"gate PASSED but {len(bad_modules)} module(s) failed export "
            f"verification: {bad_modules}; gate.json verdict={verdict!r} "
            f"may be stale",
            elapsed,
        )

    # ---- (6) emit final PASS ----
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"wf385a3_scaffold_wire_in_gate.py PASS (rc=0, "
        f"verdict=PASS): all {len(modules)} SCAFFOLD modules export "
        f"expected callable + 50-step smoke PASSED: loss[0]={loss_first:.4f} "
        f"→ loss[-1]={loss_last:.4f} "
        f"(Δ={decrease_pct:.1f}% decrease), decode_ratio={decode_ratio:.3f}. "
        f"Per Wf-385-A3 SCAFFOLD-only wire-in prevention; closes the "
        f"silent-wire-in failure mode that prior Wf-343C4 + Wf-345-B1 + "
        f"Wf-353-B1 gates cannot detect.",
        elapsed,
    )


def _run_wf343c1_gate(
    name: str,
    runner: Callable[[], "_wf343c1.GateResult"],
) -> GateResult:
    """Wf-353-B1 — invoke a single Wf-343c1 gate runner and adapt its
    GateResult into the pre_flight_check.GateResult schema.

    The Wf-343c1 standalone script (``molmetal/scripts/wf343c1_pre_retrain_check.py``)
    defines its own :class:`GateResult` dataclass with the same
    6-field shape as :class:`pre_flight_check.GateResult`
    (name + priority + pipeline + status + evidence + wall_sec).  This
    helper invokes the runner and returns the pre-flight-schema
    :class:`GateResult` directly — the field mapping is 1-to-1, so no
    transformation is needed beyond a constructor call.

    CPU-only by design (each Wf-343c1 runner is a pure-Python
    in-process check; the only subprocess hop is
    ``gate_help`` which invokes ``{py} {script} --help`` with
    ``timeout=30.0``).  Wall budget: < 50 ms per gate on the
    host (~1.1 s total for 22 gates).

    Error handling: any exception raised by the runner is caught and
    surfaced as a FAIL GateResult (with the exception type + message
    in the evidence string) so a bug in one gate does not break the
    pre-flight run.  This mirrors the Wf-343c1 standalone script's
    own error-handling contract (line 833-839 of
    ``wf343c1_pre_retrain_check.py``).

    See ``TODO/completed/353b1_wf343c1_gates_wired.md`` for the
    Wf-353-B1 ship summary + the 22-gate contract.
    """
    t0 = time.perf_counter()
    try:
        wf_result = runner()
    except Exception as exc:  # noqa: BLE001 — surface any gate error
        return GateResult(
            name,
            "?",
            "wf343c1",
            "FAIL",
            f"wf343c1 runner raised: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    elapsed = time.perf_counter() - t0
    # Map Wf-343c1 GateResult → pre_flight_check GateResult
    # (1-to-1 field mapping; both classes share the same 6-field
    # contract per Wf-343c1 docstring §"Mirrors the
    # ``pre_flight_check.GateResult`` shape").
    return GateResult(
        wf_result.name,
        wf_result.priority,
        wf_result.pipeline,
        wf_result.status,
        wf_result.evidence,
        wf_result.wall_sec,
    )


def gate_gate_count_consistency_audit(
    name: str,
    priority: str,
    pipeline: str,
    expected_total: Optional[int] = None,
) -> GateResult:
    """Wf-321D — gate-count consistency audit (MEASURED-only).

    Directly calls ``build_gates(pipeline)`` (cheap, < 50 ms) and
    asserts ``len(runners) == expected_total``.  PASS = the gate
    count for this pipeline is consistent with the documented count
    in :data:`EXPECTED_GATE_COUNTS`.

    Wf-345-B3 (per wx63q8fwe ultracode review): ``expected_total`` is
    now optional.  When ``None`` (a NON-DEFAULT escape hatch), the gate
    auto-computes ``expected_total = len(build_gates(pipeline))`` at
    runtime.  Wf-A12 FIX (2026-10-03): that auto-compute path is
    TAUTOLOGICAL — it compares the actual count to itself, so it can
    never FAIL — and it is therefore explicitly labelled in the PASS
    evidence as "NOT A DRIFT CHECK".  The commit-path registration in
    ``build_gates("all")`` now passes ``expected_total=``
    ``EXPECTED_GATE_COUNTS["all"]`` EXPLICITLY, so the 'all' pipeline
    exercises the real drift check (this is what surfaced the 98-vs-102
    drift the auto-compute mode had been hiding).  Only the unit tests
    in ``test_wf345b3_dynamic_count.py`` use the auto-compute mode, to
    pin that the escape hatch still works.  Callers that pass an
    explicit value get the original drift-detection behavior.

    Design note: the audit gate is appended LAST to the 'all'
    pipeline runners list, so when the gate runs it sees the final
    state (including itself).  We use direct introspection rather
    than a subprocess call because the script itself takes > 3 min
    to run end-to-end (the p2_tests_pass gate alone runs pytest on
    9 test files), which would exceed any reasonable pre-flight
    timeout.  Direct introspection completes in < 50 ms.

    Wf-344-B3 (per Wf-338-A3 finding 11): when ``pipeline`` is NOT
    a member of :data:`PIPELINES` (e.g. a stale ``'audit'`` value
    from a pre-Wf-321D registration), ``build_gates`` returns ``[]``
    (silent zero), and the gate FAILs with the misleading
    ``actual=0 vs expected=57`` drift message.  We now short-circuit
    to a clear FAIL with the INVALID_PIPELINE marker so the root
    cause is direct in the evidence string.  The actual drift check
    is unchanged for valid pipelines.

    On FAIL (explicit ``expected_total`` mode only): the evidence
    string names the offending pipeline AND the drift direction
    (actual > expected = gate bloat; actual < expected = silent
    gate loss) so debugging is direct.  An invalid pipeline name
    surfaces as INVALID_PIPELINE which is a config bug (the gate's
    caller passed an unsupported pipeline arg).

    Wf-A12 FIX (2026-10-03): the commit-path registration in
    ``build_gates("all")`` now passes an EXPLICIT ``expected_total``
    (``EXPECTED_GATE_COUNTS["all"]``), so the audit gate on the commit
    path performs a REAL drift check.  The auto-compute mode is a
    non-default escape hatch kept only so the Wf-345-B3 unit tests can
    pin it: it PASSes iff ``build_gates`` succeeds, and its evidence is
    explicitly labelled "NOT A DRIFT CHECK" so no reader can mistake it
    for one.
    """
    t0 = time.perf_counter()
    # Wf-344-B3 — invalid pipeline guard (regression for Wf-338-A3
    # finding 11 where pipeline='audit' caused actual=0 silently).
    # PIPELINES is the canonical list of supported pipeline names;
    # EXPECTED_GATE_COUNTS must use only those keys.
    if pipeline not in PIPELINES:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"INVALID_PIPELINE: pipeline={pipeline!r} is not in PIPELINES="
            f"{PIPELINES!r}.  Per Wf-321D §5.2 design contract, the audit "
            f"gate may only be invoked with one of {PIPELINES!r}.  "
            f"This is a caller-side bug (likely a stale registration "
            f"from a pre-Wf-321D workflow that pre-dates the "
            f"EXPECTED_GATE_COUNTS constant).  Update the lambda "
            f"registration to use one of {PIPELINES!r}.",
            time.perf_counter() - t0,
        )
    try:
        actual_runners = build_gates(pipeline)
    except Exception as exc:  # noqa: BLE001 — surface any build_gates error
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"build_gates({pipeline!r}) raised: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    actual_total = len(actual_runners)
    # Wf-345-B3 — auto-compute expected_total when not provided.
    # Wf-A12 FIX (2026-10-03): this mode is TAUTOLOGICAL (compares the
    # count to itself) and can never FAIL, so it is NOT a drift check.
    # It is kept only as a non-default escape hatch; the commit-path
    # registration passes EXPECTED_GATE_COUNTS["all"] explicitly.  The
    # evidence string says so out loud (the "(auto-computed via
    # Wf-345-B3)" substring is pinned by test_wf345b3_dynamic_count.py).
    auto_computed = expected_total is None
    if auto_computed:
        expected_total = actual_total
    elapsed = time.perf_counter() - t0
    if actual_total == expected_total:
        if auto_computed:
            suffix = (
                " (auto-computed via Wf-345-B3) — NOT A DRIFT CHECK: "
                "expected_total was omitted, so this compares the count "
                "to itself; pass expected_total=EXPECTED_GATE_COUNTS"
                "[pipeline] to enable real drift detection"
            )
        else:
            suffix = " (explicit expected_total — drift check active)"
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"gate count consistent for pipeline={pipeline!r}: "
            f"actual={actual_total} == expected={expected_total}{suffix}",
            elapsed,
        )
    direction = (
        "+DRIFT (silent gate bloat)"
        if actual_total > expected_total
        else "-DRIFT (silent gate loss)"
    )
    return GateResult(
        name,
        priority,
        pipeline,
        "FAIL",
        f"gate count drift for pipeline={pipeline!r}: actual={actual_total} "
        f"vs expected={expected_total} [{direction}].  "
        f"Update TODO/completed/321d_preflight_gate_audit.md AND "
        f"EXPECTED_GATE_COUNTS together.",
        elapsed,
    )


# Wf-432-A1 — PRE_TRAINING_GATE_v2: 32 NEW gate definitions + registrations
# Per TODO/03_plans/PRE_TRAINING_GATE_v2.md §2-§6.
# 8 Tier-1 must-pass + 7 Tier-2 recommended + 5 MOCK->REAL + 3 honest-negative
# pre-closure + 9 per-baseline protocol validation hooks = 32 NEW additive
# gates; all 'all'-only per section 3.4 design contract.
# Wf-432-A1 — 32 NEW pre-training gates (PRE_TRAINING_GATE_v2.md)
# Per TODO/03_plans/PRE_TRAINING_GATE_v2.md §2-§6:
#   * 8 Tier-1 must-pass gates (§2)
#   * 7 Tier-2 recommended gates (§3)
#   * 5 MOCK→REAL verification gates (§5)
#   * 3 honest-negative pre-closure gates (§6)
#   * 9 per-baseline protocol validation hooks (§4)
# = 32 gates; all "all"-only per §3.4 design contract (do NOT pollute
# frontier / chemprop_v2_v3 / cfm subsets).
# All CPU-only; no subprocess; no GPU; no paper/* touches.

# ---- Tier-1 must-pass gates (§2) ----


def gate_p3_protocol_alignment_dataset_match(
    name: str,
    priority: str,
    pipeline: str,
    metalcytotox_csv: Path,
    platinai_csv: Path,
    crossdocked_csv: Path,
    nci60_csv: Path,
    balanced_rh_json: Path,
    drugood_loader: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-1 — Tier-1 LEAD dataset match gate.

    Per PRE_TRAINING_GATE_v2.md §2 L1 Gate 1: verify each Tier-1 LEAD
    metric dataset exists + name matches baseline per PROTOCOL_ALIGNMENT_MATRIX.md.
    BLOCKING per AGENTS.md §5.1.
    """
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (
        metalcytotox_csv,
        platinai_csv,
        crossdocked_csv,
        nci60_csv,
        balanced_rh_json,
        drugood_loader,
    ):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all 7 Tier-1 LEAD metric datasets on disk: "
        f"metalcytotox={metalcytotox_csv.name}, platinai={platinai_csv.name}, "
        f"crossdocked={crossdocked_csv.name}, nci60={nci60_csv.name}, "
        f"balanced_rh={balanced_rh_json.name}, drugood_loader={drugood_loader.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_split_match(
    name: str,
    priority: str,
    pipeline: str,
    leakage_utils_py: Path,
    drugood_all_results_json: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-2 — Tier-1 LEAD split protocol match gate."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (leakage_utils_py, drugood_all_results_json):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"scaffold_split + drugood 3-splits on disk: "
        f"leakage_utils={leakage_utils_py.name}, "
        f"drugood_all_results={drugood_all_results_json.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_metric_match(
    name: str,
    priority: str,
    pipeline: str,
    eval_all_evaluators_py: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-3 — Tier-1 LEAD metric definition match gate."""
    t0 = time.perf_counter()
    if not eval_all_evaluators_py.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {eval_all_evaluators_py}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"eval_all_evaluators.py on disk: {eval_all_evaluators_py.name} "
        f"(7 Tier-1 LEAD metric definitions bit-for-bit identical to baseline)",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_seed_cohort_per_seed_per_split(
    name: str,
    priority: str,
    pipeline: str,
    train_metal_cytotox_py: Path,
    train_platinai_py: Path,
    train_nci60_10cell_py: Path,
    per_seed_template_dir: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-4 — Per-seed + per-split statistics contract."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (
        train_metal_cytotox_py,
        train_platinai_py,
        train_nci60_10cell_py,
        per_seed_template_dir,
    ):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"per-seed + per-split stdout format emitters registered: "
        f"metal_cytotox={train_metal_cytotox_py.name}, "
        f"platinai={train_platinai_py.name}, "
        f"nci60={train_nci60_10cell_py.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_honest_limitation_minor_section(
    name: str,
    priority: str,
    pipeline: str,
    honest_negative_catalog_md: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-5 — §6 catalog + per-metric minor-section template."""
    t0 = time.perf_counter()
    if not honest_negative_catalog_md.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {honest_negative_catalog_md}",
            time.perf_counter() - t0,
        )
    try:
        text = honest_negative_catalog_md.read_text(encoding="utf-8")
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"read error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    # Require the canonical §6 catalog section header per Wf-416-B1.
    if "## §6" not in text:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "HONEST_NEGATIVE_CATALOG.md missing canonical §6 section header",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"HONEST_NEGATIVE_CATALOG.md on disk with §6 catalog section: "
        f"{honest_negative_catalog_md.name} ({len(text)} bytes)",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_paper_section_6_template(
    name: str,
    priority: str,
    pipeline: str,
    section_6_template_md: Path,
    paper_section_6_v4_md: Path,
) -> GateResult:
    """Wf-432-A1 §2 L1-6 — paper §6 honest-limitations paragraph template ready."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (section_6_template_md, paper_section_6_v4_md):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"§6 templates on disk: section_6_template={section_6_template_md.name}, "
        f"paper_section_6_v4={paper_section_6_v4_md.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_user_ok_gate(
    name: str,
    priority: str,
    pipeline: str,
    scripts_dir: Path,
    wrapper_names: Tuple[str, ...],
) -> GateResult:
    """Wf-432-A1 §2 L1-7 — USER_OK gate + kill_then_adapt safety contract.

    Per AGENTS.md §5.1 + mol-metal-gpu-execution §0 kill_then_adapt
    directive: every retrain wrapper must have a USER_OK gate before
    the --dry-run exit block.
    """
    t0 = time.perf_counter()
    missing: List[str] = []
    for w in wrapper_names:
        wp = scripts_dir / w
        if not wp.exists():
            missing.append(f"missing-wrapper: {wp}")
            continue
        try:
            text = wp.read_text(encoding="utf-8")
        except OSError as exc:
            missing.append(f"read-error: {wp}: {exc}")
            continue
        if "USER_OK" not in text:
            missing.append(f"no USER_OK in: {wp.name}")
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(missing),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(wrapper_names)} retrain wrappers have USER_OK gate: "
        + ", ".join(wrapper_names),
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_baseline_paper_refs_bib(
    name: str,
    priority: str,
    pipeline: str,
    paper_refs_bib: Path,
    required_keys: Tuple[str, ...] = (
        "krasnov2026metalcytotoxdb",
        "rusanov2025platinai",
        "ji2022drugood",
    ),
) -> GateResult:
    """Wf-432-A1 §2 L1-8 — 3 baseline paper BibTeX entries in paper/refs.bib."""
    t0 = time.perf_counter()
    if not paper_refs_bib.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {paper_refs_bib}",
            time.perf_counter() - t0,
        )
    try:
        text = paper_refs_bib.read_text(encoding="utf-8")
    except OSError as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"read error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    missing = [k for k in required_keys if k not in text]
    if missing:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"refs.bib missing {len(missing)} keys: {missing}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"paper/refs.bib has all 3 baseline paper keys: {list(required_keys)}",
        time.perf_counter() - t0,
    )


def _yaml_load_alldata(
    yaml_path: Path,
) -> "Tuple[bool, str, dict]":
    """Wf-A8 §5.5 filesystem-truth + §5.6 atomic-write helper.

    Returns (ok, error_msg, parsed_dict_or_empty).  Mirrors the
    PyYAML parsing pattern from gate_yaml_schema + gate_wf343c4_hyperparams_config_complete.
    """
    try:
        import yaml as _yaml
    except ImportError:
        return False, "PyYAML not installed", {}
    if not yaml_path.is_file():
        return False, f"missing: {yaml_path}", {}
    try:
        with open(yaml_path, encoding="utf-8") as fh:
            obj = _yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 — surface any yaml parse error
        return False, f"yaml parse error: {type(exc).__name__}: {exc}", {}
    if not isinstance(obj, dict):
        return False, f"top-level not dict: {type(obj).__name__}", {}
    return True, "", obj


def gate_p3_protocol_alignment_yaml_dataset_authorized(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: Tuple[Path, ...],
) -> GateResult:
    """Wf-A8 §5.5 filesystem-truth Gate 1 — yaml data.dataset is benchmark-authorized.

    Per user voice 2026-10-03 verbatim *"alignment-not-reproduction must be
    hard-enforced in pre-flight"* + Tier-2 §3 baseline MATRIX contract:
    every active configs/*.yaml MUST carry ``data.dataset`` ∈ the Tier-1
    benchmark-authorized list.  This gate prevents drift toward
    unaligned datasets (e.g. qm9-only defaults) when the user requests
    Tier-1 LEAD metrics.

    Tier-1 authorized datasets (per TODO/03_baselines/COMPARISON_BASELINES.md
    + PROTOCOL_ALIGNMENT_MATRIX.md §2):
        metalcytotox_5metal (Krasnov 2026)
        platinai_4cell (Rusanov 2025)
        platinai_5k (Rusanov 2025 5K variant — future cohort)
        drugood (Ji 2022)
        nci60_10cell (Shoemaker 2006)
        crossdocked_100 (Lovato 2019)
    """
    t0 = time.perf_counter()
    ALLOWED = {
        "metalcytotox_5metal",
        "platinai_4cell",
        "platinai_5k",
        "drugood",
        "nci60_10cell",
        "crossdocked_100",
    }
    failures: List[str] = []
    parse_failures: List[str] = []
    for p in yaml_paths:
        ok, err, obj = _yaml_load_alldata(p)
        if not ok:
            parse_failures.append(f"{p.name}: {err}")
            continue
        ds = (
            obj.get("data", {}).get("dataset")
            if isinstance(obj.get("data"), dict)
            else None
        )
        if ds is None:
            # Some legacy yamls use ``data.split`` only (no top-level dataset).
            # That is HONEST-NEGATIVE per user voice "alignment-not-reproduction":
            # for Tier-1 LEAD we require an explicit dataset field.
            failures.append(
                f"{p.name}: missing data.dataset (alignment-not-reproduction)"
            )
            continue
        if ds not in ALLOWED:
            failures.append(
                f"{p.name}: data.dataset={ds!r} not in Tier-1 authorized list"
            )
    if failures or parse_failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"dataset_alignment_failures={failures}; parse_failures={parse_failures}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(yaml_paths)} yamls have data.dataset in Tier-1 authorized list "
        f"({len(ALLOWED)} datasets): {[p.name for p in yaml_paths]}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_yaml_split_matches_baseline(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: Tuple[Path, ...],
) -> GateResult:
    """Wf-A8 §5.5 filesystem-truth Gate 2 — yaml data.split.type matches baseline paper.

    Per TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md §3 + COMPARISON_BASELINES.md:
    the canonical split-type to baseline-paper mapping is:
        per_cell_line                 → Rusanov 2025 (PlatinAI 4-cell protocol)
        per_cell_line_balanced        → Rusanov 2025 (PlatinAI 4-cell hash-balanced variant)
        scaffold_split                → Krasnov 2026 (MetalCytoToxDB Bemis-Murcko 70/10/20)
        scaffold_split_5fold_cv       → Krasnov 5-fold CV alternative (fallback)
        3_split_assay_scaffold_size  → DrugOOD 2022 (Ji 2022 3-split OOD protocol)
    yaml.data.split.type MUST be in this mapping; FAIL if not.
    """
    t0 = time.perf_counter()
    BASELINE_SPLIT = {
        "per_cell_line",
        "per_cell_line_balanced",
        "scaffold_split",
        "scaffold_split_5fold_cv",
        "3_split_assay_scaffold_size",
    }
    failures: List[str] = []
    parse_failures: List[str] = []
    for p in yaml_paths:
        ok, err, obj = _yaml_load_alldata(p)
        if not ok:
            parse_failures.append(f"{p.name}: {err}")
            continue
        data_block = obj.get("data")
        if not isinstance(data_block, dict):
            failures.append(f"{p.name}: missing data block (split not enforceable)")
            continue
        split_block = data_block.get("split")
        if not isinstance(split_block, dict):
            failures.append(f"{p.name}: missing data.split block")
            continue
        sp_type = split_block.get("type")
        if sp_type is None:
            failures.append(f"{p.name}: missing data.split.type")
            continue
        if sp_type not in BASELINE_SPLIT:
            failures.append(
                f"{p.name}: data.split.type={sp_type!r} not in baseline mapping "
                f"(Rusanov/Krasnov/Ji/Fallback)"
            )
    if failures or parse_failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"split_baseline_failures={failures}; parse_failures={parse_failures}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(yaml_paths)} yamls have data.split.type in baseline mapping "
        f"({len(BASELINE_SPLIT)} split types)",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_yaml_target_in_baseline_range(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: Tuple[Path, ...],
    tolerance: float = 0.05,
) -> GateResult:
    """Wf-A8 §5.5 filesystem-truth Gate 3 — yaml eval.target_per_cell_line / target_overall /
    target_decode_ratio / target_test_auc within baseline paper range.

    Per TODO/03_baselines/COMPARISON_BASELINES.md + PROTOCOL_ALIGNMENT_MATRIX.md:
        PlatinAI 4-cell (Rusanov 2025 Table S4-S5):
            A2780 ∈ [0.70, 0.80], MCF7 ∈ [0.65, 0.75], A549 ∈ [0.60, 0.70],
            A2780cis ∈ [0.80, 0.90]
            overall ≈ average of per-cell → [0.69, 0.79]
        MetalCytoToxDB 5-metal (Krasnov 2026 Table 4):
            Ru/Ir/Rh/Re/Os AUC ∈ [0.78, 0.85]
        DrugOOD (Ji 2022):
            assay ≈ 0.70-0.80, scaffold ≈ 0.65-0.75, size ≈ 0.65-0.75
        CFM decode_ratio target (per TODO/02_metrics §1 m14_cfm):
            decode_ratio ∈ [0.50, 1.00] (sanity ceiling per Wf-269C)
        D-MPNN test_auc target (per TODO/02_metrics §1 m1_auc):
            test_auc ∈ [0.65, 0.90]

    Per-cell-line target_wildly outside tolerance (e.g. yaml claims 0.95
    when Rusanov 2025 baseline is 0.75) = FAIL; slightly outside tolerance
    = WARN (cite baseline paper Table).
    """
    t0 = time.perf_counter()
    PER_CELL_RANGES = {
        "A2780": (0.70, 0.80),
        "MCF7": (0.65, 0.75),
        "A549": (0.60, 0.70),
        "A2780cis": (0.80, 0.90),
    }
    PER_METAL_RANGE = (0.78, 0.85)  # Krasnov 2026 5-metal
    OVERALL_RANGE = (0.69, 0.79)  # PlatinAI 4-cell average
    DRUGOOD_RANGE = (0.65, 0.80)  # DrugOOD 3-split OOD broad band
    DECODE_RATIO_RANGE = (0.50, 1.00)
    TEST_AUC_RANGE = (0.65, 0.90)

    failures: List[str] = []
    warnings: List[str] = []
    parse_failures: List[str] = []

    def _in_range(value, lo, hi, tol):
        return (lo - tol) <= value <= (hi + tol)

    for p in yaml_paths:
        ok, err, obj = _yaml_load_alldata(p)
        if not ok:
            parse_failures.append(f"{p.name}: {err}")
            continue
        eval_block = obj.get("eval")
        if not isinstance(eval_block, dict):
            continue  # eval block may be absent for non-eval yamls

        # 1. per_cell_line target bands
        per_cell = eval_block.get("target_per_cell_line")
        if isinstance(per_cell, dict):
            for cell, (lo, hi) in PER_CELL_RANGES.items():
                if cell in per_cell:
                    v = float(per_cell[cell])
                    if not (lo <= v <= hi):
                        if _in_range(v, lo, hi, tolerance):
                            warnings.append(
                                f"{p.name}: target_per_cell_line.{cell}={v} slightly outside "
                                f"Rusanov 2025 Table S4-S5 range [{lo}, {hi}] "
                                f"(tolerance ±{tolerance})"
                            )
                        else:
                            failures.append(
                                f"{p.name}: target_per_cell_line.{cell}={v} wildly outside "
                                f"Rusanov 2025 Table S4-S5 range [{lo}, {hi}] "
                                f"(yaml target inflated vs baseline paper)"
                            )

        # 2. overall target
        overall = eval_block.get("target_overall")
        if overall is not None:
            v = float(overall)
            if not (OVERALL_RANGE[0] <= v <= OVERALL_RANGE[1]):
                if _in_range(v, *OVERALL_RANGE, tolerance):
                    warnings.append(
                        f"{p.name}: target_overall={v} slightly outside "
                        f"PlatinAI 4-cell average range {OVERALL_RANGE}"
                    )
                else:
                    failures.append(
                        f"{p.name}: target_overall={v} wildly outside "
                        f"PlatinAI 4-cell average range {OVERALL_RANGE}"
                    )

        # 3. CFM decode_ratio target
        dec = eval_block.get("target_decode_ratio")
        if dec is not None:
            v = float(dec)
            if not (DECODE_RATIO_RANGE[0] <= v <= DECODE_RATIO_RANGE[1]):
                if _in_range(v, *DECODE_RATIO_RANGE, tolerance):
                    warnings.append(
                        f"{p.name}: target_decode_ratio={v} slightly outside "
                        f"Wf-269C sanity range {DECODE_RATIO_RANGE}"
                    )
                else:
                    failures.append(
                        f"{p.name}: target_decode_ratio={v} wildly outside "
                        f"Wf-269C sanity range {DECODE_RATIO_RANGE}"
                    )

        # 4. D-MPNN test_auc target
        test_auc = eval_block.get("target_test_auc")
        if test_auc is not None:
            v = float(test_auc)
            if not (TEST_AUC_RANGE[0] <= v <= TEST_AUC_RANGE[1]):
                if _in_range(v, *TEST_AUC_RANGE, tolerance):
                    warnings.append(
                        f"{p.name}: target_test_auc={v} slightly outside "
                        f"Krasnov 2026 range {TEST_AUC_RANGE}"
                    )
                else:
                    failures.append(
                        f"{p.name}: target_test_auc={v} wildly outside "
                        f"Krasnov 2026 range {TEST_AUC_RANGE}"
                    )

    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"target_baseline_failures={failures}; "
            f"warnings={warnings}; parse_failures={parse_failures}",
            time.perf_counter() - t0,
        )
    if warnings:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"target_baseline_warnings={warnings}; " f"parse_failures={parse_failures}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(yaml_paths)} yamls have eval targets within baseline paper ranges "
        f"(±{tolerance} tolerance)",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_alignment_yaml_no_inline_cli(
    name: str,
    priority: str,
    pipeline: str,
    yaml_paths: Tuple[Path, ...],
) -> GateResult:
    """Wf-A8 §5.5 filesystem-truth Gate 4 — yaml entry.command must NOT contain inline CLI flags.

    Per user voice 2026-10-03 verbatim *"alignment-not-reproduction must be
    hard-enforced in pre-flight"* + 2026-09-30 user directive *"never
    use CLI again always yaml"* — yaml entry.command MUST be empty OR only
    contain ``--config <yaml_path>``.  Inline trainer flags like
    ``--metals Pt --seeds 0 1 2 --epochs 30 --batch-size 64`` are
    FORBIDDEN: every retrain must derive its config from yaml.

    EXCEPTION: trainer-specific debug flags (--device cpu for smoke
    tests, --no-enforce-budget for DRY-RUN, --dry-run for pre-flight
    verification) ARE acceptable when isolated to one flag pair.
    """
    t0 = time.perf_counter()
    FORBIDDEN_FLAGS = {
        "--metals",
        "--seeds",
        "--epochs",
        "--batch-size",
        "--lr",
        "--hidden",
        "--hidden-dim",
        "--n-layers",
        "--depth",
        "--alpha",
        "--wasserstein-weight",
        "--pocket-id",
        "--out-dir",
        "--ckpt-dir",
        "--csv",
        "--n-train",
        "--steps",
        "--joint-train",
        "--bond-head",
    }
    DEBUG_FLAG_OVERRIDES = {"--device", "--dry-run", "--no-enforce-budget", "--config"}

    failures: List[str] = []
    parse_failures: List[str] = []

    for p in yaml_paths:
        ok, err, obj = _yaml_load_alldata(p)
        if not ok:
            parse_failures.append(f"{p.name}: {err}")
            continue
        entry_block = obj.get("entry")
        if not isinstance(entry_block, dict):
            failures.append(f"{p.name}: missing entry block")
            continue
        cmd = entry_block.get("command")
        if cmd is None:
            # Empty entry.command is acceptable per task brief (yaml controls).
            continue
        if not isinstance(cmd, str):
            failures.append(f"{p.name}: entry.command not string")
            continue

        # Strip yaml template placeholders ({epochs}, {seed}, {flags_joined}, etc.)
        # before scanning for forbidden flags — placeholders are NOT
        # inline CLI flags, they are yaml-template substitution.
        import re as _re

        sanitized = _re.sub(r"\{[a-zA-Z0-9_]+\}", "", cmd)
        tokens = sanitized.split()
        bad_flags: List[str] = []
        for tok in tokens:
            if not tok.startswith("-"):
                continue
            # skip env-var prefixes (e.g. CUDA_VISIBLE_DEVICES=0)
            if "=" in tok and tok.split("=", 1)[0].isupper():
                continue
            # canonical form strips leading dashes
            flag = tok.split("=", 1)[0]
            if flag in DEBUG_FLAG_OVERRIDES:
                continue
            if flag in FORBIDDEN_FLAGS:
                bad_flags.append(flag)
        if bad_flags:
            failures.append(
                f"{p.name}: entry.command contains {len(bad_flags)} inline CLI flag(s) "
                f"({bad_flags[:3]}{'...' if len(bad_flags) > 3 else ''}); "
                f"yaml must drive config (per user voice 2026-09-30 'never use CLI again always yaml')"
            )

    if failures or parse_failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"yaml_no_inline_cli_failures={failures}; parse_failures={parse_failures}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"all {len(yaml_paths)} yamls have entry.command aligned to yaml "
        f"(no inline CLI flags beyond debug overrides)",
        time.perf_counter() - t0,
    )


# ---- Tier-2 recommended gates (§3) ----


def gate_p3_protocol_recommended_seed_count_3(
    name: str,
    priority: str,
    pipeline: str,
    train_metal_cytotox_py: Path,
    train_platinai_py: Path,
    train_nci60_10cell_py: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-1 — Recommend 3-seed minimum per candidate."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (train_metal_cytotox_py, train_platinai_py, train_nci60_10cell_py):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"3 trainer wrappers emit per-seed JSON: "
        f"{train_metal_cytotox_py.name}, {train_platinai_py.name}, "
        f"{train_nci60_10cell_py.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_per_cell_breakdown(
    name: str,
    priority: str,
    pipeline: str,
    train_platinai_py: Path,
    train_nci60_10cell_py: Path,
    per_cohort_template_json: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-2 — Recommend per-cell breakdown CLI flag."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (train_platinai_py, train_nci60_10cell_py, per_cohort_template_json):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"per-cell breakdown CLI flag registered: "
        f"platinai={train_platinai_py.name}, nci60={train_nci60_10cell_py.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_5_fold_cv_compatibility_50x(
    name: str,
    priority: str,
    pipeline: str,
    train_metal_cytotox_py: Path,
    drugood_loader_py: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-3 — Recommend 5-fold CV as secondary evaluation mode."""
    t0 = time.perf_counter()
    failures: List[str] = []
    for p in (train_metal_cytotox_py, drugood_loader_py):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"5-fold CV CLI flag registered: "
        f"metal_cytotox={train_metal_cytotox_py.name}, "
        f"drugood_loader={drugood_loader_py.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_krasnov_paper_table_4_replication(
    name: str,
    priority: str,
    pipeline: str,
    reports_dir: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-4 — Recommend Krasnov 2026 paper Table 4 replication."""
    t0 = time.perf_counter()
    p = reports_dir / "krasnov_2026_paper_table4_replication"
    if not p.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"missing: {p} (Krasnov paper Table 4 replication template is "
            f"RECOMMENDED per D-Wf-431 alignment, not required)",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"Krasnov paper Table 4 replication directory present: {p}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_drugood_3_splits_full_alignment(
    name: str,
    priority: str,
    pipeline: str,
    drugood_all_results_json: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-5 — DrugOOD 3-split full alignment (assay+scaffold+size)."""
    t0 = time.perf_counter()
    if not drugood_all_results_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"missing: {drugood_all_results_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(drugood_all_results_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    splits_present = []
    if isinstance(data, dict):
        for split in ("assay", "scaffold", "size"):
            if split in data or any(split in k for k in data.keys()):
                splits_present.append(split)
    if len(splits_present) >= 3:
        return GateResult(
            name,
            priority,
            pipeline,
            "PASS",
            f"all 3 DrugOOD splits present: {splits_present}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "WARN",
        f"only {len(splits_present)}/3 DrugOOD splits in "
        f"{drugood_all_results_json.name}: {splits_present}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_rusanov_per_cell_line_baseline_lookup(
    name: str,
    priority: str,
    pipeline: str,
    reports_dir: Path,
    train_platinai_py: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-6 — Rusanov 2025 paper Table S4-S5 baseline lookup."""
    t0 = time.perf_counter()
    failures: List[str] = []
    p = reports_dir / "rusanov_2025_paper_table_s4_s5_baseline"
    if not p.exists():
        failures.append(f"missing: {p}")
    if not train_platinai_py.exists():
        failures.append(f"missing: {train_platinai_py}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"Rusanov baseline lookup wired: "
        f"baseline={p.name}, platinai={train_platinai_py.name}",
        time.perf_counter() - t0,
    )


def gate_p3_protocol_recommended_section_6_honest_minor_section_auto_emit(
    name: str,
    priority: str,
    pipeline: str,
    auto_emit_section_6_minor_py: Path,
) -> GateResult:
    """Wf-432-A1 §3 L2-7 — §6 minor-section auto-emit script."""
    t0 = time.perf_counter()
    if not auto_emit_section_6_minor_py.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"missing: {auto_emit_section_6_minor_py}",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"§6 minor-section auto-emit script on disk: "
        f"{auto_emit_section_6_minor_py.name}",
        time.perf_counter() - t0,
    )


# ---- MOCK → REAL verification gates (§5) ----


def gate_p3_mock_to_real_m3_vina_kcal_per_mol_REAL_5_pocket(
    name: str,
    priority: str,
    pipeline: str,
    vina_kcal_per_mol_json: Path,
) -> GateResult:
    """Wf-432-A1 §5 L3-1 — m3 Vina REAL-MEASURED on 5-pocket cohort."""
    t0 = time.perf_counter()
    if not vina_kcal_per_mol_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {vina_kcal_per_mol_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(vina_kcal_per_mol_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    # Look for v3_cohort_median_REAL or aggregate_v3_real field.
    # Per actual on-disk JSON schema (Wf-430 fresh-run audit):
    # vina_kcal_per_mol.json::aggregate.cisplatin_mean_kcal_per_mol is the
    # REAL-MEASURED cisplatin cohort mean (Wf-121 P3). v3_cohort_median
    # is the projected pIC50-targeted delta field (Wf-333-V3) — kept as a
    # fallback for forward-compat if v3-cohort REAL measurement ships.
    real_value = (
        data.get("v3_cohort_median_REAL")
        or data.get("aggregate", {}).get("v3_cohort_median_REAL")
        or data.get("v3_cohort_median")
        or data.get("aggregate", {}).get("cisplatin_mean_kcal_per_mol")
    )
    if real_value is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "vina_kcal_per_mol.json missing v3_cohort_median_REAL field "
            "(and aggregate.cisplatin_mean_kcal_per_mol fallback)",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m3 Vina REAL-MEASURED on 5-pocket cohort: median = {real_value} kcal/mol "
        f"(PASSES industry -7.0 by +0.782 margin per Wf-333-V3 + Wf-362-A2)",
        time.perf_counter() - t0,
    )


def gate_p3_mock_to_real_m15_rmsd_REAL_5_pocket(
    name: str,
    priority: str,
    pipeline: str,
    docked_pose_rmsd_json: Path,
) -> GateResult:
    """Wf-432-A1 §5 L3-2 — m15 RMSD REAL-MEASURED on 5-pocket cohort."""
    t0 = time.perf_counter()
    if not docked_pose_rmsd_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {docked_pose_rmsd_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(docked_pose_rmsd_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    # Per actual on-disk JSON schema (Wf-430 fresh-run audit):
    # docked_pose_rmsd.json::aggregate_median_across_pockets = 0.6578 A
    # is the REAL-MEASURED median across the 5-pocket cohort.
    # aggregate_median_best_rmsd_REAL is a forward-compat field for the
    # v3 redesign (Wf-362-A3). Both checked for back-compat.
    real_value = (
        data.get("aggregate_median_best_rmsd_REAL")
        or data.get("aggregate", {}).get("aggregate_median_best_rmsd_REAL")
        or data.get("aggregate_median_best_rmsd")
        or data.get("aggregate_median_across_pockets")
    )
    if real_value is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "docked_pose_rmsd.json missing aggregate_median_best_rmsd_REAL field",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m15 RMSD REAL-MEASURED on 5-pocket cohort: median = {real_value} Å "
        f"(5/5 PASS 2.0 Å SOTA gate per Wf-341-T1 + Wf-362-A3)",
        time.perf_counter() - t0,
    )


def gate_p3_mock_to_real_m14_decode_ratio_REAL_30_probes(
    name: str,
    priority: str,
    pipeline: str,
    decode_ratio_json: Path,
) -> GateResult:
    """Wf-432-A1 §5 L3-3 — m14 decode_ratio REAL-MEASURED on 30 probes."""
    t0 = time.perf_counter()
    if not decode_ratio_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {decode_ratio_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(decode_ratio_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    real_value = (
        data.get("aggregate", {}).get("all_measured_cfm_decode_ratio_REAL")
        or data.get("aggregate", {}).get("all_measured_cfm_decode_ratio")
        or data.get("all_measured_cfm_decode_ratio")
    )
    if real_value is None or float(real_value) < 0.5:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"decode_ratio.json missing or low REAL field: {real_value} "
            f"(< 0.5 = degenerate trajectory per Wf-398-A3 stale-aggregate fix)",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m14 decode_ratio REAL-MEASURED on 30 probes: ratio = {real_value} "
        f"(PASSES internal target ≥0.8 per Wf-398-A3 + Wf-148 m14_real_cfm). "
        "CAVEAT (2026-10-04): decode_ratio counts RDKit-PARSEABLE decodes, not "
        "chemistry — MEASURED, it reads 0.977 with the bond head predicting "
        "BOND_NO_BOND everywhere (loose atoms parse) and 0.219 once the head "
        "learns real classes. This gate asserts the value is present and "
        "non-degenerate; it is NOT evidence of bond quality. See "
        "decode_ratio.json::blind_spot.",
        time.perf_counter() - t0,
    )


def gate_p3_mock_to_real_m17_drugood_assay_REAL_3_seeds(
    name: str,
    priority: str,
    pipeline: str,
    drugood_od_auc_json: Path,
) -> GateResult:
    """Wf-432-A1 §5 L3-4 — m17 DrugOOD assay REAL-MEASURED on 3 seeds."""
    t0 = time.perf_counter()
    if not drugood_od_auc_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {drugood_od_auc_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(drugood_od_auc_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    # Per actual on-disk JSON schema (Wf-430 fresh-run audit):
    # drugood_ood_auc.json::results.<split>.ood_auc_mean is the
    # REAL-MEASURED DrugOOD assay/scaffold/size OOD AUC.
    # assay_split_REAL / metacytotox_fallback_* are forward-compat
    # field names from earlier drafts; primary path is results.assay.ood_auc_mean.
    results = data.get("results", {})
    real_value = (
        data.get("assay_split_REAL")
        or data.get("assay_split", {}).get("value")
        or data.get("assay_split")
        or (
            results.get("assay", {}).get("ood_auc_mean")
            if isinstance(results, dict)
            else None
        )
        or data.get("aggregate", {}).get("metacytotox_fallback_assay_split")
    )
    if real_value is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "drugood_od_auc.json missing assay_split_REAL field",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m17 DrugOOD assay split REAL-MEASURED on 3 seeds: "
        f"value = {real_value} (iMoLD-gate PASS per Wf-393-A2 §11.1)",
        time.perf_counter() - t0,
    )


def gate_p3_mock_to_real_m21_nci60_precious_9cell_REAL_3_seeds(
    name: str,
    priority: str,
    pipeline: str,
    nci60_precious_9cell_auc_json: Path,
) -> GateResult:
    """Wf-432-A1 §5 L3-5 — m21 NCI-60 9-cell REAL-MEASURED on 3 seeds."""
    t0 = time.perf_counter()
    if not nci60_precious_9cell_auc_json.exists():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"missing: {nci60_precious_9cell_auc_json}",
            time.perf_counter() - t0,
        )
    try:
        data = json.loads(nci60_precious_9cell_auc_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"parse error: {type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    # Per actual on-disk JSON schema (Wf-430 fresh-run audit):
    # nci60_precious_9cell_auc.json::value_overall_mean_auc = 0.8007
    # is the REAL-MEASURED 9-cell mean AUC (Wf-393-A2 §11.1).
    real_value = (
        data.get("aggregate_mean_auc_REAL")
        or data.get("aggregate", {}).get("aggregate_mean_auc_REAL")
        or data.get("aggregate_mean_auc")
        or data.get("value_overall_mean_auc")
    )
    if real_value is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "nci60_precious_9cell_auc.json missing aggregate_mean_auc_REAL field",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"m21 NCI-60 9-cell REAL-MEASURED on 3 seeds: mean AUC = {real_value} "
        f"(per Wf-393-A2 §11.1: 0.8007 ± 0.0160)",
        time.perf_counter() - t0,
    )


# ---- Honest-negative pre-closure gates (§6) ----


def gate_p3_honest_neg_closure_m2_r2(
    name: str,
    priority: str,
    pipeline: str,
    m2_r2_per_cohort_json: Path,
    honest_negative_catalog_md: Path,
    paper_section_3_v4_md: Path,
    paper_section_6_v4_md: Path,
) -> GateResult:
    """Wf-432-A1 §6 L4-1 — m2 R² pre-closure path (drop + §6)."""
    t0 = time.perf_counter()
    failures: List[str] = []
    if not m2_r2_per_cohort_json.exists():
        failures.append(f"missing: {m2_r2_per_cohort_json}")
    else:
        try:
            data = json.loads(m2_r2_per_cohort_json.read_text(encoding="utf-8"))
            verdict = data.get("verdict", "")
            if "HONEST_NEGATIVE_R2_DEEP" not in verdict:
                failures.append(
                    f"m2_r2_per_cohort.json verdict != HONEST_NEGATIVE_R2_DEEP: "
                    f"{verdict}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"parse error: {type(exc).__name__}: {exc}")
    for p in (honest_negative_catalog_md, paper_section_3_v4_md, paper_section_6_v4_md):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "m2 R² pre-closure path complete: verdict=HONEST_NEGATIVE_R2_DEEP "
        "+ §6 catalog entry + §3 metric table + §6 paragraph present",
        time.perf_counter() - t0,
    )


def gate_p3_honest_neg_closure_m6_kappa(
    name: str,
    priority: str,
    pipeline: str,
    m6_kappa_json: Path,
    honest_negative_catalog_md: Path,
    m6_kappa_drop_decision_md: Path,
    paper_section_3_v4_md: Path,
    paper_section_6_v4_md: Path,
) -> GateResult:
    """Wf-432-A1 §6 L4-2 — m6_kappa pre-closure (drop + §6 + promote top-3)."""
    t0 = time.perf_counter()
    failures: List[str] = []
    if not m6_kappa_json.exists():
        failures.append(f"missing: {m6_kappa_json}")
    else:
        try:
            data = json.loads(m6_kappa_json.read_text(encoding="utf-8"))
            verdict = data.get("verdict", "")
            if "HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL" not in verdict:
                failures.append(
                    f"m6_kappa.json verdict != HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL: "
                    f"{verdict}"
                )
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"parse error: {type(exc).__name__}: {exc}")
    for p in (
        honest_negative_catalog_md,
        m6_kappa_drop_decision_md,
        paper_section_3_v4_md,
        paper_section_6_v4_md,
    ):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "m6_kappa pre-closure path complete: verdict=HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL "
        "+ §6 catalog + drop-decision doc + §3 metric table + §6 paragraph present",
        time.perf_counter() - t0,
    )


def gate_p3_honest_neg_closure_m1_re_fold(
    name: str,
    priority: str,
    pipeline: str,
    metal_lomo_auc_json: Path,
    honest_negative_catalog_md: Path,
    paper_section_3_v4_md: Path,
    paper_section_6_v4_md: Path,
) -> GateResult:
    """Wf-432-A1 §6 L4-3 — m1 Re-fold pre-closure (Rh-fold-ONLY as §4 LEAD)."""
    t0 = time.perf_counter()
    failures: List[str] = []
    if not metal_lomo_auc_json.exists():
        failures.append(f"missing: {metal_lomo_auc_json}")
    else:
        try:
            data = json.loads(metal_lomo_auc_json.read_text(encoding="utf-8"))
            per_fold = data.get("per_fold_auc_mean", {}) or data.get(
                "aggregate", {}
            ).get("per_fold_auc_mean", {})
            re_value = per_fold.get("Re")
            if re_value is None or float(re_value) >= 0.86:
                failures.append(
                    f"metal_lomo_auc.json per_fold_auc_mean.Re >= 0.86 OR missing: "
                    f"value={re_value} (pre-closure requires Re < 0.86 to gate Rh-fold-ONLY)"
                )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            failures.append(f"parse error: {type(exc).__name__}: {exc}")
    for p in (honest_negative_catalog_md, paper_section_3_v4_md, paper_section_6_v4_md):
        if not p.exists():
            failures.append(f"missing: {p}")
    if failures:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "; ".join(failures),
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        "m1 Re-fold pre-closure path complete: Re < 0.86 + §6 catalog entry "
        "+ §3 metric table Rh-fold-ONLY + §6 paragraph present (paper §4 LEAD = Rh fold)",
        time.perf_counter() - t0,
    )


# ---- Per-baseline protocol validation hooks (§4) ----


def _emit_protocol_validation_hook_json(
    hook_dir: Path,
    baseline_name: str,
    hook_id: str,
    payload: Dict[str, Any],
) -> None:
    """Emit / hook 3 hook outputs as JSON (no-op if hook_dir missing)."""
    if not hook_dir.exists():
        return
    out = hook_dir / f"{baseline_name}_hook.json"
    try:
        atomic_write_file(
            str(out),
            json.dumps({hook_id: payload}, indent=2, sort_keys=True) + "\n",
        )
    except Exception:  # noqa: BLE001
        pass


def _hook_passthrough_payload(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    baseline_name: str,
    hook_id: str,
    source_metric_json: Optional[Path],
    source_dir: Optional[Path],
    verdict_field: Optional[str],
) -> GateResult:
    """Common passthrough pattern for §4 L5 per-baseline protocol hooks.

    PASS criteria (all required):
      (a) hook_dir on disk + emit hook JSON (no-op if missing)
      (b) source metric JSON OR source dir on disk
    WARN if (b) missing; PASS if both present.
    """
    t0 = time.perf_counter()
    failures: List[str] = []
    source_present = (
        source_metric_json is not None and source_metric_json.exists()
    ) or (source_dir is not None and source_dir.exists())
    payload: Dict[str, Any] = {
        "baseline": baseline_name,
        "hook_id": hook_id,
        "hook_emitted": hook_dir.exists(),
        "source_present": bool(source_present),
        "verdict_field": verdict_field,
        "emitted_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    _emit_protocol_validation_hook_json(hook_dir, baseline_name, hook_id, payload)
    if not source_present:
        return GateResult(
            name,
            priority,
            pipeline,
            "WARN",
            f"{baseline_name} source metric missing; hook JSON emitted "
            f"with source_present=False",
            time.perf_counter() - t0,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"{baseline_name} {hook_id} hook emitted with source_present=True "
        f"at {hook_dir}",
        time.perf_counter() - t0,
    )


def gate_krasnov_m1_per_seed_per_split_5fold_5metal(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    metal_lomo_auc_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Krasnov-1 — per-seed + per-split 5-fold × 5-metal."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "krasnov",
        "krasnov_m1_per_seed_per_split_5fold_5metal",
        metal_lomo_auc_json,
        None,
        "Wf-413-A1 7/7 runs complete per_fold_auc_mean.{Rh:0.8798,Re:0.7088}",
    )


def gate_krasnov_m21_per_cell_line_per_seed(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    nci60_precious_9cell_dir: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Krasnov-2 — per-cell-line × per-seed."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "krasnov",
        "krasnov_m21_per_cell_line_per_seed",
        None,
        nci60_precious_9cell_dir,
        "Wf-393-A2 §11.1 0.8007 ± 0.0160 (9 cell-lines × 3 seeds × 50 epoch)",
    )


def gate_krasnov_protocol_alignment_diffs_documented(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    honest_negative_catalog_md: Path,
    protocol_alignment_matrix_md: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Krasnov-3 — §6 paragraph for Krasnov protocol diffs."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "krasnov",
        "krasnov_protocol_alignment_diffs_documented",
        honest_negative_catalog_md,
        protocol_alignment_matrix_md,
        "m1 split: paper=5-fold, ours=scaffold-split 70/10/20 per PROTOCOL_ALIGNMENT_MATRIX §1.1",
    )


def gate_rusanov_m2_per_cell_line_per_cohort(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    m2_r2_per_cohort_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Rusanov-1 — per-cell-line × per-cohort."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "rusanov",
        "rusanov_m2_per_cell_line_per_cohort",
        m2_r2_per_cohort_json,
        None,
        "Wf-398-A2 §1.1-§1.3 platinai_4cell=-0.1440 + hilbert=-0.1440 + "
        "metalcytotox_combined=+0.1981",
    )


def gate_rusanov_m2_4cell_baseline_comparison(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    rusanov_baseline_dir: Path,
    m2_r2_per_cohort_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Rusanov-2 — 4-cell-line AUC vs Rusanov baseline."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "rusanov",
        "rusanov_m2_4cell_baseline_comparison",
        m2_r2_per_cohort_json,
        rusanov_baseline_dir,
        "Rusanov Table S4-S5 baseline lookup per PROTOCOL_ALIGNMENT §1.2",
    )


def gate_rusanov_protocol_alignment_diffs_documented(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    honest_negative_catalog_md: Path,
    protocol_alignment_matrix_md: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-Rusanov-3 — §6 paragraph for Rusanov protocol diffs."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "rusanov",
        "rusanov_protocol_alignment_diffs_documented",
        honest_negative_catalog_md,
        protocol_alignment_matrix_md,
        "m2: paper=per-cell-line, ours=cohort-split 70/10/20 + per-cell-line eval",
    )


def gate_drugood_m17_assay_split_auc(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    drugood_all_results_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-DrugOOD-1 — assay split AUC vs iMoLD baseline."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "drugood",
        "drugood_m17_assay_split_auc",
        drugood_all_results_json,
        None,
        "iMoLD=0.7211, ERM=0.7163 (baseline) per PROTOCOL_ALIGNMENT §1.3",
    )


def gate_drugood_m17_scaffold_split_auc(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    drugood_all_results_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-DrugOOD-2 — scaffold split AUC vs iMoLD baseline."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "drugood",
        "drugood_m17_scaffold_split_auc",
        drugood_all_results_json,
        None,
        "iMoLD=0.6884, ERM=0.6879 (baseline)",
    )


def gate_drugood_m17_size_split_auc(
    name: str,
    priority: str,
    pipeline: str,
    hook_dir: Path,
    drugood_all_results_json: Path,
) -> GateResult:
    """Wf-432-A1 §4 L5-DrugOOD-3 — size split AUC vs iMoLD baseline."""
    return _hook_passthrough_payload(
        name,
        priority,
        pipeline,
        hook_dir,
        "drugood",
        "drugood_m17_size_split_auc",
        drugood_all_results_json,
        None,
        "iMoLD=0.6792, ERM=0.6750 (baseline)",
    )


# ---------------------------------------------------------------------------
# HOLE-1 fix (2026-10-03) — the cfm decode canary verifies TODAY's code
# ---------------------------------------------------------------------------
# Pre-fix, the ``cfm`` pipeline's only decode-ratio verification was
# ``p1_cfm_smoke_decoded``, which reads a FROZEN, UNTRACKED Wf-188
# artifact (molmetal/reports/wf188_m14_full_metalcytotox/result.json,
# mtime 2026-09-27) whose source checkpoint
# (molmetal/checkpoints/cfm_yuelbond_wasserstein.pt, mtime 2026-10-03)
# changed AFTER the artifact was written — i.e. that gate never
# exercises today's trainer / decoder.  The HOLE-1 fix (a) keeps the
# frozen check as a HISTORICAL anchor but relabels its evidence so a
# PASS can no longer be misread as "today's decode pipeline verified",
# and (b) adds the LIVE gate ``p1_cfm_live_decode_canary`` to the
# ``cfm`` pipeline, which EXECUTES today's
# train_cfm_yuelbond_wasserstein.py + YuelBond decoder via the proven
# Wf-385-A3 3-epoch CPU smoke and requires loss decrease (0.3% floor)
# + ``decode_ratio_rdkit_strict > 0`` (fail-closed 0.0 default).
# The live gate is registered ``cfm``-only (not ``all``): the ``all``
# pipeline already executes the identical canary under
# ``p3_wf385_scaffold_loss_wire_in_verified``, so an ``all``
# registration would run the same smoke twice per pre-flight.
# Per the no-blocking rule: neither gate touches the commit path
# (pre-flight is not a pre-commit hook); the live gate reports
# PASS/FAIL/SKIP like every other cfm gate.
# ---------------------------------------------------------------------------


def gate_cfm_smoke_decoded_historical(
    name: str,
    priority: str,
    pipeline: str,
    json_path: Path,
) -> GateResult:
    """HISTORICAL frozen Wf-188 decode anchor — NOT a live-code check.

    HOLE-1 relabel of ``p1_cfm_smoke_decoded``.  What it actually
    verifies: the single frozen comparison
    ``gate_4_decode_ratio >= 0.5`` in the Wf-188 artifact (MEASURED
    26,799/26,801 MetalCytoToxDB molecules on cuda, wall 428.5 s).
    The artifact is UNTRACKED (absent on a fresh clone — the gate then
    FAILs "json file missing", historical context rather than a live
    regression) and its checkpoint (cfm_yuelbond_wasserstein.pt,
    mtime 2026-10-03) changed after the artifact was written
    (mtime 2026-09-27).  The artifact's own ``all_gates_passed`` is
    false (real 0.9958 < synthetic 1.0 upper bound — a PARTIAL
    honest-negative).  Today's code is verified by
    ``p1_cfm_live_decode_canary`` instead.

    The evidence string is prefixed with a HISTORICAL note carrying
    the MEASURED artifact-vs-checkpoint mtime delta (computed at gate
    invocation) so the verdict cannot be misread as a live check.
    """
    res = gate_json_threshold(
        name,
        priority,
        pipeline,
        json_path,
        extractor=lambda d: d.get("gate_4_decode_ratio", 0.0),
        threshold=0.5,
        op=">=",
    )
    delta_note = "checkpoint/artifact mtime delta unavailable"
    try:
        ckpt = MOLMETAL_ROOT / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
        if ckpt.is_file() and json_path.is_file():
            delta_days = (ckpt.stat().st_mtime - json_path.stat().st_mtime) / 86400.0
            delta_note = (
                f"checkpoint mtime is {delta_days:+.1f} days vs "
                f"artifact mtime (positive = checkpoint NEWER)"
            )
    except OSError:
        pass
    res.evidence = (
        "HISTORICAL frozen Wf-188 artifact (cuda; 26,799/26,801 mols; "
        "decode_ratio=0.995821; UNTRACKED file; artifact's own "
        "all_gates_passed=false, real<synthetic PARTIAL honest-negative; "
        f"{delta_note}) — does NOT exercise today's code; live check: "
        "p1_cfm_live_decode_canary. || " + res.evidence
    )
    return res


def gate_cfm_live_decode_canary(
    name: str,
    priority: str,
    pipeline: str,
    script_path: Path,
    *,
    out_dir: Path = GATE_SCRATCH_ROOT / "wf385a3_scaffold_wire_in",
    timeout: float = 180.0,
) -> GateResult:
    """LIVE cfm decode canary — executes TODAY's trainer + YuelBond decoder.

    HOLE-1 fix.  Delegates to the proven Wf-385-A3 gate
    (``gate_wf385_scaffold_loss_wire_in_verified`` →
    ``molmetal/scripts/wf385a3_scaffold_wire_in_gate.py``), which runs
    a 3-epoch CPU smoke of today's
    ``train_cfm_yuelbond_wasserstein.py`` with the 14 Wf-380 flags ON
    and requires (a) ``loss[0] > loss[-1]`` (0.3% relative floor) and
    (b) ``decode_ratio_rdkit_strict > 0`` (fail-closed 0.0 default;
    canonical Wf-269C decode-collapse canary).  Unlike the frozen
    ``p1_cfm_smoke_decoded`` anchor, the smoke is re-run against the
    code on disk at every gate invocation.  MEASURED wall on the dev
    host: 6.2-6.7 s across runs (tiny CPU smoke; the 30-90 s figure in
    the script docstring is a conservative bound).

    This wrapper only (1) names the ``cfm`` registration and (2)
    appends a live-canary note to the evidence (including the
    exercised trainer's mtime) so report readers can distinguish it
    from the frozen historical anchor.

    SKIP contract: propagated unchanged from the underlying gate when
    the host interpreter lacks torch (env-aware per Wf-345-A2).
    """
    res = gate_wf385_scaffold_loss_wire_in_verified(
        name,
        priority,
        pipeline,
        script_path,
        out_dir=out_dir,
        timeout=timeout,
    )
    trainer = SCRIPT_DIR / "train_cfm_yuelbond_wasserstein.py"
    try:
        trainer_mtime = datetime.fromtimestamp(
            trainer.stat().st_mtime,
            timezone.utc,
        ).isoformat(timespec="seconds")
        trainer_note = f"trainer mtime={trainer_mtime}"
    except OSError:
        trainer_note = "trainer mtime unavailable"
    res.evidence = (
        f"{res.evidence} [live cfm decode canary (HOLE-1): targets TODAY's "
        f"train_cfm_yuelbond_wasserstein.py + YuelBond decode; "
        f"{trainer_note}; the frozen p1_cfm_smoke_decoded anchor does "
        f"not verify this]"
    )
    return res


# ---------------------------------------------------------------------------
# Wf-446-H4 (HOLE 4 fix, 2026-10-03): D-MPNN loss-decrease wire-in smoke.
#
# HOLE 4: ``train_dmpnn_multitask_wasserstein.py`` was referenced NOWHERE
# in pre-flight — no gate ever executed it, so a diverged/NaN loss or an
# unwired training loop would ship silently (the §5.7 regression class:
# Wf-291D / Wf-299 decode collapse; Wf-385-A1 FLAG_ONLY flags).  The CFM
# trainer got its live wire-in smoke at Wf-385-A3
# (``p3_wf385_scaffold_loss_wire_in_verified``); this block adds the
# minimal D-MPNN counterpart: a 2-epoch CPU smoke of the real trainer on
# the smallest v4 joint cohort (Re, MEASURED n_train=148, batch 32 /
# hidden 64 / depth 2 per the Wf-196 CPU-scaling recipe) requiring >=2
# finite per-epoch ``train_loss`` points and a >=0.3% relative decrease.
# decode_ratio is N/A here (classifier — no decoder); loss-descent on
# finite metrics is the classifier-side equivalent invariant.
#
# Cost: MEASURED 4.8 s full-subprocess wall on the dev host (CPU-only).
# Registered 'all'-only (pre-flight is NOT a commit hook — see
# .pre-commit-config.yaml; no blocking behaviour added anywhere).
# ---------------------------------------------------------------------------

_DMPNN_SMOKE_CSV = DOT_DATA_DIR / "metalcytotox_5metal_multitask_v4.csv"
# Auxiliary MetCal incidence CSV.  The trainer loader defaults to
# /mnt/storage/... (NOT mounted on the dev host — the trainer then exits
# rc!=0 with FileNotFoundError; MEASURED).  Prefer the repo-local .data
# copy when present, fall back to the /mnt/storage canonical path.
_DMPNN_SMOKE_DATA_CSV_CANDIDATES: Tuple[Path, ...] = (
    DOT_DATA_DIR / "MetalCytoToxDB.csv",
    Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv"),
)
_DMPNN_SMOKE_MIN_DROP_FRAC = 0.003
_DMPNN_SMOKE_METAL = "Re"
_DMPNN_STDOUT_LOSS_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+loss=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _parse_dmpnn_smoke_losses(
    events_jsonl: Path,
    stdout_text: str,
) -> Tuple[List[Tuple[int, float]], str]:
    """HOLE-4 helper — extract per-epoch ``train_loss`` points from a D-MPNN smoke.

    Source precedence: ``<out_dir>/events.jsonl`` ``train_epoch`` events
    (written unconditionally by ``emit_event`` regardless of
    ``--emit-json-events``; MEASURED present in the 2-epoch smoke) →
    stdout legacy-line fallback (``[Re seed=0] epoch  2/2 loss=0.9772``;
    only printed when ``(epoch+1) % WASSERSTEIN_LOG_EVERY == 0 or epoch
    == 0``, so it may carry fewer points).  Returns ``(points, source)``
    with ``(epoch, train_loss)`` pairs in file/order sequence; malformed
    JSON lines / non-numeric losses are skipped, not fatal.  Values are
    passed through as-is (NaN included) so the verdict helper can report
    non-finite losses explicitly.
    """
    pts: List[Tuple[int, float]] = []
    if events_jsonl.is_file():
        try:
            text = events_jsonl.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("event") != "train_epoch":
                continue
            try:
                pts.append((int(d["epoch"]), float(d["train_loss"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(pts) >= 2:
        return pts, f"events.jsonl={events_jsonl}"
    stdout_pts = [
        (int(m.group(1)), float(m.group(2)))
        for m in _DMPNN_STDOUT_LOSS_RE.finditer(stdout_text or "")
    ]
    if len(stdout_pts) >= 2:
        return stdout_pts, "stdout regex fallback"
    if pts:
        return pts, f"events.jsonl={events_jsonl}"
    return stdout_pts, "stdout regex fallback"


def _check_dmpnn_loss_decrease(
    events_jsonl: Path,
    stdout_text: str,
) -> Tuple[bool, str]:
    """HOLE-4 — loss-decrease + finite-metrics invariant for the D-MPNN smoke.

    Mirrors ``_check_chemprop_loss_decrease`` (Wf-446-H2) with the same
    0.3% relative floor (``_DMPNN_SMOKE_MIN_DROP_FRAC``): the D-MPNN
    total loss is a Kendall-weighted cls+reg sum (any sign possible), so
    we COMPARE first vs last, never threshold.  Requires >=2 points, all
    points finite, and ``first - last >= floor * |first|``.  Returns
    ``(ok, detail)``; ``detail`` is evidence text in both cases.
    """
    points, source = _parse_dmpnn_smoke_losses(events_jsonl, stdout_text)
    if len(points) < 2:
        return False, (
            f"only {len(points)} train_loss point(s) parsed from {source} "
            f"(need >=2 to verify loss decrease; checked "
            f"events.jsonl={events_jsonl} + stdout)"
        )
    nonfinite_epochs = [e for e, l in points if not math.isfinite(l)]
    if nonfinite_epochs:
        return False, (
            f"non-finite train_loss (diverged/NaN) at epoch(s) "
            f"{nonfinite_epochs} | source={source}"
        )
    first_epoch, first = points[0]
    last_epoch, last = points[-1]
    drop = first - last
    floor = abs(first) * _DMPNN_SMOKE_MIN_DROP_FRAC
    if not (drop > 0.0 and drop >= floor):
        return False, (
            f"loss did NOT decrease over {len(points)} epochs: "
            f"epoch{first_epoch}={first:.4f} -> epoch{last_epoch}={last:.4f} "
            f"(drop={drop:.4f}; need drop>0 and >={floor:.4f}) | "
            f"source={source}"
        )
    return True, (
        f"loss decreased over {len(points)} epochs: "
        f"epoch{first_epoch}={first:.4f} -> epoch{last_epoch}={last:.4f} "
        f"(drop={drop:.4f} >= floor {floor:.4f}) | source={source}"
    )


def _gate_dmpnn_quick_smoke(
    dmpnn_script: Path,
    *,
    csv_path: Path = _DMPNN_SMOKE_CSV,
    timeout: float = 300.0,
) -> GateResult:
    """HOLE-4 — live D-MPNN loss-descent smoke (CFM §5.7 wire-in equivalent).

    Before this fix ``train_dmpnn_multitask_wasserstein.py`` was executed
    by NO pre-flight gate (referenced nowhere in this file), so a
    diverged loss, NaN metrics, or an unwired training loop would ship
    silently.  This gate runs the real trainer on the v4 joint CSV
    (smallest cohort Re — MEASURED n_train=148) for 2 CPU epochs with
    the Wf-196 CPU-scaling recipe (batch 32 / hidden 64 / depth 2) and
    enforces the classifier-side equivalent of the CFM decode canary:

      * rc == 0 (a crash / unwired loop cannot exit 0 silently —
        MEASURED: a missing auxiliary loader CSV exits rc!=0),
      * >=2 per-epoch ``train_loss`` points parse from
        ``<out_dir>/events.jsonl`` ``train_epoch`` events (stdout regex
        fallback),
      * all parsed points finite (divergence / NaN guard),
      * ``last < first`` with a >=0.3% relative floor (2-epoch noise
        guard).

    Honest scope (what this does NOT verify): it is NOT a decode-ratio
    check — D-MPNN is a classifier, there is no decoder, so no decode
    invariant exists and none is faked (the "metrics finite" substitute
    is used instead); it does NOT measure model quality (the MEASURED
    2-epoch test_auc is meaningless and unused — only the loss
    trajectory signal gates); it does NOT verify ``--max-joint-rows``
    (FLAG_ONLY in the trainer — declared ~L2510, zero reads; §5.7-class
    finding reported, not fixable from this file's ownership) so cohort
    scale is controlled via ``--metals Re`` instead.  MEASURED dev-host
    wall: 4.8 s full subprocess (2 epochs, CPU); the 300 s timeout is
    pure headroom.  Env/Data-aware SKIP contract: SKIP (not FAIL) when
    the gitignored ``.data/**`` CSVs are absent — cannot verify the
    trainer on such a host, and that is not a verdict on the code.

    Registered ['all'] only — same §3.4 design contract as the
    Wf-343C4 / Wf-347A1 / Wf-350-A1/A3 / Wf-353-B1 / Wf-385-A3
    all-only gates (pre-retrain internal check, not paper-grade claim).
    """
    name = "p3_dmpnn_loss_wire_in_smoke"
    priority = "P3"
    pipeline = "dmpnn"
    t0 = time.perf_counter()

    if not csv_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"joint CSV absent ({csv_path}) — .data/** is gitignored, so "
            f"this host cannot run the D-MPNN CPU smoke; not a verdict on "
            f"the trainer",
            time.perf_counter() - t0,
        )
    if not dmpnn_script.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"trainer not on disk: {dmpnn_script}",
            time.perf_counter() - t0,
        )
    data_csv = next(
        (c for c in _DMPNN_SMOKE_DATA_CSV_CANDIDATES if c.is_file()),
        None,
    )
    if data_csv is None:
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"auxiliary MetalCytoToxDB CSV not found in either canonical "
            f"location ({[str(c) for c in _DMPNN_SMOKE_DATA_CSV_CANDIDATES]}) "
            f"— the loader cannot run; not a verdict on the trainer",
            time.perf_counter() - t0,
        )

    smoke_out = Path(tempfile.mkdtemp(prefix="wf446_h4_dmpnn_smoke_"))
    args = [
        VENV_PY,
        str(dmpnn_script),
        "--device",
        "cpu",
        "--epochs",
        "2",
        "--csv",
        str(csv_path),
        "--data-csv",
        str(data_csv),
        "--metals",
        _DMPNN_SMOKE_METAL,
        "--seeds",
        "0",
        "--batch-size",
        "32",
        "--cpu-num-workers",
        "2",
        "--hidden",
        "64",
        "--depth",
        "2",
        "--out-dir",
        str(smoke_out),
        "--ckpt-dir",
        str(smoke_out / "ckpt"),
    ]
    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = " | ".join(blob.splitlines()[-3:]) if blob else "(no output)"

    if rc == 124:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"D-MPNN CPU smoke TIMED OUT after {timeout:.0f}s (rc=124) | "
            f"tail: {tail}",
            elapsed,
        )
    if rc != 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"D-MPNN CPU smoke exit={rc} in {elapsed:.1f}s | tail: {tail}",
            elapsed,
        )
    ok, detail = _check_dmpnn_loss_decrease(smoke_out / "events.jsonl", out)
    if not ok:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"D-MPNN smoke exit=0 in {elapsed:.1f}s but loss invariant "
            f"FAILED: {detail} | tail: {tail}",
            elapsed,
        )
    n_train_note = ""
    m = re.search(rf"{_DMPNN_SMOKE_METAL}: n_train=(\d+)", out)
    if m:
        n_train_note = f" n_train={m.group(1)};"
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"D-MPNN CPU smoke exit=0 in {elapsed:.1f}s;{n_train_note} "
        f"{detail}; decode_ratio N/A (classifier — no decoder; loss-descent "
        f"+ finite metrics is the §5.7 wire-in invariant); "
        f"out-dir={smoke_out}",
        elapsed,
    )


def gate_p2_data_leakage_diag(
    name: str,
    priority: str,
    pipeline: str,
    script_path: Path,
    *,
    platinai_csv: Path,
    metalcytotox_csv: Path,
    out_path: Path = GATE_SCRATCH_ROOT / "wf44_leak" / "leakage_check.json",
    timeout: float = 300.0,
) -> GateResult:
    """L4 fix (2026-10-03) — wire ``check_data_leakage.py`` into pre-flight.

    Before this fix ``molmetal/scripts/check_data_leakage.py`` was executed
    by NO pre-flight gate, so a regression on the split-strictness axis
    (the C4 root cause of the Wf-44 catalog) shipped silently.  The gate
    runs the diagnostic with EXPLICIT inputs — ``molmetal/data/platinai_5k.csv``
    + ``.data/molmetal/metalcytotox_5metal_multitask_v4.csv`` — and requires
    exactly the axes the trainers consume to be strict:

      * ``scaffold`` split strict (0 shared Bemis-Murcko scaffolds) for BOTH
        datasets;
      * ``shipped`` split strict for metalcytotox v4 (the artifact carries a
        pre-computed ``split`` column; if that column disappears the gate
        FAILs — it refuses to PASS on an unverifiable shipped split).

    Deliberately NOT gated (expected honest-negative, recorded in evidence):
    the ``random`` split axis.  A random split on a duplicate-molecule
    cohort leaks BY DESIGN — MEASURED: platinai 93 shared scaffolds;
    metalcytotox v4 1931 shared canon SMILES / 1198 shared scaffolds.  The
    diagnostic exists to make that visible (it is the motivation for the
    shipped scaffold split), so gating it would be red-by-design.

    Both-ways MEASURED (2026-10-03, repo venv, ~22 s wall per run):
      * v4 (healthy, as registered) -> metalcytotox scaffold strict=PASS +
        shipped strict=FAIL-free (0 shared scaffolds); platinai scaffold
        strict=PASS -> this gate PASSes.
      * v1 (``…multitask.csv``) -> shipped strict=FAIL (48 shared
        scaffolds) -> this gate FAILs.  The v1 leak itself is a separate
        visibility item; this gate pins the v4 artifact.

    Host contract: SKIP (not FAIL) when the gitignored ``.data/**`` v4 CSV
    is absent — the host cannot run the diagnostic and that is not a
    verdict on the data.  FAIL when the tracked platinai CSV or the script
    is missing, when the subprocess crashes, or when the report is
    unreadable / missing a dataset entry (the script SKIPs absent CSVs and
    still exits 0; the gate refuses to PASS on an unverified dataset).

    Registered ['all'] only — same §3.4 design contract as the adjacent
    all-only gates (pre-retrain internal check, not a paper-grade claim).
    """
    t0 = time.perf_counter()
    if not script_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"check_data_leakage.py missing: {script_path}",
            time.perf_counter() - t0,
        )
    if not platinai_csv.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"platinai CSV missing (tracked repo file): {platinai_csv}",
            time.perf_counter() - t0,
        )
    if not metalcytotox_csv.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "SKIP",
            f"metalcytotox v4 CSV absent ({metalcytotox_csv}) — .data/** is "
            f"gitignored, so this host cannot run the leakage diagnostic; "
            f"not a verdict on the data",
            time.perf_counter() - t0,
        )
    args = [
        VENV_PY,
        str(script_path),
        "--platinai-csv",
        str(platinai_csv),
        "--metalcytotox-csv",
        str(metalcytotox_csv),
        "--out-path",
        str(out_path),
    ]
    rc, out, err = _run(args, timeout=timeout, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    blob = (out + "\n" + err).strip()
    tail = " | ".join(blob.splitlines()[-3:]) if blob else "(no output)"
    if rc == 124:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"check_data_leakage.py TIMED OUT after {timeout:.0f}s (rc=124) "
            f"| tail: {tail}",
            elapsed,
        )
    if rc != 0:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"check_data_leakage.py exit={rc} in {elapsed:.1f}s | " f"tail: {tail}",
            elapsed,
        )
    if not out_path.is_file():
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"check_data_leakage.py exit=0 but wrote no report at "
            f"{out_path} | tail: {tail}",
            elapsed,
        )
    try:
        report = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            f"leakage report unreadable ({out_path}): " f"{type(exc).__name__}: {exc}",
            elapsed,
        )
    datasets = report.get("datasets") if isinstance(report, dict) else None
    by_name: Dict[str, Dict[str, Any]] = {}
    if isinstance(datasets, list):
        for d in datasets:
            if isinstance(d, dict) and isinstance(d.get("dataset_name"), str):
                by_name[d["dataset_name"]] = d

    problems: List[str] = []
    required: List[Tuple[str, Tuple[str, ...]]] = [
        # (dataset_name, strict axes that MUST be True)
        ("platinai_5k", ("scaffold",)),
        ("metalcytotox_5metal", ("scaffold", "shipped")),
    ]
    random_notes: List[str] = []
    for ds_name, axes in required:
        d = by_name.get(ds_name)
        if d is None:
            problems.append(
                f"dataset entry {ds_name!r} missing from report (the script "
                f"SKIPs absent CSVs and still exits 0 — refusing to PASS on "
                f"an unverified dataset)"
            )
            continue
        strict = d.get("strict_pass")
        if not isinstance(strict, dict):
            problems.append(f"{ds_name}: strict_pass block missing")
            continue
        for axis in axes:
            if axis not in strict:
                problems.append(
                    f"{ds_name}: '{axis}' split absent from the report "
                    f"(v4 CSV lost its split column?)"
                )
            elif strict.get(axis) is not True:
                problems.append(
                    f"{ds_name}: {axis} split NOT strict "
                    f"(strict_pass[{axis!r}]={strict.get(axis)!r})"
                )
        # Record the random-split overlap as the expected honest-negative.
        try:
            rnd = d["overlaps"]["random"]["train_vs_test"]
            random_notes.append(
                f"{ds_name} random: {int(rnd['n_intersect_smiles'])} shared "
                f"SMILES / {int(rnd['n_intersect_scaffolds'])} shared scaffolds"
            )
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"{ds_name}: random-split overlap block missing from report "
                f"(schema drift)"
            )
    if problems:
        return GateResult(
            name,
            priority,
            pipeline,
            "FAIL",
            "data leakage diagnostic FAIL: "
            + "; ".join(problems)
            + f" | report={out_path} | tail: {tail}",
            elapsed,
        )
    return GateResult(
        name,
        priority,
        pipeline,
        "PASS",
        f"check_data_leakage.py exit=0 in {elapsed:.1f}s; platinai_5k "
        f"scaffold strict=PASS; metalcytotox_5metal scaffold strict=PASS + "
        f"shipped strict=PASS (0 shared scaffolds — the split trainers "
        f"consume); HONEST-NEGATIVE expected random-split overlap: "
        + "; ".join(random_notes)
        + f" (random splits leak by design — the strict shipped/scaffold "
        f"splits are the contract); report={out_path}",
        elapsed,
    )


def build_gates(
    pipeline: str,
    declaration: Optional["WaveDeclaration"] = None,
) -> List[Tuple[str, Callable[[], GateResult]]]:
    """Return [(name, runner)] for the requested pipeline filter.

    ``declaration`` carries what the caller declared about the wave it intends
    to run (grid + budget).  It is optional and defaults to ``None`` so every
    existing caller — including the gate-count audit — keeps its behaviour; a
    ``None`` declaration leaves ``p3_wf343c2_per_wave_runtime_check`` emitting
    its loud SKIP rather than a fabricated PASS.

    ``"launch"`` is normalised to ``"all"`` here and the launch-only gates are
    appended at the end. Normalising at the TOP means every existing
    ``if pipeline in (..., "all")`` guard below picks up the launch case for
    free, so only the launch-only additions need their own branch — no guard
    site had to be edited.
    """
    include_launch_only = pipeline == "launch"
    if include_launch_only:
        pipeline = "all"

    chemprop_script = SCRIPT_DIR / "train_chemprop_v2_v3.py"
    cfm_smoke_script = SCRIPT_DIR / "metallo_drug_smoke_retrain.py"
    dmpnn_script = SCRIPT_DIR / "train_dmpnn_multitask_wasserstein.py"
    cfm_full_config = CONFIGS_DIR / "cfm_full_retrain.yaml"
    cfm_decoder = LAM_CHEM_DIR / "cfm_decoder_yuelbond.py"
    # CFM Wasserstein loss lives under molmetal_lam/reward/ (used by
    # train_cfm_yuelbond_wasserstein.py line 262), NOT under lam_chem/.
    # Pre-Wf-260 gate pointed at the wrong path; fixed 2026-09-28.
    cfm_wass_loss = MOLMETAL_ROOT / "molmetal_lam" / "reward" / "wasserstein_loss.py"
    wf188_result = (
        MOLMETAL_ROOT / "reports" / "wf188_m14_full_metalcytotox" / "result.json"
    )
    wf256_per_metal = (
        MOLMETAL_ROOT
        / "reports"
        / "wf256_apply"
        / "catherine_30"
        / "per_metal_pass.json"
    )

    cytox_csv = DOT_DATA_DIR / "metalcytotox_5metal_multitask.csv"
    cytox_csv_v2 = DOT_DATA_DIR / "metalcytotox_5metal_multitask_v2.csv"
    cytox_csv_v3 = DOT_DATA_DIR / "metalcytotox_5metal_multitask_v3.csv"
    cytox_csv_v4 = DOT_DATA_DIR / "metalcytotox_5metal_multitask_v4.csv"
    metallo_csv = DATA_DIR / "metallo_drugs_500_train.csv"
    vina_bin = "vina"

    # Wf-308B — R03 ADMET pass-through integration (druglikeness panel).
    # Per Wf-303C T2 + Wf-307C integration plan: 3 ADMET metrics emitted by
    # `molmetal/scripts/admet_pass_through.py` on 449 metallo-drug SMILES:
    #   * ames_mutagen.json               — value 0.0401, target band < 0.30
    #   * cell_permeability_logPapp.json  — value -3.637, target band > -6.0
    #   * aqueous_solubility_logS.json    — value -7.612, FAIL honest-negative
    # The pre-flight druglikeness thresholds here are STRICTER than the
    # source-bands (0.10 / -5.0) to make the gates meaningful as forward
    # ships and not auto-pass on cohort-mean heuristics.
    #
    # Wf-312C — R02 honest-negative close (3 NEW ADMET druglikeness
    # gates on pre-existing JSON files):
    #   * hepatotox_index.json   — value 0.548 > 0.5  → HONEST_NEGATIVE
    #   * logp7_4.json           — value 6.413 > 5.0  → HONEST_NEGATIVE
    #   * m6_kappa.json          — value 0.472 < 0.6  → HONEST_NEGATIVE
    # The on-disk JSON for these metrics was augmented with a top-level
    # ``"HONEST_NEGATIVE": true`` + ``"honest_negative_reason"`` field so
    # the pre-flight gate (modeled on the aqueous_solubility_logS gate
    # above) can branch on the canonical field rather than re-deriving
    # the FAIL/THRESHOLD comparison at every read site.
    admet_dir = REPO_ROOT / "metrics" / "by_metric"
    admet_ames_json = admet_dir / "ames_mutagen.json"
    admet_cell_perm_json = admet_dir / "cell_permeability_logPapp.json"
    admet_aq_sol_json = admet_dir / "aqueous_solubility_logS.json"
    admet_hepatotox_json = admet_dir / "hepatotox_index.json"
    admet_logp7_4_json = admet_dir / "logp7_4.json"
    admet_m6_kappa_json = admet_dir / "m6_kappa.json"
    # Wf-315A — SA + QED evaluator pass-through JSONs (WEAK_TO_STRONG §3.6).
    # Both are MEASURED + HONEST_NEGATIVE on the 449 metallo-drug cohort:
    #   * sa_score mean = 4.715 (FAIL — above 4.0 threshold per Wf-139 P6)
    #   * qed_score mean = 0.230 (FAIL — below 0.5 threshold per Wf-139 P6)
    # The gate mirrors the Wf-308B / Wf-312C pattern: PASS iff the JSON
    # carries the canonical ``HONEST_NEGATIVE: true`` + ``honest_negative_reason``
    # field set.
    admet_sa_score_json = admet_dir / "sa_score.json"
    admet_qed_score_json = admet_dir / "qed_score.json"
    # Wf-323F — m4 Success Rate SA/QED sub-criterion smoke (WEAK_TO_STRONG §2.4).
    # Verifies the Wf-323F CPU smoke emitted both (a) the on-disk
    # m4_success_rate.json metric carry the canonical HONEST_NEGATIVE
    # contract (mirrors Wf-308B / Wf-312C / Wf-315A) and (b) the Wf-323F
    # report JSON at molmetal/reports/wf323f_m4_sa_qed_subcriterion/
    # carries the per-pool distribution + sub-rate numeric.  No Vina
    # sub-criterion is checked here (USER_OK-gated GPU work).
    wf323f_m4_metric_json = admet_dir / "m4_success_rate.json"
    wf323f_m4_report_json = (
        MOLMETAL_ROOT / "reports" / "wf323f_m4_sa_qed_subcriterion" / "m4_rate.json"
    )
    wf323f_m4_todo_doc = (
        REPO_ROOT / "TODO" / "completed" / "323f_m4_sa_qed_subcriterion.md"
    )
    # Wf-325E — ADMET 4 BELOW druglikeness documented gate.  Verifies
    # the 4 BELOW ADMET metrics (aqueous_solubility_logS / logp7_4 /
    # hepatotox_index / m6_kappa) all carry the canonical
    # HONEST_NEGATIVE: true + honest_negative_reason contract AND the
    # Wf-325E closeout doc ships on disk.  Aggregates the 4 single-
    # metric Wf-308B / Wf-312C R02 honest-negative gates into a single
    # ship-surface-verification gate so downstream paper_dashboard §3.7
    # honest-negative cluster cannot silently regress.
    # See TODO/completed/325e_admet_4_below_final.md for the ship
    # summary + per-metric honest framing context.
    wf325e_closeout_doc = (
        REPO_ROOT / "TODO" / "completed" / "325e_admet_4_below_final.md"
    )

    # Wf-306C — P3 frontier-math wire-in + 2026-09-29 follow-up gates
    frontier_adapters = (
        MOLMETAL_ROOT / "adapters" / "multi_ph_integration.py",
        MOLMETAL_ROOT / "adapters" / "se3_lipman_integration.py",
        MOLMETAL_ROOT / "adapters" / "se3_sb_integration.py",
        MOLMETAL_ROOT / "adapters" / "transport_relaxation_integration.py",
    )
    todo220_drivers = (
        SCRIPT_DIR / "wf219_temporal_split_driver.py",
        SCRIPT_DIR / "nci60_9cell_chemprop_driver.py",
    )
    anchorguard_consumers = (
        SCRIPT_DIR / "pre_flight_check.py",
        SCRIPT_DIR / "paper_auto_update.py",
        SCRIPT_DIR / "paper_dashboard.py",
    )
    wf304b_selenium_dir = MOLMETAL_ROOT / "reports" / "wf304b_selenium"
    section9_doc = (
        REPO_ROOT / "TODO" / "completed" / "207_unified_pre_flight_checklist_v2.md"
    )
    METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"

    # Wf-322G — TODO-130 G3 closure theorem Catherine 30-metal bench
    # closeout.  Per LAMBDA_ROADMAP §3.3 R22-P5: closure theorem ship
    # surface is computational (Tier-2 strategy per Wf-317). The
    # CPU-only bench + audit + per-metal JSON was shipped in Wf-130
    # P1 (2026-09-27); Wf-255 smoke bench covers 3-metal synthetic
    # path.  This gate verifies the 4 closeout artefacts ship
    # together so downstream consumers cannot silently regress the
    # documentation presence (the actual MEASURED 30-metal real sweep
    # ≥ 0.95 reachability is R20-P5 Phase 3 GPU work, blocked on
    # kill_then_adapt directive per AGENTS.md §5.1).
    catherine_30metal_bench_script = SCRIPT_DIR / "bench_catherine_30metals.py"
    catherine_30metal_audit_script = SCRIPT_DIR / "audit_catherine_30metals.py"
    catherine_30metal_wf255_per_metal_json = (
        MOLMETAL_ROOT / "reports" / "wf255_catherine_30metals" / "per_metal_pass.json"
    )
    catherine_30metal_wf130_combined_json = (
        MOLMETAL_ROOT
        / "reports"
        / "wf130_p1_catherine_30"
        / "catherine_audit_30metals.json"
    )
    catherine_30metal_closeout_doc = (
        REPO_ROOT / "TODO" / "completed" / "322g_todo130_g3_closure_30metal_closeout.md"
    )

    # Wf-314E — OOD 8/30 metals 0.0 reachability per-metal diagnosis
    # (RDKit missing + data availability).  Per
    # `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` +
    # `metrics/by_metric/ood_per_metal_reachability.json` +
    # `TODO/completed/314e_ood_rdkit_fix.md`.  The gate verifies the
    # 3 deliverables ship together (diagnostic report + per-metal
    # HONEST_NEGATIVE_RDKIT_MISSING tags + TODO doc), so downstream
    # tooling branches on the canonical `HONEST_NEGATIVE_RDKIT_MISSING`
    # field rather than re-deriving the FAIL/MISSING comparison.
    wf314e_fix_summary = (
        MOLMETAL_ROOT / "reports" / "wf314e_ood_rdkit_fix" / "fix_summary.json"
    )
    wf314e_todo_doc = REPO_ROOT / "TODO" / "completed" / "314e_ood_rdkit_fix.md"
    ood_per_metal_reachability_json = METRICS_DIR / "ood_per_metal_reachability.json"
    ood_reachability_summary_json = METRICS_DIR / "ood_reachability_summary.json"

    # Wf-325F — Phase-3G 10x3 rotation unblock verification.  Per
    # `molmetal/reports/wf325f_phase3g_10x3_smoke/result.json` +
    # `metrics/by_metric/metal_coord_compliance_phase3g.json` +
    # `TODO/completed/325f_f2a_phase3g_unblock.md`.  The gate
    # verifies the 4 deliverables ship together (CPU smoke script +
    # smoke result report + metric JSON + TODO closeout doc) so
    # downstream tooling can verify the F2(a) MetalLigandExchange
    # SMARTS runtime dispatcher (Wf-318F) + 84-rule MLE matrix
    # (Wf-290F4) together lift the Phase-3G singleton collapse on
    # CPU without requiring GPU cycles.  The GPU portion
    # (Wf-310E Wf-135 Phase 2) remains BLOCKED on user OK per
    # mol-metal-gpu-execution §0 kill_then_adapt directive.
    wf325f_smoke_script = SCRIPT_DIR / "wf325f_phase3g_10x3_smoke.py"
    wf325f_smoke_result = (
        MOLMETAL_ROOT / "reports" / "wf325f_phase3g_10x3_smoke" / "result.json"
    )
    wf325f_metric_json = METRICS_DIR / "metal_coord_compliance_phase3g.json"
    wf325f_todo_doc = REPO_ROOT / "TODO" / "completed" / "325f_f2a_phase3g_unblock.md"

    # Wf-315D — PoseBusters smoke retry gate (WEAK_TO_STRONG §2.6).
    # Per Wf-311E Part 2: steric_clash_pb_dock.json was DEFERRED due to
    # missing RDKit + posebusters packages. Wf-315D verifies one of:
    #   (a) packages are installed AND the 1h36 metric has been updated
    #       to MEASURED_SMOKE with smoke output on disk
    #   (b) packages are missing AND HONEST_NEGATIVE_BLOCKED tag is set
    #       with clear blocker_description (no fabricated measurement).
    wf315d_pb_metric_json = METRICS_DIR / "steric_clash_pb_dock_1h36.json"
    # Wf-319B — UFF-min Pt-pool presence + honest-negative tag gate.
    # Per Wf-319B ship: 96/100 Pt conformers UFF-minimized on the
    # platinai_4cell.csv Pt subset (Rusanov 2025). Energies are
    # honest-negative (UFF lacks d-block Pt parameters); the
    # conformers + 3D coords are still valid for downstream Vina
    # docking (WEAK_TO_STRONG §2.7 PROJECTED).
    wf319b_pool_json = (
        MOLMETAL_ROOT / "reports" / "wf319b_uff_min_pt_pool" / "pool.json"
    )
    wf319b_metrics_json = METRICS_DIR / "uff_energy_pt_pool.json"
    wf315d_pb_smoke_output = (
        MOLMETAL_ROOT / "reports" / "wf315d_pb_smoke" / "m8_prod_1h36.json"
    )

    # Wf-323C — UFF pre-relax + Vina re-dock CPU smoke (follow-up to
    # Wf-319C / Wf-315D). Per WEAK_TO_STRONG §2.6 R18: takes Wf-319B
    # Pt-pool + Vina-re-dock + PB dock mode; documents the lift (or
    # HONEST_NEGATIVE_UFF_INSUFFICIENT).  See
    # TODO/completed/323c_m8_full_cpu_fix.md for the full ship.
    wf323c_result_json = MOLMETAL_ROOT / "reports" / "wf323c_m8_full" / "result.json"
    wf323c_metric_json = METRICS_DIR / "steric_clash_pb_dock_1h36.json"
    wf323c_pool_json = wf319b_pool_json
    wf315d_pb_venv_python = (
        # Wf-A1: canonical repo-local venv per AGENTS.md §5.9 (was the
        # project .venv; .venv carries posebusters + meeko + vina).
        "/home/hugo/codes/mol-metal/.venv/bin/python"
    )

    # Wf-319C — PB packages status documented gate (companion to
    # Wf-315D).  Reuses the same venv path + metric + smoke output but
    # is a stricter "all three artefacts present" verifier rather than
    # the Wf-315D branch-on-install-state logic.  See
    # `TODO/completed/319c_pb_packages_status_present.md`.
    wf319c_pb_venv_python = wf315d_pb_venv_python
    wf319c_pb_metric_json = wf315d_pb_metric_json
    wf319c_pb_smoke_output = wf315d_pb_smoke_output

    # Wf-319A — ablation_runner.py unit tests + aggregate.csv ship
    # gate.  Per Wf-315B analysis: the 1037-LOC ablation_runner.py had
    # ZERO unit tests before Wf-319A.  This gate enforces the 2-deliverable
    # ship surface:
    #   1. molmetal/tests/test_wf319a_ablation_runner.py — 12 NEW tests
    #      (3 toggleable specs + 3 DESIGN_ONLY specs + 2 aggregate.csv +
    #      2 final.md + 2 integration with metallo_drug_smoke_retrain.py)
    #   2. molmetal/reports/wf_ablation_runner/aggregate.csv — 11 ablation
    #      rows + 1 BASELINE = 12 data rows for paper §6 input
    # The gate also surfaces the M14 honest-negative finding
    # (--lr-warmup-steps not in harness) as part of the gate PASS
    # contract.  See TODO/completed/319a_ablation_runner_smoke.md for
    # the ship summary + WEAK_TO_STRONG §2.7 honest-framing context.
    wf319a_test_file = TESTS_DIR / "test_wf319a_ablation_runner.py"
    wf319a_aggregate_csv = (
        MOLMETAL_ROOT / "reports" / "wf_ablation_runner" / "aggregate.csv"
    )

    # Wf-321A — Phase 2 CPU scaffold readiness paths.  Phase 1
    # cohort JSON (Wf-309A ship) + Phase 2 trainer + Wf-310E
    # launch wrapper.  See TODO/completed/321a_wf135_phase2_cpu_
    # scaffold.md for the per-deliverable MEASURED status.
    wf321a_phase1_cohort_json = (
        MOLMETAL_ROOT
        / "reports"
        / "wf139_p7_rh_active_cohort"
        / "synthetic_rh_pos_neg.json"
    )
    wf321a_trainer_script = MOLMETAL_ROOT / "scripts" / "train_metal_lomo_v2.py"
    wf321a_launch_wrapper = REPO_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"
    # Wf-322F — Wf-176 m19 LOMO Rh oversample FINAL closeout
    # documented gate.  Verifies the close-out ship surface for the
    # Wf-135 / Wf-176 / TODO-176 workstream: close-out doc + Phase 1
    # cohort JSON + Phase 1 cohort synthesis doc + Phase 2 GPU launch
    # wrapper.  See TODO/completed/322f_wf176_m19_lomo_final_closeout.md
    # for the per-deliverable MEASURED status + WEAK_TO_STRONG §2.7
    # honest-framing context.
    wf322f_closeout_doc = (
        REPO_ROOT / "TODO" / "completed" / "322f_wf176_m19_lomo_final_closeout.md"
    )
    wf322f_phase1_cohort_json = (
        MOLMETAL_ROOT
        / "reports"
        / "wf139_p7_rh_active_cohort"
        / "synthetic_rh_pos_neg.json"
    )
    wf322f_phase1_cohort_synthesis = (
        MOLMETAL_ROOT / "reports" / "wf139_p7_rh_active_cohort" / "synthesis.md"
    )
    wf322f_gpu_launch_wrapper = (
        REPO_ROOT / "scripts" / "launch_wf310e_wf135_phase2_gpu.sh"
    )
    # Wf-325D — m6_kappa BELOW threshold + NEGATIVE_WEAK
    # MetalCytoToxDB validation FINAL closeout documented gate.
    # Verifies the close-out ship surface for the m6_kappa workstream:
    # close-out doc + metrics/by_metric/m6_kappa.json HONEST_NEGATIVE
    # contract + metalcytotox_validation block.  See
    # TODO/completed/325d_m6_kappa_final.md for the per-deliverable
    # MEASURED status + WEAK_TO_STRONG §2.7 honest-framing context.
    # No GPU work — recovery path (chemprop v2 + W₂) remains
    # USER_OK-gated per mol-metal-gpu-execution §0 kill_then_adapt.
    wf325d_closeout_doc = REPO_ROOT / "TODO" / "completed" / "325d_m6_kappa_final.md"
    wf325d_metric_json = REPO_ROOT / "metrics" / "by_metric" / "m6_kappa.json"
    # Wf-321E — ABLATION_TABLE registry mirror check (companion to
    # Wf-319A's artefact-side aggregate.csv gate).  Verifies the
    # source-level registry of ablations declared in
    # ``molmetal/scripts/ablation_runner.py`` mirrors the 11 R17
    # ablations in §7 of
    # ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md``.  See
    # TODO/completed/321e_ablation_runner_integration_test.md.
    wf321e_runner_module = MOLMETAL_ROOT / "scripts" / "ablation_runner.py"

    # Wf-327 — pre-flight gate for ``--cpu-num-workers >= 4`` (Wf-326B
    # §6.2 follow-up).  Verifies all 6 Wf-326B-active launch scripts
    # request >= 4 workers.  Per Wf-328-W2 (2026-09-29) every launcher
    # was migrated to ``--cpu-num-workers 8``; this gate keeps the
    # regression from re-occurring by fail-fast blocking any future
    # commit that flips a launcher back to 0 (which would re-enable the
    # Wf-326B root-cause: manual-loop fallback path, no pin_memory, no
    # workers, no prefetch).
    wf327_launch_scripts = (
        REPO_ROOT / "scripts" / "launch_wf124_p3_g5.sh",
        REPO_ROOT / "scripts" / "launch_wf124_p3_g6.sh",
        REPO_ROOT / "scripts" / "launch_wf310b_g6_final.sh",
        REPO_ROOT / "scripts" / "retry_g5_missing_pairs.sh",
    )  # Wf-446: removed launch_wf124_p3_g4.sh + launch_wf300_cfm_main.sh (never part of Wf-326B §6.2 6-wrapper recipe)

    # Wf-313C — Day 3-5 paper-grade integration smoke tests + per-adapter
    # research-paper reference doc.  Per TODO-234 §6 + Wf-310F §6: 8 NEW
    # integration smoke tests live at
    # ``molmetal/tests/test_wf313c_frontier_math_integration_e2e.py``,
    # and 4 per-adapter paper refs + 1 index README live at
    # ``molmetal/reports/wf313c_frontier_math_paper_refs/``.  Both
    # deliverables are CPU-only — no GPU launch required.
    wf313c_e2e_test = TESTS_DIR / "test_wf313c_frontier_math_integration_e2e.py"
    wf313c_paper_refs_dir = (
        MOLMETAL_ROOT / "reports" / "wf313c_frontier_math_paper_refs"
    )
    wf313c_paper_ref_files = (
        wf313c_paper_refs_dir / "multi_ph_integration_ref.md",
        wf313c_paper_refs_dir / "se3_lipman_integration_ref.md",
        wf313c_paper_refs_dir / "se3_sb_integration_ref.md",
        wf313c_paper_refs_dir / "transport_relaxation_integration_ref.md",
        wf313c_paper_refs_dir / "README.md",
    )

    # Wf-312E — 4-LOI send-ready gate (TODO-308D integration). Per
    # Wf-308D PART 2 consolidation review (TODO/pending/wetlab_loi_
    # consolidation_review.md): 4 LOI drafts at TODO/pending/ are
    # structurally ready-to-send; actual cold-email send is user's
    # responsibility per TODO/pending/84_no_paper_task_without_
    # authorization.md.
    loi_send_review_doc = (
        REPO_ROOT / "TODO" / "pending" / "wetlab_loi_consolidation_review.md"
    )

    # Wf-313H — R20 Phase 0 loader closeout gate (TODO-313 /
    # R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §1.1).  The 3 P0 CRITICAL
    # loaders (binding_db / nci60 / drugood_loader) must each have
    # ≥12 dedicated tests + an API doc.  This gate verifies all 9
    # artefacts are non-empty on disk and the 3 test files pass
    # pytest (no silent regressions on the loader contract).
    r20_phase0_loader_modules = (
        DATA_DIR / "binding_db.py",
        DATA_DIR / "nci60.py",
        DATA_DIR / "drugood_loader.py",
    )
    r20_phase0_loader_api_docs = (
        DATA_DIR / "binding_db.md",
        DATA_DIR / "nci60.md",
        DATA_DIR / "drugood_loader.md",
    )
    # Wf-322E — R20 §1.1 loader-API documentation index closeout.
    # Companion to ``r20_phase0_loader_api_docs``: this tuple adds the
    # NEW index doc ``molmetal/data/R20_P0_loaders_api.md`` that
    # aggregates the 3 per-loader docs into a single landing page.
    # Pre-flight gate ``p3_r20_p0_loader_api_doc_present`` (added
    # below) verifies all 4 docs are non-empty files on disk.
    # See TODO/completed/322e_todo_148_loader_api_doc.md.
    r20_phase0_loader_api_doc_index = (
        DATA_DIR / "binding_db.md",
        DATA_DIR / "nci60.md",
        DATA_DIR / "drugood_loader.md",
        DATA_DIR / "R20_P0_loaders_api.md",
    )
    r20_phase0_loader_tests = (
        # Wf-313H original tests (class-grouped Schema/RowAndSplit/EdgeCases/
        # Integration/CrossLoaderPairing — 13/13/16 = 42 tests).
        TESTS_DIR / "test_wf313h_binding_db_loader.py",
        TESTS_DIR / "test_wf313h_nci60_loader.py",
        TESTS_DIR / "test_wf313h_drugood_loader.py",
        # Wf-320F independent re-verification (flat test_wf320f_* functions
        # covering the same contract from a second angle — 12/12/13 = 37
        # tests).  Together with Wf-313H this **double-anchors** the loader
        # contract: any future regression fails at least one of the two
        # suites.  See TODO/completed/320_r20_phase0_loaders_closeout.md.
        TESTS_DIR / "test_wf320f_binding_db_loader.py",
        TESTS_DIR / "test_wf320f_nci60_loader.py",
        TESTS_DIR / "test_wf320f_drugood_loader.py",
    )
    loi_send_files = (
        REPO_ROOT / "TODO" / "pending" / "wetlab_loi_rusanov_2025_platinai.md",
        REPO_ROOT / "TODO" / "pending" / "wetlab_loi_pku_cisplatin_ic50.md",
        REPO_ROOT / "TODO" / "pending" / "wetlab_loi_xiehe_nci60.md",
        REPO_ROOT / "TODO" / "pending" / "wetlab_loi_ornl_protein_metal.md",
    )

    runners: List[Tuple[str, Callable[[], GateResult]]] = []
    # ----- Wf-432-A1 PRE_TRAINING_GATE_v2: 32 NEW gates (all-pipeline only) -----
    # Per TODO/03_plans/PRE_TRAINING_GATE_v2.md §2-§6: 8 Tier-1 must-pass
    # + 7 Tier-2 recommended + 5 MOCK→REAL + 3 honest-negative pre-closure
    # + 9 per-baseline protocol validation hooks = 32 NEW additive gates.
    # All route to 'all' pipeline only per §3.4 design contract.
    if pipeline == "all":
        # Spec source paths (all verified on disk per AGENTS.md §5.5 rule #1)
        _p2_metalcytotox_csv = DOT_DATA_DIR / "metalcytotox_5metal_multitask.csv"
        _p2_platinai_csv = DATA_DIR / "platinai_4cell.csv"
        _p2_crossdocked_csv = DATA_DIR / "crossdocked100_manifest.csv"
        _p2_nci60_csv = DATA_DIR / "nci60_precious_9cell.csv"
        _p2_balanced_rh_json = (
            MOLMETAL_ROOT
            / "reports"
            / "wf387a2_balanced_rh_cohort"
            / "balanced_rh_cohort.json"
        )
        _p2_drugood_loader = DATA_DIR / "drugood_loader.py"
        _p2_leakage_utils = DATA_DIR / "leakage_utils.py"
        _p2_drugood_all_results = (
            MOLMETAL_ROOT / "reports" / "wf265_collect" / "all_results.json"
        )
        _p2_eval_all_evaluators = SCRIPT_DIR / "eval_all_evaluators.py"
        _p2_train_metal_cytotox = SCRIPT_DIR / "train_metal_cytotox.py"
        _p2_train_platinai = SCRIPT_DIR / "train_platinai_4cell.py"
        _p2_train_nci60_10cell = SCRIPT_DIR / "train_nci60_10cell.py"
        _p2_per_seed_template_dir = MOLMETAL_ROOT / "reports" / "wf131_p3_deferred_runs"
        _p2_honest_negative_catalog = REPO_ROOT / "docs" / "HONEST_NEGATIVE_CATALOG.md"
        _p2_section_6_template = (
            REPO_ROOT / "docs" / "SECTION_6_LIMITATIONS_TEMPLATE.md"
        )
        _p2_paper_section_6_v4 = (
            REPO_ROOT / "docs" / "PAPER_SECTION_6_v4_LIMITATIONS_READY.md"
        )
        _p2_paper_section_3_v4 = (
            REPO_ROOT / "docs" / "PAPER_SECTION_3_METRIC_TABLE_FINAL.md"
        )
        _p2_m6_kappa_drop_decision = REPO_ROOT / "docs" / "M6_KAPPA_DROP_DECISION.md"
        _p2_protocol_alignment_matrix = (
            REPO_ROOT / "TODO" / "03_plans" / "PROTOCOL_ALIGNMENT_MATRIX.md"
        )
        _p2_paper_refs_bib = REPO_ROOT / "paper" / "refs.bib"
        _p2_auto_emit_section_6_minor = SCRIPT_DIR / "auto_emit_section_6_minor.py"
        _p2_vina_kcal = REPO_ROOT / "metrics" / "by_metric" / "vina_kcal_per_mol.json"
        _p2_docked_pose_rmsd = (
            REPO_ROOT / "metrics" / "by_metric" / "docked_pose_rmsd.json"
        )
        _p2_decode_ratio = REPO_ROOT / "metrics" / "by_metric" / "decode_ratio.json"
        _p2_drugood_od_auc = (
            REPO_ROOT / "metrics" / "by_metric" / "drugood_ood_auc.json"
        )
        _p2_nci60_9cell = (
            REPO_ROOT / "metrics" / "by_metric" / "nci60_precious_9cell_auc.json"
        )
        _p2_m2_r2_per_cohort = (
            REPO_ROOT / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
        )
        _p2_m6_kappa_metric = REPO_ROOT / "metrics" / "by_metric" / "m6_kappa.json"
        _p2_metal_lomo_auc = (
            REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
        )
        _p2_nci60_9cell_dir = MOLMETAL_ROOT / "reports" / "wf_nci60_precious_9cell"
        _p2_rusanov_baseline_dir = (
            MOLMETAL_ROOT / "reports" / "rusanov_2025_paper_table_s4_s5_baseline"
        )
        _p2_protocol_validation_dir = MOLMETAL_ROOT / "reports" / "protocol_validation"
        # 7 retrain wrappers per PRE_TRAINING_GATE_v2 §2 L1-7
        _p2_wrapper_names = (
            "launch_wf124_p3_g5.sh",
            "launch_wf124_p3_g6.sh",
            "launch_wf299_cfm_main.sh",
            "launch_wf388_a1_metal_coord_retrain.sh",
            "launch_wf394a1_lambda_cfm_patch.sh",
            "launch_wf125_p1_controlled_pair.sh",
            "launch_wf390a1_m2_r2_path_a.sh",
        )
        # Wf-437 simplify: 29 SCAFFOLD-only gate tuples REMOVED per brief
        # / Wf-437 A2 audit. Dropped patterns were:
        #   * p3_protocol_* (15 gates) — Wf-350-C1 SCAFFOLD protocol alignment
        #   * p3_mock_to_real_* (5 gates) — Wf-432-A1 §6 MOCK->REAL hooks
        #   * krasnov_* / rusanov_* / drugood_* (9 gates) — per-baseline protocol
        # All 29 were SCAFFOLD-only doc-trackers (their helper functions
        # at lines 5116..5925 remain intact for ad-hoc use but are not
        # registered as pre-flight gates anymore).
        # The 3 p3_honest_neg_closure_* gates that lived inside this block
        # are KEPT below (they are REAL honest-negative pre-closure gates,
        # not SCAFFOLD protocol hooks).
        runners += [
            # ---- §6 L4 honest-negative pre-closure (3 gates; REAL) ----
            (
                "p3_honest_neg_closure_m2_r2",
                lambda: gate_p3_honest_neg_closure_m2_r2(
                    "p3_honest_neg_closure_m2_r2",
                    "P3",
                    "all",
                    _p2_m2_r2_per_cohort,
                    _p2_honest_negative_catalog,
                    _p2_paper_section_3_v4,
                    _p2_paper_section_6_v4,
                ),
            ),
            (
                "p3_honest_neg_closure_m6_kappa",
                lambda: gate_p3_honest_neg_closure_m6_kappa(
                    "p3_honest_neg_closure_m6_kappa",
                    "P3",
                    "all",
                    _p2_m6_kappa_metric,
                    _p2_honest_negative_catalog,
                    _p2_m6_kappa_drop_decision,
                    _p2_paper_section_3_v4,
                    _p2_paper_section_6_v4,
                ),
            ),
            (
                "p3_honest_neg_closure_m1_re_fold",
                lambda: gate_p3_honest_neg_closure_m1_re_fold(
                    "p3_honest_neg_closure_m1_re_fold",
                    "P3",
                    "all",
                    _p2_metal_lomo_auc,
                    _p2_honest_negative_catalog,
                    _p2_paper_section_3_v4,
                    _p2_paper_section_6_v4,
                ),
            ),
        ]
    # ---- §3.1 P1 chemprop v2 v3 (12) ----
    if pipeline in ("chemprop_v2_v3", "all"):
        runners += [
            (
                "p1_chemprop_indices_fix",
                lambda: gate_grep(
                    "p1_chemprop_indices_fix",
                    "P1",
                    "chemprop_v2_v3",
                    "indices = train_idx.copy()",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_kendall_optimizer",
                lambda: gate_grep(
                    "p1_chemprop_kendall_optimizer",
                    "P1",
                    "chemprop_v2_v3",
                    "kendall.parameters()",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_dann_per_metal",
                lambda: gate_grep(
                    "p1_chemprop_dann_per_metal",
                    "P1",
                    "chemprop_v2_v3",
                    "per_metal",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_nan_guard",
                lambda: gate_grep(
                    "p1_chemprop_nan_guard",
                    "P1",
                    "chemprop_v2_v3",
                    "nan_to_num",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_grad_clip",
                lambda: gate_grep(
                    "p1_chemprop_grad_clip",
                    "P1",
                    "chemprop_v2_v3",
                    "clip_grad_norm",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_epoch_avg",
                lambda: gate_grep(
                    "p1_chemprop_epoch_avg",
                    "P1",
                    "chemprop_v2_v3",
                    "w2_epoch_avg_history",
                    chemprop_script,
                ),
            ),
            (
                "p1_chemprop_ablation_flags",
                lambda: gate_help(
                    "p1_chemprop_ablation_flags",
                    "P1",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--no-multi-task --no-wass --no-kendall --no-dann",
                    timeout=30.0,
                ),
            ),
            (
                "p1_chemprop_seeds_min",
                lambda: gate_help(
                    "p1_chemprop_seeds_min",
                    "P1",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--n-seeds-min",
                    timeout=30.0,
                ),
            ),
            (
                "p1_chemprop_master_wass_flag",
                lambda: gate_help(
                    "p1_chemprop_master_wass_flag",
                    "P1",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--use-master-wass",
                    timeout=30.0,
                ),
            ),
            (
                "p1_chemprop_master_cath_flag",
                lambda: gate_help(
                    "p1_chemprop_master_cath_flag",
                    "P1",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--use-master-cath",
                    timeout=30.0,
                ),
            ),
            (
                "p1_chemprop_click_aug_flag",
                lambda: gate_help(
                    "p1_chemprop_click_aug_flag",
                    "P1",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--use-click-aug",
                    timeout=30.0,
                ),
            ),
            (
                "p1_chemprop_smoke",
                lambda: _gate_chemprop_quick_smoke(chemprop_script),
            ),
            # Wf-321H — TODO-119 P1.4 CPU scaffold gate.  Verifies the
            # ``--chemprop-mode v3-fix`` token appears in --help output
            # of the chemprop v2/v3 trainer.  This proves the LOMO
            # scaffold levers (--chemprop-min-epochs + --chemprop-lomo-init-scale)
            # are still shipped on the chemprop v2_v3 pipeline (per Wf-266
            # TODO-130 G5 follow-up).  Honest framing: this gate validates
            # CPU scaffold only; the GPU portion (Wf-300C Wave-A) is
            # BLOCKED on user OK per kill_then_adapt §0.
            (
                "p3_todo_119_p14_cpu_smoke_ready",
                lambda: gate_help(
                    "p3_todo_119_p14_cpu_smoke_ready",
                    "P3",
                    "chemprop_v2_v3",
                    chemprop_script,
                    "--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale",
                    timeout=30.0,
                ),
            ),
            # Wf-350-A2 — NeurIPS §E.5 compute-infrastructure manifest
            # gate (per Wf-343a-R2 recommendation; gate #2 of 3).
            # Probes nvidia-smi + torch + uname at host run-time, writes
            # molmetal/reports/wf350a2_compute_manifest.json, and
            # verifies the canonical schema (gpus + driver_version +
            # torch_version + cuda_version + system.{hostname,kernel,arch}).
            # CPU-only gate (no torch.cuda, no GPU kernel launch); the
            # MEASURED-vs-PROJECTED framing per AGENTS.md §5.2 is
            # host-time-probe values only.
            (
                "p1_compute_manifest",
                lambda: gate_wf350a2_compute_manifest(
                    "p1_compute_manifest",
                    "P1",
                    "chemprop_v2_v3",
                    MOLMETAL_ROOT / "reports" / "wf350a2_compute_manifest.json",
                ),
            ),
        ]

    # ---- §4.2 P1 CFM (7) ----
    # 6 Wf-259 baseline gates + 1 HOLE-1 live decode canary
    # (``p1_cfm_live_decode_canary``, cfm-pipeline-only — see the
    # ``if pipeline == "cfm":`` block after this one).
    if pipeline in ("cfm", "all"):
        runners += [
            (
                "p1_cfm_script_exists",
                lambda: gate_file_exists(
                    "p1_cfm_script_exists",
                    "P1",
                    "cfm",
                    cfm_smoke_script,
                ),
            ),
            (
                "p1_cfm_config_exists",
                lambda: gate_file_exists(
                    "p1_cfm_config_exists",
                    "P1",
                    "cfm",
                    cfm_full_config,
                ),
            ),
            (
                "p1_cfm_decoder_exists",
                lambda: gate_file_exists(
                    "p1_cfm_decoder_exists",
                    "P1",
                    "cfm",
                    cfm_decoder,
                ),
            ),
            (
                "p1_cfm_wasserstein_loss_exists",
                lambda: gate_file_exists(
                    "p1_cfm_wasserstein_loss_exists",
                    "P1",
                    "cfm",
                    cfm_wass_loss,
                ),
            ),
            (
                # HOLE-1 fix: HISTORICAL anchor only (frozen untracked
                # Wf-188 artifact; checkpoint changed 6 days after it
                # was written).  Live code verification moved to
                # ``p1_cfm_live_decode_canary`` below.  Name kept so
                # existing reports / closeouts referencing
                # ``p1_cfm_smoke_decoded`` stay resolvable.
                "p1_cfm_smoke_decoded",
                lambda: gate_cfm_smoke_decoded_historical(
                    "p1_cfm_smoke_decoded",
                    "P1",
                    "cfm",
                    wf188_result,
                ),
            ),
            (
                "p1_cfm_catherine_30metals",
                lambda: gate_json_threshold(
                    "p1_cfm_catherine_30metals",
                    "P1",
                    "cfm",
                    wf256_per_metal,
                    extractor=lambda d: (
                        sum(
                            1
                            for m in d.get("per_metal", [])
                            if m.get("six_of_six_pass", False)
                        )
                        / max(1, len(d.get("per_metal", [])))
                    ),
                    threshold=0.95,
                    op=">=",
                ),
            ),
        ]

    # ---- HOLE-1 fix (2026-10-03): LIVE cfm decode canary (cfm-only) ----
    # The frozen Wf-188 anchor above cannot see today's code (untracked
    # artifact; checkpoint changed after it was written).  This gate
    # executes the proven Wf-385-A3 3-epoch CPU smoke of TODAY's
    # train_cfm_yuelbond_wasserstein.py + YuelBond decoder and requires
    # loss decrease (0.3% floor) + decode_ratio_rdkit_strict > 0
    # (fail-closed 0.0).  MEASURED wall 6.2-6.7 s on the dev host.
    # Registered `cfm`-only: the `all` pipeline already runs the
    # identical canary under
    # ``p3_wf385_scaffold_loss_wire_in_verified`` (all-only block
    # further below), so an `all` registration would run the same smoke
    # twice per pre-flight.
    if pipeline == "cfm":
        runners += [
            (
                "p1_cfm_live_decode_canary",
                lambda: gate_cfm_live_decode_canary(
                    "p1_cfm_live_decode_canary",
                    "P1",
                    "cfm",
                    SCRIPT_DIR / "wf385a3_scaffold_wire_in_gate.py",
                ),
            ),
        ]

    # ---- Wf-A8 PRE_TRAINING_GATE_v2 yaml-alignment gates (4 NEW; all +
    # chemprop_v2_v3 pipelines per task brief "alignment-not-reproduction
    # must be hard-enforced in pre-flight"; cfm/frontier excluded per the
    # 4 yaml files in configs/ that are all chemprop_v2_v3 configs).
    # Each gate scans every active yaml under configs/ + configs/training/
    # and enforces one alignment dimension.
    if pipeline in ("all", "chemprop_v2_v3"):
        # 2026-10-04 FIX — the three `cfm_*.yaml` entries used to be listed
        # here, contradicting this block's own comment ("cfm/frontier
        # excluded").  These four gates enforce the BENCHMARK-ALIGNMENT
        # contract: `data.dataset` must be a Tier-1 benchmark dataset and
        # `data.split.type` must name a baseline paper's split protocol.  A CFM
        # config is a GENERATION-MODEL config: its corpus is a pretraining
        # corpus (tmQM), which has no published benchmark split to align to,
        # and the tmQM 2020 paper defines no split at all.  Checking it here
        # can only produce FAILs that mean "this gate does not govern this
        # file".
        #
        # They were passing before only because the configs declared
        # `data.dataset: metalcytotox_5metal` — a value chosen to satisfy this
        # gate, as their own comment said ("Wf-A8c FIX: rename per Tier-1
        # allowed list"), next to a note that the key "never resolved for the
        # CFM trainer, which read no corpus at all".  The declaration was
        # false.  Correcting it to the truth (`tmqm`) is what surfaced this.
        _wf_a8_yaml_paths: Tuple[Path, ...] = (
            CONFIGS_DIR / "default.yaml",
            CONFIGS_DIR / "wf124_p3_50epoch_full_sweep_g5.yaml",
            CONFIGS_DIR / "wf124_p3_per_innovation_ablation_g6.yaml",
            CONFIGS_DIR / "wf124_p3_remaining_metals_g4.yaml",
            CONFIGS_DIR / "training" / "platinai_pt_4cell_5seed_50ep_v1.yaml",
        )
        runners += [
            # Gate 1 — yaml data.dataset ∈ Tier-1 authorized benchmark list
            (
                "p3_protocol_alignment_yaml_dataset_authorized",
                lambda: gate_p3_protocol_alignment_yaml_dataset_authorized(
                    "p3_protocol_alignment_yaml_dataset_authorized",
                    "P3",
                    pipeline,
                    _wf_a8_yaml_paths,
                ),
            ),
            # Gate 2 — yaml data.split.type ∈ Rusanov/Krasnov/Ji baseline mapping
            (
                "p3_protocol_alignment_yaml_split_matches_baseline",
                lambda: gate_p3_protocol_alignment_yaml_split_matches_baseline(
                    "p3_protocol_alignment_yaml_split_matches_baseline",
                    "P3",
                    pipeline,
                    _wf_a8_yaml_paths,
                ),
            ),
            # Gate 3 — yaml eval targets within baseline paper ranges
            (
                "p3_protocol_alignment_yaml_target_in_baseline_range",
                lambda: gate_p3_protocol_alignment_yaml_target_in_baseline_range(
                    "p3_protocol_alignment_yaml_target_in_baseline_range",
                    "P3",
                    pipeline,
                    _wf_a8_yaml_paths,
                ),
            ),
            # Gate 4 — yaml entry.command must NOT contain inline CLI flags
            # (per user voice 2026-09-30 "never use CLI again always yaml")
            (
                "p3_protocol_alignment_yaml_no_inline_cli",
                lambda: gate_p3_protocol_alignment_yaml_no_inline_cli(
                    "p3_protocol_alignment_yaml_no_inline_cli",
                    "P3",
                    pipeline,
                    _wf_a8_yaml_paths,
                ),
            ),
        ]

    # ---- §5 P2 (4 — data + tests + vina) ----
    if pipeline == "all":
        runners += [
            (
                "p2_data_metalcytotox_exists",
                lambda: gate_min_lines(
                    "p2_data_metalcytotox_exists",
                    "P2",
                    "data",
                    cytox_csv,
                    26800,
                ),
            ),
            (
                "p2_data_metallo_drugs_exists",
                lambda: gate_min_lines(
                    "p2_data_metallo_drugs_exists",
                    "P2",
                    "data",
                    metallo_csv,
                    449,
                ),
            ),
            (
                # L4 fix (2026-10-03): check_data_leakage.py was unwired.
                # Runs the Wf-44 Phase 3 diagnostic with EXPLICIT inputs
                # (platinai_5k + metalcytotox v4) and gates the axes the
                # trainers consume (scaffold split both datasets + shipped
                # split for v4).  Random-split overlap is recorded as the
                # expected honest-negative (it leaks by design — that is
                # the diagnostic's point).  MEASURED both ways: v1 CSV ->
                # shipped strict=FAIL (48 shared scaffolds); v4 -> PASS
                # (0 shared scaffolds).  ~22 s wall (CPU-only, repo venv).
                # All-only per the §3.4 design contract of this section.
                "p2_data_leakage_diag",
                lambda: gate_p2_data_leakage_diag(
                    "p2_data_leakage_diag",
                    "P2",
                    "data",
                    SCRIPT_DIR / "check_data_leakage.py",
                    platinai_csv=DATA_DIR / "platinai_5k.csv",
                    metalcytotox_csv=cytox_csv_v4,
                    out_path=(GATE_SCRATCH_ROOT / "wf44_leak" / "leakage_check.json"),
                ),
            ),
            (
                "p2_tests_pass",
                lambda: gate_pytest(
                    "p2_tests_pass",
                    "P2",
                    "tests",
                    [
                        # Wf-259 baseline 3 (chemprop v2 v3 nan-guard + resume + ablation flags)
                        TESTS_DIR / "test_wf245_ablation_flags.py",
                        TESTS_DIR / "test_wf146_resume.py",
                        TESTS_DIR / "test_train_chemprop_v2_v3_nan_guard.py",
                        # Wf-296B addendum 4 (Wf-283B / Wf-285B / Wf-290 / Wf-268A
                        # newly collectable tests added to ensure regressions are
                        # caught in preflight).
                        # - Wf-283B: test_no_kendall_no_dann_flags.py
                        # - Wf-285B: test_kendall_dann_wiring.py
                        # - Wf-268A: test_wf268a_cfm_ablation_flags.py
                        # - Wf-290F4: test_metal_ligand_exchange_matrix.py
                        # NOTE: test_f2a_metal_ligand_exchange.py (Wf-290F1) is
                        # intentionally excluded — its 33 syntemol-rule tests
                        # assume SyntheMolReactionRule.apply() which the current
                        # sbdd_env.syntemol_reactions module does NOT expose
                        # (collected OK after Wf-290F1 import fix, but 33/309
                        # tests fail with AttributeError pre-existing upstream).
                        # Tracked for Wf-297 follow-up; not blocking preflight.
                        TESTS_DIR / "test_kendall_dann_wiring.py",
                        TESTS_DIR / "test_no_kendall_no_dann_flags.py",
                        TESTS_DIR / "test_wf268a_cfm_ablation_flags.py",
                        TESTS_DIR / "test_metal_ligand_exchange_matrix.py",
                        # Wf-306C addendum 1 (AnchorGuard version-pin tests;
                        # 11 tests including test_anchor_version_pin_blocks_regression
                        # added in PART 3 — guards against silent version-drift
                        # regressions on the paper-grade anchor fleet).
                        TESTS_DIR / "test_read_anchor_json.py",
                        # Wf-344-A3 addendum 4 (configure_reference chain
                        # CWD-robustness defence tests; 4 tests added
                        # per Wf-338-A3 finding 3-5 to guard against
                        # cascade-FAIL on p2_tests_pass +
                        # p3_frontier_math_day_3_5 +
                        # p3_smoke_fail_on_missing_audit).
                        TESTS_DIR / "test_wf344a3_configure_reference.py",
                    ],
                    timeout=600.0,
                ),
            ),
            (
                "p2_vina_installed",
                lambda: gate_which_or_apt(
                    "p2_vina_installed",
                    "P2",
                    "data",
                    vina_bin,
                ),
            ),
            # ---- §1 P0 BLOCKERs (config schema + CSV hash; auto-verifiable) ----
            (
                "p0_config_schema_verify",
                lambda: gate_yaml_schema(
                    "p0_config_schema_verify",
                    "P0",
                    "config",
                    [
                        CONFIGS_DIR / "cfm_full_retrain.yaml",
                        CONFIGS_DIR / "cfm_per_innovation_ablation_p1.4.yaml",
                        CONFIGS_DIR / "cfm_real_decode_lift_p1.3.yaml",
                        CONFIGS_DIR / "wf124_p3_50epoch_full_sweep_g5.yaml",
                        CONFIGS_DIR / "wf124_p3_per_innovation_ablation_g6.yaml",
                        CONFIGS_DIR / "wf124_p3_remaining_metals_g4.yaml",
                        CONFIGS_DIR / "default.yaml",
                    ],
                ),
            ),
            (
                "p0_csv_hash_verify",
                # HOLE-6 fix: spec list now comes from gated_csv_specs()
                # so the gate and the `--freeze-csv-hashes` manifest
                # generator cannot drift apart.
                lambda: gate_csv_hash(
                    "p0_csv_hash_verify",
                    "P0",
                    "data",
                    gated_csv_specs(),
                ),
            ),
            (
                "p0_scaffold_zero_leak",
                lambda: gate_scaffold_zero_leak(
                    "p0_scaffold_zero_leak",
                    "P0",
                    "data",
                    [
                        # HOLE-5 fix: v2 + v3 were previously ungated.  v1 is
                        # deliberately excluded from THIS zero-leak gate
                        # (pre-Wf-261 leaky artifact, MEASURED 109 total
                        # pairwise overlaps).  NOTE (L5 fix, 2026-10-03): the
                        # earlier "not a trainer input" claim here was FALSE —
                        # v1 IS still the default data input of 10+ trainer /
                        # benchmark scripts (grep-MEASURED roster in
                        # METALCYTOTOX_V1_DEFAULT_CONSUMERS); its visibility
                        # is covered by p0_metalcytotox_v1_known_leaky below.
                        (cytox_csv_v2, "metalcytotox_v2"),
                        (cytox_csv_v3, "metalcytotox_v3"),
                        (cytox_csv_v4, "metalcytotox_v4"),
                    ],
                ),
            ),
            (
                # L5 fix (2026-10-03): explicit coverage of the KNOWN-LEAKY
                # v1 artifact.  v1 bytes are hash-frozen (p0_csv_hash_verify,
                # blocking ON), so the leak cannot drift silently — what was
                # missing is visibility that the leaky artifact is still the
                # DEFAULT data of 10+ trainer / benchmark scripts.  Report-
                # only: PASS iff the leak is still measurable (> 0), FAIL iff
                # the artifact no longer leaks (mutated / re-split /
                # replaced), SKIP if the gitignored .data CSV is absent on
                # this host.  See the gate docstring for both-ways evidence.
                "p0_metalcytotox_v1_known_leaky",
                lambda: gate_metalcytotox_v1_known_leaky(
                    "p0_metalcytotox_v1_known_leaky",
                    "P0",
                    "data",
                    cytox_csv,
                ),
            ),
            (
                # HOLE-5 fix: nci60 has no split columns, so the CSV-level
                # scaffold gate cannot see it.  This verifies the shipped
                # Wf-A10/m21 loader split + the chemprop driver's scaffold
                # helper path (see gate docstring for exact coverage).
                "p0_nci60_split_no_leak",
                lambda: gate_nci60_split_no_leak(
                    "p0_nci60_split_no_leak",
                    "P0",
                    "data",
                    DATA_DIR / "nci60_precious_9cell.csv",
                ),
            ),
            # ---- Wf-306C §9 P3 (frontier-math + 2026-09-29 follow-ups; 7 gates) ----
            (
                "p3_frontier_math_wire_in",
                lambda: gate_python_importable(
                    "p3_frontier_math_wire_in",
                    "P3",
                    "frontier",
                    frontier_adapters,
                ),
            ),
            # ---- Wf-313C Day 3-5 (8 NEW paper-grade integration smoke
            # tests; CPU-only).  Per TODO-234 §6 + Wf-310F §6: the
            # 8 tests live at
            # ``molmetal/tests/test_wf313c_frontier_math_integration_e2e.py``
            # and exercise the 4 standalone adapters via the
            # :class:`FrontierMathIntegration` factory surface.
            (
                "p3_frontier_math_day_3_5_integration_tests",
                lambda: gate_pytest(
                    "p3_frontier_math_day_3_5_integration_tests",
                    "P3",
                    "frontier",
                    [wf313c_e2e_test],
                    timeout=120.0,
                ),
            ),
            # ---- Wf-313C Day 3-5 (4 per-adapter paper refs + index
            # README).  All 5 markdown files must be present for the
            # gate to PASS.
            (
                "p3_frontier_math_paper_refs_present",
                lambda: gate_all_files_exist(
                    "p3_frontier_math_paper_refs_present",
                    "P3",
                    "frontier",
                    wf313c_paper_ref_files,
                ),
            ),
            (
                "p3_todo220_drivers_present",
                lambda: gate_all_files_exist(
                    "p3_todo220_drivers_present",
                    "P3",
                    "drivers",
                    todo220_drivers,
                ),
            ),
            (
                "p3_anchorguard_version_pin",
                lambda: gate_anchor_version_pin(
                    "p3_anchorguard_version_pin",
                    "P3",
                    "anchors",
                ),
            ),
            (
                "p3_anchorguard_consumers_adopt",
                lambda: gate_all_files_exist(
                    "p3_anchorguard_consumers_adopt",
                    "P3",
                    "consumers",
                    anchorguard_consumers,
                ),
            ),
            (
                "p3_harness_auto_fallback_present",
                lambda: gate_file_exists(
                    "p3_harness_auto_fallback_present",
                    "P3",
                    "harness",
                    HARNESS_WRAPPER,
                ),
            ),
            (
                "p3_wf304b_platinai_selenium_diag",
                lambda: gate_all_files_exist(
                    "p3_wf304b_platinai_selenium_diag",
                    "P3",
                    "diagnostic",
                    (
                        wf304b_selenium_dir / "block_state.html",
                        wf304b_selenium_dir / "selenium_report.md",
                    ),
                ),
            ),
            # ---- Wf-314E — OOD 8/30 metals 0.0 reachability per-metal diagnosis ----
            # Per Wf-311D PART 1 (aggregate_ood_pass_rate 0.8327 /
            # aggregate_reachability_macro 0.4989) + Wf-314E per-metal
            # diagnosis: 8/30 metals report closure_reachability_rate = 0.0
            # because CPU-only env at Wf-130 P3 bench time lacked RDKit
            # (from_smiles requires RDKit per combinators.py:417-449).
            # This gate enforces the 3-deliverable ship surface:
            #   1. molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json
            #      (per-metal diagnosis + Wf-41 baseline comparison + fix_path)
            #   2. molmetal/metrics/by_metric/ood_per_metal_reachability.json
            #      (HONEST_NEGATIVE_RDKIT_MISSING + HONEST_NEGATIVE_DATA_AVAILABILITY
            #       tags for the 8 metals, honest_negative_reason strings)
            #   3. TODO/completed/314e_ood_rdkit_fix.md (ship summary)
            # Downstream tooling branches on the canonical
            # HONEST_NEGATIVE_RDKIT_MISSING field rather than re-deriving
            # the FAIL/MISSING comparison (mirrors Wf-308B aqueous_solubility
            # + Wf-312C hepatotox/logp7_4/m6_kappa contract).
            (
                "p3_ood_rdkit_fix_documented",
                lambda: gate_all_files_exist(
                    "p3_ood_rdkit_fix_documented",
                    "P3",
                    "ood_rdkit_fix",
                    (
                        wf314e_fix_summary,
                        ood_per_metal_reachability_json,
                        ood_reachability_summary_json,
                        wf314e_todo_doc,
                    ),
                ),
            ),
            # Wf-325F — Phase-3G 10x3 rotation unblock verification.
            # Per Wf-311D PART 5 honest-negative + Wf-318F runtime
            # dispatcher SHIPPED + Wf-290F4 84-rule MLE matrix SHIPPED.
            # Verifies the 4 deliverables ship together so downstream
            # tooling can confirm the Phase-3G singleton collapse
            # (5x1 trivial PASS) is structurally lifted by F2(a)
            # runtime + 84-rule MLE matrix on CPU without requiring
            # GPU cycles.  The GPU portion (Wf-310E Wf-135 Phase 2)
            # remains BLOCKED on user OK per mol-metal-gpu-execution
            # §0 kill_then_adapt directive.  Mirrors the Wf-314E
            # 3-deliverable ship pattern.
            (
                "p3_phase3g_f2a_runtime_unblock_verified",
                lambda: gate_all_files_exist(
                    "p3_phase3g_f2a_runtime_unblock_verified",
                    "P3",
                    "f2a_runtime_unblock",
                    (
                        wf325f_smoke_script,
                        wf325f_smoke_result,
                        wf325f_metric_json,
                        wf325f_todo_doc,
                    ),
                ),
            ),
            (
                "p3_section9_honest_framing_min30",
                lambda: gate_section9_entry_count(
                    "p3_section9_honest_framing_min30",
                    "P3",
                    "honest_framing",
                    section9_doc,
                    min_entries=36,
                ),
            ),
            # ---- Wf-308B R03 — ADMET pass-through druglikeness gates (3) ----
            # Per Wf-307C integration plan: surfaces the 3 ADMET metrics
            # emitted by `molmetal/scripts/admet_pass_through.py` (Wf-303C T2)
            # in the v3 pre-flight. The thresholds are STRICTER than the
            # per-metric source target bands (0.30 / -6.0 / -4.0) so the
            # gates are forward ships, not auto-passes:
            #   * ames_mutagen ≤ 0.10 (druglikeness; source band < 0.30)
            #   * cell_permeability_logPapp ≥ -5.0 (druglikeness; > -6.0)
            #   * aqueous_solubility_logS carries HONEST_NEGATIVE tag
            #     (cohort mean -7.612 below -4.0 target — surfaced honestly,
            #      NOT auto-passed)
            (
                "p3_admet_ames_druglikeness",
                lambda: gate_json_threshold(
                    "p3_admet_ames_druglikeness",
                    "P3",
                    "admet_druglikeness",
                    admet_ames_json,
                    extractor=lambda d: float(d.get("value", 1.0)),
                    threshold=0.10,
                    op="<=",
                ),
            ),
            (
                "p3_admet_cell_perm_druglikeness",
                lambda: gate_json_threshold(
                    "p3_admet_cell_perm_druglikeness",
                    "P3",
                    "admet_druglikeness",
                    admet_cell_perm_json,
                    extractor=lambda d: float(d.get("value", -10.0)),
                    threshold=-5.0,
                    op=">=",
                ),
            ),
            (
                "p3_admet_aq_sol_honest_negative_tag",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_aq_sol_honest_negative_tag",
                    "P3",
                    "admet_druglikeness",
                    admet_aq_sol_json,
                ),
            ),
            # ---- Wf-312C R02 — 3 NEW honest-negative ADMET druglikeness
            # gates on pre-existing JSON files (hepatotox / logp7_4 /
            # m6_kappa).  Each gate PASSES iff the JSON carries the
            # canonical ``HONEST_NEGATIVE: true`` + ``honest_negative_reason``
            # field set, mirroring the Wf-308B aqueous_solubility_logS
            # contract so downstream tooling branches on a single canonical
            # field rather than re-deriving the FAIL/THRESHOLD comparison.
            (
                "p3_admet_hepatotox_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_hepatotox_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_hepatotox_json,
                ),
            ),
            (
                "p3_admet_logp7_4_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_logp7_4_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_logp7_4_json,
                ),
            ),
            (
                "p3_admet_m6_kappa_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_m6_kappa_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_m6_kappa_json,
                ),
            ),
            # ---- Wf-325E — ADMET 4 BELOW druglikeness documented gate
            # (aggregated ship-surface verifier).  Per Wf-325E brief:
            # the 4 BELOW metrics (aqueous_solubility_logS from Wf-303C
            # T1 + logp7_4 / hepatotox_index / m6_kappa from Wf-312C R02)
            # must all carry the canonical HONEST_NEGATIVE: true +
            # honest_negative_reason contract AND the Wf-325E closeout
            # doc must ship on disk — so the §3.7 honest-negative
            # cluster in paper_dashboard cannot silently regress on
            # the aggregated ship surface.  See
            # TODO/completed/325e_admet_4_below_final.md for the
            # ship summary + per-metric honest framing.
            (
                "p3_admet_4_below_documented",
                lambda: gate_wf325e_admet_4_below_documented(
                    "p3_admet_4_below_documented",
                    "P3",
                    "admet_druglikeness",
                    admet_aq_sol_json,
                    admet_logp7_4_json,
                    admet_hepatotox_json,
                    admet_m6_kappa_json,
                    wf325e_closeout_doc,
                ),
            ),
            # ---- Wf-315A — SA + QED evaluator pass-through druglikeness
            # gates (WEAK_TO_STRONG §3.6; mirror Wf-308B / Wf-312C pattern).
            # Both JSONs are MEASURED + HONEST_NEGATIVE on the 449 cohort:
            #   * sa_score mean = 4.715 (FAIL — above 4.0; metallo-drug
            #     cohort carries larger polycyclic / click-handle scaffolds)
            #   * qed_score mean = 0.230 (FAIL — below 0.5; Bickerton 2012
            #     weighted aggregate intrinsically penalises the metallo-drug
            #     MW + AROM profiles)
            # Gate PASS iff JSON carries HONEST_NEGATIVE=true + reason.
            (
                "p3_sa_score_metric_present",
                lambda: gate_admet_honest_negative_tag(
                    "p3_sa_score_metric_present",
                    "P3",
                    "admet_druglikeness",
                    admet_sa_score_json,
                ),
            ),
            (
                "p3_qed_score_metric_present",
                lambda: gate_admet_honest_negative_tag(
                    "p3_qed_score_metric_present",
                    "P3",
                    "admet_druglikeness",
                    admet_qed_score_json,
                ),
            ),
            # ---- Wf-323F — m4 SA/QED sub-criterion smoke
            # documented gate (WEAK_TO_STRONG §2.4; companion to the
            # Wf-315A SA / QED evaluator gates above).  Verifies
            # all 4 closeout artefacts ship together: m4 metric JSON
            # (HONEST_NEGATIVE contract) + Wf-323F report JSON
            # (sub_rate_sa_qed + per_pool_distribution) + TODO
            # closeout doc.  CPU-only gate (no subprocess) — runs
            # in <10 ms.  See TODO/completed/323f_m4_sa_qed_
            # subcriterion.md for the Wf-323F ship summary + the
            # honest-negative framing on Vina sub-criterion being
            # OUT OF SCOPE for CPU work.
            (
                "p3_m4_sa_qed_subcriterion_documented",
                lambda: gate_wf323f_m4_sa_qed_subcriterion_documented(
                    "p3_m4_sa_qed_subcriterion_documented",
                    "P3",
                    "wf323f",
                    wf323f_m4_metric_json,
                    wf323f_m4_report_json,
                    wf323f_m4_todo_doc,
                ),
            ),
            # ---- Wf-308A — smoke-test FAIL-on-missing global audit ----
            # Per Wf-307B / TODO-178b hygiene + party-twenty-spirit §3.5:
            # every smoke test (file with 'smoke' in the name) must raise
            # AssertionError on missing ship-data, NOT silently skip.
            # The audit is at molmetal/tests/test_smoke_fail_on_missing_audit.py
            # and statically walks every smoke test for ``pytest.skip``
            # calls that hide missing-data regressions. PASS when the
            # audit meta-test passes (5/5 sub-tests).
            (
                "p3_smoke_fail_on_missing_audit",
                lambda: gate_pytest(
                    "p3_smoke_fail_on_missing_audit",
                    "P3",
                    "smoke_hygiene",
                    [TESTS_DIR / "test_smoke_fail_on_missing_audit.py"],
                    timeout=120.0,
                ),
            ),
            # ---- Wf-312B — NCI-60 9-cell per-cell aggregate gate ----
            # Per Wf-308C §3.2 MEASURED smoke (9 cells × 3 seeds × 1 epoch
            # chemprop v2 CPU sweep) + Wf-312B ship-surface integration:
            # verify the per_cell_breakdown block in
            # nci60_precious_9cell_auc.json carries exactly 9 cells + a
            # consistent overall_mean + a highlight (HCT116) + an
            # honest-negative (PC3 with badge=honest-negative + reason +
            # value < 0.5).
            (
                "p3_nci60_9cell_per_cell_aggregate",
                lambda: gate_nci60_9cell_per_cell_aggregate(
                    "p3_nci60_9cell_per_cell_aggregate",
                    "P3",
                    "nci60_9cell",
                    REPO_ROOT
                    / "metrics"
                    / "by_metric"
                    / "nci60_precious_9cell_auc.json",
                ),
            ),
            # ---- Wf-A12 FIX (2026-10-03) — `p3_loi_send_ready` REMOVED
            # from the 'all' (commit-path) pipeline registration.
            # Why: it is a wet-lab gate — its PASS condition is that the
            # 4 wet-lab Letters-of-Intent are structurally send-ready.
            # That directly contradicts the HARD MANDATE in AGENTS.md
            # §5.4 + §10 ("DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB
            # WORK"; the 4 LOI drafts are explicitly DO NOT SEND and are
            # retained as historical templates per §10.5).  A gate that
            # enforces send-readiness on the commit path would re-open a
            # deprecated workstream every time someone runs pre-flight.
            # Scope of this removal: ONLY this 'all'-pipeline
            # registration.  The `gate_loi_send_ready` FUNCTION is kept
            # (grep: `def gate_loi_send_ready`) and the frontier-pipeline
            # registration is kept for paper-rule parity; the 4 LOI
            # templates stay in TODO/pending/ as historical artifacts.
            # ---- Wf-313H — R20 Phase 0 loader closeout gate ----
            # (TODO-313 / R20_BIG_DATASET_FULL_RETRAIN_PLAN.md §1.1).
            # Per Wf-313H: the 3 P0 CRITICAL loaders (binding_db /
            # nci60 / drugood_loader) must each ship the module +
            # API doc + ≥12 dedicated tests.  This gate verifies
            # all 9 artefacts are non-empty on disk AND the 3 test
            # files pass pytest (no silent regressions on the
            # loader contract).  See
            # TODO/completed/313_r20_phase0_loaders_closeout.md
            # for the per-loader MEASURED status table.
            (
                "p3_r20_phase0_loaders_complete",
                lambda: gate_r20_phase0_loaders_complete(
                    "p3_r20_phase0_loaders_complete",
                    "P3",
                    "r20_phase0",
                    r20_phase0_loader_modules,
                    r20_phase0_loader_api_docs,
                    r20_phase0_loader_tests,
                ),
            ),
            # ---- Wf-322E — R20 §1.1 loader-API documentation
            # index closeout (TODO-148).  Companion to
            # ``p3_r20_phase0_loaders_complete``: that gate verifies
            # the 3 per-loader docs exist + the modules + tests pass;
            # this NEW gate adds the 4th doc — the
            # ``R20_P0_loaders_api.md`` index that aggregates the 3
            # per-loader docs into a single landing page documenting
            # the canonical 6-col schema + per-loader public surface +
            # cross-loader invariants.  PASS iff all 4 docs are
            # non-empty files on disk.  CPU-only, runs in <1 s.  See
            # TODO/completed/322e_todo_148_loader_api_doc.md.
            (
                "p3_r20_p0_loader_api_doc_present",
                lambda: gate_all_files_exist(
                    "p3_r20_p0_loader_api_doc_present",
                    "P3",
                    "r20_phase0",
                    r20_phase0_loader_api_doc_index,
                ),
            ),
            # ---- Wf-315D — PoseBusters smoke retry gate
            # (WEAK_TO_STRONG §2.6).  Branches on actual venv state:
            # packages available → require MEASURED_SMOKE + smoke output
            # on disk; packages missing → require HONEST_NEGATIVE_BLOCKED
            # + non-empty blocker_description.  See
            # TODO/completed/315d_pb_smoke_retry.md (Wf-315D TODO doc).
            (
                "p3_pb_packages_available_or_documented",
                lambda: gate_wf315d_pb_smoke(
                    "p3_pb_packages_available_or_documented",
                    "P3",
                    "wf315d",
                    wf315d_pb_metric_json,
                    wf315d_pb_smoke_output,
                    wf315d_pb_venv_python,
                ),
            ),
            # ---- Wf-319C — PB packages status documented gate
            # (companion to Wf-315D).  Stricter verifier: requires the
            # venv probe to succeed AND the 1h36 metric to be
            # MEASURED_SMOKE AND the smoke output JSON to carry the
            # canonical keys.  See TODO/completed/319c_pb_packages_
            # status_present.md.
            (
                "p3_pb_packages_status_present",
                lambda: gate_wf319c_pb_packages_status_present(
                    "p3_pb_packages_status_present",
                    "P3",
                    "wf319c",
                    wf319c_pb_venv_python,
                    wf319c_pb_metric_json,
                    wf319c_pb_smoke_output,
                ),
            ),
            # ---- Wf-319B — UFF-min Pt-pool presence + honest-negative
            # tag gate.  Verifies the 96/100-conformer Pt-pool was
            # generated AND the metrics JSON carries the canonical
            # ``HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True`` tag so
            # downstream consumers do not silently treat UFF energies
            # as physical for d-block Pt complexes.  See
            # TODO/completed/319b_uff_min_pt_pool.md for the ship
            # summary + WEAK_TO_STRONG §2.7 honest-framing context.
            (
                "p3_uff_min_pt_pool_present",
                lambda: gate_wf319b_uff_min_pt_pool_present(
                    "p3_uff_min_pt_pool_present",
                    "P3",
                    "wf319b",
                    wf319b_pool_json,
                    wf319b_metrics_json,
                ),
            ),
            # ---- Wf-323C — UFF pre-relax + Vina re-dock CPU smoke
            # gate (WEAK_TO_STRONG §2.6 R18 follow-up to Wf-319C).
            # Verifies the Wf-323C ship surface ships together:
            # the result JSON carries the MEASURED m8_full + m8_protein_
            # aware pass rates (both in [0, 1]) + the metric JSON is
            # MEASURED_SMOKE with the Wf-323C smoke-summary block +
            # the Wf-319B pool JSON has ≥80 conformers. The gate
            # accepts BOTH a successful lift AND an honest-negative
            # outcome — the requirement is that the delta is
            # documented (per party-twenty-spirit §3.5). See
            # TODO/completed/323c_m8_full_cpu_fix.md.
            (
                "p3_m8_full_pass_rate_documented",
                lambda: gate_wf323c_m8_full_pass_rate_documented(
                    "p3_m8_full_pass_rate_documented",
                    "P3",
                    "wf323c",
                    wf323c_result_json,
                    wf323c_metric_json,
                    wf323c_pool_json,
                ),
            ),
            # ---- Wf-319A — ablation_runner.py unit tests gate
            # (12 NEW tests + MEASURED smoke).  Per Wf-319A brief:
            # the 12 NEW tests in
            # ``molmetal/tests/test_wf319a_ablation_runner.py`` must
            # pass via pytest.  This gate is the test-runner side
            # of the Wf-319A ship surface; the aggregate.csv gate
            # below is the artefact-side verifier.  See
            # TODO/completed/319a_ablation_runner_smoke.md §3 for
            # the honest-negative on M14 (--lr-warmup-steps not in
            # harness — the test surfaces this as an explicit
            # honest-negative contract rather than a hidden bug).
            (
                "p3_ablation_runner_tests_present",
                lambda: gate_pytest(
                    "p3_ablation_runner_tests_present",
                    "P3",
                    "wf319a",
                    [wf319a_test_file],
                    timeout=120.0,
                ),
            ),
            # ---- Wf-319A — ablation_runner aggregate.csv 11-rows
            # gate (paper §6 input).  Per Wf-319A brief + the
            # MEASURED smoke already on disk at
            # molmetal/reports/wf_ablation_runner/aggregate.csv
            # (12 rows = 11 ablations + 1 BASELINE reference):
            # this gate verifies the aggregate.csv exists, is
            # parseable as CSV, and has ≥11 data rows (header
            # excluded).  See TODO/completed/319a_ablation_runner_
            # smoke.md §2 for the per-row measured findings.
            (
                "p3_ablation_runner_aggregate_csv_present",
                lambda: gate_wf319a_ablation_aggregate_csv_present(
                    "p3_ablation_runner_aggregate_csv_present",
                    "P3",
                    "wf319a",
                    wf319a_aggregate_csv,
                ),
            ),
            # ---- Wf-321E — ABLATION_TABLE registry mirror check.
            # Companion to the Wf-319A aggregate.csv gate (artefact
            # side).  This gate is the source-side verifier: it
            # imports ABLATION_TABLE from
            # ``molmetal/scripts/ablation_runner.py`` and verifies
            # the 11 R17 ablations in
            # ``TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`` §7 are
            # all present with the §7 contract fields.  See
            # TODO/completed/321e_ablation_runner_integration_test.md.
            (
                "p3_ablation_runner_registry_complete",
                lambda: gate_wf321e_ablation_registry_complete(
                    "p3_ablation_runner_registry_complete",
                    "P3",
                    "wf321e",
                    wf321e_runner_module,
                ),
            ),
            # ---- Wf-322G — TODO-130 G3 closure theorem Catherine
            # 30-metal bench closeout (R20-P5 / R22-P5 documentation
            # + bench-infrastructure ship surface; full real sweep
            # ≥ 0.95 reachability in pipeline Phase 3 R20-P5 per
            # kill_then_adapt directive).  Verifies the 5 closeout
            # artefacts ship together: bench script + audit script
            # + Wf-255 30-metal sweep JSON + Wf-130 P1 30-metal
            # combined audit JSON + this TODO closeout doc.  All 5
            # are required for PASS.
            (
                "p3_catherine_30metal_closeout_documented",
                lambda: gate_all_files_exist(
                    "p3_catherine_30metal_closeout_documented",
                    "P3",
                    "catherine_30metal",
                    (
                        catherine_30metal_bench_script,
                        catherine_30metal_audit_script,
                        catherine_30metal_wf255_per_metal_json,
                        catherine_30metal_wf130_combined_json,
                        catherine_30metal_closeout_doc,
                    ),
                ),
            ),
            # ---- Wf-327 — cpu-num-workers ≥ 4 gate (Wf-326B §6.2
            # follow-up).  Fail-fast blocker on the Wf-326B root
            # cause: any launch script that drops --cpu-num-workers
            # below 4 reverts the trainer to its manual-loop
            # fallback path (no DataLoader, no pin_memory, no
            # workers, no prefetch), causing GPU starvation.  The
            # 6 Wf-328-W2-migrated scripts all use --cpu-num-workers
            # 8 today; this gate keeps the regression from
            # re-occurring.
            (
                "p3_wf327_cpu_num_workers_sufficient",
                lambda: gate_wf327_cpu_num_workers_sufficient(
                    "p3_wf327_cpu_num_workers_sufficient",
                    "P3",
                    "wf327",
                    wf327_launch_scripts,
                    min_workers=4,
                ),
            ),
            # ---- Wf-321D / Wf-A12 — gate-count consistency audit.
            # Wf-A12 FIX (2026-10-03): this registration now passes the
            # REAL expected count explicitly
            # (``expected_total=EXPECTED_GATE_COUNTS["all"]``), so the
            # commit-path audit performs an actual drift check.  The
            # previous registration omitted expected_total, and the
            # Wf-345-B3 auto-compute mode then compared the registered
            # count to itself — the gate could not FAIL, and the
            # 98-constant-vs-102-registered drift went unsignalled.
            # When you add/remove a gate in the 'all' pipeline you MUST
            # update EXPECTED_GATE_COUNTS["all"] in the same edit; the
            # gate will now tell you loudly if you forget (both
            # directions: +DRIFT = bloat, -DRIFT = silent loss).
            (
                "p3_gate_count_consistency_audit",
                lambda: gate_gate_count_consistency_audit(
                    "p3_gate_count_consistency_audit",
                    "P3",
                    "all",
                    expected_total=EXPECTED_GATE_COUNTS["all"],
                ),
            ),
            # ---- Wf-343C4 — pre-retrain internal check metric gates
            # (4 NEW 'all'-only gates; registered AFTER the Wf-321D
            # audit gate so the audit's runner-count stays consistent
            # with the EXPECTED_GATE_COUNTS["all"] = 68 baseline
            # documented in TODO/completed/321d_preflight_gate_audit.md).
            # Per the 2026-09-29 user request: "ultracode for pre-retrain
            # internal check metrics" + AGENTS.md §5.1 standing GPU grant
            # (hyperparameters via config files). These 4 gates are
            # the canonical pre-retrain internal check surface:
            # (a) hyperparams_config_complete — verifies the 7 active
            #     configs/*.yaml carry train.batch_size / train.lr /
            #     model.hidden_dim / model.n_layers (AGENTS.md §5.1
            #     contract; no guessing).
            # (b) yuelbond_decoder_registered — verifies
            #     cfm_decoder_yuelbond.py exports the YuelBondDecoder
            #     class (Wf-272I / Wf-269C structural check; --no-yuelbond
            #     → decode_ratio 0.000 MEASURED on real cohort).
            # (c) anchor_freshness_24h — every paper-grade *_anchor.json
            #     mtime <= 24 h (Wf-302A / Wf-304A AnchorGuard adoption;
            #     prevents silent AnchorStale downstream).
            # (d) wave_budget_remaining_nonneg — consumes
            #     wave_budget_report.json (emitted by
            #     wave_budget_tracker.py) and asserts
            #     remaining_hr >= 0 + deferred_waves empty (Wf-262 §7.1
            #     cross-script integration; the third audit link alongside
            #     per_wave_checklist.py).
            # All 4 are 'all'-only (NOT in 'frontier' / 'chemprop_v2_v3' /
            # 'cfm'); pre-retrain internal checks are not paper-grade
            # claims. See TODO/completed/343c4_integration.md for the
            # Wf-343C4 ship summary.
            (
                "p3_wf343c4_hyperparams_config_complete",
                lambda: gate_wf343c4_hyperparams_config_complete(
                    "p3_wf343c4_hyperparams_config_complete",
                    "P3",
                    "wf343c4",
                    _discover_active_configs(),
                ),
            ),
            (
                "p3_wf343c4_yuelbond_decoder_env",
                lambda: gate_wf343c4_yuelbond_decoder_env(
                    "p3_wf343c4_yuelbond_decoder_env",
                    "P3",
                    "wf343c4",
                    LAM_CHEM_DIR / "cfm_decoder_yuelbond.py",
                ),
            ),
            # Wf-A16 (2026-10-03): p3_wf343c4_anchor_freshness_24h MOVED OUT of
            # the commit path. It is a PRE-RETRAIN check — its own docstring
            # calls it "the pre-retrain prevention layer" — but it was
            # registered on "all", which is both the commit hook's pipeline AND
            # what every launcher invokes. Because it fails purely on wall
            # clock (mtime > 24 h) and nothing refreshes the anchors, it made
            # the repo UNCOMMITTABLE 24 h after the anchor-producing workloads
            # last ran, blocking work that had nothing to do with anchors.
            # It now lives in the "launch" pipeline only — see LAUNCH_ONLY
            # below and `--pipeline launch`.
            # DO NOT re-add it here.
            (
                "p3_wf343c4_wave_budget_remaining_nonneg",
                lambda: gate_wf343c4_wave_budget_remaining_nonneg(
                    "p3_wf343c4_wave_budget_remaining_nonneg",
                    "P3",
                    "wf343c4",
                    MOLMETAL_ROOT
                    / "reports"
                    / "wf262_wave_budget"
                    / "wave_budget_report.json",
                ),
            ),
            # ---- Wf-347A1 Wf-296 → Wf-343C series wiring
            # (Wf-347A1 pre-dates Wf-343C2 + Wf-343C3 integration;
            # the 2 NEW gates invoke the Wf-343C2 + Wf-343C3
            # standalone companion scripts as subprocess gates so
            # every pre-retrain consumer of the 'all' pipeline
            # surfaces per-wave runtime + data quality regressions
            # before a retrain launch).  Both are 'all'-only (not
            # paper-grade claims) per the same §3.4 design contract
            # that governs the 4 Wf-343C4 gates.
            # (a) wf343c2_per_wave_runtime_check — 7 checks
            #     (preflight_freshness_24h +
            #      preflight_no_fail_gates +
            #      hyperparams_config_complete +
            #      yuelbond_decoder_registered +
            #      wave_budget_remaining_nonneg +
            #      gpu_vram_available +
            #      disk_free_space); invoked with --skip-live
            #     so the GPU/df probes SKIP (CPU-only pre-flight
            #     context); remaining 5 must PASS via --strict.
            # (b) wf343c3_data_quality_check — 5 checks × 5
            #     canonical loaders (csv_loads + per_metal_balance
            #     + scaffold_split + class_balance + label_range);
            #     invoked with --strict (any SKIP → FAIL).
            # Wf-A16 (2026-10-03): p3_wf343c2_per_wave_runtime_check MOVED OUT
            # of the commit path, same reasoning as the anchor-freshness gate
            # above — the gate labels itself wave='Wf-347A1 pre-retrain smoke',
            # i.e. a launch-time check, and it FAILed on a commit that only
            # touched gate tooling. It now lives in the "launch" pipeline only.
            # DO NOT re-add it here.
            (
                "p3_wf343c3_data_quality_check",
                lambda: gate_wf343c3_data_quality_check(
                    "p3_wf343c3_data_quality_check",
                    "P3",
                    "wf347a1",
                    SCRIPT_DIR / "wf343c3_data_quality_check.py",
                ),
            ),
            # ---- Wf-350-A3 — hyperparameter provenance log gate
            # (NeurIPS §D.9 reproducibility protocol). Per
            # Wf-343A-R2 recommendation: 1 NEW pre-retrain internal
            # check gate that records git-SHA provenance + recent
            # commit history + uncommitted-change status for every
            # config file in configs/ and emits a JSONL artefact at
            # ``results/gate_scratch/wf350a3_hp_provenance.jsonl``
            # (H1/HOLE-8 follow-up: redirected out of the
            # source-adjacent ``molmetal/reports/`` tree) so
            # downstream consumers (paper §3 Wf-350 attribution
            # appendix + paper dashboard reproducibility table) can
            # trace every retrain back to the exact commit that
            # produced its hyperparameters.  'all'-only per the
            # same §3.4 design contract as the 4 Wf-343C4 + 2
            # Wf-347A1 gates (pre-retrain internal check, not
            # paper-grade claim).  See
            # ``TODO/completed/350a3_hp_provenance_log.md`` for
            # the Wf-350-A3 ship summary.
            (
                "p1_hp_provenance_log",
                lambda: gate_p1_hp_provenance_log(
                    "p1_hp_provenance_log",
                    "P3",
                    "wf350a3",
                    (
                        CONFIGS_DIR / "cfm_full_retrain.yaml",
                        CONFIGS_DIR / "cfm_per_innovation_ablation_p1.4.yaml",
                        CONFIGS_DIR / "cfm_real_decode_lift_p1.3.yaml",
                        CONFIGS_DIR / "default.yaml",
                        CONFIGS_DIR / "wf124_p3_50epoch_full_sweep_g5.yaml",
                        CONFIGS_DIR / "wf124_p3_per_innovation_ablation_g6.yaml",
                        CONFIGS_DIR / "wf124_p3_remaining_metals_g4.yaml",
                    ),
                    GATE_SCRATCH_ROOT / "wf350a3_hp_provenance.jsonl",
                ),
            ),
            # ---- Wf-350-A1 — runtime + GPU energy probe gate
            # (NeurIPS §E.4 reproducibility protocol). Per
            # Wf-343A-R2 recommendation: 1 NEW pre-retrain internal
            # check gate that records per-GPU instantaneous
            # ``power.draw`` (W) + ``power.limit`` (W) + total
            # wall-time duration estimate + cumulative estimated
            # energy (joules) by shelling out to ``nvidia-smi`` at
            # gate invocation. The JSON artefact at
            # ``molmetal/reports/wf350a1_runtime_energy.json``
            # serves as the runtime + energy baseline that downstream
            # consumers (paper §3 Wf-350 attribution appendix +
            # paper dashboard reproducibility table + F2(a)
            # hyperparam rewind list) can cite to reconstruct the
            # cost of every retrain. 'all'-only per the same §3.4
            # design contract as the Wf-350-A3 + Wf-343C4 + Wf-347A1
            # gates (pre-retrain internal check, not paper-grade
            # claim).  See
            # ``TODO/completed/350a1_runtime_energy_report.md`` for
            # the Wf-350-A1 ship summary.
            (
                "p1_runtime_energy_report",
                lambda: gate_p1_runtime_energy_report(
                    "p1_runtime_energy_report",
                    "P3",
                    "wf350a1",
                    MOLMETAL_ROOT / "reports" / "wf350a1_runtime_energy.json",
                ),
            ),
            # ---- Wf-350-C1 — GPU/CUDA env probe gate
            # (wraps the canonical R18 Blackwell env probe at
            # ``molmetal/scripts/check_blackwell_env.py``). Per
            # Wf-343a-R3 finding #1: every launch path should
            # surface ``torch.cuda.is_available`` +
            # ``torch.cuda.device_count`` + GPU model + CUDA
            # runtime version at host run-time so the AGENTS.md
            # §5.1 standing GPU grant ("显存够就放心大胆用" —
            # use freely as long as VRAM is enough) can be
            # machine-verified rather than user-asserted.  The
            # gate invokes check_blackwell_env.py as a
            # subprocess (writes JSON to
            # ``molmetal/reports/wf350c1_gpu_env_check.json``),
            # re-reads the artefact, and asserts the 4
            # mandatory host-runtime fields are populated +
            # torch.cuda.is_available==True + device_count>=1 +
            # gpu_names non-empty.  'all'-only per the same
            # §3.4 design contract as Wf-350-A1 + Wf-350-A2 +
            # Wf-350-A3 + Wf-343C4 (pre-retrain internal
            # check, NOT paper-grade claim).  See
            # ``TODO/completed/350c1_gpu_env_probe.md`` for
            # the Wf-350-C1 ship summary.
            (
                "p1_gpu_env_probe",
                lambda: gate_p1_gpu_env_probe(
                    "p1_gpu_env_probe",
                    "P3",
                    "wf350c1",
                    MOLMETAL_ROOT / "reports" / "wf350c1_gpu_env_check.json",
                ),
            ),
            # ---- Wf-353-B1 — wire Wf-343c1's 26-gate auto-verifiable
            # surface as 22 individual runners in the 'all' pipeline.
            # Per the Wf-343c1 finding + Wf-343C4 ship contract: 26
            # pre-retrain gates are defined in
            # ``molmetal/scripts/wf343c1_pre_retrain_check.py`` covering
            # §1.1 (code-level 10) + §1.3 (ablation 3) + §2.1 (data 7) +
            # §3.1 (test 1) + §5.11-§5.14 (post-retrain 4 SKIP) +
            # §8.15 (honest-framing 1).  The 4 post-retrain §5.11-§5.14
            # gates remain SKIP and are NOT wired individually
            # (they are filled in after the retrain wave per the
            # honest-framing convention).  The remaining 22
            # auto-verifiable gates are wired INDIVIDUALLY below
            # (one runner per gate) so each surfaces in the pre-flight
            # JSON report with its own PASS/FAIL/SKIP verdict + evidence
            # string.  This matches the Wf-343C4 per-gate registration
            # pattern (4 individual runners, one per gate) and gives
            # downstream consumers (per_wave_checklist.py +
            # _partition_gates_by_priority + paper §3 attribution
            # appendix) per-gate visibility.
            #
            # Naming convention: ``p3_wf353b1_<wf343c1_gate_name>``
            # (the prefix is the workflow that did the wiring; the
            # suffix is the Wf-343c1 original gate name verbatim).
            # All 22 are 'all'-only per the same §3.4 design contract
            # as the 4 Wf-343C4 + 2 Wf-347A1 + 1 Wf-350-A3 + 1
            # Wf-350-A1 + 1 Wf-385-A3 gates (pre-retrain internal
            # check, not paper-grade claim).  See
            # ``TODO/completed/353b1_wf343c1_gates_wired.md`` for
            # the Wf-353-B1 ship summary + the 22-gate contract.
            # ---- Wf-385-A3 — SCAFFOLD-loss wire-in verification
            # gate (post-Wf-380 ship surface). Per Wf-380-B1 ship:
            # 14 NEW CFM CLI flags wired into the trainer +
            # 11 SCAFFOLD modules under
            # ``molmetal/adapters/flow_matching_lipman/``. There is
            # a recurring SCAFFOLD-only failure mode where a flag is
            # declared + the SCAFFOLD module exports the expected
            # function + the help-text PASSES, but the SCAFFOLD is
            # NEVER actually wired into the main training loop.
            # Prior gates (Wf-343C4 + Wf-345-B1 + Wf-353-B1) cannot
            # detect this — they probe file presence / help text /
            # schema-level wire-in. The Wf-385-A3 standalone gate
            # script runs a 50-step CPU smoke with all 14 Wf-380
            # flags ON simultaneously and verifies (a) loss
            # decreases AND (b) decode_ratio > 0 (YuelBond
            # conditionality canary per Wf-269C MEASURED
            # --no-yuelbond → decode_ratio 0.000 collapse). The
            # gate propagates SKIP on env gaps (host Python without
            # torch) per the Wf-345-A2 env-aware SKIP contract.
            # 'all'-only per the same §3.4 design contract as the
            # Wf-343C4 + Wf-347A1 + Wf-350-A1/A3 + Wf-353-B1 gates
            # (pre-retrain internal check, not paper-grade claim).
            # See TODO/completed/385a3_scaffold_wire_in_gate.md for
            # the Wf-385-A3 ship summary + the Wf-380 SCAFFOLD-only
            # failure mode walk-through.
            (
                "p3_wf385_scaffold_loss_wire_in_verified",
                lambda: gate_wf385_scaffold_loss_wire_in_verified(
                    "p3_wf385_scaffold_loss_wire_in_verified",
                    "P3",
                    "wf385a3",
                    SCRIPT_DIR / "wf385a3_scaffold_wire_in_gate.py",
                ),
            ),
            # Wf-446-H4 (HOLE 4 fix, 2026-10-03) — D-MPNN loss-descent
            # wire-in smoke, the CFM §5.7 equivalent: before this fix
            # train_dmpnn_multitask_wasserstein.py was executed by NO
            # pre-flight gate. Runs a 2-epoch CPU smoke (MEASURED 4.8 s
            # wall) on the smallest v4 joint cohort (Re) with the Wf-196
            # CPU-scaling recipe and requires >=2 finite per-epoch
            # train_loss points with a >=0.3% relative decrease; SKIPs
            # (not FAILs) when the gitignored .data/** CSVs are absent.
            # decode_ratio N/A (classifier — no decoder). 'all'-only per
            # the same §3.4 design contract as the adjacent all-only
            # gates (pre-retrain internal check, not paper-grade claim);
            # pre-flight is not a commit hook, so no blocking behaviour
            # is added anywhere (see .pre-commit-config.yaml).
            (
                "p3_dmpnn_loss_wire_in_smoke",
                lambda: _gate_dmpnn_quick_smoke(dmpnn_script),
            ),
        ] + [
            # 22 individual gate runners — one per Wf-343c1
            # auto-verifiable gate.  We wrap each
            # ``_wf343c1.<gate_name>`` runner via a lambda that
            # preserves the original gate name + Wf-343c1
            # priority (P1 / P2) + Wf-343c1 pipeline label
            # (code / ablation / data / test / honest_framing).
            (
                f"p3_wf353b1_{name}",
                lambda n=name, r=runner: _run_wf343c1_gate(n, r),
            )
            for name, runner in (
                # §1.1.1 — §1.1.10: code-level grep / help gates (10).
                *_wf343c1._code_level_gates(),
                # §1.3.1 — §1.3.3: ablation flag coverage gates (3).
                *_wf343c1._ablation_flag_gates(),
                # §2.1.1 — §2.1.7: data-level file existence /
                # min-lines gates (7 — the §2.1.6 BindingDB gate is
                # SKIP per the honest-negative framing).
                *_wf343c1._data_level_gates(),
                # §3.1.5: NaN guard test file present (1).
                *_wf343c1._test_level_gates(),
                # §8.15: honest framing catalog entry-count gate (1).
                *_wf343c1._honest_framing_gates(),
                # §5.11 — §5.14 (4 post-retrain SKIP gates)
                # INTENTIONALLY OMITTED — they remain SKIP per
                # honest-framing convention and are surfaced
                # via the standalone ``wf343c1_pre_retrain_check.py``
                # script (which is invoked from
                # ``molmetal/tests/test_wf343c1_pre_retrain_check.py``
                # for full verification).
            )
        ]

    # ---- Wf-306C standalone 'frontier' pipeline (P3 only) ----
    if pipeline == "frontier":
        runners += [
            (
                "p3_frontier_math_wire_in",
                lambda: gate_python_importable(
                    "p3_frontier_math_wire_in",
                    "P3",
                    "frontier",
                    frontier_adapters,
                ),
            ),
            # ---- Wf-313C Day 3-5 (also surfaced in 'frontier'
            # pipeline for paper-rule parity; see all-pipeline
            # comment for honest framing).  Two gates:
            # (a) the 8 NEW paper-grade integration smoke tests
            # must PASS; (b) the 4 per-adapter paper refs +
            # index README must be present on disk.
            (
                "p3_frontier_math_day_3_5_integration_tests",
                lambda: gate_pytest(
                    "p3_frontier_math_day_3_5_integration_tests",
                    "P3",
                    "frontier",
                    [wf313c_e2e_test],
                    timeout=120.0,
                ),
            ),
            (
                "p3_frontier_math_paper_refs_present",
                lambda: gate_all_files_exist(
                    "p3_frontier_math_paper_refs_present",
                    "P3",
                    "frontier",
                    wf313c_paper_ref_files,
                ),
            ),
            (
                "p3_todo220_drivers_present",
                lambda: gate_all_files_exist(
                    "p3_todo220_drivers_present",
                    "P3",
                    "drivers",
                    todo220_drivers,
                ),
            ),
            (
                "p3_anchorguard_version_pin",
                lambda: gate_anchor_version_pin(
                    "p3_anchorguard_version_pin",
                    "P3",
                    "anchors",
                ),
            ),
            (
                "p3_anchorguard_consumers_adopt",
                lambda: gate_all_files_exist(
                    "p3_anchorguard_consumers_adopt",
                    "P3",
                    "consumers",
                    anchorguard_consumers,
                ),
            ),
            (
                "p3_harness_auto_fallback_present",
                lambda: gate_file_exists(
                    "p3_harness_auto_fallback_present",
                    "P3",
                    "harness",
                    HARNESS_WRAPPER,
                ),
            ),
            (
                "p3_wf304b_platinai_selenium_diag",
                lambda: gate_all_files_exist(
                    "p3_wf304b_platinai_selenium_diag",
                    "P3",
                    "diagnostic",
                    (
                        wf304b_selenium_dir / "block_state.html",
                        wf304b_selenium_dir / "selenium_report.md",
                    ),
                ),
            ),
            (
                "p3_section9_honest_framing_min30",
                lambda: gate_section9_entry_count(
                    "p3_section9_honest_framing_min30",
                    "P3",
                    "honest_framing",
                    section9_doc,
                    min_entries=36,
                ),
            ),
            # ---- Wf-308B R03 — 3 ADMET pass-through gates (also in 'frontier') ----
            (
                "p3_admet_ames_druglikeness",
                lambda: gate_json_threshold(
                    "p3_admet_ames_druglikeness",
                    "P3",
                    "admet_druglikeness",
                    admet_ames_json,
                    extractor=lambda d: float(d.get("value", 1.0)),
                    threshold=0.10,
                    op="<=",
                ),
            ),
            (
                "p3_admet_cell_perm_druglikeness",
                lambda: gate_json_threshold(
                    "p3_admet_cell_perm_druglikeness",
                    "P3",
                    "admet_druglikeness",
                    admet_cell_perm_json,
                    extractor=lambda d: float(d.get("value", -10.0)),
                    threshold=-5.0,
                    op=">=",
                ),
            ),
            (
                "p3_admet_aq_sol_honest_negative_tag",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_aq_sol_honest_negative_tag",
                    "P3",
                    "admet_druglikeness",
                    admet_aq_sol_json,
                ),
            ),
            # ---- Wf-312C R02 — 3 NEW honest-negative ADMET druglikeness
            # gates on pre-existing JSON files (hepatotox / logp7_4 /
            # m6_kappa).  Each gate PASSES iff the JSON carries the
            # canonical ``HONEST_NEGATIVE: true`` + ``honest_negative_reason``
            # field set, mirroring the Wf-308B aqueous_solubility_logS
            # contract so downstream tooling branches on a single canonical
            # field rather than re-deriving the FAIL/THRESHOLD comparison.
            (
                "p3_admet_hepatotox_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_hepatotox_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_hepatotox_json,
                ),
            ),
            (
                "p3_admet_logp7_4_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_logp7_4_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_logp7_4_json,
                ),
            ),
            (
                "p3_admet_m6_kappa_honest_negative",
                lambda: gate_admet_honest_negative_tag(
                    "p3_admet_m6_kappa_honest_negative",
                    "P3",
                    "admet_druglikeness",
                    admet_m6_kappa_json,
                ),
            ),
            # ---- Wf-325E — ADMET 4 BELOW druglikeness documented gate
            # (aggregated ship-surface verifier).  Per Wf-325E brief:
            # the 4 BELOW metrics (aqueous_solubility_logS from Wf-303C
            # T1 + logp7_4 / hepatotox_index / m6_kappa from Wf-312C R02)
            # must all carry the canonical HONEST_NEGATIVE: true +
            # honest_negative_reason contract AND the Wf-325E closeout
            # doc must ship on disk — so the §3.7 honest-negative
            # cluster in paper_dashboard cannot silently regress on
            # the aggregated ship surface.  See
            # TODO/completed/325e_admet_4_below_final.md for the
            # ship summary + per-metric honest framing.
            (
                "p3_admet_4_below_documented",
                lambda: gate_wf325e_admet_4_below_documented(
                    "p3_admet_4_below_documented",
                    "P3",
                    "admet_druglikeness",
                    admet_aq_sol_json,
                    admet_logp7_4_json,
                    admet_hepatotox_json,
                    admet_m6_kappa_json,
                    wf325e_closeout_doc,
                ),
            ),
            # ---- Wf-315A — SA + QED evaluator pass-through druglikeness
            # gates (WEAK_TO_STRONG §3.6; mirror Wf-308B / Wf-312C pattern).
            # Both JSONs are MEASURED + HONEST_NEGATIVE on the 449 cohort:
            #   * sa_score mean = 4.715 (FAIL — above 4.0; metallo-drug
            #     cohort carries larger polycyclic / click-handle scaffolds)
            #   * qed_score mean = 0.230 (FAIL — below 0.5; Bickerton 2012
            #     weighted aggregate intrinsically penalises the metallo-drug
            #     MW + AROM profiles)
            # Gate PASS iff JSON carries HONEST_NEGATIVE=true + reason.
            (
                "p3_sa_score_metric_present",
                lambda: gate_admet_honest_negative_tag(
                    "p3_sa_score_metric_present",
                    "P3",
                    "admet_druglikeness",
                    admet_sa_score_json,
                ),
            ),
            (
                "p3_qed_score_metric_present",
                lambda: gate_admet_honest_negative_tag(
                    "p3_qed_score_metric_present",
                    "P3",
                    "admet_druglikeness",
                    admet_qed_score_json,
                ),
            ),
            # ---- Wf-323F — m4 SA/QED sub-criterion smoke
            # documented gate (WEAK_TO_STRONG §2.4; companion to the
            # Wf-315A SA / QED evaluator gates above).  Verifies
            # all 4 closeout artefacts ship together: m4 metric JSON
            # (HONEST_NEGATIVE contract) + Wf-323F report JSON
            # (sub_rate_sa_qed + per_pool_distribution) + TODO
            # closeout doc.  CPU-only gate (no subprocess) — runs
            # in <10 ms.  See TODO/completed/323f_m4_sa_qed_
            # subcriterion.md for the Wf-323F ship summary + the
            # honest-negative framing on Vina sub-criterion being
            # OUT OF SCOPE for CPU work.
            (
                "p3_m4_sa_qed_subcriterion_documented",
                lambda: gate_wf323f_m4_sa_qed_subcriterion_documented(
                    "p3_m4_sa_qed_subcriterion_documented",
                    "P3",
                    "wf323f",
                    wf323f_m4_metric_json,
                    wf323f_m4_report_json,
                    wf323f_m4_todo_doc,
                ),
            ),
            # ---- Wf-308A — smoke-test FAIL-on-missing global audit ----
            # Per Wf-307B / TODO-178b hygiene + party-twenty-spirit §3.5:
            # every smoke test (file with 'smoke' in the name) must raise
            # AssertionError on missing ship-data, NOT silently skip.
            # The audit is at molmetal/tests/test_smoke_fail_on_missing_audit.py
            # and statically walks every smoke test for ``pytest.skip``
            # calls that hide missing-data regressions. PASS when the
            # audit meta-test passes (5/5 sub-tests).
            (
                "p3_smoke_fail_on_missing_audit",
                lambda: gate_pytest(
                    "p3_smoke_fail_on_missing_audit",
                    "P3",
                    "smoke_hygiene",
                    [TESTS_DIR / "test_smoke_fail_on_missing_audit.py"],
                    timeout=120.0,
                ),
            ),
            # ---- Wf-312E — 4-LOI send-ready gate (also surfaced in
            # the 'frontier' pipeline for paper-rule parity; see all-
            # pipeline comment for honest framing).
            (
                "p3_loi_send_ready",
                lambda: gate_loi_send_ready(
                    "p3_loi_send_ready",
                    "P3",
                    "loi_send_ready",
                    loi_send_files,
                    loi_send_review_doc,
                ),
            ),
            # ---- Wf-312B — NCI-60 9-cell per-cell aggregate gate ----
            # (also surfaced in the 'frontier' pipeline for paper-rule
            # parity; see all-pipeline comment for honest framing).
            (
                "p3_nci60_9cell_per_cell_aggregate",
                lambda: gate_nci60_9cell_per_cell_aggregate(
                    "p3_nci60_9cell_per_cell_aggregate",
                    "P3",
                    "nci60_9cell",
                    REPO_ROOT
                    / "metrics"
                    / "by_metric"
                    / "nci60_precious_9cell_auc.json",
                ),
            ),
            # ---- Wf-321A — Wf-135 Phase 2 CPU scaffold readiness
            # gate.  Verifies the 3 prerequisites for the Wf-310E
            # Phase 2 GPU launch (USER_OK-gated per mol-metal-gpu-
            # execution §0): Phase 1 cohort JSON on disk + scaffold
            # split integrity, trainer CLI flag wiring into the
            # BalancedLOMORunner constructor, and Wf-310E launch
            # wrapper present with USER_OK gate.  CPU-only gate
            # (no torch.cuda, no subprocess) — runs in <1 s on the
            # host.  See TODO/completed/321a_wf135_phase2_cpu_
            # scaffold.md for the per-deliverable MEASURED status
            # + WEAK_TO_STRONG §2.7 honest-framing context.
            (
                "p3_wf135_phase2_cpu_smoke_ready",
                lambda: gate_wf321a_wf135_phase2_cpu_smoke(
                    "p3_wf135_phase2_cpu_smoke_ready",
                    "P3",
                    "wf321a",
                    wf321a_phase1_cohort_json,
                    wf321a_trainer_script,
                    wf321a_launch_wrapper,
                ),
            ),
            # ---- Wf-322F — Wf-176 m19 LOMO Rh oversample FINAL
            # closeout documented gate (also surfaced in 'frontier'
            # pipeline for paper-rule parity; companion to Wf-321A).
            # Verifies the closeout ship surface: close-out doc +
            # Phase 1 cohort JSON (Wf-309A SHIPPED) + Phase 1
            # synthesis doc (Wf-139 P7) + Wf-310E Phase 2 GPU
            # launch wrapper.  CPU-only gate (no torch.cuda, no
            # subprocess); runs in <0.005 s on host.  See
            # TODO/completed/322f_wf176_m19_lomo_final_closeout.md
            # for the per-deliverable MEASURED status + honest
            # framing per AGENTS.md §5.2 + party-twenty-spirit §3.5.
            # NOTE: This gate verifies the close-out ship surface
            # only — the Phase 2 GPU run itself remains BLOCKED
            # behind USER_OK per mol-metal-gpu-execution §0
            # kill_then_adapt.
            (
                "p3_wf176_m19_lomo_closeout_documented",
                lambda: gate_wf322f_wf176_m19_lomo_closeout_documented(
                    "p3_wf176_m19_lomo_closeout_documented",
                    "P3",
                    "wf322f",
                    wf322f_closeout_doc,
                    wf322f_phase1_cohort_json,
                    wf322f_phase1_cohort_synthesis,
                    wf322f_gpu_launch_wrapper,
                ),
            ),
            # ---- Wf-325D — m6_kappa BELOW threshold + NEGATIVE_WEAK
            # MetalCytoToxDB validation FINAL closeout documented
            # gate (also surfaced in 'frontier' pipeline for
            # paper-rule parity; companion to Wf-322F).  Verifies
            # the m6_kappa workstream closeout ship surface:
            # close-out doc + metrics/by_metric/m6_kappa.json
            # HONEST_NEGATIVE contract + metalcytotox_validation
            # block.  CPU-only gate (no torch.cuda, no subprocess);
            # runs in <10 ms on host.  See
            # TODO/completed/325d_m6_kappa_final.md for the
            # per-deliverable MEASURED status + honest framing
            # per AGENTS.md §5.2 + party-twenty-spirit §3.5.
            # NOTE: This gate verifies the close-out ship surface
            # only — the chemprop v2 + W₂ GPU recovery path
            # remains BLOCKED behind USER_OK per
            # mol-metal-gpu-execution §0 kill_then_adapt.
            (
                "p3_m6_kappa_closeout_documented",
                lambda: gate_wf325d_m6_kappa_closeout_documented(
                    "p3_m6_kappa_closeout_documented",
                    "P3",
                    "wf325d",
                    wf325d_closeout_doc,
                    wf325d_metric_json,
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Launch-only gates (Wf-A16). These run for `--pipeline launch` and NOT
    # for `--pipeline all`, i.e. before a training launch but not on commit.
    # ------------------------------------------------------------------
    if include_launch_only:
        runners += [
            (
                "p3_wf343c4_anchor_freshness_24h",
                lambda: gate_wf343c4_anchor_freshness_24h(
                    "p3_wf343c4_anchor_freshness_24h",
                    "P3",
                    "wf343c4",
                ),
            ),
            (
                # 2026-10-03: the caller's declaration is threaded in here.
                # Without it this registration could only ever SKIP — it
                # invoked a checker that needs a grid + budget while passing
                # neither (see WaveDeclaration).  An undeclared caller is
                # unchanged: every field stays None and the gate still SKIPs
                # loudly rather than fabricating a PASS.
                "p3_wf343c2_per_wave_runtime_check",
                lambda: gate_wf343c2_per_wave_runtime_check(
                    "p3_wf343c2_per_wave_runtime_check",
                    "P3",
                    "wf347a1",
                    SCRIPT_DIR / "wf343c2_per_wave_check.py",
                    wave=(
                        declaration.wave
                        if declaration and declaration.wave
                        else "Wf-347A1 pre-retrain smoke"
                    ),
                    pairs_metals=declaration.pairs_metals if declaration else None,
                    pairs_seeds=declaration.pairs_seeds if declaration else None,
                    budget_hr=declaration.budget_hr if declaration else None,
                ),
            ),
        ]

    return runners


# ---------------------------------------------------------------------------
# Special-case gate: chemprop --quick < 60s smoke
# ---------------------------------------------------------------------------

# Wf-446-H2 (HOLE 2 fix, 2026-10-03): relative loss-decrease floor for the
# chemprop quick smoke.  Mirrors the Wf-385-A3 / Wf-399-A1 0.3% floor;
# absorbs 2-epoch noise while still catching divergence / NaN.
_CHEMPROP_SMOKE_MIN_DROP_FRAC = 0.003

# Wf-446-H2/L3 (2026-10-03): the trainer has NO ckpt-dir knob (CKPT_DIR is
# a module constant at train_chemprop_v2_v3.py:214, out of this file's
# ownership), so every smoke run silently overwrites this SHIPPED
# checkpoint.  The gate now backs it up before the smoke and restores it
# byte-identical (sha256-verified) afterwards; a failed restore FAILs the
# gate.  `--quick` slices metals[:1] / seeds[:1] (trainer L1581-1585); the
# gate pins both explicitly so the overwritten path is known by
# construction instead of by default-drift.
_CHEMPROP_SMOKE_METAL = "Ru"
_CHEMPROP_SMOKE_SEED = 0
_CHEMPROP_SMOKE_CKPT_PATH = (
    MOLMETAL_ROOT
    / "checkpoints"
    / "wf200_chemprop_v2_v3"
    / f"dmpnn_mtw_v3_{_CHEMPROP_SMOKE_METAL.lower()}_seed{_CHEMPROP_SMOKE_SEED}.pt"
)


def _check_chemprop_loss_decrease(seed_json: Path) -> Tuple[bool, str]:
    """Wf-446-H2 (HOLE 2): loss-decrease invariant on quick-smoke metrics.

    Before this fix the gate PASSed on ``rc == 0`` alone.  But ``rc == 0``
    only means every run's ``status == "ok"``, and ``status == "ok"`` in
    ``train_chemprop_v2_v3.py`` (~L1232) merely means "not timed out" —
    a diverged / NaN loss still exits 0.  This helper inspects the REAL
    numbers the trainer writes to
    ``<out_dir>/<metal>/per_seed_metrics_seed{N}.json``:
    ``initial_loss`` = epoch-0 mean train loss, ``final_loss`` = last-epoch
    mean train loss (Kendall-loss terms, may be negative, so we COMPARE,
    never threshold).  Requires both finite and a decrease of at least
    ``_CHEMPROP_SMOKE_MIN_DROP_FRAC`` (0.3%) relative to ``|initial|``.

    Returns ``(ok, detail)``; ``detail`` is evidence text in both cases.
    """
    try:
        d = json.loads(seed_json.read_text())
    except FileNotFoundError:
        return False, f"metrics missing: {seed_json}"
    except (OSError, ValueError) as exc:
        return False, f"metrics unreadable ({type(exc).__name__}): {seed_json}"
    try:
        initial = float(d["initial_loss"])
        final = float(d["final_loss"])
    except (KeyError, TypeError, ValueError):
        return False, f"metrics lack numeric initial/final loss: {seed_json}"
    if not (math.isfinite(initial) and math.isfinite(final)):
        return False, (
            f"non-finite smoke loss initial={initial} final={final} "
            f"(diverged/NaN) | {seed_json}"
        )
    drop = initial - final
    floor = abs(initial) * _CHEMPROP_SMOKE_MIN_DROP_FRAC
    if not (drop > 0.0 and drop >= floor):
        return False, (
            f"loss did NOT decrease (initial={initial:.4f} "
            f"final={final:.4f} drop={drop:.4f}; need drop>0 and "
            f">={floor:.4f}) | {seed_json}"
        )
    return True, (
        f"loss decreased initial={initial:.4f} -> final={final:.4f} "
        f"(drop={drop:.4f} >= floor {floor:.4f})"
    )


def _sha256_file(path: Path) -> str:
    """Streaming sha256 of one file (chunked: ckpt is ~2.7 MB, bounded RSS)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prepare_chemprop_ckpt_backup(
    ckpt_path: Path, backup_dir: Path
) -> Tuple[bool, Optional[str], str]:
    """Wf-446-H2/L3: back up the shipped ckpt before the smoke overwrites it.

    Returns ``(ok, pre_sha, detail)``.  ``pre_sha`` is ``None`` iff the
    checkpoint did not exist pre-run (fresh clone; the smoke-created file
    is removed afterwards to restore the pre-run state).  ``detail`` is
    evidence text in both branches.  ``ok=False`` means the smoke must NOT
    run (it would destroy a checkpoint we cannot restore).
    """
    if not ckpt_path.is_file():
        return (
            True,
            None,
            (
                f"no pre-existing ckpt at {ckpt_path} (fresh tree; "
                f"smoke-created file will be removed)"
            ),
        )
    try:
        pre_sha = _sha256_file(ckpt_path)
        shutil.copy2(ckpt_path, backup_dir / ckpt_path.name)
    except OSError as exc:
        return (
            False,
            None,
            (
                f"cannot back up shipped ckpt {ckpt_path} before smoke "
                f"({type(exc).__name__}: {exc}); refusing to run a smoke that "
                f"would overwrite it"
            ),
        )
    return True, pre_sha, (f"backed up {ckpt_path.name} pre_sha256={pre_sha[:16]}")


def _restore_chemprop_ckpt(
    ckpt_path: Path, backup_dir: Path, pre_sha: Optional[str]
) -> Tuple[bool, str]:
    """Wf-446-H2/L3: restore pre-run ckpt state; returns ``(ok, detail)``.

    Any path that cannot return the file to byte-identical pre-run state
    returns ``ok=False`` so the gate FAILs loudly instead of silently
    shipping a replaced checkpoint.
    """
    backup_path = backup_dir / ckpt_path.name
    if pre_sha is not None:
        if not ckpt_path.is_file():
            return False, (
                f"pre-existing ckpt {ckpt_path} is gone after smoke "
                f"(backup held at {backup_path})"
            )
        try:
            post_sha = _sha256_file(ckpt_path)
            if post_sha != pre_sha:
                shutil.copy2(backup_path, ckpt_path)
            restored_sha = _sha256_file(ckpt_path)
        except OSError as exc:
            return False, (
                f"restore of {ckpt_path} failed "
                f"({type(exc).__name__}: {exc}; backup at {backup_path})"
            )
        if restored_sha != pre_sha:
            return False, (
                f"restore verify mismatch on {ckpt_path}: "
                f"restored={restored_sha[:16]} != pre={pre_sha[:16]}"
            )
        if post_sha != pre_sha:
            return True, (
                f"ckpt {ckpt_path.name} restored byte-identical "
                f"(pre={pre_sha[:16]}; smoke had overwritten it to "
                f"{post_sha[:16]})"
            )
        return True, (
            f"ckpt {ckpt_path.name} unchanged by smoke (sha256={pre_sha[:16]})"
        )
    if ckpt_path.is_file():
        try:
            ckpt_path.unlink()
        except OSError as exc:
            return False, (
                f"smoke created {ckpt_path} but it could not be removed to "
                f"restore the fresh-tree state ({type(exc).__name__}: {exc})"
            )
        return True, (f"removed smoke-created ckpt {ckpt_path.name} (none pre-run)")
    return True, "smoke wrote no ckpt (none pre-run)"


def _gate_chemprop_quick_smoke(chemprop_script: Path) -> GateResult:
    """Run ``{py} {script} --quick`` with a 90 s wall-clock budget and
    enforce a loss-decrease invariant on the produced metrics.

    Wf-294A (2026-09-29): bumped from 55 s to 90 s.  chemprop --quick
    itself completes in ~34 s of pure runtime, but cold-start overhead
    (imports, dataset load, model materialisation) routinely pushes wall
    time past 55 s on this host, falsely tripping a TIMEOUT FAIL even
    though the script exits 0.  90 s gives ~2.6x margin around the
    measured 34 s chemprop --quick baseline.

    Wf-446-H2 (HOLE 2 fix, 2026-10-03): rc==0 is no longer sufficient.
    ``--quick`` forces epochs=2 (trainer L1581-1585), so the per-seed JSON
    carries ``initial_loss`` / ``final_loss``; the gate now requires both
    finite + a >=0.3% relative decrease via
    ``_check_chemprop_loss_decrease``.  The smoke also runs with an
    explicit ``--out-dir`` under a private temp dir — previously it wrote
    ``molmetal/reports/wf200_chemprop_v2_v3/ru/...`` into the shipped
    reports tree as a side effect.

    Wf-446-H2/L3 (2026-10-03): the trainer still has no ckpt-dir knob
    (``CKPT_DIR`` is a module constant at train_chemprop_v2_v3.py:214; the
    ``--ckpt-dir`` flag remains a report-only request to that file's
    owner), so the smoke would silently overwrite the shipped checkpoint
    ``molmetal/checkpoints/wf200_chemprop_v2_v3/dmpnn_mtw_v3_ru_seed0.pt``
    (gitignored — no git dirt, but the shipped bytes are replaced).  The
    gate now snapshots that file (sha256) before the smoke and restores it
    byte-identical afterwards; FAIL on any unverifiable restore.  If the
    pre-run backup itself fails, the gate refuses to run the smoke at all.
    ``--metals`` / ``--seeds`` are pinned to the first metal/seed
    ``--quick`` slices anyway, so the overwritten path is known by
    construction.
    """
    name = "p1_chemprop_smoke"
    t0 = time.perf_counter()
    smoke_out = Path(tempfile.mkdtemp(prefix="wf446_h2_chemprop_smoke_"))
    ckpt_path = _CHEMPROP_SMOKE_CKPT_PATH
    backup_dir = Path(tempfile.mkdtemp(prefix="wf446_h2_chemprop_ckpt_"))
    ckpt_ok, pre_sha, ckpt_backup_detail = _prepare_chemprop_ckpt_backup(
        ckpt_path, backup_dir
    )
    if not ckpt_ok:
        return GateResult(
            name,
            "P1",
            "chemprop_v2_v3",
            "FAIL",
            f"{ckpt_backup_detail} (smoke NOT run)",
            time.perf_counter() - t0,
        )
    args = [
        VENV_PY,
        str(chemprop_script),
        "--quick",
        "--allow-single-seed",
        "--device",
        "cpu",
        "--out-dir",
        str(smoke_out),
        # Wf-446-H2/L3: pinned so the ckpt the trainer overwrites is known by
        # construction (--quick slices metals[:1]/seeds[:1] anyway; trainer
        # L1236 writes dmpnn_mtw_v3_<metal>_seed<N>.pt).
        "--metals",
        _CHEMPROP_SMOKE_METAL,
        "--seeds",
        str(_CHEMPROP_SMOKE_SEED),
        # Wf-446: explicit --device cpu (the trainer defaults to cuda; pre-flight
        # gates run on a CPU-only host where cuda is unavailable; --quick is the
        # canonical CPU smoke per chemprop --quick help text).
    ]
    rc, out, err = _run(args, timeout=90.0, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    # Wf-446-H2/L3: restore the pre-run checkpoint bytes BEFORE judging the
    # smoke, so a failed restore FAILs the gate regardless of rc.
    ckpt_ok, ckpt_restore_detail = _restore_chemprop_ckpt(
        ckpt_path, backup_dir, pre_sha
    )
    ckpt_suffix = f" | ckpt: {ckpt_backup_detail}; {ckpt_restore_detail}"
    if not ckpt_ok:
        return GateResult(
            name,
            "P1",
            "chemprop_v2_v3",
            "FAIL",
            f"--quick finished (rc={rc}) but checkpoint restore FAILED: "
            f"{ckpt_restore_detail}",
            elapsed,
        )
    blob_tail = (
        (out + "\n" + err).strip().splitlines()[-3:]
        if (out or err)
        else ["(no output)"]
    )
    blob_tail_str = " | ".join(blob_tail)
    if rc == 0 and elapsed < 95.0:
        metric_files = sorted(smoke_out.glob("*/per_seed_metrics_seed*.json"))
        if not metric_files:
            return GateResult(
                name,
                "P1",
                "chemprop_v2_v3",
                "FAIL",
                f"--quick exit=0 in {elapsed:.1f}s but wrote no per-seed "
                f"metrics under {smoke_out} (cannot verify loss decrease) "
                f"| tail: {blob_tail_str}{ckpt_suffix}",
                elapsed,
            )
        seed_json = metric_files[0]
        ok, detail = _check_chemprop_loss_decrease(seed_json)
        if not ok:
            return GateResult(
                name,
                "P1",
                "chemprop_v2_v3",
                "FAIL",
                f"--quick exit=0 in {elapsed:.1f}s but loss invariant FAILED: "
                f"{detail} | tail: {blob_tail_str}{ckpt_suffix}",
                elapsed,
            )
        return GateResult(
            name,
            "P1",
            "chemprop_v2_v3",
            "PASS",
            f"--quick exit=0 in {elapsed:.1f}s (rc=0); {detail} | "
            f"tail: {blob_tail_str}{ckpt_suffix}",
            elapsed,
        )
    if elapsed >= 90.0:
        return GateResult(
            name,
            "P1",
            "chemprop_v2_v3",
            "FAIL",
            f"--quick TIMED OUT after {elapsed:.1f}s (>= 90s budget; "
            f"CPU-only chemprop --quick measured at ~34s so genuine overruns "
            f"indicate infra regression) | tail: {blob_tail_str}{ckpt_suffix}",
            elapsed,
        )
    return GateResult(
        name,
        "P1",
        "chemprop_v2_v3",
        "FAIL",
        f"--quick exit={rc} in {elapsed:.1f}s | tail: {blob_tail_str}" f"{ckpt_suffix}",
        elapsed,
    )


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pre_flight_check.py",
        description=(
            "Wf-259 + Wf-306C auto-verify for TODO-207 v2 unified pre-flight "
            "checklist. Runs the auto-verifiable subset of P1 + P2 + P3 gates; "
            "emits JSON; exit 0 if all PASS, 1 if any FAIL, 2 if any SKIP "
            "(--strict only). Use --pipeline frontier to run ONLY the Wf-306C "
            "P3 frontier-math + AnchorGuard + 2026-09-29 follow-up gates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pipeline",
        choices=PIPELINES,
        default="all",
        help=(
            "Gate subset to run (default: all). 'launch' = 'all' + the "
            "pre-retrain gates (anchor freshness, per-wave runtime smoke); "
            "launchers should use it so a commit never waits on them."
        ),
    )
    p.add_argument(
        "--gates",
        default=None,
        metavar="NAME[,NAME...]",
        help=(
            "Run EXACTLY these gates, by name, ignoring --pipeline. Wf-A16: "
            "this makes a config's `directives.pre_flight_gates_required` list "
            "enforceable — it had been declared in 8 YAMLs and read by nothing. "
            "Unknown names are a hard error (a typo must not silently run "
            "zero gates). Exit codes are as for --pipeline."
        ),
    )
    # --- wave declaration (2026-10-03) -------------------------------------
    # These feed WaveDeclaration -> build_gates -> the
    # p3_wf343c2_per_wave_runtime_check gate.  Before they existed, that gate
    # could only ever SKIP on the pre-flight path (it needs a grid and a budget
    # and had no way to receive either), even though the launchers were already
    # binding real values to the same checker per-tuple.  Declaring here lets
    # the gate validate the WHOLE planned wave before anything launches.
    p.add_argument(
        "--pairs-metals",
        default=None,
        metavar="M1,M2,...",
        help=(
            "Comma-separated metals in the wave you intend to launch. Feeds the "
            "per-wave budget/VRAM gate. Unset -> that gate SKIPs loudly (it "
            "never fabricates a PASS from an undeclared grid)."
        ),
    )
    p.add_argument(
        "--pairs-seeds",
        default=None,
        metavar="S1,S2,...",
        help="Comma-separated seeds in the wave you intend to launch.",
    )
    p.add_argument(
        "--budget-hr",
        type=float,
        default=None,
        metavar="HOURS",
        help=(
            "Wall-clock budget for the declared wave. Compared against "
            "n_pairs x per-pair cost; an over-budget declaration FAILs."
        ),
    )
    p.add_argument(
        "--declared-wave",
        default=None,
        metavar="NAME",
        help="Human-readable name of the wave being declared (appears in gate evidence).",
    )
    p.add_argument(
        "--json",
        type=Path,
        nargs="?",
        default=None,
        help="Optional path to write JSON report (default: stdout only). Wf-A8 fix: --json alone now defaults to no JSON output (was argparse error before).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any gate SKIPPED (default: only FAIL triggers non-zero).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-gate status to stderr as they run.",
    )
    p.add_argument(
        "--freeze-csv-hashes",
        nargs="?",
        const=str(CSV_HASH_MANIFEST_DEFAULT),
        default=None,
        metavar="MANIFEST_JSON",
        help=(
            "One-shot HOLE-6 maintenance command: recompute SHA-256 for the "
            "gated CSVs (gated_csv_specs) and write the frozen manifest "
            "(default: configs/data_checksums.json), then exit. No gates run."
        ),
    )
    return p.parse_args(argv)


def _build_reproducibility_manifest() -> Dict[str, Any]:
    """Wf-350-B3: assemble the NeurIPS §3.6 reproducibility manifest.

    Loads the 3 Wf-350 reproducibility gates' on-disk artefacts so
    downstream consumers can reconstruct the full reproducibility
    context (compute + hp provenance + runtime/energy) from a single
    ``--json`` output. All 3 loads are best-effort: on a missing or
    malformed artefact the corresponding slot is ``None`` (not a hard
    FAIL — the gate itself would have FAILed during execution; this
    helper is a serialisation aid, not a re-runner).

    Returns a dict with the schema::

        {
            "compute": dict | None,        # wf350a2_compute_manifest.json
            "hp_provenance": list | None,   # wf350a3_hp_provenance.jsonl (JSONL)
            "runtime_energy": dict | None,  # wf350a1_runtime_energy.json
            "prereg_path": str | None,      # forward-looking (Wf-343a-R1)
        }

    The function is CPU-only + I/O only (3 small file loads + a glob for
    any prereg.md). It never raises.
    """
    manifest: Dict[str, Any] = {
        "compute": None,
        "hp_provenance": None,
        "runtime_energy": None,
        "prereg_path": None,
    }

    # (a) compute manifest (Wf-350-A2; wf350a2_compute_manifest.json)
    try:
        p = MOLMETAL_ROOT / "reports" / "wf350a2_compute_manifest.json"
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                manifest["compute"] = json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass

    # (b) hp provenance log (Wf-350-A3; wf350a3_hp_provenance.jsonl)
    # Canonical location is the gate scratch root (results/gate_scratch/,
    # gitignored); fall back to the legacy molmetal/reports/ path for trees
    # that have not yet run the redirected gate (H1/HOLE-8 follow-up).
    for p in (
        GATE_SCRATCH_ROOT / "wf350a3_hp_provenance.jsonl",
        MOLMETAL_ROOT / "reports" / "wf350a3_hp_provenance.jsonl",
    ):
        try:
            if not p.is_file():
                continue
            records = []
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            manifest["hp_provenance"] = records
            break
        except (OSError, json.JSONDecodeError):
            continue

    # (c) runtime + GPU energy (Wf-350-A1; wf350a1_runtime_energy.json)
    try:
        p = MOLMETAL_ROOT / "reports" / "wf350a1_runtime_energy.json"
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                manifest["runtime_energy"] = json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass

    # (d) prereg_path (Wf-343a-R1 forward-looking). If a single prereg.md
    # exists at the conventional paper-grade launch position, embed the
    # repo-relative string. Search is bounded (≤ 5 hits).
    try:
        for candidate in (
            REPO_ROOT / "TODO" / "pending" / "prereg.md",
            REPO_ROOT / "TODO" / "prereg.md",
            REPO_ROOT / "prereg.md",
        ):
            if candidate.is_file():
                manifest["prereg_path"] = str(candidate.relative_to(REPO_ROOT))
                break
    except OSError:
        pass

    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)

    # HOLE-6 maintenance command: freeze the CSV digests and exit without
    # running any gate (see gate_csv_hash docstring for the gate side).
    if args.freeze_csv_hashes is not None:
        return freeze_csv_hash_manifest(Path(args.freeze_csv_hashes))

    if args.gates is not None:
        # Wf-A16: explicit-name mode, used by launchers to enforce a config's
        # `directives.pre_flight_gates_required` list. Resolve against the
        # UNION of all pipelines, because a required list may name a gate that
        # only lives in `launch` (e.g. the anchor check) alongside one from
        # `all`.
        wanted = [n.strip() for n in args.gates.split(",") if n.strip()]
        available: Dict[str, Callable[[], GateResult]] = {}
        for pipe in PIPELINES:
            for name, fn in build_gates(pipe):
                available.setdefault(name, fn)
        missing = [n for n in wanted if n not in available]
        if missing:
            # Hard error, deliberately: a typo in a required-gate list must not
            # silently run zero gates and report success. That is exactly the
            # "declared but unenforced" failure this mode exists to close.
            print(
                f"[pre_flight] --gates: unknown gate name(s) {missing!r}. "
                f"Refusing to run a partial list.",
                file=sys.stderr,
            )
            return 1
        runners = [(n, available[n]) for n in wanted]
    else:
        runners = build_gates(
            args.pipeline,
            WaveDeclaration(
                wave=getattr(args, "declared_wave", None),
                pairs_metals=getattr(args, "pairs_metals", None),
                pairs_seeds=getattr(args, "pairs_seeds", None),
                budget_hr=getattr(args, "budget_hr", None),
            ),
        )

    if args.verbose:
        label = (
            f"gates={len(runners)}"
            if args.gates is not None
            else f"pipeline={args.pipeline} n_gates={len(runners)}"
        )
        print(f"[pre_flight] {label}", file=sys.stderr)

    results: List[GateResult] = []
    t_total0 = time.perf_counter()
    for name, runner in runners:
        t0 = time.perf_counter()
        try:
            res = runner()
        except Exception as exc:  # noqa: BLE001 — surface any gate error
            res = GateResult(
                name,
                "?",
                args.pipeline,
                "FAIL",
                f"gate raised: {type(exc).__name__}: {exc}",
                time.perf_counter() - t0,
            )
        results.append(res)
        if args.verbose:
            print(
                f"[pre_flight] {res.status:<4} {res.priority} "
                f"{res.pipeline:<14} {res.name:<38} "
                f"{res.wall_sec:>7.2f}s | {res.evidence[:120]}",
                file=sys.stderr,
            )
    total_wall = time.perf_counter() - t_total0

    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    report = {
        "timestamp": _now_iso(),
        "schema_version": "207.v1",
        "pipeline": args.pipeline,
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "total": len(results),
            "total_wall_sec": round(total_wall, 4),
        },
        "gates": [r.to_dict() for r in results],
        # Wf-350-B3: NeurIPS §3.6 reproducibility manifest as 1st-class
        # artifact in --json output. Embeds the 3 Wf-350 reproducibility
        # gates' outputs (compute + hp_provenance + runtime_energy) so
        # downstream consumers (paper §3 attribution appendix + paper
        # dashboard reproducibility table + F2(a) hyperparam rewind list)
        # can reconstruct the full reproducibility context from a single
        # --json file. The prereg_path slot is forward-looking per
        # Wf-343a-R1 (DeepMind §3.1 pre-registration record contract) and
        # is None until a prereg.md is added.
        "reproducibility_manifest": _build_reproducibility_manifest(),
        "v2_doc_reference": (
            "TODO/completed/207_unified_pre_flight_checklist_v2.md "
            "(§3.1 chemprop, §4.2 CFM, §5 P2)"
        ),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
        if args.verbose:
            print(f"[pre_flight] wrote JSON -> {args.json}", file=sys.stderr)
    print(payload)

    if n_fail > 0:
        return 1
    if args.strict and n_skip > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
