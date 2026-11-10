#!/usr/bin/env python3
"""_wrapper_audit.py — classifies every scripts/launch_wf*.sh wrapper as one of:

  - REAL:        USER_OK gate + DRY_RUN.log + plan.json emission + invokes an actual
                 trainer script (.venv/bin/python) via PYTHONPATH=REPO_ROOT
  - SCAFFOLD:    any wrapper that lacks the full 4-property canonical surface
                 (USER_OK + DRY_RUN.log + plan.json + canonical trainer call)
  - DUPLICATE:   target_overlap with another wrapper (detected via DOCK_PANEL
                 field emission inside the wrapper)

Per Wf-431-A2 FIX 2 (wrapper reclassification + single wrapper template) +
AGENTS.md §5.5 filesystem-truth + §5.6 atomic write + §5.9 hardcoded-PY rule.

Stdlib-only — no third-party deps. Reads every wrapper for lane >= 1, classifies
via regex matching against the canonical surface markers, emits structured JSON to
molmetal/reports/wrapper_audit.json, and returns non-zero exit code if any wrapper
violates the canonical surface (i.e. is SCAFFOLD or DUPLICATE).

Usage:
  python scripts/_wrapper_audit.py                 # default REPO_ROOT discovery
  python scripts/_wrapper_audit.py --strict       # exit 1 on any violation
  python scripts/_wrapper_audit.py --json-only    # console.log the JSON and exit 0

Cross-ref: scripts/_WRAPPER_TEMPLATE.sh is the canonical surface.

--------------------------------------------------------------------------
2026-11-04 (Wf-A269 wave-16) — the COVERAGE half of the §5.9 gate
--------------------------------------------------------------------------
The rendering above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6; the
CORRECTION is additive and dated.

THE DEFECT.  This gate enforced AGENTS.md §5.9 ("scripts/launch_*.sh MUST use
/home/hugo/codes/mol-metal/.venv/bin/python; NO bare python, NO
CadButEaas/kanzi_venv") over 31 files — ``scripts/launch_wf*.sh``.  The repo
holds 113 tracked ``.sh`` files, so 82 of them (72.6 %) were never looked at.
Three of them assign an interpreter that does not exist on this host
(``/home/hugo/codes/CadButEaas/.venv-vlm/bin/python``), and the gate could
not see them: its population was a strict subset of the sibling yaml-only
gate's, and it printed no denominator, so a reader could not tell 31/113 from
31/31.

THE REPAIR.  The population is now the SAME population the yaml-only gate
uses — every TRACKED ``*.sh`` in the repo minus an explicit, reasoned
exclusion list — and the gate PRINTS its own denominator (tracked total /
scanned / excluded / %).  The exclusion list is NOT duplicated here: it is
imported from the module that owns it (``scripts/_yaml_only_gate.py``), so the
two gates cannot drift apart (§6.269.13's class: a gate that cannot state its
denominator).

TWO SCOPES, ONE POPULATION.  The §5.9 INTERPRETER census runs over the WHOLE
population (99 files) — that is the coverage hole this closes.  The
REAL/SCAFFOLD/DUPLICATE CLASSIFICATION still runs over the launch wrappers
(``scripts/launch_wf*.sh``), because the seven canonical-surface markers are a
LAUNCH-WRAPPER contract, not a contract for e.g. ``env.sh``; reclassifying
every tracked shell file as a launch wrapper would manufacture violations
instead of closing a hole.

FAIL-CLOSED.  A degraded population (git unavailable, or the shared module
unimportable because a concurrent edit left it mid-syntax-error) cannot
certify coverage, so the verdict is FAIL and the exit code is non-zero even
when no wrapper violates the canonical surface.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
AUDIT_JSON = REPORTS_DIR / "wrapper_audit.json"

# canonical surface markers — must all be present for REAL classification.
# Patterns tuned to the REAL wrappers on disk (Wf-355-B1 USER_OK-gate family):
#   - `if [ "${USER_OK:-0}" != "1" ]` is the Wf-294B gate
#   - `PYTHONPATH=/home/hugo/codes/mol-metal ${PY}` is the Wf-431-A1 fix pattern
#   - `PY=/home/hugo/codes/mol-metal/.venv/bin/python` is the §5.9 hardcoded rule
# `export PYTHONPATH` is an acceptable alternative (catches wrappers that do
# `export PYTHONPATH=...` once at the top instead of inline).
CANONICAL_MARKERS = {
    "USER_OK gate": re.compile(r'USER_OK:-0}.*\s*!=\s*"1"', re.IGNORECASE),
    "DRY_RUN emission": re.compile(r"DRY_RUN\.log", re.IGNORECASE),
    "plan_json emission": re.compile(r"plan\.json", re.IGNORECASE),
    "pre_flight_check gate": re.compile(r"pre_flight_check\.py", re.IGNORECASE),
    "PYTHONPATH export": re.compile(
        r'PYTHONPATH\s*=\s*[/"\']?/home/hugo/codes/mol-metal(?:\s|\$|\")', re.IGNORECASE
    ),
    "PY hardcoded-via-bin/python": re.compile(
        r"(?:PY|py|python)[: =].{0,40}/home/hugo/codes/mol-metal/\.venv/bin/python",
        re.IGNORECASE,
    ),
    # NOTE (2026-10-08): this marker is satisfied by EITHER a `--user-ok` flag
    # literal OR a wrapper-level `USER_OK:-0} != "1"` env gate (see surface_check).
    # No trainer on disk declares `--user-ok` (grep verified 0 hits across
    # train_cfm_yuelbond_wasserstein / train_metal_lomo_v2 /
    # train_dmpnn_multitask_wasserstein / wf219_krasnov_aligned), so requiring the
    # literal flagged env-gated wrappers that are correctly gated at the env layer.
    "USER_OK env propagation to trainer": re.compile(r"--user-ok", re.IGNORECASE),
}
# An env-gated wrapper (Wf-294B family) satisfies the "propagation" marker even
# without a --user-ok flag, because the trainer reads no such flag.
_USER_OK_ENV_GATE_RE = re.compile(r'USER_OK:-0}.*\s*!=\s*"1"', re.IGNORECASE)

# DUPLICATE detector — DOCK_PANEL references the same trainer
DOCK_PANEL_RE = re.compile(r'DOCK_PANEL|TRAINER\s*=\s*"?([\w_./-]+\.py)"?')


def discover_wrappers(scripts_dir: Path) -> list[Path]:
    """List every launch_wf*.sh wrapper on disk."""
    return sorted(scripts_dir.glob("launch_wf*.sh"))


def surface_check(path: Path) -> dict[str, bool]:
    """Check which canonical markers are present in a wrapper."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return {name: False for name in CANONICAL_MARKERS} | {"_read_error": str(exc)}
    result = {
        name: bool(pattern.search(content))
        for name, pattern in CANONICAL_MARKERS.items()
    }
    # A correctly env-gated wrapper (Wf-294B `USER_OK:-0} != "1"`) satisfies the
    # trainer-propagation marker: no trainer on disk consumes `--user-ok`, so the
    # env gate IS the canonical propagation. Prevents a false-positive that would
    # gate CI on an unsatisfiable condition.
    if _USER_OK_ENV_GATE_RE.search(content):
        result["USER_OK env propagation to trainer"] = True
    # exclude the template from classification noise (it's the reference itself)
    result["_template"] = path.name.startswith("_")
    return result


def dock_panel(path: Path) -> str:
    """Extract DOCK_PANEL or TRAINER__name reference from a wrapper."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    m = DOCK_PANEL_RE.search(content)
    if m:
        # group(1) captures the explicit TRAINER value; if absent (DOCK_PANEL only), return ""
        return m.group(1) if m.group(1) else "<DOCK_PANEL-placeholder>"
    # Fall back: any .py reference in the trainer section
    m2 = re.search(r'(?:TRAINER|--trainer)\s+["\']?([\w_./-]+\.py)', content)
    return m2.group(1) if m2 else ""


# Wf-A269 wave-5 (2026-11-02) — REPAIR of the DUPLICATE classifier.
#
# MEASURED root cause of the false positive: the old rule counted RAW-TEXT
# occurrences of ``--device <word>`` and flagged anything with more than one.
# scripts/launch_wf310c_todo220_gpu.sh has TWO --device flags — but they belong
# to TWO DISTINCT trainers (molmetal/scripts/wf219_temporal_split_driver.py at
# the Stage-A invocation and molmetal/scripts/nci60_9cell_chemprop_driver.py at
# the Stage-B invocation), and both carry the SAME device value.  That is the
# opposite of the violation the rule is looking for.  The wave-4 hardening made
# it worse: the launcher added PROSE ``--device cuda`` inside its own
# DRY_RUN.log / plan.json heredocs, so the raw count went 2 -> 6 while the
# number of real trainer invocations stayed at 2.
#
# MEASURED second defect: the old regex ``--device\s+\w+`` (raw spelling:
# r"--device\s+\w+") cannot see the
# ``--device=<value>`` spelling at all (the ``=`` is not whitespace), so a
# wrapper carrying BOTH ``--device=cuda:0`` and ``--device=cuda:1`` was
# invisible to the gate.  Six wrappers on disk use the ``=`` spelling.
#
# CORRECTION (2026-11-02, Wf-A269 wave-5 main thread): the sentence above is
# FALSE -- EIGHT wrappers on disk use the ``=`` spelling, not six.  MEASURED
# (both trees agree, so this is not wave-5 drift): the 8 are
#   scripts/launch_wf131_p3_deferred_runs.sh
#   scripts/launch_wf146_deferred_runs.sh
#   scripts/launch_wf299_cfm_main.sh
#   scripts/launch_wf308c_todo220_gpu.sh
#   scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh
#   scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
#   scripts/launch_wf310e_wf135_phase2_gpu.sh
#   scripts/launch_wf310e_wf219_production_gpu.sh
# (`git show :scripts/<name>` for each, and the live file, both match.)
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   # invisible to the gate.  Six wrappers on disk use the ``=`` spelling.
#
# THE REPAIRED RULE (what the code below implements):
#   A wrapper violates the canonical "one resolver, one device" pattern
#   (repo commit 87b026ee — "one resolver, one device — everything targets
#   GPU 0") when it names MORE THAN ONE DISTINCT GPU DEVICE.
#     - BOTH spellings are matched: ``--device X`` and ``--device=X``.
#     - values are NORMALISED: surrounding quotes and trailing JSON/shell
#       punctuation are stripped; a shell variable reference is an opaque
#       symbolic device (``var:GPU``), so ``"${GPU}"`` and ``${GPU}`` are
#       the same device.
#     - a ``cpu`` value is EXEMPT: every wrapper on disk that carries one
#       uses it in an explicitly-CPU branch (a CPU-only scaffold smoke, or the
#       ``if [ "${gpu}" != "cpu" ]`` fallback) and never as a second GPU
#       target.
#
# COVERAGE, disclosed rather than hidden (AGENTS.md section 5.2):
#   - The rule is WRAPPER-LEVEL.  Two different invocation contexts that both
#     target GPU devices (e.g. a GPU smoke on cuda:0 and the real run on
#     cuda:1) are NOT distinguished — such a wrapper IS flagged.  That is the
#     FAIL-SAFE direction.
#   - The ``cpu`` exemption means a wrapper that names ``cpu`` AND exactly one
#     GPU device is NOT flagged, even if the cpu occurrence targets the same
#     trainer.  This is an UNDER-FLAG hole, admitted on purpose: the three
#     ``cpu`` occurrences on disk (launch_wf125_p1_controlled_pair.sh,
#     launch_wf146_deferred_runs.sh, launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh)
#     are all legitimately CPU-only branches, and flagging them would be a
#     regression in the gate's own pass rate.
# Wf-A269 wave-5 (2026-11-02) - CORRECTION: the value group's delimiter set
# now ALSO excludes the BACKTICK.
#
# MEASURED DEFECT (found by the main thread on 2026-11-02, immediately after
# repairing the todo220 launcher): the delimiter set excluded whitespace,
# quotes, comma, semicolon, `)`, `]` and `}`, but NOT the backtick.  A
# markdown-quoted device value -- `--device cuda` written inside a shell
# comment or a Python docstring -- was therefore captured as the DIFFERENT
# token "cuda`" (with the trailing backtick), and `_normalize_device_value`
# does not strip a trailing backtick.  One device (`cuda`) plus its own
# prose echo (`cuda` + backtick) counted as TWO DISTINCT devices, so a
# wrapper naming one device was reported DUPLICATE.  The rule's contract
# says "True iff the wrapper names > 1 distinct GPU DEVICE"; prose that
# names the SAME device does not violate it, so this is an IMPLEMENTATION
# defect, not a change to the standard.
#
# MEASURED CLASS: the backtick-quoted spelling occurs on 67 tracked .sh/.py
# lines in this repo (the dominant prose form is the double-backtick
# ``--device cuda`` used in docstrings).  It had been LATENT for the
# wave-5 rule because no `scripts/launch_wf*.sh` wrapper quoted the flag in
# backticks until the wave-5 CORRECTION note on the todo220 launcher did.
#
# CORRECTION (2026-11-02, the main thread, re-measured after the repair):
# the "67 tracked .sh/.py lines" figure above is NOT REPRODUCIBLE and is
# superseded by the measured table below.  The class is real and BROAD, but
# its size depends on how "the backtick-quoted spelling" is defined (the
# PROSE class) rather than on the DEFECT class (a captured device value
# that ENDS with a backtick).  The figure above was recorded without a
# reproducible command, so per AGENTS.md section 5.2 it could not be
# re-derived and is not a durable locator.
#
#   universe                        lines scanned   OLD captures a backtick-terminated value
#   all tracked files                 6 869 022     136 lines (144 matches) / 87 files
#   tracked .sh / .py                 1 228 168      52 lines ( 53 matches) / 28 files
#   the 31 gate-scanned wrappers         12 657       0 lines (  0 matches) /  0 files
#
# REPRODUCTION (match counts; the line counts come from a line-wise sweep of
# the same two regexes over the same universes):
#
#   OLD='--device(?:=|\s+)(?:"[^"]*"|'"'"'[^'"'"']*'"'"'|[^\s"'"'"',;)\]}]+)'
#   git grep -a -P -o -- "$OLD" -- '*.sh' '*.py' | grep -c '`$'          # -> 53
#   git grep -a -P -o -- "$OLD"                | grep -c '`$'          # -> 144
#   git grep -a -P -o -- "$OLD" -- 'scripts/launch_wf*.sh' | grep -c '`$' # -> 0
#
# The gate's own universe is listed LAST on purpose: it is the COVERAGE
# BOUND.  The defect had ZERO live instances inside the universe the gate
# scans -- which is precisely why it was LATENT, and why the wave-5
# CORRECTION NOTE on the todo220 launcher (the first wrapper to quote the
# flag in backticks) was the thing that tripped it.
#
# The "67" rendering above is PRESERVED VERBATIM per AGENTS.md section 5.5
# rule #6.
#
# THE FIX is ONE delimiter: the backtick is excluded, so a backtick-quoted
# device value captures as the device itself (`cuda`) and prose that names
# the SAME device no longer fabricates a second one.  No device value is
# ever legitimately written with a trailing backtick.
#
# COVERAGE LIMIT, disclosed rather than hidden (AGENTS.md section 5.2): a
# device value supplied by COMMAND SUBSTITUTION (``--device `nvidia-smi
# ...` ``) is now not counted at all.  That spelling is pathological -- no
# wrapper on disk uses it, MEASURED -- and it was previously counted as an
# opaque token, so this is a knowingly-accepted trade of one uncovered
# form for the removal of the false-positive class above.
#
# SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
#   DEVICE_FLAG_RE = re.compile(r"""--device(?:=|\s+)("[^"]*"|'[^']*'|[^\s"',;)\]}]+)""")
DEVICE_FLAG_RE = re.compile(r"""--device(?:=|\s+)("[^"]*"|'[^']*'|[^\s"',;)\]}`]+)""")


def _normalize_device_value(raw: str) -> str:
    """Normalise one captured --device value to a comparable token."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    value = value.rstrip(",;)]}\"'")
    match = re.fullmatch(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", value)
    if match:
        return "var:" + match.group(1)
    return value


#: A shell assignment of a PLAIN literal: ``NAME=value`` / ``NAME="value"``.
#: Command substitutions (``$(...)``), parameter expansions (``${...}``) and
#: backticks are NOT literals and are skipped, so the map only ever holds
#: values the file itself states.
_SHELL_ASSIGN_RE = re.compile(
    r"""^[ \t]*(?:export[ \t]+|declare[ \t]+(?:-[a-zA-Z]+[ \t]+)?)?([A-Za-z_][A-Za-z0-9_]*)="""
    r"""["']?([^"' \t#;]+)["']?[ \t]*(?:#.*)?$"""
)


def _shell_var_literals(content: str) -> dict[str, set[str]]:
    """``VAR`` -> the PLAIN literal values this file assigns to it.

    Wf-A269 wave-19 (2026-11-04).  The device rule compared the captured
    ``--device`` values as opaque tokens, so ``--device "${GPU}"`` (the
    command line) and ``--device cuda:0`` (the file's own prose about the
    same device) counted as TWO devices.  MEASURED on
    ``scripts/launch_wf310b_g6_final.sh``: the tokens were
    ``['var:GPU', 'cuda:0', 'var:GPU']`` and the wrapper names ONE device
    (``GPU="cuda:0"``, line 53).

    The fix is to RESOLVE the variable from the file's own assignment rather
    than to GUESS that it differs -- the same "decide where you can decide"
    principle as the yaml-only gate's LOOKAHEAD (§6.269.28.5).  A variable
    with NO plain-literal assignment stays unresolved (``var:NAME``), so the
    rule remains fail-closed: two DIFFERENT unresolved variables are still
    two distinct devices.
    """
    out: dict[str, set[str]] = {}
    for line in content.splitlines():
        match = _SHELL_ASSIGN_RE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        if "$" in value or "`" in value:
            # Not a plain literal (command substitution / parameter
            # expansion) -- the file does not STATE the value.
            continue
        out.setdefault(name, set()).add(value)
    return out


def _resolved_device_flag_values(path: Path) -> list[str]:
    """Every ``--device`` value, with RESOLVABLE shell variables expanded.

    A variable that the file assigns a plain literal for expands to that
    literal; a variable the file does not assign (or assigns only through a
    substitution) keeps its ``var:NAME`` token.  A variable assigned MORE
    THAN ONE literal expands to all of them -- the wrapper really does name
    two devices in that case.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []
    env = _shell_var_literals(content)
    out: list[str] = []
    for raw in DEVICE_FLAG_RE.findall(content):
        token = _normalize_device_value(raw)
        if token.startswith("var:"):
            literals = env.get(token[4:])
            if literals:
                out.extend(sorted(literals))
                continue
        out.append(token)
    return out


def device_flag_values(path: Path) -> list[str]:
    """Every --device value named by the wrapper, normalised, in file order."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []
    return [_normalize_device_value(m) for m in DEVICE_FLAG_RE.findall(content)]


def has_multiple_device_flags(path: Path) -> bool:
    r'''CANONICAL flag detection: True iff the wrapper names > 1 distinct GPU device.

    Wf-A269 wave-5 (2026-11-02) — CORRECTION.  The previous rule counted
    RAW-TEXT occurrences of ``--device <word>`` and flagged anything with more
    than one, which produced a MEASURED false positive on
    scripts/launch_wf310c_todo220_gpu.sh (two DISTINCT trainers, same device
    value) and could not see the ``--device=<value>`` spelling at all.  The
    rule is now "more than one distinct GPU device" — see the block comment
    above DEVICE_FLAG_RE for the full derivation, the coverage limits, and the
    admitted ``cpu`` exemption.

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
      def has_multiple_device_flags(path: Path) -> bool:
          """CANONICAL flag detection: return True if wrapper has > 1 --device flag invocation.

          Per AGENTS.md §5.5 filesystem-truth + Wf-A2 wrapper consolidation: a CANONICAL
          wrapper has exactly ONE --device flag invocation (the canonical entry-point).
          Multiple --device flags indicate DUPLICATE pattern (e.g. inline + for-loop).
          """
          try:
              content = path.read_text(encoding="utf-8")
          except Exception:
              return False
          return len(re.findall(r"--device\s+\w+", content)) > 1

    Wf-A269 wave-19 (2026-11-04) — CORRECTION, the SECOND false-positive
    class.  The wave-5 rule compared the captured values as OPAQUE tokens,
    so a shell variable was a device name in its own right: a wrapper that
    names ONE device twice (once through the variable, once in prose about
    that same device) counted as TWO devices.  MEASURED on
    scripts/launch_wf310b_g6_final.sh: the tokens were
    ['var:GPU', 'cuda:0', 'var:GPU'] while the file assigns GPU="cuda:0"
    — ONE device.  The rule now RESOLVES a variable from the file's own
    plain-literal assignment instead of GUESSING that it differs; an
    unresolvable variable keeps its opaque token, so two DIFFERENT
    unresolved variables are still two devices (fail-closed preserved).
    '''
    gpu_values = {
        value for value in _resolved_device_flag_values(path) if value != "cpu"
    }
    return len(gpu_values) > 1


def classify(surface: dict[str, bool], wrapper_path: Path) -> tuple[str, str]:
    """Classify a wrapper based on its canonical surface coverage.

    Returns (classification, evidence).

    Per Wf-A2 ADR docs-only sync: extend to factor in CANONICAL flag detection
    (multiple --device flags = DUPLICATE pattern per AGENTS.md §5.5 filesystem-truth).
    """
    if surface.get("_read_error"):
        return "DUPLICATE", f"read_error: {surface['_read_error']}"
    matched = [k for k, v in surface.items() if v and k != "_template"]
    missing = [k for k, v in surface.items() if not v and k != "_template"]
    if has_multiple_device_flags(wrapper_path):
        # Wf-A269 wave-5 (2026-11-02) — the evidence string now names the
        # REPAIRED rule (distinct GPU devices), not the old raw-occurrence
        # count.  The old string is preserved verbatim below.
        # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
        #   "multiple --device flag invocations (>1) — CANONICAL flag pattern violated",
        return (
            "DUPLICATE",
            "more than one distinct GPU device named — CANONICAL flag pattern "
            "violated (one resolver, one device)",
        )
    if len(matched) == len(CANONICAL_MARKERS):
        return "REAL", "all 7 canonical markers present"
    if surface.get("_template"):
        return "DUAL: TEMPLATE", "template file (excluded from classification)"
    return "SCAFFOLD", f"missing_markers: {",".join(missing)}"


def collect_duplicates(wrappers: list[Path]) -> dict[str, str]:
    """Map dock_panel → wrapper name (primary collision key)."""
    panels: dict[str, str] = {}
    for w in wrappers:
        if w.name.startswith("_"):
            continue
        panel = dock_panel(w)
        if not panel or panel.startswith("<"):
            continue
        if panel in panels:
            panels[panel] = panels[panel] + " + " + w.name
        else:
            panels[panel] = w.name
    return {k: v for k, v in panels.items() if " + " in v}


# ---------------------------------------------------------------------------
# §5.9 interpreter census — the COVERAGE half (2026-11-04, Wf-A269 wave-16)
# ---------------------------------------------------------------------------
# The §5.9 rule ("`scripts/launch_*.sh` MUST use
# /home/hugo/codes/mol-metal/.venv/bin/python; NO bare python, NO
# CadButEaas/kanzi_venv") was enforced over `scripts/launch_wf*.sh` ONLY —
# 31 of the repo's 113 tracked ``.sh`` files.  This section widens the
# INTERPRETER census to the SAME population the sibling yaml-only gate uses
# (every tracked ``.sh`` minus an explicit, reasoned exclusion list) and
# PRINTS the denominator.
#
# The exclusion list is NOT duplicated here.  It is IMPORTED from the module
# that owns it (``scripts/_yaml_only_gate.py``), so the two gates can never
# disagree about what "the population" is.  A gate that cannot state its
# denominator cannot be reviewed (§6.269.13), and a gate that cannot compute
# its population at all must FAIL rather than silently certify a subset.

CANONICAL_VENV = "/home/hugo/codes/mol-metal/.venv"
CANONICAL_PY = CANONICAL_VENV + "/bin/python"

# §5.9 names these two forbidden environments explicitly.
FORBIDDEN_VENV_NAMES = ("CadButEaas", "kanzi")

# The module that owns the population, and the symbols this audit consumes
# from it.  Reported on every run so a reviewer can see the provenance.
_YAML_GATE_PATH = SCRIPTS_DIR / "_yaml_only_gate.py"
_YAML_GATE_SYMBOLS = (
    "_population",
    "_exclusion_for",
    "_git_tracked_sh_paths",
    "EXCLUDED_WRAPPER_DIRS",
    "EXCLUDED_WRAPPER_FILES",
)

# A LITERAL absolute interpreter path, e.g. /home/hugo/x/.venv/bin/python.
# The lookbehind rejects a path that continues a variable expansion
# ("${VAR}/.venv/bin/python" -> the "/.venv/bin/python" tail is not a path in
# its own right) and a path that continues a word.
_ABS_PY_RE = re.compile(
    r"(?<![\w./~}-])(/[\w./+~-]*?/python[0-9]*(?:\.[0-9]+)?)(?![\w.-])"
)
# A LITERAL CWD-DEPENDENT interpreter path: it starts with a dot segment
# ("./.venv/bin/python", "../venv/bin/python", ".venv/bin/python").  The
# lookbehind keeps it from re-matching the TAIL of a longer path
# ("$REPO_ROOT/.venv/bin/python" is a variable-expanded path, not a relative
# one), and the leading-dot requirement keeps the "${PY:-/abs/path}" default
# out of this class.
_REL_PY_RE = re.compile(
    r"(?<![\w./~}/])((?:\.{1,2}/|\.venv/)[\w./+~-]*?/python[0-9]*(?:\.[0-9]+)?)(?![\w.-])"
)
# A forbidden venv path fragment (prefix-agnostic: it fires however the path
# is assembled).  A bare mention ("the legacy CadButEaas / kanzi venvs are
# forbidden") is NOT a path and does not match: the name must be followed by
# a "/" and at least one more path segment.
_FORBIDDEN_RE = re.compile(r"(?:CadButEaas|kanzi)(?:/[\w./+-]+)+")


def _rel(path: Path) -> str:
    """Repo-relative path, best effort."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def _load_yaml_only_gate():
    """Import ``scripts/_yaml_only_gate.py`` BY PATH.  Returns ``(module, error)``.

    ``scripts/`` has no ``__init__.py``, so the module is loaded with
    :func:`importlib.util.spec_from_file_location` rather than imported by
    name.  The import is LAZY and fault-isolated: a concurrent edit that
    leaves the yaml gate mid-syntax-error must not crash this audit — it must
    make this audit FAIL, because an audit that cannot compute its population
    cannot certify coverage.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_yaml_only_gate_wfa269", _YAML_GATE_PATH
        )
        if spec is None or spec.loader is None:
            return None, f"cannot build an import spec for {_YAML_GATE_PATH}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, ""
    except Exception as exc:  # noqa: BLE001 - any import fault must fail closed
        return None, f"{type(exc).__name__}: {exc}"


def _population_from_gate(gate) -> tuple[list[Path], list[dict], dict]:
    """``(scanned paths, excluded entries, coverage record)`` from the gate.

    The gate's own :func:`_population` is the primary source.  The fallback
    re-assembles the SAME population from the gate's other primitives — never
    from a second copy of the exclusion list.
    """
    population = getattr(gate, "_population", None)
    if callable(population):
        scanned, coverage = population()
        return list(scanned), list(coverage.get("excluded", [])), dict(coverage)

    tracked, error = gate._git_tracked_sh_paths()
    if tracked is None:
        return (
            [],
            [],
            {
                "population_source": "UNAVAILABLE",
                "population_degraded": True,
                "population_error": error,
                "n_tracked_sh_total": None,
                "n_scanned": 0,
                "n_excluded": 0,
                "coverage_pct": None,
                "excluded": [],
            },
        )
    scanned: list[Path] = []
    excluded: list[dict] = []
    for rel in tracked:
        path = REPO_ROOT / rel
        found = gate._exclusion_for(path, rel)
        if found is None:
            scanned.append(path)
        else:
            kind, reason = found
            excluded.append({"path": rel, "kind": kind, "reason": reason})
    total = len(tracked)
    return scanned, excluded, {
        "population_source": "git ls-files '*.sh' (reassembled from the yaml gate's primitives)",
        "population_degraded": False,
        "population_error": "",
        "n_tracked_sh_total": total,
        "n_scanned": len(scanned),
        "n_excluded": len(excluded),
        "coverage_pct": round(100.0 * len(scanned) / total, 1) if total else None,
        "excluded": excluded,
    }


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, quote-aware.

    A ``#`` inside quotes is data; a ``#`` glued to a word is part of that
    word.  Only a ``#`` that starts a word begins a comment.  This matters:
    several tracked wrappers carry the forbidden venv path inside a
    preserved-verbatim comment, and a scanner that reads comments would pin
    them as broken forever.
    """
    out: list[str] = []
    quote = None
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if quote:
            out.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "'\"":
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "#" and (i == 0 or line[i - 1] in " \t;&|("):
            break
        out.append(c)
        i += 1
    return "".join(out)


def interpreter_census(paths: list[Path]) -> dict:
    """§5.9 interpreter census over the given ``.sh`` files.

    The rule is LITERAL by construction: an interpreter path is adjudicated
    when it is written out in full.  A path assembled from a variable prefix
    (``${REPO_ROOT}/.venv/bin/python``) is not adjudicated by the absolute
    rule — it cannot be, without resolving shell state that the file does not
    declare — but the FORBIDDEN rule still fires on it, because it matches
    the forbidden venv NAME wherever it appears.

    Classes:

    * ``canonical``  — a literal absolute path under the repo's ``.venv``;
    * ``foreign``    — a literal absolute path outside it (§5.9 VIOLATION);
    * ``forbidden``  — the ``CadButEaas`` / ``kanzi`` venv name in a path
      (§5.9 VIOLATION — §5.9 names these two environments explicitly);
    * ``relative``   — a literal relative interpreter path (DISCLOSED: the
      path depends on the caller's CWD; not a §5.9 violation here).
    """
    counts = {"canonical": 0, "foreign": 0, "forbidden": 0, "relative": 0}
    foreign: list[dict] = []
    forbidden: list[dict] = []
    relative: list[dict] = []
    unreadable: list[str] = []
    for path in paths:
        rel = _rel(path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unreadable.append(rel)
            continue
        for lineno, raw in enumerate(text.splitlines(), 1):
            code = _strip_shell_comment(raw)
            if not code.strip():
                continue
            for match in _FORBIDDEN_RE.finditer(code):
                counts["forbidden"] += 1
                forbidden.append(
                    {"file": rel, "line": lineno, "token": match.group(0)}
                )
            for match in _ABS_PY_RE.finditer(code):
                literal = match.group(1)
                if literal.startswith(CANONICAL_VENV + "/"):
                    counts["canonical"] += 1
                else:
                    counts["foreign"] += 1
                    foreign.append(
                        {"file": rel, "line": lineno, "path": literal}
                    )
            for match in _REL_PY_RE.finditer(code):
                counts["relative"] += 1
                relative.append(
                    {"file": rel, "line": lineno, "path": match.group(1)}
                )
    return {
        "n_files": len(paths),
        "n_unreadable": len(unreadable),
        "counts": counts,
        "foreign": foreign,
        "forbidden": forbidden,
        "relative": relative,
        "rule": (
            "literal absolute interpreter path outside "
            f"{CANONICAL_VENV} => foreign; the forbidden venv names "
            f"{list(FORBIDDEN_VENV_NAMES)} in a path => forbidden"
        ),
    }


def section59_coverage() -> tuple[dict, dict]:
    """``(coverage record, census record)`` for the §5.9 interpreter census.

    TWO scopes, ONE population:

    * the IN-CONTRACT scope is the yaml gate's scanned set — the files this
      audit judges;
    * the EXCLUDED-RESIDUAL scope is the gate's exclusion set, measured
      anyway, because excluding a file from a gate's CONTRACT must not HIDE
      that it carries a defect (the yaml gate's own residual-census rule).

    The union is the whole tracked ``.sh`` population, so the §5.9 census
    covers 100 % of the repository's tracked shell scripts even though the
    in-contract population is the yaml gate's 87.6 %.
    """
    gate, error = _load_yaml_only_gate()
    if gate is None:
        coverage = {
            "population_source": "UNAVAILABLE",
            "population_degraded": True,
            "population_error": error,
            "n_tracked_sh_total": None,
            "n_scanned": 0,
            "n_excluded": 0,
            "coverage_pct": None,
            "excluded": [],
            "imported_from": str(_YAML_GATE_PATH),
            "imported_symbols": list(_YAML_GATE_SYMBOLS),
            "n_examined_total": 0,
        }
        empty = {
            "n_files": 0,
            "n_unreadable": 0,
            "counts": {},
            "foreign": [],
            "forbidden": [],
            "relative": [],
            "rule": "population unavailable",
        }
        return coverage, {"scanned": empty, "excluded_residual": empty}

    scanned, excluded_entries, coverage = _population_from_gate(gate)
    coverage["imported_from"] = _YAML_GATE_PATH.name
    coverage["imported_symbols"] = list(_YAML_GATE_SYMBOLS)
    coverage["n_examined_total"] = len(scanned) + len(excluded_entries)
    coverage["examination_scopes"] = {
        "in_contract": len(scanned),
        "excluded_residual": len(excluded_entries),
    }

    residual_paths = [
        REPO_ROOT / entry["path"] for entry in excluded_entries
    ]
    scanned_census = interpreter_census(scanned)
    residual_census = interpreter_census(residual_paths)
    return coverage, {
        "scanned": scanned_census,
        "excluded_residual": residual_census,
    }


def audit(strict: bool = False) -> dict:
    """Run the full audit and emit structured JSON."""
    wrappers = discover_wrappers(SCRIPTS_DIR)
    records = []
    summary = {"REAL": 0, "SCAFFOLD": 0, "DUPLICATE": 0, "TEMPLATE": 0}
    for w in wrappers:
        surface = surface_check(w)
        cls, evidence = classify(surface, w)
        records.append(
            {
                "wrapper": w.name,
                "path": str(w),
                "size_bytes": w.stat().st_size,
                "mode_octal": oct(w.stat().st_mode & 0o777),
                "classification": cls,
                "evidence": evidence,
                "matched_markers": [
                    k for k, v in surface.items() if v and k != "_template"
                ],
                "missing_markers": [
                    k for k, v in surface.items() if not v and k != "_template"
                ],
            }
        )
        if w.name.startswith("_"):
            summary["TEMPLATE"] += 1
        elif cls.startswith("DUAL"):
            summary["TEMPLATE"] += 1
        else:
            summary[cls] = summary.get(cls, 0) + 1

    duplicates = collect_duplicates(wrappers)
    for record in records:
        record["duplicates_panel_with"] = [
            p
            for p, who in duplicates.items()
            if record["wrapper"] in who
            and record["wrapper"] not in who.replace(record["wrapper"], "")
        ]

    coverage, census = section59_coverage()
    scanned_census = census["scanned"]
    residual_census = census["excluded_residual"]
    # A foreign interpreter in the IN-CONTRACT population is a §5.9
    # violation.  The same defect in an EXCLUDED file is a disclosed
    # residual: the exclusion is a §5.11 contract deferral, not a §5.9
    # waiver, so it is measured and printed every run (it is NOT a
    # violation, and the report never prints the word "VIOLATION" for it).
    section59_violations = list(scanned_census["foreign"]) + list(
        scanned_census["forbidden"]
    )
    violations = [
        r["wrapper"]
        for r in records
        if r["classification"] in ("SCAFFOLD", "DUPLICATE")
    ]
    degraded = bool(coverage.get("population_degraded"))
    verdict_reasons: list[str] = []
    if violations:
        verdict_reasons.append("wrapper_violations")
    if section59_violations:
        verdict_reasons.append("section59_interpreter_violations")
    if degraded:
        verdict_reasons.append("population_degraded")
    verdict = "FAIL" if verdict_reasons else "PASS"

    return {
        "audit_ts": datetime.now(timezone.utc).isoformat(),
        "wrapper_count": len(wrappers),
        "summary": summary,
        "duplicates": duplicates,
        "canonical_markers": list(CANONICAL_MARKERS.keys()),
        "template": "scripts/_WRAPPER_TEMPLATE.sh",
        "wrappers": records,
        "violations": violations,
        "coverage": coverage,
        "section59_interpreters": census,
        "section59_violations": section59_violations,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }


def write_audit_json(audit_obj: dict) -> None:
    """Atomic-write the audit JSON to molmetal/reports/wrapper_audit.json."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(AUDIT_JSON, json.dumps(audit_obj, indent=2, sort_keys=True))
    except ImportError:
        # Fallback: direct write_text (NOT atomic — only as a last resort if the
        # atomic primitive fails to import). Audit script is CI-safe.
        AUDIT_JSON.write_text(
            json.dumps(audit_obj, indent=2, sort_keys=True), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper audit per Wf-431-A2 FIX 2")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 on any SCAFFOLD/DUPLICATE"
    )
    parser.add_argument(
        "--json-only", action="store_true", help="console.log the JSON and exit 0"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="console.log the summary only"
    )
    args = parser.parse_args()

    audit_obj = audit(strict=args.strict)
    write_audit_json(audit_obj)

    if args.summary_only:
        s = audit_obj["summary"]
        total = audit_obj["wrapper_count"]
        print(
            json.dumps(
                {
                    "summary": s,
                    "total": total,
                    "violations": audit_obj["violations"],
                    "coverage": audit_obj["coverage"],
                    "section59_violations": audit_obj["section59_violations"],
                    "verdict": audit_obj["verdict"],
                    "verdict_reasons": audit_obj["verdict_reasons"],
                },
                indent=2,
            )
        )
    elif args.json_only:
        print(json.dumps(audit_obj, indent=2, sort_keys=True))
    else:
        s = audit_obj["summary"]
        total = audit_obj["wrapper_count"]
        print(f"[wrapper_audit] {total} wrappers scanned → {s}")
        if audit_obj["violations"]:
            print(f"[wrapper_audit] {len(audit_obj['violations'])} violations:")
            for v in audit_obj["violations"]:
                print(f"  - {v}")
        _print_coverage(audit_obj)
        print(f"[wrapper_audit] JSON: {AUDIT_JSON}")

    # 2026-10-08 gate rc: default run now exits nonzero when violations exist so
    # the audit can actually gate CI (previously it always returned 0 unless
    # --strict). --json-only remains an informational escape hatch (exit 0).
    # 2026-11-04 (Wf-A269 wave-16): the §5.9 interpreter census joins the
    # verdict, and a DEGRADED population fails closed — an audit that cannot
    # compute its population cannot certify coverage.
    if args.json_only:
        return 0
    return 0 if audit_obj["verdict"] == "PASS" else 1


def _merge_rows(rows: list[dict]) -> list[dict]:
    """Merge census rows that land on the same ``file:line``.

    One stale-interpreter assignment trips BOTH rules (the absolute path and
    the forbidden venv name), so the raw rows are counted once per FILE, not
    once per rule.
    """
    merged: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row.get("file", ""), row.get("line", 0))
        entry = merged.setdefault(
            key, {"file": key[0], "line": key[1], "parts": []}
        )
        label = row.get("path") or row.get("token")
        if label and label not in entry["parts"]:
            entry["parts"].append(label)
    return [merged[key] for key in sorted(merged)]


def _print_coverage(audit_obj: dict) -> None:
    """Print the §5.9 coverage block: denominator, exclusions, census."""
    cov = audit_obj.get("coverage") or {}
    census = audit_obj.get("section59_interpreters") or {}
    scanned = census.get("scanned") or {}
    residual = census.get("excluded_residual") or {}
    counts = scanned.get("counts") or {}

    print(
        f"[wrapper_audit] §5.9 COVERAGE: {cov.get('n_tracked_sh_total')} tracked "
        f"*.sh in the repo | {cov.get('n_scanned')} scanned | "
        f"{cov.get('n_excluded')} excluded | {cov.get('coverage_pct')}% scanned | "
        f"source: {cov.get('population_source')}"
    )
    print(
        "[wrapper_audit] §5.9 population IMPORTED from "
        f"{cov.get('imported_from')} (symbols: "
        f"{', '.join(cov.get('imported_symbols') or [])})"
    )
    if cov.get("population_degraded"):
        print(
            "[wrapper_audit] POPULATION DEGRADED: "
            f"{cov.get('population_error')} — the gate cannot state its "
            "denominator, so it cannot certify coverage"
        )
    for entry in cov.get("excluded", []):
        print(f"  EXCLUDED {entry['path']} [{entry['kind']}] — {entry['reason']}")
    print(
        f"[wrapper_audit] §5.9 interpreter census: "
        f"{cov.get('n_examined_total')} examined "
        f"({cov.get('examination_scopes', {}).get('in_contract')} in-contract + "
        f"{cov.get('examination_scopes', {}).get('excluded_residual')} "
        f"excluded-residual) | canonical={counts.get('canonical', 0)} "
        f"foreign={counts.get('foreign', 0)} "
        f"forbidden={counts.get('forbidden', 0)} "
        f"relative={counts.get('relative', 0)}"
    )
    for row in audit_obj.get("section59_violations", []):
        print(
            f"  VIOLATION {row['file']}:{row['line']} "
            f"[{row.get('path') or row.get('token')}] — §5.9 forbids an "
            "interpreter outside the repo venv"
        )
    residual_rows = _merge_rows(
        list(residual.get("foreign", [])) + list(residual.get("forbidden", []))
    )
    if residual_rows:
        detail = ", ".join(
            f"{row['file']}:{row['line']} ({' | '.join(row['parts'])})"
            for row in residual_rows
        )
        print(
            "[wrapper_audit] §5.9 RESIDUAL (out of this gate's contract, "
            f"disclosed every run): {len(residual_rows)} stale-interpreter "
            f"assignment(s) across the excluded file(s) — {detail}"
        )
    for tag, scope in (("in-contract", scanned), ("excluded-residual", residual)):
        rel_rows = scope.get("relative") or []
        if not rel_rows:
            continue
        files = sorted({row["file"] for row in rel_rows})
        print(
            f"[wrapper_audit] §5.9 DISCLOSED ({tag}): {len(rel_rows)} relative "
            f"interpreter path(s) in {len(files)} file(s) — {', '.join(files)}"
        )


if __name__ == "__main__":
    sys.exit(main())
