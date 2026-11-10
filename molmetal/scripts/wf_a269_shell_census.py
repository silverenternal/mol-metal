#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-A269 — the SHELL census: the .sh half of the host-bound-path class.

THE SHIPPED GATE SCANS .py ONLY.

``molmetal/tests/test_wfa269_no_hardcoded_repo_paths.py`` states this
explicitly in its docstring ("WHAT THIS GATE DOES NOT COVER: Shell
wrappers ... this gate scans .py files only").  The justification given
there is AGENTS.md §5.9, which MANDATES that ``scripts/launch_*.sh`` use
the absolute path ``/home/hugo/codes/mol-metal/.venv/bin/python``.

So the .sh half of the class is not merely uncovered — its own gate
REQUIRES the non-portable form.  This script measures how big that half
is, so §6.269.14 can scope wave-3 with a number instead of a shrug.

THE FROZEN SHELL RULE (frozen 2026-11-02, this file).  Mirrors the
Python rule's four conditions:

  BINDING      — the literal occurs in the file, outside a comment and
                  outside a pure ``echo``/``printf`` diagnostic line.
  REACHABILITY — the literal is used in a PATH position: it is assigned
                  to a variable, exported, or interpolated into a command
                  argument.  (A literal only mentioned in prose is not a
                  binding.)
  LIVENESS     — the variable it is bound to is READ somewhere later in
                  the same file.  (Bound and never read => DEAD.)
  IN-REPO      — the path, after stripping the host root, names a
                  tracked file or a directory that contains tracked
                  files — i.e. it lands inside THIS repository.

ARCHIVED is a file under ``TODO/archive/`` — historical artifact, not
live code.

Usage (CPU-only, read-only):
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        /tmp/wfa269k_shell_census.py --json /tmp/wfa269k_shell_census.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _gate_roots() -> tuple[str, ...]:
    """The forbidden host roots, read from the gate -- the SSOT for them.

    Reading them instead of spelling them here is what keeps THIS file free
    of the very literals it is a detector for.
    """
    gate = REPO / "molmetal" / "tests" / "test_wfa269_no_hardcoded_repo_paths.py"
    spec = importlib.util.spec_from_file_location("wfa269_gate", gate)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wfa269_gate"] = mod
    spec.loader.exec_module(mod)
    return tuple(mod._ROOTS)


# The same three roots the .py gate forbids.
HOST_ROOTS = _gate_roots()

# The match must cover the WHOLE literal (root + the path after it), not
# just the root prefix: `findall` returns the full match, and the tail is
# what decides whether the path lands inside this repository.
HOST_RE = re.compile(
    "(?:" + "|".join(re.escape(r) for r in HOST_ROOTS) + r")"
    r"(?:/[^\s\"':;|,)\]]*)?"
)

# A line that only prints the literal is a diagnostic, not a binding.
DIAGNOSTIC_RE = re.compile(
    r"^\s*(echo|printf|log|logger)\b", re.IGNORECASE
)

# `VAR="..."` or `VAR=...`
ASSIGN_RE = re.compile(r"^\s*(?:export\s+|declare\s+-\w+\s+|local\s+)?"
                       r"([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# A variable read: `$VAR` or `${VAR}` or `${VAR:-...}`
VAR_READ_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def tracked(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True, text=True, check=True,
    )
    return [n for n in out.stdout.split("\0") if n]


def build_index(names: list[str]) -> tuple[set[str], dict[str, int]]:
    prefixes: dict[str, int] = defaultdict(int)
    for n in names:
        parts = n.split("/")
        for i in range(1, len(parts) + 1):
            prefixes["/".join(parts[:i])] += 1
    return set(names), dict(prefixes)


def lands_in_repo(lit: str, index: set[str], prefixes: dict[str, int]) -> str | None:
    """The repo-relative path the literal lands on, or None."""
    parts = [p for p in lit.split("/") if p]
    for i in range(len(parts)):
        suffix = "/".join(parts[i:])
        if suffix in index or suffix in prefixes:
            return suffix
    return None


def strip_root(lit: str) -> str:
    for r in HOST_ROOTS:
        if lit == r:
            return ""
        if lit.startswith(r + "/"):
            return lit[len(r) + 1:]
    return lit


def classify_file(rel: str, text: str,
                  index: set[str], prefixes: dict[str, int]) -> dict:
    """Apply the frozen shell rule to one file."""
    lines = text.splitlines()
    occurrences = 0
    bound_vars: set[str] = set()
    live_vars: set[str] = set()
    in_repo_hits = 0

    # --- pass 1: binding (literal outside comments / pure diagnostics) ---
    per_line: list[tuple[int, list[str]]] = []  # (lineno, literals)
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        lits = HOST_RE.findall(line)
        if not lits:
            continue
        occurrences += len(lits)
        per_line.append((i, lits))

    # --- pass 2: which variable carries each literal, and is it read? ---
    for lineno, lits in per_line:
        line = lines[lineno - 1]
        # is this line a diagnostic (echo/printf) only?
        if DIAGNOSTIC_RE.match(line):
            continue
        m = ASSIGN_RE.match(line)
        if m:
            var, rhs = m.group(1), m.group(2)
            if any(l in rhs for l in lits):
                bound_vars.add(var)
        else:
            # a literal used directly in a command position: treat the
            # enclosing command's variables as carrying it
            for v in VAR_READ_RE.findall(line):
                bound_vars.add(v)

    # --- pass 3: liveness — is each bound variable read anywhere? ---
    for lineno, _ in per_line:
        line = lines[lineno - 1]
        if DIAGNOSTIC_RE.match(line):
            continue
        m = ASSIGN_RE.match(line)
        if m:
            var = m.group(1)
            for i, other in enumerate(lines, 1):
                if i == lineno:
                    continue
                if VAR_READ_RE.search(other) and re.search(
                    r"\$\{?" + re.escape(var) + r"\b", other
                ):
                    live_vars.add(var)
                    break
        else:
            for v in VAR_READ_RE.findall(line):
                live_vars.add(v)

    # --- pass 4: in-repo test on each literal ---
    for lineno, lits in per_line:
        line = lines[lineno - 1]
        if DIAGNOSTIC_RE.match(line):
            continue
        for lit in lits:
            tail = strip_root(lit)
            if tail and lands_in_repo(tail, index, prefixes) is not None:
                in_repo_hits += 1

    # --- pass 5: the COMPOSED form. ---
    # A bare host root bound to a variable, later read on a line that also
    # names a repo-relative path, composes to a path INSIDE this
    # repository -- e.g.
    #     export PYTHONPATH=/home/hugo/codes/mol-metal
    #     ${PY} molmetal/scripts/pre_flight_check.py
    # This is the DERIVED category of the Python census.  Without this
    # step the dominant shell shape (the bare root) would be counted as
    # "not a repo path" and the shell half of the class would look
    # seven files wide instead of the size it actually is.
    repo_prefix_re = re.compile(
        r"(?:^|[\s\"'=:/])(?:molmetal|scripts|configs|TODO|metrics|"
        r"triton_kernels|reports|\.data/molmetal)/"
    )
    composed = 0
    for lineno, lits in per_line:
        line = lines[lineno - 1]
        if DIAGNOSTIC_RE.match(line):
            continue
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        var = m.group(1)
        rhs = m.group(2)
        # the assigned value is (or contains) a BARE host root
        if not any(strip_root(l) == "" for l in lits):
            continue
        if not HOST_RE.search(rhs):
            continue
        # is the variable read on a line that also names a repo path?
        for other in lines:
            if other is line:
                continue
            if re.search(r"\$\{?" + re.escape(var) + r"\b", other) and \
                    repo_prefix_re.search(other):
                composed += 1
                break

    return {
        "file": rel,
        "occurrences": occurrences,
        "in_repo": in_repo_hits,
        "composed_in_repo": composed,
        "bound_vars": sorted(bound_vars),
        "live_vars": sorted(live_vars),
        "dead_vars": sorted(bound_vars - live_vars),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    repo = Path(args.repo)

    names = tracked(repo)
    index, prefixes = build_index(names)
    sh = [n for n in names if n.endswith(".sh")]

    rows = []
    for rel in sh:
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if not HOST_RE.search(text):
            continue
        r = classify_file(rel, text, index, prefixes)
        r["archived"] = rel.startswith("TODO/archive/")
        rows.append(r)

    live = [r for r in rows if not r["archived"]]
    arch = [r for r in rows if r["archived"]]
    live_inclass = [r for r in live if r["in_repo"] > 0 or r["composed_in_repo"] > 0]

    summary = {
        "tracked_sh_files": len(sh),
        "sh_files_with_a_host_root_string": len(rows),
        "live_files_with_a_host_root_string": len(live),
        "archived_files_with_a_host_root_string": len(arch),
        "live_files_under_scripts_dir": len([r for r in live if r["file"].startswith("scripts/")]),
        "total_occurrences_all": sum(r["occurrences"] for r in rows),
        "total_occurrences_live": sum(r["occurrences"] for r in live),
        "live_in_class_files": len(live_inclass),
        "live_in_class_occurrences": sum(
            r["in_repo"] + r["composed_in_repo"] for r in live_inclass),
        "live_in_class_direct_occurrences": sum(r["in_repo"] for r in live_inclass),
        "live_in_class_composed_occurrences": sum(
            r["composed_in_repo"] for r in live_inclass),
        "live_files_with_a_direct_in_repo_path": len(
            [r for r in live if r["in_repo"] > 0]),
        "live_files_with_a_composed_in_repo_path": len(
            [r for r in live if r["composed_in_repo"] > 0]),
        "live_files_with_a_dead_bound_var": len([r for r in live if r["dead_vars"]]),
    }

    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
