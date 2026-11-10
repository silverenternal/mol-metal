# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Regression tests for the AGENTS.md §5.1 VRAM-budget guard's `bc` dependency.

`bc` is NOT installed on this host (MEASURED 2026-11-02: `command -v bc`
-> rc=1, no binary; `awk` IS present at /usr/bin/awk).  A launcher whose
VRAM-budget guard computes `free_gb` through `bc` therefore always sees
free_gb="0" (the `|| echo "0"` fallback) and the budget comparison always
evaluates "1" (the `|| echo 1` fallback) — so the gate ALWAYS dies with
"VRAM free=0GB < budget=...GB" even on a completely idle GPU.  The OK arm
is unreachable and a GPU launch can never proceed.  The same `bc` call
empties the wall-hour field of the per-phase log line ("wall=123s ( hr)").

A guard that cannot distinguish a full GPU from an empty one is worse than
no guard, because it looks like one.

Three things are enforced here:

  1. no LIVE `bc` invocation survives in any ``scripts/*.sh`` — a
     comment-only mention is LEGAL and EXPECTED, because AGENTS.md §5.5
     rule #6 requires the superseded rendering to be preserved VERBATIM
     as a comment block (those blocks quote the old `bc` expressions);
  2. every launcher owning a VRAM budget guard computes free_gb and
     compares it against VRAM_BUDGET_GB WITHOUT `bc`;
  3. the arithmetic path is EXERCISED, not merely pattern-matched: the
     real ``vram_gate`` body is lifted out of the launcher and run under
     bash with a stubbed ``nvidia-smi`` (stubbed free_mib) plus a stub
     ``bc`` that always fails — so the test is deterministic on a host
     that HAS `bc` installed as well as on one that does not, and no GPU
     is ever touched.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# The launcher this group owns (GROUP 4 of the Wf-A269 wave).
TARGET_LAUNCHER = SCRIPTS_DIR / "launch_wf310e_r20_p4_bindingdb_1.07M.sh"

# `bc` as a standalone word.  "bc29aa90" (a git SHA in a comment) and
# "subchecklist" do NOT match: a word boundary requires a non-word
# character on both sides of "bc".
_BC_RE = re.compile(r"\bbc\b")

# Harness: the real vram_gate() body is spliced in where @@BODY@@ sits.
# The stubs replace the launcher's own log/ok/warn/die helpers so that the
# function under test can be executed in isolation.
_HARNESS = """\
set -euo pipefail
export PATH="@@STUBDIR@@:$PATH"
VRAM_BUDGET_GB=20
DRY_RUN=0
MODE=R3
die() { printf 'DIE: %s\\n' "$*" >&2; exit 3; }
ok()  { printf 'OK: %s\\n' "$*"; }
log() { printf 'LOG: %s\\n' "$*" >&2; }
warn(){ printf 'WARN: %s\\n' "$*" >&2; }
@@BODY@@
vram_gate "cuda:0"
"""


def _strip_comment(line: str) -> str:
    """Return `line` with any trailing shell comment removed.

    Quote-aware so that a '#' inside a quoted string is not mistaken for
    the start of a comment.
    """
    out: list[str] = []
    quote: str | None = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def _live_lines(text: str) -> list[tuple[int, str]]:
    """(lineno, code) for every non-comment, non-blank line of `text`."""
    out: list[tuple[int, str]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        code = _strip_comment(raw)
        if code.strip():
            out.append((idx, code))
    return out


def _shell_files() -> list[Path]:
    return sorted(SCRIPTS_DIR.rglob("*.sh"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _rel(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _extract_shell_function(text: str, name: str) -> str:
    """Return the source text of the bash function `name`()."""
    m = re.search(r"^" + re.escape(name) + r"\s*\(\)\s*\{", text, re.MULTILINE)
    assert m is not None, f"bash function {name}() not found"
    depth = 0
    for i in range(m.end() - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.start() : i + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}()")


def _make_stub_bin(tmp_path: Path) -> tuple[Path, Path]:
    """Build a stub PATH dir: a fake nvidia-smi and a `bc` that always fails.

    The failing `bc` stub makes the test deterministic on a host where `bc`
    IS installed: it reproduces exactly what a missing `bc` does to the
    launcher's `... | bc 2>/dev/null || echo "<fallback>"` expressions.
    """
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir(parents=True, exist_ok=True)

    nvidia_stub = stub_dir / "nvidia-smi"
    nvidia_stub.write_text('#!/bin/sh\nprintf \'%s\\n\' "${STUB_FREE_MIB}"\n')
    nvidia_stub.chmod(0o755)

    bc_marker = tmp_path / "bc-was-called.txt"
    bc_stub = stub_dir / "bc"
    bc_stub.write_text('#!/bin/sh\n: > "${BC_MARKER_FILE}"\nexit 127\n')
    bc_stub.chmod(0o755)
    return stub_dir, bc_marker


def _run_vram_gate(
    tmp_path: Path, free_mib: str, launcher: Path | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the launcher's real vram_gate() against a stubbed nvidia-smi."""
    if launcher is None:
        launcher = TARGET_LAUNCHER
    stub_dir, bc_marker = _make_stub_bin(tmp_path)
    body = _extract_shell_function(_read(launcher), "vram_gate")
    harness = _HARNESS.replace("@@STUBDIR@@", str(stub_dir)).replace("@@BODY@@", body)
    harness_path = tmp_path / "harness.sh"
    harness_path.write_text(harness)

    env = dict(os.environ)
    env["PATH"] = str(stub_dir) + ":" + env.get("PATH", "")
    env["STUB_FREE_MIB"] = str(free_mib)
    env["BC_MARKER_FILE"] = str(bc_marker)

    proc = subprocess.run(
        ["bash", str(harness_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc, bc_marker


# ---------------------------------------------------------------------------
# 1. no LIVE `bc` anywhere in scripts/*.sh
# ---------------------------------------------------------------------------
def test_no_live_bc_in_any_launcher_script() -> None:
    offenders: list[str] = []
    for path in _shell_files():
        for lineno, code in _live_lines(_read(path)):
            if _BC_RE.search(code):
                offenders.append(f"{_rel(path)}:{lineno}: {code.strip()}")
    assert not offenders, (
        "LIVE `bc` invocation(s) found in scripts/*.sh. `bc` is NOT installed on "
        "this host (MEASURED 2026-11-02: `command -v bc` -> rc=1, no binary), so "
        "each of these expressions silently falls back to its `|| echo` arm:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# 2. every VRAM guard computes / compares free_gb WITHOUT `bc`
# ---------------------------------------------------------------------------
def test_every_vram_guard_computes_free_gb_without_bc() -> None:
    checked: list[str] = []
    for path in _shell_files():
        text = _read(path)
        if "VRAM_BUDGET_GB" not in text:
            continue
        rel = _rel(path)
        checked.append(rel)
        guard = [
            (ln, c)
            for ln, c in _live_lines(text)
            if "free_gb" in c or "VRAM_BUDGET_GB" in c
        ]
        assert guard, f"{rel} declares VRAM_BUDGET_GB but has no live VRAM-guard lines"
        for lineno, code in guard:
            assert not _BC_RE.search(code), (
                f"{rel}:{lineno}: VRAM-guard line still shells out to `bc`: "
                f"{code.strip()}"
            )
        joined = "\n".join(c for _, c in guard)
        assert "1024" in joined, (
            f"{rel}: VRAM guard does not convert free MiB -> GB (no /1024):\n{joined}"
        )
        assert any("free_gb" in c and "VRAM_BUDGET_GB" in c for _, c in guard), (
            f"{rel}: VRAM guard never compares free_gb against VRAM_BUDGET_GB:\n{joined}"
        )
    assert checked, "no launcher with a VRAM_BUDGET_GB guard was found"


# ---------------------------------------------------------------------------
# 3. the arithmetic path is EXERCISED with a stubbed free_mib
# ---------------------------------------------------------------------------
def test_vram_gate_passes_when_free_vram_exceeds_budget(tmp_path: Path) -> None:
    proc, bc_marker = _run_vram_gate(tmp_path, 40960)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"vram_gate died on an idle-enough GPU (rc={proc.returncode}); "
        f"40 GiB free must pass a 20 GiB budget:\n{combined}"
    )
    assert "VRAM free=40GB" in combined, combined
    assert not bc_marker.exists(), "the VRAM guard shelled out to `bc`"


def test_vram_gate_dies_when_free_vram_below_budget(tmp_path: Path) -> None:
    proc, bc_marker = _run_vram_gate(tmp_path, 2048)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 3, (
        f"vram_gate must refuse a 2 GiB-free GPU against a 20 GiB budget "
        f"(rc={proc.returncode}):\n{combined}"
    )
    assert "VRAM free=2GB < budget=20GB" in combined, combined
    assert not bc_marker.exists(), "the VRAM guard shelled out to `bc`"


def test_vram_gate_dies_on_non_numeric_probe(tmp_path: Path) -> None:
    proc, _ = _run_vram_gate(tmp_path, "not-a-number")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 3, (
        f"vram_gate must refuse a non-numeric nvidia-smi reading "
        f"(rc={proc.returncode}):\n{combined}"
    )
    assert "non-numeric free VRAM" in combined, combined


# ---------------------------------------------------------------------------
# 4. the wall-hour field is rendered without `bc`
# ---------------------------------------------------------------------------
def test_wall_hour_field_is_rendered_without_bc(tmp_path: Path) -> None:
    text = _read(TARGET_LAUNCHER)
    m = re.search(r"^\s*wall_hr=\$\((.+)\)\s*$", text, re.MULTILINE)
    assert m is not None, (
        "no `wall_hr=$( ... )` rendering found — the wall-hour field must be "
        "computed without `bc` (which is absent on this host)"
    )
    expr = m.group(1)

    stub_dir, bc_marker = _make_stub_bin(tmp_path)
    script = "\n".join(
        [
            "set -euo pipefail",
            'export PATH="' + str(stub_dir) + ':$PATH"',
            "wall_sec=3600",
            "wall_hr=$(" + expr + ")",
            'printf \'%s\' "${wall_hr}"',
            "",
        ]
    )
    script_path = tmp_path / "wall_hr.sh"
    script_path.write_text(script)

    env = dict(os.environ)
    env["BC_MARKER_FILE"] = str(bc_marker)
    proc = subprocess.run(
        ["bash", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"wall-hour rendering failed: {proc.stderr}"
    assert proc.stdout == "1.00", (
        f"3600 s must render as '1.00' hr, got {proc.stdout!r} "
        f"(stderr: {proc.stderr!r})"
    )
    assert not bc_marker.exists(), "the wall-hour rendering shelled out to `bc`"
