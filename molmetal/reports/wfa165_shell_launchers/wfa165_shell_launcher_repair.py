#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
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
"""Wf-A165 branch A — the shell launchers that hard-override the GPU mask.

THE DEFECT
----------
The session's GPU lock is "run everything with ``CUDA_VISIBLE_DEVICES=""``".
Several tracked shell scripts UNCONDITIONALLY overwrite that variable, so
running them silently re-enables a GPU — exactly what the lock exists to
prevent.  Three mechanisms are in scope for THIS branch:

  (1) HARD OVERRIDE  ``export CUDA_VISIBLE_DEVICES=0``   (or ``=1``)
  (2) ``${VAR:-default}``  — the ``:-`` operator substitutes the default when
      the variable is unset **OR EMPTY**, so an ambient ``""`` (the lock) is
      REPLACED by the default.  POSIX ``${VAR-default}`` (no colon) keeps an
      explicitly-empty value.
  (3) THE YAML-DERIVED FORM in ``scripts/start.sh`` — **NOT OURS** (branch B).

THE REPAIR
----------
``export CUDA_VISIBLE_DEVICES=N``  ->  ``export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-N}"``

i.e. the ORIGINAL default is kept, but an explicitly-set ambient value
(including the empty string, i.e. the lock) wins.  The DEFAULT is never
changed: unset -> the original default, exactly as before.

WHAT THIS TOOL DOES
-------------------
  * enumerates every tracked ``*.sh`` via ``git ls-files`` (NEVER os.walk);
  * classifies every ``export CUDA_VISIBLE_DEVICES=...`` line;
  * repairs the defecting ones (atomically, via
    ``wf386a3_atomic_write.atomic_write_file``);
  * PROVES each repaired file with a three-case shell harness whose export
    line is EXTRACTED from the file on disk (never hand-typed);
  * writes ``proof.json``.

CPU-ONLY
--------
The proof harness is ``bash`` variable expansion only.  It never imports
torch, never touches CUDA, and allocates no GPU memory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

CVD = "CUDA_VISIBLE_DEVICES"

#: Files another branch owns, or that are historical artifacts rather than
#: live launchers.  Keyed by path -> the reason it is left alone.
SKIP: Dict[str, str] = {
    "scripts/start.sh": (
        "OWNED BY BRANCH B — the YAML-derived form (mechanism 3), "
        "scripts/start.sh:670-673. Not touched by branch A."
    ),
    "molmetal/reports/wf229_ablation/quick_pilot/run_7_remaining.sh": (
        "HISTORICAL ARTIFACT — an archived run script under molmetal/reports; "
        "editing it would rewrite a record of a past run. Its export FORCES "
        "the blank mask, so it cannot re-enable a GPU."
    ),
    "molmetal/reports/wf256_apply/wf229_quick_pilot_5epoch_BACKUP/run_7_remaining.sh": (
        "HISTORICAL ARTIFACT — same as above; export FORCES the blank mask."
    ),
    "scripts/_retired/launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh": (
        "export FORCES the blank mask (lock-safe); retired launcher, left alone."
    ),
}

EXPORT_RE = re.compile(r"^(?P<indent>\s*)export\s+CUDA_VISIBLE_DEVICES\s*=\s*(?P<rhs>.*)$")
ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=")

#: Classification verdicts.
SAFE_PRESERVE = "safe_preserves_ambient"
FORCES_BLANK = "safe_forces_blank"
DEFEATER = "defeater_overwrites_ambient"

COMMENT = (
    "# Wf-A165: an explicit ambient CUDA_VISIBLE_DEVICES (incl. the \"\" "
    "session lock) must win; unset -> {default}."
)


# --------------------------------------------------------------------------
# parameter-expansion parsing
# --------------------------------------------------------------------------
def _matching_brace(inner: str) -> int:
    """Index of the ``}`` closing the ``${`` that opens ``inner`` (or -1)."""
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def unquote(rhs: str) -> str:
    """Strip one layer of matching quotes from a shell right-hand side."""
    rhs = rhs.strip()
    if len(rhs) >= 2 and rhs[0] == rhs[-1] and rhs[0] in ("'", '"'):
        return rhs[1:-1]
    return rhs


def classify_export(rhs: str) -> Tuple[str, str]:
    """Classify the right-hand side of an ``export CUDA_VISIBLE_DEVICES=`` .

    Returns ``(verdict, detail)``.

    * ``safe_preserves_ambient``  — the value is the ambient variable itself
      (``${CUDA_VISIBLE_DEVICES-default}``, no colon), so an explicitly-set
      ambient value (including the empty string) survives.
    * ``safe_forces_blank``       — the value is the empty string; it can only
      ever TIGHTEN the mask, never re-enable a GPU.
    * ``defeater_overwrites_ambient`` — anything else: a literal, a
      ``${CUDA_VISIBLE_DEVICES:-default}`` (the colon replaces an explicitly
      EMPTY ambient value), or an expression that ignores the ambient value.
    """
    inner = unquote(rhs)
    if inner == "":
        return FORCES_BLANK, "value is the empty string (forces the blank mask)"
    if inner.startswith("${") and inner.endswith("}"):
        close = _matching_brace(inner)
        if close == len(inner) - 1:
            name = inner[2:close]
            if name.startswith(CVD + ":-"):
                return (
                    DEFEATER,
                    "${%s:-...} substitutes the default when the ambient value "
                    "is unset OR EMPTY" % CVD,
                )
            if name.startswith(CVD + "-"):
                return (
                    SAFE_PRESERVE,
                    "${%s-...} keeps an explicitly-set (even empty) ambient value"
                    % CVD,
                )
            return DEFEATER, "expansion of %r ignores the ambient %s" % (
                name.split("-")[0].split(":")[0],
                CVD,
            )
    return DEFEATER, "literal / unconditional override of the ambient %s" % CVD


def repaired_rhs(inner: str) -> str:
    """The lock-preserving replacement for a defeater's right-hand side."""
    if inner.startswith("${") and inner.endswith("}"):
        close = _matching_brace(inner)
        if close == len(inner) - 1:
            name = inner[2:close]
            if name.startswith(CVD + ":-"):
                # mechanism (2): drop ONLY the colon.
                return '"${%s-%s}"' % (CVD, name[len(CVD) + 2 :])
            # mechanism (1) / other-var: keep the original default as the
            # fallback, but let an explicitly-set ambient value win.
            return '"${%s-%s}"' % (CVD, inner)
    return '"${%s-%s}"' % (CVD, inner)


def default_word(inner: str) -> str:
    """The word substituted when the ambient variable is UNSET."""
    if inner.startswith("${") and inner.endswith("}"):
        close = _matching_brace(inner)
        if close == len(inner) - 1:
            name = inner[2:close]
            if name.startswith(CVD + ":-"):
                return name[len(CVD) + 2 :]
            if name.startswith(CVD + "-"):
                return name[len(CVD) + 1 :]
            return inner
    return inner


#: A command-PREFIX assignment — ``CUDA_VISIBLE_DEVICES=0 some-command`` — is
#: a FOURTH mechanism: it sets the mask for one command only, and it
#: overrides an explicitly-set ambient value exactly like an ``export`` does.
#: It is NOT an ``export`` line, so it is outside this branch's mechanism
#: (1)/(2) repair list; it is CENSUSED here so the finding is not lost.
CMD_PREFIX_RE = re.compile(
    r"^\s*(?:if\s+!?\s*|if\s+!\s+)?(?P<var>CUDA_VISIBLE_DEVICES)\s*=\s*(?P<rhs>\S+)\s+\S"
)


def command_prefix_sites(rel: str) -> List[Dict]:
    """The command-prefix form of the same defect."""
    out = []
    for i, line in enumerate(read_lines(REPO_ROOT / rel), 1):
        s = line.strip()
        if s.startswith("#") or s.startswith("log ") or s.startswith("echo "):
            continue
        m = CMD_PREFIX_RE.match(line)
        if m:
            out.append({"file": rel, "line": i, "source": s})
    return out


# --------------------------------------------------------------------------
# file helpers
# --------------------------------------------------------------------------
def tracked_sh() -> List[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.sh"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\n") if p]


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").split("\n")


def head_lines(rel: str) -> Optional[List[str]]:
    """The file's content at HEAD (the pre-repair baseline), or None."""
    proc = subprocess.run(
        ["git", "show", "HEAD:" + rel],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.split("\n")


def find_exports(lines: List[str]) -> List[Tuple[int, str, str]]:
    """-> [(line_no (1-based), full_line, rhs), ...]"""
    out = []
    for i, line in enumerate(lines, 1):
        m = EXPORT_RE.match(line)
        if m:
            out.append((i, line, m.group("rhs").strip()))
    return out


def prelude_for(lines: List[str], upto: int, inner: str) -> List[str]:
    """Assignment lines needed to evaluate ``inner`` (e.g. ``GPU=...``).

    The export line's default word may reference a variable the launcher
    defines earlier (``GPU="${GPU:-0}"``).  The proof harness must define it
    exactly as the launcher does, or the "unset -> default" case would
    measure the harness instead of the launcher.
    """
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", inner)) - {CVD}
    prelude: List[str] = []
    for name in sorted(names):
        found = None
        for i, line in enumerate(lines[: upto - 1], 1):
            m = ASSIGN_RE.match(line)
            if m and m.group("name") == name:
                found = line
        if found is not None:
            prelude.append(found)
    return prelude


# --------------------------------------------------------------------------
# the three-case proof
# --------------------------------------------------------------------------
def run_cases(export_line: str, prelude: List[str], default_probe: str) -> Dict:
    """Run the three cases with the export line EXTRACTED from the file.

    a) ``env -u CUDA_VISIBLE_DEVICES``  -> the default must survive
    b) ``CUDA_VISIBLE_DEVICES=""``       -> the lock must survive (empty)
    c) ``CUDA_VISIBLE_DEVICES=1``        -> an explicit choice must win
    """
    # NB: the printf format string is built by concatenation, never by `%`
    # interpolation — a `%s` inside the format would be eaten by the
    # interpolation and the probe would print the variable NAME.
    show = 'printf "[%s]\\n" "$' + CVD + '"'
    script = "\n".join(prelude + [export_line, show])
    cases = [
        ("a_unset_keeps_default", ["env", "-u", CVD, "bash", "-c", script]),
        ("b_empty_lock_is_respected", ["env", CVD + "=", "bash", "-c", script]),
        ("c_explicit_choice_wins", ["env", CVD + "=1", "bash", "-c", script]),
    ]
    out: Dict[str, Dict] = {}
    for name, cmd in cases:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        value = proc.stdout.strip()
        out[name] = {
            "command": " ".join(cmd[:4]) + " '<script>'",
            "stdout": proc.stdout,
            "value": value,
            "rc": proc.returncode,
            "stderr": proc.stderr.strip()[:400],
        }
    # the expected default is measured independently of the export line
    dscript = "\n".join(
        prelude + ['printf "[%s]\\n" ' + default_probe]
    )
    dproc = subprocess.run(
        ["env", "-u", CVD, "bash", "-c", dscript],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    expected_default = dproc.stdout.strip()
    out["expected_default"] = expected_default
    out["_expected_default_value"] = expected_default.strip("[]")
    return out


# --------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------
def repair_file(rel: str, dry_run: bool = False) -> List[Dict]:
    """Repair every defeater export line in one file.  -> list of site dicts."""
    path = REPO_ROOT / rel
    lines = read_lines(path)
    sites: List[Dict] = []
    out_lines: List[str] = []
    changed = False
    for i, line in enumerate(lines, 1):
        m = EXPORT_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        rhs = m.group("rhs").strip()
        verdict, detail = classify_export(rhs)
        if verdict != DEFEATER:
            out_lines.append(line)
            sites.append(
                {
                    "file": rel,
                    "line": i,
                    "original_line": line,
                    "rhs": rhs,
                    "verdict": verdict,
                    "detail": detail,
                    "action": "left alone (not a defeater)",
                }
            )
            continue
        inner = unquote(rhs)
        new_rhs = repaired_rhs(inner)
        indent = m.group("indent")
        dw = default_word(inner)
        new_line = "%sexport %s=%s" % (indent, CVD, new_rhs)
        comment = indent + COMMENT.format(default=dw)
        out_lines.append(comment)
        out_lines.append(new_line)
        changed = True
        sites.append(
            {
                "file": rel,
                "line": i,
                "original_line": line,
                "rhs": rhs,
                "verdict": verdict,
                "detail": detail,
                "action": "repaired",
                "new_line": new_line,
                "new_rhs": new_rhs,
                "comment": comment,
                "default_word": dw,
            }
        )
    if changed and not dry_run:
        content = "\n".join(out_lines)
        sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))
        from wf386a3_atomic_write import atomic_write_file, safe_overwrite_check  # noqa

        ratio = safe_overwrite_check(path, content)
        atomic_write_file(path, content)
        sites[0]["_overwrite_diff_ratio"] = ratio
    return sites


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the repairs")
    ap.add_argument(
        "--out", default=str(OUT_DIR / "proof.json"), help="proof artefact path"
    )
    args = ap.parse_args()

    files = tracked_sh()
    all_sites: List[Dict] = []
    for rel in files:
        if rel in SKIP:
            continue
        all_sites.extend(repair_file(rel, dry_run=not args.apply))

    # ---- the census baseline is HEAD (the pre-repair state) ----------------
    # Provenance is read from git so the artefact is stable across re-runs:
    # `from` = the line at HEAD, `to` = the line on disk now.
    repair_sites: List[Dict] = []
    unrepaired: List[Dict] = []
    safe_sites: List[Dict] = []
    head_export_total = 0
    for rel in files:
        head = head_lines(rel)
        if head is None:
            continue
        for ln, full, rhs in find_exports(head):
            head_export_total += 1
            if rel in SKIP:
                continue
            verdict, detail = classify_export(rhs)
            if verdict != DEFEATER:
                safe_sites.append(
                    {
                        "file": rel,
                        "line": ln,
                        "source": full.strip(),
                        "verdict": verdict,
                        "reason": "lock-safe — an export that can only TIGHTEN the "
                        "mask (it forces the blank), never re-enable a GPU",
                    }
                )
                continue
            now = read_lines(REPO_ROOT / rel)
            now_exports = find_exports(now)
            now_line = None
            for nln, nfull, nrhs in now_exports:
                if classify_export(nrhs)[0] != DEFEATER:
                    now_line = (nln, nfull)
                    break
            if now_line is None:
                unrepaired.append(
                    {
                        "file": rel,
                        "line": ln,
                        "source": full.strip(),
                        "reason": "still a defeater on disk",
                    }
                )
                continue
            inner = unquote(rhs)
            repair_sites.append(
                {
                    "file": rel,
                    "line_head": ln,
                    "line_now": now_line[0],
                    "from": full.strip(),
                    "to": now_line[1].strip(),
                    "head_verdict": verdict,
                    "head_detail": detail,
                    "default_word": default_word(inner),
                }
            )

    # ---- proof: re-read every file from disk and exercise the real line ----
    entries: List[Dict] = []
    for rel in files:
        path = REPO_ROOT / rel
        if not path.exists():
            entries.append(
                {"file": rel, "status": "not applicable", "reason": "file missing"}
            )
            continue
        lines = read_lines(path)
        exports = find_exports(lines)
        if not exports:
            entries.append(
                {
                    "file": rel,
                    "status": "not applicable",
                    "reason": "no `export CUDA_VISIBLE_DEVICES=` line",
                    "n_export_lines": 0,
                }
            )
            continue
        if rel in SKIP:
            entries.append(
                {
                    "file": rel,
                    "status": "not applicable",
                    "reason": SKIP[rel],
                    "n_export_lines": len(exports),
                    "exports": [
                        {"line": ln, "source": full, "verdict": classify_export(r)[0]}
                        for ln, full, r in exports
                    ],
                }
            )
            continue
        site_entries = []
        for ln, full, rhs in exports:
            verdict, detail = classify_export(rhs)
            inner = unquote(rhs)
            prelude = prelude_for(lines, ln, inner)
            # the default probe evaluates the ORIGINAL right-hand side (the
            # launcher's own default) with the ambient variable unset.
            dw = default_word(inner)
            probe = dw if dw != inner else inner
            cases = run_cases(full, prelude, probe)
            exp_default = cases.pop("_expected_default_value")
            a = cases["a_unset_keeps_default"]["value"]
            b = cases["b_empty_lock_is_respected"]["value"]
            c = cases["c_explicit_choice_wins"]["value"]
            ok = (
                a == "[%s]" % exp_default
                and b == "[]"
                and c == "[1]"
                and cases["a_unset_keeps_default"]["rc"] == 0
                and cases["b_empty_lock_is_respected"]["rc"] == 0
                and cases["c_explicit_choice_wins"]["rc"] == 0
            )
            site_entries.append(
                {
                    "line": ln,
                    "extracted_line": full,
                    "rhs": rhs,
                    "verdict": verdict,
                    "detail": detail,
                    "prelude": prelude,
                    "expected_default": exp_default,
                    "cases": cases,
                    "all_three_pass": ok,
                }
            )
        entries.append(
            {
                "file": rel,
                "status": "applicable",
                "n_export_lines": len(exports),
                "exports": site_entries,
                "all_three_pass": all(s["all_three_pass"] for s in site_entries),
            }
        )

    n_applicable = sum(1 for e in entries if e["status"] == "applicable")
    n_na = sum(1 for e in entries if e["status"] == "not applicable")
    n_pass = sum(1 for e in entries if e.get("all_three_pass"))
    cmd_prefix: List[Dict] = []
    for rel in files:
        cmd_prefix.extend(command_prefix_sites(rel))
    artifact = {
        "wf": "Wf-A165 branch A",
        "generated": "2026-11-01",
        "title": "shell launchers that hard-override the GPU mask — repair + three-case proof",
        "tool": {
            "script": "molmetal/reports/wfa165_shell_launchers/wfa165_shell_launcher_repair.py",
            "enumeration": "git ls-files -- '*.sh' (NEVER os.walk)",
            "python": sys.version.split()[0],
            "cwd": str(REPO_ROOT),
            "gpu": "CPU-ONLY — bash variable expansion; torch never imported; no GPU memory allocated",
            "applied": bool(args.apply),
        },
        "census": {
            "tracked_sh_files_scanned": len(files),
            "files_with_export_cvd_total": sum(
                1 for e in entries if e.get("n_export_lines", 0)
            ),
            "files_with_export_cvd_proved": n_applicable,
            "files_without_export_cvd": n_na,
            "export_lines_at_head": head_export_total,
            "export_lines_on_disk": sum(
                e.get("n_export_lines", 0) for e in entries
            ),
            "defeater_sites_at_head": len(repair_sites) + len(unrepaired),
            "defeater_sites_repaired": len(repair_sites),
            "defeater_sites_unrepaired": unrepaired,
            "sites_left_alone": [
                {
                    "file": k,
                    "source": None,
                    "verdict": "skipped",
                    "reason": v,
                }
                for k, v in SKIP.items()
            ]
            + safe_sites,
            "repair_sites": repair_sites,
            "command_prefix_census": {
                "note": (
                    "A command-PREFIX assignment (`CUDA_VISIBLE_DEVICES=0 cmd`) "
                    "is a FOURTH mechanism with the same effect as an export "
                    "override, but it is not an `export` line — it is OUTSIDE "
                    "branch A's mechanism (1)/(2) repair list and is NOT "
                    "repaired here. CENSUSED so the finding is durable."
                ),
                "sites_total": len(cmd_prefix),
                "in_scripts_live_launchers": len([c for c in cmd_prefix if c["file"].startswith("scripts/")]),
                "in_todo_archive_read_only": len([c for c in cmd_prefix if c["file"].startswith("TODO/")]),
                "in_results_artifacts": len([c for c in cmd_prefix if c["file"].startswith("results/")]),
                "sites": cmd_prefix,
            },
        },
        "proof_summary": {
            "files_proved": n_applicable,
            "files_all_three_pass": n_pass,
            "files_any_fail": n_applicable - n_pass,
            "all_three_pass": n_pass == n_applicable,
        },
        "files": entries,
    }
    out_path = Path(args.out)
    sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))
    from wf386a3_atomic_write import atomic_write_file  # noqa

    atomic_write_file(out_path, json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    print(
        "scanned %d tracked .sh | export lines %d | defeaters repaired %d | "
        "proved %d files, all-3-pass %d, any-fail %d"
        % (
            len(files),
            artifact["census"]["export_lines_at_head"],
            artifact["census"]["defeater_sites_repaired"],
            n_applicable,
            n_pass,
            n_applicable - n_pass,
        )
    )
    print("artefact:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
