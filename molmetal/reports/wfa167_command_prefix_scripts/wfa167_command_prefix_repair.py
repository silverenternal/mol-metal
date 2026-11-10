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
"""Wf-A167 branch 1 — COMMAND-PREFIX ``CUDA_VISIBLE_DEVICES=...`` assignments.

THE DEFECT (the FOURTH mechanism Wf-A165 did not cover)
-------------------------------------------------------
Wf-A165 repaired three ways a shell launcher could re-enable a GPU behind the
session's ambient ``CUDA_VISIBLE_DEVICES=""`` lock:

  (1) ``export CUDA_VISIBLE_DEVICES=0``            (hard override)
  (2) ``${VAR:-default}``                          (``:`` also fires on EMPTY)
  (3) the yaml-derived form in ``scripts/start.sh``

This module covers the fourth: the **command-prefix assignment**

    CUDA_VISIBLE_DEVICES=0 setsid nohup ${PY} ${SCRIPT}

It is not an ``export`` line, so Wf-A165's export repair cannot see it — yet
in bash an assignment that precedes the *command word* sets the variable in
the child's environment FOR THAT ONE COMMAND, silently replacing the ambient
``""`` lock with a real device.

THE LOCK SEMANTICS (the acceptance tests)
------------------------------------------
For the repaired form ``CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-<D>}"``:

  * ambient UNSET  -> the script's ORIGINAL default ``<D>`` applies
    (behaviour byte-for-byte identical to today);
  * ambient ``""`` -> the lock WINS, the command sees an EMPTY mask;
  * ambient ``"1"``-> the explicit caller choice WINS, the command sees ``1``.

The ``-`` (hyphen, no colon) is what makes the difference: ``${VAR-default}``
substitutes only when the variable is UNSET, while ``${VAR:-default}``
substitutes when it is unset **OR EMPTY** — the colon form is the original
defect.

CLASSIFICATION VOCABULARY (used by the census and by the regression test)
---------------------------------------------------------------------------
``REAL_DEFEATER``       command-prefix assignment whose value is NOT
                        lock-preserving — it FORCES a device.
``LOCK_PRESERVING``     command-prefix assignment of the form
                        ``"${CUDA_VISIBLE_DEVICES-...}"`` — benign.
``EXPORT_ASSIGNMENT``   ``export CUDA_VISIBLE_DEVICES=...`` — Wf-A165's
                        mechanism, out of scope for this branch.
``ARGV_ASSIGNMENT``     the assignment appears AFTER a command word, so bash
                        passes it as an ARGUMENT (``setsid nohup
                        CUDA_VISIBLE_DEVICES=0 cmd`` -> nohup tries to exec a
                        program literally named ``CUDA_VISIBLE_DEVICES=0``).
                        NOT a device-forcing defeater; a different defect class.
``BENIGN_COMMENT``      the occurrence is inside a ``#`` comment.
``BENIGN_QUOTED``       the occurrence is inside a quoted string (a log/echo
                        message, or an argument such as a device-query
                        ``-i <index>``).
``BENIGN_USE``          a plain use (``${CUDA_VISIBLE_DEVICES}`` in a test).
``ARCHIVED``            the file lives under ``TODO/archive/`` or ``results/``.

CPU-ONLY
--------
Stdlib only.  The proof harness is bash variable expansion against a stub
command that prints the mask it received.  torch is never imported, no CUDA
context is created, no GPU memory is allocated, and no launcher is executed.
File enumeration goes through ``git ls-files`` (NEVER ``os.walk`` — this repo
is ~201 GB).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
CVD = "CUDA_VISIBLE_DEVICES"
PROTOOL = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))
from wf386a3_atomic_write import atomic_write_file, safe_overwrite_check  # noqa: E402

# ---------------------------------------------------------------------------
# directories that are NOT live code
# ---------------------------------------------------------------------------
ARCHIVED_PREFIXES = ("TODO/archive/", "results/")

#: tokens that may legally precede a command-prefix assignment
PREFIX_ALLOWED_PREV = {
    None, ";", "&", "|", "(", "{", "!", "&&", "||", ";;",
    "then", "else", "elif", "do", "if", "while", "until", "time",
    "command", "exec", "env", "sudo",
}

REAL_DEFEATER = "REAL_DEFEATER"
LOCK_PRESERVING = "LOCK_PRESERVING"
EXPORT_ASSIGNMENT = "EXPORT_ASSIGNMENT"
ARGV_ASSIGNMENT = "ARGV_ASSIGNMENT"
BENIGN_COMMENT = "BENIGN_COMMENT"
BENIGN_QUOTED = "BENIGN_QUOTED"
BENIGN_USE = "BENIGN_USE"


# ---------------------------------------------------------------------------
# low-level scanning
# ---------------------------------------------------------------------------
def _states(line: str) -> List[str]:
    """Per-character state: 'code' | 'single' | 'double' | 'comment'."""
    states: List[str] = []
    state = "code"
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if state == "code":
            if ch == "\\" and i + 1 < n:
                states.append("code")
                states.append("code")
                i += 2
                continue
            if ch == "#" and (i == 0 or line[i - 1] in " \t"):
                state = "comment"
                states.append("comment")
                i += 1
                continue
            if ch == "'":
                state = "single"
                states.append("single")
                i += 1
                continue
            if ch == '"':
                state = "double"
                states.append("double")
                i += 1
                continue
            states.append("code")
            i += 1
            continue
        if state == "single":
            states.append("single")
            if ch == "'":
                state = "code"
            i += 1
            continue
        if state == "double":
            if ch == "\\" and i + 1 < n:
                states.append("double")
                states.append("double")
                i += 2
                continue
            states.append("double")
            if ch == '"':
                state = "code"
            i += 1
            continue
        states.append("comment")
        i += 1
    return states


def _prev_token(line: str, start: int) -> Optional[str]:
    """The whitespace-delimited token immediately before ``start``."""
    i = start - 1
    while i >= 0 and line[i] in " \t":
        i -= 1
    if i < 0:
        return None
    if line[i] in ";&|(){}":
        j = i
        while j >= 0 and line[j] in ";&|(){}":
            j -= 1
        return line[j + 1 : i + 1]
    j = i
    while j >= 0 and line[j] not in " \t;&|(){}":
        j -= 1
    return line[j + 1 : i + 1]


def _value_end(line: str, eq: int) -> int:
    """End index (exclusive) of the assignment value that starts after ``eq``."""
    i = eq + 1
    n = len(line)
    if i >= n:
        return i
    if line[i] in ("'", '"'):
        quote = line[i]
        j = i + 1
        while j < n:
            if line[j] == "\\" and quote == '"':
                j += 2
                continue
            if line[j] == quote:
                return j + 1
            j += 1
        return n
    j = i
    while j < n and line[j] not in " \t":
        if line[j] == "\\" and j + 1 < n:
            j += 2
            continue
        j += 1
    return j


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def classify_value(value: str) -> str:
    """Is this assigned value lock-preserving?

    ``LOCK_PRESERVING`` — the value references ``CUDA_VISIBLE_DEVICES`` with
    the ``-`` operator, so an explicitly-set ambient value (including the
    empty string) survives.
    ``DEFEATER``       — anything else.
    """
    inner = _unquote(value)
    if inner == "":
        return LOCK_PRESERVING
    if inner.startswith("${") and inner.endswith("}"):
        depth = 0
        close = -1
        for i, ch in enumerate(inner):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close == len(inner) - 1:
            name = inner[2:close]
            if name.startswith(CVD + "-"):
                return LOCK_PRESERVING
    return REAL_DEFEATER


def scan_line(line: str) -> List[Dict]:
    """Every ``CUDA_VISIBLE_DEVICES`` occurrence on one shell line."""
    out: List[Dict] = []
    states = _states(line)
    start = 0
    while True:
        idx = line.find(CVD, start)
        if idx < 0:
            break
        start = idx + len(CVD)
        state = states[idx]
        rec: Dict = {"start": idx, "state": state}
        if state == "comment":
            rec["classification"] = BENIGN_COMMENT
            out.append(rec)
            continue
        if state in ("single", "double"):
            rec["classification"] = BENIGN_QUOTED
            out.append(rec)
            continue
        after = idx + len(CVD)
        if after >= len(line) or line[after] != "=":
            rec["classification"] = BENIGN_USE
            out.append(rec)
            continue
        # --- it is an assignment -------------------------------------------
        vend = _value_end(line, after)
        value = line[after + 1 : vend]
        token_start = idx
        token_end = vend
        prev = _prev_token(line, token_start)
        rec.update(
            {
                "assignment_end": token_end,
                "value": value,
                "value_inner": _unquote(value),
                "prev_token": prev,
                "token": line[token_start:token_end],
            }
        )
        if prev == "export":
            rec["classification"] = EXPORT_ASSIGNMENT
        elif prev in PREFIX_ALLOWED_PREV:
            rec["classification"] = (
                LOCK_PRESERVING
                if classify_value(value) == LOCK_PRESERVING
                else REAL_DEFEATER
            )
        else:
            rec["classification"] = ARGV_ASSIGNMENT
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
def tracked_sh() -> List[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", "*.sh"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\n") if p]


def is_archived(rel: str) -> bool:
    return rel.startswith(ARCHIVED_PREFIXES)


def read_text(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def census(files: Optional[List[str]] = None) -> Dict:
    """Full occurrence census over the tracked ``*.sh`` tree."""
    files = tracked_sh() if files is None else files
    occurrences: List[Dict] = []
    for rel in files:
        fp = REPO_ROOT / rel
        if not fp.exists():
            continue
        for i, line in enumerate(read_text(rel).split("\n"), 1):
            for rec in scan_line(line):
                occurrences.append(
                    {
                        "file": rel,
                        "line": i,
                        "source": line.rstrip(),
                        "classification": rec["classification"],
                        "archived": is_archived(rel),
                        "value": rec.get("value"),
                        "token": rec.get("token"),
                    }
                )
    return {"tracked_sh_files": len(files), "occurrences": occurrences}


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------
def why_comment(inner: str) -> str:
    return (
        "# Wf-A167: a command-prefix CUDA_VISIBLE_DEVICES assignment defeats the "
        'ambient "" no-GPU lock; an explicit ambient value must win; unset -> '
        + inner
        + "."
    )


def repair_value(value: str) -> str:
    """The lock-preserving replacement for one assigned value."""
    inner = _unquote(value)
    return '"${%s-%s}"' % (CVD, inner)


def repair_file(rel: str, occs: List[Dict]) -> Tuple[str, List[Dict]]:
    """Apply every REAL_DEFEATER in one file.  -> (new_text, applied[])"""
    lines = read_text(rel).split("\n")
    ordered = sorted(occs, key=lambda r: r["line"])
    # descending line order so earlier insertions do not shift later ones
    for rec in reversed(ordered):
        idx = rec["line"] - 1
        line = lines[idx]
        scan = scan_line(line)
        target = [r for r in scan if r["classification"] == REAL_DEFEATER]
        assert len(target) == 1, (rel, rec["line"], [r["classification"] for r in scan])
        t = target[0]
        new_token = "CUDA_VISIBLE_DEVICES=" + repair_value(t["value"])
        new_line = line[: t["start"]] + new_token + line[t["assignment_end"] :]
        indent = line[: len(line) - len(line.lstrip())]
        comment = indent + why_comment(t["value_inner"])
        lines[idx] = new_line
        lines.insert(idx, comment)
        rec["from"] = t["token"]
        rec["to"] = new_token
        rec["why_comment"] = comment
        rec["structure_preserved"] = (
            new_line.replace(new_token, "@@") == line.replace(t["token"], "@@")
        )
    # the final 1-based line number of each site: its own 0-based index + 1
    # (the comment pushed it down by one) + the number of comments inserted
    # above it.
    for i, rec in enumerate(ordered):
        rec["line_now"] = rec["line"] + 1 + i
    return "\n".join(lines), ordered


def bash_n(rel: str) -> Tuple[bool, str]:
    proc = subprocess.run(
        ["bash", "-n", str(REPO_ROOT / rel)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()


def atomic_write(rel: str, text: str) -> Optional[float]:
    path = REPO_ROOT / rel
    risk = safe_overwrite_check(path, text)
    atomic_write_file(path, text)
    return risk


# ---------------------------------------------------------------------------
# three-way proof harness (bash variable expansion only — no GPU, no torch)
# ---------------------------------------------------------------------------
STUB = """#!/usr/bin/env bash
if [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ]; then
    printf 'MASK=[%s]\\n' "${CUDA_VISIBLE_DEVICES}"
else
    printf 'MASK=[<UNSET>]\\n'
fi
"""


def _eval_assignment(token: str, prelude: List[str], ambient: Optional[str],
                     extra_env: Dict[str, str]) -> str:
    """Evaluate one assignment token against the stub under one ambient."""
    with open("/tmp/wfa167_stub.sh", "w", encoding="utf-8") as fh:
        fh.write(STUB)
    os.chmod("/tmp/wfa167_stub.sh", 0o755)
    script = "\n".join(prelude + [token + " /tmp/wfa167_stub.sh"])
    cmd = ["bash", "-c", script]
    if ambient is None:
        cmd = ["env", "-u", CVD] + cmd
    else:
        cmd = ["env", CVD + "=" + ambient] + cmd
    for k, v in extra_env.items():
        cmd = ["env", k + "=" + v] + cmd
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("harness failed: %s / %s" % (proc.stderr[:400], script))
    out = proc.stdout.strip()
    m = re.match(r"^MASK=\[(.*)\]$", out)
    return m.group(1) if m else out


def prove_site(rec: Dict, prelude: List[str], extra_env: Dict[str, str]) -> Dict:
    """The three-way measurement for one repaired site."""
    before = _eval_assignment(rec["from"], prelude, None, extra_env)
    after_unset = _eval_assignment(rec["to"], prelude, None, extra_env)
    after_empty = _eval_assignment(rec["to"], prelude, "", extra_env)
    after_one = _eval_assignment(rec["to"], prelude, "1", extra_env)
    ok = (
        before == after_unset
        and after_empty == ""
        and after_one == "1"
        and rec["structure_preserved"]
    )
    return {
        "ambient_unset_original_mask": before,
        "ambient_unset_repaired_mask": after_unset,
        "ambient_empty_repaired_mask": after_empty,
        "ambient_one_repaired_mask": after_one,
        "structure_preserved": rec["structure_preserved"],
        "all_three_pass": ok,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    print("[wfa167] census ...")
    cen = census()
    occs = cen["occurrences"]
    live_real = [
        o
        for o in occs
        if o["classification"] == REAL_DEFEATER and not o["archived"]
    ]
    print(
        "[wfa167] tracked *.sh = %d ; occurrences = %d ; live REAL_DEFEATER = %d"
        % (cen["tracked_sh_files"], len(occs), len(live_real))
    )
    # The repair below MUTATES the occurrence dicts in place (the line
    # numbers move once the WHY comment is inserted), so the census artefact
    # must be written from a snapshot taken BEFORE the repair.
    census_snapshot = json.loads(json.dumps(occs))
    # Guard against a partial / repeated run silently dropping sites.
    assert len(live_real) == 20, (
        "expected exactly 20 live REAL_DEFEATER sites, found %d — the tree may "
        "already be repaired; restore before re-running" % len(live_real)
    )

    by_file: Dict[str, List[Dict]] = {}
    for o in live_real:
        by_file.setdefault(o["file"], []).append(o)

    # --- diff against Wf-A165's census ------------------------------------
    a165_path = REPO_ROOT / "molmetal" / "reports" / "wfa165_shell_launchers" / "proof.json"
    diff: Dict = {}
    if a165_path.exists():
        a165 = json.loads(a165_path.read_text(encoding="utf-8"))
        a165_sites = a165["census"]["command_prefix_census"]["sites"]
        mine = {(o["file"], o["line"]) for o in live_real}
        theirs = {(s["file"], s["line"]) for s in a165_sites if not is_archived(s["file"])}
        diff = {
            "wfa165_live_sites": len(theirs),
            "wfa165_total_sites": len(a165_sites),
            "in_wfa165_live_not_in_mine": sorted("%s:%d" % (f, l) for f, l in theirs - mine),
            "in_mine_not_in_wfa165_live": sorted(
                "%s:%d" % (f, l) for f, l in mine - theirs
            ),
        }
        print("[wfa167] diff vs Wf-A165: %s" % json.dumps(diff))

    # --- repair -----------------------------------------------------------
    print("[wfa167] repairing %d sites in %d files ..." % (len(live_real), len(by_file)))
    repaired: List[Dict] = []
    file_reports: List[Dict] = []
    for rel in sorted(by_file):
        recs = by_file[rel]
        before_text = read_text(rel)
        before_mode = os.stat(REPO_ROOT / rel).st_mode & 0o7777
        new_text, _ = repair_file(rel, recs)
        risk = atomic_write(rel, new_text)
        after_mode = os.stat(REPO_ROOT / rel).st_mode & 0o7777
        ok, err = bash_n(rel)
        print(
            "[wfa167]   %-58s sites=%d bash_n=%s mode=%s->%s risk=%s"
            % (rel, len(recs), "PASS" if ok else "FAIL", oct(before_mode), oct(after_mode), risk)
        )
        assert ok, "bash -n failed for %s: %s" % (rel, err)
        assert before_mode == after_mode, "mode changed for %s" % rel
        # recompute the post-repair line numbers
        new_lines = new_text.split("\n")
        for rec in recs:
            ln_now = rec["line_now"]
            assert rec["to"] in new_lines[ln_now - 1], (
                "cannot relocate %s:%d -> %d (line is %r)"
                % (rel, rec["line"], ln_now, new_lines[ln_now - 1])
            )
            rec["file"] = rel
            rec["line_before"] = rec["line"]
            rec["line"] = ln_now
            rec["why_comment_line"] = ln_now - 1
            repaired.append(rec)
        file_reports.append(
            {
                "file": rel,
                "sites": len(recs),
                "bash_n": "PASS",
                "mode_before": oct(before_mode),
                "mode_after": oct(after_mode),
                "overwrite_risk_ratio": risk,
            }
        )

    # --- proof ------------------------------------------------------------
    print("[wfa167] three-way proof ...")
    PRELUDE: Dict[str, List[str]] = {
        "scripts/launch_wf146_deferred_runs.sh": ['idx="${gpu##cuda:}"'],
        "scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": ['idx="${gpu##cuda:}"'],
        "scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh": ['idx="${gpu##cuda:}"'],
    }
    EXTRA_ENV: Dict[str, Dict[str, str]] = {
        "scripts/launch_wf146_deferred_runs.sh": {"gpu": "cuda:1"},
        "scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh": {"gpu": "cuda:1"},
        "scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh": {"gpu": "cuda:1"},
        "scripts/launch_wf299_cfm_main.sh": {"gpu": "cuda:1"},
        "scripts/launch_wf310f_frontier_math_gpu.sh": {"gpu": "cuda:1"},
        "scripts/retry_g5_missing_pairs.sh": {"gpu": "1"},
    }
    proof_entries: List[Dict] = []
    for rec in repaired:
        pre = PRELUDE.get(rec["file"], [])
        extra = EXTRA_ENV.get(rec["file"], {})
        res = prove_site(rec, pre, extra)
        entry = {
            "file": rec["file"],
            "line": rec["line_before"],
            "line_now": rec["line"],
            "why_comment_line": rec["why_comment_line"],
            "from": rec["from"],
            "to": rec["to"],
            "source_before": rec["source"],
            "source_now": None,
            "why_comment": rec["why_comment"],
            "prelude": pre,
            "extra_env": extra,
        }
        entry.update(res)
        entry["source_now"] = read_text(rec["file"]).split("\n")[rec["line"] - 1].rstrip()
        proof_entries.append(entry)
        print(
            "[wfa167]   %-58s unset %s->%s | empty %r | one %r | %s"
            % (
                rec["file"],
                res["ambient_unset_original_mask"],
                res["ambient_unset_repaired_mask"],
                res["ambient_empty_repaired_mask"],
                res["ambient_one_repaired_mask"],
                "PASS" if res["all_three_pass"] else "FAIL",
            )
        )

    # --- write artefacts --------------------------------------------------
    PROTOOL.mkdir(parents=True, exist_ok=True)
    proof = {
        "wave": "Wf-A167 branch 1",
        "defect_class": "command-prefix CUDA_VISIBLE_DEVICES=<value> assignment",
        "tag": "MEASURED",
        "census": {
            "tracked_sh_files_scanned": cen["tracked_sh_files"],
            "occurrences_total": len(occs),
            "live_real_defeaters_before_repair": len(live_real),
            "archived_real_defeaters_left_alone": len(
                [
                    o
                    for o in census_snapshot
                    if o["classification"] == REAL_DEFEATER and o["archived"]
                ]
            ),
            "archived_real_defeater_sites": [
                {"file": o["file"], "line": o["line"], "source": o["source"]}
                for o in census_snapshot
                if o["classification"] == REAL_DEFEATER and o["archived"]
            ],
            "diff_vs_wfa165": diff,
            "repair_sites": [
                {
                    "file": e["file"],
                    "line": e["line"],
                    "line_now": e["line_now"],
                    "from": e["from"],
                    "to": e["to"],
                }
                for e in proof_entries
            ],
        },
        "proof_summary": {
            "sites": len(proof_entries),
            "all_three_pass": all(e["all_three_pass"] for e in proof_entries),
            "pass_count": sum(1 for e in proof_entries if e["all_three_pass"]),
            "fail_count": sum(1 for e in proof_entries if not e["all_three_pass"]),
            "files_any_fail": len(
                {e["file"] for e in proof_entries if not e["all_three_pass"]}
            ),
        },
        "sites": proof_entries,
        "files": file_reports,
    }
    atomic_write_file(PROTOOL / "proof.json", json.dumps(proof, indent=2) + "\n")

    census_out = {
        "wave": "Wf-A167 branch 1",
        "tag": "MEASURED",
        "tracked_sh_files_scanned": cen["tracked_sh_files"],
        "occurrences_total": len(census_snapshot),
        "by_classification": {},
        "live_real_defeaters": [
            {"file": o["file"], "line": o["line"], "source": o["source"]}
            for o in census_snapshot
            if o["classification"] == REAL_DEFEATER and not o["archived"]
        ],
        "archived_real_defeaters": [
            {"file": o["file"], "line": o["line"], "source": o["source"]}
            for o in census_snapshot
            if o["classification"] == REAL_DEFEATER and o["archived"]
        ],
        "all_occurrences": census_snapshot,
    }
    for o in census_snapshot:
        key = o["classification"] + ("_archived" if o["archived"] else "_live")
        census_out["by_classification"][key] = (
            census_out["by_classification"].get(key, 0) + 1
        )
    atomic_write_file(PROTOOL / "census.json", json.dumps(census_out, indent=2) + "\n")

    print(
        "[wfa167] proof: %d/%d PASS"
        % (proof["proof_summary"]["pass_count"], proof["proof_summary"]["sites"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
