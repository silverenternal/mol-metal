#!/usr/bin/env python
"""AGENTS.md §5.7 — a trainer whose flag surface changed needs a fresh smoke.

WHY THIS EXISTS (BLK-I4 finding F2)
-----------------------------------
§5.7 says a new trainer ``--flag`` is INCOMPLETE until a 50-step end-to-end
smoke has run. The project already built the instrument for that
(``molmetal/scripts/wf399a1_smoke_protocol.py::run_smoke_50step``), but when the
BLK-I4 hygiene audit measured it, the helper had **zero callers repo-wide** — so
§5.7 was prose. F2's cheapest real fix is the enforcement half:

    for every trainer whose flag surface differs from HEAD (a ``--flag`` was
    added or removed), require a smoke artifact NEWER than the trainer file.

That is the same failure class the project has already been bitten by: Wf-385-A1
found 17 declared flags that were never wired into the loop, and every unit test
was green because no test ever ran the trainer end to end.

WHAT COUNTS AS A SMOKE ARTIFACT
-------------------------------
``molmetal/reports/smoke_artifacts/<trainer_stem>.json``, written by

    python molmetal/scripts/wf399a1_smoke_protocol.py \
        --trainer <trainer> [--flag-arg ...] --out-json <that path>

It must be NEWER than the trainer and carry ``smoke.returncode == 0``. A stale
file, a missing file, or a non-zero run all FAIL. The metrics payload and the
helper's own ``verdict.pass`` are deliberately NOT required — see
:func:`artifact_is_fresh_and_passing` for why requiring either would fail real
smokes (CFM writes ``per_epoch_metrics.json``; classifiers have no
``decode_ratio``).

SCOPE / HONEST LIMITS
---------------------
* Only ``molmetal/scripts/train_*.py`` are trainers.
* The comparison is against ``HEAD``. A change committed *without* a smoke is
  not retroactively caught — this gate catches the *change*, which is the point
  at which §5.7 applies. (Once committed, HEAD moves and the delta closes.)
* If ``git`` cannot read the path at HEAD (new trainer) every flag counts as
  added, so a brand-new trainer needs a smoke before it can be committed.
* No git / detached/no-HEAD environment: the trainer is skipped with a note
  rather than failed, so the gate cannot red on an environment it cannot read.

EXIT CODES
----------
0 = every changed trainer has a fresh, passing smoke artifact (or nothing
    changed).
1 = at least one changed trainer does not.

CPU-only; imports nothing heavy.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINERS_GLOB = "molmetal/scripts/train_*.py"
SMOKE_DIR = REPO_ROOT / "molmetal" / "reports" / "smoke_artifacts"

# argparse registrations look like  add_argument(\n    "--update",  or
# add_argument("--epochs", type=int)  or  add_argument("-x", "--xlong").
# \s* spans the newline in the multi-line form.  We capture the first one or two
# POSITIONAL string args (the flag names) and keep only the long form — that is
# what lets `# see --not-a-real-flag` in a comment not count, while
# `add_argument("-x", "--xlong")` still yields `--xlong`.
_FLAG_RE = re.compile(
    r"add_argument\(\s*[\"']([^\"']+)[\"']"
    r"(?:\s*,\s*[\"']([^\"']+)[\"'])?"
)


def flag_surface(source: str) -> set:
    """The set of long ``--flag`` names a trainer source registers."""
    out: set = set()
    for first, second in _FLAG_RE.findall(source):
        for tok in (first, second):
            if tok and tok.startswith("--"):
                out.add(tok)
    return out


def head_flag_surface(rel_path: str) -> set | None:
    """Flag surface at HEAD, or ``None`` if git cannot answer.

    ``None`` (not ``set()``) means "unknown" and is treated as a skip, so an
    environment without git does not manufacture a FAIL.
    """
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        # Path absent at HEAD -> a NEW trainer.  Every flag is new; the empty
        # set is the correct "before" state, NOT "unknown".
        if "does not exist" in proc.stderr or "exists on disk" in proc.stderr:
            return set()
        if "unknown revision" in proc.stderr or "ambiguous argument" in proc.stderr:
            return None
        return set()
    return flag_surface(proc.stdout)


def smoke_artifact_path(trainer: Path) -> Path:
    return SMOKE_DIR / f"{trainer.stem}.json"


def artifact_is_fresh_and_passing(artifact: Path, trainer: Path) -> tuple[bool, str]:
    """(ok, reason) for one trainer's smoke artifact.

    The required evidence is deliberately trainer-agnostic (F2's literal ask is
    "a smoke artifact newer than the trainer"):

    * the artifact must exist and be NEWER than the trainer, and
    * its recorded run must have exited 0 (``smoke.returncode == 0``).

    The metrics dict is NOT required: ``wf399a1_smoke_protocol`` locates a
    trainer's metrics via ``<out-dir>/metrics.json`` or a stdout banner, but
    the CFM trainer writes ``per_epoch_metrics.json``, so a legitimate CFM smoke
    can carry an empty metrics dict. Requiring metrics would fail a real smoke
    for a file-naming reason. ``wf399a1_smoke_protocol``'s own ``verdict.pass``
    is likewise NOT required — its decode_ratio/loss-curve invariants are
    decoder-specific and unsatisfiable for classification trainers.
    """
    if not artifact.is_file():
        return False, "no smoke artifact on disk"
    if artifact.stat().st_mtime <= trainer.stat().st_mtime:
        return False, "smoke artifact is OLDER than the trainer"
    try:
        doc = json.loads(artifact.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"smoke artifact unreadable: {exc}"
    smoke = doc.get("smoke") or {}
    if smoke.get("returncode") != 0:
        return False, f"smoke artifact smoke.returncode={smoke.get('returncode')!r}"
    return True, "fresh + ran to completion (rc=0)"


def evaluate() -> Dict[str, object]:
    """Return a structured report; ``ok`` is the gate verdict."""
    findings: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    changed_count = 0

    for trainer in sorted(REPO_ROOT.glob(TRAINERS_GLOB)):
        rel = trainer.relative_to(REPO_ROOT).as_posix()
        head_flags = head_flag_surface(rel)
        if head_flags is None:
            skipped.append({"trainer": rel, "why": "git cannot read HEAD"})
            continue
        now_flags = flag_surface(trainer.read_text())
        if now_flags == head_flags:
            continue
        changed_count += 1
        added = sorted(now_flags - head_flags)
        removed = sorted(head_flags - now_flags)
        artifact = smoke_artifact_path(trainer)
        ok, reason = artifact_is_fresh_and_passing(artifact, trainer)
        findings.append(
            {
                "trainer": rel,
                "added_flags": added,
                "removed_flags": removed,
                "artifact": artifact.relative_to(REPO_ROOT).as_posix(),
                "ok": ok,
                "reason": reason,
            }
        )

    failures = [f for f in findings if not f["ok"]]
    return {
        "n_trainers": sum(1 for _ in REPO_ROOT.glob(TRAINERS_GLOB)),
        "n_changed": changed_count,
        "findings": findings,
        "skipped": skipped,
        "ok": not failures,
    }


def _remediation(finding: Dict[str, object]) -> str:
    return (
        f"run: python molmetal/scripts/wf399a1_smoke_protocol.py "
        f"--trainer {finding['trainer']} --out-json {finding['artifact']}"
    )


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)

    report = evaluate()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    for f in report["findings"]:
        if f["ok"]:
            print(
                f"[smoke-freshness] OK   {f['trainer']} "
                f"(+{len(f['added_flags'])}/-{len(f['removed_flags'])} flag(s)) "
                f"-> {f['artifact']} ({f['reason']})"
            )
        else:
            print(
                f"[smoke-freshness] FAIL {f['trainer']} "
                f"(+{f['added_flags']}/-{f['removed_flags']}) "
                f"-> {f['reason']}.  {_remediation(f)}"
            )
    for s in report["skipped"]:
        print(f"[smoke-freshness] SKIP {s['trainer']} ({s['why']})")

    if report["ok"]:
        print(
            f"[smoke-freshness] PASS — {report['n_changed']} trainer(s) with a "
            f"changed flag surface; all have a fresh, passing §5.7 smoke."
        )
        return 0
    n_fail = sum(1 for f in report["findings"] if not f["ok"])
    print(
        f"[smoke-freshness] FAIL — {n_fail} changed trainer(s) lack a fresh "
        f"§5.7 smoke artifact."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
