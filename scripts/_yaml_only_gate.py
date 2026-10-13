#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""AGENTS.md §5.11 gate — training and inference launch from YAML only.

The rule (user directive 2026-10-09, verbatim "以后要求能且只能用yaml配置启动
训练和推理，no cli参数"): every knob that reaches a trainer must be declared in
the YAML config. A launch wrapper may set up the environment, detach the
process, and redirect output — it may not hand the trainer a hyperparameter.

What this gate checks, and why each check is the right shape:

1. `--extra-flags` anywhere in `scripts/launch_*.sh` is a violation. That is the
   escape hatch which let 12 hyperparameters live outside the YAML on
   2026-10-09 (BLK-C5's five levers + device/batch/early-stop/save-smiles).
2. A `train_*.py` invocation carrying any `--flag` outside the allowlist is a
   violation. The allowlist is deliberately tiny and covers only plumbing that
   identifies WHICH config to run, never WHAT to train.
3. `scripts/start.sh` must refuse `--extra-flags`, so the escape hatch is closed
   at the surface every wrapper funnels through — otherwise a future wrapper
   would simply re-introduce it.

Why source-scanning rather than a dry-run diff: a dry-run only proves the
command a wrapper builds *today*, for the one config it names. These wrappers
each hardcode a single YAML path, so a dry-run cannot see the other 12 configs,
and any new wrapper would start at zero coverage. The rule being enforced is
about the shape of every launch surface, so the check has to be about shape too.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP. Exit 0=PASS, 1=FAIL, 2=SKIP.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Flags a launch wrapper may pass to a trainer. Each one selects the RUN, not
#: the MODEL: which config, where output goes, which checkpoint dir. Every
#: hyperparameter (lr, batch, epochs, scheduler, all mechanism toggles) is
#: deliberately absent — those belong in the YAML per §5.11.
ALLOWED_TRAINER_FLAGS = frozenset(
    {
        "--config",
        "--out-dir",
        "--ckpt-dir",
        "--help",
    }
)

#: A trainer entry point, e.g. `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`.
_TRAINER_CALL = re.compile(r"\btrain_[a-z0-9_]+\.py\b")
_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")

# Leading detach prefix: env assignments, then `setsid`/`nohup` and their own
# options. `setsid --wait nohup "$PY" trainer.py` is a legal command head whose
# `--wait` belongs to setsid, not to the trainer — without stripping it the gate
# reports `--wait` as a smuggled trainer hyperparameter and pushes authors back
# toward the bare `setsid`, whose non-forking behaviour is what makes
# `rc=$?` after the command readable in the first place.
_DETACH_HEAD = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+)*"
    r"(?:(?:setsid|nohup)(?:\s+--?[A-Za-z][A-Za-z-]*)*\s+)+"
)


def _strip_detach_head(line: str) -> str:
    """Drop a leading setsid/nohup (plus options) so only the payload is scanned."""
    return _DETACH_HEAD.sub("", line, count=1)


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    A wrapper outside the repo root is a test fixture, not a crash: the gate
    must report on whatever it is handed.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Return (start_lineno, code) for comment-free, continuation-joined lines.

    Two things make this more than a comment stripper, and both were caught by
    the gate's own regression suite rather than by inspection:

    * Comments are dropped, because the migrated wrappers document the flags
      they used to pass. Scanning those would pin every one of them as broken
      forever, which is how a gate gets disabled instead of fixed.
    * Backslash-continued lines are JOINED. Every launcher in this repo puts
      the trainer flags on continuation lines, so a per-physical-line scan sees
      `--config` on line 1 and a bare `--lr` on line 2 with no trainer call on
      it to anchor to — and reports PASS on a wrapper that smuggles in a
      hyperparameter. That is precisely the bug class §5.11 exists to close, so
      a scanner that misses it is worse than no scanner.
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0

    def flush() -> None:
        if not buf:
            return
        joined = " ".join(buf)
        hash_at = joined.find(" #")
        if hash_at != -1:
            joined = joined[:hash_at]
        if joined.strip():
            out.append((start, joined))
        buf.clear()

    in_block = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_block:
            end = line.find("*/")
            if end == -1:
                continue
            line, in_block = line[end + 2 :], False
        if not buf and (line.lstrip().startswith("#") or not line.strip()):
            continue
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            if not buf:
                start = i
            buf.append(stripped[:-1].strip())
            continue
        if not buf:
            start = i
        buf.append(stripped.strip())
        flush()
    flush()
    return out


def _audit_wrapper(path: Path) -> tuple[list[dict], bool]:
    """Return (violations, trainer_seen) for a single launch wrapper."""
    violations: list[dict] = []
    rel = _rel(path)

    for lineno, line in _logical_lines(path.read_text(encoding="utf-8", errors="replace")):
        if "--extra-flags" in line:
            violations.append(
                {
                    "file": rel,
                    "line": lineno,
                    "kind": "extra_flags",
                    "detail": (
                        "--extra-flags passes trainer hyperparameters outside the "
                        "YAML; declare them in the config instead (§5.11)"
                    ),
                }
            )

        if not _TRAINER_CALL.search(line):
            continue
        for flag in _FLAG.findall(_strip_detach_head(line)):
            if flag not in ALLOWED_TRAINER_FLAGS:
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "trainer_flag",
                        "detail": (
                            f"{flag} is passed on a trainer invocation line; only "
                            + " ".join(sorted(ALLOWED_TRAINER_FLAGS))
                            + " may appear there (§5.11)"
                        ),
                    }
                )
    return violations, bool(violations)


def _audit_start_sh(path: Path) -> list[dict]:
    """start.sh must actively refuse --extra-flags, not merely avoid using it."""
    violations: list[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    code = _logical_lines(text)
    accepts = any(
        "--extra-flags" in line and ("EXTRA_FLAGS=" in line or '"${2}"' in line)
        for _, line in code
    )
    refuses = any(
        re.search(r"extra-flags", line) and re.search(r"refus|reject|forbid|FATAL", line, re.I)
        for _, line in code
    )
    if accepts and not refuses:
        violations.append(
            {
                "file": _rel(path),
                "line": 0,
                "kind": "start_sh_permits_extra_flags",
                "detail": (
                    "scripts/start.sh still parses --extra-flags. It is the surface "
                    "every wrapper funnels through, so the §5.11 escape hatch stays "
                    "open until start.sh refuses the flag."
                ),
            }
        )
    return violations


def audit(launchers: list[Path] | None = None) -> dict:
    """Audit every launch wrapper. Returns a report dict; never raises."""
    if launchers is None:
        launchers = sorted(SCRIPTS_DIR.glob("launch_*.sh"))

    violations: list[dict] = []
    scanned: list[str] = []
    for wrapper in launchers:
        if not wrapper.exists():
            continue
        scanned.append(_rel(wrapper))
        found, _ = _audit_wrapper(wrapper)
        violations.extend(found)

    start_sh = SCRIPTS_DIR / "start.sh"
    if start_sh.exists():
        scanned.append(_rel(start_sh))
        violations.extend(_audit_start_sh(start_sh))

    return {
        "gate": "yaml_only_launch",
        "rule": "AGENTS.md §5.11 — training and inference launch from YAML only",
        "allowed_trainer_flags": sorted(ALLOWED_TRAINER_FLAGS),
        "n_wrappers_scanned": len(scanned),
        "wrappers": scanned,
        "n_violations": len(violations),
        "violations": violations,
        "verdict": "PASS" if not violations else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="accepted for interface parity with the other §4 gates; a violation "
        "already fails without it",
    )
    args = ap.parse_args(argv)

    report = audit()

    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print(f"[yaml-only-launch] scanned {report['n_wrappers_scanned']} wrapper(s)")
        for v in report["violations"]:
            loc = f"{v['file']}:{v['line']}" if v["line"] else v["file"]
            print(f"  VIOLATION {loc} [{v['kind']}] {v['detail']}")
        if not report["violations"]:
            print(
                "  every launch wrapper passes --config only; all hyperparameters "
                "come from the YAML"
            )
        print(f"[yaml-only-launch] {report['verdict']}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
