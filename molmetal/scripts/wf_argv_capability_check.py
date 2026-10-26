#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Does a trainer's argparse declare the flags a launcher is about to pass?

WHY (2026-10-27).  Three separate waves have now lost runs to the same defect:
a launcher builds an argv the trainer cannot parse, argparse exits 2 with
"unrecognized arguments", and the wrapper either dies before the first epoch or
— worse — backgrounds a pid and reports success.  MEASURED instances:

  * ``launch_wf124_p3_g5.sh`` / ``_g6.sh`` passed ``--num-workers`` and
    ``--no-cuda-graphs``; fixed in Wf-r28 / Wf-R32.
  * ``launch_wf310b_g6_final.sh``, ``retry_g5_missing_pairs.sh``,
    ``recover_g6_failed_tuples.sh`` still passed ``--no-cuda-graphs``, and the
    last two also passed ``--num-workers``.  ``recover_g6_failed_tuples.sh`` is
    the RECOVERY path for the G6 failures, so it could not recover anything.
  * ``gpu_smoke_w2_sliced.sh`` passed five flags that do not exist
    (``--smoke``, ``--num-workers``, ``--torch-compile-mode-v2``,
    ``--no-cuda-graphs``, ``--bf16``) and died immediately after printing a pid.
  * ``launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`` passes fourteen,
    including ``--mode joint_tmqm_platinai`` and ``--per-cell-w2-out``, which
    no script in this repository implements.

The grep that would have caught this ("does the launcher mention the flag?")
does not work, because a comment mentioning the flag looks identical to a live
argument.  This asks the trainer's own parser instead.

How a flag is classified, and why it is done this way
-----------------------------------------------------
Each flag is fed to the trainer's ``parse_args`` on its own:

  * exits with "unrecognized arguments"  -> NOT DECLARED (a real finding);
  * exits with "expected one argument"   -> declared, takes a value (fine);
  * exits with a bad ``choices`` value   -> declared (fine);
  * returns normally                     -> declared boolean (fine).

Passing a value is deliberately avoided: guessing the arity would mean
maintaining a second copy of the parser.  "Unrecognized" is the only outcome
that means the launcher cannot run.

Usage
-----
    python molmetal/scripts/wf_argv_capability_check.py \\
        --trainer molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --flags "--no-cuda-graphs --num-workers"

Stdout invariant: ``PASS`` / ``FAIL`` (AGENTS.md §4).  Exit 0=PASS, 1=FAIL,
2=SKIP (trainer has no importable ``parse_args``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

_UNRECOGNIZED = "unrecognized arguments"


def load_parse_args(trainer: Path):
    """Import ``trainer`` by path and return its ``parse_args``."""
    spec = importlib.util.spec_from_file_location("_cap_check_trainer", trainer)
    if spec is None or spec.loader is None:
        raise SystemExit(2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cap_check_trainer"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "parse_args", None)
    if fn is None:
        raise SystemExit(2)
    return fn


def flag_is_declared(parse_args, flag: str) -> bool:
    """True unless the parser answers "unrecognized arguments: <flag>"."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            parse_args([flag])
    except SystemExit:
        return _UNRECOGNIZED not in buf.getvalue()
    return True


def check(trainer: Path, flags: List[str]) -> Tuple[List[str], List[str]]:
    parse_args = load_parse_args(trainer)
    seen: List[str] = []
    missing: List[str] = []
    for flag in flags:
        if flag in seen:
            continue
        seen.append(flag)
        if not flag_is_declared(parse_args, flag):
            missing.append(flag)
    return seen, missing


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer", required=True, type=Path)
    ap.add_argument(
        "--flags",
        action="append",
        default=[],
        help="Whitespace-separated flags to check; repeatable. A `--x=value` "
        "token is reduced to `--x`. `--flag --x` cannot be used directly "
        "because argparse reads `--x` as an option, not as a value.",
    )
    args = ap.parse_args(argv)

    trainer = args.trainer
    if not trainer.is_absolute():
        trainer = REPO_ROOT / trainer
    if not trainer.exists():
        print(f"FAIL — trainer not found: {trainer}", file=sys.stderr)
        return 1

    flags = [
        tok.split("=", 1)[0]
        for chunk in args.flags
        for tok in chunk.split()
        if tok.startswith("--")
    ]
    try:
        seen, missing = check(trainer, flags)
    except SystemExit:
        print(f"SKIP — {trainer.name} has no importable parse_args", file=sys.stderr)
        return 2

    for flag in missing:
        print(f"MISSING {flag}")
    if missing:
        print(
            f"FAIL — {len(missing)} of {len(seen)} flag(s) are not declared by "
            f"{trainer.name}: {', '.join(missing)}"
        )
        return 1
    print(f"PASS — all {len(seen)} flag(s) are declared by {trainer.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
