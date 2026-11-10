#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Derive a per-(metal, seed) launch config from a base YAML config.

WHY THIS EXISTS (AGENTS.md §5.11, user directive 2026-10-09, verbatim
"以后要求能且只能用yaml配置启动训练和推理，no cli参数"):

A launch wrapper may pass only ``--config`` / ``--out-dir`` / ``--ckpt-dir`` /
``--help`` to a trainer.  Every hyperparameter — including the per-pair sweep
dimensions ``metals`` and ``seeds`` — must therefore live in the YAML.

The sweep dimensions are the one case that CANNOT be satisfied by a single
static config: the LOMO / G6 trainers loop internally over
``args.metals`` x ``args.seeds`` (train_metal_lomo_wasserstein.py:4352 /
:4356), so a config that declares the full cohort would run N folds in ONE
process, while the wrapper today runs N processes (one per (metal, seed)
pair, each with its own ``--out-dir`` / ``--out-metric``).  Moving the cohort
into the base config would therefore change WHAT THE RUN DOES (process count,
per-pair output layout, per-pair log files, resume semantics).

So the cohort stays per-pair, and this helper writes the per-pair config that
carries it.  The wrapper then passes ONLY:

    <venv python> <trainer> --config <derived> --out-dir <dir> [--ckpt-dir <dir>]

which is exactly the shape AGENTS.md §5.11 permits.

THE DERIVATION IS VALUE-PRESERVING BY CONSTRUCTION: it starts from the base
config (which already carries every hyperparameter the wrapper used to pass)
and replaces ONLY the sweep dimensions plus whatever the caller explicitly
overrides.  Every other key is copied through untouched, so the trainer reads
the same values it read before the migration.

The write is atomic (temp file + ``os.replace`` in the same directory), per
AGENTS.md §5.6.

USAGE
-----
    .venv/bin/python scripts/_wf_yaml_pair_config.py \\
        --base configs/<base>.yaml \\
        --out   <derived>.yaml \\
        --metal Ru --seed 0 \\
        [--set section.key=<yaml-value>] ... \\
        [--expect section.key=<shell-value>] ...

``--set`` values are parsed with ``yaml.safe_load``, so ``true`` is a boolean,
``3`` is an integer and ``/a/b`` is a string.

``--expect`` is a FAIL-CLOSED consistency check, not an override: the base
config's value at that dotted path must already equal the value given on the
command line, or the helper exits non-zero.  It exists so a wrapper that keeps
a shell constant for display (a plan record, a file-existence check) can
prove that constant still agrees with the config the trainer actually reads.
§5.11 makes the config the single source of truth for every value that
reaches a trainer; this is what makes a stale shell copy loud instead of
merely stale.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for a dotted path, creating dicts."""
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        raise SystemExit("empty key in --set")
    node = cfg
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def derive(
    base: Path,
    metal: str,
    seed: int,
    overrides: List[str],
) -> Dict[str, Any]:
    """Return the derived config mapping for one (metal, seed) pair."""
    cfg = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"base config is not a YAML mapping: {base}")

    # The sweep dimensions.  Both are LIST-valued argparse dests
    # (nargs="+"), so they MUST be YAML sequences -- a bare string would
    # resolve and then match zero rows (AGENTS.md §5.11, 2026-10-09).
    #
    # `data.metals` and `protocol.seeds` are the two-component leaf paths
    # `_LEAF_FALLBACK_SECTIONS` allows (wfA2_config_loader._candidate_dests):
    # `data` and `protocol` are both in the allowlist, so the 2-component leaf
    # form resolves to the `metals` / `seeds` dests.
    _set_path(cfg, "data.metals", [metal])
    _set_path(cfg, "protocol.seeds", [int(seed)])

    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _set_path(cfg, dotted.strip(), value)
    return cfg


def get_path(cfg: Dict[str, Any], dotted: str) -> Any:
    """Return the value at ``a.b.c`` in ``cfg``, or ``None`` when absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_expectations(
    cfg: Dict[str, Any], expectations: List[str], base: Path
) -> None:
    """Fail closed unless every ``--expect section.key=value`` holds.

    The comparison is ``str(yaml_value) == shell_value`` so an int in the
    config matches the same int rendered by the shell (``50`` == ``"50"``).
    A missing path is a mismatch: the config must NAME the value.
    """
    problems: List[str] = []
    for item in expectations:
        if "=" not in item:
            raise SystemExit(f"--expect expects section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        key = dotted.strip()
        actual = get_path(cfg, key)
        if actual is None:
            problems.append(f"{key}: MISSING from {base} (shell expects {raw!r})")
        elif str(actual) != raw:
            problems.append(f"{key}: yaml={actual!r} != shell={raw!r}")
    if problems:
        raise SystemExit(
            "CONFIG MISMATCH between the shell constants and "
            f"{base}:\n  " + "\n  ".join(problems)
        )


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via temp-file + rename (AGENTS.md §5.6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", suffix=".part", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="base YAML config to derive from")
    ap.add_argument("--out", required=True, help="derived YAML config to write")
    ap.add_argument("--metal", required=True, help="the ONE metal for this run")
    ap.add_argument("--seed", required=True, help="the ONE seed for this run")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="extra override applied after the sweep dimensions (repeatable); "
        "VALUE is parsed as YAML",
    )
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="fail-closed consistency check: the base config's value at "
        "SECTION.KEY must already equal VALUE (repeatable)",
    )
    args = ap.parse_args(argv)

    base = Path(args.base)
    cfg = derive(
        base,
        args.metal,
        int(args.seed),
        list(args.set or []),
    )
    check_expectations(cfg, list(args.expect or []), base)
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True)
    write_atomic(Path(args.out), text)
    print(str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
