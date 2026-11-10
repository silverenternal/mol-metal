#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
"""Fail-closed auditor for a STATIC per-(metal, seed) launch config.

WHY THIS EXISTS (AGENTS.md §5.11, user directive 2026-10-09, verbatim
"以后要求能且只能用yaml配置启动训练和推理，no cli参数"):

A launch wrapper may hand the trainer ONLY ``--config`` / ``--out-dir`` /
``--ckpt-dir`` / ``--help``.  Every hyperparameter — including the sweep
dimensions ``metals`` and ``seeds`` — must live in the YAML.

The D-MPNN trainer loops over ``args.metals`` x ``args.seeds`` INSIDE one
process (``train_dmpnn_multitask_wasserstein.py:4352`` and ``:4356``), so a
wrapper that runs one PROCESS per (metal, seed) pair needs a config that
declares exactly ONE metal and ONE seed.  That config must be a FILE ON DISK:
generating it at launch time would make the run's hyperparameters unreviewable
before the run and unreproducible afterwards.

This helper is the guard that keeps a static config honest.  It answers, for
one config (or for a whole cohort of them):

  1. does the file exist and parse as a YAML mapping?
  2. does its ``entry.script`` name the trainer that is about to run?
  3. does every hyperparameter the wrapper still keeps a shell record of
     (``--flags``) hold the SAME value in the YAML that the trainer will
     actually read?  The comparison is made through the trainer's OWN
     argparse parser and the REAL loader (``apply_yaml_overrides``), so the
     flag -> dest -> YAML-leaf resolution is the trainer's, not this
     script's.
  4. does the config pin exactly ONE metal and ONE seed, and are they the
     ones the wrapper asked for?
  5. do the ``--expect section.key=value`` dotted paths hold?
  6. did the glob match at least ``--expect-min-count`` files?  A launcher
     that enumerates N tuples must find N static configs -- a glob that
     silently matched fewer would under-launch.
  7. does each config carry the ablation lever of the ARM ITS FILE NAME
     names?  ``--arm-flags no_wass=--wasserstein-weight 0.0`` adds that
     lever to the audited flag set for every config whose name carries the
     ``no_wass`` component -- so one trainer import audits all four arms.

Every check is FAIL-CLOSED: any mismatch exits 1 with the differing dest
named.  A wrapper that runs this before it launches a sweep turns "the
shell record and the YAML drifted apart" into a loud abort instead of a
silent run with the wrong hyperparameter.

Usage::

    .venv/bin/python scripts/_wf_yaml_static_config_c1.py \\
        --trainer molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --config configs/wfa269_c1_dmpnn_pair_sweep_g5_Ru_seed0.yaml \\
        --flags "--epochs 50 --batch-size 256" \\
        --metal Ru --seed 0 \\
        [--expect train.epochs=50]

    # audit a whole cohort in ONE process (one trainer import, not one
    # per config):
    ... --config-glob "configs/wfa269_c1_dmpnn_pair_sweep_g5_*.yaml" --flags "..."

    # audit every arm of a 60-tuple ablation in ONE process:
    ... --config-glob "configs/..._g6_*.yaml" --flags "<COMMON_ARGS>" \\
        --arm-flags "no_multi_task=--single-task" \\
        --arm-flags "no_wass=--wasserstein-weight 0.0" \\
        --arm-flags "no_kendall=--no-kendall" \\
        --arm-flags "no_dann=--no-dann" \\
        --expect-min-count 60

Stdout invariant (AGENTS.md §4): PASS / FAIL.  Exit 0 = every check held,
1 = at least one mismatch, 2 = the auditor could not run.
"""

from __future__ import annotations

import argparse
import glob as globmod
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_probe():
    """Import the repo's own equivalence probe (read-only; not modified)."""
    path = REPO_ROOT / "scripts/_yaml_equivalence_probe.py"
    spec = importlib.util.spec_from_file_location("wfa269_probe_static_cfg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yaml_namespace(parser, config_path: Path):
    """The namespace the trainer gets from ``--config <config_path>``.

    Mirrors what the trainers do: parse with ``--config`` on the command
    line, then let ``apply_yaml_overrides`` layer the YAML on top.  ``argv=[]``
    is passed explicitly so that no dest counts as "explicitly supplied on
    the command line" — i.e. the YAML is the single source of truth, which is
    exactly the situation this auditor models.
    """
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    ns = parser.parse_args(["--config", str(config_path)])
    return apply_yaml_overrides(parser, ns, str(config_path), argv=[])


def _flag_dests(parser, flag_tokens: List[str]) -> List[str]:
    """The argparse dests the flag tokens name (order preserved, deduped)."""
    dests: List[str] = []
    for tok in flag_tokens:
        if not tok.startswith("--"):
            continue
        action = parser._option_string_actions.get(tok)  # noqa: SLF001
        if action is None:
            raise SystemExit(f"[static-config] FAIL: unknown trainer flag {tok!r}")
        if action.dest not in dests:
            dests.append(action.dest)
    return dests


def _same_value(old: Any, new: Any) -> bool:
    if type(old) is not type(new):
        if not (isinstance(old, (list, tuple)) and isinstance(new, (list, tuple))):
            return False
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        return list(old) == list(new)
    return old == new


def _get_path(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def audit_config(
    parser,
    config_path: Path,
    flag_tokens: List[str],
    metal: Optional[str],
    seed: Optional[int],
    expectations: List[str],
    trainer_rel: str,
) -> Tuple[List[str], int]:
    """Audit one config.  Returns (problems, n_flags_verified)."""
    problems: List[str] = []

    if not config_path.exists():
        return ([f"config file missing: {config_path}"], 0)

    import yaml

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(raw_text) or {}
    except Exception as exc:  # noqa: BLE001 — any YAML error is a FAIL
        return ([f"config does not parse as YAML: {exc}"], 0)
    if not isinstance(cfg, dict):
        return (["config does not parse as a YAML mapping"], 0)

    # --- entry.script must name the trainer that is about to run ----------
    entry = cfg.get("entry")
    if isinstance(entry, dict) and entry.get("script"):
        if str(entry["script"]) != trainer_rel:
            problems.append(
                f"entry.script={entry['script']!r} != trainer {trainer_rel!r}"
            )

    yml_ns = _yaml_namespace(parser, config_path)

    # --- the shell record must agree with what the trainer will read ------
    n_flags = 0
    if flag_tokens:
        cli_ns = parser.parse_args(list(flag_tokens))
        for dest in _flag_dests(parser, flag_tokens):
            n_flags += 1
            old = getattr(cli_ns, dest, None)
            new = getattr(yml_ns, dest, None)
            if not _same_value(old, new):
                problems.append(
                    f"{dest}: shell-record={old!r} != yaml={new!r} "
                    f"(types {type(old).__name__}/{type(new).__name__})"
                )

    # --- the cohort must be exactly ONE metal and ONE seed ----------------
    yml_metals = getattr(yml_ns, "metals", None)
    yml_seeds = getattr(yml_ns, "seeds", None)
    if not isinstance(yml_metals, (list, tuple)) or len(yml_metals) != 1:
        problems.append(f"metals={yml_metals!r} is not a one-element list")
    if not isinstance(yml_seeds, (list, tuple)) or len(yml_seeds) != 1:
        problems.append(f"seeds={yml_seeds!r} is not a one-element list")
    if metal is not None and isinstance(yml_metals, (list, tuple)):
        if list(yml_metals) != [metal]:
            problems.append(f"metals={list(yml_metals)!r} != [{metal!r}]")
    if seed is not None and isinstance(yml_seeds, (list, tuple)):
        if list(yml_seeds) != [seed]:
            problems.append(f"seeds={list(yml_seeds)!r} != [{seed}]")
    if metal is None and seed is None and isinstance(yml_metals, (list, tuple)):
        if len(yml_metals) == 1 and isinstance(yml_seeds, (list, tuple)):
            if len(yml_seeds) == 1:
                # the file name must name the tuple it carries
                stem = config_path.stem
                m = re.search(r"_([A-Z][a-z]?)_seed(\d+)$", stem)
                if m:
                    if m.group(1) != str(yml_metals[0]):
                        problems.append(
                            f"file name names metal {m.group(1)!r} but the config "
                            f"carries {yml_metals[0]!r}"
                        )
                    if int(m.group(2)) != int(yml_seeds[0]):
                        problems.append(
                            f"file name names seed {m.group(2)} but the config "
                            f"carries {yml_seeds[0]!r}"
                        )

    # --- the house fail-closed dotted-path expectations --------------------
    for item in expectations:
        if "=" not in item:
            raise SystemExit(f"[static-config] FAIL: --expect expects key=value, got {item!r}")
        dotted, expected = item.split("=", 1)
        key = dotted.strip()
        actual = _get_path(cfg, key)
        if actual is None:
            problems.append(f"{key}: MISSING from {config_path.name} (expected {expected!r})")
        elif str(actual) != expected:
            problems.append(f"{key}: yaml={actual!r} != expected {expected!r}")

    return (problems, n_flags)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trainer", required=True, help="repo-relative path to the trainer")
    ap.add_argument("--config", help="path to ONE static config")
    ap.add_argument(
        "--config-glob",
        help="glob of static configs to audit in one process (mutually "
        "exclusive with --config)",
    )
    ap.add_argument(
        "--flags",
        default="",
        help="the shell record of the hyperparameters, e.g. "
        "'--epochs 50 --batch-size 256'; each flag must hold the same value "
        "in the YAML",
    )
    ap.add_argument("--metal", help="the ONE metal this config must carry")
    ap.add_argument("--seed", type=int, help="the ONE seed this config must carry")
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="fail-closed dotted-path check against the raw YAML (repeatable)",
    )
    ap.add_argument(
        "--expect-min-count",
        type=int,
        default=0,
        help="fail-closed lower bound on the number of configs the glob "
        "matched.  A launcher that enumerates N tuples must find N static "
        "configs, not 'whatever happens to be on disk' — a glob that silently "
        "matches fewer files than the sweep needs would under-launch.",
    )
    ap.add_argument(
        "--arm-flags",
        action="append",
        default=[],
        metavar="ARM=FLAG...",
        help="repeatable per-arm extra flags, e.g. "
        "'no_wass=--wasserstein-weight 0.0'.  The arm is matched against the "
        "file name's underscore-delimited components, so the arm's lever is "
        "audited against the config that actually carries it — without "
        "needing one trainer import per arm.",
    )
    args = ap.parse_args(argv)

    if bool(args.config) == bool(args.config_glob):
        print("[static-config] FAIL: pass exactly one of --config / --config-glob")
        return 2

    trainer = Path(args.trainer)
    if not trainer.is_absolute():
        trainer = REPO_ROOT / trainer
    if not trainer.exists():
        print(f"[static-config] FAIL: trainer not found: {trainer}")
        return 2
    trainer_rel = str(Path(args.trainer))

    if args.config_glob:
        configs = sorted(Path(p) for p in globmod.glob(str(REPO_ROOT / args.config_glob)))
        if not configs:
            print(f"[static-config] FAIL: --config-glob matched nothing: {args.config_glob}")
            return 1
        if args.expect_min_count and len(configs) < args.expect_min_count:
            print(
                f"[static-config] FAIL: --config-glob matched {len(configs)} config(s) "
                f"but the sweep needs at least {args.expect_min_count}: "
                f"{args.config_glob}"
            )
            return 1
    else:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        configs = [cfg_path]

    try:
        probe = _load_probe()
        parser = probe.capture_parser(trainer)
    except Exception as exc:  # noqa: BLE001
        print(f"[static-config] FAIL: could not capture the trainer parser: {exc}")
        return 2

    flag_tokens = args.flags.split()

    # ---- per-arm extra flags (the ablation lever) ------------------------
    arm_flags: List[Tuple[str, List[str]]] = []
    for item in args.arm_flags:
        if "=" not in item:
            raise SystemExit(
                f"[static-config] FAIL: --arm-flags expects ARM=FLAG..., got {item!r}"
            )
        arm, _, rest = item.partition("=")
        arm = arm.strip()
        tokens = rest.split()
        if not arm or not tokens:
            raise SystemExit(
                f"[static-config] FAIL: --arm-flags expects ARM=FLAG..., got {item!r}"
            )
        arm_flags.append((arm, tokens))

    n_fail = 0
    n_flags_total = 0
    for cfg_path in configs:
        metal, seed = args.metal, args.seed
        if args.config_glob and (metal is not None or seed is not None):
            # in cohort mode the tuple is read from the config itself
            metal, seed = None, None
        tokens = list(flag_tokens)
        if arm_flags:
            parts = cfg_path.stem.split("_")
            for arm, extra in arm_flags:
                if arm in parts:
                    tokens.extend(extra)
        problems, n_flags = audit_config(
            parser, cfg_path, tokens, metal, seed, list(args.expect), trainer_rel
        )
        n_flags_total += n_flags
        if problems:
            n_fail += 1
            print(f"[static-config] FAIL {cfg_path}:")
            for problem in problems:
                print(f"    {problem}")
        else:
            print(f"[static-config] PASS {cfg_path} (flags verified: {n_flags})")

    if n_fail:
        print(
            f"[static-config] FAIL: {n_fail}/{len(configs)} config(s) failed the audit"
        )
        return 1
    print(
        f"[static-config] PASS: {len(configs)} config(s) audited, "
        f"{n_flags_total} flag dests verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
