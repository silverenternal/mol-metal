#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""CLI↔YAML equivalence probe (Wf-A269 wave-19, 2026-11-03).

One question, one answer: **does `--config <cfg>` deliver the same value, of
the same type, to every argparse dest that the OLD command line delivered?**

Why this exists. AGENTS.md §5.11 says a launch wrapper may pass only
``--config`` / ``--out-dir`` / ``--ckpt-dir`` to a trainer; every
hyperparameter must live in the YAML. Migrating a wrapper to that shape is
easy to get wrong in a way that is INVISIBLE to every gate in this repo:

* a flag dropped during the migration silently changes the run;
* a list-valued dest (``--metals`` is ``nargs="+"``) written as a YAML string
  (``metals: Ru Ir Os Re``) resolves, passes both §5.11 gates, and then matches
  zero metals (measured 2026-10-09 — the run died in ``statistics.pstdev``);
* a YAML leaf that resolves to no dest is dropped by the loader with a warning
  on stderr, so the run proceeds with the argparse DEFAULT while the file
  claims otherwise.

None of those are visible in the gate output. This probe is the check that
makes a migration auditable: it runs the trainer's OWN parser over the OLD
argv and over the YAML, and compares dest by dest — value AND type.

How it gets a parser. The repo's trainers build their parser in three
different shapes (``_parse_args(argv)``, ``parse_args()`` reading ``sys.argv``,
and ``main(argv)`` building it inline). Importing the module and calling
``main()`` would START A TRAINING RUN — forbidden (0 GPU) and slow. So the
probe captures the parser the same way ``scripts/_config_binding_gate.py``
does: it spies on ``argparse.ArgumentParser.parse_args`` and aborts the call
before the body of ``main`` runs. The captured parser is then driven directly,
which is a pure parse: no GPU, no data, no side effects.

The YAML namespace is computed the way the trainers compute it:
``apply_yaml_overrides(parser, ns, ns.config)`` (AGENTS.md §5.11).

Stdout invariant (AGENTS.md §4): PASS / FAIL. Exit 0 = every dest equal,
1 = at least one dest differs or is missing, 2 = the probe could not run
(trainer will not import, no parser, no config).

Usage::

    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/_yaml_equivalence_probe.py \\
        --trainer molmetal/scripts/train_metal_lomo_wasserstein.py \\
        --config configs/wfa269_metal_lomo.yaml \\
        --old-argv-json '["--metals", "Ru", "--epochs", "50", "--hidden", "64"]'

    # or from a file holding the JSON list:
    ... --old-argv-file /tmp/old_argv.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Stop(Exception):
    """Abort the trainer's parse so no training body is ever reached."""


def _import(script_path: Path):
    name = "wfa269eq_" + script_path.stem
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        spec.loader.exec_module(mod)
    return mod


def capture_parser(script_path: Path):
    """Return the trainer's ArgumentParser without running its main body.

    Returns ``(parser, argv_used)``. ``argv_used`` is the argv the trainer
    itself would have parsed (usually ``[]``), so a caller can reproduce the
    trainer's own default namespace.
    """
    mod = _import(script_path)
    captured: dict = {}

    def make():
        def spy(self, args=None, namespace=None):
            captured["parser"] = self
            captured["argv"] = args
            raise _Stop()

        return spy

    import argparse as _ap

    originals = {
        "parse_args": _ap.ArgumentParser.parse_args,
        "parse_known_args": _ap.ArgumentParser.parse_known_args,
    }
    old_argv = sys.argv
    try:
        _ap.ArgumentParser.parse_args = make()
        _ap.ArgumentParser.parse_known_args = make()
        sys.argv = [str(script_path)]
        for attr in ("_parse_args", "parse_args"):
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            try:
                if len(inspect.signature(fn).parameters):
                    fn([])
                else:
                    fn()
            except BaseException:  # noqa: BLE001 — SystemExit/_Stop both fine
                pass
            if "parser" in captured:
                return captured["parser"]
        if hasattr(mod, "main"):
            try:
                mod.main([])
            except BaseException:  # noqa: BLE001
                pass
        if "parser" in captured:
            return captured["parser"]
    finally:
        sys.argv = old_argv
        _ap.ArgumentParser.parse_args = originals["parse_args"]
        _ap.ArgumentParser.parse_known_args = originals["parse_known_args"]
    raise RuntimeError("could not capture a parser namespace")


def cli_namespace(script_path: Path, argv: list[str]):
    """Namespace the trainer's parser produces for ``argv`` (no side effects)."""
    parser = capture_parser(script_path)
    return parser.parse_args(list(argv))


def yaml_namespace(script_path: Path, config_path: Path):
    """Namespace the trainer produces for ``--config <config_path>``.

    Mirrors what the trainers do: parse with ``--config`` on the command line,
    then let ``apply_yaml_overrides`` layer the YAML on top
    (AGENTS.md §5.11 — explicit CLI flag > YAML > argparse default).
    """
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    parser = capture_parser(script_path)
    ns = parser.parse_args(["--config", str(config_path)])
    return apply_yaml_overrides(parser, ns, str(config_path))


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def compare(script_path: Path, old_argv: list[str], config_path: Path) -> dict:
    """Compare the CLI namespace with the YAML namespace, dest by dest."""
    cli_ns = cli_namespace(script_path, old_argv)
    yml_ns = yaml_namespace(script_path, config_path)

    rows = []
    n_equal = 0
    n_diff = 0
    for dest, old_value in sorted(vars(cli_ns).items()):
        if dest == "config":
            continue
        if dest.startswith("yaml_"):
            continue
        if not hasattr(yml_ns, dest):
            rows.append(
                {
                    "dest": dest,
                    "old": _jsonable(old_value),
                    "new": None,
                    "equal": False,
                    "reason": "the YAML namespace has no such dest",
                }
            )
            n_diff += 1
            continue
        new_value = getattr(yml_ns, dest)
        same_value = old_value == new_value
        same_type = type(old_value) is type(new_value)
        if isinstance(old_value, (list, tuple)) and isinstance(new_value, (list, tuple)):
            same_type = same_type or (
                isinstance(old_value, list) and isinstance(new_value, list)
            )
        equal = bool(same_value and same_type)
        n_equal += int(equal)
        n_diff += int(not equal)
        rows.append(
            {
                "dest": dest,
                "old": _jsonable(old_value),
                "new": _jsonable(new_value),
                "old_type": type(old_value).__name__,
                "new_type": type(new_value).__name__,
                "equal": equal,
                "reason": None if equal else ("value differs" if not same_value else "type differs"),
            }
        )

    # A skipped YAML key is a DEFECT only when it has no documented
    # non-trainer consumer.  The loader reports metadata-section keys
    # (`entry.script` and friends) in the same list, and those are by design
    # (AGENTS.md §5.11's three dispositions).  Classifying them with the
    # loader's own helper keeps the probe's verdict and the gate's verdict
    # about the same thing.
    from molmetal.scripts import wfA2_config_loader as L

    unresolved = []
    for entry in sorted(
        getattr(yml_ns, "yaml_unresolved_keys", []) or [], key=lambda e: str(e.get("key", ""))
    ):
        if L._registered_non_trainer(entry):
            continue
        unresolved.append(entry)

    return {
        "trainer": str(script_path),
        "config": str(config_path),
        "old_argv": list(old_argv),
        "n_dests_compared": len(rows),
        "n_equal": n_equal,
        "n_different": n_diff,
        "yaml_unresolved_keys": unresolved,
        "rows": rows,
        "verdict": "PASS" if (n_diff == 0 and not unresolved) else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trainer", required=True, help="repo-relative path to the trainer")
    ap.add_argument("--config", required=True, help="repo-relative path to the YAML config")
    ap.add_argument("--old-argv-json", default=None, help="JSON list of the OLD argv")
    ap.add_argument("--old-argv-file", default=None, help="file holding the JSON list")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--only-flags",
        action="store_true",
        help="compare only the dests the OLD argv touched (default: every dest)",
    )
    args = ap.parse_args(argv)

    if args.old_argv_file and args.old_argv_json:
        print("[yaml-equivalence] FAIL: --old-argv-json and --old-argv-file are exclusive")
        return 2
    if args.old_argv_file:
        old_argv = json.loads(Path(args.old_argv_file).read_text(encoding="utf-8"))
    elif args.old_argv_json:
        old_argv = json.loads(args.old_argv_json)
    else:
        print("[yaml-equivalence] FAIL: one of --old-argv-json / --old-argv-file is required")
        return 2

    trainer = Path(args.trainer)
    if not trainer.is_absolute():
        trainer = REPO_ROOT / trainer
    config = Path(args.config)
    if not config.is_absolute():
        config = REPO_ROOT / config
    if not trainer.exists():
        print(f"[yaml-equivalence] FAIL: trainer not found: {trainer}")
        return 2
    if not config.exists():
        print(f"[yaml-equivalence] FAIL: config not found: {config}")
        return 2

    try:
        report = compare(trainer, list(old_argv), config)
    except Exception as exc:  # noqa: BLE001
        print(f"[yaml-equivalence] FAIL: {type(exc).__name__}: {exc}")
        return 2

    if args.only_flags:
        touched = set()
        for token in old_argv:
            if token.startswith("--"):
                touched.add(token[2:].replace("-", "_"))
        report["rows"] = [r for r in report["rows"] if r["dest"] in touched]
        report["n_dests_compared"] = len(report["rows"])
        report["n_equal"] = sum(1 for r in report["rows"] if r["equal"])
        report["n_different"] = sum(1 for r in report["rows"] if not r["equal"])
        report["verdict"] = (
            "PASS" if (report["n_different"] == 0 and not report["yaml_unresolved_keys"])
            else "FAIL"
        )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"[yaml-equivalence] trainer={report['trainer']}")
        print(f"[yaml-equivalence] config={report['config']}")
        print(f"[yaml-equivalence] dests compared = {report['n_dests_compared']}")
        for row in report["rows"]:
            if not row["equal"]:
                print(
                    f"  DIFF {row['dest']}: old={row['old']!r} ({row.get('old_type')}) "
                    f"new={row['new']!r} ({row.get('new_type')}) — {row.get('reason')}"
                )
        if report["yaml_unresolved_keys"]:
            print(f"  UNRESOLVED YAML KEYS: {report['yaml_unresolved_keys']}")
        print(f"[yaml-equivalence] {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
