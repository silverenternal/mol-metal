#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Launch provenance for §5.11 YAML-only launches (Wf-A269, 2026-11-04).

The problem this solves.  AGENTS.md §5.11 exists because "a knob that lives
only in a shell script is a knob nobody audits".  The YAML puts the knob in a
file — but the file records the **declared** value, and nothing in the tree
recorded **what the trainer actually read**.  A config edited after the run
silently turns the declaration into a lie, and no gate can see it.

The mechanism, and where it comes from.  Hydra writes the RESOLVED config into
the run directory (``.hydra/config.yaml``) at launch; SLSA / in-toto wrap that
in a signed statement whose ``externalParameters`` must cover every
externally-controllable influence.  This file is the wrapper-side half of
that idea, and ``molmetal/scripts/wfA2_config_loader.py::write_resolved_config``
is the trainer-side half.

TWO ARTEFACTS, TWO PROCESSES, ONE COMPARISON.

  wrapper process  ──emit──▶  <out-dir>/resolved_config.expected.yaml
                                        <out-dir>/launch_provenance.json
  trainer process  ──(loader)──▶  <out-dir>/resolved_config.actual.yaml

``verify`` compares them.  The check is NOT circular: the expected side is
computed from the wrapper's INTENDED argv, in the wrapper's process; the
actual side is computed from the argv the trainer RECEIVED, in the trainer's
process, by the trainer's OWN parser.  If the wrapper assembles a different
argv than it registered, or the config changes on disk between the two, or an
environment variable differs, the comparison FAILS.  That is the property
that makes it a check and not a transcript.

WHY THIS IS NOT A SLSA L3 BUILD PROVENANCE.  It records the configuration
namespace, the argv, the config's sha256, and the outcome-relevant
environment.  It does NOT record the host state (kernel, driver, GPU, dataset
bytes).  The statement below says so in its own ``honest_limits`` field
rather than letting a reader assume full L3 coverage.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP.  Exit 0 = MATCH,
1 = MISMATCH, 2 = SKIP (an artefact is missing).

Usage::

    # 1. the wrapper registers what it is ABOUT to run
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/_wf_launch_provenance.py emit \\
        --trainer molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --config configs/wfa269_c1_dmpnn_pair_sweep_g5_Ru_seed0.yaml \\
        --out-dir results/run/Ru/seed0 --ckpt-dir results/run/Ru/seed0/ckpts

    # 2. the wrapper launches the trainer with MOLMETAL_RESOLVED_CONFIG_PATH set
    MOLMETAL_RESOLVED_CONFIG_PATH=<out-dir>/resolved_config.actual.yaml ...

    # 3. after the run, the check
    .venv/bin/python scripts/_wf_launch_provenance.py verify --out-dir results/run/Ru/seed0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_NAME = "resolved_config.expected.yaml"
ACTUAL_NAME = "resolved_config.actual.yaml"
PROVENANCE_NAME = "launch_provenance.json"

#: SLSA buildType / builder.id are URIs.  ``.invalid`` is reserved by
#: RFC 2606 precisely so a local build type cannot be mistaken for a
#: resolvable, registered one.
BUILD_TYPE = "https://molmetal.invalid/buildtypes/molmetal-yaml-launch/v1"
BUILDER_ID = "https://molmetal.invalid/builders/molmetal-launch-provenance/v1"
PREDICATE_TYPE = "https://slsa.dev/provenance/v1"
STATEMENT_TYPE = "https://in-toto.io/Statement/v1"

#: The §5.11 legal flag set.  A §5.11-compliant launch passes ONLY these.
LEGAL_FLAGS = ("--config", "--out-dir", "--ckpt-dir")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _capture_parser(trainer: Path):
    """The trainer's own parser, captured without running its main body.

    Reuses the repo's existing probe rather than re-implementing the spy —
    a second copy of this logic would be a second thing to keep correct.
    """
    probe = _load_module(REPO_ROOT / "scripts" / "_yaml_equivalence_probe.py", "wfa269prov_probe")
    return probe.capture_parser(trainer)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _canonical_argv(config: Path, out_dir: Path, ckpt_dir: Optional[Path]) -> list[str]:
    """The argv a §5.11-compliant launch MUST pass — nothing else."""
    argv = ["--config", str(config), "--out-dir", str(out_dir)]
    if ckpt_dir is not None:
        argv += ["--ckpt-dir", str(ckpt_dir)]
    return argv


def _resolve(trainer: Path, config: Path, argv: list[str]) -> dict[str, Any]:
    """The trainer-faithful namespace for ``argv`` — exactly as the trainer
    computes it (parse_args, then the YAML layered over the CLI)."""
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    parser = _capture_parser(trainer)
    ns = parser.parse_args(list(argv))
    ns = apply_yaml_overrides(parser, ns, str(config), argv=list(argv))
    return vars(ns)


def _dump(payload: dict[str, Any]) -> str:
    import yaml

    header = (
        "# Wf-A269 launch provenance (Hydra-style resolved config).\n"
        "# Generated by scripts/_wf_launch_provenance.py.\n"
        "# MEASURED-on-this-run.  Do not hand-edit: the sha256 of this file is\n"
        "# part of what the verification compares.\n"
    )
    return header + yaml.safe_dump(payload, default_flow_style=False, sort_keys=True)


def _load_artifact(path: Path) -> Optional[dict[str, Any]]:
    import yaml

    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def cmd_emit(args: argparse.Namespace) -> int:
    # The ARGV is recorded VERBATIM — the literal strings the wrapper passes.
    # A relative-vs-absolute difference is a STRING difference, not a run
    # difference, and the trainer's argv comparison must not fire on it.  The
    # paths are therefore resolved against the CURRENT WORKING DIRECTORY (the
    # directory the trainer will inherit), not against the repo root: the
    # trainer opens the config the same way.
    cwd = Path.cwd()
    trainer = Path(args.trainer)
    config = Path(args.config)
    out_dir = Path(args.out_dir)
    if not trainer.is_absolute():
        trainer = cwd / trainer
    if not config.is_absolute():
        config = cwd / config
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None
    if ckpt_dir is not None and not ckpt_dir.is_absolute():
        ckpt_dir = cwd / ckpt_dir

    if not trainer.exists():
        print(f"[launch-provenance] FAIL: trainer not found: {trainer}")
        return 2
    if not config.exists():
        print(f"[launch-provenance] FAIL: config not found: {config}")
        return 2

    argv = _canonical_argv(
        Path(args.config), Path(args.out_dir),
        Path(args.ckpt_dir) if args.ckpt_dir else None,
    )
    argv_source = "guessed"
    if getattr(args, "argv_json", None):
        # The wrapper DECLARED the argv it will pass.  This is the real
        # check: the declared argv is compared against the argv the trainer
        # RECEIVED, so a wrapper that assembles something else is caught.
        try:
            declared_argv = json.loads(args.argv_json)
        except Exception as exc:  # noqa: BLE001
            print(f"[launch-provenance] FAIL: --argv-json is not valid JSON: {exc!r}")
            return 2
        if not isinstance(declared_argv, list) or not all(
            isinstance(x, str) for x in declared_argv
        ):
            print("[launch-provenance] FAIL: --argv-json must be a JSON array of strings")
            return 2
        argv = list(declared_argv)
        argv_source = "declared"
    try:
        ns = _resolve(trainer, config, argv)
    except Exception as exc:  # noqa: BLE001
        print(f"[launch-provenance] FAIL: cannot resolve the namespace: {exc!r}")
        return 2

    from molmetal.scripts.wfA2_config_loader import resolved_environment, resolved_namespace
    import argparse as _ap

    namespace = resolved_namespace(_ap.Namespace(**ns))
    payload = {
        "resolved": namespace,
        "argv": argv,
        "argv_source": argv_source,
        "config": {"path": str(config), "sha256": _sha256_file(config)},
        "environment": resolved_environment(),
    }
    text = _dump(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_path = out_dir / EXPECTED_NAME
    expected_path.write_text(text, encoding="utf-8")
    (out_dir / (EXPECTED_NAME + ".sha256")).write_text(
        _sha256_text(text) + "\n", encoding="utf-8"
    )

    # SLSA / in-toto shaped statement.  `externalParameters` is the resolved
    # namespace — every dest the configuration can move.
    statement = {
        "_type": STATEMENT_TYPE,
        "subject": [
            {
                "name": str(config),
                "digest": {"sha256": payload["config"]["sha256"]},
            }
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": BUILD_TYPE,
                "externalParameters": namespace,
                "internalParameters": {
                    "trainer": str(trainer),
                    "argv": argv,
                    # 2026-11-04 (Wf-A269 frontier sweep) — `command` and
                    # `command_source` are NEW.
                    #
                    # PRE-REPAIR RENDERING (PRESERVED VERBATIM per AGENTS.md
                    # §5.5 rule #6 — the SLSA statement carried NO `command`
                    # at all, and the declared-vs-guessed distinction existed
                    # ONLY in the YAML payload's `argv_source`):
                    #
                    #     "internalParameters": {
                    #         "trainer": str(trainer),
                    #         "argv": argv,
                    #         "legal_flags": list(LEGAL_FLAGS),
                    #     },
                    #
                    # WHY: a consumer of the SLSA statement ALONE could not
                    # tell a DECLARED argv from a GUESSED one — the
                    # provenance of the provenance was lost in the SLSA view.
                    # `command` is the trainer path followed by the argv; the
                    # INTERPRETER is deliberately NOT part of it (the wrapper
                    # chooses it and this record does not know it — stated in
                    # `honest_limits.does_not_cover` below).  `command_source`
                    # carries the same "declared"/"guessed" distinction as the
                    # payload's `argv_source`.
                    "command": [str(trainer), *argv],
                    "command_source": argv_source,
                    "legal_flags": list(LEGAL_FLAGS),
                },
                "resolvedDependencies": [
                    {
                        "name": str(config),
                        "digest": {"sha256": payload["config"]["sha256"]},
                    }
                ],
            },
            "runDetails": {
                "builder": {"id": BUILDER_ID},
                "metadata": {
                    "invocationId": str(uuid.uuid4()),
                    "startedOn": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                },
            },
        },
        "honest_limits": {
            "covers": [
                "the configuration namespace (every argparse dest)",
                "the argv the wrapper intended to pass",
                "the sha256 of the source config",
                "the outcome-relevant environment variables",
                # 2026-11-04 (Wf-A269 frontier sweep) — NEW.  PRE-REPAIR
                # RENDERING: the statement did not name the command, nor say
                # where the argv came from.
                "the command the wrapper assembled (the trainer path followed "
                "by the argv) and WHETHER THAT ARGV WAS DECLARED BY THE "
                "WRAPPER OR GUESSED BY THIS TOOL",
                "the trainer's post-override derivations (which dests it "
                "changed after the config was applied, and whether that set "
                "is the one the loader's authoritative table allows)",
            ],
            "does_not_cover": [
                "host state (kernel, driver, GPU, dataset bytes)",
                "the runtime transports scripts/start.sh performs at launch",
                "the RESOLVER's own correctness: BOTH sides of the namespace "
                "comparison are computed by the same loader "
                "(molmetal/scripts/wfA2_config_loader.py), so a loader bug "
                "cancels out.  The §5.11-documented list-valued defect "
                "(`metals: Ru Ir Os Re` written as one string) makes this "
                "check PASS.  The check detects DRIFT between registration "
                "and consumption; it does NOT validate the resolver.",
                "a dest on a trainer's derived allowlist can be overridden "
                "UNCONDITIONALLY without failing the check — the allowlist "
                "says WHICH dests may differ, not under which condition",
                # 2026-11-04 (Wf-A269 frontier sweep) — NEW.  The recorded
                # `command` is the trainer path followed by the argv.  It does
                # NOT name the INTERPRETER, the wrapper's environment, or any
                # argument the trainer adds AFTER it parses the argv.
                "the INTERPRETER of the command: `command` is the trainer "
                "path followed by the argv, and the WRAPPER (not this record) "
                "chooses the interpreter",
            ],
            "note": (
                "This is a CONFIGURATION provenance record, not a full SLSA L3 "
                "build provenance.  SLSA L3 requires externalParameters to "
                "cover every externally-controllable influence; the start.sh "
                "runtime transports are a known gap and are listed above."
            ),
        },
    }
    (out_dir / PROVENANCE_NAME).write_text(
        json.dumps(statement, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"[launch-provenance] PASS: registered {len(namespace)} dest(s) -> {expected_path}")
    print(f"[launch-provenance] config sha256 = {payload['config']['sha256']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    expected = _load_artifact(out_dir / EXPECTED_NAME)
    actual = _load_artifact(out_dir / ACTUAL_NAME)

    if expected is None:
        print(f"[launch-provenance] SKIP: no {EXPECTED_NAME} in {out_dir}")
        return 1 if args.require_actual else 2
    if actual is None:
        print(f"[launch-provenance] SKIP: no {ACTUAL_NAME} in {out_dir} "
              "(the trainer never wrote one — the run did not record its "
              "provenance)")
        # With --require-actual a missing actual is a FAIL, never a PASS.
        return 1 if args.require_actual else 2

    rows: list[dict[str, Any]] = []

    # 1. the argv — did the trainer receive what the wrapper registered?
    exp_argv = list(expected.get("argv") or [])
    act_argv = list(actual.get("argv") or [])
    argv_source = expected.get("argv_source") or "guessed"
    # A GUESSED argv is not what the trainer received — it is the wrapper's
    # guess at the legal flag set.  MEASURED 2026-11-04: 0 of 197
    # `entry.command_template`s contain `--ckpt-dir`, so the guess adds a flag
    # the trainer never sees and the comparison would fire on every correct
    # launch.  A guessed argv is therefore ADVISORY: reported, never able to
    # fail.  The row becomes a real check the moment the wrapper passes
    # `--argv-json` (argv_source = "declared").
    rows.append(
        {
            "check": "argv",
            "equal": exp_argv == act_argv,
            "advisory": argv_source != "declared",
            "argv_source": argv_source,
            "expected": exp_argv,
            "actual": act_argv,
        }
    )

    # 2. the config's sha256 — was the config edited between registration
    #    and the run?
    exp_sha = (expected.get("config") or {}).get("sha256")
    act_sha = (actual.get("config") or {}).get("sha256")
    rows.append(
        {
            "check": "config_sha256",
            "equal": exp_sha == act_sha and exp_sha is not None,
            "expected": exp_sha,
            "actual": act_sha,
        }
    )

    # 3. the DECLARED namespace, dest by dest, VALUE AND TYPE.
    #
    #    Like-for-like: the wrapper's side is the namespace the trainer's
    #    parser produced from the config (parse_args + apply_yaml_overrides),
    #    and the trainer's side is the namespace it recorded IMMEDIATELY
    #    AFTER the config was applied.  Comparing the wrapper's DECLARED
    #    namespace against the trainer's CONSUMED one would fire on every
    #    post-override derivation (ckpt_dir / csv / metals in the D-MPNN
    #    trainer) — i.e. always red, which is the DEAD-GATE class by another
    #    route.  The trainer's own derivations are row 3b.
    exp_ns = dict(expected.get("resolved") or {})
    act_ns = dict(actual.get("declared") or actual.get("resolved") or {})
    ns_diffs = []
    for dest in sorted(set(exp_ns) | set(act_ns)):
        e = exp_ns.get(dest, "<missing>")
        a = act_ns.get(dest, "<missing>")
        same = e == a and type(e) is type(a)
        if not same:
            ns_diffs.append({"dest": dest, "expected": e, "actual": a})
    rows.append(
        {
            "check": "namespace",
            "equal": not ns_diffs,
            "n_dests_compared": len(set(exp_ns) | set(act_ns)),
            "n_different": len(ns_diffs),
            "diffs": ns_diffs[:20],
        }
    )

    # 3b. the trainer's POST-OVERRIDE MUTATIONS — the row that makes the
    #     POST-OVERRIDE MUTATION BLINDNESS defect visible.
    #
    #     The trainer is allowed to DERIVE some dests after the config has
    #     been applied (the D-MPNN trainer derives ckpt_dir, csv and metals).
    #     A dest it changes that is NOT on its declared derived list is
    #     DRIFT: the run used a value the config never declared, and the
    #     check FAILS on it.  The derived list is validated against the
    #     loader's authoritative table at write time, so a trainer cannot
    #     widen it in its own call site.
    overrides = dict(actual.get("overrides") or {})
    derived = dict(actual.get("derived") or {})
    allowed = set(derived.get("allowed") or [])
    declared_by_trainer = set(derived.get("declared_by_trainer") or [])
    derived_state = derived.get("state")
    unexpected = sorted(set(overrides) - allowed)
    # A trainer with no table entry has NO allowed set, so every override is
    # unexpected — the check stays closed rather than open.
    rows.append(
        {
            "check": "overrides",
            "equal": not unexpected and derived_state != "REFUSED_TOO_WIDE",
            "n_overrides": len(overrides),
            "allowed": sorted(allowed),
            "declared_by_trainer": sorted(declared_by_trainer),
            "derived_state": derived_state,
            "unexpected": unexpected[:20],
            "diffs": [
                {"dest": d, "declared": overrides[d].get("declared"),
                 "consumed": overrides[d].get("consumed")}
                for d in unexpected[:20]
            ],
        }
    )

    # 4. the environment — did the trainer run with the environment the
    #    wrapper registered?
    exp_env = dict(expected.get("environment") or {})
    act_env = dict(actual.get("environment") or {})
    env_diffs = []
    for name in sorted(set(exp_env) | set(act_env)):
        if exp_env.get(name) != act_env.get(name):
            env_diffs.append(
                {"name": name, "expected": exp_env.get(name), "actual": act_env.get(name)}
            )
    rows.append(
        {"check": "environment", "equal": not env_diffs, "diffs": env_diffs[:20]}
    )

    ok = all(row["equal"] for row in rows if not row.get("advisory"))
    if args.json:
        print(json.dumps({"verdict": "PASS" if ok else "FAIL", "rows": rows}, indent=2))
    else:
        for row in rows:
            if row["equal"]:
                print(f"[launch-provenance]   {row['check']}: MATCH")
            elif row.get("advisory"):
                print(
                    f"[launch-provenance]   {row['check']}: ADVISORY "
                    f"(row cannot fail: argv_source={row.get('argv_source')})"
                )
                # The difference is still REPORTED — an advisory row that
                # cannot fail must not be a row that hides.
                if row["check"] == "argv":
                    print(f"[launch-provenance]     expected={row['expected']}")
                    print(f"[launch-provenance]     actual  ={row['actual']}")
            else:
                print(f"[launch-provenance]   {row['check']}: MISMATCH")
                for diff in row.get("diffs", []):
                    print(f"[launch-provenance]     {diff}")
                if row["check"] == "namespace":
                    print(
                        f"[launch-provenance]     {row['n_different']} of "
                        f"{row['n_dests_compared']} dest(s) differ"
                    )
                if row["check"] == "argv":
                    print(f"[launch-provenance]     expected={row['expected']}")
                    print(f"[launch-provenance]     actual  ={row['actual']}")
                if row["check"] == "config_sha256":
                    print(f"[launch-provenance]     expected={row['expected']}")
                    print(f"[launch-provenance]     actual  ={row['actual']}")
                if row["check"] == "overrides":
                    print(
                        f"[launch-provenance]     {row['n_overrides']} override(s); "
                        f"allowed={row['allowed']}; "
                        f"declared_by_trainer={row['declared_by_trainer']}; "
                        f"state={row['derived_state']}"
                    )
        print(f"[launch-provenance] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit", help="register the run's intended configuration")
    emit.add_argument("--trainer", required=True)
    emit.add_argument("--config", required=True)
    emit.add_argument("--out-dir", required=True)
    emit.add_argument("--ckpt-dir", default=None)
    emit.add_argument(
        "--argv-json",
        default=None,
        help=(
            "the EXACT argv the wrapper will pass to the trainer, as a JSON "
            "array of strings.  Without it the argv is GUESSED from the legal "
            "§5.11 flag set, and the argv row becomes ADVISORY (reported, "
            "never able to fail) — MEASURED 2026-11-04: 0 of 197 "
            "`entry.command_template`s contain `--ckpt-dir`, so the guess adds "
            "a flag the trainer never receives."
        ),
    )
    emit.set_defaults(func=cmd_emit)

    verify = sub.add_parser("verify", help="compare the registered and the consumed config")
    verify.add_argument("--out-dir", required=True)
    verify.add_argument(
        "--require-actual",
        action="store_true",
        help="treat a missing actual artefact as FAIL (use after a completed run)",
    )
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
