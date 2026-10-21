#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Config↔trainer binding gate — every YAML key must reach the trainer it names.

`wfA2_config_loader` resolves a YAML leaf to an argparse dest *of the parser it
is handed*. When a config is loaded, that parser is the one belonging to
`entry.script` — and nothing has ever checked that those two agree. A config
whose keys target a different trainer loads without complaint and silently drops
every key that trainer does not declare.

That is not hypothetical. Measured 2026-10-09 against the CFM parser:

    configs/blk_d1_attentivefp_5seed_2026_10_09.yaml    9 keys resolve to nothing
    configs/wf_fix3lines_bace_50ep_2026_10_09.yaml      8
    configs/blk_d2_chemprop_v3_frozen_5seed_2026_10_09  5
    configs/dmpnn.yaml                                 21
    configs/wf124_p3_50epoch_full_sweep_g5.yaml         26

Each is *correct* — those configs name a different `entry.script` and do resolve
against it. The gate is what turns "correct today" into "correct by
construction": it resolves every key against the parser of the trainer the config
itself points at, so a config that drifts onto the wrong trainer, or gains a key
whose dest nobody declares, fails instead of quietly training something other
than what the file says.

Why source-scanning is not enough: whether a leaf resolves depends on the
parser's *dest set*, which is a runtime property. A renamed argparse argument
breaks resolution without touching the YAML at all.

Resolving is not delivering, either. The second tier added 2026-10-09 checks the
TYPE of what lands on the dest: `nargs="+"`/`"*"` need a YAML sequence, and a
`type=` converter must be able to convert the value. The run that motivated it
resolved `--metals` perfectly and still matched zero metals, because the YAML
said `metals: Ru Ir Os Re` (a string) where the trainer expects a list.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP. Exit 0=PASS, 1=FAIL, 2=SKIP.
"""

from __future__ import annotations

import argparse
import re
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wfA2_config_loader as L  # noqa: E402

CONFIG_GLOBS = ("configs/*.yaml", "configs/training/*.yaml")


def _capture_parser(script_path: Path):
    """Import a trainer and return (namespace, actions_by_dest, error).

    A namespace (not the parser) is what `_resolve_dest` tests against, and
    building it is the only way to learn a trainer's real dest set — which
    includes dests renamed by ``dest=`` on add_argument, which a source grep
    cannot see.

    The actions are captured alongside it because a dest set alone cannot say
    what *shape* of value the trainer expects: ``--metals`` (``nargs="+"``)
    and ``--lr`` (``type=float``) both resolve, and delivering a YAML string
    to the first is a silent, run-killing bug while delivering it to the second
    is merely coerced. Only the action carries that distinction. See
    :func:`_type_violation`.
    """
    name = "cfgbind_" + script_path.stem
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        return None, None, f"cannot load {script_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - a trainer that won't import is a finding
        return None, None, f"import failed: {type(exc).__name__}: {exc}"[:80]

    import argparse as _ap

    captured: dict = {}
    originals = {
        "parse_args": _ap.ArgumentParser.parse_args,
        "parse_known_args": _ap.ArgumentParser.parse_known_args,
    }

    # Both hooks share one body: the parse is abandoned before it runs, so the
    # old recursion trap (parse_args delegating to parse_known_args) cannot fire.
    def make():
        def spy(self, args=None, namespace=None):
            # Capture BEFORE parsing. MEASURED 2026-10-21: two trainers declare
            # `--config` with required=True, so `parse_args([])` exits 2 before
            # the capture line ever ran, and the gate reported "could not
            # capture a parser namespace" — a config it then could not check at
            # all. The dest set does not depend on the values, so it is built
            # from the actions and the parse is abandoned.
            captured["actions"] = _actions_by_dest(self)
            captured["ns"] = _ap.Namespace(
                **{
                    a.dest: a.default
                    for a in self._actions
                    if a.dest not in ("help", _ap.SUPPRESS)
                }
            )
            raise _Stop()
        return spy

    _ap.ArgumentParser.parse_args = make()
    _ap.ArgumentParser.parse_known_args = make()
    old_argv = sys.argv

    def captured_result():
        return captured["ns"], captured.get("actions", {})

    try:
        import inspect

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
            except BaseException:  # noqa: BLE001 - SystemExit/_Stop both fine
                pass
            if "ns" in captured:
                return *captured_result(), None
        if hasattr(mod, "main"):
            try:
                mod.main([])
            except BaseException:  # noqa: BLE001
                pass
        if "ns" in captured:
            return *captured_result(), None
    finally:
        sys.argv = old_argv
        _ap.ArgumentParser.parse_args = originals["parse_args"]
        _ap.ArgumentParser.parse_known_args = originals["parse_known_args"]
    return None, None, "could not capture a parser namespace"


class _Stop(Exception):
    pass


def _actions_by_dest(parser) -> dict[str, dict]:
    """Map argparse dest -> the shape info the gate needs to type-check it.

    Three fields, all read off the live action rather than inferred:

    * ``nargs`` — ``"+"`` / ``"*"`` mean the trainer expects a *sequence*. This
      is the field that was missing on 2026-10-09.
    * ``type`` — the converter the trainer's own CLI would have run. YAML
      values bypass argparse entirely, so this is the only record of what the
      value was supposed to become.
    * ``option`` — the flag-style name (``--metals``) for reporting. Reported
      by CLI name, not by dest, because that is the name a person has to go
      and fix, and because a dest can be reached by several flags.
    """
    out: dict[str, dict] = {}
    for action in getattr(parser, "_actions", ()):
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        longs = [o for o in getattr(action, "option_strings", ()) if o.startswith("--")]
        out[dest] = {
            "nargs": getattr(action, "nargs", None),
            "type": getattr(action, "type", None),
            "option": max(longs, key=len) if longs else "--" + dest.replace("_", "-"),
        }
    return out


#: ``nargs`` values that mean "one or more" / "zero or more" positional-or-flag
#: tokens, i.e. argparse hands the trainer a list rather than a scalar.
SEQUENCE_NARGS = ("+", "*")


def _type_violation(path: list[str], value, dest: str, shape: dict) -> dict | None:
    """Return a violation dict if ``value`` is the wrong shape for ``dest``.

    Resolving is not enough. Measured 2026-10-09: ``configs/`` shipped
    ``metals: Ru Ir Os Re`` (a YAML *string*) against a trainer whose
    ``--metals`` is ``nargs="+"``. The key resolved, so both §5.11 gates
    passed; the trainer then searched for a metal literally named
    ``"Ru Ir Os Re"``, matched zero rows, produced an empty test set for every
    fold and emitted ``test_auc = nan`` for all 5 seeds. The run died at the
    very end in ``statistics.pstdev`` and destroyed the aggregate for folds
    that had trained fine.

    Two failure shapes, checked in this order:

    1. ``nargs`` in :data:`SEQUENCE_NARGS` — the delivered value MUST be a
       list/tuple. A string here is not merely wrong, it is silently iterable-
       per-character at best and one bogus token at worst.
    2. a ``type=`` converter on a scalar action — run it. If it raises, the
       value cannot become what the trainer expects.
    """
    dotted = ".".join(path)
    nargs = shape.get("nargs")
    conv = shape.get("type")
    option = shape.get("option") or ("--" + dest.replace("_", "-"))

    if nargs in SEQUENCE_NARGS:
        if not isinstance(value, (list, tuple)):
            return {
                "key": dotted,
                "reason": "type",
                "detail": (
                    f"{option} is nargs={nargs!r} so it delivers a list, but the "
                    f"YAML value is a {type(value).__name__} ({value!r}). "
                    f"Write it as a YAML sequence: {option} takes "
                    f"'[Ru, Ir, Os, Rh, Re]', not the string 'Ru Ir Os Rh Re'."
                ),
                "dest": dest,
                "received": type(value).__name__,
                "required": f"list (nargs={nargs!r})",
            }
        return None

    if conv is None or value is None or isinstance(value, (list, tuple, dict)):
        return None

    try:
        conv(value)
    except (TypeError, ValueError) as exc:
        target = getattr(conv, "__name__", str(conv))
        return {
            "key": dotted,
            "reason": "type",
            "detail": (
                f"{value!r} is not convertible to {target} for {option} "
                f"({type(exc).__name__}: {exc}). wfA2_config_loader.apply_yaml_overrides "
                f"keeps the argparse default for this key and records it in "
                f"last_skipped_keys() as a type mismatch, so the run would train on "
                f"a default nobody chose."
            ),
            "dest": dest,
            "received": f"{type(value).__name__} {value!r}",
            "required": target,
        }
    return None


#: Flags an `entry.command_template` may carry. Same rule as the launcher
#: wrappers: identify the run, never describe the model. `--config` selects the
#: config, `--out-dir` / `--ckpt-dir` say where output goes.
ALLOWED_TEMPLATE_FLAGS = frozenset({"--config", "--out-dir", "--ckpt-dir", "--help"})

_TEMPLATE_SCRIPT = re.compile(r"\b(molmetal/scripts/train_[a-z0-9_]+\.py)\b")
_TEMPLATE_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")


def _entry_script_and_command(entry):
    """Return (script, command_template), mirroring how start.sh finds them.

    scripts/start.sh does not require `entry.script`: it also scans
    `command_template` for a `*.py` token (its `_script_candidates`). Three
    configs in this repo rely on that — `default.yaml` and both
    configs/training/ sweeps spell the trainer only inside the template — so a
    gate that keys on `entry.script` alone would skip exactly the configs whose
    binding is least obvious.
    """
    if not isinstance(entry, dict):
        return None, ""
    script = entry.get("script")
    command = str(entry.get("command_template") or entry.get("command") or "")
    if not script and command:
        m = _TEMPLATE_SCRIPT.search(command)
        if m:
            script = m.group(1)
    return script, command


def _disallowed_template_flags(command: str) -> list[str]:
    """Trainer flags in a command_template that §5.11 does not permit.

    The distinction that matters here is placeholder vs literal, and getting it
    wrong in either direction is expensive:

    * ``--epochs {epochs}`` — the value is a start.sh substitution field that
      start.sh fills from the SAME yaml (scripts/start.sh's ``entry.format(...)``).
      That is yaml-driven delivery by another route, and it is what makes the
      wf124 G4/G5/G6 sweep configs work at all: `{metal}`/`{seed}` are the sweep
      dimensions. Flagging these would demand deleting the sweep mechanism.
    * ``--device cuda:0`` — a literal baked into the template. Nothing reads it
      from yaml; the value lives only in a shell string. That is the violation.

    Verified 2026-10-09: every placeholder used by every config resolves to a
    field start.sh actually passes, so no template raises KeyError at launch. The
    literal offenders are ``--device cuda:0`` in the two configs/training/ sweeps
    and the bare ``--use-multi-ph`` switch in fm_pocket_multiph.yaml.
    """
    if not command:
        return []
    offenders: list[str] = []
    for match in _TEMPLATE_FLAG.finditer(command):
        flag = match.group(1)
        if flag in ALLOWED_TEMPLATE_FLAGS:
            continue
        tail = command[match.end():]
        # A placeholder value (possibly preceded by spaces) means start.sh fills
        # it from the yaml. A bare switch or a literal does not.
        value = re.match(r"\s*(\{[a-zA-Z_][a-zA-Z0-9_]*\})?", tail)
        if value and value.group(1):
            continue
        offenders.append(flag)
    return offenders


def _rel(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    A config outside the repo root is a test fixture, not a crash — the gate
    must report on whatever it is handed. (Same trap as the sibling
    _yaml_only_gate.py, where it surfaced as a test failure.)
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_empty_container(value) -> bool:
    return isinstance(value, (list, dict, tuple, set)) and len(value) == 0


_DIRECT_READ = re.compile(r"""(?:\.get\(\s*|\[\s*)["']([A-Za-z0-9_]+)["']""")
#: `cfg["model"]` NOT followed by another subscript or a method call — the
#: whole sub-mapping is handed to a consumer (`model_cfg=cfg["model"]`).
_DIRECT_SUBTREE = re.compile(r"""\[\s*["']([A-Za-z0-9_]+)["']\s*\](?![\[.])""")


def _direct_loader_keys(script_path: Path) -> set[str] | None:
    """Keys a script reads out of the loaded YAML mapping by name.

    Not every config names a trainer whose hyperparameters arrive through
    ``apply_yaml_overrides``.  Some scripts load the file themselves and read
    ``cfg["train"]["epochs"]``; those keys are delivered, but they never touch
    an argparse dest, so the dest test reports them as dead.  MEASURED
    2026-10-21: the eight ``configs/msrs_dmpnn_*.yaml`` files name no
    ``entry.script`` at all, so the gate skipped them — ``n_configs_skipped``
    was 8 and ``test_every_config_is_actually_checked`` failed.  A gate that
    skips a config is not checking it.

    This is a *weaker* claim than resolution and is reported separately: a
    string literal of the right name can still be ignored downstream.  It is
    used only where no dest exists, and only to distinguish "read by the script
    the config names" from "read by nothing", which is the failure the gate is
    for.

    Returns ``None`` when the source cannot be read, so the caller falls back
    to the strict behaviour rather than silently accepting every key.
    """
    try:
        source = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return set(_DIRECT_READ.findall(source))


def _direct_subtree_keys(script_path: Path) -> set[str]:
    """Keys whose whole sub-mapping the script takes and passes on.

    ``msrs_dmpnn_protocols.py`` does ``model_cfg=cfg["model"]`` — the leaves
    under ``model:`` are delivered, but no single line mentions ``hidden``.
    Treating them as dead would be a false positive, which is how a gate gets
    switched off.  The rule is deliberately narrow: a parent counts only when
    it is subscripted and then *not* indexed or method-called further, so
    ``cfg["train"]["epochs"]`` does **not** license every key under ``train:``.
    """
    try:
        source = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return set(_DIRECT_SUBTREE.findall(source))


def audit_config(
    config_path: Path,
    ns,
    actions: dict | None = None,
    direct_keys: set[str] | None = None,
    direct_out: list[str] | None = None,
    subtree_keys: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (violations, declared_inert) for one config.

    Four dispositions, deliberately distinct:

    * resolved to a dest — the key is live.
    * read by name from the loaded mapping — the key is live, but through the
      script's own ``cfg["..."]`` access rather than an argparse dest. Weaker
      than resolution and reported separately (``direct_out``): it proves the
      script mentions the key, not that the value reaches the model.
    * registered non-trainer consumer — the key is live, read by something that
      is not the trainer (scripts/start.sh, an aggregator, pre-flight).
    * DECLARED-INERT — no dest and no consumer anywhere in the repo, but
      registered in ``DECLARED_INERT_PATHS`` because it documents sweep intent.
      Reported every run, never fatal: hiding it would recreate the blind spot
      this audit exists to close.
    * anything else — a violation. A NEW unresolvable key fails the launch.

    On top of the resolution test, every resolved leaf is TYPE-checked against
    the argparse action it lands on (:func:`_type_violation`). Resolution alone
    is not delivery: a `nargs="+"` dest fed a YAML string resolves perfectly and
    then matches zero rows. `actions` is optional so callers (and older tests)
    can audit resolution only; when omitted, no type check runs and the
    behaviour is exactly what it was before this tier existed.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return [], []
    violations: list[dict] = []
    inert: list[str] = []
    for path, value in L._iter_leaves(data):
        if _is_empty_container(value):
            continue
        entry = L._skip_entry(path, value)
        if L._registered_non_trainer(entry):
            continue
        dest, _alias = L._resolve_dest(ns, path)
        if dest is not None:
            # Resolution is necessary but not sufficient — now check the shape
            # of what is actually being delivered.
            if actions:
                shape = actions.get(dest)
                if shape is not None:
                    bad = _type_violation(path, value, dest, shape)
                    if bad is not None:
                        bad["config"] = _rel(config_path)
                        violations.append(bad)
            continue
        dotted = ".".join(path)
        # Ancestor check: _iter_leaves descends into nested mappings, so
        # registering `eval.attribution_matrix` does NOT cover its children —
        # `.description` and `.output` arrive as separate leaves and would be
        # reported as unknown. Match on any prefix instead of enumerating every
        # descendant by hand.
        prefixes = [".".join(path[:i]) for i in range(1, len(path) + 1)]
        if any(p in L.DECLARED_INERT_PATHS for p in prefixes):
            inert.append(dotted)
            continue
        if direct_keys and (
            path[-1] in direct_keys
            or (len(path) > 1 and subtree_keys and path[-2] in subtree_keys)
        ):
            # No argparse dest, but the script the config names reads this leaf
            # by name out of the loaded mapping — directly, or by taking the
            # whole parent dict (`model_cfg=cfg["model"]`) and handing it to a
            # builder. Weaker than resolution: a typo inside a passed sub-dict
            # still slips through. Reported on its own line, never counted as a
            # dest hit.
            if direct_out is not None:
                direct_out.append(dotted)
            continue
        violations.append(
            {
                "config": _rel(config_path),
                "key": dotted,
            }
        )
    return violations, inert


def audit(configs: list[Path] | None = None, strict: bool = False) -> dict:
    """Audit every config against the trainer its own entry.script names.

    ``strict`` turns a skipped config into a violation.  The flag used to be
    accepted and ignored ("interface parity"), which meant the one failure mode
    this gate cannot see — a config it never opened — was the one failure mode
    it never reported.  MEASURED 2026-10-21: eight ``configs/msrs_dmpnn_*.yaml``
    files named no ``entry.script``, so they were skipped and the gate said
    PASS while checking nothing about them.
    """
    if configs is None:
        seen: list[Path] = []
        for pattern in CONFIG_GLOBS:
            seen.extend(sorted(REPO_ROOT.glob(pattern)))
        configs = seen

    violations: list[dict] = []
    declared_inert: list[str] = []
    direct_keys_seen: list[str] = []
    checked: list[str] = []
    skipped: list[str] = []
    ns_cache: dict[Path, object] = {}
    act_cache: dict[Path, object] = {}

    for cfg in configs:
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            violations.append({"config": cfg.name, "key": f"<unparseable: {exc}>"})
            continue
        entry = data.get("entry") if isinstance(data, dict) else None
        script, command = _entry_script_and_command(entry)
        if not script:
            skipped.append(f"{cfg.name} (no entry.script and no .py in entry.command_template)")
            if strict:
                violations.append(
                    {
                        "config": cfg.name,
                        "key": "entry.script absent — config cannot be bound to a trainer",
                        "reason": "skipped",
                    }
                )
            continue

        # §5.11's third escape route. `--extra-flags` is closed in start.sh and
        # the launch wrappers are scanned by _yaml_only_gate.py, but
        # `entry.command_template` is a raw shell string inside the YAML: it can
        # carry trainer flags just as well, and did — configs/training/
        # platinai_pt_4cell_5seed_50ep_v1.yaml ships `--device cuda:0` there.
        for flag in _disallowed_template_flags(command):
            violations.append(
                {
                    "config": cfg.name,
                    "key": f"entry.command_template carries {flag}",
                }
            )

        script_path = (REPO_ROOT / str(script)).resolve()
        if not script_path.is_file():
            violations.append({"config": cfg.name, "key": f"entry.script not found: {script}"})
            continue

        if script_path not in ns_cache:
            ns, actions, err = _capture_parser(script_path)
            if ns is None and err and err.startswith("could not capture"):
                # A script with no argparse at all: it reads the mapping
                # itself. Auditing it against an empty namespace is exactly
                # right — every key must then be justified by a direct read.
                ns_cache[script_path] = argparse.Namespace()
                act_cache[script_path] = {}
            elif ns is None:
                violations.append({"config": cfg.name, "key": f"trainer: {err}"})
                ns_cache[script_path] = False
                act_cache[script_path] = {}
            else:
                ns_cache[script_path] = ns
                act_cache[script_path] = actions or {}
        ns = ns_cache[script_path]
        if ns is False:
            continue

        before_direct = len(direct_keys_seen)
        found, inert = audit_config(
            cfg,
            ns,
            act_cache[script_path],
            direct_keys=_direct_loader_keys(script_path),
            direct_out=direct_keys_seen,
            subtree_keys=_direct_subtree_keys(script_path),
        )
        for i in range(before_direct, len(direct_keys_seen)):
            direct_keys_seen[i] = f"{cfg.name}: {direct_keys_seen[i]}"
        violations.extend(found)
        for k in inert:
            declared_inert.append(f"{cfg.name}: {k}")
        checked.append(cfg.name)
        if found:
            n_type = sum(1 for f in found if f.get("reason") == "type")
            n_dead = len(found) - n_type
            tag = []
            if n_type:
                tag.append(f"{n_type} type")
            if n_dead:
                tag.append(f"{n_dead} dead")
            checked[-1] += f" ({', '.join(tag)})"

    return {
        "gate": "config_trainer_binding",
        "rule": (
            "every YAML key resolves to a dest of the parser named by that "
            "config's own entry.script (AGENTS.md §5.11 single-source-of-truth), "
            "and the value delivered to that dest has the type the argparse "
            "action declares (nargs='+'/'*' needs a list, type= must convert)"
        ),
        "n_configs_checked": len(checked),
        "n_configs_skipped": len(skipped),
        "configs": checked,
        "skipped": skipped,
        "n_declared_inert": len(declared_inert),
        "declared_inert": declared_inert,
        "n_direct_loader_keys": len(direct_keys_seen),
        "direct_loader_keys": direct_keys_seen,
        "n_type_violations": sum(1 for v in violations if v.get("reason") == "type"),
        "n_violations": len(violations),
        "violations": violations,
        "verdict": "PASS" if not violations else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="a config with no entry.script fails instead of being skipped",
    )
    args = ap.parse_args(argv)

    report = audit(strict=args.strict)
    if args.json:
        import json

        print(json.dumps(report, indent=2))
    else:
        print(
            f"[config-binding] {report['n_configs_checked']} config(s) checked, "
            f"{report['n_configs_skipped']} skipped"
        )
        for v in report["violations"]:
            if v.get("reason") == "type":
                print(f"  VIOLATION {v['config']}: '{v['key']}' TYPE — {v['detail']}")
            else:
                print(f"  VIOLATION {v['config']}: '{v['key']}' resolves to no dest")
        for k in report["declared_inert"]:
            # 2026-10-12 wording fix. This said "no consumer", which is not
            # what DECLARED-INERT means and was false for at least one entry:
            # `sweep.strengths` and `sweep.biases` ARE read, by
            # scripts/launch_cfm_l3_guidance_sweep.sh and
            # scripts/launch_cfm_aromatic_bias_sweep.sh, which iterate the grid
            # to emit one invocation per arm. A gate that prints a false
            # statement about its own entries trains its reader to distrust
            # the ones that matter.
            print(
                f"  DECLARED-INERT {k}  (no trainer dest; launcher-owned — "
                "documented sweep intent, reported every run)"
            )
        for s in report["skipped"]:
            print(f"  SKIP {s}")
        if not report["violations"]:
            print(
                "  every config key reaches a dest of its own entry.script parser, "
                "with the type that dest's argparse action declares"
            )
        print(f"[config-binding] {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
