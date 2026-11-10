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

#: Configs that are DELIBERATELY not bound to a trainer parser, and the reason.
#:
#: **WHY THIS REGISTRY EXISTS.**  A config with no ``entry.script`` (and no
#: ``.py`` token in ``entry.command_template``) cannot be checked against a
#: parser — there is no parser to check it against.  Before this registry, the
#: gate **SKIPPED** such a config, kept a PASS verdict, and printed a final
#: line that read *"every config key reaches a dest of its own entry.script
#: parser"*.  **That line was false for every skipped config** (MEASURED
#: 2026-11-04: 2 of 261 configs were skipped, and the gate still said PASS).
#: That is the §6.269.17 DEAD-GATE class in its "skipped" costume: the check
#: could PASS without checking.
#:
#: **THE THREE DISPOSITIONS, MIRRORING THE LOADER'S** (``wfA2_config_loader``
#: carries ``_NON_TRAINER_KEYS`` / ``_NON_TRAINER_PATHS`` /
#: ``DECLARED_INERT_PATHS`` for a KEY that resolves to no dest):
#:
#: 1. the config names a trainer and is bound to it  → checked, normally;
#: 2. the config is skipped **and DECLARED HERE, with a reason** → reported
#:    as ``DECLARED-SKIP`` every run, and the reason is on file;
#: 3. the config is skipped **and NOT declared** → a **VIOLATION**, always —
#:    not only under ``--strict``.
#:
#: **AN UNDECLARED SKIP IS NOW FATAL BY DEFAULT.**  ``--strict`` used to be the
#: only way to see it, and the pre-flight calls this gate WITHOUT it — so the
#: one failure mode the gate could not see was the one the gate never
#: reported.  ``--strict`` is retained for interface parity and now means the
#: same thing the default means.
#:
#: **THE POPULATION IS A MOVING TARGET — READ THE DENOMINATOR, NOT THE RATIO.**
#: The reading above ("2 of 261") was taken at 01:0x on 2026-11-04, when the
#: working tree held 261 configs.  **SUPERSEDED 2026-11-04 (04:0x, same day)
#: by a direct measurement of this gate's own globs**
#: (``configs/*.yaml`` + ``configs/training/*.yaml``):
#:
#:   * **500** configs audited (495 + 5) — **257 TRACKED**, **243 UNTRACKED**
#:     (**48.6 %** of the audited population is untracked ⇒ a clean checkout of
#:     HEAD audits 257 configs, not 500; the PASS describes a WORKING TREE);
#:   * **2** declared skips (both untracked — the cohort-c9 fixtures below);
#:   * **0** undeclared skips ⇒ ``verdict = PASS``;
#:   * **2** tracked yaml under ``configs/`` sit OUTSIDE these globs
#:     (``configs/locks/drugood.yaml``, ``configs/locks/protocol_lock.yaml``)
#:     — they are LOCK files, excluded by design, not a coverage hole.
#:
#: ⇒ the "2 of 261" reading is preserved VERBATIM above (AGENTS.md §5.5
#: rule #6) and is superseded by the measurement above, dated 2026-11-04.
#: Full record: ``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269.22.5.
#:
#: **MEASURED 2026-11-04.**  The two entries below are the Wf-A269 cohort-c9
#: fixtures.  They are ``entry:``-less **ON PURPOSE**: ``scripts/start.sh``
#: synthesises an ``entry`` block for a config that carries none
#: (``e = d.get('entry', {})`` / ``if not e:``), and the fixture's whole job is
#: to exercise that LEGACY branch.  Adding an ``entry:`` section to them would
#: defeat the fixture.  The per-config binding check they need was run
#: directly against the CFM / D-MPNN parsers instead — recorded in
#: ``molmetal/reports/wfa269_yaml_migration/start.json``.
DECLARED_SKIPPED_CONFIGS: dict[str, str] = {
    "wfa269_c9_start_sh.yaml": (
        "Wf-A269 cohort c9 — an entry:-less fixture for scripts/start.sh's "
        "LEGACY branch (start.sh synthesises the entry block when the config "
        "carries none). Its binding was checked directly against the CFM "
        "trainer's parser; the fixture has no entry.script BY CONSTRUCTION."
    ),
    "wfa269_c9_start_sh_dmpnn.yaml": (
        "Wf-A269 cohort c9 — the D-MPNN twin of wfa269_c9_start_sh.yaml; "
        "entry:-less by construction (start.sh's legacy branch). Its binding "
        "was checked directly against the D-MPNN trainer's parser."
    ),
    "cfm_smoke_ci.yaml": (
        "CFM subsystem CI smoke (molmetal/scripts/smoke/smoke_cfm.sh) — its "
        "consumer is a PYTHON HEREDOC embedded in the shell script, not an "
        "importable .py, so there is no parser for this gate to capture (the "
        "§5.11 gates' trainer predicate is the regex train_[a-z0-9_]+.py, "
        "which a heredoc-embedded trainer cannot match). The driver builds "
        "its own argparse, resolves this file through "
        "molmetal/scripts/wfA2_config_loader.py::apply_yaml_overrides and "
        "FAILS CLOSED on an unresolved key (CFM_SMOKE FAIL config_unresolved) "
        "or a missing key (CFM_SMOKE FAIL config_incomplete) — the binding "
        "check for this config is the consumer itself."
    ),
}


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


def _action_choices(action) -> tuple | None:
    """The action's declared ``choices``, as a tuple — or ``None``.

    ``None`` means "the action declares no choice set".  ``argparse``'s own
    ``_check_value`` skips the membership test in that case, so this gate must
    too.  An EMPTY tuple is deliberately NOT ``None``: ``choices=()`` means
    "no value is acceptable", and argparse rejects every value against it.

    MEASURED 2026-11-04: 98 actions across the 32 trainers that a config's
    ``entry.script`` names declare ``choices``.  Several declare it as
    ``choices=list(SOME_CONSTANT)``, so the value is only knowable from the
    LIVE action — never from the source text alone.
    """
    choices = getattr(action, "choices", None)
    if choices is None:
        return None
    try:
        return tuple(choices)
    except TypeError:
        return None


def _actions_by_dest(parser) -> dict[str, dict]:
    """Map argparse dest -> the shape info the gate needs to type-check it.

    PRE-REPAIR RENDERING (2026-11-04, preserved VERBATIM per AGENTS.md §5.5
    rule #6 — the gate modelled THREE fields, and this docstring said so):

        Three fields, all read off the live action rather than inferred:

        * ``nargs`` — ``"+"`` / ``"*"`` mean the trainer expects a *sequence*. This
          is the field that was missing on 2026-10-09.
        * ``type`` — the converter the trainer's own CLI would have run. YAML
          values bypass argparse entirely, so this is the only record of what the
          value was supposed to become.
        * ``option`` — the flag-style name (``--metals``) for reporting. Reported
          by CLI name, not by dest, because that is the name a person has to go
          and fix, and because a dest can be reached by several flags.

    2026-11-04 (Wf-A269 frontier sweep) — a FOURTH field was added:

    * ``choices`` — the action's declared value set, or ``None`` when the action
      declares none.  ``argparse`` enforces ``choices`` on the command line,
      but the YAML path never reaches argparse (see :func:`_choices_violation`),
      so before this field existed a config could select a value the trainer's
      own parser FORBIDS and every §5.11 gate in the repo passed.  The first
      three fields are unchanged.
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
            "choices": _action_choices(action),
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


def _choices_violation(path: list[str], value, dest: str, shape: dict) -> dict | None:
    """Return a violation dict if ``value`` is outside the dest's declared choices.

    THE DEFECT CLASS (a new one, in the §6.269.17 DEAD-GATE family — a check
    that cannot fail because the check does not exist)

    ``argparse``'s ``choices=`` is a **DECLARED CONTRACT**: it enumerates the
    values the trainer claims to accept.  MEASURED 2026-11-04 (AST census over
    the 32 trainers a config's ``entry.script`` names): **98** actions declare
    it.  But the YAML path **BYPASSES argparse entirely** —
    ``wfA2_config_loader.apply_yaml_overrides`` writes the value straight onto
    the namespace with ``setattr`` — so before this check existed a config
    could select a value the trainer's own parser FORBIDS, and **every §5.11
    gate in the repo passed**.

    MEASURED 2026-11-04 on the real trainer: with ``choices=("argmax",
    "threshold")`` declared, a config's ``flow.bond_decision: per_class``
    reached ``args.bond_decision`` unchanged — so the CLI could not express
    what the YAML could, and the declaration was the only thing out of step.

    So the defect is not "the value is wrong"; it is **"a declared contract
    that no gate checked"**.

    WHAT IS CHECKED — the value the loader DELIVERS, not the raw YAML token:

    * a SCALAR dest: ``apply_yaml_overrides`` runs the action's own ``type=``
      converter, so the delivered value is the CONVERTED one.  ``argparse``
      validates in the same order (CPython 3.12 ``_get_values``: ``_get_value``
      converts, then ``_check_value`` tests membership), so the converted value
      is the faithful thing to test.
    * a SEQUENCE dest (``nargs`` in :data:`SEQUENCE_NARGS`): the loader does
      NOT convert list elements (``apply_yaml_overrides`` skips conversion for
      lists), so the delivered elements are the RAW ones; argparse tests
      membership on EACH element, so this does too.

    A value whose CONVERSION FAILS is not reported here: :func:`_type_violation`
    already reports it under ``reason="type"``, and reporting it twice under
    two reasons would make the gate's own counts lie.
    """
    choices = shape.get("choices")
    if choices is None:
        return None
    nargs = shape.get("nargs")
    conv = shape.get("type")
    option = shape.get("option") or ("--" + dest.replace("_", "-"))
    dotted = ".".join(path)

    if nargs in SEQUENCE_NARGS and isinstance(value, (list, tuple)):
        delivered = list(value)
    else:
        delivered = [value]
        if (
            conv is not None
            and delivered[0] is not None
            and not isinstance(delivered[0], (list, tuple, dict))
        ):
            try:
                delivered[0] = conv(delivered[0])
            except (TypeError, ValueError):
                # The conversion fails, so the loader KEEPS the argparse
                # default and never delivers this value at all.
                return None

    bad = [d for d in delivered if d not in choices]
    if not bad:
        return None

    rendered = ", ".join(repr(d) for d in choices)
    return {
        "key": dotted,
        "reason": "choices",
        "detail": (
            f"{option} declares choices=({rendered}), but the YAML value "
            f"{bad[0]!r} is not one of them. The YAML path bypasses argparse's "
            f"choices check entirely (apply_yaml_overrides writes the value "
            f"straight onto the namespace), so the run would train on a value "
            f"the trainer's own command line rejects."
        ),
        "dest": dest,
        "received": f"{type(bad[0]).__name__} {bad[0]!r}",
        "required": f"one of ({rendered})",
    }


#: Flags an `entry.command_template` may carry. Same rule as the launcher#: Flags an `entry.command_template` may carry. Same rule as the launcher
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


#: Dest pointers that are DELIBERATELY inert, and the reason.
#:
#: **WHAT A ``dest:`` POINTER IS.**  A schema-metadata block in a YAML carries
#: a ``dest:`` child whose value is a machine-readable ``script.py::dest``
#: reference — e.g. ``bench_cfm_decode_real.py::strict_large_frag``.  It is a
#: BINDING CLAIM, not documentation: it names a script and an argparse dest.
#:
#: **WHY THE REGISTRY EXISTS.**  ``wfA2_config_loader._iter_leaves`` skips a
#: mapping whose EVERY child key is in ``_NON_TRAINER_SUBKEYS`` — and ``dest``
#: is one of those keys — so the whole block, pointer included, never reached
#: this gate.  MEASURED 2026-11-09: the live config population holds **5**
#: schema-metadata blocks and **all 5** carry a ``dest:`` child ⇒ the skip's
#: entire observable effect was to hide exactly those pointers, and **5 of 5**
#: of them are FALSE (4 name a dest absent from the named script's parser, 1
#: names a script that does not exist anywhere in the repo).  The gate
#: PASSed the config anyway, while its own closing line claimed that every
#: config key reaches a dest of its own ``entry.script`` parser.
#:
#: **THE DISPOSITION MIRRORS ``DECLARED_SKIPPED_CONFIGS``** (one layer down):
#: a pointer that is declared HERE, with a reason, is REPORTED every run and
#: is not fatal; a pointer that is NOT declared is a **VIOLATION, always** —
#: the same rule the config-level registry uses.  Declaring a pointer is not
#: free: the reason has to say what the pointer currently resolves to.
#:
#: KEY FORMAT: ``"<config file name>::<block path>"`` — the same file-name
#: keying as :data:`DECLARED_SKIPPED_CONFIGS`, with the block's dotted path
#: appended so two blocks in one file are distinguishable.
#:
#: MEASURED 2026-11-09: the pointer ``script.py::dest`` is parsed by NOTHING
#: in the repo (no component reads it) ⇒ every pointer in the population is
#: an UNVERIFIED CLAIM until this gate checks it.
DECLARED_INERT_DEST_POINTERS: dict[str, str] = {
    "cfm_patience10_seed0_2026_10_12.yaml::strict_large_frag": (
        "Track E intent declaration — the m14 per-atom decode-ratio gate was "
        "NEVER wired into bench_cfm_decode_real.py: MEASURED 2026-11-09, that "
        "script's parser declares 14 dests (ckpt_path, data_config, "
        "data_path, device, drugood_path, jitter, n_rows, n_samples, "
        "nci60_path, out_dir, out_json, pocket_id, seeds, smoke) and "
        "'strict_large_frag' is NOT among them. The block is kept so the "
        "record of what Track E (Wf-60231cff-03e-15) intended survives; the "
        "pointer is surfaced by this gate every run."
    ),
    "cfm_patience10_seed0_2026_10_12.yaml::largest_frag_min_frac": (
        "Track E intent declaration — 'largest_frag_min_frac' is NOT a dest of "
        "bench_cfm_decode_real.py's parser (MEASURED 2026-11-09: 14 dests, "
        "none of them this) and the name appears nowhere in "
        "molmetal/eval/scoring/decode_gate.py either. Kept as the record of "
        "the intended per-atom fragment threshold."
    ),
    "cfm_patience10_seed0_2026_10_12.yaml::min_atoms_per_mol": (
        "Track E intent declaration — 'min_atoms_per_mol' is NOT a dest of "
        "bench_cfm_decode_real.py's parser (MEASURED 2026-11-09: 14 dests, "
        "none of them this) and appears nowhere in "
        "molmetal/eval/scoring/decode_gate.py either. Kept as the record of "
        "the intended heavy-atom floor."
    ),
    "cfm_patience10_seed0_2026_10_12.yaml::emit_admet8": (
        "Track E intent declaration — 'emit_admet8' is NOT a dest of "
        "cfm_property_panel.py's parser (MEASURED 2026-11-09: 8 dests — "
        "ckpt_path, decode_chemistry_validity, decode_seed, last_epoch, "
        "n_samples, out_json, out_smiles_csv, smiles_csv — and none of them "
        "is this). The 8-metric ADMET emission was never implemented; the "
        "block is kept as the record of that intent."
    ),
    "cfm_patience10_seed0_2026_10_12.yaml::paper_table_cohort": (
        "Track E intent declaration — the named script bench_cfm_paper_table.py "
        "DOES NOT EXIST anywhere in the repo (MEASURED 2026-11-09 by a "
        "bounded find over the repo), so the pointer cannot resolve at all. "
        "Kept as the record of the intended paper-table cohort key."
    ),
}


#: Where a ``dest:`` pointer's bare script name may live, in probe order.
#: The first candidate is the directory of the config's OWN ``entry.script`` —
#: a pointer names a sibling of the script the config launches — after which
#: the two repo script directories and the repo root are tried.
_POINTER_SCRIPT_DIRS = ("molmetal/scripts", "scripts", "")


def _resolve_pointer_script(
    name: str, entry_script_path: Path | None
) -> Path | None:
    """Resolve a pointer's bare script name to a repo path, or ``None``.

    ``name`` is the part before ``::`` — e.g. ``bench_cfm_decode_real.py``.
    """
    candidates: list[Path] = []
    if entry_script_path is not None:
        candidates.append(entry_script_path.parent / name)
    for sub in _POINTER_SCRIPT_DIRS:
        candidates.append(REPO_ROOT / sub / name if sub else REPO_ROOT / name)
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def _audit_dest_pointers(
    cfg: Path,
    data: dict,
    entry_script_path: Path | None,
    parser_cache: dict,
) -> list[dict]:
    """Adjudicate every ``dest:`` pointer in one config.

    Returns one record per pointer: ``{"config", "block", "pointer",
    "state", "detail"}`` where ``state`` is one of

    * ``"OK"`` — the named script exists AND its parser declares the dest;
    * ``"DECLARED-INERT"`` — the pointer does not resolve, but it is declared
      in :data:`DECLARED_INERT_DEST_POINTERS` with a reason ⇒ reported, not
      fatal;
    * ``"VIOLATION"`` — the pointer does not resolve and nobody declared it.

    A pointer that names a script which does not exist is a false claim.  A
    pointer that names an existing script whose parser does not declare the
    dest is ALSO a false claim.  Both are violations unless declared.
    """
    records: list[dict] = []
    cfg_name = cfg.name
    for block_path, pointer in L.iter_dest_pointers(data):
        key = f"{cfg_name}::{'.'.join(block_path)}"
        record = {
            "config": cfg_name,
            "block": ".".join(block_path),
            "pointer": pointer,
            "state": "OK",
            "detail": "",
        }
        script_name, sep, dest_name = str(pointer).partition("::")
        if not sep or not script_name.strip() or not dest_name.strip():
            record["state"] = "VIOLATION"
            record["detail"] = (
                f"the dest: pointer {pointer!r} is not of the form "
                "'<script>.py::<argparse_dest>'"
            )
        else:
            script_name = script_name.strip()
            dest_name = dest_name.strip()
            script_path = _resolve_pointer_script(script_name, entry_script_path)
            if script_path is None:
                record["state"] = "VIOLATION"
                record["detail"] = (
                    f"the dest: pointer names {script_name!r}, which does NOT "
                    "exist anywhere in the repo"
                )
            else:
                if script_path not in parser_cache:
                    ns, actions, err = _capture_parser(script_path)
                    if ns is None:
                        # A script with no argparse at all: its dest set is
                        # empty, so any pointer into it is false.
                        parser_cache[script_path] = set()
                    else:
                        parser_cache[script_path] = set(vars(ns).keys())
                dests = parser_cache[script_path]
                if dest_name not in dests:
                    record["state"] = "VIOLATION"
                    record["detail"] = (
                        f"the dest: pointer names {dest_name!r}, which is NOT "
                        f"a dest of {script_path.name}'s parser "
                        f"(it declares {len(dests)} dest(s))"
                    )
                else:
                    record["detail"] = (
                        f"{dest_name} is a dest of {script_path.name}'s parser"
                    )
        if record["state"] == "VIOLATION":
            declared = DECLARED_INERT_DEST_POINTERS.get(key)
            if declared:
                record["state"] = "DECLARED-INERT"
                record["declared_reason"] = declared
        records.append(record)
    return records


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
                    # 2026-11-04 (Wf-A269 frontier sweep) — the PRE-REPAIR
                    # rendering ran ONE check here (`_type_violation`).  The
                    # choices check was added beside it; both are run, and
                    # each violation carries its OWN reason so the counters
                    # below can tell them apart.
                    for _check in (_type_violation, _choices_violation):
                        bad = _check(path, value, dest, shape)
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

    ``strict`` used to turn a skipped config into a violation.  The flag used
    to be accepted and ignored ("interface parity"), which meant the one
    failure mode this gate cannot see — a config it never opened — was the one
    failure mode it never reported.  MEASURED 2026-10-21: eight
    ``configs/msrs_dmpnn_*.yaml`` files named no ``entry.script``, so they were
    skipped and the gate said PASS while checking nothing about them.

    **CHANGED 2026-11-04 (Wf-A269):** an UNDECLARED skip is now a violation
    **by default** — ``strict`` is retained for interface parity and now means
    exactly what the default means.  A skip that IS declared in
    :data:`DECLARED_SKIPPED_CONFIGS` (with a reason) is reported as
    ``DECLARED-SKIP`` and is not counted among the unchecked configs.  ⇒ the
    gate can no longer PASS while leaving a config unchecked and unnamed.
    """
    # ``strict`` is now the DEFAULT behaviour; the flag is kept so that a
    # caller passing it is not surprised, and so the interface does not move.
    del strict  # kept for interface parity; see the docstring
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
    declared_skipped: list[str] = []
    ns_cache: dict[Path, object] = {}
    act_cache: dict[Path, object] = {}
    # 2026-11-09 — the dest-pointer tier.  Separate cache from ns_cache: a
    # pointer names its OWN script, which is generally NOT the config's
    # entry.script.
    pointer_cache: dict[Path, set] = {}
    dest_pointer_records: list[dict] = []

    for cfg in configs:
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            violations.append({"config": cfg.name, "key": f"<unparseable: {exc}>"})
            continue
        entry = data.get("entry") if isinstance(data, dict) else None
        script, command = _entry_script_and_command(entry)
        if not script:
            declared_reason = DECLARED_SKIPPED_CONFIGS.get(cfg.name)
            if declared_reason is not None:
                # Disposition 2 — a DECLARED skip: reported every run, with its
                # reason on file, and NOT counted as an unchecked config.
                declared_skipped.append(f"{cfg.name} (DECLARED: {declared_reason})")
                continue
            # Disposition 3 — an UNDECLARED skip is a VIOLATION, always.
            # `--strict` used to be the only way to see it and the pre-flight
            # calls this gate without it, so the one failure mode the gate
            # could not see was the one it never reported (MEASURED
            # 2026-11-04: 2 of 261 configs were skipped while the gate said
            # PASS and its closing line claimed EVERY config was bound —
            # SUPERSEDED 2026-11-04 by the population measurement at
            # `DECLARED_SKIPPED_CONFIGS`: 500 audited, 257 tracked, 243
            # untracked, 2 declared skips, 0 undeclared).
            skipped.append(f"{cfg.name} (no entry.script and no .py in entry.command_template)")
            violations.append(
                {
                    "config": cfg.name,
                    "key": "entry.script absent — config cannot be bound to a trainer "
                    "(declare it in DECLARED_SKIPPED_CONFIGS with a reason, or "
                    "give it an entry.script)",
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

        # 2026-11-09 — the dest-pointer tier runs BEFORE the parser capture,
        # so a pointer is adjudicated even when the config's own entry.script
        # has no importable parser.
        for rec in _audit_dest_pointers(cfg, data, script_path, pointer_cache):
            dest_pointer_records.append(rec)
            if rec["state"] == "VIOLATION":
                violations.append(
                    {
                        "config": rec["config"],
                        "key": rec["block"],
                        "reason": "dest_pointer",
                        "detail": rec["detail"],
                    }
                )

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
            # 2026-11-04 (Wf-A269 frontier sweep) — the PRE-REPAIR rendering
            # was `n_dead = len(found) - n_type`, which counted a CHOICES
            # violation as "dead" (a key that resolves to no dest) and so
            # MISLABELLED it in this very tag.  The three reasons are now
            # counted separately.
            n_choices = sum(1 for f in found if f.get("reason") == "choices")
            n_dead = len(found) - n_type - n_choices
            tag = []
            if n_type:
                tag.append(f"{n_type} type")
            if n_choices:
                tag.append(f"{n_choices} choices")
            if n_dead:
                tag.append(f"{n_dead} dead")
            checked[-1] += f" ({', '.join(tag)})"

    return {
        "gate": "config_trainer_binding",
        "rule": (
            "every YAML key resolves to a dest of the parser named by that "
            "config's own entry.script (AGENTS.md §5.11 single-source-of-truth), "
            "and the value delivered to that dest has the type the argparse "
            "action declares (nargs='+'/'*' needs a list, type= must convert) "
            "and is inside the choices that action declares"
        ),
        "n_configs_checked": len(checked),
        "n_configs_skipped": len(skipped),
        "configs": checked,
        "skipped": skipped,
        "n_configs_declared_skipped": len(declared_skipped),
        "declared_skipped": declared_skipped,
        "n_declared_inert": len(declared_inert),
        "declared_inert": declared_inert,
        "n_direct_loader_keys": len(direct_keys_seen),
        "direct_loader_keys": direct_keys_seen,
        "n_type_violations": sum(1 for v in violations if v.get("reason") == "type"),
        # 2026-11-04 (Wf-A269 frontier sweep) — new counter for the choices
        # check.  The PRE-REPAIR report had no such key; the two counters above
        # are unchanged.
        "n_choices_violations": sum(
            1 for v in violations if v.get("reason") == "choices"
        ),
        # 2026-11-09 — the dest-pointer tier's own counters.  A counter that
        # is 0 because the tier never fired is not evidence, so the pointer
        # COUNT is reported alongside the violation count.
        "n_dest_pointers": len(dest_pointer_records),
        "n_dest_pointer_violations": sum(
            1 for r in dest_pointer_records if r["state"] == "VIOLATION"
        ),
        "n_declared_inert_dest_pointers": sum(
            1 for r in dest_pointer_records if r["state"] == "DECLARED-INERT"
        ),
        "dest_pointers": [
            f"{r['config']}: {r['block']} -> {r['pointer']} ({r['state']})"
            for r in dest_pointer_records
        ],
        "dest_pointer_records": dest_pointer_records,
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
        help=(
            "retained for interface parity — an UNDECLARED skip (a config with "
            "no entry.script and no .py in entry.command_template) is now a "
            "violation by DEFAULT; declare it in DECLARED_SKIPPED_CONFIGS with "
            "a reason to keep it skipped"
        ),
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
            elif v.get("reason") == "choices":
                # 2026-11-04 (Wf-A269 frontier sweep) — new branch.  The
                # PRE-REPAIR rendering had only the "type" branch and an
                # "else" that printed "resolves to no dest" for EVERY other
                # reason, so a choices violation would have been reported as a
                # DEAD KEY — a different defect with a different fix.
                print(
                    f"  VIOLATION {v['config']}: '{v['key']}' CHOICES — {v['detail']}"
                )
            elif v.get("reason") == "dest_pointer":
                # 2026-11-09 — the dest-pointer tier's own branch.  The
                # PRE-REPAIR rendering had only the "type" / "choices" /
                # "else" branches, so a dest-pointer violation would have been
                # printed as a DEAD KEY — a different defect with a different
                # fix.
                print(
                    f"  VIOLATION {v['config']}: '{v['key']}' DEST-POINTER — "
                    f"{v['detail']}"
                )
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
        for s in report["declared_skipped"]:
            print(f"  DECLARED-SKIP {s}")
        for rec in report["dest_pointer_records"]:
            if rec["state"] != "DECLARED-INERT":
                continue
            # 2026-11-09 — the prefix deliberately does NOT contain the
            # substring "DECLARED-INERT".  `pre_flight_check.py::gate_config_trainer_binding`
            # counts stdout lines containing that substring and labels the
            # count "declared-inert KEY(s) reported, not hidden" — so a
            # pointer line carrying the substring would inflate a KEY count
            # with pointers (MEASURED 2026-11-09 on the live run: 391 = 386
            # keys + 5 pointers).  A count whose label no longer matches its
            # population is the §6.269.40(6) class.  The pointer disclosure is
            # therefore prefixed with a distinct token, and the KEY count
            # stays at its pre-repair value.
            print(
                f"  DECLARED-POINTER-INERT {rec['config']}: "
                f"{rec['block']} -> {rec['pointer']}  ({rec['detail']})  "
                f"[{rec.get('declared_reason', '')}]"
            )
        if not report["violations"]:
            n_checked = report["n_configs_checked"]
            n_decl = report["n_configs_declared_skipped"]
            if n_decl:
                # 2026-11-04 (Wf-A269 frontier sweep) — the PRE-REPAIR closing
                # line read (PRESERVED VERBATIM per AGENTS.md §5.5 rule #6):
                #     "  every config key reaches a dest of its own entry.script "
                #     "parser, with the type that dest's argparse action declares "
                #     "({n_checked} config(s) bound; {n_decl} declared-skipped, "
                #     "each with a reason on file and NOT checked against a "
                #     "parser — they name none)"
                # It is now extended, ADDITIVELY, with the choices contract.
                # 2026-11-09 — extended ADDITIVELY again, with the dest-pointer
                # contract.  The PRE-REPAIR (2026-11-04) rendering is
                # PRESERVED VERBATIM above; the pointer clause is appended.
                print(
                    f"  every config key reaches a dest of its own entry.script "
                    f"parser, with the type that dest's argparse action declares "
                    f"and inside the choices that dest's argparse action declares "
                    f"({n_checked} config(s) bound; {n_decl} declared-skipped, "
                    f"each with a reason on file and NOT checked against a "
                    f"parser — they name none); and every schema-metadata "
                    f"dest: pointer names a dest of the script it names "
                    f"({report['n_dest_pointers']} pointer(s), "
                    f"{report['n_declared_inert_dest_pointers']} declared-inert)"
                )
            else:
                # 2026-11-04 (Wf-A269 frontier sweep) — the PRE-REPAIR closing
                # line read (PRESERVED VERBATIM per AGENTS.md §5.5 rule #6):
                #     "  every config key reaches a dest of its own entry.script parser, "
                #     "with the type that dest's argparse action declares"
                # It is now extended, ADDITIVELY, with the choices contract the
                # gate also checks.
                # 2026-11-09 — extended ADDITIVELY again, with the dest-pointer
                # contract.  The PRE-REPAIR (2026-11-04) rendering is
                # PRESERVED VERBATIM above; the pointer clause is appended.
                print(
                    "  every config key reaches a dest of its own entry.script parser, "
                    "with the type that dest's argparse action declares and inside "
                    "the choices that dest's argparse action declares; and every "
                    "schema-metadata dest: pointer names a dest of the script it "
                    f"names ({report['n_dest_pointers']} pointer(s), "
                    f"{report['n_declared_inert_dest_pointers']} declared-inert)"
                )
        print(f"[config-binding] {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
