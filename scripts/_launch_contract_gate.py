#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""AGENTS.md §5.11 launch-contract gate — what may a wrapper pass to THIS trainer?

Two gates, one rule, and this is the second one.

``scripts/_yaml_only_gate.py`` answers the §5.11 question: *is this flag one
of the four that may appear on a trainer invocation line?*  Its allowlist is
FIXED — the same four strings for every trainer.

That is the wrong oracle, and the repo's own run logs say so:

    molmetal/reports/wf125_p1_controlled_pair/run_synthetic/run.log:18
    train_metal_lomo_wasserstein.py: error: unrecognized arguments:
      --ckpt-dir .../ckpts --no-cuda-graphs --cuda-visible-devices 0

``--ckpt-dir`` is in §5.11's allowlist.  ``train_metal_lomo_wasserstein.py``
does not declare it.  **A wrapper can obey §5.11 literally and still die at
argument parsing.**  MEASURED 2026-11-03 over the 38 trainers whose parser
the repo can capture: ``--ckpt-dir`` is declared by **8**, ``--out-dir`` by
**29**, ``--config`` by **16**.

SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    argument parsing.**  MEASURED 2026-11-03 over the 38 trainers whose parser
    the repo can capture: ``--ckpt-dir`` is declared by **8**, ``--out-dir`` by
    **29**, ``--config`` by **16**.

CORRECTION (2026-11-04, MEASURED on this tree): the superseded rendering above
is STALE in its trainer COUNT — "the 38 trainers whose parser the repo can
capture" is no longer true.  MEASURED 2026-11-04 by driving this gate's own
``_declared_flags()`` over every ``molmetal/scripts/train_*.py``:
    n_trainer_files = 43, parsers captured = 43/43, coverage gaps = 0
All five trainers the superseded rendering names as unreadable are readable
now (MEASURED 2026-11-04, one line each):
    train_hybrid.py: parser_captured=True
    train_metal_hybrid_v4.py: parser_captured=True
    train_metal_hybrid_v4_mb1.py: parser_captured=True
    train_property_predictor.py: parser_captured=True
    train_symbolic_regression.py: parser_captured=True
(the zero-argument-``main()`` fix further down in this file is what made them
so).  The per-flag counts are UNCHANGED and still MEASURED-correct —
``--ckpt-dir`` **8**, ``--out-dir`` **29**, ``--config`` **16** — but they are
now out of 43, not out of 38.

So the legal flag set is PER-TRAINER, and the only authority on it is the
trainer's own parser.  This gate resolves each wrapper's trainer invocation
— including the flags that live inside a bash array, which is where a
launch surface hides them — and asserts that every flag is DECLARED by the
parser of the trainer that invocation runs.

Why a separate gate and not another check inside the yaml-only gate: the
yaml-only gate is a SOURCE scan and runs in the pre-flight, whose budget is
already contended (MEASURED 2026-11-01: 129 gates, 411 s).  This gate has to
IMPORT each trainer to read its parser.  MEASURED 2026-11-03: importing all
43 trainers and capturing every parser costs **9.7 s** in total (one trainer
pays 6.2 s for torch; the rest are lazy), so it is affordable — but it is a

SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:
    IMPORT each trainer to read its parser.  MEASURED 2026-11-03: importing all
    43 trainers and capturing every parser costs **9.7 s** in total (one trainer
    pays 6.2 s for torch; the rest are lazy), so it is affordable — but it is a

CORRECTION (2026-11-04, MEASURED on this tree): the **9.7 s** and the **6.2 s**
in the superseded rendering above are STALE for this host.  The 43-trainer
COUNT is still correct.  MEASURED 2026-11-04, seven consecutive CPU-only runs of
the import+capture pass over all 43 trainers: **3.264, 3.655, 4.189, 4.221,
4.241, 4.332, 4.581 s** — median **4.22 s**, range **3.3–4.6 s**.  The slowest SINGLE
trainer now costs **2.2 s** (train_affinity_regression.py), not 6.2 s; the
second is train_cfm_yuelbond_wasserstein.py at **1.9 s**; every other trainer
is under **0.02 s**.
MEASURED 2026-11-04 command (CPU-only, 0 GPU):
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python -c "
    import importlib.util, sys, time, pathlib
    REPO = pathlib.Path('/home/hugo/codes/mol-metal')
    spec = importlib.util.spec_from_file_location('lcg', REPO/'scripts/_launch_contract_gate.py')
    lcg = importlib.util.module_from_spec(spec); sys.modules['lcg'] = lcg
    spec.loader.exec_module(lcg)
    ts = sorted((REPO/'molmetal/scripts').glob('train_*.py'))
    t0 = time.perf_counter()
    ok = sum(1 for t in ts if lcg._declared_flags(t)[0] is not None)
    print(len(ts), ok, round(time.perf_counter()-t0, 3))"
    -> 43 43 <3.7-4.6>
The figure is host-load dependent (two agents work in this repo concurrently),
so the durable statement is "a few seconds", not a fixed constant.
different cost profile and a different failure mode (a trainer that will not
import is a COVERAGE GAP, not a violation), and it deserves its own tool and
its own report.

SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6
(Wf-A269 census repair, 2026-11-04):
    Fail-closed rule.  A flag the trainer does not declare is a violation
    (`undeclared_flag`).  A trainer whose parser cannot be captured is NOT a
    violation — it is a gap, and it is counted and listed in the report under
    `coverage`.  A gate that reported "the trainer would not import" as a
    violation would be right by accident and useless in practice; a gate that
    reported it as clean would be lying.  The report says which it is.

CORRECTION (Wf-A269 census repair, 2026-11-04): the coverage-gap half of
that rule was REVERSED on 2026-11-03 and the sentence above is STALE in it.
A trainer whose parser cannot be captured is now a VIOLATION
(`kind="coverage_gap"`), and a trainer INVOCATION whose trainer file cannot
be resolved is a VIOLATION too (`kind="unresolved_trainer"`).  The reasoning
the superseded rendering gives for the old rule ("a gate that called the
trainer's import failure a defect would be right by accident") does not
survive the repair of the underlying capture: with 43 of 43 parsers
readable, a gap means something else again, and a gate that skipped what it
could not read and still reported PASS is a check that can PASS WITHOUT
CHECKING (§6.269.19.5's defect class).

THE CENSUS, AND WHAT IT COUNTS (Wf-A269 census repair, 2026-11-04).

``_trainer_call`` — the shared shell-parsing helper — returns non-None for
ANY line whose TEXT mentions a ``train_*.py``.  That includes shell
assignments (``TRAINER="${SCRIPTS_DIR}/train_chemprop_v2_v3.py"``), JSON
receipt lines inside heredocs (``"trainer": "molmetal/scripts/train_cfm.py",``),
python assignments inside heredocs (``legacy_script = '...'``) and
comparisons (``Path(w).name == '...'``).  Counting those as "trainer calls"
made the counter claim a coverage the gate does not have.

⇒ **THE COUNTER NOW COUNTS WHAT IT CLAIMS TO COUNT.**  The report carries
three numbers instead of one:

``n_trainer_lines_seen``
    Every line on which ``_trainer_call`` returned non-None — the MENTION
    population.
``n_trainer_invocations`` (and its historical alias ``n_trainer_calls``)
    The lines whose COMMAND POSITION runs a trainer — the INVOCATION
    population, and exactly the population whose flags are checked.
``n_mentions``
    The difference: the lines that mention a trainer without invoking one.

⇒ A MENTION contributes NO flags, so it contributed NO violation: **the
verdicts are unchanged by this repair; only the census is corrected.**
MEASURED 2026-11-04 on this tree: 87 lines seen, 31 invocations, 56 mentions
(34 with no script in command position, 22 whose command position is not a
trainer).  The distinction is made by :func:`_trainer_name`, which tests the
ANCHORED basename of the command-position token.

Stdout invariant (AGENTS.md §4): PASS / FAIL / SKIP.  Exit 0=PASS, 1=FAIL,
2=SKIP (no wrapper to audit / no parser capturable at all).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_gate_module():
    """Import `scripts/_yaml_only_gate.py` — the shell-parsing home.

    It is imported by path (not as a package) because `scripts/` is not a
    package and the gate is a script, not a library.
    """
    path = REPO_ROOT / "scripts" / "_yaml_only_gate.py"
    spec = importlib.util.spec_from_file_location("_yaml_only_gate_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GATE = _load_gate_module()


#: A trainer FILE NAME — ANCHORED.  The shared helper's ``_TRAINER_CALL``
#: (``\btrain_[a-z0-9_]+\.py\b``) is a TEXT test: "does this line mention a
#: trainer?".  This one is a NAME test: "does this token name a trainer
#: FILE?".  The difference between the two is the difference between a
#: MENTION and an INVOCATION.
#:
#: The test is applied to the BASENAME of the expanded token, because the
#: trainer test is a file-NAME test: an unresolvable directory prefix cannot
#: turn ``helper_check.py`` into ``train_x.py``.  It is ANCHORED at both ends
#: so that a token which merely CONTAINS a trainer name — a runtime command
#: string, a JSON receipt value, a python assignment — does not pass: the
#: basename of such a token is not a bare trainer file name.
_TRAINER_NAME = re.compile(r"^train_[a-z0-9_]+\.py$")


# ---------------------------------------------------------------------------
# Trainer parser capture
# ---------------------------------------------------------------------------
def _declared_flags(script_path: Path) -> tuple[set[str] | None, str | None]:
    """Every long option the trainer's parser declares, or an error.

    The set is read off ``parser._actions[*].option_strings`` — the FULL list.
    The repo's ``_config_binding_gate._actions_by_dest`` keeps ONE option per
    dest (``max(longs, key=len)``), and for an ``argparse.BooleanOptionalAction``
    that drops one of the pair: MEASURED 2026-11-03,
    ``train_metal_lomo_wasserstein.py`` declares ``--triton-ot`` with
    ``BooleanOptionalAction`` (so the parser accepts BOTH ``--triton-ot`` and
    ``--no-triton-ot``), while the dest-keyed view reports only
    ``--no-triton-ot`` — the longer of the two.  A launch-contract check built
    on the dest-keyed view would report a LEGAL flag as undeclared.
    """
    spec = importlib.util.spec_from_file_location(
        "wfa269_lc_" + script_path.stem, script_path
    )
    if spec is None or spec.loader is None:
        return None, f"cannot load {script_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — a trainer that won't import is a gap
        return None, f"import failed: {type(exc).__name__}: {exc}"[:120]

    import argparse as _ap

    captured: dict = {}
    originals = {
        "parse_args": _ap.ArgumentParser.parse_args,
        "parse_known_args": _ap.ArgumentParser.parse_known_args,
    }

    def make():
        def spy(self, args=None, namespace=None):
            captured["parser"] = self
            raise _Stop()

        return spy

    class _Stop(Exception):
        pass

    _ap.ArgumentParser.parse_args = make()
    _ap.ArgumentParser.parse_known_args = make()
    old_argv = sys.argv
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
            except BaseException:  # noqa: BLE001
                pass
            if "parser" in captured:
                break
        if "parser" not in captured and hasattr(mod, "main"):
            # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5
            # rule #6 (Wf-A269, 2026-11-03):
            #     if "parser" not in captured and hasattr(mod, "main"):
            #         try:
            #             mod.main([])
            #         except BaseException:  # noqa: BLE001
            #             pass
            #
            # `mod.main([])` was called UNCONDITIONALLY, but most trainers
            # declare `def main():` with NO parameters — so the call raised
            # `TypeError: main() takes 0 positional arguments but 1 was
            # given`, the `except BaseException` swallowed it, and the trainer
            # was reported as "could not capture a parser namespace".
            #
            # MEASURED 2026-11-03 over the corpus
            # `molmetal/scripts/train_*.py` (43 files): 5 trainers were
            # unreadable for exactly this reason —
            #   train_hybrid.py, train_metal_hybrid_v4.py,
            #   train_metal_hybrid_v4_mb1.py, train_property_predictor.py,
            #   train_symbolic_regression.py
            # — all five with a zero-argument `main()`.  The other 38
            # captured their parser.  This is the §6.269.18.11 class: a tool
            # that is wrong in a way only a live test catches (the file
            # *looks* fine; the parser is simply never reached).
            #
            # THE FIX: try the zero-argument call FIRST, then the one-argument
            # form, so BOTH `def main():` and `def main(argv):` are reachable.
            for _call in (lambda: mod.main(), lambda: mod.main([])):
                try:
                    _call()
                except TypeError:
                    continue  # wrong arity for THIS main — try the other form
                except BaseException:  # noqa: BLE001
                    pass
                if "parser" in captured:
                    break
                break
    finally:
        sys.argv = old_argv
        _ap.ArgumentParser.parse_args = originals["parse_args"]
        _ap.ArgumentParser.parse_known_args = originals["parse_known_args"]

    parser = captured.get("parser")
    if parser is None:
        return None, "could not capture a parser namespace"
    flags: set[str] = set()
    for action in getattr(parser, "_actions", ()):
        for opt in getattr(action, "option_strings", ()) or ():
            if opt.startswith("--"):
                flags.add(opt)
    return flags, None


# ---------------------------------------------------------------------------
# Trainer resolution
# ---------------------------------------------------------------------------
def _trainer_name(script_token: str, env: dict) -> str | None:
    """The trainer FILE NAME a script token names, or None.

    The token is expanded against the wrapper's own assignments and its
    BASENAME is tested against :data:`_TRAINER_NAME`.  A token that names a
    trainer file returns that name; anything else — a JSON receipt key, a
    python assignment, a bash array literal, a runtime command string —
    returns None.

    MEASURED 2026-11-04 over the live tree: of the 87 lines the shared
    ``_trainer_call`` accepts, this function names a trainer on **31** and
    returns None on **56** (34 whose command position holds no script at all,
    22 whose command position holds something that is not a trainer).
    """
    expanded = GATE._expand(GATE._unquote(script_token), env)
    basename = expanded.rstrip("/").split("/")[-1]
    return basename if _TRAINER_NAME.match(basename) else None


def _trainer_file(script_token: str, env: dict) -> Path | None:
    """The trainer file a script token names, or None."""
    expanded = GATE._unquote(GATE._expand(script_token, env))
    if not GATE._TRAINER_CALL.search(expanded):
        return None
    candidate = Path(expanded)
    if candidate.is_file():
        return candidate
    # A relative or ${SCRIPTS_DIR}-style path: fall back to the repo's own
    # trainer directory, matched on the basename.
    name = candidate.name
    for root in (REPO_ROOT / "molmetal" / "scripts", REPO_ROOT / "scripts"):
        p = root / name
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(launchers: list[Path] | None = None) -> dict:
    """Audit every launch wrapper against the parser of the trainer it runs."""
    coverage: dict
    if launchers is None:
        launchers, coverage = GATE._population()
    else:
        _default_paths, coverage = GATE._population()
        coverage = dict(coverage)
        coverage["population_source"] = "explicit --wrapper list"
        coverage["population_degraded"] = False
        coverage["n_scanned"] = len(launchers)
        coverage["n_excluded"] = 0
        coverage["coverage_pct"] = None
        coverage["excluded"] = []

    violations: list[dict] = []
    scanned: list[str] = []
    trainer_cache: dict[str, tuple[set[str] | None, str | None]] = {}
    coverage_gaps: dict[str, str] = {}
    mentions: list[dict] = []
    trainer_lines_seen = 0
    trainer_calls = 0
    unresolved_trainers = 0
    flags_checked = 0

    for wrapper in launchers:
        if not wrapper.exists():
            continue
        rel = GATE._rel(wrapper)
        if rel not in scanned:
            scanned.append(rel)
        text = wrapper.read_text(encoding="utf-8", errors="replace")
        env = GATE._collect_assignments(text)

        for lineno, line in GATE._logical_lines(text):
            via = GATE._trainer_call(line, env)
            if via is None:
                continue
            # A line the shared helper accepts: it MENTIONS a trainer.
            trainer_lines_seen += 1

            toks = GATE._shell_tokens(GATE._strip_detach_head(line))
            idx = GATE._script_index(toks, env)
            script_token = toks[idx] if idx is not None else None
            if script_token is None:
                # A MENTION, not an invocation: the line names a trainer but
                # its command position holds NO SCRIPT — a shell assignment
                # (`TRAINER=".../train_x.py"`), an `echo` of the command, a
                # `printf`.  Such a line contributes no argv and therefore no
                # flag; counting it as a "trainer call" overstated the
                # gate's coverage (Wf-A269 census repair, 2026-11-04).
                mentions.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "mention_no_script_in_command_position",
                        "via": via,
                    }
                )
                continue

            name = _trainer_name(script_token, env)
            if name is None:
                # A MENTION: the command position holds a script, but not a
                # trainer's — a JSON receipt key (`"trainer": "..."`), a python
                # assignment (`legacy_script = '...'`), a comparison
                # (`Path(w).name == '...'`), a bash array literal, or a
                # runtime command string (`bash -c "$TRAINER_CMD ..."`).
                mentions.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "mention_command_is_not_a_trainer",
                        "via": via,
                    }
                )
                continue

            # A real INVOCATION: the command position runs a trainer.  This is
            # the population whose flags are checked, and the population the
            # `n_trainer_invocations` counter reports.
            trainer_calls += 1
            trainer = _trainer_file(script_token, env)
            if trainer is None:
                # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md
                # section 5.5 rule #6 (Wf-A269 census repair, 2026-11-04):
                #     if trainer is None:
                #         continue  # not a trainer this gate can name (e.g. a helper)
                #
                # That branch was FAIL-OPEN in the §6.269.19.5 sense: a
                # trainer INVOCATION whose trainer file cannot be resolved was
                # SKIPPED, and the gate still reported PASS, having checked
                # nothing for it.  A check that can PASS WITHOUT CHECKING is
                # the defect class this gate exists to close, and it had the
                # defect inside itself.
                #
                # The branch also conflated two populations: a line whose
                # command position does not run a trainer at all (a MENTION)
                # and a line that DOES run a trainer the gate cannot find on
                # disk (a renamed trainer, a typo'd path, a deleted trainer).
                # The first is not this gate's business; the second is exactly
                # its business and must not be skipped.
                #
                # THE FIX: the MENTION population is now classified BEFORE
                # this branch (above), so the only lines that reach it are
                # genuine INVOCATIONS — and an invocation whose trainer cannot
                # be resolved is a VIOLATION (fail-closed), because the gate
                # cannot read that trainer's parser and therefore cannot
                # prove the flags on the line are declared.
                unresolved_trainers += 1
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "unresolved_trainer",
                        "via": via,
                        "trainer": name,
                        "detail": (
                            f"the invocation runs the trainer {name}, but this "
                            "gate cannot find that file on disk, so it cannot "
                            "read the trainer's parser and cannot prove that "
                            "the flags on this invocation — nor any other "
                            "flag on it — are declared; a check that cannot "
                            "check must not certify the launch"
                        ),
                    }
                )
                continue

            key = str(trainer)
            if key not in trainer_cache:
                trainer_cache[key] = _declared_flags(trainer)
            declared, err = trainer_cache[key]

            argv, unresolved, _excluded = GATE._trainer_argv(line, env)
            flags: set[str] = set()
            for tok in argv:
                flags.update(GATE._FLAG.findall(GATE._unquote(tok)))

            if declared is None:
                # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section
                # 5.5 rule #6 (Wf-A269, 2026-11-03):
                #     if declared is None:
                #         coverage_gaps[rel] = err or "unknown"
                #         continue
                #
                # That branch was FAIL-OPEN: an unreadable parser made the
                # gate SKIP the wrapper's flags entirely and still report
                # PASS — a check that can PASS WITHOUT CHECKING, which is the
                # §6.269.19.5 defect class (the same mistake the migration
                # wave made with "0 yaml-only violations").  The flags below
                # were never counted, so the report's `n_flags_checked` did
                # not even reveal the omission.
                #
                # THE FIX: a coverage gap is now a VIOLATION (fail-closed).
                # The gate cannot prove those flags are legal, so it must not
                # certify the wrapper.
                coverage_gaps[rel] = err or "unknown"
                for flag in sorted(flags):
                    violations.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "kind": "coverage_gap",
                            "via": via,
                            "trainer": GATE._rel(trainer),
                            "detail": (
                                f"the gate could not read the parser of "
                                f"{GATE._rel(trainer)} ({err or 'unknown'}), so it "
                                f"cannot prove that {flag} — nor any other flag on "
                                "this invocation — is declared; a check that cannot "
                                "check must not certify the launch"
                            ),
                        }
                    )
                continue

            for flag in sorted(flags):
                flags_checked += 1
                if flag in declared:
                    continue
                violations.append(
                    {
                        "file": rel,
                        "line": lineno,
                        "kind": "undeclared_flag",
                        "via": via,
                        "trainer": GATE._rel(trainer),
                        "detail": (
                            f"{flag} is passed to {GATE._rel(trainer)}, whose parser "
                            f"does not declare it; the launch would abort with "
                            f'"unrecognized arguments" before the run starts '
                            "(§5.11: the legal flag set is defined by the "
                            "trainer's own parser, not by a fixed allowlist)"
                        ),
                    }
                )

    return {
        "gate": "launch_contract",
        "rule": (
            "AGENTS.md §5.11 — every flag a launch wrapper passes to a trainer "
            "must be declared by that trainer's parser"
        ),
        "n_wrappers_scanned": len(scanned),
        # The gate states its own coverage — the same block the yaml-only
        # gate prints — so the two gates can never disagree about what they
        # audited (§6.269.13: a gate that cannot report its denominator).
        "coverage": coverage,
        "excluded_residual": GATE._residual_census(coverage.get("excluded", [])),
        "wrappers": scanned,
        # The INVOCATION count — what the gate actually checks.  It used to
        # count every line that MENTIONED a trainer (see the module
        # docstring); the historical key name is kept, its meaning is not.
        "n_trainer_calls": trainer_calls,
        "n_trainer_invocations": trainer_calls,
        # The MENTION population the old counter was really measuring.
        "n_trainer_lines_seen": trainer_lines_seen,
        "n_mentions": len(mentions),
        "mentions": mentions,
        "n_flags_checked": flags_checked,
        "n_trainers_resolved": len(trainer_cache),
        "n_unresolved_trainers": unresolved_trainers,
        "coverage_gaps": coverage_gaps,
        "n_coverage_gaps": len(coverage_gaps),
        "n_violations": len(violations),
        "violations": violations,
        # A DEGRADED population cannot certify coverage, so it cannot PASS
        # (the yaml-only gate's rule, applied identically here).
        "verdict": (
            "FAIL"
            if (violations or coverage.get("population_degraded"))
            else "PASS"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    ap.add_argument(
        "--wrapper",
        action="append",
        default=None,
        metavar="PATH",
        help="audit only this wrapper (repeatable)",
    )
    args = ap.parse_args(argv)

    launchers = None
    if args.wrapper:
        launchers = [Path(w) for w in args.wrapper]
    report = audit(launchers)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        kinds: dict[str, int] = {}
        for m in report["mentions"]:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        breakdown = (
            ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
            or "none"
        )
        GATE._print_coverage(report, tag="launch-contract")
        print(
            f"[launch-contract] scanned {report['n_wrappers_scanned']} wrapper(s), "
            f"{report['n_trainer_invocations']} trainer invocation(s), "
            f"{report['n_flags_checked']} flag(s) checked"
        )
        print(
            f"[launch-contract] CENSUS: {report['n_trainer_lines_seen']} line(s) "
            f"mention a trainer; {report['n_trainer_invocations']} are invocations, "
            f"{report['n_mentions']} are mentions ({breakdown})"
        )
        for v in report["violations"]:
            loc = f"{v['file']}:{v['line']}"
            print(f"  VIOLATION {loc} [{v['kind']}] {v['detail']}")
        if report["n_coverage_gaps"]:
            print(
                f"[launch-contract] COVERAGE GAP: {report['n_coverage_gaps']} "
                "wrapper(s) name a trainer whose parser could not be captured; "
                "their flags were NOT checked, so the gate CANNOT certify them "
                "and the verdict is FAIL"
            )
            for f, err in sorted(report["coverage_gaps"].items()):
                print(f"    {f}: {err}")
        print(f"[launch-contract] {report['verdict']}")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
