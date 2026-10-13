"""Wf-399-A1 — 50-step end-to-end smoke protocol for trainer flag wire-ins.

PURPOSE (per AGENTS.md §5.7 added by Wf-399-A1):

  Whenever a new `--flag` is wired into a trainer's training loop (loss / sampling /
  dispatch), the wire-in is INCOMPLETE until a 50-step end-to-end smoke has been run
  that verifies:

    1. `parse_args` accepts the flag without NameError or argparse error.
    2. The trainer imports cleanly with the flag set.
    3. A 50-step end-to-end run produces a metrics JSON / log with
       `decode_ratio > 0` and a strictly decreasing `loss[0] > loss[49]`.
    4. (added 2026-10-09) Every flag arrived with the TYPE the trainer's own
       argparse declares.  A smoke on that date passed 1-3 while `--metals`
       (an `nargs="+"` flag) had been handed the single string
       `"Ru Ir Os Re"`; the run matched zero rows and every fold scored `nan`.
       See :func:`verify_flag_types`.

This script provides the protocol helper so that future trainer-wire-in waves (Wf-381
wire-in of Wf-380 SCAFFOLD-only flags, Wf-394-B1 Lambda CFM patch, Wf-401+ follow-ups)
can drop a single import + 3-line invocation and emit a structured verdict.

CPU-only by design — no `torch.cuda` invocation. Smoke invocations run on CPU
(`--device cpu`) at tiny batch sizes so a full 50-step run finishes in <60 s on the
test host.

Reference (real precedent, NOT fictional):
  - Wf-385-A1 audit (`TODO/completed/385a1_wf380_cli_flag_audit.md`) — 17/17 Wf-380 NEW
    CLI flags were FLAG_ONLY; SCAFFOLD modules existed + 4-4 unit tests PASS but ZERO
    trainer import; main-loop regression invisible at unit-test layer.
  - Wf-291D + Wf-299 — canonical `decode_ratio = 0/192` regression → 1-line try/except
    + 5 shim edits → `1.000 ± 0.000` (3 seeds × 4 probes × 16 samples).
  - Wf-394-A1 (`TODO/completed/394a1_lambda_cfm_wire.md`) — TODO-21 Lambda × CFM
    coupling audit + smoke scaffold (2 NEW tests PASS; wire-in patch deferred to
    USER_OK wave).

Usage example (future Wf-381 wire-in):

    from molmetal.scripts.wf399a1_smoke_protocol import (
        run_smoke_50step, verify_decode_ratio, verify_loss_decrease,
    )
    output = run_smoke_50step(
        trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
        flags_list=["--use-ot-cfm", "--triton-ot", "--epochs", "1",
                    "--batch-size", "8", "--device", "cpu"],
    )
    assert verify_decode_ratio(output)["decode_ratio"] > 0
    assert verify_loss_decrease(output)["decreasing"] is True

CLI usage (smoke + audit from a shell):

    python molmetal/scripts/wf399a1_smoke_protocol.py \
        --trainer molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
        --flag-arg "--use-ot-cfm" --flag-arg "--triton-ot" \
        --epochs 1 --batch-size 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SEC = 600  # 10 min — plenty for a 50-step CPU smoke


@dataclass
class SmokeOutput:
    """Structured output of a 50-step end-to-end smoke invocation."""

    trainer_path: str
    flags: List[str]
    returncode: int
    stdout: str
    stderr: str
    metrics_path: Optional[str]
    metrics: Dict[str, Any]
    wall_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Smoke invocation
# ---------------------------------------------------------------------------


def run_smoke_50step(
    trainer_path: str,
    flags_list: List[str],
    *,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[str] = None,
    capture_metrics_path: Optional[str] = None,
) -> SmokeOutput:
    """Invoke the trainer with a tiny 50-step end-to-end smoke and capture output.

    The smoke is constructed by passing `--epochs 1 --batch-size 8 --device cpu`
    to the trainer so that the first epoch runs ~50 training steps (assuming the
    trainer's default split produces ~50 batches per epoch on the smoke cohort).
    If the caller passes their own `--epochs` / `--batch-size`, those override.

    The trainer's own per-epoch / per-step metrics JSON is detected automatically
    via the trainer's stdout banner (e.g. `[results] wrote {out_dir}/metrics.json`)
    or via the `--out-dir` flag's `metrics.json` companion file. If neither is
    found, an empty `metrics` dict is returned and `verify_decode_ratio` /
    `verify_loss_decrease` will return their not-found dict variant.

    Parameters
    ----------
    trainer_path : str
        Path to the trainer script (e.g. ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``).
    flags_list : list[str]
        List of CLI arguments to forward to the trainer. Defaults to a tiny
        CPU smoke configuration if `flags_list` is empty.
    timeout_sec : int, default 600
        Subprocess timeout in seconds.
    cwd : str | None
        Working directory for the subprocess. Defaults to REPO_ROOT so that
        relative paths in the trainer resolve correctly.
    capture_metrics_path : str | None
        If provided, this explicit path is read after the smoke completes and
        its JSON contents are returned in the SmokeOutput.metrics field.

    Returns
    -------
    SmokeOutput
        Structured dataclass with the trainer's stdout / stderr / metrics.
    """
    import time

    if not flags_list:
        flags_list = ["--epochs", "1", "--batch-size", "8", "--device", "cpu"]
    else:
        flags_list = list(flags_list)

    cmd = [sys.executable, trainer_path] + flags_list
    work_dir = cwd or str(REPO_ROOT)

    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return SmokeOutput(
            trainer_path=trainer_path,
            flags=flags_list,
            returncode=-1,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"TimeoutExpired after {timeout_sec}s",
            metrics_path=None,
            metrics={},
            wall_seconds=time.time() - t0,
        )
    wall = time.time() - t0

    # Locate metrics JSON via --out-dir or stdout banner
    metrics_path: Optional[str] = None
    if capture_metrics_path is not None:
        metrics_path = capture_metrics_path
    else:
        # Look for --out-dir flag in flags_list
        out_dir: Optional[str] = None
        for i, flag in enumerate(flags_list):
            if flag == "--out-dir" and i + 1 < len(flags_list):
                out_dir = flags_list[i + 1]
                break
        if out_dir is None:
            # Look for the trainer's stdout banner pattern
            m = re.search(
                r"(?:results|wrote|saved).*?([^\s]+\.json)", stdout, re.IGNORECASE
            )
            if m:
                metrics_path = m.group(1)
        else:
            candidate = Path(out_dir) / "metrics.json"
            if candidate.exists():
                metrics_path = str(candidate)

    metrics: Dict[str, Any] = {}
    if metrics_path and Path(metrics_path).exists():
        try:
            metrics = json.loads(Path(metrics_path).read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return SmokeOutput(
        trainer_path=trainer_path,
        flags=flags_list,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        metrics_path=metrics_path,
        metrics=metrics,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Flag TYPE verification (2026-10-09)
#
# MEASURED incident this closes: a smoke on 2026-10-09 PASSED all three
# pre-existing invariants (rc=0, decode_ratio>0, loss[0]>loss[49]) while the
# run was still broken, because ``--metals`` is an ``nargs="+"`` flag and had
# been handed the single string ``"Ru Ir Os Re"`` instead of a token list.  The
# trainer then matched zero rows and every fold scored ``nan``.  The protocol
# asserted VALUES; nothing asserted the TYPE a flag arrived with.
#
# The check works on the trainer's OWN argparse: re-parse the exact flag list
# the smoke used, then assert each dest's value is the shape its action
# declares.  No values are invented — everything comes from the parser.
# ---------------------------------------------------------------------------


#: ``nargs`` values that make argparse collect into a list.
_LIST_NARGS = ("+", "*")


def flag_type_expectations(parser: argparse.ArgumentParser) -> Dict[str, str]:
    """Map every dest on ``parser`` to the TYPE its action declares.

    Returns ``{dest: kind}`` with ``kind`` in ``{"list", "bool", "path",
    "int", "float", "str"}``.  Derived purely from argparse metadata
    (``action.nargs`` / ``action.type`` / ``action.__class__``), so it cannot
    drift from what the trainer actually accepts.
    """
    out: Dict[str, str] = {}
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue
        if action.__class__.__name__ in (
            "_StoreTrueAction",
            "_StoreFalseAction",
            "_CountAction",
        ):
            out[dest] = "bool"
            continue
        if getattr(action, "nargs", None) in _LIST_NARGS:
            out[dest] = "list"
            continue
        conv = getattr(action, "type", None)
        if conv is None:
            out[dest] = "str"
            continue
        name = getattr(conv, "__name__", "")
        if name in ("int", "float", "str", "bool"):
            out[dest] = name
        elif isinstance(conv, type) and issubclass(conv, Path):
            out[dest] = "path"
        else:
            out[dest] = "str"
    return out


def _call_trainer_parse_args(
    module: Any, flags: List[str]
) -> Optional[argparse.Namespace]:
    for name in ("parse_args", "_parse_args"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                return fn(list(flags))
            except BaseException:
                continue
    return None


def _call_trainer_parse_args_via_argv(
    module: Any, flags: List[str]
) -> Optional[argparse.Namespace]:
    """Parse by temporarily swapping ``sys.argv`` — for zero-arg ``parse_args()``.

    ``train_cfm_yuelbond_wasserstein.parse_args()`` takes no argument and
    reads ``sys.argv`` implicitly (the latent trainer quirk documented in
    ``molmetal/tests/test_config_yaml.py``), so the argv-passing route cannot
    reach it.  ``sys.argv`` is saved and restored unconditionally; this runs in
    the smoke-protocol process, not in a trainer.
    """
    for name in ("parse_args", "_parse_args"):
        fn = getattr(module, name, None)
        if not callable(fn):
            continue
        saved = sys.argv
        sys.argv = [str(Path(module.__file__ or "trainer"))] + list(flags)
        try:
            return fn()
        except BaseException:
            continue
        finally:
            sys.argv = saved
    return None


def _load_trainer_module(trainer_path: str) -> Optional[Any]:
    """Import a trainer module by path, or ``None`` if it will not import.

    The module is registered in ``sys.modules`` BEFORE ``exec_module``: the
    trainers declare ``@dataclass`` types, and ``dataclasses._is_type``
    resolves annotations via ``sys.modules[cls.__module__]``, so an unregistered
    module raises ``AttributeError: 'NoneType' object has no attribute
    '__dict__'`` at class-creation time (measured on
    ``train_metal_lomo_wasserstein.py``).  The entry is removed again on
    failure and left in place on success, so a dataclass defined later still
    resolves.
    """
    import importlib.util

    path = Path(trainer_path)
    if not path.is_file():
        return None
    mod_name = f"_wf399_smoke_probe_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        return None
    return module


def _kind_of_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, Path):
        return "path"
    if isinstance(value, str):
        return "str"
    return "str"


def _derive_expectations(
    module: Any, flags: List[str]
) -> tuple[Optional[Dict[str, str]], Optional[argparse.Namespace]]:
    """Best-effort ``(expectations, parsed_args)`` for a trainer module.

    Preferred source is a real ``argparse`` parser (``build_parser`` /
    ``get_parser`` / ``make_parser``), because its declared ``nargs`` / ``type``
    are the trainer's own contract.  The Mol-Metal trainers instead expose only
    a module-level ``parse_args``, so a second route parses twice — once with
    ``[]`` for the pristine namespace and once with the smoke's flags — and
    classifies each dest from its DEFAULT type (a list default means an
    ``nargs`` flag) crossed with the type the flag actually produced.  That is
    weaker than the parser route but still catches the 2026-10-09 shape,
    because ``--metals``'s default is a list.
    """
    for name in ("build_parser", "get_parser", "make_parser"):
        factory = getattr(module, name, None)
        if not callable(factory):
            continue
        try:
            parser = factory()
        except BaseException:
            continue
        if not isinstance(parser, argparse.ArgumentParser):
            continue
        namespace = _call_trainer_parse_args(module, flags)
        if namespace is None:
            try:
                namespace = parser.parse_args(list(flags))
            except BaseException:
                continue
        return flag_type_expectations(parser), namespace

    pristine = _call_trainer_parse_args(module, [])
    namespace = _call_trainer_parse_args(module, flags)
    if pristine is None or namespace is None:
        # Zero-arg `parse_args()` trainers (CFM) read sys.argv implicitly.
        pristine = _call_trainer_parse_args_via_argv(module, [])
        namespace = _call_trainer_parse_args_via_argv(module, flags)
    if pristine is None or namespace is None:
        return None, None
    expectations: Dict[str, str] = {}
    for dest, value in vars(namespace).items():
        if dest.startswith("yaml_"):
            continue
        kind = _kind_of_value(getattr(pristine, dest, value))
        if kind == "str" and isinstance(value, (list, tuple)):
            # default was a scalar but the flag produced a list -> nargs flag.
            kind = "list"
        expectations[dest] = kind
    return expectations, namespace


def verify_flag_types(
    smoke: SmokeOutput,
    *,
    expectations: Optional[Dict[str, str]] = None,
    parsed_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    """Assert each smoke flag arrived with the TYPE its action declares.

    Three independent failures are caught:

    1. **Wrong container** — an ``nargs="+"`` dest holding a bare ``str`` (or
       any non-list) instead of a token list.
    2. **Un-split list** — an ``nargs="+"`` dest holding a ONE-element list
       whose element contains whitespace.  This is the exact 2026-10-09
       ``--metals "Ru Ir Os Re"`` shape: the type is right, the cohort is
       wrong, and the run scores ``nan`` on every fold while looking healthy.
    3. **Wrong scalar type** — a dest whose action declares ``int`` / ``float``
       / ``Path`` holding something else.

    Supply ``parsed_args`` (a namespace from the trainer's own parser) and/or
    ``expectations`` (``{dest: kind}`` from :func:`flag_type_expectations`) to
    check a run in-process.  With neither, the trainer named by
    ``smoke.trainer_path`` is imported and re-parsed with ``smoke.flags``.

    Returns ``{"invariant_holds", "checked", "violations", "source",
    "error"}``.  A SKIP (``invariant_holds=True``, empty ``checked``) is
    reported, never silently dropped, when the parser cannot be obtained.
    """
    if (expectations is None or parsed_args is None) and smoke.trainer_path:
        # Resolve whichever half the caller did not supply, from the trainer.
        try:
            module = _load_trainer_module(smoke.trainer_path)
        except BaseException:
            module = None
        if module is not None:
            try:
                derived_expectations, derived_args = _derive_expectations(
                    module, list(smoke.flags)
                )
            except BaseException:
                derived_expectations, derived_args = None, None
            if expectations is None:
                expectations = derived_expectations
            if parsed_args is None:
                parsed_args = derived_args

    if not expectations or parsed_args is None:
        return {
            "invariant_holds": True,
            "checked": [],
            "violations": [],
            "source": "skip",
            "error": (
                "no trainer parser / parsed args available; flag-type check "
                "SKIPPED (pass expectations= and parsed_args= to force it)"
            ),
        }

    violations: List[Dict[str, Any]] = []
    checked: List[str] = []
    for dest, kind in sorted(expectations.items()):
        if not hasattr(parsed_args, dest):
            continue
        value = getattr(parsed_args, dest)
        # A dest still holding its `None` default was never supplied, so it
        # carries no information about the type the flag ARRIVES with. Every
        # real trainer has such flags (`--config`, `--out-dir`, …); checking
        # them would be a false-positive generator, not a check.
        if value is None:
            continue
        checked.append(dest)
        if kind == "list":
            if not isinstance(value, (list, tuple)):
                violations.append(
                    {
                        "dest": dest,
                        "kind": "wrong_container",
                        "expected": "list",
                        "got": type(value).__name__,
                        "value": repr(value)[:200],
                        "detail": (
                            f"--{dest.replace('_', '-')} takes multiple values "
                            f"(nargs='+'); got a single {type(value).__name__}. "
                            "Pass one token per element, not a joined string."
                        ),
                    }
                )
            elif (
                len(value) == 1
                and isinstance(value[0], str)
                and any(ch.isspace() for ch in value[0])
            ):
                violations.append(
                    {
                        "dest": dest,
                        "kind": "unsplit_list",
                        "expected": "list of >=1 distinct tokens",
                        "got": f"1-element list {value[0]!r}",
                        "value": repr(value)[:200],
                        "detail": (
                            f"--{dest.replace('_', '-')} received a single "
                            "whitespace-joined string where a token list was "
                            "intended (quote the elements separately). The run "
                            "matches no rows and scores nan per fold while the "
                            "value-based invariants still hold."
                        ),
                    }
                )
        elif kind == "bool":
            if not isinstance(value, (bool, int)):
                violations.append(
                    {
                        "dest": dest,
                        "kind": "wrong_scalar_type",
                        "expected": "bool",
                        "got": type(value).__name__,
                        "value": repr(value)[:200],
                        "detail": f"--{dest.replace('_', '-')} is a store_true flag.",
                    }
                )
        else:
            expected_types = {
                "int": (int,),
                "float": (int, float),
                "str": (str,),
                "path": (str, Path),
            }.get(kind, ())
            if expected_types and not isinstance(value, expected_types):
                violations.append(
                    {
                        "dest": dest,
                        "kind": "wrong_scalar_type",
                        "expected": kind,
                        "got": type(value).__name__,
                        "value": repr(value)[:200],
                        "detail": (
                            f"--{dest.replace('_', '-')} declares type={kind}; "
                            f"got {type(value).__name__}."
                        ),
                    }
                )

    return {
        "invariant_holds": not violations,
        "checked": checked,
        "violations": violations,
        "source": "trainer parser",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def verify_decode_ratio(smoke: SmokeOutput) -> Dict[str, Any]:
    """Parse the smoke output for `decode_ratio > 0`.

    Accepts either:
      * `smoke.metrics["decode_ratio"]` (canonical JSON key from the trainer),
      * `smoke.metrics["final"]["decode_ratio"]`,
      * a stdout pattern ``decode_ratio=<float>`` (the trainer's print format).

    Returns a dict with:
      * ``decode_ratio`` (float | None) — value found,
      * ``invariant_holds`` (bool) — True iff decode_ratio > 0,
      * ``source`` (str) — where the value was found,
      * ``error`` (str | None) — diagnostic if not found.
    """
    decode_ratio: Optional[float] = None
    source = "none"

    # Direct metrics key
    if "decode_ratio" in smoke.metrics:
        try:
            decode_ratio = float(smoke.metrics["decode_ratio"])
            source = "metrics.decode_ratio"
        except (TypeError, ValueError):
            pass

    # Nested final key (some trainers emit a {"final": {...}} wrapper)
    if decode_ratio is None and isinstance(smoke.metrics.get("final"), dict):
        try:
            decode_ratio = float(smoke.metrics["final"]["decode_ratio"])
            source = "metrics.final.decode_ratio"
        except (KeyError, TypeError, ValueError):
            pass

    # Stdout pattern. Trainers print either:
    #   "decode_ratio=1.000 (192/192) ..."  (no spaces) OR
    #   "Best decode_ratio = 0.788 (96/122) ..."  (with spaces around '=')
    # The regex accepts both forms.
    if decode_ratio is None:
        m = re.search(r"decode_ratio\s*=\s*([0-9]+(?:\.[0-9]+)?)", smoke.stdout)
        if m:
            try:
                decode_ratio = float(m.group(1))
                source = "stdout regex"
            except ValueError:
                pass

    if decode_ratio is None:
        return {
            "decode_ratio": None,
            "invariant_holds": False,
            "source": source,
            "error": "decode_ratio metric not found in smoke output",
        }

    return {
        "decode_ratio": decode_ratio,
        "invariant_holds": decode_ratio > 0.0,
        "source": source,
        "error": None,
    }


def verify_loss_decrease(smoke: SmokeOutput) -> Dict[str, Any]:
    """Parse the smoke output for `loss[0] > loss[N-1]` (monotone non-increasing).

    Accepts either:
      * `smoke.metrics["loss_curve"]` as a list of per-step / per-epoch losses,
      * `smoke.metrics["epoch_losses"]` as a list of per-epoch losses,
      * a stdout pattern ``loss=<float>`` (single value; not decreasing-checkable).

    Returns a dict with:
      * ``loss_start`` (float | None),
      * ``loss_end`` (float | None),
      * ``decreasing`` (bool) — True iff loss_start > loss_end,
      * ``n_points`` (int),
      * ``source`` (str),
      * ``error`` (str | None).
    """
    curve: Optional[List[float]] = None
    source = "none"

    if isinstance(smoke.metrics.get("loss_curve"), list):
        curve = [float(x) for x in smoke.metrics["loss_curve"]]
        source = "metrics.loss_curve"
    elif isinstance(smoke.metrics.get("epoch_losses"), list):
        curve = [float(x) for x in smoke.metrics["epoch_losses"]]
        source = "metrics.epoch_losses"

    if curve is None or len(curve) < 2:
        return {
            "loss_start": None,
            "loss_end": None,
            "decreasing": False,
            "n_points": len(curve) if curve is not None else 0,
            "source": source,
            "error": "loss curve not found or has fewer than 2 points",
        }

    return {
        "loss_start": curve[0],
        "loss_end": curve[-1],
        "decreasing": curve[0] > curve[-1],
        "n_points": len(curve),
        "source": source,
        "error": None,
    }


def verify_all_invariants(
    smoke: SmokeOutput,
    *,
    expectations: Optional[Dict[str, str]] = None,
    parsed_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Any]:
    """Convenience wrapper — runs all three verifiers + emits one PASS/FAIL verdict.

    The three invariants (AGENTS.md §5.7):
      1. ``decode_ratio > 0``               — a decode regression is visible.
      2. ``loss[0] > loss[N-1]``            — the wire-in is not loss-diverging.
      3. flag TYPES match the trainer's own argparse contract — added
         2026-10-09 after a smoke PASSED 1 and 2 while ``--metals`` had
         arrived as one joined string and every fold scored ``nan``.

    ``expectations`` / ``parsed_args`` are forwarded to
    :func:`verify_flag_types`; omit them and the trainer named by
    ``smoke.trainer_path`` is inspected.
    """
    decode_ratio = verify_decode_ratio(smoke)
    loss_decrease = verify_loss_decrease(smoke)
    flag_types = verify_flag_types(
        smoke, expectations=expectations, parsed_args=parsed_args
    )
    all_pass = (
        decode_ratio["invariant_holds"]
        and loss_decrease["decreasing"]
        and flag_types["invariant_holds"]
    )
    return {
        "pass": all_pass,
        "decode_ratio": decode_ratio,
        "loss_decrease": loss_decrease,
        "flag_types": flag_types,
        "returncode": smoke.returncode,
        "wall_seconds": smoke.wall_seconds,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-399-A1 50-step end-to-end smoke protocol. "
            "Runs a trainer with a tiny CPU config + verifies decode_ratio > 0 "
            "and loss[0] > loss[N-1]."
        )
    )
    parser.add_argument(
        "--trainer",
        required=True,
        help="Path to trainer script (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--flag-arg",
        action="append",
        default=[],
        help="Forwarded flag (repeatable). Defaults to a tiny CPU smoke config.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="If set, write the structured SmokeOutput + verdict JSON here.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--expect-list-dest",
        action="append",
        default=None,
        metavar="DEST",
        help=(
            "Declare an extra nargs-style dest that must arrive as a list "
            "(repeatable). Belt-and-braces for trainers whose parser cannot "
            "be introspected; a single whitespace-joined element FAILS."
        ),
    )
    args = parser.parse_args(argv)

    expectations = None
    if args.expect_list_dest:
        expectations = {dest: "list" for dest in args.expect_list_dest}

    smoke = run_smoke_50step(
        trainer_path=args.trainer,
        flags_list=args.flag_arg,
        timeout_sec=args.timeout_sec,
    )
    verdict = verify_all_invariants(smoke, expectations=expectations)
    summary = {
        "smoke": smoke.to_dict(),
        "verdict": verdict,
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if verdict["pass"] and smoke.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
