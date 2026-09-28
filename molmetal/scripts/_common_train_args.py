#!/usr/bin/env python3
"""Wf-35 Phase 1 — Shared argparse + GPU env + MEASURED/PROJECTED tag.

=====================================================================
Centralised helpers for the **R18 / R19 trainer scripts**.  The eight
shipped trainers

  * ``train_se3sb.py``        (Wf-29 fix)
  * ``train_transport.py``    (Wf-29)
  * ``train_multiph.py``      (Wf-29)
  * ``train_ru_5metal_se3sb.py`` (Wf-31)
  * ``train_cisplatin_resistance_v2.py`` (Wf-31)
  * ``train_affinity_regression.py``   (Wf-32)
  * ``train_multimetal_lomo.py``       (Wf-32)
  * ``bench_pb_valid.py``              (Wf-32)

duplicate ~80 lines of argparse plumbing, GPU env setup, and JSON
tagging.  This module collects the duplication so Wf-35 Phase 2 can
swap it in without touching the eight trainers' core logic.

The module exposes four primitives:

1. :func:`add_common_args`     — recipe-level CLI flags
   (``--data``, ``--seeds``, ``--epochs``, ``--batch``, ``--lr``,
   ``--hidden``, ``--depth``, ``--init``, …).
2. :func:`add_gpu_args`        — GPU env flags
   (``--cuda-visible-devices``, ``--mixed-precision``,
   ``--num-workers``).
3. :func:`add_benchmark_args`  — benchmark / CV flags
   (``--n-folds``, ``--cv-strategy``, …).
4. :func:`setup_gpu_env`       — apply the GPU env side-effects
   (CUDA_VISIBLE_DEVICES, torch deterministic, torch seed).

Plus the JSON-tagging machinery:

* :func:`MEASURED_vs_PROJECTED_decorator`  — wraps ``main()`` so the
  metrics JSON at ``args.metrics_path`` is auto-tagged with
  ``{"_provenance": "MEASURED"}`` (training actually executed) or
  ``{"_provenance": "PROJECTED"}`` (dry-run / historical numbers).
* :func:`tag_metrics_payload`  — explicitly tag a payload dict.
* :data:`MEASURED_TAG`, :data:`PROJECTED_TAG` — string constants.

Design notes
------------
* **Argparse groups.**  Three helpers attach args to independent
  ``argparse.ArgumentParser.add_argument_group`` sections so the
  ``--help`` output stays organised: ``[Common recipe]`` /
  ``[GPU env]`` / ``[Benchmark / CV]``.
* **Idempotent re-use.**  Each helper is safe to call on the same
  parser from different scripts; the parser keeps the first
  registration and ignores duplicates.
* **Defaults from the W2.1 / W2.2 / W2.3 recipes.**  All defaults
  match the values in :data:`RECIPE_EPOCHS`, :data:`RECIPE_BATCH`,
  …  so swapping ``add_common_args`` into the existing trainers
  is a one-line edit per script.
* **No side effects on import.**  :func:`setup_gpu_env` is opt-in;
  importing this module never touches ``CUDA_VISIBLE_DEVICES`` or
  torch's RNG state.

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_common_train_args.py -v
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Public provenance-tag constants
# ---------------------------------------------------------------------------
MEASURED_TAG: str = "MEASURED"
"""Provenance tag for metric values produced by an actually-executed run."""

PROJECTED_TAG: str = "PROJECTED"
"""Provenance tag for metric values extrapolated / projected from a prior run."""

# ---------------------------------------------------------------------------
# Recipe defaults — must mirror the eight trainers' RECIPE_* constants
# ---------------------------------------------------------------------------
RECIPE_EPOCHS: int = 50
RECIPE_BATCH: int = 128
RECIPE_LR: float = 1e-3
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
RECIPE_HIDDEN: int = 300
RECIPE_DEPTH: int = 3
RECIPE_EVAL_EVERY: int = 5
RECIPE_DROPOUT: float = 0.2
RECIPE_TRAIN_MAX_PER_METAL: int = 4000

# Encoder-init choices — same set as ``train_ru_multitask.py`` /
# ``train_affinity_regression.py``.
INIT_CHOICES: Tuple[str, ...] = (
    "tmqm",      # R17 default (with arch-mismatch fallback to random)
    "random",    # R17 random init
    "se3sb",     # R18 W2.1 attach
    "transport", # R18 W2.2 attach
    "multiph",   # R18 W2.3 attach
)
INIT_DEFAULT: str = "tmqm"

# Mixed-precision choices — same vocabulary as PyTorch autocast.
MIXED_PRECISION_CHOICES: Tuple[str, ...] = ("bf16", "fp16", "none")
MIXED_PRECISION_DEFAULT: str = "bf16"

# Cross-validation strategy choices.
CV_STRATEGY_CHOICES: Tuple[str, ...] = ("scaffold", "random", "temporal", "lomo")
CV_STRATEGY_DEFAULT: str = "scaffold"

# ---------------------------------------------------------------------------
# Defaults for paths — overridable on every CLI
# ---------------------------------------------------------------------------
DEFAULT_DATA: Path = Path(".data/molmetal/metalcytotox_5metal_multitask.csv")
DEFAULT_ENCODER_CKPT: Path = Path("models/tmqm_pretrained/tmqm_pretrained_encoder.pt")
DEFAULT_OUT_DIR: Path = Path(".ckpt")
DEFAULT_METRICS_PATH: Path = Path("metrics/by_metric/r18_train.json")
DEFAULT_SUMMARY_PATH: Path = Path("metrics/by_metric/r18_train_summary.json")


# ---------------------------------------------------------------------------
# Low-level: ensure each arg appears at most once
# ---------------------------------------------------------------------------
def _existing_dest(root_parser: argparse.ArgumentParser, dest: str) -> bool:
    """Return True iff ``root_parser`` already has an action with ``dest``.

    The check is run against the *root* parser — argparse groups keep
    their own ``_actions`` list which starts empty for each call to
    :meth:`add_argument_group`, so we cannot trust the per-group view
    for cross-call idempotency.  The root parser's ``_actions``
    accumulates *all* actions from every group ever added (each
    independently), so it is the canonical idempotency key.
    """
    for action in root_parser._actions:
        if action.dest == dest:
            return True
    return False


def _add(
    root_parser: argparse.ArgumentParser,
    group: argparse.ArgumentParser,
    *args: Any,
    **kwargs: Any,
) -> None:
    """``add_argument`` wrapper that silently skips already-registered dests.

    ``root_parser`` is used for the idempotency check; ``group`` is the
    argparse group that should own the new argument when it is added.

    The dest is derived from the option strings (the same way
    :meth:`argparse.ArgumentParser.add_argument` does) so callers that
    pass only ``--foo`` (without an explicit ``dest=``) get the same
    ``foo`` dest that argparse would assign.
    """
    explicit_dest = kwargs.get("dest")
    if explicit_dest is None and args:
        # Mimic argparse's auto-dest: strip leading dashes from the
        # first option string (positional args have no leading dash).
        first = args[0]
        if isinstance(first, str) and first.startswith("-"):
            derived = first.lstrip("-").replace("-", "_")
            explicit_dest = derived
    if explicit_dest and _existing_dest(root_parser, explicit_dest):
        return
    group.add_argument(*args, **kwargs)


# ---------------------------------------------------------------------------
# add_common_args — recipe-level flags
# ---------------------------------------------------------------------------
def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    data_default: Path = DEFAULT_DATA,
    encoder_ckpt_default: Path = DEFAULT_ENCODER_CKPT,
    out_dir_default: Path = DEFAULT_OUT_DIR,
    metrics_path_default: Path = DEFAULT_METRICS_PATH,
    summary_path_default: Path = DEFAULT_SUMMARY_PATH,
    include_init: bool = True,
    include_train_max_per_metal: bool = True,
    include_eval_every: bool = True,
    include_dropout: bool = True,
) -> argparse.ArgumentParser:
    """Attach the standard R18 / R19 recipe-level CLI flags.

    Parameters
    ----------
    parser :
        The argparse parser to mutate.
    data_default :
        Default for ``--data``.  Defaults to the canonical 5-metal CSV
        path; ``train_affinity_regression`` etc. override per script.
    encoder_ckpt_default :
        Default for ``--encoder-ckpt``.
    out_dir_default :
        Default for ``--out-dir``.
    metrics_path_default :
        Default for ``--metrics-path``.
    summary_path_default :
        Default for ``--summary-path``.
    include_init :
        Set ``False`` for scripts that don't support the ``--init``
        choices (e.g. LOMO trainers where ``--init`` is fixed).
    include_train_max_per_metal :
        Set ``False`` for scripts that aren't metal-subset-aware.
    include_eval_every :
        Set ``False`` for non-eval-driven scripts.
    include_dropout :
        Set ``False`` for scripts that don't expose dropout.

    Returns
    -------
    argparse.ArgumentParser
        The input parser (mutated in place).  Returning the parser
        allows fluent chaining::

            p = argparse.ArgumentParser()
            p = add_common_args(p)
            p = add_gpu_args(p)
            p = add_benchmark_args(p)
    """
    group = parser.add_argument_group(
        "Common recipe",
        "R18/R19 training recipe — data, seeds, optimiser, encoder init.",
    )

    # --- I/O paths ---------------------------------------------------------
    _add(parser, group, "--data", type=Path, default=data_default,
         help="CSV / dataset path (default: %(default)s).")
    _add(parser, group, "--encoder-ckpt", type=Path, default=encoder_ckpt_default,
         help="Optional pretrained encoder checkpoint (tmQM default).")
    _add(parser, group, "--out-dir", type=Path, default=out_dir_default,
         help="Where to write per-seed ckpt files.")
    _add(parser, group, "--metrics-path", type=Path, default=metrics_path_default,
         help="Per-seed metrics JSON.")
    _add(parser, group, "--summary-path", type=Path, default=summary_path_default,
         help="Aggregate (mean±std) summary JSON.")

    # --- Training recipe ---------------------------------------------------
    _add(parser, group, "--seeds", type=int, nargs="+", default=list(RECIPE_SEEDS),
         help="List of RNG seeds (default: %(default)s).")
    _add(parser, group, "--epochs", type=int, default=RECIPE_EPOCHS,
         help="Number of training epochs per seed.")
    _add(parser, group, "--batch", type=int, default=RECIPE_BATCH,
         help="Mini-batch size.")
    _add(parser, group, "--lr", type=float, default=RECIPE_LR,
         help="AdamW learning rate.")
    _add(parser, group, "--hidden", type=int, default=RECIPE_HIDDEN,
         help="D-MPNN hidden width.")
    _add(parser, group, "--depth", type=int, default=RECIPE_DEPTH,
         help="D-MPNN message-passing depth.")

    if include_eval_every:
        _add(parser, group, "--eval-every", type=int, default=RECIPE_EVAL_EVERY,
             help="Evaluate on val every N epochs.")

    if include_dropout:
        _add(parser, group, "--dropout", type=float, default=RECIPE_DROPOUT,
             help="Dropout probability on the multi-task heads.")

    if include_train_max_per_metal:
        _add(parser, group, "--train-max-per-metal", type=int,
             default=RECIPE_TRAIN_MAX_PER_METAL,
             help="Subsample cap per metal per epoch (0 = full). Default 4000.")

    if include_init:
        _add(parser, group, "--init", choices=list(INIT_CHOICES), default=INIT_DEFAULT,
             help=(
                 "Encoder initialisation. R17: 'tmqm' (with arch-mismatch "
                 "fallback to random), 'random'. R18: 'se3sb' / 'transport' "
                 "/ 'multiph' attach R18 adapter components."
             ))

    return parser


# ---------------------------------------------------------------------------
# add_gpu_args — GPU env flags
# ---------------------------------------------------------------------------
def add_gpu_args(
    parser: argparse.ArgumentParser,
    *,
    default_mixed_precision: str = MIXED_PRECISION_DEFAULT,
    default_num_workers: int = 4,
    default_cuda_visible_devices: str = "1",
) -> argparse.ArgumentParser:
    """Attach GPU-environment CLI flags.

    Flags
    -----
    --cuda-visible-devices  : int or comma-separated list
        Sets ``CUDA_VISIBLE_DEVICES`` *before* importing torch.
        Default ``"1"`` (Wf-130: RTX 5090 on kanzi — GPU 0 is occupied
        by the VLM venv process).  Pass ``"0"`` to override.
    --mixed-precision       : {bf16, fp16, none}
        Autocast dtype.  Default ``bf16``.
    --num-workers           : int
        DataLoader worker count.  Default ``4``.
    """
    group = parser.add_argument_group(
        "GPU env",
        "Hardware selection + precision / dataloader knobs.",
    )

    _add(parser, group, "--cuda-visible-devices", type=str,
         default=default_cuda_visible_devices,
         help=(
             "Comma-separated GPU ids to expose.  Default '1' "
             "(Wf-130 — RTX 5090 on kanzi, GPU 0 reserved for VLM "
             "venv).  Empty string '' = inherit from the shell.  "
             "Applied via CUDA_VISIBLE_DEVICES before importing torch."
         ))
    _add(parser, group, "--mixed-precision", choices=list(MIXED_PRECISION_CHOICES),
         default=default_mixed_precision,
         help="Autocast dtype (default: %(default)s).")
    _add(parser, group, "--num-workers", type=int, default=default_num_workers,
         help="DataLoader worker count (default: %(default)s).")

    return parser


# ---------------------------------------------------------------------------
# add_benchmark_args — benchmark / CV flags
# ---------------------------------------------------------------------------
def add_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    default_n_folds: int = 5,
    default_cv_strategy: str = CV_STRATEGY_DEFAULT,
    include_folds_seed: bool = True,
) -> argparse.ArgumentParser:
    """Attach benchmark / cross-validation flags.

    Flags
    -----
    --n-folds       : int
        Number of cross-validation folds.  ``0`` disables CV.  Default ``5``.
    --cv-strategy   : {scaffold, random, temporal, lomo}
        Splitter strategy.  Default ``scaffold``.
    --folds-seed    : int
        RNG seed for the fold splitter.  Default ``0``.  Only added when
        ``include_folds_seed=True``.
    """
    group = parser.add_argument_group(
        "Benchmark / CV",
        "Cross-validation + benchmarking knobs.",
    )

    _add(parser, group, "--n-folds", type=int, default=default_n_folds,
         help="Number of CV folds (0 disables CV). Default: %(default)s.")
    _add(parser, group, "--cv-strategy", choices=list(CV_STRATEGY_CHOICES),
         default=default_cv_strategy,
         help="Splitter strategy (default: %(default)s).")

    if include_folds_seed:
        _add(parser, group, "--folds-seed", type=int, default=0,
             help="RNG seed for the fold splitter (default: %(default)s).")

    return parser


# ---------------------------------------------------------------------------
# setup_gpu_env — apply GPU env side-effects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuEnvState:
    """Snapshot of the GPU environment after :func:`setup_gpu_env` ran."""

    cuda_visible_devices: str
    mixed_precision: str
    num_workers: int
    seed: int
    deterministic: bool


def setup_gpu_env(
    args: argparse.Namespace,
    *,
    default_seed: int = 0,
    deterministic: bool = True,
) -> GpuEnvState:
    """Apply GPU env side-effects and seed torch.

    Reads from ``args``:

    * ``args.cuda_visible_devices`` — exported as ``CUDA_VISIBLE_DEVICES``.
      Wf-130 default is ``"1"`` (RTX 5090 on kanzi — GPU 0 is the
      VLM venv).  Pass ``""`` to inherit from the shell, ``"0"`` for
      the first GPU, or ``"0,1"`` for both.
    * ``args.mixed_precision``       — accepted (kept for downstream
      code that looks at the autocast dtype).
    * ``args.num_workers``           — accepted (no side effect here;
      DataLoader construction reads it directly).
    * ``args.seeds``                 — first seed becomes the torch seed
      if ``args.seed`` is missing.

    Parameters
    ----------
    args :
        Parsed argparse namespace from :func:`add_common_args` +
        :func:`add_gpu_args`.
    default_seed :
        Used if neither ``args.seed`` nor ``args.seeds[0]`` is set.
    deterministic :
        If True, call :func:`torch.use_deterministic_algorithms` (with
        the warning-only fallback for builds that lack it).  Sets
        ``CUBLAS_WORKSPACE_CONFIG`` to the documented deterministic
        value so cuBLAS doesn't reject the request.

    Returns
    -------
    GpuEnvState
        Snapshot of what was actually applied (useful for tagging the
        metrics JSON via :func:`tag_metrics_payload`).
    """
    cvd = getattr(args, "cuda_visible_devices", "") or ""
    if cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)
        # Wf-130: log so the caller can see what was applied.
        sys.stderr.write(
            f"[setup_gpu_env] CUDA_VISIBLE_DEVICES={cvd} (from "
            f"--cuda-visible-devices flag).\n"
        )

    mixed_precision = getattr(args, "mixed_precision", MIXED_PRECISION_DEFAULT)
    num_workers = int(getattr(args, "num_workers", 4))

    # Pick a torch seed: explicit ``args.seed`` > first seed in --seeds > default.
    seed: int = int(default_seed)
    explicit_seed = getattr(args, "seed", None)
    if explicit_seed is not None:
        seed = int(explicit_seed)
    else:
        seeds_attr = getattr(args, "seeds", None)
        if isinstance(seeds_attr, (list, tuple)) and seeds_attr:
            seed = int(seeds_attr[0])

    # Import torch lazily so the rest of this module is usable from
    # any environment, including ones without torch.
    import torch  # noqa: WPS433 — intentional lazy import for env isolation

    torch.manual_seed(seed)

    if deterministic:
        # cuBLAS needs this env var *before* the first CUDA call.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover — older torch
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    return GpuEnvState(
        cuda_visible_devices=str(cvd),
        mixed_precision=str(mixed_precision),
        num_workers=num_workers,
        seed=seed,
        deterministic=bool(deterministic),
    )


# ---------------------------------------------------------------------------
# tag_metrics_payload — explicit JSON tagging
# ---------------------------------------------------------------------------
def tag_metrics_payload(
    payload: Mapping[str, Any],
    *,
    provenance: str = MEASURED_TAG,
    gpu_env: Optional[GpuEnvState] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a *copy* of ``payload`` annotated with provenance metadata.

    The annotated payload gets three top-level keys:

    * ``_provenance`` — :data:`MEASURED_TAG` or :data:`PROJECTED_TAG`.
    * ``_gpu_env``     — :class:`GpuEnvState` as a plain dict (omitted
      if ``gpu_env`` is None).
    * ``_meta``        — merged ``extra_meta`` (defaults to {}).

    The input ``payload`` is **not mutated**.
    """
    out: Dict[str, Any] = dict(payload)
    out["_provenance"] = str(provenance)
    if gpu_env is not None:
        out["_gpu_env"] = {
            "cuda_visible_devices": gpu_env.cuda_visible_devices,
            "mixed_precision": gpu_env.mixed_precision,
            "num_workers": gpu_env.num_workers,
            "seed": gpu_env.seed,
            "deterministic": gpu_env.deterministic,
        }
    out["_meta"] = dict(extra_meta or {})
    return out


def write_tagged_metrics(
    metrics_path: Path,
    payload: Mapping[str, Any],
    *,
    provenance: str = MEASURED_TAG,
    gpu_env: Optional[GpuEnvState] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write ``payload`` to ``metrics_path`` with provenance tagging.

    The parent directory is created automatically.  Returns the
    tagged payload that was actually written.
    """
    tagged = tag_metrics_payload(
        payload,
        provenance=provenance,
        gpu_env=gpu_env,
        extra_meta=extra_meta,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_to_jsonable(tagged), indent=2))
    return tagged


# ---------------------------------------------------------------------------
# MEASURED_vs_PROJECTED_decorator — auto-tag the metrics JSON
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecoratorPolicy:
    """Policy knobs for :func:`MEASURED_vs_PROJECTED_decorator`.

    Attributes
    ----------
    measured_when_ckpt_exists :
        When True (default), the wrapper tags the JSON ``MEASURED`` iff
        at least one ``*.pt`` ckpt under ``--out-dir`` is non-empty
        after ``main()`` finishes.  This is the canonical signal that
        a training run actually executed.
    measured_when_seeds_nonempty :
        When True (default ``False`` — must opt-in), the wrapper also
        tags ``MEASURED`` whenever ``--seeds`` is non-empty.  Useful
        for scripts that don't write ckpts (e.g. ``bench_pb_valid.py``).
        Without this flag, ``--seeds`` alone is not enough evidence
        for ``MEASURED``.
    ckpt_glob :
        Glob pattern for ckpt files (default ``*.pt``).
    """

    measured_when_ckpt_exists: bool = True
    measured_when_seeds_nonempty: bool = False
    ckpt_glob: str = "*.pt"


def _list_ckpts(out_dir: Path, glob: str) -> List[Path]:
    if not out_dir.exists():
        return []
    return sorted(p for p in out_dir.glob(glob) if p.is_file() and p.stat().st_size > 0)


def _decide_provenance(
    args: argparse.Namespace,
    result: Any,
    policy: DecoratorPolicy,
) -> str:
    """Decide MEASURED vs PROJECTED for a finished run.

    Decision tree
    -------------
    1. If ``policy.measured_when_ckpt_exists`` AND ``--out-dir`` has a
       non-empty ckpt → ``MEASURED``.
    2. Else if ``policy.measured_when_seeds_nonempty`` AND
       ``--seeds`` is non-empty → ``MEASURED``.
    3. Otherwise → ``PROJECTED``.
    """
    out_dir = getattr(args, "out_dir", None)
    seeds = getattr(args, "seeds", None)

    if policy.measured_when_ckpt_exists and isinstance(out_dir, Path):
        if _list_ckpts(out_dir, policy.ckpt_glob):
            return MEASURED_TAG

    if (
        policy.measured_when_seeds_nonempty
        and isinstance(seeds, (list, tuple))
        and len(seeds) > 0
    ):
        return MEASURED_TAG

    return PROJECTED_TAG


def MEASURED_vs_PROJECTED_decorator(
    policy: Optional[DecoratorPolicy] = None,
    *,
    default_provenance: str = MEASURED_TAG,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap ``main()`` so the metrics JSON at ``args.metrics_path`` is
    auto-tagged with ``_provenance`` (and ``_gpu_env`` if applicable).

    Usage
    -----
    ::

        from molmetal.scripts._common_train_args import (
            add_common_args, add_gpu_args, setup_gpu_env,
            MEASURED_vs_PROJECTED_decorator,
        )

        @MEASURED_vs_PROJECTED_decorator()
        def main(argv=None):
            p = argparse.ArgumentParser()
            p = add_common_args(p)
            p = add_gpu_args(p)
            args = p.parse_args(argv)
            gpu_env = setup_gpu_env(args)
            # ... training happens, metrics written ...
            return 0

        if __name__ == "__main__":
            sys.exit(main())

    Behaviour
    ---------
    1. Calls the wrapped ``main(argv)`` to completion.
    2. After the wrapped function returns, reads ``args.metrics_path``
       (if it exists) and re-writes it with the provenance tag.
    3. Falls back to :data:`PROJECTED_TAG` if the JSON doesn't exist
       or no ckpts were produced (so a dry-run / historical sweep
       doesn't lie about being MEASURED).
    4. If the wrapped main() returns a tuple ``(exit_code, args_ns)``,
       the second element is used as the source of truth for tagging
       (no re-parse needed).  Otherwise the decorator re-parses argv.
    """
    policy = policy or DecoratorPolicy()

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)

            # Allow the wrapped main() to short-circuit by returning
            # ``(exit_code, args_namespace)``.
            explicit_args: Optional[argparse.Namespace] = None
            if isinstance(result, tuple) and len(result) == 2:
                exit_code, maybe_ns = result
                if isinstance(maybe_ns, argparse.Namespace):
                    explicit_args = maybe_ns
                    result = exit_code

            try:
                p_args = explicit_args
                if p_args is None:
                    argv = kwargs.get("argv")
                    if argv is None and args:
                        argv = args[0]
                    if argv is None:
                        argv = sys.argv[1:]
                    # Build a *minimal* parser containing only the
                    # arguments the decorator needs for tagging
                    # (``--metrics-path`` / ``--out-dir`` / ``--seeds`` /
                    # ``--init`` / ``--cuda-visible-devices`` /
                    # ``--mixed-precision`` / ``--num-workers`` / CV
                    # knobs).  Trainer-specific flags (e.g. Wf-32
                    # ``--dataset`` / ``--hela-csv``) are unknown to
                    # this parser, so we use ``parse_known_args`` to
                    # silently ignore them.  Previously this raised
                    # ``SystemExit(2)`` for any trainer that added
                    # args on top of ``add_common_args`` /
                    # ``add_gpu_args`` / ``add_benchmark_args``.
                    p = argparse.ArgumentParser(add_help=False)
                    p = add_common_args(p)
                    p = add_gpu_args(p)
                    p = add_benchmark_args(p)
                    p_args, _unknown = p.parse_known_args(argv)
                _tag_metrics_json(p_args, policy, result, extra_meta)
            except SystemExit:
                # argparse --help / parse error → leave JSON alone.
                raise
            except Exception as exc:  # pragma: no cover — defensive
                sys.stderr.write(
                    f"[MEASURED_vs_PROJECTED] tagging failed: {exc!r}\n"
                )
            return result

        return wrapper

    return decorator


def _tag_metrics_json(
    args: argparse.Namespace,
    policy: DecoratorPolicy,
    result: Any,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> None:
    """Read ``args.metrics_path`` and re-write with provenance tagging."""
    metrics_path = getattr(args, "metrics_path", None)
    if not isinstance(metrics_path, Path) or not metrics_path.exists():
        return
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    provenance = _decide_provenance(args, result, policy)
    tagged = tag_metrics_payload(
        payload,
        provenance=provenance,
        gpu_env=None,
        extra_meta=extra_meta,
    )
    metrics_path.write_text(json.dumps(_to_jsonable(tagged), indent=2))


# ---------------------------------------------------------------------------
# Internal JSON helper (mirrors what the eight trainers use)
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """Recursive converter for JSON-safe scalar types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    try:
        import numpy as np  # noqa: WPS433
        if isinstance(obj, np.floating):
            return float(obj.item())
        if isinstance(obj, np.integer):
            return int(obj.item())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, float):
        try:
            import math
            if math.isnan(obj):
                return None
        except Exception:
            pass
    return obj


__all__ = [
    "MEASURED_TAG",
    "PROJECTED_TAG",
    "RECIPE_EPOCHS",
    "RECIPE_BATCH",
    "RECIPE_LR",
    "RECIPE_SEEDS",
    "RECIPE_HIDDEN",
    "RECIPE_DEPTH",
    "RECIPE_EVAL_EVERY",
    "RECIPE_DROPOUT",
    "RECIPE_TRAIN_MAX_PER_METAL",
    "INIT_CHOICES",
    "INIT_DEFAULT",
    "MIXED_PRECISION_CHOICES",
    "MIXED_PRECISION_DEFAULT",
    "CV_STRATEGY_CHOICES",
    "CV_STRATEGY_DEFAULT",
    "DEFAULT_DATA",
    "DEFAULT_ENCODER_CKPT",
    "DEFAULT_OUT_DIR",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_SUMMARY_PATH",
    "GpuEnvState",
    "DecoratorPolicy",
    "add_common_args",
    "add_gpu_args",
    "add_benchmark_args",
    "setup_gpu_env",
    "tag_metrics_payload",
    "write_tagged_metrics",
    "MEASURED_vs_PROJECTED_decorator",
]
