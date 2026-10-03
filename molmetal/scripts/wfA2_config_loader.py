"""Wf-A2 unified YAML config loader.

Per AGENTS.md §1.2 + party-twenty-spirit §3.5: MEASURED-vs-PROJECTED tag discipline;
every numeric value traces to a JSON / report path on disk.

Contract:
- load_config(path) -> dict: load + flatten nested YAML.
- apply_yaml_overrides(p, args, config_path) -> Namespace:
  Pre-parse + post-parse pattern that augments argparse defaults from YAML
  while preserving CLI-explicit values when they differ from defaults.

Schema convention (per configs/default.yaml + configs/wf124_p3_50epoch_full_sweep_g5.yaml):
  model.{hidden_dim,n_layers,backbone,ablation_config,task,...}
  train.{epochs,batch_size,lr,grad_clip_norm,bf16,...}
  data.{csv,n_rows_expected,scaffold_split,cohort_filter.metals,...}
  protocol.{seeds,enumerated_runs,...}
  waves.{total_waves,wave_a/wave_b/...}
  eval.{per_pair_metrics,target_test_auc,...}
  output.{base_dir,per_metal_dir,...}
  compute.{gpu,cpu_workers,bf16,...}
  directives.{kill_then_adapt,user_ok_required,...}

Single source of truth: trainer argparse defaults remain canonical; YAML is
a transparent override layer.

HONEST FRAMING:
- MEASURED-on-real-job test: Wf-A2 only added wire-in (no GPU).
- Bit-for-bit backward compat when --config not supplied (argparse-only path).
- The M17 canonical config is SCAFFOLD-only — MEASURED-on-real-job launch
  deferred to Wf-A2 follow-up per AGENTS.md §5.1 standing GPU grant.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a hard dep of this repo
    yaml = None


# Sentinel used to distinguish "default" vs "user-set" via comparison.
class _MissingSentinel:
    """Sentinel distinct from any legitimate argparse value."""

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def _normalize_key(key: str) -> str:
    """YAML key -> argparse dest name.

    Convention: yaml-key '--lr' or 'lr' or 'LR' -> dest 'lr';
    yaml-key 'wasserstein_weight' or 'wasserstein-weight' -> dest 'wasserstein_weight'.
    Nested dict keys like 'model.hidden_dim' (post-flatten) -> 'model_hidden_dim'.
    """
    k = str(key).strip().lstrip("-")
    k = k.replace("-", "_").replace(".", "_")
    return k.lower()


#: Top-level YAML sections whose *leaf* keys may map onto a same-named
#: trainer argparse dest.
#:
#: Wf-A9 FIX (2026-10-03) — sections describing the actual run.  Sections
#: that describe *variants* (``waves.*``, ``ablation_configs.*``) or pure
#: metadata (``honest_framing``, ``cross_references``, ``entry``,
#: ``directives``) are deliberately excluded so a per-wave value can never
#: silently win over the protocol-level one.
_LEAF_FALLBACK_SECTIONS = frozenset({
    "protocol", "model", "flow", "data", "compute", "eval", "output",
})


def _candidate_dests(path: list[str]) -> list[str]:
    """Candidate argparse dests for one YAML leaf, most specific first.

    Wf-A9 FIX (2026-10-03) — the pre-fix loader only tried the *fully
    flattened* key, so a YAML written with the repo-wide section schema
    (``protocol.epochs``, ``protocol.batch_size``, ``model.hidden_dim`` …)
    flattened to ``protocol_epochs`` and found no matching argparse dest
    (the trainer's dest is plain ``epochs``).  ``apply_yaml_overrides``
    then hit its ``hasattr`` guard and skipped the key *silently*, so the
    YAML was decorative: every hyperparameter came from the argparse
    default.  Measured before the fix, with
    ``configs/cfm_per_innovation_ablation_p1.4.yaml`` (yaml says
    epochs=3, batch_size=16, lr=1e-4): ``args`` came back as the argparse
    defaults 50 / 64 / 3e-4.

    We now also try the leaf of a **two-component** path (``protocol.epochs``
    → ``epochs``), restricted to the run-describing sections whitelisted in
    :data:`_LEAF_FALLBACK_SECTIONS`.  The two-component rule is load-bearing:
    it is what keeps ``protocol.scheduler.warmup_epochs`` (3 components)
    from collapsing to ``epochs`` and clobbering the real epoch count.
    The fully flattened form is still tried first and still wins, so configs
    that spell out ``model_hidden_dim`` at top level keep working unchanged.
    """
    flat = _normalize_key(".".join(path))
    cands: list[str] = [flat]
    if len(path) == 2 and path[0].strip().lower() in _LEAF_FALLBACK_SECTIONS:
        leaf = _normalize_key(path[1])
        if leaf and leaf not in cands:
            cands.append(leaf)
    return cands


def _iter_leaves(
    mapping: Mapping[str, Any], prefix: Optional[list[str]] = None
):
    """Yield ``(path, value)`` for every leaf of a nested YAML mapping.

    ``path`` is the list of keys from the root, e.g.
    ``{'protocol': {'epochs': 3}}`` → ``(['protocol', 'epochs'], 3)``.
    Non-empty nested mappings recurse; everything else (scalars, lists,
    empty mappings) is a leaf.
    """
    prefix = list(prefix or [])
    for k, v in mapping.items():
        path = prefix + [str(k)]
        if isinstance(v, Mapping) and v:
            yield from _iter_leaves(v, path)
        else:
            yield path, v


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into single-level underscore-joined keys.

    Example: {'model': {'hidden_dim': 128}} -> {'model_hidden_dim': 128}.
    Lists are kept as-is (no further flattening).
    """
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        path = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:  # non-empty nested dict
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


def _load_nested(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Read a YAML config file and return its raw (still nested) mapping.

    Same error contract as :func:`load_config`: a missing file is a no-op
    (``{}``), a parse error propagates with line context, and a non-mapping
    root raises :class:`yaml.YAMLError`.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML required for --config YAML override")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise yaml.YAMLError(
            f"config root must be a mapping; got {type(raw).__name__} in {p}"
        )
    return raw


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load + normalize + flatten a YAML config file.

    Returns a flat dict mapping canonical argparse-dest names to override values.

    Raises:
      yaml.YAMLError: YAML parse error (propagates with line context).
    FileNotFoundError -> returns {} (callers treat as no-op).
    """
    raw = _load_nested(path)
    if not raw:
        return {}
    return _flatten(raw)


def _dest_converters(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Map argparse dest -> its ``type=`` converter, where it has one.

    Wf-A9 FIX (2026-10-03) — YAML values land in ``args`` without passing
    through argparse, so they skipped the ``type=`` coercion too.  A YAML
    ``output.ckpt_dir: molmetal/checkpoints`` therefore arrived as ``str``
    where the trainer expects a ``Path`` and the next ``args.ckpt_dir.mkdir()``
    died with ``AttributeError: 'str' object has no attribute 'mkdir'``.
    Reusing the action's own converter keeps YAML and CLI values identical
    in type, including Path-valued flags (``--ckpt-dir``, ``--out-dir``,
    ``--csv``, …) and numeric ones (``--epochs``, ``--lr``, …).
    """
    out: dict[str, Any] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        conv = getattr(action, "type", None)
        if dest and conv is not None and dest not in out:
            out[dest] = conv
    return out


def apply_yaml_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_path: Optional[str],
) -> argparse.Namespace:
    """If config_path is set, override args with YAML values where the user
    did NOT explicitly pass the corresponding CLI flag (i.e., value matches
    the default).

    Algorithm:
      1. Re-parse with [] to capture pure-default namespace.
      2. For each YAML (key, value):
         - normalize key -> argparse dest
         - if args.dest exists:
           cur = getattr(args, dest)
           dft = getattr(defaults, dest, _MISSING)
           if cur == dft (user did not pass): setattr(args, dest, value)
           else: keep CLI value (explicit override)

    Args:
      parser: the full argparse.ArgumentParser that produced args.
      args: post-parse namespace.
      config_path: path to YAML config (None = no-op).

    Returns: same args object (mutated).
    """
    if not config_path:
        return args
    # Wf-A9 FIX (2026-10-03) — walk the *nested* structure rather than the
    # flattened dict, so a leaf can be told apart from a deeper path
    # (``protocol.epochs`` vs ``protocol.scheduler.warmup_epochs``);
    # underscore-flattening alone cannot distinguish the two.
    nested = _load_nested(config_path)
    if not nested:
        return args

    # Re-parse with [] to get a pure-default namespace.
    defaults = parser.parse_args([])
    converters = _dest_converters(parser)

    for path, value in _iter_leaves(nested):
        # Resolve the YAML leaf to an argparse dest via the candidate
        # chain (flattened form first, then the two-component leaf form
        # for run-describing sections).  See _candidate_dests.
        dest = None
        for cand in _candidate_dests(path):
            if hasattr(args, cand):
                dest = cand
                break
        if dest is None:
            # YAML key has no argparse counterpart. Forward-compat: skip silently.
            continue
        cur = getattr(args, dest)
        dft = getattr(defaults, dest, _MISSING)
        if dft is _MISSING:
            continue
        # Special case: for LIST-valued args, YAML always wins when provided.
        # Rationale: list defaults (e.g. metals cohort = ['Ru','Ir','Rh','Os','Re'])
        # are rarely the user's actual intent; YAML cohorts override by default.
        if isinstance(value, list) and isinstance(cur, list):
            setattr(args, dest, value)
            continue
        if cur == dft:
            # Wf-A9 FIX (2026-10-03) — run the value through the action's
            # own ``type=`` converter so YAML and CLI values have the same
            # type (notably Path-valued flags, which would otherwise blow
            # up downstream on the first attribute access).
            conv = converters.get(dest)
            if conv is not None and not isinstance(value, list):
                try:
                    value = conv(value)
                except (TypeError, ValueError):
                    # Not convertible to the argparse type — keep the
                    # default rather than crashing the launch.
                    continue
            # User did NOT pass CLI -> YAML applies
            setattr(args, dest, value)
    return args
