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


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load + normalize + flatten a YAML config file.

    Returns a flat dict mapping canonical argparse-dest names to override values.

    Raises:
      yaml.YAMLError: YAML parse error (propagates with line context).
    FileNotFoundError -> returns {} (callers treat as no-op).
    """
    p = Path(path)
    if not p.is_file():
        # Treat as no-op; callers don't need to gate on existence.
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
    return _flatten(raw)


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
    cfg = load_config(config_path)
    if not cfg:
        return args

    # Re-parse with [] to get a pure-default namespace.
    defaults = parser.parse_args([])

    for raw_key, value in cfg.items():
        dest = _normalize_key(raw_key)
        if not hasattr(args, dest):
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
            # User did NOT pass CLI -> YAML applies
            setattr(args, dest, value)
    return args
