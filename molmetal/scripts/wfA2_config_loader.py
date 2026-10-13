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
import sys
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
_LEAF_FALLBACK_SECTIONS = frozenset(
    {
        "protocol",
        "model",
        "flow",
        "data",
        "compute",
        "eval",
        "output",
        # Wf-A11 FIX (2026-10-03) — "train" added: configs/default.yaml carries
        # the real run hyperparameters (batch_size / lr / epochs / warmup_steps /
        # grad_clip_norm) under `train:` and the whole block was silently dropped
        # before this fix (see TODO/completed/config_dead_key_remediation.md).
        "train",
    }
)


# ---------------------------------------------------------------------------
# Wf-A11 FIX (2026-10-03) — dead-key visibility + alias resolution.
#
# PROBLEM (measured by the Wf-A11 audit, 2026-10-03): keys with no argparse
# dest were dropped by the loop below with a bare `continue`.  Configs
# DECLARED one value (e.g. `model.hidden_dim: 300`) while the trainer RAN the
# argparse default (128) — a silent lie, with no warning and no way for a
# gate to notice.
#
# FIX: (1) every skipped key is now collected, attached to the returned
# namespace as `args.yaml_unresolved_keys`, exposed via
# `last_skipped_keys()`, and printed to stderr (suppress with
# WFA2_CONFIG_QUIET=1); (2) `WFA2_CONFIG_STRICT=1` raises
# UnresolvedConfigKey instead of skipping — keys inside _NON_TRAINER_SECTIONS
# or named in _NON_TRAINER_KEYS are exempt because a NON-trainer consumer is
# documented for them; (3) unambiguous schema-name aliases are resolved via
# _LEAF_ALIASES.
# ---------------------------------------------------------------------------


class UnresolvedConfigKey(RuntimeError):
    """Strict-mode failure: YAML key(s) resolved to no argparse dest.

    Raised by :func:`apply_yaml_overrides` when ``WFA2_CONFIG_STRICT`` is a
    truthy value and at least one key (outside the documented non-trainer
    allowlists) had no argparse counterpart.
    """


#: Leaf-name aliases: YAML leaf -> argparse dests tried *after* the exact and
#: leaf forms, for leaves under :data:`_LEAF_FALLBACK_SECTIONS` (any depth).
#: Each alias is a same-meaning schema name for a dest that exists in at
#: least one of the two trainers.  An alias NEVER overrides a YAML key that
#: names the dest directly (two-pass logic in apply_yaml_overrides), and an
#: alias whose target dest does not exist on the active parser stays
#: unresolved (reported, not force-fitted).
_LEAF_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_dim": ("hidden",),  # D-MPNN: --hidden (CFM declares --hidden-dim itself)
    "grad_clip_norm": (
        "max_norm",
    ),  # both trainers: --max-norm (no --grad-clip-norm exists)
    "warmup_steps": (
        "lr_warmup_steps",
    ),  # CFM: --lr-warmup-steps (D-MPNN has epochs-based warmup only)
    "warmup_epochs": ("lr_warmup_epochs",),  # both trainers: --lr-warmup-epochs
    "min_lr": ("lr_min_lr",),  # both trainers: --lr-min-lr
    "cpu_workers": (
        "cpu_num_workers",
    ),  # D-MPNN: --cpu-num-workers (CFM has no workers dest)
}

#: Top-level sections that describe run metadata / enumeration, not trainer
#: hyperparameters.  A skipped key inside one of these is by design, so
#: strict mode does not fail on it.
_NON_TRAINER_SECTIONS = frozenset(
    {
        "entry",
        "directives",
        "honest_framing",
        "cross_references",
        "waves",
        "ablation_configs",
        "sweep_pairs",
        "runs",
        "pending_pairs",
    }
)

#: Leaf keys (any section) whose consumer is NOT the trainer, with the
#: verified consumer named.  Skips of these are by design.
_NON_TRAINER_KEYS: dict[str, str] = {
    "n_seeds": "scripts/start.sh N_SEEDS env (reads protocol.n_seeds)",
    "gpu": "scripts/start.sh GPU env (reads compute.gpu)",
    "base_dir": "scripts/start.sh OUT_DIR env (reads output.base_dir)",
    "log_master": "scripts/start.sh LOG_MASTER env (reads output.log_master)",
    "training_summary": "scripts/start.sh TRAINING_SUMMARY env (reads output.training_summary)",
    "events_jsonl_path": "scripts/start.sh EVENTS_JSONL_PATH env (reads output.events_jsonl_path)",
    "dry_run": "run-control metadata; no consumer found (Wf-A11 audit); launch mode comes from launcher flags",
    "user_ok_required": "run-control metadata; start.sh reads directives.user_ok_required, not the top-level key",
    "user_ok_granted": "run-control metadata",
    "per_wave_user_ok_required": "run-control metadata",
}

#: Dotted-path form of the same idea, for keys whose leaf name is too generic
#: to key on safely (`protocol.scheduler.type` has leaf `type`, and excusing
#: every `*.type` would hide real dead keys).  Every entry was verified by
#: reading the named consumer — same bar as `_NON_TRAINER_KEYS`.
_NON_TRAINER_PATHS: dict[str, str] = {
    "protocol.n_seeds_min": "scripts/start.sh N_SEEDS_MIN env (reads protocol.n_seeds_min)",
    "protocol.scheduler.type": (
        "scripts/start.sh SCHEDULER env (reads protocol.scheduler.type, "
        "passes it back as --lr-scheduler)"
    ),
    "protocol.scheduler.pct_start": (
        "scripts/start.sh LR_PCT_START env (reads "
        "protocol.scheduler.pct_start, passes it back as --lr-pct-start; "
        "omitted entirely when the yaml does not declare it)"
    ),
    "output.events_jsonl": (
        "scripts/start.sh events_jsonl_disabled (reads output.events_jsonl)"
    ),
}

# 2026-10-09 (config↔trainer binding audit).  These resolve to no trainer dest
# but DO have a verified non-trainer consumer, found by grepping for the literal
# key across molmetal/scripts/*.py + scripts/*.sh and reading each hit:
#   eval.target_test_auc      -> pre_flight_check.py
#   protocol.enumerated_runs  -> scripts/start.sh
#   protocol.n_runs_total     -> wf229_ablation_launcher / wf240_ablation_launcher
#                                / wf311b_paper4_hard_metric_aggregate
#   protocol.total_runs       -> gpu_launch_aggregator.py + launch_wf124_p3_g5.sh
#                                + gpu_launch_dashboard.sh (+ 3 more launchers)
#   scaffold_split            -> train_nci60_10cell / train_platinai_4cell
#                                / pre_flight_check / check_data_leakage
#                                / wf343c3_data_quality_check
# Before this audit they were indistinguishable from genuinely dead keys, which
# is what made a "0 dead keys" claim uncheckable for the wf124 sweep configs.
_NON_TRAINER_PATHS.update(
    {
        "eval.target_test_auc": "molmetal/scripts/pre_flight_check.py",
        "protocol.enumerated_runs": "scripts/start.sh",
        "protocol.n_runs_total": "wf229/wf240_ablation_launcher, wf311b_aggregate",
        "protocol.total_runs": "gpu_launch_aggregator.py, gpu_launch_dashboard.sh",
        "data.split.scaffold_split": "train_nci60_10cell.py, train_platinai_4cell.py",
        "model.scaffold_split": "train_nci60_10cell.py, train_platinai_4cell.py",
        "protocol.scaffold_split": "train_nci60_10cell.py, train_platinai_4cell.py",
    }
)

#: Keys that resolve to no dest and have NO consumer anywhere in the repo —
#: grepped 2026-10-09 for the literal key across every *.py and *.sh under
#: molmetal/ and scripts/ (0 hits each).  They document sweep/ablation INTENT in
#: the wf124_p3_* configs; nothing reads them.
#:
#: These are NOT excused from the skip report — a key being inert is a fact a
#: reader needs, and silencing it would recreate the blind spot this audit was
#: run to close (AGENTS.md memory: the `entry:` section was a dead-key blind
#: spot precisely because its skips were invisible).  They are registered here
#: only so `scripts/_config_binding_gate.py` can label them DECLARED-INERT and
#: print them every run instead of treating them as unknown.
#:
#: TODO: either give each a consumer (the G5/G6 aggregators read the sibling
#: `protocol.total_runs`, so the rest of this family was probably intended for
#: them) or delete it from the yamls. Left in place, declared, until someone
#: decides which — deleting a config key that a human reads costs more than
#: keeping an inert one that is loudly reported.
DECLARED_INERT_PATHS: frozenset[str] = frozenset(
    {
        # --- split-protocol documentation (configs/training/*) ---------------
        # Describe HOW the cohort was split, for reproducibility of the number.
        # The trainer takes the resulting split from the data loader, not from
        # these keys, so they are prose that happens to be machine-readable.
        "data.split",
        "data.split.type",
        "data.split.metals",
        "data.split.implementation",
        "data.split.strategy",
        "data.split.seed",
        "data.split.grouped_on",
        "data.split.train_frac",
        "data.split.val_frac",
        "data.split.test_frac",
        "data.split.split_column",
        "data.split_intent",
        "data.n_rows_expected",
        "data.expected_columns_subset",
        "data.cohort_filter",
        # --- sweep / ablation structure (wf124_p3_*) ------------------------
        "protocol.wave_decomposition",
        "protocol.enumerated_runs",
        "eval.per_pair_metrics",
        "eval.per_run_metrics",
        "eval.per_cell_line_metrics",
        "eval.comparison_metrics",
        "eval.attribution_matrix",
        "output.per_metal_dir",
        "output.per_config_dir",
        "output.per_cell_line_dir",
        "output.wasserstein_decay_dir",
        "output.comparison_30vs50",
        "output.ablation_report",
        "output.comparison_matrix",
        # --- MEASURED baselines + gates, recorded as the comparison target ---
        # These are the numbers a retrain is judged against, written into the
        # config so the comparison cannot drift the way prose does
        # (AGENTS.md §5.2: every numeric value must trace to a path on disk).
        # They are read by humans and by the report generators, not by the
        # trainer — removing them would destroy the provenance chain, so they
        # are declared inert, never deleted.
        "eval.baseline_per_cell_line_MEASURED",
        "eval.baseline_overall_MEASURED",
        "eval.baseline_per_cell_line",
        "eval.krasnov_2026_baseline_auc",
        "eval.gate_target",
        "eval.target_per_cell_line",
        "eval.target_overall",
        # --- sweep GRID, read by the launcher rather than by any trainer ----
        # A list of values to iterate is a property of the RUNNER, not of a
        # training run: no trainer has a --strengths dest because a trainer is
        # invoked once. The launcher (scripts/launch_cfm_l3_guidance_sweep.sh)
        # reads these to emit one invocation per arm, which is why the grid
        # cannot live on the command line and must be in the YAML — the whole
        # point is that it cannot drift from the config documenting it.
        #
        # Declared inert, not deleted: removing them would move the grid back
        # into a shell script, which is exactly how use_fm_objective went
        # missing from five of seven CFM configs (AGENTS.md 5.11).
        "sweep.strengths",
        "sweep.arm_prefix",
        # 2026-10-12 (BLK-C-M4) — the aromatic logit-offset grid, same
        # rationale as sweep.strengths one line up: a list of values to
        # iterate belongs to the RUNNER, not to a trainer, which is invoked
        # once. Read by scripts/launch_cfm_aromatic_bias_sweep.sh to emit one
        # invocation per arm, which is why the grid cannot live on the command
        # line — it would drift from the config that documents it.
        "sweep.biases",
    }
)

#: Report of the most recent apply_yaml_overrides() call (module-global so
#: tests / gates can inspect it without re-parsing).
_LAST_SKIPPED: list[dict[str, Any]] = []


def last_skipped_keys() -> list[dict[str, Any]]:
    """Keys that resolved to no argparse dest in the last apply call.

    Each entry: {"key": dotted-yaml-path, "value": raw value, "section":
    top-level section, "leaf": normalized leaf name, optional "reason"}.
    """
    return list(_LAST_SKIPPED)


def _strict_enabled() -> bool:
    return os.environ.get("WFA2_CONFIG_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _quiet_enabled() -> bool:
    return os.environ.get("WFA2_CONFIG_QUIET", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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


def _resolve_dest(
    args: argparse.Namespace, path: list[str]
) -> tuple[Optional[str], bool]:
    """Resolve one YAML leaf path to ``(dest, is_alias)``.

    Tries the exact/leaf candidate chain first (``is_alias=False``); then,
    for leaves under a run-describing section, the same-meaning schema
    aliases from :data:`_LEAF_ALIASES` (``is_alias=True``).  Returns
    ``(None, False)`` when nothing matches on this parser.
    """
    for cand in _candidate_dests(path):
        if hasattr(args, cand):
            return cand, False
    if len(path) >= 2 and path[0].strip().lower() in _LEAF_FALLBACK_SECTIONS:
        leaf = _normalize_key(path[-1])
        for alt in _LEAF_ALIASES.get(leaf, ()):
            if hasattr(args, alt):
                return alt, True
    return None, False


def _skip_entry(
    path: list[str], value: Any, reason: Optional[str] = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": ".".join(path),
        "value": value,
        "section": path[0].strip().lower() if path else "",
        "leaf": _normalize_key(path[-1]) if path else "",
    }
    if reason:
        entry["reason"] = reason
    return entry


def _non_trainer_consumer(entry: Mapping[str, Any]) -> Optional[str]:
    """Name the non-trainer consumer of a skipped key, or ``None``.

    Checked in order: the section is metadata by design, the full dotted path
    is registered, or the (normalised) leaf is registered.
    """
    if entry.get("section") in _NON_TRAINER_SECTIONS:
        return "metadata section"
    path = str(entry.get("key", ""))
    if path in _NON_TRAINER_PATHS:
        return _NON_TRAINER_PATHS[path]
    return _NON_TRAINER_KEYS.get(str(entry.get("leaf", "")))


def _registered_non_trainer(entry: Mapping[str, Any]) -> bool:
    """True when the skip is by design (metadata section or named consumer)."""
    return _non_trainer_consumer(entry) is not None


def _iter_leaves(mapping: Mapping[str, Any], prefix: Optional[list[str]] = None):
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


def _values_equal(a: Any, b: Any) -> bool:
    """``==`` that never raises (numpy-ish values can return an array)."""
    try:
        return bool(a == b)
    except Exception:  # pragma: no cover - exotic value types
        return a is b


def _argv_probe_disabled() -> bool:
    return os.environ.get("WFA2_CONFIG_ARGV_PROBE", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _supplied_dests(parser: argparse.ArgumentParser, argv: list[str]) -> Optional[set[str]]:
    """Dests the caller ACTUALLY typed on ``argv``.

    Implemented the documented-argparse way: temporarily swap every action's
    ``default`` for ``argparse.SUPPRESS``, parse, and read off which attributes
    appeared.  ``SUPPRESS`` is never set by argparse itself, so a dest that is
    present in the namespace can only have come from a real occurrence on the
    command line.  Returns ``None`` when the parse could not be done.
    """
    saved: list[tuple[Any, Any]] = [
        (action, action.default) for action in parser._actions if hasattr(action, "default")
    ]
    try:
        for action, _default in saved:
            action.default = argparse.SUPPRESS
        namespace, _unknown = parser.parse_known_args(list(argv))
    except BaseException:
        # SystemExit (required args / --help), argparse errors, anything odd:
        # the caller gets no explicitness signal and we fall back.
        return None
    finally:
        for action, default in saved:
            action.default = default
    return {
        dest
        for dest, value in vars(namespace).items()
        if value is not argparse.SUPPRESS
    }


def _explicit_cli_dests(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Optional[list[str]],
) -> Optional[set[str]]:
    """Dests explicitly supplied on the command line, or ``None`` if unknown.

    Why not just compare against the default?  2026-10-09 — ``cur == dft``
    conflates "user passed nothing" with "user passed exactly the default",
    which is precisely the case where YAML must NOT win (a launcher that
    spells out ``--epochs 50`` because that is what it wants, against a config
    whose ``train.epochs`` is 3).  ``--epochs 50`` is silently rewritten to 3.

    The real signal is whether argparse saw the token.  It can only be read
    back from the argv the caller parsed, which this function does not
    receive, so the argv is re-parsed here — but ONLY after two guards prove
    that argv is in fact the argv that produced ``args``:

      1. the probe parse leaves no unrecognised tokens (an argv carrying the
         ambient test-runner flags fails this immediately), and
      2. the probe namespace equals ``args`` on every parser dest.

    If either guard fails — or the caller opts out with
    ``WFA2_CONFIG_ARGV_PROBE=0`` — the answer is ``None`` and
    :func:`apply_yaml_overrides` falls back to the historical
    "value equals default" proxy, so callers that pass nothing keep their
    current behaviour byte for byte.
    """
    if argv is None:
        if _argv_probe_disabled():
            return None
        argv = list(sys.argv[1:])
    try:
        probe, unknown = parser.parse_known_args(list(argv))
    except BaseException:
        return None
    if unknown:
        # Not this parser's argv (e.g. the ambient pytest command line).
        return None
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == argparse.SUPPRESS:
            continue
        if not _values_equal(getattr(probe, dest, _MISSING), getattr(args, dest, _MISSING)):
            return None
    return _supplied_dests(parser, argv)


def apply_yaml_overrides(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config_path: Optional[str],
    *,
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """If config_path is set, override args with YAML values where the user
    did NOT explicitly pass the corresponding CLI flag.

    Algorithm:
      1. Re-parse with [] to capture pure-default namespace.
      2. Determine which dests were EXPLICITLY supplied on the command line
         (``_explicit_cli_dests``); when that cannot be established, fall back
         to the legacy proxy "value still equals the argparse default".
      3. For each YAML (key, value):
         - normalize key -> argparse dest
         - if args.dest exists and the dest was not explicitly supplied:
           setattr(args, dest, value)
           else: keep the CLI value (explicit override)

    Args:
      parser: the full argparse.ArgumentParser that produced args.
      args: post-parse namespace.
      config_path: path to YAML config (None = no-op).
      argv: the argv ``args`` was parsed from.  Omit and the loader re-probes
        ``sys.argv[1:]`` (guarded); pass it explicitly to remove all doubt.

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

    # 2026-10-09 — which dests did the caller actually TYPE?  ``None`` means
    # "cannot tell", in which case each dest falls back to the historical
    # "still equals the argparse default" proxy.
    explicit = _explicit_cli_dests(parser, args, argv)

    def _user_supplied(dest: str, cur: Any, dft: Any) -> bool:
        if explicit is not None:
            return dest in explicit
        return not _values_equal(cur, dft)

    # Wf-A11 PASS 1 — resolve every leaf.  ``direct_dests`` records dests a
    # non-alias YAML key writes; aliases (_LEAF_ALIASES) yield to those
    # regardless of file order, so ``protocol.grad_clip_norm`` can never
    # silently beat an explicit ``protocol.max_norm`` in the same file.
    leaves: list[tuple[list[str], Any, Optional[str], bool]] = []
    direct_dests: set[str] = set()
    for path, value in _iter_leaves(nested):
        dest, is_alias = _resolve_dest(args, path)
        leaves.append((path, value, dest, is_alias))
        if dest is not None and not is_alias:
            direct_dests.add(dest)

    # Wf-A11 PASS 2 — apply, collecting every skip for the report.
    skipped: list[dict[str, Any]] = []
    yaml_written: set[str] = set()
    for path, value, dest, is_alias in leaves:
        if dest is None:
            # Wf-A11 FIX (2026-10-03) — was `continue` (silent).  Now
            # collected and emitted; see report block below.
            skipped.append(_skip_entry(path, value))
            continue
        if is_alias and dest in direct_dests:
            # Alias loses to a direct YAML key on the same dest.  Recorded
            # (and strict-fatal) so the redundancy is visible, never silent.
            skipped.append(
                _skip_entry(
                    path,
                    value,
                    reason=f"alias suppressed: '{dest}' is set by a direct YAML key",
                )
            )
            continue
        cur = getattr(args, dest)
        dft = getattr(defaults, dest, _MISSING)
        if dft is _MISSING:
            continue
        # Special case: for LIST-valued args, YAML wins over the argparse
        # DEFAULT but NOT over an explicit CLI flag.  Rationale: a list
        # default (e.g. the built-in metals cohort ['Ru','Ir','Rh','Os','Re'])
        # is rarely the caller's intent, so YAML beats it — but a launcher
        # that enumerated one (metal, seed) pair per invocation passes
        # `--metals Ru --seeds 0` EXPLICITLY, and clobbering that with the
        # yaml cohort would silently run all 5 seeds x the yaml metals on
        # every pair.  Wf-R27 FIX 4 (2026-10-07): the old branch set the value
        # unconditionally, which made `cur == dft` unreachable for lists.
        # Guarding on `cur == dft` restores the documented precedence
        # (explicit CLI > YAML > argparse default) for lists too.
        #
        # 2026-10-09: the guard is now `_user_supplied`, which reads the REAL
        # signal (did argparse see the token?) whenever it can be established,
        # and falls back to the `cur == dft` proxy otherwise.
        if isinstance(value, list) and isinstance(cur, list):
            if not _user_supplied(dest, cur, dft):
                setattr(args, dest, value)
                yaml_written.add(dest)
            continue
        # `dest in yaml_written` keeps last-key-wins among YAML keys mapping
        # to the same dest (a prior YAML write must not look like a CLI pass).
        if not _user_supplied(dest, cur, dft) or dest in yaml_written:
            # Wf-A9 FIX (2026-10-03) — run the value through the action's
            # own ``type=`` converter so YAML and CLI values have the same
            # type (notably Path-valued flags, which would otherwise blow
            # up downstream on the first attribute access).
            conv = converters.get(dest)
            if conv is not None and not isinstance(value, list):
                try:
                    value = conv(value)
                except (TypeError, ValueError) as exc:
                    # 2026-10-09 — was a bare `continue`, which made a
                    # TYPE-MISMATCHED value indistinguishable from an absent
                    # key: the run silently trained on the argparse default
                    # that the config author never chose, with no warning and
                    # no entry in the skip report. Every other skip in this
                    # function is collected, classified and (under
                    # WFA2_CONFIG_STRICT=1) made fatal — this one path
                    # bypassed all of it.
                    #
                    # Measured on 2026-10-09 against
                    # train_cfm_yuelbond_wasserstein.py:
                    #     assemble_loss_weight: 0.25          -> 0.25  ok
                    #     assemble_loss_weight: "0.25"        -> 0.25  ok (coerced)
                    #     assemble_loss_weight: "nonsense"    -> 0.0   SILENT
                    #     assemble_loss_weight: null          -> 0.0   SILENT
                    #
                    # Keeping the default is still the right behaviour (a bad
                    # value in one key should not abort a 30-epoch run), but
                    # it must now be VISIBLE: recorded in `skipped` with a
                    # reason, so it reaches the stderr report, the strict mode,
                    # and scripts/_config_binding_gate.py's view of the file.
                    skipped.append(
                        _skip_entry(
                            path,
                            value,
                            reason=(
                                f"type mismatch: cannot convert {value!r} to "
                                f"{getattr(conv, '__name__', conv)} for dest "
                                f"'{dest}'; kept the argparse default"
                            ),
                        )
                    )
                    continue
            # User did NOT pass CLI -> YAML applies
            setattr(args, dest, value)
            yaml_written.add(dest)

    # Wf-A11 FIX — publish the skip report: namespace attr + module global +
    # one stderr line per key (WFA2_CONFIG_QUIET=1 silences per-key lines).
    # Default remains non-fatal; WFA2_CONFIG_STRICT=1 raises instead of
    # skipping any key without a documented non-trainer consumer.
    global _LAST_SKIPPED
    _LAST_SKIPPED = skipped
    setattr(args, "yaml_unresolved_keys", skipped)
    if skipped and not _quiet_enabled():
        # 2026-10-04 — split the report by intent.
        #
        # A key inside `_NON_TRAINER_SECTIONS` (entry / directives /
        # honest_framing / cross_references / waves / ablation_configs) has no
        # trainer dest BY DESIGN — that is what the section is for.  Warning
        # per key made a routine load print 40-78 lines, and a genuinely dead
        # key (`data.dataset`) drowned in them: `cfm_full_retrain.yaml`
        # declared a dataset it never read and nobody noticed, because the
        # noise floor was already 46 lines high.  Metadata sections now get one
        # aggregated line; only keys that COULD have resolved — or that sit in
        # a section where a trainer dest is expected — get a per-key WARNING.
        # Everything still lands in `last_skipped_keys()` and strict mode still
        # fails on it.
        metadata: dict[str, list[str]] = {}
        actionable: list[dict[str, Any]] = []
        for entry in skipped:
            section = str(entry.get("key", "")).split(".")[0]
            if section in _NON_TRAINER_SECTIONS:
                metadata.setdefault(section, []).append(str(entry.get("key", "")))
            else:
                actionable.append(entry)
        by_design = 0
        for entry in actionable:
            consumer = _non_trainer_consumer(entry)
            if consumer is not None and "reason" not in entry:
                # Verified non-trainer consumer (typically a start.sh env var
                # that is passed straight back to the trainer as a CLI flag).
                # Not a warning: the key IS consumed, just not by argparse.
                by_design += 1
                continue
            why = entry.get("reason") or "no argparse dest"
            print(
                f"[wfA2_config_loader] WARNING: YAML key "
                f"'{entry['key']}' (value={entry['value']!r}) skipped — {why}",
                file=sys.stderr,
            )
        if by_design:
            print(
                f"[wfA2_config_loader] note: {by_design} key(s) outside "
                "metadata sections have a verified non-trainer consumer "
                "(scripts/start.sh); see _NON_TRAINER_PATHS / _NON_TRAINER_KEYS.",
                file=sys.stderr,
            )
        for section, keys in sorted(metadata.items()):
            print(
                f"[wfA2_config_loader] note: {len(keys)} key(s) under "
                f"'{section}:' have no trainer dest by design (run-control "
                f"metadata, read by start.sh where applicable).",
                file=sys.stderr,
            )
        print(
            f"[wfA2_config_loader] {len(skipped)} YAML key(s) resolved to no "
            f"argparse dest ({len(actionable)} outside metadata sections); "
            "set WFA2_CONFIG_STRICT=1 to fail on these.",
            file=sys.stderr,
        )
    if _strict_enabled():
        fatal = [e for e in skipped if not _registered_non_trainer(e)]
        if fatal:
            raise UnresolvedConfigKey(
                "WFA2_CONFIG_STRICT: YAML key(s) with no argparse dest: "
                + ", ".join(f"{e['key']}={e['value']!r}" for e in fatal)
            )
    return args
