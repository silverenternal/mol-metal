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
_LEAF_FALLBACK_SECTIONS = frozenset({
    "protocol", "model", "flow", "data", "compute", "eval", "output",
    # Wf-A11 FIX (2026-10-03) — "train" added: configs/default.yaml carries
    # the real run hyperparameters (batch_size / lr / epochs / warmup_steps /
    # grad_clip_norm) under `train:` and the whole block was silently dropped
    # before this fix (see TODO/completed/config_dead_key_remediation.md).
    "train",
})


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
    "hidden_dim": ("hidden",),               # D-MPNN: --hidden (CFM declares --hidden-dim itself)
    "grad_clip_norm": ("max_norm",),         # both trainers: --max-norm (no --grad-clip-norm exists)
    "warmup_steps": ("lr_warmup_steps",),    # CFM: --lr-warmup-steps (D-MPNN has epochs-based warmup only)
    "warmup_epochs": ("lr_warmup_epochs",),  # both trainers: --lr-warmup-epochs
    "min_lr": ("lr_min_lr",),                # both trainers: --lr-min-lr
    "cpu_workers": ("cpu_num_workers",),     # D-MPNN: --cpu-num-workers (CFM has no workers dest)
}

#: Top-level sections that describe run metadata / enumeration, not trainer
#: hyperparameters.  A skipped key inside one of these is by design, so
#: strict mode does not fail on it.
_NON_TRAINER_SECTIONS = frozenset({
    "entry", "directives", "honest_framing", "cross_references",
    "waves", "ablation_configs", "sweep_pairs", "runs", "pending_pairs",
})

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
        "1", "true", "yes", "on",
    }


def _quiet_enabled() -> bool:
    return os.environ.get("WFA2_CONFIG_QUIET", "").strip().lower() in {
        "1", "true", "yes", "on",
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


def _registered_non_trainer(entry: Mapping[str, Any]) -> bool:
    """True when the skip is by design (metadata section or named key)."""
    return (
        entry.get("section") in _NON_TRAINER_SECTIONS
        or entry.get("leaf") in _NON_TRAINER_KEYS
    )


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
            skipped.append(_skip_entry(
                path, value,
                reason=f"alias suppressed: '{dest}' is set by a direct YAML key",
            ))
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
            yaml_written.add(dest)
            continue
        # `dest in yaml_written` keeps last-key-wins among YAML keys mapping
        # to the same dest (a prior YAML write must not look like a CLI pass).
        if cur == dft or dest in yaml_written:
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
            yaml_written.add(dest)

    # Wf-A11 FIX — publish the skip report: namespace attr + module global +
    # one stderr line per key (WFA2_CONFIG_QUIET=1 silences per-key lines).
    # Default remains non-fatal; WFA2_CONFIG_STRICT=1 raises instead of
    # skipping any key without a documented non-trainer consumer.
    global _LAST_SKIPPED
    _LAST_SKIPPED = skipped
    setattr(args, "yaml_unresolved_keys", skipped)
    if skipped and not _quiet_enabled():
        for entry in skipped:
            why = entry.get("reason") or "no argparse dest"
            consumer = _NON_TRAINER_KEYS.get(entry.get("leaf", ""))
            if consumer and "reason" not in entry:
                why = f"{why} (consumer: {consumer})"
            print(
                f"[wfA2_config_loader] WARNING: YAML key "
                f"'{entry['key']}' (value={entry['value']!r}) skipped — {why}",
                file=sys.stderr,
            )
        print(
            f"[wfA2_config_loader] {len(skipped)} YAML key(s) resolved to no "
            "argparse dest; set WFA2_CONFIG_STRICT=1 to fail on these.",
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
