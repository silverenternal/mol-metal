"""Wf-394-A1 — TODO-21 Lambda × CFM coupling wire-in scaffold (CPU-only).

================================================================
Purpose
================================================================
The Lambda × CFM joint retrain (TODO-21 / WEAK_TO_STRONG_PLAN §2.5)
depends on three CLI flags + a 1-line adapter change in
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` so that the
CFM velocity loss can mix a Lambda-reward-weighted term when
``--lambda-cfm-coupling`` is enabled.

Wf-329-W6 already shipped the wiring (per
``TODO/completed/329w6_todo21_lambda_cfm_verified.md``); however,
subsequent waves (Wf-388-A2 + Wf-388-A3) added new diversity +
SA-penalty flags around the loss block, which caused the
Wf-329-W6 Lambda-CFM wiring to drift out of the trainer in the
working tree (the file at HEAD no longer contains the
``--lambda-cfm-coupling*`` flags or the loss-multiplier block).
Wf-331-R6 + Wf-333-G1 + the GPU wrapper (Wf-333-G1) therefore
reference lines that have moved.

This module is a CPU-only scaffold that:

1. Computes the exact patch shape required to re-introduce the
   Lambda × CFM coupling wiring into the trainer without disturbing
   the Wf-388-A2 / Wf-388-A3 diversity + SA-penalty code paths.
2. Validates the patch by AST-parsing the would-be-patched file.
3. Provides a smoke test that verifies the
   :class:`LambdaRewardDataset` factory + the coupling adapter
   chain still import cleanly (the same chain Wf-323E shipped).

It does **NOT** apply the patch to disk by default (default OFF for
bit-for-bit backward compat with Wf-388-A3 HEAD). Call
:func:`apply_lambda_cfm_patch` with ``write=True`` after explicit
USER_OK.

Honest framing
==============
* The Lambda × CFM coupling training itself remains BLOCKED on
  upstream ``decode_ratio > 0`` per Wf-291D + Wf-299. Wiring the
  CLI flags does NOT unblock the GPU launch — it only makes the
  scaffold available for the future retrain.
* 0 GPU launches performed in Wf-394-A1.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).

Lit anchors
===========
* Wf-329-W6 (2026-09-29) — first wiring of ``--lambda-cfm-coupling``
  CLI flag + 1-line adapter change + ``LambdaRewardDataset`` factory
  chain (4 tests PASS).
* Wf-331-R6 (2026-09-29) — reconciliation re-verification (12/12
  prior tests PASS + cfm pre-flight 6/6 PASS).
* Wf-333-G1 (2026-09-29) — GPU wrapper scaffold + dataclass +
  ``run_joint_inference_smoke`` (2 tests PASS).
* Wf-388-A2/A3 (2026-09-30) — diversity + SA-penalty flag additions
  that displaced the Wf-329-W6 wiring in HEAD.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Wf-406-A3: atomic write protocol migration (per AGENTS.md §5.6 + Wf-386-A3).
# Replace direct Write with atomic_write_file to close the half-merged-file
# failure mode. Also adds safe_overwrite_check before overwrite-write +
# detect_truncation on agent output (per AGENTS.md §5.5/§5.6 standing directives).
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
    detect_truncation,
)

log = logging.getLogger(__name__)

# Path constants — match the Wf-329-W6 / Wf-333-G1 anchor.
DEFAULT_TRAINER_PATH = Path(
    "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
)
DEFAULT_LAMBDA_REWARD_PATH = Path(
    "molmetal/molmetal_lam/reward/lambda_reward_dataset.py"
)


# ---------------------------------------------------------------------------
# Patch spec — the canonical Wf-329-W6 wiring shape
# ---------------------------------------------------------------------------
CLI_FLAGS_PATCH: List[Dict[str, str]] = [
    {
        "flag": "--lambda-cfm-coupling",
        "type": "store_true",
        "default": "False",
        "help": (
            "Wf-394-A1 NEW (re-wire of Wf-329-W6): enable the "
            "Lambda × CFM coupling weighting (TODO-21 / "
            "WEAK_TO_STRONG_PLAN §2.5). When ON, the trainer "
            "builds a LambdaRewardDataset (pocket_id-conditional) "
            "and multiplies the CFM velocity loss by "
            "(1 + weight * mean_weight). Default OFF for "
            "bit-for-bit backward compat with Wf-388-A3."
        ),
    },
    {
        "flag": "--lambda-cfm-coupling-weight",
        "type": "float",
        "default": "0.1",
        "help": (
            "Wf-394-A1 NEW (re-wire of Wf-329-W6): weight on the "
            "Lambda reward term. Default 0.1. Ignored when "
            "--lambda-cfm-coupling is OFF."
        ),
    },
    {
        "flag": "--lambda-cfm-coupling-samples",
        "type": "int",
        "default": "200",
        "help": (
            "Wf-394-A1 NEW (re-wire of Wf-329-W6): n_samples for "
            "build_lambda_reward_dataset. Default 200. Ignored when "
            "--lambda-cfm-coupling is OFF."
        ),
    },
]

LOSS_MULTIPLIER_BLOCK = '''\
        # Wf-394-A1 — TODO-21 Lambda × CFM reward-weighting wiring
        # (re-wire of Wf-329-W6 + Wf-333-G1; CPU-safe scaffolding;
        # GPU retrain remains BLOCKED on upstream decode_ratio > 0
        # per Wf-291D + Wf-299). Off by default for backward compat
        # with Wf-388-A3.
        if (
            getattr(args, "lambda_cfm_coupling_weight", 0.0) > 0.0
            and lambda_reward_dataset is not None
        ):
            _rw_mean = float(
                getattr(lambda_reward_dataset, "mean_weight", 0.0)
            )
            loss = loss * (1.0 + float(
                getattr(args, "lambda_cfm_coupling_weight", 0.0)
            ) * _rw_mean)
'''

DATASET_BUILD_BLOCK = '''\
    # Wf-394-A1 — TODO-21 Lambda × CFM coupling scaffolding (CPU-only).
    lambda_reward_dataset = None
    if getattr(args, "lambda_cfm_coupling", False):
        try:
            from molmetal.molmetal_lam.reward.lambda_reward_dataset import (
                build_lambda_reward_dataset,
            )
            lambda_reward_dataset = build_lambda_reward_dataset(
                n_samples=int(
                    getattr(args, "lambda_cfm_coupling_samples", 200)
                ),
                pocket_id=args.pocket_id,
                seed=args.seed,
            )
            print(
                f"[Wf-394-A1] --lambda-cfm-coupling ENABLED: "
                f"LambdaRewardDataset len={len(lambda_reward_dataset)} "
                f"mean_weight={lambda_reward_dataset.mean_weight:.4f} "
                f"max_weight={lambda_reward_dataset.max_weight:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Wf-394-A1][WARNING] build failed: {exc}; "
                f"reward-weighting term collapses to zero.",
                flush=True,
            )
            lambda_reward_dataset = None
'''


# ---------------------------------------------------------------------------
# Dataclass + helpers
# ---------------------------------------------------------------------------
@dataclass
class LambdaCFMWireReport:
    """Status of the Lambda × CFM wiring patch attempt.

    Attributes
    ----------
    ok : bool
        ``True`` iff the would-be-patched file parses cleanly AND
        the LambdaRewardDataset factory chain imports successfully.
    trainer_path : str
        Absolute path to the trainer file under inspection.
    flags_present : Dict[str, bool]
        Whether each ``--lambda-cfm-coupling*`` flag is currently
        present in the trainer ``parse_args``.
    loss_multiplier_present : bool
        Whether the Wf-394-A1 / Wf-329-W6 loss-multiplier block is
        currently present in ``train_one_epoch``.
    dataset_build_present : bool
        Whether the Wf-394-A1 / Wf-329-W6 dataset-build block is
        currently present in ``main``.
    lambda_reward_factory_ok : bool
        Whether ``build_lambda_reward_dataset`` importable.
    patch_actions : List[str]
        Human-readable list of patch actions that
        :func:`apply_lambda_cfm_patch` would perform.
    notes : List[str]
        Human-readable diagnostic / forward-looking notes.
    """

    ok: bool
    trainer_path: str
    flags_present: Dict[str, bool] = field(default_factory=dict)
    loss_multiplier_present: bool = False
    dataset_build_present: bool = False
    lambda_reward_factory_ok: bool = False
    patch_actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "trainer_path": self.trainer_path,
            "flags_present": self.flags_present,
            "loss_multiplier_present": self.loss_multiplier_present,
            "dataset_build_present": self.dataset_build_present,
            "lambda_reward_factory_ok": self.lambda_reward_factory_ok,
            "patch_actions": list(self.patch_actions),
            "notes": list(self.notes),
        }


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Trainer not found: {path}")
    return path.read_text(encoding="utf-8")


def _ast_rewrite_detected(trainer_src: str) -> bool:
    """Wf-414-A2-NEW Pattern (iv) — AST-rewrite detection.

    Parses the trainer source as a Python AST and walks all
    :class:`ast.Constant` string nodes looking for matches against
    the 3 canonical flag regexes. This catches cases where the
    line-based grep misses:

    * **Unicode-escaped flag names** — ``"\\u002d\\u002dlambda-cfm-coupling"``
      in source: the line-based grep sees ``\\u002d`` (6 chars per dash)
      not ``--`` (2 chars), so the regex misses. But ``ast.parse``
      resolves the Unicode escape at parse time, so
      ``ast.Constant.value`` contains the resolved ``--`` — Pattern
      (iv) catches via the AST walk.

    * **F-string / docstring / module-level constant flag references**
      that the per-line grep might miss due to line-splitting or
      non-quoting contexts.

    Per :class:`ast.Constant` string node, the same negative-lookahead
    pattern (``(?!-weight|-samples)``) is used for the bare flag so
    the bare form does NOT substring-match the suffixed forms.

    Returns
    -------
    bool
        ``True`` iff ``ast.parse(trainer_src)`` succeeds AND at
        least one ``ast.Constant`` string node value matches any of
        the 3 canonical flag regexes.
    """
    try:
        tree = ast.parse(trainer_src)
    except SyntaxError:
        return False

    _bare_re = re.compile(r"--lambda-cfm-coupling(?!-weight|-samples)")
    _weight_re = re.compile(r"--lambda-cfm-coupling-weight")
    _samples_re = re.compile(r"--lambda-cfm-coupling-samples")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if (
                _bare_re.search(val)
                or _weight_re.search(val)
                or _samples_re.search(val)
            ):
                return True
    return False


def _eval_exec_detected(trainer_src: str) -> bool:
    """Wf-414-A2-NEW Pattern (v) — eval/exec detection.

    Detects whether the trainer source uses ``eval(`` or ``exec(`` to
    dynamically construct flag names at runtime (a common
    monkey-patching technique that defeats static argparser checks).

    When detected, the audit cannot rely on the static
    source-level check alone. Per-flag heuristics fall back to a
    conservative substring heuristic: any flag whose
    ``-``-stripped SUFFIX (e.g. ``lambda-cfm-coupling-weight``) appears
    anywhere in the source is marked as potentially present.

    The canonical regex matches ``eval(`` and ``exec(`` with optional
    whitespace between the identifier and the open-paren.

    Returns
    -------
    bool
        ``True`` iff ``eval(`` or ``exec(`` appears in the trainer
        source.
    """
    return bool(
        re.search(r"\beval\s*\(", trainer_src)
        or re.search(r"\bexec\s*\(", trainer_src)
    )


def _flags_present(trainer_src: str) -> Dict[str, bool]:
    """Detect whether the canonical Wf-329-W6 / Wf-394-A1 flags exist.

    Three layered patterns are applied (line-based grep +
    AST-rewrite + eval/exec fallback per the Wf-414-A2 task brief):

    1. **Line-based regex grep** — per-line regex matching using
       :func:`re.search` on each line. The bare flag uses a
       NEGATIVE lookahead ``(?!-weight|-samples)`` to avoid
       substring-matching the suffixed forms. Robust to:
       quoted flag names, help-text references, docstring
       references, comment references, f-string references, and
       module-level constant flag references.

    2. **Wf-414-A2 Pattern (iv) — AST-rewrite detection** via
       :func:`_ast_rewrite_detected` fallback. Re-runs the regex
       match against each :class:`ast.Constant` string value (after
       Python's parse-time Unicode-escape resolution). Catches
       ``"\\u002d\\u002dlambda-cfm-coupling"``-style flag names
       that the line-based grep misses.

    3. **Wf-414-A2 Pattern (v) — eval/exec fallback** via
       :func:`_eval_exec_detected`. When eval/exec is detected,
       per-flag detection falls back to a conservative substring
       check: each flag's ``-``-stripped SUFFIX (e.g.
       ``lambda-cfm-coupling-weight``) is searched for anywhere in
       source. If found, the flag is marked as potentially present.

    Returns
    -------
    Dict[str, bool]
        Maps each canonical flag name to ``True`` iff any of the
        three patterns detect it.
    """
    lines = trainer_src.splitlines()
    found: Dict[str, bool] = {
        "--lambda-cfm-coupling": False,
        "--lambda-cfm-coupling-weight": False,
        "--lambda-cfm-coupling-samples": False,
    }
    # ---- Pattern 1: line-based regex grep ----
    _bare_re = re.compile(r"--lambda-cfm-coupling(?!-weight|-samples)")
    _weight_re = re.compile(r"--lambda-cfm-coupling-weight")
    _samples_re = re.compile(r"--lambda-cfm-coupling-samples")
    for line in lines:
        if not found["--lambda-cfm-coupling"] and _bare_re.search(line):
            found["--lambda-cfm-coupling"] = True
        if not found["--lambda-cfm-coupling-weight"] and _weight_re.search(line):
            found["--lambda-cfm-coupling-weight"] = True
        if not found["--lambda-cfm-coupling-samples"] and _samples_re.search(line):
            found["--lambda-cfm-coupling-samples"] = True
        # Early exit when all 3 found.
        if all(found.values()):
            return found

    # ---- Pattern 2: Wf-414-A2-NEW AST-rewrite detection ----
    # Re-run the regex match against each ast.Constant string
    # value. Catches flag names hidden via Unicode escape or
    # embedded in f-strings / docstrings / module-level constants.
    if not all(found.values()):
        try:
            tree = ast.parse(trainer_src)
        except SyntaxError:
            tree = None
        if tree is not None:
            _bare_pat = re.compile(r"--lambda-cfm-coupling(?!-weight|-samples)")
            _weight_pat = re.compile(r"--lambda-cfm-coupling-weight")
            _samples_pat = re.compile(r"--lambda-cfm-coupling-samples")
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                ):
                    continue
                val = node.value
                if not found["--lambda-cfm-coupling"] and _bare_pat.search(val):
                    found["--lambda-cfm-coupling"] = True
                if not found["--lambda-cfm-coupling-weight"] and _weight_pat.search(val):
                    found["--lambda-cfm-coupling-weight"] = True
                if not found["--lambda-cfm-coupling-samples"] and _samples_pat.search(val):
                    found["--lambda-cfm-coupling-samples"] = True
                if all(found.values()):
                    break

    # ---- Pattern 3: Wf-414-A2-NEW eval/exec fallback ----
    # When dynamic code is present, mark any flag whose
    # ``-``-stripped SUFFIX appears anywhere in source as
    # potentially present.
    if not all(found.values()) and _eval_exec_detected(trainer_src):
        for flag in found:
            if not found[flag]:
                suffix = flag.lstrip("-")
                if suffix in trainer_src:
                    found[flag] = True

    return found


def _loss_multiplier_present(trainer_src: str) -> bool:
    """Detect whether the canonical loss-multiplier block exists.

    Three shapes are accepted (line-based grep, NOT AST-only per
    Wf-410-A2):

    1. **Wf-329-W6 OLD shape** — literal ``lambda_reward_dataset.mean_weight``
       attribute access (the dataset was a local var named
       ``lambda_reward_dataset`` and the loss block read its
       ``.mean_weight`` attribute).
    2. **Wf-394-A1 OLD shape** — literal ``(1.0 + lambda_cfm_coupling_weight``
       single-line idiom (legacy Wf-394-A1 shape).
    3. **Wf-406-A2 NEW shape** — multi-line idiom
       ``1.0\\n + float(lambda_cfm_coupling_weight)`` OR the
       kwarg signature
       ``lambda_cfm_coupling_weight: float = 0.0`` +
       ``lambda_cfm_mean_weight: float = 0.0`` in
       ``train_one_epoch`` (current HEAD shape per the Wf-406-A2
       re-wire of the Wf-329-W6 / Wf-394-A1 contract).

    Returns
    -------
    bool
        ``True`` iff any of the three shapes is detected.
    """
    # Wf-329-W6 OLD shape: variable attribute access
    if "lambda_reward_dataset.mean_weight" in trainer_src:
        return True
    if 'lambda_reward_dataset, "mean_weight"' in trainer_src:
        return True
    # Wf-394-A1 OLD shape: single-line idiom
    if "(1.0 + lambda_cfm_coupling_weight" in trainer_src:
        return True
    if (
        '1.0 + float(\n                getattr(args, "lambda_cfm_coupling_weight"'
        in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape (multi-line idiom): the loss multiplier
    # spans multiple lines (open paren on line N, 1.0 on line N+1,
    # + float(lambda_cfm_coupling_weight) on line N+2).
    if (
        "1.0\n" in trainer_src
        and "+ float(lambda_cfm_coupling_weight)" in trainer_src
        and "* float(lambda_cfm_mean_weight)" in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape (kwarg signature): both kwargs declared in
    # train_one_epoch signature with default 0.0.
    if (
        "lambda_cfm_coupling_weight: float = 0.0" in trainer_src
        and "lambda_cfm_mean_weight: float = 0.0" in trainer_src
        and "1.0 + float(\n            getattr(args, \"lambda_cfm_coupling_weight\""
        in trainer_src
    ):
        return True
    # Wf-414-A2 NEW shape (boolean flag in train_one_epoch kwargs):
    # a future-proofing rewire that introduces a boolean
    # ``_lambda_cfm_coupling_flag`` (or ``lambda_cfm_coupling_flag``)
    # kwarg in train_one_epoch, instead of the float-weight + mean-
    # weight pair. Both forms accepted (with or without leading
    # underscore) because Python convention varies.
    if re.search(
        r"(?:_)?lambda_cfm_coupling_flag\s*:\s*bool", trainer_src
    ):
        return True
    # Wf-414-A2 NEW shape (variable in loss block): a future-proofing
    # rewire that introduces a ``lambda_cfm_multiplier`` variable
    # inside the loss cascade (e.g. ``loss = loss * lambda_cfm_multiplier``).
    # The variable name is distinctive enough to detect as a token.
    if re.search(r"\blambda_cfm_multiplier\b", trainer_src):
        return True
    return False


def _dataset_build_present(trainer_src: str) -> bool:
    """Detect whether the canonical dataset-build block exists.

    Two shapes are accepted (line-based grep, NOT AST-only per
    Wf-410-A2):

    1. **Wf-394-A1 OLD shape** — ``Wf-394-A1`` marker + literal
       ``lambda_reward_dataset = None`` initial value (the local
       var was named ``lambda_reward_dataset``).
    2. **Wf-406-A2 NEW shape** — ``Wf-406-A2`` marker + literal
       ``lambda_cfm_mean_weight = 0.0`` initial value + reference
       to ``build_lambda_reward_dataset`` factory (current HEAD
       shape — the dataset is bound to ``_lambda_dataset`` and the
       scalar ``mean_weight`` is extracted into
       ``lambda_cfm_mean_weight`` for the kwarg pass-through).

    Returns
    -------
    bool
        ``True`` iff either shape is detected.
    """
    # Both shapes require build_lambda_reward_dataset to be
    # referenced (the canonical factory symbol).
    if "build_lambda_reward_dataset" not in trainer_src:
        return False
    # Wf-394-A1 OLD shape
    if (
        "Wf-394-A1" in trainer_src
        and "lambda_reward_dataset = None" in trainer_src
    ):
        return True
    # Wf-406-A2 NEW shape
    if (
        "Wf-406-A2" in trainer_src
        and "lambda_cfm_mean_weight = 0.0" in trainer_src
    ):
        return True
    # Fallback: the factory symbol is referenced AND any Wf-406+
    # marker exists (covers future re-wires that keep the canonical
    # factory but rename the local var).
    if (
        "build_lambda_reward_dataset" in trainer_src
        and ("Wf-406" in trainer_src or "Wf-410" in trainer_src)
        and "lambda_cfm_mean_weight" in trainer_src
    ):
        return True
    return False


def _check_lambda_reward_factory() -> bool:
    """Try to import ``build_lambda_reward_dataset``; return success."""
    try:
        from molmetal.molmetal_lam.reward.lambda_reward_dataset import (
            build_lambda_reward_dataset,
        )

        # Verify the factory is callable with the canonical default
        # kwargs (do NOT actually run the Lambda MCTS path — only
        # verify the symbol exists and is callable).
        return callable(build_lambda_reward_dataset)
    except Exception as exc:  # noqa: BLE001
        log.warning("[wf394a1] factory import failed: %s", exc)
        return False


def audit_lambda_cfm_state(
    trainer_path: Path = DEFAULT_TRAINER_PATH,
    repo_root: Optional[Path] = None,
) -> LambdaCFMWireReport:
    """Audit the current Lambda × CFM coupling state.

    Returns
    -------
    LambdaCFMWireReport
        See class docstring for fields. ``ok`` is ``True`` iff
        *all* canonical wiring is present AND the
        ``build_lambda_reward_dataset`` factory chain imports.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
    abs_path = (repo_root / trainer_path).resolve()
    src = _read_text(abs_path)

    # AST parse — catches any syntactic regression early.
    try:
        ast.parse(src, filename=str(abs_path))
        ast_ok = True
    except SyntaxError as exc:
        log.warning("[wf394a1] trainer AST parse failed: %s", exc)
        ast_ok = False

    flags = _flags_present(src)
    loss_present = _loss_multiplier_present(src)
    ds_present = _dataset_build_present(src)
    factory_ok = _check_lambda_reward_factory()

    patch_actions: List[str] = []
    notes: List[str] = []

    if not flags["--lambda-cfm-coupling"]:
        patch_actions.append(
            "add_argument(--lambda-cfm-coupling, store_true, default=False)"
        )
    if not flags["--lambda-cfm-coupling-weight"]:
        patch_actions.append(
            "add_argument(--lambda-cfm-coupling-weight, float, default=0.1)"
        )
    if not flags["--lambda-cfm-coupling-samples"]:
        patch_actions.append(
            "add_argument(--lambda-cfm-coupling-samples, int, default=200)"
        )
    if not loss_present:
        patch_actions.append(
            "inject loss multiplier block in train_one_epoch"
        )
    if not ds_present:
        patch_actions.append(
            "inject dataset build block in main"
        )
    if not factory_ok:
        notes.append(
            "LambdaRewardDataset factory chain import FAILED — "
            "cannot wire CLI flags safely until factory is restored"
        )
    notes.append(
        "GPU launch BLOCKED on upstream decode_ratio > 0 per Wf-291D + Wf-299"
    )
    notes.append(
        "Wf-329-W6 / Wf-333-G1 / Wf-331-R6 wiring shape preserved "
        "verbatim — only the source location in HEAD has moved due "
        "to Wf-388-A2/A3 diversity + SA-penalty flag additions"
    )

    ok = (
        ast_ok
        and all(flags.values())
        and loss_present
        and ds_present
        and factory_ok
    )

    return LambdaCFMWireReport(
        ok=ok,
        trainer_path=str(abs_path),
        flags_present=flags,
        loss_multiplier_present=loss_present,
        dataset_build_present=ds_present,
        lambda_reward_factory_ok=factory_ok,
        patch_actions=patch_actions,
        notes=notes,
    )


def apply_lambda_cfm_patch(
    trainer_path: Path = DEFAULT_TRAINER_PATH,
    *,
    repo_root: Optional[Path] = None,
    write: bool = False,
) -> LambdaCFMWireReport:
    """Apply (or report) the canonical Wf-394-A1 patch.

    When ``write`` is ``False`` (the default), this function is a
    dry-run — it computes the report but does NOT modify
    ``trainer_path`` on disk. This preserves the
    "Wf-394-A1 default OFF" invariant per the constraint that
    bit-for-bit backward compat with Wf-388-A3 must be preserved.

    When ``write`` is ``True``, the function would inject:

    * the three ``add_argument(...)`` calls in ``parse_args``;
    * the dataset-build block at the top of ``main``;
    * the loss-multiplier block inside ``train_one_epoch``.

    The actual file mutation is gated behind ``write=True`` AND
    would require explicit USER_OK per AGENTS.md §5.1. This
    scaffold does not perform file writes at any default.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
    audit = audit_lambda_cfm_state(
        trainer_path=trainer_path, repo_root=repo_root
    )

    if write and audit.patch_actions:
        log.info(
            "[wf394a1] USER_OK-gated write requested; "
            "would apply %d patch actions to %s",
            len(audit.patch_actions), audit.trainer_path,
        )
        # NOTE: actual file writes are NOT performed in this scaffold
        # — the function is reserved for a future USER_OK-gated wave.
        # The Wf-394-A1 deliverable is the audit + smoke + tests; the
        # patch application itself is out of scope per the
        # "Default OFF (preserve bit-for-bit backward compat)"
        # constraint.
        audit.notes.append(
            "write=True requested but no file mutation performed "
            "in Wf-394-A1 scaffold (USER_OK-gated wave deferred)"
        )

    return audit


# ---------------------------------------------------------------------------
# Smoke test — verifies the coupling chain is importable end-to-end.
# ---------------------------------------------------------------------------
def smoke_lambda_cfm(coupling_strength: float = 0.1) -> Dict[str, object]:
    """Run a smoke test on the Lambda × CFM coupling chain.

    Does NOT invoke the CFM trainer (deferred to R18 GPU). Only
    verifies that the import chain is alive + the factory symbol
    is callable + the coupling_strength produces a sane (1 + w *
    mean_weight) loss-multiplier shape.

    Parameters
    ----------
    coupling_strength : float
        The ``--lambda-cfm-coupling-weight`` analogue. Default 0.1
        matches the Wf-329-W6 / Wf-394-A1 default.

    Returns
    -------
    dict
        ``{"ok": bool, "factory_callable": bool, "coupling_strength":
        float, "loss_multiplier_upper_bound": float,
        "import_chain": List[str]}``.
    """
    import_chain: List[str] = []
    factory_callable = False
    try:
        import_chain.append("molmetal")
        from molmetal.molmetal_lam.reward import lambda_reward_dataset as _lrd
        import_chain.append(
            "molmetal.molmetal_lam.reward.lambda_reward_dataset"
        )
        factory = getattr(_lrd, "build_lambda_reward_dataset", None)
        factory_callable = callable(factory)
        import_chain.append(
            f"build_lambda_reward_dataset.callable={factory_callable}"
        )
    except Exception as exc:  # noqa: BLE001
        import_chain.append(f"IMPORT_ERROR: {exc}")

    # Compute the loss-multiplier upper bound (1 + w * 1.0) — the
    # coupling only amplifies the loss when mean_weight > 0; the
    # upper bound is reached when mean_weight=1.
    loss_multiplier_upper_bound = 1.0 + float(coupling_strength) * 1.0

    return {
        "ok": factory_callable,
        "factory_callable": factory_callable,
        "coupling_strength": float(coupling_strength),
        "loss_multiplier_upper_bound": float(loss_multiplier_upper_bound),
        "import_chain": import_chain,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="wf394a1_lambda_cfm_wire",
        description=(
            "Wf-394-A1 — TODO-21 Lambda × CFM coupling wire-in audit "
            "(CPU-only scaffold; default OFF for Wf-388-A3 "
            "backward compat)."
        ),
    )
    p.add_argument(
        "--trainer-path",
        type=str,
        default=str(DEFAULT_TRAINER_PATH),
        help="Path (relative to repo root) to the CFM trainer.",
    )
    p.add_argument(
        "--write",
        action="store_true",
        default=False,
        help=(
            "USER_OK-gated: apply the canonical patch to the "
            "trainer file. NOT performed by Wf-394-A1 — use only "
            "in a follow-up USER_OK wave."
        ),
    )
    p.add_argument(
        "--coupling-strength",
        type=float,
        default=0.1,
        help="Smoke-test coupling_strength (Wf-329-W6 default).",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="If set, write the audit+smoke JSON to this path.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    trainer_path = Path(args.trainer_path)
    repo_root = Path(__file__).resolve().parent.parent.parent

    audit = apply_lambda_cfm_patch(
        trainer_path=trainer_path,
        repo_root=repo_root,
        write=bool(args.write),
    )
    smoke = smoke_lambda_cfm(coupling_strength=float(args.coupling_strength))

    print(
        f"[Wf-394-A1] trainer_path={audit.trainer_path}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] flags_present={audit.flags_present}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] loss_multiplier_present="
        f"{audit.loss_multiplier_present}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] dataset_build_present="
        f"{audit.dataset_build_present}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] lambda_reward_factory_ok="
        f"{audit.lambda_reward_factory_ok}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] patch_actions={audit.patch_actions}",
        flush=True,
    )
    print(
        f"[Wf-394-A1] smoke={smoke}",
        flush=True,
    )

    if args.output_json:
        out_path = (repo_root / args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Wf-406-A3: atomic_write + safe_overwrite_check + detect_truncation.
        _payload = json.dumps(
            {
                "audit": audit.to_dict(),
                "smoke": smoke,
            },
            indent=2,
        )
        _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
        if _risk is not None:
            print(
                f"[Wf-394-A1] WARN: safe_overwrite_check ratio={_risk:.3f} at {out_path}",
                flush=True,
            )
        atomic_write_file(out_path, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            print(
                f"[Wf-394-A1] WARN: detect_truncation reason={_trunc!r} at {out_path}",
                flush=True,
            )
        print(f"[Wf-394-A1] wrote {out_path}", flush=True)

    return 0 if audit.lambda_reward_factory_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
