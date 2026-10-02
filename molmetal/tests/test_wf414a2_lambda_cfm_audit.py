"""Wf-414-A2 — TODO-21 Lambda × CFM coupling audit heuristic fix tests.

================================================================
Background
================================================================
Wf-394-A1 (2026-09-30) shipped the original audit + smoke scaffold
for the TODO-21 Lambda × CFM coupling re-wire. Wf-410-A2 (2026-09-30)
fixed two FALSE-NEGATIVES in the original heuristic (the Wf-406-A2
patch shape uses ``lambda_cfm_mean_weight`` kwarg + ``Wf-406-A2``
marker, NOT the Wf-394-A1 OLD shapes).

Wf-414-A2 (this wave) extends the audit heuristic with THREE NEW
detection patterns per the Wf-414-A2 task brief:

1. **Line-based grep for the 3 canonical CLI flags** (in
   ``_flags_present``) — replaces the prior pure-substring search
   with per-line regex. Catches help-text references, docstring
   references, comment references, and f-string references.
2. **Pattern match for ``_lambda_cfm_coupling_flag`` boolean kwarg**
   (in ``_loss_multiplier_present``) — future-proofs the audit
   against a hypothetical rewire that introduces a boolean
   ``_lambda_cfm_coupling_flag`` kwarg in ``train_one_epoch``
   instead of the float-weight + mean-weight pair.
3. **Pattern match for ``lambda_cfm_multiplier`` variable** (in
   ``_loss_multiplier_present``) — future-proofs the audit
   against a hypothetical rewire that uses a named
   ``lambda_cfm_multiplier`` variable inside the loss cascade.

Honest framing
==============
* 0 GPU launches performed in Wf-414-A2.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
* The Wf-414-A2 heuristic update is **backward-compatible**: the
  OLD Wf-329-W6 + Wf-394-A1 + Wf-406-A2 shapes continue to be
  detected. The NEW line-based grep + the NEW hypothetical kwarg
  patterns are ADDITIONAL acceptance criteria, not replacements.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_PATH = (
    REPO_ROOT / "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
)
sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf394a1_lambda_cfm_wire import (  # noqa: E402
    DEFAULT_TRAINER_PATH,
    _dataset_build_present,
    _flags_present,
    _loss_multiplier_present,
    audit_lambda_cfm_state,
)


def test_heuristic_catches_patched_trainer() -> None:
    """Wf-414-A2 — line-based grep + new loss-multiplier patterns MUST
    detect the Wf-406-A2 patch in HEAD (the canonical "no
    FALSE-NEGATIVE" success criterion).

    Asserts
    -------
    1. ``_flags_present(trainer_src)`` returns all 3 CLI flags True
       via the new line-based grep (not via substring match).
    2. ``_loss_multiplier_present(trainer_src)`` returns True via
       either the Wf-406-A2 multi-line idiom OR the Wf-414-A2
       hypothetical ``lambda_cfm_multiplier`` variable pattern.
    3. ``_dataset_build_present(trainer_src)`` returns True (the
       Wf-406-A2 + Wf-410-A2 detection still works).
    4. End-to-end ``audit_lambda_cfm_state`` report has all 4
       contracts PASS + ``patch_actions == []`` (the canonical
       fix-the-FALSE-NEGATIVE success criterion).
    """
    # ---- Sanity: the trainer file MUST be present ----
    assert TRAINER_PATH.exists(), (
        f"CFM trainer not found at {TRAINER_PATH}; "
        "test cannot exercise the Wf-414-A2 heuristic fix."
    )
    trainer_src = TRAINER_PATH.read_text(encoding="utf-8")

    # ---- Check 1: line-based grep detects all 3 CLI flags ----
    flags = _flags_present(trainer_src)
    assert flags == {
        "--lambda-cfm-coupling": True,
        "--lambda-cfm-coupling-weight": True,
        "--lambda-cfm-coupling-samples": True,
    }, (
        f"Wf-414-A2 line-based grep MUST detect all 3 canonical "
        f"--lambda-cfm-coupling* flags in HEAD (the patched trainer). "
        f"Got flags_present={flags}. If this fails, either the patch "
        f"has regressed OR the line-based grep regression has been "
        f"introduced."
    )

    # ---- Check 2: loss-multiplier heuristic detects the Wf-406-A2
    # multi-line idiom (the canonical regression-of-focus) ----
    assert _loss_multiplier_present(trainer_src), (
        "Wf-414-A2 heuristic fix: _loss_multiplier_present MUST return "
        "True for the Wf-406-A2 multi-line idiom "
        "(`1.0\\n + float(lambda_cfm_coupling_weight)`) OR the new "
        "Wf-414-A2 hypothetical `lambda_cfm_multiplier` variable pattern. "
        "If this assertion fails, the Wf-414-A2 heuristic update has "
        "regressed."
    )

    # ---- Check 3: dataset-build heuristic detects the Wf-406-A2
    # initial-value pattern (Wf-410-A2 detection still works) ----
    assert _dataset_build_present(trainer_src), (
        "Wf-414-A2 heuristic fix: _dataset_build_present MUST return "
        "True for the Wf-406-A2 initial-value pattern "
        "(`lambda_cfm_mean_weight = 0.0` + Wf-406-A2 marker + "
        "`build_lambda_reward_dataset` factory reference). The "
        "Wf-410-A2 detection must still work post-Wf-414-A2 update."
    )

    # ---- Check 4: end-to-end audit report has all 4 contracts PASS
    # + empty patch_actions (canonical fix-the-FALSE-NEGATIVE success
    # criterion) ----
    audit = audit_lambda_cfm_state(trainer_path=DEFAULT_TRAINER_PATH)
    # All 3 CLI flags MUST be present (sanity check on the trainer
    # source — these are the canonical Wf-329-W6 / Wf-394-A1 /
    # Wf-406-A2 / Wf-414-A2 flags that the audit reports on).
    assert audit.flags_present == {
        "--lambda-cfm-coupling": True,
        "--lambda-cfm-coupling-weight": True,
        "--lambda-cfm-coupling-samples": True,
    }, (
        f"Wf-414-A2 audit sanity: all 3 --lambda-cfm-coupling* flags "
        f"MUST be present in HEAD; got flags_present="
        f"{audit.flags_present}"
    )
    # Loss-multiplier MUST be True (the Wf-410-A2 fix + Wf-414-A2
    # additional patterns preserve the TRUE).
    assert audit.loss_multiplier_present is True, (
        f"Wf-414-A2 audit fix: loss_multiplier_present MUST be True "
        f"(post-Wf-410-A2 + Wf-414-A2 heuristics); got "
        f"{audit.loss_multiplier_present}."
    )
    assert audit.dataset_build_present is True, (
        f"Wf-414-A2 audit fix: dataset_build_present MUST be True "
        f"(post-Wf-410-A2 heuristic); got {audit.dataset_build_present}."
    )
    assert audit.lambda_reward_factory_ok is True, (
        f"Wf-414-A2 audit fix: lambda_reward_factory_ok MUST be True "
        f"(the build_lambda_reward_dataset factory chain is "
        f"importable in this venv); got {audit.lambda_reward_factory_ok}."
    )
    # The patch_actions list MUST be empty (no patch needed because
    # the Wf-406-A2 patch is already in HEAD).
    assert audit.patch_actions == [], (
        f"Wf-414-A2 audit fix: patch_actions MUST be empty (the "
        f"Wf-406-A2 patch is already applied); got patch_actions="
        f"{audit.patch_actions}. An empty patch_actions list is the "
        f"canonical fix-the-FALSE-NEGATIVE success criterion."
    )
    # The audit.ok flag MUST be True (the canonical "all 4 contracts
    # PASS" success criterion per the Wf-414-A2 task brief).
    assert audit.ok is True, (
        f"Wf-414-A2 audit fix: audit.ok MUST be True (all 4 contracts "
        f"PASS per the task brief). Got audit.ok={audit.ok}. "
        f"flags_present={audit.flags_present}, "
        f"loss_multiplier_present={audit.loss_multiplier_present}, "
        f"dataset_build_present={audit.dataset_build_present}, "
        f"lambda_reward_factory_ok={audit.lambda_reward_factory_ok}."
    )


def test_heuristic_rejects_unpatched() -> None:
    """Wf-414-A2 — line-based grep + new loss-multiplier patterns MUST
    NOT detect the Lambda × CFM coupling in a SYNTHETIC unpatched
    trainer source.

    This is the canonical "regression-of-correctness" guard: the
    line-based grep + new hypothetical kwarg patterns are ADDITIONAL
    acceptance criteria, but they MUST NOT introduce FALSE-POSITIVES
    on a trainer that does not contain ANY Lambda × CFM coupling
    patterns.

    The synthetic unpatched source uses a realistic-looking CFM
    trainer shape (argparse + train_one_epoch + main) with NONE of
    the Lambda × CFM coupling patterns:
    * No ``--lambda-cfm-coupling*`` flag in any form (quoted /
      help-text / docstring / comment / f-string).
    * No ``_lambda_cfm_coupling_flag`` boolean kwarg.
    * No ``lambda_cfm_multiplier`` variable.
    * No ``build_lambda_reward_dataset`` factory reference.

    Asserts
    -------
    1. ``_flags_present(unpatched_src)`` returns all 3 flags False
       (the line-based grep MUST NOT introduce FALSE-POSITIVES).
    2. ``_loss_multiplier_present(unpatched_src)`` returns False.
    3. ``_dataset_build_present(unpatched_src)`` returns False.
    """
    unpatched_src = '''\
#!/usr/bin/env python3
"""Synthetic unpatched CFM trainer (Wf-414-A2 regression test).

This trainer has NONE of the Lambda × CFM coupling patterns. The
audit heuristic MUST correctly report ``False`` for all 3 CLI
flags + loss-multiplier + dataset-build contracts.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    # Generic CFM flags only — no Lambda coupling patterns.
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for training",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="training device (cuda or cpu)",
    )
    return p.parse_args()


def train_one_epoch(loss, *, learning_rate: float = 1e-4):
    # Generic training loop — no Lambda coupling.
    loss = loss * (1.0 + learning_rate)
    return loss


def main():
    args = parse_args()
    loss = 1.0
    for _ in range(args.epochs):
        loss = train_one_epoch(loss, learning_rate=args.batch_size * 1e-5)
    print(f"final loss = {loss:.6f}")


if __name__ == "__main__":
    main()
'''

    # ---- Sanity: the unpatched source has no Lambda × CFM patterns ----
    assert "--lambda-cfm-coupling" not in unpatched_src, (
        "synthetic unpatched source must not contain any "
        "--lambda-cfm-coupling* flag (test setup error)"
    )
    assert "build_lambda_reward_dataset" not in unpatched_src, (
        "synthetic unpatched source must not contain the "
        "LambdaRewardDataset factory reference (test setup error)"
    )
    assert "_lambda_cfm_coupling_flag" not in unpatched_src, (
        "synthetic unpatched source must not contain the "
        "_lambda_cfm_coupling_flag boolean kwarg (test setup error)"
    )
    assert "lambda_cfm_multiplier" not in unpatched_src, (
        "synthetic unpatched source must not contain the "
        "lambda_cfm_multiplier variable (test setup error)"
    )

    # ---- Check 1: line-based grep MUST return all 3 flags False ----
    flags = _flags_present(unpatched_src)
    assert flags == {
        "--lambda-cfm-coupling": False,
        "--lambda-cfm-coupling-weight": False,
        "--lambda-cfm-coupling-samples": False,
    }, (
        f"Wf-414-A2 regression guard: line-based grep MUST NOT "
        f"introduce FALSE-POSITIVES on the synthetic unpatched "
        f"trainer. Got flags_present={flags}. If this fails, the "
        f"line-based grep regex is over-matching (e.g. matching "
        f"unrelated content like comment words or generic tokens)."
    )

    # ---- Check 2: loss-multiplier heuristic MUST return False ----
    assert _loss_multiplier_present(unpatched_src) is False, (
        "Wf-414-A2 regression guard: _loss_multiplier_present MUST "
        "return False for the synthetic unpatched trainer (no "
        "Lambda × CFM coupling patterns). If this fails, the new "
        "Wf-414-A2 hypothetical kwarg patterns "
        "(`_lambda_cfm_coupling_flag` + `lambda_cfm_multiplier`) are "
        "over-matching on generic Python code."
    )

    # ---- Check 3: dataset-build heuristic MUST return False ----
    assert _dataset_build_present(unpatched_src) is False, (
        "Wf-414-A2 regression guard: _dataset_build_present MUST "
        "return False for the synthetic unpatched trainer. If this "
        "fails, the dataset-build heuristic is matching generic code."
    )


# ---------------------------------------------------------------------------
# Wf-414-A2 expansion — Patterns (iv) AST-rewrite + (v) eval/exec
# ---------------------------------------------------------------------------
SYNTHETIC_ADV_SRC = '''\\
#!/usr/bin/env python3
"""Synthetic adversarial trainer — exercises Wf-414-A2 Patterns (iv) + (v).

Designed to DEFEAT the original Wf-414-A2 line-based grep by combining:

* Unicode-escape flag name for the weight flag. The source contains
  the literal 6-char escape sequence (backslash + u002d + u002d) for
  the two leading dashes, NOT the dashes themselves. The line-based
  grep sees the escape sequence and misses. ast.parse resolves the
  Unicode escape at parse time so the corresponding ast.Constant
  value contains the resolved dashes — Pattern (iv) catches via AST
  walk.
* eval() + chr() split for the bare coupling flag. Neither line-grep
  nor the Wf-414-A2 line-based grep can find the full bare flag name
  as a contiguous substring on any line (the dashes come from chr(45)
  at runtime; the suffix is in a separate string literal). Pattern
  (v) eval/exec detection catches via the eval+substring heuristic.
* Line-grep-direct substring for the samples flag on a literal
  assignment line. Line-grep catches it directly.

This trainer verifies that the Wf-414-A2 expansion correctly catches
flags that the Wf-414-A2 original heuristic would miss.
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser()

    # Pattern (v) target (bare flag): eval() + chr() split. The bare
    # flag substring cannot be found as a contiguous literal on any
    # line (the leading two dashes come from chr(45) which is a
    # function call, NOT a string constant; the suffix lives in a
    # separate string literal). Line-grep misses. Pattern (iv) AST
    # walk misses (no ast.Constant value contains the bare flag name
    # as a contiguous substring — only ``--`` and ``lambda-cfm-coupling``
    # separately). Pattern (v) catches via eval detection +
    # flag-suffix-in-source heuristic.
    flag_coupling_v4 = eval(chr(45) + chr(45) + "lambda-cfm-coupling")

    # Pattern (iv) target: the weight flag is hidden via Unicode escape.
    # The Python source on this line contains the literal 6-char
    # escape sequence (backslash + u002d + u002d) for the two leading
    # dashes — line-grep cannot match the regex ``--lambda-cfm-coupling-weight``
    # against this source. ast.parse resolves the Unicode escape at
    # parse time so the corresponding ast.Constant value contains the
    # resolved dashes and Pattern (iv) catches the flag via AST walk.
    flag_weight_v4 = eval("\\u002d\\u002dlambda-cfm-coupling-weight")

    # Line-grep target: the samples flag is literally on the line.
    flag_samples_v4 = "--lambda-cfm-coupling-samples"

    p.add_argument(flag_coupling_v4, action="store_true")
    p.add_argument(flag_weight_v4, type=float, default=0.1)
    p.add_argument(flag_samples_v4, type=int, default=200)
    return p.parse_args()


def train_one_epoch(loss):
    return loss


def main():
    args = parse_args()
    print(f"args = {args}")


if __name__ == "__main__":
    main()
'''


def test_heuristic_catches_ast_rewritten_trainer() -> None:
    """Wf-414-A2 expansion — Patterns (iv) AST-rewrite + (v) eval/exec
    MUST detect the synthetic adversarial trainer that defeats the
    Wf-414-A2 line-based grep alone.

    The synthetic trainer at module-level ``SYNTHETIC_ADV_SRC`` is
    designed so that:

    * The bare flag ``--lambda-cfm-coupling`` is hidden via Unicode
      escape (``\\u002d`` in source). ``ast.parse`` resolves the
      Unicode escape so ``ast.Constant.value`` contains ``--`` —
      Pattern (iv) catches via AST walk.
    * ``--lambda-cfm-coupling-weight`` is embedded in an ``eval()``
      argument string — line-grep catches it directly AND Pattern
      (v) catches via the eval/exec fallback.
    * ``--lambda-cfm-coupling-samples`` is literally on a line —
      line-grep catches it directly.

    Asserts
    -------
    1. **Sanity check** — the synthetic trainer source has NO
       ``--lambda-cfm-coupling`` contiguous substring on any line.
       If this fails, the synthetic trainer setup is broken (line-grep
       alone would suffice and we wouldn't be exercising Pattern (iv)).
    2. **Sanity check** — the synthetic trainer source contains an
       ``eval(`` call (Pattern v target).
    3. **Sanity check** — the synthetic trainer source parses cleanly
       via ``ast.parse`` (no SyntaxError). This is required for both
       Pattern (iv) AST walk to work.
    4. **Sanity check** — the synthetic trainer source has a
       ``ast.Constant`` node whose ``value`` contains
       ``--lambda-cfm-coupling`` (proving Pattern (iv) AST walk has
       a target to find).
    5. **Main check** — ``_flags_present(SYNTHETIC_ADV_SRC)`` returns
       all 3 flags as True (canonical "patterns (iv) + (v) work"
       success criterion).
    """
    # ---- Sanity 1: line-grep MUST miss the bare flag ----
    # The bare flag should NOT appear as a contiguous substring on
    # ANY line (otherwise line-grep alone would suffice and we would
    # not be exercising Pattern (iv) on the weight flag).
    _bare_re = re.compile(r"--lambda-cfm-coupling(?!-weight|-samples)")
    _weight_re = re.compile(r"--lambda-cfm-coupling-weight")
    bare_flag_line_caught = any(
        _bare_re.search(line) for line in SYNTHETIC_ADV_SRC.splitlines()
    )
    assert not bare_flag_line_caught, (
        "Synthetic trainer setup error: the bare flag should NOT "
        "appear as a contiguous substring on any line (line-grep "
        "would catch it directly). If this assertion fails, the "
        "Unicode escape pattern is broken (likely the test source "
        "is missing the raw-string ``r'''...'''`` wrapping)."
    )

    # ---- Sanity 2: eval( is present (Pattern v target) ----
    assert re.search(r"\beval\s*\(", SYNTHETIC_ADV_SRC), (
        "Synthetic trainer setup error: ``eval(`` should be present "
        "to exercise Pattern (v) eval/exec detection. If this fails, "
        "the synthetic trainer source is missing the eval() call."
    )

    # ---- Sanity 3: ast.parse SUCCEEDS (no SyntaxError) ----
    # Pattern (iv) AST walk requires the source to be valid Python.
    try:
        ast.parse(SYNTHETIC_ADV_SRC)
    except SyntaxError as exc:
        raise AssertionError(
            f"Synthetic trainer setup error: ast.parse failed with "
            f"SyntaxError={exc}. The synthetic trainer source must "
            f"be valid Python for Pattern (iv) AST walk to work."
        )

    # ---- Sanity 4: ast.Constant node contains resolved "--" for the
    # weight flag (this is what Pattern (iv) matches against) ----
    # The Unicode escape ``-`` in source must resolve to ``-``
    # in the parsed AST Constant value (Pattern (iv) target).
    tree = ast.parse(SYNTHETIC_ADV_SRC)
    found_in_ast = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _weight_re.search(node.value)
        ):
            found_in_ast = True
            break
    assert found_in_ast, (
        "Synthetic trainer setup error: the Unicode-escaped weight "
        "flag does NOT resolve to ``--lambda-cfm-coupling-weight`` "
        "in any ast.Constant value. This means the test source is "
        "missing the ``r'''...'''`` raw-string wrapping OR the "
        "Unicode escape pattern is broken. Without this, Pattern "
        "(iv) AST walk has nothing to match."
    )

    # ---- Main check: patterns (iv) + (v) detect all 3 flags ----
    flags = _flags_present(SYNTHETIC_ADV_SRC)
    assert flags == {
        "--lambda-cfm-coupling": True,
        "--lambda-cfm-coupling-weight": True,
        "--lambda-cfm-coupling-samples": True,
    }, (
        f"Wf-414-A2 expansion: patterns (iv) AST-rewrite + "
        f"(v) eval/exec MUST catch all 3 flags in the synthetic "
        f"adversarial trainer. Got flags_present={flags}. The "
        f"bare flag was hidden via chr() split (line-grep misses) — "
        f"only Pattern (v) eval+substring heuristic catches it. The "
        f"weight flag is hidden via Unicode escape — only Pattern "
        f"(iv) AST walk catches it (after Unicode resolution). The "
        f"samples flag is literally on a line — line-grep catches "
        f"it directly. If this assertion fails, either Pattern (iv) "
        f"AST walk or Pattern (v) eval/exec detection has regressed."
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "--tb=short"])
    )
