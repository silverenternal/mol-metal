"""Wf-410-A2 — TODO-21 Lambda × CFM coupling audit heuristics update tests.

================================================================
Background
================================================================
Wf-394-A1 (2026-09-30) shipped the audit + smoke scaffold for the
TODO-21 Lambda × CFM coupling re-wire (3 CLI flags + dataset-build
block + loss-multiplier block). The original Wf-394-A1 heuristics
were written to detect the OLD Wf-329-W6 / Wf-394-A1 shape:

* ``_loss_multiplier_present`` matched the literal
  ``lambda_reward_dataset.mean_weight`` attribute access OR the
  single-line ``(1.0 + lambda_cfm_coupling_weight`` idiom.
* ``_dataset_build_present`` required the ``Wf-394-A1`` marker
  AND the literal ``lambda_reward_dataset = None`` initial value.

Wf-406-A2 (2026-09-30) re-wired the same contract in HEAD using a
different (but bit-equivalent) shape:

* Loss multiplier spans multiple lines with kwargs ``lambda_cfm_coupling_weight``
  + ``lambda_cfm_mean_weight`` passed into ``train_one_epoch`` —
  the local variable is no longer named ``lambda_reward_dataset``.
* Dataset is bound to ``_lambda_dataset`` and the scalar
  ``mean_weight`` is extracted into ``lambda_cfm_mean_weight`` for
  the kwarg pass-through — the initial value is
  ``lambda_cfm_mean_weight = 0.0``, not
  ``lambda_reward_dataset = None``.

The Wf-394-A1 heuristics produced FALSE-NEGATIVES against HEAD
after the Wf-406-A2 patch (verified by
``molmetal/reports/wf406a2_lambda_cfm_patch/wf394a1_audit.json``
reporting ``loss_multiplier_present: false`` + ``dataset_build_present: false``
despite the patch being physically present).

Wf-410-A2 (this wave) updates the heuristics to detect BOTH the
OLD Wf-329-W6 / Wf-394-A1 shape AND the NEW Wf-406-A2 shape (line-
based grep rather than AST-only per the Wf-410-A2 task brief).

Honest framing
==============
* 0 GPU launches performed in Wf-410-A2.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
* The heuristic update is **backward-compatible**: any future
  patch that re-introduces the OLD Wf-329-W6 shape (e.g. a future
  re-wire that restores the ``lambda_reward_dataset`` local var)
  will still be detected — the Wf-406-A2 NEW shape is added in
  addition to the OLD shape, not replacing it.
"""

from __future__ import annotations

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
    _loss_multiplier_present,
    audit_lambda_cfm_state,
)


def test_heuristic_catches_patched_trainer() -> None:
    """Wf-410-A2 — line-based heuristics MUST detect the Wf-406-A2 patch.

    Pre-Wf-410-A2 the Wf-394-A1 heuristics produced FALSE-NEGATIVES
    on the Wf-406-A2 patch (verified by the
    ``molmetal/reports/wf406a2_lambda_cfm_patch/wf394a1_audit.json``
    report). This test asserts that the updated line-based
    heuristics correctly detect the Wf-406-A2 patch shape in HEAD.

    Asserts
    -------
    1. ``_loss_multiplier_present`` returns ``True`` — the Wf-406-A2
       multi-line idiom ``1.0\\n + float(lambda_cfm_coupling_weight)``
       is detected.
    2. ``_dataset_build_present`` returns ``True`` — the Wf-406-A2
       dataset-build block (Wf-406-A2 marker +
       ``lambda_cfm_mean_weight = 0.0`` initial value +
       ``build_lambda_reward_dataset`` factory reference) is detected.
    3. The end-to-end ``audit_lambda_cfm_state`` report has
       ``patch_actions == []`` (no patch actions needed because the
       patch is already in HEAD) — this is the canonical fix-the-false-
       negative success criterion.
    """
    # ---- Sanity: the trainer file MUST be present ----
    assert TRAINER_PATH.exists(), (
        f"CFM trainer not found at {TRAINER_PATH}; "
        "test cannot exercise the Wf-410-A2 heuristic fix."
    )
    trainer_src = TRAINER_PATH.read_text(encoding="utf-8")

    # ---- Check 1: loss-multiplier heuristic detects the Wf-406-A2
    # multi-line idiom (the canonical regression-of-focus) ----
    assert _loss_multiplier_present(trainer_src), (
        "Wf-410-A2 heuristic fix: _loss_multiplier_present MUST return "
        "True for the Wf-406-A2 multi-line idiom "
        "(`1.0\\n + float(lambda_cfm_coupling_weight)`) — the OLD "
        "Wf-394-A1 heuristic returned False on this shape (FALSE-NEGATIVE). "
        "If this assertion fails, the heuristic fix has regressed."
    )

    # ---- Check 2: dataset-build heuristic detects the Wf-406-A2
    # initial-value pattern ----
    assert _dataset_build_present(trainer_src), (
        "Wf-410-A2 heuristic fix: _dataset_build_present MUST return "
        "True for the Wf-406-A2 initial-value pattern "
        "(`lambda_cfm_mean_weight = 0.0` + Wf-406-A2 marker + "
        "`build_lambda_reward_dataset` factory reference) — the OLD "
        "Wf-394-A1 heuristic returned False on this shape (FALSE-NEGATIVE). "
        "If this assertion fails, the heuristic fix has regressed."
    )

    # ---- Check 3: end-to-end audit report has empty patch_actions
    # when the Wf-406-A2 patch is already applied (canonical fix-the-
    # false-negative success criterion) ----
    audit = audit_lambda_cfm_state(trainer_path=DEFAULT_TRAINER_PATH)
    # All 3 CLI flags MUST be present (sanity check on the trainer
    # source — these are the canonical Wf-329-W6 / Wf-394-A1 / Wf-406-A2
    # flags that the audit reports on).
    assert audit.flags_present == {
        "--lambda-cfm-coupling": True,
        "--lambda-cfm-coupling-weight": True,
        "--lambda-cfm-coupling-samples": True,
    }, (
        f"Wf-410-A2 audit sanity: all 3 --lambda-cfm-coupling* flags "
        f"MUST be present in HEAD; got flags_present="
        f"{audit.flags_present}"
    )
    # Loss-multiplier + dataset-build MUST both be True (this is the
    # canonical Wf-410-A2 fix — previously these were False despite
    # the patch being in HEAD).
    assert audit.loss_multiplier_present is True, (
        f"Wf-410-A2 audit fix: loss_multiplier_present MUST be True "
        f"(post-Wf-410-A2 heuristic); got {audit.loss_multiplier_present}. "
        f"This is the Wf-406-A2 honest-negative FALSE-NEGATIVE that "
        f"Wf-410-A2 fixes."
    )
    assert audit.dataset_build_present is True, (
        f"Wf-410-A2 audit fix: dataset_build_present MUST be True "
        f"(post-Wf-410-A2 heuristic); got {audit.dataset_build_present}. "
        f"This is the Wf-406-A2 honest-negative FALSE-NEGATIVE that "
        f"Wf-410-A2 fixes."
    )
    # The patch_actions list MUST be empty (no patch needed because
    # the Wf-406-A2 patch is already in HEAD).
    assert audit.patch_actions == [], (
        f"Wf-410-A2 audit fix: patch_actions MUST be empty (the "
        f"Wf-406-A2 patch is already applied); got patch_actions="
        f"{audit.patch_actions}. An empty patch_actions list is the "
        f"canonical fix-the-FALSE-NEGATIVE success criterion."
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-v", "--tb=short"])
    )