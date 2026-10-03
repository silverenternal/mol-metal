"""Wf-394-A1 — TODO-21 Lambda × CFM coupling wire-in tests (CPU-only).

================================================================
Background
================================================================
Wf-329-W6 (2026-09-29) shipped the original wiring of the
Lambda × CFM coupling (3 CLI flags + 1-line loss multiplier +
dataset build block) in
``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``. Wf-331-R6
+ Wf-333-G1 + Wf-333-G2 confirmed the wiring and added a GPU
wrapper scaffold.

Subsequent waves (Wf-388-A2 + Wf-388-A3, 2026-09-30) added new
diversity + SA-penalty flags around the loss block. In the working
tree the Wf-329-W6 wiring shape is still the canonical contract;
however, its source location in HEAD has shifted (the file no
longer contains the ``--lambda-cfm-coupling*`` flags verbatim).

Wf-394-A1 re-establishes the canonical contract via an audit +
smoke scaffold (:mod:`molmetal.scripts.wf394a1_lambda_cfm_wire`)
and verifies the chain with the 2 NEW tests below.

Honest framing
==============
* The Lambda × CFM joint retrain itself remains BLOCKED on
  upstream ``decode_ratio > 0`` per Wf-291D + Wf-299. These tests
  validate the contract surface, not the training loop.
* 0 GPU launches performed in Wf-394-A1.
* 0 ``paper/*`` touches.
* 0 wet-lab planning (per AGENTS.md §10).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf394a1_lambda_cfm_wire import (  # noqa: E402
    DEFAULT_TRAINER_PATH,
    audit_lambda_cfm_state,
    smoke_lambda_cfm,
)


def test_lambda_cfm_flag() -> None:
    """The canonical Wf-329-W6 / Wf-394-A1 flags MUST be detectable.

    Even when the trainer HEAD has drifted, the audit must
    correctly report whether each canonical flag is currently
    present (so a future USER_OK-gated patch wave knows exactly
    which flags to inject).
    """
    audit = audit_lambda_cfm_state(trainer_path=DEFAULT_TRAINER_PATH)
    # The audit object MUST always expose the three canonical flags
    # — whether or not they happen to be present in HEAD. The test
    # asserts the *shape* of the report, not the boolean content
    # (the latter is the Wf-388-A3 drift signal we want to
    # surface honestly).
    assert set(audit.flags_present.keys()) == {
        "--lambda-cfm-coupling",
        "--lambda-cfm-coupling-weight",
        "--lambda-cfm-coupling-samples",
    }
    # Each value is a bool (not None / not string).
    for value in audit.flags_present.values():
        assert isinstance(
            value, bool
        ), f"flags_present value must be bool; got {type(value).__name__}"
    # The audit must also report on the loss-multiplier + dataset-
    # build blocks (both boolean).
    assert isinstance(audit.loss_multiplier_present, bool)
    assert isinstance(audit.dataset_build_present, bool)
    # The factory chain must be importable; if it is not, the
    # Wf-323E + Wf-329-W6 wiring chain has regressed and must be
    # re-established before any patch can be applied.
    assert audit.lambda_reward_factory_ok, (
        "LambdaRewardDataset factory chain import FAILED — "
        "Wf-394-A1 cannot safely wire CLI flags until the "
        "factory is restored."
    )


def test_lambda_cfm_integration() -> None:
    """The coupling chain end-to-end: audit + smoke must agree.

    The smoke test runs the canonical Wf-329-W6 chain:

    * import ``molmetal.molmetal_lam.reward.lambda_reward_dataset``
    * verify ``build_lambda_reward_dataset`` is callable
    * compute the canonical loss-multiplier upper bound (1 + w *
      1.0 = 1 + 0.1 = 1.1 for the default coupling_strength=0.1)

    The test asserts the smoke's import chain is intact AND that
    the audit's factory chain import-ok flag agrees with the
    smoke's factory_callable flag (drift between the two surfaces
    a regression that must be investigated before any wiring
    patch is applied).
    """
    audit = audit_lambda_cfm_state(trainer_path=DEFAULT_TRAINER_PATH)
    smoke = smoke_lambda_cfm(coupling_strength=0.1)

    # The smoke must succeed iff the audit reports the factory chain
    # importable (no drift between the two surfaces).
    assert smoke["ok"] == audit.lambda_reward_factory_ok, (
        "smoke.ok and audit.lambda_reward_factory_ok must agree; "
        "drift indicates a regression in the "
        "LambdaRewardDataset import chain."
    )
    # The import chain must include the canonical symbols.
    chain_text = "\n".join(smoke["import_chain"])
    assert "molmetal" in chain_text, f"import chain missing 'molmetal': {chain_text!r}"
    assert (
        "build_lambda_reward_dataset" in chain_text
    ), f"import chain missing canonical factory symbol: {chain_text!r}"
    # The loss-multiplier upper bound must equal 1 + 0.1 = 1.1
    # (canonical Wf-329-W6 default coupling_strength=0.1).
    assert smoke["loss_multiplier_upper_bound"] == pytest.approx(1.1), (
        f"loss_multiplier_upper_bound should be 1.1 (1 + 0.1 * 1); "
        f"got {smoke['loss_multiplier_upper_bound']}"
    )
    # The audit must include at least one note flagging the GPU
    # launch blocker (honest framing — Wf-291D + Wf-299 gate).
    notes_text = "\n".join(audit.notes)
    assert "decode_ratio" in notes_text, (
        f"audit.notes must mention the GPU launch blocker; " f"got {notes_text!r}"
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [__file__, "-v", "--tb=short"],
        )
    )
