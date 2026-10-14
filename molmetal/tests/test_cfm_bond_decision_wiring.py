"""The decode path and the metric path must obey the SAME decision rule.

2026-10-05.  ``configs/cfm_tmqm_all6.yaml`` declares::

    flow:
      bond_decision: threshold
      bond_decision_threshold: 0.5

That reached ``evaluate_bond_accuracy`` (which calls the trainer's
``_bond_decision``) but never reached the DECODER:
``YuelBondBondDecoder._select_bonds`` hardcoded
``torch.argmax(logits, dim=-1)``, and ``__init__`` had no ``decision``
parameter to configure.  So the reported bond F1 / exact-type and the
molecules actually assembled into an RDKit ``Mol`` came from two different
rules.

MEASURED, and this is the part that matters for honesty: on the epoch-9
checkpoint over real tmQM geometry the two rules agree PAIR-FOR-PAIR
(624/2141 bonded under argmax, 0.5 and 0.3 alike).  So this gap did not
produce the fragmented decodes — it is a latent coupling bug, invisible in
the numbers only for as long as the two rules happen to coincide.  The test
below therefore pins the WIRING, not a number.

Two definitions of the rule also existed (one in the trainer, one now in the
library).  That is the same duplication that let the YuelBond forward drift
apart from its monkey-patch, so the shared function is asserted to be the
same object on both paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECODER = (
    PROJECT_ROOT
    / "molmetal"
    / "molmetal_lam"
    / "lam_chem"
    / "cfm_decoder_yuelbond.py"
)
LIB = (
    PROJECT_ROOT
    / "molmetal"
    / "molmetal_lam"
    / "lam_chem"
    / "yuelbond_decoder.py"
)
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lib():
    return _load(LIB, "_t_decw_lib")


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------
def test_threshold_decouples_existence_from_type(lib):
    """`threshold` decides EXISTENCE by P(bonded), TYPE by argmax over the rest.

    MEASURED on the 64-cloud tmQM cohort the two rules differ by 0.45 F1 on
    the SAME logits (argmax 0.079 vs threshold 0.570 unweighted at 4 epochs),
    and class weighting swung argmax to the other failure mode (recall 0.97 /
    precision 0.32). Four of five classes are overwhelmingly NO_BOND, so the
    argmax boundary sits nowhere near a useful operating point.
    """
    # 5 classes, strongly favouring NO_BOND but with a real bonded minority.
    logits = torch.tensor(
        [
            [6.0, 0.0, 0.0, 0.0, 0.0],  # clearly no bond
            [0.0, 6.0, 0.0, 0.0, 0.0],  # clearly single
            [1.0, 2.0, 3.0, 0.0, 0.0],  # existence marginal, type=double
            [2.0, 0.5, 0.0, 4.0, 0.0],  # existence marginal, type=triple
        ]
    )
    out = lib.bond_decision(logits, "threshold", 0.5)
    assert out[0] == 0, "an overwhelming NO_BOND must stay NO_BOND"
    assert out[1] == 1, "an overwhelming SINGLE must be a bond"
    # The point of the rule: at a low enough threshold these become bonds, and
    # the TYPE comes from the argmax over the non-zero columns, not from a
    # plain argmax over all five (which would have said NO_BOND).
    low = lib.bond_decision(logits, "threshold", 0.10)
    assert low[2] == 2, "type must be DOUBLE from the non-zero argmax"
    assert low[3] == 3, "type must be TRIPLE from the non-zero argmax"


def test_argmax_is_unchanged(lib):
    logits = torch.tensor([[6.0, 0, 0, 0, 0], [0, 6.0, 0, 0, 0]])
    assert lib.bond_decision(logits, "argmax", 0.5) == [0, 1]


def test_an_unknown_rule_raises_rather_than_silently_falling_back(lib):
    """A typo must not decode with argmax and look fine."""
    logits = torch.tensor([[0.0, 6.0, 0, 0, 0]])
    with pytest.raises(ValueError, match="unknown bond decision rule"):
        lib.bond_decision(logits, "treshold", 0.5)


# ---------------------------------------------------------------------------
# The wiring — this is the regression that was actually fixed
# ---------------------------------------------------------------------------
def test_decoder_accepts_and_stores_the_decision_rule():
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1)
    d = YuelBondBondDecoder(
        head=head, decision="threshold", decision_threshold=0.3
    )
    assert d.decision == "threshold"
    assert d.decision_threshold == pytest.approx(0.3)


def test_decoder_rejects_a_bad_rule_at_construction():
    """Construction time, not first-decode time.

    A yaml typo that only surfaces when a molecule comes out wrong is found
    days later, or never.
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    with pytest.raises(ValueError, match="unknown bond decision rule"):
        YuelBondBondDecoder(head=YuelBondDecoderHead(hidden_dim=8, n_layers=1),
                            decision="treshold")


def test_decoder_selector_is_not_hardcoded_argmax():
    """The source must route through the shared rule.

    A behavioural test would pass whenever the two rules coincide — which is
    the case at the current operating point — so this pins the STRUCTURE.
    """
    src = DECODER.read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "_bond_decision(" in code, (
        "_select_bonds must call the shared decision rule, not "
        "torch.argmax(logits, dim=-1)"
    )
    assert "torch.argmax(logits, dim=-1)" not in code, (
        "a hardcoded argmax is back in the selector; the configured rule "
        "(flow.bond_decision) would not reach the decoder"
    )


def test_trainer_passes_the_configured_rule_to_the_decoder():
    """The trainer must forward --bond-decision to YuelBondBondDecoder.

    Wf-A35-1 (2026-10-14) C3 routed the read through a single helper
    (:func:`molmetal.eval.bond_decision.bond_decision_settings`) instead
    of the prior inline ``getattr(args, ...)`` chain — the contract
    here (yaml setting reaches the decoder) is preserved, only the
    literal pattern changed.
    """
    src = TRAINER.read_text()
    assert "_bond_decision_settings(vars(args))" in src, (
        "the trainer must read (decision, threshold, class_logit_bias) "
        "via the C3 helper; otherwise the yaml setting only reaches the metrics"
    )
    assert "decision=_dec_decision," in src, (
        "the trainer must forward the helper's `decision` to "
        "YuelBondBondDecoder; otherwise the yaml setting only reaches the metrics"
    )
    assert "decision_threshold=_dec_threshold," in src, (
        "the trainer must forward the helper's `decision_threshold` to "
        "YuelBondBondDecoder; otherwise the yaml setting only reaches the metrics"
    )


def test_metric_and_decode_share_one_definition():
    """Two copies of the rule is how the forward drifted apart earlier today.

    Compared by identity on the module the trainer ACTUALLY imported, rather
    than against a second hand-loaded copy of the library: loading a module
    twice under two names yields two distinct function objects that behave
    identically, so an identity check against a private copy would fail for a
    reason that has nothing to do with the wiring.
    """
    import molmetal.molmetal_lam.lam_chem.yuelbond_decoder as libmod
    import molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond as decmod

    tmod = _load(TRAINER, "_t_decw_trainer")
    assert tmod._bond_decision is libmod.bond_decision, (
        "the trainer must import the library's rule, not define its own"
    )
    # The decoder module imports the same object under a private alias.
    assert decmod._bond_decision is libmod.bond_decision, (
        "the decoder must import the library's rule, not define its own"
    )


def test_trainer_no_longer_defines_its_own_copy():
    src = TRAINER.read_text()
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "\ndef _bond_decision(" not in code, (
        "the trainer still defines its own _bond_decision"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
