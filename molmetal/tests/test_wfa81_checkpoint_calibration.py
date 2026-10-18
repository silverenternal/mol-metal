"""Wf-A81 (2026-10-18) — the checkpoint must carry the rule it was decoded with.

``args.bond_decision`` in a checkpoint says "per_class" and carries no
thresholds, so a reader that rebuilt a decoder from the .pt fell back to
argmax: the weights were read with a different rule than the one that
produced the run's own gen_validity.  These tests pin the stamp and its
reader, including the cases where the stamp must be IGNORED rather than
half-applied.
"""

import unittest

from molmetal.eval.bond_threshold_calibration import (
    N_CLASSES,
    calibration_from_decoder,
    decoder_kwargs_from_checkpoint,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
)


def _decoder(**kw) -> YuelBondBondDecoder:
    # head=None is enough: the constructor resolves the rule and the
    # thresholds, and no forward pass is run here.
    return YuelBondBondDecoder(head=None, hidden_dim=8, n_layers=1, **kw)


class StampTest(unittest.TestCase):
    def test_stamp_records_the_resolved_rule(self) -> None:
        t = [0.3, 0.2, 0.1, 0.15, 0.25]
        cal = calibration_from_decoder(
            _decoder(decision="per_class", per_class_thresholds=t), epoch=7
        )
        self.assertEqual(cal["decision"], "per_class")
        self.assertEqual(cal["per_class_thresholds"], t)
        self.assertEqual(cal["source_epoch"], 7)

    def test_stamp_of_a_plain_decoder_has_no_thresholds(self) -> None:
        cal = calibration_from_decoder(_decoder())
        self.assertEqual(cal["decision"], "argmax")
        self.assertIsNone(cal["per_class_thresholds"])
        self.assertEqual(cal["decision_threshold"], 0.5)
        self.assertNotIn("source_epoch", cal)

    def test_stamp_is_json_serialisable(self) -> None:
        import json

        cal = calibration_from_decoder(
            _decoder(decision="per_class", per_class_thresholds=[0.5] * 5)
        )
        json.dumps(cal)


class ReadBackTest(unittest.TestCase):
    def test_round_trip_per_class(self) -> None:
        t = [0.3, 0.2, 0.1, 0.15, 0.25]
        cal = calibration_from_decoder(
            _decoder(decision="per_class", per_class_thresholds=t)
        )
        kw = decoder_kwargs_from_checkpoint({"decode_calibration": cal})
        self.assertEqual(kw["decision"], "per_class")
        self.assertEqual(kw["per_class_thresholds"], t)
        rebuilt = _decoder(**kw)
        self.assertEqual(rebuilt.decision, "per_class")
        self.assertEqual(rebuilt.per_class_thresholds, t)

    def test_round_trip_threshold_rule(self) -> None:
        cal = calibration_from_decoder(
            _decoder(decision="threshold", decision_threshold=0.35)
        )
        kw = decoder_kwargs_from_checkpoint({"decode_calibration": cal})
        self.assertEqual(kw["decision"], "threshold")
        self.assertAlmostEqual(kw["decision_threshold"], 0.35)

    def test_old_checkpoint_without_a_stamp_is_untouched(self) -> None:
        self.assertEqual(decoder_kwargs_from_checkpoint({}), {})
        self.assertEqual(decoder_kwargs_from_checkpoint(None), {})
        self.assertEqual(
            decoder_kwargs_from_checkpoint({"args": {"bond_decision": "per_class"}}),
            {},
        )

    def test_per_class_without_thresholds_is_refused(self) -> None:
        # A half-applied stamp would build a decoder that raises on the
        # first decode; falling back to defaults is the only safe read.
        for t in (None, [], [0.5, 0.5], [0.5] * (N_CLASSES + 1)):
            self.assertEqual(
                decoder_kwargs_from_checkpoint(
                    {"decode_calibration": {"decision": "per_class",
                                            "per_class_thresholds": t}}
                ),
                {},
                msg=f"thresholds={t!r}",
            )

    def test_unknown_decision_is_refused(self) -> None:
        self.assertEqual(
            decoder_kwargs_from_checkpoint(
                {"decode_calibration": {"decision": "softmax_topk"}}
            ),
            {},
        )

    def test_logit_bias_rides_along(self) -> None:
        cal = {
            "decision": "per_class",
            "per_class_thresholds": [0.5] * N_CLASSES,
            "class_logit_bias": [0.0, 0.0, 0.0, 0.0, -0.7],
        }
        kw = decoder_kwargs_from_checkpoint({"decode_calibration": cal})
        self.assertEqual(kw["class_logit_bias"], [0.0, 0.0, 0.0, 0.0, -0.7])


if __name__ == "__main__":
    unittest.main(verbosity=2)
