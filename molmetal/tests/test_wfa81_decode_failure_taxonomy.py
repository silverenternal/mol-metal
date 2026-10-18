"""Wf-A81 (2026-10-18) — the decode failure taxonomy must be ABLE to fail.

Why this file exists
--------------------
``_rdkit_strict_reason`` tested ``not smi`` before ``not sanitized``, and
``cfm_decoder_yuelbond`` sets ``smiles=""`` in the very ``except`` branch
that sets ``sanitized=False``.  So the only failure tuple that could ever
occur was ``("", False)``, and every failure — a kekulizer refusal, an
assembly error, a head that produced no edges — was counted as
``"no_smiles"``.  MEASURED on results/cfm_5metal_wfa75a_w000/events.jsonl:
the ep-3/ep-4 gen_validity collapse is recorded as ``{"no_smiles": 51}``.

These tests pin the buckets apart, using the exact ``error`` strings the
decoder writes, so a future refactor cannot re-collapse them.
"""

import importlib.util
import pathlib
import sys
import unittest

_TRAINER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "train_cfm_yuelbond_wasserstein.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_wfa81_trainer", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves annotations through sys.modules[cls.__module__],
    # so the module must be registered BEFORE exec_module runs.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


class SanitizeFailureIsNotNoSmiles(unittest.TestCase):
    def test_unsanitized_empty_smiles_is_a_sanitize_failure(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", False), "sanitize_failed"
        )

    def test_no_smiles_bucket_still_exists(self) -> None:
        # sanitized=True with nothing to show is the only route to it.
        self.assertEqual(M._rdkit_strict_reason("", True), "no_smiles")

    def test_ok_is_still_none(self) -> None:
        self.assertIsNone(M._rdkit_strict_reason("CCO", True, ""))
        self.assertIsNone(M._rdkit_strict_reason("CCO", True))


class DecoderErrorIsUsed(unittest.TestCase):
    def test_empty_decode_is_reachable(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", False, "yuelbond:no_candidate_edges"),
            "empty_decode",
        )

    def test_assembly_error_is_reachable(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", False, "RWMol.GetMol failed: boom"),
            "assembly_error",
        )

    def test_sanitize_failure_carries_the_exception_class(self) -> None:
        # The exact string _assemble_mol writes: f"{type(exc).__name__}: {exc}"
        self.assertEqual(
            M._rdkit_strict_reason(
                "",
                False,
                "KekulizeException: Can't kekulize mol.  Unkekulized atoms: 0",
            ),
            "sanitize_failed:KekulizeException",
        )

    def test_atom_kekulize_class_is_kept_apart(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason(
                "",
                False,
                "AtomKekulizeException: non-ring atom 3 marked aromatic",
            ),
            "sanitize_failed:AtomKekulizeException",
        )

    def test_error_without_a_colon_does_not_crash(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", False, "weird"),
            "sanitize_failed:weird",
        )

    def test_sanitized_success_ignores_a_stale_error_string(self) -> None:
        # A caller that passes a leftover error must not be able to turn a
        # good molecule into a failure.
        self.assertIsNone(
            M._rdkit_strict_reason("c1ccccc1", True, "KekulizeException: x")
        )

    def test_round_trip_failures_are_unchanged(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("C(((", True), "reparse_failed"
        )


class BucketsAreDisjoint(unittest.TestCase):
    def test_five_decoder_outcomes_land_in_five_buckets(self) -> None:
        cases = [
            ("CCO", True, "", "ok"),
            ("", False, "yuelbond:cutoff", "empty_decode"),
            ("", False, "RWMol.GetMol failed: x", "assembly_error"),
            ("", False, "KekulizeException: x", "sanitize_failed:KekulizeException"),
            ("", False, "", "sanitize_failed"),
        ]
        seen = []
        for smi, san, err, expect in cases:
            got = M._rdkit_strict_reason(smi, san, err) or "ok"
            self.assertEqual(got, expect)
            seen.append(got)
        self.assertEqual(len(set(seen)), len(seen))


if __name__ == "__main__":
    unittest.main(verbosity=2)
