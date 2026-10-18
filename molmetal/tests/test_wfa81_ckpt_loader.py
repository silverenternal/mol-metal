"""Wf-A81 (2026-10-18) — the panel must load BOTH checkpoint serialisations.

The trainer writes checkpoints as a raw pickle through
`wf386a3_atomic_write.atomic_write_file` (since Wf-A30-1, 2026-10-14).
`cfm_property_panel` read them with `torch.load`, which raises
`RuntimeError: Invalid magic number; corrupt file?` on a raw pickle, and the
helper caught that and returned an empty SMILES list.  MEASURED: the helper
returned [] on `/tmp/wfa81_smoke/ckpt/cfm_yuelbond_wasserstein_seed0.pt`, a
checkpoint the trainer had written seconds earlier.

These tests pin both formats and pin that a genuinely broken file still raises.
"""

import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from molmetal.scripts.cfm_property_panel import _load_checkpoint_any


class BothSerialisations(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.payload = {
            "model_state_dict": {"a": torch.zeros(2)},
            "decode_calibration": {"decision": "per_class",
                                   "per_class_thresholds": [0.5] * 5},
            "epoch": 3,
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_raw_pickle_as_the_trainer_writes_it(self) -> None:
        p = self.dir / "atomic.pt"
        with p.open("wb") as fh:
            pickle.dump(self.payload, fh)
        got = _load_checkpoint_any(p)
        self.assertEqual(got["epoch"], 3)
        self.assertEqual(
            got["decode_calibration"]["decision"], "per_class"
        )
        self.assertTrue(torch.equal(got["model_state_dict"]["a"], torch.zeros(2)))

    def test_torch_save_as_the_legacy_path_writes_it(self) -> None:
        p = self.dir / "legacy.pt"
        torch.save(self.payload, p)
        got = _load_checkpoint_any(p)
        self.assertEqual(got["epoch"], 3)

    def test_truncated_file_still_raises(self) -> None:
        p = self.dir / "truncated.pt"
        with p.open("wb") as fh:
            pickle.dump(self.payload, fh)
        blob = p.read_bytes()
        p.write_bytes(blob[: len(blob) // 2])
        with self.assertRaises(Exception):
            _load_checkpoint_any(p)

    def test_random_bytes_raise_rather_than_return_none(self) -> None:
        p = self.dir / "garbage.pt"
        p.write_bytes(b"not a checkpoint at all")
        with self.assertRaises(Exception):
            _load_checkpoint_any(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
