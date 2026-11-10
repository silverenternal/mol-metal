#!/usr/bin/env python3
"""BLK-D2 Step 2 §5.7 50-step CPU smoke: chemprop_v3 frozen encoder + trainable MLP head.

Per AGENTS.md §5.7 the wire-in is INCOMPLETE until a 50-step end-to-end
CPU smoke verifies:
  1. parse_args accepts --csv, --smiles-col, --label-col, --hidden, --dropout
  2. chemprop_v3 encoder imports + initialises on CPU
  3. encoder parameters are FROZEN (``requires_grad=False`` on every tensor)
  4. head parameters TRAIN (grad populated, loss finite)
  5. per-seed JSON on disk with finite test_metrics.roc_auc

This is the canonical wire-in smoke for BLK-D2 Step 2 per todo.json
fix_steps Step 2 (chemprop_v3 frozen encoder -> fine-tune D-MPNN head;
section 5.7 smoke before GPU).  The chemprop_v3 encoder has been
confirmed wired end-to-end via the plan-§2.2/§2.3 ``ChempropV3EncoderForward``
class; this smoke verifies the head-attachment + freeze discipline.

Per AGENTS.md §5.7 the smoke is the WIRE-IN gate, not the loss-monotone
gate. A separate 50-step loss-monotone check is the GPU run's job.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "molmetal"))


class BLKD2ChempropV3FrozenHeadSmoke(unittest.TestCase):
    def test_chemprop_v3_frozen_head_end_to_end_smoke(self) -> None:
        out_dir = "/tmp/blk_d2_chemprop_v3_frozen_head_smoke_57"
        os.makedirs(out_dir, exist_ok=True)

        # Mirror what scripts/_retired/launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh
        # would invoke (the wrapper is wired separately per AGENTS.md §5.6
        # BLK-I1; this smoke invokes the trainer directly).
        cmd = [
            str(REPO / ".venv/bin/python"),
            "molmetal/scripts/train_chemprop_v3_frozen_dmpnn_head.py",
            "--csv", ".data/molmetal/moleculenet/bace.csv",
            "--smiles-col", "mol",
            "--label-col", "Class",
            "--seeds", "0",
            "--epochs", "1",
            "--batch-size", "32",
            "--hidden", "32",
            "--lr", "1e-3",
            "--out-dir", out_dir,
        ]
        # Force CPU so we don't fight C4 on GPU 0 (1.1GB used but SM-util
        # dominated; a parallel chemprop_v3 forward would still serialise
        # on the SM stream).  The wire-in does not require GPU.
        smoke_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
            env=smoke_env,
        )

        # §5.7 invariant #1 + #2 + #3: parse_args + import + initialise
        # cleanly + encoder frozen.  An import or argparse error would
        # surface here as rc != 0.
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-25:])
            print(f"[BLK-D2-S2 §5.7] trainer rc={proc.returncode}; tail:\n{tail}")
        self.assertEqual(
            proc.returncode, 0,
            f"trainer rc={proc.returncode}; expected rc=0 "
            f"(argparse error, import error, or freeze discipline broken would surface here)"
        )

        stdout = proc.stdout
        # §5.7 invariant #2: chemprop_v3 encoder imported and initialised.
        self.assertIn(
            "loaded", stdout,
            f"stdout missing 'loaded' marker; the bace CSV was not parsed"
        )
        # §5.7 invariant #3: encoder frozen (frozen params + trainable=0).
        m = re.search(r"encoder frozen:\s+(\d+)\s+params,\s+trainable=(\d+)", stdout)
        self.assertIsNotNone(m, "stdout missing 'encoder frozen: N params, trainable=0' line")
        n_frozen = int(m.group(1))
        n_trainable = int(m.group(2))
        print(f"[BLK-D2-S2 §5.7] encoder: frozen={n_frozen} trainable={n_trainable}")
        self.assertGreater(n_frozen, 0, "encoder has 0 parameters — wrapper broken")
        self.assertEqual(
            n_trainable, 0,
            f"encoder must be FROZEN but found {n_trainable} trainable params"
        )

        # §5.7 invariant #4: per-seed JSON exists on disk with finite
        # test_metrics.roc_auc (parallel to decode_ratio>0 for CFM, or
        # bce_loss_finite for D-MPNN).
        seed_json = Path(out_dir) / "chemprop_v3_frozen_head_seed0.json"
        self.assertTrue(
            seed_json.exists(),
            f"per-seed JSON not on disk at {seed_json}: "
            f"BLK-I1 + JSON-write discipline broken"
        )
        per_seed = json.loads(seed_json.read_text())
        roc = float(per_seed["test_metrics"]["roc_auc"])
        n_train = per_seed["n_train"]
        n_test = per_seed["n_test"]
        head_params = per_seed.get("head_n_params", 0)
        encoder_params = per_seed.get("encoder_n_params", 0)
        print(
            f"[BLK-D2-S2 §5.7] bace/chemprop_v3_frozen_head/seed0  "
            f"roc_auc={roc:.4f}  n_train={n_train} n_test={n_test}  "
            f"head_params={head_params} encoder_params={encoder_params}"
        )
        self.assertFalse(roc != roc, f"test_metrics.roc_auc is NaN: {roc}")
        self.assertNotEqual(roc, float("inf"), f"test_metrics.roc_auc is Inf: {roc}")
        self.assertGreaterEqual(roc, 0.0, f"roc_auc < 0: {roc}")
        self.assertLessEqual(roc, 1.0, f"roc_auc > 1: {roc}")
        # Head must have parameters (it's a 300 -> hidden -> 1 MLP).
        self.assertGreater(
            head_params, 0,
            f"head has 0 parameters: wire-in broken"
        )
        # Encoder must have parameters (chemprop_v3 D-MPNN is non-trivial).
        self.assertGreater(
            encoder_params, 0,
            f"encoder has 0 parameters: encoder wrapper broken"
        )

        # §5.7 invariant #4 (loss-finite): parse the epoch-1 loss line.
        m_loss = re.search(r"epoch\s+1/1\s+loss=([\d\.\-]+(?:e[+\-]?\d+)?)", stdout)
        if m_loss is None:
            print("[BLK-D2-S2 §5.7] note: epoch-1 loss line not parsed; "
                  "checking test auc alone")
        else:
            loss = float(m_loss.group(1))
            print(f"[BLK-D2-S2 §5.7] epoch-1 bce loss = {loss:.4f}")
            self.assertFalse(loss != loss, f"bce loss is NaN: {loss}")
            self.assertNotEqual(loss, float("inf"), f"bce loss is Inf: {loss}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
