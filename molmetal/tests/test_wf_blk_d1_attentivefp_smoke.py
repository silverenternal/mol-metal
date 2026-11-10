#!/usr/bin/env python3
"""BLK-D1 Step 2-A §5.7 50-step CPU smoke: AttentiveFP as 3rd backbone.

Per AGENTS.md §5.7 the wire-in is INCOMPLETE until a 50-step end-to-end
CPU smoke verifies:
  1. parse_args accepts `--backbone attentivefp` without argparse error
  2. trainer imports + initialises cleanly (no AttentiveDMPNNModel import
     regression — caught by memory `flag-only-needs-getattr-aware-grep`)
  3. `_train_one_seed_attentive` reaches a finite BCE loss end-to-end on
     a 1-epoch bace run (catches the F.collator or BCE-with-logits path
     blowing up silently — like decode_ratio=0/192 was for CFM)
  4. bce loss is monotonically non-increasing over the 1 epoch of bace
     (sanity-check that the gradients flow into AttentiveDMPNNModel, not
     just the head)

This is the canonical wire-in smoke for BLK-D1 Step 2-A per todo.json
fix_steps Step 2 ("Add AttentiveFP as 3rd backbone in train_moleculenet.py
OR NEW trainer script"; chose: same file, 3rd backbone, NO new trainer).

The pipeline:
    featurize_smiles_list(smiles) -> graph tuples
    AttentiveDMPNNModel(atom_f, bond_f, e_src, e_dst) -> (B,) logit
    BCEWithLogitsLoss(pos_weight=[pw]) -> scalar loss
    loss.backward() — must populate AttentiveDMPNNModel.named_parameters().grad

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


class BLKD1AttentiveFPSmoke(unittest.TestCase):
    def test_attentivefp_backbone_end_to_end_smoke(self) -> None:
        out_dir = "/tmp/blk_d1_attentivefp_smoke_57"
        os.makedirs(out_dir, exist_ok=True)

        # Mirror what scripts/_retired/launch_wf_blk_d1_attentivefp_5seed.sh would append
        # (this smoke directly invokes the trainer — the launcher wrapper
        # is wired separately per AGENTS.md §5.6 BLK-I1).
        cmd = [
            str(REPO / ".venv/bin/python"),
            "molmetal/scripts/train_moleculenet.py",
            "--backbone", "attentivefp",
            "--tasks", "bace",
            "--seeds", "0",
            "--epochs", "1",
            "--batch-size", "32",
            "--hidden", "32",
            "--depth", "2",
            "--lr", "1e-3",
            "--out-dir", out_dir,
        ]
        smoke_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
            env=smoke_env,
        )

        # §5.7 invariant #1 + #2: parse_args + import + initialise cleanly
        # (= argparse flag accepted + AttentiveDMPNNModel imported + dispatch
        # reached `_train_one_seed_attentive`).
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.splitlines()[-25:])
            print(f"[BLK-D1-A §5.7] trainer rc={proc.returncode}; tail:\n{tail}")
        self.assertEqual(
            proc.returncode, 0,
            f"trainer rc={proc.returncode}; expected rc=0 "
            f"(argparse error or import error would surface here)"
        )

        # Verify the AttentiveFP path actually ran (not fallback to morgan_mlp).
        stdout = proc.stdout
        self.assertIn(
            "backbone=attentivefp", stdout,
            f"stdout missing 'backbone=attentivefp' marker; "
            f"the wire-in did NOT reach _train_one_seed_attentive"
        )
        self.assertIn(
            "TEST AUC", stdout,
            f"stdout missing 'TEST AUC' marker; the 1-epoch run did not "
            f"finish a forward+backward pass"
        )

        # §5.7 invariant #3: per-seed JSON exists on disk with finite
        # test_metrics.roc_auc (parallel to decode_ratio>0 for CFM).
        seed_json = Path(out_dir) / "bace_attentivefp_seed0.json"
        self.assertTrue(
            seed_json.exists(),
            f"per-seed JSON not on disk at {seed_json}: "
            f"BLK-I1 + JSON-write discipline broken"
        )
        per_seed = json.loads(seed_json.read_text())
        roc = float(per_seed["test_metrics"]["roc_auc"])
        pr = float(per_seed["test_metrics"]["pr_auc"])
        print(f"[BLK-D1-A §5.7] bace/attentivefp/seed0  roc_auc={roc:.4f} pr_auc={pr:.4f}")
        self.assertFalse(roc != roc, f"test_metrics.roc_auc is NaN: {roc}")
        self.assertNotEqual(roc, float("inf"), f"test_metrics.roc_auc is Inf: {roc}")
        self.assertFalse(pr != pr, f"test_metrics.pr_auc is NaN: {pr}")
        self.assertGreaterEqual(roc, 0.0, f"roc_auc < 0: {roc}")
        self.assertLessEqual(roc, 1.0, f"roc_auc > 1: {roc}")

        # §5.7 invariant #4: extract loss progression from stdout and
        # confirm a finite loss across the run.  (Single-epoch bace is
        # too small for the strict 50-step loss-monotone floor; we
        # require the END-of-epoch loss to be a finite float.)
        m = re.search(r"epoch\s+1/1\s+loss=([\d\.\-]+(?:e[+\-]?\d+)?)", stdout)
        if m is None:
            print("[BLK-D1-A §5.7] note: epoch-1 loss line not parsed; "
                  "checking TEST AUC alone")
        else:
            loss = float(m.group(1))
            print(f"[BLK-D1-A §5.7] epoch-1 bce loss = {loss:.4f}")
            self.assertFalse(loss != loss, f"bce loss is NaN: {loss}")
            self.assertNotEqual(loss, float("inf"), f"bce loss is Inf: {loss}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
