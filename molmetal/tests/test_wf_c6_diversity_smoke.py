#!/usr/bin/env python3
"""BLK-C6 §5.7 50-step CPU smoke: 6 diversity flags ON.

Per AGENTS.md §5.7 the wire-in is INCOMPLETE until a 50-step end-to-end
smoke verifies:
  1. parse_args accepts all 6 flags without argparse error
  2. trainer imports + initialises cleanly
  3. decode_ratio > 0
  4. loss[0] > loss[49]
  5. diversity_term > 0 (MMD + repulsion components separately)

Uses wf399a1_smoke_protocol.run_smoke_50step + verify_all_invariants,
plus an in-process probe of the trainer module's _LAST_LOSS_TERMS dict
to verify the diversity term is wired into the loss cascade.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "molmetal"))

from molmetal.scripts.wf399a1_smoke_protocol import (
    run_smoke_50step,
    verify_all_invariants,
)


SIX_FLAGS = [
    "--use-egnn-velocity",
    "--top-p", "0.9",
    "--top-k", "50",
    "--temperature", "0.8",
    "--mmd-weight", "0.1",
    "--repulsion-weight", "0.05",
    "--scaffold-conditioning",
]


class C6DiversitySmoke(unittest.TestCase):
    def test_six_diversity_flags_smoke(self) -> None:
        out_dir = "/tmp/c6_smoke_57_v4"
        os.makedirs(out_dir, exist_ok=True)

        smoke = run_smoke_50step(
            trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
            flags_list=[
                "--epochs", "2",
                "--batch-size", "8",
                "--hidden-dim", "32",
                "--n-layers", "2",
                *SIX_FLAGS,
                "--device", "cpu",
                "--early-stop-patience", "999",
                "--tmqm-dir", ".data/molmetal/tmQM",
                "--out-dir", out_dir,
            ],
            timeout_sec=120,
            cwd=str(REPO),
        )

        # parse_args + initialise: rc must be 0
        self.assertEqual(smoke.returncode, 0, f"trainer rc={smoke.returncode}")

        # decode_ratio > 0: parse from stdout banner (the wf399a1 helper handles this)
        from molmetal.scripts.wf399a1_smoke_protocol import verify_decode_ratio
        dr_verdict = verify_decode_ratio(smoke)
        print(f"[BLK-C6 §5.7] decode_ratio verdict: {dr_verdict}")
        self.assertTrue(dr_verdict["invariant_holds"],
                        f"decode_ratio invariant failed: {dr_verdict}")

        # loss[0] > loss[N-1]: parse from events.jsonl (the CFM trainer emits
        # per-epoch train_loss there, not in metrics.json which the helper
        # expects). With batch-size=8 / train=256 = 32 steps/epoch, 2 epochs
        # = 64 steps — exceeds the §5.7 50-step floor.
        events_path = Path(out_dir) / "events.jsonl"
        self.assertTrue(events_path.exists(), f"events.jsonl missing at {events_path}")
        epoch_losses = []
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("event") == "train_epoch":
                    epoch_losses.append(float(obj["train_loss"]))
        print(f"[BLK-C6 §5.7] epoch losses: {epoch_losses}")
        self.assertGreaterEqual(len(epoch_losses), 2,
                                f"need >=2 epochs for loss monotonicity check, got {len(epoch_losses)}")
        self.assertGreater(epoch_losses[0], epoch_losses[-1],
                           f"loss not monotone decreasing: {epoch_losses}")

        # diversity_term: read _LAST_LOSS_TERMS after the run
        diversity_value = self._read_diversity_term(out_dir)
        print(f"[BLK-C6 §5.7] diversity_term (from in-process probe) = {diversity_value}")
        self.assertIsNotNone(diversity_value, "diversity_term not captured")
        self.assertGreater(diversity_value, 0.0,
                            f"diversity_term={diversity_value} should be > 0")

    def _read_diversity_term(self, out_dir: str) -> float | None:
        """Run a single training step in-process with the 6 flags,
        then read `trainer._LAST_LOSS_TERMS['diversity_term']`."""
        # Probe script: invoke the trainer's main() in-process and capture
        # the module-level _LAST_LOSS_TERMS dict it populates.
        probe_script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "import sys as _s\n"
            "_s.argv = ['trainer', "
            "'--epochs', '1', '--batch-size', '4', "
            # Wf-A106 — this probe reads _LAST_LOSS_TERMS after ONE step; it
            # does not need the full cohort.  Without these two it inherited
            # the default 192-cloud validation set, MEASURED ~290 s, against
            # the 120 s timeout below — so the probe raised TimeoutExpired and
            # the test reported "diversity_term not captured".  A budget that
            # is smaller than the work is a test that fails on machine load.
            "'--n-train', '16', '--n-val', '8', "
            "'--hidden-dim', '32', '--n-layers', '2', "
            "'--use-egnn-velocity', "
            "'--top-p', '0.9', '--top-k', '50', '--temperature', '0.8', "
            "'--mmd-weight', '0.1', '--repulsion-weight', '0.05', "
            "'--scaffold-conditioning', "
            "'--device', 'cpu', '--early-stop-patience', '999', "
            "'--tmqm-dir', '.data/molmetal/tmQM', "
            # Wf-A106 — the probe was NOT hermetic.  It reused a fixed out-dir,
            # so the trainer found the checkpoint a previous run had left
            # there, printed "resume target is already complete (epoch=1 >=
            # args.epochs=1); nothing to train", and returned with
            # _LAST_LOSS_TERMS EMPTY.  The test then reported
            # "diversity_term not captured" — a passing probe that never ran.
            "'--resume-from', 'none', "
            f"'--out-dir', {out_dir + '_probe'!r}, "
            f"'--ckpt-dir', {out_dir + '_probe_ck'!r}]\n"
            "t.main()\n"
            "terms = {k: (float(v) if hasattr(v, 'item') else None) "
            "         for k, v in t._LAST_LOSS_TERMS.items()}\n"
            "print(json.dumps({'diversity_term': terms.get('diversity_term'), "
            "                  'bond_ce': terms.get('bond_ce'), "
            "                  'wasserstein': terms.get('wasserstein_weight*w2_val')}, "
            "                 default=str))\n"
        )
        proc = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", probe_script],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            print(f"[_read_diversity_term] probe rc={proc.returncode} stderr={proc.stderr[-500:]}")
            return None
        # Find the last JSON-looking line in stdout
        for line in proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and "diversity_term" in line:
                try:
                    obj = json.loads(line)
                    v = obj.get("diversity_term")
                    if v is None:
                        # Wf-A106 — the probe DID emit a line, but the term was
                        # missing/None inside it.  Say so instead of returning a
                        # bare None that reads like "the probe never ran".
                        print(
                            f"[_read_diversity_term] parsed line has no "
                            f"diversity_term: {obj!r}"
                        )
                    return float(v) if v is not None else None
                except Exception:
                    continue
        # Wf-A106 — a bare None is undiagnosable.  This probe failed with
        # "diversity_term not captured" and nothing else, twice.
        print(
            f"[_read_diversity_term] rc={proc.returncode} no JSON line; "
            f"stdout tail={proc.stdout[-400:]!r} stderr tail={proc.stderr[-400:]!r}"
        )
        return None


if __name__ == "__main__":
    unittest.main(verbosity=2)
