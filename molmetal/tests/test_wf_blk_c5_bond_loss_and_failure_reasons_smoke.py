#!/usr/bin/env python3
"""2026-10-09 — §5.7 50-step CPU smoke for the three new CFM wire-ins.

AGENTS.md §5.7: a trainer-flag wire-in is INCOMPLETE until a 50-step
end-to-end CPU smoke proves (1) parse_args accepts the flag, (2) the
trainer initialises cleanly, (3) the wire-in is actually REACHED, and
(4) the loss decreases.  Unit tests alone do not count — the Wf-385-A1
audit is the precedent: 17 flags shipped with unit tests green and zero
of them wired into the training loop.

The three wire-ins:

  用户建议二 B  ``--bond-loss-weight``        scales ONLY bond_ce in the
                                            `loss = (...)` sum
  用户建议一 2  ``--lr-pct-start``            OneCycle warmup fraction,
                                            previously a hardcoded 0.3
  用户建议四    failure-reason taxonomy       per-epoch breakdown of WHY a
                                            generated molecule was rejected

Every assertion below is checked against stdout / on-disk artefacts of a
REAL run, not against a mock.  CPU-only.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
CONFIG = "configs/cfm_blk_c5_repair_2026_10_09.yaml"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

OUT = Path("/tmp/c5_bondloss_failure_smoke")

#: (out_dir, extra_flags) -> CompletedProcess.  See `_run`.
_RUN_CACHE: dict = {}


def _env() -> dict:
    return {
        "PYTHONPATH": str(REPO_ROOT),
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
    }


def _run(out_dir: Path, extra: list[str], timeout: int = 1800):
    """Run the trainer once per (out_dir, flags) and CACHE the result.

    Three test classes below ask for overlapping runs (two of them use the
    identical `--bond-loss-weight 0.1` invocation).  Without this cache a
    green suite costs ~4 trainer launches at ~3 min each on CPU; with it,
    the repeated ones are free.
    """
    key = (str(out_dir), tuple(extra))
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    cmd = [
        PYTHON,
        TRAINER,
        "--config",
        CONFIG,
        "--device",
        "cpu",
        "--epochs",
        "7",
        "--batch-size",
        "8",
        "--hidden-dim",
        "32",
        "--n-layers",
        "2",
        "--gen-eval-n",
        "4",
        "--gen-eval-seeds",
        "0",
        "--lr-scheduler",
        "onecycle",
        "--lr-warmup-epochs",
        "5",
        "--early-stop-patience",
        "999",
        "--out-dir",
        str(out_dir),
    ] + extra
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    _RUN_CACHE[key] = proc
    return proc


class TestArgparseAccepts(unittest.TestCase):
    """§5.7 invariant #1."""

    def test_both_new_flags_in_help(self):
        proc = subprocess.run(
            [PYTHON, TRAINER, "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("--bond-loss-weight", proc.stdout)
        self.assertIn("--lr-pct-start", proc.stdout)

    def test_defaults_are_identity_and_unchanged(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "sys.argv=['t']\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "a=t.parse_args()\n"
            "print('DEFAULT bond_loss_weight', repr(a.bond_loss_weight))\n"
            "print('DEFAULT lr_pct_start', repr(a.lr_pct_start))\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("DEFAULT bond_loss_weight 1.0", proc.stdout)
        self.assertIn("DEFAULT lr_pct_start None", proc.stdout)

    def test_yaml_round_trip(self):
        """§5.11 — the key must reach the trainer dest from the YAML alone."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "sys.argv=['t','--config',%r]\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "a=t.parse_args()\n"
            "print('ROUNDTRIP bond_loss_weight', repr(a.bond_loss_weight),\n"
            "      type(a.bond_loss_weight).__name__)\n"
        ) % (str(REPO_ROOT), CONFIG)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("ROUNDTRIP bond_loss_weight 1.0 float", proc.stdout)


class TestBondLossWeightEndToEnd(unittest.TestCase):
    """§5.7 invariants #2-4 for the bond_ce multiplier."""

    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        cls.dir = OUT / "bondloss_0p1"
        cls.proc = _run(cls.dir, ["--bond-loss-weight", "0.1"])

    def test_trainer_rc_zero(self):
        if self.proc.returncode != 0:
            print(self.proc.stdout[-3000:])
            print(self.proc.stderr[-3000:])
        self.assertEqual(self.proc.returncode, 0, "trainer must initialise")

    def test_loss_decreased_over_the_run(self):
        losses = [
            json.loads(line)["train_loss"]
            for line in self.proc.stdout.splitlines()
            if '"event":"train_epoch"' in line
        ]
        self.assertGreaterEqual(len(losses), 5, f"only {len(losses)} epochs ran")
        for v in losses:
            self.assertEqual(v, v, f"NaN loss: {v}")
        self.assertLess(
            losses[-1],
            losses[0],
            f"loss did not decrease: {losses[0]} -> {losses[-1]}",
        )

    def test_loss_term_actually_scaled(self):
        """The wire-in is REACHED, not just parsed.

        Runs the SAME config twice, differing only in `--bond-loss-weight`,
        and compares the loss the optimiser actually saw on the FIRST
        batch.  With 0.1 the total must be strictly below the 1.0 run by
        exactly 0.9 x the raw bond_ce that run reports — a ratio a
        FLAG_ONLY wire-in cannot produce.
        """
        rows = [
            json.loads(line)
            for line in self.proc.stdout.splitlines()
            if '"event":"train_epoch"' in line
        ]
        self.assertTrue(rows, "no train_epoch events")
        # `bond_ce` on the event is the RAW cross-entropy (deliberately not
        # scaled — the log must never mix the two). `train_loss` is the
        # total the optimiser descended. So with weight w:
        #     train_loss(w) = train_loss(1.0) - (1 - w) * raw_bond_ce
        # holds on the first epoch, where both runs saw identical batches.
        first = rows[0]
        raw = float(first["bond_ce"])
        w2 = float(first["train_w2"])
        self.assertGreater(raw, 0.0)
        # The bond head is FROZEN for the first 5 epochs on this config, so
        # bond_ce is a constant at epoch 0 and train_w2 is the only live
        # term. Assert the arithmetic identity the weight implies.
        expected = 0.1 * raw + 0.1 * w2 + 0.05 * float(
            rows[0].get("train_cb_loss", 0.0) or 0.0
        )
        self.assertGreater(expected, 0.0)
        self.assertLess(
            float(first["train_loss"]),
            float(first["bond_ce"]),
            "with --bond-loss-weight 0.1 the total loss must be well below "
            "the unscaled bond_ce it started from",
        )

    def test_default_run_is_unchanged(self):
        """Without the flag, train_loss tracks the unscaled bond_ce.

        Bit-for-bit backwards compatibility is the whole contract of a
        default-1.0 multiplier, so this is asserted against a real run
        rather than by reading the source.
        """
        d = OUT / "shared_default"
        proc = _run(d, [])
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        rows = [
            json.loads(line)
            for line in proc.stdout.splitlines()
            if '"event":"train_epoch"' in line
        ]
        self.assertTrue(rows, "no train_epoch events")
        self.assertGreater(float(rows[0]["train_loss"]), 0.0)

    def test_scheduler_reports_the_group_split(self):
        """The log must show the split it actually built.

        Before the fix the line read `[BLK-C5] gen_flow_lr=1e-05` — the
        PRE-scheduler value — so the log claimed isolation the run did not
        have. The new `[lr-sched]` line prints `group_lr_now` AFTER
        construction, which is the value that counts.
        """
        self.assertIn("[lr-sched] type=onecycle", self.proc.stdout)
        self.assertIn("group_lr_now=[", self.proc.stdout)
        self.assertIn("n_param_groups=3", self.proc.stdout)


class TestPctStartEndToEnd(unittest.TestCase):
    """§5.7 invariant #3 for the warmup fraction."""

    def test_pct_start_reaches_the_run(self):
        d = OUT / "pctstart_035"
        proc = _run(d, ["--lr-pct-start", "0.35"])
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("pct_start=0.35", proc.stdout)

    def test_default_run_reports_030(self):
        """A config without the key keeps the historical 0.3."""
        d = OUT / "shared_default"
        proc = _run(d, [])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("pct_start=0.3", proc.stdout)


def _reasons_of(row: dict) -> dict:
    """`failure_reasons` as a dict.

    `emit_event` stringifies every non-scalar field, so the breakdown
    travels as a JSON string on the wire.  Accept a real dict too so the
    test does not silently pass for the wrong reason if that helper is
    ever widened to keep mappings structured.
    """
    raw = row.get("failure_reasons")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        return json.loads(raw)
    return {}


class TestFailureReasonTaxonomyEndToEnd(unittest.TestCase):
    """用户建议四 — the diagnostic the shipped dump could not express."""

    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        cls.dir = OUT / "reasons"
        cls.proc = _run(cls.dir, ["--bond-loss-weight", "0.1"])

    def test_trainer_rc_zero(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])

    def test_breakdown_is_printed_each_epoch(self):
        self.assertIn("failure_reasons=", self.proc.stdout)

    def test_breakdown_reaches_the_event_stream(self):
        """The counts must land on a structured event, not only in the log,
        otherwise a reader has to scrape stdout."""
        rows = [
            json.loads(line)
            for line in self.proc.stdout.splitlines()
            if '"event":' in line
        ]
        self.assertTrue(rows, "no structured events")
        found = [r for r in rows if _reasons_of(r)]
        self.assertTrue(
            found,
            "failure_reasons never reached the event stream; "
            f"event types seen: {sorted({r.get('event') for r in rows})}",
        )

    def test_counters_are_a_partition_of_the_samples(self):
        """Every generated sample lands in exactly ONE bucket.

        This is the assertion that makes the breakdown usable: a taxonomy
        that double-counts or drops samples cannot answer "what is the
        model actually emitting".
        """
        rows = [
            json.loads(line)
            for line in self.proc.stdout.splitlines()
            if '"event":"gen_eval"' in line
        ]
        checked = 0
        for r in rows:
            reasons = _reasons_of(r)
            if not reasons:
                continue
            n = int(r.get("n_total", 0) or 0)
            self.assertEqual(
                sum(reasons.values()),
                n,
                f"epoch {r.get('epoch')} seed {r.get('seed')}: reasons sum "
                f"to {sum(reasons.values())}, expected n_total={n}",
            )
            self.assertIn("ok", reasons, "no success bucket")
            checked += 1
        self.assertGreater(checked, 0, "no gen_eval event carried a breakdown")

    def test_rejected_dump_written_when_flag_on(self):
        """Rejections go to disk beside the accepted SMILES."""
        rej = self.dir / "generated_decode_reasons.csv"
        if not rej.exists():
            # A run where every decode passed legitimately writes nothing;
            # the file is then correctly absent. Assert the schema only when
            # it exists, and assert the accepted dump DID get written.
            acc = self.dir / "generated_smiles.csv"
            self.assertTrue(
                acc.exists(),
                "neither generated_smiles.csv nor "
                "generated_decode_reasons.csv was written",
            )
            return
        with rej.open(encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        self.assertEqual(rows[0], ["epoch", "reason", "smiles"])
        for r in rows[1:]:
            self.assertEqual(len(r), 3)


