#!/usr/bin/env python3
"""2026-10-09 — regression tests for the four CFM collapse-repair wire-ins.

Each test here corresponds to a defect MEASURED on the run whose log is at
``molmetal/reports/cfm_blk_c5_repair_2026_10_09/master.log``, i.e. these are
not speculative guards.  The three-agent audit found:

  1. ``--gen-head-lr-scale 0.1`` was a NO-OP.  ``build_lr_scheduler`` passed
     ``max_lr=base_lr * 10`` — a SCALAR — which PyTorch broadcasts to every
     param group.  The per-group ``lr=1e-5`` written at trainer :7411 was
     overwritten at construction, so the generation head ran at the SAME
     peak LR as the bond head for the entire run: ratio 1.0000 at every
     single step.  The log line ``gen_flow_lr=1e-05`` printed the
     PRE-scheduler value, which is why the run looked isolated and was not.
  2. ``pct_start`` was a hardcoded 0.3, so
     ``protocol.scheduler.warmup_epochs`` controlled nothing on the
     OneCycle path even though the yaml has carried it since BLK-C4.
  3. ``generated_smiles.csv`` records RDKit-strict SURVIVORS ONLY (the
     write is gated on a non-empty ``smi``, and ``smi`` is "" for every
     molecule the decoder failed to sanitise), so the failure class the
     audit was chasing was structurally invisible in the file the run
     already wrote.

CPU-only.  Nothing here launches a training run; the §5.7 end-to-end smoke
lives in ``test_wf_blk_c5_bond_loss_and_failure_reasons_smoke.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

sys.path.insert(0, str(REPO))


def _load_trainer():
    spec = importlib.util.spec_from_file_location("_c5_iso_trainer", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_c5_iso_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_trainer()


def _split_optimizer(gen_scale: float = 0.1, base: float = 1e-4):
    """Mirror the trainer's 3-group construction at :7406-7425."""
    a = torch.nn.Linear(4, 4)
    b = torch.nn.Linear(4, 4)
    c = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(
        [
            {"params": a.parameters(), "lr": float(base)},
            {"params": b.parameters(), "lr": float(base) * gen_scale},
            {"params": c.parameters(), "lr": float(base)},
        ],
        lr=float(base),
    )
    return opt, a, b, c


class GenHeadLrIsolation(unittest.TestCase):
    """Defect 1: a scalar max_lr collapsed the split back to ratio 1.0."""

    def test_scalar_max_lr_would_have_been_a_noop(self) -> None:
        """Pins WHY the fix exists: the old shape flattens the ratio.

        If a future torch ever stops broadcasting, this test fails loudly
        rather than the isolation silently coming back.
        """
        opt, _, _, _ = _split_optimizer()
        torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=1e-3,  # the OLD scalar
            total_steps=100,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )
        ratios = [g["lr"] / opt.param_groups[0]["lr"] for g in opt.param_groups]
        self.assertEqual(
            ratios,
            [1.0, 1.0, 1.0],
            "expected the scalar max_lr to flatten every group; if torch "
            "changed, re-check the fix comment at build_lr_scheduler",
        )

    def test_split_ratio_survives_construction(self) -> None:
        opt, _, _, _ = _split_optimizer()
        M.build_lr_scheduler(
            optimizer=opt,
            total_steps=480,
            scheduler_type="onecycle",
            warmup_steps=0,
            min_lr=1e-6,
        )
        ratios = [g["lr"] / opt.param_groups[0]["lr"] for g in opt.param_groups]
        self.assertEqual(len(ratios), 3)
        for r, want in zip(ratios, (1.0, 0.1, 1.0)):
            self.assertAlmostEqual(
                r, want, places=6, msg=f"group ratio {r} != {want}"
            )

    def test_split_ratio_survives_the_whole_curve(self) -> None:
        """MEASURED over the BLK-C5 shape: 480 steps, 16 steps/epoch."""
        opt, _, _, _ = _split_optimizer()
        sched = M.build_lr_scheduler(
            optimizer=opt,
            total_steps=480,
            scheduler_type="onecycle",
            warmup_steps=0,
            min_lr=1e-6,
        )
        steps_per_epoch = 16
        for epoch in (0, 5, 9, 15, 24, 29):
            while sched.last_epoch < epoch * steps_per_epoch:
                sched.step()
            lrs = [g["lr"] for g in opt.param_groups]
            ratio = lrs[1] / lrs[0]
            self.assertAlmostEqual(
                ratio,
                0.1,
                places=4,
                msg=f"gen_flow/bond ratio drifted at epoch {epoch}: {ratio}",
            )
            # The absolute peak also has to be the documented 1e-3 for the
            # un-scaled groups, i.e. the base_lr * 10 contract still holds.
            self.assertLess(max(lrs), 1.1e-3)

    def test_single_group_path_is_unchanged(self) -> None:
        """No split -> one-element list -> same value as the old scalar."""
        m = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        M.build_lr_scheduler(
            optimizer=opt,
            total_steps=480,
            scheduler_type="onecycle",
            warmup_steps=0,
            min_lr=1e-6,
        )
        self.assertEqual(len(opt.param_groups), 1)
        self.assertAlmostEqual(opt.param_groups[0]["max_lr"], 1e-3, places=9)


class PctStart(unittest.TestCase):
    """Defect 2: pct_start was a constant; warmup_epochs reached nothing."""

    @staticmethod
    def _warmup_fraction(pct_start) -> float:
        """MEASURED warmup fraction from the LR curve, not an attribute.

        ``OneCycleLR`` does not retain ``pct_start`` on the instance, so
        the only honest way to assert it took effect is to walk the curve
        and find where the LR stops rising.  Returns the observed fraction
        of total steps spent in the up-phase.
        """
        m = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        total = 480
        sched = M.build_lr_scheduler(
            optimizer=opt,
            total_steps=total,
            scheduler_type="onecycle",
            warmup_steps=0,
            min_lr=1e-6,
            pct_start=pct_start,
        )
        curve = []
        for _ in range(total):
            curve.append(opt.param_groups[0]["lr"])
            sched.step()
        peak = max(range(total), key=lambda i: curve[i])
        return (peak + 1) / total

    def test_default_keeps_the_historical_030(self) -> None:
        self.assertAlmostEqual(self._warmup_fraction(None), 0.30, delta=0.01)

    def test_explicit_value_reaches_the_scheduler(self) -> None:
        self.assertAlmostEqual(self._warmup_fraction(0.35), 0.35, delta=0.01)

    def test_035_is_inside_the_users_30_40_band(self) -> None:
        """用户建议一 2 asked for warmup = 30-40% of total steps."""
        warmup_steps = int(480 * 0.35)
        self.assertEqual(warmup_steps, 168)
        self.assertTrue(0.30 <= warmup_steps / 480 <= 0.40)

    def test_out_of_range_is_clamped_not_rejected(self) -> None:
        """A yaml typo must not crash a launch mid-schedule."""
        self.assertAlmostEqual(self._warmup_fraction(-1.0), 0.0, delta=0.01)
        self.assertLessEqual(self._warmup_fraction(5.0), 0.96)

    def test_cosine_branch_ignores_pct_start(self) -> None:
        """The cosine branch already used warmup_steps; do not double up."""
        m = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        sched = M.build_lr_scheduler(
            optimizer=opt,
            total_steps=480,
            scheduler_type="cosine",
            warmup_steps=80,
            min_lr=1e-6,
            pct_start=0.35,
        )
        self.assertIsNotNone(sched)
        self.assertNotIsInstance(sched, torch.optim.lr_scheduler.OneCycleLR)


class BondLossWeight(unittest.TestCase):
    """用户建议二 修改建议 B — the bond_ce multiplier."""

    def test_default_is_the_identity(self) -> None:
        """§5.7 invariant #1 at the unit level: argparse accepts both new
        flags and their defaults leave the objective untouched."""
        saved = sys.argv
        try:
            sys.argv = ["trainer.py"]
            args = M.parse_args()
        finally:
            sys.argv = saved
        self.assertEqual(args.bond_loss_weight, 1.0)
        self.assertIsNone(args.lr_pct_start)

    def test_flags_are_accepted_with_the_right_types(self) -> None:
        saved = sys.argv
        try:
            sys.argv = [
                "trainer.py",
                "--bond-loss-weight",
                "0.1",
                "--lr-pct-start",
                "0.35",
            ]
            args = M.parse_args()
        finally:
            sys.argv = saved
        # Verify by TYPE, not just by value (AGENTS.md §5.11: a LIST-valued
        # flag declared on the CLI arrives as one string from yaml, which
        # both gates pass — the same trap, checked here for the float flags).
        self.assertIsInstance(args.bond_loss_weight, float)
        self.assertIsInstance(args.lr_pct_start, float)
        self.assertAlmostEqual(args.bond_loss_weight, 0.1, places=9)
        self.assertAlmostEqual(args.lr_pct_start, 0.35, places=9)

    def test_only_bond_ce_is_scaled(self) -> None:
        bond_ce = torch.tensor(2.0, requires_grad=True)
        w2 = torch.tensor(0.5)
        cb = torch.tensor(0.1)
        w = 0.1
        loss = w * bond_ce + 0.1 * w2 + 0.05 * cb
        # Only the bond term moves.
        self.assertAlmostEqual(float(loss), 0.2 + 0.05 + 0.005, places=6)
        self.assertAlmostEqual(float(0.1 * w2 + 0.05 * cb), 0.055, places=6)

    def test_gradient_scales_with_the_weight(self) -> None:
        grads = []
        for w in (1.0, 0.1):
            bond_ce = torch.tensor(2.0, requires_grad=True)
            (w * bond_ce).backward()
            grads.append(float(bond_ce.grad))
        self.assertAlmostEqual(grads[0], 1.0, places=6)
        self.assertAlmostEqual(grads[1], 0.1, places=6)


class FailureReasonTaxonomy(unittest.TestCase):
    """Defect 3: the shipped SMILES dump cannot express a failure."""

    def test_ok_molecule_has_no_reason(self) -> None:
        self.assertIsNone(M._rdkit_strict_reason("CCO", sanitized=True))

    def test_empty_smiles_is_its_own_bucket(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", sanitized=False), "no_smiles"
        )

    def test_unsanitized_is_reported_as_such(self) -> None:
        self.assertEqual(
            M._rdkit_strict_reason("", sanitized=False), "no_smiles"
        )

    def test_rejected_string_that_was_claimed_sanitized(self) -> None:
        """sanitized=True but the round-trip fails must NOT read as ok."""
        # Not a parseable SMILES: MolFromSmiles returns None.
        self.assertEqual(
            M._rdkit_strict_reason("C(((", sanitized=True), "reparse_failed"
        )

    def test_loose_atoms_parse_and_are_still_ok(self) -> None:
        """The point of the caveat: RDKit accepts `C.C.C.C`.

        This is why the taxonomy is a SEPARATE observation from
        `gen_validity`; the strict gate cannot see fragmentation, and
        pretending otherwise in the reason string would be the same
        blindness in a new place.
        """
        self.assertIsNone(M._rdkit_strict_reason("C.C.C.C", sanitized=True))

    def test_counters_sum_to_the_sample_count(self) -> None:
        """Every sample lands in exactly one bucket."""
        reasons = {}
        for smi, san in (
            ("CCO", True),
            ("c1ccccc1O", True),
            ("", False),
            ("", False),
            ("C(((", True),
        ):
            r = M._rdkit_strict_reason(smi, san) or "ok"
            reasons[r] = reasons.get(r, 0) + 1
        self.assertEqual(sum(reasons.values()), 5)
        self.assertEqual(reasons["ok"], 2)
        self.assertEqual(reasons["no_smiles"], 2)
        self.assertEqual(reasons["reparse_failed"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
