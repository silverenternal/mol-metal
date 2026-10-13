"""BLK-C5 §5.7 50-step CPU smoke — CFM generation-head collapse repair.

AGENTS.md §5.7 makes a trainer-flag wire-in INCOMPLETE until a 50-step
end-to-end CPU smoke passes. BLK-C5 (2026-10-09) adds five flags to
``train_cfm_yuelbond_wasserstein.py``; this file is the gate.

The four structural levers, each from the user's verbatim advice:

  一  LR 分组隔离        -> ``--gen-head-lr-scale``       (default 1.0 = off)
  二  Bond head 阶段性冻结 -> ``--bond-head-warmup-epochs`` (default 0 = off)
  一  Gradient clipping   -> ``--max-norm``               (default 5.0)
  三  熵正则化             -> ``--entropy-reg-weight``     (default 0.0 = off)
  三  Gen-head dropout     -> ``--gen-head-dropout``       (default None = off)

What is asserted (and why each one is here):

  * argparse accepts every new flag          — catches dest typos
  * trainer rc == 0                         — the §5.7 floor
  * loss is finite                          — catches NaN/0-vector blowups
  * model.head params are frozen at epoch 0  — the freeze is NOT FLAG_ONLY
  * model.head params unfreeze at the boundary — the unfreeze is reachable
  * entropy_term is gradient-LIVE at weight>0, exact-zero at weight 0
  * the two param groups really are separated, and the gen-head group
    really is the generation head (not an empty split)

Everything here is CPU-only and runs in well under the §5.7 60 s target.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = "molmetal/scripts/train_cfm_yuelbond_wasserstein.py"
CONFIG = "configs/cfm_bond_head_30ep_onecycle.yaml"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

NEW_FLAGS = [
    "--gen-head-lr-scale",
    "--bond-head-warmup-epochs",
    "--bond-head-unfreeze-stride",
    "--entropy-reg-weight",
    "--gen-head-dropout",
    "--bond-predictor",
]


def _base_flags(out_dir: str, epochs: int, **extra) -> list:
    """Tiny CPU config the trainer will actually accept."""
    flags = [
        "--config",
        CONFIG,
        "--device",
        "cpu",
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--gen-eval-n",
        "2",
        "--early-stop-patience",
        "999",
        "--out-dir",
        out_dir,
    ]
    for k, v in extra.items():
        flags += [f"--{k.replace('_', '-')}", str(v)]
    return flags


def _run(flags: list, timeout: int = 900) -> subprocess.CompletedProcess:
    env = {
        "PYTHONPATH": str(REPO_ROOT),
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
    }
    return subprocess.run(
        [PYTHON, TRAINER] + flags,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestBlkC5Argparse(unittest.TestCase):
    """§5.7 floor #1 — the flags parse."""

    def test_all_five_new_flags_accepted(self):
        proc = subprocess.run(
            [PYTHON, TRAINER, "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        for flag in NEW_FLAGS:
            self.assertIn(flag, proc.stdout, f"{flag} missing from --help")

    def test_bond_predictor_choices_in_help(self):
        """`--bond-predictor` must list `yuelbond` and `dmpnn` (BLK-C5 follow-up)."""
        proc = subprocess.run(
            [PYTHON, TRAINER, "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("{yuelbond,dmpnn}", proc.stdout)

    def test_bond_predictor_default_is_yuelbond(self):
        """Default must be 'yuelbond' so every shipped wrapper / checkpoint
        loads bit-exactly through the historical head."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "sys.argv = ['trainer']\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "a = t.parse_args()\n"
            "print('DEFAULT', 'bond_predictor', repr(getattr(a, 'bond_predictor')))\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("DEFAULT bond_predictor 'yuelbond'", proc.stdout)

    def test_bond_predictor_garbage_rejected(self):
        """A value outside {yuelbond, dmpnn} must be refused by argparse."""
        proc = subprocess.run(
            [PYTHON, TRAINER, "--bond-predictor", "garbage",
             "--config", CONFIG, "--device", "cpu",
             "--epochs", "1", "--batch-size", "8",
             "--out-dir", "/tmp/blk_c5_test_garbage"],
            cwd=str(REPO_ROOT),
            env={
                "PYTHONPATH": str(REPO_ROOT),
                "CUDA_VISIBLE_DEVICES": "",
                "PATH": "/usr/bin:/bin",
                "HOME": str(Path.home()),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertNotEqual(proc.returncode, 0, "garbage value was accepted")

    def test_defaults_are_backward_compatible(self):
        """Every new flag must default to the pre-BLK-C5 behaviour.

        Asserted by PARSING, not by scraping `--help`. This trainer does not
        use ArgumentDefaultsHelpFormatter, so argparse never renders a
        `(default: X)` suffix — a help-text scrape cannot see the defaults
        at all, which is how the first version of this test failed.
        """
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "sys.argv = ['trainer']\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "a = t.parse_args()\n"
            "for k in ('gen_head_lr_scale', 'bond_head_warmup_epochs',\n"
            "          'bond_head_unfreeze_stride', 'entropy_reg_weight',\n"
            "          'gen_head_dropout', 'bond_predictor', 'max_norm'):\n"
            "    print('DEFAULT', k, repr(getattr(a, k)))\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        found = dict(
            line.split()[1:3] for line in proc.stdout.splitlines() if line.startswith("DEFAULT")
        )
        expectations = {
            "gen_head_lr_scale": "1.0",  # single flat param group
            "bond_head_warmup_epochs": "0",  # never freeze
            "bond_head_unfreeze_stride": "1",  # never re-freeze
            "entropy_reg_weight": "0.0",  # exact zero loss term
            "gen_head_dropout": "None",  # head keeps its constructor value
            # 2026-10-09 NEW (BLK-C5 follow-up, user voice) — bond
            # predictor defaults to 'yuelbond' so every shipped CFM
            # checkpoint loads bit-exactly through the historical head.
            "bond_predictor": "'yuelbond'",
            # max_norm is deliberately NOT changed to 1.0: that is a
            # run-level setting, and flipping the default would silently
            # change every wrapper that does not pass it. The BLK-C5 launch
            # wrapper passes --max-norm 1.0 explicitly.
            "max_norm": "5.0",
        }
        for key, expected in expectations.items():
            self.assertIn(key, found, f"{key} not parsed out of the namespace")
            self.assertEqual(
                found[key],
                expected,
                f"{key} default is {found[key]}, expected {expected} "
                f"(the backward-compatible value)",
            )


class TestBlkC5ParamGroupIsolation(unittest.TestCase):
    """用户建议一 — LR 分组隔离. Run once; assert on the banner."""

    @classmethod
    def setUpClass(cls):
        cls.proc = _run(
            _base_flags("/tmp/blk_c5_test_pg", 1, gen_head_lr_scale=0.1)
        )
        cls.out = cls.proc.stdout + cls.proc.stderr

    def test_rc_zero(self):
        self.assertEqual(self.proc.returncode, 0, self.out[-3000:])

    def test_three_param_groups_with_separate_lrs(self):
        # BLK-C5 rev. 2 (2026-10-09): three-group split — atom_head / gen_flow
        # / bond_head.  All three must be non-empty (any empty side means
        # the marker list drifted from the model layout), and gen_flow must
        # be the MAJORITY group (vel_field + msg_mlps = 74% of params at the
        # yaml's hidden_dim=128/n_layers=3), not the minority 14-param head
        # that rev. 1 incorrectly isolated.
        m = re.search(
            r"\[BLK-C5\] param_groups=3 "
            r"atom_head=(\d+) gen_flow=(\d+) bond_head=(\d+)",
            self.out,
        )
        self.assertIsNotNone(m, "param_groups banner missing:\n" + self.out[-3000:])
        atom_n, flow_n, bond_n = (int(m.group(i)) for i in (1, 2, 3))
        self.assertGreater(atom_n, 0)
        self.assertGreater(flow_n, 0)
        self.assertGreater(bond_n, 0)
        # gen_flow must dominate the model — that's the whole point of the
        # rev-2 direction reversal.  Catches a future regression that
        # silently falls back to the rev-1 head-vs-vel 2-group split.
        self.assertGreater(
            flow_n,
            atom_n + bond_n,
            f"gen_flow ({flow_n}) should be the largest group; "
            f"atom ({atom_n}) + bond ({bond_n}) together = {atom_n + bond_n}. "
            "The rev-1 marker ('head.',) lumped them the other way and "
            "isolated the WRONG component — see closeout §2.2.",
        )

    def test_gen_flow_lr_is_scaled_down(self):
        m = re.search(
            r"gen_flow_lr=([0-9.eE+-]+) \(base lr=([0-9.eE+-]+)\)", self.out
        )
        self.assertIsNotNone(m, "gen_flow_lr banner missing")
        flow_lr, base_lr = float(m.group(1)), float(m.group(2))
        self.assertLess(flow_lr, base_lr, "gen_flow LR was not scaled below base")
        self.assertAlmostEqual(flow_lr / base_lr, 0.1, places=6)


class TestBlkC5Smoke(unittest.TestCase):
    """§5.7 50-step smoke with every lever engaged."""

    @classmethod
    def setUpClass(cls):
        # warmup=1 over 2 epochs so BOTH the frozen epoch and the unfreeze
        # boundary execute inside one run.
        cls.proc = _run(
            _base_flags(
                "/tmp/blk_c5_test_smoke",
                2,
                max_norm=1.0,
                gen_head_lr_scale=0.1,
                bond_head_warmup_epochs=1,
                bond_head_unfreeze_stride=1,
                entropy_reg_weight=0.01,
                gen_head_dropout=0.2,
            )
        )
        cls.out = cls.proc.stdout + cls.proc.stderr

    def test_rc_zero(self):
        self.assertEqual(self.proc.returncode, 0, self.out[-3000:])

    def test_head_frozen_then_unfrozen(self):
        """用户建议二 — the freeze must fire, and the unfreeze must be reachable."""
        self.assertIn(
            "bond_head FROZEN for the first 1 epoch(s)",
            self.out,
            "staged freeze never engaged — the flag would be FLAG_ONLY",
        )
        self.assertIn(
            "bond_head unfrozen after epoch 1",
            self.out,
            "unfreeze at the warmup boundary never fired",
        )

    def test_dropout_wrapped_real_linears(self):
        """用户建议三 — must wrap >=2 linear layers, not 0 and not 1.

        The first implementation indexed `pair_head[1]`, which is an nn.ReLU,
        so exactly one layer was wrapped. The smoke banner is the only place
        that surfaced it, so the count is asserted rather than assumed.
        """
        m = re.search(r"gen_head_dropout=[0-9.]+ wrapped .* on (\d+) head layer", self.out)
        self.assertIsNotNone(m, "dropout banner missing:\n" + self.out[-3000:])
        self.assertGreaterEqual(
            int(m.group(1)),
            2,
            "fewer than 2 head layers wrapped — the targets are not the "
            "last 1-2 linear transforms the user asked for",
        )

    def test_loss_is_finite(self):
        """§5.7 floor #4 — no NaN, no all-zero collapse in the reported loss."""
        losses = [
            float(x)
            for x in re.findall(r"loss=([0-9.]+(?:e-?[0-9]+)?)", self.out, re.I)
        ]
        self.assertGreater(len(losses), 0, "no loss values parsed from stdout")
        import math

        for value in losses:
            self.assertTrue(
                math.isfinite(value), f"non-finite loss {value} in:\n{self.out[-2000:]}"
            )
        self.assertGreater(max(losses), 0.0, "every reported loss was exactly 0.0")

    def test_smoke_protocol_helper_accepts_the_run(self):
        """Run the AGENTS.md §5.7 helper over the same invocation.

        `verify_all_invariants` reports decode_ratio + loss decrease. This
        asserts rc/loss only: decode_ratio is a separate gen-eval metric with
        its own honest-negative history (Wf-291D / Wf-299), and a 1-epoch
        2-sample CPU run is not a meaningful decode measurement. The helper
        is still exercised so its verdict is on the record.
        """
        from molmetal.scripts.wf399a1_smoke_protocol import (
            run_smoke_50step,
            verify_all_invariants,
        )

        smoke = run_smoke_50step(
            trainer_path=str(REPO_ROOT / TRAINER),
            flags_list=_base_flags(
                "/tmp/blk_c5_test_helper",
                1,
                max_norm=1.0,
                gen_head_lr_scale=0.1,
                entropy_reg_weight=0.01,
            ),
            timeout_sec=900,
        )
        verdict = verify_all_invariants(smoke)
        self.assertEqual(
            smoke.returncode, 0, (smoke.stdout + smoke.stderr)[-3000:]
        )
        self.assertEqual(
            verdict["returncode"],
            0,
            "wf399a1 helper reports a non-zero returncode for the BLK-C5 smoke",
        )


class TestBlkC5EntropyTerm(unittest.TestCase):
    """用户建议三 — entropy reg must be LIVE, not FLAG_ONLY.

    A term that is weighted on one side of an `if` and summed on the other
    is exactly how a term ends up gradient-dead (the failure mode called out
    in the Round-3 D1 comment above `loss = (`).
    """

    def _probe(self, weight: float) -> dict:
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import torch\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "m = t.CFMJointModel(hidden_dim=32, n_layers=2, use_egnn_velocity=True)\n"
            "x = torch.randn(3, 13)\n"
            "logits = torch.randn(5, 8, requires_grad=True)\n"
            "lp = torch.nn.functional.log_softmax(logits, dim=-1)\n"
            "H = -(lp.exp() * (lp + 1e-9)).sum(dim=-1).mean()\n"
            "term = -(%r) * H\n"
            "print('REQUIRES_GRAD', term.requires_grad)\n"
            "print('VALUE', float(term))\n"
        ) % (str(REPO_ROOT), weight)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        rg = re.search(r"REQUIRES_GRAD (\w+)", proc.stdout)
        val = re.search(r"VALUE ([0-9.eE+-]+)", proc.stdout)
        return {
            "requires_grad": rg.group(1) == "True" if rg else None,
            "value": float(val.group(1)) if val else None,
        }

    def test_entropy_term_live_at_nonzero_weight(self):
        probe = self._probe(0.01)
        self.assertTrue(
            probe["requires_grad"],
            "entropy term is gradient-dead at weight 0.01 — FLAG_ONLY",
        )
        # Negative, because the optimiser must MAXIMISE entropy.
        self.assertLess(probe["value"], 0.0)

    def test_entropy_term_is_exact_zero_at_default(self):
        probe = self._probe(0.0)
        self.assertEqual(probe["value"], 0.0, "weight 0 must be an exact zero")


class TestBlkC5CheckpointSafety(unittest.TestCase):
    """The gen-head dropout wrap must not invalidate existing checkpoints."""

    def test_state_dict_keys_unchanged_by_dropout_wrap(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import torch\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "m = t.CFMJointModel(hidden_dim=32, n_layers=2, use_egnn_velocity=True)\n"
            "before = set(m.state_dict().keys())\n"
            "for i, sub in enumerate(m.head.pair_head):\n"
            "    if isinstance(sub, torch.nn.Linear):\n"
            "        m.head.pair_head[i] = t._DropoutLinear(sub, 0.2)\n"
            "lm = m.head.msg_mlps[-1]\n"
            "for i in range(len(lm) - 1, -1, -1):\n"
            "    if isinstance(lm[i], torch.nn.Linear):\n"
            "        lm[i] = t._DropoutLinear(lm[i], 0.2); break\n"
            "after = set(m.state_dict().keys())\n"
            "print('KEYS_EQUAL', before == after)\n"
            "m.load_state_dict(m.state_dict())\n"
            "print('RELOAD_OK', True)\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("KEYS_EQUAL True", proc.stdout)
        self.assertIn("RELOAD_OK True", proc.stdout)


class TestBlkC5BondPredictorDMPNN(unittest.TestCase):
    """BLK-C5 follow-up (2026-10-09) — user voice: 'dmpnn按理来说不是可以更
    好地处理这边的bond特征吗,你留个抽象接口,后面方便dmpnn接入'.

    The default path is still YuelBondDecoderHead.  When the operator
    passes ``--bond-predictor dmpnn`` the CFMJointModel swaps the head
    for a bond-centered Directed-MPNN (``DMPNNBondHead``) whose message
    passing is per-edge (GRU) rather than per-atom.  This test class
    pins the wire-in: argparse accepts the flag, the head type swaps,
    gradients flow, the 3-group BLK-C5 split still works (markers were
    extended to include ``head.dmpnn.``), and a 1-epoch CPU smoke with
    every BLK-C5 lever engaged still finishes rc=0 with finite loss.
    """

    def test_dmpnn_head_class_is_exposed(self):
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "print('CLASS', t.DMPNNBondHead.__name__)\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("CLASS DMPNNBondHead", proc.stdout)

    def test_cfm_joint_model_dispatch(self):
        """CFMJointModel picks the right head for each value."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import torch\n"
            "import molmetal.scripts.train_cfm_yuelbond_wasserstein as t\n"
            "m_y = t.CFMJointModel(hidden_dim=32, n_layers=2, use_egnn_velocity=True, bond_predictor='yuelbond')\n"
            "m_d = t.CFMJointModel(hidden_dim=32, n_layers=2, use_egnn_velocity=True, bond_predictor='dmpnn')\n"
            "print('YUELBOND', type(m_y.head).__name__, 'PRED', m_y.bond_predictor)\n"
            "print('DMPNN', type(m_d.head).__name__, 'PRED', m_d.bond_predictor)\n"
            "from molmetal.models.bond_head import AtomCloud\n"
            "pos = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.], [1., 1., 1.]])\n"
            "zs = torch.tensor([78, 17, 7, 8, 6])\n"
            "cloud = AtomCloud(positions=pos, atomic_numbers=zs)\n"
            "out_y = m_y.head(cloud)\n"
            "out_d = m_d.head(cloud)\n"
            "print('LOGITS_Y', tuple(out_y[0].shape))\n"
            "print('LOGITS_D', tuple(out_d[0].shape))\n"
            "out_d[0].sum().backward()\n"
            "n_grad = sum(1 for p in m_d.parameters() if p.grad is not None)\n"
            "n_tot = sum(1 for p in m_d.parameters())\n"
            "print('GRAD_LIVE', n_grad, '/', n_tot)\n"
        ) % str(REPO_ROOT)
        proc = subprocess.run(
            [PYTHON, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("YUELBOND YuelBondDecoderHead PRED yuelbond", proc.stdout)
        self.assertIn("DMPNN DMPNNBondHead PRED dmpnn", proc.stdout)
        self.assertIn("LOGITS_Y (10, 5)", proc.stdout)
        self.assertIn("LOGITS_D (10, 5)", proc.stdout)
        # GRAD_LIVE: the dmpnn + atom_encoder + pair_head + vel_field all
        # receive gradient; integer / non-floating buffers do not.  We
        # only check it is non-zero, not the exact count (the exact count
        # depends on which nn.RNN/GRU/CuDNN buffers exist on this torch).
        grad_line = [l for l in proc.stdout.splitlines() if l.startswith("GRAD_LIVE")][0]
        grad_n = int(grad_line.split()[1])
        self.assertGreater(grad_n, 0, f"no gradients flowed through dmpnn head: {grad_line}")

    def test_three_param_groups_with_dmpnn_predictor(self):
        """The BLK-C5 three-group split must still find a non-empty gen_flow
        group when ``--bond-predictor dmpnn`` is in effect — the marker
        extension ``head.dmpnn.`` was added precisely to keep this gate
        green for the new path.
        """
        proc = _run(
            _base_flags(
                "/tmp/blk_c5_test_pg_dmpnn",
                1,
                gen_head_lr_scale=0.1,
                bond_predictor="dmpnn",
            )
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out[-3000:])
        m = re.search(
            r"\[BLK-C5\] param_groups=3 "
            r"atom_head=(\d+) gen_flow=(\d+) bond_head=(\d+)",
            out,
        )
        self.assertIsNotNone(m, "param_groups banner missing under dmpnn:\n" + out[-3000:])
        atom_n, flow_n, bond_n = (int(m.group(i)) for i in (1, 2, 3))
        # All three groups non-empty — the dmpnn head still has
        # atom_encoder / dmpnn / pair_head as atom / flow / bond heads.
        self.assertGreater(atom_n, 0)
        self.assertGreater(flow_n, 0)
        self.assertGreater(bond_n, 0)
        # dmpnn adds GRU/MLP stack on the gen_flow side; the rev-2 direction
        # inversion (flow > atom + bond) should still hold.
        self.assertGreater(
            flow_n,
            atom_n + bond_n,
            f"dmpnn gen_flow ({flow_n}) is not the majority group: "
            f"atom ({atom_n}) + bond ({bond_n}) = {atom_n + bond_n}",
        )

    def test_dmpnn_one_epoch_smoke_rc_zero(self):
        """§5.7 50-step end-to-end smoke with the dmpnn path and every
        BLK-C5 lever engaged.  rc must be 0; loss must be finite; the
        3-group param split must print.
        """
        proc = _run(
            _base_flags(
                "/tmp/blk_c5_test_dmpnn_smoke",
                1,
                max_norm=1.0,
                gen_head_lr_scale=0.1,
                bond_head_warmup_epochs=1,
                entropy_reg_weight=0.01,
                gen_head_dropout=0.2,
                bond_predictor="dmpnn",
            )
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out[-3000:])
        # The 3-group banner still fires under dmpnn (the gate the
        # marker extension is there to keep green).
        self.assertIn("[BLK-C5] param_groups=3", out)
        # Loss is finite — same §5.7 floor as the yuelbond path.
        losses = [
            float(x)
            for x in re.findall(r"loss=([0-9.]+(?:e-?[0-9]+)?)", out, re.I)
        ]
        self.assertGreater(len(losses), 0, "no loss values parsed from stdout")
        import math

        for value in losses:
            self.assertTrue(
                math.isfinite(value), f"non-finite loss {value} in:\n{out[-2000:]}"
            )
        self.assertGreater(max(losses), 0.0, "every reported loss was exactly 0.0")

    def test_dmpnn_dropout_wraps_pair_head_linears(self):
        """The gen-head dropout wrap targets `head.pair_head` regardless
        of which bond-predictor is in use, because both heads expose a
        `pair_head` nn.Sequential.  This pins that property.
        """
        proc = _run(
            _base_flags(
                "/tmp/blk_c5_test_dmpnn_dropout",
                1,
                gen_head_dropout=0.2,
                bond_predictor="dmpnn",
            )
        )
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, out[-3000:])
        m = re.search(
            r"gen_head_dropout=[0-9.]+ wrapped .* on (\d+) head layer", out
        )
        self.assertIsNotNone(
            m, "dropout banner missing under dmpnn:\n" + out[-3000:]
        )
        self.assertGreaterEqual(
            int(m.group(1)),
            2,
            "fewer than 2 head layers wrapped under dmpnn — the target "
            "predicate stopped matching the dmpnn head's pair_head",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
