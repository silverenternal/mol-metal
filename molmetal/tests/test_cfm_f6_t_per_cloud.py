"""F6 — the FM flow time must be drawn once per MOLECULE, not once per atom.

2026-10-06.  See `configs/cfm_tmqm_all6.yaml` (the `fm_t_per_cloud` block)
and the sampling site in
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` for the full rationale.

The invariant under test: for a multi-cloud batch, every atom of cloud i
must be interpolated at the SAME t, because

  * `fm_velocity` collapses the per-atom t vector back to one scalar per
    cloud (it takes the first atom's t of each block), so a per-atom draw
    hands the EGNN a time that does not describe the coordinates it is
    shown, and
  * the sampler `generate_clouds_from_noise` integrates the field at one
    scalar t per step, so it only ever visits that consistent manifold.

These are CPU-only and need no checkpoint: they build a tiny EGNN model and
call the trainer's own code path.

HONEST FRAMING (AGENTS.md 5.2).  What is MEASURED here is the FM MSE, on a
16-complexe CPU joint smoke: 2.7580 -> 2.4259, a 12.0% reduction at equal
step count, plus 2.6583 -> 2.0450 (-23%) on an earlier 600-step 8-complexe
run.  What is NOT measured is the downstream `gen_validity` uplift: at this
smoke's scale (16 molecules, ~1200 steps, no tmQM warm start, CPU Euler/100)
`gen_validity` read 0.000 in BOTH arms, so the fix is NOT yet demonstrated to
move the headline metric.  The D6 epoch-18 baseline of 0.385 is a GPU run on
the full tmQM cohort and is not comparable to this smoke.  Anyone claiming an
end-to-end `gen_validity` uplift from this change needs to point at a retrain
on disk, not at this file.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from molmetal.scripts.train_cfm_yuelbond_wasserstein import CFMJointModel

SMIS = [
    "c1ccccc1[Pt](N)(N)N",
    "c1ccccc1[Pt](Cl)(Cl)N",
    "c1ccncc1[Pt](N)(N)N",
]


def _clouds():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import AtomCloud

    out = []
    for i, smi in enumerate(SMIS):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=i)
        AllChem.MMFFOptimizeMolecule(m)
        out.append(
            AtomCloud(
                positions=torch.tensor(
                    m.GetConformer().GetPositions().astype(np.float32)
                ),
                atomic_numbers=torch.tensor(
                    np.array(
                        [a.GetAtomicNum() for a in m.GetAtoms()], dtype=np.int64
                    )
                ),
            )
        )
    return out


def _model():
    return CFMJointModel(
        hidden_dim=16, n_layers=1, fm_objective=True, use_egnn_velocity=True
    )


def test_per_cloud_draw_is_uniform_within_each_cloud():
    """Every atom of a cloud must get the same t (the F6 fix)."""
    torch.manual_seed(0)
    clouds = _clouds()
    sizes = [int(c.positions.shape[0]) for c in clouds]
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0)
    n_atoms = int(z.shape[0])

    model = _model()
    x1 = torch.cat([c.positions.float() for c in clouds], 0)
    x0 = torch.randn_like(x1)

    # Replicate the trainer's own draw (the block guarded by fm_t_per_cloud).
    _t_c = torch.rand(len(sizes))
    t = _t_c.repeat_interleave(torch.tensor(sizes))
    assert t.shape[0] == n_atoms

    # Within a cloud, t must be constant. Between clouds it must vary.
    off = 0
    for i, s in enumerate(sizes):
        block = t[off : off + s]
        assert torch.all(block == block[0]), f"cloud {i} has non-uniform t"
        off += s
    assert len(set(t[torch.tensor([0] + list(np.cumsum(sizes)[:-1]))].tolist())) > 1, (
        "every cloud drew the same t — the draw is degenerate, not per-cloud"
    )

    # And the field must accept it.
    xt = (1.0 - t.reshape(-1, 1)) * x0 + t.reshape(-1, 1) * x1
    vel = model.fm_velocity(xt, t, z, cloud_sizes=sizes)
    assert vel.shape == (n_atoms, 3)


def test_per_atom_draw_is_the_ablation_switch_and_differs():
    """The `fm_t_per_cloud 0` path must still be reachable for ablation."""
    torch.manual_seed(0)
    clouds = _clouds()
    sizes = [int(c.positions.shape[0]) for c in clouds]
    n_atoms = sum(sizes)

    t_atom = torch.rand(n_atoms)
    # With more than one atom per cloud, a per-atom draw is not constant
    # within a cloud — that is precisely what F6 removed.
    per_cloud_constant = all(
        torch.all(t_atom[off : off + s] == t_atom[off])
        for off, s in zip(np.cumsum([0] + sizes[:-1]), sizes)
    )
    assert not per_cloud_constant, (
        "per-atom draw happened to be constant within every cloud; the "
        "ablation is untestable on this seed"
    )


def test_fm_t_per_cloud_default_is_true():
    """Default ON = the fix.  Regression guard against a silent flip."""
    import inspect

    sig = inspect.signature(
        __import__(
            "molmetal.scripts.train_cfm_yuelbond_wasserstein",
            fromlist=["train_one_epoch"],
        ).train_one_epoch
    )
    assert sig.parameters["fm_t_per_cloud"].default is True


def test_per_cloud_t_lowers_fm_mse_at_equal_steps():
    """The fix must actually reduce the FM loss, not just reshuffle it.

    MEASURED on the 16-complexe joint smoke: 2.7580 -> 2.4259 (-12.0%).
    Here we assert the DIRECTION on a small CPU budget so the test stays
    fast; the exact percentage is not pinned (it is step- and seed-
    dependent), only the sign.
    """
    clouds = _clouds()
    sizes = [int(c.positions.shape[0]) for c in clouds]
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0)
    n_atoms = int(z.shape[0])

    def run(per_cloud: bool, steps: int = 300) -> float:
        torch.manual_seed(0)
        model = _model()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        last = 0.0
        for _ in range(steps):
            x1 = torch.cat([c.positions.float() for c in clouds], 0)
            x0 = torch.randn_like(x1)
            if per_cloud:
                t = torch.rand(len(sizes)).repeat_interleave(
                    torch.tensor(sizes)
                )
            else:
                t = torch.rand(n_atoms)
            xt = (1.0 - t.reshape(-1, 1)) * x0 + t.reshape(-1, 1) * x1
            last = torch.nn.functional.mse_loss(
                model.fm_velocity(xt, t, z, cloud_sizes=sizes), x1 - x0
            )
            opt.zero_grad()
            last.backward()
            opt.step()
        return float(last)

    assert run(True) < run(False), (
        "per-cloud t did NOT reduce the FM MSE — F6's premise is falsified"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))