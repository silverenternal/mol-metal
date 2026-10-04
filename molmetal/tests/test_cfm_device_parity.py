"""The CUDA-vs-CPU generation gap, and what it actually was.

2026-10-05.  MEASURED on the 26,715-cloud run (results/cfm_tmqm_all6): the
same checkpoint reported ``gen_validity`` 0/128 on CUDA and 28/72 on CPU.

That gap was NOT a modelling result.  ``generate_clouds_from_noise`` returns
``.cpu()`` clouds by design, ``YuelBondDecoderHead.forward`` fed those tensors
straight into a ``nn.Linear`` whose weights were on CUDA, and the forward
raised::

    RuntimeError: Expected all tensors to be on the same device, but got
    mat1 is on cpu, different from other tensors on cuda:0

``YuelBondDecoder.featurise`` catches every exception and returns
``decode_succeeded=False`` — which is the same value the model returns when it
has legitimately produced a molecule nobody can decode.  So the crash was
indistinguishable from a chemistry result, and an entire CUDA run reported 0
for the half of the CFM that exists to produce molecules.

The three device sites that had to be fixed, all of which default to CPU:

* ``z_to_onehot`` — ``torch.zeros(n, dtype=long)`` with no ``device=``
* ``forward`` — ``torch.triu_indices(n, n, offset=1)`` with no ``device=``
* the trainer's ``_device_safe_yuelbond_forward`` monkey-patch, which routed
  device from ``cloud.positions`` rather than from the weights

These tests pin the fixed behaviour.  The parity test is the one that matters:
a device that silently reports zero must never be able to again.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

cuda_required = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="no CUDA device; the parity claim is about CUDA vs CPU",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    # Importing the trainer is what installs the `_device_safe_yuelbond_forward`
    # monkey-patch, so it must be the module under test — testing the library
    # in isolation would miss the third of the three sites.
    return _load(
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py",
        "_t_device_parity",
    )


def _needs_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


needs_rdkit = pytest.mark.skipif(not _needs_rdkit(), reason="rdkit not installed")


# ---------------------------------------------------------------------------
# The one that would have caught it
# ---------------------------------------------------------------------------
@cuda_required
def test_the_library_forward_is_device_safe_on_its_own():
    """The LIBRARY must be device-safe, not merely covered by the monkey-patch.

    MEASURED while fixing this: the trainer installs
    ``_device_safe_yuelbond_forward`` over ``YuelBondDecoderHead.forward`` at
    import time, so any test that imports the trainer cannot tell a correct
    library from a broken one — the patch covers it.  That is precisely why the
    duplication is dangerous, and why the fix cannot be verified through the
    production entry point alone.

    This test therefore imports ``yuelbond_decoder`` WITHOUT the trainer, so it
    exercises the shipped ``forward``.  Verified to fail on the pre-fix code
    (3 failed / 7 passed) and pass after.
    """
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem import yuelbond_decoder as yb

    head = yb.YuelBondDecoderHead(hidden_dim=16, n_layers=1)
    cloud = AtomCloud(
        positions=torch.randn(5, 3), atomic_numbers=torch.tensor([78, 7, 7, 17, 17])
    )
    cpu_logits, (cpu_i, _) = head(cloud)
    head = head.cuda()
    cuda_logits, (cuda_i, _) = head(cloud)

    assert cpu_i.shape == cuda_i.shape
    assert torch.allclose(cpu_logits, cuda_logits.cpu(), atol=1e-4)


@cuda_required
def test_same_cloud_decodes_identically_on_cpu_and_cuda(trainer):
    """A CPU cloud through a CUDA head must give the CUDA head's answer.

    This is the exact shape of the production failure — the sampler emits CPU
    clouds, the model lives on CUDA — and the exact shape of the production
    fix.  Asserting on the LOGITS as well as the SMILES matters: a decode can
    coincidentally agree while the underlying forward is wrong, and the logits
    are what every downstream metric reads.
    """
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    head.eval()
    clouds = [
        AtomCloud(
            positions=torch.randn(6, 3),
            atomic_numbers=torch.tensor([78, 7, 7, 17, 17, 8]),
        )
        for _ in range(5)
    ]

    def _run(device):
        h = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
        h.load_state_dict(head.state_dict())
        h.eval().to(device)
        dec = YuelBondBondDecoder(head=h)
        logits, smiles = [], []
        for c in clouds:
            res = dec.decode(c)
            logits.append(res.yuelbond_result.logits.detach().cpu())
            smiles.append(getattr(getattr(res, "decoded", None), "smiles", "") or "")
        return logits, smiles

    cpu_logits, cpu_smiles = _run("cpu")
    cuda_logits, cuda_smiles = _run("cuda")

    assert len(cpu_logits) == len(cuda_logits) == 5
    for i, (a_t, b_t) in enumerate(zip(cpu_logits, cuda_logits)):
        assert a_t.shape == b_t.shape, f"cloud {i}: edge count differs by device"
        if a_t.numel():
            # float32 atomics make bitwise equality the wrong ask; a tight
            # tolerance still catches a *wrong* computation, which is what a
            # device bug produces (an exception, or a differently-built graph).
            assert torch.allclose(a_t, b_t, atol=1e-4), f"cloud {i}: logits differ"
    assert cpu_smiles == cuda_smiles, "same clouds decoded to different molecules"


@cuda_required
def test_generate_clouds_from_noise_decodes_on_cuda(trainer):
    """End-to-end: the sampler's own output must survive its own decoder.

    `generate_clouds_from_noise` returns `.cpu()` clouds.  If the decoder
    cannot consume those while the model is on CUDA, the CFM's generation half
    is unmeasurable on every GPU — which is where all training happens.
    """
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    model = trainer.CFMJointModel(
        hidden_dim=32,
        n_layers=2,
        cutoff=3.5,
        z_table=get_z_table("all6"),
        use_egnn_velocity=True,
        max_atomic_number=100,
    )
    model.eval().to("cuda")
    dec = YuelBondBondDecoder(head=model.head)
    tpl = [
        AtomCloud(
            positions=torch.randn(6, 3),
            atomic_numbers=torch.tensor([78, 7, 7, 17, 17, 8]),
        )
        for _ in range(3)
    ]
    clouds = trainer.generate_clouds_from_noise(
        model, tpl, torch.device("cuda"), method="euler", n_steps=3, seed=0
    )
    assert len(clouds) == 3
    for c in clouds:
        assert c.positions.device.type == "cpu", "premise: the sampler emits CPU"
        res = dec.decode(c)  # the step that used to raise
        assert res.yuelbond_result is not None


# ---------------------------------------------------------------------------
# The swallowing that hid it
# ---------------------------------------------------------------------------
@needs_rdkit
def test_a_crashed_decode_is_not_reported_as_a_chemistry_result(trainer):
    """gen_validity=0 must be separable from "the decoder never ran".

    The whole bug survived a day because an exception and a bad molecule both
    produced `0`.  `n_decode_error` is the field that tells them apart, so its
    absence — or its absence being *reported as 0 successes* without
    qualification — is the failure being guarded.
    """
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    class _ExplodingDecoder:
        def decode(self, cloud):
            raise RuntimeError("simulated head failure")

    tpl = [
        AtomCloud(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([78, 7, 7, 17, 17]),
        )
        for _ in range(3)
    ]
    model = trainer.CFMJointModel(
        hidden_dim=32,
        n_layers=2,
        z_table=get_z_table("all6"),
        use_egnn_velocity=True,
        max_atomic_number=100,
    )
    out: dict = {}
    n_ok, n_tot = trainer.evaluate_generation_validity(
        model=model,
        decoder=_ExplodingDecoder(),
        templates=tpl,
        device=torch.device("cpu"),
        n_samples=3,
        chemistry_out=out,
    )
    assert (n_ok, n_tot) == (0, 3)
    assert out["n_decode_error"] == 3, "the crash must be counted, not absorbed"
    assert "simulated head failure" in out["decode_error_example"]


@needs_rdkit
def test_a_working_decode_reports_no_errors(trainer):
    """The counter must not fire on a healthy run — or it teaches nothing."""
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
        get_z_table,
    )

    torch.manual_seed(0)
    # A velocity field is required: `evaluate_generation_validity` samples
    # before it decodes, and a model without one has no sampler to run.
    model = trainer.CFMJointModel(
        hidden_dim=32,
        n_layers=2,
        z_table=get_z_table("all6"),
        use_egnn_velocity=True,
        max_atomic_number=100,
    )
    dec = YuelBondBondDecoder(head=model.head)
    tpl = [
        AtomCloud(
            positions=torch.randn(5, 3) * 2.0,
            atomic_numbers=torch.tensor([78, 7, 7, 17, 17]),
        )
        for _ in range(3)
    ]
    out: dict = {}
    trainer.evaluate_generation_validity(
        model=model,
        decoder=dec,
        templates=tpl,
        device=torch.device("cpu"),
        n_samples=3,
        chemistry_out=out,
    )
    assert out["n_decode_error"] == 0
    assert "decode_error_example" not in out


def test_featurise_reports_the_exception_message_not_only_its_type():
    """A bare class name is what made the original unreadable.

    `head_forward_failed:RuntimeError` says nothing about which of the many
    ways a forward can fail happened; the device error's entire text is the
    diagnosis, and it was being thrown away.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoder,
        YuelBondDecoderHead,
    )
    from molmetal.models.bond_head import AtomCloud
    import torch

    class _ExplodingHead(YuelBondDecoderHead):
        def forward(self, cloud):
            raise RuntimeError("mat1 is on cpu, different from other tensors on cuda")

    dec = YuelBondDecoder(head=_ExplodingHead(hidden_dim=8, n_layers=1))
    res = dec.featurise(
        AtomCloud(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([78, 7, 7, 17]),
        )
    )
    assert res.decode_succeeded is False
    assert "mat1 is on cpu" in res.error, "the message must survive"


# ---------------------------------------------------------------------------
# The three device sites
# ---------------------------------------------------------------------------
def test_z_to_onehot_follows_its_input_device():
    """It used to allocate on CPU regardless of `z`'s device."""
    import torch

    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import z_to_onehot

    z = torch.tensor([78, 7, 7, 17], dtype=torch.long)
    assert z_to_onehot(z).device == z.device
    if torch.cuda.is_available():
        assert z_to_onehot(z.cuda()).device.type == "cuda"


@cuda_required
def test_head_forward_handles_a_cpu_cloud_on_cuda_weights():
    """The production shape: CUDA module, CPU cloud."""
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1).cuda()
    cloud = AtomCloud(
        positions=torch.randn(5, 3), atomic_numbers=torch.tensor([78, 7, 7, 17, 17])
    )
    logits, (i_idx, j_idx) = head(cloud)
    assert logits.shape[0] == i_idx.shape[0] == j_idx.shape[0]
    assert logits.device.type == "cuda", "the head answers on its own device"


@cuda_required
def test_triu_indices_are_built_on_the_weight_device():
    """`torch.triu_indices` defaults to CPU; that alone re-raised the bug."""
    import torch

    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    head = YuelBondDecoderHead(hidden_dim=16, n_layers=1).cuda()
    cloud = AtomCloud(
        positions=torch.randn(5, 3), atomic_numbers=torch.tensor([78, 7, 7, 17, 17])
    )
    _, (i_idx, _j) = head(cloud)
    assert i_idx.device.type == "cuda"


@cuda_required
def test_atom_cloud_to_carries_its_bond_truth():
    """A move that dropped the labels would silently unsupervise the cloud."""
    import torch

    from molmetal.models.bond_head import AtomCloud

    c = AtomCloud(
        positions=torch.randn(4, 3),
        atomic_numbers=torch.tensor([78, 7, 7, 17]),
        bonds=torch.tensor([[0, 1], [1, 2]]),
        bond_types=torch.tensor([1, 1]),
    )
    moved = c.cuda()
    assert moved.positions.device.type == "cuda"
    assert moved.bonds is not None and moved.bonds.device.type == "cuda"
    assert moved.bond_types is not None and moved.bond_types.device.type == "cuda"
    assert moved.cuda().positions.device.type == "cuda"
    assert moved.cpu().positions.device.type == "cpu"


def test_atom_cloud_to_preserves_the_original():
    """`.to()` must not mutate in place — callers hold the source cloud."""
    import torch

    from molmetal.models.bond_head import AtomCloud

    c = AtomCloud(
        positions=torch.randn(4, 3), atomic_numbers=torch.tensor([78, 7, 7, 17])
    )
    before = c.positions.clone()
    c.to(torch.device("cpu"))
    assert torch.equal(c.positions, before)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
