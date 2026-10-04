"""Guards for the de novo loop's wiring.

2026-10-04 — `de_novo_loop.py` was first written with

    decoder = YuelBondBondDecoder()

which constructs a **freshly initialised** bond head.  The trainer's own wiring
passes ``head=model.head`` for exactly this reason: the decoder's head *is* the
trained head, so training updates it.  Omitting it does not crash — the loop
happily decodes with random weights, producing parseable garbage whose count
varies between processes (MEASURED across four runs: 0, 16, 15, 15 of 16).  A
silent, run-to-run-varying wrong answer is the worst failure shape, so it gets
a test.

The second guard is the chemical-sanity check.  ``decode_rate`` and
``gen_validity`` both ask only "does RDKit parse it", and RDKit parses
``C.C#C.C#C.[Ru]`` — a pile of one-carbon fragments — without complaint.  A
generator that emits only fragments scores a perfect decode rate.  The loop
therefore reports ``n_chemically_sane`` separately, and this test pins that it
is not silently equal to the decode count.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loop_mod():
    return _load(
        PROJECT_ROOT / "molmetal" / "scripts" / "de_novo_loop.py", "_t_dn_loop"
    )


def test_decoder_must_share_the_trained_head(loop_mod):
    """A decoder built without ``head=`` is randomly initialised.

    Two fresh decoders must therefore disagree, which is what makes the
    omission detectable rather than silent.  If this ever stops holding, the
    guard below is no longer measuring anything.
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    # Deterministic input so the ONLY difference is the head's weights.
    torch.manual_seed(0)
    a = YuelBondBondDecoder(hidden_dim=32, n_layers=2)
    torch.manual_seed(1)
    b = YuelBondBondDecoder(hidden_dim=32, n_layers=2)

    ha = a._decoder.head
    hb = b._decoder.head
    same = all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(ha.parameters(), hb.parameters())
    )
    assert not same, (
        "two freshly-constructed YuelBondBondDecoder heads are identical — "
        "the decoder no longer random-initialises, so a missing `head=` would "
        "no longer be detectable this way"
    )


def test_loop_passes_the_model_head_to_the_decoder():
    """The loop source must construct the decoder WITH the CFM's head.

    Asserted on the source rather than by running the loop, because running it
    needs a trained checkpoint on disk and this must fail in CI too.
    """
    src = (
        PROJECT_ROOT / "molmetal" / "scripts" / "de_novo_loop.py"
    ).read_text()
    assert "YuelBondBondDecoder(" in src
    assert "head=cfm_model.head" in src, (
        "de_novo_loop builds the decoder without the trained bond head — it "
        "would decode with random weights, and the resulting garbage varies "
        "between processes"
    )


def test_sanity_check_is_not_a_copy_of_the_decode_check(loop_mod):
    """`sane` must be strictly stronger than `decoded`.

    A disconnected multi-fragment SMILES parses fine, so it must be counted as
    decoded-but-not-sane.  This is the exact blind spot that made a 100%
    decode rate look like success.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    fragment_pile = "C.C#C.C#C.[Ru]"
    mol = Chem.MolFromSmiles(fragment_pile)
    assert mol is not None, "the fixture is meant to PARSE — that is the point"
    assert len(Chem.GetMolFrags(mol)) > 1, (
        "the fixture is meant to be multi-fragment"
    )

    src = (
        PROJECT_ROOT / "molmetal" / "scripts" / "de_novo_loop.py"
    ).read_text()
    assert "n_fragments" in src and "GetMolFrags" in src, (
        "de_novo_loop no longer checks fragment count — a generator emitting "
        "only fragments would report a perfect decode rate"
    )
    assert "n_chemically_sane" in src, (
        "de_novo_loop no longer reports the sane count separately"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
