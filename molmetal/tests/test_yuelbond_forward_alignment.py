"""One forward, and an atom encoded as itself.

Two failures that only show up under ``--yuelbond-vocab all6``, which is the
vocabulary every shipped CFM config uses.

1. The trainer used to install ``_device_safe_yuelbond_forward`` over
   ``YuelBondDecoderHead.forward`` at import time.  Its comment claimed it
   "mirrors the original 1:1"; it did not — it also changed the model's maths
   (ReLU, bidirectional aggregation, residual).  Every entry point imported the
   trainer, so every checkpoint was trained under the patch while the library
   advertised something else.  The module is now aligned to the patch, so
   aligning is a no-op for existing weights.

2. That same patch replaced ``z_to_onehot`` with a ``searchsorted`` over the
   z-table.  ``searchsorted`` requires a SORTED table.  ``get_z_table("all6")``
   is ``(..., 53, 78, 44, 77, 45, 76, 75)`` — the five extra metals are
   APPENDED, so it is not sorted, and every metal was silently encoded as a
   different element:

       Z=78 (Pt) -> column 17, which holds 75 (Ir)
       Z=44 (Ru) -> column 11, which holds 53 (I)
       Z=77 (Ir) -> column 16, which holds 76 (Os)

   For a project whose entire subject is six precious metals, that is the
   atom encoder learning that Pt is Ir.  The dict lookup in the library is
   order-preserving and correct; the tests below pin it.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECODER = (
    PROJECT_ROOT
    / "molmetal"
    / "molmetal_lam"
    / "lam_chem"
    / "yuelbond_decoder.py"
)
TRAINER = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"

_TRAINER_SRC = TRAINER.read_text()
_DECODER_SRC = DECODER.read_text()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The atom encoder must encode the atom that is actually there
# ---------------------------------------------------------------------------
def _onehot(z, table):
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import z_to_onehot

    import torch

    return z_to_onehot(torch.tensor(z, dtype=torch.long), tuple(table))


def test_z_to_onehot_encodes_each_metal_as_itself():
    """A metal must land in ITS OWN column, for every vocabulary.

    This is the whole point of the `all6` preset.  Under the removed
    `searchsorted` patch every metal decoded to a different element, so the
    encoder could not tell Pt from Ir from Ru.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    for name in ("pt", "all6"):
        table = get_z_table(name)
        for z in table:
            if z == 0:
                continue
            out = _onehot([z], table)
            col = int(out[0].argmax())
            assert table[col] == z, (
                f"vocab {name}: Z={z} encoded as column {col}, which holds "
                f"Z={table[col]}. A metal must be encoded as itself."
            )


def test_all6_vocab_is_not_sorted_so_searchsorted_is_unusable():
    """State the trap explicitly.

    `searchsorted` needs a sorted table.  `all6` appends the five extra metals
    to the Pt table, so it is not sorted — the precondition the removed patch
    silently violated.  If someone "optimises" `z_to_onehot` with searchsorted
    again, this is the test that should stop them.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    all6 = list(get_z_table("all6"))
    assert all6 != sorted(all6), (
        "all6 is now sorted, so the searchsorted trap no longer applies and "
        "this test's premise has changed — re-check before deleting it"
    )
    assert list(get_z_table("pt")) == sorted(get_z_table("pt")), (
        "the Pt table IS sorted, which is why the bug was invisible there and "
        "only appeared on all6"
    )


def test_no_searchsorted_in_the_decoder():
    src = _DECODER_SRC
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "searchsorted" not in code, (
        "searchsorted over z_table requires a sorted table; the all6 preset "
        "is not sorted and every metal would be encoded as another element"
    )


# ---------------------------------------------------------------------------
# One definition, not two
# ---------------------------------------------------------------------------
def test_trainer_no_longer_replaces_the_library_forward():
    """The monkey-patch must stay gone.

    Two implementations of the same forward is what let the library and the
    trainer disagree for as long as they did, with a comment asserting they
    were identical.
    """
    code = "\n".join(
        ln for ln in _TRAINER_SRC.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "YuelBondDecoderHead.forward =" not in code, (
        "the trainer is monkey-patching the library's forward again; there "
        "must be ONE definition"
    )
    assert "_device_safe_yuelbond_forward" not in code
    assert "_device_safe_z_to_onehot" not in code


def test_importing_the_trainer_does_not_mutate_the_library_class():
    """The strongest form: import it and check the class is untouched.

    A source-level grep can be satisfied by a comment; this cannot be.
    """
    pytest.importorskip("rdkit")
    from molmetal.molmetal_lam.lam_chem import yuelbond_decoder as yb

    before = yb.YuelBondDecoderHead.forward
    _load(TRAINER, "_t_align_probe")
    after = yb.YuelBondDecoderHead.forward
    assert before is after, (
        "importing the trainer replaced YuelBondDecoderHead.forward; the "
        "library must be the single definition"
    )


def test_forward_docstring_records_the_alignment_decision():
    """The choice is load-bearing: the other direction invalidates weights.

    Going the other way (module -> original) would change the maths every
    shipped checkpoint was trained under.  That reasoning has to live in the
    code, not only in a commit message.
    """
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    doc = inspect.getdoc(YuelBondDecoderHead.forward) or ""
    assert "relu" in doc.lower() or "ReLU" in doc, (
        "the forward's docstring must state which message-passing variant it "
        "implements"
    )


# ---------------------------------------------------------------------------
# The three steps of the update itself
# ---------------------------------------------------------------------------
def _message_passing_source() -> str:
    """The loop body, with comments stripped, from the LIBRARY forward."""
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )

    src = inspect.getsource(YuelBondDecoderHead.forward)
    body = src[src.index("for k in range(self.n_layers):"):]
    body = body[: body.index("pair_in = torch.cat")]
    code = []
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            continue
        code.append(ln)
    return "\n".join(code)


def test_forward_aggregates_each_edge_in_both_directions():
    """Standard MPNN: an edge informs both its endpoints.

    The removed module version scattered into ``i_idx`` only, so an atom's
    state depended on its index rather than on its chemistry.
    """
    body = _message_passing_source()
    assert "index_add_(" in body, "aggregation is missing entirely"
    assert body.count("index_add_(") >= 2, (
        "both endpoints must receive the message; only "
        f"{body.count('index_add_(')} scatter site(s) found"
    )
    assert "i_idx" in body and "j_idx" in body, (
        "the two scatter sites must target i_idx and j_idx"
    )


def test_forward_applies_relu_and_a_residual():
    body = _message_passing_source()
    assert "relu(msg)" in body.replace("torch.relu", "relu"), (
        "messages must be passed through a ReLU before aggregation"
    )
    assert "h = h +" in body or "h + torch" in body, (
        "the update must be residual; without it the representation is "
        "recomputed from scratch each layer"
    )


def test_forward_is_se3_invariant_in_the_aggregated_part():
    """A rotation must not change the per-edge features beyond the distance.

    Cheap but meaningful: the message is built from h, the distance d, and
    scalar flags — all rotation-invariant — so only `d` may depend on the
    embedding.  If a coordinate ever leaked into the message, this breaks.
    """
    import torch

    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        YuelBondDecoderHead,
    )
    from molmetal.models.bond_head import AtomCloud

    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    head.eval()
    n = 7
    pos = torch.randn(n, 3) * 2.0
    z = torch.tensor([78, 7, 7, 17, 17, 8, 6])
    a, _ = head(AtomCloud(positions=pos, atomic_numbers=z))

    # Rotate about z by 37 degrees.
    th = torch.tensor(0.6458)
    c, s = torch.cos(th), torch.sin(th)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    b, _ = head(AtomCloud(positions=pos @ R.T, atomic_numbers=z))

    if a.shape != b.shape:
        pytest.skip("rotation changed the edge set; nothing to compare")
    assert torch.allclose(a, b, atol=1e-4), (
        "rotating the input changed the logits — the message passing is not "
        "SE(3)-invariant"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
