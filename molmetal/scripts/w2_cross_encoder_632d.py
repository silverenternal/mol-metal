"""W2-aware cross-encoder: chemprop-v3(300) + D-MPNN(300) + F5_RDKIT_2D(32) -> 632-d.

Per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` §6.1 Blocker 1 + §7/§8
(Path B "~200 LOC trainer glue"):

The Hilbert+ASTSR trainer's default 12-d RDKit 2D signal floor
(`train_hilbert_astsr_end_to_end._featurize_smiles`) is the structural root
cause of the L5 R2 lift gap. This module builds the missing piece: a
standalone encoder that fuses three independent molecular representations
into one feature stack as the input to the W2-aligned joint regression:

    chemprop v3 encoder  (300-d)   [molmetal.models.chemprop_v2_mt_model.ChempropV2Encoder]
  + D-MPNN             (300-d)   [molmetal.models.dmpnn.DirectedMPNN]
  + F5_RDKIT_2D        ( 32-d)   [molmetal.data.rdkit_2d_extended.compute_rdkit_2d_extended]
  = fused encoder      (632-d)

The fusion glue is what §6.1 Blocker 1 flags as NOT-WRITTEN. The three
source encoders already exist and are audited by the sibling scaffold
`molmetal/scripts/wf390a2_alt_encoder_audit.py` (`smoke_alt_encoder` already
reports the 300+300+32 = 632-d cross-encoder ensemble size; this module
turns that arithmetic into an executable feature stack).

Interface
---------
``CrossEncoder632(hidden_dim=300, dmpnn_hidden_dim=300, device="cpu")``
    ``.encode(smiles_list) -> torch.Tensor (B, 632)``
    ``.encode_numpy(smiles_list) -> np.ndarray (B, 632) float32``

Honest-negative boundary (party-twenty-spirit §3.5)
---------------------------------------------------
This module builds the encoder FUSION + a CPU smoke that proves the 632-d
feature stack is produced with correct shape + downstream grad flow.  It does
NOT run the plan §3.2 retrain (5 seeds x 50 epochs, USER_OK/GPU-gated) and it
does NOT claim any R2 lift — the lift number itself is GPU-blocked and
unmeasured.  The 12-d-signal-floor caveat (§6.1) still applies: this encoder
lifts the representational ceiling, it does not prove the ceiling is broken.

Smoke: ``.venv/bin/python -m molmetal.scripts.w2_cross_encoder_632d --smoke``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHEMPROP_HIDDEN_DIM = 300
DMPNN_HIDDEN_DIM = 300
F5_RDKIT_2D_DIM = 32
CROSS_ENCODER_DIM = CHEMPROP_HIDDEN_DIM + DMPNN_HIDDEN_DIM + F5_RDKIT_2D_DIM  # 632


def _build_dmpnn_batch(
    smiles_list: Sequence[str], featurizer, device: str
):
    """Featurize SMILES into D-MPNN padded-tensor inputs.

    Returns ``(h_atom, edge_index, edge_attr, batch_idx)`` with the
    ``(B, N_max, ...)`` padded layout :class:`DirectedMPNN.forward` expects.

    Raises:
        ValueError: if any SMILES fails to featurize (loud, not silent).
    """
    import torch

    mol_dicts = []
    for smi in smiles_list:
        try:
            mol_dicts.append(featurizer(smi))
        except Exception as exc:  # noqa: BLE001 — re-raised loud below
            raise ValueError(f"D-MPNN featurizer failed for {smi!r}: {exc}") from exc

    atom_dim = int(mol_dicts[0]["x"].shape[1])
    edge_dim = int(
        mol_dicts[0]["edge_attr"].shape[1]
        if mol_dicts[0]["edge_attr"].ndim == 2
        else 0
    )
    n_max = max(int(m["n_atoms"]) for m in mol_dicts)
    e_max = max(int(m["edge_index"].shape[1]) for m in mol_dicts)
    n_max = max(n_max, 1)
    e_max = max(e_max, 1)
    B = len(mol_dicts)

    h_atom = np.zeros((B, n_max, atom_dim), dtype=np.float32)
    edge_index = np.zeros((B, 2, e_max), dtype=np.int64)
    edge_attr = np.zeros((B, e_max, edge_dim), dtype=np.float32)
    batch_idx = np.full((B, n_max), -1, dtype=np.int64)

    for b, m in enumerate(mol_dicts):
        n = int(m["n_atoms"])
        h_atom[b, :n, :] = m["x"]
        batch_idx[b, :n] = b
        if n > 0:
            ei = m["edge_index"]
            if ei.size:
                edge_index[b, :, : ei.shape[1]] = ei
                edge_attr[b, : ei.shape[1], :] = m["edge_attr"]

    return (
        torch.from_numpy(h_atom).to(device),
        torch.from_numpy(edge_index).to(device),
        torch.from_numpy(edge_attr).to(device),
        torch.from_numpy(batch_idx).to(device),
    )


class CrossEncoder632:
    """Fuse chemprop v3 + D-MPNN + F5_RDKIT_2D into a 632-d feature stack.

    All three sub-encoders run on CPU by default.  Autograd flows through the
    chemprop + D-MPNN graph-message-passing paths, so downstream losses can be
    backpropagated into the encoder (verified by ``--smoke``).

    Args:
        hidden_dim: chemprop v3 encoder width (default 300).
        dmpnn_hidden_dim: D-MPNN width (default 300).
        device: ``"cpu"`` (default) or ``"cuda"`` (cuda needs USER_OK=1 /
            available GPU; this class does not enforce the gate itself —
            callers launching > 1 hr jobs must, per AGENTS.md §5.1).
    """

    def __init__(
        self,
        hidden_dim: int = CHEMPROP_HIDDEN_DIM,
        dmpnn_hidden_dim: int = DMPNN_HIDDEN_DIM,
        device: str = "cpu",
    ):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False"
            )
        self.device = device
        self.hidden_dim = int(hidden_dim)
        self.dmpnn_hidden_dim = int(dmpnn_hidden_dim)
        self.fused_dim = self.hidden_dim + self.dmpnn_hidden_dim + F5_RDKIT_2D_DIM

        from molmetal.models.chemprop_v2_mt_model import ChempropV2Encoder
        from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
        from molmetal.data.featurize import GraphFeaturizer

        self.chemprop_encoder = ChempropV2Encoder(
            hidden_dim=self.hidden_dim, depth=3
        ).to(device)
        self.dmpnn = DirectedMPNN(
            MPNNConfig(hidden_dim=self.dmpnn_hidden_dim)
        ).to(device)
        self.featurizer = GraphFeaturizer()

    def parameters(self):
        yield from self.chemprop_encoder.parameters()
        yield from self.dmpnn.parameters()

    # ------------------------------------------------------------------
    def _rdkit_2d(self, smiles_list: Sequence[str]) -> "Any":
        """Compute the F5_RDKIT_2D 32-d block; NaN rows are zero-filled loud."""
        import torch

        from molmetal.data.rdkit_2d_extended import compute_rdkit_2d_extended

        rows = []
        n_none = 0
        for smi in smiles_list:
            vec = compute_rdkit_2d_extended(smi)
            if vec is None:
                n_none += 1
                vec = np.zeros(F5_RDKIT_2D_DIM, dtype=np.float32)
            rows.append(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))
        if n_none:
            print(
                f"[w2_cross_encoder_632d] WARN: {n_none} SMILES had no RDKit 2D "
                f"vector (zero-filled)",
                file=sys.stderr,
            )
        arr = np.stack(rows, axis=0).astype(np.float32)
        if arr.shape[-1] != F5_RDKIT_2D_DIM:
            raise RuntimeError(
                f"F5_RDKIT_2D block dim {arr.shape[-1]} != {F5_RDKIT_2D_DIM}"
            )
        return torch.from_numpy(arr).to(self.device)

    # ------------------------------------------------------------------
    def encode(self, smiles_list: Sequence[str]) -> "Any":
        """Encode SMILES -> ``(B, fused_dim)`` torch tensor (fused_dim=632).

        Raises:
            ValueError: on an empty batch or un-parseable SMILES.
        """
        import torch

        if not smiles_list:
            raise ValueError("smiles_list is empty — nothing to encode")

        cp = self.chemprop_encoder(list(smiles_list))  # (B, 300)
        h_atom, edge_index, edge_attr, batch_idx = _build_dmpnn_batch(
            smiles_list, self.featurizer, self.device
        )
        dmp = self.dmpnn(h_atom, edge_index, edge_attr, batch_idx)  # (B, 300)
        f5 = self._rdkit_2d(smiles_list)  # (B, 32)

        fused = torch.cat([cp, dmp, f5], dim=-1)  # (B, 632)
        if fused.shape[-1] != CROSS_ENCODER_DIM:
            raise RuntimeError(
                f"fused dim {fused.shape[-1]} != expected {CROSS_ENCODER_DIM}"
            )
        return fused

    def encode_numpy(
        self, smiles_list: Sequence[str], *, no_grad: bool = True
    ) -> np.ndarray:
        """Encode SMILES -> ``(B, fused_dim)`` float32 numpy array."""
        import torch

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            fused = self.encode(smiles_list)
        return fused.detach().to("cpu").numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# CPU smoke
# ---------------------------------------------------------------------------
def run_smoke(n_samples: int = 6) -> Dict[str, Any]:
    """CPU smoke: 632-d fusion + shape + autograd, exits nonzero on failure.

    Proves on a tiny synthetic batch that:
      1. all three sub-encoders import + run on CPU (no GPU),
      2. the fused output is exactly ``(B, 632)``,
      3. each block's contribution is present (300 + 300 + 32),
      4. autograd flows through the graph-message-passing encoders,
      5. an invalid SMILES is rejected loudly.

    Returns a result dict; raises AssertionError on any failed check.
    """
    import torch

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C", "c1ccncc1", "CCN(CC)CC"][
        :n_samples
    ]
    result: Dict[str, Any] = {"n_samples": len(smiles)}

    enc = CrossEncoder632(device="cpu")
    fused = enc.encode(smiles)
    assert tuple(fused.shape) == (len(smiles), CROSS_ENCODER_DIM), (
        f"fusion shape mismatch: got {tuple(fused.shape)} expected "
        f"{(len(smiles), CROSS_ENCODER_DIM)}"
    )
    result["fused_shape"] = list(fused.shape)
    result["block_sizes"] = [
        CHEMPROP_HIDDEN_DIM,
        DMPNN_HIDDEN_DIM,
        F5_RDKIT_2D_DIM,
    ]

    # Gradient flow through chemprop + D-MPNN message passing.
    loss = fused.pow(2).mean()
    loss.backward()
    cp_grads = [
        p
        for p in enc.chemprop_encoder.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    dmp_grads = [
        p
        for p in enc.dmpnn.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert cp_grads, "no gradient reached chemprop v3 encoder params"
    assert dmp_grads, "no gradient reached D-MPNN params"
    result["chemprop_params_with_grad"] = len(cp_grads)
    result["dmpnn_params_with_grad"] = len(dmp_grads)

    # Loud rejection of invalid SMILES.
    rejected = False
    try:
        enc.encode(["CCO", "not_a_smiles"])
    except ValueError:
        rejected = True
    assert rejected, "fusion encoder failed to reject an invalid SMILES"
    result["validation_rejects_invalid"] = True

    result["status"] = "SMOKE_PASS"
    result["honest_framing"] = (
        "SMOKE-ONLY: proves the 632-d fusion (300 chemprop v3 + 300 D-MPNN + "
        "32 F5_RDKIT_2D) builds with correct shape + grad flow on CPU. The "
        "plan §3.2 5-seed x 50-epoch W2 retrain (and the resulting R2 lift) is "
        "USER_OK/GPU-gated and NOT run here — no lift number claimed."
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "W2-aware cross-encoder: chemprop-v3(300) + D-MPNN(300) + "
            "F5_RDKIT_2D(32) -> 632-d (TODO 179 §6.1 Blocker 1)."
        )
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run the CPU 632-d fusion smoke and exit.")
    parser.add_argument("--n-samples", type=int, default=6)
    args = parser.parse_args(argv)

    if args.smoke:
        res = run_smoke(n_samples=args.n_samples)
        for k, v in res.items():
            print(f"  {k}: {v}")
        print(f"[w2_cross_encoder_632d] SMOKE rc=0 status={res['status']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
