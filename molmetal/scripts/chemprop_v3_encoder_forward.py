"""chemprop v3 D-MPNN encoder forward pass -> 300-d per-molecule embedding.

Per `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` §2.2/§2.3 (Path B): the
spec'd script that maps

    SMILES -> chemprop v3 D-MPNN encoder -> (B, 300) per-molecule embedding

which is the input stage of the Path B remediation
(`SMILES -> chemprop v3 encoder (300-d) -> per-cell one-hot -> W2 best-axis
-> Hilbert primal-dual regression -> per-cohort R2`).

What this module provides (CPU-runnable today)
----------------------------------------------
- ``ChempropV3EncoderForward``: wraps
  :class:`molmetal.models.chemprop_v2_mt_model_v3.ChempropV2MTModelV3`'s
  shared :class:`molmetal.models.chemprop_v2_mt_model.ChempropV2Encoder`
  backbone and exposes a clean ``forward(smiles_list) -> (B, hidden_dim)``
  encoder-only path (no per-metal heads).  ``hidden_dim`` defaults to 300
  per the plan.
- ``validate_smiles_batch`` / ``validate_csv_schema``: loud input
  validation (SMILES parse, CSV schema, non-empty batch) so a malformed
  cohort fails fast rather than silently producing shape-wrong embeddings.
- ``encode_smiles_file``: read a SMILES column (default the on-disk
  ``molmetal/data/platinai_4cell.csv``) and encode a cohort to a
  ``(N, 300)`` array.
- ``dump_encoder_npy``: write the ``*_encoder_300d.npy`` artifact named in
  plan §3.2.

Honest-negative boundary (party-twenty-spirit §3.5)
---------------------------------------------------
This module builds the CODE + input-validation path only.  It does NOT
fabricate encoder outputs.  Two plan-§2.3 sub-items are GPU/USER_OK-gated
and deliberately OUT of scope here:

  1. The **~28 GB VRAM peak** full-cohort forward.  The forward path here
     is device-agnostic (``--device cpu|cuda``); the production run needs a
     GPU (Blackwell sm_120 98 GB or RTX 5090 32 GB) and ``USER_OK=1`` per
     AGENTS.md §5.1.
  2. **Backprop through message-passing for training** (plan §2.3 item 1).
     This module is a forward encoder; training-glue is the trainer's job.
     The forward path DOES support ``loss.backward()`` (autograd flows
     through the message-passing layers) — proven by the ``--smoke`` grad
     check — but no optimizer loop is run here.

NOTE on wrapper mismatch (surfaced in the scope): the plan's §4.1 wrapper
`scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` invokes
`train_dmpnn_multitask_wasserstein.py`, NOT this script — the plan's §2.3
script and §4.1 wrapper point at two different code paths.  This module is
therefore a standalone encoder utility (spec-only, no live caller), built to
the plan's §2.2/§2.3 interface so the CPU forward + shape + grad-flow can be
verified today.

Smoke: ``.venv/bin/python -m molmetal.scripts.chemprop_v3_encoder_forward --smoke``
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_HIDDEN_DIM = 300
ENCODER_ARTIFACT_SUFFIX = "_encoder_300d.npy"

# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Must run BEFORE any torch import: CUDA reads CUDA_VISIBLE_DEVICES
# once, at init.  See molmetal/scripts/_cuda_default.py.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
from _cuda_default import ensure_cuda_visible_devices_default as _cvd

_cvd()
del _cvd
# --- end CUDA default --------------------------------------------


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
@dataclass
class SmilesValidation:
    """Result of a SMILES-batch validation pass.

    ``valid`` holds the parseable SMILES in input order; ``invalid`` holds
    ``(index, smiles, reason)`` triples.  ``valid`` is MEASURED (RDKit parse).
    """

    n_input: int = 0
    valid: List[str] = field(default_factory=list)
    invalid: List[tuple] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def ok(self) -> bool:
        return self.n_input > 0 and not self.invalid


def validate_smiles_batch(smiles_list: Sequence[str]) -> SmilesValidation:
    """Parse-validate a SMILES batch with RDKit (fail-loud, no silent drop).

    Raises:
        ValueError: if the batch is empty (an empty cohort is a caller bug,
            not a silently-encodable input).
    """
    if not smiles_list:
        raise ValueError("smiles_list is empty — nothing to encode")
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out = SmilesValidation(n_input=len(smiles_list))
    for i, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            out.invalid.append((i, smi, "empty/non-string SMILES"))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            out.invalid.append((i, smi, "RDKit parse returned None/empty"))
            continue
        out.valid.append(smi)
    return out


def validate_csv_schema(csv_path: Path, smiles_column: str = "smiles") -> List[str]:
    """Read a cohort CSV and return the SMILES column (validated non-empty).

    Raises:
        FileNotFoundError: if ``csv_path`` is absent.
        ValueError: if the header lacks ``smiles_column`` or no rows load.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"cohort CSV not found: {csv_path}")
    smiles: List[str] = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or smiles_column not in reader.fieldnames:
            raise ValueError(
                f"CSV {csv_path} header {reader.fieldnames} missing "
                f"required column {smiles_column!r}"
            )
        for row in reader:
            val = row.get(smiles_column, "")
            if val:
                smiles.append(val)
    if not smiles:
        raise ValueError(f"CSV {csv_path} yielded 0 rows in column {smiles_column!r}")
    return smiles


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------
class ChempropV3EncoderForward:
    """chemprop v3 D-MPNN encoder forward -> (B, hidden_dim) embeddings.

    Thin, encoder-only wrapper around the shared
    :class:`ChempropV2Encoder` backbone used by
    :class:`ChempropV2MTModelV3`.  No metal heads are built or run, so the
    output is the pure per-molecule 300-d representation the Path B
    pipeline consumes (plan §2.2).

    Autograd flows through the message-passing layers, so
    ``loss = f(embeds); loss.backward()`` works (verified by ``--smoke``).

    Args:
        hidden_dim: encoder width (default 300 per plan §2.3).
        depth: message-passing iterations (default 3).
        dropout: encoder dropout (default 0.0).
        device: ``"cpu"`` (default) or ``"cuda"``.  cuda requires
            ``USER_OK=1`` per AGENTS.md §5.1.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        depth: int = 3,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested but torch.cuda.is_available() is False"
            )
        self.device = device
        self.hidden_dim = int(hidden_dim)
        from molmetal.models.chemprop_v2_mt_model import ChempropV2Encoder

        self.encoder = ChempropV2Encoder(
            hidden_dim=self.hidden_dim, depth=int(depth), dropout=float(dropout)
        ).to(device)

    def parameters(self):
        return self.encoder.parameters()

    def forward(self, smiles_list: Sequence[str]) -> "Any":
        """Encode SMILES -> ``(B, hidden_dim)`` torch tensor.

        Raises:
            ValueError: on an empty batch or any un-parseable SMILES (loud,
                not silent — a bad row must not become a zero row).
        """
        validation = validate_smiles_batch(smiles_list)
        if not validation.ok:
            raise ValueError(
                f"SMILES validation failed: {len(validation.invalid)} invalid "
                f"of {validation.n_input} (first: {validation.invalid[0]})"
            )
        emb = self.encoder(list(smiles_list))
        if emb.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"encoder returned dim {emb.shape[-1]} != expected {self.hidden_dim}"
            )
        return emb

    def encode_numpy(
        self, smiles_list: Sequence[str], *, no_grad: bool = True
    ) -> np.ndarray:
        """Encode SMILES -> ``(B, hidden_dim)`` float32 numpy array."""
        import torch

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            emb = self.forward(smiles_list)
        return emb.detach().to("cpu").numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Cohort encoding + artifact dump
# ---------------------------------------------------------------------------
def encode_smiles_file(
    csv_path: Path = DEFAULT_CSV,
    smiles_column: str = "smiles",
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    device: str = "cpu",
) -> np.ndarray:
    """Read a cohort CSV and encode its SMILES column to ``(N, hidden_dim)``.

    Honest-negative: on a GPU-blocked host this still runs on CPU.  It does
    NOT perform the plan's full-cohort ~28 GB VRAM forward — pass a small
    slice or run on a GPU host for the production artifact.
    """
    smiles = validate_csv_schema(csv_path, smiles_column=smiles_column)
    enc = ChempropV3EncoderForward(hidden_dim=hidden_dim, device=device)
    return enc.encode_numpy(smiles)


def dump_encoder_npy(
    out_dir: Path, cohort_name: str, embeddings: np.ndarray
) -> Path:
    """Write ``<cohort>_encoder_300d.npy`` (plan §3.2 artifact name)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cohort_name}{ENCODER_ARTIFACT_SUFFIX}"
    np.save(path, embeddings)
    return path


# ---------------------------------------------------------------------------
# CPU smoke
# ---------------------------------------------------------------------------
def run_smoke(n_samples: int = 8, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> Dict[str, Any]:
    """CPU smoke: forward + shape + autograd + validation, exits nonzero on failure.

    Proves, on a tiny synthetic batch, that:
      1. the encoder imports + runs on CPU with no GPU,
      2. output shape is ``(B, hidden_dim)``,
      3. autograd flows through the message-passing layers
         (``.grad`` is populated after ``backward()``),
      4. validation rejects a malformed SMILES loudly.

    Returns a result dict; raises AssertionError on any failed check.
    """
    import torch

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C", "c1ccncc1",
              "CCN(CC)CC", "O=C(O)c1ccccc1O", "C1CCCCC1"][:n_samples]
    result: Dict[str, Any] = {"n_samples": len(smiles), "hidden_dim": hidden_dim}

    enc = ChempropV3EncoderForward(hidden_dim=hidden_dim, device="cpu")

    # 1 + 2: shape check (grad enabled so we can check autograd next).
    emb = enc.forward(smiles)
    assert tuple(emb.shape) == (len(smiles), hidden_dim), (
        f"shape mismatch: got {tuple(emb.shape)} expected "
        f"{(len(smiles), hidden_dim)}"
    )
    result["output_shape"] = list(emb.shape)

    # 3: autograd flow through message passing.
    loss = emb.pow(2).mean()
    loss.backward()
    grad_params = [
        p for p in enc.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(grad_params) > 0, (
        "autograd did not reach any encoder parameter — backward through "
        "message-passing is broken"
    )
    result["n_params_with_grad"] = len(grad_params)

    # 4: validation rejects a malformed SMILES loudly.
    rejected = False
    try:
        enc.forward(["CCO", "this is not a smiles"])
    except ValueError:
        rejected = True
    assert rejected, "validation failed to reject an invalid SMILES"
    result["validation_rejects_invalid"] = True

    result["status"] = "SMOKE_PASS"
    result["honest_framing"] = (
        "SMOKE-ONLY: proves the CPU 300-d forward path imports, shapes, and "
        "backprops. The plan §2.3 full-cohort ~28 GB VRAM forward + training "
        "are GPU/USER_OK-gated and NOT run here. No numbers fabricated."
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "chemprop v3 D-MPNN encoder forward -> 300-d per-molecule "
            "embeddings (plan 376a3_m2_r2 §2.2/§2.3). CPU smoke by default."
        )
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run the CPU forward+shape+autograd smoke and exit.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Cohort CSV (default platinai_4cell.csv).")
    parser.add_argument("--smiles-column", type=str, default="smiles")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["cpu", "cuda", "auto"]
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Encode at most N rows (0 = all).")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "chemprop_v3_encoder")
    parser.add_argument("--cohort-name", type=str, default="platinai_4cell")
    args = parser.parse_args(argv)

    if args.smoke:
        res = run_smoke(hidden_dim=args.hidden_dim)
        for k, v in res.items():
            print(f"  {k}: {v}")
        print(f"[chemprop_v3_encoder_forward] SMOKE rc=0 status={res['status']}")
        return 0

    smiles = validate_csv_schema(args.csv, smiles_column=args.smiles_column)
    if args.limit > 0:
        smiles = smiles[: args.limit]
    if args.device == "auto":
        from molmetal.molmetal_lam.sbdd_env.training_utils import resolve_default_device

        device = str(resolve_default_device())
    else:
        device = args.device
    enc = ChempropV3EncoderForward(
        hidden_dim=args.hidden_dim, device=device
    )
    emb = enc.encode_numpy(smiles)
    path = dump_encoder_npy(args.out_dir, args.cohort_name, emb)
    print(
        f"[chemprop_v3_encoder_forward] encoded {len(smiles)} SMILES -> "
        f"{tuple(emb.shape)} written to {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
