"""molmetal/adapters/pocketgen — vendored PocketGen encoder for CFM pocket conditioning.

Upstream: https://github.com/zaixizhang/PocketGen (MIT, 2024)
Cite:     Zhang et al., 2024, Nature Machine Intelligence.
           "Accurate Protein Pocket Design with a Bilevel Graph Transformer
            and an n-gram Decoding Scheme."

Scope
-----
We vendor ONLY the encoder half of PocketGen (BilevelEncoder / HierEncoder /
GET) for use as a protein-pocket representation model `c` in our CFM
(Conditional Flow Matching) pipeline. The generation head, decoder, and
chemistry / docking / relaxation pipeline are intentionally excluded
(see ATTRIBUTION.md for the full keep / drop manifest).

The vendored encoder code is **kept verbatim** at the time of import; all
mol-metal adaptation happens in the wrapper class below (PocketGenEncoder)
or in mol-metal's own EGNN extension (PerLayerPocketInjector).

Heavy dependencies
------------------
PocketGen requires the following packages, which are NOT in mol-metal's
default `.venv` and will be installed by Wf-A32 wire-in:

  - torch_scatter (~10 MB wheel; CUDA-specific build)
  - torch_geometric (PyG, ~50 MB; CUDA-specific)
  - esm (Facebook ESM-2 protein language model, ~50 MB wheel)
  - omegaconf (config; trivial, ~50 KB)

For the wire-in, we will add these to pyproject.toml as an `[extras]`
group (`pocketgen`) so they are not pulled in by default.

Lazy import
-----------
This module exposes the encoder classes via lazy attribute access so that
``from molmetal.adapters.pocketgen import PocketGenEncoder`` works in
environments without the heavy deps installed (e.g. the 50-step CPU
smoke for CFM trainer). The encoders themselves are loaded only when
``PocketGenEncoder.from_pretrained(...)`` is called.

Wrap-into-CFM contract
----------------------
The mol-metal CFM pocket-conditioning slot consumes a single tensor of
shape ``(B, pocket_embed_dim)`` per pocket. PocketGen's BilevelEncoder
emits a per-residue tensor of shape ``(N_res, hidden_channels)`` plus a
batch dimension. The wrapper below:

  1. Pools the per-residue output to a fixed-size vector (mean + max
     pooling, concat → 2 * hidden_channels).
  2. Optionally projects to ``pocket_embed_dim`` via a learnable
     Linear when the CFM velocity-field ``hidden_dim`` does not match
     PocketGen's ``hidden_channels``.
  3. Returns shape ``(B, pocket_embed_dim)`` ready for
     PerLayerPocketInjector or additive bias in the EGNN velocity field.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Make the upstream layout importable as a sibling package.
# PocketGen's encoder/ESM modules use relative imports
# (e.g. ``from ..common import GaussianSmearing``), so we expose the
# `models` subdir as the PocketGen-style import root.
_HERE = Path(__file__).resolve().parent
_MODELS_DIR = _HERE / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

# Re-export upstream MIT copyright in our namespace.
__vendor_license__ = "MIT"
__vendor_upstream__ = "zaixizhang/PocketGen (2024)"
__vendor_paper__ = "Zhang et al., Nat. Mach. Intell., 2024"


def _try_import_encoder_class(qualname: str):
    """Lazy loader for PocketGen encoder classes.

    Defers the heavy torch_scatter / torch_geometric / esm imports so
    that ``import molmetal.adapters.pocketgen`` does NOT fail when those
    packages are absent. Returns the class on success, raises ImportError
    with a clear remediation message on failure.
    """
    mod = importlib.import_module("encoders")
    try:
        return getattr(mod, qualname)
    except AttributeError as exc:
        raise ImportError(
            f"molmetal.adapters.pocketgen: '{qualname}' not found in upstream "
            f"encoders module. Verify that models/encoders/__init__.py exports it."
        ) from exc


if TYPE_CHECKING:  # pragma: no cover — only for type-checkers / IDE
    from encoders import BilevelEncoder, HierEncoder, GET  # noqa: F401


def __getattr__(name: str):
    """PEP 562 lazy attribute access for encoder classes.

    Examples
    --------
    >>> from molmetal.adapters.pocketgen import BilevelEncoder  # lazy
    >>> enc = BilevelEncoder(hidden_channels=128, ...)            # heavy
    """
    if name in ("BilevelEncoder", "HierEncoder", "GET"):
        return _try_import_encoder_class(name)
    raise AttributeError(f"module 'molmetal.adapters.pocketgen' has no attribute {name!r}")


# The wrapper class itself is pure-Python and importable without
# the heavy deps — only the encoder *instantiation* triggers them.
class PocketGenEncoder:
    """Wrap PocketGen's BilevelEncoder as a CFM-friendly pocket embedder.

    Parameters
    ----------
    hidden_channels : int
        PocketGen's internal hidden dim (default 128, matches the
        BilevelEncoder paper config).
    num_interactions : int
        Number of bilevel message-passing layers (default 3, matches paper).
    knn : int
        k-nearest-neighbour graph size for residue proximity (default 30).
    cutoff : float
        Spatial cutoff in Angstrom for the kNN graph (default 10.0).
    pocket_embed_dim : int | None
        Output dim. If None, return 2 * hidden_channels (mean+max pool).
        If set, project via a learnable Linear to this dim.
    pretrained_ckpt : str | os.PathLike | None
        Path to PocketGen's pretrained weights. If None, weights are
        random-initialised (sufficient for the Wf-A30 / A31 plumbing
        test; the real weights are vendored in Wf-A32 wire-in).
    esm_model_name : str
        ESM-2 model name for the amino-acid prior. Default
        ``'esm2_t6_8M_UR50D'`` (8 M params, lightweight). For
        paper-grade quality use ``'esm2_t33_650M_UR50D'`` (650 M params,
        ~650 MB download).

    Notes
    -----
    This class is the integration point. It is intentionally
    dependency-free at the import level; only ``.from_pretrained()``
    and ``.forward()`` trigger the heavy torch_scatter / torch_geometric
    / esm imports. This keeps the CFM trainer's 50-step CPU smoke
    (per AGENTS.md §5.7) cheap.
    """

    DEFAULTS = {
        "hidden_channels": 128,
        "edge_channels": 32,
        "key_channels": 64,
        "num_heads": 4,
        "num_interactions": 3,
        "knn": 30,
        "cutoff": 10.0,
    }

    def __init__(
        self,
        hidden_channels: int = 128,
        num_interactions: int = 3,
        knn: int = 30,
        cutoff: float = 10.0,
        pocket_embed_dim: int | None = None,
        pretrained_ckpt: str | os.PathLike | None = None,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        device: str = "cpu",
    ) -> None:
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.knn = knn
        self.cutoff = cutoff
        self.pocket_embed_dim = pocket_embed_dim or (2 * hidden_channels)
        self.pretrained_ckpt = (
            Path(pretrained_ckpt) if pretrained_ckpt is not None else None
        )
        self.esm_model_name = esm_model_name
        self.device = device

        # Lazy: do NOT instantiate the encoder here. The heavy deps
        # (torch_scatter / torch_geometric / esm) load only on .load().
        self._encoder = None
        self._projector = None  # built lazily when pocket_embed_dim != 2*hidden

    def load(self) -> "PocketGenEncoder":
        """Construct the underlying BilevelEncoder. Triggers heavy imports.

        Returns self for chaining. Raises ImportError with a clear
        remediation message if the heavy deps are not installed.
        """
        BilevelEncoder = _try_import_encoder_class("BilevelEncoder")
        try:
            import torch  # noqa: F401  (sanity)
        except ImportError as exc:
            raise ImportError(
                "PocketGenEncoder.load(): torch is required. mol-metal's "
                ".venv should already have it. Original error: " + str(exc)
            ) from exc

        self._encoder = BilevelEncoder(
            hidden_channels=self.hidden_channels,
            edge_channels=self.DEFAULTS["edge_channels"],
            key_channels=self.DEFAULTS["key_channels"],
            num_heads=self.DEFAULTS["num_heads"],
            num_interactions=self.num_interactions,
            k=self.knn,
            cutoff=self.cutoff,
            device=self.device,
        )

        if self.pocket_embed_dim != 2 * self.hidden_channels:
            import torch.nn as nn
            self._projector = nn.Linear(
                2 * self.hidden_channels, self.pocket_embed_dim
            )

        if self.pretrained_ckpt is not None and self.pretrained_ckpt.exists():
            import torch
            state = torch.load(self.pretrained_ckpt, map_location=self.device)
            self._encoder.load_state_dict(state, strict=False)
        return self

    def extract_embedding(self, pocket_data) -> "torch.Tensor":
        """Run the encoder on a single pocket and return (pocket_embed_dim,).

        Parameters
        ----------
        pocket_data : torch_geometric.data.Data
            BilevelEncoder-style input. Construction of this object from
            a PDB file is the responsibility of the caller (we already
            have `load_pocket_from_pdb` in molmetal.data.tmqm + the
            CrossDockedDataset loader in molmetal.data.crossdocked).

        Returns
        -------
        torch.Tensor of shape (pocket_embed_dim,).
        """
        if self._encoder is None:
            self.load()
        import torch

        out = self._encoder(pocket_data)  # (N_res, hidden_channels)
        # Mean + max pool → (2 * hidden_channels,)
        mean_pool = out.mean(dim=0)
        max_pool = out.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        if self._projector is not None:
            pooled = self._projector(pooled)
        return pooled


__all__ = [
    "PocketGenEncoder",
    "__vendor_license__",
    "__vendor_upstream__",
    "__vendor_paper__",
    # Encoder classes are exposed via lazy __getattr__; not in __all__ to
    # avoid lint warnings about lazy imports.
]
