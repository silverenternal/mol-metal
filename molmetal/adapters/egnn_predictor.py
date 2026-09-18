"""EGNN-based PropertyPredictor stub.

Goal: predict **binding affinity pIC50** for a (ligand, pocket) pair using
an Equivariant Graph Neural Network (Satorras et al. 2021).  This is the
GPU complement to :class:`molmetal.adapters.rdkit_predictor.RDKitPropertyPredictor`
which only computes 2D descriptors.

This file is a **stub**.  When a pretrained checkpoint is supplied via
``checkpoint_path`` the model is loaded and ``predict()`` returns a
``PropertyPrediction`` with ``binding_affinity_pic50`` filled.  Without
a checkpoint the predictor returns ``None`` for the affinity and leaves
the 2D fields as defaults — useful for orchestrator plumbing tests.

Training script (stub for now) lives at
``molmetal/scripts/train_property_predictor.py`` and will be implemented
in a later milestone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from molmetal.domain import Complex, Molecule
from molmetal.ports import PropertyPrediction, PropertyPredictor
from molmetal.utils.device import get_device


__all__ = ["EGNNPropertyPredictor", "EGNNConfig"]


# ---------------------------------------------------------------------------
# Config (kept simple — fill out when the training script lands)
# ---------------------------------------------------------------------------
class EGNNConfig:
    """Minimal EGNN config used by the predictor stub.

    Attributes
    ----------
    in_node_dim : int
        Input node-feature dimension (default 11: atom-type one-hot +
        formal charge + chirality — to be aligned with the trainer).
    in_edge_dim : int
        Input edge-feature dimension (default 4: bond type one-hot).
    hidden_dim : int
        EGNN message-passing hidden width.
    n_layers : int
        Number of message-passing layers.
    checkpoint_path : str | None
        If set, ``setup()`` loads these weights.  If ``None``, the
        predictor runs in **stub mode** (returns ``None`` pIC50).
    """

    def __init__(
        self,
        in_node_dim: int = 11,
        in_edge_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 5,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.in_node_dim = int(in_node_dim)
        self.in_edge_dim = int(in_edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.checkpoint_path = (
            str(checkpoint_path) if checkpoint_path is not None else None
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
class EGNNPropertyPredictor(PropertyPredictor):
    """EGNN-based PropertyPredictor stub for binding affinity pIC50.

    Parameters
    ----------
    config : EGNNConfig | None
        Optional config.  If ``None``, defaults are used (stub mode).
    """

    name = "EGNNPropertyPredictor_v0"

    def __init__(self, config: Optional[EGNNConfig] = None) -> None:
        self._config = config or EGNNConfig()
        # ROCm PyTorch exposes the CUDA device API; use the project-wide
        # resolver so this adapter follows HIP_VISIBLE_DEVICES and CPU
        # fallback consistently with the rest of the stack.
        self._device = str(get_device())
        self._model = None           # lazily allocated in setup()
        self._checkpoint_loaded = False

    # ------------------------------------------------------------------
    def setup(self, device: Optional[str] = None) -> None:
        """Allocate the EGNN model on ``device`` and load the checkpoint if any."""
        self._device = str(get_device() if device is None else device)
        # Allocate model.  We do NOT require torch_geometric at import
        # time — only at setup.  If torch_geometric isn't installed we
        # fall back to a no-op stub so the orchestrator can still call
        # ``predict()`` and get ``None`` for pIC50.
        try:
            import torch  # noqa: WPS433

            self._model = _build_egnn_stub(
                self._config.in_node_dim,
                self._config.in_edge_dim,
                self._config.hidden_dim,
                self._config.n_layers,
            ).to(self._device)
        except Exception:
            # torch_geometric missing or another import error — stay in stub mode.
            self._model = None

        # Optionally load a pretrained checkpoint.
        if (
            self._config.checkpoint_path is not None
            and self._model is not None
        ):
            ckpt_path = Path(self._config.checkpoint_path)
            if ckpt_path.is_file():
                try:
                    import torch  # noqa: WPS433

                    state = torch.load(ckpt_path, map_location=self._device)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    self._model.load_state_dict(state)
                    self._checkpoint_loaded = True
                except Exception:
                    # Bad checkpoint — leave the model randomly init'd but flag it.
                    self._checkpoint_loaded = False
            else:
                self._checkpoint_loaded = False

    # ------------------------------------------------------------------
    def predict(
        self,
        molecule: Molecule,
        complex: Optional[Complex] = None,  # noqa: A002
    ) -> PropertyPrediction:
        """Predict binding affinity pIC50 for ``molecule``.

        Without a loaded checkpoint, this returns a default
        ``PropertyPrediction`` with ``binding_affinity_pic50 = None``.
        With a checkpoint, the stub returns ``None`` (real inference
        loop is TODO once the training script lands).
        """
        if not self._checkpoint_loaded or self._model is None:
            return PropertyPrediction(
                binding_affinity_pic50=None,
                metal_binding_score=None,
            )

        # TODO(milestone-2): implement forward pass + pIC50 head.
        # The training script is at molmetal/scripts/train_property_predictor.py
        # (currently a stub).  When available, the forward pass will:
        #   1. RDKit-featurize the molecule into (node_x, edge_index, edge_attr)
        #   2. RDKit/BioPython-featurize the pocket (from complex.pocket if given)
        #   3. concatenate the two graphs into a single complex graph
        #   4. mean-pool node embeddings → MLP → pIC50 scalar
        return PropertyPrediction(
            binding_affinity_pic50=None,
            metal_binding_score=None,
        )

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "egnn-stub",
            "device": self._device,
            "config": {
                "in_node_dim": self._config.in_node_dim,
                "in_edge_dim": self._config.in_edge_dim,
                "hidden_dim": self._config.hidden_dim,
                "n_layers": self._config.n_layers,
                "checkpoint_path": self._config.checkpoint_path,
            },
            "checkpoint_loaded": self._checkpoint_loaded,
            "training_script": "molmetal/scripts/train_property_predictor.py",
            "description": (
                "EGNN-based PropertyPredictor stub. Allocates an EGNN on "
                "the requested device and loads a pretrained checkpoint "
                "when available. Currently returns None for pIC50 — "
                "forward pass + training script are TODO in a later "
                "milestone."
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_egnn_stub(in_node_dim: int, in_edge_dim: int, hidden_dim: int, n_layers: int):
    """Construct a minimal EGNN-style module.

    Tries to import :class:`torch_geometric.nn.EGNN`.  If unavailable
    (no torch_geometric install), returns a tiny torch.nn.Module that
    just keeps a parameter buffer so ``load_state_dict`` doesn't crash.
    """
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
    except Exception as e:  # pragma: no cover — torch always present
        raise RuntimeError("EGNN predictor requires torch") from e

    try:
        from torch_geometric.nn import EGNN  # type: ignore  # noqa: WPS433

        return EGNN(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    except Exception:
        # No torch_geometric — return a torch stub with the right state-dict keys.
        class _StubEGNN(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                # A single learnable scalar so load_state_dict with the
                # *real* checkpoint shape will fail loudly (which we
                # catch in setup() and downgrade to "not loaded").
                self.dummy = nn.Parameter(torch.zeros(1))

            def forward(self, *args, **kwargs):  # pragma: no cover
                raise NotImplementedError(
                    "EGNN stub cannot run forward — torch_geometric not installed."
                )

        return _StubEGNN()
