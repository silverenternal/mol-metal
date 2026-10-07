# SPDX-License-Identifier: Apache-2.0
"""Plan A Step 1 — DMPNN pIC50 value adapter for the Lambda MCTS search.

Replaces the hand-crafted discrete binding reward (``bonus_binder`` which
yields leaf values quantised to {0, 0.5, 1.0}) with a *continuous* learned
value channel from the trained multi-task D-MPNN metal-cytotoxicity
regressor (Plan A Step 1, ``TODO/pending/plan_a_step1_dmpnn_value.md``).

Design constraints:

* **Lazy load** — ``torch`` and the checkpoint are only touched on the first
  forward, so importing this module does not slow down the MCTS bench's
  import time and does not pull torch into headless test contexts.
* **Graceful degradation** — every failure path (missing checkpoint, missing
  RDKit, unparseable SMILES, torch exception) returns ``0.0`` rather than
  raising, matching the ``_safe(channel)`` contract that
  :class:`~molmetal_lam.search_alg.proof_search.RewardAggregator` expects
  from every reward channel.
* **Cache** — MCTS re-visits the same leaves many times (n_simulations=100
  over a small molecule space), so canonical-SMILES keyed
  ``functools.lru_cache`` collapses most forward passes.

Honest-negative: the DMPNN checkpoint was trained on *flat* cytotoxicity
cohorts (metal + cell line + pIC50), never on reaction-tree leaves. Leaf
molecules produced by the MCTS are therefore **out of distribution**; the
value channel is an engineering wiring validation, not a metric claim.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "DEFAULT_CKPT",
    "FIVE_METALS",
    "DMPNNValueFn",
    "load_checkpoint",
    "make_value_fn",
    "predict_pic50",
]

#: Default checkpoint.  Ir seed0 is the largest, most-recently-trained
#: checkpoint on disk (MEASURED 2026-10-07: 31.7 MB, mtime Oct 3).
DEFAULT_CKPT = Path("molmetal/checkpoints/dmpnn_mtw_ir_seed0.pt")

#: Metal order matches ``FIVE_METALS`` in
#: ``molmetal/scripts/train_dmpnn_multitask_wasserstein.py``; the ``n``-th
#: entry indexes ``model.cls_heads[n]``.
FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")


class DMPNNValueFn:
    """Callable adapter ``(smiles) -> pIC50`` backed by the trained DMPNN.

    Parameters
    ----------
    ckpt_path:
        Path to a ``train_dmpnn_multitask_wasserstein.py`` checkpoint.
    metal:
        One of :data:`FIVE_METALS`, or ``"wildcard"`` to average the pIC50
        head across all metals (the default — MCTS leaves carry no metal
        label, so per-metal selection would be an unearned assumption).
    device:
        ``"cpu"`` or ``"cuda"``.  Defaults to CPU so the MCTS bench (a CPU
        workload by construction) does not contend with in-flight training
        jobs for VRAM.
    """

    def __init__(
        self,
        ckpt_path: Path = DEFAULT_CKPT,
        *,
        metal: str = "wildcard",
        device: str = "cpu",
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.metal = str(metal)
        self.device = str(device)
        self._model: Any = None
        self._n_metals: int = len(FIVE_METALS)
        self._loaded = False
        self.n_forward = 0
        self.n_cache_hit = 0

    # ------------------------------------------------------------------
    # lazy loading
    # ------------------------------------------------------------------
    def _load(self) -> bool:
        """Import torch + build the model + load weights.  Returns success."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not self.ckpt_path.exists():
            self._model = None
            return False
        try:
            import torch  # local import: keeps module import cheap

            from molmetal.scripts.train_dmpnn_multitask_wasserstein import (
                DEFAULT_DEPTH,
                DEFAULT_DROPOUT,
                DEFAULT_HIDDEN,
                MultiTaskWassersteinModel,
            )

            ck = torch.load(
                self.ckpt_path, map_location=self.device, weights_only=False
            )
            state = ck.get("model_state_dict") if isinstance(ck, dict) else None
            if not isinstance(state, dict):
                self._model = None
                return False
            n_heads = sum(
                1 for k in state if k.startswith("cls_heads.") and k.endswith(".0.weight")
            )
            self._n_metals = n_heads or len(FIVE_METALS)
            # Wf-143 — checkpoints trained with the Kendall UW + DANN
            # gradient-reversal head carry extra ``domain_head.*`` keys.
            # Build with ``use_dann=True`` so the state dict matches
            # exactly; ``strict=False`` then absorbs any residual drift
            # between training-time and inference-time module shapes.
            has_domain_head = any(k.startswith("domain_head.") for k in state)
            model = MultiTaskWassersteinModel(
                n_metals=self._n_metals,
                hidden=int(ck.get("hidden") or DEFAULT_HIDDEN),
                depth=int(ck.get("depth") or DEFAULT_DEPTH),
                dropout=float(ck.get("dropout", DEFAULT_DROPOUT)),
                use_dann=has_domain_head,
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            # A non-empty *missing* list means the inference model is
            # missing weights the checkpoint supplies — random init would
            # silently corrupt the value channel, so refuse instead.
            if missing:
                self._model = None
                return False
            model.eval()
            model.to(self.device)
            self._model = model
            self._torch = torch
            return True
        except Exception:
            self._model = None
            return False

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def __call__(self, smiles: str) -> float:
        if not isinstance(smiles, str) or not smiles:
            return 0.0
        return self._cached(smiles)

    @functools.lru_cache(maxsize=256)
    def _cached(self, smiles: str) -> float:
        return self._forward_uncached(smiles)

    def _forward_uncached(self, smiles: str) -> float:
        if not self._load():
            return 0.0
        self.n_forward += 1
        try:
            from molmetal.baselines.dmpnn import featurize_smiles_list
            from molmetal.training.collate_multimetal import collate_multimetal
            import numpy as np

            graphs = featurize_smiles_list([smiles], metal_mode="retain")
            if not graphs or graphs[0] is None:
                return 0.0
            ids = self._metal_ids()
            pic50 = self._predict_graphs(graphs, ids)
            return float(pic50[0])
        except Exception:
            return 0.0

    def _metal_ids(self) -> "list":
        """Metal head indices: one per metal for ``wildcard``, else just one."""
        if self.metal == "wildcard":
            return list(range(self._n_metals))
        if self.metal in FIVE_METALS:
            return [FIVE_METALS.index(self.metal)]
        return list(range(self._n_metals))

    def _predict_graphs(self, graphs: "list", metal_ids: "list") -> "list":
        """Run one forward per metal head and return the per-mol pIC50 values.

        For the wildcard case the heads are averaged; for a specific metal
        only that head runs.  Averaging the *regression* outputs (rather than
        running a mean over heads) keeps the code path identical to what the
        trainer did at eval time.
        """
        from molmetal.training.collate_multimetal import collate_multimetal
        import numpy as np

        torch = self._torch
        outs: "list" = []
        for mid in metal_ids:
            batch = collate_multimetal(
                graphs, [mid] * len(graphs), np.zeros(len(graphs)), np.zeros(len(graphs))
            )
            if batch is None:
                outs.append(0.0)
                continue
            atom_f, bond_f, e_src, e_dst, mids_t, _p, _a = batch
            with torch.no_grad():
                pic50, _logits = self._model(
                    atom_f.to(self.device),
                    bond_f.to(self.device),
                    e_src.to(self.device),
                    e_dst.to(self.device),
                    mids_t.to(self.device),
                )
            outs.append(float(pic50[0].item()))
        return [sum(outs) / max(1, len(outs))]

    def cache_info(self):
        return self._cached.cache_info()


def load_checkpoint(path: Path = DEFAULT_CKPT) -> bool:
    """Return True iff ``path`` is a loadable multi-task DMPNN checkpoint."""
    return DMPNNValueFn(path)._load()


def predict_pic50(smiles: str, ckpt_path: Path = DEFAULT_CKPT, metal: str = "wildcard") -> float:
    """One-shot convenience wrapper (uncached).  Returns 0.0 on any failure."""
    return DMPNNValueFn(ckpt_path, metal=metal)(smiles)


def make_value_fn(
    ckpt_path: Path = DEFAULT_CKPT,
    *,
    metal: str = "wildcard",
    device: str = "cpu",
) -> Callable[[Any], float]:
    """Build a ``state -> pIC50`` callable for ``RewardAggregator``.

    The returned callable accepts either a raw SMILES string or a
    ``MoleculeClosedTerm`` (duck-typed via ``canonical_smiles()``), which is
    what :class:`RewardAggregator` passes to every reward channel.  Any
    exception raised inside is swallowed and mapped to ``0.0``.
    """
    inner = DMPNNValueFn(ckpt_path, metal=metal, device=device)

    def _value_fn(state: Any) -> float:
        smi = state
        if not isinstance(smi, str):
            getter = getattr(state, "canonical_smiles", None)
            if callable(getter):
                try:
                    smi = getter()
                except Exception:
                    return 0.0
            else:
                return 0.0
        try:
            return float(inner(smi))
        except Exception:
            return 0.0

    _value_fn.dmpnn = inner  # type: ignore[attr-defined]
    return _value_fn