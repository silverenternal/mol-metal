"""Continuous-BO evaluator (Wf-A27 step-3, 2026-10-13).

Per-batch evaluator that takes decoded SMILES + a corresponding bond-truth
list and emits a small dictionary of MAE-style metrics.  Mirror of the
:mod:`molmetal.eval.scoring.natural_charge_aggregator` aggregation pattern
(loads the cloud's :attr:`bond_orders_continuous` truth, scores against an
RDKit-parsed SMILES, never NaN).

Backward-compat rule
--------------------
``evaluate_bo_continuous`` returns ``None`` when there are no SMILES OR
every SMILES fails to parse; never NaN, never a Tensor.  This keeps the
``metals_panel`` / ablation bench scripts from blowing up on a 0-batch
edge case.

Honest-negative note
--------------------
This evaluator is a SUPERVISION helper for the standalone BO regression
loss (:class:`molmetal.loss.chemistry_losses.ContinuousBOLoss`); it is
NOT consumed by the trainer's loss path today — only by bench /
post-hoc decoder-quality analyses.  When the Wf-A22 panel scripts opt
to call it, the evaluator reads truth+pred pairs and emits a
``mean_bo_continuous_mae`` plus a ``coverage`` ratio (fraction of
inputs we successfully scored).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Lazy / tolerant RDKit import — already convention in this package.
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdMolDescriptors  # type: ignore
    _RDKIT_AVAILABLE = True
except Exception:
    _RDKIT_AVAILABLE = False


def _safe_bond_orders_from_smiles(smi: str) -> Optional[List[float]]:
    """Return a per-bond Wiberg-style surrogate from a SMILES string.

    We do NOT have a Wiberg-2 engine at hand; the function emits a
    SMILES-class-proxy: every detected bond gets a 1.0 for SINGLE, 2.0
    for DOUBLE, 3.0 for TRIPLE, 1.5 for AROMATIC, and 0.0 for the
    BOND_NO_BOND placeholder.  The 1.5 aromatic midpoint is the
    standard Wiberg band the project's bond head reads from tmQM.

    Returns ``None`` when ``smi`` is empty, ``None``, or RDKit fails
    to parse it — matching the surrounding evaluators' tolerance
    contract.
    """
    if not smi or not _RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    out: List[float] = []
    for b in mol.GetBonds():
        bt = b.GetBondType()
        if bt.name == "SINGLE":
            out.append(1.0)
        elif bt.name == "DOUBLE":
            out.append(2.0)
        elif bt.name == "TRIPLE":
            out.append(3.0)
        elif bt.name == "AROMATIC":
            out.append(1.5)
        else:
            # ZERO (BOND_NO_BOND placeholder), quadruple, etc.
            out.append(0.0)
    if not out:
        return None
    return out


def evaluate_bo_continuous(
    decoded_smiles_batch: Sequence[Optional[str]],
    truth_bond_orders: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[Dict[str, float]]:
    """Per-batch aggregator of BO continuous MAE.

    Parameters
    ----------
    decoded_smiles_batch
        Sequence of SMILES strings (one per cloud / per decode).  Empty
        strings, ``None`` entries, and unparseable SMILES are silently
        skipped — they do not contribute to ``coverage`` and never
        raise.
    truth_bond_orders
        Optional per-cloud WIBERG BO ground-truth sequences.  When
        provided, the evaluator MEASURES per-pair absolute error
        ``|pred - truth|``.  When ``None`` (the default for post-hoc
        decoder panels without Wiberg truth) the function still
        returns a dict but with the BO prediction summary (mean / max /
        coverage) instead of the MAE.

    Returns
    -------
    Optional[Dict[str, float]] with at least:
        * ``mean_bo_continuous`` — float mean predicted BO across all
          successfully parsed decodes (always present).
        * ``coverage`` — float fraction of inputs we successfully
          parsed via RDKit.
        * ``n_pairs_pred`` — int total predicted bond pairs parsed.
        * ``mean_bo_continuous_mae`` — float MAE vs ``truth`` (only
          when truth is supplied).
        * ``n_pairs_evaluated`` — int total pairs that contributed to
          the MAE.

    Returns ``None`` when every input is empty / unparseable (the
    0-batch edge case).
    """
    n_total = len(decoded_smiles_batch)
    if n_total == 0:
        return None
    parsed_pred: List[List[float]] = []
    for smi in decoded_smiles_batch:
        orders = _safe_bond_orders_from_smiles(smi if smi else "")
        if orders is not None:
            parsed_pred.append(orders)
    if not parsed_pred:
        # Every SMILES failed to parse — sentinel.
        return None
    flat_pred = [v for sub in parsed_pred for v in sub]
    n_pairs_pred = len(flat_pred)
    coverage = float(len(parsed_pred)) / float(n_total)
    out: Dict[str, float] = {
        "mean_bo_continuous": float(sum(flat_pred)) / max(1, n_pairs_pred),
        "max_bo_continuous": float(max(flat_pred)),
        "coverage": coverage,
        "n_pairs_pred": int(n_pairs_pred),
    }
    # Truth-side MAE (only when truth supplied + same length as batch).
    if truth_bond_orders is not None and len(truth_bond_orders) == len(parsed_pred):
        abs_errs: List[float] = []
        n_pairs_truth_aligned = 0
        for pred_sub, truth_sub in zip(parsed_pred, truth_bond_orders):
            n = min(len(pred_sub), len(truth_sub))
            if n == 0:
                continue
            for i in range(n):
                abs_errs.append(abs(float(pred_sub[i]) - float(truth_sub[i])))
            n_pairs_truth_aligned += n
        if abs_errs:
            out["mean_bo_continuous_mae"] = float(sum(abs_errs)) / len(abs_errs)
            out["n_pairs_evaluated"] = int(n_pairs_truth_aligned)
    return out


__all__ = [
    "evaluate_bo_continuous",
    "_safe_bond_orders_from_smiles",
]
