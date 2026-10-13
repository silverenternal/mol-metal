"""Track-A per-atom NBO natural-charge aggregator (Wf-A15 / Layer 8).

Computes four metrics from paired (predicted, target) per-atom NBO natural
charge tensors:

  * ``q_mae``               — Mean absolute error (in electrons, typical
                              q ∈ [-1.5, +1.5])
  * ``q_mean_signed_error`` — signed mean (predicted − target), a positive
                              value indicates systematic over-prediction
  * ``q_per_atom_rmse``     — RMSE per atom (sqrt(mean(error²)))
  * ``n_atoms``             — number of atoms actually scored (after
                              padding mask removal)

The aggregator masks out atoms whose target charge equals the
:data:`molmetal.data.batch.PAD_CHARGE` sentinel (the pad-token default).
This is necessary because :class:`molmetal.data.batch.Batch` flattens
padded clouds into a single ``(N_total,)`` tensor — without the mask the
MAE would be dominated by padding zeros that the q-head correctly
predicts as ``~0`` (i.e. the metric would say "0 error" on the padded
positions and look misleadingly good).

Pure numpy — no torch dependency, no GPU. The panel script that calls
this aggregator is responsible for tensor→numpy conversion. The
aggregator is intentionally pure-Python so the panel can run on a
CPU-only eval box and so the metrics JSON is reproducible.

YAML key surface
----------------
None — this module is opt-in via ``molmetal/scripts/cfm_natural_charge_panel.py``.
No CLI flag is exposed. Trainers that wish to log per-atom q-MAE inline
during training should call :func:`compute_q_metrics` directly with the
flat ``(N,)`` tensors.

Bit-exact contract
------------------
The aggregator has no state and no parameters — every call is a pure
function of its inputs. Padding atoms contribute nothing to any metric.
A degenerate input (``n_atoms == 0``) returns zeros with ``n_atoms=0`` —
never NaN, never a divide-by-zero. The panel script writes
``{"q_mae": 0.0, "q_mean_signed_error": 0.0, "q_per_atom_rmse": 0.0,
"n_atoms": 0}`` when the underlying cloud has no atoms.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Sentinel import: keeps this module decoupled from torch / data loaders
# at import time so the panel can run on a CPU-only eval box.  The
# constant itself lives in :mod:`molmetal.data.batch`.
try:
    from molmetal.data.batch import PAD_CHARGE as _PAD_CHARGE
except Exception:  # pragma: no cover - module unavailable in pure-numpy tests
    _PAD_CHARGE = 0.0


def _to_numpy(
    arr: Any, *, name: str = "input"
) -> np.ndarray:
    """Coerce a torch tensor / numpy array / list to ``float32`` numpy.

    Falls back to ``np.asarray`` so list-of-floats works.  Empty input
    becomes an empty ``(0,)`` array, NOT a scalar — downstream code relies
    on ``arr.ndim == 1``.
    """
    if arr is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(arr, "detach") and hasattr(arr, "cpu"):
        # torch.Tensor — keep this check BEFORE np.asarray so we get the
        # right dtype + device handling.
        try:
            arr = arr.detach().cpu().numpy()
        except Exception:
            arr = np.asarray(arr)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim == 0:
        # Scalar: promote to (1,) so downstream shape checks pass.
        arr = arr.reshape(1)
    return arr.astype(np.float32, copy=False)


def compute_q_metrics(
    pred_charges: Any,
    target_charges: Any,
    *,
    pad_charge: Optional[float] = None,
) -> Dict[str, float]:
    """Compute per-atom q metrics from paired ``(N,)`` tensors.

    Parameters
    ----------
    pred_charges, target_charges
        Tensors / arrays of shape ``(N,)`` with per-atom predicted and
        target natural charges.  Either may be ``None`` (returns zeros)
        or empty ``(0,)`` (returns zeros with ``n_atoms=0``).
    pad_charge : float, optional
        Sentinel value marking padded atoms.  Defaults to
        :data:`molmetal.data.batch.PAD_CHARGE` (``0.0``).  Atoms whose
        target equals this sentinel are masked out before computing
        metrics — they are not part of any real molecule.

    Returns
    -------
    dict
        ``{"q_mae": float, "q_mean_signed_error": float,
        "q_per_atom_rmse": float, "n_atoms": int}``.

        All three metric fields are zero when ``n_atoms == 0``.
        The returned dict is JSON-serialisable as-is.
    """
    if pad_charge is None:
        pad_charge = float(_PAD_CHARGE)

    pred = _to_numpy(pred_charges, name="pred_charges")
    target = _to_numpy(target_charges, name="target_charges")

    if pred.size == 0 or target.size == 0:
        return {
            "q_mae": 0.0,
            "q_mean_signed_error": 0.0,
            "q_per_atom_rmse": 0.0,
            "n_atoms": 0,
        }

    if pred.shape != target.shape:
        # Shape mismatch: cannot score.  Return zeros with the smaller
        # of the two sizes as ``n_atoms`` so the panel sees a non-zero
        # count to investigate — never raises, per the INERT-by-default
        # contract (cf. ``NaturalChargeLoss.compute``).
        n = int(min(pred.shape[0], target.shape[0]))
        return {
            "q_mae": 0.0,
            "q_mean_signed_error": 0.0,
            "q_per_atom_rmse": 0.0,
            "n_atoms": n,
        }

    # Mask out padding atoms: their target charge equals pad_charge.
    keep = target != float(pad_charge)
    if not keep.any():
        return {
            "q_mae": 0.0,
            "q_mean_signed_error": 0.0,
            "q_per_atom_rmse": 0.0,
            "n_atoms": 0,
        }

    p = pred[keep]
    t = target[keep]
    diff = p - t
    n_atoms = int(p.shape[0])
    q_mae = float(np.mean(np.abs(diff)))
    q_mse = float(np.sqrt(np.mean(diff * diff)))
    q_signed = float(np.mean(diff))
    return {
        "q_mae": q_mae,
        "q_mean_signed_error": q_signed,
        "q_per_atom_rmse": q_mse,
        "n_atoms": n_atoms,
    }


__all__ = ["compute_q_metrics"]


if __name__ == "__main__":  # pragma: no cover - smoke
    # Smoke: bit-exact-instant zero with weight=0 / pad-only inputs.
    out = compute_q_metrics(None, None)
    assert out == {
        "q_mae": 0.0,
        "q_mean_signed_error": 0.0,
        "q_per_atom_rmse": 0.0,
        "n_atoms": 0,
    }, f"None inputs must yield zeros, got {out}"

    # Smoke: empty inputs.
    out = compute_q_metrics([], [])
    assert out["n_atoms"] == 0
    assert out["q_mae"] == 0.0

    # Smoke: pad atoms are masked out.
    # All targets == 0.0 (PAD_CHARGE) → all masked → n_atoms=0.
    pred = np.array([0.1, 0.0, 0.2], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    out = compute_q_metrics(pred, target)
    assert out["n_atoms"] == 0, f"all-pad should be masked, got {out}"

    # Smoke: real atoms (non-zero target) are scored.
    pred = np.array([0.10, 0.05, 0.20], dtype=np.float32)
    target = np.array([0.05, 0.10, 0.20], dtype=np.float32)
    out = compute_q_metrics(pred, target)
    # diff = [0.05, -0.05, 0.0]; MAE = 0.10/3; signed = 0.0
    assert out["n_atoms"] == 3
    assert abs(out["q_mae"] - 0.10 / 3) < 1e-6, f"q_mae={out['q_mae']}"
    assert abs(out["q_mean_signed_error"] - 0.0) < 1e-6, f"q_signed={out['q_mean_signed_error']}"

    print("natural_charge_aggregator PASS")