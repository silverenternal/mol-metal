#!/usr/bin/env python3
"""Wf-A15 — Track-A CFM per-atom NBO natural-charge panel.

======================================================================
Computes per-atom q-MAE + q-mean-signed-error + q-per-atom-RMSE for a
CFM run's q-prediction output against the NBO Loewdin q target loaded
from :file:`.data/molmetal/tmQM/tmQM_X.q` (via
:func:`molmetal.data.tmqm.parse_natural_charges`).
======================================================================

Pipeline
--------
1. Load ``pred_charges`` from a ``.npz`` file holding the CFM run's
   per-atom q prediction (key: ``q_pred``, shape ``(N,)`` float32).
2. Load ``target_charges`` from either:
   * A ``.npz`` file with key ``q_target`` (shape ``(N,)`` float32), OR
   * A tmQM cohort loader via :mod:`molmetal.data.tmqm_clouds` (when
     the CFM run was a tmQM-pretrained checkpoint).  When the loader
     cannot find ``.data/molmetal/tmQM/tmQM_X.q(.gz)``, the target is
     the per-atom all-zero tensor (i.e. ``PAD_CHARGE`` everywhere) —
     every atom is masked out by :func:`compute_q_metrics` and the
     emitted ``n_atoms`` is ``0``.  This is the INERT-by-default
     contract documented in :class:`molmetal.loss.chemistry_losses.NaturalChargeLoss`.
3. Call :func:`molmetal.eval.scoring.natural_charge_aggregator.compute_q_metrics`.
4. Emit JSON to ``metrics/by_metric/cfm_natural_charge_mae.json``.

CLI surface
-----------
``--pred-npz PATH``        Path to a .npz holding ``q_pred`` (shape (N,)).
``--target-npz PATH``      Optional.  Path to a .npz holding ``q_target``.
                           When omitted, the panel uses zeros.
``--target-loader NAME``   Optional.  One of ``"tmqm"`` to load the
                           tmQM_X.q via :mod:`molmetal.data.tmqm_clouds`.
                           Reserved for future runs — currently inert
                           because the Wf-A14 host does not have
                           ``.data/molmetal/tmQM/tmQM_X.q(.gz)``
                           (network-blocked, see
                           :class:`molmetal.loss.chemistry_losses.NaturalChargeLoss`).
``--out-dir DIR``          Where to write the JSON.  Default:
                           ``metrics/by_metric``.

Output JSON schema
------------------
::

    {
      "q_mae": <float>,
      "q_mean_signed_error": <float>,
      "q_per_atom_rmse": <float>,
      "n_atoms": <int>,
      "source": "<pred_npz basename>:<target source>",
      "n_pred": <int>,        # raw input sizes for forensics
      "n_target": <int>,
      "wf": "Wf-A15",
      "tag": "Track-A natural-charge panel"
    }

Why this panel exists
---------------------
* The Track-A wiring (Layers 3 + 5 + 6 + 7 + 8) closes the loop where
  ``Batch.natural_charge`` reaches the trainer and is dropped on the
  floor.  After the wire-in, the CFM emits per-atom q predictions;
  this panel measures them against the NBO Loewdin ground truth.
* A persistent q-MAE > 0.30 e would be a useful honest-negative signal
  for paper §3.x (Track-A L8).  A q-MAE <= 0.10 e would be a positive
  but probably overstated signal given the 50%-atom padding in the
  CrossDocked cohort.

Bit-exact contract
------------------
The panel has no parameters and never any side effect beyond emitting
JSON to ``--out-dir``.  The aggregator (Layer 8) is pure-numpy and
returns zeros for empty / mismatched / all-pad inputs.  When the .q
file is absent the emitted ``n_atoms=0`` — explicit honest-negative
record, not a silent skip.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger("cfm_natural_charge_panel")

#: Path within the repo where the metric JSON is written by every other
#: panel.  Mirrored from :mod:`molmetal.scripts.cfm_property_panel`.
DEFAULT_OUT_DIR = "metrics/by_metric"

#: Output filename.
DEFAULT_OUT_NAME = "cfm_natural_charge_mae.json"


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    """Load a single ``(N,)`` float32 array from a .npz file.

    Falls back to an empty array when the file is missing or the key
    is absent — never raises, per the panel's INERT-by-default contract.
    """
    if not path.exists():
        LOG.warning("[cfm_natural_charge_panel] %s missing — using empty array", path)
        return np.zeros((0,), dtype=np.float32)
    try:
        with np.load(path) as data:
            if key not in data.files:
                LOG.warning(
                    "[cfm_natural_charge_panel] %s lacks key %s — using empty array",
                    path, key,
                )
                return np.zeros((0,), dtype=np.float32)
            arr = np.asarray(data[key])
    except Exception as exc:
        LOG.warning(
            "[cfm_natural_charge_panel] %s unreadable (%s) — using empty array",
            path, exc,
        )
        return np.zeros((0,), dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32, copy=False)


def _load_target(
    target_npz: Optional[Path],
    target_loader: Optional[str],
) -> Tuple[np.ndarray, str]:
    """Resolve ``target_charges`` from the explicit available paths.

    Returns ``(array, source)`` where ``source`` is a short string
    suitable for the JSON output.  Returns zeros + ``"absent"`` when no
    target is available — the aggregator then masks all atoms and emits
    ``n_atoms=0``.
    """
    if target_npz is not None:
        arr = _load_npz_array(target_npz, "q_target")
        return arr, f"npz:{target_npz.name}"
    if target_loader == "tmqm":
        try:
            from molmetal.data.tmqm import parse_natural_charges
            from molmetal.data.tmqm_clouds import load_tmqm_clouds  # noqa: F401
        except Exception as exc:
            LOG.warning(
                "[cfm_natural_charge_panel] tmqm loader import failed (%s) — "
                "target stays empty",
                exc,
            )
            return np.zeros((0,), dtype=np.float32), "tmqm:import-failed"
        try:
            from molmetal.data import tmqm as _tmqm_mod
            q_path = _tmqm_mod.DEFAULT_Q_FILE
            if q_path is None or not Path(q_path).exists():
                LOG.warning(
                    "[cfm_natural_charge_panel] tmQM_X.q(.gz) not present at "
                    "%s — target stays empty (INERT-by-default contract)",
                    q_path,
                )
                return np.zeros((0,), dtype=np.float32), "tmqm:absent"
            q_map = parse_natural_charges(q_path=q_path)
            if not q_map:
                return np.zeros((0,), dtype=np.float32), "tmqm:empty"
            # Concatenate in insertion order (deterministic for the same
            # file across runs).  The aggregator masks pad atoms (0.0)
            # so we don't need the original cloud-major ordering here.
            arr = np.concatenate(
                [np.asarray(v, dtype=np.float32) for v in q_map.values()],
                axis=0,
            )
            return arr, "tmqm:loaded"
        except Exception as exc:
            LOG.warning(
                "[cfm_natural_charge_panel] tmqm loader raised (%s) — target "
                "stays empty",
                exc,
            )
            return np.zeros((0,), dtype=np.float32), "tmqm:error"
    return np.zeros((0,), dtype=np.float32), "absent"


def run_panel(
    pred_npz: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    target_npz: Optional[Path] = None,
    target_loader: Optional[str] = None,
    out_name: str = DEFAULT_OUT_NAME,
) -> Dict[str, Any]:
    """Run the panel end-to-end and write the JSON.

    Returns the dict that was written.
    """
    # Lazy import so the panel does not pull torch unless the aggregator
    # needs numpy — the aggregator itself is pure numpy.
    from molmetal.eval.scoring.natural_charge_aggregator import compute_q_metrics

    pred = _load_npz_array(pred_npz, "q_pred")
    target, target_src = _load_target(target_npz, target_loader)
    metrics = compute_q_metrics(pred, target)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    payload = {
        **metrics,
        "source": f"pred={pred_npz.name}:target={target_src}",
        "n_pred": int(pred.shape[0]),
        "n_target": int(target.shape[0]),
        "wf": "Wf-A15",
        "tag": "Track-A natural-charge panel",
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOG.info(
        "[cfm_natural_charge_panel] wrote %s (q_mae=%.4f n_atoms=%d)",
        out_path, payload["q_mae"], payload["n_atoms"],
    )
    return payload


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cfm_natural_charge_panel",
        description=(
            "Track-A per-atom NBO natural-charge panel (Wf-A15 L8).  "
            "Reads q_pred from --pred-npz and q_target from --target-npz "
            "or the tmqm loader; emits JSON to --out-dir."
        ),
    )
    p.add_argument(
        "--pred-npz",
        type=Path,
        required=True,
        help="Path to a .npz holding the per-atom q_pred (key: q_pred, shape (N,)).",
    )
    p.add_argument(
        "--target-npz",
        type=Path,
        default=None,
        help="Path to a .npz holding q_target (key: q_target, shape (N,)).",
    )
    p.add_argument(
        "--target-loader",
        choices=("tmqm",),
        default=None,
        help="Optional loader for the q_target.  Currently only 'tmqm' "
             "(resolves to .data/molmetal/tmQM/tmQM_X.q(.gz)); absent on "
             "the Wf-A14 host by design.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help=f"Where to write the JSON.  Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default=DEFAULT_OUT_NAME,
        help=f"Output filename.  Default: {DEFAULT_OUT_NAME}",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    payload = run_panel(
        pred_npz=args.pred_npz,
        out_dir=args.out_dir,
        target_npz=args.target_npz,
        target_loader=args.target_loader,
        out_name=args.out_name,
    )
    print(
        f"[cfm_natural_charge_panel] q_mae={payload['q_mae']:.4f} "
        f"q_signed={payload['q_mean_signed_error']:+.4f} "
        f"q_rmse={payload['q_per_atom_rmse']:.4f} "
        f"n_atoms={payload['n_atoms']} "
        f"target={payload['source']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - smoke below
    # Smoke: synthesize a tiny (N,) pred/target npz pair and run the
    # panel end-to-end.  This is what the IMPL closeout uses as the
    # §5.7 wire-in evidence — the panel runs cleanly without a
    # trained model or a real .q file on disk.
    import tempfile

    if len(sys.argv) > 1 and sys.argv[1] != "--smoke":
        sys.exit(main())

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pred_arr = np.array([0.10, 0.20, 0.30, -0.10], dtype=np.float32)
        target_arr = np.array([0.10, 0.20, 0.30, -0.10], dtype=np.float32)
        np.savez(td / "pred.npz", q_pred=pred_arr)
        np.savez(td / "target.npz", q_target=target_arr)
        payload = run_panel(
            pred_npz=td / "pred.npz",
            out_dir=td / "out",
            target_npz=td / "target.npz",
        )
        assert payload["n_atoms"] == 4
        assert abs(payload["q_mae"]) < 1e-6
        assert abs(payload["q_mean_signed_error"]) < 1e-6
        out_json = td / "out" / DEFAULT_OUT_NAME
        assert out_json.exists()
        print("[cfm_natural_charge_panel] smoke PASS", payload)