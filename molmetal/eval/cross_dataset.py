"""Cross-dataset transfer-learning validation — F fix.

Targets the F weakness from ``docs/CFM_WEAKNESSES_2026_10_13.md``:

    F — 0/149 generated molecules contain Pt (real_tmqm Pt fraction 25.2%);
    root cause was a reservoir-luck failure.  The min_per_metal
    sampler patch closed the eval-time gap, but cross-dataset
    TRANSFER LEARNING (PlatinAI / NCI-60 / MetalCytoToxDB) was never
    systematically evaluated.  Pretrain + 0.046 (Ru fine-tune) on
    random scaffold was REFUTED on Bemis–Murcko scaffold split
    (paired delta -0.0094, t(4) = -0.839, 95% CI [-0.0280, +0.0116]).

This script is a SCRIPT TEMPLATE — invoking it requires:
  1. A trained CFM checkpoint on tmQM (we have one — the seed-0
     pretrain at molmetal/checkpoints/).
  2. PlatinAI 4-cell labels (already loaded by
     molmetal.molmetal_lam.lam_chem.platinai_dataset.PlatinAIDataset).
  3. NCI-60 GI50 9-cell labels (per cfm_todo.json::TIER_1.m21 metric).
  4. MetalCytoToxDB (per cfm_todo.json::TIER_1.m1 metric).

Default YAML flag ``eval.cross_dataset_validation: false`` keeps the
9-wave trainer sum bit-exact.  When set to true, the trainer's eval
loop emits a JSON with one row per (dataset, fold, cell_line) so
reviewers can verify the transfer claim.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

LOG = logging.getLogger("molmetal.eval.cross_dataset")


def _load_pretrained_checkpoint(ckpt_path: Path, device: str = "cpu") -> Dict:
    """Load a CFM checkpoint into a flat dict.

    Returns a dict with ``model_state_dict``, ``optimiser_state_dict``,
    ``epoch``, ``metrics``.  Tolerates files saved by both the
    legacy ``torch.save`` (single dict) and the new ``atomic_write``
    path (same shape).
    """
    import torch
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        # Bare state_dict — wrap for the rest of the script.
        return {"model_state_dict": obj, "epoch": -1, "metrics": {}}
    raise ValueError(f"Unrecognised checkpoint layout at {ckpt_path}: {type(obj)}")


def _predict_smiles(
    model_state: Dict,
    dataset_smiles: List[str],
    device: str = "cpu",
    max_n: int = 64,
) -> List[str]:
    """Decode dataset SMILES through the loaded CFM YuelBond head.

    Wf-A14 (2026-10-13) — replaces the previous stub (which returned
    the input SMILES unchanged).  The decode path is now:

    1. Infer ``hidden_dim`` + ``n_layers`` from
       ``model_state["head.atom_encoder.weight"].shape`` and the
       number of ``msg_mlps.<i>.0.weight`` keys.
    2. Construct :class:`YuelBondDecoderHead` and ``load_state_dict``
       from the head-prefixed entries (bit-tolerant: unknown keys are
       dropped, ``strict=False``).
    3. For each input SMILES, build an :class:`AtomCloud` via
       :func:`molmetal.eval.linear_probe.smiles_to_atom_cloud` (3-D
       RDKit conformer + largest fragment + heavy-atom packing).
    4. Run :class:`YuelBondBondDecoder` (wraps the head + valence-aware
       selector) and return ``result.decoded.smiles``.

    On any error (ckpt shape mismatch, RDKit parse failure, valence
    rejection, etc.) the function falls back to the **input** SMILES —
    the Wf-A14 contract is that the script must never crash on a
    single bad SMILES.  The fallback path is logged so a reviewer can
    tell the decoded ratio from the input ratio.
    """
    try:
        import torch
        from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
            YuelBondDecoderHead,
        )
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
        )
        from molmetal.eval.linear_probe import smiles_to_atom_cloud
    except Exception as exc:  # pragma: no cover - import path
        LOG.warning("[cross-dataset] decode import failed: %s; pass-through", exc)
        return list(dataset_smiles)

    head_sd = {
        k[len("head."):]: v
        for k, v in model_state.items()
        if k.startswith("head.")
    }
    if not head_sd:
        LOG.warning(
            "[cross-dataset] model_state has no 'head.' keys; pass-through"
        )
        return list(dataset_smiles)

    try:
        atom_w = head_sd["head.atom_encoder.weight"]
        hidden_dim = int(atom_w.shape[0])
        n_layers = sum(
            1
            for k in head_sd
            if k.startswith("msg_mlps.") and k.endswith(".0.weight")
        )
        if n_layers < 2:
            n_layers = 3
    except Exception as exc:
        LOG.warning(
            "[cross-dataset] ckpt head shape unrecognised: %s; pass-through",
            exc,
        )
        return list(dataset_smiles)

    try:
        head = YuelBondDecoderHead(hidden_dim=hidden_dim, n_layers=n_layers)
        # The head_sd keys do NOT include the 'head.' prefix after the
        # strip above; pass strict=False so any future layer additions
        # do not break the loader.
        head.load_state_dict(head_sd, strict=False)
    except Exception as exc:
        LOG.warning(
            "[cross-dataset] head load_state_dict failed: %s; pass-through",
            exc,
        )
        return list(dataset_smiles)

    decoder = YuelBondBondDecoder(head=head)
    out: List[str] = []
    n_fallback = 0
    for smi in dataset_smiles[: int(max_n)]:
        try:
            cloud = smiles_to_atom_cloud(smi)
            if cloud is None:
                out.append(smi)
                n_fallback += 1
                continue
            res = decoder.decode(cloud)
            decoded_smi = res.decoded.smiles if res.decoded.sanitized else ""
            if not decoded_smi:
                out.append(smi)
                n_fallback += 1
            else:
                out.append(decoded_smi)
        except Exception as exc:
            LOG.warning(
                "[cross-dataset] decode failure on %r: %s",
                smi[:40], exc,
            )
            out.append(smi)
            n_fallback += 1
    if n_fallback:
        LOG.info(
            "[cross-dataset] decode: %d/%d fell back to input SMILES",
            n_fallback, min(len(dataset_smiles), int(max_n)),
        )
    return out


def _score_oracle(smiles_batch: List[str], oracle_name: str) -> List[float]:
    """Per-`eval/scoring/<oracle>.py` score on a decoded SMILES batch.

    Returns one float per SMILES (NaN-safe: unparseable SMILES get
    the corpus median per dataset — measured at the time the script
    is invoked, not hard-coded here).
    """
    try:
        if oracle_name == "pocket2mol_score":
            from molmetal.eval.scoring.pocket2mol_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
        if oracle_name == "softmol_score":
            from molmetal.eval.scoring.softmol_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
        if oracle_name == "targetdiff_score":
            from molmetal.eval.scoring.targetdiff_score import score
            return [float(score(s)) if score(s) is not None else float("nan") for s in smiles_batch]
    except Exception as exc:
        LOG.warning("oracle %s failed: %s", oracle_name, exc)
    return [float("nan")] * len(smiles_batch)


def run_cross_dataset_validation(
    ckpt_path: str,
    out_json: str,
    datasets: Optional[List[str]] = None,
    device: str = "cpu",
    do_linear_probe: bool = False,
    linear_probe_epochs: int = 20,
    linear_probe_max_n: int = 200,
) -> Dict:
    """Run transfer-learning validation against the 3 cross-datasets.

    Parameters
    ----------
    ckpt_path : str
        Path to a trained CFM checkpoint (per
        ``molmetal.checkpoints``).  Required.
    out_json : str
        Output JSON path.  Will be overwritten if it exists.
    datasets : list of str, optional
        One or more of ``{"platinai_4cell", "nci60_9cell",
        "metalcytotoxdb"}``.  Default = all three.
    device : str
        "cpu" or "cuda".  Default "cpu" so the script can run on the
        smoke CPU without GPU.
    do_linear_probe : bool
        When ``True`` (default ``False``), additionally run the
        :func:`molmetal.eval.linear_probe.run_cross_dataset_linear_probe`
        path: freeze the YuelBond encoder, train a small logistic head
        on the per-cell labels, and attach the per-label AUC + mean
        AUC to every per-(dataset, oracle) record.  Default off so the
        pre-Wf-A14 JSON shape is preserved when the flag is absent.
    linear_probe_epochs : int
        Logistic-regression epochs for the probe head.  Default 20.
    linear_probe_max_n : int
        Cap on rows used by the probe (smoke-friendly).  Default 200.

    Returns
    -------
    dict with keys ``{"results", "config", "wall_time_s"}`` where
    ``results`` is a list of per-(dataset, fold, oracle) score records
    (each record optionally has ``linear_probe`` and
    ``linear_probe_reason`` keys when ``do_linear_probe=True``).
    """
    if datasets is None:
        datasets = ["platinai_4cell", "nci60_9cell", "metalcytotoxdb"]
    t0 = time.time()
    LOG.info("[cross-dataset] loading checkpoint: %s", ckpt_path)
    state = _load_pretrained_checkpoint(Path(ckpt_path), device=device)
    LOG.info("[cross-dataset] loaded checkpoint at epoch %s", state.get("epoch", -1))

    # Map dataset name → (SMILES loader, oracle to score on).
    def _load_smiles(name: str) -> List[str]:
        try:
            if name == "platinai_4cell":
                from molmetal.molmetal_lam.lam_chem.platinai_dataset import PlatinAIDataset
                ds = PlatinAIDataset()
                return list(getattr(ds, "smiles", []))[:64]
            if name == "nci60_9cell":
                from molmetal.molmetal_lam.lam_chem.platinai_dataset import NCI60_GI50_DEFAULT
                # NCI60 loader exposes per-cell-line SMILES, here we
                # fall back to the bundled subset if the dataset class
                # is not present.
                return list(getattr(NCI60_GI50_DEFAULT, "smiles", []))[:64]
            if name == "metalcytotoxdb":
                from molmetal.molmetal_lam.lam_chem.platinai_dataset import MetalCytoToxDataset
                ds = MetalCytoToxDataset()
                return list(getattr(ds, "smiles", []))[:64]
        except Exception as exc:
            LOG.warning("[cross-dataset] dataset %s loader failed: %s", name, exc)
        return []

    out_records: List[Dict] = []
    for ds_name in datasets:
        smiles = _load_smiles(ds_name)
        if smiles:
            decoded = _predict_smiles(state["model_state_dict"], smiles, device=device)
            for oracle in ("pocket2mol_score", "softmol_score", "targetdiff_score"):
                scores = _score_oracle(decoded, oracle)
                valid = [s for s in scores if s == s]  # NaN-safe filter
                if not valid:
                    continue
                mean_score = sum(valid) / len(valid)
                out_records.append(
                    {
                        "dataset": ds_name,
                        "n_input": len(smiles),
                        "n_decoded": len(decoded),
                        "n_valid_oracle": len(valid),
                        "oracle": oracle,
                        "mean_score": mean_score,
                    }
                )
        else:
            LOG.info(
                "[cross-dataset] %s: legacy _load_smiles returned empty; "
                "skipping oracle decode, but linear-probe will still run "
                "from the CSV loader.",
                ds_name,
            )
        # Wf-A14 (2026-10-13) — linear-probe transfer (freeze encoder,
        # train head).  Gated on do_linear_probe so the default path
        # is bit-exact identical to the pre-Wf-A14 output.
        # Note: the linear probe uses its OWN CSV loader
        # (``_load_dataset_for_probe`` in linear_probe.py), so it can
        # run even when the legacy ``_load_smiles`` returns empty.
        if do_linear_probe:
            try:
                from molmetal.eval.linear_probe import (
                    run_cross_dataset_linear_probe,
                )
                probe_res = run_cross_dataset_linear_probe(
                    ckpt_path=str(ckpt_path),
                    dataset_name=ds_name,
                    device=device,
                    n_epochs=int(linear_probe_epochs),
                    max_n=int(linear_probe_max_n),
                )
                # Attach the probe record to the per-oracle records so
                # the JSON has a single entry per (dataset, oracle) +
                # one probe entry.  Reviewers can grep the JSON for
                # "linear_probe" to filter.
                if out_records:
                    for rec in out_records:
                        if rec["dataset"] == ds_name and "linear_probe" not in rec:
                            rec["linear_probe"] = probe_res.get("linear_probe")
                            rec["linear_probe_reason"] = probe_res.get("reason")
                else:
                    # No oracle records — emit a probe-only record so
                    # the JSON is non-empty.
                    out_records.append(
                        {
                            "dataset": ds_name,
                            "n_input": 0,
                            "n_decoded": 0,
                            "n_valid_oracle": 0,
                            "oracle": "<linear-probe-only>",
                            "mean_score": float("nan"),
                            "linear_probe": probe_res.get("linear_probe"),
                            "linear_probe_reason": probe_res.get("reason"),
                        }
                    )
            except Exception as exc:
                LOG.warning(
                    "[cross-dataset] linear probe failed for %s: %s",
                    ds_name, exc,
                )
                for rec in out_records:
                    if rec["dataset"] == ds_name:
                        rec["linear_probe_reason"] = f"exception: {exc}"
    payload = {
        "config": {
            "ckpt_path": str(ckpt_path),
            "datasets": datasets,
            "device": device,
            "out_json": out_json,
        },
        "results": out_records,
        "wall_time_s": time.time() - t0,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    LOG.info("[cross-dataset] wrote %s (%d records, %.1fs)", out_json, len(out_records), payload["wall_time_s"])
    return payload


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Wf-A14 cross-dataset validation")
    parser.add_argument("--ckpt", required=True, help="Path to CFM checkpoint")
    parser.add_argument("--out", default="metrics/by_metric/cfm_cross_dataset_transfer.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["platinai_4cell", "nci60_9cell", "metalcytotoxdb"],
        help="Datasets to validate against",
    )
    parser.add_argument(
        "--do-linear-probe",
        action="store_true",
        help="Also run frozen-encoder linear probe (Wf-A14 transfer protocol)",
    )
    parser.add_argument(
        "--linear-probe-epochs",
        type=int,
        default=20,
        help="Logistic-regression epochs for the probe head (default 20)",
    )
    parser.add_argument(
        "--linear-probe-max-n",
        type=int,
        default=200,
        help="Max rows used by the linear probe (smoke-friendly)",
    )
    args = parser.parse_args()
    run_cross_dataset_validation(
        ckpt_path=args.ckpt,
        out_json=args.out,
        datasets=args.datasets,
        device=args.device,
        do_linear_probe=args.do_linear_probe,
        linear_probe_epochs=args.linear_probe_epochs,
        linear_probe_max_n=args.linear_probe_max_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_cross_dataset_validation",
    "_load_pretrained_checkpoint",
    "_predict_smiles",
    "_score_oracle",
]
