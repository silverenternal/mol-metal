#!/usr/bin/env python
# Apache-2.0 header per §5.10 — see LICENSE.
"""Wf-A53 (2026-10-19) — score a CFM checkpoint on a HELD-OUT split.

Why this exists
---------------
``build_csv_cohort`` has always accepted ``split="test"``, but the trainer
only ever calls it twice — ``split=args.data_split`` (train) and
``split=args.val_split`` (val).  There is no ``--test-split`` flag, so the
5,736-row test split of the shipped corpus had **no consumer at all**: the
data existed, was split, and was never scored.  A split nobody reads is
indistinguishable from a split that does not exist.

This script is that consumer.  It deliberately does NOT edit
``train_cfm_yuelbond_wasserstein.py`` — the trainer is owned by another
workstream.  Instead it imports the trainer's own public evaluation
functions, so the numbers it reports are the same code paths the training
loop runs on ``val``:

    evaluate_bond_accuracy       — predicted bond graph vs corpus truth
    evaluate_decode_ratio        — RDKit-strict decode of the REAL geometry
    evaluate_generation_validity — sample from noise, decode, RDKit-strict

Nothing here retrains, fine-tunes, or writes a checkpoint.  The checkpoint
is opened read-only.

What this does NOT measure
--------------------------
There is no held-out *loss* number.  The trainer's flow-matching /
Wasserstein loss is computed inline inside ``train_one_epoch`` rather than
in a callable that takes a cloud list, so re-deriving it here would be a
second implementation of the objective — exactly the kind of parallel
definition this project has been burned by.  The three metrics above are
the held-out proxies that already ship with the trainer.

Usage
-----
    .venv/bin/python molmetal/scripts/wf_a53_eval_cfm_test_split.py \\
        --ckpt-path results/cfm_5metal_default/checkpoints/\\
cfm_yuelbond_wasserstein_seed0_last.pt \\
        --csv .data/molmetal/metalcytotox_5metal_multitask_v5.csv \\
        --split test \\
        --out-json metrics/by_metric/cfm_5metal_v5_test_split_eval.json

    # cohort-only sanity pass, no model load, no GPU:
    .venv/bin/python molmetal/scripts/wf_a53_eval_cfm_test_split.py \\
        --ckpt-path ... --csv ... --split test --dry-run \\
        --out-json /tmp/test_cohort.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str) -> Path:
    """Relative paths resolve against the repo root, matching the YAMLs."""
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wf_a53_eval_cfm_test_split.py",
        description=(
            "Score a CFM checkpoint on a held-out split (default: v5 test). "
            "Read-only: no training, no checkpoint writes."
        ),
    )
    p.add_argument(
        "--ckpt-path",
        type=Path,
        required=True,
        help="CFM checkpoint. Both serialisations the trainer emits are read.",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Corpus CSV. Default: the checkpoint's own data.csv (which is "
            "the file it TRAINED on — pass --csv explicitly when evaluating "
            "a held-out corpus that is not the training file)."
        ),
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Value of the 'split' column to score. Default 'test'.",
    )
    p.add_argument(
        "--n-clouds",
        type=int,
        default=None,
        help=(
            "Cap on embedded clouds. Default None = every embeddable row in "
            "the split. The report records how many rows the split had vs "
            "how many were actually embedded, so a capped run cannot be read "
            "as a full-split result."
        ),
    )
    p.add_argument(
        "--max-bond-clouds",
        type=int,
        default=512,
        help="Cap for evaluate_bond_accuracy (default 512).",
    )
    p.add_argument(
        "--max-decode-clouds",
        type=int,
        default=256,
        help="Cap for evaluate_decode_ratio (default 256).",
    )
    p.add_argument(
        "--gen-samples",
        type=int,
        default=64,
        help=(
            "Compositions sampled from noise by evaluate_generation_validity "
            "(default 64). 0 skips the generation pass."
        ),
    )
    p.add_argument(
        "--gen-steps",
        type=int,
        default=50,
        help="ODE steps for the generation pass (default 50; trainer yaml uses 100).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the forward passes. Default 'cpu'.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="ETKDG / noise seed. Default 0.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output JSON path; created or overwritten.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the cohort and report its composition only. No model load, "
            "no forward pass."
        ),
    )
    return p


_METAL_Z = {"Ru": 44, "Rh": 45, "Re": 75, "Os": 76, "Ir": 77}


def _cohort_summary(clouds: Sequence[Any]) -> Dict[str, Any]:
    """Composition of the embedded cohort, so a claim about coverage is checkable.

    The per-metal counts are read off the CLOUD's atomic numbers, not off the
    CSV.  That is the point: the loader is supposed to INSERT the metal atom
    (Wf-A66), and a CSV row whose metal never reached the cloud would train a
    ligand-only model while every row-level count still looked correct.
    """
    n_atoms = [int(c.atomic_numbers.numel()) for c in clouds]
    by_metal: Dict[str, int] = {sym: 0 for sym in _METAL_Z}
    no_metal = 0
    with_truth = 0
    for c in clouds:
        z = set(int(v) for v in c.atomic_numbers.tolist())
        found = False
        for sym, zz in _METAL_Z.items():
            if zz in z:
                by_metal[sym] += 1
                found = True
        if not found:
            no_metal += 1
        if getattr(c, "bonds", None) is not None and int(c.bonds.shape[1]) > 0:
            with_truth += 1
    return {
        "n_clouds": len(clouds),
        "n_with_bond_truth": with_truth,
        "atoms_min": min(n_atoms) if n_atoms else 0,
        "atoms_max": max(n_atoms) if n_atoms else 0,
        "atoms_mean": (sum(n_atoms) / len(n_atoms)) if n_atoms else 0.0,
        # Counted from cloud atomic numbers (see docstring).
        "clouds_by_metal": dict(sorted(by_metal.items())),
        "clouds_with_no_metal": no_metal,
    }


def _split_rows(csv_path: Path, split: str) -> Dict[str, Any]:
    """Row counts straight from the CSV, independent of embedding success."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return {"available": False}
    df = pd.read_csv(csv_path, low_memory=False)
    if "split" not in df.columns:
        return {"available": False, "reason": "no 'split' column"}
    smi_col = "smiles" if "smiles" in df.columns else (
        "SMILES_Ligands" if "SMILES_Ligands" in df.columns else None
    )
    sub = df[df["split"].astype(str) == str(split)]
    out: Dict[str, Any] = {
        "available": True,
        "n_rows": int(len(sub)),
        "n_unique_smiles": int(sub[smi_col].astype(str).nunique()) if smi_col else None,
        "n_total_rows": int(len(df)),
        "split_value_counts": {
            str(k): int(v) for k, v in df["split"].astype(str).value_counts().items()
        },
    }
    if "Metal" in sub.columns:
        out["rows_by_metal"] = {
            str(k): int(v) for k, v in sub["Metal"].astype(str).value_counts().items()
        }
    return out


def _canon_set(csv_path: Path, split: str) -> Optional[set]:
    """Canonical SMILES of one split, or ``None`` if the file cannot answer."""
    try:
        import pandas as pd
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize

        RDLogger.DisableLog("rdApp.*")
    except ImportError:  # pragma: no cover
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    if "split" not in df.columns:
        return None
    col = "smiles" if "smiles" in df.columns else (
        "SMILES_Ligands" if "SMILES_Ligands" in df.columns else None
    )
    if col is None:
        return None
    sub = df[df["split"].astype(str) == str(split)][col].astype(str).unique()
    out = set()
    for smi in sub:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            if len(Chem.GetMolFrags(mol)) > 1:
                mol = rdMolStandardize.FragmentParent(mol)
            out.add(Chem.MolToSmiles(mol))
        except Exception:
            out.add(smi)
    return out


def _training_overlap(
    ckpt_args: Dict[str, Any], eval_csv: Path, eval_split: str
) -> Dict[str, Any]:
    """Is the split actually held out relative to THIS checkpoint?

    A "test" split is only held out with respect to a particular training
    file.  v5 was rebuilt from v3 by deduplication + re-assignment, so a
    checkpoint trained on v3's ``train`` split has almost certainly seen the
    molecules that v5 places in ``test``.  Reporting a number from that
    combination under the word "held-out" would be false, so this measures
    the overlap and states the verdict rather than leaving it to the reader.
    """
    train_csv = ckpt_args.get("data_csv")
    train_split = str(ckpt_args.get("data_split", "train"))
    if not train_csv:
        return {"verdict": "UNKNOWN", "reason": "checkpoint carries no data.csv"}
    train_path = _resolve(str(train_csv))
    if not train_path.is_file():
        return {"verdict": "UNKNOWN", "reason": f"{train_path} is missing"}

    a = _canon_set(train_path, train_split)
    b = _canon_set(eval_csv, eval_split)
    if a is None or b is None:
        return {"verdict": "UNKNOWN", "reason": "could not canonicalise one side"}

    shared = a & b
    return {
        "verdict": "HELD_OUT" if not shared else "NOT_HELD_OUT",
        "training_csv": str(train_path),
        "training_split": train_split,
        "n_training_unique": len(a),
        "n_eval_unique": len(b),
        "n_shared_unique": len(shared),
        "frac_eval_seen_in_training": (len(shared) / len(b)) if b else 0.0,
        "jaccard": (len(shared) / len(a | b)) if (a | b) else 0.0,
    }


def _smiles_fragment_stats(smiles: Sequence[str]) -> Dict[str, Any]:
    """Connectivity of a decoded SMILES list.

    ``evaluate_decode_ratio`` asks only "did RDKit sanitise the string".  A
    cloud of loose atoms sanitises perfectly, so a decoder that predicts
    BOND_NO_BOND everywhere scores 1.000.  MEASURED 2026-10-19 on the
    shipped checkpoint: 256/256 decode "successes", each molecule 10-18
    fragments including ``[HH]`` — the ratio was measuring nothing.

    This function is descriptive only.  It does not re-implement the
    pass/fail rule: ``evaluate_decode_ratio`` remains the single definition
    of that, and the two are reported side by side.
    """
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError:  # pragma: no cover
        return {"available": False}

    frag_counts: List[int] = []
    n_with_metal = 0
    metal_z = set(_METAL_Z.values())
    for smi in smiles:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        frag_counts.append(len(Chem.GetMolFrags(mol)))
        if any(a.GetAtomicNum() in metal_z for a in mol.GetAtoms()):
            n_with_metal += 1

    if not frag_counts:
        return {"available": True, "n_decoded": 0}
    n_single = sum(1 for n in frag_counts if n == 1)
    return {
        "available": True,
        "n_decoded": len(frag_counts),
        "n_single_fragment": n_single,
        "single_fragment_ratio": n_single / len(frag_counts),
        "fragments_mean": sum(frag_counts) / len(frag_counts),
        "fragments_max": max(frag_counts),
        "n_containing_metal": n_with_metal,
        "metal_ratio": n_with_metal / len(frag_counts),
    }


def _decode_fragment_stats(decoder: Any, clouds: Sequence[Any]) -> Dict[str, Any]:
    """Re-decode the same clouds (deterministic) and measure connectivity."""
    smiles: List[str] = []
    for cloud in clouds:
        try:
            result = decoder.decode(cloud)
        except Exception:
            continue
        decoded = getattr(result, "decoded", None)
        smi = getattr(decoded, "smiles", "") or ""
        if smi:
            smiles.append(smi)
    return _smiles_fragment_stats(smiles)


def _load_model(ckpt: Any, device: Any) -> Any:
    """Rebuild the architecture from ``ckpt['args']`` and load the weights."""
    import argparse as _argparse

    import torch

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_joint_model

    args_dict = ckpt.get("args") if isinstance(ckpt, dict) else None
    if not isinstance(args_dict, dict):
        raise SystemExit(
            "checkpoint carries no serialised 'args' dict; the architecture "
            "cannot be rebuilt without guessing shapes"
        )
    args = _argparse.Namespace(**args_dict)
    model = build_joint_model(args)
    state = ckpt.get("model_state_dict")
    if state is None:
        raise SystemExit("checkpoint has no 'model_state_dict'")
    model.load_state_dict(state, strict=True)
    # The trainer evaluates with the EMA snapshot swapped in when it exists
    # (``_ema_swapped = model.ema_swap_in()`` before the val pass).  Use the
    # same API rather than ``load_state_dict`` on the snapshot: the snapshot
    # is a dict of tensors with no buffers guarantee, and ``ema_swap_in``
    # keeps the live weights recoverable.
    ema = ckpt.get("ema_state_dict")
    if isinstance(ema, dict) and ema:
        if model.load_ema_state(ema) and model.ema_swap_in():
            print("[wfA53] evaluating with EMA weights (ema_state_dict present)")
        else:
            print("[wfA53] WARNING: EMA snapshot present but not applied")
    model.to(device)
    model.eval()
    print(
        f"[wfA53] model rebuilt: hidden_dim={getattr(args, 'hidden_dim', '?')} "
        f"n_layers={getattr(args, 'n_layers', '?')} "
        f"vocab={getattr(args, 'yuelbond_vocab', '?')} "
        f"egnn_velocity={getattr(args, 'use_egnn_velocity', '?')}"
    )
    return model, args


def _build_decoder(model: Any, args: Any, ckpt: Any) -> Any:
    """Construct the decoder with the rule the checkpoint was scored under."""
    from molmetal.eval.bond_threshold_calibration import (
        decoder_kwargs_from_checkpoint,
    )
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    try:
        dec_kw = decoder_kwargs_from_checkpoint(ckpt)
    except Exception as exc:  # pragma: no cover - safety net
        dec_kw = {}
        print(f"[wfA53] WARNING: decode_calibration not applied ({exc!r})")
    if not dec_kw:
        # Older checkpoints carry no calibration stamp; fall back to the
        # trainer's argparse defaults so behaviour is the documented default.
        dec_kw = {
            "decision": str(getattr(args, "bond_decision", "argmax")),
            "decision_threshold": float(
                getattr(args, "bond_decision_threshold", 0.5)
            ),
        }
    hidden_dim = int(model.head.atom_encoder.weight.shape[0])
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=hidden_dim,
        n_layers=int(getattr(args, "n_layers", 3)),
        **dec_kw,
    )
    inner = getattr(decoder, "_decoder", None)
    if hasattr(inner, "eval"):
        inner.eval()
    print(
        f"[wfA53] decoder: decision={dec_kw.get('decision')} "
        f"threshold={dec_kw.get('decision_threshold', 0.5)}"
    )
    return decoder


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    t0 = time.time()

    ckpt_path = _resolve(str(args.ckpt_path))
    if not ckpt_path.is_file():
        raise SystemExit(f"--ckpt-path '{ckpt_path}' is not a file")

    # The checkpoint is opened before the CSV default is resolved because the
    # default IS the checkpoint's own training file.
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        load_cfm_checkpoint,
    )

    ckpt = load_cfm_checkpoint(str(ckpt_path), map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    csv_str = args.csv or ckpt_args.get("data_csv")
    if not csv_str:
        raise SystemExit("no --csv given and the checkpoint carries no data.csv")
    csv_path = _resolve(str(csv_str))
    if not csv_path.is_file():
        raise SystemExit(f"corpus CSV '{csv_path}' is not a file")

    report: Dict[str, Any] = {
        "script": "wf_a53_eval_cfm_test_split",
        "generated_at_unix": int(time.time()),
        "tag": "MEASURED",
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": _sha256(ckpt_path),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "best_epoch": ckpt.get("best_epoch") if isinstance(ckpt, dict) else None,
            "training_data_csv": ckpt_args.get("data_csv"),
            "training_n_train": ckpt_args.get("n_train"),
            "training_n_val": ckpt_args.get("n_val"),
        },
        "corpus": {
            "path": str(csv_path),
            "sha256": _sha256(csv_path),
            "split": str(args.split),
            "n_clouds_requested": args.n_clouds,
            "split_rows": _split_rows(csv_path, str(args.split)),
        },
    }

    # Is this split held out relative to THIS checkpoint?  Measured before
    # any model work, because the answer decides how the numbers may be read.
    overlap = _training_overlap(ckpt_args, csv_path, str(args.split))
    report["held_out_relative_to_checkpoint"] = overlap
    if overlap.get("verdict") == "NOT_HELD_OUT":
        print(
            "[wfA53] WARNING: this split is NOT held out relative to the "
            f"checkpoint — {overlap['n_shared_unique']}/"
            f"{overlap['n_eval_unique']} "
            f"({100.0 * overlap['frac_eval_seen_in_training']:.1f}%) of the "
            "evaluated molecules are in the checkpoint's own training split. "
            "The numbers below are an in-sample measurement, not a "
            "generalisation result."
        )
    elif overlap.get("verdict") == "HELD_OUT":
        print(
            f"[wfA53] held-out check: 0/{overlap['n_eval_unique']} molecules "
            "shared with the checkpoint's training split"
        )
    else:
        print(f"[wfA53] held-out check UNKNOWN: {overlap.get('reason')}")

    # ------------------------------------------------------------------
    # 1. Cohort
    # ------------------------------------------------------------------
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
    )

    print(
        f"[wfA53] building cohort: csv={csv_path} split={args.split!r} "
        f"n_clouds={args.n_clouds}"
    )
    clouds = build_csv_cohort(
        str(csv_path),
        split=str(args.split),
        n_clouds=args.n_clouds,
        seed=int(args.seed),
        metal_placement=str(ckpt_args.get("metal_placement", "bonded")),
        dative_distance_cutoff=float(
            ckpt_args.get("dative_distance_cutoff", 2.6)
        ),
    )
    summary = _cohort_summary(clouds)
    report["cohort"] = summary
    print(
        f"[wfA53] cohort built: n_clouds={summary['n_clouds']} "
        f"with_bond_truth={summary['n_with_bond_truth']} "
        f"atoms={summary['atoms_min']}-{summary['atoms_max']} "
        f"(mean {summary['atoms_mean']:.1f}) "
        f"metals={summary['clouds_by_metal']} "
        f"no_metal={summary['clouds_with_no_metal']}"
    )
    if summary["n_clouds"] == 0:
        # A split with no embeddable row is a data problem, not a model
        # result. Fail rather than emit a report full of zeros.
        _write(args.out_json, report)
        print("[wfA53] FAIL: the split embedded 0 clouds")
        return 1

    if args.dry_run:
        report["wall_seconds"] = round(time.time() - t0, 2)
        _write(args.out_json, report)
        print("[wfA53] PASS (dry-run: cohort only)")
        return 0

    # ------------------------------------------------------------------
    # 2. Model + decoder
    # ------------------------------------------------------------------
    import torch

    device = torch.device(args.device)
    model, model_args = _load_model(ckpt, device)
    decoder = _build_decoder(model, model_args, ckpt)

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _bond_decision_settings,
        _to_device_cloud,
        evaluate_bond_accuracy,
        evaluate_decode_ratio,
        evaluate_generation_validity,
    )

    clouds_d = [_to_device_cloud(c, device) for c in clouds]

    decision, threshold, bias = _bond_decision_settings(vars(model_args))
    metric_decision = "threshold" if str(decision) == "per_class" else decision

    # ------------------------------------------------------------------
    # 3. Bond accuracy (real geometry vs corpus truth)
    # ------------------------------------------------------------------
    n_bond = min(len(clouds_d), max(0, int(args.max_bond_clouds)))
    bond_metrics: Dict[str, float] = {}
    if n_bond:
        print(f"[wfA53] bond accuracy on {n_bond}/{len(clouds_d)} clouds ...")
        # The trainer's eval helpers do not carry @torch.no_grad(); the
        # training loop's per-cloud calls free their graphs between
        # iterations, but the generation pass integrates tens of ODE steps
        # in ONE graph. MEASURED 2026-10-19: without this guard the
        # generation pass reached 93.4 GiB allocated and OOM'd on a 95 GiB
        # GPU. Inference must not build a graph at all.
        with torch.no_grad():
            bond_metrics = evaluate_bond_accuracy(
                model=model,
                val_clouds=clouds_d[:n_bond],
                decision=metric_decision,
                threshold=threshold,
                class_logit_bias=bias,
                bond_decision_threshold_sweep=list(
                    getattr(model_args, "bond_decision_threshold_sweep", [0.5])
                    or [0.5]
                ),
            )
        print(
            f"[wfA53] bond: prec={bond_metrics.get('bond_precision', 0):.3f} "
            f"rec={bond_metrics.get('bond_recall', 0):.3f} "
            f"F1={bond_metrics.get('bond_f1', 0):.3f} "
            f"exact={bond_metrics.get('bond_exact_match', 0):.3f} "
            f"(n_scored={int(bond_metrics.get('n_scored', 0))})"
        )
    report["bond_accuracy"] = {
        "n_clouds_scored": n_bond,
        "n_clouds_available": len(clouds_d),
        "decision_rule": metric_decision,
        "threshold": threshold,
        "metrics": bond_metrics,
    }

    # ------------------------------------------------------------------
    # 4. Decode ratio (real geometry -> SMILES)
    # ------------------------------------------------------------------
    n_dec = min(len(clouds_d), max(0, int(args.max_decode_clouds)))
    dec_ok, dec_total = 0, 0
    if n_dec:
        print(f"[wfA53] decode ratio on {n_dec}/{len(clouds_d)} clouds ...")
        with torch.no_grad():
            dec_ok, dec_total = evaluate_decode_ratio(
                model=model,
                decoder=decoder,
                val_clouds=clouds_d[:n_dec],
                device=device,
                closure_bounded=bool(getattr(model_args, "closure_bounded", True)),
                closure_bias=ckpt.get("closure_bias"),
            )
        print(f"[wfA53] decode_ratio={dec_ok}/{dec_total}")
        frag = _decode_fragment_stats(decoder, clouds_d[:n_dec])
        if frag.get("n_decoded"):
            print(
                f"[wfA53] decoded connectivity: "
                f"single_fragment={frag['n_single_fragment']}/"
                f"{frag['n_decoded']} "
                f"({100.0 * frag['single_fragment_ratio']:.1f}%), "
                f"fragments mean={frag['fragments_mean']:.1f} "
                f"max={frag['fragments_max']}, "
                f"metal_containing={100.0 * frag['metal_ratio']:.1f}%"
            )
        else:
            frag = {"available": True, "n_decoded": 0}
    else:
        frag = {"available": False}
    report["decode_ratio"] = {
        "n_ok": int(dec_ok),
        "n_total": int(dec_total),
        "ratio": (dec_ok / dec_total) if dec_total else 0.0,
        "n_clouds_available": len(clouds_d),
        # A high ratio with a low single_fragment_ratio is the loose-atom
        # artifact, not a good decode.  Kept adjacent so the pair is read
        # together rather than the ratio alone.
        "decoded_connectivity": frag,
    }

    # ------------------------------------------------------------------
    # 5. Generation validity (noise -> geometry -> SMILES)
    # ------------------------------------------------------------------
    gen_ok, gen_total = 0, 0
    if int(args.gen_samples) > 0:
        print(
            f"[wfA53] generation validity on {int(args.gen_samples)} "
            f"compositions ({int(args.gen_steps)} ODE steps) ..."
        )
        gen_smiles: List[str] = []
        with torch.no_grad():
            gen_ok, gen_total = evaluate_generation_validity(
                model=model,
                decoder=decoder,
                templates=clouds_d,
                device=device,
                method=str(getattr(model_args, "method", "euler")),
                n_steps=int(args.gen_steps),
                n_samples=int(args.gen_samples),
                seed=int(args.seed),
                noise_sigma=float(getattr(model_args, "noise_sigma", 1.0)),
                pad_groups=True,
                pad_group_bucket=8,
                smiles_out=gen_smiles,
            )
        print(f"[wfA53] gen_validity={gen_ok}/{gen_total}")
        gen_frag = _smiles_fragment_stats(gen_smiles)
        if gen_frag.get("n_decoded"):
            print(
                f"[wfA53] generated connectivity: "
                f"single_fragment={gen_frag['n_single_fragment']}/"
                f"{gen_frag['n_decoded']} "
                f"({100.0 * gen_frag['single_fragment_ratio']:.1f}%), "
                f"fragments mean={gen_frag['fragments_mean']:.1f} "
                f"max={gen_frag['fragments_max']}, "
                f"metal_containing={100.0 * gen_frag['metal_ratio']:.1f}%"
            )
    else:
        gen_frag = {"available": False}
    report["generation_validity"] = {
        "n_ok": int(gen_ok),
        "n_total": int(gen_total),
        "ratio": (gen_ok / gen_total) if gen_total else 0.0,
        "n_steps": int(args.gen_steps),
        "n_samples": int(args.gen_samples),
        "decoded_connectivity": gen_frag,
    }

    report["wall_seconds"] = round(time.time() - t0, 2)
    _write(args.out_json, report)
    print(f"[wfA53] PASS -> {args.out_json} ({report['wall_seconds']:.1f}s)")
    return 0


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path = path if path.is_absolute() else (_REPO_ROOT / path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
