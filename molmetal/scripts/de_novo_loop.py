#!/usr/bin/env python3
"""De novo metal-drug design loop.

==============================================================================
CFM (noise -> 3D complex) -> YuelBond decode -> SMILES -> D-MPNN pIC50
==============================================================================

This closes the loop the project's architecture describes as the "3D generation
chain" (M5 EGNN + M6 CFM + M7 YuelBond) but never wired to a scorer.

Before this script, the two halves existed and nothing joined them:

  * `train_cfm_yuelbond_wasserstein.py` trains the generator and had, until
    2026-10-04, NO sampling path at all -- `generate_clouds_from_noise` is new,
    as is the `EGNNVelocityField` it integrates.
  * `train_dmpnn_multitask_wasserstein.py` trains the pIC50 predictor and
    writes `dmpnn_mtw_{metal}_seed{N}.pt`.
  * `molmetal/scoring/score_any.py` routes 24 metrics, ALL of them docking /
    pose / synthesis / OOD / QED / SA -- there is no activity route, so nothing
    could ask "what is this molecule's predicted potency?".

Each stage is therefore a real, previously-unconnected component, and this
script is the connections.

==============================================================================
Stages
==============================================================================

1. COMPOSE  -- draw atom compositions from a corpus.  With `--tmqm-dir` these
   are real tmQM complexes of the target metal (so the atom counts and element
   mixes are those of real chemistry).  Without it, the synthetic Pt-click
   template is used and the run is a pipeline smoke, not a design run.
2. GENERATE -- integrate the EGNN velocity field from x0 ~ N(0, I) to t = 1.
   The radius graph is rebuilt at every ODE step.
3. DECODE   -- YuelBond bond-graph decoder, RDKit-strict.
4. SCORE    -- featurise the decoded SMILES and run the D-MPNN's per-metal
   regression head.  The metal selects which checkpoint is loaded.

==============================================================================
Honest framing (AGENTS.md §5.2)
==============================================================================

Every number this prints is a MODEL OUTPUT of the specific checkpoints named in
the header it writes.  It is not a measurement of anything in a lab, and a
predicted pIC50 is not an activity.  The script always records, per run:

  * the CFM checkpoint path and the epoch it was saved at,
  * the D-MPNN checkpoint path and ITS recorded val_auc / epoch,
  * how many generated clouds decoded to an RDKit-valid molecule,

so a reader can tell an under-trained smoke from a real run without reading
this docstring.  A checkpoint saved at epoch 1 will produce a nearly flat score
distribution -- that is expected and is reported, not hidden.

Usage::

    MOLMETAL_DATA_DIR=.data/molmetal \\
    python -m molmetal.scripts.de_novo_loop \\
        --metal Ru --n-samples 64 --tmqm-dir .data/molmetal/tmQM \\
        --cfm-ckpt molmetal/checkpoints/cfm_yuelbond_wasserstein_seed0.pt \\
        --dmpnn-ckpt molmetal/checkpoints/dmpnn_mtw_ru_seed0.pt \\
        --out-dir molmetal/reports/de_novo_ru
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Must run BEFORE any torch / molmetal import: CUDA reads
# CUDA_VISIBLE_DEVICES once, at init.  See molmetal/scripts/_cuda_default.py.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))
# PROJECT_ROOT must precede the package import in the --device auto branch.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from _cuda_default import ensure_cuda_visible_devices_default as _cvd

_cvd()
del _cvd
# --- end CUDA default --------------------------------------------


# ---------------------------------------------------------------------------
# Module loading — both model classes live inside their trainer scripts.
# ---------------------------------------------------------------------------
def _load_module(path: Path, name: str):
    """Import a script as a module (they guard their entry points)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfm_module():
    return _load_module(
        PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py",
        "_dn_cfm",
    )


def _dmpnn_module():
    return _load_module(
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_dmpnn_multitask_wasserstein.py",
        "_dn_dmpnn",
    )


# ---------------------------------------------------------------------------
# Stage 4 — the pIC50 scorer
# ---------------------------------------------------------------------------
def load_pic50_scorer(ckpt_path: Path, device: torch.device):
    """Build the D-MPNN and load ``ckpt_path`` into it.

    The checkpoint carries everything needed to reconstruct the architecture
    (``hidden``, ``depth``) plus its own provenance (``epoch``, ``val_auc``),
    so nothing has to be guessed from the filename.
    """
    dmp = _dmpnn_module()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state_dict" not in ck:
        raise SystemExit(
            f"{ckpt_path} does not look like a D-MPNN checkpoint (keys: "
            f"{list(ck)[:6] if isinstance(ck, dict) else type(ck).__name__})"
        )
    model = dmp.MultiTaskWassersteinModel(
        n_metals=1,  # one metal per checkpoint; see the load check below
        hidden=int(ck.get("hidden", dmp.DEFAULT_HIDDEN)),
        depth=int(ck.get("depth", dmp.DEFAULT_DEPTH)),
        use_dann=any(
            k.startswith("domain_head.") for k in ck["model_state_dict"]
        ),  # key form is domain_head.classifier.0.weight, not domain_head.0.*
    )
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"architecture mismatch loading {ckpt_path}: {len(missing)} missing "
            f"and {len(unexpected)} unexpected keys (e.g. {list(missing)[:3]}). "
            "Refusing to score with a partially-initialised model."
        )
    model.eval().to(device)
    return model, {
        "path": str(ckpt_path),
        "epoch": ck.get("epoch"),
        "val_auc": ck.get("val_auc"),
        "test_auc": ck.get("test_auc"),
        "metal": ck.get("metal"),
        "n_train": ck.get("n_train") or ck.get("n_joint_train"),
    }


def score_pic50(
    model,
    smiles: Sequence[str],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(pic50, active_prob)`` for ``smiles``.

    ``predict_joint`` drops rows whose ``pic50`` argument is NaN (it needs the
    labels to build the collated batch), so a real placeholder is passed and
    the returned ``active_true`` ignored.  Passing NaN here silently returns
    all zeros — that is a real failure mode this signature exists to avoid.
    """
    dmp = _dmpnn_module()
    from molmetal.baselines.dmpnn import featurize_smiles_list

    graphs = featurize_smiles_list([str(s) for s in smiles])
    placeholder = np.zeros(len(graphs), dtype=float)  # never NaN — see above
    active = np.zeros(len(graphs), dtype=float)
    _, active_prob, pic50 = dmp.predict_joint(
        model,
        graphs,
        [0] * len(graphs),
        placeholder,
        active,
        batch_size=batch_size,
        device=device,
    )
    return np.asarray(pic50, dtype=float), np.asarray(active_prob, dtype=float)


# ---------------------------------------------------------------------------
# Stages 1-3 — compose, generate, decode
# ---------------------------------------------------------------------------
def compose_templates(
    cfm_mod,
    metal: str,
    n_samples: int,
    tmqm_dir: Optional[str],
) -> Tuple[List[Any], str]:
    """Return ``(templates, source_description)``.

    With ``tmqm_dir`` the compositions are real complexes of ``metal``; without
    it the synthetic Pt-click layout is used and the caller must say so.
    """
    if tmqm_dir:
        from molmetal.data.tmqm_clouds import load_tmqm_clouds

        clouds = load_tmqm_clouds(
            tmqm_dir,
            metals=[metal],
            coord_number=None,
            n_clouds=n_samples,
            seed=0,
        )
        if clouds:
            return list(clouds), f"tmQM real compositions ({metal}, n={len(clouds)})"
        print(
            f"[de-novo][WARNING] tmQM returned no clouds for {metal!r}; "
            "falling back to the synthetic template",
            file=sys.stderr,
            flush=True,
        )
    base = cfm_mod.build_synthetic_pt_click_clouds(n_clouds=max(1, n_samples), seed=0)
    return list(base), "SYNTHETIC Pt-click template (pipeline smoke, not a design run)"


def generate_and_decode(
    cfm_mod,
    model,
    decoder,
    templates: Sequence[Any],
    device: torch.device,
    method: str,
    n_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Noise -> velocity-field ODE -> decode.  Returns one record per template."""
    clouds = cfm_mod.generate_clouds_from_noise(
        model, templates, device, method=method, n_steps=n_steps, seed=seed
    )
    out: List[Dict[str, Any]] = []
    for i, cloud in enumerate(clouds):
        rec: Dict[str, Any] = {
            "index": i,
            "n_atoms": int(cloud.atomic_numbers.shape[0]),
            "decoded": False,
            "smiles": "",
            "sanitized": False,
        }
        try:
            res = decoder.decode(cloud)
            dec = getattr(res, "decoded", None)
            if dec is not None:
                rec["smiles"] = str(getattr(dec, "smiles", "") or "")
                rec["sanitized"] = bool(getattr(dec, "sanitized", False))
                rec["decoded"] = bool(rec["smiles"])
        except Exception as exc:  # noqa: BLE001 — a decode failure is data
            rec["error"] = f"{type(exc).__name__}: {exc}"

        # Chemical sanity, which `decoded` alone does NOT establish.
        #
        # RDKit will happily parse `C.C#C.C#C.[Ru]` — a pile of disconnected
        # one-carbon fragments — so a decode rate of 15/16 can coexist with
        # every candidate being meaningless.  A generated metal complex should
        # come out as ONE connected molecule containing the template's metal,
        # so both are checked and reported separately from the parse rate.
        if rec["decoded"]:
            try:
                from rdkit import Chem, RDLogger

                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(rec["smiles"])
                if mol is None:
                    rec["sane"] = False
                    rec["sane_reason"] = "reparse_failed"
                else:
                    n_frag = len(Chem.GetMolFrags(mol))
                    rec["n_fragments"] = int(n_frag)
                    rec["has_metal"] = any(
                        a.GetSymbol() == metal for a in mol.GetAtoms()
                    )
                    rec["sane"] = bool(n_frag == 1 and rec["has_metal"])
                    if not rec["sane"]:
                        rec["sane_reason"] = (
                            f"{n_frag}_fragments" if n_frag != 1 else "no_metal"
                        )
            except Exception as exc:  # noqa: BLE001
                rec["sane"] = False
                rec["sane_reason"] = f"{type(exc).__name__}"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metal", default="Ru", help="Pt/Ru/Ir/Rh/Os/Re")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--tmqm-dir", default=None)
    ap.add_argument(
        "--cfm-ckpt",
        default="molmetal/checkpoints/cfm_yuelbond_wasserstein_seed0.pt",
    )
    ap.add_argument("--dmpnn-ckpt", default=None, help="defaults to dmpnn_mtw_{metal}_seed0.pt")
    ap.add_argument("--method", default="rk4", choices=("euler", "midpoint", "heun", "rk2", "rk4"))
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument(
        "--device", default="auto", help="auto / cpu / cuda (auto -> cuda if allocatable)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if args.device == "auto":
        from molmetal.molmetal_lam.sbdd_env.training_utils import resolve_default_device

        device = resolve_default_device()
    else:
        device = torch.device(args.device)
    metal = str(args.metal).strip()
    cfm_ckpt = PROJECT_ROOT / args.cfm_ckpt
    dmpnn_ckpt = (
        Path(args.dmpnn_ckpt)
        if args.dmpnn_ckpt
        else Path("molmetal") / "checkpoints" / f"dmpnn_mtw_{metal.lower()}_seed0.pt"
    )
    dmpnn_ckpt = dmpnn_ckpt if dmpnn_ckpt.is_absolute() else PROJECT_ROOT / dmpnn_ckpt

    if not cfm_ckpt.is_file():
        raise SystemExit(f"CFM checkpoint not found: {cfm_ckpt}")
    if not dmpnn_ckpt.is_file():
        raise SystemExit(
            f"D-MPNN checkpoint not found: {dmpnn_ckpt}.  Train one, or pass "
            "--dmpnn-ckpt.  Without a trained predictor there is nothing to "
            "rank candidates against."
        )

    print(f"[de-novo] metal={metal} device={device}", flush=True)
    cfm_mod = _cfm_module()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import YuelBondBondDecoder

    # --- CFM + decoder ---------------------------------------------------
    cfm_state = torch.load(cfm_ckpt, map_location="cpu", weights_only=False)
    cfm_args = cfm_state.get("args", {}) if isinstance(cfm_state, dict) else {}
    # The bond head's atom encoder width is set by the YuelBond vocab preset
    # (13 entries for "pt", 18 for "all6" — the preset that adds Ru/Ir/Rh/Os/Re).
    # Rebuilding from the wrong preset is a shape mismatch on
    # head.atom_encoder.weight, so read it back out of the checkpoint.
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        get_z_table as _get_z_table,
    )

    _vocab = str(cfm_args.get("yuelbond_vocab", "pt") or "pt")
    cfm_model = cfm_mod.CFMJointModel(
        hidden_dim=int(cfm_args.get("hidden_dim", cfm_mod.DEFAULT_HIDDEN_DIM)),
        n_layers=int(cfm_args.get("n_layers", cfm_mod.DEFAULT_N_LAYERS)),
        cutoff=float(cfm_args.get("edge_cutoff", cfm_mod.DEFAULT_EDGE_CUTOFF)),
        fm_objective=bool(cfm_args.get("use_fm_objective", False)),
        z_table=_get_z_table(_vocab),
        use_egnn_velocity=bool(cfm_args.get("use_egnn_velocity", False)),
        max_atomic_number=int(cfm_args.get("egnn_max_atomic_number", 100)),
    )
    cfm_missing, cfm_unexpected = cfm_model.load_state_dict(
        cfm_state.get("model_state_dict", cfm_state), strict=False
    )
    if cfm_missing or cfm_unexpected:
        raise SystemExit(
            f"architecture mismatch loading {cfm_ckpt}: {len(cfm_missing)} "
            f"missing / {len(cfm_unexpected)} unexpected (e.g. "
            f"{list(cfm_missing)[:3]}).  A checkpoint written before "
            "--use-egnn-velocity has vel_probe keys and no vel_field keys."
        )
    # 2026-10-04 — the sampler NEEDS an EGNN velocity field.  Every CFM
    # checkpoint written before --use-egnn-velocity existed carries only the
    # 2-layer MLP probes, which were never a sampler: the trainer had no
    # generation path at all until this pass.  Fail with the reason rather
    # than an AttributeError from inside the ODE solver.
    if not getattr(cfm_model, "use_egnn_velocity", False):
        raise SystemExit(
            f"{cfm_ckpt} was trained without --use-egnn-velocity, so it has no "
            "velocity field to integrate: its CFMJointModel holds only the "
            "vel_probe/fm_vel_head placeholders, and the trainer had no "
            "sampling path before 2026-10-04.  Train a generator first, e.g.\n"
            "  " + str(PROJECT_ROOT / ".venv/bin/python") + " -m "
            "molmetal.scripts.train_cfm_yuelbond_wasserstein \\\n"
            "    --config configs/cfm_tmqm_all6.yaml --seed 0 --epochs 2 \\\n"
            "    --n-train 64 --n-val 32 --out-dir /tmp/cfm_egnn \\\n"
            "    --ckpt-dir /tmp/cfm_egnn/ck --device cuda\n"
            "then pass --cfm-ckpt /tmp/cfm_egnn/ck/cfm_yuelbond_wasserstein_seed0.pt"
        )
    cfm_model.eval().to(device)
    # The decoder MUST share the trained bond head.  `YuelBondBondDecoder()`
    # with no `head=` argument constructs a FRESHLY INITIALISED head, and the
    # trainer's own wiring passes `head=model.head` for exactly this reason
    # ("The decoder's head *is* model.head — we share weights so training
    # updates the decoder too").  Getting this wrong is not a crash: it
    # silently decodes with random weights, which varies between processes and
    # produces parseable garbage, so the run looks like it worked.
    decoder = YuelBondBondDecoder(
        head=cfm_model.head,
        hidden_dim=int(cfm_args.get("hidden_dim", cfm_mod.DEFAULT_HIDDEN_DIM)),
        n_layers=int(cfm_args.get("n_layers", cfm_mod.DEFAULT_N_LAYERS)),
    )

    # --- compositions ----------------------------------------------------
    templates, source = compose_templates(cfm_mod, metal, args.n_samples, args.tmqm_dir)

    # --- generate + decode ----------------------------------------------
    records = generate_and_decode(
        cfm_mod, cfm_model, decoder, templates, device,
        method=args.method, n_steps=args.n_steps, seed=args.seed,
    )
    n_decoded = sum(1 for r in records if r["decoded"])
    n_sanitized = sum(1 for r in records if r["sanitized"])
    n_sane = sum(1 for r in records if r.get("sane"))
    print(
        f"[de-novo] generated {len(records)}  decoded={n_decoded}  "
        f"rdkit_strict={n_sanitized}  CHEMICALLY_SANE={n_sane}  source={source}",
        flush=True,
    )
    if n_decoded and not n_sane:
        print(
            "[de-novo][WARNING] every candidate parsed but none is a single "
            "connected metal-containing molecule — the generator produced "
            "fragments. Check the CFM checkpoint's epoch before reading the "
            "scores below.",
            flush=True,
        )

    # --- score -----------------------------------------------------------
    scorer, predictor_meta = load_pic50_scorer(dmpnn_ckpt, device)
    scored: List[Dict[str, Any]] = []
    # Rank the chemically sensible ones; fall back to merely-parseable
    # only so the run still reports something, and say when that happened.
    valid = [r for r in records if r.get("sane")] or [
        r for r in records if r["decoded"]
    ]
    ranking_basis = "sane" if any(r.get("sane") for r in valid) else "parseable_only"
    if valid:
        pics, probs = score_pic50(scorer, [r["smiles"] for r in valid], device)
        for rec, p, a in zip(valid, pics, probs):
            rec["pred_pic50"] = float(p)
            rec["pred_active_prob"] = float(a)
            scored.append(rec)
        scored.sort(key=lambda r: r["pred_pic50"], reverse=True)

    # --- report ----------------------------------------------------------
    verdict = {
        "metal": metal,
        "composition_source": source,
        "generation": {
            "solver": f"{args.method}/{args.n_steps}",
            "n_generated": len(records),
            "n_decoded": n_decoded,
            "n_rdkit_strict": n_sanitized,
            "n_chemically_sane": n_sane,
            "sane_rate": n_sane / max(len(records), 1),
            "ranking_basis": ranking_basis,
            "decode_rate": n_decoded / max(len(records), 1),
        },
        "cfm_checkpoint": {
            "path": str(cfm_ckpt),
            "epoch": cfm_state.get("epoch"),
            "decode_ratio_at_save": cfm_state.get("decode_ratio"),
            "use_egnn_velocity": bool(cfm_args.get("use_egnn_velocity", False)),
        },
        "pic50_predictor": predictor_meta,
        "score_summary": (
            {
                "n_scored": len(scored),
                "min": float(min(r["pred_pic50"] for r in scored)),
                "max": float(max(r["pred_pic50"] for r in scored)),
                "mean": float(np.mean([r["pred_pic50"] for r in scored])),
            }
            if scored
            else {"n_scored": 0}
        ),
        "honest_framing": (
            "Every pIC50 here is a MODEL OUTPUT of the checkpoints named above, "
            "not a measured activity. A predictor whose recorded epoch is 1 "
            "produces a near-flat score distribution — that is reported in "
            "`pic50_predictor.epoch`, not hidden. A synthetic composition "
            "source means this run exercised the pipeline, it did not design a "
            "molecule."
        ),
        "records": scored or records,
    }

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "molmetal" / "reports" / f"de_novo_{metal.lower()}"
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "de_novo.json").write_text(json.dumps(verdict, indent=2, default=str))
    with (out_dir / "candidates.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "smiles", "n_atoms", "rdkit_strict", "n_fragments", "sane", "pred_pic50", "pred_active_prob"])
        for i, r in enumerate(scored, 1):
            w.writerow([i, r["smiles"], r["n_atoms"], int(r["sanitized"]),
                        r.get("n_fragments", ""), int(bool(r.get("sane"))),
                        f"{r['pred_pic50']:.4f}", f"{r['pred_active_prob']:.4f}"])
    print(f"[de-novo] wrote {out_dir / 'de_novo.json'} + candidates.csv", flush=True)
    if scored:
        print(f"[de-novo] top-3 predicted pIC50: "
              f"{[round(r['pred_pic50'], 3) for r in scored[:3]]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
