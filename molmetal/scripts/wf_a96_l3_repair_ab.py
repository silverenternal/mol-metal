#!/usr/bin/env python3
"""Wf-A96 (2026-10-19) -- causal A/B of the decode-side aromatic repairs.

WHY THIS EXISTS
---------------
cfm_todo.json::L3_A_B_Phase_3_redo asked for a re-measure of the L3 control
arm's 0.109 decode rate after Wf-A79/83/84 shipped.  Re-running the probe is
not enough to attribute the change: the arm JSON on disk is dated 2026-10-12
while the checkpoint at the path it names has mtime 2026-10-13 20:17, so the
file measured then is not the file on disk now.  A before/after across two
different checkpoints would be a checkpoint delta wearing a repair's name.

This script isolates the repair instead.  It integrates the ODE ONCE on the
current checkpoint, then decodes the SAME 64 clouds twice:

  arm "shipped"     -- the shipped path (Wf-A79 + Wf-A83 + Wf-A84)
  arm "no_repairs"  -- the same weights with the three repair functions
                       monkey-patched to identity

Everything else (weights, coordinates, decision rule, dative cutoff) is
identical between the arms, so the delta is the repair's and nothing else.
The monkey-patch is process-local; it is reverted before the process exits
and it touches no file.

OUTPUT
------
JSON with, per arm: decode_rate, n_with_smiles, error classes, aromatic
share of survivors, single-fragment share.  Written atomically (AGENTS.md
5.6) and stamped with the checkpoint's sha256 + mtime so a later reader can
tell WHICH file this measured (AGENTS.md 5.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    got = [r for r in rows if r.get("smiles")]
    errs: Counter = Counter()
    for r in rows:
        if r.get("smiles"):
            continue
        e = str(r.get("decode_error") or "")
        errs[e.split(":")[0] if e else (r.get("error") or "unknown")] += 1
    arom = [r.get("n_aromatic_atoms") for r in got if isinstance(r.get("n_aromatic_atoms"), int)]
    return {
        "n_attempted": len(rows),
        "n_with_smiles": len(got),
        "decode_rate": (len(got) / len(rows)) if rows else 0.0,
        "error_classes": dict(errs.most_common()),
        "single_fragment_ratio": (
            sum(1 for r in got if r.get("n_fragments") == 1) / len(got) if got else None
        ),
        "frac_any_aromatic": (sum(1 for v in arom if v > 0) / len(arom)) if arom else None,
        "mean_aromatic_atoms": (sum(arom) / len(arom)) if arom else None,
        "n_repaired": sum(1 for r in got if r.get("repair")),
        "repair_counts": dict(Counter(str(r.get("repair") or "<none>") for r in got)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aromatic-repair A/B on one checkpoint.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-clouds", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ns = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(ns.config.read_text(encoding="utf-8")) or {}
    model_cfg = cfg.get("model", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    ckpt = Path(model_cfg["ckpt"])
    n_clouds = int(ns.n_clouds or eval_cfg.get("n_clouds", 64))
    n_steps = int(ns.n_steps or eval_cfg.get("n_steps", 50))
    method = str(ns.method or eval_cfg.get("method", "midpoint"))
    seed = int(ns.seed or eval_cfg.get("seed", 1234))

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "molmetal" / "scripts"))
    from wf_blkc_l3_guidance_probe import decode, integrate, load_clouds, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a, state = load_model(ckpt, device)
    clouds = load_clouds(a, n_clouds)
    sampled = integrate(
        model,
        clouds,
        device,
        method=method,
        n_steps=n_steps,
        seed=seed,
        noise_sigma=float(a.get("fm_noise_sigma", 1.0)),
        guidance_strength=0.0,
    )

    import molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond as _dec_mod

    _names = (
        "_demote_aromatic_outside_rings",
        "_demote_small_ring_aromatic",
        "_demote_all_aromatic",
    )
    _orig = {n: getattr(_dec_mod, n) for n in _names}

    run_args = dict(a)
    shipped = decode(model, sampled, run_args, state=state)

    for n in _names:
        setattr(_dec_mod, n, lambda mol: mol)
    try:
        unpatched = decode(model, sampled, run_args, state=state)
    finally:
        for n, fn in _orig.items():
            setattr(_dec_mod, n, fn)

    report = {
        "metric": "cfm_l3_aromatic_repair_ab",
        "tag": "MEASURED",
        "date": "2026-10-19",
        "checkpoint": {
            "path": str(ckpt),
            "sha256": _sha256(ckpt),
            "mtime_iso": __import__("datetime").datetime.fromtimestamp(
                ckpt.stat().st_mtime
            ).isoformat(timespec="seconds"),
        },
        "cohort": {
            "source": "tmQM real, split=val (composition templates)",
            "n_clouds": len(clouds),
            "method": method,
            "n_steps": n_steps,
            "seed": seed,
            "device": str(device),
            "mode": "eval() + one forward ODE integration, decoded twice",
        },
        "arms": {
            "shipped": _summarise(shipped),
            "no_repairs": _summarise(unpatched),
        },
    }
    report["delta"] = {
        "decode_rate": (
            report["arms"]["shipped"]["decode_rate"]
            - report["arms"]["no_repairs"]["decode_rate"]
        ),
        "n_with_smiles": (
            report["arms"]["shipped"]["n_with_smiles"]
            - report["arms"]["no_repairs"]["n_with_smiles"]
        ),
    }

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = ns.out.with_suffix(ns.out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    os.replace(tmp, ns.out)
    print(json.dumps(report, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
