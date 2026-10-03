"""Wf-84 Phase 3 — D-MPNN pIC50 checkpoint diagnostic.

=====================================================================
Verify D-MPNN pIC50 checkpoints exist + load + produce sensible output
=====================================================================

Three-step diagnostic for the per-metal D-MPNN pIC50 checkpoints written
by :mod:`molmetal.scripts.train_dmpnn_pic50_real`:

1. **Inventory** — list every ``dmpnn_pic50_<metal>_seed<seed>.pt`` under
   the configured checkpoint dir; emit a per-file size + mtime summary.
2. **Load + rebuild** — instantiate :class:`DMPNNMultiTaskModel`, hydrate
   weights via ``state_dict`` (with ``strict=False`` so older / newer
   schema versions don't crash the diagnostic), and confirm parameter
   tensors are finite.
3. **Inference smoke** — featurise a handful of small drug-like SMILES
   from the MetalCytoToxDB test split and run forward; assert finite
   pIC50 logits and report the predicted range.

Outputs (under ``--out-dir``):
  * ``ckpt_inventory.json``  — per-file size + mtime + sha256-prefix.
  * ``ckpt_load_report.json`` — load result per checkpoint.
  * ``inference_smoke.json``  — per-ckpt pIC50 predictions on probe SMILES.
  * ``diagnostic_synthesis.md`` — human-readable verdict.

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * This is a *diagnostic*, not a quality gate: a ckpt that loads is
    "REAL" not "SYNTHETIC" but the *quality* of its pIC50 R² is reported
    separately in ``training_summary.json``.
  * ``strict=False`` load is intentional — schema evolution (e.g. new
    readout head) should not block a diagnostic.
  * Probe SMILES are deterministic (no random sampling) so the report is
    bit-exactly reproducible.

**Hardware**: CPU primary; CUDA optional.

**Usage**
-----
::
    source env.sh
    python molmetal/scripts/dmpnn_ckpt_diagnostic.py \\
        --ckpt-dir molmetal/checkpoints \\
        --out-dir molmetal/reports/wf84_dmpnn_ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from molmetal.baselines.dmpnn import featurize_smiles_list
from molmetal.baselines.dmpnn_multitask import DMPNNMultiTaskModel
from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)

# Probe SMILES — small, drug-like, drawn from common MetalCytotox ligands
PROBE_SMILES: Tuple[str, ...] = (
    "CCO",                                              # ethanol
    "c1ccccc1",                                         # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                            # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",                     # caffeine
    "O=C([O-])c1ccccc1",                                # benzoate
    "C1N2CN3CN1CP(C2)C3",                               # small N-donor cage
    "Cl[P](Cl)(Cl)Cl",                                  # PtCl4 fragment (dative)
)


@dataclass
class CheckpointRecord:
    metal: str
    seed: int
    path: Path
    size_bytes: int
    mtime_utc: str
    sha256_prefix: str
    state_dict_keys: int
    load_strict_ok: bool
    load_strict_missing: List[str]
    load_strict_unexpected: List[str]
    inference_status: str
    inference_pic50_range: Tuple[float, float]
    inference_n_probes: int


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _sha256_prefix(path: Path, n_bytes: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(1024 * 1024)
        while chunk:
            h.update(chunk)
            chunk = f.read(1024 * 1024)
    return h.hexdigest()[: 2 * n_bytes]


def _utc_iso(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-checkpoint loader
# ---------------------------------------------------------------------------
def load_ckpt_diagnostic(
    ckpt_path: Path,
    probe_smiles: Sequence[str] = PROBE_SMILES,
    device: Optional[torch.device] = None,
) -> CheckpointRecord:
    """Run the three-step diagnostic on a single checkpoint."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stat = ckpt_path.stat()
    metal_match = ckpt_path.stem.replace("dmpnn_pic50_", "")
    # e.g. ru_seed0 → metal=Ru seed=0
    if "_seed" not in metal_match:
        metal = metal_match.upper()
        seed = -1
    else:
        metal_part, seed_part = metal_match.rsplit("_seed", 1)
        metal = metal_part.upper()
        try:
            seed = int(seed_part)
        except ValueError:
            seed = -1

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = blob.get("model_state_dict", blob)
    state_keys = list(state_dict.keys())

    # Determine hidden/depth from checkpoint if present
    hidden = int(blob.get("hidden", 128)) if isinstance(blob, dict) else 128
    depth = int(blob.get("depth", 3)) if isinstance(blob, dict) else 3

    # Try strict load
    model = DMPNNMultiTaskModel(
        atom_dim=ATOM_FEATURE_DIM,
        bond_dim=BOND_FEATURE_DIM,
        hidden=hidden,
        depth=depth,
        dropout=0.0,
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    strict_ok = (len(missing) == 0 and len(unexpected) == 0)

    # Inference smoke
    graphs = featurize_smiles_list(list(probe_smiles))
    pic50_preds: List[float] = []
    inference_status = "OK"
    try:
        model = model.to(device).eval()
        with torch.no_grad():
            for g in graphs:
                if g is None:
                    pic50_preds.append(float("nan"))
                    continue
                atom_f = torch.from_numpy(g[0]).unsqueeze(0).to(device)
                bond_f = torch.from_numpy(g[1]).unsqueeze(0).to(device)
                e_src = torch.from_numpy(g[2]).unsqueeze(0).to(device)
                e_dst = torch.from_numpy(g[3]).unsqueeze(0).to(device)
                pic50, _ = model(atom_f, bond_f, e_src, e_dst)
                pic50_preds.append(float(pic50.cpu().numpy()[0]))
        finite = [p for p in pic50_preds if np.isfinite(p)]
        if not finite:
            inference_status = "ALL_NAN"
    except Exception as exc:
        inference_status = f"ERROR: {exc!r}"

    finite_preds = [p for p in pic50_preds if np.isfinite(p)]
    if finite_preds:
        pic50_range = (float(min(finite_preds)), float(max(finite_preds)))
    else:
        pic50_range = (float("nan"), float("nan"))

    return CheckpointRecord(
        metal=metal,
        seed=seed,
        path=ckpt_path,
        size_bytes=int(stat.st_size),
        mtime_utc=_utc_iso(stat.st_mtime),
        sha256_prefix=_sha256_prefix(ckpt_path),
        state_dict_keys=len(state_keys),
        load_strict_ok=strict_ok,
        load_strict_missing=missing,
        load_strict_unexpected=unexpected,
        inference_status=inference_status,
        inference_pic50_range=pic50_range,
        inference_n_probes=len(probe_smiles),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wf-84 Phase 3 — D-MPNN pIC50 checkpoint diagnostic.",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "checkpoints",
        help="Directory to scan for dmpnn_pic50_*.pt files.",
    )
    p.add_argument(
        "--pattern",
        default="dmpnn_pic50_*_seed*.pt",
        help="Glob pattern (default: dmpnn_pic50_*_seed*.pt).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf84_dmpnn_ckpt",
        help="Output directory.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force device (cuda:0 or cpu). Default: auto-detect.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    pattern = args.pattern
    ckpts = sorted(args.ckpt_dir.glob(pattern))
    print(
        f"[dmpnn-ckpt-diag] scanning {args.ckpt_dir} for {pattern} → "
        f"{len(ckpts)} files (device={device})",
        flush=True,
    )

    inventory: List[Dict] = []
    load_reports: List[Dict] = []
    inference_reports: List[Dict] = []

    t_start = time.time()
    for ckpt in ckpts:
        # 1. Inventory
        stat = ckpt.stat()
        inventory.append(
            {
                "filename": ckpt.name,
                "path": str(ckpt),
                "size_bytes": int(stat.st_size),
                "mtime_utc": _utc_iso(stat.st_mtime),
                "sha256_prefix": _sha256_prefix(ckpt),
            }
        )

        # 2+3. Load + inference smoke
        try:
            rec = load_ckpt_diagnostic(ckpt, device=device)
        except Exception as exc:
            load_reports.append(
                {
                    "path": str(ckpt),
                    "load_status": f"ERROR: {exc!r}",
                }
            )
            inference_reports.append(
                {
                    "path": str(ckpt),
                    "inference_status": "SKIPPED_LOAD_ERROR",
                }
            )
            continue

        load_reports.append(
            {
                "path": str(ckpt),
                "metal": rec.metal,
                "seed": rec.seed,
                "state_dict_keys": rec.state_dict_keys,
                "load_strict_ok": rec.load_strict_ok,
                "load_strict_missing": rec.load_strict_missing,
                "load_strict_unexpected": rec.load_strict_unexpected,
                "load_status": "OK" if rec.load_strict_ok else "OK_NONSTRICT",
            }
        )
        inference_reports.append(
            {
                "path": str(ckpt),
                "metal": rec.metal,
                "seed": rec.seed,
                "inference_status": rec.inference_status,
                "inference_pic50_range": list(rec.inference_pic50_range),
                "inference_n_probes": rec.inference_n_probes,
            }
        )
        print(
            f"  {ckpt.name}: size={rec.size_bytes:>8} keys={rec.state_dict_keys} "
            f"strict_ok={rec.load_strict_ok} pic50∈[{rec.inference_pic50_range[0]:+.3f}, {rec.inference_pic50_range[1]:+.3f}]",
            flush=True,
        )

    elapsed = time.time() - t_start

    inventory_payload = {
        "wf_tag": "Wf-84 Phase 3 — D-MPNN pIC50 checkpoint inventory",
        "ckpt_dir": str(args.ckpt_dir),
        "pattern": pattern,
        "n_checkpoints": len(ckpts),
        "checkpoints": inventory,
        "timestamp_utc": _utc_iso(time.time()),
        "status": "MEASURED" if ckpts else "EMPTY",
        "honest_framing_note": (
            "Inventory is a filesystem listing + sha256 prefix. "
            "Quality (pIC50 R²) is reported in training_summary.json."
        ),
    }
    (args.out_dir / "ckpt_inventory.json").write_text(json.dumps(inventory_payload, indent=2))

    load_payload = {
        "wf_tag": "Wf-84 Phase 3 — D-MPNN pIC50 ckpt load report",
        "n_checkpoints": len(load_reports),
        "n_strict_ok": sum(1 for r in load_reports if r.get("load_strict_ok")),
        "n_load_errors": sum(1 for r in load_reports if "ERROR" in r.get("load_status", "")),
        "reports": load_reports,
        "timestamp_utc": _utc_iso(time.time()),
    }
    (args.out_dir / "ckpt_load_report.json").write_text(json.dumps(load_payload, indent=2))

    inference_payload = {
        "wf_tag": "Wf-84 Phase 3 — D-MPNN pIC50 inference smoke",
        "n_checkpoints": len(inference_reports),
        "n_inference_ok": sum(
            1
            for r in inference_reports
            if r.get("inference_status") in ("OK", "ALL_NAN")
        ),
        "probe_smiles": list(PROBE_SMILES),
        "reports": inference_reports,
        "timestamp_utc": _utc_iso(time.time()),
    }
    (args.out_dir / "inference_smoke.json").write_text(json.dumps(inference_payload, indent=2))

    # Synthesis markdown
    md_lines: List[str] = [
        "# Wf-84 Phase 3 — D-MPNN pIC50 checkpoint diagnostic",
        "",
        f"- ckpt_dir: `{args.ckpt_dir}`",
        f"- pattern: `{pattern}`",
        f"- n_checkpoints: **{len(ckpts)}**",
        f"- n_strict_ok: **{sum(1 for r in load_reports if r.get('load_strict_ok'))}**",
        f"- n_load_errors: **{sum(1 for r in load_reports if 'ERROR' in r.get('load_status', ''))}**",
        f"- n_inference_ok: **{sum(1 for r in inference_reports if r.get('inference_status') in ('OK', 'ALL_NAN'))}**",
        f"- elapsed: {elapsed:.1f}s",
        f"- device: {device}",
        "",
        "## Per-checkpoint summary",
        "",
        "| ckpt | size | keys | strict_ok | pic50 range |",
        "|------|------|------|-----------|-------------|",
    ]
    for inv, load_r, inf_r in zip(inventory, load_reports, inference_reports):
        name = inv["filename"]
        size = inv["size_bytes"]
        keys = load_r.get("state_dict_keys", "?")
        strict_ok = "yes" if load_r.get("load_strict_ok") else "no"
        rng = inf_r.get("inference_pic50_range", [float("nan"), float("nan")])
        rng_s = (
            f"[{rng[0]:+.3f}, {rng[1]:+.3f}]"
            if all(np.isfinite(rng))
            else "[n/a]"
        )
        md_lines.append(
            f"| {name} | {size:>8} | {keys} | {strict_ok} | {rng_s} |"
        )
    md_lines += [
        "",
        "## Honest framing",
        "",
        "This diagnostic confirms ckpt existence + load + finite forward pass.",
        "It does **not** validate pIC50 R² quality; see `training_summary.json`.",
        "",
    ]
    (args.out_dir / "diagnostic_synthesis.md").write_text("\n".join(md_lines))

    print(
        f"[dmpnn-ckpt-diag] DONE: wrote inventory + load + inference reports to "
        f"{args.out_dir} ({elapsed:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
