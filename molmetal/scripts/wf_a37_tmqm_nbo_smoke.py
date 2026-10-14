#!/usr/bin/env python3
"""Wf-A37-6 — tmQM NBO natural-charge file availability smoke.

======================================================================
The bond head's per-atom NBO Loewdin q supervision slot requires
``.data/molmetal/tmQM/tmQM_X.q`` (or ``.gz``) on disk.  Wf-A37-6 ships
that file (95 MB; 108,541 complexes; NBO summary per CSD block) and
this smoke records its availability + parses a sample so the reviewer
sees the data flow end-to-end without booting the trainer.

Output JSON schema
------------------
::

    {
      "wf": "Wf-A37-6",
      "tag": "tmQM NBO natural-charge smoke",
      "q_path": "<absolute>",
      "q_exists": <bool>,
      "q_size_bytes": <int>,
      "q_n_blocks": <int>,
      "parser_status": "AVAILABLE" | "MISSING_FILE" | "PARSE_ERROR",
      "sample_complex": "<csd_code>",
      "sample_n_atoms": <int>,
      "sample_q_min": <float>,
      "sample_q_max": <float>,
      "sample_q_mean": <float>,
      "loader_smoke": {
        "metals": ["Pt"],
        "n_clouds_loaded": <int>,
        "clouds_with_natural_charges": <int>,
        "first_cloud_n_atoms": <int>,
        "first_cloud_q_sum": <float>
      },
      "bond_head_degradation_risk": <bool>
    }

Run::

    /home/hugo/codes/mol-metal/.venv/bin/python -m molmetal.scripts.wf_a37_tmqm_nbo_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "metrics" / "by_metric"


def _default_q_path() -> Path:
    return REPO_ROOT / ".data" / "molmetal" / "tmQM" / "tmQM_X.q"


def _probe(q_path: Path) -> dict:
    """Read the file's existence + size; do NOT load it (95 MB)."""
    info: dict = {
        "q_path": str(q_path),
        "q_exists": q_path.is_file(),
        "q_size_bytes": int(q_path.stat().st_size) if q_path.is_file() else 0,
        "q_n_blocks": 0,
        "parser_status": "MISSING_FILE",
    }
    if not info["q_exists"]:
        return info
    # Count CSD_code block headers — O(1) since the file is line-oriented.
    n_blocks = 0
    with open(q_path, "r", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("CSD_code"):
                n_blocks += 1
    info["q_n_blocks"] = n_blocks
    info["parser_status"] = "AVAILABLE" if n_blocks > 0 else "PARSE_ERROR"
    return info


def _sample(q_path: Path, code: str = "WELROW") -> dict:
    """Stream the q file and return the first block's charges for the sample code."""
    from molmetal.data.tmqm import parse_natural_charges

    full = parse_natural_charges(q_path=q_path)
    codes = list(full.keys())
    sample_code = code if code in full else codes[0]
    q_list = full[sample_code]
    return {
        "sample_complex": sample_code,
        "sample_n_atoms": len(q_list),
        "sample_q_min": float(min(q_list)),
        "sample_q_max": float(max(q_list)),
        "sample_q_mean": float(sum(q_list) / max(1, len(q_list))),
    }


def _loader_smoke(metals: list[str], n_clouds: int, seed: int) -> dict:
    """Mimic the trainer's batch_ctx wire-in for a small cohort."""
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        tmqm_dir=str(REPO_ROOT / ".data" / "molmetal" / "tmQM"),
        metals=metals,
        n_clouds=n_clouds,
        seed=seed,
    )
    n_with = 0
    first_n = -1
    first_q_sum = float("nan")
    for c in clouds:
        nat = getattr(c, "natural_charges", None)
        if nat is not None:
            n_with += 1
            if first_n == -1:
                first_n = int(c.positions.shape[0])
                first_q_sum = float(nat.sum().item())
    return {
        "metals": list(metals),
        "n_clouds_loaded": len(clouds),
        "clouds_with_natural_charges": n_with,
        "first_cloud_n_atoms": first_n,
        "first_cloud_q_sum": first_q_sum,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--q-path",
        type=Path,
        default=_default_q_path(),
        help="Path to tmQM_X.q (default: .data/molmetal/tmQM/tmQM_X.q)",
    )
    p.add_argument(
        "--metals",
        type=str,
        nargs="+",
        default=["Pt"],
        help="Metals cohort for loader smoke (default: Pt)",
    )
    p.add_argument("--n-clouds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Where to write the JSON metric",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="tmqm_natural_charges_status.json",
    )
    args = p.parse_args()

    t0 = time.perf_counter()
    info = _probe(args.q_path)

    sample: dict = {}
    if info["parser_status"] == "AVAILABLE":
        try:
            sample = _sample(args.q_path)
            info.update(sample)
        except Exception as exc:
            info["parser_status"] = "PARSE_ERROR"
            info["parse_error"] = f"{type(exc).__name__}: {exc}"

    loader_info = {}
    bond_head_risk = True
    if info["parser_status"] == "AVAILABLE":
        try:
            loader_info = _loader_smoke(args.metals, args.n_clouds, args.seed)
            # Degradation risk = "any loaded cloud has natural_charges=None".
            # With the .q file shipped + parser working, risk is False.
            n_loaded = loader_info["n_clouds_loaded"]
            n_with = loader_info["clouds_with_natural_charges"]
            bond_head_risk = (n_loaded > 0 and n_with < n_loaded)
        except Exception as exc:
            loader_info = {"error": f"{type(exc).__name__}: {exc}"}
            bond_head_risk = True

    payload = {
        "wf": "Wf-A37-6",
        "tag": "tmQM NBO natural-charge smoke",
        **info,
        "loader_smoke": loader_info,
        "bond_head_degradation_risk": bond_head_risk,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    # Stdout summary (5 lines max, per §5.8 token-budget discipline)
    print(f"WF: Wf-A37-6 status={payload['parser_status']}")
    print(f"q_file: exists={payload['q_exists']} size_MB={payload['q_size_bytes']/1e6:.1f} n_blocks={payload['q_n_blocks']}")
    print(f"loader: n={payload['loader_smoke'].get('n_clouds_loaded', '?')} with_nat={payload['loader_smoke'].get('clouds_with_natural_charges', '?')}")
    print(f"bond_head_degradation_risk: {payload['bond_head_degradation_risk']}")
    print(f"json_on_disk: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())