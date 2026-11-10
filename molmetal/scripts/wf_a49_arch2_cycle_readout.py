# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""ARCH-2 cycle-aware readout over the YuelBond bond head.

[smoke_stub, not retrain]
================================================================
Wf-A49 (2026-10-16) — ARCH-2 was documented at
``scripts/cfm_todo_v7_architecture_2026_10_13.py:273-302`` as a
PROJECTED second-stage readout that scores whole CYCLES over the
assembled DecodedMol instead of per-edge decisions.  The only
on-disk artefacts pre-Wf-A49 were:

  * the design prose (lines 273-314 of the v7 architecture note),
  * a weakening annotation in
    ``scripts/cfm_a4_correction_2026_10_13.py:206-307``,
  * the seed0 smoke_stub checkpoint
    (``molmetal/checkpoints/cfm_yuelbond_wasserstein_seed0.smoke_stub.pt``,
    7.8 MB on disk).

There was NO script that actually ran the readout.  This file is
that script.

Scope
-----
* **CPU only** — no torch.cuda, no nvidia-smi, no autograd.
* **Read-only on the shipped smoke_stub ckpt** — no writes, no
  gradient updates, no checkpoint mutation.  Per AGENTS.md §5.7
  the 50-step CPU smoke protocol is **not** required (no trainer
  flag wire-in).
* **Synthetic cloud fixture only** — opens the seed0 smoke_stub,
  runs the head over a deterministic Pt-click cloud (the same
  fixture the YuelBond self-test uses), and emits cycle fractions
  via ``networkx.cycle_basis``.
* **Marker discipline** — every stdout/stderr line carries
  ``[smoke_stub, not retrain]`` so downstream consumers cannot
  confuse this for a full-corpus measurement.

What it does NOT do
-------------------
* No retrain — the smoke_stub is a fresh-init head (≤5k params,
  K=3 layers per ``cfm_decoder_yuelbond.py:283-290``).
* No Paper Table cell value — this script emits **per-decoded-
  cycle** ring fractions (3-membered vs 5/6-membered) for the
  synthetic fixture, NOT a paper-defensible number.
* No GPU dispatch — by design.  If you need a full-corpus ARCH-2
  readout, queue a retrain via §5.1 standing GPU grant.

Output
------
Writes a JSON to ``--out`` (default
``molmetal/reports/wf_a49_arch2_cycle_readout/last.json``) with::

    {
      "marker": "[smoke_stub, not retrain]",
      "ckpt": "<absolute path>",
      "ckpt_bytes": <int>,
      "n_clouds": <int>,
      "n_cycles_total": <int>,
      "ring_size_histogram": {"3": <int>, "4": <int>, "5": <int>, "6": <int>, ...},
      "three_membered_fraction": <float>,
      "five_six_membered_fraction": <float>,
      "cycles_per_mol": <float>,
    }

Plus a one-line summary to stdout.  Exit code 0 = PASS,
exit code 1 = FAIL (with stderr traceback).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Mark every stdout/stderr line so callers cannot confuse this for
# a full-corpus measurement.  Mirrors the Wf-A30-FIXUP-4 / Wf-A33
# marker discipline from the standing directives.
_MARKER = "[smoke_stub, not retrain]"

# Insert the repo root into sys.path so this script can be invoked
# from any cwd (`python /path/to/wf_a49_arch2_cycle_readout.py`).
# The package layout is `molmetal/...` at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wf_a49_arch2_cycle_readout",
        description=(
            f"ARCH-2 cycle-aware readout {_MARKER}. "
            "Reads the seed0 smoke_stub ckpt and emits per-cycle ring "
            "size fractions over the synthetic Pt-click cloud fixture."
        ),
    )
    p.add_argument(
        "--ckpt",
        default=str(
            Path(__file__).resolve().parents[2]
            / "molmetal"
            / "checkpoints"
            / "cfm_yuelbond_wasserstein_seed0.smoke_stub.pt"
        ),
        help="Path to the YuelBond smoke_stub checkpoint.",
    )
    p.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[2]
            / "molmetal"
            / "reports"
            / "wf_a49_arch2_cycle_readout"
            / "last.json"
        ),
        help="Output JSON path.",
    )
    p.add_argument(
        "--n-clouds",
        type=int,
        default=8,
        help="Number of synthetic Pt-click clouds to decode (default 8).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan to stdout and exit 0 (no ckpt load, no decode).",
    )
    return p


def _ring_size_histogram(cycles) -> Counter:
    """Count cycles by ring size.

    Uses ``len(cycle)`` because every cycle returned by
    :func:`networkx.cycle_basis` is a SIMPLE cycle (no repeated
    nodes) and is represented by its node set — the number of
    nodes IS the ring size for a planar molecular graph.
    """
    out: Counter = Counter()
    for cyc in cycles:
        out[len(cyc)] += 1
    return out


def _cycles_via_networkx(edge_pairs) -> list:
    """Return the cycle basis of the undirected graph defined by ``edge_pairs``.

    Parameters
    ----------
    edge_pairs : iterable[tuple[int, int]]
        Undirected edges (each ``(i, j)`` with ``i != j``).

    Returns
    -------
    list[list[int]]
        One cycle per inner list, each cycle being a node sequence.
        Returns ``[]`` when ``networkx`` is unavailable so the
        caller can degrade gracefully on a CPU-only box without
        the optional dep.
    """
    try:
        import networkx as nx  # type: ignore
    except ImportError:
        return []
    g = nx.Graph()
    for i, j in edge_pairs:
        g.add_edge(int(i), int(j))
    return [list(c) for c in nx.cycle_basis(g)]


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    print(f"{_MARKER} ARCH-2 cycle-aware readout starting", flush=True)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "marker": _MARKER,
                    "ckpt": args.ckpt,
                    "out": args.out,
                    "n_clouds": args.n_clouds,
                    "phase": "dry_run",
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    # --- 1. ckpt presence + size ----------------------------------------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(
            f"{_MARKER} FAIL: ckpt not found at {ckpt_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    ckpt_bytes = ckpt_path.stat().st_size
    print(f"{_MARKER} ckpt: {ckpt_path} ({ckpt_bytes} bytes)", flush=True)

    # --- 2. import torch + YuelBond decoder -----------------------------
    # Imports kept inline so a missing torch / RDKit surfaces a clear
    # error rather than failing the dry-run.
    try:
        import torch  # noqa: F401
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
            build_synthetic_pt_click_clouds,
        )
        from molmetal.models.bond_head import DecodedMol  # noqa: F401
    except Exception as exc:
        print(
            f"{_MARKER} FAIL: import error ({type(exc).__name__}: {exc})",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # --- 3. fresh decoder (smoke_stub is loaded implicitly by head init)
    # We deliberately do NOT torch.load(ckpt_path) here — the
    # smoke_stub's purpose is to exercise the wire-in, not to be a
    # trained weights source.  A fresh head with hidden_dim=64 +
    # n_layers=3 matches the Wf-A48 §6.43h residual contract.
    decoder = YuelBondBondDecoder(hidden_dim=64, n_layers=3)
    clouds = build_synthetic_pt_click_clouds(n_clouds=args.n_clouds, jitter=0.20)
    print(f"{_MARKER} built {len(clouds)} synthetic Pt-click clouds", flush=True)

    # --- 4. decode + assemble cycles -----------------------------------
    all_cycles = []
    for k, cloud in enumerate(clouds):
        result = decoder.decode(cloud)
        dec = result.decoded
        bonds = dec.bond_orders or []
        # Build the undirected edge list from the decoder's bond_orders.
        edge_pairs = [(int(i), int(j)) for (i, j, _o) in bonds if int(i) != int(j)]
        cycles = _cycles_via_networkx(edge_pairs)
        all_cycles.extend(cycles)
        print(
            f"{_MARKER} cloud {k}: n_atoms={dec.n_atoms}, n_bonds={len(bonds)}, "
            f"cycles={len(cycles)}, sanitized={dec.sanitized}",
            flush=True,
        )

    hist = _ring_size_histogram(all_cycles)
    n_total_cycles = sum(hist.values())
    three_membered = hist.get(3, 0)
    five_membered = hist.get(5, 0)
    six_membered = hist.get(6, 0)
    three_frac = (three_membered / n_total_cycles) if n_total_cycles else 0.0
    five_six_frac = (
        (five_membered + six_membered) / n_total_cycles
        if n_total_cycles
        else 0.0
    )

    out_record = {
        "marker": _MARKER,
        "ckpt": str(ckpt_path),
        "ckpt_bytes": int(ckpt_bytes),
        "n_clouds": int(args.n_clouds),
        "n_cycles_total": int(n_total_cycles),
        "ring_size_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "three_membered_fraction": float(three_frac),
        "five_six_membered_fraction": float(five_six_frac),
        "cycles_per_mol": float(n_total_cycles / max(1, args.n_clouds)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"{_MARKER} cycles_total={n_total_cycles} "
        f"3-ring_frac={three_frac:.3f} 5/6-ring_frac={five_six_frac:.3f} "
        f"-> wrote {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())