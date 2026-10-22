"""Wf-A108g — peak VRAM of the FM forward as a function of batch width.

Measures what one FM forward costs at several cloud counts and several
``fm_batch_chunk_ratio`` values, on the real MetalCytoToxDB cohort the
``wfa108_vector_channels`` arm trains on.  The point is not that a kernel
got faster: it is that the peak is set by how many atoms one padded batch
carries, and grouping by size lowers that number without changing what is
computed.

Writes ``metrics/by_metric/wfa108g_batch_chunk_memory.json``.  Every
number the config comments or the catalog cite must come from that file.

    .venv/bin/python molmetal/scripts/wfa108g_batch_chunk_memory.py \
        --config configs/wfa108_vector_channels.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: E402
    build_csv_cohort,
    build_joint_model,
    parse_args as parse_trainer_args,
)

OUT = _ROOT / "metrics" / "by_metric" / "wfa108g_batch_chunk_memory.json"


def _peak_mb(fn) -> tuple[float, object]:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wfa108_vector_channels.yaml")
    ap.add_argument("--batches", default="8,16,32,64")
    ap.add_argument("--ratios", default="0,1.25,1.5,2.0")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    ns_ap = ap.parse_args(argv)

    batches = [int(v) for v in ns_ap.batches.split(",") if v]
    ratios = [float(v) for v in ns_ap.ratios.split(",") if v]

    sys.argv = ["x", "--config", ns_ap.config]
    ns = parse_trainer_args()
    dev = torch.device("cuda:0")
    model = build_joint_model(ns).to(dev).eval()

    rows = []
    for n_clouds in batches:
        clouds = build_csv_cohort(
            ns.data_csv, split="train", n_clouds=n_clouds, seed=0,
            add_hydrogens=True, metal_placement="bonded",
            dative_distance_cutoff=float(ns.dative_distance_cutoff),
        )
        sizes = [int(c.positions.shape[0]) for c in clouds]
        x1 = torch.cat(
            [torch.as_tensor(c.positions, dtype=torch.float32) for c in clouds]
        ).to(dev)
        z = torch.cat(
            [torch.as_tensor(c.atomic_numbers, dtype=torch.long) for c in clouds]
        ).to(dev)
        t = torch.rand(len(sizes), generator=torch.Generator().manual_seed(0))
        t_atom = t.repeat_interleave(torch.tensor(sizes)).to(dev)

        # Memory and numerics per ratio.
        ref = None
        peaks: dict[float, float] = {}
        outs: dict[float, torch.Tensor] = {}
        with torch.no_grad():
            for ratio in ratios:
                model.fm_batch_chunk_ratio = ratio
                # Warm-up: autotune and allocator growth must not land in
                # the measured call.
                model.fm_velocity(x1, t_atom, z, cloud_sizes=sizes)
                torch.cuda.empty_cache()
                peak, out = _peak_mb(
                    lambda: model.fm_velocity(  # noqa: E731
                        x1, t_atom, z, cloud_sizes=sizes
                    )
                )
                peaks[ratio] = peak
                outs[ratio] = out
                if ref is None:
                    ref = out

        # Timing: ratios must be INTERLEAVED.  Two sequential runs of this
        # script measured the same unchunked B=8 forward at 4.14 ms and
        # 7.48 ms, so a block-at-a-time sweep compares clock state as much
        # as it compares code.  Alternating inside one loop holds that
        # fixed; only ratios from the same row are comparable.
        stamps: dict[float, list[float]] = {r: [] for r in ratios}
        with torch.no_grad():
            for _ in range(ns_ap.rounds):
                for ratio in ratios:
                    model.fm_batch_chunk_ratio = ratio
                    for _ in range(ns_ap.repeats):
                        start = time.perf_counter()
                        model.fm_velocity(x1, t_atom, z, cloud_sizes=sizes)
                        torch.cuda.synchronize()
                        stamps[ratio].append((time.perf_counter() - start) * 1e3)

        for ratio in ratios:
            peak = peaks[ratio]
            out = outs[ratio]
            delta = 0.0 if out is ref else float((out - ref).abs().max().item())
            ordered = sorted(stamps[ratio])
            ms = ordered[len(ordered) // 2]
            rows.append(
                {
                    "n_clouds": len(sizes),
                    "ratio": ratio,
                    "peak_mb": round(peak, 2),
                    "median_ms": round(ms, 2),
                    "ms_stdev": round(
                        (sum((v - ms) ** 2 for v in ordered) / len(ordered)) ** 0.5,
                        2,
                    ),
                    "atoms": int(sum(sizes)),
                    "padded_slots": int(len(sizes) * max(sizes)),
                    "max_abs_delta": delta,
                    "cloud_sizes": sizes,
                }
            )
            print(
                f"B={len(sizes):3d} ratio={ratio:4.2f}  peak {peak:8.2f} MB  "
                f"{ms:7.2f} ms  "
                f"atoms {sum(sizes):5d}/{len(sizes) * max(sizes):5d}  "
                f"max|d| {delta:.2e}",
                flush=True,
            )

    base = {r["n_clouds"]: r for r in rows if r["ratio"] == 0.0}
    for r in rows:
        b = base[r["n_clouds"]]
        r["memory_ratio_vs_unchunked"] = (
            round(b["peak_mb"] / r["peak_mb"], 2) if r["peak_mb"] else None
        )
        r["time_ratio_vs_unchunked"] = round(r["median_ms"] / b["median_ms"], 2)

    payload = {
        "work": "Wf-A108g",
        "date": time.strftime("%Y-%m-%d"),
        "config": ns_ap.config,
        "device": torch.cuda.get_device_name(0),
        "triton_dispatch": _dispatch(),
        "rows": rows,
        "note": (
            "Peak is max_memory_allocated minus the allocation live before "
            "the call, on one forward with no autograd.  ratio 0.0 is the "
            "pre-Wf-A108g path (one padded forward).  Median_ms is taken "
            "from ratios INTERLEAVED inside one loop: two sequential runs "
            "measured the same unchunked B=8 forward at 4.14 ms and 7.48 ms, "
            "so only rows from the same n_clouds are comparable."
        ),
    }
    out_path = Path(ns_ap.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


def _dispatch() -> dict:
    try:
        from triton_kernels import backend

        return backend.describe()
    except Exception:  # noqa: BLE001
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
