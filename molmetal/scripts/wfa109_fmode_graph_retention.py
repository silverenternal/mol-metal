"""Wf-A109 — does ``FmOdeTerm`` retain across steps, or is 16 graphs the budget?

D11 says the FM arm at ``fm_ode_term_weight = 0.05`` reaches 42.5 GB and OOMs
the host, while the same arm at 0.0 peaks at 5.28 GB.  ``FmOdeTerm`` runs four
RK4 stages over four steps — 16 velocity-field forwards per training step —
and every one of their autograd graphs is alive until backward.

Two hypotheses, and they make different predictions:

* **budget** — the 16 graphs cost what they cost and are released at backward.
  RSS then saws: up inside the step, back down after.  The curve across steps
  is flat, and the peak is set by batch size, not by step count.
* **leak** — something keeps a reference past the step.  RSS then rises
  monotonically and never returns to its pre-step value.

This measures both, on the real model and the real term:

1. per-graph cost, as the MARGINAL slope of holding n graphs.  The n=1 row is
   not used for this: the first call also allocates the allocator's arena and
   the model's workspace, so it overstates the marginal cost by 3x (MEASURED:
   147.6 MB at n=1 against 20 MB marginal at 8 clouds).
2. 30 full steps (forward -> term -> backward -> zero_grad) with the term on,
   sampling RSS after every step, and the same 30 steps with it off.

Writes ``metrics/by_metric/wfa109_fmode_graph_retention.json``.

    .venv/bin/python molmetal/scripts/wfa109_fmode_graph_retention.py \\
        --config configs/wfa108j_fixed_cpu_2026_10_25.yaml --n-clouds 64
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "metrics" / "by_metric" / "wfa109_fmode_graph_retention.json"


def _rss_mb() -> float:
    """Resident set size of this process, in MiB.

    Read from ``/proc/self/status`` rather than ``resource.getrusage`` because
    ``ru_maxrss`` is a high-water mark and cannot show a release.
    """
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def _peak_rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def _live_tensors() -> int:
    """How many ``torch.Tensor`` objects are reachable right now.

    This is the statistic that separates the two hypotheses, and RSS is not:
    glibc keeps freed arenas mapped, so a run that allocates and frees 2 GB
    per step sits at a flat high-water mark whether or not anything is
    retained.  A reference leak, by contrast, is exactly a live-object count
    that grows with the step index.
    """
    gc.collect()
    return sum(1 for o in gc.get_objects() if torch.is_tensor(o))


def _slope_per_step(values: Sequence[float]) -> Optional[float]:
    """Least-squares slope of a series against step index, in units/step.

    The first three points are excluded: the allocator grows its arena while
    it learns the workload's size, and that one-off would read as growth.
    """
    ys = [float(v) for v in values[3:]]
    if len(ys) < 4:
        return None
    xs = list(range(len(ys)))
    n = float(len(ys))
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-clouds", type=int, default=64)
    ap.add_argument("--max-graphs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--only",
        choices=("all", "control", "plain", "ckpt"),
        default="all",
        help=(
            "Run ONE arm in this process.  RSS is a high-water mark and glibc "
            "does not return freed arenas, so a second arm in the same process "
            "starts from the first arm's peak and cannot show a reduction.  "
            "The peak comparison has to be across processes."
        ),
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    sys.argv = ["wf_a109_retention", "--config", str(args.config)]
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_csv_cohort,
        build_joint_model,
        parse_args as _pta,
    )
    from molmetal.loss.regularization_losses import FmOdeTerm

    ns = _pta()
    dev = torch.device(args.device)
    model = build_joint_model(ns).to(dev).train()
    clouds = build_csv_cohort(
        str(ns.data_csv), split="train", n_clouds=int(args.n_clouds), seed=0,
        metal_placement=str(getattr(ns, "metal_placement", "bonded")),
        dative_distance_cutoff=float(ns.dative_distance_cutoff),
    )
    sizes = [int(c.atomic_numbers.shape[0]) for c in clouds]
    x1 = torch.cat([c.positions.float() for c in clouds], 0).to(dev)
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0).to(dev)
    sigma = float(getattr(ns, "fm_noise_sigma", 1.0))
    g = torch.Generator(device="cpu").manual_seed(0)
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(dev)
    t_atom = torch.full((int(x1.shape[0]),), 0.5, device=dev)
    x_t = (1.0 - t_atom[:, None]) * x0 + t_atom[:, None] * x1

    base = _rss_mb()
    print(
        f"[retention] clouds={len(clouds)} atoms={int(x1.shape[0])} "
        f"base_rss={base:.0f} MB device={dev}",
        flush=True,
    )

    def _pred(x: torch.Tensor) -> torch.Tensor:
        return model.fm_velocity(x, t_atom, z, cloud_sizes=sizes)

    # ---- 1. per-graph cost -------------------------------------------------
    def _hold(n_graphs: int, checkpoint: bool) -> Dict[str, Any]:
        before = _rss_mb()
        outs: List[torch.Tensor] = []
        for _ in range(int(n_graphs)):
            if checkpoint:
                from torch.utils.checkpoint import checkpoint as _ckpt

                out = _ckpt(_pred, x1, use_reentrant=False)
            else:
                out = _pred(x1)
            outs.append(out)
        during = _rss_mb()
        loss = sum(o.sum() for o in outs)
        loss.backward()
        after_bwd = _rss_mb()
        del outs, loss
        gc.collect()
        after_free = _rss_mb()
        model.zero_grad(set_to_none=True)
        return {
            "n_graphs": int(n_graphs),
            "checkpointed": bool(checkpoint),
            "rss_before_mb": round(before, 1),
            "rss_with_graphs_alive_mb": round(during, 1),
            "rss_after_backward_mb": round(after_bwd, 1),
            "rss_after_free_mb": round(after_free, 1),
            "delta_mb": round(during - before, 1),
            "residual_after_free_mb": round(after_free - before, 1),
        }

    plain: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    marginal = None
    if args.only != "all":
        # A single-arm run is about one step curve's peak, and the per-graph
        # probe would move that peak before the curve starts.
        pass
    for n in (() if args.only != "all" else (1, 2, 4, 8, 16)):
        if n > int(args.max_graphs):
            break
        row = _hold(n, checkpoint=False)
        plain.append(row)
        print(
            f"[retention] plain n={n:>2} delta={row['delta_mb']:>8.1f} MB "
            f"left={row['residual_after_free_mb']:>7.1f} MB",
            flush=True,
        )

    # Marginal slope over the last two sizes: both are past the arena warm-up.
    if len(plain) >= 2:
        a, b = plain[-2], plain[-1]
        dn = b["n_graphs"] - a["n_graphs"]
        if dn > 0:
            marginal = round((b["delta_mb"] - a["delta_mb"]) / dn, 2)

    for n in (() if args.only != "all" else (4, 16)):
        if n > int(args.max_graphs):
            break
        row = _hold(n, checkpoint=True)
        ckpt_rows.append(row)
        print(
            f"[retention] ckpt  n={n:>2} delta={row['delta_mb']:>8.1f} MB "
            f"left={row['residual_after_free_mb']:>7.1f} MB",
            flush=True,
        )

    # ---- 2. across-step curve ---------------------------------------------
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(args.lr)
    )

    def _step_curve(weight: float, use_checkpoint: bool = True) -> Dict[str, Any]:
        term = FmOdeTerm(
            weight=float(weight),
            enabled=float(weight) > 0.0,
            use_checkpoint=bool(use_checkpoint),
        )
        after_bwd: List[float] = []
        inside: List[float] = []
        live: List[int] = []
        after_gc: List[float] = []
        losses: List[float] = []
        for step in range(int(args.steps)):
            opt.zero_grad(set_to_none=True)
            before = _rss_mb()
            loss = term.compute(
                pred_fn=_pred if float(weight) > 0.0 else None,
                x0=x_t,
                x1=x1,
                n_steps=4,
                device=dev,
            )
            peak_inside = _rss_mb()
            if float(weight) <= 0.0:
                # Control arm still needs a graph to backward through.
                loss = _pred(x_t).pow(2).mean()
                peak_inside = _rss_mb()
            loss.backward()
            opt.step()
            after_bwd.append(_rss_mb())
            inside.append(peak_inside)
            losses.append(float(loss.detach()))
            # Census first: it ends in gc.collect(), so the RSS sampled after
            # it is post-collection and is the honest per-step level.  The
            # sample above is taken with the step's garbage still resident.
            live.append(_live_tensors())
            after_gc.append(_rss_mb())
            if step < 3 or step % 5 == 0:
                print(
                    f"[retention] w={weight:.2f} step={step:>2} "
                    f"before={before:>8.1f} inside={peak_inside:>8.1f} "
                    f"after={after_bwd[-1]:>8.1f} live={live[-1]:>6d} "
                    f"loss={losses[-1]:.4g}",
                    flush=True,
                )
        gc.collect()
        return {
            "fm_ode_term_weight": float(weight),
            "use_checkpoint": bool(use_checkpoint),
            "rss_after_step_mb": [round(v, 1) for v in after_bwd],
            "rss_after_gc_mb": [round(v, 1) for v in after_gc],
            "rss_inside_step_mb": [round(v, 1) for v in inside],
            "live_tensors_after_step": live,
            "loss": [round(v, 6) for v in losses],
            "rss_slope_mb_per_step": (
                round(_slope_per_step(after_bwd), 3)
                if _slope_per_step(after_bwd) is not None else None
            ),
            "rss_after_gc_slope_mb_per_step": (
                round(_slope_per_step(after_gc), 3)
                if _slope_per_step(after_gc) is not None else None
            ),
            "live_tensors_slope_per_step": (
                round(_slope_per_step(live), 4)
                if _slope_per_step(live) is not None else None
            ),
            "first_mb": round(after_bwd[0], 1),
            "last_mb": round(after_bwd[-1], 1),
            "total_growth_mb": round(after_bwd[-1] - after_bwd[0], 1),
        }

    # Control first: its RSS is then measured on a clean process, so it is
    # comparable to the 5.28 GB peak observed on the real arm.  The term arm
    # starts from the control's high-water mark and its RSS is therefore not
    # comparable in absolute terms — only its live-tensor count and its
    # inside-step peak above the control's are used.
    if args.only == "all":
        without_term = _step_curve(0.0, use_checkpoint=False)
        print(
            f"[retention] without term: rss_slope={without_term['rss_slope_mb_per_step']} "
            f"live_slope={without_term['live_tensors_slope_per_step']}/step "
            f"growth={without_term['total_growth_mb']} MB",
            flush=True,
        )
        with_term = _step_curve(0.05, use_checkpoint=False)
        print(
            f"[retention] with term (no ckpt): rss_slope={with_term['rss_slope_mb_per_step']} "
            f"MB/step live_slope={with_term['live_tensors_slope_per_step']}/step "
            f"growth={with_term['total_growth_mb']} MB over {args.steps} steps",
            flush=True,
        )
        with_term_ckpt = _step_curve(0.05, use_checkpoint=True)
        print(
            f"[retention] with term (ckpt): rss_slope={with_term_ckpt['rss_slope_mb_per_step']} "
            f"MB/step peak={max(with_term_ckpt['rss_inside_step_mb']):.0f} MB "
            f"vs plain peak={max(with_term['rss_inside_step_mb']):.0f} MB",
            flush=True,
        )
    elif args.only == "control":
        without_term = _step_curve(0.0, use_checkpoint=False)
        with_term = with_term_ckpt = None
    elif args.only == "plain":
        with_term = _step_curve(0.05, use_checkpoint=False)
        without_term = with_term_ckpt = None
    else:  # ckpt
        with_term_ckpt = _step_curve(0.05, use_checkpoint=True)
        with_term = without_term = None

    if args.only != "all":
        # One arm, one process: the peak is the deliverable, because the
        # comparison across arms is the only one that is not contaminated by
        # the allocator's high-water mark.
        arm = {"control": without_term, "plain": with_term, "ckpt": with_term_ckpt}[args.only]
        doc = {
            "script": "molmetal/scripts/wfa109_fmode_graph_retention.py",
            "addresses": ["D11", "T2"],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arm": args.only,
            "config": str(args.config),
            "device": str(dev),
            "n_clouds": len(clouds),
            "n_atoms": int(x1.shape[0]),
            "steps": int(args.steps),
            "base_rss_mb": round(base, 1),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
            "curve": arm,
            "inside_step_peak_mb": round(max(arm["rss_inside_step_mb"]), 1),
            "wall_time_s": round(time.time() - t0, 2),
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2))
        print(
            f"[retention] arm={args.only} inside_step_peak={doc['inside_step_peak_mb']} MB "
            f"base={doc['base_rss_mb']} MB -> {out}",
            flush=True,
        )
        return 0

    slope = with_term["rss_slope_mb_per_step"]
    control_slope = without_term["rss_slope_mb_per_step"]
    live_slope = with_term["live_tensors_slope_per_step"]
    control_live = without_term["live_tensors_slope_per_step"]

    # The verdict rests on the OBJECT CENSUS, not on RSS.  glibc keeps freed
    # arenas mapped, so RSS sits at a flat high-water mark whether or not
    # anything is retained, and a 12-step window can show a 14 MB/step slope
    # that is one arena step.  A reference leak is a live-tensor count that
    # grows with the step index; nothing else is.
    leaks = bool(
        live_slope is not None
        and live_slope > 1.0
        and (control_live is None or live_slope > control_live + 0.5)
    )

    # Per-step peak attributable to the term: the inside-step RSS of the term
    # arm against the same quantity in the control arm, divided by the 16
    # graphs the term asks for.  This is the number that matters, because it
    # is measured inside a real step rather than on an idle process.
    inside_peak = (
        max(with_term["rss_inside_step_mb"][3:]) - max(without_term["rss_inside_step_mb"][3:])
        if len(with_term["rss_inside_step_mb"]) > 3 else None
    )
    per_graph_in_step = round(inside_peak / 16.0, 2) if inside_peak is not None else None

    doc = {
        "script": "molmetal/scripts/wfa109_fmode_graph_retention.py",
        "addresses": ["D11", "T2"],
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(args.config),
        "device": str(dev),
        "n_clouds": len(clouds),
        "n_atoms": int(x1.shape[0]),
        "base_rss_mb": round(base, 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "per_graph": plain,
        "per_graph_marginal_mb": marginal,
        "per_graph_marginal_is_contaminated": (
            "RSS is a high-water mark: after the n=1 row frees its graph, "
            "glibc keeps the arena, so the n=2 row measures growth from an "
            "already-inflated baseline and the marginal slope reads low.  The "
            "n=1 delta and per_graph_in_step_mb are the usable numbers."
        ),
        "checkpointed": ckpt_rows,
        "projected_16_graphs_mb": (
            round(16.0 * marginal, 1) if marginal is not None else None
        ),
        "per_graph_in_step_mb": per_graph_in_step,
        "step_peak_attributable_to_term_mb": (
            round(inside_peak, 1) if inside_peak is not None else None
        ),
        "across_steps": {
            "with_term": with_term,
            "with_term_checkpointed": with_term_ckpt,
            "without_term": without_term,
        },
        "checkpoint_peak_reduction_mb": round(
            max(with_term["rss_inside_step_mb"])
            - max(with_term_ckpt["rss_inside_step_mb"]),
            1,
        ),
        "checkpoint_comparison_note": (
            "These two arms ran in the same process, so the checkpointed arm "
            "starts from the plain arm's high-water mark and cannot show a "
            "reduction.  Use --only plain and --only ckpt in separate "
            "processes for the peak comparison."
        ),
        "observed_peak_gb_with_fm_ode": 42.52,
        "observed_peak_gb_without_fm_ode": 5.28,
        "retention_across_steps": bool(leaks),
        "reading": (
            "The live-tensor count grows with the step index and the term is "
            "responsible, so the retainer is still to be found."
            if leaks else
            "The live-tensor count is flat across steps with the term on, so "
            "nothing is retained past backward: the 16 graphs are released.  "
            "RSS sits at a high-water mark because glibc keeps freed arenas "
            "mapped, which is why the 42.5 GB peak looks like growth and is "
            "not.  The peak is a per-step budget set by batch size and the "
            "field's activation size, and the lever is checkpointing the "
            "stage forwards, not hunting a leak."
        ),
        "wall_time_s": round(time.time() - t0, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(
        f"[retention] live_tensors slope={live_slope}/step (control "
        f"{control_live}) leak={leaks} | per_graph_in_step={per_graph_in_step} "
        f"MB | 16 graphs ~ {doc['step_peak_attributable_to_term_mb']} MB -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
