#!/usr/bin/env python
# Apache-2.0
"""Wf-A108g — Triton Blackwell (sm_120) op-by-op dispatch table.

For every fused kernel the CFM path can reach, measure it against the
PyTorch op it replaces at the shapes that path actually uses, and decide
whether to enable it.  Writes ``metrics/by_metric/triton_blackwell_bench.json``,
which :data:`triton_kernels.config.MEASURED_OP_ALLOWLIST` is derived from.

Why this exists
---------------
``triton_kernels`` was written for gfx1101 and every grid in it was tuned
for that target.  On 2026-09-27 the global gate was closed
(``TRITON_USE_FUSED=0``) because those kernels ran 3-11x slower than
PyTorch on Blackwell, and closing the gate was cheaper than retuning.
That left the whole Triton surface dead on the hardware the project now
runs on.  This script is the measurement that reopens it op by op instead
of all at once.

Decision rule (encoded, not judged per run): enable an op iff it is
numerically correct on every measured shape AND its worst measured
speed-up over PyTorch is > 1.0x.

Usage::

    .venv/bin/python molmetal/scripts/wfa108g_triton_blackwell_bench.py
    .venv/bin/python molmetal/scripts/wfa108g_triton_blackwell_bench.py --quick

Exit codes: 0 = wrote the table, 1 = no CUDA device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import triton  # noqa: E402

import triton_kernels as tk  # noqa: E402
from triton_kernels import backend as tk_backend  # noqa: E402


def bench(fn, quick: bool) -> float:
    """Median wall time in microseconds."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return (
        triton.testing.do_bench(
            fn, warmup=25 if quick else 100, rep=100 if quick else 500,
            return_mode="median",
        )
        * 1e3
    )


def measure(name, shape, torch_fn, tri_fn, *, tol, quick):
    t_torch = bench(torch_fn, quick)
    t_tri = bench(tri_fn, quick)
    a, b = torch_fn(), tri_fn()
    err = (a - b).abs().max().item()
    scale = a.abs().max().item() or 1.0
    rel = err / scale
    speed = t_torch / t_tri
    print(
        f"{name:22s} {shape:20s} torch {t_torch:8.1f}us  triton {t_tri:8.1f}us  "
        f"x{speed:5.2f}  relerr {rel:.2e}"
    )
    return {
        "op": name,
        "shape": shape,
        "torch_us": round(t_torch, 2),
        "triton_us": round(t_tri, 2),
        "speedup": round(speed, 3),
        "rel_err": float(f"{rel:.3e}"),
        "numerically_ok": bool(rel < tol),
    }


def collect(quick: bool) -> list[dict]:
    dev = "cuda:0"
    torch.manual_seed(0)
    rows: list[dict] = []

    # --- fused SiLU MLP.  M = B*N atoms; 405 = the 8-cloud CFM batch. ------
    mlp_shapes = (405, 2048) if quick else (405, 736, 2048, 8192, 32768)
    for m in mlp_shapes:
        x = torch.randn(m, 128, device=dev)
        w1 = torch.randn(128, 128, device=dev) * 0.08
        b1 = torch.randn(128, device=dev) * 0.01
        w2 = torch.randn(128, 128, device=dev) * 0.08
        b2 = torch.randn(128, device=dev) * 0.01
        # Reference is the nn.Linear layout (out, in); the fused API takes
        # (in, out), hence the .T on both weights.
        rows.append(
            measure(
                "fused_silu_mlp",
                f"M={m},H=128",
                lambda x=x, w1=w1, b1=b1, w2=w2, b2=b2: torch.nn.functional.linear(
                    torch.nn.functional.silu(
                        torch.nn.functional.linear(x, w1.T, b1)
                    ),
                    w2.T,
                    b2,
                ),
                lambda x=x, w1=w1, b1=b1, w2=w2, b2=b2: tk.fused_silu_mlp(
                    x, w1, b1, w2, b2
                ),
                tol=5e-3,
                quick=quick,
            )
        )

    # --- the activation+second-projection kernel --------------------------
    # `_fused_mlp_act_proj_kernel` was disabled repo-wide because it
    # corrupted tiles on gfx1101.  It is now routed on CUDA only, so this
    # row is the evidence that the route is worth keeping: the fused SiLU
    # MLP above goes through it on sm_120.
    from triton_kernels.fused_mlp import _act_proj

    for m in (405, 2048) if quick else (405, 736, 2048, 8192, 32768):
        pre = torch.randn(m, 128, device=dev)
        w2 = torch.randn(128, 128, device=dev) * 0.08
        b2 = torch.randn(128, device=dev) * 0.01
        rows.append(
            measure(
                "act_proj",
                f"M={m},H=128",
                lambda pre=pre, w2=w2, b2=b2: (
                    torch.nn.functional.silu(pre) @ w2 + b2
                ),
                # ``act`` is an int constexpr: 0 = SiLU, anything else = GELU.
                # Passing the string "silu" silently selects GELU and the
                # row then reads as a 0.7 relative error.
                lambda pre=pre, w2=w2, b2=b2: _act_proj(pre, w2, b2, 0),
                tol=5e-3,
                quick=quick,
            )
        )

    # --- layer norm -------------------------------------------------------
    for m, d in (((405, 128),) if quick else ((405, 128), (2048, 128), (8192, 256))):
        x = torch.randn(m, d, device=dev)
        ln = torch.nn.LayerNorm(d).to(dev)
        g, bb = ln.weight.detach(), ln.bias.detach()
        rows.append(
            measure(
                "fused_layer_norm",
                f"M={m},D={d}",
                lambda ln=ln, x=x: ln(x),
                lambda x=x, g=g, bb=bb: tk.fused_layer_norm(x, g, bb, eps=1e-5),
                tol=1e-5,
                quick=quick,
            )
        )

    # --- softmax ----------------------------------------------------------
    for m, d in (((405, 128),) if quick else ((405, 128), (2048, 128), (8192, 256))):
        x = torch.randn(m, d, device=dev)
        rows.append(
            measure(
                "softmax_last_dim",
                f"M={m},D={d}",
                lambda x=x: torch.softmax(x, dim=-1),
                lambda x=x: tk.softmax_last_dim(x),
                tol=1e-6,
                quick=quick,
            )
        )

    # --- residual add -----------------------------------------------------
    for m in ((405,) if quick else (405, 8192, 32768)):
        a = torch.randn(m, 128, device=dev)
        r = torch.randn(m, 128, device=dev)
        rows.append(
            measure(
                "fused_residual_add",
                f"N={m}x128",
                lambda a=a, r=r: a + 0.5 * r,
                lambda a=a, r=r: tk.fused_residual_add(a, r, 1.0, 0.5),
                tol=1e-6,
                quick=quick,
            )
        )

    # --- cross entropy ----------------------------------------------------
    for m, c in (((405, 5),) if quick else ((405, 5), (8192, 5))):
        logits = torch.randn(m, c, device=dev)
        tgt = torch.randint(0, c, (m,), device=dev)
        rows.append(
            measure(
                "fused_cross_entropy",
                f"M={m},C={c}",
                lambda logits=logits, tgt=tgt: torch.nn.functional.cross_entropy(
                    logits, tgt
                ),
                lambda logits=logits, tgt=tgt: tk.fused_cross_entropy(logits, tgt),
                tol=1e-5,
                quick=quick,
            )
        )

    # --- scatter ----------------------------------------------------------
    scatter_shapes = ((736, 1200),) if quick else ((736, 1200), (2048, 16384), (16384, 131072))
    for n_atoms, n_edges in scatter_shapes:
        ei = torch.stack(
            [
                torch.randint(0, n_atoms, (n_edges,), device=dev),
                torch.randint(0, n_atoms, (n_edges,), device=dev),
            ]
        )
        feat = torch.randn(n_edges, 3, device=dev)
        rows.append(
            measure(
                "aggregate_vectors",
                f"N={n_atoms},E={n_edges}",
                lambda ei=ei, feat=feat, n=n_atoms: torch.zeros(
                    n, 3, device=dev
                ).index_add_(0, ei[1], feat),
                lambda ei=ei, feat=feat, n=n_atoms: tk.aggregate_vectors(
                    ei, feat, n
                ),
                tol=1e-5,
                quick=quick,
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="3 shapes instead of the full set")
    ap.add_argument(
        "--out",
        default=str(_REPO / "metrics" / "by_metric" / "triton_blackwell_bench.json"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device; this table is a Blackwell measurement.")
        return 1

    print(f"{torch.cuda.get_device_name(0)}  {tk_backend.describe()}")
    print(f"triton {triton.__version__}  torch {torch.__version__}\n")
    rows = collect(args.quick)

    enabled, disabled = [], []
    print()
    for op in sorted({r["op"] for r in rows}):
        rs = [r for r in rows if r["op"] == op]
        ok = all(r["numerically_ok"] for r in rs)
        worst = min(r["speedup"] for r in rs)
        keep = ok and worst > 1.0
        (enabled if keep else disabled).append(op)
        print(
            f"  {op:22s} correct={ok}  worst speedup={worst:5.2f}x  "
            f"-> {'ENABLE' if keep else 'DISABLE'}"
        )
    print("\nenabled :", enabled)
    print("disabled:", disabled)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "triton": triton.__version__,
        "torch": torch.__version__,
        "measured_on": "2026-10-23",
        "quick": bool(args.quick),
        "rule": "enable iff numerically_ok on every shape AND min speedup > 1.0",
        "rows": rows,
        "enabled_ops": enabled,
        "disabled_ops": disabled,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
