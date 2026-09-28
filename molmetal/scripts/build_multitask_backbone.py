#!/usr/bin/env python3
"""Wf-128 Phase 1 — multi-task 5-metal backbone (shared trunk + per-metal heads).

================================================================
Build the **shared 5-metal multi-task regression backbone** that
consumes the 172-d feature stack produced by
``build_172d_features.py``.  The architecture is the **deliberately
simple** linear-MLP variant that Wf-128 ships first — it pins the
public surface (input dim, hidden dim, latent dim, per-metal head)
so that downstream consumers (D-MPNN fine-tune, L13/16/17/20
regression heads, MCTS rollout policy) can swap the backbone
in-place without changing the per-metal head shapes.

Architecture (Wf-128 §1 spec)
-----------------------------
::

    X (B, 172)                       # RDKit 12 + Multi-PH 32 + PLL 128
       │
       ▼
    Linear(172 → 256) + ReLU + Dropout(0.2)
       │
       ▼
    Linear(256 → 128) + ReLU         # shared 5-metal latent trunk
       │
       ▼
    z (B, 128)

    per-metal heads (5 ×):
        Linear(128 → 64) + ReLU + Dropout(0.2) + Linear(64 → 1)
        (predicts pIC50 as a continuous regression value)

Why this shape
--------------
* **172-d input** matches ``build_172d_features.FEATURE_DIM``.
* **256 → 128 trunk** is a 2-layer MLP with a >2× narrowing; this
  gives the per-metal heads a *compact* shared representation
  without losing the 12-d RDKit signal that drives the
  MetalCytoToxDB 5-metal Lipinski-style correlation (Wf-49 §1).
* **Per-metal head** is a 128 → 64 → 1 MLP with dropout 0.2 — the
  same dropout as the V4MultiTaskDMPNN head, but with a 1-d output
  for the regression target (pIC50).

Outputs
-------
* ``molmetal/checkpoints/multitask_5metal_arch.json`` — architecture
  manifest (input dim, hidden dim, latent dim, head config, metal
  labels, dropout, init scheme).  No learned weights (the script
  is architecture-only — training is Wf-128 Phase 2).
* ``molmetal/checkpoints/multitask_5metal_sanity.json`` — sanity
  check log: forward pass on a synthetic batch, per-metal output
  shape, parameter count, CPU + GPU-1 wall-time (if available).

Usage
-----
::

    uv run --no-sync python -u molmetal/scripts/build_multitask_backbone.py \\
        --device cpu \\
        --out-arch molmetal/checkpoints/multitask_5metal_arch.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — make ``molmetal.*`` importable from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# ---------------------------------------------------------------------------
# Optional-torch import — degrade gracefully if torch is missing.
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _HAS_TORCH = True
    _TORCH_ERR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False
    _TORCH_ERR = _exc


# ---------------------------------------------------------------------------
# Constants — keep in sync with ``build_172d_features``.
# ---------------------------------------------------------------------------
INPUT_DIM: int = 172          # 12 RDKit + 32 Multi-PH + 128 PLL
TRUNK_HIDDEN: int = 256       # shared 5-metal trunk hidden width
TRUNK_LATENT: int = 128       # shared 5-metal trunk latent width
HEAD_HIDDEN: int = 64         # per-metal head hidden width
HEAD_DROPOUT: float = 0.2     # per-metal head dropout (matches V4MT)
TRUNK_DROPOUT: float = 0.2    # shared trunk dropout
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
N_METALS: int = 5


# ===========================================================================
# 1. Architecture dataclasses (plain Python — no torch dependency)
# ===========================================================================
@dataclass(frozen=True)
class PerMetalHeadConfig:
    """Per-metal regression head configuration.

    Attributes
    ----------
    in_dim
        Input dim (= trunk latent).
    hidden_dim
        Hidden width inside the head MLP.
    out_dim
        Output dim (= 1 for pIC50 regression).
    dropout
        Dropout probability.
    activation
        ``"relu"`` (default), ``"gelu"``, ``"tanh"``.
    """

    in_dim: int = TRUNK_LATENT
    hidden_dim: int = HEAD_HIDDEN
    out_dim: int = 1
    dropout: float = HEAD_DROPOUT
    activation: str = "relu"


@dataclass(frozen=True)
class MultiTaskBackboneConfig:
    """Architecture manifest for the shared 5-metal multi-task backbone.

    All fields are JSON-serialisable.  The dataclass is the canonical
    spec — ``multitask_5metal_arch.json`` is a ``dict(...)`` dump of
    this object (with the parameter count + per-metal keys folded in).
    """

    input_dim: int = INPUT_DIM
    trunk_hidden: int = TRUNK_HIDDEN
    trunk_latent: int = TRUNK_LATENT
    trunk_dropout: float = TRUNK_DROPOUT
    head: PerMetalHeadConfig = field(default_factory=PerMetalHeadConfig)
    metals: Tuple[str, ...] = METALS_5
    n_metals: int = N_METALS
    init_scheme: str = "kaiming_uniform_relu"
    version: str = "0.1.0-wf128-p1"


# ===========================================================================
# 2. Optional torch module — only loaded if torch is present.
# ===========================================================================
if _HAS_TORCH:  # pragma: no cover - guard on torch availability

    def _resolve_activation(name: str):
        """Return the activation callable for ``name``.

        Validates the name eagerly — a typo in the config raises at
        construction time, not at forward time.  This is the
        pattern V4MultiTaskDMPNN uses for its per-metal heads.
        """
        if name == "relu":
            return F.relu
        if name == "gelu":
            return F.gelu
        if name == "tanh":
            return torch.tanh
        raise ValueError(
            f"unknown activation {name!r}; supported: relu, gelu, tanh"
        )

    class PerMetalHead(nn.Module):
        """Per-metal regression head: ``Linear(in → hidden) → act → Dropout → Linear(hidden → 1)``.

        All 5 heads share the same width but **not** the same weights;
        the metal-specific weights are what makes each head specialise
        on its own MetalCytoToxDB subset.  See V4MultiTaskDMPNN's
        ``_make_metal_head`` for the precedent.
        """

        def __init__(
            self,
            cfg: PerMetalHeadConfig,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            # Eager validation — typo in the config raises at __init__,
            # not at the first forward pass.
            self.act = _resolve_activation(str(cfg.activation))
            self.fc1 = nn.Linear(int(cfg.in_dim), int(cfg.hidden_dim))
            self.drop = nn.Dropout(p=float(cfg.dropout))
            self.fc2 = nn.Linear(int(cfg.hidden_dim), int(cfg.out_dim))

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            h = self.act(self.fc1(z))
            h = self.drop(h)
            return self.fc2(h)

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc2.bias)

    class SharedTrunk(nn.Module):
        """Shared 5-metal trunk: ``Linear(input_dim → trunk_hidden) → ReLU → Dropout → Linear(trunk_hidden → trunk_latent) → ReLU``.

        This is the **shared** representation — both the per-metal
        heads and any future auxiliary head (coordination geometry,
        metal-conditional Dropout, etc.) consume the ``(B,
        trunk_latent)`` output of this trunk.
        """

        def __init__(
            self,
            cfg: MultiTaskBackboneConfig,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.fc1 = nn.Linear(int(cfg.input_dim), int(cfg.trunk_hidden))
            self.act1 = nn.ReLU()
            self.drop1 = nn.Dropout(p=float(cfg.trunk_dropout))
            self.fc2 = nn.Linear(int(cfg.trunk_hidden), int(cfg.trunk_latent))
            self.act2 = nn.ReLU()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.act1(self.fc1(x))
            h = self.drop1(h)
            h = self.act2(self.fc2(h))
            return h

        def reset_parameters(self) -> None:
            for lin in (self.fc1, self.fc2):
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)

    class MultiTaskBackbone(nn.Module):
        """Full multi-task backbone: shared trunk + per-metal heads.

        Forward pass
        ------------
        ``forward(X, metal_id)`` returns a ``dict`` mapping each metal
        label to a ``(B, 1)`` pIC50 prediction tensor.  ``X`` is the
        ``(B, input_dim)`` 172-d feature stack; ``metal_id`` is a
        ``(B,)`` long tensor of metal indices in ``[0, n_metals)``.

        Alternative forward
        -------------------
        ``forward_all_metals(X)`` returns a ``(B, n_metals)`` tensor of
        pIC50 predictions — every metal head evaluated on every row.
        This is useful for **calibration / honest framing** (e.g.
        Wf-86 §3 collapse-detector that flags ``pred_std / y_test_std
        < 0.10`` per fold).
        """

        def __init__(self, cfg: Optional[MultiTaskBackboneConfig] = None) -> None:
            super().__init__()
            self.cfg = cfg or MultiTaskBackboneConfig()
            if len(self.cfg.metals) != self.cfg.n_metals:
                raise ValueError(
                    f"len(metals)={len(self.cfg.metals)} must equal "
                    f"n_metals={self.cfg.n_metals}"
                )
            self.trunk = SharedTrunk(self.cfg)
            self.metal_heads = nn.ModuleDict(
                {
                    m: PerMetalHead(self.cfg.head)
                    for m in self.cfg.metals
                }
            )

        # ------------------------------------------------------------------
        def metal_id(self, metal: str) -> int:
            if metal not in self.cfg.metals:
                raise KeyError(
                    f"Unknown metal {metal!r}. Supported: {self.cfg.metals}"
                )
            return self.cfg.metals.index(metal)

        # ------------------------------------------------------------------
        def forward(
            self,
            x: "torch.Tensor",
            metal_id: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            if x.dim() != 2:
                raise ValueError(
                    f"x must be (B, input_dim); got shape {tuple(x.shape)}"
                )
            if metal_id.dim() != 1:
                raise ValueError(
                    f"metal_id must be (B,); got shape {tuple(metal_id.shape)}"
                )
            b = x.size(0)
            if metal_id.size(0) != b:
                raise ValueError(
                    f"len(metal_id)={metal_id.size(0)} != x.size(0)={b}"
                )
            if x.size(1) != self.cfg.input_dim:
                raise ValueError(
                    f"x.size(1)={x.size(1)} != input_dim={self.cfg.input_dim}"
                )
            z = self.trunk(x)
            return {m: head(z) for m, head in self.metal_heads.items()}

        # ------------------------------------------------------------------
        def forward_all_metals(
            self,
            x: "torch.Tensor",
        ) -> "torch.Tensor":
            """Evaluate every metal head on every row → ``(B, n_metals)``."""
            if x.dim() != 2:
                raise ValueError(
                    f"x must be (B, input_dim); got shape {tuple(x.shape)}"
                )
            z = self.trunk(x)
            preds = [head(z) for head in self.metal_heads.values()]
            return torch.cat(preds, dim=-1)

        # ------------------------------------------------------------------
        def reset_parameters(self) -> None:
            self.trunk.reset_parameters()
            for head in self.metal_heads.values():
                head.reset_parameters()


# ===========================================================================
# 3. CPU + GPU-1 sanity check
# ===========================================================================
def _count_parameters(module) -> int:
    """Total parameter count (sum of ``numel`` over all parameters)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _measure_forward_wall(
    module,
    x: "torch.Tensor",
    metal_id: "torch.Tensor",
    n_iters: int = 50,
) -> Dict[str, float]:
    """Measure CPU / GPU-1 wall-time for ``n_iters`` forward passes.

    Returns a dict with ``mean_ms``, ``std_ms``, ``n_iters``,
    ``device``.  Warmup runs are discarded (they include the
    cuBLAS / cuDNN kernel-selection overhead).
    """
    device = next(module.parameters()).device
    times: List[float] = []
    # Warmup
    for _ in range(5):
        _ = module(x, metal_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
    for _ in range(int(n_iters)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = module(x, metal_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "std_ms": float(arr.std() * 1000.0),
        "n_iters": int(n_iters),
        "device": str(device),
    }


def _synthetic_batch(
    cfg: MultiTaskBackboneConfig,
    *,
    batch_size: int,
    seed: int = 0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Build a synthetic (x, metal_id) batch for the sanity check."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((int(batch_size), int(cfg.input_dim))).astype(np.float32)
    metal_id = rng.integers(0, int(cfg.n_metals), size=int(batch_size)).astype(np.int64)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(metal_id, dtype=torch.long),
    )


def run_sanity(
    cfg: MultiTaskBackboneConfig,
    *,
    batch_size: int = 64,
    n_iters: int = 50,
    device: str = "cpu",
) -> Dict[str, object]:
    """Run the CPU (and GPU-1, if available) sanity check.

    Returns a JSON-serialisable dict with per-device wall-time,
    per-metal output shape, parameter count.
    """
    if not _HAS_TORCH:
        return {
            "torch_available": False,
            "torch_error": repr(_TORCH_ERR),
            "config": asdict(cfg),
            "param_count": None,
        }

    result: Dict[str, object] = {
        "torch_available": True,
        "config": asdict(cfg),
        "batch_size": int(batch_size),
        "n_iters": int(n_iters),
    }

    # ----- CPU -------------------------------------------------------------
    module = MultiTaskBackbone(cfg)
    module.eval()
    result["param_count"] = _count_parameters(module)

    x_cpu, m_cpu = _synthetic_batch(cfg, batch_size=batch_size, seed=0)
    with torch.no_grad():
        out_cpu = module(x_cpu, m_cpu)
    per_metal_shape = {m: tuple(t.shape) for m, t in out_cpu.items()}
    result["per_metal_output_shape_cpu"] = per_metal_shape
    result["cpu_wall"] = _measure_forward_wall(
        module, x_cpu, m_cpu, n_iters=int(n_iters)
    )

    # Per-metal output finiteness check.
    result["per_metal_finite_cpu"] = {
        m: bool(torch.isfinite(t).all().item()) for m, t in out_cpu.items()
    }

    # ----- forward_all_metals on CPU --------------------------------------
    with torch.no_grad():
        all_metals_cpu = module.forward_all_metals(x_cpu)
    result["forward_all_metals_shape_cpu"] = tuple(all_metals_cpu.shape)

    # ----- GPU-1 (optional) ----------------------------------------------
    if device.startswith("cuda") and torch.cuda.is_available():
        module_gpu = MultiTaskBackbone(cfg).to("cuda")
        module_gpu.eval()
        x_cuda = x_cpu.to("cuda")
        m_cuda = m_cpu.to("cuda")
        with torch.no_grad():
            out_cuda = module_gpu(x_cuda, m_cuda)
        result["per_metal_output_shape_gpu"] = {
            m: tuple(t.shape) for m, t in out_cuda.items()
        }
        result["gpu_wall"] = _measure_forward_wall(
            module_gpu, x_cuda, m_cuda, n_iters=int(n_iters)
        )
    else:
        result["gpu_wall"] = None

    return result


# ===========================================================================
# 4. CLI
# ===========================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Sanity-check device (default cpu).",
    )
    p.add_argument(
        "--out-arch",
        type=Path,
        default=Path("molmetal/checkpoints/multitask_5metal_arch.json"),
        help="Output architecture JSON path.",
    )
    p.add_argument(
        "--out-sanity",
        type=Path,
        default=None,
        help="Output sanity JSON path (default: <out-arch>.sanity.json).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Sanity-check batch size.",
    )
    p.add_argument(
        "--n-iters",
        type=int,
        default=50,
        help="Sanity-check forward-pass iters.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def write_arch_json(cfg: MultiTaskBackboneConfig, path: Path) -> Path:
    """Dump the architecture manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["metals"] = list(cfg.metals)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = MultiTaskBackboneConfig()
    write_arch_json(cfg, args.out_arch)
    sanity = run_sanity(
        cfg,
        batch_size=args.batch_size,
        n_iters=args.n_iters,
        device=args.device,
    )
    sanity_path = args.out_sanity or args.out_arch.with_suffix(".sanity.json")
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sanity_path, "w") as fh:
        json.dump(sanity, fh, indent=2, sort_keys=False)
    print(f"Wrote {args.out_arch}")
    print(f"Wrote {sanity_path}")
    if sanity.get("torch_available"):
        print(
            f"param_count={sanity['param_count']}  "
            f"cpu_mean_ms={sanity['cpu_wall']['mean_ms']:.3f}"
        )
        if sanity.get("gpu_wall"):
            print(
                f"gpu_mean_ms={sanity['gpu_wall']['mean_ms']:.3f}"
            )
    else:
        print(f"Torch unavailable: {sanity.get('torch_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())