"""VRAM measurement harness for Mol-Metal CFM training.

================================================================
Why this script exists
================================================================
The VRAM TOP #1 fix is BF16 autocast + per-EGNN-layer gradient
checkpointing (see ``molmetal/reports/wf_vram_fix/01_amp.md`` and
``molmetal/reports/wf_vram_fix/02_checkpoint.md``).  Both halves
shipped CPU-verified — but neither one shipped with **measured VRAM
deltas on real GPU**.

This script is the missing measurement harness: it builds a small
CFM model with ``LipmanFlowMatchingAdapter`` (using existing
init helpers — *not* new training code), runs forward + backward
for 3 steps under four precision/activation regimes, and reports:

* ``peak_allocated_mb``  : ``torch.cuda.max_memory_allocated()`` / 1 MiB
* ``peak_reserved_mb``   : ``torch.cuda.max_memory_reserved()`` / 1 MiB
* ``current_allocated_mb``  : at end of measurement
* ``fragmentation_mb``  : ``peak_reserved - peak_allocated`` (unusable
                          reserved blocks in the caching allocator)
* ``step_time_mean_ms``  : mean wall time per (forward + backward) step

Usage
-----
    # all 4 modes side-by-side
    uv run python molmetal/scripts/measure_vram.py

    # single mode for CI / smoke
    uv run python molmetal/scripts/measure_vram.py --mode bf16+checkpoint

    # just baseline (FP32 + no checkpoint)
    uv run python molmetal/scripts/measure_vram.py --mode baseline

CPU fallback
------------
On hosts without a usable GPU (RX 7800 XT SMU hang or CI), the
script prints a single ``SKIPPED`` line and exits 0 — the per-mode
``measure_vram()`` function still returns a dict so callers can
branch on the ``"available"`` key.

Citations
---------
* Adapter init: ``molmetal/adapters/flow_matching_lipman/__init__.py:1726-1981``
* Adapter ``train_step``: ``molmetal/adapters/flow_matching_lipman/__init__.py:1986-2250``
* AMP wrapper: ``molmetal/adapters/flow_matching_lipman/amp.py:83-99`` (``recommended_amp_kwargs``)
* AMP context: ``molmetal/adapters/flow_matching_lipman/amp.py`` (``CFMAMPContext``)
* Checkpoint wrapper: ``molmetal/adapters/egnn_rocm_checkpoint.py`` (per-layer wrap, ``use_reentrant=False``)
* Device auto-detect: ``molmetal/utils/device.py`` (``get_device``)
* Sister verdicts: ``molmetal/reports/wf_vram_fix/01_amp.md``, ``02_checkpoint.md``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_ATOMS = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_N_STEPS = 3

ALL_MODES = ("baseline", "bf16", "checkpoint", "bf16+checkpoint")


# ---------------------------------------------------------------------------
# Synthetic batch
# ---------------------------------------------------------------------------

def _build_synthetic_batch(
    batch_size: int,
    n_atoms: int,
    device: torch.device,
) -> List[Any]:
    """Return a list of ``Molecule`` objects with deterministic coords.

    We synthesise the batch in-place rather than depending on RDKit /
    CSV / checkpoint files so the harness is hermetic and runs on a
    clean checkout with zero external data.

    Coords are on the unit sphere (norm 1) so the CFM path stays in
    a well-conditioned regime (no overflow / NaN risk from very large
    coords).  Atom-types are sampled from the canonical metallodrug
    vocab {C, N, O, F, S, P, Cl, Br, I, Pt, Pd, Au, Ir, Ru} = 6..78.

    Bonds are fully connected within each mol so the
    :class:`BondOrderHead` forward (when ``use_bond_head=True``) is
    exercised end-to-end.

    Cite: Molecule class — ``molmetal/domain.py``
    """
    from molmetal.domain import Molecule

    # Z = atomic number; index into adapter.atom_vocab at line ~1825
    # of flow_matching_lipman/__init__.py is {1,6,7,8,9,15,16,17,34,35,53,78}.
    # For the metallodrug set we extend with 46=Pd, 79=Au, 77=Ir, 44=Ru
    # to match the smoke-retrain vocab (see metallo_drug_smoke_retrain.py:46).
    vocab_z = (6, 7, 8, 9, 15, 16, 17, 35, 53, 78, 46, 79, 77, 44)

    g = torch.Generator(device="cpu").manual_seed(0)
    mols: List[Molecule] = []
    for b in range(batch_size):
        n = n_atoms
        # deterministic coords: permutation of n points on a circle
        # plus a small per-axis jitter (RNG seeded) so the CFM target
        # velocity has finite, non-zero magnitude.
        t = torch.linspace(0.0, 6.283185307179586, n)
        jitter = torch.randn(n, 3, generator=g) * 0.05
        coords = torch.zeros(n, 3)
        coords[:, 0] = torch.cos(t) + jitter[:, 0]
        coords[:, 1] = torch.sin(t) + jitter[:, 1]
        coords[:, 2] = jitter[:, 2]
        # atom_types: round-robin through vocab
        at = torch.tensor(
            [vocab_z[(b * n + i) % len(vocab_z)] for i in range(n)],
            dtype=torch.long,
        )
        # bonds: fully connected (n*(n-1)/2 undirected edges, both dirs)
        src_list: List[int] = []
        dst_list: List[int] = []
        bt_list: List[int] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                src_list.append(i)
                dst_list.append(j)
                bt_list.append(1)  # RDKit SINGLE=1
        bonds = torch.tensor([src_list, dst_list], dtype=torch.long)
        bond_types = torch.tensor(bt_list, dtype=torch.long)
        m = Molecule(
            coords=coords.to(device),
            atom_types=at.to(device),
            bonds=bonds,
            bond_types=bond_types,
            formal_charges=torch.zeros(n, dtype=torch.long, device=device),
        )
        mols.append(m)
    return mols


# ---------------------------------------------------------------------------
# Per-mode measurement
# ---------------------------------------------------------------------------

def measure_vram(
    mode: str = "baseline",
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_layers: int = DEFAULT_N_LAYERS,
    n_atoms: int = DEFAULT_N_ATOMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_steps: int = DEFAULT_N_STEPS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a 3-step CFM forward+backward under ``mode`` and report VRAM.

    Parameters
    ----------
    mode : str
        One of ``"baseline"``, ``"bf16"``, ``"checkpoint"``,
        ``"bf16+checkpoint"``.
    hidden_dim : int
        Adapter hidden width (default 64 — production-safe floor).
    n_layers : int
        Adapter EGNN depth (default 2).
    n_atoms : int
        Atoms per molecule in the synthetic batch.
    batch_size : int
        Number of molecules per train_step call.
    n_steps : int
        Forward+backward steps to time (3 by default; 1 warmup).
    seed : int
        torch.manual_seed seed for the adapter init.

    Returns
    -------
    dict
        ``{mode, available, peak_allocated_mb, peak_reserved_mb,
        current_allocated_mb, fragmentation_mb, step_time_mean_ms,
        n_steps, hidden_dim, n_layers, n_atoms, batch_size, error}``

    On a host without a usable GPU, returns ``{"available": False, ...}``
    with peak fields set to ``None``.
    """
    if mode not in ALL_MODES:
        raise ValueError(
            f"mode must be one of {ALL_MODES}, got {mode!r}"
        )

    use_bf16 = mode in ("bf16", "bf16+checkpoint")
    use_checkpoint = mode in ("checkpoint", "bf16+checkpoint")

    out: Dict[str, Any] = {
        "mode": mode,
        "available": False,
        "peak_allocated_mb": None,
        "peak_reserved_mb": None,
        "current_allocated_mb": None,
        "fragmentation_mb": None,
        "step_time_mean_ms": None,
        "n_steps": n_steps,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_atoms": n_atoms,
        "batch_size": batch_size,
        "error": None,
    }

    if not torch.cuda.is_available():
        out["error"] = "torch.cuda.is_available() == False"
        return out

    # Local imports — keep the harness import-light so callers that
    # only need the dict-shape contract on CPU don't pay for the
    # adapter's reference-repo import chain.
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    from molmetal.adapters.flow_matching_lipman.amp import CFMAMPContext

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory.empty_cache() if hasattr(torch.cuda, "memory") else None

    # Set the EGNN-checkpoint process-wide default BEFORE building the
    # adapter so each layer is wrapped.  See
    # ``molmetal/adapters/egnn_rocm_checkpoint.py`` for the env-driven
    # default; we override here for the harness.
    from molmetal.adapters.egnn_rocm_checkpoint import (
        egnn_checkpoint_default_enabled,
        set_egnn_checkpoint_default_enabled,
    )
    prev_egnn_ckpt = egnn_checkpoint_default_enabled()
    set_egnn_checkpoint_default_enabled(use_checkpoint)

    # Force-enable AMP at the adapter level too — the env-var path
    # (``MOLMETAL_CFM_AMP``) only fires inside the adapter's own
    # ``maybe_enable_amp()``.  The harness owns the autocast context
    # so we use the public ``CFMAMPContext`` directly.
    from molmetal.adapters.flow_matching_lipman import maybe_enable_amp
    prev_amp_setting = None  # we don't toggle the env; we own the context

    try:
        adapter = LipmanFlowMatchingAdapter(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            lr=1e-4,
            use_bond_head=True,         # exercise the bond path
            joint_train=True,           # bond head params in optimizer
            bond_loss_weight=1.0,
            bond_pattern_mask=True,
            vocab_mask=True,
            cfg_scale=1.0,              # skip the double-pass CFG
            context_dropout=0.1,
            pocket_embed_scale=0.1,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        adapter.setup(device=device)
        mols = _build_synthetic_batch(batch_size, n_atoms, adapter.device)

        # Reset stats AFTER setup + batch construction so the baseline
        # only captures the forward+backward cost, not init.
        torch.cuda.reset_peak_memory_stats()

        amp_ctx = CFMAMPContext(
            enabled=use_bf16,
            dtype=torch.bfloat16,
            device="cuda",
            cache_enabled=True,
        )

        step_times_ms: List[float] = []
        for step in range(n_steps):
            # Warm-up step: torch's first run allocates workspace
            # (cuBLAS handles, NCCL buffers, etc.) and is usually
            # 5-10x slower than steady-state.  We still record it so
            # callers can see the gap, but flag it in the dict.
            t0 = time.perf_counter()
            with amp_ctx:
                # Forward + backward via train_step — this is the same
                # path the real retrain uses (see
                # metallo_drug_smoke_retrain.py:361).  train_step
                # returns a Python float after running
                # ``self.optimizer.zero_grad()`` + the loss scalar's
                # ``.backward()`` internally (see
                # flow_matching_lipman/__init__.py:2010-2250), so the
                # activation tensors that drive ``max_memory_allocated``
                # have already been materialised and released by the
                # time we record the step time.  ``reset_peak_memory_stats``
                # BEFORE the loop + ``max_memory_allocated`` AFTER the
                # loop captures the high-water mark across all steps.
                loss_value = adapter.train_step(None, mols)
            _ = loss_value  # currently a float; float() coercion is the contract
            step_times_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
        cur_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        frag = peak_resv - peak_alloc
        # Drop warm-up step from the mean — report steady-state.
        steady = step_times_ms[1:] if len(step_times_ms) > 1 else step_times_ms
        mean_ms = sum(steady) / max(1, len(steady))

        out.update(
            available=True,
            peak_allocated_mb=float(peak_alloc),
            peak_reserved_mb=float(peak_resv),
            current_allocated_mb=float(cur_alloc),
            fragmentation_mb=float(frag),
            step_time_mean_ms=float(mean_ms),
            warmup_step_ms=float(step_times_ms[0]) if step_times_ms else None,
            device=str(adapter.device),
            torch_version=torch.__version__,
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Restore process-wide defaults so we don't leak state to the
        # caller / test suite.
        try:
            set_egnn_checkpoint_default_enabled(prev_egnn_ckpt)
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    return out


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _print_table(rows: List[Dict[str, Any]], baseline_row: Optional[Dict[str, Any]] = None) -> None:
    """Render the comparison table to stdout."""
    if not rows:
        print("[measure_vram] no rows to display")
        return

    headers = [
        "mode", "peak_alloc_mb", "peak_resv_mb", "frag_mb",
        "cur_alloc_mb", "step_ms", "delta_vs_baseline_mb",
    ]
    print("[measure_vram] " + " | ".join(f"{h:>22}" for h in headers))
    print("[measure_vram] " + "-" * (24 * len(headers)))

    base_peak = (baseline_row or {}).get("peak_allocated_mb") if baseline_row else None

    for row in rows:
        if not row.get("available"):
            err = row.get("error", "no GPU")
            print(
                f"[measure_vram] {row['mode']:<22} | "
                f"{'SKIPPED':>22} | err={err}"
            )
            continue
        delta = ""
        if base_peak is not None and row.get("peak_allocated_mb") is not None:
            d = row["peak_allocated_mb"] - base_peak
            sign = "+" if d >= 0 else ""
            delta = f"{sign}{d:+.2f}"
        cells = [
            row["mode"],
            f"{row['peak_allocated_mb']:.2f}",
            f"{row['peak_reserved_mb']:.2f}",
            f"{row['fragmentation_mb']:.2f}",
            f"{row['current_allocated_mb']:.2f}",
            f"{row['step_time_mean_ms']:.2f}",
            delta,
        ]
        print("[measure_vram] " + " | ".join(f"{c:>22}" for c in cells))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all",) + ALL_MODES,
        default="all",
        help=(
            "Which mode(s) to run.  'all' (default) runs all four "
            "regimes side-by-side and prints a delta-vs-baseline table.  "
            "Pass a single mode name for a one-shot smoke."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n-atoms", type=int, default=DEFAULT_N_ATOMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Optional path to dump the per-mode results as JSON.",
    )
    args = parser.parse_args(argv)

    print(f"[measure_vram] torch.cuda.is_available()={torch.cuda.is_available()}"
          f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"[measure_vram] hidden_dim={args.hidden_dim} n_layers={args.n_layers}"
          f" batch={args.batch_size} n_atoms={args.n_atoms} steps={args.n_steps}")

    modes_to_run = ALL_MODES if args.mode == "all" else (args.mode,)
    rows: List[Dict[str, Any]] = []
    for mode in modes_to_run:
        print(f"[measure_vram] --- running mode={mode} ---")
        row = measure_vram(
            mode=mode,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_atoms=args.n_atoms,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        rows.append(row)
        if row.get("available"):
            print(
                f"[measure_vram]   peak_alloc={row['peak_allocated_mb']:.2f} MB  "
                f"peak_resv={row['peak_reserved_mb']:.2f} MB  "
                f"frag={row['fragmentation_mb']:.2f} MB  "
                f"step_ms={row['step_time_mean_ms']:.2f}"
            )
        else:
            print(f"[measure_vram]   SKIPPED: {row.get('error')}")

    baseline_row = next((r for r in rows if r["mode"] == "baseline"), None)
    _print_table(rows, baseline_row=baseline_row)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"[measure_vram] wrote {args.json_out}")

    # Exit 0 even if no GPU — callers / CI should treat the absence
    # of "available": True rows as the signal, not the exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
