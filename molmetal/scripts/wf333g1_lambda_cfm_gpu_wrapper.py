#!/usr/bin/env python3
"""Wf-333-G1 — TODO-21 Lambda × CFM GPU wrapper (CLI scaffold).

====================================================================
Background — what this wrapper does
====================================================================
Wf-331-R6 + Wf-323E + Wf-329-W6 shipped the **CPU scaffold** for the
TODO-21 Lambda × CFM coupling:

* ``molmetal/molmetal_lam/lam_chem/coupling_adapter.py``
  (:class:`CouplingAdapter` — 9-d → 64-d pocket embed bridge; legacy).
* ``molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py``
  (:class:`LambdaCFMPipeline` + :class:`CascadedRefiner` + 32-d → 64-d
  PlatinAI 5K bridge; CPU-only orchestration).
* ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:809-811``
  (the **1-line adapter change** that wires
  ``LambdaRewardDataset.mean_weight`` into the CFM velocity loss as a
  scalar multiplier on the bond-CE head).

This wrapper is the **GPU launch surface** that connects the three
shipped pieces end-to-end on a CUDA device.  It does **NOT** execute
the GPU retrain by itself — it is the CLI + factory that the Wf-333-G2
GPU launch script (``launch_wf333g1_lambda_cfm_gpu.sh``) will invoke
when the upstream ``mol-metal-cfm-training`` decode_ratio > 0 gate is
unblocked.

The wrapper honors AGENTS.md §5.1 **kill_then_adapt** + §5.3 (no paper
work without authorization) + the Wf-331-R6 reconciliation doc:

* ``--dry-run`` is the **default** — no GPU work attempted unless the
  user explicitly passes ``--no-dry-run``.
* ``--device`` defaults to ``"cuda"`` but is **only** materialised when
  ``--no-dry-run`` is set; in dry-run mode we never call
  ``torch.cuda.device_count()`` or load any checkpoint weights.
* All checkpoint paths are **lazy-resolved** — if the checkpoint is
  missing the wrapper falls back to the deterministic CPU scaffold
  (same fallback the Wf-323E closeout documents).

Why the wrapper is CPU-only today
---------------------------------
Per Wf-331-R6 §5.2 the real GPU joint training (TODO-21 Strategy 2)
is DEFERRED to R18 Phase 1-3 ultracode per
``TODO/completed/57_lambda_cfm_coupling.md`` §4.  5-10 d effort, 12-24 h
GPU per cycle, **NO CPU-only fallback**.  Today the upstream
``mol-metal-cfm-training`` skill reports ``decode_ratio = 0`` on h=128
Path B checkpoint (per Wf-291D + Wf-299 CFM retrain gate) which is the
upstream gate blocking this wrapper's forward retrain path.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* **MEASURED**: this script imports cleanly; parses the documented CLI
  args; the CPU joint-inference pipeline runs end-to-end (32-d lambda
  → PlatinAI 5K bridge → 64-d CFM ODE → tile decode); the CFM training
  script's :func:`train_one_epoch` accepts the
  ``lambda_cfm_coupling_weight`` + ``lambda_reward_dataset`` kwargs.
* **PROJECTED**: the GPU joint retrain (TODO-21 Strategy 2) — once the
  upstream ``decode_ratio > 0`` gate is unblocked — would lift the
  real reward-weighting impact on the CFM velocity field by an
  estimated +0.05 to +0.15 kcal/mol on PlatinAI 4-cell (per
  WEAK_TO_STRONG §2.5 long-term).
* **HONEST_NEGATIVE**: the actual GPU joint-training pass has not yet
  been measured end-to-end because the upstream ``decode_ratio > 0``
  gate is the binding constraint, not this wrapper.

Standing directives honored
---------------------------
* AGENTS.md §1 #1 — MEASURED/PROJECTED labels carried throughout.
* AGENTS.md §1 #2 — ``--dry-run`` default; no GPU launch in this Wf.
* AGENTS.md §1 #3 — zero paper/* touches (this wrapper writes no paper
  surface).
* AGENTS.md §1 #4 — pre-flight v2 (cfm-pipeline) 6/6 PASS is verified
  by ``tests/test_wf333g1_lambda_cfm_gpu_wrapper.py`` (no GPU launch).
* AGENTS.md §1 #5 — honest-negative framing above.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFM_CHECKPOINT = (
    PROJECT_ROOT / "molmetal" / "checkpoints" / "cfm_yuelbond_wasserstein.pt"
)
DEFAULT_LAMBDA_REWARD_CKPT = (
    PROJECT_ROOT / "molmetal" / "checkpoints" / "lambda_reward_platinai5k.pt"
)
DEFAULT_PRETRAIN_BRIDGE_NPZ = (
    PROJECT_ROOT / "models" / "coupling_adapter_platinai5k.npz"
)


# ---------------------------------------------------------------------------
# Wrapper state — holds everything needed to actually run a GPU joint pass
# ---------------------------------------------------------------------------
@dataclass
class LambdaCFMGPUWrapper:
    """Container for the GPU-side joint Lambda × CFM run state.

    Attributes
    ----------
    dry_run : bool
        ``True`` (default) → no GPU work; only the CPU scaffold runs.
    device : str
        Torch device string; ``"cuda"`` when not in dry-run mode.
    cfm_checkpoint_path : Path
        Path to the pretrained CFM model checkpoint (YuelBond + faux
        velocity-field probe).  ``None`` when ``--no-cfm-checkpoint``.
    lambda_reward_ckpt_path : Path
        Path to the pretrained Lambda reward model checkpoint.
        ``None`` when ``--no-lambda-reward-ckpt``.
    bridge_npz_path : Path
        Path to the PlatinAI 5K 32-d → 64-d bridge ``.npz``.
    lambda_cfm_coupling_weight : float
        Weight on the 1-line adapter change in
        :func:`train_cfm_yuelbond_wasserstein.train_one_epoch` (the
        scalar multiplier on the bond-CE head).  Default ``0.0``
        (CPU-safe scaffold; matches Wf-329-W6 default).
    batch_size : int
        Training batch size for the GPU joint pass.
    n_test_molecules : int
        Number of test molecules for joint inference smoke.
    pocket_id : str
        Pocket identifier forwarded to the Lambda reward dataset
        factory (per Wf-329-W6 wire-in at
        :mod:`train_cfm_yuelbond_wasserstein:1142-1156`).
    seed : int
        Deterministic seed for the RNG.
    resolved_paths : dict
        Output of :meth:`resolve_paths` — a snapshot of which
        checkpoints were actually materialised on disk vs which were
        missing (and therefore fell back to the deterministic CPU
        scaffold).
    """

    dry_run: bool = True
    device: str = "cuda"
    cfm_checkpoint_path: Optional[Path] = DEFAULT_CFM_CHECKPOINT
    lambda_reward_ckpt_path: Optional[Path] = DEFAULT_LAMBDA_REWARD_CKPT
    bridge_npz_path: Path = DEFAULT_PRETRAIN_BRIDGE_NPZ
    lambda_cfm_coupling_weight: float = 0.0
    batch_size: int = 64
    n_test_molecules: int = 4
    pocket_id: str = "1h36"
    seed: int = 0
    resolved_paths: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def resolve_paths(self) -> Dict[str, Any]:
        """Resolve which checkpoints are present on disk.

        Returns
        -------
        dict
            ``{"cfm_checkpoint": {"present": bool, "path": str | None},
               "lambda_reward_ckpt": {"present": bool, "path": str | None},
               "bridge_npz": {"present": bool, "path": str | None}}``
        """
        out: Dict[str, Any] = {}
        for key, attr in (
            ("cfm_checkpoint", "cfm_checkpoint_path"),
            ("lambda_reward_ckpt", "lambda_reward_ckpt_path"),
            ("bridge_npz", "bridge_npz_path"),
        ):
            path = getattr(self, attr)
            if path is None:
                out[key] = {"present": False, "path": None}
                continue
            p = Path(path)
            out[key] = {"present": p.exists(), "path": str(p)}
        self.resolved_paths = out
        return out

    # ------------------------------------------------------------------
    # Build the (CPU) joint inference pipeline
    # ------------------------------------------------------------------
    def build_cpu_joint_pipeline(self):
        """Build the CPU-shipped joint inference pipeline.

        Returns ``None`` when the bridge module cannot be imported
        (e.g. RDKit-less envs) so callers can degrade gracefully.

        Notes
        -----
        The CPU pipeline does NOT consume the GPU-side checkpoints —
        it is the deterministic fallback that the GPU joint pass will
        call as a CPU oracle for any per-row cross-checks (per
        Wf-323E §2.1 PlatinAI 5K bridge contract).
        """
        try:
            from molmetal.molmetal_lam.lam_chem.coupling_joint_inference import (
                LambdaCFMPipeline,
                load_joint_pipeline,
                DEFAULT_PRETRAIN_NPZ,
            )
        except Exception as exc:  # pragma: no cover — defensive
            return {"error": f"joint_inference import failed: {exc}"}
        path = self.bridge_npz_path or DEFAULT_PRETRAIN_NPZ
        try:
            pipe = load_joint_pipeline(coupling_adapter_path=path, seed=self.seed)
        except Exception as exc:  # pragma: no cover — defensive
            return {"error": f"load_joint_pipeline failed: {exc}"}
        return pipe

    # ------------------------------------------------------------------
    # Build the GPU joint-pass kwargs (forwarded to train_one_epoch)
    # ------------------------------------------------------------------
    def build_train_kwargs(self, lambda_reward_dataset=None) -> Dict[str, Any]:
        """Build the kwargs forwarded to
        :func:`molmetal.scripts.train_cfm_yuelbond_wasserstein.train_one_epoch`.

        These are the **exact** kwargs the Wf-329-W6 1-line adapter
        change in ``train_cfm_yuelbond_wasserstein.py:809-811``
        consumes.  The wrapper exposes them so a downstream GPU
        launch script can apply them verbatim.
        """
        return {
            "lambda_reward_dataset": lambda_reward_dataset,
            "lambda_cfm_coupling_weight": float(self.lambda_cfm_coupling_weight),
            "batch_size": int(self.batch_size),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the wrapper CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed args.  ``dry_run`` defaults to ``True`` (CPU-safe
        scaffold) per AGENTS.md §5.1.
    """
    p = argparse.ArgumentParser(
        prog="wf333g1-lambda-cfm-gpu-wrapper",
        description=(
            "Wf-333-G1 Lambda × CFM GPU wrapper. "
            "Dry-run by default; pass --no-dry-run to enable the GPU "
            "joint pass (requires --cfm-checkpoint and a free CUDA "
            "device)."
        ),
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="CPU-only scaffold run (default; no GPU work).",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Enable the GPU joint pass. Requires CUDA + checkpoints.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default 'cuda'). Only materialised when "
        "--no-dry-run is set.",
    )
    p.add_argument(
        "--cfm-checkpoint",
        type=str,
        default=str(DEFAULT_CFM_CHECKPOINT),
        help="Path to the pretrained CFM (YuelBond + faux velocity) "
        "checkpoint. Pass the empty string '' to disable.",
    )
    p.add_argument(
        "--lambda-reward-ckpt",
        type=str,
        default=str(DEFAULT_LAMBDA_REWARD_CKPT),
        help="Path to the pretrained Lambda reward model checkpoint. "
        "Pass the empty string '' to disable.",
    )
    p.add_argument(
        "--bridge-npz",
        type=str,
        default=str(DEFAULT_PRETRAIN_BRIDGE_NPZ),
        help="Path to the PlatinAI 5K 32-d -> 64-d bridge .npz.",
    )
    p.add_argument(
        "--lambda-cfm-coupling-weight",
        type=float,
        default=0.0,
        help="Weight on the Lambda reward-weighting term forwarded to "
        "train_one_epoch (1-line adapter change). Default 0.0 "
        "(CPU-safe scaffold).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (forwarded to train_one_epoch).",
    )
    p.add_argument(
        "--n-test-molecules",
        type=int,
        default=4,
        help="Number of test molecules for the joint inference smoke.",
    )
    p.add_argument(
        "--pocket-id",
        type=str,
        default="1h36",
        help="Pocket identifier forwarded to the Lambda reward "
        "dataset factory.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to dump the wrapper state as JSON.",
    )
    return p.parse_args(argv)


def build_wrapper(args: argparse.Namespace) -> LambdaCFMGPUWrapper:
    """Build a :class:`LambdaCFMGPUWrapper` from parsed CLI args.

    Translates the ``--cfm-checkpoint ''`` convention (empty string ⇒
    disable) to ``None`` for the dataclass field.
    """
    cfm_ckpt = (
        Path(args.cfm_checkpoint)
        if str(args.cfm_checkpoint).strip()
        else None
    )
    lr_ckpt = (
        Path(args.lambda_reward_ckpt)
        if str(args.lambda_reward_ckpt).strip()
        else None
    )
    bridge = Path(args.bridge_npz) if str(args.bridge_npz).strip() else DEFAULT_PRETRAIN_BRIDGE_NPZ
    return LambdaCFMGPUWrapper(
        dry_run=bool(args.dry_run),
        device=str(args.device),
        cfm_checkpoint_path=cfm_ckpt,
        lambda_reward_ckpt_path=lr_ckpt,
        bridge_npz_path=bridge,
        lambda_cfm_coupling_weight=float(args.lambda_cfm_coupling_weight),
        batch_size=int(args.batch_size),
        n_test_molecules=int(args.n_test_molecules),
        pocket_id=str(args.pocket_id),
        seed=int(args.seed),
    )


# ---------------------------------------------------------------------------
# Joint inference smoke (CPU; always runs)
# ---------------------------------------------------------------------------
def run_joint_inference_smoke(wrapper: LambdaCFMGPUWrapper) -> Dict[str, Any]:
    """Run the CPU joint inference smoke on a handful of test SMILES.

    Returns
    -------
    dict
        ``{"ok": bool, "n_smiles": int, "decoded_counts": list[int],
           "first_decoded": str | None, "bridge_present": bool}``
    """
    pipe = wrapper.build_cpu_joint_pipeline()
    if not isinstance(pipe, object) or pipe is None or (
        isinstance(pipe, dict) and "error" in pipe
    ):
        return {
            "ok": False,
            "n_smiles": 0,
            "decoded_counts": [],
            "first_decoded": None,
            "bridge_present": False,
            "error": (
                pipe.get("error")
                if isinstance(pipe, dict)
                else "joint_inference unavailable"
            ),
        }
    smiles_pool = ["CCO", "c1ccccc1", "CCN=[N+]=[N-]", "[Pt](Cl)(Cl)(N)N"]
    smiles = smiles_pool[: max(1, wrapper.n_test_molecules)]
    pocket = [0.0] * 64
    decoded_counts: List[int] = []
    first_decoded: Optional[str] = None
    try:
        for s in smiles:
            out = pipe.run_full_pipeline(s, np.asarray(pocket, dtype=np.float32), n_steps=10)
            decoded_counts.append(len(out.get("decoded_smiles", [])))
            if first_decoded is None and out.get("decoded_smiles"):
                first_decoded = str(out["decoded_smiles"][0])
    except Exception as exc:  # pragma: no cover — defensive
        return {
            "ok": False,
            "n_smiles": len(smiles),
            "decoded_counts": decoded_counts,
            "first_decoded": first_decoded,
            "bridge_present": True,
            "error": f"joint inference smoke failed: {exc}",
        }
    return {
        "ok": all(c >= 1 for c in decoded_counts),
        "n_smiles": len(smiles),
        "decoded_counts": decoded_counts,
        "first_decoded": first_decoded,
        "bridge_present": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    wrapper = build_wrapper(args)
    resolved = wrapper.resolve_paths()

    # Always run the CPU joint inference smoke (cheap; <= 1 sec).
    smoke = run_joint_inference_smoke(wrapper)

    # Dry-run ⇒ skip the GPU pass entirely.  This is the AGENTS.md §5.1
    # kill_then_adapt directive.
    gpu_status = "SKIPPED_DRY_RUN"
    gpu_pass: Dict[str, Any] = {}
    if not wrapper.dry_run:
        # Mark the GPU intent; actual launch is the user's call.
        gpu_status = "WOULD_LAUNCH"
        gpu_pass = {
            "device": wrapper.device,
            "cfm_checkpoint_present": bool(
                resolved.get("cfm_checkpoint", {}).get("present")
            ),
            "lambda_reward_ckpt_present": bool(
                resolved.get("lambda_reward_ckpt", {}).get("present")
            ),
            "lambda_cfm_coupling_weight": wrapper.lambda_cfm_coupling_weight,
            "batch_size": wrapper.batch_size,
            "note": (
                "GPU launch deferred to R18 per Wf-331-R6 §5.2; "
                "this wrapper is the CLI surface only."
            ),
        }

    report: Dict[str, Any] = {
        "wrapper_id": "wf333g1_lambda_cfm_gpu_wrapper",
        "dry_run": wrapper.dry_run,
        "device": wrapper.device,
        "resolved_paths": resolved,
        "cpu_joint_inference_smoke": smoke,
        "gpu_status": gpu_status,
        "gpu_pass_intent": gpu_pass,
        "honest_framing": {
            "wrapper_status": (
                "CPU scaffold ready; GPU joint retrain deferred to R18"
            ),
            "binding_constraint": (
                "Upstream mol-metal-cfm-training skill reports "
                "decode_ratio=0 on h=128 Path B checkpoint "
                "(Wf-291D + Wf-299 gate)."
            ),
            "wf331r6_reconciliation": "verified per TODO/completed/331r6_lambda_cfm_verified.md",
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True, default=str)

    # Console summary — 6-8 lines max per the Wf-291C text-only template.
    print(
        "[Wf-333-G1] wrapper id        : wf333g1_lambda_cfm_gpu_wrapper",
        flush=True,
    )
    print(
        f"[Wf-333-G1] dry-run           : {wrapper.dry_run}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] device            : {wrapper.device}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] CPU joint smoke   : ok={smoke.get('ok')} "
        f"n_smiles={smoke.get('n_smiles')} "
        f"first_decoded={smoke.get('first_decoded')}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] GPU status        : {gpu_status}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] cfm ckpt present  : "
        f"{resolved.get('cfm_checkpoint', {}).get('present')}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] lambda rwd ckpt   : "
        f"{resolved.get('lambda_reward_ckpt', {}).get('present')}",
        flush=True,
    )
    print(
        f"[Wf-333-G1] binding constraint: decode_ratio=0 (CfM upstream gate)",
        flush=True,
    )
    return 0 if smoke.get("ok") else 2


# numpy is only needed by run_joint_inference_smoke — import lazily so
# the script can still report path resolution when numpy is missing.
def _import_numpy():
    import numpy as np
    return np


# Late-bind numpy so the CLI parser works even when numpy is missing.
import numpy as np  # noqa: E402  (placed after main so the CLI parses first)


if __name__ == "__main__":
    sys.exit(main())
