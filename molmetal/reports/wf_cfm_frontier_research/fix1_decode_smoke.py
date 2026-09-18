"""Decode smoke for WF-CFM-Frontier-Research Fix #1.

Loads the h=128 5000-step diagnostic checkpoint from
``molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt``,
constructs a :class:`LipmanFlowMatchingAdapter` with Fix #1 defaults
active (``use_bond_head=True``, ``joint_train=True``), generates 8
samples at 200 ODE steps, and counts how many of them produce a
non-empty bond graph + a valid SMILES (i.e. ``decode_ratio > 0``).

Run on GPU (RX 7800 XT) per `wf_gpu_recovery_now/final.md` (cuda=True
device_count=2 confirmed 2026-09-15).  Falls back to CPU if GPU is
unavailable.

Usage::

    uv run python molmetal/reports/wf_cfm_frontier_research/fix1_decode_smoke.py

Exit code is 0 if any decoded > 0 (Fix #1 worked at the wiring level),
1 otherwise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CKPT_PATH = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf_cfm_gpu_retrain"
    / "diagnostic"
    / "checkpoint_seed0.pt"
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> int:
    device = _device()
    print(f"[smoke] device={device}  cuda_available={torch.cuda.is_available()}")
    if not CKPT_PATH.exists():
        print(f"[smoke] FAIL: checkpoint missing: {CKPT_PATH}")
        return 1

    print(f"[smoke] loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    vf_state = ckpt["velocity_field"]
    pe_state = ckpt["pocket_encoder"]
    print(f"[smoke] checkpoint keys: velocity_field ({len(vf_state)} tensors), "
          f"pocket_encoder ({len(pe_state)} tensors)")

    # Build the adapter with Fix #1 defaults active.  The adapter's
    # constructor-level default for joint_train is now True (Fix #1);
    # we still pass it explicitly so the smoke is self-documenting.
    print("[smoke] constructing LipmanFlowMatchingAdapter (use_bond_head=True, joint_train=True)")
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=128,
        n_layers=3,
        max_atomic_number=100,
        use_bond_head=True,
        joint_train=True,
        vocab_mask=True,
    )
    try:
        adapter.setup(device=str(device))
    except Exception as exc:
        print(f"[smoke] adapter.setup() failed: {type(exc).__name__}: {exc}")
        return 1

    # Confirm Fix #1 is functional at the wiring level (regardless of
    # whether the trained checkpoint can produce a non-degenerate
    # coord cloud).
    print(f"[smoke] Fix #1 wiring checks:")
    print(f"[smoke]   adapter._use_bond_head = {adapter._use_bond_head}")
    print(f"[smoke]   adapter._joint_train   = {adapter._joint_train}")
    print(f"[smoke]   adapter.bond_head is None? {adapter.bond_head is None}")
    if adapter.bond_head is not None:
        n_bond_head_params = sum(p.numel() for p in adapter.bond_head.parameters())
        print(f"[smoke]   bond_head param count = {n_bond_head_params}")
    print(f"[smoke]   adapter.bond_head in optimizer? "
          f"{any(any(id(p) == id(bp) for bp in adapter.bond_head.parameters()) for g in adapter.optimizer.param_groups for p in g['params']) if adapter.bond_head is not None else 'n/a'}")
    # atom_vocab mask
    print(f"[smoke]   adapter._atom_vocab   = {adapter._atom_vocab}")

    # Load weights into the velocity_field and pocket_encoder submodules
    # (the adapter is not an nn.Module itself; the submodules are).
    vf_result = adapter.velocity_field.load_state_dict(vf_state, strict=False)
    if vf_result.missing_keys:
        print(f"[smoke] velocity_field: {len(vf_result.missing_keys)} missing keys "
              f"(first 3: {vf_result.missing_keys[:3]})")
    if vf_result.unexpected_keys:
        print(f"[smoke] velocity_field: {len(vf_result.unexpected_keys)} unexpected keys "
              f"(first 3: {vf_result.unexpected_keys[:3]})")
    pe_result = adapter.pocket_encoder.load_state_dict(pe_state, strict=False)
    if pe_result.missing_keys:
        print(f"[smoke] pocket_encoder: {len(pe_result.missing_keys)} missing keys "
              f"(first 3: {pe_result.missing_keys[:3]})")
    if pe_result.unexpected_keys:
        print(f"[smoke] pocket_encoder: {len(pe_result.unexpected_keys)} unexpected keys "
              f"(first 3: {pe_result.unexpected_keys[:3]})")
    # Set submodules to eval mode (the adapter itself doesn't expose .eval())
    if hasattr(adapter.velocity_field, "eval"):
        adapter.velocity_field.eval()
    if hasattr(adapter.pocket_encoder, "eval"):
        adapter.pocket_encoder.eval()
    if hasattr(adapter, "bond_head") and adapter.bond_head is not None and hasattr(adapter.bond_head, "eval"):
        adapter.bond_head.eval()

    # Build a realistic-ish 8-atom pocket stand-in.  We use a non-zero
    # coordinate cloud with bond-length-scale distances so the
    # BondAwareDecoder's hard 2.4 Å cutoff has a fair chance of
    # finding pairs.  The smoke's primary goal is to verify the
    # wired BondAwareDecoder path FIRES (decode_ratio > 0).  Per
    # synthesis §2.1 honest caveat: on the existing h=128 5000-step
    # checkpoint, even an activated path may produce 0/8 if the
    # trained coord outputs are degenerate — that's a retrain gate,
    # not a wiring gate.
    from molmetal.domain import Pocket
    n_pocket_atoms = 8
    # Synthetic but realistic-ish bond-distance coords (carbon chain
    # at 1.5 Å increments along x, with some y/z jitter so the cloud
    # is not collinear).
    pocket_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.5, 0.0],
        [6.0, 0.0, 0.5],
        [7.5, -0.5, 0.0],
        [9.0, 0.0, -0.5],
        [10.5, 0.5, 0.0],
    ], dtype=torch.float32)
    pocket = Pocket(
        pdb_id="smoke_1h36",
        coords=pocket_coords,
        atom_types=torch.tensor([6, 6, 7, 8, 6, 7, 8, 6], dtype=torch.long),
        residue_ids=torch.arange(n_pocket_atoms),
        chain_ids=torch.zeros(n_pocket_atoms, dtype=torch.long),
        mask=torch.ones(n_pocket_atoms, dtype=torch.bool),
        center=torch.zeros(3),
        radius=12.0,
    )

    # Build the GenerationConfig.  The harness uses a 19-atom CNOF
    # ligand cardinality (per SizedGenerationConfig.n_atoms=19 in
    # r10_cfg_real_crossdocked.py:27); we mirror that here so the
    # trained h=128 checkpoint's coord outputs are not rejected at
    # the cardinality gate.
    from molmetal.ports import GenerationConfig

    config = GenerationConfig(
        n_samples=8,
        n_steps=200,
        seed=42,
    )
    # Patch n_atoms via the dataclass replace trick
    try:
        from dataclasses import replace
        config = replace(config, n_atoms=19)
    except (TypeError, ImportError):
        # If GenerationConfig is frozen and doesn't accept n_atoms, we
        # just fall back to the default.
        pass

    print(f"[smoke] generating 8 samples × 200 ODE steps...")
    t0 = time.monotonic()
    try:
        mols = adapter.generate(pocket, config)
    except Exception as exc:
        print(f"[smoke] generate() failed: {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.monotonic() - t0

    # Count decoded samples.  Per the synthesis pass criterion: at
    # least 1/8 sample must produce a non-empty bond graph AND a
    # SMILES that doesn't start with [DISCONNECTED.
    n_decoded = 0
    n_disconnected = 0
    n_empty_bonds = 0
    n_no_smiles = 0
    for i, m in enumerate(mols):
        bonds = m.bonds
        smiles = (m.smiles or "").strip()
        if bonds.shape[-1] == 0:
            n_empty_bonds += 1
        if smiles.startswith("[DISCONNECTED"):
            n_disconnected += 1
        if not smiles:
            n_no_smiles += 1
        if bonds.shape[-1] > 0 and smiles and not smiles.startswith("[DISCONNECTED"):
            n_decoded += 1
            print(f"[smoke]   sample {i}: bonds={bonds.shape[-1]} smiles={smiles[:60]}")
        else:
            print(f"[smoke]   sample {i}: bonds={bonds.shape[-1]} "
                  f"smiles={smiles[:60]!r}")

    print()
    print(f"[smoke] decode_ratio = {n_decoded}/8 = {n_decoded/8:.3f}")
    print(f"[smoke]   empty_bonds:    {n_empty_bonds}/8")
    print(f"[smoke]   no_smiles:      {n_no_smiles}/8")
    print(f"[smoke]   disconnected:   {n_disconnected}/8")
    print(f"[smoke] wall_time: {elapsed:.2f}s")
    print()
    if n_decoded > 0:
        print(f"[smoke] PASS: Fix #1 wired BondAwareDecoder path; "
              f"decode_ratio > 0 confirms the head is no longer frozen.")
        return 0
    else:
        print(f"[smoke] FAIL: decode_ratio == 0; the wired path did not fire.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())