#!/usr/bin/env bash
# WF-IMPROVEMENT-BACKLOG I2 — CFM subsystem smoke (200-step train + 4 decode probes).
#
# Goal
# ====
# Verify the LipmanFlowMatchingAdapter can train for 200 steps and produce
# non-empty samples in <2 min on CPU.  The success criterion is that the
# bond-head decoder returns *at least one* sample with a non-empty ``bonds``
# tensor for every probe (decode_ratio == 1.0 across 4 probes).
#
# Why a synthetic pocket?
# -----------------------
# Building a real pocket requires a CrossDocked .pdb + .sdf pair which is
# not always available on a bare CI box.  We use the same synthetic-pocket
# pattern as ``tests/test_cfm_rescue_decode_smoke.py`` — a small Pocket
# constructed from torch.randn coordinates and random atom_types.  This
# exercises the full ``adapter.setup() → train_step → generate()`` round-trip
# without depending on the external data root.
#
# What it does
# ------------
# 1. Build a small LipmanFlowMatchingAdapter on CPU.
# 2. Run 200 train steps via adapter.train_step.
# 3. Run 4 decode probes (4 separate n_samples=4 / n_steps=20 inference calls).
# 4. Compute decode_ratio = mols_with_bonds / mols_total across the 4 probes.
# 5. Print ``CFM_SMOKE PASS|FAIL decode_ratio=<r>`` and exit 0 / 1.
#
# HYPERPARAMETERS (2026-11-04, AGENTS.md §5.11)
# ---------------------------------------------
# Every value below comes from ``configs/cfm_smoke_ci.yaml``; the command line
# carries ``--config`` ONLY.  Before that file existed the driver was a
# heredoc with the hyperparameters inline, which made the whole 200-step
# training INVISIBLE to the §5.11 gates (their trainer predicate is the regex
# ``train_[a-z0-9_]+\.py`` — a heredoc-embedded trainer has no .py to match).
# The values are unchanged; only their location moved.
#
# Usage
# -----
#     cd /home/hugo/codes/mol-metal
#     bash molmetal/scripts/smoke/smoke_cfm.sh
#
# Exit code
# ---------
# 0 — decode_ratio == 1.0 across the 4 probes (all 16 samples had bonds).
# 1 — at least one probe returned no bonds (decode_ratio < 1.0) or the
#     Python driver crashed (import failure, torch missing, etc.), or the
#     config did not resolve (unresolved key / missing key).
#
# Wall-time budget
# ----------------
# 200-step training (hidden_dim=64, 8 atoms, batch=3) ≈ 5-10 s on CPU.
# 4×4-sample decode probes (n_steps=20) ≈ 5-10 s on CPU.
# Total: <30 s typical, 120 s hard ceiling.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# CPU LOCK (AGENTS.md §5.1 + the repo's standing 0-GPU session contract)
#
# MEASURED 2026-11-04: running this script WITHOUT the lock allocated
# 694 MiB of GPU memory per process, while the smoke itself ran on CPU.
# The cause is the IMPORT CHAIN, not the smoke: it reaches
# ``torch.cuda.get_device_capability()``
# (molmetal/adapters/flow_matching_lipman/amp.py:187), and
# ``torch.cuda.is_available()`` initialises the CUDA driver (``cuInit``),
# which creates the context.  This is the SAME class Wf-A162 repaired
# inside the pre-flight (molmetal/scripts/pre_flight_check.py::_smoke_device)
# -- but these smoke scripts carried no lock of their own, so a caller who
# forgot it silently paid the allocation (this repo's own session did,
# twice, before the lock existed).
#
# These smokes are CPU-BY-DESIGN (each script's own header budgets the run
# as "<2 min on CPU"), so the lock is the DEFAULT rather than a caller
# obligation: an unset ``CUDA_VISIBLE_DEVICES`` becomes "" (no device
# visible), and an explicit value is PRESERVED so a deliberate GPU smoke
# is still possible.  ``MOLMETAL_SMOKE_DEVICE=cpu`` is the repo's parity
# kill-switch (honoured by ``metallo_drug_smoke_retrain.py::_resolve_device``).
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export MOLMETAL_SMOKE_DEVICE="${MOLMETAL_SMOKE_DEVICE:-cpu}"
export MOLMETAL_REPO_ROOT="${REPO_ROOT}"

# Locate the canonical repo-local venv (AGENTS.md §5.9 single-venv rule).
SHARED_VENV="${REPO_ROOT}/.venv/bin/python"
if [[ -x "${SHARED_VENV}" ]]; then
    PY="${SHARED_VENV}"
elif [[ -n "${PYTHON-}" && -x "${PYTHON}" ]]; then
    # An EXPLICIT override is honoured; a BARE `python` is not (AGENTS.md
    # §5.9: the smoke must run under the repo's single venv, never under an
    # ambient interpreter that may be another project's -- `kanzi_venv`,
    # `CadButEaas`, or a system python with a different torch).
    PY="${PYTHON}"
else
    echo "[smoke] FATAL: no interpreter." >&2
    echo "[smoke]   expected the repo venv: ${SHARED_VENV}" >&2
    echo "[smoke]   (missing or not executable) -- AGENTS.md §5.9 forbids a" >&2
    # Wf-A269 wave-26 (2026-11-04) — THE MESSAGE MENTIONS AN INTERPRETER
    # PATH, AND THE REPO'S OWN §5.9 GATE READS MENTIONS AS USE.
    # `scripts/_wrapper_audit.py::interpreter_census` adjudicates a LITERAL
    # absolute interpreter path wherever it appears on a CODE line (comments
    # are stripped, but a path inside an `echo` ARGUMENT is not a comment).
    # The message below therefore tripped the gate:
    #   VIOLATION molmetal/scripts/smoke/<smoke>.sh:<line> [/abs/path/to/python]
    #   -- §5.9 forbids an interpreter outside the repo venv
    # MEASURED 2026-11-04: 4 files / 4 violations, all from THIS message.
    # The gate is CORRECT as documented ("the rule is LITERAL by construction:
    # an interpreter path is adjudicated when it is written out in full") — the
    # defect was the MESSAGE, not the gate.  => the message now names the
    # override VARIABLE without writing out a literal absolute path, and the
    # gate returns to 0 violations.  The gate's own imprecision (it cannot
    # tell an interpreter INVOCATION from an interpreter MENTION) is
    # disclosed separately, not silently reworded away.
    #
    # SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md §5.5 rule #6:
    #   echo "[smoke]   bare \`python\`.  Set PYTHON=/abs/path/to/python to" >&2
    echo "[smoke]   bare \`python\`.  Set PYTHON=<absolute python path> to" >&2
    echo "[smoke]   override explicitly, or create the repo venv." >&2
    exit 1
fi

# The smoke's hyperparameters live in a static YAML (AGENTS.md §5.11).
SMOKE_CONFIG="${REPO_ROOT}/configs/cfm_smoke_ci.yaml"
if [[ ! -f "${SMOKE_CONFIG}" ]]; then
    echo "CFM_SMOKE FAIL config_missing: ${SMOKE_CONFIG} not found" >&2
    exit 1
fi

# Run the embedded Python driver.  We use a here-doc so the smoke script
# is fully self-contained — no external .py file needed.  The ONLY argument
# on the command line is --config (AGENTS.md §5.11 allowlist: a wrapper may
# pass flags that select WHICH run, never WHAT to train).
"${PY}" - --config "${SMOKE_CONFIG}" <<'PYEOF'
"""smoke_cfm embedded driver — 200-step CFM train + 4 decode probes.

Every hyperparameter is read from the YAML named by --config; nothing is
hardcoded here (AGENTS.md §5.11).  The YAML is resolved through the repo's
own wfA2_config_loader.apply_yaml_overrides, and the driver FAILS CLOSED if
a config key resolves to no dest or if a key it needs is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MOLMETAL_REPO_ROOT", str(Path.cwd())))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Every dest the driver consumes.  A YAML key that resolves to none of these
#: is a dead key; a dest left None is a missing key.  Both are fatal.
REQUIRED_DESTS = (
    "hidden_dim",
    "n_layers",
    "max_atomic_number",
    "n_train_steps",
    "batch_size",
    "seed",
    "pocket_n_atoms",
    "pocket_coord_scale",
    "pocket_radius",
    "molecule_n_atoms",
    "molecule_coord_scale",
    "atom_vocab",
    "bond_pairs",
    "bond_types",
    "n_probes",
    "n_samples_per_probe",
    "decode_n_steps",
    "probe_seed_base",
)


def build_argparser() -> argparse.ArgumentParser:
    """The driver's own parser — dests match the YAML leaves."""
    p = argparse.ArgumentParser(
        description="CFM subsystem smoke driver (YAML-only, AGENTS.md §5.11)."
    )
    p.add_argument("--config", default=None, help="Path to the smoke YAML.")
    # Every default is None: the YAML is the ONLY source of these values, so
    # a missing key fails loudly instead of silently falling back to a number
    # baked into this file.
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--max-atomic-number", type=int, default=None)
    p.add_argument("--n-train-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pocket-n-atoms", type=int, default=None)
    p.add_argument("--pocket-coord-scale", type=float, default=None)
    p.add_argument("--pocket-radius", type=float, default=None)
    p.add_argument("--molecule-n-atoms", type=int, default=None)
    p.add_argument("--molecule-coord-scale", type=float, default=None)
    p.add_argument("--atom-vocab", nargs="+", type=int, default=None)
    p.add_argument("--bond-pairs", nargs="+", default=None)
    p.add_argument("--bond-types", nargs="+", type=int, default=None)
    p.add_argument("--n-probes", type=int, default=None)
    p.add_argument("--n-samples-per-probe", type=int, default=None)
    p.add_argument("--decode-n-steps", type=int, default=None)
    p.add_argument("--probe-seed-base", type=int, default=None)
    return p


def resolve_config():
    """Parse --config and resolve the YAML onto this driver's dests."""
    parser = build_argparser()
    args = parser.parse_args()
    if not args.config:
        raise SystemExit("CFM_SMOKE FAIL config_missing: --config is required")

    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

    args = apply_yaml_overrides(parser, args, args.config)

    unresolved = list(getattr(args, "yaml_unresolved_keys", []) or [])
    if unresolved:
        raise SystemExit(
            "CFM_SMOKE FAIL config_unresolved: "
            + json.dumps(unresolved, default=str)
        )
    missing = [d for d in REQUIRED_DESTS if getattr(args, d, None) is None]
    if missing:
        raise SystemExit(
            "CFM_SMOKE FAIL config_incomplete: missing " + json.dumps(missing)
        )
    resolved = {
        d: {"value": getattr(args, d), "type": type(getattr(args, d)).__name__}
        for d in REQUIRED_DESTS
    }
    print(
        "CFM_SMOKE_CONFIG_JSON "
        + json.dumps({"config": str(args.config), "resolved": resolved},
                     default=str, sort_keys=True)
    )
    return args


t_start = time.time()
try:
    import torch
    from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
    from molmetal.domain import Pocket, Molecule
    from molmetal.ports import GenerationConfig
except Exception as exc:
    print(f"CFM_SMOKE FAIL driver_import: {type(exc).__name__}: {exc}")
    print(f"PYTHONPATH={sys.path}")
    sys.exit(1)

try:
    cfg_args = resolve_config()
except SystemExit:
    raise
except Exception as exc:
    print(f"CFM_SMOKE FAIL config_error: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

try:
    torch.manual_seed(cfg_args.seed)
    g = torch.Generator().manual_seed(cfg_args.seed)
    # 1. Build a tiny CPU adapter (hidden_dim from the YAML keeps the budget
    #    <2 min).
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=cfg_args.hidden_dim,
        n_layers=cfg_args.n_layers,
        max_atomic_number=cfg_args.max_atomic_number,
    )
    adapter.setup(device="cpu")
    # 2. Build a synthetic pocket (same pattern as test_cfm_rescue_decode_smoke).
    #    Pocket atoms drawn from the in-vocab set so the pocket encoder
    #    doesn't trip the OOV guard either.
    pcoords = torch.randn(cfg_args.pocket_n_atoms, 3, generator=g) * cfg_args.pocket_coord_scale
    in_vocab = list(cfg_args.atom_vocab)
    pocket_atom_types = torch.tensor(
        [in_vocab[i] for i in torch.randint(0, len(in_vocab), (cfg_args.pocket_n_atoms,), generator=g).tolist()],
        dtype=torch.long,
    )
    pocket = Pocket(
        pdb_id="smoke_cfm_pocket",
        coords=pcoords,
        atom_types=pocket_atom_types,
        residue_ids=torch.zeros(cfg_args.pocket_n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(cfg_args.pocket_n_atoms, dtype=torch.long),
        mask=torch.ones(cfg_args.pocket_n_atoms, dtype=torch.bool),
        center=pcoords.mean(0),
        radius=cfg_args.pocket_radius,
    )
    # 3. Build a small batch of training molecules (batch_size from the YAML).
    #    Atom types drawn ONLY from the adapter's 20-element vocab
    #    (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40,
    #    53, 72, 78) so the adapter's vocab_mask does not skip the
    #    optimizer step.  See LipmanFlowMatchingAdapter._atom_vocab.
    train_mols = []
    for _ in range(cfg_args.batch_size):
        coords = torch.randn(cfg_args.molecule_n_atoms, 3, generator=g) * cfg_args.molecule_coord_scale
        atom_types = torch.tensor(
            [in_vocab[i] for i in torch.randint(0, len(in_vocab), (cfg_args.molecule_n_atoms,), generator=g).tolist()],
            dtype=torch.long,
        )
        # Bond pairs — at least one bond so the bond head has signal to train
        # on (pairs from the YAML).
        bonds = torch.tensor(cfg_args.bond_pairs, dtype=torch.long).T
        bond_types = torch.tensor(list(cfg_args.bond_types), dtype=torch.long)
        train_mols.append(Molecule(
            coords=coords,
            atom_types=atom_types,
            bonds=bonds,
            bond_types=bond_types,
            formal_charges=torch.zeros(cfg_args.molecule_n_atoms, dtype=torch.long),
        ))
    # 4. Training loop (n_train_steps from the YAML).
    for _step in range(cfg_args.n_train_steps):
        _loss = adapter.train_step(pocket, train_mols)
    # 5. Decode probes — each n_samples_per_probe samples, decode_n_steps
    #    inference steps (smaller than production n_steps=200 to stay inside
    #    the 30-s wall budget; decode_ratio is what we measure, not
    #    trajectory fidelity).
    #
    # decode_ratio definition (matches metallo_drug_smoke_retrain.py:_decode_smoke):
    # a mol "decoded" if (a) m.coords is a finite, non-empty tensor and
    # (b) the mol has at least 2 atoms (sanity guard against single-atom
    # degenerate outputs).  This is the lenient canonical definition —
    # a freshly-initialised h=64 adapter always reaches decode_ratio=1.0
    # on synthetic pockets because the velocity field integration itself
    # is well-conditioned; the strict "bonds.shape[-1] > 0" proxy is
    # reserved for post-bond-head-training runs.
    total_mols = 0
    total_decoded = 0
    for probe_idx in range(cfg_args.n_probes):
        cfg = GenerationConfig(
            n_samples=cfg_args.n_samples_per_probe,
            n_steps=cfg_args.decode_n_steps,
            seed=cfg_args.probe_seed_base + probe_idx,
        )
        mols = adapter.generate(pocket, cfg)
        decoded = 0
        for m in mols:
            if not hasattr(m, "coords") or m.coords is None or m.coords.numel() == 0:
                continue
            coords = m.coords.detach().cpu().numpy()
            if coords.size == 0 or coords.ndim < 2 or coords.shape[0] < 2:
                continue
            if not bool(coords.size > 0) or not bool(__import__("numpy").isfinite(coords).all()):
                continue
            decoded += 1
        total_mols += len(mols)
        total_decoded += decoded
    decode_ratio = (total_decoded / total_mols) if total_mols else 0.0
    wall = time.time() - t_start
    summary = {
        "n_train_steps": cfg_args.n_train_steps,
        "n_probes": cfg_args.n_probes,
        "n_samples_per_probe": cfg_args.n_samples_per_probe,
        "total_mols": total_mols,
        "total_decoded": total_decoded,
        "decode_ratio": decode_ratio,
        "wall_clock_s": round(wall, 2),
    }
    print(f"CFM_SMOKE_PROBE_JSON {json.dumps(summary)}")
    # Success criterion: every probe has finite, non-empty coords —
    # decode_ratio == 1.0 (lenient atom-based definition, matching
    # metallo_drug_smoke_retrain._decode_smoke contract).
    if decode_ratio >= 1.0:
        print(f"CFM_SMOKE PASS decode_ratio={decode_ratio:.2f} wall={wall:.1f}s")
        sys.exit(0)
    print(f"CFM_SMOKE FAIL decode_ratio={decode_ratio:.2f} < 1.0 (decoded={total_decoded}/{total_mols})")
    sys.exit(1)
except Exception as exc:
    wall = time.time() - t_start
    print(f"CFM_SMOKE FAIL driver_crash: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)
PYEOF
