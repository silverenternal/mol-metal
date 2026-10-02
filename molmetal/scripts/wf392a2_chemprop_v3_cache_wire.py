"""Wf-392-A2 — chemprop v3 featurization cache wire-in scaffold.

TODO cross-ref: TODO-155 §3 (chemprop v2 cache pivot) +
``molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py``
(Wf-382-A2 SCAFFOLD — the cache itself, 467 LOC) + Wf-328-W10
(D-MPNN on-disk pickle cache pattern) + Wf-320D
(``molmetal/scripts/wf320d_featurization_fix.py`` chemprop v2
``PerEpochFeatureCache``).

This module is the **trainer wire-in harness** for Wf-382-A2's
``ChempropV3FeaturizationCache``. It does NOT replace the trainer's
existing ``_batch_molgraphs_to(bmols, device)`` path; it adds a
``--chemprop-v3-cache`` CLI flag (default OFF, preserves bit-for-bit
backward compat) and a small helper block. Full integration into the
training loop (replacing ``_batch_molgraphs_to`` calls) is a follow-up
wave — see ``TODO/completed/392a2_chemprop_v3_cache_wire.md`` §4.

Honest framing per ``party-twenty-spirit`` §3.5
-----------------------------------------------
- This module is CPU-only engineering. No GPU launches.
- The cache is **stateless** with respect to chemprop v3 model
  parameters (per ``wf382a2_chemprop_v3_featurization_cache.py`` §0)
  so the per-batch tier is safe to reuse across epochs.
- The wire-in adds the flag **but does not modify training loop** —
  full integration requires replacing ``_batch_molgraphs_to`` with
  ``cache.get_or_build_batch(...)`` calls at lines 751 + 1031 of
  ``train_chemprop_v2_v3.py`` (the two existing per-batch assembly
  points). This is a SCAFFOLD only; the flag is wired + smoke-tested
  but end-to-end training-time cache use requires the follow-up wave.
- Default OFF — pre-existing trainer behaviour is bit-for-bit
  preserved when the flag is not passed.
- Cache savings PROJECTED 5-10× on featurisation speed on cross-
  epoch reuse (per Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline for
  the chemprop v2 + D-MPNN caches; chemprop v3 cache uses the same
  two-tier pattern).

CLI smoke::

    python molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py \\
        --wire-trainer \\
        molmetal/scripts/train_chemprop_v2_v3.py
    python molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py \\
        --smoke-cache-hit-rate \\
        --cohort-size 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Repo-root importability.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation
# closes the half-merged-file failure mode for the trainer backup + JSON
# Write sites (per AGENTS.md §5.5/§5.6 standing directives). The shared
# atomic_write_file primitive at molmetal/scripts/wf386a3_atomic_write.py
# is the canonical I/O substrate for all closeout-drafting scripts
# (Wf-386-A3 §6.1 adoption sequence step 4).
try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF414_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF414_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants — anchor markers used to detect prior wiring.
# ---------------------------------------------------------------------------
WIRED_MARKER: str = "# Wf-392-A2 wired: --chemprop-v3-cache CLI flag inserted."
BACKUP_SUFFIX: str = ".bak.wf392a2"

# The two regex-free anchor markers used to locate the insert position
# inside the trainer file. Both markers exist verbatim in the trainer
# (verified at Wf-392-A2 author time).
ARGPARSE_END_MARKER: str = (
    'p.add_argument(\n'
    '        "--allow-single-seed", action="store_true",\n'
    '        help=(\n'
    '            "Bypass the --n-seeds-min check (for smoke tests; not for "\n'
    '            "paper claims).  Wf-245 Fix."\n'
    '        ),\n'
    '    )\n'
    '    args = p.parse_args(argv)\n'
)

ARGPARSE_END_MARKER_INSERT: str = (
    'p.add_argument(\n'
    '        "--allow-single-seed", action="store_true",\n'
    '        help=(\n'
    '            "Bypass the --n-seeds-min check (for smoke tests; not for "\n'
    '            "paper claims).  Wf-245 Fix."\n'
    '        ),\n'
    '    )\n'
    '    # Wf-392-A2 wired: --chemprop-v3-cache CLI flag inserted.\n'
    '    p.add_argument(\n'
    '        "--chemprop-v3-cache", action="store_true", default=False,\n'
    '        help=(\n'
    '            "Enable chemprop v3 featurization cache (default: OFF; "\n'
    '            "preserves bit-for-bit backward compat). When ON, the "\n'
    '            "trainer initializes a ChempropV3FeaturizationCache "\n'
    '            "and uses it for per-SMILES + per-batch featurization "\n'
    '            "reuse across epochs. SCAFFOLD: full training-loop "\n'
    '            "wire-in is a follow-up wave; this flag is wired + "\n'
    '            "smoke-tested only."\n'
    '        ),\n'
    '    )\n'
    '    args = p.parse_args(argv)\n'
)


# ---------------------------------------------------------------------------
# wire_cache_to_trainer — idempotent trainer patch.
# ---------------------------------------------------------------------------
def wire_cache_to_trainer(trainer_path: str) -> bool:
    """Add the ``--chemprop-v3-cache`` CLI flag to ``trainer_path``.

    Args:
        trainer_path: path to ``train_chemprop_v2_v3.py``.

    Returns:
        True if a NEW wire-in was performed; False if the trainer
        was already wired or the trainer path could not be found.

    Side effects:
        - On first wire: writes ``{trainer_path}.bak.wf392a2``
          containing the original (un-wired) trainer content.
        - Idempotent: subsequent calls detect the WIRED_MARKER in
          the trainer file and skip (no-op).
    """
    trainer_file = Path(trainer_path)
    if not trainer_file.is_file():
        return False
    content = trainer_file.read_text()
    if WIRED_MARKER in content:
        return False

    backup_path = trainer_file.with_suffix(trainer_file.suffix + BACKUP_SUFFIX)
    if not backup_path.exists():
        # Wf-414-A3 migration: backup file Write site → atomic_write_file.
        if _WF414_HAS_ATOMIC:
            atomic_write_file(backup_path, content, encoding="utf-8")
        else:  # pragma: no cover
            backup_path.write_text(content)

    if ARGPARSE_END_MARKER not in content:
        raise RuntimeError(
            f"Cannot find argparser-end anchor in {trainer_path}. "
            "Trainer file may have been edited since Wf-392-A2 author time. "
            "Please re-author the wire-in marker or update the anchor."
        )

    new_content = content.replace(ARGPARSE_END_MARKER, ARGPARSE_END_MARKER_INSERT, 1)
    # Wf-414-A3 migration: trainer edit Write site → atomic_write_file.
    if _WF414_HAS_ATOMIC:
        atomic_write_file(trainer_file, new_content, encoding="utf-8")
    else:  # pragma: no cover
        trainer_file.write_text(new_content)
    return True


# ---------------------------------------------------------------------------
# smoke_cache_hit_rate — CPU-only smoke that exercises the cache.
# ---------------------------------------------------------------------------
def smoke_cache_hit_rate(
    trainer_path: Optional[str] = None,
    cohort_size: int = 10,
    epochs: int = 2,
    batches_per_epoch: int = 5,
    batch_size: int = 4,
) -> Dict[str, object]:
    """Exercise the Wf-382-A2 cache on a synthetic cohort.

    Args:
        trainer_path: optional path to trainer (NOT invoked — only
            verified to exist so the smoke confirms the wire-in
            target file is reachable).
        cohort_size: number of synthetic SMILES in the cohort
            (default 10).
        epochs: number of synthetic epochs to simulate (default 2).
        batches_per_epoch: number of batches per epoch (default 5).
        batch_size: SMILES per batch (default 4).

    Returns:
        Dict with keys: ``cache_info``, ``stats``, ``hit_rate``,
        ``build_wall_s``, ``lookup_wall_s``, ``cohort_size``,
        ``epochs``, ``batches_per_epoch``, ``batch_size``,
        ``trainer_path``, ``trainer_exists``.

    Note:
        This is a synthetic smoke — the per-batch tier may raise on
        a host that doesn't have ``chemprop`` installed. We catch
        ImportError and degrade gracefully (the tier-1 SMILES cache
        still completes, hit_rate reports 0.0 because tier-2 cannot
        be exercised). Requires rdkit (chemprop v3's featurizer uses
        rdkit.Chem.MolFromSmiles for canonicalisation).
    """
    from molmetal.scripts.wf382a2_chemprop_v3_featurization_cache import (
        ChempropV3FeaturizationCache,
    )

    # Synthetic SMILES cohort (canonical, diverse drug-like).
    synthetic = [
        "CCO", "c1ccccc1", "CC(=O)O", "CN", "CCN", "CC(C)O",
        "C1CCCCC1", "c1ccc(O)cc1", "CC(=O)N", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    cohort = synthetic[: int(cohort_size)]

    cache = ChempropV3FeaturizationCache(use_fast_featurizer=False)

    # Tier 1: pre-featurise cohort exactly once.
    n_added = cache.precompute_smiles(cohort)

    tier2_exercised = False
    tier2_error: Optional[str] = None
    try:
        from chemprop.data.molgraph import BatchMolGraph  # type: ignore
        for ep in range(int(epochs)):
            cache.begin_epoch(ep)
            for bi in range(int(batches_per_epoch)):
                batch = [
                    cohort[(bi * int(batch_size) + j) % len(cohort)]
                    for j in range(int(batch_size))
                ]
                cache.get_or_build_batch(ep, bi, batch, "cpu", BatchMolGraph)
        tier2_exercised = True
    except ImportError as e:
        tier2_error = str(e)
    except Exception as e:  # noqa: BLE001
        tier2_error = f"unexpected: {type(e).__name__}: {e}"

    info = cache.cache_info()
    trainer_exists = (
        Path(trainer_path).is_file() if trainer_path is not None else None
    )
    return {
        "cache_info": info,
        "hit_rate": info["stats"]["hit_rate"],
        "build_wall_s": info["stats"]["build_wall_s"],
        "lookup_wall_s": info["stats"]["lookup_wall_s"],
        "n_added": n_added,
        "cohort_size": int(cohort_size),
        "epochs": int(epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "tier2_exercised": tier2_exercised,
        "tier2_error": tier2_error,
        "trainer_path": trainer_path,
        "trainer_exists": trainer_exists,
    }


# ---------------------------------------------------------------------------
# CLI runner.
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wire-trainer",
        metavar="PATH",
        help="Path to trainer script to wire with --chemprop-v3-cache CLI flag.",
    )
    p.add_argument(
        "--smoke-cache-hit-rate",
        action="store_true",
        help="Exercise the Wf-382-A2 cache on a synthetic cohort.",
    )
    p.add_argument(
        "--cohort-size", type=int, default=10,
        help="Synthetic cohort size for --smoke-cache-hit-rate (default 10).",
    )
    p.add_argument(
        "--epochs", type=int, default=2,
        help="Synthetic epochs for --smoke-cache-hit-rate (default 2).",
    )
    p.add_argument(
        "--batches-per-epoch", type=int, default=5,
        help="Synthetic batches per epoch (default 5).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="SMILES per batch (default 4).",
    )
    p.add_argument(
        "--out", default="/tmp/wf392a2_chemprop_v3_cache_wire_smoke.json",
        help="Output JSON path for --smoke-cache-hit-rate.",
    )
    args = p.parse_args(argv)

    if not args.wire_trainer and not args.smoke_cache_hit_rate:
        p.error("Pass --wire-trainer PATH and/or --smoke-cache-hit-rate.")

    exit_code = 0
    out: Dict[str, object] = {"actions": []}

    if args.wire_trainer:
        wired = wire_cache_to_trainer(args.wire_trainer)
        action = {
            "action": "wire_trainer",
            "trainer_path": args.wire_trainer,
            "wired": wired,
        }
        out["actions"].append(action)
        print(f"wire_trainer: {args.wire_trainer} wired={wired}")
        if not wired:
            # Already wired OR path missing.
            if Path(args.wire_trainer).is_file():
                print("  → already wired (WIRED_MARKER present); no-op.")
            else:
                print("  → trainer path not found; no-op.")
                exit_code = 1

    if args.smoke_cache_hit_rate:
        t0 = time.time()
        result = smoke_cache_hit_rate(
            trainer_path=args.wire_trainer,
            cohort_size=args.cohort_size,
            epochs=args.epochs,
            batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size,
        )
        result["smoke_wall_s"] = time.time() - t0
        out["actions"].append({"action": "smoke_cache_hit_rate", **result})
        print(
            f"smoke_cache_hit_rate: hit_rate={result['hit_rate']:.2%} "
            f"n_added={result['n_added']} "
            f"tier2_exercised={result['tier2_exercised']}"
        )
        if result["tier2_error"]:
            print(f"  tier2_error: {result['tier2_error']}")
        print(
            f"  build_wall_s={result['build_wall_s']:.4f} "
            f"lookup_wall_s={result['lookup_wall_s']:.4f} "
            f"smoke_wall_s={result['smoke_wall_s']:.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Wf-414-A3 migration: smoke report Write site → atomic_write_file +
    # safe_overwrite_check + detect_truncation.
    _payload = json.dumps(out, indent=2, default=str)
    if _WF414_HAS_ATOMIC:
        _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
        if _risk is not None:
            print(
                f"[Wf-392-A2] WARN: safe_overwrite_check ratio={_risk:.3f} at {out_path}",
                file=sys.stderr,
            )
        atomic_write_file(out_path, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            print(
                f"[Wf-392-A2] WARN: detect_truncation reason={_trunc!r} at {out_path}",
                file=sys.stderr,
            )
    else:  # pragma: no cover
        out_path.write_text(_payload, encoding="utf-8")
    print(f"Output written to: {out_path}")
    return exit_code


__all__ = [
    "WIRED_MARKER",
    "BACKUP_SUFFIX",
    "wire_cache_to_trainer",
    "smoke_cache_hit_rate",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())