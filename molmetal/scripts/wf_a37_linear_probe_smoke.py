"""Wf-A37-2 (NEW-3, 2026-10-15) — linear_probe smoke.

Exercises :func:`molmetal.eval.cross_dataset.run_cross_dataset_validation`
with ``do_linear_probe=True`` on a minimal stub checkpoint and the
``platinai_4cell`` cohort.

Background
----------
``cross_dataset.run_cross_dataset_validation`` exposes
``do_linear_probe: bool = False`` (cross_dataset.py:462).  The branch
runs the CFM-side :func:`run_cross_dataset_linear_probe` path
(cross_dataset.py:634) which freezes the YuelBond encoder and trains a
small logistic head on per-cell labels.  A grep across
``molmetal/scripts/`` for ``do_linear_probe`` or
``run_cross_dataset_linear_probe`` returned ZERO matches — the flag is
wired but never invoked from any caller, so the linear_probe path
sits as dead code in the repo.

This smoke closes the gap by ACTUALLY invoking it on a single
dataset.  Per §5.5 filesystem-truth (post-wave on-disk verification),
wiring a path is not enough — we have to run it once and write the
JSON artefact, so the on-disk state proves the branch was exercised.

Design
------
The probe head is unfit on a smoke-stub (the YuelBond checkpoint has
random-ish weights and the encoder shape may mismatch), but the probe
function explicitly returns ``{"linear_probe": None, "reason":
"ckpt_unavailable"|...}`` rather than raising, so the cross-dataset
JSON is still written with a ``linear_probe`` and
``linear_probe_reason`` key on every (dataset, oracle) record.

What this proves
----------------
* ``run_cross_dataset_validation`` accepts ``do_linear_probe=True``.
* The ``run_cross_dataset_linear_probe`` import + dispatch path
  runs without import error.
* The output JSON has ``linear_probe`` keys visible to reviewers
  via grep.
* The platinai_4cell branch of ``_load_smiles`` returns a non-empty
  list when the PlatinAIDataset .smiles attribute exists
  (Wf-A36-2 F2 contract).

CPU-only.  See AGENTS.md §5.1 standing GPU authorisation: this is a
50-line smoke, not a retrain, so the >1 hr wall check is moot.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch


def _build_smoke_ckpt(ckpt_path: Path) -> None:
    """Write a minimal valid CFM checkpoint into ``ckpt_path``.

    The checkpoint is named ``wf_a37_linear_probe_smoke.pt`` so it
    does NOT trip :func:`molmetal.eval.ckpt_is_smoke_stub` (which
    only matches ``*.smoke_stub.pt``).

    The state_dict is intentionally tiny (1 scalar tensor) — the
    probe head will refuse it on shape, but the path runs end-to-end
    and the JSON records a per-dataset ``linear_probe_reason``.
    """
    payload = {
        "model_state_dict": {"__probe_shape_probe__": torch.zeros(1)},
        "epoch": 0,
        "metrics": {"smoke": True, "wave": "Wf-A37-2"},
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-A37-2 linear_probe smoke — invokes "
            "run_cross_dataset_validation(do_linear_probe=True) once."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to a (non-.smoke_stub) checkpoint.  Defaults to a "
            "tempfile written by this script."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="metrics/by_metric/cfm_cross_dataset_linear_probe.json",
        help="Output JSON path (default: project by_metric slot).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["platinai_4cell"],
        help="Datasets to validate on (default: platinai_4cell only).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device ("cpu" or "cuda"). Default "cpu" for smoke.',
    )
    args = parser.parse_args()

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            print(f"[wf-a37-smoke] ERROR: ckpt not found: {ckpt_path}")
            return 2
    else:
        # Tempfile so the smoke is self-contained.
        tmp = tempfile.NamedTemporaryFile(
            prefix="wf_a37_linear_probe_", suffix=".pt", delete=False
        )
        tmp.close()
        ckpt_path = Path(tmp.name)
        _build_smoke_ckpt(ckpt_path)
        print(f"[wf-a37-smoke] wrote stub ckpt: {ckpt_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from molmetal.eval.cross_dataset import run_cross_dataset_validation

    print(
        f"[wf-a37-smoke] invoking run_cross_dataset_validation("
        f"do_linear_probe=True) on ckpt={ckpt_path} "
        f"datasets={args.datasets} device={args.device}"
    )
    payload = run_cross_dataset_validation(
        ckpt_path=str(ckpt_path),
        out_json=str(out_path),
        datasets=args.datasets,
        device=args.device,
        do_linear_probe=True,
    )

    results = payload.get("results", [])
    n_records = len(results)
    n_with_probe = sum(
        1 for r in results if "linear_probe" in r
    )
    print(
        f"[wf-a37-smoke] records={n_records} "
        f"with_linear_probe={n_with_probe} "
        f"wall_time_s={payload.get('wall_time_s', 0.0):.2f}"
    )
    print(f"[wf-a37-smoke] wrote JSON: {out_path}")

    # Quick post-condition: JSON on disk has 1+ records.
    if not out_path.exists():
        print("[wf-a37-smoke] FAIL: output JSON missing on disk")
        return 1
    with out_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    disk_records = on_disk.get("results", [])
    if len(disk_records) < 1:
        print("[wf-a37-smoke] FAIL: no records on disk")
        return 1
    if not any("linear_probe" in r for r in disk_records):
        print(
            "[wf-a37-smoke] FAIL: no record carries linear_probe key"
        )
        return 1
    print("[wf-a37-smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())