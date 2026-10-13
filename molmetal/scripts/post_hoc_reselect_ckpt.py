"""Post-hoc re-selection of the gate-selected checkpoint (Wf-A30-4 D2 helper).

Background
----------
``train_cfm_yuelbond_wasserstein.py`` writes TWO checkpoint paths per seed:

    cfm_yuelbond_wasserstein_seed{N}.pt        # the gate-selected one (overwrites on improvement)
    cfm_yuelbond_wasserstein_seed{N}_last.pt   # the periodic save (distinct filename)

Pre-Wf-A30-1 historical runs recorded the gate-selected epoch in
``per_epoch_metrics.json::best.epoch`` while the surviving ``.pt`` held the FINAL
epoch's weights — the gate picked an epoch and the last save then replaced those
weights.  A30-1 fixes this by routing the periodic save to ``..._last.pt``; this
helper is the post-hoc remedy for any historical run whose selected checkpoint is
already lost.

What it does
------------
Given a run directory containing ``per_epoch_metrics.json`` + ``checkpoints/``:

1. Reads ``gate_metric`` + ``best.epoch`` from the JSON.
2. Lists every ``.pt`` in ``checkpoints/`` and locates the one whose embedded
   ``epoch`` matches ``best.epoch`` (we read the header with ``torch.load`` on
   ``map_location="cpu"`` so this works without a GPU).
3. Copies that file to ``checkpoints/<stem>_best_epoch{N}.pt``.

If no checkpoint matches (typical: every surviving ``.pt`` is the FINAL epoch,
not the gate-selected one), the script prints a one-line honest-negative and
exits 0 — the user has just demonstrated that the selected epoch is unrecoverable
from this run directory and must be re-trained.

The actual post-hoc reselect of the 2 affected pre-A30-1 waves (the ones
whose ``.pt`` files were renamed ``*.smoke_stub.pt``) is DEFERRED per the Wf-A30-4
brief — this helper is shipped so the reselect is a one-liner when the follow-up
is authorised.  See HONEST_NEGATIVE_CATALOG.md §6.52e.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _list_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(p for p in ckpt_dir.glob("*.pt"))


def reselect(run_dir: Path) -> int:
    """Post-hoc re-select the gate-selected checkpoint for a single run dir.

    Returns 0 on success, 1 on honest-negative (no matching epoch found), 2
    on missing inputs (no JSON / no checkpoints dir).
    """
    run_dir = run_dir.resolve()
    pem_path = run_dir / "per_epoch_metrics.json"
    ckpt_dir = run_dir / "checkpoints"
    if not pem_path.is_file():
        print(f"[post-hoc-reselect] NO per_epoch_metrics.json at {pem_path}", flush=True)
        return 2
    if not ckpt_dir.is_dir():
        print(f"[post-hoc-reselect] NO checkpoints/ dir at {run_dir}", flush=True)
        return 2

    data = _load_json(pem_path)
    best = data.get("best", {}) or {}
    best_epoch = int(best.get("epoch", -1))
    gate_metric = str(best.get("gate_metric", "?"))
    if best_epoch < 0:
        print(
            f"[post-hoc-reselect] no best.epoch in {pem_path.name} "
            f"(early stop did not trigger); nothing to do",
            flush=True,
        )
        return 0

    import torch  # deferred: this helper is smoke-imported in unit tests

    matched: list[tuple[Path, int]] = []
    for p in _list_ckpts(ckpt_dir):
        # weights_only=False is required to read pre-Wf-A30-1 trainer
        # checkpoints whose state dict carries non-Tensor entries (e.g.
        # ``args`` as an argparse Namespace, ``pocket_id`` as a string, and
        # the older pickle protocol used by ``torch.save``).  The post-hoc
        # use-case is RECOVERING those exact blobs, so we cannot be
        # pickier than the trainer itself was at write-time.  Per
        # AGENTS.md §5.5 filesystem-truth, we only read what the trainer
        # wrote — there is no untrusted code path here.
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            # Pre-2.x torch has no weights_only kwarg.
            blob = torch.load(p, map_location="cpu")
        except Exception as exc:
            print(f"[post-hoc-reselect] skip {p.name}: {type(exc).__name__}", flush=True)
            continue
        epoch_in_blob = int(blob.get("epoch", -1)) if isinstance(blob, dict) else -1
        if epoch_in_blob == best_epoch:
            matched.append((p, epoch_in_blob))

    if not matched:
        print(
            f"[post-hoc-reselect] HONEST-NEGATIVE: best.epoch={best_epoch} "
            f"(gate={gate_metric}) but no surviving .pt holds that epoch in "
            f"{ckpt_dir}. The selected weights are unrecoverable from this run "
            f"directory; re-training is required. See "
            f"HONEST_NEGATIVE_CATALOG.md §6.52e.",
            flush=True,
        )
        return 1

    for src, _ in matched:
        dst = src.with_name(f"{src.stem}_best_epoch{best_epoch}.pt")
        shutil.copy2(src, dst)
        print(
            f"[post-hoc-reselect] {src.name} (epoch={best_epoch}, "
            f"gate={gate_metric}) -> {dst.name}",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: post_hoc_reselect_ckpt.py RUN_DIR [RUN_DIR ...]",
            file=sys.stderr,
        )
        return 2
    rc = 0
    for raw in argv:
        rc = max(rc, reselect(Path(raw)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
