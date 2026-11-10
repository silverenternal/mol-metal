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

1. Reads ``gate_metric`` + ``best.epoch`` from the JSON -- but ONLY after the
   record has been ATTESTED as the run of record's own file (the owner is
   ``molmetal/scripts/cfm_pool_epoch_rule.py``); an unattested record is
   REFUSED, because a clobbered ``per_epoch_metrics.json`` names a sibling
   population's epoch.
2. Lists every ``.pt`` in ``checkpoints/`` and locates the one whose embedded
   ``epoch`` matches ``best.epoch`` (we read the header with ``torch.load`` on
   ``map_location="cpu"`` so this works without a GPU).
3. Copies that file to ``checkpoints/<stem>_best_epoch{N}.pt``.

If no checkpoint matches (typical: every surviving ``.pt`` is the FINAL epoch,
not the gate-selected one), the script prints a one-line honest-negative and
exits 1 — the user has just demonstrated that the selected epoch is unrecoverable
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The run census -- the SSOT of which run wrote what.
CENSUS_PATH = REPO_ROOT / "metrics" / "by_metric" / "cfm_run_census.json"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _list_ckpts(ckpt_dir: Path) -> list[Path]:
    return sorted(p for p in ckpt_dir.glob("*.pt"))


def _census_row(run_name: str) -> dict | None:
    """The run census's row for ``run_name``, or None when it carries none."""
    from molmetal.scripts.cfm_pool_epoch_rule import census_row

    if not CENSUS_PATH.exists():
        return None
    try:
        census = json.loads(CENSUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return census_row(census, run_name)


def attest_record(run_dir: Path) -> tuple[bool, str]:
    """Is this directory's ``per_epoch_metrics.json`` the RUN OF RECORD's file?

    ``per_epoch_metrics.json`` is written ONCE, at the END of training, to a
    FIXED path, so two runs launched into the same ``--out-dir`` clobber each
    other's record and the survivor is whichever finished LAST.  A
    ``best.epoch`` read out of a clobbered record belongs to a SIBLING
    POPULATION, and re-selecting a checkpoint against it copies the wrong
    epoch's weights.

    The owner is ``molmetal/scripts/cfm_pool_epoch_rule.py``: it resolves the
    run of record of the directory and publishes ``metrics_file_is_run_of_record``
    (the census's own answer when the census carries the run, the local
    epoch-set test otherwise).  ``None`` means "nothing on disk to compare
    against" -- that is NOT an attestation, and the record is refused.
    """
    from molmetal.scripts.cfm_pool_epoch_rule import resolve_population

    try:
        population = resolve_population(
            run_dir, None, census_row=_census_row(run_dir.name)
        )
    except Exception as exc:  # pragma: no cover - the owner must not be fatal
        return False, (
            f"the run-of-record owner raised {type(exc).__name__}: {exc}; "
            f"the recorded file cannot be attested"
        )

    is_ro = population.get("metrics_file_is_run_of_record")
    cohort = population.get("cohort") or {}
    if is_ro is True:
        return True, (
            f"the recorded per-epoch file IS the run of record's own file "
            f"(resolution={population.get('resolution')}, "
            f"epochs={population.get('epoch_indices')}, "
            f"cohort_source={cohort.get('source')})"
        )
    if is_ro is False:
        return False, (
            f"the recorded per-epoch file is NOT the run of record's own file "
            f"(resolution={population.get('resolution')}, "
            f"run-of-record epochs={population.get('run_of_record_epoch_indices')}"
            f", cohort_source={cohort.get('source')})"
        )
    return False, (
        f"no run of record could be resolved for this directory "
        f"(resolution={population.get('resolution')}, "
        f"events_present={population.get('events_present')}), so the recorded "
        f"file cannot be attested as the run of record's own"
    )


def reselect(run_dir: Path) -> int:
    """Post-hoc re-select the gate-selected checkpoint for a single run dir.

    Returns 0 on success, 1 on honest-negative (no matching epoch found, or a
    record that cannot be attested), 2 on missing inputs (no JSON / no
    checkpoints dir).
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

    # The best.epoch must come from the run of record's OWN record.  A
    # clobbered record names a sibling population's epoch, and re-selecting
    # against it copies the wrong weights -- so refuse, and say why.  This
    # script publishes no metric artifact, so refusing is safe.
    attested, why = attest_record(run_dir)
    if not attested:
        print(
            f"[post-hoc-reselect] REFUSED: {pem_path} cannot be attested as "
            f"the run of record's own record -- {why}. No checkpoint is "
            f"re-selected; the selected epoch would belong to another "
            f"population.",
            flush=True,
        )
        return 1

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
