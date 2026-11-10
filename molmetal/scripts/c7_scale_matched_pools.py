#!/usr/bin/env python3
"""C7 — the synthesisability numbers were measured on the wrong generator.

``cfm_sa_yield_audit.py`` scored three pools and concluded the generator cannot
reach SA <= 4.5.  Two things were not checked:

1. **Training scale.**  Two of the three pools come from runs whose recorded
   ``n_train`` is 256.  The corpus is 26,715 complexes.  The THIRD pool
   (``cfm_fullcorpus_2026_10_10``) is a FREE-GENERATION pool whose run of
   record trained on all 26,715 -- MEASURED three ways: the free-generation
   panel's checkpoint string ("epoch 10, n_train 26715"), the census's
   ts-attested cohort (26715/192, ``run_log:ts_matched``), and the fact that
   the two later 1-epoch CPU smokes' ``done`` events POST-DATE the pool file.
   Its own ``per_epoch_metrics.json`` reads ``n_train = 64`` because a LATER
   smoke wrote into the same ``--out-dir``: that file is written ONCE at the
   END of training, so the last writer wins and the run of record's cohort is
   lost.
2. **Pool selection.**  The two pools the audit skipped
   (``cfm_objective_terms_2026_10_11``, ``cfm_fullcorpus_b64_2026_10_11``)
   are RECONSTRUCTION pools from full-corpus runs -- and
   ``cfm_fullcorpus_2026_10_10`` is a THIRD full-corpus pool, the only
   full-corpus FREE-GENERATION one (v1.5 below decides a pool's kind by
   CONTENT, not by its header).

This script scores every pool it finds under ``molmetal/reports/cfm_*``,
attaches each to its run's recorded training size, and reports SA / QED by
scale tier.  It changes no model and launches nothing.

!! V1.1 CORRECTION (2026-10-20): TIER IS NOT THE ONLY THING THAT DIFFERS !!
    The v1.0 table compared ``full_corpus`` against ``small`` and attributed the
    gap to training scale.  That is wrong: the two cells were produced by two
    different pipelines.  ``molmetal/scripts/wf_cfm_decode_dump_smiles.py``
    writes a CSV with a ``metal_z`` column, and it wrote that ONE header in
    TWO eras: before 2026-10-12 it loaded REAL tmQM geometries and ran only
    the bond head -- a RECONSTRUCTION, never the sampler -- and from
    2026-10-12 it calls the trainer's own ODE sampler -- FREE GENERATION.
    The trainer's own path (``evaluate_generation_validity``) writes
    ``n_frags_raw``.  Reconstruction inherits the SA of real chemistry, so its
    SA mean lands within 0.005 of the tmQM population -- the agreement is the
    fingerprint of the artefact, not evidence of quality.

    v1.1 therefore classifies each pool and only aggregates within a pipeline.
    It also carries ``gen_validity`` from the run's events.jsonl, the one
    number both scales compute through the same code path.

!! V1.5 CORRECTION (Wf-A114D, 2026-10-30): THE HEADER ALONE DOES NOT DECIDE !!
    v1.1 read the pool's KIND from its HEADER and said that a ``metal_z``
    header means a reconstruction.  That is true for the two pools of
    2026-10-11 and FALSE for ``cfm_fullcorpus_2026_10_10``: the decode-dump
    script wrote ONE header in TWO eras, and from 2026-10-12 it calls the
    trainer's own ODE sampler (FREE GENERATION).  The kind is now decided by
    the ONE shared rule (``molmetal/scripts/cfm_pool_kind.py``), which reads
    the CONTENT of a ``metal_z`` pool; a pool whose signals disagree is
    UNKNOWN and is refused, never silently bucketed.

    MEASURED on disk (the CONTENT signals that decide the kind):
    ``cfm_fullcorpus_2026_10_10`` -- 66 rows, ``sanitized`` == 0 on 66/66 and
    ``metal_z`` empty on 66/66 -> FREE GENERATION (the ODE path);
    ``cfm_fullcorpus_b64_2026_10_11`` -- 40 rows, ``sanitized`` == 1 on 40/40
    and ``metal_z`` populated on 40/40 -> RECONSTRUCTION;
    ``cfm_objective_terms_2026_10_11`` -- 43 rows, ``sanitized`` == 1 on
    43/43 and ``metal_z`` populated on 43/43 -> RECONSTRUCTION.
    ``wf124_p2_cfm_full`` carries neither marker and no content signal, so it
    is UNKNOWN under the shared rule; this script's discovery globs
    ``cfm_*`` directories, so it is never scored here and lands in no bucket.

!! V1.6 CORRECTION (Wf-A115 P2, 2026-10-31): TWO LOOKUPS WERE WRONG !!
    DEFECT 1 -- THE COHORT WAS READ FROM A CLOBBERED RECORD.
      ``run_config`` preferred the run's own ``per_epoch_metrics.json``.  That
      file is written ONCE at the END of training to a FIXED path, so a later
      run launched into the same ``--out-dir`` OVERWRITES it.  For
      ``cfm_fullcorpus_2026_10_10`` it holds a 1-epoch CPU smoke's config
      (``n_train = 64``), NOT the run of record's 26,715 -- and the clobbered
      file WON, so the fallbacks behind it never fired.  The cohort now has
      ONE owner: the census's own ts-attested resolution, with
      ``molmetal/scripts/cfm_cohort_recovery.py::recover_cohort`` as the
      fallback.  A genuinely unattested cohort is reported as ``unattested``;
      the clobbered value is never used.

    DEFECT 2 -- ``gen_validity`` WAS TAKEN AT THE WRONG EPOCH.
      ``generation_validity`` kept the LAST row of the append-only
      ``events.jsonl`` that carried a ``gen_validity``.  For
      ``cfm_fullcorpus_2026_10_10`` that is the SMOKE's epoch 0 (117/128 =
      0.9140625) -- published as a full-corpus generation quality for a pool
      whose own epoch is 10.  The same defect hit the two RECONSTRUCTION
      pools, where the run was NOT re-launched: there the run's LAST epoch was
      published instead of the pool's epoch, so a validity from a DIFFERENT
      model sat beside that pool's SA / QED.

    THE RULE (v1.6): the ``gen_validity`` published on a pool row is the
    resolved POPULATION's value AT THE POOL'S OWN LAST EPOCH -- the epoch
    whose molecules the pool contains.  The SA / QED / novelty on the same row
    describe THAT epoch, so a validity from another epoch describes a
    different model.  Within the population the AGGREGATE row is taken (the
    one carrying ``gen_validity_n``), never the per-seed rows.  The population
    is the RUN OF RECORD, segmented by the census's own ``_event_segments``
    and ``_population_decision``; a run the census does not carry falls back
    to the segment that CONTAINS the pool's epoch.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/c7_scale_matched_pools.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_UNATTESTED,
    recover_cohort,
)
# The COHORT-SCALE rule (the thresholds, the tier vocabulary and the
# classifier) has ONE owner: molmetal/scripts/cfm_cohort_scale.py.  The names
# are RE-EXPORTED from it so the existing importers keep working unchanged;
# they are the rule owner's VOCABULARY, so a consumer may LABEL a cohort with
# them but must never re-derive the thresholds.
from molmetal.scripts.cfm_cohort_scale import (  # noqa: E402
    FULL_CORPUS_N_TRAIN,
    SMALL_TRAIN_CUTOFF,
    TIERS,
    tier_of,
)
# The POOL-EPOCH rule (which POPULATION a pool's numbers belong to, and the
# aggregate validity AT THE POOL'S OWN EPOCH) has ONE owner too:
# molmetal/scripts/cfm_pool_epoch_rule.py.  The names are RE-EXPORTED from it
# so the existing importers keep working unchanged.
from molmetal.scripts.cfm_pool_epoch_rule import (  # noqa: E402
    _epoch_of,
    census_row,
    generation_validity_at_epoch as generation_validity,
    resolve_population_census as resolve_population,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    GENERATION_MARKER,  # noqa: F401  (re-exported, see the note below)
    POOL_KIND_RULE,
    RECONSTRUCTION_MARKER,  # noqa: F401  (re-exported, see the note below)
    UnknownPoolKindError,
    classify_pool,
    require_known_kind,
)
from molmetal.scripts.cfm_sa_yield_audit import (  # noqa: E402
    ALL_POOLS as SA_AUDIT_POOLS,
    NOVELTY_GATE,
    SA_GATE,
    THRESHOLDS,
    pool_report,
    wilson,
)
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402

# The POPULATION segmentation and the run-of-record decision have ONE owner:
# molmetal/scripts/wf_cfm_run_census.py.  v1.6 REUSES them instead of
# re-implementing them: ``_read_events`` calls the census's own
# ``_event_segments``, ``_population_decision`` is the census's own
# cross-check of the per-epoch file against the event stream, and the
# census's PUBLISHED ``integrity.run_of_record_segment_index`` is the
# resolution this script reads.  No census number is changed here.
from molmetal.scripts.wf_cfm_run_census import (  # noqa: E402
    _load_series as _census_load_series,
    _population_decision as _census_population_decision,
    _read_events as _census_read_events,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

#: The artifact's own one-line statement of the tier rule.  The shared rule
#: owner's string also states the ``unknown`` case, so the two are NOT
#: character-identical; the artifact keeps its own wording -- and the two are
#: ASSERTED to agree on the thresholds, so they cannot drift apart.
TIER_RULE_TEXT = "full_corpus if n_train >= 26715; small if < 1024; else mid"
assert TIER_RULE_TEXT == (
    f"full_corpus if n_train >= {FULL_CORPUS_N_TRAIN}; small if < "
    f"{SMALL_TRAIN_CUTOFF}; else mid"
), "the artifact's tier_rule drifted from the shared cohort-scale thresholds"

# The pool-kind rule (the marker constants, the classifier and the per-kind
# SEMANTICS) has ONE owner: molmetal/scripts/cfm_pool_kind.py.  The marker
# names above are RE-EXPORTED from it so the existing importers keep working
# unchanged; they are the rule owner's VOCABULARY, so a consumer may LABEL a
# pool with them but must never BRANCH on them to re-decide the kind (see
# :func:`pipeline_of`, which asks the rule owner).


def discover_pools(reports: Path) -> List[Path]:
    return sorted(p for p in reports.glob("cfm_*/generated_smiles.csv") if p.is_file())


def pipeline_of(csv_path: Path) -> str:
    """Which code path produced this CSV: ``reconstruction``, ``generation`` or ``unknown``.

    Compatibility seam.  The rule itself lives in
    ``molmetal/scripts/cfm_pool_kind.py`` (:func:`classify_pool`); this
    function is kept because existing importers call it by name.
    """
    return classify_pool(csv_path).kind


#: The census's own verdicts under which the per-epoch metrics file IS the
#: run of record's own record -- and may therefore attest its cohort.  Two of
#: the census's classes are EXCLUDED on purpose: ``class_a_other_population``
#: (the file is another population's record) and ``class_a_truncated_prefix``
#: (the file is a shorter run's prefix); in both the file's cohort belongs
#: to the OTHER run, which is exactly the clobbering this script must not
#: inherit.
FILE_IS_RUN_OF_RECORD_VERDICTS = frozenset(
    {
        "events_absent",
        "events_no_done",
        "no_conflict",
        "class_b_resume_prehistory",
        "class_b_resume_tail",
        "class_c_events_shorter",
        "class_d_foreign_launch",
    }
)


def file_is_run_of_record(run_dir: Path) -> bool:
    """Does the census's own cross-check say the per-epoch file IS the run of record's?

    The census's ``_population_decision`` is the ONE owner of that question:
    it compares the per-epoch file against the append-only event stream and
    names the class.  This reuses it instead of re-deriving "is this file
    stale?" from scratch.
    """
    try:
        rows, _source, _jd = _census_load_series(run_dir)
    except OSError:
        return False
    decision = _census_population_decision(rows, _census_read_events(run_dir))
    return str(decision.get("verdict") or "") in FILE_IS_RUN_OF_RECORD_VERDICTS


def resolve_cohort(
    run_dir: Path,
    census: Dict[str, object],
    population_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """The cohort of the RUN OF RECORD, with the artifact that attests it.

    Wf-A115 P2 (v1.6) -- ONE owner for the cohort.  The run's own
    ``per_epoch_metrics.json`` is written ONCE at the END of training to a
    FIXED path, so a later run launched into the same ``--out-dir``
    OVERWRITES it: for ``cfm_fullcorpus_2026_10_10`` it holds a 1-epoch CPU
    smoke's config (64/32), not the run of record's 26,715/192.  The old
    ``run_config`` chain let that clobbered file WIN, so the fallbacks behind
    it never fired.

    The order is therefore:
      1. the census's own ts-attested cohort (``run["n_train"]`` /
         ``run["n_val"]`` with its ``cohort_source``);
      2. ``cfm_cohort_recovery.recover_cohort`` on the resolved population's
         timestamps -- the same ts-matching rule the census uses;
      3. the per-epoch metrics file, but ONLY when the census's own
         cross-check says that file IS the run of record's own record
         (:func:`file_is_run_of_record`);
      4. the checkpoint's own ``args``.

    A cohort that no artifact attests is reported as ``unattested`` with
    ``n_train = None`` -- never filled in from the clobbered file.
    """
    row = census_row(census, run_dir.name)

    if row is not None:
        source = str(row.get("cohort_source") or "")
        n_train = row.get("n_train")
        n_val = row.get("n_val")
        if n_train is not None and source and source != COHORT_SOURCE_UNATTESTED:
            return {
                "n_train": int(n_train),
                "n_val": int(n_val) if n_val is not None else None,
                "source": f"census:{source}",
                "reason": str(row.get("cohort_reason") or ""),
            }

    population_ts = [
        float(r["ts"]) for r in population_rows if isinstance(r.get("ts"), (int, float))
    ]
    recovered = recover_cohort(run_dir, population_ts)
    if recovered.get("n_train") is not None:
        return {
            "n_train": int(recovered["n_train"]),
            "n_val": (
                int(recovered["n_val"]) if recovered.get("n_val") is not None else None
            ),
            "source": f"cohort_recovery:{recovered.get('source')}",
            "reason": str(recovered.get("reason") or ""),
        }

    if file_is_run_of_record(run_dir):
        path = run_dir / "per_epoch_metrics.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = {}
            cfg = dict(payload.get("config") or {})
            if cfg.get("n_train") is not None:
                return {
                    "n_train": int(cfg["n_train"]),
                    "n_val": (
                        int(cfg["n_val"]) if cfg.get("n_val") is not None else None
                    ),
                    "source": "per_epoch_metrics.json:config",
                    "reason": (
                        "the per-epoch metrics file is the run of record's own "
                        "record"
                    ),
                }

    ckpt = config_from_checkpoint(run_dir)
    if ckpt.get("n_train") is not None:
        return {
            "n_train": int(ckpt["n_train"]),
            "n_val": int(ckpt["n_val"]) if ckpt.get("n_val") is not None else None,
            "source": "checkpoint:args",
            "reason": (
                "the checkpoint carries the run's args verbatim and is written "
                "every epoch"
            ),
        }

    return {
        "n_train": None,
        "n_val": None,
        "source": COHORT_SOURCE_UNATTESTED,
        "reason": (
            "no artifact on disk ties a cohort to this run's run of record"
        ),
    }


def pool_own_draw(
    panel: Optional[Dict[str, object]],
    run_name: str,
    pool_epoch: Optional[int],
) -> Optional[Dict[str, object]]:
    """The pool's OWN draw -- how many attempts yielded a molecule.

    The run-of-record ``gen_validity`` and the pool's own draw are DIFFERENT
    measurements: the first is the trainer's generation eval on the
    validation cohort, the second is how many of the pool's own attempts
    produced a usable molecule.  They are published in SEPARATE fields, each
    with its own source string, so neither can be read as the other.

    MEASURED: ``cfm_fullcorpus_2026_10_10``'s pool holds the 66 survivors of
    192 attempts (66/192 = 0.34375); the free-generation panel records the
    same draw and rounds it to 0.344.
    """
    if not isinstance(panel, dict):
        return None
    checkpoint = str(panel.get("checkpoint") or "")
    match = re.search(r"reports/([^/]+)/", checkpoint)
    if not match or match.group(1) != run_name:
        return None
    epoch_match = re.search(r"epoch (\d+)", checkpoint)
    if epoch_match and pool_epoch is not None and int(epoch_match.group(1)) != int(pool_epoch):
        return None
    generation = panel.get("generation") or {}
    n_valid = generation.get("n_valid")
    n_attempted = generation.get("n_attempted")
    if n_valid is None or n_attempted is None or not int(n_attempted):
        return None
    return {
        "value": float(n_valid) / float(n_attempted),
        "n": int(n_valid),
        "n_attempted": int(n_attempted),
        "source": (
            "metrics/by_metric/cfm_free_generation_panel_seed0.json -> "
            "generation.n_valid (66) / generation.n_attempted (192); the "
            "panel also records generation.gen_validity = 0.344 for the same "
            "draw"
        ),
    }


def config_from_checkpoint(run_dir: Path) -> Dict[str, object]:
    """Read a run's config out of its own checkpoint.

    A run that dies before its first ``per_epoch_metrics.json`` write leaves no
    metrics file and therefore no census row, so ``tier_of`` would return
    "unknown" and its pool would silently fall out of every cross-scale table.
    The checkpoint carries ``args`` verbatim and is written every epoch, so it
    is the only surviving record of what such a run actually trained.

    MEASURED 2026-10-20: the C7 full-corpus launch died inside epoch 4 on a
    fatal CUDA OOM after producing 4 epochs / 2,816 generated SMILES and no
    metrics file.  Without this fallback that pool — the first full-corpus
    free-generation pool on disk — was invisible to the C7 verdict.

    Unpickling a checkpoint executes code, so this only ever reads files under
    this repo's own ``molmetal/reports`` tree, which the trainer wrote.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return {}
    candidates = sorted(ckpt_dir.glob("*_last.pt")) or sorted(ckpt_dir.glob("*.pt"))
    for ckpt in candidates:
        blob: object = None
        try:
            import pickle as _pickle

            with ckpt.open("rb") as handle:
                blob = _pickle.load(handle)
        except Exception:
            # The trainer writes raw pickle via atomic_write_file; a run that
            # took the torch.save fallback wrote a zip instead.  Try both.
            try:
                import torch as _torch

                blob = _torch.load(ckpt, map_location="cpu", weights_only=False)
            except Exception:
                continue
        if not isinstance(blob, dict):
            continue
        args = blob.get("args")
        if isinstance(args, dict) and args.get("n_train"):
            return dict(args)
    return {}


def run_config(run_dir: Path, census: Dict[str, object]) -> Dict[str, object]:
    """Prefer the run's own per_epoch_metrics.json, then the census, then its checkpoint.

    v1.6: this is NOT the cohort owner any more.  The run's own
    ``per_epoch_metrics.json`` is written ONCE at the END of training to a
    FIXED path, so a later run launched into the same ``--out-dir``
    OVERWRITES it -- for ``cfm_fullcorpus_2026_10_10`` it holds a 1-epoch CPU
    smoke's config, not the run of record's.  The cohort therefore comes from
    :func:`resolve_cohort` (the census's ts-attested resolution).  What this
    function still supplies is the run's OTHER settings (``epochs``, the bond
    head's metrics), which are reported verbatim and are not used to bucket a
    pool.
    """
    path = run_dir / "per_epoch_metrics.json"
    cfg: Dict[str, object] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            payload = {}
        cfg = dict(payload.get("config") or {})
        best = payload.get("best") or {}
        for key in ("bond_exact_type_rate", "bond_type_macro_f1", "bond_f1"):
            if key in best:
                cfg[key] = best[key]
    if not cfg.get("n_train"):
        for row in census.get("runs", []):
            if row.get("run") == run_dir.name:
                cfg.update({k: v for k, v in row.get("config", {}).items() if k not in cfg})
                break
    if not cfg.get("n_train"):
        cfg.update({k: v for k, v in config_from_checkpoint(run_dir).items() if k not in cfg})
    return cfg


def sa_per_molecule(smiles: Sequence[str]) -> Dict[str, float]:
    """SA score per input SMILES, so it can be joined with novelty."""
    from rdkit import Chem, RDLogger

    from molmetal.adapters.flow_matching_lipman.sa_penalty import compute_sa_score

    RDLogger.DisableLog("rdApp.*")
    out: Dict[str, float] = {}
    for text in smiles:
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) else None
        if mol is None:
            continue
        sa = compute_sa_score(mol)
        if sa is not None:
            out[text] = float(sa)
    return out


def novelty_per_molecule(smiles: Sequence[str], train_path: Path) -> Dict[str, float]:
    """1 - max Tanimoto to the training corpus, one value per parseable molecule.

    ``cfm_sa_yield_audit.novelty_of`` only returns the mean, which cannot answer
    the joint question the C7 gate asks (SA <= 4.5 *and* novelty >= 0.90).
    """
    import pandas as pd

    from molmetal.scripts.cfm_generation_metrics import max_tanimoto_to_reference, morgan

    train = pd.read_csv(train_path, low_memory=False)
    if "split" in train.columns:
        train = train[train["split"] == "train"]
    ref = [f for f in (morgan(s) for s in train["smiles"].dropna().astype(str)) if f is not None]
    keys, fps = [], []
    for text in smiles:
        fp = morgan(text) if isinstance(text, str) else None
        if fp is not None:
            keys.append(text)
            fps.append(fp)
    if not fps or not ref:
        return {}
    sims = max_tanimoto_to_reference(fps, ref)
    return {k: float(1.0 - s) for k, s in zip(keys, sims)}


def per_epoch_stats(
    rows: Sequence[Dict[str, str]],
    sa_map: Dict[str, float],
    nov_map: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    """SA / joint-gate rate per ``epoch`` value, for pools that span epochs.

    ``cfm_pcgrad_*`` wrote 30 epochs into one CSV; pooling them reports a rate
    for a model that spent most of those epochs unable to emit bonds. The last
    epoch is the model anyone would ship, so it gets its own row.
    """
    buckets: Dict[str, Dict[str, object]] = {}
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None or epoch == "":
            continue
        smiles = row.get("smiles", "")
        sa = sa_map.get(smiles)
        if sa is None:
            continue
        bucket = buckets.setdefault(
            str(epoch), {"n": 0, "n_sa_le_gate": 0, "n_joint": 0, "sa_sum": 0.0}
        )
        bucket["n"] = int(bucket["n"]) + 1
        bucket["sa_sum"] = float(bucket["sa_sum"]) + float(sa)
        if sa <= SA_GATE:
            bucket["n_sa_le_gate"] = int(bucket["n_sa_le_gate"]) + 1
            nov = nov_map.get(smiles)
            if nov is not None and nov >= NOVELTY_GATE:
                bucket["n_joint"] = int(bucket["n_joint"]) + 1
    for bucket in buckets.values():
        n_e = int(bucket["n"])
        bucket["sa_mean"] = round(float(bucket["sa_sum"]) / n_e, 4) if n_e else None
        bucket["rate_sa_le_gate"] = round(int(bucket["n_sa_le_gate"]) / n_e, 4) if n_e else None
        bucket["wilson_ci95"] = wilson(int(bucket["n_sa_le_gate"]), n_e)
        bucket.pop("sa_sum", None)
    return buckets


def dedupe_pools(
    pools: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Drop pools that repeat rows already counted by a larger pool.

    Two directories holding the same molecules are ONE pool, not two.

    MEASURED 2026-10-21: a second session preserved the crashed full-corpus
    run's pool under ``cfm_c7_fullgen_ep4_partial_2026_10_20`` while
    ``cfm_c7_fullgen_2026_10_20`` still existed.  Identical file, identical
    sha256, but the aggregates counted 2,816 molecules twice — the full-corpus
    tier reported 5,715 scored instead of 2,899 and the generation total 7,974
    instead of 5,158.

    The same run then RESUMED and finished 18 epochs, so the two files stopped
    being byte-identical: the partial is now a strict PREFIX of the full pool
    (epochs 0-3 of 0-17).  A sha256 rule cannot see that, and the tier again
    double-counted 2,816 molecules — this time as 15,484 instead of 12,668.
    Hence the second rule: a pool whose ``(epoch, smiles)`` row set is a
    subset of a kept pool's is a prefix of it, not an independent sample.

    Keying on content rather than on the directory name also catches a re-run
    that happens to regenerate the same strings, which a name rule would miss.
    Pools with no sha256/row keys are kept: an unprovable duplicate is the
    worse error to drop silently.
    """
    seen: Dict[str, str] = {}
    kept: List[Dict[str, object]] = []
    duplicates: List[Dict[str, object]] = []
    # Largest first, so a prefix always meets its superset already kept,
    # whatever order the filesystem handed us.
    for pool in sorted(pools, key=lambda p: (-int(p.get("n_rows") or 0), str(p.get("label")))):
        sha = str(pool.get("sha256") or "")
        if sha and sha in seen:
            duplicates.append(
                {
                    "label": pool.get("label"),
                    "same_content_as": seen[sha],
                    "reason": "identical_sha256",
                    "n_rows": pool.get("n_rows"),
                    "sha256": sha,
                }
            )
            continue
        keys = pool.get("_row_keys")
        prefix_of = None
        if isinstance(keys, (set, frozenset)) and keys:
            for other in kept:
                other_keys = other.get("_row_keys")
                if not isinstance(other_keys, (set, frozenset)):
                    continue
                if len(keys) < len(other_keys) and keys <= other_keys:
                    prefix_of = other
                    break
        if prefix_of is not None:
            duplicates.append(
                {
                    "label": pool.get("label"),
                    "same_content_as": prefix_of.get("label"),
                    "reason": "strict_prefix_of",
                    "n_rows": pool.get("n_rows"),
                    "n_rows_of_kept": prefix_of.get("n_rows"),
                }
            )
            continue
        if sha:
            seen[sha] = str(pool.get("label"))
        kept.append(pool)
    return kept, duplicates


def verdict_reading(generation: Dict[str, object]) -> str:
    """Build the verdict sentence from the aggregates instead of hard-coding it.

    The previous version was a literal string.  When the first full-corpus
    free-generation pool appeared on disk the flag
    ``full_corpus_generation_pool_exists`` — which IS computed from the data —
    flipped to True, while the prose kept saying "There is no full-corpus
    free-generation pool".  A reader taking the sentence at face value would
    have retracted a result that had just been measured.  Nothing could catch
    it, because no test compared the two.

    MEASURED 2026-10-20: cfm_c7_fullgen_2026_10_20, 2,816 molecules at
    n_train=26,715, SA<=4.5 = 10 (0.0036).
    """
    tiers = generation.get("by_tier") or {}
    full = tiers.get("full_corpus")
    small = tiers.get("small")
    parts = [
        "Only pools from the same pipeline are comparable: a bond-head "
        "reconstruction of a real tmQM geometry inherits real chemistry, so its "
        "SA score is not generator quality."
    ]
    if full:
        parts.append(
            f"A full-corpus free-generation pool exists "
            f"({full['n_scored']} molecules, SA<={SA_GATE} = "
            f"{full['rate_sa_le_gate']:.4f} [{full['wilson_ci95'][0]:.4f}, "
            f"{full['wilson_ci95'][1]:.4f}])."
        )
        if small:
            better = full["rate_sa_le_gate"] > small["rate_sa_le_gate"]
            parts.append(
                f"The small tier (n_train < {SMALL_TRAIN_CUTOFF}) scores "
                f"{small['rate_sa_le_gate']:.4f} [{small['wilson_ci95'][0]:.4f}, "
                f"{small['wilson_ci95'][1]:.4f}], so scaling to the full corpus is "
                f"{'better' if better else 'not better'} on this gate."
            )
    else:
        parts.append(
            "There is no full-corpus free-generation pool; the largest "
            "generation pool is below the full-corpus tier."
        )
    parts.append(
        "gen_validity is the one metric both scales compute through the same "
        "path, and it does not improve with scale on the runs that have it."
    )
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "molmetal" / "reports")
    parser.add_argument(
        "--census",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "cfm_run_census.json",
    )
    parser.add_argument(
        "--train-reference",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "c7_scale_matched_pools.json",
    )
    parser.add_argument(
        "--free-generation-panel",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "cfm_free_generation_panel_seed0.json",
        help=(
            "the free-generation panel that records a pool's OWN draw "
            "(n_valid / n_attempted); it is published SEPARATELY from the "
            "run of record's gen_validity so neither can be read as the other"
        ),
    )
    args = parser.parse_args(argv)

    # v1.4 (OPT-IN): kind-separated ``by_tier``.  OFF by default; the
    # environment variable is the switch (repo precedent:
    # WFA2_CONFIG_STRICT=1, WF343C4_GATE=1) -- NOT a new argparse flag, so
    # the default launch path is byte-identical.
    strict_pool_kind = os.environ.get("CFM_POOL_KIND_STRICT", "") == "1"

    census = json.loads(args.census.read_text()) if args.census.exists() else {}
    panel: Optional[Dict[str, object]] = None
    if args.free_generation_panel.exists():
        try:
            loaded = json.loads(args.free_generation_panel.read_text())
        except json.JSONDecodeError:
            loaded = None
        panel = loaded if isinstance(loaded, dict) else None
    pools: List[Dict[str, object]] = []
    for path in discover_pools(args.reports):
        report = pool_report(path.parent.name, path)
        if report is None:
            continue
        cfg = run_config(path.parent, census)
        report["epochs_configured"] = cfg.get("epochs")
        report["bond_exact_type_rate"] = cfg.get("bond_exact_type_rate")
        report["bond_type_macro_f1"] = cfg.get("bond_type_macro_f1")
        if strict_pool_kind:
            # Strict mode REFUSES an unclassified pool instead of bucketing
            # it into an "unknown" tier that later reads like a number.
            try:
                kind = require_known_kind(path)
            except UnknownPoolKindError as exc:
                print(f"[c7] REFUSING to score an unclassified pool: {exc}", file=sys.stderr)
                return 2
        else:
            kind = classify_pool(path)
        report["pipeline"] = kind.kind
        report["pool_kind_reason"] = kind.reason
        # joint gate: SA <= 4.5 AND novelty >= 0.90, per molecule
        import csv as _csv

        with path.open(newline="") as handle:
            rows = list(_csv.DictReader(handle))
        # The pool's OWN last epoch -- the epoch whose molecules the pool
        # actually contains.  It is read from the CSV's ``epoch`` column, so
        # it is known BEFORE anything is looked up in the event stream (v1.6
        # reordering: the validity used to be resolved first, from a stream
        # that holds SEVERAL populations).
        pool_epochs = [
            e for e in (_epoch_of({"epoch": r.get("epoch")}) for r in rows) if e is not None
        ]
        pool_epoch = max(pool_epochs) if pool_epochs else None
        report["pool_epoch"] = pool_epoch
        # The POPULATION the pool's numbers belong to (the run of record),
        # segmented by the census's own machinery.
        population = resolve_population(path.parent, census, pool_epoch)
        report["population_source"] = population["source"]
        # The COHORT has ONE owner (v1.6): the ts-attested census /
        # cohort-recovery resolution, never the clobbered per-epoch file.
        cohort = resolve_cohort(path.parent, census, population["rows"])
        n_train = cohort.get("n_train")
        report["n_train"] = n_train
        report["n_val"] = cohort.get("n_val")
        report["cohort_source"] = cohort.get("source")
        report["tier"] = tier_of(n_train)
        # The validity AT THE POOL'S OWN EPOCH (v1.6), from the resolved
        # population's AGGREGATE row -- never the last row of the file.
        report["gen_validity"] = generation_validity(
            path.parent, epoch=pool_epoch, rows=population["rows"]
        )
        report["gen_validity_source"] = (
            f"{population['source']}: the aggregate gen_validity row (the one "
            f"carrying gen_validity_n) at epoch {pool_epoch} of the resolved "
            f"population"
        )
        # The pool's OWN draw, published SEPARATELY so the run of record's
        # validity cannot be read as the pool's yield.
        own_draw = pool_own_draw(panel, path.parent.name, pool_epoch)
        if own_draw is not None:
            report["gen_validity_pool"] = own_draw
        # Row identity for prefix detection: a resumed run rewrites the same
        # (epoch, smiles) pairs the partial run already had, so a subset test
        # is the only way to see that one pool is the head of the other.
        report["_row_keys"] = {(r.get("epoch", ""), r.get("smiles", "")) for r in rows}
        texts = [r.get("smiles", "") for r in rows]
        sa_map = sa_per_molecule(texts)
        nov_map = novelty_per_molecule(list(sa_map), args.train_reference)
        joint = [k for k, s_ in sa_map.items() if s_ <= SA_GATE and nov_map.get(k, 0.0) >= NOVELTY_GATE]
        gate_subset = [nov_map[k] for k, s_ in sa_map.items() if s_ <= SA_GATE and k in nov_map]
        report["joint_gate"] = {
            "n": len(sa_map),
            "n_sa_and_novel": len(joint),
            "rate": round(len(joint) / len(sa_map), 4) if sa_map else None,
            "wilson_ci95": wilson(len(joint), len(sa_map)),
            "mean_novelty_of_sa_subset": round(sum(gate_subset) / len(gate_subset), 4) if gate_subset else None,
        }
        # Per-epoch breakdown. Pooling every epoch mixes a model that has not
        # learned to emit bonds with one that has, so the headline for a
        # multi-epoch pool should be the LAST epoch, not the pooled rate.
        per_epoch = per_epoch_stats(rows, sa_map, nov_map)
        report["by_epoch"] = per_epoch
        if per_epoch:
            last = max(per_epoch, key=lambda k: int(k))
            report["last_epoch"] = {"epoch": int(last), **per_epoch[last]}
        report["train_fraction_of_corpus"] = (
            round(float(n_train) / FULL_CORPUS_N_TRAIN, 4) if n_train else None
        )
        pools.append(report)

    # Two directories holding the same molecules are ONE pool, not two.
    #
    # MEASURED 2026-10-21: a second session preserved the crashed full-corpus
    # run's pool under cfm_c7_fullgen_ep4_partial_2026_10_20 while the original
    # cfm_c7_fullgen_2026_10_20 still existed.  Identical file, identical
    # sha256, but the aggregates counted 2,816 molecules twice — the full-corpus
    # tier reported 5,715 scored instead of 2,899.  Deduplicating on content
    # (not on name) also catches a re-run that happens to regenerate the same
    # strings, which a name-based rule would miss.
    _deduped, _duplicates = dedupe_pools(pools)
    pools = _deduped

    # Wf-A121 / I3 (R7) -- the MOLECULE-weighted companion to the pool mean.
    #
    # `pool_mean_sa` / `pool_mean_qed` are the UNWEIGHTED mean over the tier's
    # POOLS: every pool counts once, whatever its size.  `n_scored` is a SUM
    # over molecules, so the two live on different scales -- a tier holding one
    # 12,668-molecule pool and one 66-molecule pool reports an unweighted mean
    # the 66 molecules cannot move, while the molecule-level mean is dominated
    # by the 12,668.  Both are published; NEITHER replaces the other, and
    # `pool_mean_sa` / `pool_mean_qed` are left exactly as they were.
    _POOL_MEAN_WEIGHTING_NOTE = (
        "pool_mean_sa / pool_mean_qed are the UNWEIGHTED mean over the "
        "tier's pools; n_scored is a SUM over molecules — read "
        "molecule_weighted_mean_* for the molecule-level mean"
    )

    def _molecule_weighted(rows: List[Dict[str, object]], key: str) -> Optional[float]:
        """sum(pool.<key> * pool.n_scored) / sum(pool.n_scored).

        A pool with no scored molecule contributes nothing: it has no mean
        to weight and no weight to contribute.
        """
        num = 0.0
        den = 0
        for p in rows:
            value = p.get(key)
            n = int(p.get("n_scored") or 0)
            if value is None or n <= 0:
                continue
            num += float(value) * n
            den += n
        return round(num / den, 4) if den else None

    def pooled(rows: List[Dict[str, object]]) -> Dict[str, object]:
        n_scored = sum(int(p.get("n_scored") or 0) for p in rows)
        k_gate = sum(int((p.get("sa_counts_at_or_below") or {}).get(str(SA_GATE), 0)) for p in rows)
        k_joint = sum(int((p.get("joint_gate") or {}).get("n_sa_and_novel") or 0) for p in rows)
        sa_means = [p["sa_mean"] for p in rows if p.get("sa_mean") is not None]
        qed_means = [p["qed_mean"] for p in rows if p.get("qed_mean") is not None]
        return {
            "n_pools": len(rows),
            "n_scored": n_scored,
            "n_sa_le_gate": k_gate,
            "n_joint_gate": k_joint,
            "rate_joint_gate": round(k_joint / n_scored, 4) if n_scored else None,
            "rate_sa_le_gate": round(k_gate / n_scored, 4) if n_scored else None,
            "wilson_ci95": wilson(k_gate, n_scored),
            "pool_mean_sa": round(sum(sa_means) / len(sa_means), 4) if sa_means else None,
            "pool_mean_qed": round(sum(qed_means) / len(qed_means), 4) if qed_means else None,
            "molecule_weighted_mean_sa": _molecule_weighted(rows, "sa_mean"),
            "molecule_weighted_mean_qed": _molecule_weighted(rows, "qed_mean"),
            "pool_mean_sa_weighting": _POOL_MEAN_WEIGHTING_NOTE,
        }

    def _tier_rows(subset: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        """``tier -> aggregate`` over ``subset``, plus the pipeline mix.

        ``comparable`` is False exactly when the tier's rows come from more
        than one pipeline -- a mixed cell averages decoder fidelity with
        generator quality and is not a number anyone should read.
        """
        out = {t: pooled([p for p in subset if p["tier"] == t]) for t in TIERS}
        out = {k: v for k, v in out.items() if v["n_pools"]}
        for tier, row in out.items():
            row["pipeline_mix"] = {
                pl: sum(
                    int(p.get("n_scored") or 0)
                    for p in subset
                    if p["tier"] == tier and p["pipeline"] == pl
                )
                for pl in sorted({p["pipeline"] for p in subset if p["tier"] == tier})
            }
            row["comparable"] = len(row["pipeline_mix"]) == 1
        return out

    tiers = _tier_rows(pools)

    by_pipeline: Dict[str, Dict[str, object]] = {}
    for pl in sorted({str(p["pipeline"]) for p in pools}):
        rows = [p for p in pools if p["pipeline"] == pl]
        sub = {t: pooled([p for p in rows if p["tier"] == t]) for t in TIERS}
        sub = {k: v for k, v in sub.items() if v["n_pools"]}
        gv = [p["gen_validity"]["value"] for p in rows if p.get("gen_validity")]
        by_pipeline[pl] = {
            "n_pools": len(rows),
            "n_scored": sum(int(p.get("n_scored") or 0) for p in rows),
            "by_tier": sub,
            "tiers_present": sorted(sub),
            "tiers_missing": [t for t in ("full_corpus", "mid", "small") if t not in sub],
            "gen_validity_mean": round(sum(gv) / len(gv), 4) if gv else None,
        }

    # v1.4 (OPT-IN): the mixed ``by_tier`` cell is replaced by the
    # kind-separated view.  The mixed cell averages decoder fidelity with
    # generator quality; the kind-separated one never mixes them.  The
    # default path is UNCHANGED -- this switch is OFF unless the environment
    # asks for it (repo precedent: WFA2_CONFIG_STRICT=1, WF343C4_GATE=1).
    by_tier_by_kind = {
        pl: _tier_rows([p for p in pools if p["pipeline"] == pl])
        for pl in sorted({str(p["pipeline"]) for p in pools})
    }
    # Under strict mode the mixed table is REPLACED by the kind-separated one;
    # otherwise ``by_tier`` stays exactly as it has always been.
    by_tier_out = by_tier_by_kind if strict_pool_kind else tiers

    # Which pools the SA audit scores.  DERIVED from the SA audit's own pool
    # set, so the two cannot drift: a hardcoded list here would silently
    # claim a pool is "omitted" from an audit that now scores it.
    used_by_sa_audit = {Path(rel).parent.name for _label, rel in SA_AUDIT_POOLS}
    omitted = [
        p["label"]
        for p in pools
        if p["label"] not in used_by_sa_audit and int(p.get("n_scored") or 0) > 0
    ]
    for pool in pools:
        pool.pop("_row_keys", None)

    payload = {
        "artifact": "c7_scale_matched_pools",
        "version": "1.6" if strict_pool_kind else "1.5",
        "strict_pool_kind": strict_pool_kind,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/c7_scale_matched_pools.py",
            "git_commit": _git_commit(),
            "census": "metrics/by_metric/cfm_run_census.json",
            "scoring": "reuses cfm_sa_yield_audit.pool_report (SA + QED, unscored counted)",
            "v1_1_correction": (
                "v1.0 aggregated by tier alone. The full_corpus cell and the "
                "small cell came from two different pipelines -- a bond-head "
                "reconstruction of REAL tmQM geometries versus free generation "
                "-- so v1.0's 'training scale' reading compared two pipelines."
            ),
            "v1_2_correction": (
                "pools are deduplicated on file content (sha256), because two "
                "directories holding the same molecules were counted twice and "
                "doubled the full-corpus tier."
            ),
            "v1_3_correction": (
                "sha256 alone stopped working once the full run resumed to 18 "
                "epochs: the 4-epoch partial is then a strict prefix, not a copy. "
                "A pool whose (epoch, smiles) rows are a subset of a kept pool's "
                "is now dropped as a prefix."
            ),
            "v1_4_pool_kind": (
                "the pool-kind rule has ONE owner: "
                "molmetal/scripts/cfm_pool_kind.py.  The mixed by_tier cell is "
                "replaced by the kind-separated by_tier_by_kind view; by_tier "
                "itself is only replaced under CFM_POOL_KIND_STRICT=1."
            ),
            "v1_5_pool_kind": (
                "the pool kind is decided by CONTENT, not by the header.  "
                "MEASURED: cfm_fullcorpus_2026_10_10 has sanitized == 0 on "
                "66/66 rows and an empty metal_z on 66/66 -> FREE GENERATION "
                "(the ODE path); cfm_fullcorpus_b64_2026_10_11 (40 rows) and "
                "cfm_objective_terms_2026_10_11 (43 rows) have sanitized == 1 "
                "and a populated metal_z on every row -> RECONSTRUCTION.  A "
                "pool whose signals disagree is UNKNOWN and is never pooled "
                "into a kind."
            ),
            "v1_6_cohort_owner": (
                "the cohort (n_train / n_val) has ONE owner: the census's own "
                "ts-attested resolution (run.n_train / run.n_val with its "
                "cohort_source), with "
                "molmetal/scripts/cfm_cohort_recovery.py::recover_cohort as "
                "the fallback.  The run's own per_epoch_metrics.json is "
                "written ONCE at the END of training to a FIXED path, so a "
                "later run launched into the same --out-dir overwrites it; "
                "that clobbered file can no longer win.  MEASURED: "
                "cfm_fullcorpus_2026_10_10's file reads n_train=64 / n_val=32 "
                "(a 1-epoch CPU smoke) while the run of record's cohort is "
                "26715/192 (census, run_log:ts_matched)."
            ),
            "v1_6_epoch_aligned_gen_validity": (
                "the gen_validity published on a pool row is the resolved "
                "POPULATION's value AT THE POOL'S OWN LAST EPOCH -- the epoch "
                "whose molecules the pool contains -- taken from the "
                "AGGREGATE row (the one carrying gen_validity_n), never from "
                "the per-seed rows and never from another epoch.  The SA / "
                "QED on the same row describe that epoch, so a validity from "
                "another epoch describes a different model.  The population is "
                "the RUN OF RECORD, segmented by the census's own "
                "_event_segments / _population_decision; a run the census does "
                "not carry falls back to the segment that CONTAINS the pool's "
                "epoch.  MEASURED: cfm_fullcorpus_2026_10_10's pool is epoch "
                "10 and its run of record's epoch-10 aggregate is "
                "0.3255208333333333 (n=768); the value the old 'last logged "
                "row' rule published was the epoch-0 row of a LATER 1-epoch "
                "CPU smoke that wrote into the same directory (0.9140625, "
                "n=128)."
            ),
            "v1_6_pool_own_draw": (
                "the pool's OWN draw (how many of its attempts yielded a "
                "molecule) is published SEPARATELY, in gen_validity_pool, with "
                "its own source string, so it can never be read as the run of "
                "record's gen_validity.  MEASURED: cfm_fullcorpus_2026_10_10 "
                "holds the 66 survivors of 192 attempts (66/192 = 0.34375); "
                "the free-generation panel records the same draw and rounds "
                "it to 0.344."
            ),
        },
        "duplicate_pools_dropped": _duplicates,
        "design": {
            "sa_gate": SA_GATE,
            "novelty_gate": NOVELTY_GATE,
            "thresholds": list(THRESHOLDS),
            "full_corpus_n_train": FULL_CORPUS_N_TRAIN,
            "small_train_cutoff": SMALL_TRAIN_CUTOFF,
            "tier_rule": TIER_RULE_TEXT,
            "pipeline_rule": POOL_KIND_RULE,
            "pool_kind_source": "molmetal/scripts/cfm_pool_kind.py",
            "comparable_across_tiers_only_within_a_pipeline": True,
        },
        "pools": pools,
        "by_tier": by_tier_out,
        "by_tier_by_kind": by_tier_by_kind,
        "by_pipeline": by_pipeline,
        "selection_gap": {
            "pools_scored_here": len(pools),
            "pools_used_by_cfm_sa_guided": sorted(used_by_sa_audit),
            "pools_omitted_there": omitted,
            "note": (
                "cfm_sa_guided.json scores the generator's pools (2,396 "
                "molecules from three runs with n_train 26715/256/256) and, "
                "separately, the two full-corpus RECONSTRUCTION pools in their "
                "own block -- the two are never pooled into one rate."
            ),
        },
        "verdict": {
            "generation_pools": by_pipeline.get("generation", {}).get("n_pools", 0),
            "generation_molecules": by_pipeline.get("generation", {}).get("n_scored", 0),
            "generation_tiers_missing": by_pipeline.get("generation", {}).get("tiers_missing", []),
            "reconstruction_molecules": by_pipeline.get("reconstruction", {}).get("n_scored", 0),
            "full_corpus_generation_pool_exists": "full_corpus"
            in by_pipeline.get("generation", {}).get("by_tier", {}),
            "reading": verdict_reading(by_pipeline.get("generation", {})),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    for p in pools:
        gv = (p.get("gen_validity") or {}).get("value")
        last = p.get("last_epoch") or {}
        print(
            f"[c7] {p['label'][:38]:38s} {p['pipeline'][:14]:14s} n={p.get('n_scored', 0):5d} "
            f"n_train={p['n_train']} sa_mean={p.get('sa_mean')} qed={p.get('qed_mean')} "
            f"gate={((p.get('sa_counts_at_or_below') or {}).get(str(SA_GATE)))} "
            f"last_epoch={last.get('epoch')} last_gate={last.get('rate_sa_le_gate')} "
            f"gen_validity={gv}"
        )
    for tier, row in tiers.items():
        print(
            f"[c7] tier={tier:11s} pools={row['n_pools']} n={row['n_scored']} "
            f"rate={row['rate_sa_le_gate']} ci={row['wilson_ci95']} "
            f"mix={row['pipeline_mix']} comparable={row['comparable']}"
        )
    for pl, sub in by_tier_by_kind.items():
        for tier, row in sub.items():
            print(
                f"[c7] kind={pl:14s} tier={tier:11s} pools={row['n_pools']} "
                f"n={row['n_scored']} rate={row['rate_sa_le_gate']} "
                f"ci={row['wilson_ci95']} comparable={row['comparable']}"
            )
    for pl, row in by_pipeline.items():
        print(f"[c7] pipeline={pl:14s} n={row['n_scored']} tiers={row['tiers_present']} missing={row['tiers_missing']}")
    print(f"[c7] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
