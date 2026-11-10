#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Wf-A155 (2026-11-01) -- a PoseBusters validity column for the CFM pool.

WHAT THIS EXISTS FOR
--------------------
The frontier band we are compared against (92.7--94 % connected / 99.9 % valid /
91.9 % PoseBusters-valid, per arXiv 2609.08333 and FlowMol3 arXiv 2508.12629)
is quoted as a *percentage* -- but the PROVENANCE of the 92.7--94 % band is NOT
settled on disk: the frontier post-processing matrix record
(``frontier_matrix.json``) holds TWO competing readings of which frontier
cells the band names -- the first under ``band_provenance``, the second under
``sources``.``EF-TALFM`` -- and they do not even agree on the numerals, so
the attribution above is ONE reading, not a settled provenance, and only the
invariants that hold under BOTH readings may be published (the band is
EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale).  We had
no PB column at all: every PoseBusters
artifact under ``metrics/by_metric/`` either measured a single molecule, or
covered at most a couple of dozen molecules on a 1-pocket smoke, or was
SKIPPED / DEFERRED because the package was not installed at the time.  That
enumeration is NOT transcribed here: it is READ FROM DISK at run time and
rendered into the artifact (Wf-A164).  So this is the FIRST pool-scale
PoseBusters measurement in this repo.

The DEFAULT pool is ``molmetal/reports/cfm_c8_fullgen_2026_10_24/
generated_smiles.csv``.  Its row count, epoch structure, seed count and
checkpoint identity are MEASURED from the pool and its run directory at run
time and rendered into the artifact; they are NOT literals in this script
(Wf-A164).  A run against ANY other pool reports THAT pool's numbers and
paths, and says so when the pool's provenance is not derivable.

WHAT IT IS AND WHAT IT IS NOT
-----------------------------
* It IS a molecule-geometry validity measurement over the emitted SMILES:
  RDKit parse -> ETKDGv3 embedding -> MMFF94 optimisation -> PoseBusters
  ``mol`` config (the ``config="mol"`` module surface: loading, sanitisation,
  InChI convertibility, connectivity, radicals, bond lengths, bond angles,
  internal steric clash, ring flatness, double-bond flatness, internal
  energy).
* It is NOT a restatement of the published END-TO-END median headline
  (``metrics/by_metric/cfm_fragmentation_audit.json`` ->
  ``/summary/generation_side/single_molecule_rate_end_to_end``).  That value
  AND its run count are READ FROM THAT ARTIFACT at run time, not asserted
  here.  This artifact is a SINGLE-POOL GEOMETRY rate.  Different quantity,
  different population.
* It is NOT a restatement of the single-pool fragment count
  (``n_frags == 1``).  That count is COMPUTED from the pool's own
  ``n_frags_raw`` column and cross-checked against the published artifacts
  that name THIS pool; it is a 2D graph-fragment count, while this is a 3D
  geometry verdict.
* It says NOTHING about whether the metal is coordinated correctly.
  PoseBusters has **no metal-coordination-sphere check**.  A molecule can be
  PB-valid and still have a chemically absurd metal environment.

DESIGN NOTES
------------
* PoseBusters is NOT re-implemented here.  The existing surface
  ``molmetal.validation.posebusters_runner.check_posebusters`` is called
  molecule-by-molecule; ``posebusters_available()`` is the probe;
  ``pb_pass_rate()`` is the aggregation helper for the "strict" rate.
* The module fails CLOSED: a check that could not be computed (``None``)
  counts as a failure.  That matters here -- MMFF94 cannot parameterise most
  metal centres, so PoseBusters' ``internal_energy`` check comes back
  ``None`` for metal-containing molecules, which then makes the molecule
  PB-invalid.  That is an honest negative, not a bug in this script.
* TWO denominators are reported, and both are in the artifact:
    - STRICT:  denominator = every row in scope (unparseable + embed-failed
               + skipped all count as NOT valid).
    - SCORED:  denominator = rows that produced a PoseBusters verdict.
  Neither is presented alone.
* CPU-only.  This module must never import ``torch`` and must never touch a
  GPU; the wave that owns it runs with ``CUDA_VISIBLE_DEVICES=""``.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wf_a155_posebusters_validity.py

    # documented subset mode (labelled SUBSET in the artifact, never as pool %)
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wf_a155_posebusters_validity.py --sample-size 200 --sample-seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_POOL = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c8_fullgen_2026_10_24"
    / "generated_smiles.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "metrics" / "by_metric" / "wfa155_posebusters_validity.json"
)

DEFAULT_N_CONFORMERS = 5
DEFAULT_EMBED_METHOD = "ETKDGv3"
DEFAULT_OPTIMIZE_METHOD = "MMFF94"

#: Per-row status vocabulary.
STATUS_PB_VALID = "pb_valid"
STATUS_PB_INVALID = "pb_invalid"
STATUS_EMBED_FAILED = "embed_failed"
STATUS_UNPARSEABLE = "unparseable"
STATUS_SKIPPED = "skipped_posebusters_missing"
#: A row whose PoseBusters call exceeded the hard per-row wall-clock cap.
#: Counted as NOT pb_valid (fail closed) and reported separately, because a
#: timeout is a censoring event, not a chemistry verdict.
STATUS_TIMEOUT = "timeout"

#: Marker used by the child process to hand its JSON payload back to the
#: parent.  The child prints other lines (the ``molmetal`` env banner) to
#: stdout, so the parent keys on this marker rather than on "last line".
CHILD_MARKER = "WFA155_JSON:"

#: The frontier numbers being compared against, for reference only.  These
#: are the FRONTIER's own published numbers -- they describe the papers we
#: are compared against, NOT our population -- and they are carried as
#: MACHINE fields so that every narrative that quotes them is backed by a
#: number this artifact actually holds (Wf-A164).
FRONTIER_BAND = {
    "band_connected_pct": [92.7, 94.0],
    "frontier_denominator_draws": "10,000-15,000",
    "frontier_denominator_draws_min": 10000,
    "frontier_denominator_draws_max": 15000,
    "ef_talfm_arxiv_2609_08333": {
        "arxiv_id": "2609.08333",
        "connected_pct": 92.98,
        "sanitized_pct": 98.25,
        "unique_sanitized_pct": 98.21,
        "novel_unique_sanitized_pct": 94.33,
        "n_generated": 10000,
    },
    "flowmol3_arxiv_2508_12629": {
        "arxiv_id": "2508.12629",
        "pct_valid": [99.9, 0.1],
        "pct_pb_valid": [91.9, 0.7],
        "venue": "Digital Discovery 2026, 5(5):2052-2066",
        "dataset": "GEOM-Drugs with explicit hydrogens",
    },
}

#: MEASURED 2026-11-01: the wall-clock time of ONE molecule-level
#: ``check_posebusters`` call for the three LONGEST SMILES in the published
#: pool.  The per-row timeout rationale quotes these three numbers, so
#: they are carried as MACHINE fields and the prose is rendered from them
#: (Wf-A181): a number rendered in a narrative that no machine field of
#: the same artifact backs is a defect, and dropping the numerals would
#: DESTROY the measurement they record.
#:
#: Wf-A182 (2026-11-01) -- THE DEFECT: a module CONSTANT cannot say WHICH
#: pool it measured.  Left unbound, the three numbers CERTIFIED a
#: measurement for whatever pool the artifact names -- the Wf-A164 defect
#: class (see the block comment below) that the narrative guard exists to
#: stop: repointing the tool at a different pool would publish
#: 86.7 / 60.5 / 0.1 under that pool's name, and the NUMBER guard would
#: stay SILENT, because the machine field backing them IS the constant.
#: :func:`longest_pool_measurement_binding` therefore emits the three
#: numbers ONLY for the pool they were measured on, and says so otherwise.
LONGEST_POOL_SMILES_CALL_SECONDS: Tuple[float, float, float] = (86.7, 60.5, 0.1)

#: Wf-A182 (2026-11-01): the PROVENANCE of the three numbers above, cited
#: from disk.  The measurement is recorded in the Wf-A164 PoseBusters
#: report, in the ``protocol.row_timeout_rationale`` field of
#: ``molmetal/reports/wfa164_posebusters_narrative/default_pool_bounded_original.json``
#: -- the same record that names the pool it was measured on (its
#: ``source_pool.path`` / ``source_pool.sha256``), the tool that took it
#: (``protocol.backend_module``), and whose ``date`` / ``generated_at_utc``
#: are the measurement date.
LONGEST_POOL_SMILES_CALL_SECONDS_PROVENANCE: Dict[str, Any] = {
    "measured_on_pool_path": (
        "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
    ),
    "measured_on_pool_sha256": (
        "4b3f7687b1ad3a9af71525b815922a34b0ccb2b4edc87a041871900837bb5e46"
    ),
    "measurement_date": "2026-11-01",
    "measurement_record_path": (
        "molmetal/reports/wfa164_posebusters_narrative/"
        "default_pool_bounded_original.json"
    ),
    "measurement_record_field": "protocol.row_timeout_rationale",
    "measurement_tool": (
        "molmetal.validation.posebusters_runner.check_posebusters"
    ),
    "measurement_scope": (
        "the three LONGEST SMILES of that pool, ONE molecule-level "
        "check_posebusters call each"
    ),
    "certification_rule": (
        "these three numbers are a MEASUREMENT OF THE POOL NAMED ABOVE and "
        "may be certified for that pool only; for any other pool this block "
        "reports that they do not apply and the numbers are not emitted"
    ),
}

#: Wf-A182 (2026-11-01): the superseded field name and its value in the
#: PUBLISHED artifact, preserved VERBATIM per AGENTS.md §5.5 rule #6.
#: The OLD name was unscoped: it read as a repo-wide maximum sample size,
#: but the value is scoped to the PoseBusters candidate files enumerated
#: in the same block.
SUPERSEDED_LARGEST_N_SAMPLES_FIELD_2026_11_01: Dict[str, Any] = {
    "field": "largest_n_samples",
    "value": 26,
    "why_superseded": (
        "the OLD name was UNSCOPED and read as a repo-wide maximum sample "
        "size, but it is scoped to the PoseBusters candidate files "
        "enumerated in the same block (a DEFERRED artifact is excluded "
        "because it carries the size it WOULD have measured). The value is "
        "unchanged and is preserved here verbatim (AGENTS.md §5.5 rule #6)"
    ),
}

#: Where the earlier PoseBusters artifacts live.  A machine field, so the
#: prose that points at the directory is backed by it.
PRIOR_POSE_BUSTERS_DIR = "metrics/by_metric"

#: The rate fields the pre-Wf-A164 enumeration prose quoted from each
#: earlier PoseBusters artifact.  They are READ FROM THE FILE ON DISK --
#: never from a literal -- and carried as machine fields, so that any
#: narrative quoting them is backed by the artifact itself.
QUOTED_PRIOR_RATE_FIELDS: Tuple[Tuple[str, ...], ...] = (
    ("aggregate", "best_measured_smoke"),
    ("value", "rate_protein_aware_subset"),
)

#: The published metric artifacts that carry a single-fragment rate for a
#: named pool.  An artifact is a candidate ONLY when its own ``source_pool``
#: names THIS pool -- so pointing the tool at a different pool yields no
#: cross-check rather than a comparison against someone else's number.
PUBLISHED_POOLED_SOURCES: Tuple[
    Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...
] = (
    (
        "wfa155_ligand_connected_rate",
        ("metal_inclusive", "single_fragment_rate"),
        ("metal_inclusive", "n_single_fragment"),
    ),
    (
        "cfm_fragmentation_postproc_arm",
        ("raw_axis", "single_fragment_rate_over_parsed"),
        ("raw_axis", "n_single"),
    ),
)

#: Candidate files under ``metrics/by_metric/`` that may carry an earlier
#: PoseBusters measurement.  Enumerated from DISK at run time (Wf-A164):
#: the old artifact transcribed this list by hand, so it could not notice
#: that a file had been added, removed or re-measured.
PRIOR_POSE_BUSTERS_ARTIFACTS: Tuple[str, ...] = (
    "wf32_pb_valid.json",
    "wf32_pb_valid_summary.json",
    "wf32_pb_valid_verify.json",
    "wf32_pb_valid_verify_summary.json",
    "pb_pass_rate.json",
    "steric_clash_pb_dock.json",
    "steric_clash_pb_dock_test_001.json",
    "steric_clash_pb_dock_1h36.json",
    "docked_pose_rmsd.json",
    "r13_100x3_partial.json",
    "tier2_summary.json",
    "vina_kcal_per_mol.json",
    "_audit/m5_real_per_metric.json",
)


# ----------------------------------------------------------------
# Wf-A164 -- COMPUTED pool profile, published cross-checks, and the
#            narrative self-consistency guard.
#
# THE DEFECT CLASS (Wf-A155 -> Wf-A162 -> Wf-A163 -> Wf-A164):
# a measurement tool embeds POOL-SPECIFIC CLAIMS as string literals
# inside a ``tag: MEASURED`` artifact.  For the tool's DEFAULT pool the
# literals are TRUE; for ANY OTHER pool the tool emits FALSE statements.
# The repair is the Wf-A162 pattern: COMPUTE every descriptive number
# from the run, and add a guard that FAILS the run when a narrative
# number is not backed by a machine field of the SAME artifact.
# ----------------------------------------------------------------


class NarrativeSelfConsistencyError(Exception):
    """Raised when a narrative string carries a number that no machine
    field of the same artifact backs.

    Wf-A164 (2026-11-01): the offending artifact is carried on the
    exception so the caller can write it out as an explicit ``status:
    FAIL`` artifact rather than silently publishing a false claim.
    """

    def __init__(self, artifact: Dict[str, Any], guard: Dict[str, Any]) -> None:
        super().__init__(
            "narrative self-consistency FAILED: unbacked numbers "
            f"{sorted(guard.get('unbacked_numbers', {}))} "
            f"unclassified leaves {guard.get('unclassified_leaves', {})}"
        )
        self.artifact = artifact
        self.guard = guard


def _relative_to_project(path: Path) -> str:
    """Return ``path`` relative to the project root when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def longest_pool_measurement_binding(
    *, pool_path: str, pool_sha256: str
) -> Dict[str, Any]:
    """The provenance binding for the longest-pool-SMILES call seconds.

    Wf-A182 (2026-11-01): the three numbers in
    :data:`LONGEST_POOL_SMILES_CALL_SECONDS` are a MEASUREMENT OF ONE
    POOL, and a constant that cannot say WHICH pool it measured
    CERTIFIES a measurement for whatever pool the artifact names -- the
    Wf-A164 defect class the narrative guard exists to stop.

    This function returns the binding for the pool at hand.  The three
    numbers are attached ONLY when that pool IS the pool the measurement
    was taken on (both the path and the sha256 must match, so a DIFFERENT
    file that happens to sit at the same path is not certified either).
    For any other pool the binding says that the numbers do not apply and
    they are not emitted at all.
    """
    binding = dict(LONGEST_POOL_SMILES_CALL_SECONDS_PROVENANCE)
    same_pool = bool(
        pool_path == binding["measured_on_pool_path"]
        and pool_sha256 == binding["measured_on_pool_sha256"]
    )
    binding["applicable_to_this_pool"] = same_pool
    if same_pool:
        binding["seconds"] = list(LONGEST_POOL_SMILES_CALL_SECONDS)
    else:
        binding["why_not_applicable"] = (
            "this run's pool is NOT the pool the three numbers were measured "
            "on, so they are NOT certified for it and are not reported"
        )
    return binding


def read_run_provenance(pool_path: Path) -> Dict[str, Any]:
    """Read the RUN's own provenance from the pool's sibling files.

    The pool CSV carries NO seed column and NO checkpoint column, so the
    decode-seed count and the checkpoint identity are NOT derivable from
    the pool itself.  They ARE derivable from the run directory that
    produced it: ``events.jsonl`` records the decode seeds and
    ``checkpoints/`` names the checkpoint.

    When the sibling files are absent -- e.g. a COPY of the pool placed in
    a scratch directory -- the provenance is reported as UNAVAILABLE
    rather than silently inherited from the default pool.  That is the
    difference between "measured" and "copied from the default pool".
    """
    out: Dict[str, Any] = {
        "events_jsonl": None,
        "n_seeds": None,
        "seeds": None,
        "n_seeds_source": None,
        "checkpoint_files": None,
        "n_checkpoints": None,
        "checkpoint_names": None,
        "available": False,
    }
    directory = Path(pool_path).parent
    events = directory / "events.jsonl"
    if events.is_file():
        seeds: set = set()
        n_seeds_field: set = set()
        try:
            with open(events, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if isinstance(record.get("seed"), int):
                        seeds.add(int(record["seed"]))
                    if isinstance(record.get("n_seeds"), int):
                        n_seeds_field.add(int(record["n_seeds"]))
        except OSError:
            pass
        out["events_jsonl"] = _relative_to_project(events)
        if seeds:
            out["seeds"] = sorted(seeds)
            out["n_seeds"] = len(seeds)
            out["n_seeds_source"] = out["events_jsonl"]
        elif n_seeds_field:
            out["n_seeds"] = max(n_seeds_field)
            out["n_seeds_source"] = out["events_jsonl"]
    checkpoint_dir = directory / "checkpoints"
    if checkpoint_dir.is_dir():
        try:
            files = sorted(
                entry.name for entry in checkpoint_dir.iterdir() if entry.is_file()
            )
        except OSError:
            files = []
        out["checkpoint_files"] = files
        # A run directory usually holds "<ckpt>.pt" AND "<ckpt>_last.pt" --
        # the SAME checkpoint at two points in time.  Count distinct
        # checkpoint NAMES, not files, or the count would be inflated.
        names = sorted({Path(name).stem.removesuffix("_last") for name in files})
        out["checkpoint_names"] = names
        out["n_checkpoints"] = len(names)
    out["available"] = out["n_seeds"] is not None or out["n_checkpoints"] is not None
    return out


def build_pool_profile(
    *,
    pool_path: Path,
    pool_sha256: str,
    n_data_rows: int,
    columns: Sequence[str],
    epoch_labels: Sequence[str],
    fragment_counts: Sequence[int],
    composition: Dict[str, Any],
) -> Dict[str, Any]:
    """Everything the artifact says ABOUT OUR POPULATION -- all computed.

    Wf-A164 (2026-11-01): this block used to be four hard-coded strings
    ("2112 rows", "704 rows each, 6 seeds per epoch",
    "cfm_c8_fullgen_2026_10_24/generated_smiles.csv").  Every field below
    is read from the pool, from the pool's run directory, or from the
    measured composition -- so pointing the tool at a different pool
    produces that pool's numbers.
    """
    epoch_counts = Counter(epoch_labels)
    rows_per_epoch = [
        {"epoch": label, "n_rows": int(count)}
        for label, count in sorted(epoch_counts.items(), key=lambda kv: str(kv[0]))
    ]
    per_epoch_values = [entry["n_rows"] for entry in rows_per_epoch]
    provenance = read_run_provenance(pool_path)
    n_single_fragment = sum(1 for value in fragment_counts if value == 1)
    n_frags_histogram = Counter(
        str(value) for value in fragment_counts if value >= 0
    )
    return {
        "path": _relative_to_project(Path(pool_path)),
        "sha256": pool_sha256,
        "n_data_rows": int(n_data_rows),
        "columns": list(columns),
        "epoch_column_distribution": {
            label: int(count)
            for label, count in sorted(epoch_counts.items(), key=lambda kv: str(kv[0]))
        },
        "n_distinct_epochs": len(rows_per_epoch),
        "rows_per_epoch": rows_per_epoch,
        "n_rows_per_epoch_min": min(per_epoch_values) if per_epoch_values else 0,
        "n_rows_per_epoch_max": max(per_epoch_values) if per_epoch_values else 0,
        "epoch_counts_uniform": len(set(per_epoch_values)) <= 1,
        "n_seeds": provenance["n_seeds"],
        "seeds": provenance["seeds"],
        "n_seeds_source": provenance["n_seeds_source"],
        "n_checkpoints": provenance["n_checkpoints"],
        "checkpoint_names": provenance["checkpoint_names"],
        "checkpoint_files": provenance["checkpoint_files"],
        "run_provenance_available": provenance["available"],
        "n_single_fragment_rows": int(n_single_fragment),
        "single_fragment_fraction": (
            float(n_single_fragment) / n_data_rows if n_data_rows else 0.0
        ),
        "single_fragment_pct": (
            100.0 * float(n_single_fragment) / n_data_rows if n_data_rows else 0.0
        ),
        "n_frags_histogram": dict(sorted(n_frags_histogram.items(), key=lambda kv: int(kv[0]))),
        "n_rows_parsed_by_rdkit": composition.get("n_rows_parsed_by_rdkit"),
        "n_rows_with_a_metal_atom": composition.get(
            "n_rows_with_at_least_one_metal_atom"
        ),
        "fraction_rows_with_a_metal_atom": composition.get(
            "fraction_rows_with_a_metal_atom"
        ),
        "n_rows_with_an_explicit_hydrogen_atom_in_the_smiles": composition.get(
            "n_rows_with_an_explicit_hydrogen_atom_in_the_smiles"
        ),
    }


def _pool_phrase(profile: Dict[str, Any]) -> str:
    """One computed sentence describing THIS pool's structure."""
    n_rows = profile["n_data_rows"]
    n_epochs = profile["n_distinct_epochs"]
    per_epoch = profile["rows_per_epoch"]
    if n_epochs <= 1:
        epoch_phrase = "a single epoch label"
    else:
        counts = [int(entry["n_rows"]) for entry in per_epoch]
        if len(set(counts)) <= 1:
            epoch_phrase = (
                f"{n_epochs} epoch labels with {counts[0]} row"
                f"{'' if counts[0] == 1 else 's'} each"
            )
        else:
            joined = ", ".join(str(value) for value in counts)
            epoch_phrase = (
                f"{n_epochs} epoch labels with {joined} rows respectively"
            )
    n_seeds = profile.get("n_seeds")
    if n_seeds is None:
        seed_phrase = (
            "the decode seed count is NOT derivable from this pool directory "
            "(no sibling events.jsonl), so no seed count is claimed"
        )
    else:
        seed_phrase = f"with {n_seeds} decode seeds"
    n_checkpoints = profile.get("n_checkpoints")
    if n_checkpoints is None:
        checkpoint_phrase = (
            "the checkpoint count is NOT derivable from this pool directory "
            "(no sibling checkpoints/ directory), so none is claimed"
        )
    else:
        checkpoint_phrase = (
            f"decoded from {n_checkpoints} checkpoint"
            f"{'' if n_checkpoints == 1 else 's'}"
        )
    return (
        f"ONE pool of {n_rows} rows spanning {epoch_phrase}, "
        f"{checkpoint_phrase}, {seed_phrase}"
    )


def build_comparability_breaks(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """The breaks between our numbers and the published frontier band.

    Wf-A164 (2026-11-01): every number that describes OUR population is
    COMPUTED from the run -- the pool's row count, its epoch structure,
    its seed count, its checkpoint count -- so this text cannot go stale
    when the tool is pointed at a different pool.  It used to be a module
    constant carrying "2112 rows", "704 rows each, 6 seeds per epoch" and
    the DEFAULT pool's path, which is FALSE for every pool except that
    one -- including the wrong pool PATH in a ``tag: MEASURED`` artifact.
    """
    ours = _pool_phrase(profile)
    return (
        {
            "id": 1,
            "axis": "DENOMINATOR",
            "frontier": (
                f"over {FRONTIER_BAND['frontier_denominator_draws']} generated "
                "molecules drawn from a training-set-distribution sampling"
            ),
            "ours": f"{ours} ({profile['path']})",
            "consequence": "different populations; the two rates are not exchangeable",
        },
        {
            "id": 2,
            "axis": "METRIC",
            "frontier": (
                "'connected' = the DECODER's own connectivity claim; 'valid' = "
                "RDKit sanitisation; '% PB-Valid' = PoseBusters geometry"
            ),
            "ours": (
                "'connected' = an RDKit fragment count of the emitted SMILES; "
                "'pb_valid' = the PoseBusters 'mol' module verdict on an "
                "ETKDGv3/MMFF94 conformer ensemble"
            ),
            "consequence": (
                "the denominators are different events even before the "
                "population differs"
            ),
        },
        {
            # Wf-A176 (2026-11-01) CORRECTION — the superseded text, PRESERVED
            # VERBATIM per AGENTS.md §5.5 rule #6:
            #     "frontier": "measured on the published POST-PROCESSED stream",
            #     "consequence": "our number is not post-processed and theirs is",
            #
            # WHY: FALSE per the repo's OWN band-provenance record —
            # molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json
            # `band_provenance.endpoint_candidates` carries
            # `post_processed: false` for BOTH band endpoints (GeoLDM
            # Validity & Uniq 92.7 %, HLTF PB-Valid 94.0 %).  The consequence
            # was false for the same reason: the band is RAW too.  The band's
            # real separation from our number is the METRIC-TYPE break.
            #
            # Wf-A177 (2026-11-01) CORRECTION 2 — the correction-1 text below,
            # PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and itself
            # SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record) and neither endpoint is "
            #         "a connected-component count"
            #     ),
            #     "consequence": (
            #         "post-processing is a THIRD, separate axis on top of the METRIC "
            #         "break, not the break itself: both arms are RAW, and the "
            #         "decisive break is that neither band endpoint is a "
            #         "connected-component count"
            #     ),
            #
            # WHY correction 1 was SUPERSEDED (adversarial verify, Wf-A176):
            # it asserted "neither endpoint is a connected-component count" as
            # if settled.  The repo holds TWO competing provenance readings
            # for the band (frontier_matrix.json -> `band_provenance` :51-89
            # vs the same file's `frontier_axis_mapping` :360/:378-379,
            # EF-TALFM arXiv:2609.08333v4) and under the SECOND the lower
            # endpoint IS a connectivity count (EF-TALFM's own raw `C.` cell,
            # 92.98 %).  The two readings do not even agree on the numerals
            # (92.7/94.0 vs 92.98/94.33), so the band's source cells are NOT
            # settled on disk.  Only the invariants that hold under BOTH
            # readings may be published.
            "id": 3,
            "axis": "POST-PROCESSING",
            # Wf-A177 (2026-11-01) CORRECTION 3 — the correction-2 rendering
            # below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
            # itself SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record); which frontier cells the "
            #         "band names is NOT settled on disk -- the repo holds TWO "
            #         "competing provenance readings for it (frontier_matrix.json "
            #         "`band_provenance` :51-89 vs the same file's "
            #         "`frontier_axis_mapping` :360/:378-379, EF-TALFM "
            #         "arXiv:2609.08333v4) that do not even agree on the numerals "
            #         "(92.7/94.0 vs 92.98/94.33)"
            #     ),
            #     "consequence": (
            #         "post-processing is a THIRD, separate axis on top of the METRIC "
            #         "break, not the break itself: both arms are RAW, and the "
            #         "decisive break is that under BOTH readings the band is "
            #         "external, is NOT one homogeneous quantity, and is NOT on our "
            #         "scale (OUR number is a connected-component count on the RAW "
            #         "emitted SMILES, and nothing else)"
            #     ),
            #
            # WHY correction 2 was SUPERSEDED: the emitters' OWN narrative
            # self-consistency guard (the Wf-A162 class-level guard) REJECTED
            # it — six of its numeric tokens were backed by NO machine field
            # of this artifact.  The guard is right: a `tag: MEASURED`
            # artifact must not render a number it cannot back.  The fix is
            # to keep the CLAIM and drop the numerals from the PROSE — the
            # concrete readings live in the SOURCE RECORD named below, at the
            # two named key paths.  This keeps the correction TEXT-ONLY: 0
            # numeric leaves change and 0 keys are added.
            #
            # Wf-A182 (2026-11-01) CORRECTION 4 — the correction-3 rendering
            # below, PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
            # itself SUPERSEDED:
            #     "frontier": (
            #         "the published band's two endpoints are RAW (post_processed: "
            #         "false in the band-provenance record); which frontier cells the "
            #         "band names is NOT settled on disk -- the repo holds TWO "
            #         "competing provenance readings for it, both recorded in "
            #         "the frontier post-processing matrix record "
            #         "(`frontier_matrix.json`)"
            #         " -- the first under `band_provenance`, the second under "
            #         "`sources`.`EF-TALFM` -- and they do not even agree on the "
            #         "numerals"
            #     ),
            #     "consequence": (
            #         "post-processing is a THIRD, separate axis on top of the METRIC "
            #         "break, not the break itself: both arms are RAW, and the "
            #         "decisive break is that under BOTH readings the band is "
            #         "external, is NOT one homogeneous quantity, and is NOT on our "
            #         "scale (OUR number is a connected-component count on the RAW "
            #         "emitted SMILES, and nothing else)"
            #     ),
            #
            # WHY correction 3 was SUPERSEDED (adversarial verify, Wf-A182):
            # it asserted, WITHOUT qualification, that the band's two
            # endpoints are RAW and that "both arms are RAW".  Under the
            # SECOND competing reading the SAME record holds -- its
            # `sources`.`EF-TALFM` path, EF-TALFM arXiv:2609.08333v4
            # Table 3(a) -- the frontier arm is MIXED: a RAW connectivity
            # cell (`C.` = 92.98 %), counted BEFORE the largest-component
            # retention, paired with a POST-PROCESSED companion cell
            # (`N. & U. & S.` = 94.33 %).  A blanket "the band is RAW" is
            # therefore FALSE under one of the two readings the record
            # itself holds, and it is the MIRROR of the blanket "the band is
            # POST-PROCESSED" claim that Wf-A175 already refuted.  The
            # corrected text scopes `post_processed: false` to the
            # `band_provenance` reading and states that whether the band AS A
            # WHOLE is RAW is NOT settled on disk.  The invariants that hold
            # under BOTH readings are untouched, and NO new numeral is
            # introduced.
            "frontier": (
                "under the `band_provenance` reading the published band "
                "names two cells that are RAW (post_processed: false in the "
                "band-provenance record), but whether the band AS A WHOLE is "
                "RAW is NOT settled on disk -- the repo holds TWO "
                "competing provenance readings for it, both recorded in "
                "the frontier post-processing matrix record "
                "(`frontier_matrix.json`)"
                " -- the first under `band_provenance`, the second under "
                "`sources`.`EF-TALFM` -- and they do not even agree on the "
                "numerals"
            ),
            "ours": "RAW -- no post-processing was applied to this pool",
            "consequence": (
                "post-processing is a THIRD, separate axis on top of the METRIC "
                "break, not the break itself: OUR arm is RAW, and whether the "
                "band's arm is RAW as a WHOLE is NOT settled on disk; the "
                "decisive break is that under BOTH readings the band is "
                "external, is NOT one homogeneous quantity, and is NOT on our "
                "scale (OUR number is a connected-component count on the RAW "
                "emitted SMILES, and nothing else)"
            ),
        },
        {
            "id": 4,
            "axis": "HYDROGENS",
            "frontier": "the frontier dataset carries EXPLICIT hydrogens",
            "ours": (
                "our pool carries IMPLICIT hydrogens (RDKit AddHs is applied "
                "inside the embedding step, but the pool itself has no explicit-H "
                "atoms written into the SMILES)"
            ),
            "consequence": (
                "hydrogens change both sanitisation and geometry; this alone "
                "breaks the comparison"
            ),
        },
    )


def _dig(doc: Any, path: Sequence[str]) -> Any:
    """Follow a jsonpath of KEYS through nested dicts; ``None`` if absent."""
    cur = doc
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_metric(metric: str) -> Optional[Dict[str, Any]]:
    """Read ``metrics/by_metric/<metric>.json``; ``None`` if absent."""
    path = PROJECT_ROOT / "metrics" / "by_metric" / f"{metric}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None


def _published_cross_check(
    profile: Dict[str, Any], rate: float, n_single: int
) -> Dict[str, Any]:
    """Compare this pool's single-fragment count against every PUBLISHED
    metric artifact that names THIS pool.

    Wf-A164 (2026-11-01): the old artifact ASSERTED which published
    number it reproduced ("the 8.51 % headline" / "the 25.05 % count"),
    which was FALSE on every pool except its default one -- and it named
    the WRONG POOL PATH while doing it.  The comparison is now COMPUTED:
    an artifact is a candidate only when its own ``source_pool`` names
    this pool, and the delta is measured, never asserted.
    """
    pool_name = Path(profile["path"]).parent.name
    pool_file = Path(profile["path"]).name
    considered: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    for metric, rate_path, n_path in PUBLISHED_POOLED_SOURCES:
        doc = _load_metric(metric)
        if doc is None:
            considered.append({"metric": metric, "present": False})
            continue
        src = doc.get("source_pool")
        src_name = Path(str(src)).parent.name if src else None
        pub_rate = _dig(doc, rate_path)
        pub_n = _dig(doc, n_path)
        # A candidate applies ONLY when it names THIS pool.  The
        # full-path equality is the strong rule; the directory-name rule
        # applies only when OUR pool lives inside the repo, so that a COPY
        # of a pool in a scratch directory is never mistaken for it (every
        # pool CSV in this project is called generated_smiles.csv, so the
        # FILE name carries no identity at all).
        same_path = bool(src and str(src) == profile["path"])
        same_dir_name = bool(src_name and src_name == pool_name)
        names_this_pool = bool(
            same_path
            or (same_dir_name and profile["path"].startswith("molmetal/"))
        )
        considered.append(
            {
                "metric": metric,
                "present": True,
                "names_this_pool": names_this_pool,
                # The other pool's identity and numbers are recorded ONLY
                # when the candidate applies to THIS pool.  Otherwise this
                # artifact would carry another pool's row counts, which is
                # exactly the defect class being repaired.
                "source_pool": src if names_this_pool else None,
                "source_pool_name": src_name if names_this_pool else None,
                "published_rate": pub_rate if names_this_pool else None,
                "published_n_single": pub_n if names_this_pool else None,
                "applicability": (
                    "applies to THIS pool"
                    if names_this_pool
                    else "NOT applicable -- this published artifact measures "
                    "a different pool"
                ),
            }
        )
        if names_this_pool and isinstance(pub_rate, (int, float)):
            matches.append(
                {
                    "metric": metric,
                    "published_rate": pub_rate,
                    "published_n_single": pub_n,
                    "this_run_rate": rate,
                    "this_run_n_single": n_single,
                    "abs_delta": abs(float(pub_rate) - float(rate)),
                    "rate_identical": pub_rate == rate,
                    "numerator_identical": (
                        None if pub_n is None else pub_n == n_single
                    ),
                }
            )
    if not matches:
        verdict = (
            "NO PUBLISHED ARTIFACT NAMES THIS POOL -- no cross-check was "
            "possible, and none is claimed"
        )
    elif all(m["rate_identical"] for m in matches):
        verdict = (
            "THIS RUN REPRODUCES the published rate for THIS pool exactly "
            "(bit-for-bit)"
        )
    else:
        verdict = (
            "THIS RUN DIFFERS from the published rate for THIS pool; the "
            "delta is recorded above and is NOT reconciled here"
        )
    return {
        "rule": (
            "a published artifact is compared ONLY when its own source_pool "
            "names THIS pool; the comparison is computed, never asserted"
        ),
        "this_pool_name": pool_name,
        "this_run_rate": rate,
        "this_run_n_single": n_single,
        "published_artifacts_considered": considered,
        "matches": matches,
        "n_matches": len(matches),
        "verdict": verdict,
    }


def _end_to_end_headline() -> Dict[str, Any]:
    """The END-TO-END median headline, read from the published audit.

    This is the quantity the old prose called "the 8.51 % headline".  It
    is a MEDIAN over several RUNS of a DIFFERENT quantity ("one connected
    parseable molecule per DRAWN sample"), so it is not comparable to a
    single-pool per-ROW rate -- but its value and its run count are carried
    here so the prose can quote them from the machine field instead of
    from a literal.
    """
    doc = _load_metric("cfm_fragmentation_audit")
    source = "metrics/by_metric/cfm_fragmentation_audit.json"
    jsonpath = "/summary/generation_side/single_molecule_rate_end_to_end"
    if doc is None:
        return {
            "source": source,
            "jsonpath": jsonpath,
            "present": False,
            "median": None,
            "median_pct": None,
            "n_runs": None,
        }
    node = _dig(
        doc, ("summary", "generation_side", "single_molecule_rate_end_to_end")
    )
    if not isinstance(node, dict):
        return {
            "source": source,
            "jsonpath": jsonpath,
            "present": False,
            "median": None,
            "median_pct": None,
            "n_runs": None,
        }
    return {
        "source": source,
        "jsonpath": jsonpath,
        "present": True,
        "median": node.get("median"),
        "median_pct": (
            float(node["median"]) * 100.0
            if isinstance(node.get("median"), (int, float))
            else None
        ),
        # Wf-A182: `median_pct` is a UNIT CONVERSION of `median`, not a
        # number read from disk.  It is labelled as derived so that a
        # reader (and the guard) can tell it from a measured value.
        "median_pct_derivation": (
            "DERIVED, not read from disk: this is the SAME machine field "
            "`median` expressed in PERCENT; the value read from disk is "
            "`median`"
        ),
        "n_runs": node.get("n"),
        "definition": (
            "'one connected, parseable molecule per DRAWN sample', median "
            "over the generation-side runs"
        ),
    }


#: A JSON key that scopes a subtree to the PoseBusters measurement.
_PB_KEY_RE = re.compile(r"(^|_)pb(_|$)|posebusters", re.IGNORECASE)


def _collect_sample_sizes(node: Any, *, root_is_pb: bool) -> List[int]:
    """Collect the POSE BUSTERS sample-size integers in a JSON document.

    The keys are matched EXACTLY (``n_samples``, ``n_evaluations``,
    ``n_total``) so that a CHECK count (``n_pb_checks_total``) is never
    mistaken for a sample count, and the search is SCOPED to the
    PoseBusters sub-trees: an artifact that also reports Vina, SA and
    diversity sample sizes must not have those counted as PoseBusters
    samples.
    """
    wanted = ("n_samples", "n_evaluations", "n_total")
    found: List[int] = []

    def walk(current: Any, in_pb: bool) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                child_in_pb = in_pb or bool(_PB_KEY_RE.search(str(key)))
                if (
                    child_in_pb
                    and key in wanted
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                ):
                    found.append(int(value))
                walk(value, child_in_pb)
        elif isinstance(current, list):
            for item in current:
                walk(item, in_pb)

    walk(node, root_is_pb)
    return found


def _quoted_rates(doc: Any) -> Dict[str, float]:
    """The RATE fields of a prior artifact, read from the document itself.

    Wf-A181: the pre-Wf-A164 enumeration prose quoted two rates from the
    earlier PoseBusters artifacts (``best_measured_smoke`` and
    ``rate_protein_aware_subset``).  The rates are read from the file on
    disk -- never from a literal -- so that a narrative quoting them is
    backed by a machine field of THIS artifact.
    """
    out: Dict[str, float] = {}
    if not isinstance(doc, dict):
        return out
    for keys in QUOTED_PRIOR_RATE_FIELDS:
        node: Any = doc
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            out[keys[-1]] = node
    return out


def _prior_posebusters_artifacts() -> Dict[str, Any]:
    """Enumerate the earlier PoseBusters artifacts FROM DISK.

    Wf-A164 (2026-11-01): the old artifact transcribed this enumeration
    by hand -- file names, sample counts, rates and statuses.  A
    hand-transcribed enumeration cannot notice that a file was added,
    removed or re-measured.  It is now read from disk, and the prose that
    quotes it is rendered from the values read.
    """
    base = PROJECT_ROOT / "metrics" / "by_metric"
    rows: List[Dict[str, Any]] = []
    for name in PRIOR_POSE_BUSTERS_ARTIFACTS:
        path = base / name
        entry: Dict[str, Any] = {
            "file": f"metrics/by_metric/{name}",
            "present": path.is_file(),
        }
        if entry["present"]:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw = ""
            try:
                doc = json.loads(raw)
            except Exception:
                doc = None
            entry["mentions_posebusters"] = "posebusters" in raw.lower()
            if isinstance(doc, dict):
                entry["metric"] = doc.get("metric")
                entry["status"] = doc.get("status")
                # A DEFERRED artifact carries the size it WOULD have
                # measured, not a measurement.  Excluding it is what makes
                # "the largest sample recorded anywhere" mean a sample that
                # was actually measured.
                deferred = str(doc.get("status") or "").upper().startswith(
                    "DEFERRED"
                )
                entry["deferred"] = deferred
                root_is_pb = bool(
                    _PB_KEY_RE.search(str(doc.get("metric") or ""))
                )
                samples = _collect_sample_sizes(doc, root_is_pb=root_is_pb)
                entry["sample_sizes_present"] = sorted(set(samples))
                entry["largest_sample_size"] = (
                    max(samples) if samples and not deferred else None
                )
                quoted = _quoted_rates(doc)
                if quoted:
                    entry["quoted_rates"] = quoted
        rows.append(entry)
    present = [entry for entry in rows if entry["present"]]
    mentioning = [entry for entry in present if entry.get("mentions_posebusters")]
    with_samples = [
        entry
        for entry in present
        if isinstance(entry.get("largest_sample_size"), int)
    ]
    largest = (
        max(with_samples, key=lambda entry: entry["largest_sample_size"])
        if with_samples
        else None
    )
    return {
        "rule": (
            "the earlier PoseBusters artifacts are enumerated from DISK; "
            "this block was read on 2026-11-01, when the published artifact "
            "was patched additively (the emitter reads the same files at "
            "run time), and every file path, sample size and rate below "
            "was read from the file it names"
        ),
        "n_candidates_scanned": len(rows),
        "n_present": len(present),
        "n_mentioning_posebusters": len(mentioning),
        "n_deferred": sum(1 for entry in present if entry.get("deferred")),
        # Wf-A182: the OLD name `largest_n_samples` was UNSCOPED -- it read
        # as a repo-wide maximum sample size, but it is scoped to the
        # PoseBusters candidate files enumerated in `files`.  The name now
        # carries the scope; the old name and its value are preserved
        # verbatim in the dated key below (AGENTS.md §5.5 rule #6).
        "largest_n_samples_among_posebusters_candidates": (
            None if largest is None else largest["largest_sample_size"]
        ),
        "largest_n_samples_file": None if largest is None else largest["file"],
        "largest_n_samples_scope": (
            "the largest sample size among the PoseBusters candidate files "
            "enumerated in `files`; a DEFERRED artifact is excluded because "
            "it carries the size it WOULD have measured. This is NOT a "
            "repo-wide maximum -- other, non-PoseBusters artifacts under "
            "metrics/by_metric/ report larger sample sizes"
        ),
        "superseded_field_name_2026_11_01": dict(
            SUPERSEDED_LARGEST_N_SAMPLES_FIELD_2026_11_01
        ),
        "files": rows,
    }


# --------------------------------------------------------------------------
# Wf-A164 -- the narrative self-consistency GUARD
#
# The guard is for the CLASS of defect, not for this one instance:
# a narrative string in a ``tag: MEASURED`` artifact must not carry a
# number that no machine field of the SAME artifact backs, and it must
# not quote a repo path that no machine field of the SAME artifact
# carries.  A token with no backing is a DEFECT and FAILS the run.
#
# Two design points that the Wf-A162 reference implementation taught:
#   * a narrative path that designates a LIST must resolve to its
#     ELEMENTS (a naive ``path + "."`` prefix silently classifies the
#     elements as machine fields and checks nothing);
#   * a narrative path that resolves to NOTHING must FAIL the guard
#     (a renamed key would otherwise disable the check silently).
# --------------------------------------------------------------------------

#: A number-like token: digits, optional thousands separators, optional
#: decimal part.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

#: A maximal run of "word" characters.  Used to MASK identifier-like
#: tokens (names that contain digits) before the number scan: ``Wf-A155``,
#: ``ETKDGv3`` and ``mol.yml`` are NAMES, not numeric claims.
_WORD_RE = re.compile(r"[A-Za-z0-9_.:@/+-]+")

#: A repo-relative path that may be quoted in prose.  Every such token in a
#: narrative must appear verbatim inside a machine string field of the same
#: artifact -- this is what makes a STALE POOL PATH a hard failure rather
#: than an unbacked number.
_REPO_PATH_RE = re.compile(
    r"(?:molmetal|metrics|configs|TODO|docs|paper|scripts)/[A-Za-z0-9_./-]*[A-Za-z0-9_]"
)

#: An arXiv identifier: FOUR digits, a dot, then FOUR or FIVE digits -- the
#: OLD-style arXiv number (``2508.12629``).  It is an IDENTIFIER, not a
#: measurement, so it must not be read as a numeric claim.
#:
#: The VERSIONED form (``2508.12629v1``) is unambiguous -- a decimal
#: measurement cannot carry a ``v<N>`` suffix -- so it is masked wherever
#: it appears.
_ARXIV_VERSIONED_ID_RE = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}v\d+(?![\d])")

#: The marker that says a BARE ``NNNN.NNNNN`` token is an arXiv id rather
#: than a decimal measurement.  The marker must be ADJACENT: a genuine
#: measurement that happens to have an arXiv-id shape (e.g. ``2508.12629``
#: seconds) is NOT preceded by the word "arXiv", so it is still read as a
#: claim and still has to be backed.  Masking the shape alone would
#: WEAKEN the guard, not strengthen it.
_ARXIV_MARKER_RE = re.compile(
    r"(?i)(arxiv(?:\.org)?[:\s]*)(\d{4}\.\d{4,5})(?![\d])"
)

#: The artifact's PROSE paths.  Every leaf under these is prose; its
#: numbers must be backed by a machine field.
NARRATIVE_PATHS: Tuple[str, ...] = (
    "purpose",
    "source_pool.composition.explicit_hydrogen_note",
    "source_pool.composition.metal_definition",
    "source_pool.pool_structure_note",
    "source_pool.pool_caveat",
    "protocol.runner_fails_closed",
    "protocol.execution",
    "protocol.row_timeout_policy",
    "protocol.row_timeout_rationale",
    "scope.sampling.method",
    "scope.sampling.population",
    "scope.label",
    "counts.strict_rate_cross_check",
    "counts.forcefield_parameterisation_note",
    "counts.denominators.strict.definition",
    "counts.denominators.scored.definition",
    "counts.denominators.scored_plus_embed_failed.definition",
    "counts.denominators.completed_only.definition",
    "counts.by_pool_fragment_count._why",
    "counts.by_pool_fragment_count._conditioning_variable",
    "counts.by_pool_epoch._why",
    "counts.by_pool_epoch._conditioning_variable",
    "conformer_sensitivity_probe.method",
    "conformer_sensitivity_probe.why",
    "conformer_sensitivity_probe.sample_method",
    "timeout_recheck.bias_direction",
    "headline.strict_denominator_definition",
    "headline.scored_denominator_definition",
    "headline.one_line",
    "headline.single_fragment_note",
    "not_a_restatement_of.end_to_end_median_headline",
    "not_a_restatement_of.single_pool_fragment_count",
    "not_a_restatement_of.prior_posebusters_artifacts",
    "published_cross_check.rule",
    "published_cross_check.verdict",
    "end_to_end_headline.definition",
    # Wf-A182: the DERIVED-value label.  It is prose about a number, so it
    # is checked like prose -- and it carries no numeral of its own.
    "end_to_end_headline.median_pct_derivation",
    "comparability_vs_frontier_band.statement",
    "comparability_vs_frontier_band.breaks",
    "limitations",
)

#: Paths whose STRING leaves are DATA, not prose (file names, paths, hashes,
#: SMILES, status vocabularies, tool versions).  A string leaf that is
#: neither under a narrative path nor under one of these is UNCLASSIFIED and
#: FAILS the guard -- that is the check that stops a new prose string from
#: being added without being audited.
IDENTIFIER_PATHS: Tuple[str, ...] = (
    "metric",
    "tag",
    "date",
    "wave",
    "branch",
    "script",
    "generated_at_utc",
    "status",
    "failure",
    "source_pool.path",
    "source_pool.sha256",
    "source_pool.columns",
    "source_pool.pool_kind",
    "source_pool.measured_profile",
    "protocol.embed_method",
    "protocol.optimize_method",
    "protocol.posebusters_config",
    "protocol.backend_module",
    "protocol.embed_random_seed_source",
    "protocol.device",
    "scope.kind",
    "scope.pool_sha256_at_run_time",
    "tool_versions",
    "conformer_sensitivity_probe.disagreements",
    "timeout_recheck.rows",
    "published_cross_check.published_artifacts_considered",
    "published_cross_check.matches",
    "published_cross_check.this_pool_name",
    # Wf-A182: the PROVENANCE BINDING for the three longest-pool-SMILES
    # call seconds.  It is DATA (which pool, which date, which tool, which
    # record), not prose -- and it is a machine STRING, so a path quoted in
    # the prose can be backed by it.
    "protocol.longest_pool_smiles_call_seconds_provenance",
    "end_to_end_headline.source",
    "end_to_end_headline.jsonpath",
    "prior_posebusters_artifacts_measured",
    "prose_backing_measurements",
    "comparability_vs_frontier_band.frontier_band",
    "examples",
    "pb_valid_smiles_sample",
    "narrative_self_consistency",
)

#: Wf-A181: the PUBLISHED artifact was produced by the PRE-Wf-A164
#: emitter, and that rendering named two of the prose leaves differently.
#: The alias says where the legacy rendering put the prose, so the legacy
#: prose is STILL CHECKED (its numbers must be backed) instead of being
#: left unclassified.  The alias only fires when the current key does not
#: resolve, so a fresh run is unaffected.
NARRATIVE_PATH_ALIASES: Dict[str, str] = {
    "not_a_restatement_of.end_to_end_median_headline": (
        "not_a_restatement_of.the_8_51_pct_headline"
    ),
    "not_a_restatement_of.single_pool_fragment_count": (
        "not_a_restatement_of.the_25_05_pct_fragment_count"
    ),
}

#: DATED PROVENANCE records: the Wf-A177 / Wf-A180 correction records the
#: artifact carries.  They are prose (they explain a correction), but
#: their numerals are DOCUMENT REFERENCES -- a date, a section number, a
#: rule number -- and not measurements of this pool, so they are not
#: number-checked.  They are classified explicitly rather than left
#: unclassified, because an unclassified string leaf is a guard failure.
PROVENANCE_PATHS: Tuple[str, ...] = (
    "comparability_vs_frontier_band.superseded_reading_2026_11_01",
    "comparability_vs_frontier_band.superseded_reading_2026_11_01_wfa180",
    "comparability_vs_frontier_band.superseded_reading_2026_11_01_wfa182",
    "published_cross_check_superseded_2026_11_01",
    "wfa181_narrative_backing_repair_2026_11_01",
    "wfa182_residual_defect_repair_2026_11_01",
)


def _leaf_map(artifact: Any) -> Dict[str, Any]:
    """Flatten an artifact to ``dotted.path -> leaf``."""
    out: Dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                walk(val, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for idx, val in enumerate(node):
                walk(val, f"{path}[{idx}]")
        else:
            out[path] = node

    walk(artifact, "")
    return out


def _resolve(artifact: Any, path: str) -> List[str]:
    """The leaf paths under a dotted narrative path.

    A path may designate a LEAF (``headline.one_line``), a MAPPING
    (``not_a_restatement_of`` -> ``not_a_restatement_of.<key>``) or a LIST
    (``limitations`` -> ``limitations[0]``, ``limitations[1]``, ...).  The
    list form matters: with a naive ``path + "."`` prefix the list elements
    would never match, they would be silently classified as MACHINE fields,
    and every number inside them would go unchecked.  That hole was found
    by a negative control in Wf-A162; Wf-A164 keeps the fix AND adds the
    "a narrative path that resolves to nothing FAILS the guard" rule.
    """
    leaves = _leaf_map(artifact)
    if path in leaves:
        return [path]
    prefixes = (path + ".", path + "[")
    return sorted(p for p in leaves if p.startswith(prefixes))


def _mask_identifier_tokens(text: str) -> str:
    """Blank out identifier-like tokens (names that contain digits).

    ``Wf-A155``, ``ETKDGv3``, ``mol.yml`` and ``posebusters/config/mol.yml``
    are NAMES, not numeric claims.  A token containing BOTH a letter and a
    digit is masked; a pure number is not, so ``8.51`` and ``2112`` are
    still read as claims.

    Wf-A181: an OLD-STYLE arXiv id (``NNNN.NNNNN``) is also an IDENTIFIER
    and not a measurement, but it is pure digits, so the letter+digit rule
    did not catch it -- ``FlowMol3 arXiv 2508.12629`` was reported as an
    unbacked number, which is a FALSE POSITIVE.  The id is now masked:

    * the VERSIONED form (``2508.12629v1``) is unambiguous and is masked
      wherever it appears;
    * the BARE form is masked only when the text marks it as an arXiv
      reference (the word ``arXiv`` immediately before it).  A genuine
      measurement that merely HAS that shape is NOT masked, so the guard
      keeps catching it.
    """

    def _repl(match: "re.Match[str]") -> str:
        token = match.group(0)
        if any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token):
            return " "
        return token

    masked = _WORD_RE.sub(_repl, text)
    # The versioned id first: the bare pattern would match its numeric
    # part and leave the ``v<N>`` suffix behind as a stray number.
    masked = _ARXIV_VERSIONED_ID_RE.sub(" ", masked)
    masked = _ARXIV_MARKER_RE.sub(lambda mo: mo.group(1) + " ", masked)
    return masked


def _parent_is_null(leaves: Dict[str, Any], path: str) -> bool:
    """True when the path's PARENT is an explicit ``null`` leaf.

    A narrative path under a parent the artifact explicitly recorded as
    ``null`` has NO prose to check: the rendering said "this block was
    not produced" (the published cross-check, the end-to-end headline).
    Requiring such a path to resolve would force the artifact to claim a
    measurement it never performed.  A RENAMED parent -- absent, or a
    mapping -- is still a hard failure, so the rename protection that
    this rule exists for is untouched.
    """
    parts = path.split(".")
    for index in range(len(parts) - 1, 0, -1):
        ancestor = ".".join(parts[:index])
        if ancestor in leaves and leaves[ancestor] is None:
            return True
    return False


def _standalone_in_any(token: str, strings: List[str]) -> bool:
    """True when ``token`` appears in one of ``strings`` as a STANDALONE
    number -- delimited by characters that are neither digits nor a decimal
    point.

    This is deliberately stricter than a substring test.  The token ``1``
    must NOT count as "backed" by the ``155`` in a file name; with a plain
    substring test it would, and the guard would be measurably looser than
    it looks.
    """
    pattern = re.compile(r"(?<![0-9.])" + re.escape(token) + r"(?![0-9.])")
    return any(pattern.search(text) for text in strings)


def _narrative_self_consistency(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """THE GUARD for the class of defect, not just this instance.

    Wf-A164 (2026-11-01): the tool hard-coded pool-specific claims as
    string literals inside a ``tag: MEASURED`` artifact.  Every one of them
    was true for the DEFAULT pool and FALSE for any other -- including the
    wrong pool PATH.  The class is stopped here:

    * every number rendered in a NARRATIVE string must be backed by a
      MACHINE field of the same artifact (as a standalone number inside a
      machine string, or equal -- to the token's own decimal count -- to a
      machine number);
    * every repo path quoted in a NARRATIVE string must appear verbatim in
      a machine string field of the same artifact;
    * every string leaf must be CLASSIFIED (narrative, identifier or
      dated provenance);
    * every designated narrative path must RESOLVE to at least one leaf.

    Wf-A181 (2026-11-01) added the two resolutions that the PUBLISHED
    (pre-Wf-A164) rendering needs, and neither of them WEAKENS the
    number guard:

    * a narrative path whose PARENT is an explicit ``null`` leaf is
      VACUOUSLY satisfied -- the rendering said "this block was not
      produced", so there is no prose there to check.  A RENAMED parent
      (absent, or a mapping) is still a hard failure, so the rename
      protection is untouched;
    * a narrative path that does not resolve may ALIAS onto the key the
      legacy rendering used, so the legacy prose is checked rather than
      skipped.

    A token with no backing is a DEFECT and fails the run.
    """
    leaves = _leaf_map(artifact)

    narrative_leaves: set = set()
    unresolved_paths: List[str] = []
    for path in NARRATIVE_PATHS:
        resolved = _resolve(artifact, path)
        if not resolved:
            alias = NARRATIVE_PATH_ALIASES.get(path)
            if alias:
                resolved = _resolve(artifact, alias)
        if not resolved and not _parent_is_null(leaves, path):
            unresolved_paths.append(path)
        narrative_leaves.update(resolved)

    identifier_leaves: set = set()
    for path in IDENTIFIER_PATHS:
        identifier_leaves.update(_resolve(artifact, path))

    provenance_leaves: set = set()
    for path in PROVENANCE_PATHS:
        provenance_leaves.update(_resolve(artifact, path))

    machine = {p: v for p, v in leaves.items() if p not in narrative_leaves}
    machine_numbers: List[float] = []
    machine_strings: List[str] = []
    for value in machine.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            machine_numbers.append(float(value))
        elif isinstance(value, str):
            machine_strings.append(value)

    unbacked: Dict[str, List[str]] = {}
    stale_paths: Dict[str, List[str]] = {}
    n_prose_numbers = 0
    for path in sorted(narrative_leaves):
        text = leaves.get(path)
        if not isinstance(text, str):
            continue
        # (a) PATH guard -- a repo path quoted in prose must be a machine field
        for token in _REPO_PATH_RE.findall(text):
            if not any(token in candidate for candidate in machine_strings):
                stale_paths.setdefault(token, []).append(path)
        # (b) NUMBER guard
        for token in _NUM_RE.findall(_mask_identifier_tokens(text)):
            n_prose_numbers += 1
            norm = token.replace(",", "")
            try:
                value = float(norm)
            except ValueError:  # pragma: no cover - defensive
                continue
            decimals = len(norm.split(".")[1]) if "." in norm else 0
            tol = 0.5 * (10.0 ** (-decimals)) if decimals else 0.0
            backed = _standalone_in_any(token, machine_strings)
            if not backed:
                backed = any(
                    abs(mv - value) <= tol + 1e-12 for mv in machine_numbers
                )
            if not backed:
                unbacked.setdefault(token, []).append(path)

    # (c) CLASSIFICATION guard -- every string leaf must be accounted for.
    unclassified: Dict[str, str] = {}
    for path, value in sorted(leaves.items()):
        if not isinstance(value, str):
            continue
        if (
            path in narrative_leaves
            or path in identifier_leaves
            or path in provenance_leaves
        ):
            continue
        unclassified[path] = value

    passed = not unbacked and not stale_paths and not unclassified and not unresolved_paths
    return {
        "rule": (
            "every number rendered in a narrative string must be backed by a "
            "machine field of the same artifact -- as a standalone number "
            "inside a machine string, or equal (to the token's own decimal "
            "count) to a machine number; every repo path quoted in a "
            "narrative must appear in a machine string field; every string "
            "leaf must be classified as narrative, identifier or dated "
            "provenance; and every designated narrative path must resolve "
            "to at least one leaf (a path whose parent is an explicit null "
            "is vacuously satisfied)"
        ),
        "narrative_paths": list(NARRATIVE_PATHS),
        "narrative_path_aliases": dict(NARRATIVE_PATH_ALIASES),
        "identifier_paths": list(IDENTIFIER_PATHS),
        "provenance_paths": list(PROVENANCE_PATHS),
        "n_prose_leaves": len(narrative_leaves),
        "n_prose_numbers": n_prose_numbers,
        "n_machine_leaves": len(machine),
        "n_machine_numbers": len(machine_numbers),
        "n_machine_strings": len(machine_strings),
        "n_unclassified_string_leaves": len(unclassified),
        "unresolved_narrative_paths": list(unresolved_paths),
        "unbacked_numbers": {
            tok: sorted(set(paths)) for tok, paths in sorted(unbacked.items())
        },
        "stale_paths": {
            tok: sorted(set(paths)) for tok, paths in sorted(stale_paths.items())
        },
        "unclassified_leaves": {
            path: value for path, value in sorted(unclassified.items())
        },
        "pass": passed,
    }


# ----------------------------------------------------------------
# Small IO helpers
# ----------------------------------------------------------------
#: Wf-A189 (2026-11-01) -- the dated rule-#6 records this
#: artefact carries.  AGENTS.md §5.5 rule #6 requires every
#: superseded rendering to survive VERBATIM; emitting them
#: HERE is what makes a re-run REPRODUCE the published
#: artefact instead of silently DROPPING the dated history.

COMPARABILITY_SUPERSEDED_READING_2026_11_01: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A177',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'superseded': [{'id': 3, 'field': 'frontier', 'text': 'measured on the published POST-PROCESSED stream'}, {'id': 3, 'field': 'consequence', 'text': 'our number is not post-processed and theirs is'}],
    'why_superseded': "the published band-provenance claim asserted ONE of TWO competing readings as settled.  The repo holds two competing readings of the published 92.7-94 % band's source cells (frontier_matrix.json `band_provenance` vs `frontier_axis_mapping`), and they do not even agree on the numerals.  The corrected text asserts only the invariants that hold under BOTH readings: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves (including the wall-clock-censored protocol.wall_seconds and counts.n_timeout) are UNCHANGED — the tool was NOT re-run, because a re-run would produce a NEW measurement, not a correction.",
}

COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA180: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A180',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'superseded': [{'id': 3, 'field': 'frontier', 'text': "the published band's two endpoints are RAW (post_processed: false in the band-provenance record); which frontier cells the band names is NOT settled on disk -- the repo holds TWO competing provenance readings for it, both recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- under `band_provenance` and under `frontier_axis_mapping` -- and they do not even agree on the numerals"}],
    'why_superseded': 'the pointer `frontier_axis_mapping` is NOT a JSON key of molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json — a recursive key-walk finds that token only as a STRING VALUE, twice (inside band_provenance.competing_reading_and_what_this_does_not_settle.canonical_reading_safe_band_statement and inside ...readings[1].locator).  The record\'s OWN band_provenance.competing_reading_and_what_this_does_not_settle.line_number_note states: "The ROBUST locators are the JSON PATHS -- `band_provenance` and `sources`.`EF-TALFM` -- not the line numbers."  The corrected string names the two REAL key paths.  The CLAIM is unchanged: two competing provenance readings, numerals disagree, invariants hold under BOTH.',
    'note_on_the_wfa177_record': "the sibling key `superseded_reading_2026_11_01` (Wf-A177's own record) is preserved UNTOUCHED — including its `why_superseded` prose, which repeats the same defective pointer.  That defect is corrected ADDITIVELY here rather than by editing a dated record.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves (including the wall-clock-censored protocol.wall_seconds and counts.n_timeout) are UNCHANGED — the tool was NOT re-run, because a re-run would produce a NEW measurement, not a correction.",
}

COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA182: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A182',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'superseded': [{'id': 3, 'field': 'frontier', 'text': "the published band's two endpoints are RAW (post_processed: false in the band-provenance record); which frontier cells the band names is NOT settled on disk -- the repo holds TWO competing provenance readings for it, both recorded in the frontier post-processing matrix record (`frontier_matrix.json`) -- the first under `band_provenance`, the second under `sources`.`EF-TALFM` -- and they do not even agree on the numerals"}, {'id': 3, 'field': 'consequence', 'text': 'post-processing is a THIRD, separate axis on top of the METRIC break, not the break itself: both arms are RAW, and the decisive break is that under BOTH readings the band is external, is NOT one homogeneous quantity, and is NOT on our scale (OUR number is a connected-component count on the RAW emitted SMILES, and nothing else)'}, {'id': 1, 'field': 'ours', 'text': 'ONE pool of 2112 rows from THREE epoch-wise decodes of ONE checkpoint (704 rows each, 6 seeds per epoch; molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv)'}],
    'why_superseded': "the POST-PROCESSING break asserted, WITHOUT qualification, that the published band's two endpoints are RAW and that both arms are RAW. That is FALSE under the SECOND competing provenance reading the frontier post-processing matrix record itself holds -- its `sources`.`EF-TALFM` path (EF-TALFM arXiv:2609.08333v4, Table 3(a)) pairs a RAW connectivity cell (`C.` = 92.98 %), counted BEFORE the largest-component retention, with a POST-PROCESSED companion cell (`N. & U. & S.` = 94.33 %), so the frontier arm is MIXED there. The blanket claim is the MIRROR of the blanket 'the band is POST-PROCESSED' claim that Wf-A175 already refuted. The corrected text scopes `post_processed: false` to the `band_provenance` reading and states that whether the band AS A WHOLE is RAW is NOT settled on disk; the invariants that hold under BOTH readings (the band is external, is NOT one homogeneous quantity, and is NOT on our scale) are untouched, and NO new numeral is introduced. The DENOMINATOR break's `ours` text is superseded for a different reason: the published rendering carried a HAND-WRITTEN description of the pool, and the producer's COMPUTED rendering replaces it -- the same pool and the same numbers, derived from the pool on disk instead of transcribed.",
    'number_preserving': "TEXT-ONLY patch: this artefact's numeric leaves are UNCHANGED apart from the two values named in the D2 and D4 records — the tool was NOT re-run, because a re-run would produce a NEW measurement, not a correction.",
}

#: Wf-A194 (2026-11-02) -- the dated rule-#6 record for the MODULE DOCSTRING
#: this wave replaced.  AGENTS.md §5.5 rule #6 requires every superseded
#: rendering to survive VERBATIM; a DATED NAMED CONSTANT in the producer is
#: the sanctioned carrier for a rendering that lives in the producer's own
#: prose (the module docstring is NOT read by the emitter, so it cannot be
#: carried in the artefact -- and publishing it there would change the
#: artefact's bytes, its sha256 and its registry row for no measurement
#: reason).

DOCSTRING_SUPERSEDED_2026_11_02_WFA194: Dict[str, str] = {
    'date': '2026-11-02',
    'wave': 'Wf-A194',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'supersedes': 'the module docstring of molmetal/scripts/wf_a155_posebusters_validity.py (the `WHAT THIS EXISTS FOR` paragraph)',
    'field': 'module docstring',
    'rendering': "The frontier band we are compared against (92.7--94 % connected / 99.9 % valid /\n91.9 % PoseBusters-valid, per arXiv 2609.08333 and FlowMol3 arXiv 2508.12629)\nis quoted as a *percentage*.",
    'why_superseded': "the superseded rendering asserted ONE provenance reading as settled: it attributed the 92.7--94 % connected band to arXiv 2609.08333 (EF-TALFM) as a fact, with no reading named.  The repo's own frontier post-processing matrix record (molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json) holds TWO competing readings of which frontier cells the band names -- the first under `band_provenance` (GeoLDM Validity & Uniq + HLTF PB-Valid) and the second under `sources`.`EF-TALFM` (EF-TALFM's own RAW connectivity cell + a post-processed companion cell) -- and they do not even agree on the numerals, so the band's source cells are NOT settled on disk.  The repo's own frontier-labelling record (molmetal/scripts/wfa161_frontier_labelling.py) states that ONLY the invariants that hold under BOTH readings may be published: the band is EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale.  The live docstring is now READING-SCOPED; it introduces NO numeral and keeps every numeral the superseded rendering carried.",
    'number_preserved': "TEXT-ONLY patch to the module docstring.  The published artefact metrics/by_metric/wfa155_posebusters_validity.json is UNCHANGED by this wave -- the module docstring is not read by the emitter, so the emitted artefact is byte-identical before and after: every numeric leaf and the artefact's sha256 are unchanged and its registry row stays valid.",
    'emitted_into_the_artifact': "no -- deliberately.  A module docstring is not part of the emitted artefact; adding it to the artefact would change the artefact's bytes (and therefore its sha256 and its registry row) without changing a single measurement.",
}

PUBLISHED_CROSS_CHECK_SUPERSEDED_2026_11_01: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A182',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'superseded': {'published_cross_check': None},
    'why_superseded': "the field was published as `null`, while the producer's own `_published_cross_check()` computes a POPULATED block for this pool. The `null` is also what made the `published_cross_check.*` narrative paths vacuously satisfied under the guard's null-parent rule, so a field that was FALSE was doing guard-work. Wf-A182 replaces it with the producer's COMPUTED block.",
    'replaced_by': 'the block the producer computes for this pool, reproduced verbatim in `published_cross_check`: one published artifact names THIS pool and its published rate is reproduced bit-for-bit',
    'number_preserving': 'the block is COMPUTED from the published artifacts on disk — `metrics/by_metric/wfa155_ligand_connected_rate.json` and `metrics/by_metric/cfm_fragmentation_postproc_arm.json` — and NOT re-measured: the tool was NOT re-run.',
}

WFA181_NARRATIVE_BACKING_REPAIR_2026_11_01: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A181',
    'rule': 'AGENTS.md §5.5 rule #6 -- the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'what_was_added': ['comparability_vs_frontier_band.frontier_band.frontier_denominator_draws', 'comparability_vs_frontier_band.frontier_band.frontier_denominator_draws_min', 'comparability_vs_frontier_band.frontier_band.frontier_denominator_draws_max', 'protocol.longest_pool_smiles_call_seconds', 'scope.label', 'end_to_end_headline', 'published_cross_check', 'prior_posebusters_artifacts_measured', 'prose_backing_measurements', 'narrative_self_consistency'],
    'why': 'the artifact carries its OWN narrative self-consistency guard (_narrative_self_consistency in the producer), and the published rendering FAILED it: six numerals were rendered in prose with no machine field of this same artifact behind them (0.846, 0.6875, 15,000, 60.5, 86.7, and the FlowMol3 arXiv id). The numerals are MEASURED quantities -- dropping them would destroy measured information -- so the repair is ADDITIVE machine fields that carry the same values, not a deletion',
    'the_arxiv_id': "the FlowMol3 arXiv id (the digits are deliberately NOT repeated here, so that the id is masked by the guard's identifier rule rather than accidentally backed by this record: a genuine measurement with the same shape must still be caught) is an IDENTIFIER, not a measurement. The producer's guard now MASKS arXiv ids before the number scan -- the versioned form everywhere, the bare form only when the text marks it as an arXiv reference -- so the id is no longer reported as an unbacked number, while a genuine measurement that merely has that shape is still checked",
    'no_existing_leaf_changed': 'TEXT-ONLY-plus-machine-fields patch: every leaf that was present before this patch is present afterwards with the SAME value; the leaves listed under what_was_added are NEW. The tool was NOT re-run -- a re-run would produce a NEW measurement, not a correction -- so every numeric leaf of the published run is UNCHANGED',
    'values_are_the_prose_values': 'every value added is LITERALLY the number the prose already renders (86.7, 60.5, 0.1, 15000, 0.846, 0.6875) or a number read from disk at patch time (the end-to-end median and run count, the prior-artifact rates and sample sizes). Nothing was rounded, re-derived or invented',
    'disclosure_2026_11_01_wfa182': {'date': '2026-11-01', 'wave': 'Wf-A182', 'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated', 'what': "the `published_cross_check` field listed in `what_was_added` above was added with the value `null`, while the producer's own `_published_cross_check()` computes a POPULATED block for this pool", 'why': "a `null` field is what made the `published_cross_check.*` narrative paths vacuously satisfied under the guard's null-parent rule, so the published field was doing guard-work while being FALSE. Wf-A182 replaces the `null` with the producer's COMPUTED block — the `null` is preserved verbatim under `published_cross_check_superseded_2026_11_01` — and the computed block ALSO passes the guard, so the repair does not weaken it"},
}

WFA182_RESIDUAL_DEFECT_REPAIR_2026_11_01: Dict[str, str] = {
    'date': '2026-11-01',
    'wave': 'Wf-A182',
    'rule': 'AGENTS.md §5.5 rule #6 — the superseded rendering is preserved VERBATIM and the correction is additive + dated',
    'what_was_repaired': ['D1 — the three longest-pool-SMILES call seconds were a module CONSTANT that certified a measurement for whatever pool the artifact names; they now carry a provenance binding (the pool path and sha256 they were measured on, the measurement date, the measurement record and the measurement tool) and are emitted ONLY for the pool they were measured on', "D2 — `published_cross_check` was published as `null` while the producer computes a populated block; it now carries the producer's computed block, with the `null` preserved verbatim", 'D3 — `end_to_end_headline.median_pct` is a unit conversion of `end_to_end_headline.median`, not a value read from disk; it is now labelled as derived, in the machine field `end_to_end_headline.median_pct_derivation`', 'D4 — `prior_posebusters_artifacts_measured.largest_n_samples` was an unscoped name for a PoseBusters-candidate-scoped value; it is renamed `largest_n_samples_among_posebusters_candidates`, the scope is carried in `largest_n_samples_scope`, and the old name and value are preserved verbatim', "D5 — the Wf-A181 record's `what_was_added` listed `published_cross_check` as added without disclosing that the value was `null`; the disclosure is appended under `wfa181_narrative_backing_repair_2026_11_01.disclosure_2026_11_01_wfa182`", "C1 — the POST-PROCESSING comparability break asserted that the published band's two endpoints are RAW and that both arms are RAW; the claim is now scoped to the `band_provenance` reading and states that whether the band AS A WHOLE is RAW is NOT settled on disk"],
    'why': "the adversarial verify of wave Wf-A181 found five residual defects in this artifact and its producer, two of them MATERIAL: a hard-coded constant that certified a measurement (D1) and a published machine field that was FALSE while doing guard-work (D2). The remaining three (D3, D4, D5) and the POST-PROCESSING break's blanket RAW claim (C1) were MINOR but each is a claim a `tag: MEASURED` artifact must not make without the evidence for it.",
    'number_preserving': "no numeric leaf of the MEASUREMENT moved: the three longest-pool-SMILES call seconds keep their values, the end-to-end median and its derived percent keep their values, the counts block is untouched, and the tool was NOT re-run — a re-run would produce a NEW measurement, not a correction. The numeric leaves that DID change are the two named in D2 and D4 (both old values preserved verbatim in their dated records) and the guard block's OWN counters (`n_prose_leaves`, `n_prose_numbers`, `n_machine_leaves`, `n_machine_numbers`, `n_machine_strings`), which describe the guard's view of this very artifact; the guard block as published before this patch is preserved verbatim under `narrative_self_consistency.superseded_2026_11_01_wfa182`.",
}

#: Wf-A189 (2026-11-01) -- the Wf-A182-era guard block, preserved
#: VERBATIM per AGENTS.md §5.5 rule #6 and re-emitted here so that a
#: re-run REPRODUCES the published artefact's guard block instead of
#: dropping the superseded guard output.
SUPERSEDED_GUARD_2026_11_01_WFA182: Dict[str, Any] = {
    'rule': "every number rendered in a narrative string must be backed by a machine field of the same artifact -- as a standalone number inside a machine string, or equal (to the token's own decimal count) to a machine number; every repo path quoted in a narrative must appear in a machine string field; every string leaf must be classified as narrative, identifier or dated provenance; and every designated narrative path must resolve to at least one leaf (a path whose parent is an explicit null is vacuously satisfied)",
    'narrative_paths': ['purpose', 'source_pool.composition.explicit_hydrogen_note', 'source_pool.composition.metal_definition', 'source_pool.pool_structure_note', 'source_pool.pool_caveat', 'protocol.runner_fails_closed', 'protocol.execution', 'protocol.row_timeout_policy', 'protocol.row_timeout_rationale', 'scope.sampling.method', 'scope.sampling.population', 'scope.label', 'counts.strict_rate_cross_check', 'counts.forcefield_parameterisation_note', 'counts.denominators.strict.definition', 'counts.denominators.scored.definition', 'counts.denominators.scored_plus_embed_failed.definition', 'counts.denominators.completed_only.definition', 'counts.by_pool_fragment_count._why', 'counts.by_pool_fragment_count._conditioning_variable', 'counts.by_pool_epoch._why', 'counts.by_pool_epoch._conditioning_variable', 'conformer_sensitivity_probe.method', 'conformer_sensitivity_probe.why', 'conformer_sensitivity_probe.sample_method', 'timeout_recheck.bias_direction', 'headline.strict_denominator_definition', 'headline.scored_denominator_definition', 'headline.one_line', 'headline.single_fragment_note', 'not_a_restatement_of.end_to_end_median_headline', 'not_a_restatement_of.single_pool_fragment_count', 'not_a_restatement_of.prior_posebusters_artifacts', 'published_cross_check.rule', 'published_cross_check.verdict', 'end_to_end_headline.definition', 'comparability_vs_frontier_band.statement', 'comparability_vs_frontier_band.breaks', 'limitations'],
    'narrative_path_aliases': {'not_a_restatement_of.end_to_end_median_headline': 'not_a_restatement_of.the_8_51_pct_headline', 'not_a_restatement_of.single_pool_fragment_count': 'not_a_restatement_of.the_25_05_pct_fragment_count'},
    'identifier_paths': ['metric', 'tag', 'date', 'wave', 'branch', 'script', 'generated_at_utc', 'status', 'failure', 'source_pool.path', 'source_pool.sha256', 'source_pool.columns', 'source_pool.pool_kind', 'source_pool.measured_profile', 'protocol.embed_method', 'protocol.optimize_method', 'protocol.posebusters_config', 'protocol.backend_module', 'protocol.embed_random_seed_source', 'protocol.device', 'scope.kind', 'scope.pool_sha256_at_run_time', 'tool_versions', 'conformer_sensitivity_probe.disagreements', 'timeout_recheck.rows', 'published_cross_check.published_artifacts_considered', 'published_cross_check.matches', 'published_cross_check.this_pool_name', 'end_to_end_headline.source', 'end_to_end_headline.jsonpath', 'prior_posebusters_artifacts_measured', 'prose_backing_measurements', 'comparability_vs_frontier_band.frontier_band', 'examples', 'pb_valid_smiles_sample', 'narrative_self_consistency'],
    'provenance_paths': ['comparability_vs_frontier_band.superseded_reading_2026_11_01', 'comparability_vs_frontier_band.superseded_reading_2026_11_01_wfa180', 'wfa181_narrative_backing_repair_2026_11_01'],
    'n_prose_leaves': 64,
    'n_prose_numbers': 87,
    'n_machine_leaves': 734,
    'n_machine_numbers': 284,
    'n_machine_strings': 343,
    'n_unclassified_string_leaves': 0,
    'unresolved_narrative_paths': [],
    'unbacked_numbers': {},
    'stale_paths': {},
    'unclassified_leaves': {},
    'pass': True,
}


def sha256_file(path: "os.PathLike[str] | str") -> str:
    """Return the sha256 hex digest of ``path`` (streamed, no full read)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pool(path: "os.PathLike[str] | str") -> Tuple[List[str], List[str]]:
    """Read the pool CSV, preserving file order.

    Returns ``(smiles_list, column_names)``.  The header row is consumed by
    :class:`csv.DictReader`; only DATA rows are returned, so the row count
    here is the "n_data_rows" quoted in the artifact -- computed, never
    asserted, and different for a different pool.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        smiles: List[str] = []
        for row in reader:
            value = row.get("smiles")
            if value is None:
                continue
            smiles.append(value)
    return smiles, columns


# ----------------------------------------------------------------
# Per-row classification
# ----------------------------------------------------------------
def pool_composition(smiles: Sequence[str]) -> Dict[str, Any]:
    """Measure the pool's metal content and explicit-hydrogen content.

    Pure w.r.t. the filesystem -- it only reads the SMILES strings.  This
    is the measured backbone of the artifact's central honest-negative:
    PoseBusters has no metal-coordination-sphere check, and MMFF94/UFF
    cannot parameterise most metal centres, so a pool that is 100 % metal
    is a pool whose PB verdict is largely a FORCE-FIELD verdict.

    Metals are identified by atomic number (all elements outside the
    non-metal set), which covers the alkali/alkaline, transition,
    lanthanide/actinide, and post-transition metals.
    """
    from rdkit import Chem  # local import: keeps module import light
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    metal_atomic_numbers = {
        3, 4, 11, 12, 13,
        *range(19, 32),
        *range(37, 52),
        *range(55, 84),
        *range(87, 104),
    }
    n_rows_with_metal = 0
    total_metal_atoms = 0
    metal_symbol_counts: Counter = Counter()
    n_rows_parsed = 0
    for row_smiles in smiles:
        if "[H]" in row_smiles:
            pass
        mol = Chem.MolFromSmiles(row_smiles)
        if mol is None:
            continue
        n_rows_parsed += 1
        row_metal_atoms = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in metal_atomic_numbers:
                row_metal_atoms += 1
                metal_symbol_counts[atom.GetSymbol()] += 1
        if row_metal_atoms:
            n_rows_with_metal += 1
            total_metal_atoms += row_metal_atoms
    n_explicit_h_rows = sum(1 for row_smiles in smiles if "[H]" in row_smiles)
    return {
        "n_rows_parsed_by_rdkit": n_rows_parsed,
        "n_rows_with_at_least_one_metal_atom": n_rows_with_metal,
        "fraction_rows_with_a_metal_atom": (
            n_rows_with_metal / len(smiles) if smiles else 0.0
        ),
        "total_metal_atoms": total_metal_atoms,
        "metal_element_counts": dict(sorted(metal_symbol_counts.items())),
        "n_rows_with_an_explicit_hydrogen_atom_in_the_smiles": n_explicit_h_rows,
        "explicit_hydrogen_note": (
            "0 means the pool SMILES carry IMPLICIT hydrogens only; RDKit "
            "AddHs is applied inside the runner's embedding step. The "
            "frontier dataset carries explicit hydrogens, which is one of "
            "the four comparability breaks."
        ),
        "metal_definition": (
            "atomic number in the metal set (all elements except B, C, N, O, "
            "F, Si, P, S, Cl, Ge, As, Se, Br, Te, I, Xe, At, Rn)"
        ),
    }


def classify_result(
    result: Optional[Dict[str, Any]],
    *,
    parseable: bool,
) -> str:
    """Map a :func:`check_posebusters` result dict onto a status string.

    Pure function -- no I/O, no RDKit, no PoseBusters.  Unit-tested
    directly.

    * ``parseable=False``  -> ``unparseable`` (the row was skipped before
      PoseBusters was ever called).
    * ``result["skipped"]`` -> ``skipped_posebusters_missing`` (the package
      is not importable on this host).
    * ``failures == ["embed_failed"]`` -> ``embed_failed``.  This is the
      "parsed fine but RDKit could not produce a conformer" case; it is
      distinct from ``unparseable`` because we parse independently first.
    * ``pb_valid is True`` -> ``pb_valid``.
    * everything else -> ``pb_invalid`` (fail closed).
    """
    if not parseable:
        return STATUS_UNPARSEABLE
    if not isinstance(result, dict):
        return STATUS_EMBED_FAILED
    if result.get("skipped"):
        return STATUS_SKIPPED
    failures = list(result.get("failures") or [])
    if failures and set(failures) == {"embed_failed"}:
        return STATUS_EMBED_FAILED
    if result.get("pb_valid") is True:
        return STATUS_PB_VALID
    return STATUS_PB_INVALID


def _parseable(smiles: str) -> bool:
    """Return ``True`` iff RDKit can parse ``smiles`` into a molecule."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _run_one(
    payload: Tuple[int, str, int, str, str],
) -> Dict[str, Any]:
    """Worker: check ONE pool row with PoseBusters.

    Module-level so it can be dispatched to a worker process; in the
    shipped configuration it runs inside the isolated child process that
    :func:`_run_one_subprocess` spawns.
    """
    row_index, smiles, n_conformers, embed_method, optimize_method = payload
    from molmetal.validation.posebusters_runner import check_posebusters

    started = time.perf_counter()
    parseable = _parseable(smiles)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    if parseable:
        try:
            result = check_posebusters(
                smiles,
                n_conformers=n_conformers,
                embed_method=embed_method,
                optimize_method=optimize_method,
            )
        except Exception as exc:  # pragma: no cover - defensive
            error = f"{type(exc).__name__}: {exc}"
            result = None
    elapsed = time.perf_counter() - started

    status = classify_result(result, parseable=parseable)
    if error is not None and status == STATUS_PB_INVALID:
        status = STATUS_EMBED_FAILED

    checks = dict(result.get("checks") or {}) if isinstance(result, dict) else {}
    return {
        "row_index": int(row_index),
        "smiles": smiles,
        "status": status,
        "pb_valid": (result.get("pb_valid") if isinstance(result, dict) else None),
        "n_conformers": int(result.get("n_conformers") or 0) if isinstance(result, dict) else 0,
        "n_pb_rows": len(result.get("row_checks") or []) if isinstance(result, dict) else 0,
        "checks": checks,
        "failures": list(result.get("failures") or []) if isinstance(result, dict) else [],
        "missing_checks": (
            list(result.get("missing_checks") or []) if isinstance(result, dict) else []
        ),
        "elapsed_s": float(elapsed),
        "error": error,
    }


# ----------------------------------------------------------------
# Child-process execution (hard per-row wall-clock cap)
# ----------------------------------------------------------------
def _child_main(argv: Sequence[str]) -> int:
    """Child mode: check ONE molecule and print its JSON payload.

    Run as ``<python> <this-script> --child <smiles> <n_conformers>
    <embed_method> <optimize_method>``.  The parent reads the line that
    starts with :data:`CHILD_MARKER`.

    Why a separate process: a handful of pool molecules have pathological
    ETKDG geometry -- MEASURED 2026-11-01, the three longest SMILES in the
    pool took 86.7 s / 60.5 s / 0.1 s on one molecule-level call -- and an
    in-process run has no way to abandon one.  A child process can be
    killed on a deadline, which is what makes a FULL-POOL run terminate at
    all.
    """
    smiles = argv[0]
    n_conformers = int(argv[1])
    embed_method = argv[2]
    optimize_method = argv[3]
    record = _run_one((0, smiles, n_conformers, embed_method, optimize_method))
    sys.stdout.write(CHILD_MARKER + json.dumps(record) + "\n")
    sys.stdout.flush()
    return 0


def _run_one_subprocess(
    smiles: str,
    *,
    n_conformers: int,
    embed_method: str,
    optimize_method: str,
    timeout_s: float,
) -> Dict[str, Any]:
    """Run one molecule in an isolated child process with a hard timeout.

    Returns the same record shape as :func:`_run_one`, with
    ``status == "timeout"`` when the child exceeded ``timeout_s``.
    """
    script = Path(__file__).resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    cmd = [
        sys.executable,
        str(script),
        "--child",
        smiles,
        str(n_conformers),
        embed_method,
        optimize_method,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "row_index": -1,
            "smiles": smiles,
            "status": STATUS_TIMEOUT,
            "pb_valid": False,
            "n_conformers": 0,
            "n_pb_rows": 0,
            "checks": {},
            "failures": ["row_timeout"],
            "missing_checks": [],
            "elapsed_s": time.perf_counter() - started,
            "error": f"timeout after {timeout_s}s",
        }

    elapsed = time.perf_counter() - started
    payload = None
    for line in (completed.stdout or "").splitlines():
        if line.startswith(CHILD_MARKER):
            try:
                payload = json.loads(line[len(CHILD_MARKER) :])
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        return {
            "row_index": -1,
            "smiles": smiles,
            "status": STATUS_TIMEOUT,
            "pb_valid": False,
            "n_conformers": 0,
            "n_pb_rows": 0,
            "checks": {},
            "failures": ["child_failed"],
            "missing_checks": [],
            "elapsed_s": elapsed,
            "error": (
                f"child exited {completed.returncode} without a payload "
                f"(stderr tail: {(completed.stderr or '')[-300:]!r})"
            ),
        }
    payload["row_index"] = -1
    payload["elapsed_s"] = elapsed
    return payload


# ----------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------
def strict_rate_via_repo_helper(records: Sequence[Dict[str, Any]]) -> float:
    """Cross-check the strict rate with the repo's OWN aggregation helper.

    ``molmetal.validation.posebusters_runner.pb_pass_rate`` is the existing
    surface for "what fraction of these rows did PoseBusters pass".  It keeps
    failures, skips and missing verdicts in the denominator -- which is
    exactly the STRICT denominator this artifact uses.  Reconstructing
    runner-shaped result dicts from the records and calling it proves the
    strict rate in this artifact is the SAME number the existing repo
    surface produces, not a re-derivation that happens to agree.

    Pure w.r.t. the filesystem.  Unit-tested directly.
    """
    from molmetal.validation.posebusters_runner import pb_pass_rate

    shaped = [
        {
            "pb_valid": (
                bool(record.get("pb_valid"))
                if record.get("status") in (STATUS_PB_VALID, STATUS_PB_INVALID)
                else False
            ),
            "skipped": record.get("status") == STATUS_SKIPPED,
        }
        for record in records
    ]
    return float(pb_pass_rate(shaped))


def read_pool_epoch_labels(path: "os.PathLike[str] | str") -> List[str]:
    """Read the pool's own ``epoch`` column, in file order.

    The published pool is NOT a single decode: MEASURED 2026-11-01, the
    ``epoch`` column holds 704 rows for epoch 0, 704 for epoch 1 and 704 for
    epoch 2 -- three epoch-wise decodes of ONE checkpoint
    (``cfm_yuelbond_wasserstein_seed0``), 6 seeds per epoch.  Reading the
    column instead of asserting the structure keeps the artifact's
    provenance measured.
    """
    labels: List[str] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels.append((row.get("epoch") or "").strip() or "unknown")
    return labels


def read_pool_fragment_counts(path: "os.PathLike[str] | str") -> List[int]:
    """Read the pool's own ``n_frags_raw`` column (its 2D graph-fragment count).

    Returns one integer per data row, in file order.  ``-1`` marks a row
    whose column is missing or unparsable.  This is the conditioning
    variable for the single-fragment vs multi-fragment split: PoseBusters'
    ``all_atoms_connected`` check is exactly the check a multi-fragment row
    cannot pass, so the overall rate mixes a GEOMETRY question with a
    FRAGMENTATION question and the conditional rate separates them.
    """
    counts: List[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("n_frags_raw") or "").strip()
            try:
                counts.append(int(raw))
            except (TypeError, ValueError):
                counts.append(-1)
    return counts


def aggregate(
    records: Sequence[Dict[str, Any]],
    *,
    row_timeout_s: Optional[float] = None,
    pool_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate per-row records into the counts block of the artifact.

    Pure function -- no I/O.  Unit-tested directly.

    Two rates are produced and BOTH are reported:

    * ``pb_valid_rate_strict`` -- denominator is every row in scope
      (unparseable, embed-failed and skipped rows all count as NOT valid).
    * ``pb_valid_rate_scored`` -- denominator is the rows that produced a
      PoseBusters verdict (``pb_valid`` + ``pb_invalid``).

    A third number, ``pb_valid_rate_scored_plus_embed_failed``, treats an
    embed failure as a failure while still excluding the unparseable and
    skipped rows; it is reported for completeness and is NOT a headline.
    """
    n_rows = len(records)
    counts = Counter(record["status"] for record in records)
    n_pb_valid = int(counts.get(STATUS_PB_VALID, 0))
    n_pb_invalid = int(counts.get(STATUS_PB_INVALID, 0))
    n_embed_failed = int(counts.get(STATUS_EMBED_FAILED, 0))
    n_unparseable = int(counts.get(STATUS_UNPARSEABLE, 0))
    n_skipped = int(counts.get(STATUS_SKIPPED, 0))
    n_timeout = int(counts.get(STATUS_TIMEOUT, 0))
    n_scored = n_pb_valid + n_pb_invalid

    strict_denominator = n_rows
    scored_plus_embed_denominator = n_scored + n_embed_failed

    # Per-check breakdown over the SCORED rows only -- an unparseable row
    # has no check verdicts at all, so counting it under every check would
    # silently inflate the failure counts.
    check_not_true: Counter = Counter()
    check_false: Counter = Counter()
    check_missing: Counter = Counter()
    conformer_hist: Counter = Counter()
    pb_row_hist: Counter = Counter()
    partial_embed = 0
    for record in records:
        if record["status"] not in (STATUS_PB_VALID, STATUS_PB_INVALID):
            continue
        checks = record.get("checks") or {}
        for name, value in checks.items():
            if value is True:
                continue
            check_not_true[name] += 1
            if value is False:
                check_false[name] += 1
            else:
                check_missing[name] += 1
        n_conf = int(record.get("n_conformers") or 0)
        conformer_hist[str(n_conf)] += 1
        pb_row_hist[str(int(record.get("n_pb_rows") or 0))] += 1
        if 0 < n_conf < DEFAULT_N_CONFORMERS:
            partial_embed += 1

    elapsed_total = sum(float(record.get("elapsed_s") or 0.0) for record in records)
    elapsed_sorted = sorted(
        float(record.get("elapsed_s") or 0.0) for record in records
    )

    def _pct(fraction: float) -> float:
        if not elapsed_sorted:
            return 0.0
        index = int(round(fraction * (len(elapsed_sorted) - 1)))
        index = min(max(index, 0), len(elapsed_sorted) - 1)
        return elapsed_sorted[index]

    elapsed_percentiles = {
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p99": _pct(0.99),
        "max": elapsed_sorted[-1] if elapsed_sorted else 0.0,
    }
    n_near_cap = (
        sum(1 for value in elapsed_sorted if value >= 0.9 * float(row_timeout_s))
        if row_timeout_s
        else None
    )

    # --- conditional on the pool's own graph-fragment count -------------
    # PoseBusters' ``all_atoms_connected`` check is exactly the check a
    # multi-fragment row cannot pass, so the pooled rate mixes a GEOMETRY
    # question with a FRAGMENTATION question.  Splitting on the pool's own
    # ``n_frags_raw`` separates them.
    groups: Dict[str, List[Dict[str, Any]]] = {
        "single_fragment": [],
        "multi_fragment": [],
        "fragment_count_unknown": [],
    }
    for record in records:
        fragment_count = record.get("n_frags_raw")
        if fragment_count == 1:
            groups["single_fragment"].append(record)
        elif isinstance(fragment_count, int) and fragment_count > 1:
            groups["multi_fragment"].append(record)
        else:
            groups["fragment_count_unknown"].append(record)

    def _group_stats(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        group_counts = Counter(record["status"] for record in subset)
        group_rows = len(subset)
        group_valid = int(group_counts.get(STATUS_PB_VALID, 0))
        group_scored = group_valid + int(group_counts.get(STATUS_PB_INVALID, 0))
        return {
            "n_rows": group_rows,
            "n_pb_valid": group_valid,
            "n_pb_invalid": int(group_counts.get(STATUS_PB_INVALID, 0)),
            "n_embed_failed": int(group_counts.get(STATUS_EMBED_FAILED, 0)),
            "n_unparseable": int(group_counts.get(STATUS_UNPARSEABLE, 0)),
            "n_timeout": int(group_counts.get(STATUS_TIMEOUT, 0)),
            "n_scored": group_scored,
            "pb_valid_rate_strict": (
                float(group_valid) / group_rows if group_rows else 0.0
            ),
            "pb_valid_rate_scored": (
                float(group_valid) / group_scored if group_scored else 0.0
            ),
        }

    by_fragment_count = {name: _group_stats(subset) for name, subset in groups.items()}
    by_fragment_count["_why"] = (
        "PoseBusters' 'all_atoms_connected' check is exactly the check a "
        "multi-fragment row cannot pass, so the pooled rate mixes a GEOMETRY "
        "question with a FRAGMENTATION question. The single_fragment group "
        "is the rate that speaks to 3D geometry alone."
    )
    by_fragment_count["_conditioning_variable"] = (
        "the pool's own n_frags_raw column (2D graph-fragment count)"
    )

    # --- conditional on the pool's own epoch column --------------------
    epoch_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        epoch_groups.setdefault(str(record.get("epoch", "unknown")), []).append(record)
    by_pool_epoch = {
        name: _group_stats(subset) for name, subset in sorted(epoch_groups.items())
    }
    # Wf-A164: the epoch structure of OUR pool is COMPUTED, never asserted.
    if pool_profile:
        n_epochs = int(pool_profile.get("n_distinct_epochs") or 0)
        per_epoch = pool_profile.get("rows_per_epoch") or []
        counts_phrase = ", ".join(
            f"{entry['n_rows']}" for entry in per_epoch
        ) or "no per-epoch counts"
        epoch_why = (
            f"the pool spans {n_epochs} epoch labels of its own `epoch` "
            f"column ({counts_phrase} rows respectively), so this block shows "
            "the epoch-to-epoch spread of the same pool's decode."
        )
    else:
        epoch_why = (
            "the pool's own epoch column splits the rows; the pool profile "
            "was not supplied, so no structural claim is made here."
        )
    by_pool_epoch["_why"] = epoch_why
    by_pool_epoch["_conditioning_variable"] = "the pool's own epoch column"

    return {
        "n_rows_in_scope": n_rows,
        "n_pb_valid": n_pb_valid,
        "n_pb_invalid": n_pb_invalid,
        "n_embed_failed": n_embed_failed,
        "n_unparseable": n_unparseable,
        "n_skipped_posebusters_missing": n_skipped,
        "n_timeout": n_timeout,
        "n_scored": n_scored,
        "status_counts": dict(sorted(counts.items())),
        "pb_valid_rate_strict": (
            float(n_pb_valid) / strict_denominator if strict_denominator else 0.0
        ),
        "pb_valid_rate_strict_via_repo_helper": strict_rate_via_repo_helper(
            records
        ),
        "strict_rate_cross_check": (
            "pb_valid_rate_strict_via_repo_helper is the same number produced "
            "by molmetal.validation.posebusters_runner.pb_pass_rate() on "
            "runner-shaped result dicts -- the repo's own aggregation helper. "
            "It MUST equal pb_valid_rate_strict."
        ),
        "pb_valid_rate_scored": (
            float(n_pb_valid) / n_scored if n_scored else 0.0
        ),
        "pb_valid_rate_scored_plus_embed_failed": (
            float(n_pb_valid) / scored_plus_embed_denominator
            if scored_plus_embed_denominator
            else 0.0
        ),
        "denominators": {
            "strict": {
                "n": strict_denominator,
                "definition": (
                    "every row in scope; unparseable + embed_failed + skipped "
                    "all count as NOT pb_valid"
                ),
            },
            "scored": {
                "n": n_scored,
                "definition": (
                    "rows that produced a PoseBusters verdict "
                    "(pb_valid + pb_invalid)"
                ),
            },
            "scored_plus_embed_failed": {
                "n": scored_plus_embed_denominator,
                "definition": (
                    "scored rows plus embed-failed rows, with embed failure "
                    "counting as NOT pb_valid; unparseable + skipped excluded"
                ),
                "is_headline": False,
            },
            "completed_only": {
                "n": n_scored + n_timeout,
                "definition": (
                    "rows that produced a PoseBusters verdict plus rows "
                    "that hit the per-row wall-clock cap; a timeout counts "
                    "as NOT pb_valid. Reported so the censoring effect of "
                    "the timeout policy is visible."
                ),
                "is_headline": False,
            },
        },
        "pb_valid_rate_completed_only": (
            float(n_pb_valid) / (n_scored + n_timeout)
            if (n_scored + n_timeout)
            else 0.0
        ),
        "n_forcefield_parameterisation_failures": int(
            check_missing.get("internal_energy", 0)
        ),
        "forcefield_parameterisation_note": (
            "n_forcefield_parameterisation_failures counts the scored rows "
            "whose PoseBusters 'internal_energy' check came back MISSING "
            "(None) because neither MMFF94 nor UFF could parameterise the "
            "molecule -- this is the metal-coordination blind spot of the "
            "force field, and the runner fails closed on it, so those rows "
            "count as NOT pb_valid."
        ),
        "per_check_not_true_counts": dict(sorted(check_not_true.items())),
        "per_check_explicit_false_counts": dict(sorted(check_false.items())),
        "per_check_missing_counts": dict(sorted(check_missing.items())),
        "n_conformers_embedded_histogram": dict(sorted(conformer_hist.items(), key=lambda kv: int(kv[0]))),
        "n_pb_rows_evaluated_histogram": dict(sorted(pb_row_hist.items(), key=lambda kv: int(kv[0]))),
        "n_rows_with_partial_embed": partial_embed,
        "sum_row_elapsed_s": elapsed_total,
        "row_elapsed_s_percentiles": elapsed_percentiles,
        "row_timeout_s": row_timeout_s,
        "n_rows_at_or_above_90pct_of_timeout_cap": n_near_cap,
        "by_pool_fragment_count": by_fragment_count,
        "by_pool_epoch": by_pool_epoch,
    }


# ----------------------------------------------------------------
# Artifact assembly
# ----------------------------------------------------------------
def summarize_conformer_probe(
    probe_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarise the conformer-sensitivity probe.

    Pure function -- no I/O.  A row counts as comparable only when BOTH
    runs produced a PoseBusters verdict (status ``pb_valid`` or
    ``pb_invalid``); embed failures are reported separately rather than
    being counted as agreement.
    """
    comparable = [
        row
        for row in probe_rows
        if row["status_n"] in (STATUS_PB_VALID, STATUS_PB_INVALID)
        and row["status_1"] in (STATUS_PB_VALID, STATUS_PB_INVALID)
    ]
    agree = [row for row in comparable if row["status_n"] == row["status_1"]]
    disagree = [row for row in comparable if row["status_n"] != row["status_1"]]
    return {
        "n_rows_probed": len(probe_rows),
        "n_comparable": len(comparable),
        "n_agree": len(agree),
        "n_disagree": len(disagree),
        "agreement_rate": (float(len(agree)) / len(comparable)) if comparable else 0.0,
        "n_embed_failed_on_either_arm": sum(
            1
            for row in probe_rows
            if STATUS_EMBED_FAILED in (row["status_n"], row["status_1"])
        ),
        "disagreements": [
            {
                "row_index": row["row_index"],
                "smiles": row["smiles"],
                "status_n_conformers": row["status_n"],
                "status_1_conformer": row["status_1"],
            }
            for row in disagree[:20]
        ],
        "method": (
            "the same molecule run twice through the public runner surface: "
            "check_posebusters(n_conformers=1) vs "
            "check_posebusters(n_conformers=<protocol>)"
        ),
        "why": (
            "PoseBusters' 'mol' config evaluates ONE conformer of the "
            "molecule (posebusters/config/mol.yml -> "
            "modules/distance_geometry.py uses mol.GetConformer()). The "
            "extra conformers are embedded and MMFF94-optimised but only "
            "one is judged, so this probe measures whether that choice "
            "changes the verdict."
        ),
    }


def tool_versions() -> Dict[str, Optional[str]]:
    """Return the installed versions of the tools that produced the numbers."""
    versions: Dict[str, Optional[str]] = {}
    for package in ("rdkit", "posebusters", "numpy", "pandas"):
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        except Exception:
            versions[package] = None
    return versions


def build_artifact(
    *,
    pool_path: Path,
    pool_sha256: str,
    pool_columns: Sequence[str],
    n_data_rows: int,
    records: Sequence[Dict[str, Any]],
    protocol: Dict[str, Any],
    scope: Dict[str, Any],
    timestamp: str,
    conformer_probe: Optional[Dict[str, Any]] = None,
    timeout_recheck: Optional[Dict[str, Any]] = None,
    row_timeout_s: Optional[float] = None,
    composition: Optional[Dict[str, Any]] = None,
    epoch_labels: Optional[Sequence[str]] = None,
    fragment_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Assemble the full artifact dict.

    Wf-A164 (2026-11-01): every number that describes OUR population is
    COMPUTED here from the pool, the pool's run directory and the measured
    composition.  Nothing about the DEFAULT pool is written as a literal,
    so a run against a different pool reports THAT pool's numbers.
    """
    composition = dict(composition or {})
    epoch_labels = list(epoch_labels or [])
    fragment_counts = list(fragment_counts or [])

    profile = build_pool_profile(
        pool_path=pool_path,
        pool_sha256=pool_sha256,
        n_data_rows=n_data_rows,
        columns=pool_columns,
        epoch_labels=epoch_labels,
        fragment_counts=fragment_counts,
        composition=composition,
    )
    counts = aggregate(
        records, row_timeout_s=row_timeout_s, pool_profile=profile
    )
    n_pb_valid = counts["n_pb_valid"]
    n_scored = counts["n_scored"]
    n_rows = counts["n_rows_in_scope"]

    breaks = build_comparability_breaks(profile)
    n_breaks = len(breaks)
    cross_check = _published_cross_check(
        profile,
        profile["single_fragment_fraction"],
        profile["n_single_fragment_rows"],
    )
    end_to_end = _end_to_end_headline()
    prior = _prior_posebusters_artifacts()

    # --- computed prose fragments -------------------------------------
    n_explicit = int(
        profile.get("n_rows_with_an_explicit_hydrogen_atom_in_the_smiles") or 0
    )
    metal_fraction = profile.get("fraction_rows_with_a_metal_atom")
    metal_rows = profile.get("n_rows_with_a_metal_atom")
    single_pct = profile["single_fragment_fraction"] * 100.0
    elapsed_percentiles = counts.get("row_elapsed_s_percentiles") or {}
    max_elapsed = float(elapsed_percentiles.get("max") or 0.0)
    p99_elapsed = float(elapsed_percentiles.get("p99") or 0.0)
    n_conformers = int(protocol.get("n_conformers") or DEFAULT_N_CONFORMERS)
    frontier = FRONTIER_BAND
    pb_pct = frontier["flowmol3_arxiv_2508_12629"]["pct_pb_valid"][0]
    pb_sd = frontier["flowmol3_arxiv_2508_12629"]["pct_pb_valid"][1]
    band_lo = frontier["band_connected_pct"][0]
    band_hi = frontier["band_connected_pct"][1]
    axes_phrase = ", ".join(str(item["axis"]) for item in breaks)

    structure_note = (
        "MEASURED from THIS pool's own columns and run directory at run "
        f"time: {_pool_phrase(profile)}. The per-epoch row counts are in "
        "source_pool.measured_profile.rows_per_epoch. It is NOT a "
        "training-set-distribution sampling."
    )
    pool_caveat = (
        f"{_pool_phrase(profile)}, not a training-set-distribution sampling "
        f"and not a benchmark at the frontier band's "
        f"{frontier['frontier_denominator_draws']} scale. The frontier band "
        f"is over {frontier['frontier_denominator_draws']} such draws, so "
        "even a like-for-like metric would not be a like-for-like population."
    )
    composition["explicit_hydrogen_note"] = (
        f"{n_explicit} rows of this pool carry an explicit [H] in the "
        "SMILES; the rest carry IMPLICIT hydrogens only. RDKit AddHs is "
        "applied inside the runner's embedding step. The frontier dataset "
        f"carries explicit hydrogens, which is one of the {n_breaks} "
        "comparability breaks."
    )
    composition.setdefault(
        "metal_definition",
        "atomic number in the metal set (all elements except B, C, N, O, "
        "F, Si, P, S, Cl, Ge, As, Se, Br, Te, I, Xe, At, Rn)",
    )

    if end_to_end.get("present") and end_to_end.get("median") is not None:
        e2e_phrase = (
            f"NOT a restatement of {end_to_end['source']} -> "
            f"{end_to_end['jsonpath']} (median {end_to_end['median']}, over "
            f"{end_to_end['n_runs']} "
            "runs). That is an END-TO-END single-molecule rate over MANY "
            "RUNS; this artifact is a SINGLE-POOL GEOMETRY rate over "
            f"{n_rows} rows. Different quantity, different population."
        )
    else:
        e2e_phrase = (
            "NOT a restatement of the published END-TO-END median headline: "
            f"{end_to_end['source']} was not readable at run time, so no "
            "value and no run count is quoted here."
        )

    fragment_phrase = (
        "NOT a restatement of the single-pool single-fragment count (the "
        "rows whose n_frags_raw column equals one): "
        f"{profile['n_single_fragment_rows']}/{n_rows} = "
        f"{single_pct:.2f} %. That is a 2D graph-fragment count; this "
        "artifact is a 3D geometry verdict. NOTE that the single_fragment "
        "conditional rate reported under counts.by_pool_fragment_count is "
        "NOT that fraction either: it is the PoseBusters-valid rate WITHIN "
        f"the {profile['n_single_fragment_rows']}-row single-fragment "
        "subpopulation. Two different quantities over the same partition."
    )

    if prior.get("largest_n_samples") is None:
        prior_phrase = (
            f"{prior['n_candidates_scanned']} candidate files under "
            f"{PRIOR_POSE_BUSTERS_DIR} were scanned at run time; "
            f"{prior['n_present']} are present and "
            f"{prior['n_mentioning_posebusters']} mention PoseBusters. No "
            "sample size was recorded by any of them, so no largest sample "
            "is claimed."
        )
    else:
        prior_phrase = (
            f"{prior['n_candidates_scanned']} candidate files under "
            f"{PRIOR_POSE_BUSTERS_DIR} were scanned at run time; "
            f"{prior['n_present']} are present and "
            f"{prior['n_mentioning_posebusters']} mention PoseBusters. The "
            f"LARGEST PoseBusters sample among them is "
            f"{prior['largest_n_samples']} "
            f"({prior['largest_n_samples_file']}). This artifact IS the "
            "first POOL-SCALE PoseBusters measurement in the repo: all "
            f"{n_rows} rows of the pool named in source_pool."
        )

    metal_phrase = (
        "EVERY row of this pool carries a metal centre, so the force-field "
        "parameterisation shortfall above is not a corner case, it is the "
        "whole pool."
        if isinstance(metal_fraction, (int, float)) and metal_fraction >= 1.0
        else (
            f"{metal_rows} of {n_rows} rows of this pool carry a metal "
            "centre, so the force-field parameterisation shortfall above "
            "applies to that fraction of the pool."
        )
    )

    statement = (
        f"Our % PB-Valid is NOT comparable to the frontier {pb_pct} +/- "
        f"{pb_sd} % (% PB-Valid, FlowMol3 arXiv "
        f"{frontier['flowmol3_arxiv_2508_12629']['arxiv_id']}) or to the "
        f"{band_lo}-{band_hi} % connected band. {n_breaks} independent "
        f"breaks ({axes_phrase}) separate the two. Do NOT write 'our X % "
        f"vs their {pb_pct} %' without this statement."
    )

    probe_arm_n = n_conformers
    probe_arm_1 = 1
    probe_method = (
        "the same molecule run twice through the public runner surface: "
        f"check_posebusters(n_conformers={probe_arm_1}) vs "
        f"check_posebusters(n_conformers={probe_arm_n})"
    )
    probe_why = (
        "PoseBusters' 'mol' config evaluates ONE conformer of the molecule "
        "(posebusters/config/mol.yml -> modules/distance_geometry.py uses "
        "mol.GetConformer()). The extra conformers are embedded and "
        "MMFF94-optimised but only one is judged, so this probe measures "
        "whether that choice changes the verdict."
    )
    probe_sample_method = (
        "random.Random(probe_seed).sample(range(n_pool), probe_size)"
    )
    if conformer_probe is None:
        conformer_probe = {
            "enabled": False,
            "n_rows_probed": 0,
            "n_comparable": 0,
            "n_agree": 0,
            "n_disagree": 0,
            "agreement_rate": 0.0,
            "n_embed_failed_on_either_arm": 0,
            "disagreements": [],
        }
    conformer_probe["enabled"] = bool(conformer_probe.get("n_rows_probed"))
    conformer_probe["method"] = probe_method
    conformer_probe["why"] = probe_why
    conformer_probe["sample_method"] = probe_sample_method
    conformer_probe["probe_arm_n_conformers"] = probe_arm_n
    conformer_probe["probe_arm_1_conformers"] = probe_arm_1

    if timeout_recheck is None:
        timeout_recheck = {
            "enabled": False,
            "n_timeouts_in_main_pass": int(counts.get("n_timeout") or 0),
            "n_rechecked": 0,
            "recheck_cap_s": None,
            "recheck_max_rows": 0,
            "recheck_wall_seconds": 0.0,
            "recheck_status_counts": {},
            "n_that_would_be_pb_valid": 0,
            "n_that_would_be_pb_invalid": 0,
            "n_still_embed_failed": 0,
            "n_still_timed_out": 0,
            "rows": [],
        }
    timeout_recheck.setdefault(
        "bias_direction",
        "the censored rows were counted as NOT pb_valid in the strict rate; "
        "if the re-check turns some of them into pb_valid rows, the strict "
        "rate is a LOWER bound on the pool rate, and the size of the effect "
        "is n_that_would_be_pb_valid",
    )

    # The scope block must always carry the same narrative leaves, otherwise
    # the guard would resolve to nothing for a full-pool run.
    scope = dict(scope)
    scope.setdefault(
        "label",
        "FULL POOL -- every row of the pool was checked"
        if not scope.get("is_subset")
        else "SUBSET estimate -- NOT the pool rate",
    )
    if not scope.get("sampling"):
        scope["sampling"] = {
            "method": "every row of the pool was taken (no sampling)",
            "seed": None,
            "population": "all pool rows",
        }

    # Examples are spread ACROSS statuses (not just the first rows), so the
    # artifact shows what each outcome actually looks like.
    examples: List[Dict[str, Any]] = []
    status_order = (
        STATUS_PB_VALID,
        STATUS_PB_INVALID,
        STATUS_EMBED_FAILED,
        STATUS_UNPARSEABLE,
        STATUS_SKIPPED,
        STATUS_TIMEOUT,
    )
    by_status: Dict[str, List[Dict[str, Any]]] = {status: [] for status in status_order}
    for record in records:
        by_status.setdefault(record["status"], []).append(record)
    for status in status_order:
        for record in by_status.get(status, [])[:4]:
            if len(examples) >= 12:
                break
            examples.append(
                {
                    "row_index": record["row_index"],
                    "smiles": record["smiles"],
                    "status": record["status"],
                    "pb_valid": record.get("pb_valid"),
                    "n_conformers": record.get("n_conformers"),
                    "failures": record.get("failures"),
                    "missing_checks": record.get("missing_checks"),
                }
            )

    valid_smiles = [
        record["smiles"] for record in records if record["status"] == STATUS_PB_VALID
    ]

    n_checkpoints = profile.get("n_checkpoints")
    pool_kind = "post_hoc_decode_pool"
    pool_kind += f"_{profile['n_distinct_epochs']}_epoch_labels"
    pool_kind += (
        f"_{n_checkpoints}_checkpoints"
        if isinstance(n_checkpoints, int)
        else "_checkpoint_count_unknown"
    )

    artifact: Dict[str, Any] = {
        "metric": "wfa155_posebusters_validity",
        "tag": "MEASURED",
        "date": "2026-11-01",
        "wave": "Wf-A155 (branch POSE) + Wf-A164 (computed narrative)",
        "branch": "POSE",
        "script": "molmetal/scripts/wf_a155_posebusters_validity.py",
        "purpose": (
            "PoseBusters ('mol' config) validity rate over the CFM decode "
            "pool named in source_pool -- the frontier option (c) of the "
            "Wf-A155 wave: we had NO PoseBusters column at all before this "
            "artifact."
        ),
        "generated_at_utc": timestamp,
        "source_pool": {
            "path": profile["path"],
            "sha256": pool_sha256,
            "n_data_rows": profile["n_data_rows"],
            "columns": list(pool_columns),
            "pool_kind": pool_kind,
            "composition": composition,
            "epoch_column_distribution": profile["epoch_column_distribution"],
            "n_distinct_epochs": profile["n_distinct_epochs"],
            "measured_profile": profile,
            "pool_structure_note": structure_note,
            "pool_caveat": pool_caveat,
        },
        "protocol": protocol,
        "scope": scope,
        "tool_versions": tool_versions(),
        "counts": counts,
        "conformer_sensitivity_probe": conformer_probe,
        "timeout_recheck": timeout_recheck,
        "end_to_end_headline": end_to_end,
        "published_cross_check": cross_check,
        "prior_posebusters_artifacts_measured": prior,
        "prose_backing_measurements": {
            "rule": (
                "the numbers this artifact's PROSE quotes are carried here "
                "as MACHINE fields, so that every number rendered in a "
                "narrative string is backed by a machine field of this same "
                "artifact -- a number in a `tag: MEASURED` narrative that "
                "nothing backs is a defect (Wf-A181)"
            ),
            "end_to_end_headline": {
                "source": end_to_end.get("source"),
                "jsonpath": end_to_end.get("jsonpath"),
                "median": end_to_end.get("median"),
                "n_runs": end_to_end.get("n_runs"),
                "definition": end_to_end.get("definition"),
            },
            "single_fragment_fraction_pct_2dp": round(single_pct, 2),
            "n_single_fragment_rows": profile["n_single_fragment_rows"],
        },
        # Wf-A189 (2026-11-01): the Wf-A181 dated record is emitted HERE, in
        # the SAME POSITION it occupies in the published artefact, so a re-run
        # REPRODUCES the published artefact instead of silently DROPPING the
        # dated history (AGENTS.md §5.5 rule #6).
        "wfa181_narrative_backing_repair_2026_11_01": dict(
            WFA181_NARRATIVE_BACKING_REPAIR_2026_11_01
        ),
        "headline": {
            "pb_valid_rate_strict_denominator": counts["pb_valid_rate_strict"],
            "pb_valid_rate_scored_denominator": counts["pb_valid_rate_scored"],
            "n_pb_valid": n_pb_valid,
            "n_scored": n_scored,
            "n_rows_in_scope": n_rows,
            "n_embed_failed": counts["n_embed_failed"],
            "n_unparseable": counts["n_unparseable"],
            "n_timeout": counts["n_timeout"],
            "n_forcefield_parameterisation_failures": counts[
                "n_forcefield_parameterisation_failures"
            ],
            "strict_denominator_definition": (
                "every row in scope, including rows that could not be parsed "
                "or embedded"
            ),
            "scored_denominator_definition": (
                "rows that produced a PoseBusters verdict"
            ),
            "one_line": (
                f"{n_pb_valid}/{n_rows} rows are PoseBusters-valid under the "
                f"strict denominator; {n_pb_valid}/{n_scored} under the "
                "scored denominator."
            ),
            "pb_valid_rate_scored_single_fragment": counts[
                "by_pool_fragment_count"
            ]["single_fragment"]["pb_valid_rate_scored"],
            "pb_valid_rate_scored_multi_fragment": counts[
                "by_pool_fragment_count"
            ]["multi_fragment"]["pb_valid_rate_scored"],
            "single_fragment_note": (
                "the single-fragment conditional rate is the number that "
                "speaks to 3D GEOMETRY alone: a multi-fragment row cannot "
                "pass PoseBusters' 'all_atoms_connected' check for reasons "
                "that belong to the fragmentation axis, not to the geometry."
            ),
        },
        "not_a_restatement_of": {
            "end_to_end_median_headline": e2e_phrase,
            "single_pool_fragment_count": fragment_phrase,
            "prior_posebusters_artifacts": prior_phrase,
        },
        "comparability_vs_frontier_band": {
            "frontier_band": FRONTIER_BAND,
            "n_independent_comparability_breaks": n_breaks,
            "breaks": [dict(item) for item in breaks],
            "statement": statement,
            # Wf-A189 (2026-11-01): the dated rule-#6 records are emitted
            # HERE, in the SAME POSITION they occupy in the published
            # artefact, so a re-run REPRODUCES the published artefact instead
            # of silently DROPPING the dated history (AGENTS.md §5.5 rule #6).
            # They are classified as PROVENANCE_PATHS by the guard, so their
            # numerals (document references, not measurements of this pool)
            # are not number-checked.
            "superseded_reading_2026_11_01": dict(
                COMPARABILITY_SUPERSEDED_READING_2026_11_01
            ),
            "superseded_reading_2026_11_01_wfa180": dict(
                COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA180
            ),
            "superseded_reading_2026_11_01_wfa182": dict(
                COMPARABILITY_SUPERSEDED_READING_2026_11_01_WFA182
            ),
        },
        "limitations": [
            "PoseBusters has NO metal-coordination-sphere check. A % "
            "PB-Valid for this pool says NOTHING about whether the metal is "
            "coordinated correctly (oxidation state, coordination number, "
            "dative bonds, trans-effect). This metric is a MOLECULE-GEOMETRY "
            "check, not a coordination-chemistry check.",
            "MMFF94 cannot parameterise most metal centres. The PoseBusters "
            "'internal_energy' check then returns None (missing), and the "
            "runner fails CLOSED, so a metal-containing molecule is reported "
            "as PB-invalid for a force-field reason rather than a geometry "
            "reason. The per_check_missing_counts block quantifies this.",
            "The pooled PB verdict is NOT dominated by 'bad 3D geometry' in "
            "the usual sense. Two check failures dominate and both are "
            "structural rather than geometry-quality failures: "
            "'all_atoms_connected' (a multi-fragment row cannot pass it) and "
            "'internal_energy' (missing, because neither MMFF94 nor UFF can "
            "parameterise the metal centre). Read the pooled rate together "
            "with counts.by_pool_fragment_count and "
            "counts.per_check_not_true_counts before interpreting it.",
            f"MEASURED (source_pool.composition): "
            f"fraction_rows_with_a_metal_atom = {metal_fraction} -- "
            + metal_phrase,
            f"The pool carries IMPLICIT hydrogens in {n_rows - n_explicit} "
            f"of its {n_rows} rows; RDKit AddHs is applied inside the "
            "embedding step but the pool SMILES themselves have no "
            "explicit-H atoms in those rows. The frontier dataset has "
            "explicit hydrogens. This alone breaks the comparison. MEASURED "
            "(source_pool.composition): "
            f"n_rows_with_an_explicit_hydrogen_atom_in_the_smiles = "
            f"{n_explicit} of {n_rows}.",
            f"The pool is {_pool_phrase(profile)}. It is not a "
            "training-set-distribution sampling, it carries no "
            "checkpoint-to-checkpoint variance, and it is smaller than the "
            f"frontier band's {frontier['frontier_denominator_draws']} draws.",
            "The conformer ensemble is seeded (ETKDGv3 randomSeed = "
            f"{protocol.get('embed_random_seed')} in "
            "molmetal/validation/posebusters_runner.py), so the result is "
            "reproducible for a fixed RDKit/PoseBusters version pair but is "
            "not a seed-averaged statistic.",
            "PoseBusters' 'mol' config evaluates ONE conformer per molecule "
            "(posebusters/config/mol.yml -> modules/distance_geometry.py "
            "reads mol.GetConformer()); the other embedded conformers are "
            "optimised but not judged. The conformer_sensitivity_probe "
            "block measures how often that choice changes the verdict.",
            "Rows that fail to embed are counted and reported under BOTH "
            "denominators; they are not silently dropped.",
            "A per-row wall-clock cap (protocol.row_timeout_s) kills any row "
            "whose PoseBusters call overruns it. A timeout is a CENSORING "
            "event, not a chemistry verdict: it counts as NOT pb_valid in "
            "the strict rate, and counts.n_timeout plus the completed-only "
            "rate expose the censoring. The rows that hit the cap are the "
            "pathological ETKDG geometries, so the censoring is NOT random "
            "with respect to molecule size.",
            "The censoring is MEASURED, not only asserted. The "
            "timeout_recheck block re-runs the timed-out rows under a longer "
            "per-row cap and reports what verdicts they would have "
            "produced, so the direction and size of the timeout bias is "
            "quantified rather than described.",
        ],
        "examples": examples,
        "pb_valid_smiles_sample": valid_smiles[:300],
        "pb_valid_smiles_sample_truncated": len(valid_smiles) > 300,
    }

    # The row-timing sentence is rendered from THIS run's own percentiles,
    # EXCEPT when the historical, POOL-BOUND single-call measurement applies
    # (Wf-A182): then the MEASURED-2026-11-01 rendering stands, because it
    # is the measurement that pool's own record carries.  For any other
    # pool the three numbers are not certified, so the run's own percentiles
    # are the only timing statement the artifact may make.
    longest_provenance = (
        artifact["protocol"].get("longest_pool_smiles_call_seconds_provenance")
        or {}
    )
    if not longest_provenance.get("applicable_to_this_pool"):
        artifact["protocol"]["row_timeout_rationale"] = (
            f"MEASURED in THIS run: the slowest row took {max_elapsed:.1f} s and "
            f"the p99 row took {p99_elapsed:.1f} s "
            "(counts.row_elapsed_s_percentiles), and an in-process pool blocks "
            "indefinitely on such a row. The child-process cap is what makes a "
            "FULL-POOL run terminate."
        )

    # -- THE GUARD for the class of defect, not just this instance --------
    guard = _narrative_self_consistency(artifact)
    artifact["narrative_self_consistency"] = guard
    # Wf-A189 (2026-11-01): the two remaining dated rule-#6 records, and the
    # Wf-A182-era guard block that Wf-A182 preserved inside the guard block,
    # are emitted AFTER the guard -- in the SAME POSITION they occupy in the
    # published artefact -- so a re-run REPRODUCES the published artefact's
    # PATH SET instead of silently DROPPING the dated history (AGENTS.md §5.5
    # rule #6).  They are classified as PROVENANCE_PATHS (and the guard block
    # itself as an IDENTIFIER_PATH), so the guard stays green.
    artifact["published_cross_check_superseded_2026_11_01"] = dict(
        PUBLISHED_CROSS_CHECK_SUPERSEDED_2026_11_01
    )
    artifact["wfa182_residual_defect_repair_2026_11_01"] = dict(
        WFA182_RESIDUAL_DEFECT_REPAIR_2026_11_01
    )
    artifact["narrative_self_consistency"]["superseded_2026_11_01_wfa182"] = dict(
        SUPERSEDED_GUARD_2026_11_01_WFA182
    )
    if not guard["pass"]:
        raise NarrativeSelfConsistencyError(artifact, guard)
    return artifact


# ----------------------------------------------------------------
# Driver
# ----------------------------------------------------------------
def select_rows(
    smiles: Sequence[str],
    *,
    sample_size: Optional[int],
    sample_seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Return the row indices to check plus a description of the selection."""
    total = len(smiles)
    if sample_size is None or sample_size >= total:
        return list(range(total)), {
            "kind": "full_pool",
            "n_selected": total,
            "n_pool": total,
            "sampling": {
                "method": "every row of the pool was taken (no sampling)",
                "seed": None,
                "population": "all pool rows",
            },
            "is_subset": False,
            "label": "FULL POOL -- every row of the pool was checked",
        }
    rng = random.Random(sample_seed)
    indices = sorted(rng.sample(range(total), sample_size))
    return indices, {
        "kind": "seeded_random_sample",
        "n_selected": sample_size,
        "n_pool": total,
        "sampling": {
            "method": "random.Random(sample_seed).sample(range(n_pool), n)",
            "seed": sample_seed,
            "population": "all pool row indices, uniform without replacement",
        },
        "is_subset": True,
        "label": "SUBSET estimate -- NOT the pool rate",
    }


def run(
    *,
    pool_path: Path,
    out_path: Path,
    n_conformers: int,
    embed_method: str,
    optimize_method: str,
    sample_size: Optional[int],
    sample_seed: int,
    workers: int,
    timestamp: str,
    row_timeout_s: float = 90.0,
    probe_size: int = 25,
    probe_seed: int = 155,
    timeout_recheck_s: float = 0.0,
    timeout_recheck_max: int = 30,
) -> Dict[str, Any]:
    """Run the measurement and write the artifact.  Returns the artifact."""
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )
    from molmetal.validation.posebusters_runner import posebusters_available

    if not posebusters_available():
        raise RuntimeError(
            "posebusters is not importable on this host -- the metric cannot "
            "be measured. This artifact REQUIRES a real measurement; do not "
            "write a null artifact."
        )

    pool_sha256 = sha256_file(pool_path)
    smiles, columns = read_pool(pool_path)
    fragment_counts = read_pool_fragment_counts(pool_path)
    epoch_labels = read_pool_epoch_labels(pool_path)
    composition = pool_composition(smiles)
    indices, scope = select_rows(
        smiles, sample_size=sample_size, sample_seed=sample_seed
    )
    scope["pool_sha256_at_run_time"] = pool_sha256

    payloads = [
        (index, smiles[index], n_conformers, embed_method, optimize_method)
        for index in indices
    ]

    started = time.perf_counter()
    records = []
    if workers <= 1:
        for index, payload in enumerate(payloads, start=1):
            record = _run_one(payload)
            record["row_index"] = index
            record["n_frags_raw"] = (
                fragment_counts[index] if index < len(fragment_counts) else -1
            )
            record["epoch"] = (
                epoch_labels[index] if index < len(epoch_labels) else "unknown"
            )
            records.append(record)
            if index % 100 == 0:
                LOG.info("progress: %d/%d rows", index, len(payloads))
    else:
        from concurrent.futures import ThreadPoolExecutor

        def _task(item: Tuple[int, str, int, str, str]) -> Dict[str, Any]:
            row_index, row_smiles, n_conf, embed, optimize = item
            record = _run_one_subprocess(
                row_smiles,
                n_conformers=n_conf,
                embed_method=embed,
                optimize_method=optimize,
                timeout_s=row_timeout_s,
            )
            record["row_index"] = row_index
            record["n_frags_raw"] = (
                fragment_counts[row_index] if row_index < len(fragment_counts) else -1
            )
            record["epoch"] = (
                epoch_labels[row_index] if row_index < len(epoch_labels) else "unknown"
            )
            return record

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, record in enumerate(pool.map(_task, payloads), start=1):
                records.append(record)
                if done % 100 == 0:
                    LOG.info(
                        "progress: %d/%d rows (elapsed %.0fs)",
                        done,
                        len(payloads),
                        time.perf_counter() - started,
                    )
    wall_seconds = time.perf_counter() - started
    LOG.info("main pass done: %d rows in %.1fs", len(records), wall_seconds)

    records.sort(key=lambda record: record["row_index"])

    # --- censoring measurement ------------------------------------------
    # A row that hit the per-row cap is a CENSORED row, not a chemistry
    # verdict.  Re-running those rows under a longer cap MEASURES what the
    # censoring did to the rate instead of merely describing it.
    timeout_recheck: Optional[Dict[str, Any]] = None
    timed_out = [record for record in records if record["status"] == STATUS_TIMEOUT]
    if timeout_recheck_s and timeout_recheck_s > row_timeout_s and timed_out:
        chosen = timed_out[: max(0, int(timeout_recheck_max))]
        recheck_started = time.perf_counter()

        def _recheck_task(record: Dict[str, Any]) -> Dict[str, Any]:
            fresh = _run_one_subprocess(
                record["smiles"],
                n_conformers=n_conformers,
                embed_method=embed_method,
                optimize_method=optimize_method,
                timeout_s=timeout_recheck_s,
            )
            return {
                "row_index": record["row_index"],
                "smiles": record["smiles"],
                "status": fresh["status"],
                "pb_valid": fresh.get("pb_valid"),
                "failures": fresh.get("failures"),
                "elapsed_s": fresh.get("elapsed_s"),
            }

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, workers)) as recheck_pool:
            rechecks = list(recheck_pool.map(_recheck_task, chosen))
        rechecks.sort(key=lambda item: item["row_index"])
        status_counts = Counter(item["status"] for item in rechecks)
        n_became_valid = int(status_counts.get(STATUS_PB_VALID, 0))
        timeout_recheck = {
            "n_timeouts_in_main_pass": len(timed_out),
            "n_rechecked": len(rechecks),
            "recheck_cap_s": timeout_recheck_s,
            "recheck_max_rows": int(timeout_recheck_max),
            "recheck_wall_seconds": time.perf_counter() - recheck_started,
            "recheck_status_counts": dict(sorted(status_counts.items())),
            "n_that_would_be_pb_valid": n_became_valid,
            "n_that_would_be_pb_invalid": int(status_counts.get(STATUS_PB_INVALID, 0)),
            "n_still_embed_failed": int(status_counts.get(STATUS_EMBED_FAILED, 0)),
            "n_still_timed_out": int(status_counts.get(STATUS_TIMEOUT, 0)),
            "bias_direction": (
                "the censored rows were counted as NOT pb_valid in the strict "
                "rate; if the re-check turns some of them into pb_valid rows, "
                "the strict rate is a LOWER bound on the pool rate, and the "
                "size of the effect is n_that_would_be_pb_valid"
            ),
            "rows": rechecks,
        }
        LOG.info(
            "timeout recheck: %d/%d re-checked, %d would be pb_valid",
            len(rechecks),
            len(timed_out),
            n_became_valid,
        )

    # --- conformer-sensitivity probe (small, documented sample) --------
    probe_summary: Optional[Dict[str, Any]] = None
    if probe_size and probe_size > 0:
        probe_rng = random.Random(probe_seed)
        probe_pool = sorted(probe_rng.sample(range(len(smiles)), min(probe_size, len(smiles))))
        probe_started = time.perf_counter()

        def _probe_task(index: int) -> Dict[str, Any]:
            arm_n = _run_one_subprocess(
                smiles[index],
                n_conformers=n_conformers,
                embed_method=embed_method,
                optimize_method=optimize_method,
                timeout_s=row_timeout_s,
            )
            arm_1 = _run_one_subprocess(
                smiles[index],
                n_conformers=1,
                embed_method=embed_method,
                optimize_method=optimize_method,
                timeout_s=row_timeout_s,
            )
            return {
                "row_index": index,
                "smiles": smiles[index],
                "status_n": arm_n["status"],
                "status_1": arm_1["status"],
                "pb_valid_n": arm_n.get("pb_valid"),
                "pb_valid_1": arm_1.get("pb_valid"),
            }

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, workers)) as probe_pool_exec:
            probe_rows = list(probe_pool_exec.map(_probe_task, probe_pool))
        probe_summary = summarize_conformer_probe(probe_rows)
        probe_summary["sample_seed"] = probe_seed
        probe_summary["sample_method"] = (
            "random.Random(probe_seed).sample(range(n_pool), probe_size)"
        )
        probe_summary["wall_seconds"] = time.perf_counter() - probe_started
        probe_summary["method"] = (
            "the same molecule checked twice through the public runner "
            "surface: check_posebusters(n_conformers=1) vs "
            "check_posebusters(n_conformers=<protocol>), each under the "
            "same hard per-row wall-clock cap as the main pass"
        )
        probe_summary["why"] = (
            "PoseBusters' 'mol' config evaluates ONE conformer of the "
            "molecule (posebusters/config/mol.yml -> "
            "modules/distance_geometry.py uses mol.GetConformer()); the "
            "extra conformers are embedded and MMFF94-optimised but only "
            "one is judged."
        )

    # Wf-A182: bind the three longest-pool-SMILES call seconds to the pool
    # they were measured on, BEFORE the protocol block renders them.
    longest_binding = longest_pool_measurement_binding(
        pool_path=_relative_to_project(Path(pool_path)),
        pool_sha256=pool_sha256,
    )

    protocol = {
        "n_conformers": n_conformers,
        "embed_method": embed_method,
        "optimize_method": optimize_method,
        "posebusters_config": "mol",
        "backend_module": "molmetal.validation.posebusters_runner.check_posebusters",
        "embed_random_seed": 42,
        "embed_random_seed_source": (
            "molmetal/validation/posebusters_runner.py::_embed_conformers "
            "(params.randomSeed = 42, hard-coded in the existing runner)"
        ),
        "runner_fails_closed": (
            "a check the runner could not compute (None) counts as a "
            "failure -- this is the existing module's semantics and is "
            "NOT changed here"
        ),
        "wall_seconds": wall_seconds,
        "n_workers": workers,
        "execution": (
            "one isolated child process per pool row (ThreadPoolExecutor "
            "driving subprocess.run), CPU-only"
        ),
        "row_timeout_s": row_timeout_s,
        "row_timeout_policy": (
            "a row whose PoseBusters call exceeds row_timeout_s is KILLED and "
            "recorded with status 'timeout'. A timeout is a CENSORING event, "
            "not a chemistry verdict: it counts as NOT pb_valid in the "
            "strict rate, and the counts block reports n_timeout and a "
            "completed-only rate so the censoring is visible."
        ),
        "row_timeout_rationale": (
            "MEASURED 2026-11-01: the three longest pool SMILES took "
            + " / ".join(
                f"{value} s" for value in LONGEST_POOL_SMILES_CALL_SECONDS
            )
            + " on a single molecule-level "
            "check_posebusters call, and an in-process pool blocks "
            "indefinitely on such a row. The child-process cap is what "
            "makes a FULL-POOL run terminate."
        )
        if longest_binding.get("applicable_to_this_pool")
        else "",
        "longest_pool_smiles_call_seconds": (
            list(LONGEST_POOL_SMILES_CALL_SECONDS)
            if longest_binding.get("applicable_to_this_pool")
            else None
        ),
        # Wf-A182: the binding is emitted WITH the numbers, so a re-run
        # cannot drop it -- and the numbers are emitted only for the pool
        # the measurement was taken on.
        "longest_pool_smiles_call_seconds_provenance": longest_binding,
        "device": "CPU",
    }

    artifact = build_artifact(
        pool_path=pool_path,
        pool_sha256=pool_sha256,
        pool_columns=columns,
        n_data_rows=len(smiles),
        records=records,
        protocol=protocol,
        scope=scope,
        timestamp=timestamp,
        conformer_probe=probe_summary,
        timeout_recheck=timeout_recheck,
        row_timeout_s=row_timeout_s,
        composition=composition,
        epoch_labels=epoch_labels,
        fragment_counts=fragment_counts,
    )

    content = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    risk = safe_overwrite_check(out_path, content)
    if risk is not None:
        raise RuntimeError(
            f"safe_overwrite_check refused the write to {out_path} "
            f"(diff_ratio={risk:.3f}); pick a different filename"
        )
    truncation = detect_truncation(content)
    if truncation is not None:
        raise RuntimeError(
            f"detect_truncation flagged the artifact payload: {truncation}"
        )
    atomic_write_file(out_path, content)

    # Read back and assert equality (AGENTS.md 5.5 filesystem-truth).
    reread = out_path.read_text(encoding="utf-8")
    if reread != content:
        raise RuntimeError(f"read-back mismatch for {out_path}")
    return artifact


def _write_fail_artifact(
    out_path: Path, exc: "NarrativeSelfConsistencyError"
) -> List[str]:
    """Write the ``status: FAIL`` artifact for a rejected run.

    Wf-A164: the failure must be ON DISK, not only on stdout.  The FAIL
    artifact is written to ``out_path`` when that is safe; if
    ``safe_overwrite_check`` refuses (e.g. a good artifact is already
    there), it is written beside it as ``<stem>.FAIL.json`` so a valid
    published artifact is never silently destroyed by a guard failure.
    """
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
    )

    bad = dict(exc.artifact)
    bad["status"] = "FAIL"
    bad["failure"] = (
        "the narrative self-consistency guard rejected this artifact: "
        f"unbacked_numbers={exc.guard.get('unbacked_numbers')} "
        f"stale_paths={exc.guard.get('stale_paths')} "
        f"unresolved_narrative_paths={exc.guard.get('unresolved_narrative_paths')} "
        f"unclassified_leaves={sorted(exc.guard.get('unclassified_leaves', {}))}"
    )
    content = json.dumps(bad, indent=2, ensure_ascii=False) + "\n"
    candidates = [Path(out_path)]
    sibling = Path(out_path).with_name(Path(out_path).stem + ".FAIL.json")
    if sibling != Path(out_path):
        candidates.append(sibling)
    written: List[str] = []
    for candidate in candidates:
        if candidate.exists():
            risk = safe_overwrite_check(candidate, content)
            if risk is not None:
                continue
        atomic_write_file(candidate, content)
        written.append(str(candidate))
        break
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Child mode is checked BEFORE argparse: the child gets a SMILES on the
    # command line, which argparse would try to interpret.
    if argv is not None and len(argv) >= 1 and argv[0] == "--child":
        return _child_main(argv[1:])
    if argv is None:
        argv = sys.argv[1:]
        if argv and argv[0] == "--child":
            return _child_main(argv[1:])

    parser = argparse.ArgumentParser(
        description=(
            "Wf-A155 PoseBusters validity column over the published CFM pool."
        )
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-conformers", type=int, default=DEFAULT_N_CONFORMERS)
    parser.add_argument("--embed-method", default=DEFAULT_EMBED_METHOD)
    parser.add_argument("--optimize-method", default=DEFAULT_OPTIMIZE_METHOD)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="documented subset mode; omit for a full-pool run",
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--row-timeout-s",
        type=float,
        default=90.0,
        help=(
            "hard wall-clock cap for ONE row's PoseBusters call; a row that "
            "exceeds it is killed and recorded with status 'timeout'"
        ),
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=25,
        help="rows used for the conformer-sensitivity probe (0 disables it)",
    )
    parser.add_argument("--probe-seed", type=int, default=155)
    parser.add_argument(
        "--timeout-recheck-s",
        type=float,
        default=0.0,
        help=(
            "if > row_timeout_s, re-run the timed-out rows under this longer "
            "per-row cap and MEASURE the censoring bias (0 disables it)"
        ),
    )
    parser.add_argument(
        "--timeout-recheck-max",
        type=int,
        default=30,
        help="max number of timed-out rows re-checked",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="override the artifact timestamp (default: UTC now, ISO-8601)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    try:
        artifact = run(
            pool_path=args.pool,
            out_path=args.out,
            n_conformers=args.n_conformers,
            embed_method=args.embed_method,
            optimize_method=args.optimize_method,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            workers=args.workers,
            timestamp=timestamp,
            row_timeout_s=args.row_timeout_s,
            probe_size=args.probe_size,
            probe_seed=args.probe_seed,
            timeout_recheck_s=args.timeout_recheck_s,
            timeout_recheck_max=args.timeout_recheck_max,
        )
    except NarrativeSelfConsistencyError as exc:
        # Wf-A164: a narrative number (or a quoted repo path) with no machine
        # backing is a DEFECT.  The offending artifact is written out as an
        # explicit FAIL so the failure is on disk, not only on stdout, and
        # the run exits non-zero.
        fail_paths = _write_fail_artifact(args.out, exc)
        print(f"FAIL: {exc}", file=sys.stderr)
        for tok, paths in sorted(exc.guard.get("unbacked_numbers", {}).items()):
            print(f"  unbacked {tok!r} in {paths}", file=sys.stderr)
        for tok, paths in sorted(exc.guard.get("stale_paths", {}).items()):
            print(f"  stale path {tok!r} in {paths}", file=sys.stderr)
        for path in exc.guard.get("unresolved_narrative_paths", []):
            print(f"  unresolved narrative path {path!r}", file=sys.stderr)
        for path in sorted(exc.guard.get("unclassified_leaves", {})):
            print(f"  unclassified string leaf {path!r}", file=sys.stderr)
        for path in fail_paths:
            print(f"  FAIL artifact: {path}", file=sys.stderr)
        return 1

    headline = artifact["headline"]
    counts = artifact["counts"]
    LOG.info(
        "wrote %s -- pb_valid %s/%s strict, %s/%s scored (wall %.1fs)",
        args.out,
        headline["n_pb_valid"],
        headline["n_rows_in_scope"],
        headline["n_pb_valid"],
        headline["n_scored"],
        artifact["protocol"]["wall_seconds"],
    )
    print(json.dumps({
        "out": str(args.out),
        "pb_valid_rate_strict": counts["pb_valid_rate_strict"],
        "pb_valid_rate_scored": counts["pb_valid_rate_scored"],
        "pb_valid_rate_completed_only": counts["pb_valid_rate_completed_only"],
        "n_pb_valid": counts["n_pb_valid"],
        "n_scored": counts["n_scored"],
        "n_rows_in_scope": counts["n_rows_in_scope"],
        "n_embed_failed": counts["n_embed_failed"],
        "n_unparseable": counts["n_unparseable"],
        "n_timeout": counts["n_timeout"],
        "n_forcefield_parameterisation_failures": counts[
            "n_forcefield_parameterisation_failures"
        ],
        "wall_seconds": artifact["protocol"]["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
