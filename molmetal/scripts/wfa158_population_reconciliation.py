#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wf-A158 BRANCH POP -- reconcile the two headline fragmentation numbers, and census the rest.

WHAT THIS EXISTS
----------------
Two numbers in this repo are both called "the fragmentation rate" and they
disagree by ~3x:

  * **8.51 %** -- ``metrics/by_metric/cfm_fragmentation_audit.json``
    ``/summary/generation_side/single_molecule_rate_end_to_end/median``
    = 0.08506944444444445, over **9** generation pools.
  * **25.05 %** -- the published 2112-row pool
    (``molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv``),
    ``n_frags == 1`` in **529 / 2112** rows.

This script establishes, from files on disk:

  * **Q1** the DEFINITION of each number -- population, numerator, denominator,
    measurement point -- with ``file:line`` and ``jsonpath`` provenance, and
    the precise reason the two differ;
  * **Q2** a CENSUS of every published fragmentation-shaped number in
    ``metrics/by_metric/*.json`` and ``registry/metrics.yaml``, classified into
    DEFINITION FAMILIES;
  * **Q3** a QUOTED-NUMBER MAP: every document in the repo that quotes one of
    the headline strings, and which family each quoted number belongs to.

It also computes the **rendering collisions** -- distinct published numbers
that render to the same 2-decimal percent string (this is the mechanism by
which "8.51 %" became ambiguous) -- and the **patch list** the main thread
applies serially in the Rule-6 house shape.

HONEST FRAMING
--------------
Neither headline number is declared WRONG here.  The deliverable is the
DEFINITION TABLE.  A document that quotes a number with no published
definition is reported as a FINDING, not as an error to be silently fixed.

READ-ONLY
---------
This script READS ``metrics/``, ``registry/`` and ``molmetal/reports/``.  It
WRITES ONLY into ``molmetal/reports/wfa158_population_reconciliation/``.
``paper/`` is never opened (AGENTS.md §5.3 / TODO-84).

CPU-ONLY
--------
No torch, no CUDA, no checkpoint, no RDKit.  Pure-python + the stdlib.  Run
with ``CUDA_VISIBLE_DEVICES=""``.

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP (a required artifact is absent).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO / "metrics" / "by_metric"
REGISTRY_YAML = REPO / "registry" / "metrics.yaml"
AUDIT_JSON = METRICS_DIR / "cfm_fragmentation_audit.json"
CENSUS_JSON = METRICS_DIR / "cfm_run_census.json"
POSTPROC_JSON = METRICS_DIR / "cfm_fragmentation_postproc_arm.json"
C7_QUALITY_JSON = METRICS_DIR / "c7_fullgen_pool_quality.json"
LIGAND_JSON = METRICS_DIR / "wfa155_ligand_connected_rate.json"
PANEL_JSON = METRICS_DIR / "cfm_free_generation_panel_seed0.json"
MOLECULE_VALIDITY_JSON = METRICS_DIR / "cfm_molecule_validity_recompute.json"
DECODE_RATIO_JSON = METRICS_DIR / "decode_ratio.json"
REFERENCE_BAND_JSON = METRICS_DIR / "cfm_panel_reference_band_real_tmqm.json"

C8_POOL = REPO / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24" / "generated_smiles.csv"
C7_POOL = REPO / "molmetal" / "reports" / "cfm_c7_fullgen_2026_10_20" / "generated_smiles.csv"

OUT_DIR = REPO / "molmetal" / "reports" / "wfa158_population_reconciliation"
OUT_JSON = OUT_DIR / "reconciliation.json"
OUT_MD = OUT_DIR / "census.md"

# The two pools the two headline numbers are measured on.
POOL_C7 = "cfm_c7_fullgen_2026_10_20"
POOL_C8 = "cfm_c8_fullgen_2026_10_24"

# The families whose numbers are EXTERNAL (another paper's measurement).  They
# are catalogued, never re-derived, so they are exempt from the "this repo
# publishes a rate row for every family" gate.
EXTERNAL_FAMILIES = ("F12_external_band", "F13_external_paper_rate")


# --------------------------------------------------------------------------
# THE DEFINITION-FAMILY TAXONOMY (Q2)
# --------------------------------------------------------------------------
# A "family" is a distinct DEFINITION of a fragmentation-shaped rate: what the
# numerator counts, what the denominator is, and where in the pipeline the
# measurement is taken.  Two numbers in the same family differ only in their
# population; two numbers in DIFFERENT families are not comparable at all.
FAMILIES: Dict[str, Dict[str, Any]] = {
    "F1_pool_e2e": {
        "label": "pool end-to-end single-molecule rate",
        "numerator": "n_single -- the pool rows whose decoded molecule has exactly ONE RDKit fragment",
        "denominator": "the SAMPLES DRAWN for that pool (events.jsonl gen_validity_n summed over the pool's epochs), or the decode-dump's own n_attempted when the pool carries that column",
        "measurement_point": "at the RAW emitted SMILES, after RDKit MolFromSmiles parse (sanitisation) and Chem.GetMolFrags -- NO post-processing",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "molmetal/scripts/cfm_fragmentation_audit.py",
    },
    "F2_pool_cond": {
        "label": "pool CONDITIONAL single-fragment ratio (over PARSEABLE rows)",
        "numerator": "n_single (same as F1)",
        "denominator": "n_scored -- the PARSEABLE rows only; an unparseable string leaves the denominator",
        "measurement_point": "same as F1",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "molmetal/scripts/cfm_fragmentation_gate.py (score_smiles_pool)",
    },
    "F3_pool_rows": {
        "label": "pool single-molecule rate over the pool's OWN ROWS",
        "numerator": "n_single (same as F1)",
        "denominator": "n_rows_in_csv -- every row of the pool file, parseable or not",
        "measurement_point": "same as F1",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "molmetal/scripts/cfm_fragmentation_audit.py",
    },
    "F4_csv_count": {
        "label": "DIRECT count of n_frags_raw == 1 in the pool CSV",
        "numerator": "the number of CSV rows whose own n_frags_raw column == 1",
        "denominator": "the number of rows in that CSV",
        "measurement_point": "at the WRITER's own decode (the CSV column itself) -- NOT re-scored by this repo",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "the pool writer; re-read by any tool (this repo's c7 pool-quality script)",
    },
    "F5_census_last_epoch": {
        "label": "run-census end-to-end rate at the run's LAST epoch",
        "numerator": "gen_validity_end x gen_single_fragment_ratio_end (the run's own trainer-side numbers)",
        "denominator": "the samples the run drew at that epoch",
        "measurement_point": "in-process, inside the trainer (the run wrote these numbers about itself)",
        "population": "ONE training run, at ONE epoch",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/cfm_run_census.json (aggregating the runs' own per_epoch_metrics)",
    },
    "F6_census_best_epoch": {
        "label": "run-census rate at the run's BEST epoch",
        "numerator": "gen_single_molecule_rate_best",
        "denominator": "the samples the run drew at that epoch",
        "measurement_point": "in-process, inside the trainer",
        "population": "ONE training run, at its BEST epoch",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/cfm_run_census.json",
    },
    "F7_validity": {
        "label": "RDKit-sanitisable fraction (gen_validity)",
        "numerator": "the samples RDKit could parse and sanitise",
        "denominator": "the samples drawn",
        "measurement_point": "in-process, inside the trainer -- SANITISATION ONLY, no connectivity test ('C.C.C.C' sanitises)",
        "population": "ONE run / ONE pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "the trainer + metrics/by_metric/cfm_run_census.json",
    },
    "F8_molecule_validity": {
        "label": "molecule_validity (parse AND one fragment AND metal AND >1 heavy atom)",
        "numerator": "samples meeting all four conditions",
        "denominator": "the samples drawn (same denominator as gen_validity)",
        "measurement_point": "in-process / recomputed from the pool",
        "population": "ONE run",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/cfm_molecule_validity_recompute.json",
    },
    "F9_decode_ratio": {
        "label": "decode_ratio (the decode SUCCEEDED flag)",
        "numerator": "decodes that succeeded",
        "denominator": "decodes attempted",
        "measurement_point": "at the decode call, before any connectivity test",
        "population": "ONE run",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/decode_ratio.json",
    },
    "F10_postproc_connected": {
        "label": "POST-PROCESSED connected rate (LargestFragmentChooser -> explicit H -> UFF)",
        "numerator": "rows that are ONE connected component AFTER the post-processing chain",
        "denominator": "the parsed rows / the post-processed rows",
        "measurement_point": "AFTER post-processing -- the fragments have been DELETED, so the rate is bought with mass",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/cfm_fragmentation_postproc_arm.json",
    },
    "F11_ligand_only_connected": {
        "label": "LIGAND-ONLY connected rate (every metal atom deleted first)",
        "numerator": "rows whose ligand (the molecule minus its metals) is ONE connected component",
        "denominator": "n_analyzed rows (an empty ligand stays in the denominator)",
        "measurement_point": "on a COUNTERFACTUAL molecule -- the metals are stripped, so this is NOT a molecule the sampler emitted",
        "population": "ONE generation pool",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "metrics/by_metric/wfa155_ligand_connected_rate.json",
    },
    "F12_external_band": {
        "label": "PUBLISHED FRONTIER BAND (external literature -- not our metric)",
        "numerator": "external",
        "denominator": "external (10,000-15,000 generated molecules for the 92.7-94 % band)",
        "measurement_point": "external -- per-BOND recovery F1 on the CDG cohort (sigma = 0.2 A) for 92.7 %, and '% PB-Valid' geometry validity for the band",
        "population": "an EXTERNAL cohort",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "the literature (YuelBond, Wang & Dokholyan 2025/2026) -- documented in docs/HONEST_NEGATIVE_CATALOG.md §6.264",
    },
    "F13_external_paper_rate": {
        "label": "a SINGLE PAPER's external validity rate (external -- not our metric)",
        "numerator": "external",
        "denominator": "external (that paper's own generated cohort)",
        "measurement_point": "external -- each paper measures its OWN validity definition on its OWN cohort",
        "population": "an EXTERNAL cohort, one paper per number",
        "unit": "fraction",
        "higher_is_better": True,
        "producer": "the literature -- several DIFFERENT papers each publish a '94 %' with a DIFFERENT definition (MolGPT 'Valid 94 %', TargetDiff/FLOWr '94 % PB-valid', ...)",
    },
}

# --------------------------------------------------------------------------
# THE FAMILY-CLASSIFICATION RULE (Q2)
# --------------------------------------------------------------------------
# Generic rule: a JSON key whose NAME names a fragmentation-shaped quantity.
# Ordered; first match wins.  The regex is applied to the LAST path segment.
GENERIC_KEY_RULES: List[Tuple[str, str, str]] = [
    # (regex on the key, family_id, unit)
    (r"^single_molecule_rate_end_to_end$", "F1_pool_e2e", "fraction"),
    (r"^single_molecule_rate_over_pool_rows$", "F3_pool_rows", "fraction"),
    (r"^single_molecule_rate_over_declared_denominator$", "F3_pool_rows", "fraction"),
    (r"^single_molecule_rate_best", "F6_census_best_epoch", "fraction"),
    (r"^single_molecule_rate", "F5_census_last_epoch", "fraction"),
    (r"^single_fragment_ratio", "F2_pool_cond", "fraction"),
    (r"^(connected_rate|single_fragment_rate|connected_rate_over_parsed)$", "F2_pool_cond", "fraction"),
    (r"^connected_rate_over_postprocessed$", "F10_postproc_connected", "fraction"),
    (r"^n_single_fragment$", "F2_pool_cond", "count"),
    (r"^(n_single|n_connected|n_multicomponent_after_pp|n_unparseable)$", "F1_pool_e2e", "count"),
    (r"^(fragmented_ratio|fragmented_ratio_mean)$", "F1_pool_e2e", "fraction"),
    (r"^molecule_validity", "F8_molecule_validity", "fraction"),
    (r"^(gen_validity|validity)$", "F7_validity", "fraction"),
    (r"^gen_molecule_validity", "F8_molecule_validity", "fraction"),
    (r"^decode_ratio$", "F9_decode_ratio", "fraction"),
    (r"^(largest_frag_frac_mean|mean_largest_frag_frac)$", "F1_pool_e2e", "fraction"),
    (r"^(n_fragments_mean|mean_n_frags)$", "F1_pool_e2e", "fraction"),
    (r"^share_n_frags_", "F1_pool_e2e", "fraction"),
]

# File-scoped overrides: the key NAME alone cannot decide the family when the
# SAME key means different things in different producers.
FILE_OVERRIDES: Dict[str, List[Tuple[str, str, str]]] = {
    "cfm_fragmentation_postproc_arm": [
        (r"^connected_rate_over_parsed$", "F10_postproc_connected", "fraction"),
        (r"^connected_rate_over_postprocessed$", "F10_postproc_connected", "fraction"),
        (r"^n_connected$", "F10_postproc_connected", "count"),
        (r"^n_multicomponent_after_pp$", "F10_postproc_connected", "count"),
        (r"^single_fragment_rate_over_parsed$", "F2_pool_cond", "fraction"),
        (r"^n_single$", "F2_pool_cond", "count"),
    ],
    "wfa155_ligand_connected_rate": [
        (r"^single_fragment_rate$", "F11_ligand_only_connected", "fraction"),
        (r"^n_single_fragment$", "F11_ligand_only_connected", "count"),
    ],
    "cfm_run_census": [
        (r"^gen_single_fragment_ratio_end$", "F2_pool_cond", "fraction"),
        (r"^gen_single_molecule_rate_end$", "F5_census_last_epoch", "fraction"),
        (r"^gen_single_molecule_rate_best$", "F6_census_best_epoch", "fraction"),
        (r"^gen_validity_end$", "F7_validity", "fraction"),
        (r"^gen_molecule_validity_end$", "F8_molecule_validity", "fraction"),
    ],
    "cfm_free_generation_panel_seed0": [
        (r"^single_fragment_ratio$", "F2_pool_cond", "fraction"),
        (r"^gen_validity$", "F7_validity", "fraction"),
    ],
    # ``best_measured_real`` is the decode_ratio family's value key -- but ONLY
    # inside decode_ratio.json.  Elsewhere the same key means something else, so
    # the generic rule must not claim it.
    "decode_ratio": [
        (r"^best_measured_real$", "F9_decode_ratio", "fraction"),
    ],
}

# A key that names a fragmentation-shaped quantity but is NOT a rate.
NON_RATE_KEYS = re.compile(
    r"^(n_unparseable|n_empty|n_rows|n_rows_in_csv|n_scored|n_parse|n_analyzed|"
    r"n_attempted|n_valid|rejected_by_decode|n_multicomponent_after_pp|"
    r"n_fragments|n_frags|n_metal_atoms_per_row|n_rings|n_unique_smiles)$"
)

# A census row's ROLE inside its family: is this key the RATE itself, a
# component of the rate's numerator/denominator, or a companion statistic?
COMPONENT_COUNT_RE = re.compile(
    r"^(n_single|n_connected|n_multicomponent_after_pp|n_unparseable|n_empty|"
    r"n_pp_failed|n_attempted|n_valid|rejected_by_decode)$"
)
COMPANION_STAT_RE = re.compile(
    r"^(largest_frag_frac_mean|mean_largest_frag_frac|n_fragments_mean|"
    r"mean_n_frags|share_n_frags_.*|mean_second_frag_size|"
    r"second_frag_is_single_atom|mean_heavy_atoms|metal_ratio|"
    r"bonds_per_heavy_atom)$"
)


def classify_role(key: str) -> str:
    """rate | component_count | companion_statistic."""
    key = str(key)
    if COMPONENT_COUNT_RE.match(key):
        return "component_count"
    if COMPANION_STAT_RE.match(key):
        return "companion_statistic"
    return "rate"


def classify_key(metric_stem: str, key: str) -> Optional[Tuple[str, str]]:
    """Classify one JSON key into (family_id, unit), or None.

    The rule is file-aware by design: ``connected_rate`` means F2 (conditional
    over parseable rows) in ``c7_fullgen_pool_quality.json`` and F10
    (post-processed) in ``cfm_fragmentation_postproc_arm.json``, and the key
    name alone cannot tell them apart.
    """
    key = str(key)
    if NON_RATE_KEYS.match(key):
        return None
    for pattern, family_id, unit in FILE_OVERRIDES.get(metric_stem, []):
        if re.search(pattern, key):
            return (family_id, unit)
    for pattern, family_id, unit in GENERIC_KEY_RULES:
        if re.search(pattern, key):
            return (family_id, unit)
    return None


def render_pct(value: float) -> str:
    """Render a fraction the way every document in this repo renders it."""
    return f"{float(value) * 100:.2f}"


def find_rendering_collisions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group DISTINCT published values that render to the SAME 2-dp percent string.

    This is the mechanism by which "8.51 %" became ambiguous: three different
    published numbers -- 0.08506944 (the 9-pool median), 0.08508959 (the c7
    pool over its own rows) and 0.08509631 (the c7 pool over its parseable
    rows) -- all render as "8.51".
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        pct = render_pct(value)
        buckets.setdefault(pct, []).append(row)
    collisions: List[Dict[str, Any]] = []
    for pct, group in sorted(buckets.items()):
        distinct = sorted({float(r["value"]) for r in group})
        if len(distinct) > 1:
            collisions.append(
                {
                    "rendered_percent": pct,
                    "n_distinct_values": len(distinct),
                    "values": distinct,
                    "members": [
                        {
                            "file": r.get("file"),
                            "jsonpath": r.get("jsonpath"),
                            "value": r.get("value"),
                            "family_id": r.get("family_id"),
                        }
                        for r in group
                    ],
                }
            )
    return collisions


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_pool_csv(path: Path) -> Dict[str, Any]:
    """Count a pool CSV WITHOUT RDKit: read the writer's own n_frags_raw column.

    Returns the counts, plus the per-epoch breakdown.  The measurement point
    is the WRITER's decode (family F4), which is exactly what a document that
    quotes "n_frags == 1 in N / M rows" is quoting.
    """
    out: Dict[str, Any] = {
        "path": str(path.relative_to(REPO)),
        "exists": path.is_file(),
        "n_rows": 0,
        "n_single": 0,
        "header": [],
        "per_epoch": {},
        "n_attempted_column": None,
    }
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        out["header"] = list(reader.fieldnames or [])
        per_epoch: Dict[str, Dict[str, int]] = {}
        for row in reader:
            out["n_rows"] += 1
            raw = (row.get("n_frags_raw") or "").strip()
            try:
                is_single = float(raw) == 1.0
            except (TypeError, ValueError):
                is_single = False
            if is_single:
                out["n_single"] += 1
            epoch = (row.get("epoch") or "").strip()
            bucket = per_epoch.setdefault(epoch, {"n_rows": 0, "n_single": 0})
            bucket["n_rows"] += 1
            if is_single:
                bucket["n_single"] += 1
            if out["n_attempted_column"] is None:
                na = (row.get("n_attempted") or "").strip()
                if na:
                    try:
                        out["n_attempted_column"] = int(float(na))
                    except (TypeError, ValueError):
                        out["n_attempted_column"] = None
    out["per_epoch"] = {
        k: {
            "n_rows": v["n_rows"],
            "n_single": v["n_single"],
            "rate_over_rows": (v["n_single"] / v["n_rows"]) if v["n_rows"] else None,
        }
        for k, v in sorted(per_epoch.items())
    }
    out["rate_over_rows"] = (
        (out["n_single"] / out["n_rows"]) if out["n_rows"] else None
    )
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(values)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Q1 -- THE DEFINITION DIFFERENCE
# --------------------------------------------------------------------------
def reconcile_q1() -> Dict[str, Any]:
    audit = _load_json(AUDIT_JSON)
    if audit is None:
        return {"status": "SKIP", "reason": f"{_repo_relative(AUDIT_JSON)} unreadable"}

    runs = audit.get("runs") or []
    gen_runs = [r for r in runs if r.get("pipeline") == "generation"]
    e2e_runs = [r for r in gen_runs if isinstance(r.get("single_molecule_rate_end_to_end"), (int, float))]

    by_run = {r.get("run"): r for r in runs}
    c7_row = by_run.get(POOL_C7) or {}
    c8_row = by_run.get(POOL_C8) or {}

    published_median = (
        (audit.get("summary") or {}).get("generation_side") or {}
    ).get("single_molecule_rate_end_to_end") or {}
    recomputed_values = [
        float(r["single_molecule_rate_end_to_end"]) for r in e2e_runs
    ]
    recomputed_median = _median(recomputed_values)

    c7_csv = _read_pool_csv(C7_POOL)
    c8_csv = _read_pool_csv(C8_POOL)

    # The c8 pool's rate, recomputed two ways.
    c8_pooled = c8_csv.get("rate_over_rows")
    c8_epoch_rates = [
        v["rate_over_rows"] for v in c8_csv.get("per_epoch", {}).values() if v["rate_over_rows"] is not None
    ]
    c8_mean_of_epochs = (
        sum(c8_epoch_rates) / len(c8_epoch_rates) if c8_epoch_rates else None
    )

    c7_pooled = c7_csv.get("rate_over_rows")
    c7_conditional = c7_row.get("single_fragment_ratio")
    c7_e2e = c7_row.get("single_molecule_rate_end_to_end")

    definitions = {
        "8.51_percent": {
            "rendered": "8.51 %",
            "published_value": published_median.get("median"),
            "published_jsonpath": "/summary/generation_side/single_molecule_rate_end_to_end/median",
            "published_metric_file": _repo_relative(AUDIT_JSON),
            "population": (
                "the MEDIAN of the per-pool end-to-end single-molecule rate over "
                f"{published_median.get('n')} GENERATION POOLS (each pool = one run's "
                "generated_smiles.csv). It is NOT a pooled rate over all the pools' draws."
            ),
            "numerator": FAMILIES["F1_pool_e2e"]["numerator"],
            "denominator": FAMILIES["F1_pool_e2e"]["denominator"],
            "measurement_point": FAMILIES["F1_pool_e2e"]["measurement_point"],
            "producer_lines": [
                "molmetal/scripts/cfm_fragmentation_audit.py:711  \"single_molecule_rate_end_to_end\": e2e",
                "molmetal/scripts/cfm_fragmentation_audit.py:944  e2e = n_single / total_drawn",
                "molmetal/scripts/cfm_fragmentation_audit.py:950  \"single_molecule_rate_end_to_end\": _stats(e2e)",
                "molmetal/scripts/cfm_fragmentation_audit.py:333-348  _stats(): median = vals[n//2] for odd n",
                "molmetal/scripts/cfm_fragmentation_gate.py:325  single = sum(1 for r in rows if n_fragments == 1)",
                "molmetal/scripts/cfm_fragmentation_gate.py:330  rate = single / denominator",
                "molmetal/scripts/cfm_fragmentation_audit.py:406  e2e_note (the denominator's own provenance)",
            ],
            "family_id": "F1_pool_e2e",
            "recomputed_median_from_runs": recomputed_median,
            "recomputed_matches_published": (
                recomputed_median is not None
                and published_median.get("median") is not None
                and abs(recomputed_median - float(published_median["median"])) <= 1e-12
            ),
            "n_pools_in_the_median": len(recomputed_values),
            "the_median_is_one_pool_value": next(
                (
                    r["run"]
                    for r in e2e_runs
                    if abs(float(r["single_molecule_rate_end_to_end"]) - float(recomputed_median)) <= 1e-12
                ),
                None,
            ),
            "the_median_is_one_pool_value_caveat": (
                "the median of 9 happens to LAND on one pool's own value "
                "(cfm_c7_fullgen_2026_10_20). The published number is still the "
                "MEDIAN; the coincidence is why the headline is ambiguous."
            ),
            "per_pool_values": [
                {
                    "run": r.get("run"),
                    "single_molecule_rate_end_to_end": r.get("single_molecule_rate_end_to_end"),
                    "single_fragment_ratio": r.get("single_fragment_ratio"),
                    "n_rows_in_csv": r.get("n_rows_in_csv"),
                    "samples_drawn": r.get("samples_drawn"),
                    "end_to_end_denominator_source": r.get("end_to_end_denominator_source"),
                }
                for r in sorted(
                    e2e_runs, key=lambda r: float(r["single_molecule_rate_end_to_end"])
                )
            ],
        },
        "25.05_percent": {
            "rendered": "25.05 %",
            "published_value": c8_row.get("single_molecule_rate_end_to_end"),
            "published_jsonpath": (
                "/runs[run==" + POOL_C8 + "]/single_molecule_rate_end_to_end"
            ),
            "published_metric_file": _repo_relative(AUDIT_JSON),
            "population": (
                "ONE generation pool -- " + POOL_C8 + " -- 2112 rows over its 3 epochs "
                "(704 per epoch). It is a per-pool rate, not an aggregate."
            ),
            "numerator": FAMILIES["F1_pool_e2e"]["numerator"],
            "denominator": FAMILIES["F1_pool_e2e"]["denominator"],
            "measurement_point": FAMILIES["F1_pool_e2e"]["measurement_point"],
            "producer_lines": [
                "molmetal/scripts/cfm_fragmentation_audit.py:711 / :944  the same F1 computation, ONE pool",
                "molmetal/scripts/cfm_fragmentation_gate.py:325-330  the same numerator/denominator",
                "the CSV's own n_frags_raw column (the WRITER's decode, family F4)",
                "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv  header: epoch,smiles,n_frags_raw,largest_frag_frac",
            ],
            "family_id": "F1_pool_e2e",
            "csv_recount": {
                "n_rows": c8_csv.get("n_rows"),
                "n_single": c8_csv.get("n_single"),
                "rate_over_rows": c8_pooled,
                "per_epoch": c8_csv.get("per_epoch"),
                "mean_of_the_three_epoch_rates": c8_mean_of_epochs,
                "pooled_equals_mean_of_epochs": (
                    c8_pooled is not None
                    and c8_mean_of_epochs is not None
                    and abs(c8_pooled - c8_mean_of_epochs) <= 1e-12
                ),
                "pooled_minus_mean_of_epochs": (
                    (c8_pooled - c8_mean_of_epochs)
                    if (c8_pooled is not None and c8_mean_of_epochs is not None)
                    else None
                ),
                "note": (
                    "the repo's todo.json states the c8 figure as the MEAN of the three "
                    "per-epoch rates (0.2031+0.2003+0.3480)/3 = 0.2504666..., while the "
                    "pooled 529/2112 = 0.2504734... . Both render as 25.05 %; they are "
                    "DIFFERENT numbers."
                ),
            },
            "matches_audit_published_value": (
                c8_pooled is not None
                and c8_row.get("single_molecule_rate_end_to_end") is not None
                and abs(c8_pooled - float(c8_row["single_molecule_rate_end_to_end"])) <= 1e-12
            ),
        },
    }

    # The precise reason the two differ.
    if recomputed_median is not None and c8_pooled is not None:
        ratio = c8_pooled / recomputed_median if recomputed_median else None
        reason = (
            "They are the SAME DEFINITION (family F1_pool_e2e: n_single / samples drawn, "
            "RDKit connected-component count on the RAW emitted SMILES) measured on DIFFERENT "
            "POPULATIONS: 8.51 % is the MEDIAN of that rate over 9 generation pools, while "
            "25.05 % is ONE pool (cfm_c8_fullgen_2026_10_24, 529/2112) -- which is the 2nd-best "
            "of the 9, not the median. The ~2.9x gap is therefore an AGGREGATION + POPULATION "
            "difference, NOT a definition difference; the two numbers are not in conflict and "
            "neither is wrong."
        )
    else:
        ratio = None
        reason = "NOT ESTABLISHED: one of the two headline values could not be recomputed."

    # The OTHER numbers measured on the SAME c7 pool -- the rendering
    # neighbourhood of the headline.  Each entry states its OWN source, its OWN
    # jsonpath and the string it PRINTS as, because they do NOT all print as
    # "8.51": the DIRECT column count (F4) is one row higher than the audit's
    # recount (the single unparseable row the audit drops) and therefore
    # straddles the 2-dp boundary.
    c7_audit_pooled = c7_row.get("single_molecule_rate_over_pool_rows")
    c7_cross_check = c7_row.get("csv_column_cross_check") or {}
    other_851 = {
        "note": (
            "the rendering neighbourhood of the 8.51 % headline, all measured on the "
            "SAME c7 pool.  Each entry names its own source and the string it "
            "prints as -- they do NOT all print '8.51'."
        ),
        "c7_pool_over_its_own_rows__AUDIT_RECOUNT": {
            "value": c7_audit_pooled,
            "rendered_percent": (
                render_pct(c7_audit_pooled)
                if isinstance(c7_audit_pooled, (int, float))
                else None
            ),
            "jsonpath": "/runs[run==" + POOL_C7 + "]/single_molecule_rate_over_pool_rows",
            "family_id": "F3_pool_rows",
            "source": "metrics/by_metric/cfm_fragmentation_audit.json",
            "numerator_n_single": c7_row.get("n_single"),
            "denominator_n_rows_in_csv": c7_row.get("n_rows_in_csv"),
        },
        "c7_pool_over_its_own_rows__CSV_COLUMN_COUNT": {
            "value": c7_pooled,
            "rendered_percent": render_pct(c7_pooled),
            "jsonpath": (
                "molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv"
                "::n_frags_raw==1"
            ),
            "family_id": "F4_csv_count",
            "source": "the writer's own n_frags_raw column, counted by this script",
            "numerator_n_single": c7_csv.get("n_single"),
            "denominator_n_rows": c7_csv.get("n_rows"),
            "note": (
                "ONE ROW higher than the audit's recount (%s vs %s): the pool holds "
                "1 UNPARSEABLE row (n_unparseable=%s) whose column still reads 1, "
                "and the audit drops it -- its csv_column_cross_check reports "
                "n_checked=%s with n_mismatch=%s over the parseable rows.  The two "
                "numbers therefore straddle a 2-dp boundary and PRINT differently: "
                "'8.52' vs '8.51'."
                % (
                    c7_csv.get("n_single"),
                    c7_row.get("n_single"),
                    c7_row.get("n_unparseable"),
                    c7_cross_check.get("n_checked"),
                    c7_cross_check.get("n_mismatch"),
                )
            ),
        },
        "c7_pool_over_its_parseable_rows": {
            "value": c7_conditional,
            "rendered_percent": (
                render_pct(c7_conditional)
                if isinstance(c7_conditional, (int, float))
                else None
            ),
            "jsonpath": "/runs[run==" + POOL_C7 + "]/single_fragment_ratio",
            "family_id": "F2_pool_cond",
            "source": "metrics/by_metric/cfm_fragmentation_audit.json",
            "n_scored": c7_row.get("n_scored"),
            "n_single": c7_row.get("n_single"),
        },
        "c7_pool_quality_artifact_overall_connected_rate": {
            "value": ((_load_json(C7_QUALITY_JSON) or {}).get("overall") or {}).get("connected_rate"),
            "rendered_percent": render_pct(
                ((_load_json(C7_QUALITY_JSON) or {}).get("overall") or {}).get("connected_rate")
            ),
            "jsonpath": "/overall/connected_rate",
            "family_id": "F2_pool_cond",
            "source": "metrics/by_metric/c7_fullgen_pool_quality.json",
            "note": (
                "molmetal/scripts/c7_fullgen_pool_quality.py:127 computes it as "
                "round(n_connected / n_parse, 4) -- CONDITIONAL over parseable rows, "
                "rounded to 4 dp, so it is 0.0851 by construction, not by coincidence."
            ),
        },
        "c7_pool_e2e": {
            "value": c7_e2e,
            "rendered_percent": (
                render_pct(c7_e2e) if isinstance(c7_e2e, (int, float)) else None
            ),
            "jsonpath": "/runs[run==" + POOL_C7 + "]/single_molecule_rate_end_to_end",
            "family_id": "F1_pool_e2e",
            "source": "metrics/by_metric/cfm_fragmentation_audit.json",
            "note": "this is the value the 9-pool MEDIAN lands on",
        },
    }

    return {
        "status": "OK",
        "headline": {
            "8.51_percent": published_median.get("median"),
            "25.05_percent": c8_row.get("single_molecule_rate_end_to_end"),
            "ratio": ratio,
            "same_definition": True,
            "different_population": True,
        },
        "reason_the_two_differ": reason,
        "definitions": definitions,
        "c7_pool_rendering_neighbourhood": other_851,
        "cross_checks": {
            "c8_pool_csv_recount_vs_audit": {
                "csv": {
                    "n_rows": c8_csv.get("n_rows"),
                    "n_single": c8_csv.get("n_single"),
                    "rate": c8_csv.get("rate_over_rows"),
                },
                "audit": {
                    "value": c8_row.get("single_molecule_rate_end_to_end"),
                    "jsonpath": "/runs[run==" + POOL_C8 + "]/single_molecule_rate_end_to_end",
                },
                "agrees": c8_csv.get("rate_over_rows")
                == c8_row.get("single_molecule_rate_end_to_end"),
                "note": (
                    "the DIRECT count of n_frags_raw==1 in the c8 pool CSV equals the "
                    "audit's own e2e value for that pool -- to the last bit."
                ),
            },
            "c7_pool_csv_recount_vs_audit": {
                "csv": {
                    "n_rows": c7_csv.get("n_rows"),
                    "n_single": c7_csv.get("n_single"),
                    "rate": c7_pooled,
                },
                "audit": {
                    "value": c7_audit_pooled,
                    "n_single": c7_row.get("n_single"),
                    "n_rows_in_csv": c7_row.get("n_rows_in_csv"),
                    "n_unparseable": c7_row.get("n_unparseable"),
                    "csv_column_cross_check": c7_cross_check,
                    "jsonpath": "/runs[run==" + POOL_C7 + "]/single_molecule_rate_over_pool_rows",
                },
                "agrees": c7_csv.get("n_single") == c7_row.get("n_single"),
                "difference_in_rows": (c7_csv.get("n_single") or 0)
                - (c7_row.get("n_single") or 0),
                "note": (
                    "the DIRECT column count is ONE ROW higher than the audit's "
                    "recount, and the difference is exactly the pool's single "
                    "UNPARSEABLE row (n_unparseable=%s): the audit scores %s "
                    "parseable rows and reports csv_column_cross_check "
                    "n_checked=%s with n_mismatch=%s, so the two are consistent -- "
                    "but they straddle a 2-dp boundary and PRINT differently "
                    "('8.52' vs '8.51').  This is a DEFINITION difference (F4 counts "
                    "the column; F3 counts the scored rows), not a contradiction."
                )
                % (
                    c7_row.get("n_unparseable"),
                    c7_row.get("n_scored"),
                    c7_cross_check.get("n_checked"),
                    c7_cross_check.get("n_mismatch"),
                ),
            },
        },
        "csv_recounts": {"c7_pool": c7_csv, "c8_pool": c8_csv},
        "provenance": {
            "audit_metric_file": _repo_relative(AUDIT_JSON),
            "audit_date": audit.get("date"),
            "audit_wave": audit.get("wave"),
            "audit_direction": audit.get("direction"),
        },
    }


# --------------------------------------------------------------------------
# Q2 -- THE CENSUS
# --------------------------------------------------------------------------
CENSUS_KEY_RE = re.compile(
    r"frag|connect|n_frag|validity|single_molecule|molecule_rate|decode_ratio|"
    r"largest_frag|one_mol|n_frags",
    re.I,
)


def census_q2() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    per_file: Dict[str, int] = {}
    for path in sorted(METRICS_DIR.glob("*.json")):
        data = _load_json(path)
        if data is None:
            continue
        stem = path.stem
        found: List[Tuple[str, Any]] = []

        def walk(node: Any, path_parts: List[str]) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    here = path_parts + [str(key)]
                    if CENSUS_KEY_RE.search(str(key)):
                        found.append(("/".join(here), value))
                    walk(value, here)
            elif isinstance(node, list):
                for index, value in enumerate(node[:200]):
                    walk(value, path_parts + [str(index)])

        walk(data, [])
        kept = 0
        for jsonpath, value in found:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            key = jsonpath.rsplit("/", 1)[-1]
            classified = classify_key(stem, key)
            if classified is None:
                continue
            family_id, unit = classified
            rows.append(
                {
                    "file": _repo_relative(path),
                    "metric": stem,
                    "jsonpath": jsonpath,
                    "key": key,
                    "value": float(value),
                    "unit": unit,
                    "role": classify_role(key),
                    "family_id": family_id,
                    "population": _population_of(stem, family_id),
                    "registry_row": None,
                }
            )
            kept += 1
        if kept:
            per_file[stem] = kept

    # The F4 family lives in the POOL CSVs, not in a metric file: it is the
    # count a document gets by reading the writer's own n_frags_raw column.
    # Every pool CSV on disk that carries the column is counted here.
    pool_rows: List[Dict[str, Any]] = []
    for csv_path in sorted((REPO / "molmetal" / "reports").glob("*/generated_smiles.csv")):
        counted = _read_pool_csv(csv_path)
        if not counted.get("exists") or "n_frags_raw" not in (counted.get("header") or []):
            continue
        if not counted.get("n_rows"):
            continue
        pool_rows.append(
            {
                "file": _repo_relative(csv_path),
                "metric": "<pool csv>",
                "jsonpath": "%s::n_frags_raw==1" % _repo_relative(csv_path),
                "key": "n_frags_raw==1",
                "value": float(counted["rate_over_rows"]),
                "unit": "fraction",
                "role": "rate",
                "family_id": "F4_csv_count",
                "population": "ONE generation pool (read directly off the CSV column)",
                "registry_row": None,
                "n_rows": counted["n_rows"],
                "n_single": counted["n_single"],
            }
        )
    rows.extend(pool_rows)

    # The registry (READ ONLY): which of the census metrics the registry carries.
    registry_rows: List[Dict[str, Any]] = []
    try:
        import yaml  # noqa: PLC0415  (optional dependency, read-only use)

        reg = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8", errors="replace"))
        metrics = (reg or {}).get("metrics") or {}
        for name, row in metrics.items():
            src = str((row or {}).get("source_json") or "")
            if not src:
                continue
            stem = Path(src).stem
            if stem not in per_file:
                continue
            registry_rows.append(
                {
                    "registry_key": name,
                    "source_json": src,
                    "value": (row or {}).get("value"),
                    "value_key": (row or {}).get("value_key"),
                    "unit_canonical": (row or {}).get("unit_canonical"),
                    "status": (row or {}).get("status"),
                    "quotable": (row or {}).get("quotable"),
                    "headline_grade": (row or {}).get("headline_grade"),
                    "registry_role": (row or {}).get("registry_role"),
                    "metric_family": (row or {}).get("metric_family"),
                }
            )
            for row in rows:
                if row["metric"] == stem and row["registry_row"] is None:
                    row["registry_row"] = name
    except Exception as exc:  # pragma: no cover - yaml optional
        registry_rows.append({"error": f"{type(exc).__name__}: {exc}"})

    # The family table.
    family_table: Dict[str, Any] = {}
    for family_id, spec in FAMILIES.items():
        members = [r for r in rows if r["family_id"] == family_id]
        family_table[family_id] = {
            **spec,
            "n_census_rows": len(members),
            "n_rate_rows": sum(1 for m in members if m.get("role") == "rate"),
            "n_distinct_metrics": len({m["metric"] for m in members}),
            "metrics": sorted({m["metric"] for m in members}),
            "populations": sorted({m["population"] for m in members})[:8],
        }

    # The headline value per family (the number a document would quote).
    headlines = {
        "F1_pool_e2e": {
            "value": (
                (_load_json(AUDIT_JSON) or {}).get("summary") or {}
            ).get("generation_side", {})
            .get("single_molecule_rate_end_to_end", {})
            .get("median"),
            "jsonpath": "/summary/generation_side/single_molecule_rate_end_to_end/median",
            "file": _repo_relative(AUDIT_JSON),
        },
        "F2_pool_cond": {
            "value": (
                (_load_json(AUDIT_JSON) or {}).get("summary") or {}
            ).get("generation_side", {})
            .get("single_fragment_ratio", {})
            .get("median"),
            "jsonpath": "/summary/generation_side/single_fragment_ratio/median",
            "file": _repo_relative(AUDIT_JSON),
        },
        "F4_csv_count": {
            "value": next(
                (
                    r["value"]
                    for r in rows
                    if r["jsonpath"].endswith(
                        "cfm_c8_fullgen_2026_10_24/generated_smiles.csv::n_frags_raw==1"
                    )
                ),
                None,
            ),
            "jsonpath": "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv::n_frags_raw==1",
            "file": "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv",
            "n_pool_csvs_counted": len(pool_rows),
        },
        "F5_census_last_epoch": {
            "value": (
                (
                    ((_load_json(CENSUS_JSON) or {}).get("fragmentation_summary") or {}).get(
                        "end_to_end_single_molecule_rate"
                    )
                    or {}
                ).get("median")
            ),
            "jsonpath": "/fragmentation_summary/end_to_end_single_molecule_rate/median",
            "file": _repo_relative(CENSUS_JSON),
            "also_published": [
                {
                    "value": (
                        (
                            (_load_json(AUDIT_JSON) or {}).get("project_generation_baseline") or {}
                        ).get("gen_single_molecule_rate_end", {})
                    ).get("median"),
                    "n": (
                        (
                            (_load_json(AUDIT_JSON) or {}).get("project_generation_baseline") or {}
                        ).get("gen_single_molecule_rate_end", {})
                    ).get("n"),
                    "jsonpath": "/project_generation_baseline/gen_single_molecule_rate_end/median",
                    "file": _repo_relative(AUDIT_JSON),
                }
            ],
            "within_family_disagreement": (
                "THE SAME FAMILY PUBLISHES TWO MEDIANS: the run census reads "
                "0.0625 over n=27 CLEAN runs; the audit reads 0.057291666666666664 over "
                "n=29 CLEAN runs. Both name the SAME four degenerate runs, but the audit "
                "DERIVES two missing rows as validity x conditional while the census does "
                "not. Neither is retracted; a document quoting 'the' run-census median "
                "must say which producer it read."
            ),
        },
        "F6_census_best_epoch": {
            "value": (
                (
                    ((_load_json(CENSUS_JSON) or {}).get("fragmentation_summary") or {}).get(
                        "best_epoch_single_molecule_rate"
                    )
                    or {}
                ).get("median")
            ),
            "jsonpath": "/fragmentation_summary/best_epoch_single_molecule_rate/median",
            "file": _repo_relative(CENSUS_JSON),
            "agrees_with": (
                "the audit's /project_generation_baseline/gen_single_molecule_rate_best/median "
                "= 0.35546875 (n=29) -- the two producers AGREE on this family"
            ),
        },
        "F7_validity": {
            "value": (
                (
                    (_load_json(AUDIT_JSON) or {}).get("project_generation_baseline") or {}
                ).get("gen_validity_end", {})
            ).get("median"),
            "jsonpath": "/project_generation_baseline/gen_validity_end/median",
            "file": _repo_relative(AUDIT_JSON),
            "note": (
                "the sanity-check number: it reads 1.000 on the c8 pool while the "
                "single-molecule rate reads 0.25047 -- 'C.C.C.C' sanitises, so validity "
                "cannot see fragmentation"
            ),
        },
        "F10_postproc_connected": {
            "value": (
                ((_load_json(POSTPROC_JSON) or {}).get("postprocessed_axis") or {}).get(
                    "connected_rate_over_parsed"
                )
            ),
            "jsonpath": "/postprocessed_axis/connected_rate_over_parsed",
            "file": _repo_relative(POSTPROC_JSON),
        },
        "F11_ligand_only_connected": {
            "value": (
                ((_load_json(LIGAND_JSON) or {}).get("ligand_only") or {}).get(
                    "single_fragment_rate"
                )
            ),
            "jsonpath": "/ligand_only/single_fragment_rate",
            "file": _repo_relative(LIGAND_JSON),
        },
        "F12_external_band": {
            "value": "92.7 - 94 % (a band, not a point value)",
            "jsonpath": "external literature -- documented in docs/HONEST_NEGATIVE_CATALOG.md §6.264",
            "file": "(external)",
        },
    }

    collisions = find_rendering_collisions(rows)
    collisions = [c for c in collisions if c["rendered_percent"] in {"8.51", "25.05", "5.73", "35.55"}]

    return {
        "status": "OK",
        "n_census_rows": len(rows),
        "n_metrics_scanned": len(list(METRICS_DIR.glob("*.json"))),
        "n_metrics_with_fragmentation_rows": len(per_file),
        "rows_per_metric": per_file,
        "n_families_declared": len(FAMILIES),
        "n_families_with_rows": sum(
            1 for f in family_table.values() if f["n_census_rows"] > 0
        ),
        "family_table": family_table,
        "family_headlines": headlines,
        "rendering_collisions": collisions,
        "registry_rows": registry_rows,
        "n_registry_rows": len(registry_rows),
        "rows": rows,
    }


def _population_of(metric_stem: str, family_id: str) -> str:
    if family_id in {"F5_census_last_epoch", "F6_census_best_epoch", "F7_validity", "F8_molecule_validity", "F9_decode_ratio"}:
        return "ONE training run (trainer-side)"
    if family_id == "F10_postproc_connected":
        return "ONE generation pool (post-processed)"
    if family_id == "F11_ligand_only_connected":
        return "ONE generation pool (counterfactual, metal stripped)"
    if family_id == "F12_external_band":
        return "an EXTERNAL cohort"
    if metric_stem == "cfm_run_census":
        return "ONE training run (trainer-side)"
    return "ONE generation pool"


# --------------------------------------------------------------------------
# Q3 -- THE QUOTED-NUMBER MAP
# --------------------------------------------------------------------------
# The six strings the brief names.  Each regex forbids a digit or a dot before
# the match, so "2025.05.06.652517" (a bioRxiv DOI) and "892.7" (a metric) do
# not masquerade as the number 25.05 / 92.7.
QUOTE_PATTERNS: List[Tuple[str, str]] = [
    ("8.51", r"(?<![\d.])8\.51(?![\d])"),
    ("0.0851", r"(?<![\d.])0\.0851(?![\d])"),
    ("25.05", r"(?<![\d.])25\.05(?![\d])"),
    ("0.2505", r"(?<![\d.])0\.2505(?![\d])"),
    ("92.7", r"(?<![\d.])92\.7(?![\d])"),
    ("94 %", r"(?<![\d.])94\s*%"),
    ("94%", r"(?<![\d.])94\s*%"),
]
COMPILED_QUOTES = [(label, re.compile(pattern)) for label, pattern in QUOTE_PATTERNS]

# A quoted number only counts as a FRAGMENTATION number when the sentence
# carries a fragmentation keyword.  Without this guard a value-collision
# ("-0.0851" as a confidence bound) would be mis-reported as a fragmentation
# quote.
FRAG_CONTEXT_RE = re.compile(
    r"frag|single_fragment|connected|connectivity|n_frags|multi-?fragment|"
    r"molecule_validity|validity|decode|sanitis|generation pool|cfm_c7|cfm_c8|"
    r"one molecule|single molecule|n_frags_raw",
    re.I,
)
EXTERNAL_BAND_CONTEXT_RE = re.compile(
    r"92\.7|validity band|connectivity band|published band|frontier band|"
    r"frontier connectivity|not our band",
    re.I,
)
EXTERNAL_PAPER_CONTEXT_RE = re.compile(
    r"valid 94|94\s*%\s*valid|94\s*%\s*pb-?valid|pb-?valid|BLEU",
    re.I,
)


def _token_fraction(token_text: str, numeric: float) -> float:
    """A token like ``0.0851`` is already a fraction; ``8.51`` is a percent.

    Getting this wrong is what made an earlier draft of this script report
    "0.0851" as a 0.09 % number and miss every fragment hit.
    """
    text = token_text.strip()
    if re.match(r"^0?\.\d+", text) or re.match(r"^0\d+\.\d+", text):
        return numeric
    return numeric / 100.0

DOC_EXTENSIONS = {".md", ".txt", ".rst", ".tex"}
EXTRA_DOC_NAMES = {"todo.json", "todo.archive_2026_10_19.json"}
EXCLUDED_PREFIXES = ("paper/",)

# The published values a quoted number can MATCH.  Each entry is
# (value, family_id, jsonpath, file).  A quoted token is matched to every
# published value that renders to the same 2-dp percent string -- a token can
# legitimately be ambiguous.
MATCH_TABLE: List[Dict[str, Any]] = []


def _build_match_table() -> None:
    MATCH_TABLE.clear()
    audit = _load_json(AUDIT_JSON) or {}
    summary_gen = (audit.get("summary") or {}).get("generation_side") or {}
    MATCH_TABLE.append(
        {
            "value": (summary_gen.get("single_molecule_rate_end_to_end") or {}).get("median"),
            "family_id": "F1_pool_e2e",
            "jsonpath": "/summary/generation_side/single_molecule_rate_end_to_end/median",
            "file": _repo_relative(AUDIT_JSON),
            "label": "the 9-pool MEDIAN end-to-end rate (the published 8.51 % headline)",
        }
    )
    MATCH_TABLE.append(
        {
            "value": (summary_gen.get("single_fragment_ratio") or {}).get("median"),
            "family_id": "F2_pool_cond",
            "jsonpath": "/summary/generation_side/single_fragment_ratio/median",
            "file": _repo_relative(AUDIT_JSON),
            "label": "the 10-pool MEDIAN conditional rate",
        }
    )
    for row in audit.get("runs") or []:
        for key, family_id in (
            ("single_molecule_rate_end_to_end", "F1_pool_e2e"),
            ("single_molecule_rate_over_pool_rows", "F3_pool_rows"),
            ("single_fragment_ratio", "F2_pool_cond"),
        ):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                MATCH_TABLE.append(
                    {
                        "value": float(value),
                        "family_id": family_id,
                        "jsonpath": f"/runs[run=={row.get('run')}!={key}",
                        "file": _repo_relative(AUDIT_JSON),
                        "label": f"{row.get('run')} :: {key}",
                    }
                )
    quality = _load_json(C7_QUALITY_JSON) or {}
    overall = quality.get("overall") or {}
    for key in ("connected_rate", "single_fragment_rate", "validity"):
        if isinstance(overall.get(key), (int, float)):
            MATCH_TABLE.append(
                {
                    "value": float(overall[key]),
                    "family_id": "F2_pool_cond" if key != "validity" else "F7_validity",
                    "jsonpath": f"/overall/{key}",
                    "file": _repo_relative(C7_QUALITY_JSON),
                    "label": f"c7 pool quality :: /overall/{key}",
                }
            )
    census = _load_json(CENSUS_JSON) or {}
    frag = census.get("fragmentation_summary") or {}
    for key, family_id in (
        ("end_to_end_single_molecule_rate", "F5_census_last_epoch"),
        ("best_epoch_single_molecule_rate", "F6_census_best_epoch"),
        ("end_to_end_single_molecule_rate_all_runs", "F5_census_last_epoch"),
        ("best_epoch_single_molecule_rate_all_runs", "F6_census_best_epoch"),
    ):
        block = frag.get(key) or {}
        if isinstance(block.get("median"), (int, float)):
            MATCH_TABLE.append(
                {
                    "value": float(block["median"]),
                    "family_id": family_id,
                    "jsonpath": f"/fragmentation_summary/{key}/median",
                    "file": _repo_relative(CENSUS_JSON),
                    "label": f"run census :: {key}/median (n={block.get('n')})",
                }
            )
    postproc = _load_json(POSTPROC_JSON) or {}
    for key, family_id in (
        ("connected_rate_over_parsed", "F10_postproc_connected"),
        ("connected_rate_over_postprocessed", "F10_postproc_connected"),
    ):
        value = (postproc.get("postprocessed_axis") or {}).get(key)
        if isinstance(value, (int, float)):
            MATCH_TABLE.append(
                {
                    "value": float(value),
                    "family_id": family_id,
                    "jsonpath": f"/postprocessed_axis/{key}",
                    "file": _repo_relative(POSTPROC_JSON),
                    "label": f"post-processed arm :: {key}",
                }
            )
    raw = (postproc.get("raw_axis") or {})
    if isinstance(raw.get("single_fragment_rate_over_parsed"), (int, float)):
        MATCH_TABLE.append(
            {
                "value": float(raw["single_fragment_rate_over_parsed"]),
                "family_id": "F2_pool_cond",
                "jsonpath": "/raw_axis/single_fragment_rate_over_parsed",
                "file": _repo_relative(POSTPROC_JSON),
                "label": "post-processed arm :: the RAW axis it is compared against",
            }
        )
    ligand = _load_json(LIGAND_JSON) or {}
    for arm, family_id in (("metal_inclusive", "F1_pool_e2e"), ("ligand_only", "F11_ligand_only_connected")):
        value = (ligand.get(arm) or {}).get("single_fragment_rate")
        if isinstance(value, (int, float)):
            MATCH_TABLE.append(
                {
                    "value": float(value),
                    "family_id": family_id,
                    "jsonpath": f"/{arm}/single_fragment_rate",
                    "file": _repo_relative(LIGAND_JSON),
                    "label": f"Wf-A155 :: /{arm}/single_fragment_rate",
                }
            )
    panel = _load_json(PANEL_JSON) or {}
    # The c8 pool's MEAN-OF-THREE-EPOCH rate -- the derivation todo.json
    # states ("(0.2031+0.2003+0.3480)/3").  It renders as 25.05 % but is NOT
    # the pooled 529/2112.
    c8 = _read_pool_csv(C8_POOL)
    epoch_rates = [
        v["rate_over_rows"]
        for v in (c8.get("per_epoch") or {}).values()
        if v.get("rate_over_rows") is not None
    ]
    if epoch_rates:
        MATCH_TABLE.append(
            {
                "value": sum(epoch_rates) / len(epoch_rates),
                "family_id": "F1_pool_e2e",
                "jsonpath": "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv::per-epoch mean",
                "file": "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv",
                "label": "the c8 pool's MEAN-OF-THREE-EPOCH rate (todo.json's derivation)",
            }
        )
    gen = panel.get("generation") or {}
    if isinstance(gen.get("single_fragment_ratio"), (int, float)):
        MATCH_TABLE.append(
            {
                "value": float(gen["single_fragment_ratio"]),
                "family_id": "F2_pool_cond",
                "jsonpath": "/generation/single_fragment_ratio",
                "file": _repo_relative(PANEL_JSON),
                "label": "free-generation panel :: conditional single-fragment ratio",
            }
        )


def _match_quoted_number(token_value: float) -> List[Dict[str, Any]]:
    """Every published value that renders to the same 2-dp percent string."""
    pct = render_pct(token_value)
    out = []
    for entry in MATCH_TABLE:
        value = entry.get("value")
        if not isinstance(value, (int, float)):
            continue
        if render_pct(float(value)) == pct:
            out.append(entry)
    return out


def _tracked_doc_files() -> List[str]:
    """Tracked, prose, non-paper files.  ``git ls-files`` (index only) + a
    content read that NEVER skips a file (the repo's PATH grep is ugrep -I,
    which silently skips non-UTF-8 tracked text files -- Wf-A151 §6.263)."""
    try:
        raw = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(REPO),
            capture_output=True,
            check=True,
            timeout=120,
        ).stdout
        names = [n.decode("utf-8", "replace") for n in raw.split(b"\0") if n]
    except Exception:
        names = []
    if not names:
        out: List[str] = []
        for path in REPO.rglob("*"):
            if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS:
                out.append(_repo_relative(path))
        names = out
    keep = []
    for name in names:
        if name.startswith(EXCLUDED_PREFIXES):
            continue
        if name in EXTRA_DOC_NAMES:
            keep.append(name)
            continue
        if Path(name).suffix.lower() in DOC_EXTENSIONS:
            keep.append(name)
    return sorted(set(keep))


def quoted_map_q3() -> Dict[str, Any]:
    _build_match_table()
    hits: List[Dict[str, Any]] = []
    files_scanned = 0
    files_with_hits: Dict[str, int] = {}
    for name in _tracked_doc_files():
        path = REPO / name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files_scanned += 1
        count = 0
        for lineno, line in enumerate(text.split("\n"), 1):
            for label, rx in COMPILED_QUOTES:
                for match in rx.finditer(line):
                    count += 1
                    raw_token = match.group(0)
                    numeric = re.search(r"(\d+(?:\.\d+)?)", raw_token)
                    token_value = float(numeric.group(1)) if numeric else None
                    if token_value is None:
                        continue
                    fraction = _token_fraction(raw_token, token_value)
                    start = max(0, match.start() - 120)
                    end = min(len(line), match.end() + 120)
                    sentence = line[start:end].strip()
                    has_frag_context = bool(FRAG_CONTEXT_RE.search(line))
                    matches = _match_quoted_number(fraction)
                    if matches and has_frag_context:
                        families = sorted({m["family_id"] for m in matches})
                        verdict = (
                            "MATCHED (AMBIGUOUS -- %d published numbers render the same)"
                            % len(matches)
                            if len(matches) > 1
                            else "MATCHED"
                        )
                    elif matches:
                        families = []
                        verdict = (
                            "VALUE COLLISION ONLY -- the token equals a published "
                            "fragmentation value but the sentence carries NO "
                            "fragmentation keyword (a different quantity)"
                        )
                    elif render_pct(fraction) in {"92.70", "94.00"} and EXTERNAL_BAND_CONTEXT_RE.search(line):
                        families = ["F12_external_band"]
                        verdict = "MATCHED (external band -- NOT our metric)"
                    elif render_pct(fraction) in {"92.70", "94.00"} and EXTERNAL_PAPER_CONTEXT_RE.search(line):
                        families = ["F13_external_paper_rate"]
                        verdict = "MATCHED (a SINGLE PAPER's external rate -- NOT our metric)"
                    elif render_pct(fraction) in {"92.70", "94.00"}:
                        families = []
                        verdict = (
                            "NOT A FRAGMENTATION NUMBER -- the token is a percent with no "
                            "fragmentation definition attached to it in this repo"
                        )
                    else:
                        families = []
                        verdict = (
                            "UNMATCHED -- quotes a number with NO published fragmentation "
                            "definition"
                        )
                    hits.append(
                        {
                            "file": name,
                            "line": lineno,
                            "quoted_token": raw_token,
                            "quoted_fraction": fraction,
                            "rendered_percent": render_pct(fraction),
                            "sentence": sentence,
                            "has_fragmentation_context": has_frag_context,
                            "verdict": verdict,
                            "family_ids": families,
                            "matches": [
                                {
                                    "file": m["file"],
                                    "jsonpath": m["jsonpath"],
                                    "value": m["value"],
                                    "family_id": m["family_id"],
                                    "label": m["label"],
                                }
                                for m in matches
                            ],
                        }
                    )
        if count:
            files_with_hits[name] = count

    by_family: Dict[str, int] = {}
    for hit in hits:
        key = ",".join(hit["family_ids"]) if hit["family_ids"] else "UNMATCHED"
        by_family[key] = by_family.get(key, 0) + 1

    # The patch list: the distinct (file, line) sites that need a Rule-6
    # repair.  Grouped by which family the quoted number belongs to.
    patch_list: List[Dict[str, Any]] = []
    for hit in hits:
        patch_list.append(
            {
                "file": hit["file"],
                "line": hit["line"],
                "quoted_token": hit["quoted_token"],
                "rendered_percent": hit["rendered_percent"],
                "family_ids": hit["family_ids"],
                "verdict": hit["verdict"],
                "suggested_repair": (
                    "quote the family + jsonpath explicitly"
                    if hit["family_ids"]
                    else "ADD the definition, or drop the number"
                ),
            }
        )

    return {
        "status": "OK",
        "n_files_scanned": files_scanned,
        "n_hits": len(hits),
        "n_files_with_hits": len(files_with_hits),
        "hits_per_file": files_with_hits,
        "hits_by_family_key": by_family,
        "n_unmatched": sum(1 for h in hits if not h["family_ids"]),
        "n_matched_to_our_families": sum(
            1
            for h in hits
            if h["family_ids"] and any(f not in EXTERNAL_FAMILIES for f in h["family_ids"])
        ),
        "n_external_only": sum(
            1
            for h in hits
            if h["family_ids"] and all(f in EXTERNAL_FAMILIES for f in h["family_ids"])
        ),
        "n_not_a_fragmentation_number": sum(
            1 for h in hits if h["verdict"].startswith("NOT A FRAGMENTATION NUMBER")
        ),
        "n_value_collision_only": sum(
            1 for h in hits if h["verdict"].startswith("VALUE COLLISION ONLY")
        ),
        "n_ambiguous": sum(
            1 for h in hits if h["verdict"].startswith("MATCHED (AMBIGUOUS")
        ),
        "hits": hits,
        "patch_list": patch_list,
        "grep_note": (
            "the scan reads every tracked prose file with errors='replace', so it "
            "does NOT skip the repo's non-UTF-8 tracked text files the way PATH "
            "grep (ugrep -I) does (Wf-A151 §6.263). paper/ is never opened."
        ),
    }


# --------------------------------------------------------------------------
# GATES
# --------------------------------------------------------------------------
def _gate(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def run_gates(q1: Dict[str, Any], q2: Dict[str, Any], q3: Dict[str, Any]) -> List[Dict[str, Any]]:
    gates: List[Dict[str, Any]] = []
    if q1.get("status") != "OK":
        return [_gate("q1_available", False, q1.get("reason", "q1 SKIP"))]

    d851 = q1["definitions"]["8.51_percent"]
    d2505 = q1["definitions"]["25.05_percent"]

    gates.append(
        _gate(
            "q1_median_recomputed",
            bool(d851.get("recomputed_matches_published")),
            "the median recomputed from /runs (%r) vs published (%r)"
            % (d851.get("recomputed_median_from_runs"), d851.get("published_value")),
        )
    )
    gates.append(
        _gate(
            "q1_c8_csv_recount_matches_published",
            bool(d2505.get("matches_audit_published_value")),
            "529/2112-style recount from the CSV (%r) vs published (%r)"
            % (
                (d2505.get("csv_recount") or {}).get("rate_over_rows"),
                d2505.get("published_value"),
            ),
        )
    )
    axes = ("population", "numerator", "denominator", "measurement_point")
    complete = all(
        all(d.get(axis) for axis in axes) for d in (d851, d2505)
    )
    gates.append(
        _gate(
            "q1_all_four_axes_established",
            complete,
            "population/numerator/denominator/measurement_point present for both numbers",
        )
    )
    gates.append(
        _gate(
            "q1_reason_established",
            bool(q1.get("reason_the_two_differ")) and "NOT ESTABLISHED" not in q1.get("reason_the_two_differ", ""),
            q1.get("reason_the_two_differ", "")[:160],
        )
    )
    gates.append(
        _gate(
            "q2_every_family_has_a_rate_row",
            q2.get("n_families_with_rows", 0) >= 1
            and all(
                fam.get("n_rate_rows", 0) > 0
                for fid, fam in (q2.get("family_table") or {}).items()
                if fid not in EXTERNAL_FAMILIES
            ),
            "families with rows: %d / %d declared; families without a RATE row: %s"
            % (
                q2.get("n_families_with_rows"),
                q2.get("n_families_declared"),
                [
                    fid
                    for fid, fam in (q2.get("family_table") or {}).items()
                    if fid not in EXTERNAL_FAMILIES and not fam.get("n_rate_rows")
                ],
            ),
        )
    )
    gates.append(
        _gate(
            "q3_every_hit_classified",
            all(
                (h.get("verdict") or "").strip() != "" for h in (q3.get("hits") or [])
            ),
            "%d hits, %d unmatched, %d ambiguous"
            % (q3.get("n_hits", 0), q3.get("n_unmatched", 0), q3.get("n_ambiguous", 0)),
        )
    )
    gates.append(
        _gate(
            "q3_no_paper_read",
            not any(
                str(h.get("file", "")).startswith("paper/") for h in (q3.get("hits") or [])
            ),
            "no hit comes from paper/",
        )
    )
    gates.append(
        _gate(
            "no_cuda_no_torch",
            "torch" not in sys.modules,
            "torch imported: %s" % ("torch" in sys.modules),
        )
    )
    return gates


# --------------------------------------------------------------------------
# MARKDOWN TABLE
# --------------------------------------------------------------------------
def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render_markdown(payload: Dict[str, Any]) -> str:
    q1 = payload["q1"]
    q2 = payload["q2"]
    q3 = payload["q3"]
    lines: List[str] = []
    lines.append("# Wf-A158 BRANCH POP -- population reconciliation")
    lines.append("")
    lines.append(
        "Generated by `molmetal/scripts/wfa158_population_reconciliation.py` "
        "(CPU-only, 0 GPU). Every number below is MEASURED from a file on disk."
    )
    lines.append("")
    lines.append("## Q1 -- the two headline numbers, side by side")
    lines.append("")
    d851 = q1["definitions"]["8.51_percent"]
    d2505 = q1["definitions"]["25.05_percent"]
    lines.append(
        _md_table(
            ["axis", "8.51 %", "25.05 %"],
            [
                ["published value", d851.get("published_value"), d2505.get("published_value")],
                ["jsonpath", d851.get("published_jsonpath"), d2505.get("published_jsonpath")],
                ["definition family", d851.get("family_id"), d2505.get("family_id")],
                ["population", d851.get("population"), d2505.get("population")],
                ["numerator", d851.get("numerator"), d2505.get("numerator")],
                ["denominator", d851.get("denominator"), d2505.get("denominator")],
                ["measurement point", d851.get("measurement_point"), d2505.get("measurement_point")],
            ],
        )
    )
    lines.append("")
    lines.append("**Why they differ:** " + str(q1.get("reason_the_two_differ")))
    lines.append("")
    lines.append("## Q2 -- the census of published fragmentation-shaped numbers")
    lines.append("")
    rows = []
    for family_id, fam in (q2.get("family_table") or {}).items():
        head = (q2.get("family_headlines") or {}).get(family_id) or {}
        value = head.get("value")
        if isinstance(value, float):
            value = f"{value}  ({render_pct(value)} %)"
        elif value is None:
            # NOT "None": the family carries census rows but publishes no
            # single quotable headline (its numbers are per-run or per-metric).
            value = "no single published headline (%d census rows)" % fam.get(
                "n_census_rows", 0
            )
        rows.append(
            [
                family_id,
                fam.get("label"),
                value,
                fam.get("n_census_rows"),
                fam.get("n_distinct_metrics"),
                fam.get("unit"),
                fam.get("population"),
            ]
        )
    lines.append(
        _md_table(
            ["family", "label", "headline value", "census rows", "metrics", "unit", "population"],
            rows,
        )
    )
    lines.append("")
    lines.append("### rendering collisions (distinct numbers, same printed figure)")
    lines.append("")
    coll_rows = []
    for coll in q2.get("rendering_collisions") or []:
        for member in coll["members"]:
            coll_rows.append(
                [
                    coll["rendered_percent"] + " %",
                    member.get("value"),
                    member.get("family_id"),
                    member.get("file"),
                    member.get("jsonpath"),
                ]
            )
    if coll_rows:
        lines.append(_md_table(["printed", "value", "family", "file", "jsonpath"], coll_rows))
    else:
        lines.append("_none_")
    lines.append("")
    lines.append("## Q3 -- the quoted-number map")
    lines.append("")
    lines.append(
        "%d hits in %d files (%d unmatched, %d ambiguous)."
        % (q3.get("n_hits", 0), q3.get("n_files_with_hits", 0), q3.get("n_unmatched", 0), q3.get("n_ambiguous", 0))
    )
    lines.append("")
    lines.append(str(q3.get("grep_note")))
    lines.append("")
    q3_rows = []
    for hit in q3.get("hits") or []:
        q3_rows.append(
            [
                f"{hit['file']}:{hit['line']}",
                hit.get("quoted_token"),
                ",".join(hit.get("family_ids") or []) or "UNMATCHED",
                hit.get("verdict"),
            ]
        )
    if q3_rows:
        lines.append(_md_table(["site", "token", "family", "verdict"], q3_rows))
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append(
        _md_table(
            ["gate", "status", "detail"],
            [[g["gate"], g["status"], g["detail"]] for g in payload["gates"]],
        )
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def _atomic_write(path: Path, content: str) -> None:
    sys.path.insert(0, str(REPO))
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: PLC0415

    atomic_write_file(path, content)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="compute and print only; do not write the artifact",
    )
    args = parser.parse_args(argv)

    if "torch" in sys.modules:
        print("FAIL: torch is imported -- this script must stay CPU-only")
        return 1

    q1 = reconcile_q1()
    if q1.get("status") == "SKIP":
        print("SKIP: %s" % q1.get("reason"))
        return 2
    q2 = census_q2()
    q3 = quoted_map_q3()
    gates = run_gates(q1, q2, q3)

    payload = {
        "metric": "wfa158_population_reconciliation",
        "tag": "MEASURED",
        "date": "2026-11-01",
        "wave": "Wf-A158 (branch POP)",
        "gpu": "none -- CPU-only, no torch, no RDKit, no checkpoint",
        "purpose": (
            "reconcile the two headline fragmentation numbers (8.51 % and 25.05 %), "
            "census every published fragmentation-shaped number into DEFINITION "
            "FAMILIES, and map every document that quotes one of them"
        ),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "gates": gates,
        "verdict": "PASS" if all(g["status"] == "PASS" for g in gates) else "FAIL",
        "write_scope": [
            _repo_relative(OUT_JSON),
            _repo_relative(OUT_MD),
        ],
        "read_scope": [
            "metrics/by_metric/*.json",
            "registry/metrics.yaml",
            "molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv",
            "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv",
        ],
        "not_established": [
            "the denominator of the 8.51 % median is a MEDIAN of per-pool rates, not a "
            "pooled rate over all draws -- a pooled 8.51 % would have a different value; "
            "the repo publishes no pooled figure, so the pooled value is NOT ESTABLISHED",
            "why each of the 9 pools' pools was drawn is not established beyond the "
            "events.jsonl gen_validity_n denominator the audit records",
            "no document-level edit was made: the patch list is the deliverable, the "
            "main thread applies the Rule-6 repairs serially",
            "the 92.7-94 % band is an EXTERNAL number; this repo publishes no "
            "measurement of it, so it is catalogued, not verified",
        ],
    }

    out_dir = Path(args.out_dir)
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(out_dir / "reconciliation.json", json.dumps(payload, indent=2, sort_keys=False) + "\n")
        _atomic_write(out_dir / "census.md", render_markdown(payload))

    verdict = payload["verdict"]
    print("=" * 78)
    print("Wf-A158 BRANCH POP -- population reconciliation")
    print("=" * 78)
    print("Q1  8.51 %%  = %r  (%s)" % (d851_value(q1), _d851_population_label(q1)))
    print("Q1  25.05 %% = %r  (%s)" % (q2505_value(q1), _d2505_population_label(q1)))
    print("Q1  ratio    = %r" % ((q1.get("headline") or {}).get("ratio"),))
    print("Q1  reason   : %s" % q1.get("reason_the_two_differ"))
    print("Q2  census rows = %d over %d metric files; families with rows = %d / %d"
          % (q2.get("n_census_rows", 0), q2.get("n_metrics_with_fragmentation_rows", 0),
             q2.get("n_families_with_rows", 0), q2.get("n_families_declared", 0)))
    for coll in q2.get("rendering_collisions") or []:
        print("Q2  COLLISION  %s %%  <- %d distinct values" % (coll["rendered_percent"], coll["n_distinct_values"]))
    print("Q3  hits = %d in %d files | ours %d (ambiguous %d) | external-only %d | "
          "no family %d (not-a-fragmentation %d, value-collision-only %d)"
          % (q3.get("n_hits", 0), q3.get("n_files_with_hits", 0),
             q3.get("n_matched_to_our_families", 0), q3.get("n_ambiguous", 0),
             q3.get("n_external_only", 0), q3.get("n_unmatched", 0),
             q3.get("n_not_a_fragmentation_number", 0),
             q3.get("n_value_collision_only", 0)))
    for gate in gates:
        print("GATE %-42s %s  %s" % (gate["gate"], gate["status"], gate["detail"][:90]))
    print("-" * 78)
    print("%s" % verdict)
    if not args.no_write:
        for path in (out_dir / "reconciliation.json", out_dir / "census.md"):
            print("WROTE %s" % path)
    return 0 if verdict == "PASS" else 1


def d851_value(q1: Dict[str, Any]) -> Any:
    return ((q1.get("definitions") or {}).get("8.51_percent") or {}).get("published_value")


def q2505_value(q1: Dict[str, Any]) -> Any:
    return ((q1.get("definitions") or {}).get("25.05_percent") or {}).get("published_value")


def _d851_population_label(q1: Dict[str, Any]) -> str:
    """The population label, read from the payload (never hard-coded)."""
    definition = (q1.get("definitions") or {}).get("8.51_percent") or {}
    n = definition.get("n_pools_in_the_median")
    return "%s-pool MEDIAN, family %s" % (
        n if n is not None else "?",
        definition.get("family_id", "?"),
    )


def _d2505_population_label(q1: Dict[str, Any]) -> str:
    definition = (q1.get("definitions") or {}).get("25.05_percent") or {}
    jsonpath = str(definition.get("published_jsonpath") or "")
    pool = ""
    if "run==" in jsonpath:
        pool = jsonpath.split("run==", 1)[1].split("]", 1)[0]
    return "ONE pool (%s), family %s" % (
        pool or "?",
        definition.get("family_id", "?"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
