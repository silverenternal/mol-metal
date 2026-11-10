#!/usr/bin/env python3
"""STR-6 — SA yield audit: what fraction of generated molecules is synthesizable?

The plan asked for "3 seeds x 200, SA penalty on/off, gate SA <= 4.5 and mean
novelty >= 0.90".  Before spending GPU on that A/B, three facts rule it out:

1. ``--sa-penalty`` is inert.  ``sa_penalty.py`` says so in its own docstring:
   sascorer is a fragment-frequency lookup with no gradient, and the trainer's
   proxy is wrapped in ``torch.tensor(float(...))`` (``requires_grad=False``).
   Parameter updates with the flag on are bit-for-bit identical to off.
2. There is no differentiable SA surrogate to guide decoding with.  The one
   that existed was deleted on 2026-10-04 because it was fake.
3. Only one checkpoint exists per run (``..._seed0.pt``), so a 3-seed sweep
   would first require training two more seeds.

What remains measurable is the yield: generate an unguided pool, score SA per
molecule, and report the fraction that clears 4.5.  That is what this script
does, over the pool of record plus every large dump on disk.

The gate is only meaningful with a yield attached.  "SA <= 4.5" over a 0.7%
hand-picked subset is a property of the filter, not of the generator.

THE POOLS ARE NOT ALL THE SAME KIND
------------------------------------
``molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv`` carries a
``metal_z`` column, and so do the two full-corpus pools of 2026-10-11.  That
column ALONE does not say what wrote the pool: ``wf_cfm_decode_dump_smiles.py``
emitted the same header in two eras -- before 2026-10-12 it loaded REAL tmQM
geometries and ran only the bond head (a RECONSTRUCTION), from 2026-10-12 it
calls the trainer's own ODE sampler (FREE GENERATION).  The rule owner
(``molmetal/scripts/cfm_pool_kind.py``) therefore decides a ``metal_z`` pool
from its CONTENT, and the pool of record comes out as GENERATION: its 66 rows
all carry ``sanitized = 0`` and an empty ``metal_z``, which is the structural
signature of the ODE path.  The project's own panel artifact agrees
(``cfm_free_generation_panel_seed0.json``: 192 attempts, 66 valid,
gen_validity 0.344).

The two kinds are not comparable.  A reconstruction pool inherits the
properties of the real chemistry it started from, so one rate over pools of
both kinds is neither a generator number nor a decoder number.  This script
therefore:

  * classifies every pool with ``molmetal/scripts/cfm_pool_kind.py`` (the ONE
    owner of the rule) BEFORE scoring it;
  * pools only pools of the SAME kind -- ``combined_by_kind`` holds one block
    per kind, each with its own denominator;
  * keeps ``combined`` as the GENERATION-side rate (the artifact's subject),
    with ``pools_excluded`` naming the reconstruction pools that are NOT in
    it;
  * REFUSES an unclassified pool: it is not scored, it enters no rate, and
    the refusal (with the reason) is recorded in ``refused_pools``.  A pool
    whose provenance is not established is never silently bucketed.

The reconstruction pools are REPORTED, not refused: their SA numbers are
correct (as the decoder's fidelity on real chemistry) and refusing them would
delete a measurement.  What changes is that they stop being pooled with the
generator's numbers.

THE GENERATION POOLS DO NOT SHARE A COHORT EITHER
--------------------------------------------------
The three GENERATION pools were trained on cohorts of very different sizes:
the panel of record comes from the n_train=26715 full-corpus run, while the
two pcgrad pools come from n_train=256 runs.  A rate over all three is a
property of the MIXTURE, not of the generator: all 16 SA<=4.5 hits come from
the two n_train=256 pools, and the 66-molecule full-corpus pool contributes
ZERO of them.  Publishing 16/2396 = 0.0067 as "the generator's SA yield"
therefore says more about the small cohorts than about the generator.

So this script ALSO tiers its numbers by cohort scale, using the ONE owner of
that rule (``molmetal/scripts/cfm_cohort_scale.py``):

  * every pool row carries its ``tier`` (``full_corpus`` / ``mid`` / ``small``
    / ``unknown``), its ``n_train`` and where that number came from;
  * ``by_pool`` publishes each pool's OWN rate and interval -- the view a
    single-cohort table needs;
  * ``combined_by_tier`` publishes the generation rate per cohort tier, each
    with its own denominator;
  * ``combined`` KEEPS its historical numbers (so no consumer breaks) but is
    LABELLED ``scale_label="mixed_cohort_scale"``, ``comparable=false``, and
    its note says what the denominator mixes;
  * the VERDICT is taken over the pool of record's cohort tier, never over
    the mixed denominator.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/cfm_sa_yield_audit.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.cfm_cohort_scale import (  # noqa: E402
    COHORT_SCALE_SOURCE,
    FULL_CORPUS_N_TRAIN,
    SMALL_TRAIN_CUTOFF,
    TIER_RULE,
    TIERS,
    cohort_scale_fields,
    run_name_of_pool,
)
from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    GENERATION_MARKER,
    KIND_GENERATION,
    KIND_RECONSTRUCTION,
    KIND_UNKNOWN,
    MEASUREMENT_MEANING,
    POOL_KIND_RULE,
    RECONSTRUCTION_MARKER,
    VERDICT_SCOPE,
    PoolKind,
    classify_pool,
)
from molmetal.scripts.msrs_surface_model import _git_commit  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

SA_GATE = 4.5
NOVELTY_GATE = 0.90
THRESHOLDS = (4.0, 4.5, 5.0, 5.5, 6.0)

#: Where the pool-kind rule lives.  It has ONE owner; this script imports it
#: rather than re-deciding "did the sampler produce this pool?".
POOL_KIND_SOURCE = "molmetal/scripts/cfm_pool_kind.py"

#: The label of the pool the paper table's SA row is about: the free-generation
#: panel of record (the decode-dump's ODE path, 192 attempts -> 66 survivors).
#: It is also the REFERENCE for the comparability labelling: a pool's rate may
#: be read next to the panel of record's only if it is a generation pool
#: trained on the SAME cohort tier.
POOL_OF_RECORD = "panel_of_record"

#: The corpus the novelty half of the gate is measured against.
TRAIN_REFERENCE = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)

#: The pools the SA gate is about.  ``panel_of_record`` is the free-generation
#: panel of record (the decode-dump's ODE path, 192 attempts -> 66
#: survivors); the two pcgrad pools are the trainer's own free-generation
#: path.  All three are GENERATION pools, so all three are in the generator's
#: denominator.
DEFAULT_POOLS = (
    ("panel_of_record", "molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv"),
    ("pcgrad_on", "molmetal/reports/cfm_pcgrad_on_2026_10_14/generated_smiles.csv"),
    ("pcgrad_off", "molmetal/reports/cfm_pcgrad_off_2026_10_15/generated_smiles.csv"),
)

#: The two REAL reconstruction pools on disk (era A of the decode-dump: the
#: bond head applied to real tmQM geometries).  They are scored and REPORTED
#: -- their SA numbers are the decoder's fidelity on real chemistry -- but
#: they are in the RECONSTRUCTION block, never in the generator's rate.
RECONSTRUCTION_POOLS = (
    (
        "recon_fullcorpus_b64",
        "molmetal/reports/cfm_fullcorpus_b64_2026_10_11/generated_smiles.csv",
    ),
    (
        "recon_objective_terms",
        "molmetal/reports/cfm_objective_terms_2026_10_11/generated_smiles.csv",
    ),
)

#: Every pool the audit scores.  The kind is DECIDED by the rule owner, not
#: by the list a pool sits in: the two lists say which pools are EXPECTED to
#: be of which kind, and the classification decides where each one lands.
ALL_POOLS = DEFAULT_POOLS + RECONSTRUCTION_POOLS


def _pool_specs() -> List[tuple]:
    """``(label, path, expected kind)`` for every pool the audit scores.

    Read at CALL time so a caller (a test) can replace either list.  The
    EXPECTED kind is only a cross-check: the shared rule decides where the
    pool lands, and a disagreement is reported, never silently resolved.
    """
    return [
        (label, rel, KIND_GENERATION) for label, rel in DEFAULT_POOLS
    ] + [
        (label, rel, KIND_RECONSTRUCTION) for label, rel in RECONSTRUCTION_POOLS
    ]


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score interval; well behaved at k = 0, where the normal one is not."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


#: Morgan fingerprints of the training corpus, computed once per process.
#: The gate subset is scored once per pool KIND and every call needs the same
#: reference set, so recomputing it per kind would double the cost of the
#: audit for no new information.
_REFERENCE_CACHE: Dict[str, List[object]] = {}


def _reference_fingerprints(train_path: Path) -> List[object]:
    """The training corpus as Morgan fingerprints (cached per process)."""
    key = str(train_path)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached
    import pandas as pd

    from molmetal.scripts.cfm_generation_metrics import morgan

    train = pd.read_csv(train_path, low_memory=False)
    if "split" in train.columns:
        train = train[train["split"] == "train"]
    ref = [f for f in (morgan(s) for s in train["smiles"].dropna().astype(str)) if f is not None]
    _REFERENCE_CACHE[key] = ref
    return ref


def novelty_of(smiles: Sequence[str], train_path: Path) -> Optional[Dict[str, object]]:
    """Mean 1 - max Tanimoto to the training corpus, for a selected subset."""
    if not smiles or not train_path.exists():
        return None

    from molmetal.scripts.cfm_generation_metrics import (
        max_tanimoto_to_reference,
        morgan,
    )

    ref = _reference_fingerprints(train_path)
    fps = [f for f in (morgan(s) for s in smiles) if f is not None]
    if not fps or not ref:
        return None
    sims = max_tanimoto_to_reference(fps, ref)
    novelty = 1.0 - sims
    return {
        "n": int(len(novelty)),
        "mean_novelty": round(float(novelty.mean()), 4),
        "median_novelty": round(float(sorted(novelty)[len(novelty) // 2]), 4),
        "underpowered": bool(len(novelty) < 20),
        "train_reference": str(train_path.relative_to(PROJECT_ROOT)),
        "fingerprint": "Morgan radius 2, 1024 bits",
    }


def score_smiles(smiles: Sequence[str]) -> Dict[str, object]:
    """Per-molecule SA / QED.  Unparseable rows are counted, never dropped."""
    from rdkit import Chem, RDLogger

    from molmetal.adapters.flow_matching_lipman.sa_penalty import compute_sa_score
    from molmetal.adapters.flow_matching_lipman.qed_penalty import compute_qed_score

    RDLogger.DisableLog("rdApp.*")
    sas: List[float] = []
    qeds: List[float] = []
    gate_smiles: List[str] = []
    n_bad = 0
    for text in smiles:
        mol = Chem.MolFromSmiles(text) if isinstance(text, str) else None
        if mol is None:
            n_bad += 1
            continue
        sa = compute_sa_score(mol)
        if sa is None:
            n_bad += 1
            continue
        sas.append(float(sa))
        if float(sa) <= SA_GATE:
            gate_smiles.append(Chem.MolToSmiles(mol))
        qed = compute_qed_score(mol)
        if qed is not None:
            qeds.append(float(qed))
    return {"sas": sas, "qeds": qeds, "n_unscored": n_bad, "gate_smiles": gate_smiles}


def _rel(path: Path) -> str:
    """Repo-relative path, or the absolute path for a pool outside the repo.

    A pool under ``tmp_path`` (a test fixture, an archived run) has no
    repo-relative form; ``relative_to`` would raise and take the audit down.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pool_report(label: str, path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = score_smiles([r.get("smiles", "") for r in rows])
    sas = scored["sas"]
    if not sas:
        return {
            "label": label,
            "path": _rel(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": len(rows),
            "n_scored": 0,
        }
    ordered = sorted(sas)
    return {
        "label": label,
        "path": _rel(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "n_scored": len(sas),
        "n_unscored": int(scored["n_unscored"]),
        "sa_mean": round(sum(sas) / len(sas), 4),
        "sa_median": round(ordered[len(ordered) // 2], 4),
        "sa_min": round(ordered[0], 4),
        "sa_max": round(ordered[-1], 4),
        "sa_counts_at_or_below": {str(t): sum(1 for s in sas if s <= t) for t in THRESHOLDS},
        "qed_mean": round(sum(scored["qeds"]) / len(scored["qeds"]), 4) if scored["qeds"] else None,
        "gate_smiles": list(scored["gate_smiles"]),
    }


def pool_kind_fields(kind: PoolKind) -> Dict[str, object]:
    """The pool-kind labelling that travels NEXT TO the numbers.

    A reader who sees only a rate must be able to tell which pipeline wrote
    the molecules behind it, so the kind, the evidence for the kind, the
    meaning that kind carries and the source of the rule all travel with
    every pool row.
    """
    return {
        "pool_kind": kind.kind,
        "pool_kind_marker": kind.marker,
        "pool_kind_reason": kind.reason,
        "pool_kind_source": POOL_KIND_SOURCE,
        "measurement_meaning": MEASUREMENT_MEANING[kind.kind],
        "verdict_scope": VERDICT_SCOPE[kind.kind],
    }


def _rate_block(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The SA gate rate over ONE kind of pool.  Never over two kinds."""
    n_total = sum(int(p.get("n_scored") or 0) for p in rows)
    n_gate = sum(
        int((p.get("sa_counts_at_or_below") or {}).get(str(SA_GATE), 0)) for p in rows
    )
    rate = (n_gate / n_total) if n_total else None
    return {
        "n_pools": len(rows),
        "pools": [p.get("label") for p in rows],
        "n_scored": n_total,
        "n_sa_le_gate": n_gate,
        "rate_sa_le_gate": round(rate, 4) if rate is not None else None,
        "wilson_ci95": wilson(n_gate, n_total),
    }


def combined_by_kind(pools: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """The gate rate, SEPARATED by pool kind.

    A reconstruction pool inherits the real chemistry it started from, so a
    rate that pools it with a generation pool is neither a generator number
    nor a decoder number.  Each kind gets its own block and its own
    denominator.  An UNKNOWN pool is refused upstream and therefore appears
    in no block at all.
    """
    for pool in pools:
        kind = pool.get("pool_kind")
        if kind not in (KIND_GENERATION, KIND_RECONSTRUCTION):
            # A pool that does not declare a known kind must never vanish
            # into a denominator: it has to be REFUSED, with its reason,
            # before it reaches the aggregation.  Failing loudly here is
            # cheaper than publishing a rate over a silently reduced set.
            raise ValueError(
                "pool "
                f"{pool.get('label')!r} reached the rate without a known pool "
                f"kind (kind={kind!r}); it must be refused upstream"
            )
    out: Dict[str, Dict[str, object]] = {}
    for kind in (KIND_GENERATION, KIND_RECONSTRUCTION):
        rows = [p for p in pools if p.get("pool_kind") == kind]
        block = _rate_block(rows)
        block["measurement_meaning"] = MEASUREMENT_MEANING[kind]
        block["verdict_scope"] = VERDICT_SCOPE[kind]
        out[kind] = block
    return out


def combined_by_tier(
    pools: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """The gate rate, SEPARATED by cohort tier (GENERATION pools only).

    The pool kind and the cohort scale are two different axes and BOTH of
    them change what a rate means.  A rate over two training cohorts of
    different sizes is a property of the MIXTURE, not of the generator: the
    n_train=26715 panel of record contributes 0 SA<=4.5 hits in 66 molecules
    while the two n_train=256 pools contribute 16 hits in 2330 molecules, so
    the pooled rate is a small-cohort number wearing a full-corpus label.

    Each tier therefore gets its own block with its own denominator.  A
    RECONSTRUCTION pool is never in this map (it is the decoder's fidelity
    on real chemistry, and it keeps its own ``combined_by_kind`` block); the
    tiers here are the GENERATION cohorts only.
    """
    generation = [p for p in pools if p.get("pool_kind") == KIND_GENERATION]
    out: Dict[str, Dict[str, object]] = {}
    for tier in TIERS:
        rows = [p for p in generation if p.get("tier") == tier]
        if not rows:
            # A tier with no pool is not a rate; publishing an empty block
            # would only invite a reader to divide by its zero denominator.
            continue
        block = _rate_block(rows)
        n_trains = sorted(
            {int(p["n_train"]) for p in rows if p.get("n_train") is not None}
        )
        block["tier"] = tier
        block["n_train"] = n_trains[0] if len(n_trains) == 1 else None
        block["n_trains"] = n_trains
        block["cohort_scale_source"] = COHORT_SCALE_SOURCE
        block["cohort_scale_rule"] = TIER_RULE
        out[tier] = block
    return out


def by_pool_block(pools: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """The per-POOL view: one block per pool, with its own denominator.

    This is the view a single-cohort table needs: the rate of ONE pool,
    next to the training cohort that produced it.  The pool kind travels
    with every block, so a reconstruction pool's number can never be read as
    a generator number.
    """
    out: Dict[str, Dict[str, object]] = {}
    for pool in pools:
        out[str(pool.get("label"))] = {
            "n_scored": pool.get("n_scored"),
            "n_sa_le_gate": pool.get("n_sa_le_gate"),
            "rate_sa_le_gate": pool.get("rate_sa_le_gate"),
            "wilson_ci95": pool.get("wilson_ci95"),
            "tier": pool.get("tier"),
            "n_train": pool.get("n_train"),
            "n_train_source": pool.get("n_train_source"),
            "pool_path": pool.get("path"),
            "pool_kind": pool.get("pool_kind"),
            "comparable": pool.get("comparable"),
            "comparable_reference": pool.get("comparable_reference"),
        }
    return out


def _reference_pool_label(pools: Sequence[Dict[str, object]]) -> Optional[str]:
    """The pool every comparability statement is made against.

    The pool of record when it is present, else the first GENERATION pool
    (so the labelling is defined even for a reduced pool set).
    """
    for pool in pools:
        if pool.get("label") == POOL_OF_RECORD:
            return POOL_OF_RECORD
    for pool in pools:
        if pool.get("pool_kind") == KIND_GENERATION:
            return str(pool.get("label"))
    return None


def apply_comparability(pools: Sequence[Dict[str, object]]) -> None:
    """Label every pool row with whether it may be read next to the panel.

    Two pools' SA rates may be compared only when they come from the same
    PIPELINE (both generation -- a reconstruction pool's rate is the
    decoder's fidelity on real chemistry) AND the same COHORT SCALE (the
    same tier).  The reason string names which of the two failed.
    """
    reference = _reference_pool_label(pools)
    ref = next((p for p in pools if p.get("label") == reference), None)
    ref_tier = ref.get("tier") if ref else None
    for pool in pools:
        kind = pool.get("pool_kind")
        tier = pool.get("tier")
        same_kind = kind == KIND_GENERATION
        same_tier = tier == ref_tier
        pool["comparable"] = bool(same_kind and same_tier)
        pool["comparable_reference"] = reference
        if not same_kind:
            pool["comparable_reason"] = (
                f"pool kind is {kind!r}, not 'generation': a reconstruction "
                f"pool's rate is the decoder's fidelity on real geometries "
                f"and is never comparable to a generation rate"
            )
        elif not same_tier:
            pool["comparable_reason"] = (
                f"cohort tier {tier!r} differs from the pool of record's "
                f"{ref_tier!r}: the two rates come from different training "
                f"cohorts and must not be pooled"
            )
        else:
            pool["comparable_reason"] = (
                f"same pool kind (generation) and the same cohort tier "
                f"({tier!r}) as the pool of record: the rates may be read "
                f"together"
            )


def _mixed_scale_note(tier_blocks: Dict[str, Dict[str, object]]) -> str:
    """One sentence naming WHAT the mixed denominator mixes.

    The sentence is built from the numbers it describes, so it cannot drift
    away from the block it annotates.
    """
    parts = []
    for tier, block in tier_blocks.items():
        n_trains = block.get("n_trains") or []
        shown = ", ".join(str(n) for n in n_trains) if n_trains else "unrecorded"
        n_pools = int(block.get("n_pools") or 0)
        pools_str = "1 pool" if n_pools == 1 else f"{n_pools} pools"
        parts.append(
            f"n_train={shown} ({tier}, {pools_str}): "
            f"{block.get('n_sa_le_gate')} SA<=4.5 hits in "
            f"{block.get('n_scored')} molecules"
        )
    return (
        "COHORT SCALE: the denominator MIXES cohort scales -- it is a "
        "mixed-cohort rate (scale_label=mixed_cohort_scale): "
        + "; ".join(parts)
        + " -- so this rate is NOT the full-corpus generator's rate; the "
        "full-corpus-only rate is in combined_by_tier.full_corpus "
        "(combined_by_tier holds the GENERATION pools only; rule: " + TIER_RULE + ")"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "cfm_sa_guided.json",
    )
    args = parser.parse_args(argv)

    pools: List[Dict[str, object]] = []
    refused: List[Dict[str, object]] = []
    for label, rel, expected_kind in _pool_specs():
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        kind = classify_pool(path)
        if kind.kind == KIND_UNKNOWN:
            # REFUSE, never bucket.  An unclassified pool has no established
            # pipeline, so its rows cannot enter a rate of EITHER kind: they
            # would be silently attributed to the generator.  The refusal and
            # its reason go to stderr AND into the payload.
            print(
                f"[str6] REFUSING to score an unclassified pool: {rel} "
                f"(reason: {kind.reason})",
                file=sys.stderr,
            )
            refused.append(
                {
                    "label": label,
                    "path": rel,
                    "reason": kind.reason,
                    "header": list(kind.header),
                    "refusal": "pool_kind_unknown",
                }
            )
            continue
        if kind.kind != expected_kind:
            # The shared rule wins; the expectation is only a cross-check, so
            # a disagreement is reported rather than resolved in either
            # direction.
            print(
                f"[str6] NOTE: {label} was expected to be a {expected_kind} "
                f"pool but the shared rule says {kind.kind} "
                f"(reason: {kind.reason}); the rule decides.",
                file=sys.stderr,
            )
        report = pool_report(label, path)
        if report is None:
            continue
        report.update(pool_kind_fields(kind))
        report["expected_pool_kind"] = expected_kind
        # The cohort scale of the pool: which TRAINING cohort produced these
        # molecules.  It is read from the run census through the one owner
        # of the rule, and it travels with the pool's own numbers.
        report.update(cohort_scale_fields(run_name_of_pool(path)))
        n_scored = int(report.get("n_scored") or 0)
        n_gate = int((report.get("sa_counts_at_or_below") or {}).get(str(SA_GATE), 0))
        report["n_sa_le_gate"] = n_gate
        report["rate_sa_le_gate"] = round(n_gate / n_scored, 4) if n_scored else None
        report["wilson_ci95"] = wilson(n_gate, n_scored)
        pools.append(report)
        print(
            f"[str6] {label:16s} kind={kind.kind:14s} "
            f"tier={report.get('tier'):11s} n_train={report.get('n_train')} "
            f"n={report['n_scored']:5d} "
            f"sa_mean={report.get('sa_mean')} "
            f"<=4.5={report.get('sa_counts_at_or_below', {}).get('4.5')}"
        )

    # The rate is computed per KIND.  A reconstruction pool is scored and
    # reported in its own block; it is never pooled with a generation pool.
    by_kind = combined_by_kind(pools)
    generation = by_kind.get(KIND_GENERATION, {})
    reconstruction = by_kind.get(KIND_RECONSTRUCTION, {})
    n_total = int(generation.get("n_scored") or 0)
    n_gate = int(generation.get("n_sa_le_gate") or 0)
    rate = (n_gate / n_total) if n_total else None
    ci = wilson(n_gate, n_total)

    # The rate is ALSO computed per COHORT TIER, because two generation
    # pools can be trained on cohorts of different sizes, and a rate over
    # both is a property of the mixture rather than of the generator.
    by_tier = combined_by_tier(pools)
    apply_comparability(pools)
    pool_view = by_pool_block(pools)

    gate_smiles_by_kind = {
        kind: [
            s
            for p in pools
            if p.get("pool_kind") == kind
            for s in p.get("gate_smiles", [])
        ]
        for kind in (KIND_GENERATION, KIND_RECONSTRUCTION)
    }
    novelty_by_kind = {
        kind: novelty_of(smiles, TRAIN_REFERENCE)
        for kind, smiles in gate_smiles_by_kind.items()
    }
    gen_gate_smiles = gate_smiles_by_kind[KIND_GENERATION]
    gen_novelty = novelty_by_kind[KIND_GENERATION]

    # The VERDICT is taken over ONE cohort, never over the mixed denominator:
    # the pool of record's tier.  The mixed rate stays in the artifact as the
    # historical number it is, labelled as mixed.
    reference = _reference_pool_label(pools)
    reference_tier = next(
        (str(p.get("tier")) for p in pools if p.get("label") == reference),
        "",
    )
    ref_block = by_tier.get(reference_tier) or {}
    verdict_n = int(ref_block.get("n_scored") or 0)
    verdict_rate = ref_block.get("rate_sa_le_gate")
    if verdict_n:
        verdict = "KILL" if float(verdict_rate) < 0.05 else "PARTIAL"
        verdict_basis = (
            f"the pool of record's cohort tier ({ref_block.get('tier')}, "
            f"{reference}, n={verdict_n}) -- a single training cohort"
        )
    elif n_total:
        # The pool of record's cohort could not be established (its tier has
        # no molecules); fall back to the mixed rate but say that the verdict
        # is over a MIXED denominator.
        verdict = "KILL" if rate < 0.05 else "PARTIAL"
        verdict_basis = (
            "the MIXED-cohort combined rate: the pool of record's cohort tier "
            "has no molecules, so no single-cohort rate is available"
        )
        verdict_rate = rate
        verdict_n = n_total
    else:
        # No generation-side pool survived (every pool was refused or absent):
        # there is no generator yield to report, so the artifact says so
        # instead of publishing a rate of 0.
        verdict = "UNMEASURED"
        verdict_basis = "no generation-side pool survived (all refused or absent)"
        verdict_rate = None
        verdict_n = 0

    payload = {
        "artifact": "cfm_sa_guided",
        # 2.1 adds the cohort-scale tiering: by_pool, combined_by_tier and
        # the scale_label / comparable labelling of `combined`.  The numbers
        # in `combined` are UNCHANGED from 2.0.
        "version": "2.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/cfm_sa_yield_audit.py",
            "git_commit": _git_commit(),
            "pool_kind_source": POOL_KIND_SOURCE,
        },
        "design": {
            "sa_gate": SA_GATE,
            "novelty_gate": NOVELTY_GATE,
            "thresholds": list(THRESHOLDS),
            "guided_decoding": "UNAVAILABLE",
            "blockers": [
                "--sa-penalty is inert: sascorer has no gradient and the trainer proxy is wrapped in torch.tensor(float(...)) with requires_grad=False",
                "no differentiable SA surrogate exists; the previous one was deleted on 2026-10-04 as fake",
                "one checkpoint per run (..._seed0.pt), so a 3-seed sweep needs two more training runs",
            ],
            "pool_of_record": "cfm_free_generation_panel_seed0.json (66 molecules, 192 attempts)",
            "pools_scored": [
                {"label": label, "path": rel, "expected_kind": "generation"}
                for label, rel in DEFAULT_POOLS
            ]
            + [
                {"label": label, "path": rel, "expected_kind": "reconstruction"}
                for label, rel in RECONSTRUCTION_POOLS
            ],
            "pooling_rule": (
                "a reconstruction pool is NEVER pooled with a generation pool: "
                "the gate rate is computed per kind, each with its own denominator"
            ),
            "pool_kind_source": POOL_KIND_SOURCE,
        },
        "pool_kind_rule": {
            "source": POOL_KIND_SOURCE,
            "rule": POOL_KIND_RULE,
            "markers": {
                "generation": GENERATION_MARKER,
                "reconstruction": RECONSTRUCTION_MARKER,
            },
            "measurement_meaning": MEASUREMENT_MEANING,
            "verdict_scope": VERDICT_SCOPE,
        },
        "cohort_scale_rule": {
            "source": COHORT_SCALE_SOURCE,
            "rule": TIER_RULE,
            "tiers": list(TIERS),
            "full_corpus_n_train": FULL_CORPUS_N_TRAIN,
            "small_train_cutoff": SMALL_TRAIN_CUTOFF,
            "n_train_source": "metrics/by_metric/cfm_run_census.json:runs[run=*].n_train",
        },
        "pools": [{k: v for k, v in p.items() if k != "gate_smiles"} for p in pools],
        "refused_pools": refused,
        "n_refused_pools": len(refused),
        "combined_by_kind": by_kind,
        "by_pool": pool_view,
        "combined_by_tier": by_tier,
        "combined": {
            "scope": "generation_only",
            "n_scored": n_total,
            "n_sa_le_gate": n_gate,
            "rate_sa_le_gate": round(rate, 4) if rate is not None else None,
            "wilson_ci95": ci,
            "pools_included": list(generation.get("pools") or []),
            "pools_excluded": list(reconstruction.get("pools") or []),
            # The numbers above are UNCHANGED (they are the artifact's
            # historical mixed-cohort rate); what changes is that the reader
            # is now told the denominator mixes cohort scales.
            "scale_label": "mixed_cohort_scale",
            "comparable": False,
            "cohort_scales_included": {
                str(p.get("label")): p.get("n_train")
                for p in pools
                if p.get("pool_kind") == KIND_GENERATION
            },
            "tiers_included": list(by_tier.keys()),
            "cohort_scale_source": COHORT_SCALE_SOURCE,
            "cohort_scale_rule": TIER_RULE,
            "note": (
                "the rate is over the GENERATION pools only.  A reconstruction "
                "pool inherits the real chemistry it started from, so pooling "
                "it with a generation pool would produce a rate that is "
                "neither a generator number nor a decoder number; its own "
                "numbers live in combined_by_kind.reconstruction "
                "(scope: decoder_fidelity_on_real_geometries).  Pool kind: "
                + POOL_KIND_SOURCE
                + ".  "
                + _mixed_scale_note(by_tier)
            ),
        },
        "gate_subset_by_kind": {
            kind: {
                "n": len(gate_smiles_by_kind[kind]),
                "novelty": novelty_by_kind[kind],
                "measurement_meaning": MEASUREMENT_MEANING[kind],
                "verdict_scope": VERDICT_SCOPE[kind],
            }
            for kind in (KIND_GENERATION, KIND_RECONSTRUCTION)
        },
        "gate_subset": {
            "scope": "generation_only",
            "n": len(gen_gate_smiles),
            "novelty": gen_novelty,
            "note": "the gate asks for SA <= 4.5 AND mean novelty >= 0.90; both can hold for a "
            "0.7% hand-picked subset while the generator itself is unchanged",
        },
        "verdict": verdict,
        "verdict_detail": {
            "gate_met": bool(verdict_rate is not None and float(verdict_rate) >= 0.05),
            "note": "the gate is a property of the generator only if the yield is reported next to it",
            "scope": "generation_only",
            "verdict_basis": verdict_basis,
            "cohort_tier": ref_block.get("tier"),
            "n_scored": verdict_n,
            "rate_sa_le_gate": verdict_rate,
            "mixed_scale_rate_sa_le_gate": round(rate, 4) if rate is not None else None,
            "mixed_scale_note": (
                "the combined rate is NOT used for the verdict: its denominator "
                "mixes cohort scales (see combined.scale_label and "
                "combined_by_tier)"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    for tier, block in by_tier.items():
        print(
            f"[str6] GENERATION tier={tier:11s} n_train={block.get('n_train')} "
            f"n={block.get('n_scored')} <=4.5={block.get('n_sa_le_gate')} "
            f"rate={block.get('rate_sa_le_gate')} ci={block.get('wilson_ci95')}"
        )
    print(
        f"[str6] GENERATION MIXED-SCALE combined n={n_total} <=4.5={n_gate} "
        f"rate={rate if rate is None else round(rate, 4)} ci={ci} -- NOT the "
        f"full-corpus generator's rate: the denominator mixes cohort scales "
        f"(see combined_by_tier)"
    )
    print(
        f"[str6] VERDICT {verdict} (basis: {verdict_basis}; "
        f"rate={verdict_rate} n={verdict_n})"
    )
    if reconstruction.get("n_pools"):
        print(
            f"[str6] RECONSTRUCTION (decoder fidelity on real geometries, NOT a "
            f"generator rate) n={reconstruction['n_scored']} "
            f"<=4.5={reconstruction['n_sa_le_gate']} "
            f"rate={reconstruction['rate_sa_le_gate']}"
        )
    for row in refused:
        print(f"[str6] REFUSED {row['label']}: {row['reason']}", file=sys.stderr)
    print(f"[str6] wrote {args.out}")
    # A refusal is loud: exit 2 (the repo's SKIP code) so a launcher sees it.
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
