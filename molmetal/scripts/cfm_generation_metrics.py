#!/usr/bin/env python3
"""MET-9 — generation-side metrics: novelty, diversity, SA/QED, metal compliance.

The generation line already has validity (0.344 yield on 192 attempts),
chemistry validity (1.000 of survivors), SA 6.4423 and QED 0.0944 in
``cfm_free_generation_panel_seed0.json``.  What is missing is:

* **novelty** — 1 - max Tanimoto to the training corpus;
* **uniqueness** — distinct canonical structures;
* **IntDiv1** recomputed on this cohort (the canonical m7 value is at n=37);
* **metal compliance** — does each generated molecule carry exactly one metal;
* **docking** — explicitly reported as blocked, with the evidence.

THE SUBJECT POOL IS THE GENERATOR'S OUTPUT
-----------------------------------------
The default pool, ``molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv``,
is the FREE-GENERATION panel of record: 66 survivors of the 192 attempts the
decode-dump script's ODE path drew.

The header alone does not say so.  The pool carries a ``metal_z`` column, but
``wf_cfm_decode_dump_smiles.py`` wrote ONE header in TWO eras: before
2026-10-12 it loaded REAL tmQM geometries and ran only the bond head (a
RECONSTRUCTION); from 2026-10-12 it calls the trainer's own ODE sampler.  The
rule owner (``molmetal/scripts/cfm_pool_kind.py``) therefore decides a
``metal_z`` pool from its CONTENT, and this pool's rows all carry
``sanitized = 0`` with an empty ``metal_z`` — the structural signature of the
ODE path.  The project's own panel artifact says the same
(``cfm_free_generation_panel_seed0.json``: 192 attempted, 66 valid,
gen_validity 0.344).

A reconstruction pool would be a different measurement: it inherits the
properties of the real chemistry it started from, so its novelty, diversity
and metal-compliance numbers would be the DECODER's fidelity on real
chemistry, never the generator's quality.  So this script:

* classifies the subject pool with ``molmetal/scripts/cfm_pool_kind.py`` (the
  ONE owner of the rule) before it is scored;
* carries the kind, the evidence for it and the meaning that kind has
  (MEASUREMENT_MEANING / VERDICT_SCOPE) NEXT TO every number it publishes;
* publishes the generation-side verdict when the pool IS a generation pool —
  the verdict is re-derived from the checks, not carried over;
* refuses the generation-side verdict when the pool IS a reconstruction — the
  artifact says the generation-side metrics are NOT published and why, and the
  verdict becomes ``MEASURED_NOT_A_CLAIM`` rather than a PASS/NEGATIVE about
  the generator;
* REFUSES an unclassified pool outright: it is not scored, no number is
  published from it, and the artifact records the refusal and its reason.

The measured numbers themselves are NOT changed by any of this.  They are
correct as the pool's numbers; what changes is the claim they are filed under.

Inputs (all on disk, no generation re-run):

``molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv``  66 rows (GENERATION pool)
``.data/molmetal/metalcytotox_5metal_multitask_v4.csv``             split == train
``metrics/by_metric/cfm_free_generation_panel_seed0.json``          SA / QED
``metrics/by_metric/m7_diversity.json``                             canonical IntDiv1
``metrics/by_metric/cfm_vina_generated.json``                       docking status

Output: ``metrics/by_metric/cfm_generation_metrics.json``.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/cfm_generation_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.cfm_pool_kind import (  # noqa: E402
    GENERATION_MARKER,
    KIND_GENERATION,
    KIND_UNKNOWN,
    MEASUREMENT_MEANING,
    POOL_KIND_RULE,
    RECONSTRUCTION_MARKER,
    VERDICT_SCOPE,
    PoolKind,
    classify_pool,
)
from molmetal.scripts.msrs_surface_model import _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METALS = ("Ru", "Ir", "Rh", "Os", "Re", "Pt")
NOVELTY_THRESHOLD = 0.9
INTDIV_TARGET = 0.85
SA_TARGET = 4.5
N_MATCHED_SUBSAMPLES = 40

#: Where the pool-kind rule lives (it has ONE owner).
POOL_KIND_SOURCE = "molmetal/scripts/cfm_pool_kind.py"


def canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else Chem.MolToSmiles(mol)


def morgan(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def max_tanimoto_to_reference(fps: Sequence[object], ref: Sequence[object]) -> np.ndarray:
    from rdkit.Chem import DataStructs

    return np.array(
        [max(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in fps], dtype=float
    )


def intdiv1(fps: Sequence[object]) -> float:
    """MOSES IntDiv1 = 1 - mean pairwise Tanimoto, self-pairs included."""
    from rdkit.Chem import DataStructs

    total = 0.0
    n = 0
    for f in fps:
        sims = DataStructs.BulkTanimotoSimilarity(f, list(fps))
        total += float(np.sum(sims))
        n += len(sims)
    return 1.0 - total / max(n, 1)


def metal_profile(smiles_list: Sequence[str]) -> Dict[str, object]:
    from rdkit import Chem

    counts: List[int] = []
    symbols: List[str] = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            counts.append(-1)
            continue
        found = [a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in METALS]
        counts.append(len(found))
        symbols.extend(found)
    counts_arr = np.asarray(counts)
    return {
        "n_parsed": int((counts_arr >= 0).sum()),
        "n_exactly_one_metal": int((counts_arr == 1).sum()),
        "compliance": round(float((counts_arr == 1).mean()), 4),
        "counts_histogram": {str(k): int(v) for k, v in pd.Series(counts).value_counts().sort_index().items()},
        "metal_distribution": {str(k): int(v) for k, v in pd.Series(symbols).value_counts().items()},
    }


def pool_kind_block(kind: PoolKind) -> Dict[str, object]:
    """The pool-kind record that travels NEXT TO the numbers.

    The kind, the evidence for it (the header it was read from), the meaning
    that kind carries and the source of the rule all sit in one block, so a
    reader cannot take a decoder-fidelity number for a generator number.
    """
    return {
        "kind": kind.kind,
        "marker": kind.marker,
        "reason": kind.reason,
        "header": list(kind.header),
        "source": POOL_KIND_SOURCE,
        "rule": POOL_KIND_RULE,
        "markers": {
            "generation": GENERATION_MARKER,
            "reconstruction": RECONSTRUCTION_MARKER,
        },
        "measurement_meaning": MEASUREMENT_MEANING[kind.kind],
        "verdict_scope": VERDICT_SCOPE[kind.kind],
        "path": str(kind.path) if kind.path is not None else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "cfm_fullcorpus_2026_10_10" / "generated_smiles.csv",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "cfm_generation_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Classify the subject pool BEFORE it is scored.  The kind decides what
    # the numbers mean and whether they may be filed as generation-side
    # metrics at all.
    kind = classify_pool(args.generated)
    if kind.kind == KIND_UNKNOWN:
        # REFUSE, never bucket.  An unclassified pool has no established
        # pipeline: scoring it and filing the result as a generation metric
        # would attribute a decoder's output to the generator.  No number is
        # published; the artifact records the refusal and its reason.
        payload = {
            "artifact": "cfm_generation_metrics",
            "version": "2.0",
            "status": "UNMEASURED",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": {
                "generated_csv": str(args.generated),
                "train_csv": str(args.train),
                "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "pool_kind_source": POOL_KIND_SOURCE,
            },
            "pool_kind": pool_kind_block(kind),
            "refusal": {
                "refusal": "pool_kind_unknown",
                "reason": kind.reason,
                "header": list(kind.header),
                "path": str(args.generated),
                "note": (
                    "the shared rule could not establish which code path wrote "
                    "the subject pool ("
                    + kind.reason
                    + ").  It was not scored and no metric is published from "
                    "it.  Pool kind: "
                    + POOL_KIND_SOURCE
                ),
            },
            "generation_side": {
                "published": False,
                "reason": "the subject pool is unclassified, so no pool kind may be claimed",
            },
            "verdict": "UNMEASURED",
            "verdict_detail": {
                "note": (
                    "no metric is published: the subject pool's provenance is "
                    "not established"
                ),
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
        print(
            f"[met9] REFUSING to score an unclassified pool: {args.generated} "
            f"(reason: {kind.reason})",
            file=sys.stderr,
        )
        print(f"[met9] wrote {args.out}")
        return 2

    is_generation = kind.kind == KIND_GENERATION
    scope = VERDICT_SCOPE[kind.kind]
    meaning = MEASUREMENT_MEANING[kind.kind]

    gen = pd.read_csv(args.generated)
    raw = [str(s) for s in gen["smiles"]]
    parsed = [canonical(s) for s in raw]
    parsed = [s for s in parsed if s is not None]

    train = pd.read_csv(args.train)
    train = train[train["split"] == "train"]
    train_canon = {canonical(s) for s in train["smiles"]}
    train_canon.discard(None)

    gen_fps = [morgan(s) for s in parsed]
    gen_fps = [f for f in gen_fps if f is not None]
    train_fps = [morgan(s) for s in sorted(train_canon)]
    train_fps = [f for f in train_fps if f is not None]

    sims = max_tanimoto_to_reference(gen_fps, train_fps)
    novelty = 1.0 - sims

    rng = np.random.default_rng(args.seed)
    matched = []
    n_sub = min(len(gen_fps), len(train_fps))
    for _ in range(N_MATCHED_SUBSAMPLES):
        idx = rng.choice(len(train_fps), size=n_sub, replace=False)
        matched.append(intdiv1([train_fps[i] for i in idx]))
    matched = np.asarray(matched)

    gen_intdiv = intdiv1(gen_fps)

    panel = json.loads(
        (PROJECT_ROOT / "metrics" / "by_metric" / "cfm_free_generation_panel_seed0.json").read_text()
    )
    m7 = json.loads(
        (PROJECT_ROOT / "metrics" / "by_metric" / "m7_diversity.json").read_text()
    )
    vina = json.loads(
        (PROJECT_ROOT / "metrics" / "by_metric" / "cfm_vina_generated.json").read_text()
    )

    sa = float(panel["panel"]["sa_mean"])
    qed = float(panel["panel"]["qed_mean"])
    frac_novel = float(np.mean(sims < NOVELTY_THRESHOLD))

    docking = {
        "status": "BLOCKED",
        "reason": vina.get("RUN_STATUS", {}).get("state", "UNKNOWN"),
        "pockets_planned": vina.get("RUN_STATUS", {}).get("planned_pockets"),
        "pockets_completed": vina.get("RUN_STATUS", {}).get("pockets_completed"),
        "source": "cfm_vina_generated.json:RUN_STATUS",
        "note": (
            "1 of 10 pockets docked before the run died; no success rate can "
            "be computed.  The precious-metal force-field gap is separate and "
            "still open (vina_status.json)."
        ),
    }

    checks = {
        "novelty_fraction_at_0.9": {
            "value": round(frac_novel, 4),
            "target": 0.90,
            "passed": bool(frac_novel >= 0.90),
            "definition": f"fraction of generated molecules with max Tanimoto < {NOVELTY_THRESHOLD}",
        },
        "mean_novelty": {
            "value": round(float(novelty.mean()), 4),
            "target": 0.90,
            "passed": bool(float(novelty.mean()) >= 0.90),
            "definition": "1 - mean(max Tanimoto to train); stricter than the fraction",
        },
        "intdiv1": {
            "value": round(gen_intdiv, 4),
            "target": INTDIV_TARGET,
            "passed": bool(gen_intdiv >= INTDIV_TARGET),
            "n_matched_training": round(float(matched.mean()), 4),
            "n_matched_sd": round(float(matched.std(ddof=1)), 4),
            "delta_vs_n_matched": round(gen_intdiv - float(matched.mean()), 4),
            "canonical_m7_value": m7.get("aggregate", {}).get("best_measured"),
            "canonical_m7_n": m7.get("aggregate", {}).get("best_measured_n"),
            "note": (
                "IntDiv1 is strongly n-dependent; compare with the n-matched "
                "training subsample, not with the MOSES 0.857 reference."
            ),
        },
        "sa": {
            "value": round(sa, 4),
            "target": SA_TARGET,
            "passed": bool(sa <= SA_TARGET),
            "source": "cfm_free_generation_panel_seed0.json:panel.sa_mean",
        },
        "qed": {
            "value": round(qed, 4),
            "target": None,
            "passed": None,
            "source": "cfm_free_generation_panel_seed0.json:panel.qed_mean",
        },
        "docking": docking,
    }
    numeric_ok = all(
        v["passed"] for k, v in checks.items() if isinstance(v, dict) and v.get("passed") is not None
    )
    passed = bool(numeric_ok and docking["status"] != "BLOCKED")
    if is_generation:
        verdict = "PASS" if passed else "NEGATIVE"
    else:
        # The generation-side verdict is NOT established by a reconstruction
        # pool: the sampler was never invoked on these molecules, so a
        # PASS/NEGATIVE here would be a verdict about the generator reached
        # through the decoder.  The numbers stay; the claim does not.
        verdict = "MEASURED_NOT_A_CLAIM"

    # Every check carries the scope of the pool it was measured on, so a
    # decoder-fidelity number cannot be read as a generator number.  The
    # docking check is the exception: it reports the docking RUN's status
    # (cfm_vina_generated.json), not a property of the subject pool.
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        if name == "docking":
            check["scope"] = "generation_docking_run"
            check["pool_kind_independent"] = True
            continue
        check["scope"] = scope
        check["pool_kind"] = kind.kind

    payload = {
        "artifact": "cfm_generation_metrics",
        "version": "2.0",
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "generated_csv": str(args.generated),
            "generated_sha256": _sha256(args.generated),
            "train_csv": str(args.train),
            "train_sha256": _sha256(args.train),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "pool_kind_source": POOL_KIND_SOURCE,
        },
        "pool_kind": pool_kind_block(kind),
        "subject": {
            "pool_kind": kind.kind,
            "pool": str(args.generated),
            "these_numbers_are": meaning,
            "these_numbers_are_not": (
                "a claim about the generator: the sampler was never invoked on "
                "this pool, so nothing here is a property of the generator's "
                "own output"
                if not is_generation
                else "a decoder-fidelity number: this pool IS the sampler's "
                "own free-generation output"
            ),
            "generation_side_published": is_generation,
            "generation_side_refusal_reason": None
            if is_generation
            else (
                "the subject pool is a RECONSTRUCTION pool "
                f"({POOL_KIND_SOURCE}): {kind.reason}.  Its numbers are the "
                "decoder's fidelity on real geometries, so they are not "
                "published as generation-side metrics."
            ),
        },
        "generation_side": {
            "published": is_generation,
            "reason": (
                "the subject pool is the trainer's free-generation output"
                if is_generation
                else (
                    "the subject pool is a RECONSTRUCTION pool, so this artifact "
                    "does NOT publish generation-side metrics: the sampler was "
                    "never invoked on it.  The measured numbers are reported "
                    "with the scope of the kind they belong to."
                )
            ),
            "what_would_be_needed": (
                "a pool the shared rule classifies as a generation pool "
                f"({POOL_KIND_SOURCE}) -- e.g. the trainer's own "
                f"{GENERATION_MARKER!r} pool, or a decode-dump pool whose "
                "content shows the ODE path"
            ),
            "pool_kind_source": POOL_KIND_SOURCE,
        },
        "design": {
            "cohort": "cfm_fullcorpus_2026_10_10 epoch 10 seed 0, 192 attempts -> 66 survivors",
            "train_reference": "metalcytotox_5metal_multitask_v4.csv, split == train (18,137 rows)",
            "fingerprint": "Morgan radius 2, 1024 bits",
            "novelty": "1 - max Tanimoto to any training molecule",
            "intdiv1": "MOSES: 1 - mean pairwise Tanimoto, self-pairs included",
            "n_matched_subsamples": N_MATCHED_SUBSAMPLES,
            "seed": int(args.seed),
            "single_seed_caveat": (
                "One seed, one checkpoint, n=66 survivors.  This is a first "
                "measurement, not a settled number."
            ),
            "pool_kind_source": POOL_KIND_SOURCE,
            "cohort_note": (
                "the subject pool is a RECONSTRUCTION pool, so '66 survivors' "
                "counts real tmQM geometries the bond head decoded, not "
                "molecules the sampler drew"
                if not is_generation
                else "the subject pool is the sampler's own free-generation output"
            ),
        },
        "cohort": {
            "n_generated": int(len(raw)),
            "n_parsed": int(len(parsed)),
            "parse_rate": round(len(parsed) / max(len(raw), 1), 4),
            "n_unique_canonical": int(len(set(parsed))),
            "uniqueness": round(len(set(parsed)) / max(len(parsed), 1), 4),
            "yield_from_attempts": 0.344,
            "yield_source": "cfm_free_generation_panel_seed0.json:generation.gen_validity",
            "pool_kind": kind.kind,
            "scope": scope,
        },
        "novelty": {
            "mean_max_tanimoto": round(float(sims.mean()), 4),
            "median_max_tanimoto": round(float(np.median(sims)), 4),
            "min_max_tanimoto": round(float(sims.min()), 4),
            "max_max_tanimoto": round(float(sims.max()), 4),
            "mean_novelty": round(float(novelty.mean()), 4),
            "fraction_below_0.9": round(frac_novel, 4),
            "fraction_below_0.7": round(float(np.mean(sims < 0.7)), 4),
            "pool_kind": kind.kind,
            "scope": scope,
        },
        "metal_compliance": {**metal_profile(parsed), "pool_kind": kind.kind, "scope": scope},
        "checks": checks,
        "verdict": verdict,
        "verdict_detail": {
            "scope": scope,
            "gate_met": bool(passed) if is_generation else None,
            "note": (
                "the gate is evaluated over the sampler's own output"
                if is_generation
                else (
                    "no generation-side verdict is published: the subject pool "
                    "is a reconstruction, so the numbers below are the "
                    "decoder's fidelity on real geometries (" + scope + ")"
                )
            ),
        },
        "verdict_rule": (
            "novelty fraction >= 0.90 AND IntDiv1 >= 0.85 AND SA <= 4.5 AND "
            "docking >= 50% on >= 20 pockets; docking is BLOCKED so the overall "
            "verdict cannot be PASS"
            + (
                ""
                if is_generation
                else (
                    ".  This rule applies to a GENERATION pool only; the "
                    "subject pool is a reconstruction, so the verdict is "
                    "MEASURED_NOT_A_CLAIM and the numbers are published with "
                    "scope " + scope + "."
                )
            )
        ),
    }
    atomic_write_file(args.out, json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"[met9] pool_kind={kind.kind} scope={scope} "
        f"n={len(parsed)} unique={len(set(parsed))} "
        f"novelty_mean={novelty.mean():.4f} frac<0.9={frac_novel:.3f} "
        f"intdiv1={gen_intdiv:.4f} (train n-matched {matched.mean():.4f}) "
        f"SA={sa:.4f} QED={qed:.4f} verdict={verdict}"
    )
    print(f"[met9] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
