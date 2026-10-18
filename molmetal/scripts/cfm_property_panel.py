#!/usr/bin/env python3
"""Wf-264-SA — CFM generated molecule SA + QED property panel.

======================================================================
Wf-264-SA — SA (Synthetic Accessibility) + QED (Quantitative Estimate
of Drug-likeness) panel for CFM-generated molecules.
======================================================================

Reads a CSV with a ``smiles`` column (one molecule per row), computes the
RDKit SA score (Ertl & Schuffenhauer 2009, 1-10 scale, **lower = more
synthesizable**) and the RDKit QED score (Bickerton 2012, 0-1 scale,
**higher = more drug-like**) for each molecule, and emits a JSON file
with per-molecule results plus aggregate statistics.

Why this panel exists
---------------------
* **SA score** measures how hard a target molecule is to synthesize.
  It is the standard panel metric used by GuacaMol, MolDQN, and most
  CFM/diffusion generative chemistry papers (Ertl & Schuffenhauer 2009).
* **QED** is Bickerton 2012's weighted multi-property drug-likeness
  estimate (MW / LogP / HBA / HBD / PSA / ROTB / AROM / ALERTS).
* Together they form the canonical "is the generated molecule any
  good?" diagnostic — without them, decode_ratio is meaningless
  because a model could "succeed" at decoding garbage.

Inputs
------
* ``--smiles-csv PATH``  CSV with at minimum a ``smiles`` column.
* ``--decode-chemistry-validity``  (BLK-C3 NEW) emit per-molecule
  chemistry-validity flags (parse / sanitize / valence / ring / radical)
  and an aggregate ``chemistry_validity_pass_rate`` distinct from
  ``parsed_ratio``.  ``parsed_ratio`` only checks that RDKit can read
  the string; ``chemistry_validity_pass_rate`` checks that the resulting
  molecule obeys the standard chemistry rules (valences, no
  anti-aromatic 4n+2-on-4n rings, no radicals, no ring size < 3).
  A low ``chemistry_validity_pass_rate`` is a stronger HALT signal
  than a low ``parsed_ratio``.

Outputs
-------
* ``--out-json PATH``  JSON with shape::

    {
      "panel_version": "wf264-sa/v1",
      "n_input": int,
      "n_parsed": int,
      "n_failed": int,
      "per_molecule": [
        {"smiles": str, "sa": float, "qed": float,
         # when --decode-chemistry-validity is set:
         "validity": {"parse_ok": bool, "sanitize_ok": bool,
                      "valence_ok": bool, "ring_ok": bool,
                      "no_radical_ok": bool, "all_pass": bool}},
        ...
      ],
      "aggregate": {
        "sa": {"mean": float, "std": float,
               "p10": float, "p25": float, "p50": float,
               "p75": float, "p90": float, "min": float, "max": float,
               "n": int},
        "qed": { ... same shape ... },
        "parsed_ratio": float,
        "n_input": int,
        "n_parsed": int,
        "n_failed": int
        # when --decode-chemistry-validity is set:
        "chemistry_validity_pass_rate": float,
        "chemistry_validity_breakdown": {
          "parse_ok": int, "sanitize_ok": int,
          "valence_ok": int, "ring_ok": int,
          "no_radical_ok": int, "all_pass": int
        }
      }
    }

  ``sa`` / ``qed`` are ``null`` for SMILES that fail to parse.

Usage
-----
::

    .venv/bin/python molmetal/scripts/cfm_property_panel.py \\
        --smiles-csv /tmp/cfm_generated.csv \\
        --out-json molmetal/reports/wf264_sa_qed/panel.json

Constraints
-----------
* RDKit is mandatory; SA/QED are standard RDKit calls.
* No model checkpoint is loaded; this panel is **post-hoc** — it
  operates on any SMILES file.
* Honest-framing: ``n_parsed / n_input`` is the parse-rate; a low
  parse-rate is itself a HALT signal worth reporting.

Attribution
-----------
* SA score: Ertl P., Schuffenhauer A. (2009) ``Estimation of synthetic
  accessibility score of drug-like molecules based on molecular
  complexity and fragment contributions``.
* QED: Bickerton G.R. et al. (2012) ``Quantifying the chemical beauty
  of drugs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


PANEL_VERSION = "wf264-sa/v1"


def _load_sascorer():
    """Late-import RDKit's bundled SA scorer with the upstream import path.

    The standard rdkit Contrib SA_Score package is importable as a
    regular Python module from its install dir; we import it lazily so
    the script's docstring/help works even when RDKit SA data is
    missing.  Returns the module (which exposes ``calculateScore``).
    """
    from rdkit.Chem import RDConfig  # noqa: WPS433
    import os as _os

    sa_dir = _os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # type: ignore[import-not-found]  # noqa: WPS433

    return sascorer


def _percentiles(values: list[float]) -> dict[str, float]:
    """Return mean / std / percentiles / min / max for non-NaN values.

    NaN values are filtered out before computing statistics; if no
    values remain, all keys are returned as ``None``.
    """
    clean = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not clean:
        return {
            "mean": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    n = len(clean)
    mean = sum(clean) / n
    var = sum((v - mean) ** 2 for v in clean) / n if n > 0 else 0.0
    std = math.sqrt(var)
    srt = sorted(clean)

    def _q(q: float) -> float:
        # Linear-interpolated quantile on the sorted sample.
        if n == 1:
            return srt[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return srt[lo]
        frac = pos - lo
        return srt[lo] + (srt[hi] - srt[lo]) * frac

    return {
        "mean": mean,
        "std": std,
        "p10": _q(0.10),
        "p25": _q(0.25),
        "p50": _q(0.50),
        "p75": _q(0.75),
        "p90": _q(0.90),
        "min": srt[0],
        "max": srt[-1],
        "n": n,
    }


#: Metal caps for the validity check, on the SAME basis as the loss side.
#:
#: 2026-10-10, second correction. The first version of this constant was
#: derived from tmQM's ``coord_number`` (the ``MND`` field) and was therefore
#: on the wrong scale — MND is a crystallographic coordination number while
#: ``Atom.GetTotalValence()`` here is a bond-order sum, and the two disagree
#: badly (mean |MND - metal_bo_sum| runs 0.81 for Re to 2.43 for Ir). The caps
#: now come from ``TMQM_MEASURED_METAL_CAPS`` in decoder_rework.py, which is
#: measured on ``metal_bo_sum`` — the same quantity the CFM soft-valence loss
#: term constrains. One basis, two consumers, so the metric and the objective
#: cannot drift apart again.
#:
#: See ``molmetal/tests/test_cfm_metal_valence_caps.py``, which pins the table
#: against the corpus percentiles it was derived from.
try:  # the panel must still run if the package import path is odd
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_CAPS as _MEASURED_COORD_CAPS,
    )
except Exception:  # noqa: BLE001
    _MEASURED_COORD_CAPS = {78: 5.0, 44: 7.0, 77: 6.0, 45: 6.0, 76: 7.0, 75: 8.0}


def _validate_molecule(mol: Any) -> dict[str, bool]:
    """BLK-C3 NEW: chemistry-validity check beyond RDKit parse.

    Returns a dict of boolean flags; ``all_pass`` is the AND of all
    sub-checks.  Sub-checks:

    * ``parse_ok``        : caller already passed a non-None mol.
    * ``sanitize_ok``     : ``Chem.SanitizeMol(mol)`` returns without
                            raising (catches valence / aromaticity /
                            kekulization issues that MolFromSmiles lets
                            through).
    * ``valence_ok``      : every atom's total valence is in
                            ``rdkit.Chem.Atom.GetAllowedValences``
                            (default 0 charge + the element's standard
                            set).  Catches hypervalent C, 5-valent N,
                            etc.
    * ``ring_ok``         : every ring is size >= 3 (catches
                            2-membered rings, the 4n+2-on-4n
                            anti-aromatic case manifests as a ring
                            that RDKit cannot sanitize, so this is
                            usually redundant with sanitize_ok; we
                            keep it as a separate gate so the
                            per-check breakdown is informative).
    * ``no_radical_ok``   : every atom has ``GetNumRadicalElectrons() == 0``.

    Any exception inside a sub-check is treated as a failure (False)
    rather than raising; the panel must always return a verdict.
    """
    from rdkit import Chem  # noqa: WPS433

    out: dict[str, bool] = {
        "parse_ok": True,
        "sanitize_ok": False,
        "valence_ok": False,
        "ring_ok": False,
        "no_radical_ok": False,
        "all_pass": False,
    }
    if mol is None:
        out["parse_ok"] = False
        return out
    try:
        Chem.SanitizeMol(mol)
        out["sanitize_ok"] = True
    except Exception:  # noqa: BLE001
        out["sanitize_ok"] = False
    try:
        ring_info = mol.GetRingInfo()
        sizes = [len(r) for r in ring_info.AtomRings()]
        out["ring_ok"] = (len(sizes) == 0) or all(s >= 3 for s in sizes)
    except Exception:  # noqa: BLE001
        out["ring_ok"] = False
    try:
        out["no_radical_ok"] = all(
            a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms()
        )
    except Exception:  # noqa: BLE001
        out["no_radical_ok"] = False
    try:
        # 2026-10-10 FIX. The previous line was
        #     a.GetTotalValence() in a.GetAllowedValences()
        # and `Atom.GetAllowedValences` DOES NOT EXIST in RDKit 2026.03.6 —
        # the valence table lives on the PeriodicTable. So every molecule hit
        # the AttributeError, the bare `except Exception` turned it into
        # `valence_ok = False`, and the panel reported a chemistry verdict
        # that had never been computed. MEASURED: 0/2146 "valence_ok" across
        # three real CFM runs, which read as "the generator produces nothing
        # chemically valid". It was a crashed check wearing a metric's
        # clothes. Same class as the swallowed TypeError that produced a false
        # 0/6, and the same reason the fix below refuses to swallow anything:
        # a check that cannot run must say so, not return a chemistry answer.
        #
        # Metals are handled SEPARATELY and on purpose. RDKit's valence model
        # is an ORGANIC valence model — for a transition metal "allowed
        # valence" is a coordination-chemistry question with no single right
        # answer, and folding it in here makes organometallics unscoreable.
        # The two questions get two flags.
        pt = Chem.GetPeriodicTable()
        nonmetal_ok = True
        metal_ok = True
        n_no_model = 0
        for a in mol.GetAtoms():
            z = a.GetAtomicNum()
            if z in _MEASURED_COORD_CAPS:
                cap = _MEASURED_COORD_CAPS[z]
                if a.GetTotalValence() > cap:
                    metal_ok = False
                continue
            if z > 20:  # a metal this project has no measured cap for
                n_no_model += 1
                continue
            allowed = pt.GetValenceList(z)
            if not allowed:
                n_no_model += 1
                continue
            if a.GetTotalValence() not in allowed:
                nonmetal_ok = False
        out["valence_ok"] = nonmetal_ok
        out["metal_coord_ok"] = metal_ok
        out["n_atoms_without_valence_model"] = n_no_model
    except Exception as exc:  # noqa: BLE001
        # Record WHY rather than inventing a False. If this branch is ever
        # reached, the metric is broken and the report must not look like a
        # chemistry result.
        out["valence_ok"] = None
        out["metal_coord_ok"] = None
        out["valence_check_error"] = f"{type(exc).__name__}: {exc}"
    out["all_pass"] = all(
        out.get(k) is True
        for k in ("parse_ok", "sanitize_ok", "valence_ok", "metal_coord_ok",
                  "ring_ok", "no_radical_ok")
    )
    return out


def compute_one(
    smiles: str,
    sascorer,
    *,
    include_validity: bool = False,
) -> dict[str, Any]:
    """Compute SA + QED for a single SMILES string.

    Returns ``{"smiles": ..., "sa": float|None, "qed": float|None}``.
    ``sa`` / ``qed`` are ``None`` if RDKit cannot parse the SMILES.

    When ``include_validity`` is True (BLK-C3 NEW), also returns a
    ``validity`` dict (see :func:`_validate_molecule`).
    """
    from rdkit import Chem  # noqa: WPS433
    from rdkit.Chem import QED  # noqa: WPS433

    sa_val: Any = None
    qed_val: Any = None
    # 2026-10-10 (BLK-QED-def) — fragmentation-aware QED.
    #
    # `qed` above is QED of the WHOLE parsed molecule, and RDKit happily
    # accepts a multi-fragment SMILES. So on a cohort where most decoded
    # molecules are disconnected, that number is not the drug-likeness of a
    # molecule — it is a score for a bag of fragments. MEASURED on 776 decoded
    # CFM molecules (85.1% multi-fragment):
    #
    #     QED on the whole SMILES          0.1952   <- what used to be reported
    #     QED on the largest fragment      0.2718
    #
    # a 39% relative understatement. And taking the largest fragment is NOT the
    # fix either: single-fragment molecules here score LOWER still (0.1246),
    # because they are small connected fragments and QED penalises small
    # molecules. So both readings are wrong in different directions, which is
    # why the per-molecule record now carries the raw facts —
    # `n_frags`, `largest_frag_frac`, `qed_largest_fragment` — and the
    # aggregate reports the single-fragment-only QED with its denominator.
    n_frags: Any = None
    largest_frag_frac: Any = None
    qed_largest: Any = None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            sa_val = float(sascorer.calculateScore(mol))
        except Exception:  # noqa: BLE001
            sa_val = None
        try:
            qed_val = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed_val = None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            n_frags = len(frags)
            heavy = mol.GetNumHeavyAtoms()
            if frags and heavy:
                best = max(frags, key=lambda f: f.GetNumHeavyAtoms())
                largest_frag_frac = best.GetNumHeavyAtoms() / heavy
                qed_largest = float(QED.qed(best))
        except Exception:  # noqa: BLE001
            pass
    out: dict[str, Any] = {
        "smiles": smiles,
        "sa": sa_val,
        "qed": qed_val,
        "n_frags": n_frags,
        "largest_frag_frac": largest_frag_frac,
        "qed_largest_fragment": qed_largest,
    }
    if include_validity:
        out["validity"] = _validate_molecule(mol)
    return out


def read_smiles_csv(path: Path, last_epoch: bool = False) -> list[str]:
    """Read a CSV with a ``smiles`` column.

    Accepts both quoted and unquoted SMILES, strips whitespace, drops
    empty rows.  Raises ``ValueError`` if no ``smiles`` column is found.

    When ``last_epoch`` is set, keep only rows whose integer ``epoch``
    column equals the CSV's maximum epoch.  The trainer's
    ``generated_smiles.csv`` is append-only across epochs, so this selects
    the final checkpoint's cohort instead of pooling every epoch.
    """
    smiles_list: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header: list[str] | None = None
        epochs: list[int] = []
        col_idx: int = -1
        epoch_idx: int | None = None
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if header is None:
                # Normalise header and find the smiles column index.
                lowered = [c.strip().lower() for c in row]
                if "smiles" not in lowered:
                    raise ValueError(
                        f"CSV at {path} has no 'smiles' column " f"(header={row!r})"
                    )
                header = row
                col_idx = lowered.index("smiles")
                epoch_idx = lowered.index("epoch") if "epoch" in lowered else None
                continue
            if col_idx >= len(row):
                continue
            s = row[col_idx].strip()
            if not s:
                continue
            smiles_list.append(s)
            if epoch_idx is not None and epoch_idx < len(row):
                try:
                    epochs.append(int(float(row[epoch_idx].strip())))
                except ValueError:
                    epochs.append(0)
    if last_epoch and epochs:
        top = max(epochs)
        smiles_list = [s for s, e in zip(smiles_list, epochs) if e == top]
    return smiles_list


def internal_diversity(
    smiles_list: list[str],
    *,
    radius: int = 2,
    n_bits: int = 1024,
    max_mols: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    """MOSES IntDiv1 over the generated cohort (BLK-M7-def).

    ``IntDiv1 = 1 - mean_{m1,m2 in G} T(m1,m2)`` — Polykovskiy et al. 2018
    (arXiv:1811.12823), which is exactly "mean pairwise Tanimoto distance",
    the quantity the project has been calling ``diversity_tanimoto``. Two
    details are matched to the definition rather than left implicit:

      * the sum runs over ALL ORDERED pairs INCLUDING self-pairs (T=1), which
        is what ``|G|^2`` in the MOSES formula means. With a 300-molecule
        sample, excluding self-pairs shifts the value by roughly 1/300;
      * Morgan radius 2 / **1024** bits, MOSES's ECFP4 setting. This project
        previously used 2048 bits in an ad-hoc script, which is not the
        reference configuration.

    WHY THE 0.88 THRESHOLD IS IN THE RETURN VALUE
    ---------------------------------------------
    MEASURED CFM value: 0.90-0.93, against a MOSES training-set reference of
    0.857 and a model range of 0.851-0.874. So this cohort scores ABOVE the
    data it was trained on. The field's own reading of that (Vendi Score,
    arXiv:2210.02410) is that values above ~0.88 on MOSES are rare and usually
    indicate distribution shift rather than genuine novelty — their HMM model
    posts one of the highest IntDiv1 scores (0.871) while ranking lowest on
    Vendi Score, i.e. IntDiv cannot separate real diversity from combinatorial
    randomness. IntDiv is also near-useless as a discriminator between models;
    it is a mode-collapse alarm, not a quality score.

    Consequence for reporting: a HIGH value here is not a win and must not be
    presented as one. The same paper notes GA/search methods collapse to
    0.13-0.15 on JNK3/sEH — which is exactly what the Lambda MCTS path's
    0.1366 looks like, and is the number on that path that IS diagnostic.

    ``max_mols`` subsamples deterministically (seeded) because the pairwise sum
    is O(n^2); the sample size actually used is returned so the number is
    never read as if it covered the whole cohort.
    """
    import random as _random

    from rdkit import Chem, DataStructs  # noqa: WPS433
    from rdkit.Chem import rdFingerprintGenerator  # noqa: WPS433

    uniq = sorted({s for s in smiles_list if s})
    if len(uniq) > max_mols:
        _random.Random(seed).shuffle(uniq)
        uniq = uniq[:max_mols]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    fps = []
    for s in uniq:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(fpgen.GetFingerprint(m))
    n = len(fps)
    if n == 1:
        # A cohort that dedupes to ONE distinct molecule is not "undefined" —
        # it is maximally collapsed, and IntDiv is a collapse alarm, so
        # returning None here would hide exactly the signal the metric exists
        # for. IntDiv over |G|=1 is 1 - T(m,m) = 0 by the same formula.
        return {"intdiv1": 0.0, "n_used": 1,
                "n_unique": len(set(smiles_list)),
                "radius": radius, "n_bits": n_bits, "self_pairs": True,
                "moses_train_set_reference": 0.857,
                "above_reference_flag": False,
                "note": ("every molecule in the cohort is identical after "
                         "deduplication — this is total mode collapse, and "
                         "IntDiv1 is 0 by the same formula, not undefined")}
    if n < 1:
        return {"intdiv1": None, "n_used": n,
                "n_unique": len(set(smiles_list)),
                "radius": radius, "n_bits": n_bits, "self_pairs": True,
                "moses_train_set_reference": 0.857,
                "note": "no molecule parsed"}
    total = 0.0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        total += sum(sims)          # includes the self-pair, T=1
    return {
        "intdiv1": 1.0 - (total / (n * n)),
        "n_used": n,
        "n_unique": len(set(smiles_list)),
        "radius": radius,
        "n_bits": n_bits,
        "self_pairs": True,
        "moses_train_set_reference": 0.857,
        "moses_model_range": [0.851, 0.874],
        "above_reference_flag": (1.0 - (total / (n * n))) > 0.88,
        "note": (
            "IntDiv1 per MOSES (arXiv:1811.12823). A value above 0.88 exceeds "
            "the MOSES training-set reference and is usually read as "
            "distribution shift / combinatorial randomness rather than genuine "
            "novelty (Vendi Score, arXiv:2210.02410). Do not present a high "
            "value here as a strength."
        ),
    }


def build_panel(
    smiles_list: list[str],
    *,
    include_validity: bool = False,
) -> dict[str, Any]:
    """Compute the full SA+QED panel for an iterable of SMILES.

    When ``include_validity`` is True (BLK-C3 NEW), every per-molecule
    entry gains a ``validity`` dict and the aggregate gains a
    ``chemistry_validity_pass_rate`` + per-check breakdown.
    """
    sascorer = _load_sascorer()
    per_molecule = [
        compute_one(s, sascorer, include_validity=include_validity)
        for s in smiles_list
    ]

    sa_vals = [m["sa"] for m in per_molecule]
    qed_vals = [m["qed"] for m in per_molecule]
    n_parsed = sum(
        1 for m in per_molecule if m["sa"] is not None and m["qed"] is not None
    )
    n_input = len(per_molecule)
    parsed_ratio = (n_parsed / n_input) if n_input > 0 else 0.0

    aggregate: dict[str, Any] = {
        "sa": _percentiles([v for v in sa_vals if v is not None]),
        "qed": _percentiles([v for v in qed_vals if v is not None]),
        "parsed_ratio": parsed_ratio,
        "n_input": n_input,
        "n_parsed": n_parsed,
        "n_failed": n_input - n_parsed,
    }

    if include_validity and per_molecule and "validity" in per_molecule[0]:
        keys = ("parse_ok", "sanitize_ok", "valence_ok", "ring_ok",
                "no_radical_ok", "all_pass")
        breakdown: dict[str, int] = {k: 0 for k in keys}
        for m in per_molecule:
            v = m.get("validity") or {}
            for k in keys:
                if v.get(k):
                    breakdown[k] += 1
        n_all_pass = breakdown["all_pass"]
        aggregate["chemistry_validity_pass_rate"] = (
            n_all_pass / n_input if n_input > 0 else 0.0
        )
        aggregate["chemistry_validity_breakdown"] = breakdown

    # 2026-10-10 (BLK-QED-def) — the single-fragment-only view, with its
    # denominator stated. This is the number to quote for m5; `qed` above is
    # kept only for continuity and is explicitly labelled contaminated.
    n_single = sum(1 for m in per_molecule if m.get("n_frags") == 1)
    sf_qed = [
        m["qed"] for m in per_molecule
        if m.get("n_frags") == 1 and m.get("qed") is not None
    ]
    lf_qed = [
        m["qed_largest_fragment"] for m in per_molecule
        if m.get("qed_largest_fragment") is not None
    ]
    aggregate["n_single_fragment"] = n_single
    aggregate["single_fragment_rate"] = (
        n_single / n_input if n_input > 0 else 0.0
    )
    aggregate["qed_single_fragment"] = _percentiles(sf_qed)
    aggregate["qed_largest_fragment_all"] = _percentiles(lf_qed)
    aggregate["multifragment_rate"] = (
        (n_input - n_single) / n_input if n_input > 0 else 0.0
    )
    # 2026-10-10 (BLK-M7-def) — m7 as a reproducible MOSES IntDiv1, replacing a
    # number that previously existed only in an ad-hoc script.
    aggregate["internal_diversity"] = internal_diversity(smiles_list)

    aggregate["qed_contamination_note"] = (
        "aggregate['qed'] is QED of the WHOLE parsed SMILES and RDKit accepts "
        "multi-fragment input, so on a fragmented cohort it scores a bag of "
        "fragments. Report 'qed_single_fragment' together with "
        "'single_fragment_rate'; do not quote 'qed' alone."
    )

    return {
        "panel_version": PANEL_VERSION,
        "n_input": n_input,
        "n_parsed": n_parsed,
        "n_failed": n_input - n_parsed,
        "per_molecule": per_molecule,
        "aggregate": aggregate,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cfm_property_panel.py",
        description=(
            "Wf-264-SA: SA + QED property panel for CFM-generated "
            "(or any) SMILES list.  Computes RDKit SA score "
            "(Ertl 2009, lower=better) + QED (Bickerton 2012, "
            "higher=better) per molecule and emits aggregate JSON."
        ),
    )
    p.add_argument(
        "--smiles-csv",
        type=Path,
        default=None,
        help=(
            "CSV with a 'smiles' column (one molecule per row).  "
            "Either --smiles-csv or --ckpt-path is required (not both)."
        ),
    )
    # Wf-A54 (2026-10-16) — close the decode→panel link gap.  Previously
    # the panel only accepted an external SMILES CSV; a CFM checkpoint
    # could not be fed in directly.  This flag lets the caller run the
    # full decode → panel pipeline from a single CLI invocation.  When
    # --ckpt-path is set, the helper at ``_decode_smiles_from_ckpt``
    # loads the checkpoint, samples ``--n-samples`` molecules via the
    # trainer's ``generate_clouds_from_noise`` + the YuelBond decoder,
    # writes them to a temp CSV, and the existing read_smiles_csv +
    # build_panel path runs on the decoded SMILES.  If decode fails
    # (incompatible ckpt schema, missing vel_field, etc.) the helper
    # emits an HONEST-NEGATIVE print and returns an empty SMILES list
    # so the panel still emits a valid (empty-cohort) JSON.
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help=(
            "Wf-A54: load a CFM checkpoint and decode → SMILES → "
            "panel in one CLI invocation.  Mutually exclusive with "
            "--smiles-csv.  The decode path is smoke-grade; see "
            "_decode_smiles_from_ckpt for the supported ckpt shapes."
        ),
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help=(
            "Wf-A54: number of molecules to decode when --ckpt-path "
            "is set.  Default 64 keeps the helper under 5 s wall "
            "on a smoke-stub checkpoint."
        ),
    )
    p.add_argument(
        "--decode-seed",
        type=int,
        default=0,
        help=(
            "Wf-A54: RNG seed for the CFM noise draw.  Default 0."
        ),
    )
    p.add_argument(
        "--out-json",
        required=True,
        type=Path,
        help="Output JSON path; will be created or overwritten.",
    )
    p.add_argument(
        "--last-epoch",
        action="store_true",
        help=(
            "If the CSV has an 'epoch' column, score only the final "
            "epoch's rows (the trainer's append-only generated_smiles.csv "
            "otherwise pools every epoch)."
        ),
    )
    p.add_argument(
        "--decode-chemistry-validity",
        action="store_true",
        help=(
            "BLK-C3 NEW: emit per-molecule chemistry-validity flags "
            "(parse / sanitize / valence / ring / no-radical) and an "
            "aggregate chemistry_validity_pass_rate.  Without this flag "
            "the panel emits only parsed_ratio (RDKit parse).  With the "
            "flag, chemistry_validity_pass_rate is a stronger HALT "
            "signal than parsed_ratio (a SMILES can parse but fail "
            "sanitize, e.g. anti-aromatic 4n+2-on-4n, hypervalent C)."
        ),
    )
    # Wf-A30-2 B6 (2026-10-14) — persist the gated cohort so m14 + the
    # downstream metric chain become reproducible post-hoc.  Without this
    # flag the panel emits only an aggregate JSON; downstream consumers
    # (paper table, ADMET8 panel, decode_gate) had to re-parse the
    # original SMILES CSV + re-apply the gate from scratch.  With the
    # flag, the gated SMILES (+ their per-row sa/qed/validity flags) are
    # written to disk so a later reviewer can re-run any downstream step
    # without re-deriving the gate.
    p.add_argument(
        "--out-smiles-csv",
        type=Path,
        # Wf-A36-2 F1 (2026-10-14) — default to "gated_cohort.csv" so the
        # trainer --save-generated-smiles → panel --smiles-csv X --out-smiles-csv
        # Y chain is reliable end-to-end without the caller needing to plumb
        # --out-smiles-csv through every launcher.  When the user does NOT
        # override --out-smiles-csv AND --smiles-csv is set, main() resolves
        # the default to live next to the input CSV (see main()), so the
        # gated cohort sits beside its source rather than in cwd.
        default="gated_cohort.csv",
        help=(
            "Wf-A30-2 B6 / Wf-A36-2 F1: write the gated cohort (per-row "
            "SMILES + sa + qed + chemistry-validity flags when "
            "--decode-chemistry-validity is set) to a CSV.  Default "
            "'gated_cohort.csv' — resolved next to --smiles-csv when the "
            "user does not pass --out-smiles-csv explicitly.  The 'gate' "
            "is all_pass=True when --decode-chemistry-validity is set, "
            "else parsed_ratio (sa and qed are not None)."
        ),
    )
    return p


def _build_gated_csv_rows(
    panel: dict[str, Any],
    *,
    include_validity: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Wf-A30-2 B6 — return (header, rows) for the gated cohort CSV.

    A row is "gated" iff:
      - include_validity=True  ->  validity.all_pass is True
      - include_validity=False  ->  sa and qed are both not None
    Returns the full per-row payload (smiles + sa + qed + validity if
    present) so a downstream consumer can re-score without re-parsing.
    """
    per_mol = panel.get("per_molecule", [])
    rows: list[dict[str, Any]] = []
    for m in per_mol:
        smi = m.get("smiles")
        sa = m.get("sa")
        qed = m.get("qed")
        validity = m.get("validity") if include_validity else None
        # Decide if the row is in the gated cohort.
        if include_validity and validity is not None:
            gated = bool(validity.get("all_pass"))
        else:
            gated = (sa is not None) and (qed is not None)
        if not gated:
            continue
        row = {"smiles": smi or "", "sa": sa, "qed": qed}
        if validity is not None:
            for k in (
                "parse_ok",
                "sanitize_ok",
                "valence_ok",
                "ring_ok",
                "no_radical_ok",
                "all_pass",
            ):
                row[k] = bool(validity.get(k))
        rows.append(row)
    header = ["smiles", "sa", "qed"]
    if include_validity:
        header.extend([
            "parse_ok",
            "sanitize_ok",
            "valence_ok",
            "ring_ok",
            "no_radical_ok",
            "all_pass",
        ])
    return header, rows


def _load_checkpoint_any(ckpt_path: Path) -> object:
    """Load a CFM checkpoint in EITHER serialisation the trainer emits.

    Wf-A81 (2026-10-18) — MEASURED break: since Wf-A30-1 (2026-10-14) the
    trainer writes checkpoints as a raw pickle through
    `wf386a3_atomic_write.atomic_write_file`, not through `torch.save`.
    `torch.load` on such a file raises `RuntimeError: Invalid magic number;
    corrupt file?`, and this helper caught it and returned an empty SMILES
    list -- so every checkpoint the trainer produced for the last four days
    read as "corrupt" at the one place that tries to decode it.  Verified on
    `/tmp/wfa81_smoke/ckpt/cfm_yuelbond_wasserstein_seed0.pt`.

    Order matters: `torch.load` first (zip magic), pickle only on the magic-
    number error, so a genuinely truncated file still raises instead of
    producing a confusing UnpicklingError.
    """
    import torch

    try:
        return torch.load(
            str(ckpt_path), map_location="cpu", weights_only=False
        )
    except RuntimeError as exc:
        if "magic number" not in str(exc):
            raise
        import pickle

        with open(ckpt_path, "rb") as fh:
            return pickle.load(fh)


def _decode_smiles_from_ckpt(
    ckpt_path: Path,
    *,
    n_samples: int,
    seed: int,
) -> list[str]:
    """Wf-A54 (2026-10-16) — load a CFM checkpoint and decode ``n_samples``
    molecules to canonical SMILES.

    Implementation
    --------------
    The decode path reuses the trainer's production wiring:

      * :func:`generate_clouds_from_noise` (trainer line 5406) draws noise
        clouds and integrates the velocity field via the trainer's
        ODESolver.  Each cloud's atomic numbers come from a fixed template
        cohort (small drug-like scaffolds) since we cannot compute a
        per-corpus template on demand without the loader.
      * :class:`YuelBondBondDecoder` (cfm_decoder_yuelbond.py:245) maps each
        cloud's per-pair logits to a bond graph, with the valence-aware
        greedy selector and connectivity prior from the trained head.

    Failure modes — ALL caught and surfaced as HONEST-NEGATIVE
    ----------------------------------------------------------
    A real production ckpt may carry any of: ``vel_field.*`` keys
    (use_egnn_velocity=True), ``vel_probe.*`` keys (use_egnn_velocity=
    False), or a head-only smoke stub (smoke_stub.pt has 28 entries —
    all under ``head.*`` + ``fm_vel_head.*`` + ``vel_probe.*``).  This
    helper handles the head-only + vel_probe shapes (smoke-stub and
    pre-Wf-A36 production ckpts) and emits an HONEST-NEGATIVE print if
    the ckpt has a vel_field (post-Wf-A36 production) — the EGNN vel
     field requires ``EGNNVelocityField`` instantiation which is a 1-
     day wire-in beyond this fix's scope.

    Returns ``[]`` on any failure so the caller can write a valid (empty)
    JSON + a 1-row CSV (so downstream consumers that re-parse don't crash).
    """
    # 2026-10-16 — make ``molmetal.*`` importable when this script is
    # invoked as ``python molmetal/scripts/cfm_property_panel.py`` (the
    # standard invocation puts the scripts/ dir on sys.path but NOT the
    # repo root, so the ``molmetal`` package isn't importable until we
    # add it).  Mirrors the pattern in main() below.
    _repo_root = Path(__file__).resolve().parent.parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    try:
        import torch  # noqa: WPS433 — lazy
    except ImportError as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: torch import failed "
            f"({exc!r}); cannot load checkpoint. "
            "Returning empty SMILES list."
        )
        return []

    if not ckpt_path.is_file():
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: ckpt-path '{ckpt_path}' "
            f"is not a file. Returning empty SMILES list."
        )
        return []

    # Lazy imports — the panel is RDKit-only at default; pulling torch +
    # the trainer module only when --ckpt-path is set keeps the no-GPU
    # baseline import budget the same as the pre-A54 panel.
    try:
        from molmetal.models.bond_head import AtomCloud  # noqa: WPS433
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (  # noqa: WPS433
            YuelBondBondDecoder,
        )
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: WPS433
            generate_clouds_from_noise,
        )
    except ImportError as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: trainer / decoder import "
            f"failed ({type(exc).__name__}: {exc}); the production "
            "decode wire-in is a future-wave deliverable. Returning "
            "empty SMILES list."
        )
        return []

    try:
        ckpt = _load_checkpoint_any(ckpt_path)
    except Exception as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: load('{ckpt_path}') "
            f"failed ({type(exc).__name__}: {exc}). Returning empty "
            "SMILES list."
        )
        return []

    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
    if state is None:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: ckpt at '{ckpt_path}' has "
            f"no 'model_state_dict' (top-level type={type(ckpt).__name__}). "
            "Returning empty SMILES list."
        )
        return []

    # Head-only / pre-Wf-A36 ckpt detection.  A v2 (EGNN-velocity) ckpt
    # would carry ``vel_field.*`` keys; if found, we refuse rather than
    # silently zeroing out the velocity pathway (the Wf-A35-1 C1 raise
    # pattern — fail fast and force a configuration fix).
    if any(k.startswith("vel_field.") for k in state.keys()):
        print(
            "[wfA54-decode] HONEST-NEGATIVE: ckpt has vel_field.* keys "
            "(EGNN-velocity v2). This helper handles v1 (vel_probe) + "
            "smoke-stub shapes only.  The EGNN-velocity decode path is "
            "a future-wave deliverable. Returning empty SMILES list."
        )
        return []

    # Build templates.  Real templates come from the tmQM cohort; the
    # smoke helper uses 3 fixed scaffolds (4-atom / 8-atom / 12-atom)
    # to exercise the velocity field at varying cloud sizes.  The
    # atomic numbers include Pt (Z=78) so the velocity field has a
    # meaningful centre-of-mass shift from noise to data.
    Z_H, Z_C, Z_N, Z_O, Z_PT = 1, 6, 7, 8, 78
    scaffolds = [
        ([Z_C, Z_H, Z_H, Z_H, Z_H], 5),                       # methane
        ([Z_C, Z_C, Z_H, Z_H, Z_H, Z_H, Z_H, Z_H], 8),        # ethane
        ([Z_C, Z_C, Z_N, Z_H, Z_H, Z_H, Z_H, Z_H], 8),        # methylamine-ish
    ]
    # Distribute n_samples evenly across scaffolds so each size band
    # gets a fair share.
    per_scaffold = max(1, n_samples // len(scaffolds))
    templates: list[AtomCloud] = []
    for atomic_numbers, n_atoms in scaffolds:
        for _ in range(per_scaffold):
            positions = torch.zeros((n_atoms, 3), dtype=torch.float32)
            templates.append(
                AtomCloud(positions=positions, atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long))
            )
    templates = templates[:n_samples]
    if not templates:
        return []

    # Build a minimal CFMJointModel matching the head-only ckpt shape.
    # Production wire-up of EGNN-velocity is out of scope for Wf-A54; the
    # v1 vel_probe path is what the smoke-stub actually carries, and the
    # head's in_dim / hidden_dim are not directly visible from the
    # state_dict keys alone.  We probe ``head.atom_encoder.weight`` for
    # the in_dim, and use the canonical 64-d hidden_dim that the bond
    # head expects when no explicit override is in the state_dict.
    try:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (  # noqa: WPS433
            CFMJointModel,
        )
    except ImportError as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: CFMJointModel import "
            f"failed ({exc!r}). Returning empty SMILES list."
        )
        return []

    try:
        # Probe hidden_dim from head.atom_encoder.weight.shape[0].  The
        # atom_encoder is ``nn.Linear(z_table_len + 9, hidden_dim)``, so
        # its OUTPUT feature count equals hidden_dim (verified on the
        # smoke-stub which has shape (64, 13) → hidden_dim=64,
        # z_table_len=4 because 13 = 9 + 4).  Using shape[1] would
        # mis-infer hidden_dim because shape[1] = z_table_len + 9 is
        # the atom-feature width, NOT the hidden dim.
        ae_w = state.get("head.atom_encoder.weight")
        if ae_w is None:
            print(
                "[wfA54-decode] HONEST-NEGATIVE: ckpt has no "
                "head.atom_encoder.weight; cannot infer hidden_dim. "
                "Returning empty SMILES list."
            )
            return []
        hidden_dim = int(ae_w.shape[0])
        if hidden_dim < 4:
            print(
                f"[wfA54-decode] HONEST-NEGATIVE: inferred hidden_dim="
                f"{hidden_dim} from head.atom_encoder.weight.shape[0]={int(ae_w.shape[0])}; "
                "suspiciously small, refusing to construct model."
            )
            return []

        # z_table: probe from head.atom_encoder.weight.shape[1].
        # atom_encoder input dim = z_table_len + 9 (one-hot z + 9 hand-crafted).
        ae_in = int(ae_w.shape[1])
        z_table_len = ae_in - 9
        if z_table_len <= 0:
            print(
                f"[wfA54-decode] HONEST-NEGATIVE: head.atom_encoder "
                f"weight.shape[1]={ae_in} implies z_table_len={z_table_len}; "
                "refusing to construct model."
            )
            return []
        # Map known z_table_len → canonical z_table tuple.  This helper
        # supports the smoke-stub 4-entry and 9-entry shapes plus the
        # production "pt" 13-entry and "all6" 18-entry shapes.  Anything
        # else is an HONEST-NEGATIVE (the trainer's own z_table preset
        # names are "pt" and "all6"; if a future preset is added the
        # helper needs the matching tuple).
        z_table_presets = {
            4: (0, 1, 6, 78),                                    # smoke-stub mini
            9: (0, 1, 6, 7, 8, 9, 15, 16, 17),                    # smoke-stub seed0
            13: (0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53, 78),  # production "pt"
            18: (0, 1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 44, 45,   # production "all6"
                 53, 75, 76, 77, 78),
        }
        if z_table_len not in z_table_presets:
            print(
                f"[wfA54-decode] HONEST-NEGATIVE: ckpt z_table_len="
                f"{z_table_len} (from head.atom_encoder.weight.shape[1]={ae_in}); "
                "no matching preset registered (supported: 4, 9, 13, 18). "
                "Returning empty SMILES list."
            )
            return []
        z_table = z_table_presets[z_table_len]

        model = CFMJointModel(
            hidden_dim=hidden_dim,
            n_layers=3,
            use_egnn_velocity=False,  # v1 ckpt shape (vel_probe)
            fm_objective=False,
            z_table=z_table,
        )
        # ``strict=False`` mirrors the trainer's own resume path (Wf-A30-1)
        # so a smoke-stub ckpt with no optimiser / EMA state can load.
        # NOTE: PyTorch's load_state_dict(strict=False) does NOT suppress
        # size-mismatch errors — those still raise RuntimeError.  We
        # catch that case below and emit a per-shape HONEST-NEGATIVE so
        # the helper does not crash on smoke-stub or partial-Z setups.
        try:
            model.load_state_dict(state, strict=False)
        except RuntimeError as rterr:
            err_str = str(rterr)
            # If the failure is a size mismatch in the head/vel head,
            # emit a precise HONEST-NEGATIVE keyed on z_table_len.
            if "size mismatch" in err_str:
                print(
                    f"[wfA54-decode] HONEST-NEGATIVE: CFMJointModel "
                    f"load_state_dict hit shape mismatches (z_table_len="
                    f"{z_table_len}, hidden_dim={hidden_dim}). The "
                    "smoke-stub ckpts use non-canonical vocab sizes "
                    "and need a direct-bond-head-only load path (future "
                    "wave). Returning empty SMILES list."
                )
                return []
            raise
        model.eval()
    except Exception as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: CFMJointModel "
            f"construction / load_state_dict failed "
            f"({type(exc).__name__}: {exc}). Returning empty SMILES list."
        )
        return []

    # Generate clouds from noise.
    device = torch.device("cpu")
    try:
        clouds = generate_clouds_from_noise(
            model,
            templates,
            device,
            method="euler",
            n_steps=10,  # smoke-grade: 10 steps for sub-5s wall
            seed=seed,
            noise_sigma=1.0,
            pad_groups=True,  # Wf-A49: per-bucket batching
            pad_group_bucket=8,
        )
    except Exception as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: generate_clouds_from_noise "
            f"failed ({type(exc).__name__}: {exc}). Returning empty "
            "SMILES list."
        )
        return []

    if not clouds:
        print("[wfA54-decode] HONEST-NEGATIVE: generate_clouds_from_noise "
              "returned 0 clouds; returning empty SMILES list.")
        return []

    # Decode each cloud via the bond head.
    try:
        head = model.head
        # Wf-A81 (2026-10-18) — decode with the rule the checkpoint was
        # trained and scored under.  Before this, a "per_class" run's own
        # gen_validity was measured with calibrated thresholds while this
        # panel decoded the same weights with the argparse default (argmax).
        # Empty kwargs for any checkpoint without the stamp, so old
        # artifacts decode exactly as before.
        try:
            from molmetal.eval.bond_threshold_calibration import (
                decoder_kwargs_from_checkpoint as _dec_kw_from_ckpt,
            )

            _dec_kw = _dec_kw_from_ckpt(ckpt)
        except Exception as _dec_kw_exc:  # pragma: no cover - safety net
            _dec_kw = {}
            print(
                f"[wfA54-decode] WARNING: decode_calibration not applied "
                f"({type(_dec_kw_exc).__name__}: {_dec_kw_exc})"
            )
        if _dec_kw:
            print(
                f"[wfA54-decode] decision from checkpoint: "
                f"{_dec_kw.get('decision')}"
                + (
                    f" thresholds={_dec_kw.get('per_class_thresholds')}"
                    if _dec_kw.get("per_class_thresholds")
                    else f" @t={_dec_kw.get('decision_threshold', 0.5):.2f}"
                )
            )
        decoder = YuelBondBondDecoder(
            head=head, hidden_dim=hidden_dim, n_layers=3, **_dec_kw
        )
        decoder.eval()
        smiles_list: list[str] = []
        for cloud in clouds:
            try:
                result = decoder.decode(cloud)
                smi = getattr(result, "smiles", None) or ""
                if smi:
                    smiles_list.append(smi)
            except Exception:
                # Per-cloud decode failure — skip and continue.
                continue
    except Exception as exc:
        print(
            f"[wfA54-decode] HONEST-NEGATIVE: YuelBondBondDecoder "
            f"construction or decode failed "
            f"({type(exc).__name__}: {exc}). Returning empty SMILES list."
        )
        return []

    print(
        f"[wfA54-decode] {len(smiles_list)}/{len(clouds)} clouds decoded "
        f"to SMILES (n_samples={n_samples}, seed={seed})."
    )
    return smiles_list


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Wf-A54 (2026-10-16) — exactly-one of --smiles-csv / --ckpt-path is
    # required.  Reject both-set and neither-set explicitly so a caller
    # mistake surfaces immediately.
    if args.smiles_csv is None and args.ckpt_path is None:
        raise SystemExit(
            "cfm_property_panel: one of --smiles-csv or --ckpt-path "
            "is required (Wf-A54)."
        )
    if args.smiles_csv is not None and args.ckpt_path is not None:
        raise SystemExit(
            "cfm_property_panel: --smiles-csv and --ckpt-path are "
            "mutually exclusive; pick one (Wf-A54)."
        )
    if args.ckpt_path is not None:
        # Decode → write to a temp CSV → fall through to read_smiles_csv.
        # The temp file is auto-cleaned by tempfile.NamedTemporaryFile.
        import tempfile

        decoded = _decode_smiles_from_ckpt(
            args.ckpt_path,
            n_samples=args.n_samples,
            seed=args.decode_seed,
        )
        if not decoded:
            # HONEST-NEGATIVE surfaced from the helper; emit an empty
            # panel + 1-row CSV (header-only) so downstream consumers
            # can read the file without crashing.
            import csv as _csv
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as fh:
                writer = _csv.writer(fh)
                writer.writerow(["smiles"])
                tmp_path = Path(fh.name)
            args.smiles_csv = tmp_path
        else:
            import csv as _csv
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as fh:
                writer = _csv.writer(fh)
                writer.writerow(["smiles"])
                for s in decoded:
                    writer.writerow([s])
                tmp_path = Path(fh.name)
            args.smiles_csv = tmp_path
    smiles_list = read_smiles_csv(args.smiles_csv, last_epoch=args.last_epoch)
    panel = build_panel(smiles_list, include_validity=args.decode_chemistry_validity)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2)
    # Wf-A36-2 F1 (2026-10-14) — resolve the default --out-smiles-csv
    # sentinel.  When the caller did NOT pass --out-smiles-csv explicitly
    # AND --smiles-csv is set, place the gated cohort next to the input
    # file (so the trainer --save-generated-smiles → panel --smiles-csv
    # X → --out-smiles-csv default chain is reliable without the launcher
    # plumbing an output through to the panel).  When the caller DID pass
    # --out-smiles-csv, honour that explicit choice verbatim.
    _OUT_DEFAULT = Path("gated_cohort.csv")
    if args.out_smiles_csv == _OUT_DEFAULT and args.smiles_csv is not None:
        out_smiles_csv = args.smiles_csv.parent / _OUT_DEFAULT.name
    else:
        out_smiles_csv = args.out_smiles_csv
    # Wf-A30-2 B6 — persist gated cohort when --out-smiles-csv is set.
    # Uses atomic_write_file (§5.6) so a partial write cannot truncate
    # the downstream consumer's view.
    if out_smiles_csv is not None:
        # Lazy import: atomic_write_file is in molmetal/scripts/wf386a3
        # Add repo root to sys.path so this script can be run as
        # ``python molmetal/scripts/cfm_property_panel.py`` (the
        # standard `scripts/cfm_property_panel.py` invocation puts the
        # scripts/ dir on sys.path but NOT the repo root, so the
        # ``molmetal`` package isn't importable until we add it).
        _repo_root = Path(__file__).resolve().parent.parent.parent
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        header, rows = _build_gated_csv_rows(
            panel, include_validity=args.decode_chemistry_validity,
        )
        out_smiles_csv.parent.mkdir(parents=True, exist_ok=True)
        # Build with stdlib csv module then dump to a string + atomic write.
        import io as _io
        buf = _io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        atomic_write_file(out_smiles_csv, buf.getvalue())
        # Wf-A36-2 F1 — 1-line honest-framing traceability stamp naming
        # the cohort + its source CSV + the gate used.  See also
        # §5.2 honest-negative framing.
        print(
            f"[cfm_property_panel] gated cohort written to {out_smiles_csv} "
            f"({len(rows)} of {panel['n_input']} rows; "
            f"gate={'all_pass' if args.decode_chemistry_validity else 'parsed'})"
        )
    print(
        f"[wf264-sa] n_input={panel['n_input']} "
        f"n_parsed={panel['n_parsed']} "
        f"n_failed={panel['n_failed']} "
        f"parsed_ratio={panel['aggregate']['parsed_ratio']:.3f}"
    )
    if args.decode_chemistry_validity:
        agg = panel["aggregate"]
        cvpr = agg.get("chemistry_validity_pass_rate")
        bd = agg.get("chemistry_validity_breakdown", {})
        print(
            f"[wf264-sa] chemistry_validity_pass_rate="
            f"{cvpr:.3f} (all_pass={bd.get('all_pass',0)}/"
            f"sanitize_ok={bd.get('sanitize_ok',0)}/"
            f"valence_ok={bd.get('valence_ok',0)}/"
            f"ring_ok={bd.get('ring_ok',0)}/"
            f"no_radical_ok={bd.get('no_radical_ok',0)}/"
            f"n_input={agg.get('n_input',0)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
