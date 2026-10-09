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
    out: dict[str, Any] = {"smiles": smiles, "sa": sa_val, "qed": qed_val}
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
        required=True,
        type=Path,
        help="CSV with a 'smiles' column (one molecule per row).",
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    smiles_list = read_smiles_csv(args.smiles_csv, last_epoch=args.last_epoch)
    panel = build_panel(smiles_list, include_validity=args.decode_chemistry_validity)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(panel, fh, indent=2)
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
