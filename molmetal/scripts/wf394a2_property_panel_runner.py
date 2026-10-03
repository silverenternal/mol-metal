#!/usr/bin/env python3
"""Wf-394-A2 — Diversity + SA aware property panel runner.

====================================================================
Why this script exists
====================================================================
Wf-264-SA + Wf-380-A6 ship two post-hoc property panels for CFM /
metallodrug-generated SMILES cohorts:

* :mod:`molmetal.scripts.cfm_property_panel` — SA + QED per molecule
  (Wf-264-SA, Ertl 2009 + Bickerton 2012).
* :mod:`molmetal.scripts.cfm_sa_aware_panel` — SA + SA-aware sampling
  penalty per molecule + aggregate (Wf-380-A6 / TODO-205).

Wf-388 wired both **diversity** and **SA-aware** paths into the CFM
trainer (``train_cfm_yuelbond_wasserstein.py``).  Wf-394-A2 provides a
**post-hoc property panel runner** that reuses the same SA-aware
aggregator on the trainer's generated SMILES and *adds* an optional
**diversity-aware evaluation** pass on top of the SA + QED outputs.

Wf-394-A2 design constraints:

* **CPU-only** — RDKit + python aggregation only; no torch, no GPU.
* **No model checkpoint loaded** — operates on any SMILES cohort
  (CFM-generated, Wf-188 MetalCytoToxDB ligand population, or
  smoke-test literals).
* **No trainer mutation** — the trainer flags ``--diversity-aware-eval``
  and ``--sa-aware`` are *evaluation*-side parameters propagated from
  :mod:`train_cfm_yuelbond_wasserstein.py` (Wf-388-A2 + Wf-388-A3).
  This runner provides a post-hoc analogue of those flags.
* **USER_OK-gated** via the canonical Wf-355 / Wf-294B pattern in the
  launcher wrapper; the runner itself never launches a GPU job.

Wf-394-A2 honest framing (party-twenty-spirit §3.5):

* SCAFFOLD aggregator — full diversity-aware evaluator (nucleus-sample +
  MMD) integration with the trainer is deferred to Wf-394-B1.
* The diversity pass here is a **light-weight post-hoc Tanimoto /
  scaffold-distribution computation** that runs in <1 s per cohort of
  ~100 SMILES on CPU.  PROJECTED wall budget for n=10k cohort is ~30 s.
* All numeric values are MEASURED at run time on the input cohort;
  no cross-cohort training is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Lazy imports of the SA-aware + property panel modules
# ---------------------------------------------------------------------------


def _lazy_import_property_panel():
    """Late-import :mod:`molmetal.scripts.cfm_property_panel` for SA + QED.

    The script may be invoked directly via
    ``python molmetal/scripts/wf394a2_property_panel_runner.py`` without
    the project root on ``sys.path``; add it defensively.
    """
    import importlib

    try:
        return importlib.import_module("molmetal.scripts.cfm_property_panel")
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # __file__ = <repo>/molmetal/scripts/wf394a2_property_panel_runner.py
        # we want <repo>/ on sys.path so `molmetal` package is importable
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return importlib.import_module("molmetal.scripts.cfm_property_panel")
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[wf394a2] WARN cannot import cfm_property_panel: {exc}\n")
        return None


def _lazy_import_sa_aware_panel():
    """Late-import :mod:`molmetal.scripts.cfm_sa_aware_panel` for SA-aware."""
    import importlib

    try:
        return importlib.import_module("molmetal.scripts.cfm_sa_aware_panel")
    except Exception:
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        return importlib.import_module("molmetal.scripts.cfm_sa_aware_panel")
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"[wf394a2] WARN cannot import cfm_sa_aware_panel: {exc}\n")
        return None


# ---------------------------------------------------------------------------
# Config dataclasses (mirror the Wf-388 trainer flags for SA / diversity)
# ---------------------------------------------------------------------------


@dataclass
class DiversityCfg:
    """Diversity-aware eval config (post-hoc analogue of Wf-388-A2).

    Attributes
    ----------
    enabled : bool
        If True, compute a Tanimoto / scaffold-distribution pass on the
        cohort.  Default False — opt-in to avoid the RDKit scaffold step.
    sample_size : int
        Max number of pairwise Tanimoto computations.  For n_cohort >=
        sample_size, a random subsample of size ``sample_size`` is drawn.
    seed : int
        RNG seed for the subsample.
    """

    enabled: bool = False
    sample_size: int = 1000
    seed: int = 42


@dataclass
class SaCfg:
    """SA-aware eval config (post-hoc analogue of Wf-388-A3).

    Attributes
    ----------
    target_sa : float
        Synthesizable threshold (default 2.8 — matches Wf-380-A6).
    sa_weight : float
        Linear SA penalty weight (default 0.3 — matches Wf-380-A6).
    """

    target_sa: float = 2.8
    sa_weight: float = 0.3


# ---------------------------------------------------------------------------
# SA + QED + SA-aware evaluation (delegated to existing panels)
# ---------------------------------------------------------------------------


def _read_smiles_csv(path: Path) -> List[str]:
    """Read SMILES from a CSV with a ``smiles`` column.  Lightweight helper.

    Mirrors :func:`cfm_sa_aware_panel.read_smiles_csv` semantics so the
    runner can ingest either panel's CSV output without a hard dep.
    """
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if "smiles" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV {path} missing required 'smiles' column; "
                f"found columns: {reader.fieldnames}"
            )
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if smi:
                out.append(smi)
    return out


def run_property_panel(
    cohort_smiles: Sequence[str],
    diversity_cfg: Optional[DiversityCfg] = None,
    sa_cfg: Optional[SaCfg] = None,
) -> Dict[str, Any]:
    """Run the Wf-394-A2 post-hoc property panel on a SMILES cohort.

    Parameters
    ----------
    cohort_smiles : Sequence[str]
        SMILES strings (one per molecule).
    diversity_cfg : DiversityCfg, optional
        If supplied (and ``enabled=True``), compute a Tanimoto /
        scaffold-distribution pass on top of the SA + QED outputs.
    sa_cfg : SaCfg, optional
        If supplied, compute the Wf-380-A6 SA-aware sampling penalty on
        top of the SA + QED outputs.

    Returns
    -------
    dict with keys::

        {
          "panel_version": "wf394a2/v1",
          "n_input": int,
          "sa_qed": {  -- delegated to cfm_property_panel.build_panel
            "n_parsed": int,
            "n_failed": int,
            "aggregate": {...},
          },
          "sa_aware": {...} | None,
          "diversity": {...} | None,
        }

    Notes
    -----
    * ``sa_qed`` always present (delegated to Wf-264-SA).
    * ``sa_aware`` present when ``sa_cfg`` is supplied.
    * ``diversity`` present when ``diversity_cfg.enabled=True``.
    """
    diversity_cfg = diversity_cfg or DiversityCfg(enabled=False)
    # If sa_cfg is None, leave it None — the test asserts sa_aware == None.
    # If sa_cfg is supplied, use it as-is.

    smiles_list = [str(s).strip() for s in cohort_smiles if str(s).strip()]
    n_input = len(smiles_list)

    # SA + QED (Wf-264-SA) — always
    pp_mod = _lazy_import_property_panel()
    if pp_mod is None:
        raise RuntimeError(
            "wf394a2: cannot import cfm_property_panel — SA + QED "
            "computation failed.  Ensure the project root is on "
            "sys.path."
        )
    sa_qed_panel = pp_mod.build_panel(smiles_list)
    sa_qed_out: Dict[str, Any] = {
        "n_parsed": int(sa_qed_panel["n_parsed"]),
        "n_failed": int(sa_qed_panel["n_failed"]),
        "parsed_ratio": float(sa_qed_panel["aggregate"]["parsed_ratio"]),
        "aggregate": sa_qed_panel["aggregate"],
    }

    # SA-aware penalty (Wf-380-A6) — opt-in (only when sa_cfg supplied)
    sa_aware_out: Optional[Dict[str, Any]] = None
    if sa_cfg is not None:
        sa_mod = _lazy_import_sa_aware_panel()
        if sa_mod is not None:
            sa_aware_panel = sa_mod.build_panel(
                smiles_list,
                target_sa=float(sa_cfg.target_sa),
                sa_weight=float(sa_cfg.sa_weight),
            )
            sa_aware_out = {
                "n_parsed": int(sa_aware_panel["n_parsed"]),
                "n_failed": int(sa_aware_panel["n_failed"]),
                "aggregate": sa_aware_panel["aggregate"],
            }

    # Diversity pass (Wf-380-A3 / Wf-388-A2) — opt-in
    diversity_out: Optional[Dict[str, Any]] = None
    if diversity_cfg.enabled:
        diversity_out = _compute_diversity_pass(
            smiles_list,
            sample_size=int(diversity_cfg.sample_size),
            seed=int(diversity_cfg.seed),
        )

    return {
        "panel_version": "wf394a2/v1",
        "n_input": n_input,
        "sa_qed": sa_qed_out,
        "sa_aware": sa_aware_out,
        "diversity": diversity_out,
    }


# ---------------------------------------------------------------------------
# Diversity pass — Tanimoto / scaffold-distribution post-hoc
# ---------------------------------------------------------------------------


def _compute_diversity_pass(
    smiles_list: List[str],
    sample_size: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute a Tanimoto / scaffold-distribution pass on the cohort.

    Implementation notes
    -------------------
    * Uses :func:`molmetal.adapters.flow_matching_lipman.scaffold_conditioning.murcko_scaffold_from_smiles`
      for scaffold extraction (Wf-380-A3 module).
    * Tanimoto computed via RDKit ``DataStructs.TanimotoSimilarity`` on
      Morgan fingerprints (radius=2, 2048 bits) — the standard
      GuacaMol / MolDQN diversity metric.
    * Subsampling is uniform-random WITHOUT replacement when
      n > sample_size; full pairwise Tanimoto on the subsample only
      (O(sample_size^2) bounded).

    Returns
    -------
    dict with::

        {
          "n_input": int,
          "n_parsed_mol": int,
          "n_unique_scaffold": int,
          "fraction_unique_scaffold": float,
          "tanimoto": {
            "n_pairs": int,
            "mean": float|None,
            "median": float|None,
            "p10": float|None,
            "p25": float|None,
            "p75": float|None,
            "p90": float|None,
            "min": float|None,
            "max": float|None,
          },
          "subsample_size": int,
          "seed": int,
        }
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    # Lazy import of the Wf-380-A3 scaffold helper
    try:
        from molmetal.adapters.flow_matching_lipman import scaffold_conditioning as _sc

        scaffold_fn = _sc.murcko_scaffold_from_smiles
    except Exception as exc:
        sys.stderr.write(f"[wf394a2] WARN scaffold_conditioning import failed: {exc}\n")
        scaffold_fn = None

    mols: List[Any] = []
    scaffolds: List[str] = []
    n_parsed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n_parsed += 1
        mols.append(mol)
        if scaffold_fn is not None:
            try:
                sc = scaffold_fn(smi) or ""
            except Exception:
                sc = ""
        else:
            try:
                from rdkit.Chem.Scaffolds import MurckoScaffold

                sc = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            except Exception:
                sc = ""
        scaffolds.append(sc)

    unique_sc = set(s for s in scaffolds if s)
    n_unique_scaffold = len(unique_sc)
    frac_unique = (n_unique_scaffold / n_parsed) if n_parsed > 0 else 0.0

    # Subsample for Tanimoto
    import random

    rng = random.Random(seed)
    if len(mols) > sample_size:
        idxs = rng.sample(range(len(mols)), sample_size)
        sub_mols = [mols[i] for i in idxs]
    else:
        sub_mols = mols

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in sub_mols
    ]

    pairs: List[float] = []
    n_pairs = 0
    n_sub = len(sub_mols)
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            try:
                t = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
                pairs.append(t)
                n_pairs += 1
            except Exception:  # noqa: BLE001
                # skip failed pairwise sim
                pass

    tanimoto_block: Dict[str, Any]
    if not pairs:
        tanimoto_block = {
            "n_pairs": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    else:
        srt = sorted(pairs)
        n_p = len(srt)

        def _q(q: float) -> float:
            if n_p == 1:
                return srt[0]
            pos = q * (n_p - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return srt[lo]
            frac = pos - lo
            return srt[lo] + (srt[hi] - srt[lo]) * frac

        mid = n_p // 2
        median = srt[mid] if n_p % 2 == 1 else 0.5 * (srt[mid - 1] + srt[mid])
        mean = sum(srt) / n_p
        var = sum((x - mean) ** 2 for x in srt) / n_p
        tanimoto_block = {
            "n_pairs": n_pairs,
            "mean": float(mean),
            "median": float(median),
            "p10": float(_q(0.10)),
            "p25": float(_q(0.25)),
            "p75": float(_q(0.75)),
            "p90": float(_q(0.90)),
            "min": float(srt[0]),
            "max": float(srt[-1]),
            "std": float(math.sqrt(var)),
        }

    return {
        "n_input": len(smiles_list),
        "n_parsed_mol": n_parsed,
        "n_unique_scaffold": n_unique_scaffold,
        "fraction_unique_scaffold": float(frac_unique),
        "tanimoto": tanimoto_block,
        "subsample_size": int(n_sub),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Aggregation over per-mol results
# ---------------------------------------------------------------------------


def aggregate_panel_metrics(
    per_mol_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-property metrics from a list of per-mol results.

    Parameters
    ----------
    per_mol_results : Sequence[Dict[str, Any]]
        Each entry has the Wf-264-SA per-mol shape::

            {"smiles": str, "sa": float|None, "qed": float|None}

        Plus optional ``penalty`` (Wf-380-A6) and ``scaffold`` (Wf-380-A3).

    Returns
    -------
    dict with aggregate per-property stats::

        {
          "sa": {"mean":..., "median":..., "std":..., "min":..., "max":...,
                 "n_parsed": int, "n_failed": int},
          "qed": {...},
          "penalty": {...} | None,
          "parsed_ratio": float,
          "n_input": int,
        }
    """
    n_input = len(per_mol_results)
    sa_vals: List[float] = []
    qed_vals: List[float] = []
    pen_vals: List[float] = []
    n_sa = n_qed = n_pen = 0
    for r in per_mol_results:
        sa = r.get("sa")
        qed = r.get("qed")
        pen = r.get("penalty")
        if sa is not None and not (isinstance(sa, float) and math.isnan(sa)):
            sa_vals.append(float(sa))
            n_sa += 1
        if qed is not None and not (isinstance(qed, float) and math.isnan(qed)):
            qed_vals.append(float(qed))
            n_qed += 1
        if pen is not None and not (isinstance(pen, float) and math.isnan(pen)):
            pen_vals.append(float(pen))
            n_pen += 1

    def _agg(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "n_parsed": 0,
                "n_failed": n_input,
            }
        srt = sorted(vals)
        n_v = len(srt)
        mid = n_v // 2
        median = srt[mid] if n_v % 2 == 1 else 0.5 * (srt[mid - 1] + srt[mid])
        mean = sum(srt) / n_v
        var = sum((x - mean) ** 2 for x in srt) / n_v
        return {
            "mean": float(mean),
            "median": float(median),
            "std": float(math.sqrt(var)),
            "min": float(srt[0]),
            "max": float(srt[-1]),
            "n_parsed": n_v,
            "n_failed": n_input - n_v,
        }

    out: Dict[str, Any] = {
        "sa": _agg(sa_vals),
        "qed": _agg(qed_vals),
        "parsed_ratio": float((n_sa + n_qed) / (2 * n_input)) if n_input > 0 else 0.0,
        "n_input": n_input,
    }
    if pen_vals:
        out["penalty"] = _agg(pen_vals)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wf394a2_property_panel_runner",
        description=(
            "Wf-394-A2 — Diversity + SA-aware post-hoc property panel "
            "runner for CFM / metallodrug generated SMILES.  CPU-only. "
            "Delegates SA + QED to Wf-264-SA; SA-aware penalty to "
            "Wf-380-A6; diversity pass to Wf-380-A3 / Wf-388-A2."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles-csv", type=str, help="CSV with a 'smiles' column.")
    src.add_argument(
        "--smiles-literal", type=str, help="Comma-separated SMILES list (smoke mode)."
    )
    p.add_argument("--out-json", type=str, required=True, help="Output JSON path.")
    # Diversity flags (Wf-388-A2 post-hoc analogue)
    p.add_argument(
        "--diversity-aware-eval",
        action="store_true",
        help="If set, run Tanimoto / scaffold diversity pass.",
    )
    p.add_argument(
        "--diversity-sample-size",
        type=int,
        default=1000,
        help="Max pairwise Tanimoto sample size (default 1000).",
    )
    p.add_argument(
        "--diversity-seed",
        type=int,
        default=42,
        help="RNG seed for diversity subsample (default 42).",
    )
    # SA-aware flags (Wf-388-A3 post-hoc analogue)
    p.add_argument(
        "--sa-aware",
        action="store_true",
        help="If set, compute Wf-380-A6 SA-aware sampling penalty.",
    )
    p.add_argument(
        "--sa-target",
        type=float,
        default=2.8,
        help="SA target threshold (default 2.8).",
    )
    p.add_argument(
        "--sa-weight",
        type=float,
        default=0.3,
        help="Linear SA penalty weight (default 0.3).",
    )
    return p


def _parse_smiles_literal(literal: str) -> List[str]:
    return [s.strip() for s in literal.split(",") if s.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.smiles_csv:
        smiles_list = _read_smiles_csv(Path(args.smiles_csv))
    else:
        smiles_list = _parse_smiles_literal(args.smiles_literal)

    diversity_cfg = DiversityCfg(
        enabled=bool(args.diversity_aware_eval),
        sample_size=int(args.diversity_sample_size),
        seed=int(args.diversity_seed),
    )
    sa_cfg = (
        SaCfg(
            target_sa=float(args.sa_target),
            sa_weight=float(args.sa_weight),
        )
        if args.sa_aware
        else None
    )

    panel = run_property_panel(
        cohort_smiles=smiles_list,
        diversity_cfg=diversity_cfg,
        sa_cfg=sa_cfg,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(
            panel,
            fh,
            indent=2,
            default=lambda o: float(o) if isinstance(o, float) else str(o),
        )

    sa_qed = panel.get("sa_qed", {})
    sa_aware = panel.get("sa_aware")
    diversity = panel.get("diversity")
    sa_mean = sa_qed.get("aggregate", {}).get("sa", {}).get("mean") if sa_qed else None
    sa_mean_str = f"{sa_mean:.4f}" if sa_mean is not None else "n/a"
    sys.stdout.write(
        f"[wf394a2] wrote {out_path} (n_input={panel['n_input']} "
        f"sa_mean={sa_mean_str} "
        f"sa_aware={'yes' if sa_aware else 'no'} "
        f"diversity={'yes' if diversity else 'no'})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
