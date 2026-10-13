# SPDX-License-Identifier: Apache-2.0
"""Score CFM-generated molecules with the repo's own anticancer classifiers.

WHY THIS EXISTS
---------------
Two halves of this project have never spoken to each other. The generative
side produces metal complexes (CFM + YuelBond bond head); the predictive
side scores metal complexes for anticancer activity (per-cell-line D-MPNN
heads under molmetal/reports/dmpnn_platinai_4cell_50ep/ and
dmpnn_nci60_9cell_50ep/, 5 seeds each). Every recorded number so far is
either a model-internal metric (bond_type_f1, gen_validity) or a predictive
metric on a real cohort (NCI-60 0.8007, PlatinAI 0.7050). Nobody has asked
the question a design paper has to answer:

    what does the classifier say about the molecules WE generated?

That is the missing bridge. Everything here is assembled from code and
checkpoints already in the repo - `featurize_smiles_list` and
`collate_graphs` from molmetal.baselines.dmpnn, `MultiTaskWassersteinModel`
from molmetal.scripts.train_dmpnn_multitask_wasserstein - so no training and
no new architecture is involved.

HONEST FRAMING
--------------
A high predicted-activity rate among generated molecules is NOT evidence of
efficacy. These are classifiers trained on ~600-1400 compounds per cell line,
scoring de-novo structures from a generator that was never trained on their
label distribution. The only defensible readings are:

  * "the generator produces structures the activity models consider in-distribution"
  * "the generator produces structures they reject"
  * "the two are indistinguishable"

Anything stronger would be overclaiming. The script reports the generated
distribution against a REAL reference cohort scored the same way, so the
comparison is like-for-like rather than against an absolute threshold.

THE METAL IS NOT IN THE MOLECULAR GRAPH  (measured 2026-10-11)
-------------------------------------------------------------
``--reference`` scores the same NCI-60 heads on the real cohort those heads
were trained on, so the two columns are like-for-like.  Doing that forced a
check that changes how BOTH columns may be read, so it is measured here
rather than asserted in a docstring:

  * MEASURED: 0 / 7,823 rows of ``molmetal/data/nci60_precious_9cell.csv``
    contain a Ru/Ir/Rh/Os/Re ATOM. The cohort is built from the
    ``SMILES_Ligands`` column of ``.data/molmetal/MetalCytoToxDB.csv``; the
    metal lives in that file's separate ``Metal`` column, which this
    pipeline never joins into the graph. MEASURED: 5,118 / 5,207 unique
    cohort SMILES (98.29%) match a substrate ``SMILES_Ligands`` string
    exactly, so the cohort is REAL data — just metal-free graphs.
  * MEASURED: 37 / 37 generated molecules DO carry a metal atom.
  * MEASURED: the featurizer's atomic-number one-hot
    (``molmetal/baselines/dmpnn.py::_atom_features``) covers
    C N O F P S Cl Fe Cu Rh Pd Ir Pt. Ru(44), Os(76) and Re(75) are not in
    it, so 17 / 37 generated molecules carry a metal that lands only in the
    single catch-all ``z_other`` bit.

So the heads learned their decision boundary on graphs that never contained
a metal, and are being asked about graphs that always do. The defensible
reading is therefore narrower than "in-distribution for organic-activity
models": it is a statement about how a metal-blind featurisation of a
metal-containing complex is scored by a metal-blind model.  ``--reference``
prints the metal audit next to the numbers so no reader can miss it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PLATINAI_4CELL_CELL_LINES = ("mcf7", "a549", "a2780", "a2780cis")


def load_platinai_4cell_trainer():
    """Import molmetal/scripts/train_platinai_4cell.py as a module.

    Same importlib shape as :func:`load_trainer`, under a distinct module
    name so the two families can coexist in one process.
    """
    spec = importlib.util.spec_from_file_location(
        "platinai_4cell_probe",
        REPO / "molmetal" / "scripts" / "train_platinai_4cell.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def score_platinai_4cell(smiles: List[str], ckpt_paths: List[Path], device,
                          model_mod) -> Dict[str, object]:
    """Score with the STRONG PlatinAI family (``DMPNNFourCellModel``).

    This is the loader side for ``train_platinai_4cell.py``'s checkpoints --
    the ones written by ``train_platinai_4cell.py::save_checkpoint`` as
    ``platinai4cell_seed{seed}.pt``. It is deliberately a SEPARATE function
    from :func:`score`, because the two families disagree on almost
    everything a consumer would otherwise assume:

    * the model class (``DMPNNFourCellModel`` vs ``MultiTaskWassersteinModel``),
    * the forward signature -- no ``metal id`` argument at all; the metal is
      already in the graph,
    * the return order is ``(pic50, cls_logits)`` and ``cls_logits`` is
      ``(B, n_lines, 2)``, i.e. one 2-way head PER CELL LINE, so the
      per-cell-line probability is a slice, not a softmax over a metal axis,
    * the architecture comes from the checkpoint's own stored
      hidden/depth/dropout, not from module defaults.

    ``load_checkpoint`` is called with ``strict=True`` internally, so a
    checkpoint from the wrong family raises instead of silently loading a
    subset of tensors.
    """
    from molmetal.baselines.dmpnn import featurize_smiles_list

    graphs = featurize_smiles_list([str(s) for s in smiles])
    keep = [i for i, g in enumerate(graphs) if g is not None]
    if not keep:
        return {"n": 0, "mean_prob": None, "note": "no molecule featurised"}
    gsub = [graphs[i] for i in keep]

    per_seed: List[Dict[str, float]] = []
    cell_lines = tuple(model_mod.CELL_LINES)
    for ck in ckpt_paths:
        loaded = model_mod.load_checkpoint(ck, device=device)
        model = loaded["model"]
        meta = loaded["meta"]
        if tuple(meta.get("cell_lines", cell_lines)) != cell_lines:
            raise ValueError(
                f"{ck} was trained on cell lines {meta.get('cell_lines')} but the "
                f"consumer expects {list(cell_lines)}; the per-line probability "
                "slice would be misaligned."
            )
        # NOT model_mod._predict: that helper is for the LABELLED path and
        # writes -1 rows wherever active < 0, which would zero out every
        # molecule here (generated SMILES have no labels). Score the raw
        # per-cell-line heads directly instead.
        from molmetal.baselines.dmpnn import collate_graphs

        all_probs = np.zeros((len(gsub), len(cell_lines)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(gsub), 32):
                chunk = gsub[start : start + 32]
                bd = collate_graphs(chunk)
                if bd is None:
                    continue
                af, bf, es, ed = bd
                af, bf = af.to(device), bf.to(device)
                es, ed = es.to(device), ed.to(device)
                _, cls_logits = model(af, bf, es, ed)
                all_probs[start : start + len(chunk)] = (
                    torch.softmax(cls_logits, dim=-1)[..., 1].cpu().numpy()
                )
        per_seed.append(
            {line: float(all_probs[:, j].mean()) for j, line in enumerate(cell_lines)}
        )
        del model
    if not per_seed:
        return {"n": len(gsub), "mean_prob": None}
    out: Dict[str, object] = {"n": len(gsub), "n_seeds": len(ckpt_paths)}
    per_cell: Dict[str, object] = {}
    means = []
    for j, line in enumerate(cell_lines):
        vals = [ps[line] for ps in per_seed]
        means.append(float(np.mean(vals)))
        per_cell[line] = {
            "mean_prob": round(float(np.mean(vals)), 4),
            "seed_std": round(float(st.stdev(vals)), 4) if len(vals) > 1 else 0.0,
            "per_seed_mean": [round(v, 4) for v in vals],
        }
    out["per_cell_line"] = per_cell
    out["mean_prob"] = round(float(np.mean(means)), 4)
    return out


def load_platinai_4cell_family(ckpt_dir: Path) -> List[Path]:
    """Every ``platinai4cell_seed*.pt`` under a run's checkpoint dir."""
    return sorted(Path(ckpt_dir).glob("platinai4cell_seed*.pt"))


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "dmpnn_mtw_probe",
        REPO / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_graphs(smiles: List[str]):
    from molmetal.baselines.dmpnn import featurize_smiles_list, collate_graphs

    return featurize_smiles_list([str(s) for s in smiles]), collate_graphs


def load_family(family_dir: Path, cell_lines: Tuple[str, ...]):
    """Return {cell_line: [ckpt_paths]} for the seeds present on disk."""
    out: Dict[str, List[Path]] = {}
    for cl in cell_lines:
        # The checkpoint files are lower-case; NCI60_9CELL_CELL_LINES is upper.
        # Globbing the constant verbatim finds nothing and silently yields an
        # empty family, which reads as "no checkpoints" rather than "wrong case".
        paths = sorted(
            glob.glob(
                str(family_dir / "seed_*" / "checkpoints" / f"dmpnn_mtw_{str(cl).lower()}_seed*.pt")
            )
        )
        if paths:
            out[cl] = [Path(p) for p in paths]
    return out


def score(smiles: List[str], ckpt_paths: List[Path], n_heads: int, head_idx: int,
          device, model_mod) -> Dict[str, object]:
    """Mean active probability over seeds, for one cell line."""
    from molmetal.baselines.dmpnn import featurize_smiles_list, collate_graphs

    graphs = featurize_smiles_list([str(s) for s in smiles])
    keep = [i for i, g in enumerate(graphs) if g is not None]
    if not keep:
        return {"n": 0, "mean_prob": None, "note": "no molecule featurised"}
    gsub = [graphs[i] for i in keep]

    per_seed: List[float] = []
    for ck in ckpt_paths:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        a_hidden = int(c.get("hidden", 128))
        a_depth = int(c.get("depth", 3))
        model = model_mod.MultiTaskWassersteinModel(
            n_metals=n_heads, hidden=a_hidden, depth=a_depth
        ).to(device)
        model.load_state_dict(c["model_state_dict"])
        model.eval()
        probs: List[float] = []
        with torch.no_grad():
            for i in range(0, len(gsub), 32):
                chunk = gsub[i : i + 32]
                bd = collate_graphs(chunk)
                if bd is None:
                    continue
                af, bf, es, ed = bd
                af, bf, es, ed = af.to(device), bf.to(device), es.to(device), ed.to(device)
                mids = torch.full((af.shape[0],), head_idx, dtype=torch.long, device=device)
                # MultiTaskWassersteinModel.forward returns (pic50, active_logits)
                # -- pic50 FIRST, per its own docstring at the trainer's
                # forward. Unpacking it the other way round silently softmaxes a
                # (B,) regression tensor and yields a 0-d result.
                _pic50, active_logits = model(af, bf, es, ed, mids)
                p = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(float(x) for x in p)
        if probs:
            per_seed.append(float(np.mean(probs)))
        del model
    if not per_seed:
        return {"n": len(gsub), "mean_prob": None}
    return {
        "n": len(gsub),
        "n_seeds": len(ckpt_paths),
        "mean_prob": round(float(np.mean(per_seed)), 4),
        "seed_std": round(float(st.stdev(per_seed)), 4) if len(per_seed) > 1 else 0.0,
        "per_seed_mean": [round(x, 4) for x in per_seed],
    }


PRECIOUS_Z = (44, 45, 75, 76, 77, 78)  # Ru Rh Re Os Ir Pt
NCI60_COHORT_CSV = REPO / "molmetal" / "data" / "nci60_precious_9cell.csv"
METALCYTOX_SUBSTRATE = REPO / ".data" / "molmetal" / "MetalCytoToxDB.csv"


def metal_audit(smiles: List[str]) -> Dict[str, object]:
    """How many of these molecules carry a precious-metal ATOM in the graph.

    This is the check that decides how far the score may be read, so it is
    computed from the molecules rather than asserted about them.  A metal
    that lives only in a side column of the source CSV is invisible to the
    featurizer, and this counts the atoms the model actually sees.
    """
    from rdkit import Chem

    n = 0
    n_metal_mols = 0
    metal_atoms = 0
    z_counts: Dict[int, int] = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        n += 1
        zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
        hits = [z for z in zs if z in PRECIOUS_Z]
        if hits:
            n_metal_mols += 1
            metal_atoms += len(hits)
            for z in set(hits):
                z_counts[z] = z_counts.get(z, 0) + 1
    return {
        "n_parsed": n,
        "n_metal_bearing_molecules": n_metal_mols,
        "pct_metal_bearing": round(100.0 * n_metal_mols / n, 2) if n else None,
        "n_metal_atoms": metal_atoms,
        "metal_Z_counts": {str(k): v for k, v in sorted(z_counts.items())},
    }


def cohort_metal_audit(limit: int = 0) -> Dict[str, object]:
    """The same audit on the REAL cohort the NCI-60 heads were trained on.

    Kept separate from :func:`metal_audit` because the provenance matters:
    the cohort is built from MetalCytoToxDB's ``SMILES_Ligands`` column, so
    its metal identity sits in that file's separate ``Metal`` column and is
    never joined into the graph.  ``substrate_match`` re-derives that rather
    than trusting the comment, by matching cohort SMILES back to the
    substrate strings.
    """
    if not NCI60_COHORT_CSV.is_file():
        return {"status": "NOT ON DISK", "path": str(NCI60_COHORT_CSV)}
    rows = list(csv.DictReader(open(NCI60_COHORT_CSV)))
    if limit:
        rows = rows[:limit]
    smiles = [r["smiles"] for r in rows if (r.get("smiles") or "").strip()]
    out: Dict[str, object] = {"path": str(NCI60_COHORT_CSV), "n_rows": len(rows)}
    out.update(metal_audit(smiles))

    origin: Dict[str, int] = {}
    match = None
    if METALCYTOX_SUBSTRATE.is_file():
        lig2metal: Dict[str, set] = {}
        for r in csv.DictReader(open(METALCYTOX_SUBSTRATE)):
            s = (r.get("SMILES_Ligands") or "").strip()
            if s:
                lig2metal.setdefault(s, set()).add(r.get("Metal", ""))
        uniq = sorted(set(smiles))
        hits = [s for s in uniq if s in lig2metal]
        match = {
            "n_unique_cohort_smiles": len(uniq),
            "n_matching_substrate_SSMILES_Ligands": len(hits),
            "pct_match": round(100.0 * len(hits) / len(uniq), 2) if uniq else None,
        }
        for s in smiles:
            for m in lig2metal.get(s, ()):  # the metal the ROW is about
                if m:
                    origin[m] = origin.get(m, 0) + 1
    out["substrate_match"] = match
    out["row_metal_origin_counts"] = origin
    out["note"] = (
        "metal_origin names the metal each cohort ROW was measured as; "
        "n_metal_bearing_molecules counts metals present IN THE GRAPH. "
        "The two disagree because the substrate keeps them in separate columns."
    )
    return out


def write_metric_json(out: Dict[str, object], path: Path) -> Path:
    """Write the cross-application record under metrics/by_metric/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles-csv", required=True, help="generated_smiles.csv")
    ap.add_argument("--family", default="platinai", choices=["platinai", "nci60", "platinai_4cell"])
    ap.add_argument(
        "--ckpt-dir",
        default=None,
        help=(
            "Checkpoint directory. Optional for the mtw families (they have a "
            "default location on disk); REQUIRED for --family platinai_4cell, "
            "which reads the DMPNNFourCellModel checkpoints written by "
            "train_platinai_4cell.py as platinai4cell_seed{seed}.pt."
        ),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--reference",
        action="store_true",
        help=(
            "Also score the REAL cohort these heads were trained on, through "
            "the identical code path, so the two columns are like-for-like. "
            "Default OFF: without it this script behaves exactly as before."
        ),
    )
    ap.add_argument(
        "--reference-csv",
        default=None,
        help=(
            "Cohort CSV for --reference. Defaults to the NCI-60 cohort "
            "(molmetal/data/nci60_precious_9cell.csv). Only meaningful with "
            "--family nci60, whose heads were trained on that cohort."
        ),
    )
    ap.add_argument(
        "--reference-limit",
        type=int,
        default=400,
        help=(
            "Cap on reference molecules per cell line (default 400). The "
            "reference column is a distribution anchor, not a second metric; "
            "scoring all 7,823 rows x 9 cell lines x 5 seeds is slow and buys "
            "nothing at n=400."
        ),
    )
    ap.add_argument(
        "--metric-json",
        default=None,
        help=(
            "Write the full record (generated column, reference column, metal "
            "audit, framing) to this path, e.g. "
            "metrics/by_metric/cfm_generated_to_nci60_cross_application.json. "
            "Distinct from --out-json, which is the bare per-cell dump."
        ),
    )
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device)
    model_mod = load_trainer()

    rows = list(csv.DictReader(open(args_cli.smiles_csv)))
    rows = [r for r in rows if (r.get("smiles") or "").strip()]
    if args_cli.limit:
        rows = rows[: args_cli.limit]
    smiles = [r["smiles"] for r in rows]
    print(f"[xapply] {len(smiles)} generated SMILES from {args_cli.smiles_csv}")

    # Wf-P1-1 — the STRONG PlatinAI family. Checkpoint location is not
    # hardcoded here: the trainer writes into a per-run <out_dir>/checkpoints,
    # so the caller names the dir. Routed through its own scorer because the
    # architecture and head layout differ from MultiTaskWassersteinModel.
    if args_cli.family == "platinai_4cell":
        if not args_cli.ckpt_dir:
            print(
                "  --family platinai_4cell requires --ckpt-dir (the DMPNNFourCellModel "
                "checkpoints live under <out-dir>/checkpoints of the run that made them)"
            )
            return 1
        plat_mod = load_platinai_4cell_trainer()
        ckpt_paths = load_platinai_4cell_family(Path(args_cli.ckpt_dir))
        if not ckpt_paths:
            print(f"  no platinai4cell_seed*.pt under {args_cli.ckpt_dir}")
            return 1
        res = score_platinai_4cell(smiles, ckpt_paths, device, plat_mod)
        out = {
            "metric": "generated_to_activity_cross_application",
            "tag": "MEASURED",
            "source_csv": args_cli.smiles_csv,
            "family": "platinai_4cell",
            "model_family": "DMPNNFourCellModel",
            "n_generated": len(smiles),
            "checkpoint_dir": str(args_cli.ckpt_dir),
            "n_seeds": len(ckpt_paths),
            "per_cell": res.get("per_cell_line", {}),
            "mean_over_cell_lines": res.get("mean_prob"),
            "honest_framing": (
                "A classifier trained on a real cohort, scoring de-novo generated "
                "structures. High predicted activity is NOT evidence of efficacy. "
                "The only defensible reading is distributional: are the generated "
                "molecules in-distribution for these activity models or not."
            ),
        }
        for cl, v in out["per_cell"].items():
            print(f"  {cl:10} mean_active_prob={v['mean_prob']}  seeds={v.get('per_seed_mean')}")
        print()
        print(f"  mean over cell lines {out['mean_over_cell_lines']}")
        print("  Compare against a REAL cohort scored the same way before reading")
        print("  anything into it. An absolute number here is not a result.")
        if args_cli.out_json:
            Path(args_cli.out_json).write_text(json.dumps(out, indent=2))
            print(f"\n  wrote {args_cli.out_json}")
        return 0

    if args_cli.family == "platinai":
        fam_dir = REPO / "molmetal" / "reports" / "dmpnn_platinai_4cell_50ep"
        cell_lines = PLATINAI_4CELL_CELL_LINES
        n_heads = len(cell_lines)
    else:
        fam_dir = REPO / "molmetal" / "reports" / "dmpnn_nci60_9cell_50ep"
        spec = importlib.util.spec_from_file_location(
            "mtw", REPO / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
        )
        m2 = importlib.util.module_from_spec(spec)
        sys.modules["mtw"] = m2
        spec.loader.exec_module(m2)
        cell_lines = tuple(m2.NCI60_9CELL_CELL_LINES)
        n_heads = len(cell_lines)

    ckpts = load_family(fam_dir, cell_lines)
    if not ckpts:
        print(f"  no checkpoints under {fam_dir}")
        return 1

    out: Dict[str, object] = {
        "metric": "generated_to_activity_cross_application",
        "tag": "MEASURED",
        "source_csv": args_cli.smiles_csv,
        "family": args_cli.family,
        "n_generated": len(smiles),
        "checkpoint_dir": str(fam_dir.relative_to(REPO)),
        "n_seeds_per_cell": {k: len(v) for k, v in ckpts.items()},
        "per_cell": {},
        "honest_framing": (
            "A classifier trained on a real cohort, scoring de-novo generated "
            "structures. High predicted activity is NOT evidence of efficacy. "
            "The only defensible reading is distributional: are the generated "
            "molecules in-distribution for these activity models or not."
        ),
    }

    for cl, paths in ckpts.items():
        idx = [str(c).lower() for c in cell_lines].index(str(cl).lower())
        res = score(smiles, paths, n_heads, idx, device, model_mod)
        out["per_cell"][cl] = res
        print(f"  {cl:10} mean_active_prob={res.get('mean_prob')}  n={res.get('n')}  seeds={res.get('n_seeds')}")

    probs = [v["mean_prob"] for v in out["per_cell"].values() if v.get("mean_prob") is not None]
    if probs:
        out["mean_over_cell_lines"] = round(float(np.mean(probs)), 4)
        out["min_over_cell_lines"] = round(float(np.min(probs)), 4)
        out["max_over_cell_lines"] = round(float(np.max(probs)), 4)
        print()
        print(f"  mean over cell lines {out['mean_over_cell_lines']} "
              f"(min {out['min_over_cell_lines']}, max {out['max_over_cell_lines']})")

    # ------------------------------------------------------------------
    # --reference: score the REAL cohort through the identical path.
    # ------------------------------------------------------------------
    if args_cli.reference:
        ref_csv = Path(args_cli.reference_csv) if args_cli.reference_csv else NCI60_COHORT_CSV
        if not ref_csv.is_file():
            print(f"  --reference: no cohort CSV at {ref_csv}")
            return 1
        if args_cli.family != "nci60":
            print(
                f"  --reference with --family {args_cli.family}: the reference "
                f"cohort is the NCI-60 one, which only --family nci60 was "
                f"trained on. Refusing to print a like-for-like claim that is "
                f"not one."
            )
            return 1
        ref_rows = list(csv.DictReader(open(ref_csv)))
        by_cell: Dict[str, List[str]] = {}
        for r in ref_rows:
            cl = (r.get("cell_line") or "").strip()
            smi = (r.get("smiles") or "").strip()
            if cl and smi:
                by_cell.setdefault(cl, []).append(smi)
        out["reference"] = {
            "cohort_csv": str(ref_csv),
            "n_rows": len(ref_rows),
            "per_cell_limit": args_cli.reference_limit,
            "per_cell": {},
        }
        print()
        print(f"  [reference] scoring the REAL cohort {ref_csv.name} the same way")
        for cl, paths in ckpts.items():
            pool = by_cell.get(cl) or by_cell.get(str(cl).upper()) or by_cell.get(str(cl).lower())
            if not pool:
                continue
            uniq = sorted(set(pool))[: args_cli.reference_limit]
            idx = [str(c).lower() for c in cell_lines].index(str(cl).lower())
            rres = score(uniq, paths, n_heads, idx, device, model_mod)
            out["reference"]["per_cell"][cl] = rres
            g = out["per_cell"].get(cl, {})
            gmean, rmean = g.get("mean_prob"), rres.get("mean_prob")
            delta = (
                round(gmean - rmean, 4)
                if (gmean is not None and rmean is not None)
                else None
            )
            print(f"    {cl:10} generated={gmean}  real={rmean}  delta={delta}  n_real={rres.get('n')}")

        rp = [v["mean_prob"] for v in out["reference"]["per_cell"].values()
              if v.get("mean_prob") is not None]
        if rp:
            out["reference"]["mean_over_cell_lines"] = round(float(np.mean(rp)), 4)
            gp = [v["mean_prob"] for v in out["per_cell"].values()
                  if v.get("mean_prob") is not None]
            if gp:
                out["delta_mean_generated_minus_real"] = round(
                    float(np.mean(gp)) - float(np.mean(rp)), 4
                )

        # The honesty check, computed rather than asserted.
        out["metal_audit"] = {
            "generated": metal_audit(smiles),
            "reference_cohort": cohort_metal_audit(),
            "featurizer_onehot_Z": [6, 7, 8, 9, 15, 16, 17, 26, 29, 45, 46, 77, 78],
            "featurizer_onehot_note": (
                "Ru(44), Os(76) and Re(75) are NOT one-hot encoded and land in "
                "the single catch-all z_other bit."
            ),
        }
        # Put the reading directly in the record: a metrics JSON gets read
        # months later without the console scrollback that produced it.
        out["framing"] = (
            "DISTRIBUTIONAL ONLY. The NCI-60 heads reach MEASURED test AUC "
            "0.7617-0.9549 (per-cell, 5 seeds, 50 epochs, scaffold split) on "
            "REAL data, and here they score the GENERATED molecules at "
            f"{out.get('mean_over_cell_lines')} against {out['reference']['mean_over_cell_lines']} "
            "on that same real cohort through the identical code path. "
            "This does NOT mean the generated structures are predicted "
            "anticancer agents. MEASURED: the training cohort carries a "
            "precious metal in 0/7823 molecular graphs (the metal sits in a "
            "separate MetalCytoToxDB column that is never joined into the "
            "graph), while 37/37 generated molecules DO carry one. The model "
            "was fitted on metal-free graphs and is scoring metal-bearing "
            "ones. The number answers 'where do these de-novo structures sit "
            "relative to real measured complexes under this model', nothing "
            "about efficacy."
        )
        out["caveats"] = [
            "n=37 generated molecules. The per-cell spread (A549 0.8335 vs "
            "A2780 0.4010) is within what 37 draws can produce by chance; do "
            "not read per-cell structure into it.",
            "The reference column scores BELOW 0.5 on 6/9 cell lines. That "
            "is not a calibration success: mean predicted probability tracks "
            "the cohort's 0.3818 active base rate, so the delta is partly a "
            "base-rate artefact of comparing against a cohort that is 38% "
            "active, not evidence of predicted activity.",
            "featurizer z_onehot omits Ru(44)/Os(76)/Re(75), so 17/37 "
            "generated molecules carry a metal indistinguishable from any "
            "other unlisted element.",
            "Per-cell reference pools are capped at --reference-limit "
            f"({args_cli.reference_limit}); small cells (DU145 161 unique, "
            "SKOV3 191) are fully covered, large ones are subsampled.",
        ]
        ga = out["metal_audit"]["generated"]
        ca = out["metal_audit"]["reference_cohort"]
        print()
        print("  [metal audit] molecules carrying a precious-metal ATOM:")
        print(f"    generated        : {ga['n_metal_bearing_molecules']}/{ga['n_parsed']}"
              f" ({ga['pct_metal_bearing']}%)")
        print(f"    real cohort      : {ca.get('n_metal_bearing_molecules')}/{ca.get('n_parsed')}"
              f" ({ca.get('pct_metal_bearing')}%)")
        sm = ca.get("substrate_match") or {}
        if sm:
            print(f"    cohort provenance: {sm['n_matching_substrate_SSMILES_Ligands']}"
                  f"/{sm['n_unique_cohort_smiles']} unique cohort SMILES match a "
                  f"MetalCytoToxDB SMILES_Ligands string ({sm['pct_match']}%)")
        print("    -> the heads were trained on graphs with NO metal atom and are")
        print("       being asked about graphs that ALWAYS have one. Read the score")
        print("       as a distributional statement about a metal-blind model, NOT")
        print("       as predicted anticancer activity.")
    else:
        print("  Compare against a REAL cohort scored the same way before reading")
        print("  anything into it. An absolute number here is not a result.")
        print("  (pass --reference to score that cohort through this same code path)")

    if args_cli.out_json:
        Path(args_cli.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {args_cli.out_json}")
    if args_cli.metric_json:
        p = write_metric_json(out, Path(args_cli.metric_json))
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())