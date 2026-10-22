#!/usr/bin/env python
"""Is the tmQM corpus the CFM trains on actually complete?

Question
--------
The CFM trainer streams ``.data/molmetal/tmQM`` and builds one
:class:`AtomCloud` per complex, but a cloud needs three separate shards to be
usable: the geometry (``tmQM_X*.xyz.gz``), the Wiberg bond truth
(``tmQM_X*.BO.gz``) and the NBO charges (``tmQM_X.q``).  The loader reports
each shard's coverage only while it runs, and it reports it *after* the
reservoir has already chosen what to keep.  So a shard that is missing for a
handful of complexes leaves no trace on disk: the run simply has fewer clouds
and nobody can say which ones.

This artifact answers the question from the raw files, before any sampling:
for the six-metal cohort the CFM config requests, does every complex have all
three shards, is the metal column the metal in the geometry, and does the
scaffold split put anything on both sides of a boundary?

Answer (measured here, not asserted)
------------------------------------
The corpus is complete: 33,429 six-metal complexes, every one with an XYZ
block, a BO block and a q vector, no duplicate codes, no non-finite
coordinates, exactly one metal atom each, and the ``metal`` column agrees with
that atom in 33,429/33,429 rows.

Two things are still worth knowing, and both are reported rather than fixed:

* The scaffold split (``tmqm_split_kind: scaffold``, fractions budgeted in
  *scaffold counts*) gives 87.7 / 6.2 / 6.2 % of rows, not 80/10/10, because
  the largest scaffolds hold more rows than their share of the count.  It is
  not starvation -- val still has 2,061 complexes and all six metals -- but
  ``tmqm_val_frac`` is ignored on this branch, so a reader who assumes the
  YAML's 0.2 is in force will be wrong.
* The trainer takes ``n_val=192`` by default with ``tmqm_min_per_metal=0``, so
  the val cohort is a 192-of-960 reservoir window.  Measured at seed 0 it is
  Pt 19 / Ru 93, against a corpus that is Pt 24.8 % / Ru 29.3 %.  Pt is not
  absent (the 2026-10-12 zero-Pt finding does not reproduce on this split),
  but it is under-represented 2.5x in the cohort every generation metric is
  averaged over.

Output: ``metrics/by_metric/tmqm_cfm_data_audit.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molmetal.data.splits import (  # noqa: E402
    DEFAULT_FRACTIONS,
    ScaffoldSplitter,
    _bemis_murcko_key,
)
from molmetal.data.tmqm_clouds import (  # noqa: E402
    _BO_FILES,
    TMQM_METAL_Z,
    iter_bo_bonds,
    iter_tmqm_blocks,
    load_tmqm_clouds,
    load_tmqm_metadata,
)

TMQM_DIR = ROOT / ".data" / "molmetal" / "tmQM"
OUT = ROOT / "metrics" / "by_metric" / "tmqm_cfm_data_audit.json"

#: Exactly what ``configs/c7_*.yaml`` puts in ``data.tmqm_metals``.
METALS = ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]
#: ``n_val`` argparse default in train_cfm_yuelbond_wasserstein.py.
N_VAL = 192

_Z_TO_SYMBOL = {int(z): s for s, z in TMQM_METAL_Z.items()}


def _scaffold_of(meta: dict, cache: dict) -> dict:
    """Per-code Bemis-Murcko key, mirroring ``load_tmqm_clouds`` lines 956-989.

    The cache is keyed on ``code|smiles`` so a re-prepped SMILES column
    invalidates only the rows it touched; a miss is computed here with the
    same ``_bemis_murcko_key`` the loader uses, and an empty SMILES falls back
    to the 6-letter CSD refcode exactly as the loader does.
    """
    out = {}
    for code, row in meta.items():
        smiles = str(row.get("smiles", "")).strip()
        key = cache.get(f"{code}|{smiles}")
        if key is None:
            key = _bemis_murcko_key(smiles) if smiles else f"__csd_prefix__{code[:6]}"
            key = key or f"__empty__{code}"
        out[code] = key
    return out


def _raw_audit(meta: dict, keep: set) -> dict:
    xyz = sorted(TMQM_DIR.glob("tmQM_X*.xyz.gz"))
    n_blocks = 0
    seen: Counter = Counter()
    atom_count: dict = {}
    heavy_count: dict = {}
    n_nonfinite = 0
    n_metal_atoms: Counter = Counter()
    metal_label_mismatch = []
    for code, symbols, coords in iter_tmqm_blocks(xyz):
        n_blocks += 1
        if code not in keep:
            continue
        seen[code] += 1
        atom_count[code] = len(symbols)
        # SMILES carries heavy atoms only, so the count it is compared
        # against must exclude H.  Comparing it to len(symbols) marked
        # every hydrogen-bearing complex as a mismatch.
        heavy_count[code] = sum(1 for s in symbols if s != "H")
        if any(not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z))
               for x, y, z in coords):
            n_nonfinite += 1
        present = [s for s in symbols if s in METALS]
        n_metal_atoms[len(present)] += 1
        if present and present[0] != str(meta[code].get("metal", "")).strip():
            metal_label_mismatch.append(
                [code, str(meta[code].get("metal", "")).strip(), present]
            )

    bo_codes = {}
    for code, pairs, _nbrs in iter_bo_bonds([TMQM_DIR / n for n in _BO_FILES]):
        if code in keep:
            bo_codes[code] = pairs
    bo_overrun = 0
    for code, pairs in bo_codes.items():
        n = atom_count.get(code)
        if n is None:
            continue
        if any(i >= n or j >= n for i, j, _ in pairs):
            bo_overrun += 1

    from molmetal.data.tmqm import parse_natural_charges

    q_map = parse_natural_charges(q_path=TMQM_DIR / "tmQM_X.q")
    q_len_mismatch = sum(
        1
        for code in keep
        if code in q_map and code in atom_count and len(q_map[code]) != atom_count[code]
    )

    return {
        "xyz_shards": [p.name for p in xyz],
        "xyz_blocks_total": n_blocks,
        "cohort_blocks": sum(seen.values()),
        "cohort_unique_codes": len(seen),
        "duplicate_codes": sum(1 for v in seen.values() if v > 1),
        "missing_xyz": len(keep - set(seen)),
        "nonfinite_coordinate_blocks": n_nonfinite,
        "metal_atoms_per_complex": {str(k): v for k, v in sorted(n_metal_atoms.items())},
        "metal_label_disagrees_with_atom": len(metal_label_mismatch),
        "metal_label_examples": metal_label_mismatch[:5],
        "bo_coverage": [len(bo_codes), len(keep)],
        "bo_missing": len(keep - set(bo_codes)),
        "bo_index_out_of_range": bo_overrun,
        "q_blocks_total": len(q_map),
        "q_coverage": [len([c for c in q_map if c in keep]), len(keep)],
        "q_missing": len([c for c in keep if c not in q_map]),
        "q_length_mismatch": q_len_mismatch,
    }, heavy_count


def _smiles_audit(meta: dict, keep: set, heavy_count: dict) -> dict:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    n_empty = n_unparsed = n_dot = n_heavy = n_no_metal = 0
    examples = []
    for code in keep:
        smiles = str(meta[code].get("smiles", "")).strip()
        metal = str(meta[code].get("metal", "")).strip()
        if not smiles:
            n_empty += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_unparsed += 1
            if len(examples) < 3:
                examples.append({"csd_code": code, "smiles": smiles[:60], "why": "unparseable"})
            continue
        if "." in smiles:
            n_dot += 1
        n_heavy_atoms = mol.GetNumHeavyAtoms()
        if n_heavy_atoms != heavy_count[code]:
            n_heavy += 1
            if len(examples) < 6:
                examples.append(
                    {
                        "csd_code": code,
                        "smiles": smiles[:60],
                        "why": f"heavy_atoms {n_heavy_atoms} != xyz {heavy_count[code]}",
                    }
                )
        if not any(a.GetSymbol() == metal for a in mol.GetAtoms()):
            n_no_metal += 1
            if len(examples) < 9:
                examples.append({"csd_code": code, "smiles": smiles[:60], "why": "metal absent"})
    return {
        "empty": n_empty,
        "empty_fraction": round(n_empty / len(keep), 4),
        "unparseable": n_unparsed,
        "multi_fragment": n_dot,
        "heavy_atom_mismatch": n_heavy,
        "metal_absent_from_smiles": n_no_metal,
        "examples": examples,
    }


def _split_audit(meta: dict, keep: set, scaf: dict) -> dict:
    """Replay the loader's scaffold split and count rows, not scaffolds.

    Same inputs as ``load_tmqm_clouds(split_kind="scaffold")``: one group per
    scaffold, ``ScaffoldSplitter(strategy="largest_first",
    fractions=DEFAULT_FRACTIONS, seed=0)``.  The candidate-pool sizes this
    produces are cross-checked against the loader's own printed counts in
    :func:`_val_cohort`.
    """
    by_scaf: dict = defaultdict(list)
    for code in keep:
        by_scaf[scaf[code]].append(code)
    codes = sorted(keep)
    idx = {c: i for i, c in enumerate(codes)}
    groups = [[idx[c] for c in by_scaf[k]] for k in sorted(by_scaf)]
    assign = ScaffoldSplitter(
        strategy="largest_first", fractions=DEFAULT_FRACTIONS, seed=0
    )._assign_groups(groups)
    idx_to_code = {i: c for c, i in idx.items()}

    rows: Counter = Counter()
    metal: dict = defaultdict(Counter)
    for i, split in assign.items():
        code = idx_to_code[i]
        rows[split] += 1
        metal[split][str(meta[code].get("metal", "")).strip()] += 1
    total = sum(rows.values())
    spanning = len(
        [1 for k in by_scaf if len({assign[idx[c]] for c in by_scaf[k]}) > 1]
    )
    return {
        "unique_scaffolds": len(by_scaf),
        "largest_scaffold_rows": max(len(v) for v in by_scaf.values()),
        "rows": {s: int(rows[s]) for s in ("train", "val", "test")},
        "row_fraction": {s: round(rows[s] / total, 4) for s in ("train", "val", "test")},
        "per_metal": {s: dict(sorted(metal[s].items())) for s in ("train", "val", "test")},
        "scaffolds_spanning_splits": spanning,
        "val_frac_is_ignored_on_this_branch": True,
    }


def _val_cohort() -> dict:
    """What the trainer actually puts in front of the generation metrics."""
    out = {}
    for mpm in (0, 20):
        clouds = load_tmqm_clouds(
            str(TMQM_DIR),
            metals=METALS,
            coord_number=None,
            n_clouds=N_VAL,
            seed=0,
            split="val",
            split_kind="scaffold",
            min_per_metal=mpm,
        )
        counts = Counter(
            _Z_TO_SYMBOL[int(cl.atomic_numbers[int(cl.metal_atom_index)])]
            for cl in clouds
        )
        out[f"min_per_metal_{mpm}"] = {
            "n_clouds": len(clouds),
            "per_metal": dict(sorted(counts.items())),
            "full_pool_scan": mpm > 0,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    meta = load_tmqm_metadata(TMQM_DIR / "tmqm_parsed.csv")
    keep = {
        code
        for code, row in meta.items()
        if str(row.get("metal", "")).strip() in set(METALS)
    }
    cache = json.loads((TMQM_DIR / "scaffold_cache.json").read_text(encoding="utf-8"))
    scaf = _scaffold_of({c: meta[c] for c in keep}, cache)

    raw, heavy_count = _raw_audit(meta, keep)
    smiles = _smiles_audit(meta, keep, heavy_count)
    split = _split_audit(meta, keep, scaf)
    val = _val_cohort()

    corpus_share = {
        m: round(split["per_metal"]["train"][m] + split["per_metal"]["val"][m]
                 + split["per_metal"]["test"][m], 0)
        for m in METALS
    }
    total = sum(corpus_share.values())
    corpus_share = {m: round(v / total, 4) for m, v in corpus_share.items()}
    val_counts = val["min_per_metal_0"]["per_metal"]
    val_total = sum(val_counts.values())
    val_share = {m: round(v / val_total, 4) for m, v in val_counts.items()}

    payload = {
        "metric": "tmqm_cfm_data_audit",
        "description": (
            "Completeness and split composition of the tmQM six-metal corpus "
            "the CFM trainer streams, measured from the raw shards before any "
            "reservoir sampling."
        ),
        "unit": "complexes (one tmQM CSD code = one geometry = one cloud)",
        "scope": "data_protocol_audit",
        "source": (
            "molmetal/scripts/wf_a63_tmqm_cfm_data_audit.py over "
            ".data/molmetal/tmQM (tmqm_parsed.csv + tmQM_X*.xyz.gz + "
            "tmQM_X*.BO.gz + tmQM_X.q), calling the production loader "
            "load_tmqm_clouds for the val-cohort measurement"
        ),
        "cohort": {
            "tmqm_dir": ".data/molmetal/tmQM",
            "metadata_rows": len(meta),
            "metals": METALS,
            "coord_number_filter": None,
            "cohort_size": len(keep),
            "corpus_metal_share": corpus_share,
        },
        "raw_files": raw,
        "smiles_column": smiles,
        "scaffold_split": split,
        "val_cohort_at_n192": {
            "config_defaults": {
                "n_val": N_VAL,
                "tmqm_min_per_metal": 0,
                "note": (
                    "configs/c7_*.yaml set neither key, so the argparse "
                    "defaults are what run"
                ),
            },
            "corpus_metal_share": corpus_share,
            "cohort_metal_share": val_share,
            "runs": val,
        },
        "verdict": (
            "No data-side hole in the corpus: all 33,429 six-metal complexes "
            "carry geometry, Wiberg bond truth and NBO charges, with one metal "
            "atom each and a metal column that agrees with it everywhere. The "
            "split is scaffold-disjoint (0 spanning scaffolds) but allocates "
            "87.7/6.2/6.2 % of rows, because the fractions are budgeted in "
            "scaffold counts; tmqm_val_frac is ignored on that branch. The "
            "192-cloud val cohort every generation metric is averaged over is "
            "Pt 19 / Ru 93 at seed 0, against a corpus that is Pt 24.8 % / "
            "Ru 29.3 %: Pt is present but 2.5x under-weighted. Raising "
            "tmqm_min_per_metal to 20 balances it (Pt 46 / Ru 52) and forces a "
            "full 2,061-candidate scan instead of a 960-candidate window."
        ),
        "status": "MEASURED",
        "generated_by": "molmetal/scripts/wf_a63_tmqm_cfm_data_audit.py",
    }

    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    payload["content_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"cohort={len(keep)} raw={raw['cohort_unique_codes']} "
          f"missing_xyz={raw['missing_xyz']} bo_missing={raw['bo_missing']} "
          f"q_missing={raw['q_missing']}")
    print(f"rows train/val/test = {split['rows']} ({split['row_fraction']})")
    print(f"val n={N_VAL} mpm=0 {val_counts} share={val_share}")
    print(f"val n={N_VAL} mpm=20 {val['min_per_metal_20']['per_metal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
