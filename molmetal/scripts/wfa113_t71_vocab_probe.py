# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T71 — the YuelBond vocab is Pt-only while the cohort is 5-metal.

WHY THIS EXISTS
---------------
``YuelBondDecoderHead`` encodes an atom by ``z_to_onehot(Z, self.z_table)``,
and ``z_to_onehot`` maps every Z that is not in the table to **index 0**,
the padding slot (``yuelbond_decoder.py:282``).

The default preset is ``pt``: 13 entries whose only metal is Pt (78)
(``get_z_table("pt")``).  ``all6`` widens it to 18 entries covering
Pt/Ru/Ir/Rh/Os/Re.

``cfm_5metal_default.yaml`` and ``cfm_5metal_v5_full.yaml`` declare
``atom_vocab_size: 13`` and never set ``yuelbond_vocab``, and their cohort
is Ru/Ir/Os/Rh/Re with **no Pt at all** (the config's own note records
"Ru 176 / Ir 48 / Os 12 / Rh 10 / Re 10").  So every metal atom in those
runs is encoded as padding.

The ``atom_vocab_size`` guard only compares the COUNT (13 == 13), so the
mismatch is invisible.

This probe measures the OOV fraction on a real cohort and audits every
config's (cohort metals) x (z_table metals) pair.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t71_vocab_probe.py \
        --out metrics/by_metric/wfa113_t71_vocab.json
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[2]

_METAL_SYMBOL_TO_Z = {
    "Ru": 44, "Rh": 45, "Pd": 46, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
}


def _load_trainer():
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_wfa113_t71_trainer",
        str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_wfa113_t71_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _z_table(preset: str):
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table
    return tuple(get_z_table(preset))


def _csv_metals(path: Path) -> Optional[List[str]]:
    if not path.is_file():
        return None
    import csv as _csv
    seen = collections.Counter()
    with path.open(newline="") as fh:
        rd = _csv.DictReader(fh)
        key = None
        for cand in ("metal", "metal_symbol", "metal_element"):
            if rd.fieldnames and cand in rd.fieldnames:
                key = cand
                break
        if key is None:
            return None
        for row in rd:
            v = (row.get(key) or "").strip()
            if v:
                seen[v] += 1
    return [k for k, _ in seen.most_common()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, default=64)
    args = ap.parse_args()

    import yaml

    tr = _load_trainer()
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import z_to_onehot

    # ---- 1. static audit over every config ---------------------------
    audit: List[Dict[str, Any]] = []
    for p in sorted((_REPO / "configs").glob("*.yaml")):
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except Exception:
            continue
        flow = doc.get("flow") or {}
        model = doc.get("model") or {}
        data = doc.get("data") or {}
        if not isinstance(flow, dict):
            flow = {}
        if not isinstance(model, dict):
            model = {}
        if not isinstance(data, dict):
            data = {}
        preset = str(flow.get("yuelbond_vocab") or model.get("yuelbond_vocab") or "pt")
        try:
            table = _z_table(preset)
        except ValueError:
            continue
        table_metals = {z for z in table if z in _METAL_SYMBOL_TO_Z.values()}
        cohort: Optional[List[str]] = None
        src = None
        if data.get("dataset") == "tmqm":
            m = data.get("tmqm_metals") or flow.get("tmqm_metals")
            if isinstance(m, list):
                cohort, src = [str(x) for x in m], "tmqm_metals"
        if cohort is None:
            cf = data.get("cohort_filter") or {}
            m = cf.get("metals") if isinstance(cf, dict) else None
            if isinstance(m, list):
                cohort, src = [str(x) for x in m], "cohort_filter.metals"
        if cohort is None and data.get("dataset") == "csv":
            csv_rel = data.get("csv")
            if isinstance(csv_rel, str):
                got = _csv_metals(_REPO / csv_rel)
                if got:
                    cohort, src = got, "csv.metal column"
        if not cohort:
            continue
        cohort_z = {_METAL_SYMBOL_TO_Z[s] for s in cohort if s in _METAL_SYMBOL_TO_Z}
        oov = sorted(cohort_z - table_metals)
        audit.append({
            "config": p.name,
            "dataset": data.get("dataset"),
            "preset": preset,
            "declared_atom_vocab_size": model.get("atom_vocab_size"),
            "actual_table_size": len(table),
            "cohort_metals": sorted(cohort),
            "cohort_source": src,
            "table_metals": sorted(table_metals),
            "metals_oov": oov,
            "all_metals_oov": bool(cohort_z) and len(oov) == len(cohort_z),
        })

    affected = [a for a in audit if a["metals_oov"]]

    # ---- 2. measure the OOV fraction on one real cohort --------------
    measured: Dict[str, Any] = {}
    for cfg_name in ("cfm_5metal_default.yaml", "c8_fullgen_b64_2026_10_24.yaml"):
        cfg_path = _REPO / "configs" / cfg_name
        if not cfg_path.is_file():
            continue
        sys.argv = ["probe", "--config", str(cfg_path), "--device", "cpu",
                    "--out-dir", "/tmp/wfa113_t71", "--ckpt-dir", "/tmp/wfa113_t71",
                    "--n-train", str(args.n_clouds), "--n-val", "2"]
        try:
            cfg = tr.parse_args()
        except SystemExit:
            continue
        preset = str(getattr(cfg, "yuelbond_vocab", "pt"))
        table = _z_table(preset)
        table_set = set(table)
        to_idx = {z: i for i, z in enumerate(table)}
        clouds = []
        try:
            if str(getattr(cfg, "dataset", "")) == "csv":
                csv_rel = str(getattr(cfg, "data_csv", ""))
                cpath = _REPO / csv_rel
                if not cpath.is_file():
                    continue
                clouds = tr.build_csv_cohort(
                    str(cpath),
                    split=str(getattr(cfg, "data_split", "train")),
                    n_clouds=int(args.n_clouds),
                    seed=int(cfg.seed),
                    metal_placement=str(getattr(cfg, "metal_placement", "bonded")),
                    dative_distance_cutoff=float(
                        getattr(cfg, "dative_distance_cutoff", 2.6)),
                )
        except Exception as exc:  # noqa: BLE001
            measured[cfg_name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if not clouds:
            continue
        metal_zs = set(_METAL_SYMBOL_TO_Z.values())
        n_atoms = n_oov = n_metal = n_metal_oov = 0
        pad_idx_hits = collections.Counter()
        for c in clouds:
            z = c.atomic_numbers.tolist()
            n_atoms += len(z)
            for zz in z:
                if zz not in table_set:
                    n_oov += 1
                    pad_idx_hits[int(to_idx.get(zz, 0))] += 1
                if zz in metal_zs:
                    n_metal += 1
                    if zz not in table_set:
                        n_metal_oov += 1
        measured[cfg_name] = {
            "preset": preset,
            "table_size": len(table),
            "table_metals": sorted(table_set & metal_zs),
            "n_clouds": len(clouds),
            "n_atoms": n_atoms,
            "n_oov_atoms": n_oov,
            "oov_atom_frac": (n_oov / n_atoms) if n_atoms else None,
            "n_metal_atoms": n_metal,
            "n_metal_oov": n_metal_oov,
            "metal_oov_frac": (n_metal_oov / n_metal) if n_metal else None,
            "oov_mapped_to_index": dict(pad_idx_hits),
            # Direct proof that an OOV metal and the padding slot are the same
            # vector after z_to_onehot.
            "onehot_Ru_equals_onehot_pad": bool(
                torch.equal(
                    z_to_onehot(torch.tensor([44]), table),
                    z_to_onehot(torch.tensor([0]), table),
                )
            ),
        }

    out = {
        "probe": "wfa113_t71_vocab",
        "z_to_onehot_oov_behaviour": "unknown Z -> index 0 (padding)",
        "configs_audited": len(audit),
        "configs_with_oov_metal": len(affected),
        "affected": affected,
        "measured": measured,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2) + "\n")

    print(f"configs audited={len(audit)} with OOV metal={len(affected)}")
    for a in affected[:20]:
        print(f"  {a['config']}: preset={a['preset']} cohort={a['cohort_metals']} "
              f"table_metals={a['table_metals']} oov={a['metals_oov']}")
    for k, v in measured.items():
        if "error" in v:
            print(f"  MEASURED {k}: ERROR {v['error']}")
        else:
            print(f"  MEASURED {k}: preset={v['preset']} metal_oov="
                  f"{v['n_metal_oov']}/{v['n_metal_atoms']} "
                  f"({100 * (v['metal_oov_frac'] or 0):.1f}%) "
                  f"oov_atoms={v['n_oov_atoms']}/{v['n_atoms']} "
                  f"Ru==pad:{v['onehot_Ru_equals_onehot_pad']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
