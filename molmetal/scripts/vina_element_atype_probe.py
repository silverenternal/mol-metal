# SPDX-License-Identifier: Apache-2.0
"""Measure which element symbols Vina 1.2.7 accepts as a PDBQT atom type.

WHY THIS EXISTS
---------------
``metal_pdbqt_overrides.VINA_REJECTED_ELEMENTS`` is a MEASURED table, not
a hand-written guess. This script is how it was measured, and how you
re-measure it after a Vina upgrade. It answers one question per element
symbol that occurs in the tmQM corpus:

    does ``Vina(sf_name="vina").set_ligand_from_file()`` accept a ligand
    whose atype column carries this symbol?

Method (per symbol)
-------------------
1. Take a REAL meeko-emitted ligand PDBQT
   (``molmetal/reports/wf121_cisplatin_real_vina/830c/cisplatin.pdbqt``).
   A real file matters: a hand-rolled synthetic row is parsed by Vina's
   rigid-body reader, which rejects it for reasons unrelated to the
   atom type, and you get 0/44 accepted with no signal.
2. Substitute the symbol into the atom-name field ``[12:16]`` and the
   AutoDock atype field ``[77:79]``.
3. Call ``set_ligand_from_file``. A raise means Vina has no row.

A receptor is NOT needed — the parse error fires before the receptor is
consulted — so this is fast (44 subprocess-free in-process probes).

Usage
-----
    .venv/bin/python -m molmetal.scripts.vina_element_atype_probe

Writes ``metrics/by_metric/vina_127_element_atypes.json``.
"""

from __future__ import annotations

import gzip
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TMQM_X1 = PROJECT_ROOT / ".data" / "molmetal" / "tmQM" / "tmQM_X1.xyz.gz"
TEMPLATE_PDBQT = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
    / "830c" / "cisplatin.pdbqt"
)
OUT_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "vina_127_element_atypes.json"


def tmqm_element_inventory(xyz_gz: Path = TMQM_X1) -> List[Dict[str, int]]:
    """Stream the tmQM extended XYZ and count element symbols.

    Counts both atoms and the number of complexes containing each, because
    the per-complex number is what tells you whether a missing atype is a
    corner case (La: 1.22%) or a real blocker (B: 6.22%).
    """
    atoms: Counter = Counter()
    per_complex: Counter = Counter()
    n_complexes = 0
    with gzip.open(xyz_gz, "rt") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                break
            raw = raw.strip()
            if not raw:
                continue
            n_atoms = int(raw)
            fh.readline()  # comment line
            symbols = []
            for _ in range(n_atoms):
                ln = fh.readline()
                symbols.append(ln.split()[0] if ln.strip() else "?")
            n_complexes += 1
            atoms.update(symbols)
            per_complex.update(set(symbols))
    return [
        {"symbol": s, "n_atoms": c, "n_complexes": per_complex[s]}
        for s, c in sorted(atoms.items(), key=lambda kv: -kv[1])
    ]


def probe_symbol(symbol: str, tmpl: str, tmpl_line: str, path: Path) -> str:
    """Return ``"ACCEPTED"`` or ``"REJECTED"`` for one element symbol."""
    from vina import Vina  # imported lazily so --help works without vina

    body = tmpl_line[:12] + symbol.ljust(4)[:4] + tmpl_line[16:]
    body = body[:77] + symbol.ljust(2)[:2] + body[79:]
    path.write_text(tmpl.replace(tmpl_line, body))
    try:
        v = Vina(sf_name="vina", cpu=1, verbosity=0)
        v.set_ligand_from_file(str(path))
    except Exception:
        return "REJECTED"
    return "ACCEPTED"


def main() -> int:
    tmpl = TEMPLATE_PDBQT.read_text()
    tmpl_line = next(l for l in tmpl.splitlines(keepends=True) if l.startswith("ATOM"))
    assert len(tmpl_line) >= 79, "template ATOM line shorter than 79 cols"

    inv = tmqm_element_inventory()
    n_complexes_total = None
    verdict: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.pdbqt"
        for row in inv:
            sym = row["symbol"]
            if sym == "?":
                continue
            verdict[sym] = probe_symbol(sym, tmpl, tmpl_line, p)

    accepted = sorted(s for s, v in verdict.items() if v == "ACCEPTED")
    rejected = sorted(s for s, v in verdict.items() if v == "REJECTED")

    # Cross-check the measurement against the committed constants so a
    # Vina upgrade that changes the accepted set fails loudly instead of
    # silently invalidating the rewrite rule.
    from molmetal.molmetal_lam.sbdd_env.metal_pdbqt_overrides import (
        VINA_ACCEPTED_ELEMENTS,
        VINA_NONMETAL_SURROGATES,
        VINA_REJECTED_ELEMENTS,
    )

    drift = {
        "missing_from_accepted_const": sorted(set(accepted) - set(VINA_ACCEPTED_ELEMENTS)),
        "missing_from_rejected_const": sorted(set(rejected) - set(VINA_REJECTED_ELEMENTS)),
        "surrogate_not_accepted": sorted(
            e for e, s in VINA_NONMETAL_SURROGATES.items() if s not in accepted
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "vina_version": "1.2.7",
                "template_pdbqt": str(TEMPLATE_PDBQT.relative_to(PROJECT_ROOT)),
                "corpus": str(TMQM_X1.relative_to(PROJECT_ROOT)),
                "n_symbols_probed": len(verdict),
                "n_accepted": len(accepted),
                "n_rejected": len(rejected),
                "accepted": accepted,
                "rejected": rejected,
                "verdict_per_symbol": verdict,
                "tmqm_inventory": inv,
                "constant_drift": drift,
                "measurement": "MEASURED",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_JSON}")
    print(f"accepted={len(accepted)} rejected={len(rejected)}")
    if any(drift.values()):
        print("DRIFT vs committed constants:", json.dumps(drift))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())