"""Isolated REINVENT4 JSON-lines bridge used for protocol validation.

The upstream REINVENT4 command line launches a TOML-configured RL program;
it is not a line-oriented scorer.  This small process therefore exists as a
safe bridge for the adapter contract and CI smoke tests.  It only uses RDKit,
never imports REINVENT4 or PyTorch, and can run from a separate environment.
It is a protocol bridge, not a claim that REINVENT4's learned plugins are
available.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def _score(smiles: str) -> dict[str, float | None]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, QED
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return {"qed": None, "sa": None, "binding": None, "novelty": None}
        # These are deterministic protocol-level proxies.  The real scorer
        # can replace this worker without changing the adapter wire format.
        qed = float(max(0.0, min(1.0, QED.qed(mol))))
        sa = float(max(0.0, min(1.0, 1.0 - Descriptors.MolWt(mol) / 600.0)))
        binding = float(max(0.0, min(1.0, 0.5 - Crippen.MolLogP(mol) / 10.0)))
        novelty = float(max(0.0, min(1.0, 1.0 - mol.GetNumHeavyAtoms() / 80.0)))
        return {"qed": qed, "sa": sa, "binding": binding, "novelty": novelty}
    except Exception:
        return {"qed": None, "sa": None, "binding": None, "novelty": None}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    op = request.get("op")
    if op == "ping":
        return {"ok": True, "bridge": "rdkit-proxy", "protocol": "reinvent4-jsonl-v1"}
    if op == "score":
        smiles = request.get("smiles", [])
        if not isinstance(smiles, list):
            return {"error": "smiles must be a list", "results": []}
        return {"results": [_score(s) for s in smiles]}
    return {"error": f"unsupported op: {op}", "results": []}


def main() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle(request if isinstance(request, dict) else {})
        except Exception as exc:  # keep stdout JSON-lines clean
            response = {"error": str(exc), "results": []}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
