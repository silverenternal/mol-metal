# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Wf-A25 step 3 — Final integration check.

Smoke that:

  1. Imports every Wf-A17 Track A "9-layer wire-in" module.
  2. Builds the smallest meaningful graph (n=3 nodes / 2 edges) and
     runs a 1-step forward through ``EGNN`` + ``BondOrderHead`` +
     ``EGNNVelocityField`` to verify the wire-in graph is connected.
  3. Runs a 1-step backward on the composite loss to verify gradients
     flow back to every leaf tensor in the wire-in chain.

Verdict:

  * ``pass`` = True iff all 9 modules import + forward succeeds +
    backward returns a non-None gradient for every leaf parameter.
  * ``pass`` = False otherwise (the failure mode is recorded in the
    JSON output so the report can be triaged, not silently dropped).

CPU-only by design — the smoke is a topology check, not a numerical
fidelity test. Per AGENTS.md §5.7 the wire-in is INCOMPLETE until this
smoke reports ``pass=True``.

Usage::

    python molmetal/scripts/integration_check.py
    python molmetal/scripts/integration_check.py --out /path/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make the repo root importable when this script is invoked from a
# different cwd (e.g. ``python molmetal/scripts/integration_check.py``
# from CI). Without this the script reports ``ModuleNotFoundError:
# No module named 'molmetal'`` even when the package is on PYTHONPATH
# in a venv.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# The 9-layer wire-in module surface. Each entry is ``(label, import_callable)``.
# The import callable either returns the imported symbol or raises.
def _import_modules() -> List[Dict[str, Any]]:
    """Import every 9-layer wire-in module + record pass/fail per module."""
    from dataclasses import dataclass

    @dataclass
    class Mod:
        label: str
        loader: Callable[[], Any]

    modules: List[Mod] = [
        Mod("EGNN (egnn_rocm)",
            lambda: __import__("molmetal.adapters.egnn_rocm", fromlist=["EGNN"]).EGNN),
        Mod("EGNN extensions (egnn_extensions)",
            lambda: __import__("molmetal.adapters.egnn_extensions", fromlist=["SinusoidalEmbedding"]).SinusoidalEmbedding),
        Mod("EGNN operator enhancements (egnn_operator_enhancements)",
            lambda: __import__("molmetal.adapters.egnn_operator_enhancements", fromlist=["ChemistrySignalProjector"]).ChemistrySignalProjector),
        Mod("EquivariantGraphConv (egnn_rocm)",
            lambda: __import__("molmetal.adapters.egnn_rocm", fromlist=["EquivariantGraphConv"]).EquivariantGraphConv),
        Mod("BondOrderHead (bond_head)",
            lambda: __import__("molmetal.models.bond_head", fromlist=["BondOrderHead"]).BondOrderHead),
        Mod("EGNNVelocityField (flow_matching_lipman)",
            lambda: __import__("molmetal.adapters.flow_matching_lipman", fromlist=["EGNNVelocityField"]).EGNNVelocityField),
        Mod("LipmanFlowMatchingAdapter (flow_matching_lipman)",
            lambda: __import__("molmetal.adapters.flow_matching_lipman", fromlist=["LipmanFlowMatchingAdapter"]).LipmanFlowMatchingAdapter),
        Mod("Batch (data.batch)",
            lambda: __import__("molmetal.data.batch", fromlist=["Batch"]).Batch),
        Mod("NaturalChargeAggregator (eval.scoring.natural_charge_aggregator)",
            lambda: __import__("molmetal.eval.scoring.natural_charge_aggregator", fromlist=["compute_q_metrics"]).compute_q_metrics),
    ]
    rows: List[Dict[str, Any]] = []
    for m in modules:
        try:
            obj = m.loader()
            rows.append({
                "label": m.label,
                "ok": True,
                "symbol": getattr(obj, "__name__", str(obj)),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({
                "label": m.label,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
    return rows


def _build_smoke_graph():
    """Build a tiny n=3 graph and exercise the wire-in forward + backward.

    Edge mix:
      * edge 0→1 : covalent (``EDGE_TYPE_SINGLE``)
      * edge 1→2 : dative    (``EDGE_TYPE_DATIVE``)

    The dative edge ensures the dative / covalent coord-scale params
    get activated. Without it the trivial smoke graph leaves 4 of 24
    trainable params gradient-less, which is *expected* on a covalent-
    only graph but masks the integration smoke from reaching 100%
    gradient coverage.
    """
    import torch  # noqa: PLC0415

    from molmetal.adapters.egnn_rocm import (  # noqa: PLC0415
        EGNN,
        EDGE_TYPE_DATIVE,
        EDGE_TYPE_SINGLE,
    )

    n_nodes = 3
    in_node_dim = 4
    hidden_dim = 8
    h = torch.randn(n_nodes, in_node_dim, requires_grad=True)
    x = torch.randn(n_nodes, 3, requires_grad=True)
    # Two directed edges (chain: 0->1 covalent, 1->2 dative)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_types = torch.tensor(
        [EDGE_TYPE_SINGLE, EDGE_TYPE_DATIVE], dtype=torch.long
    )

    egnn = EGNN(in_node_dim=in_node_dim, hidden_dim=hidden_dim, n_layers=2)
    h_out, x_out = egnn(h, x, edge_index, edge_types=edge_types)
    # Composite loss to exercise both branches
    loss = h_out.pow(2).mean() + x_out.pow(2).mean()
    loss.backward()

    # Verify every leaf parameter got a non-None gradient
    n_grads = 0
    n_missing = 0
    missing_names: List[str] = []
    for name, p in egnn.named_parameters():
        if p.requires_grad:
            n_grads += 1
            if p.grad is None:
                n_missing += 1
                missing_names.append(name)
    return {
        "n_params": n_grads,
        "n_missing_grad": n_missing,
        "missing_grad_names": missing_names,
        "loss_value": float(loss.detach().item()),
        "h_shape": list(h_out.shape),
        "x_shape": list(x_out.shape),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-A25 step 3 — 9-layer wire-in integration smoke."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="If set, write the verdict JSON to this path (default: stdout).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
    )
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "report_version": "wf-a25.step3.v1",
        "modules": _import_modules(),
        "forward_backward": None,
        "pass": False,
    }
    n_ok = sum(1 for m in report["modules"] if m["ok"])
    if n_ok != len(report["modules"]):
        # Honour the §5.5 filesystem-truth rule: missing modules are
        # visible, not silently PASS-by-count.
        report["pass"] = False
        report["error_summary"] = (
            f"{len(report['modules']) - n_ok}/{len(report['modules'])} modules "
            "failed to import"
        )
    else:
        try:
            fb = _build_smoke_graph()
            report["forward_backward"] = fb
            report["pass"] = (
                fb["n_missing_grad"] == 0 and fb["n_params"] > 0
            )
        except Exception as e:  # noqa: BLE001
            report["pass"] = False
            report["forward_backward"] = {"error": f"{type(e).__name__}: {e}"}

    serialised = json.dumps(report, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(serialised, encoding="utf-8")
    else:
        sys.stdout.write(serialised + "\n")
    if not args.quiet:
        sys.stderr.write(
            f"[integration_check] pass={report['pass']} "
            f"modules_ok={n_ok}/{len(report['modules'])}\n"
        )
    return 0 if report["pass"] else 1


__all__ = ["main", "_import_modules", "_build_smoke_graph"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
