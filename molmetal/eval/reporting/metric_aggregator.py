# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Multi-metric roll-up that preserves MEASURED/PROJECTED tags.

``MetricAggregator.aggregate(metric_ids)`` reads each ``metrics/by_metric/<id>.json``,
returns ``{metric_id: {"value", "provenance", "source_path"}}``. Missing files
return HONEST-NEGATIVE per AGENTS.md §5.2 honest framing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List


class MetricAggregator:
    def __init__(
        self,
        metrics_dir: str = str(
            Path(__file__).resolve().parents[3] / "metrics" / "by_metric"
        ),
    ) -> None:
        self.metrics_dir = metrics_dir

    def aggregate(self, metric_ids: List[str]) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for mid in metric_ids:
            path = os.path.join(self.metrics_dir, f"{mid}.json")
            if not os.path.isfile(path):
                out[mid] = {"value": None, "provenance": "HONEST-NEGATIVE", "source_path": path}
                continue
            with open(path) as f:
                data = json.load(f)
            prov = str(data.get("provenance", "MEASURED")).upper()
            if "MEASURED" not in prov and "PROJECTED" not in prov:
                prov = "HONEST-NEGATIVE"
            val = data.get("value", data.get("auc"))
            out[mid] = {"value": val, "provenance": prov, "source_path": path}
        return out


__all__ = ["MetricAggregator"]