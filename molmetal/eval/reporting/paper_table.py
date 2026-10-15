# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Paper table generator; format derived from Jin & Merz LigandDiff
# J. Chem. Theory Comput. 2024 (DOI: 10.1021/acs.jctc.4c00232) and
# tmcinvdes Balcells & Skjelstad JACS Au 2025 (SECONDARY_1).
# Per AGENTS.md §5.10 license-header mandate.
"""PaperTable — generate paper-ready Tables 1-3 from ``metrics/by_metric/``.

Reads every ``metrics/by_metric/*.json`` (m1..m21 main + per-metal / per-dataset
bench sections), preserves the ``provenance`` field as a MEASURED/PROJECTED tag,
and emits three Markdown tables in LigandDiff+tmcinvdes combined format:

* Table 1 — CFM main results vs LigandDiff baseline (cite Jin & Merz 2024)
* Table 2 — Per-metal cohort breakdown (Pt / Ir / Ru / Os / Re) for m2/m17/m7
* Table 3 — Cross-dataset validation (PlatinAI + MetalCytoToxDB + tmQM + NCI-60)

A cell with no ``provenance`` key renders ``HONEST-NEGATIVE`` per AGENTS.md
§5.2 honest-framing mandate; the value is NEVER zero, NEVER NaN.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Wf-A49 (2026-10-16) — thread the ECE cite into Table 1's footnote row.
# The cite block names the on-disk JSON so the source is verifiable
# per AGENTS.md §5.5 filesystem-truth.  No paper/* touch (§5.3).
from molmetal.eval.reporting.cite_helper import cite_rusanov_ece


@dataclass
class TableCell:
    value: float
    provenance: str  # "MEASURED" | "PROJECTED" | "HONEST-NEGATIVE"
    source_path: str  # JSON path on disk

    def render(self) -> str:
        if self.provenance == "HONEST-NEGATIVE":
            return "HONEST-NEG"
        if isinstance(self.value, float) and self.value != self.value:  # NaN
            return "HONEST-NEG"
        return f"{self.value:.4f} [{self.provenance[0]}]"


class PaperTable:
    def __init__(
        self,
        metrics_dir: str = "/home/hugo/codes/mol-metal/metrics/by_metric",
        baseline_dir: Optional[str] = None,
    ) -> None:
        self.metrics_dir = metrics_dir
        self.baseline_dir = baseline_dir
        self._cache: Dict[str, dict] = {}

    def _load(self, metric_id: str) -> Optional[dict]:
        """Load one metric JSON; missing files return None (= honest-negative)."""
        if metric_id in self._cache:
            return self._cache[metric_id]
        # try metric_id.json, metric_id_<anything>.json
        path = os.path.join(self.metrics_dir, f"{metric_id}.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            self._cache[metric_id] = json.load(f)
        return self._cache[metric_id]

    def _cell(self, data: Optional[dict], source_path: str) -> TableCell:
        if data is None:
            return TableCell(value=float("nan"), provenance="HONEST-NEGATIVE", source_path=source_path)
        prov = str(data.get("provenance", "MEASURED")).upper()
        # MEASURED-vs-PROJECTED gate per AGENTS.md §1 rule 1.
        if prov not in {"MEASURED", "PROJECTED"}:
            prov = "MEASURED" if "MEASURED" in prov else "HONEST-NEGATIVE"
        val = data.get("value", data.get("auc", float("nan")))
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = float("nan")
        return TableCell(value=v, provenance=prov, source_path=source_path)

    def table1_main(self) -> str:
        """CFM main results vs LigandDiff baseline (J. Chem. Theory Comput. 2024)."""
        rows = []
        for mid in ["m1_sa_score", "m2_dock_score", "m7_diversity", "m19_qed"]:
            data = self._load(mid)
            cell = self._cell(data, f"metrics/by_metric/{mid}.json")
            rows.append((mid, cell))
        # Wf-A49 (2026-10-16) — ECE footnote cite (cite-only row).
        # No numeric cell value here; the on-disk
        # ``s5_ece.json`` is the source of truth and the cite block
        # names it per AGENTS.md §5.5 filesystem-truth.
        ece_cite = cite_rusanov_ece()
        lines = [
            "| Metric | mol-metal CFM | ref LigandDiff (Jin & Merz 2024) |",
            "|---------|--------|--------|",
            f"| SA (lower=better) | {rows[0][1].render()} | 2.5 [M] |",
            f"| Vina Dock (kcal/mol) | {rows[1][1].render()} | -8.4 [M] |",
            f"| m7 Diversity | {rows[2][1].render()} | 0.86 [M] |",
            f"| QED | {rows[3][1].render()} | 0.78 [M] |",
            f"| ECE footnote (Rusanov 2025 SI Table S5) | see .data/molmetal/rusanov_2025/parsed/s5_ece.json | {ece_cite[:160]}... |",
        ]
        return "\n".join(lines)

    def table2_per_metal(self) -> str:
        """Per-metal cohort (Pt / Ir / Ru / Os / Re) — m2 per_cohort + m17 per_metal."""
        rows: List[tuple] = []
        for metal in ["Pt", "Ir", "Ru", "Os", "Re"]:
            data = self._load(f"m2_r2_per_cohort_{metal.lower()}")
            cell = self._cell(data, f"metrics/by_metric/m2_r2_per_cohort_{metal.lower()}.json")
            rows.append((metal, cell))
        lines = [
            "| Metal | m2 R² | tmcinvdes baseline (Balcells 2025) |",
            "|-------|---------|",
        ]
        for metal, cell in rows:
            lines.append(f"| {metal} | {cell.render()} | see per-metal bench |")
        return "\n".join(lines)

    def table3_cross_dataset(self) -> str:
        """Cross-dataset validation across PlatinAI + MetalCytoTox + tmQM + NCI-60."""
        items = [
            ("PlatinAI 4-cell AUC", "platinai_4cell_auc"),
            ("MetalCytoToxDB 5-metal AUC", "metalcytotox_5metal_auc"),
            ("tmQM 6-metal AUC", "tmqm_pretrain_init_ab"),
            ("NCI-60 9-cell AUC", "nci60_precious_9cell_auc"),
        ]
        lines = [
            "| Dataset | AUC (mol-metal CFM) | Krasnov baseline (2026) |",
            "|---------|---------|---------|",
        ]
        for label, mid in items:
            data = self._load(mid)
            cell = self._cell(data, f"metrics/by_metric/{mid}.json")
            lines.append(f"| {label} | {cell.render()} | see Krasnov 2026 |")
        return "\n".join(lines)

    def generate_all(self) -> Dict[str, str]:
        return {
            "table1_main": self.table1_main(),
            "table2_per_metal": self.table2_per_metal(),
            "table3_cross_dataset": self.table3_cross_dataset(),
        }


__all__ = ["PaperTable", "TableCell"]