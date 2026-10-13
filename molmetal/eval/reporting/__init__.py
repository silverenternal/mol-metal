# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Paper-table + metric-aggregation reporting surface (END-A5 fix).

Implements ``PaperTable`` (LigandDiff/tmcinvdes-formatted tables 1-3 over
``metrics/by_metric/*.json``), ``MetricAggregator`` (multi-metric roll-up
preserving MEASURED/PROJECTED tags), and ``cite_helper`` (4 canonical
citation blocks: LigandDiff, tmcinvdes, PlatinAI/Rusanov, Krasnov
MetalCytoToxDB). Every numeric value traces to a JSON / report path on
disk per AGENTS.md §1 rule 1 (MEASURED-vs-PROJECTED).
"""
from __future__ import annotations

from molmetal.eval.reporting.cite_helper import cite_liganddiff, cite_tmcinvdes, cite_platinai, cite_metalcytotox
from molmetal.eval.reporting.metric_aggregator import MetricAggregator
from molmetal.eval.reporting.paper_table import PaperTable

__all__ = [
    "PaperTable",
    "MetricAggregator",
    "cite_liganddiff",
    "cite_tmcinvdes",
    "cite_platinai",
    "cite_metalcytotox",
]