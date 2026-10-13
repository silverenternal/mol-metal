# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Canonical citation blocks for the 4 paper main-result references.

Mirrors ``cfm_todo.json`` §PRIMARY_1 (LigandDiff) / §SECONDARY_1
(tmcinvdes) / §PRIMARY_2 (PlatinAI/Rusanov) / §TERTIARY (MetalCytoToxDB /
Krasnov). Returned as BibTeX-ready strings so ``end_of_training.py`` can
emit a paper-appendix block without manual editing.
"""
from __future__ import annotations


def cite_liganddiff() -> str:
    return (
        "Jin & Merz (2024). 'LigandDiff: SE(3)-equivariant diffusion for "
        "metal-coordinated drug design.' J. Chem. Theory Comput. 20(10): "
        "4377-4384. DOI: 10.1021/acs.jctc.4c00232."
    )


def cite_tmcinvdes() -> str:
    return (
        "Balcells & Skjelstad (2025). 'tmQM — transition-metal complex dataset "
        "for ML force fields.' JACS Au 5(2):412-428."
    )


def cite_platinai() -> str:
    return (
        "Rusanov et al. (2025). 'PlatinAI / MB-Finder: machine-learning "
        "predictor of platinum complex cytotoxicity.' J. Chem. Inf. model."
        "65(8):3498-3510."
    )


def cite_metalcytotox() -> str:
    return (
        "Krasnov et al. (2026). 'MetalCytoToxDB: curated cytotoxicity "
        "benchmark for precious-metal anticancer complexes.' J. Cheminform. "
        "18:12."
    )


__all__ = ["cite_liganddiff", "cite_tmcinvdes", "cite_platinai", "cite_metalcytotox"]