# Apache-2.0 License
# Copyright 2026 Mol-Metal contributors
# Per AGENTS.md §5.10 license-header mandate.
"""Canonical citation blocks for the 4 paper main-result references.

Mirrors ``todo.json`` §PRIMARY_1 (LigandDiff) / §SECONDARY_1
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


def cite_rusanov_ece(per_cell: dict = None, per_split: dict = None) -> str:
    """Cite the per-cell / per-split ECE block from Rusanov 2025 SI Table S5.

    Wf-A49 (2026-10-16) — the on-disk JSON
    ``.data/molmetal/rusanov_2025/parsed/s5_ece.json`` carries
    Expected Calibration Errors for two models (MKANO / KANO)
    across four cell lines (A2780 / A2780cis / MCF-7 / A549) and
    three split types (scaffold-based / random / cluster-based).
    Numbers are ECE percentages — lower is better.

    The function accepts optional ``per_cell`` (e.g. ``"A2780"``)
    and ``per_split`` (e.g. ``"scaffold-based"``) keyword
    arguments so the cite line can name the row of interest;
    neither is required.  Returns a one-line citation that names
    the on-disk JSON path so the source is verifiable per
    AGENTS.md §5.5 filesystem-truth.

    Parameters
    ----------
    per_cell : str, optional
        One of ``{"A2780", "A2780cis", "MCF-7", "A549"}``.  When
        ``None`` the cite line reports the aggregate (the on-disk
        JSON is per-cell/per-split, so there is no aggregate row;
        the aggregate framing is just "see Table S5 in
        supplementary_information_v2.pdf").
    per_split : str, optional
        One of ``{"scaffold-based", "random", "cluster-based"}``.
        When ``None`` the cite line lists all three split types.

    Returns
    -------
    str
        A single-line BibTeX-ready citation that points at
        ``.data/molmetal/rusanov_2025/parsed/s5_ece.json`` (1.6 KB
        on disk; 4 cell lines × 2 models × 3 split types).

    Notes
    -----
    This cite block is **READ-ONLY on the parsed JSON** — it does
    not validate against it.  Callers wanting a per-cell /
    per-split ECE number must read the JSON themselves; this
    function only emits the citation line that names it.  Per
    AGENTS.md §5.3 no paper/* touch: this cite is consumed by
    ``molmetal/eval/reporting/paper_table.table1_main()`` as a
    footnote row, NOT injected into paper/main.tex.
    """
    cell = str(per_cell) if per_cell else "all 4 cell lines (A2780 / A2780cis / MCF-7 / A549)"
    split = (
        str(per_split)
        if per_split
        else "all 3 split types (scaffold-based / random / cluster-based)"
    )
    return (
        "Rusanov et al. (2025). 'PlatinAI / MB-Finder ECE per cell line × split type.' "
        f"ChemRxiv 10.26434/chemrxiv-2025-pp32k v2 SI Table S5. "
        f"Cell: {cell}; Split: {split}. "
        f"Source JSON on disk: .data/molmetal/rusanov_2025/parsed/s5_ece.json (1594 B). "
        f"S7 Pt-wet-lab cross-ref: .data/molmetal/rusanov_2025/parsed/s7_pt_wet_lab.json (9715 B)."
    )


__all__ = [
    "cite_liganddiff",
    "cite_tmcinvdes",
    "cite_platinai",
    "cite_metalcytotox",
    "cite_rusanov_ece",
]