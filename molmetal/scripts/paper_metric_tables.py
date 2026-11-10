#!/usr/bin/env python3
"""Generate every paper metric table from the on-disk artifacts.

The tables are declarations, not transcriptions: each cell names the JSON file
and JSON pointer it reads.  A cell whose file or pointer does not resolve is
rendered as ``MISSING`` and recorded in the manifest, so a table can never
quietly disagree with the artifact it claims to summarise.

Usage
-----
    .venv/bin/python molmetal/scripts/paper_metric_tables.py
    .venv/bin/python molmetal/scripts/paper_metric_tables.py --strict
    .venv/bin/python molmetal/scripts/paper_metric_tables.py --only T1 --only T3

Outputs ``<out-dir>/<table>.md``, ``<out-dir>/<table>.csv`` and
``<out-dir>/manifest.json``.  Exit code 0 on success; 1 with ``--strict`` when
any cell is MISSING.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

METRICS_DIR = REPO_ROOT / "metrics" / "by_metric"
DEFAULT_OUT = REPO_ROOT / "metrics" / "paper_tables"

MISSING = "MISSING"

# ---------------------------------------------------------------------------
# artifact access
# ---------------------------------------------------------------------------


class ArtifactCache:
    """Load each artifact once per run; a missing file is a MISSING cell."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._docs: dict[str, Any] = {}
        self._errors: dict[str, str] = {}

    def get(self, name: str) -> Any:
        if name in self._docs:
            return self._docs[name]
        if name in self._errors:
            return None
        path = self.root / name
        if not path.exists():
            self._errors[name] = "file not found"
            self._docs[name] = None
            return None
        try:
            self._docs[name] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:  # a truncated artifact is missing
            self._errors[name] = f"invalid json: {exc}"
            self._docs[name] = None
        return self._docs[name]

    def error(self, name: str) -> str:
        return self._errors.get(name, "")

    def sources(self) -> dict[str, str]:
        out = {}
        for name in sorted(self._docs):
            path = self.root / name
            if path.exists():
                out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out


_FILTER = re.compile(r"^(?P<key>[^\[\]]*)\[(?P<field>[^=\]]+)=(?P<value>[^\]]+)\]$")


def resolve(doc: Any, pointer: str) -> Any:
    """Resolve ``/a/b/0`` or ``/curve[coverage=0.5]/model`` against ``doc``.

    Raises ``KeyError``/``IndexError``/``TypeError``/``ValueError`` when the
    pointer does not resolve; callers turn that into MISSING.
    """
    if doc is None:
        raise KeyError("artifact missing")
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"pointer must start with '/': {pointer!r}")

    cur = doc
    for raw in pointer.split("/")[1:]:
        seg = raw.replace("~1", "/").replace("~0", "~")
        if seg == "":
            continue
        match = _FILTER.match(seg)
        if match:
            key = match.group("key")
            if key:
                cur = cur[key]
            if not isinstance(cur, list):
                raise TypeError(f"filter on non-list at {seg!r}")
            field_name = match.group("field")
            want = match.group("value")
            hits = [x for x in cur if str(x.get(field_name)) == want]
            if not hits:
                raise KeyError(f"no list item with {field_name}={want}")
            cur = hits[0]
            continue
        if isinstance(cur, list):
            if not re.fullmatch(r"-?\d+", seg):
                raise TypeError(f"non-integer index {seg!r} on a list")
            cur = cur[int(seg)]
        else:
            cur = cur[seg]
    return cur


# ---------------------------------------------------------------------------
# cells and formatting
# ---------------------------------------------------------------------------


def _fmt(value: Any, style: str) -> str:
    if style == "raw":
        return str(value)
    if style == "int":
        return f"{int(value)}"
    if style == "pct":
        return f"{float(value) * 100:.2f}%"
    if style == "ci":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"ci needs a 2-element list, got {value!r}")
        return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"
    digits = {"f2": 2, "f3": 3, "f4": 4, "f6": 6}.get(style)
    if digits is None:
        raise ValueError(f"unknown format {style!r}")
    return f"{float(value):.{digits}f}"


@dataclass
class Cell:
    """One table cell.

    ``kind`` is one of:
      * ``path``  — read ``pointer`` from ``file``
      * ``delta`` — ``pointer - other`` within the same file
      * ``none``  — definitionally empty for this row (rendered as ``-``)

    ``optional`` cells render as ``-`` when the pointer is absent and are
    listed under ``optional_absent`` in the manifest rather than counted as
    MISSING.  Use it only for fields that are absent by design.
    """

    kind: str = "path"
    file: str = ""
    pointer: str = ""
    other: str = ""
    file_b: str = ""
    fmt: str = "f4"
    optional: bool = False

    def value(self, cache: ArtifactCache) -> tuple[str, str, str]:
        """Return ``(text, status, where)`` with status in ok/missing/absent."""
        if self.kind == "none":
            return "-", "ok", ""
        doc = cache.get(self.file)
        if doc is None:
            return MISSING, "missing", f"{self.file} ({cache.error(self.file)})"
        try:
            if self.kind == "delta":
                doc_b = cache.get(self.file_b) if self.file_b else doc
                if doc_b is None:
                    raise KeyError(f"{self.file_b} missing")
                raw = float(resolve(doc, self.pointer)) - float(resolve(doc_b, self.other))
            else:
                raw = resolve(doc, self.pointer)
            return _fmt(raw, self.fmt), "ok", ""
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            where = f"{self.file}:{self.pointer} ({exc})"
            if self.optional:
                return "-", "absent", where
            return MISSING, "missing", where

    def source(self) -> str:
        if self.kind == "none":
            return ""
        if self.kind == "delta":
            return f"{self.file}:{self.pointer} - {self.other}"
        return f"{self.file}:{self.pointer}"


def P(file: str, pointer: str, fmt: str = "f4", optional: bool = False) -> Cell:
    return Cell(kind="path", file=file, pointer=pointer, fmt=fmt, optional=optional)


def D(file: str, a: str, b: str, fmt: str = "f4") -> Cell:
    return Cell(kind="delta", file=file, pointer=a, other=b, fmt=fmt)


def DX(file_a: str, a: str, file_b: str, b: str, fmt: str = "f4") -> Cell:
    """Cross-artifact difference: ``a`` from ``file_a`` minus ``b`` from ``file_b``."""
    return Cell(kind="delta", file=file_a, pointer=a, other=b, file_b=file_b, fmt=fmt)


def NONE() -> Cell:
    return Cell(kind="none")


# ---------------------------------------------------------------------------
# table declarations
# ---------------------------------------------------------------------------

SEED = "msrs_seed_stability.json"
GEN = "msrs_generalization.json"
DELTA = "msrs_delta_metrics.json"
CAL = "msrs_calibration.json"
ABL = "msrs_metal_ablation.json"
DEC = "msrs_decision_curve.json"
POWER = "msrs_power.json"
CARD = "msrs_data_card.json"
GENM = "cfm_generation_metrics.json"
LOMO = "metal_lomo_auc_wasserstein.json"
EXT = "msrs_external_dataset.json"
ENS = "msrs_seed_ensemble.json"
FEXP = "msrs_feature_expansion_5seed.json"
FFUS = "msrs_feature_fusion.json"
DDIR = "msrs_delta_direct.json"

PROTOCOLS = ("ligand", "doi", "scaffold", "year")
ABL_ARMS = (
    "full",
    "no_metal_code",
    "no_state",
    "no_coord",
    "no_state_coord",
    "metal_blind",
    "no_complex",
    "cell_blind",
    "ligand_only",
)
SEED_ARMS = ("lgb_tree", "gnn_metal_state", "blend_equal", "blend_foldwise", "stack_foldwise")
CAL_VARIANTS = ("global", "metal", "difficulty")


@dataclass
class Row:
    label: str
    cells: list[Cell]
    unit: str = ""
    tag: str = "MEASURED"

    def source(self) -> str:
        parts = [c.source() for c in self.cells if c.source()]
        return "; ".join(parts) if parts else "-"


@dataclass
class Table:
    ident: str
    title: str
    columns: list[str]
    rows: list[Row] = field(default_factory=list)
    note: str = ""
    show_unit: bool = False

    def header(self) -> list[str]:
        cols = ["Metric"] + (["Unit"] if self.show_unit else []) + self.columns + ["Source", "Tag"]
        return cols


def table_main() -> Table:
    t = Table(
        "T1_main_predictive",
        "Main predictive metrics",
        ["Value"],
        show_unit=True,
    )
    spec = [
        ("Canonical R2, stack (5-seed mean)", P(SEED, "/splits/canonical_grouped/arms/stack_foldwise/r2/mean"), "R2"),
        ("Canonical R2, stack (5-seed SD)", P(SEED, "/splits/canonical_grouped/arms/stack_foldwise/r2/std"), "R2"),
        ("Canonical R2, LightGBM tree (5-seed mean)", P(SEED, "/splits/canonical_grouped/arms/lgb_tree/r2/mean"), "R2"),
        ("Canonical R2, D-MPNN alone (5-seed mean)", P(SEED, "/splits/canonical_grouped/arms/gnn_metal_state/r2/mean"), "R2"),
        ("R2, ligand protocol", P(GEN, "/splits/ligand/arms/ligand+both/r2"), "R2"),
        ("R2, DOI protocol", P(GEN, "/splits/doi/arms/ligand+both/r2"), "R2"),
        ("R2, scaffold protocol", P(GEN, "/splits/scaffold/arms/ligand+both/r2"), "R2"),
        ("R2, year protocol", P(GEN, "/splits/year/arms/ligand+both/r2"), "R2"),
        ("dR2, DOI minus ligand", P(GEN, "/delta_r2_doi_vs_ligand"), "dR2"),
        ("Delta sign accuracy, DOI at |d|>=0.2091", P(DELTA, "/protocols/doi/thresholds/0.2091/sign_accuracy"), "accuracy"),
        ("MAE(delta), DOI", P(DELTA, "/protocols/doi/thresholds/0.2091/mae_delta"), "log10 IC50"),
        ("Decision regret, DOI full coverage", P(DEC, "/protocols/doi/full_coverage/model/mean_regret"), "log10 IC50"),
        ("Top-1 accuracy, DOI full coverage", P(DEC, "/protocols/doi/full_coverage/model/top1_accuracy"), "accuracy"),
        ("LOMO AUC (5-fold x 5-seed)", P(LOMO, "/value"), "ROC-AUC"),
    ]
    for label, cell, unit in spec:
        t.rows.append(Row(label, [cell], unit=unit))
    return t


def table_generalization() -> Table:
    t = Table(
        "T2_generalization",
        "Four-protocol generalization matrix (arm ligand+both)",
        ["Groups", "Rows scored", "R2", "95% CI", "dR2 vs ligand"],
        note="Each protocol holds out a different unit: ligand / DOI / Bemis-Murcko scaffold / publication year.",
    )
    for proto in PROTOCOLS:
        base = f"/splits/{proto}"
        t.rows.append(
            Row(
                proto,
                [
                    P(GEN, f"{base}/n_groups", "int"),
                    P(GEN, f"{base}/n_rows_evaluated", "int"),
                    P(GEN, f"{base}/arms/ligand+both/r2"),
                    P(GEN, f"{base}/arms/ligand+both/bootstrap_r2/ci95", "ci"),
                    D(GEN, f"{base}/arms/ligand+both/r2", "/splits/ligand/arms/ligand+both/r2"),
                ],
            )
        )
    return t


def table_calibration() -> Table:
    t = Table(
        "T3_calibration",
        "Conformal calibration (alpha=0.10, nominal 90%)",
        ["Coverage", "Median width", "Winkler", "PIT KS", "Slope", "R2(fit)", "Bias"],
        note="Coverage alone is guaranteed by the conformal construction; width and PIT are the informative columns.",
    )
    for proto in ("doi", "ligand"):
        base = f"/protocols/{proto}"
        t.rows.append(
            Row(
                f"{proto} / point calibration",
                [
                    NONE(),
                    NONE(),
                    NONE(),
                    NONE(),
                    P(CAL, f"{base}/point_calibration/slope"),
                    P(CAL, f"{base}/point_calibration/r2_fit"),
                    P(CAL, f"{base}/point_calibration/mean_bias"),
                ],
            )
        )
        for variant in CAL_VARIANTS:
            vb = f"{base}/variants/{variant}"
            t.rows.append(
                Row(
                    f"{proto} / {variant}",
                    [
                        P(CAL, f"{vb}/coverage"),
                        P(CAL, f"{vb}/median_width"),
                        P(CAL, f"{vb}/winkler"),
                        P(CAL, f"{vb}/pit/ks_stat"),
                        NONE(),
                        NONE(),
                        NONE(),
                    ],
                )
            )
    return t


def table_mechanism() -> Table:
    t = Table(
        "T4_mechanism",
        "Metal ablation, LightGBM (R2 and paired group bootstrap)",
        ["R2", "dR2 vs full", "95% CI", "p (one-sided)", "Features"],
        note="Permutation nulls are in msrs_metal_ablation.json: metal_label p and metal_block p are both 0.0099 at n_perm=100.",
    )
    for proto in ("ligand", "doi"):
        for arm in ABL_ARMS:
            base = f"/protocols/{proto}/arms/{arm}"
            is_full = arm == "full"
            t.rows.append(
                Row(
                    f"{proto} / {arm}",
                    [
                        P(ABL, f"{base}/r2"),
                        NONE() if is_full else P(ABL, f"{base}/delta_r2_vs_full"),
                        NONE() if is_full else P(ABL, f"{base}/delta_ci/ci95", "ci"),
                        NONE() if is_full else P(ABL, f"{base}/delta_ci/p_one_sided", "f4"),
                        P(ABL, f"{base}/n_features", "int"),
                    ],
                )
            )
    return t


def table_decision() -> Table:
    t = Table(
        "T5_decision",
        "Decision curve: model vs global_best at two coverage levels",
        ["Mean regret", "Top-1", "NB(0.5)", "p (Holm)"],
        note="Net benefit uses hit = regret <= 0.2091 and cost ratio p_t/(1-p_t).",
    )
    for proto in ("doi", "ligand"):
        for scorer in ("model", "global_best"):
            base = f"/protocols/{proto}/curve[coverage=0.5]/{scorer}"
            t.rows.append(
                Row(
                    f"{proto} / cov=0.5 / {scorer}",
                    [
                        P(DEC, f"{base}/mean_regret"),
                        P(DEC, f"{base}/top1_accuracy"),
                        P(DEC, f"{base}/net_benefit/0.5"),
                        P(DEC, f"/protocols/{proto}/curve[coverage=0.5]/model_vs_global_best/p_holm")
                        if scorer == "model"
                        else NONE(),
                    ],
                )
            )
        for scorer in ("model", "global_best"):
            base = f"/protocols/{proto}/full_coverage/{scorer}"
            t.rows.append(
                Row(
                    f"{proto} / cov=1.0 / {scorer}",
                    [
                        P(DEC, f"{base}/mean_regret"),
                        P(DEC, f"{base}/top1_accuracy"),
                        P(DEC, f"/protocols/{proto}/full_coverage_net_benefit/{scorer}/0.5"),
                        NONE(),
                    ],
                )
            )
    return t


def table_power() -> Table:
    t = Table(
        "T6_power",
        "Minimum detectable effect at 80% power, alpha=0.05 one-sided",
        ["Observed", "95% CI", "MDE", "n clusters", "n required", "Powered", "Target effect"],
        note="MDE = 2.4865 * sd, sd = CI width / (2*1.96). A null is only interpretable against its MDE.",
    )
    doc = ArtifactCache(METRICS_DIR).get(POWER) or {}
    for name in (doc.get("metrics") or {}):
        base = f"/metrics/{name}"
        t.rows.append(
            Row(
                name,
                [
                    P(POWER, f"{base}/observed"),
                    P(POWER, f"{base}/ci95", "ci"),
                    P(POWER, f"{base}/mde_80pct"),
                    P(POWER, f"{base}/n_groups", "int"),
                    P(POWER, f"{base}/n_required_for_observed", "int"),
                    P(POWER, f"{base}/powered_for_observed", "raw"),
                    P(POWER, f"{base}/target_effect", optional=True),
                ],
            )
        )
    return t


def table_generation() -> Table:
    t = Table(
        "T7_generation",
        "Generation-side metrics (66 survivors from 192 attempts)",
        ["Value", "Target", "Passed"],
        note="Novelty is 1 - max Tanimoto to the CFM train split (Morgan r=2, 1024 bits).",
    )
    spec = [
        ("Yield from attempts", "/cohort/yield_from_attempts", "", "", "f4"),
        ("Parse rate", "/cohort/parse_rate", "", "", "f4"),
        ("Uniqueness", "/cohort/uniqueness", "", "", "f4"),
        ("Mean max Tanimoto", "/novelty/mean_max_tanimoto", "", "", "f4"),
        ("Mean novelty", "/checks/mean_novelty/value", "/checks/mean_novelty/target", "/checks/mean_novelty/passed", "f4"),
        (
            "Fraction max T < 0.9",
            "/checks/novelty_fraction_at_0.9/value",
            "/checks/novelty_fraction_at_0.9/target",
            "/checks/novelty_fraction_at_0.9/passed",
            "f4",
        ),
        ("IntDiv1", "/checks/intdiv1/value", "/checks/intdiv1/target", "/checks/intdiv1/passed", "f4"),
        ("SA", "/checks/sa/value", "/checks/sa/target", "/checks/sa/passed", "f4"),
        ("QED", "/checks/qed/value", "", "", "f4"),
        ("Metal compliance", "/metal_compliance/compliance", "", "", "f4"),
        ("Docking status", "/checks/docking/status", "", "/checks/docking/status", "raw"),
    ]
    for label, value_ptr, target_ptr, passed_ptr, fmt in spec:
        t.rows.append(
            Row(
                label,
                [
                    P(GENM, value_ptr, fmt),
                    P(GENM, target_ptr, fmt) if target_ptr else NONE(),
                    P(GENM, passed_ptr, "raw") if passed_ptr else NONE(),
                ],
            )
        )
    t.rows.append(
        Row(
            # The panel of record's OWN rate.  Every other row of this table
            # is over the same 66-molecule panel, so the SA row must be too:
            # the mixed-cohort rate (2,396 molecules from three training
            # cohorts of two orders of magnitude) is not this panel's rate,
            # and the label names the cohort it is about.
            "SA<=4.5 yield (panel of record, 66 survivors)",
            [
                P("cfm_sa_guided.json", "/by_pool/panel_of_record/rate_sa_le_gate", "f4"),
                NONE(),
                NONE(),
            ],
        )
    )
    return t


def table_strong_generalization() -> Table:
    """The post-STR-4 matrix: ECFP4 tree / MACCS+RDKit tree / +GNN fusion.

    T2 reports the original single-seed ECFP4 tree.  This table reports the
    five-seed arms, so a reader can see that the DOI gain comes from the
    feature block (0.3380 -> 0.3767) and then from fusion (0.3811) without
    having to reconcile two different seeds.
    """
    t = Table(
        "T11_generalization_strong",
        "Five-seed generalization matrix: ECFP4 tree, MACCS+RDKit tree, GNN fusion",
        ["ECFP4 tree", "MACCS+RDKit tree", "MACCS+GNN fusion", "dR2 (MACCS - ECFP4)"],
        note="All three arms are seed-averaged over the same five seeds; DOI R2 gate is 0.36. The dR2 column is the arithmetic difference of the two columns - the paired DOI-clustered delta lives in the artifact.",
    )
    for proto in PROTOCOLS:
        t.rows.append(
            Row(
                proto,
                [
                    P(ENS, f"/protocols/{proto}/ensemble/r2"),
                    P(FEXP, f"/protocols/{proto}/r2_seed_averaged"),
                    P(FFUS, f"/protocols/{proto}/fused_r2"),
                    DX(
                        FEXP,
                        f"/protocols/{proto}/r2_seed_averaged",
                        ENS,
                        f"/protocols/{proto}/ensemble/r2",
                    ),
                ],
            )
        )
    return t


def table_ranking() -> Table:
    """The delta-sign line under DOI holdout, all three routes that were tried."""
    t = Table(
        "T12_ranking",
        "Delta-sign under DOI holdout: three routes, one gate (0.65)",
        ["Sign@|d|>=0.2091", "95% CI", "MAE(d)", "Pairs"],
        note="Differencing and fusion have no DOI-clustered interval for the sign in their artifacts; direct regression does.",
    )
    delta = "/protocols/doi/ensemble/delta/0.2091"
    t.rows.append(
        Row(
            "differencing two level predictions",
            [
                P(ENS, f"{delta}/sign_accuracy"),
                P(ENS, f"{delta}/bootstrap/sign_accuracy/ci95", "ci"),
                P(ENS, f"{delta}/mae_delta"),
                P(ENS, f"{delta}/n_pairs", "int"),
            ],
        )
    )
    t.rows.append(
        Row(
            "fusion (MACCS tree + GNN)",
            [
                P(FFUS, "/protocols/doi/delta_sign/sign_accuracy"),
                NONE(),
                P(FFUS, "/protocols/doi/delta_sign/mae_delta"),
                P(FFUS, "/protocols/doi/delta_sign/n_pairs", "int"),
            ],
        )
    )
    t.rows.append(
        Row(
            "direct delta regression",
            [
                P(DDIR, "/gated/sign_accuracy"),
                P(DDIR, "/gated/bootstrap/sign_accuracy/ci95", "ci"),
                P(DDIR, "/gated/mae_delta"),
                P(DDIR, "/gated/n_pairs", "int"),
            ],
        )
    )
    return t


def table_data() -> Table:
    t = Table(
        "T8_data",
        "Corpus and split contamination",
        ["Value"],
        note="Cross-fold rate = share of test rows whose key value also appears in that fold's training rows.",
    )
    spec = [
        ("Rows in source corpus", P(CARD, "/corpus/n_rows_total", "int")),
        ("Potency rows", P(CARD, "/corpus/potency_rows", "int")),
        ("Model rows after filter", P(CARD, "/corpus/model_rows", "int")),
        ("Distinct metals", P(CARD, "/corpus/n_metals", "int")),
        ("Distinct ligands", P(CARD, "/corpus/n_ligands", "int")),
        ("Distinct cell lines", P(CARD, "/corpus/n_cell_lines", "int")),
        ("Distinct DOIs", P(CARD, "/corpus/n_dois", "int")),
        ("Rows with metal in SMILES", P(CARD, "/corpus/metal_in_smiles_rows", "int")),
        ("Right-censored potency rows dropped", P(CARD, "/corpus/potency_censoring_counts/right", "int")),
        ("Repeated keys", P(CARD, "/duplicates/n_keys_repeated", "int")),
        ("Rows in repeated keys", P(CARD, "/duplicates/n_rows_in_repeated_keys", "int")),
    ]
    for proto in PROTOCOLS:
        spec.append((f"{proto}: ligand seen in train", P(CARD, f"/protocols/{proto}/ligand_cross_fold/rate", "pct")))
        spec.append((f"{proto}: DOI seen in train", P(CARD, f"/protocols/{proto}/doi_cross_fold/rate", "pct")))
        spec.append((f"{proto}: cell line seen in train", P(CARD, f"/protocols/{proto}/cell_cross_fold/rate", "pct")))
    for label, cell in spec:
        t.rows.append(Row(label, [cell]))
    return t


def table_seed() -> Table:
    t = Table(
        "T9_seed_stability",
        "Five-seed stability, canonical ligand-grouped split",
        ["R2 mean", "R2 SD", "MAE mean", "MAE SD", "dR2 vs tree", "p (one-sided)"],
        note="Folds are fixed by the ligand key; only model seeds vary.",
    )
    for arm in SEED_ARMS:
        base = f"/splits/canonical_grouped/arms/{arm}"
        is_tree = arm == "lgb_tree"
        t.rows.append(
            Row(
                arm,
                [
                    P(SEED, f"{base}/r2/mean"),
                    P(SEED, f"{base}/r2/std"),
                    P(SEED, f"{base}/mae/mean"),
                    P(SEED, f"{base}/mae/std"),
                    NONE() if is_tree else P(SEED, f"{base}/vs_tree/mean_delta"),
                    NONE() if is_tree else P(SEED, f"{base}/vs_tree/p_one_sided", "f6"),
                ],
            )
        )
    return t


def table_external() -> Table:
    t = Table(
        "T10_external",
        "External cohort availability (MET-7)",
        ["Value"],
        note="The pre-registered estimand is same ligand + same metal + same cell line. Metal is the key that reaches zero.",
    )
    spec = [
        ("Verdict", P(EXT, "/verdict", "raw")),
        ("Metal intersection", P(EXT, "/intersection/metal", "int")),
        ("Ligand main-fragment intersection", P(EXT, "/intersection/ligand", "int")),
        ("Cell-line intersection", P(EXT, "/intersection/cell_line", "int")),
        ("Triple same-metal rows", P(EXT, "/intersection/triple_same_metal_rows", "int")),
        (
            "PlatinAI complexes sharing a ligand",
            P(EXT, "/candidates/platin_ai_4cell/complexes_with_a_shared_ligand", "int"),
        ),
        (
            "Labels on shared cells, ignoring metal",
            P(EXT, "/intersection/label_cells_available_ignoring_metal", "int"),
        ),
        ("NCI-60 usable rows", P(EXT, "/candidates/nci60_gi50/usable_rows", "int")),
    ]
    for label, cell in spec:
        t.rows.append(Row(label, [cell]))
    return t


TABLES = (
    table_main,
    table_generalization,
    table_calibration,
    table_mechanism,
    table_decision,
    table_power,
    table_generation,
    table_data,
    table_seed,
    table_external,
    table_strong_generalization,
    table_ranking,
)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _row_values(
    row: Row, cache: ArtifactCache, missing: list[str], absent: list[str]
) -> list[str]:
    out = []
    for cell in row.cells:
        text, status, where = cell.value(cache)
        if status == "missing":
            missing.append(where)
        elif status == "absent":
            absent.append(where)
        out.append(text)
    return out


def render_md(table: Table, cache: ArtifactCache) -> tuple[str, list[str], list[str]]:
    header = table.header()
    # No timestamp in the body: it would make every rebuild a diff.  The
    # generation time and the source hashes live in manifest.json.
    lines = [
        f"# {table.title}",
        "",
        "Generated from `metrics/by_metric/*.json`; see `manifest.json` for the timestamp and source hashes.",
        "",
    ]
    if table.note:
        lines.extend([table.note, ""])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    missing: list[str] = []
    absent: list[str] = []
    for row in table.rows:
        values = _row_values(row, cache, missing, absent)
        cells = [row.label]
        if table.show_unit:
            cells.append(row.unit or "-")
        cells.extend(values)
        cells.extend([row.source(), row.tag])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines), missing, absent


def render_csv(table: Table, cache: ArtifactCache) -> tuple[str, list[str], list[str]]:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(table.header())
    missing: list[str] = []
    absent: list[str] = []
    for row in table.rows:
        values = _row_values(row, cache, missing, absent)
        out = [row.label]
        if table.show_unit:
            out.append(row.unit or "-")
        out.extend(values)
        out.extend([row.source(), row.tag])
        writer.writerow(out)
    return buf.getvalue(), missing, absent


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (e.g. a tmp out-dir)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", default=None, help="table id prefix, repeatable")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any cell is MISSING")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cache = ArtifactCache(METRICS_DIR)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"generated_utc": now, "tables": [], "sources": {}}
    total_missing = 0
    total_absent = 0
    for build in TABLES:
        table = build()
        if args.only and not any(table.ident.startswith(p) for p in args.only):
            continue
        md, missing_md, absent_md = render_md(table, cache)
        csv_text, missing_csv, absent_csv = render_csv(table, cache)
        missing = sorted(set(missing_md) | set(missing_csv))
        absent = sorted(set(absent_md) | set(absent_csv))
        total_missing += len(missing)
        total_absent += len(absent)
        md_path = args.out_dir / f"{table.ident}.md"
        csv_path = args.out_dir / f"{table.ident}.csv"
        atomic_write_file(md_path, md)
        atomic_write_file(csv_path, csv_text)
        manifest["tables"].append(
            {
                "id": table.ident,
                "title": table.title,
                "rows": len(table.rows),
                "columns": table.header(),
                "markdown": _display_path(md_path),
                "csv": _display_path(csv_path),
                "missing": missing,
                "optional_absent": absent,
            }
        )
        status = "PASS" if not missing else f"MISSING {len(missing)}"
        if absent:
            status += f" (optional absent: {len(absent)})"
        print(f"[paper_metric_tables] {table.ident}: {len(table.rows)} rows, {status}")

    manifest["sources"] = cache.sources()
    manifest["missing_total"] = total_missing
    manifest["optional_absent_total"] = total_absent

    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        # Keep the old timestamp when nothing but the clock changed, so a
        # rebuild is byte-identical and does not dirty the working tree.
        try:
            old = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            old = {}
        if {k: v for k, v in old.items() if k != "generated_utc"} == {
            k: v for k, v in manifest.items() if k != "generated_utc"
        }:
            manifest["generated_utc"] = old.get("generated_utc", now)
    atomic_write_file(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"[paper_metric_tables] wrote {len(manifest['tables'])} tables to "
        f"{_display_path(args.out_dir)} (missing cells: {total_missing})"
    )
    if args.strict and total_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
