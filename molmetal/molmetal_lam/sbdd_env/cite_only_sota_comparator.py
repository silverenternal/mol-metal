"""Cite-only SOTA comparator for paper §4 SOTA-comparison column.

Background
----------
TODO-22 (data-gap alignment plan) §4c ships a 9-row cite-only SOTA
table at ``molmetal/reports/wf_3_citeonly_sota.tex`` plus 7
protocol-mismatch flags (M1)-(M7).  This module makes the same
data programmatically queryable so the r4_c_full_sweep driver can
emit per-pocket SOTA-comparison rows WITHOUT re-running any SOTA
baseline.

Honest framing
--------------
* All 9 SOTA rows are CITED-ONLY (taken from the original papers,
  NOT re-run by Mol-Metal).
* ``compare(num_samples, protocol_flags)`` returns per-row gap
  information (our_value - target_value) plus the protocol-mismatch
  flags that apply.  Direct head-to-head deltas are NOT supported
  when the protocol-mismatch flags differ from the user's setup.
* The comparator is comparison vs literature, NOT head-to-head.

Architecture
------------
* Hardcoded 9-row inventory is the authoritative source (parsed
  directly from ``wf_3_citeonly_sota.tex`` at design time and
  embedded as a Python dataclass list).  This avoids coupling the
  paper-render path to LaTeX-table parsing.
* ``_parse_tex_rows`` provides a parser for the LaTeX table as a
  consistency check — if the .tex is hand-edited the parser should
  produce the same 9 rows; if not, the discrepancy is surfaced.

Public surface
--------------
* ``CiteOnlySOTAComparator`` (class)
    * ``rows`` — list of ``SOTARow`` dataclass (9 items)
    * ``flags`` — list of 7 ``ProtocolMismatchFlag`` dataclass
    * ``compare(our_value, num_samples, protocol_flags) -> ComparisonReport``
    * ``row_by_name(name) -> SOTARow | None``
* ``SOTARow`` (dataclass)
    paper, dataset, docking_engine, validity, novelty, diversity,
    vina_mean, vina_std, sa, qed, pb_pass, n_pockets, n_seeds,
    is_ours=False
* ``ProtocolMismatchFlag`` (dataclass)
    code, description, applies_when
* ``ComparisonReport`` (dataclass)
    row, gap_vina, gap_sa, gap_qed, gap_pb,
    applicable_flags, n_samples_ours, n_samples_target,
    fairness_verdict, notes
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Protocol-mismatch flags (M1)-(M7) from wf_3_citeonly_sota.tex §2
# ---------------------------------------------------------------------------
# These flags document systematic differences between SOTA protocols
# and the Mol-Metal Lambda arm protocol.  ``applies_when`` is a
# predicate that returns True if the flag applies to a given SOTA row
# given the user's protocol_flags.
DEFAULT_PROTOCOL_FLAGS = {
    "crossdocked_luo2021": True,  # use Luo 2021 30%-seq-id split
    "docking_engine": "vina",  # one of vina/qvina/diffdock-l/quickvina2
    "validity_def": "rdkit_only",  # one of rdkit_only/pb_augmented/pb_only
    "novelty_def": "tanimoto_lt_0.4",
    "diversity_def": "tanimoto_morgan_r2",
    "n_seeds": 1,  # number of seeds per pocket
    "n_pockets": 100,  # number of pockets
}


@dataclass(frozen=True)
class ProtocolMismatchFlag:
    code: str
    description: str
    applies_when: str  # human-readable predicate

    def to_dict(self) -> dict:
        return {"code": self.code, "description": self.description,
                "applies_when": self.applies_when}


PROTOCOL_MISMATCH_FLAGS: List[ProtocolMismatchFlag] = [
    ProtocolMismatchFlag("M1", "CrossDocked2020 split version (Luo 2021 vs SPINDR vs PDBbind time-split).",
                         "row.dataset != 'CrossDocked2020 (Luo 2021)'"),
    ProtocolMismatchFlag("M2", "Docking engine and version (QVina vs QuickVina 2 vs DiffDock-L vs Vina 1.2.7).",
                         "row.docking_engine != user.engine"),
    ProtocolMismatchFlag("M3", "Validity definition (RDKit-only vs PB-augmented); structurally incomparable.",
                         "row.validity != user.validity_def"),
    ProtocolMismatchFlag("M4", "Novelty definition (random vs scaffold vs time split); random-split numbers inflated.",
                         "row.novelty != user.novelty_def"),
    ProtocolMismatchFlag("M5", "Diversity definition (Tanimoto-based vs Homotype); incomparable on constitutional isomers.",
                         "row.diversity != user.diversity_def"),
    ProtocolMismatchFlag("M6", "Number of seeds (1 vs >=3); per-seed sigma ~ 0.46 kcal/mol at n_docked=4.",
                         "row.n_seeds != user.n_seeds"),
    ProtocolMismatchFlag("M7", "Pocket count (single vs >=10 vs 100); pocket-specific variance ~ +-2 kcal/mol.",
                         "row.n_pockets != user.n_pockets"),
]


# ---------------------------------------------------------------------------
# SOTA row inventory — parsed from wf_3_citeonly_sota.tex at design
# time.  Values are cited-only and ARE NOT measured by Mol-Metal.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SOTARow:
    paper: str
    dataset: str
    docking_engine: str
    validity: str
    novelty: str
    diversity: str
    vina_mean: Optional[float]  # kcal/mol; lower is more negative = better
    vina_std: Optional[float]
    sa: Optional[float]  # lower is better
    qed: Optional[float]  # higher is better
    pb_pass: Optional[float]  # higher is better
    n_pockets: int
    n_seeds: int
    is_ours: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Hardcoded 9-row SOTA inventory.  All values are cited-only.
# vina_mean is parsed from "$-X.XX \\pm$ NA" patterns in the .tex;
# std is None unless explicitly given.
SOTA_ROWS: List[SOTARow] = [
    SOTARow(
        paper="DiffSBDD (2023, ICML)",
        dataset="CrossDocked2020 (Luo 2021)",
        docking_engine="QVina (exh=8)",
        validity="RDKit-only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-7.62, vina_std=None,
        sa=2.81, qed=None, pb_pass=0.246,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="Pocket2Mol (2022, ICML)",
        dataset="CrossDocked2020 (Luo 2021)",
        docking_engine="QuickVina 2 (exh=16)",
        validity="RDKit-only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-7.07, vina_std=None,
        sa=2.51, qed=0.55, pb_pass=0.498,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="TargetDiff (2023, ICLR)",
        dataset="CrossDocked2020 (Luo 2021)",
        docking_engine="QVina (exh=8, UFF)",
        validity="RDKit-only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-8.45, vina_std=None,
        sa=2.65, qed=0.48, pb_pass=0.351,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="MolDiff (2023, AAAI)",
        dataset="CrossDocked2020 (Luo 2021)",
        docking_engine="QVina (exh=8)",
        validity="RDKit-only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-7.55, vina_std=None,
        sa=2.71, qed=0.51, pb_pass=0.291,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="DecompDiff (2024, ICLR)",
        dataset="CrossDocked2020 (Luo 2021)",
        docking_engine="QVina (exh=8)",
        validity="RDKit-only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-8.39, vina_std=None,
        sa=2.71, qed=None, pb_pass=0.390,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="FLOWr (2025, Nat Comput Sci)",
        dataset="CrossDocked2020 (SPINDR)",
        docking_engine="DiffDock-L pose ranker",
        validity="PoseBusters only",
        novelty="Tanimoto <0.4",
        diversity="Tanimoto (Morgan r=2)",
        vina_mean=-6.93, vina_std=None,
        sa=2.86, qed=None, pb_pass=0.94,
        n_pockets=100, n_seeds=1,
    ),
    SOTARow(
        paper="DiffDock (2022, ICML / 2024 ICLR)",
        dataset="PDBbind 2019 (time)",
        docking_engine="DiffDock-L conf.-ranked",
        validity="RMSD-to-native <2 A",
        novelty="out-of-PDBbind",
        diversity="pose-mode coverage",
        vina_mean=None, vina_std=None,
        sa=None, qed=None, pb_pass=None,
        n_pockets=363, n_seeds=10,
    ),
    SOTARow(
        paper="BindNet (2024, NeurIPS)",
        dataset="PDBbind v2020 (time)",
        docking_engine="Vina score head",
        validity="RMSD-to-native <2 A",
        novelty="out-of-PDBbind",
        diversity="ligand-mode coverage",
        vina_mean=None, vina_std=None,
        sa=None, qed=None, pb_pass=None,
        n_pockets=340, n_seeds=3,
    ),
    SOTARow(
        paper="RoseTTAFold-AA (2024, Nat Methods)",
        dataset="PDB + AF2 clusters",
        docking_engine="side-chain rotamer",
        validity="all-atom clash-free",
        novelty="sequence-only",
        diversity="backbone RMSD",
        vina_mean=None, vina_std=None,
        sa=None, qed=None, pb_pass=None,
        n_pockets=150, n_seeds=5,
    ),
]


# ---------------------------------------------------------------------------
# Comparator result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GapMetrics:
    """Per-metric gap (our_value - target_value)."""
    vina: Optional[float] = None
    sa: Optional[float] = None
    qed: Optional[float] = None
    pb: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class ComparisonReport:
    """Per-row comparison result from ``CiteOnlySOTAComparator.compare``."""
    row: SOTARow
    our_value: float  # Vina score (our_value)
    gap: GapMetrics
    applicable_flags: List[str]  # list of M1-M7 codes that apply
    n_samples_ours: int
    n_samples_target: int
    fairness_verdict: str  # 'comparable', 'flag_only', 'incomparable'
    notes: str

    def to_dict(self) -> dict:
        return {
            "row": self.row.to_dict(),
            "our_value": self.our_value,
            "gap": self.gap.to_dict(),
            "applicable_flags": list(self.applicable_flags),
            "n_samples_ours": self.n_samples_ours,
            "n_samples_target": self.n_samples_target,
            "fairness_verdict": self.fairness_verdict,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Comparator class
# ---------------------------------------------------------------------------
class CiteOnlySOTAComparator:
    """Cite-only SOTA comparator.

    Holds the 9 SOTA rows + 7 protocol-mismatch flags.  Computes
    per-row gaps and emits a per-pocket comparison row.
    """

    DEFAULT_TEX_PATH = ("/home/hugo/codes/try_triton_on_rocm/molmetal/reports/"
                        "wf_3_citeonly_sota.tex")

    def __init__(self, tex_path: Optional[str] = None,
                 rows: Optional[List[SOTARow]] = None,
                 flags: Optional[List[ProtocolMismatchFlag]] = None):
        """Initialise the comparator.

        ``tex_path`` — optional LaTeX file path.  If supplied, the
        parser is invoked as a sanity check that the hardcoded
        inventory matches the on-disk .tex file.  Discrepancies are
        recorded but do NOT raise (the hardcoded inventory is the
        authoritative source).

        ``rows`` — override the SOTA row inventory (mostly for tests).

        ``flags`` — override the protocol-mismatch flag list.
        """
        self.rows: List[SOTARow] = list(rows) if rows is not None else list(SOTA_ROWS)
        self.flags: List[ProtocolMismatchFlag] = list(flags) if flags is not None else list(PROTOCOL_MISMATCH_FLAGS)
        self._parser_discrepancies: List[str] = []
        if tex_path is not None and Path(tex_path).is_file():
            self._parser_discrepancies = self._cross_check_with_tex(tex_path)

    @property
    def parser_discrepancies(self) -> List[str]:
        """Discrepancies between the hardcoded inventory and the .tex file."""
        return list(self._parser_discrepancies)

    def _cross_check_with_tex(self, tex_path: str) -> List[str]:
        """Run the .tex parser and compare row count to hardcoded.

        Returns a list of discrepancy strings; empty list means the
        parser agrees with the hardcoded inventory.  This is a
        *consistency* check, not a correctness gate.
        """
        try:
            parsed_rows = _parse_tex_rows(tex_path)
        except Exception as exc:
            return [f"parser failure: {exc}"]
        discrepancies: List[str] = []
        if len(parsed_rows) != len(self.rows):
            discrepancies.append(
                f"row count mismatch: hardcoded={len(self.rows)} parsed={len(parsed_rows)}"
            )
        return discrepancies

    def row_by_name(self, name: str) -> Optional[SOTARow]:
        """Find a SOTA row by paper name (case-insensitive substring match)."""
        lowered = name.lower()
        for row in self.rows:
            if lowered in row.paper.lower():
                return row
        return None

    def _applicable_flags(self, row: SOTARow,
                          protocol_flags: Dict[str, object]) -> List[str]:
        """Return the list of protocol-mismatch flag codes that apply.

        Heuristic:
        * M1: row.dataset does not mention 'Luo 2021' (different split)
        * M2: row.docking_engine does not match protocol_flags['docking_engine']
        * M3: row.validity does not match protocol_flags['validity_def']
        * M4: row.novelty does not match protocol_flags['novelty_def']
        * M5: row.diversity does not match protocol_flags['diversity_def']
        * M6: row.n_seeds != protocol_flags['n_seeds']
        * M7: row.n_pockets != protocol_flags['n_pockets']
        """
        codes: List[str] = []
        if "Luo 2021" not in row.dataset:
            codes.append("M1")
        user_engine = str(protocol_flags.get("docking_engine", "")).lower()
        # Normalise engine-name overlap: vina and qvina/qvina-quickvina2
        # share a per-engine family.  We compare on the *first word*.
        row_engine = row.docking_engine.lower().split()[0] if row.docking_engine else ""
        if user_engine and row_engine and not row_engine.startswith(user_engine.split()[0]):
            # DiffDock-L and DiffDock are not strictly Vina/QVina; flag M2.
            codes.append("M2")
        if (str(protocol_flags.get("validity_def", "")).lower()
                and protocol_flags.get("validity_def") != row.validity):
            codes.append("M3")
        if (str(protocol_flags.get("novelty_def", "")).lower()
                and protocol_flags.get("novelty_def") != row.novelty):
            codes.append("M4")
        if (str(protocol_flags.get("diversity_def", "")).lower()
                and protocol_flags.get("diversity_def") != row.diversity):
            codes.append("M5")
        user_seeds = protocol_flags.get("n_seeds")
        if isinstance(user_seeds, int) and user_seeds != row.n_seeds:
            codes.append("M6")
        user_pockets = protocol_flags.get("n_pockets")
        if isinstance(user_pockets, int) and user_pockets != row.n_pockets:
            codes.append("M7")
        return codes

    def _fairness_verdict(self, applicable: List[str],
                          our_value: float,
                          row: SOTARow) -> str:
        """Compute the fairness verdict.

        * 'comparable' — 0 protocol flags apply AND row has Vina score
        * 'flag_only' — 1-3 flags apply; cite-only context
        * 'incomparable' — 4+ flags apply; do NOT cite as comparison
        """
        if not applicable and row.vina_mean is not None:
            return "comparable"
        if len(applicable) <= 3:
            return "flag_only"
        return "incomparable"

    def _compute_gap(self, our_value: float, row: SOTARow,
                     our_sa: Optional[float] = None,
                     our_qed: Optional[float] = None,
                     our_pb: Optional[float] = None) -> GapMetrics:
        """Compute per-metric gaps (our_value - target_value).

        For vina: lower (more negative) is better, so a positive gap
        means we are WORSE than the target.  For SA: same convention
        (lower=better).  For QED/PB: higher=better, so a positive gap
        means we are BETTER.
        """
        gap_vina = (our_value - row.vina_mean) if row.vina_mean is not None else None
        gap_sa = (our_sa - row.sa) if (our_sa is not None and row.sa is not None) else None
        gap_qed = (our_qed - row.qed) if (our_qed is not None and row.qed is not None) else None
        gap_pb = (our_pb - row.pb_pass) if (our_pb is not None and row.pb_pass is not None) else None
        return GapMetrics(vina=gap_vina, sa=gap_sa, qed=gap_qed, pb=gap_pb)

    def compare(self, our_value: float, num_samples: int,
                protocol_flags: Optional[Dict[str, object]] = None,
                *, row_name: Optional[str] = None,
                our_sa: Optional[float] = None,
                our_qed: Optional[float] = None,
                our_pb: Optional[float] = None) -> List[ComparisonReport]:
        """Compare our_value against every (or one named) SOTA row.

        Parameters
        ----------
        our_value : float
            Our measured Vina score (kcal/mol) for this pocket/seed.
        num_samples : int
            Number of samples our_value is averaged over (per-pocket).
        protocol_flags : dict, optional
            The user's protocol description (engine, validity def,
            novelty def, diversity def, n_seeds, n_pockets).  See
            ``DEFAULT_PROTOCOL_FLAGS`` for the expected keys.
        row_name : str, optional
            If set, restrict comparison to the row whose paper name
            contains this substring (case-insensitive).  Otherwise
            compare against every row.
        our_sa, our_qed, our_pb : optional floats
            Optional other-metric values for the same pocket, used
            to fill in the multi-metric gap.

        Returns
        -------
        list of ComparisonReport
        """
        flags = protocol_flags if protocol_flags is not None else DEFAULT_PROTOCOL_FLAGS
        target_rows = self.rows
        if row_name is not None:
            matched = self.row_by_name(row_name)
            if matched is None:
                raise ValueError(f"No SOTA row matches {row_name!r}")
            target_rows = [matched]
        reports: List[ComparisonReport] = []
        for row in target_rows:
            applicable = self._applicable_flags(row, flags)
            gap = self._compute_gap(our_value, row, our_sa, our_qed, our_pb)
            verdict = self._fairness_verdict(applicable, our_value, row)
            notes_parts = [
                "cited-only; not re-run by Mol-Metal",
                f"{len(applicable)}/7 protocol-mismatch flags apply",
            ]
            if row.vina_mean is None:
                notes_parts.append("row uses non-Vina metric (e.g. RMSD); gap_vina undefined")
            report = ComparisonReport(
                row=row,
                our_value=our_value,
                gap=gap,
                applicable_flags=applicable,
                n_samples_ours=num_samples,
                n_samples_target=row.n_pockets * row.n_seeds,
                fairness_verdict=verdict,
                notes="; ".join(notes_parts),
            )
            reports.append(report)
        return reports

    def compare_per_pocket(self, pocket_id: str, our_value: float,
                           num_samples: int,
                           protocol_flags: Optional[Dict[str, object]] = None,
                           **kwargs) -> List[Dict[str, object]]:
        """Convenience wrapper that returns per-pocket CSV-ready dicts.

        Each dict has keys: pocket_id, paper, dataset, vina_target,
        gap_vina, applicable_flags, fairness_verdict.
        """
        reports = self.compare(our_value, num_samples, protocol_flags, **kwargs)
        return [
            {
                "pocket_id": pocket_id,
                "paper": r.row.paper,
                "dataset": r.row.dataset,
                "vina_target": r.row.vina_mean,
                "vina_std_target": r.row.vina_std,
                "gap_vina": r.gap.vina,
                "n_samples_ours": r.n_samples_ours,
                "n_samples_target": r.n_samples_target,
                "applicable_flags": ",".join(r.applicable_flags) if r.applicable_flags else "",
                "fairness_verdict": r.fairness_verdict,
                "notes": r.notes,
            }
            for r in reports
        ]


# ---------------------------------------------------------------------------
# LaTeX-table parser (consistency-check only)
# ---------------------------------------------------------------------------
# Pattern captures the 9 data rows of the cite-only table.
# Each row looks like:
#   DiffSBDD~\cite{...} (2023, ICML)         & CrossDocked2020 (Luo 2021)  & ...
#   DiffDock$^\dagger$ (2022, ICML) ...     & PDBbind 2019 (time) & ... & NA (RMSD<2...: 0.382 top-1) & ...
#
# Some rows use "NA" instead of "$-X.XX \pm$ NA" for vina_mean; the
# pattern handles both forms via the optional prefix.
TEX_ROW_PATTERN = re.compile(
    r"^(?P<paper>[^&]+?)&\s*"
    r"(?P<dataset>[^&]+?)&\s*"
    r"(?P<engine>[^&]+?)&\s*"
    r"(?P<validity>[^&]+?)&\s*"
    r"(?P<novelty>[^&]+?)&\s*"
    r"(?P<diversity>[^&]+?)&\s*"
    r"(?P<vina>[^&]+?)&\s*"
    r"(?P<sa>[^&]+?)&\s*"
    r"(?P<qed>[^&]+?)&\s*"
    r"(?P<pb>[^&]+?)&\s*"
    r"(?P<n_pockets>[^&]+?)\s*\\\\",
    re.MULTILINE,
)


def _parse_tex_rows(tex_path: str) -> List[Dict[str, str]]:
    """Parse the cite-only SOTA LaTeX table.

    Returns a list of dicts with the columns extracted as raw strings.
    The 9 data rows are matched; the header row (with ``\\textbf``)
    and the 2 Mol-Metal rows (``\\textbf{Mol-Metal...}``) are
    filtered out.
    """
    text = Path(tex_path).read_text()
    rows: List[Dict[str, str]] = []
    for m in TEX_ROW_PATTERN.finditer(text):
        paper = m.group("paper").strip()
        # Skip header row (starts with \\textbf and is the column-header)
        if paper.startswith("%") or "\\textbf{Dataset}" in paper:
            continue
        # Skip Mol-Metal rows (they are MEASURED rows, not CITED-ONLY)
        if "Mol-Metal" in paper or "textbf{Mol-Metal" in paper:
            continue
        rows.append({k: m.group(k).strip() for k in (
            "paper", "dataset", "engine", "validity", "novelty",
            "diversity", "vina", "sa", "qed", "pb", "n_pockets",
        )})
    return rows


def _safe_float(value: str) -> Optional[float]:
    """Parse a float from a LaTeX table cell; None for NA / empty."""
    v = (value or "").strip().replace("$", "").replace("\\", "")
    if not v or v.lower() == "na":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _safe_int(value: str) -> Optional[int]:
    """Parse an int from a LaTeX table cell (handles 'sim'150' prefixes)."""
    v = (value or "").strip()
    if not v or v.lower() == "na":
        return None
    m = re.search(r"\d+", v)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Convenience entry-point for CLI integration
# ---------------------------------------------------------------------------
def build_per_pocket_comparison_rows(pocket_id: str, our_value: float,
                                      num_samples: int,
                                      protocol_flags: Optional[Dict[str, object]] = None
                                      ) -> List[Dict[str, object]]:
    """One-shot helper for ``r4_c_full_sweep.py`` integration.

    Returns the CSV-ready per-row dicts from
    ``CiteOnlySOTAComparator.compare_per_pocket``.
    """
    cmp = CiteOnlySOTAComparator()
    return cmp.compare_per_pocket(pocket_id, our_value, num_samples, protocol_flags)


__all__ = [
    "CiteOnlySOTAComparator",
    "SOTARow",
    "ProtocolMismatchFlag",
    "ComparisonReport",
    "GapMetrics",
    "SOTA_ROWS",
    "PROTOCOL_MISMATCH_FLAGS",
    "DEFAULT_PROTOCOL_FLAGS",
    "build_per_pocket_comparison_rows",
    "_parse_tex_rows",
]
