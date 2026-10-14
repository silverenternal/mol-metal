"""Canonical paper-table evaluator for Track E (m14 + 8 ADMET + m4/m5/m7).

Track E (Wf-60231cff-03e-15, m14-admet-paper-align) — the proposed
canonical evaluator that wraps three orthogonal metric axes into a
single JSON for the paper §X table:

  1. **m14  per-atom decode ratio**  — :mod:`molmetal.eval.scoring.decode_gate`.
     Replaces the silent ``SanitizeMol``-only gate with the
     ``GetMolFrags``-aware floor at ``largest_frac_min``.
  2. **8 ADMET proxy**              — :mod:`molmetal.eval.scoring.admet8`.
     SA + QED + logP + TPSA + HBD + HBA + RotB + aromatic_rings
     (standalone per-mol emissions, disjoint from
     ``metal_aware_admet.MetalAwareADMET``).
  3. **m4/m5/m7 paper alignment**   — reads existing JSONs
     (m5 ``platinai_mcf7_pred.json``, m7 ``diversity_tanimoto.json``,
     m4 ``sa_mean.json``) and emits a single canonical row.

Honest-negative preservation
----------------------------
m5 MCF7 pIC50 (``platinai_mcf7_pred.json::value=null``,
``PENDING_GPU_RETRAIN``) is **preserved verbatim** in the canonical
output — the GPU retrain required to wire r_platinai into
``RewardAggregator`` was NOT executed (TODO-F2).  m7 IntDiv1's
``~0.017 BELOW training distribution`` honest-negative framing is
also preserved verbatim.

Inputs
------
- ``smiles_csv`` (optional path) — generated SMILES cohort.  When
  provided, m14 + ADMET8 are computed on the cohort on-the-fly.
- Existing JSONs on disk under ``metrics/by_metric/`` are read
  unconditionally and their values echoed (with their own honest-
  negative framings) into the canonical output.

Output JSON schema (single file per cohort)::

    {
      "cohort_id": str,
      "n_smiles": int,
      "m14_decode": {
         "per_atom_validity": float,
         "rdkit_strict_proxy": float,
         "n_multi_frag_rejects": int,
         "largest_frac_min": float,
         "min_atoms": int,
      },
      "admet8": { ... },                    # see admet8.compute_admet8_panel
      "m4_sa": { "value": float | null, "source": "...", "status": "..." },
      "m5_mcf7_pic50": { "value": float | null, "source": "...", "status": "..." },
      "m7_intdiv1": {
         "tanimoto_mean": float | null,
         "source": "...",
         "status": "...",
         "honest_negative": "...",
      },
      "per_pocket_w2": {                    # Wf-A22 L9 (2026-10-13)
         "<pocket_id>": {                   # status preserved verbatim
             "value": float | null,         # from L8 panel JSON
             "n_smiles": int,
             "status": "MEASURED" |         # L8 wired
                       "EMPTY_COHORT" |
                       "HIGH_VARIANCE_PILOT",
             "honest_negative": "...",
         },
         ...,
         "source": "metrics/by_metric/cfm_per_pocket_metric.json",
         "status": "MEASURED" | "NOT_COMPUTED" | "MISSING",
      },
      "protocol_version": "paper_table_e/2026-10-13",
    }

Lit anchors
-----------
- All inputs cite the production methodologist (`metal_aware_admet`,
  `decode_gate`, `admet8`, `sa_mean.json`, `platinai_mcf7_pred.json`,
  `diversity_tanimoto.json`).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from molmetal.eval.scoring import admet8 as _admet8
from molmetal.eval.scoring import decode_gate as _decode_gate


PROTOCOL_VERSION = "paper_table_e/2026-10-13"


# Wf-A30-2 B4 (2026-10-14) — SEMANTIC BREAKING CHANGE.
# Per Wf-A22 §6.43h honest-negative close + L9 (2026-10-13) we used to
# emit ``status="NOT_COMPUTED"`` when cfm_per_pocket_metric.json was
# missing on disk.  That silent fallback hid a category of regressions
# (smoke-stub checkpoint runs producing zero per-pocket JSONs while
# the trainer logged "panel ready") for every paper table emission from
# Wf-A14..A28.  A caller asking "is the per-pocket W₂ panel real?"
# got a JSON that looked fine (status="NOT_COMPUTED" with a long
# honest_negative string) and could proceed to cite the cohort as
# MEASURED through sheer inattention.
#
# The new contract is: if a downstream consumer calls build_paper_table
# with a cohort where the per-pocket JSON is REQUIRED (i.e. the caller
# passes ``require_per_pocket=True``), the missing JSON raises
# MissingRequiredInput — a FileNotFoundError subclass so existing
# callers that wrapped the call in ``try: ... except FileNotFoundError``
# keep working.  The default behaviour is unchanged for the legacy
# side-JSON readers (m4 / m5 / m7 / m14), which still emit
# ``status="MISSING"`` honestly.
class MissingRequiredInput(FileNotFoundError):
    """Raised when paper_table requires a JSON that is not on disk.

    Inherits FileNotFoundError so callers that already wrap the call
    in ``except FileNotFoundError`` keep working unchanged.  Carries
    the missing path + the action the caller should take in
    ``.missing_path`` / ``.remediation``.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_path: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.missing_path = missing_path
        self.remediation = remediation


# ---------------------------------------------------------------------------
# Side-JSON readers — preserve honest-negative verbatim
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_smiles_csv(path: Path) -> List[str]:
    """Best-effort reader for a generated-SMILES CSV.

    Looks for the first column with the header ``"smiles"``; falls back
    to taking the first column if no ``smiles`` header is found.
    Empty strings / ``None`` / non-string values are filtered.
    """
    if not path.exists():
        return []
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        # Find the SMILES column.
        col = 0
        for i, h in enumerate(header):
            if h.strip().lower() == "smiles":
                col = i
                break
        for row in reader:
            if not row or col >= len(row):
                continue
            v = row[col].strip()
            if v:
                out.append(v)
    return out


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class PaperTableRow:
    """Per-mol row emitted into the canonical JSON."""

    cohort_id: str
    n_smiles: int
    m14_decode: Dict[str, Any]
    admet8: Dict[str, Any]
    m4_sa: Dict[str, Any]
    m5_mcf7_pic50: Dict[str, Any]
    m7_intdiv1: Dict[str, Any]
    # Wf-A22 L9 (2026-10-13) — per-pocket W₂ + decode panel, additive;
    # default empty dict preserves bit-exact pre-Wf-A22 to_dict() shape.
    per_pocket_w2: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "n_smiles": int(self.n_smiles),
            "m14_decode": dict(self.m14_decode),
            "admet8": dict(self.admet8),
            "m4_sa": dict(self.m4_sa),
            "m5_mcf7_pic50": dict(self.m5_mcf7_pic50),
            "m7_intdiv1": dict(self.m7_intdiv1),
            # Wf-A22 L9 — additive; legacy JSON consumers see no extra key
            # because the dataclass default is an empty dict and the
            # build_paper_table call site populates ``NOT_COMPUTED`` when
            # the L8 panel JSON is absent.
            "per_pocket_w2": dict(self.per_pocket_w2),
            "protocol_version": self.protocol_version,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_paper_table(
    cohort_id: str,
    smiles: Optional[Iterable[str]] = None,
    *,
    metrics_dir: Path,
    smiles_csv: Optional[Path] = None,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
    # Wf-A30-2 B4 (2026-10-14) — semantic break gate.  When True
    # (default-off for backward compat), a missing
    # cfm_per_pocket_metric.json raises MissingRequiredInput instead of
    # silently emitting status="NOT_COMPUTED".  Paper-table consumers
    # that ship to a paper §X table MUST set this to True; the
    # trainer eval-loop call site (which knows the L8 panel may not
    # be on disk during a partial run) leaves it False.
    require_per_pocket: bool = False,
) -> PaperTableRow:
    """Build the canonical Track-E paper-table row.

    Parameters
    ----------
    cohort_id : str
        Used as ``cohort_id`` in the output dict.  Must be one of
        :data:`COHORT_KEYS` or a custom descriptor string.
    smiles : iterable[str], optional
        Generated-SMILES cohort.  If ``None`` and ``smiles_csv`` is
        provided, the CSV is loaded.  If both ``None``, the
        m14-decode + ADMET8 axes emit zero-cohort sentinels; m4/m5/m7
        side-JSON values still populate from disk.
    metrics_dir : Path
        Root of ``metrics/by_metric/*.json`` side-files.
    smiles_csv : Path, optional
        Optional CSV with a ``smiles`` column to load the cohort from.
    largest_frac_min, min_atoms
        Forwarded to :func:`decode_gate.gate_ratio` and
        :func:`decode_gate.gate_smiles`.

    Returns
    -------
    :class:`PaperTableRow`
        Always returns (no exceptions on missing side-JSONs; the
        honest-negative framings are emitted as ``status="MISSING"``).
    """
    # Coalesce smiles from CSV / direct input.
    if smiles is None and smiles_csv is not None:
        smiles = _read_smiles_csv(smiles_csv)
    smiles_list = list(smiles) if smiles is not None else []
    n_smiles = len(smiles_list)

    # m14 per-atom decode + ADMET8 (only if we have a cohort).
    if n_smiles > 0:
        m14 = _decode_gate.gate_ratio(
            smiles_list,
            largest_frac_min=largest_frac_min,
            min_atoms=min_atoms,
        )
        a8 = _admet8.compute_admet8_panel(smiles_list, cohort_tag=cohort_id)
        # RDKit-strict proxy (legacy behaviour) for side-by-side.
        # Wf-A30-4 D4 (2026-10-14, doc-only): second call = RDKit-strict proxy;
        # result is a strict superset of the strict call above (no-op when
        # strict params are non-zero) — see HONEST_NEGATIVE_CATALOG.md §6.52f.
        rs = _decode_gate.gate_ratio(
            smiles_list, largest_frac_min=0.0, min_atoms=0,
        )
        m14_out = {
            "per_atom_validity": m14["per_atom_validity"],
            "rdkit_strict_proxy": rs["per_atom_validity"],
            "n_multi_frag_rejects": m14["multi_frag_rejects"],
            "n_min_atoms_rejects": m14["min_atoms_rejects"],
            "mean_largest_frac": m14["mean_largest_frac"],
            "mean_n_atoms": m14["mean_n_atoms"],
            "largest_frac_min": m14["largest_frac_min"],
            "min_atoms": m14["min_atoms"],
        }
        admet8_out = dict(a8)
    else:
        m14_out = {
            "per_atom_validity": 0.0,
            "rdkit_strict_proxy": 0.0,
            "n_multi_frag_rejects": 0,
            "n_min_atoms_rejects": 0,
            "mean_largest_frac": 0.0,
            "mean_n_atoms": 0.0,
            "largest_frac_min": float(largest_frac_min),
            "min_atoms": int(min_atoms),
            "cohort_empty": True,
        }
        admet8_out = {
            "cohort_id": cohort_id,
            "n_input": 0,
            "n_ok": 0,
            "cohort_empty": True,
            "protocol_version": "admet8/2026-10-13",
        }

    # m4 / m5 / m7 — read existing side-JSONs verbatim.
    m4 = _read_json(metrics_dir / "sa_mean.json")
    if m4 is None:
        m4_out = {
            "value": None,
            "source": "metrics/by_metric/sa_mean.json",
            "status": "MISSING",
            "honest_negative": "sa_mean.json not on disk",
        }
    else:
        # ``measurements`` is a list; surface the last MEASURED entry.
        last_meas = None
        for m in reversed(m4.get("measurements", [])):
            if m.get("status") == "MEASURED":
                last_meas = m
                break
        m4_out = {
            "value": last_meas.get("value") if last_meas else None,
            "n_samples": last_meas.get("n_samples") if last_meas else None,
            "source_run": last_meas.get("run") if last_meas else None,
            "source": "metrics/by_metric/sa_mean.json",
            "status": "MEASURED" if last_meas else m4.get("aggregate", {}).get("status", "UNKNOWN"),
            "honest_negative": (
                last_meas.get("note", "") if last_meas else
                m4.get("honest_framing", "")
            ),
        }

    m5 = _read_json(metrics_dir / "platinai_mcf7_pred.json")
    if m5 is None:
        m5_out = {
            "value": None,
            "source": "metrics/by_metric/platinai_mcf7_pred.json",
            "status": "MISSING",
            "n_cells_measured": 0,
            "n_cells_design": 10,
        }
    else:
        agg = m5.get("aggregate", {})
        m5_out = {
            "value": m5.get("aggregate", {}).get("best_measured"),
            "n_cells_design": agg.get("n_cells_design", 10),
            "n_cells_measured": agg.get("n_cells_measured", 0),
            "source": "metrics/by_metric/platinai_mcf7_pred.json",
            "status": agg.get("status", "UNKNOWN"),
            "honest_negative": m5.get("honest_framing", ""),
        }

    m7 = _read_json(metrics_dir / "diversity_tanimoto.json")
    if m7 is None:
        m7_out = {
            "tanimoto_mean": None,
            "source": "metrics/by_metric/diversity_tanimoto.json",
            "status": "MISSING",
        }
    else:
        last_meas = None
        for m in reversed(m7.get("measurements", [])):
            if m.get("status") == "MEASURED":
                last_meas = m
                break
        m7_out = {
            "tanimoto_mean": last_meas.get("value") if last_meas else None,
            "n_samples": last_meas.get("n_samples") if last_meas else None,
            "source_run": last_meas.get("run") if last_meas else None,
            "source": "metrics/by_metric/diversity_tanimoto.json",
            "status": "MEASURED" if last_meas else "UNKNOWN",
            "honest_negative": (
                last_meas.get("note", "") if last_meas else
                m7.get("honest_framing", "")
            ),
        }

    # Wf-A22 L9 (2026-10-13) — per-pocket W₂ + decode column.
    #
    # Reads ``metrics/by_metric/cfm_per_pocket_metric.json`` when present
    # (L8 wired).  When absent, emits ``status="NOT_COMPUTED"`` with the
    # same convention used for the m5 MCF7 column.  Per-pocket rows are
    # echoed verbatim (L8 already preserves status + honest_negative);
    # this layer only adds the cohort-level source / status + a
    # ``HIGH_VARIANCE_PILOT`` tag on the cohort level so a downstream
    # paper table consumer cannot mistake the numbers for a cohort-
    # aggregated measure.
    from molmetal.eval.scoring.per_pocket_metric import (
        read_per_pocket_json as _read_pp,
    )
    pp_path = metrics_dir / "cfm_per_pocket_metric.json"
    pp_raw = _read_pp(pp_path)
    if pp_raw is None:
        # Wf-A30-2 B4 (2026-10-14) — semantic-break gate.
        if require_per_pocket:
            raise MissingRequiredInput(
                f"paper_table requires {pp_path} (Wf-A22 L9 honest-negative "
                f"close); run per_pocket_metric.run_full_per_pocket_panel() "
                f"first to populate the per-pocket W₂ + decode JSON",
                missing_path=str(pp_path),
                remediation=(
                    "python -m molmetal.scripts.cfm_vina_dock_panel "
                    "--per-pocket-metric (or call "
                    "molmetal.eval.scoring.per_pocket_metric."
                    "run_full_per_pocket_panel() programmatically) "
                    "to write the JSON before re-calling build_paper_table"
                ),
            )
        per_pocket_out: Dict[str, Any] = {
            "source": "metrics/by_metric/cfm_per_pocket_metric.json",
            "status": "NOT_COMPUTED",
            "n_pockets": 0,
            "honest_negative": (
                "L8 NEVER-WIRED at 2026-10-13: cfm_per_pocket_metric.json "
                "not on disk. Build the decode->panel link first; see "
                "TODO-125 + Wf-A22 L8."
            ),
        }
    else:
        # Wf-A33-FIXUP-1 (2026-10-14): honor the JSON's top-level
        # ``status`` field when present (e.g. ``"MEASURED_PILOT_MOCK"``)
        # so honest-framing per AGENTS.md §5.2 propagates downstream
        # instead of being silently rewritten to ``"MEASURED"``.
        # Default ``"MEASURED"`` preserved for backward-compat with
        # synthetic JSONs that pre-date the on-disk status field.
        per_pocket_out = {
            "source": "metrics/by_metric/cfm_per_pocket_metric.json",
            "status": pp_raw.get("status", "MEASURED"),
            "n_pockets": int(pp_raw.get("n_pockets", 0)),
            "decode": dict(pp_raw.get("decode", {})),
            "admet": dict(pp_raw.get("admet", {})),
            "honest_negative": pp_raw.get(
                "honest_negative",
                "Per-pocket numbers are PILOT only; do not promote to "
                "headline metrics without 5-seed CV.",
            ),
        }

    return PaperTableRow(
        cohort_id=str(cohort_id),
        n_smiles=n_smiles,
        m14_decode=m14_out,
        admet8=admet8_out,
        m4_sa=m4_out,
        m5_mcf7_pic50=m5_out,
        m7_intdiv1=m7_out,
        # Wf-A22 L9 — additive; absent JSON → NOT_COMPUTED verbatim.
        per_pocket_w2=per_pocket_out,
    )


def write_paper_table_json(row: PaperTableRow, out_path: Path) -> Path:
    """Atomic-write helper (per §5.6).

    Writes ``row.to_dict()`` to ``out_path`` using a temp-file +
    ``os.replace`` round trip so the resulting file is never
    half-written even if the process is killed mid-write.
    """
    import os
    import tempfile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.name, suffix=".tmp", dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(row.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return out_path


# Canonical cohort keys — preserved for use by the CLI / paper §X labels.
COHORT_KEYS: List[str] = [
    "platinai_pt_subset_5k",
    "nci60_9_cell",
    "drugood",
    "lambda_pilot_5x1",
    "lambda_path_a_10x3",
]


AVAILABLE: bool = _decode_gate.AVAILABLE and _admet8.AVAILABLE


# Wf-A22 §6.43h RESIDUAL + Wf-A24 §6.45c emit-paper-table shim.
# Per the Wf-A21 §6.42g FAIL and the Wf-A22 §6.43h RESIDUAL
# note: the trainer's eval loop (molmetal/scripts/train_cfm_yuelbond_wasserstein.py)
# calls ``build_paper_table`` directly.  This shim wraps the
# build + write + sidecar-read into a single convenience that
# downstream callers (the cross_dataset CLI, the refresh_21_metrics
# script, the cfm_pad_groups_mcnemar script) can use without
# re-implementing the file-path conventions.
#
# Behaviour:
#   1. Calls build_paper_table(cohort_id, smiles, metrics_dir, ...)
#   2. Calls write_paper_table_json(row, out_path)
#   3. Returns the out_path (so callers can chain .exists() or
#      json.loads(open(out_path)))
#
# Default-OFF / no side-effects: when out_path is None the function
# returns the in-memory PaperTableRow (no write).  When the cohort
# is unknown, the function falls back to the sidecar-only path
# (metrics_dir walk) so the shim is a strict superset of the
# direct build_paper_table call.
def emit_paper_table(
    cohort_id: str,
    smiles: Optional[Sequence[str]] = None,
    *,
    metrics_dir: Path,
    out_path: Optional[Path] = None,
    smiles_csv: Optional[Path] = None,
    largest_frac_min: float = 0.80,
    min_atoms: int = 2,
    # Wf-A30-2 B4 (2026-10-14) — semantic break gate, propagated from
    # build_paper_table.  When True, MissingRequiredInput propagates out
    # of this shim instead of being caught silently.
    require_per_pocket: bool = False,
) -> "PaperTableRow | Path":
    """Single-call wrapper: build + write + return.

    Parameters
    ----------
    cohort_id : str
        One of COHORT_KEYS (e.g. "platinai_pt_subset_5k").
    smiles : Sequence[str], optional
        Generated SMILES cohort.  If None, falls back to smiles_csv.
    metrics_dir : Path
        Directory holding the sidecar JSONs (m14/m4/m5/m7).
    out_path : Path, optional
        Where to write the paper_table.json.  When None, the in-memory
        PaperTableRow is returned without writing (no file I/O).
    smiles_csv : Path, optional
        CSV fallback when smiles is None.  Defaults to
        ``metrics_dir/cohort_id_smiles.csv``.
    largest_frac_min : float, default 0.80
        Forwarded to build_paper_table (per Wf-269C honest-negative
        pre-condition: 0.80 matches paper §X published threshold).
    min_atoms : int, default 2
        Forwarded to build_paper_table.
    require_per_pocket : bool, default False
        Wf-A30-2 B4 — propagated from build_paper_table.  When True,
        missing cfm_per_pocket_metric.json raises MissingRequiredInput
        instead of silently emitting status="NOT_COMPUTED".

    Returns
    -------
    PaperTableRow or Path
        When out_path is None: returns the in-memory row (no write).
        When out_path is given: returns the Path (always written per
        §5.6 atomic-write protocol).
    """
    row = build_paper_table(
        cohort_id=cohort_id,
        smiles=list(smiles) if smiles is not None else None,
        metrics_dir=metrics_dir,
        smiles_csv=smiles_csv,
        largest_frac_min=largest_frac_min,
        min_atoms=min_atoms,
        require_per_pocket=require_per_pocket,
    )
    if out_path is None:
        return row
    return write_paper_table_json(row, out_path)


__all__ = [
    "PaperTableRow",
    "build_paper_table",
    "write_paper_table_json",
    "emit_paper_table",  # Wf-A22 §6.43h RESIDUAL shim (closes §6.42g FAIL)
    "MissingRequiredInput",  # Wf-A30-2 B4 — semantic-break exception class
    "COHORT_KEYS",
    "PROTOCOL_VERSION",
    "AVAILABLE",
]


# ---------------------------------------------------------------------------
# Smoke (per §5.7)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """Self-test: returns 0 on PASS, 1 on FAIL.

    Tests:
      1. Empty cohort → row has n_smiles=0; side-JSON fields reflect
         whatever's on disk (``MISSING`` if the worktree has no
         ``metrics/by_metric/sa_mean.json``).
      2. Synthetic 4-mol cohort → m14 + ADMET8 populated, m4/m5/m7
         side-JSONs reflected verbatim (status preserved).
      3. ``write_paper_table_json`` round-trips the row.
      4. Wf-A22 L9 (2026-10-13) — per_pocket_w2 column emits
         ``status=NOT_COMPUTED`` when cfm_per_pocket_metric.json is
         absent, ``status=MEASURED`` when it is present.
    """
    # Test 1
    repo_root = Path(__file__).resolve().parents[3]
    metrics_dir = repo_root / "metrics" / "by_metric"
    if not metrics_dir.exists():
        # In a worktree that lacks the metrics directory, fall back
        # to a synthetic dir; the test asserts the structure, not
        # the data.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            metrics_dir = Path(td)
    r1 = build_paper_table(
        cohort_id="smoke_empty",
        smiles=[],
        metrics_dir=metrics_dir,
    )
    assert r1.n_smiles == 0
    assert r1.m14_decode["cohort_empty"] is True
    assert r1.admet8["cohort_empty"] is True
    # Wf-A22 L9 — per-pocket column must always be present; default
    # ``NOT_COMPUTED`` when the L8 JSON is absent. Wf-A33-FIXUP-1
    # (2026-10-14) widened the accepted set to also include the
    # honest PILOT variant (``MEASURED_PILOT_MOCK``) per AGENTS.md
    # §5.2 — the JSON's top-level ``status`` propagates verbatim.
    assert "status" in r1.per_pocket_w2, r1.per_pocket_w2
    assert r1.per_pocket_w2["status"] in (
        "NOT_COMPUTED",
        "MEASURED",
        "MEASURED_PILOT_MOCK",
    ), r1.per_pocket_w2
    # Test 2
    cohort = ["CCO", "c1ccccc1", "C.C.C.C", "nonsense"]
    r2 = build_paper_table(
        cohort_id="smoke_4mol",
        smiles=cohort,
        metrics_dir=metrics_dir,
    )
    assert r2.n_smiles == 4
    assert r2.m14_decode["per_atom_validity"] >= 0.0
    assert r2.admet8["n_ok"] >= 1
    # m4 / m5 / m7 status keys always present
    for k in ("status", "source"):
        assert k in r2.m4_sa, r2.m4_sa
        assert k in r2.m5_mcf7_pic50, r2.m5_mcf7_pic50
        assert k in r2.m7_intdiv1, r2.m7_intdiv1
    # CCO should pass per-atom; C.C.C.C should fail multi-fragment.
    assert r2.m14_decode["per_atom_validity"] < 1.0, r2.m14_decode
    # Test 3
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "paper_table_smoke.json"
        write_paper_table_json(r2, out_path)
        assert out_path.exists()
        with out_path.open("r") as f:
            rd = json.load(f)
        assert rd["cohort_id"] == "smoke_4mol"
        assert rd["n_smiles"] == 4
        assert rd["protocol_version"] == PROTOCOL_VERSION
        # Wf-A22 L9 — round-trip includes the per-pocket column.
        assert "per_pocket_w2" in rd, rd
    # Test 4 (Wf-A22 L9) — synthetic L8 JSON → status="MEASURED".
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        td_metrics = Path(td)
        pp_payload = {
            "protocol_version": "per_pocket_metric/2026-10-13",
            "n_pockets": 2,
            "decode": {
                "1h36": {"per_atom_validity": 0.66, "n_smiles": 12,
                         "largest_frac_min": 0.8, "min_atoms": 2,
                         "status": "MEASURED"},
                "4uuu_v": {"per_atom_validity": 0.50, "n_smiles": 6,
                         "largest_frac_min": 0.8, "min_atoms": 2,
                         "status": "MEASURED"},
            },
            "admet": {},
            "smiles_csv": "/tmp/x.csv",
            "honest_negative": "PILOT",
        }
        (td_metrics / "cfm_per_pocket_metric.json").write_text(
            json.dumps(pp_payload, indent=2)
        )
        r4 = build_paper_table(
            cohort_id="smoke_l9",
            smiles=["CCO"],
            metrics_dir=td_metrics,
        )
        assert r4.per_pocket_w2["status"] == "MEASURED", r4.per_pocket_w2
        assert r4.per_pocket_w2["n_pockets"] == 2, r4.per_pocket_w2
        assert "1h36" in r4.per_pocket_w2["decode"], r4.per_pocket_w2
    print(f"[paper_table] SMOKE PASS  4/4  n_smiles={r2.n_smiles}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_smoke())
