"""wetlab_bayesian.py — wet-lab-in-the-loop Bayesian optimisation (L16 / R22-P4).

================================================================
Wf-78 Phase 3 (TODO-108 / R22-P4 / L16 Lambda gap closure)
================================================================
Wet-lab-in-the-loop Bayesian optimisation that closes the gap between
in-silico SBDD (D-MPNN top-20 candidates) and prospective wet-lab IC50
measurement via a Gaussian-process surrogate + q-EI batch acquisition +
per-institution protocol outreach (ORNL / PKU / 协和).

This module is the **CPU-only scaffold** for the L16 Lambda gap
(`TODO/pending/r22_l13_l16_gap_analysis.md` §4 — 70 scan signals, HIGH
priority); Wf-78 Phase 3 ships only the architecture + synthetic mock +
per-institution protocol data class.  Actual wet-lab outreach happens in
R23-P3 per `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §5.

Lit anchors
-----------
* Williams & Rasmussen (2006) GP for ML, MIT Press.
* Hernández-Lobato et al. (2017) q-EI batch BO, ICML.
* Frazier, Powell & Day (2009) surrogate-based optimisation.
* Letham et al. (2019) BO at scale with batch q-KG / q-EI.
* Snoek et al. (2012) practical BO, NeurIPS.
* Lookman et al. (2019) closed-loop materials discovery, Nature.

Honest framing
--------------
* **CPU-only docs scaffold** — no real GP fit, no real wet-lab assay, no real
  collaborator outreach.  All numerics are reproducible on synthetic mocks.
* The **6-month wet-lab cycle is DEFERRED** to R23-P3 (per
  `TODO/completed/101_r22_plan_l13_l16.md` §6).
* Per `TODO/pending/84_no_paper_task_without_authorization.md`, no paper work
  touched — synthesis.md is internal Wf-78 cycle documentation only.
* Per `TODO/completed/97b_wetlab_prework_docs.md`, 3 LOIs + 3 SOPs shipped
  on disk; the wetlab_bayesian.py **does not trigger outreach** (no
  external API calls, no email automation, no IRB paperwork).

Public surface
--------------
* :class:`WetLabConfig`            — config (kernel type, acquisition, batch size, cycle count).
* :class:`Institution`             — dataclass per-ORNL / PKU / 协和.
* :func:`build_institution_table`  — 3-row institution lookup table.
* :class:`GaussianProcessSurrogate`— minimal Matern-5/2 GP surrogate (CPU-only).
* :class:`qExpectedImprovement`    — q-EI batch acquisition (Hernández-Lobato 2017).
* :func:`run_iterative_loop`       — top-level iteration loop (CPU-only mock).
* :func:`build_synthetic_in_silico_top20` — deterministic mock D-MPNN top-20.
* :func:`build_synthetic_wetlab_measurements` — deterministic mock IC50 panel.
* :func:`compute_silico_wetlab_agreement` — Pearson r on (predicted vs measured).
* :func:`synthesize_wetlab_loop`   — top-level orchestrator (returns KPIs + per-institution report).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure the project root is on sys.path so `molmetal.*` resolves when the
# module is run standalone (`python wetlab_bayesian.py`).
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
_molmetal_pkg = os.path.abspath(os.path.join(_here, "..", ".."))
for _candidate in (_repo_root, _molmetal_pkg):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


# ---------------------------------------------------------------------------
# 1. Config + per-institution protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WetLabConfig:
    """Per-cycle configuration for the wet-lab-in-the-loop BO."""

    kernel: str = "matern_5_2"  # matern 5/2 (Williams + Rasmussen 2006 §4.2)
    acquisition: str = "q_ei"  # q-EI (Hernández-Lobato et al. 2017)
    q: int = 5  # batch size = 5 candidates per wet-lab cycle
    cycles: int = 5  # 5 wet-lab cycles per r22_p1_p2_p3_p4_p5_p6_plan.md §5.3
    metric: str = "pic50"  # target metric (pIC50 in µM)
    ard_lengthscales: int = 4  # 4 axes (metal × ligand × cell × cohort)
    noise_var: float = 0.05  # observation noise (Snoek et al. 2012 §2.2)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Institution:
    """Single wet-lab collaborator (per-institution protocol)."""

    code: str  # 'ORNL' / 'PKU' / 'XIEHE'
    name: str  # human-readable institution name
    assay_type: str  # 'protein_metal_binding' / 'ic50_curves' / 'nci60'
    target_axes: Tuple[str, ...]  # (metal, ligand_class, cell_line, cohort)
    cycle_weeks: int  # typical cycle latency (synthesis + assay + analysis)
    irb_required: bool  # whether IRB paperwork is required
    loi_path: str  # path to Letter-of-Intent on disk
    sop_path: str  # path to SOP on disk
    safety_constraints: str  # bsl-2 / bsl-1 / ehs-review-required

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "assay_type": self.assay_type,
            "target_axes": list(self.target_axes),
            "cycle_weeks": self.cycle_weeks,
            "irb_required": self.irb_required,
            "loi_path": self.loi_path,
            "sop_path": self.sop_path,
            "safety_constraints": self.safety_constraints,
        }


def build_institution_table() -> Dict[str, Institution]:
    """Build the 3-institution lookup table (ORNL + PKU + 协和)."""
    return {
        "ORNL": Institution(
            code="ORNL",
            name="Oak Ridge National Laboratory — CNMS",
            assay_type="protein_metal_binding",
            target_axes=("metal", "ligand_class", "target_protein", "buffer"),
            cycle_weeks=8,  # protein-metal binding SPR / ITC
            irb_required=False,
            loi_path="TODO/completed/97b_wetlab_prework_docs.md#ornl-loi",
            sop_path="TODO/completed/97b_wetlab_prework_docs.md#ornl-sop",
            safety_constraints="bsl-2 (Pt/Ru/Os chem stock)",
        ),
        "PKU": Institution(
            code="PKU",
            name="Peking University — College of Chemistry",
            assay_type="ic50_curves",
            target_axes=("metal", "ligand_class", "cell_line", "cohort"),
            cycle_weeks=7,  # synthesis (2w) + assay (4w) + analysis (1w)
            irb_required=False,
            loi_path="TODO/completed/97b_wetlab_prework_docs.md#pku-loi",
            sop_path="TODO/completed/97b_wetlab_prework_docs.md#pku-sop",
            safety_constraints="bsl-2 + chem-hygiene-plan",
        ),
        "XIEHE": Institution(
            code="XIEHE",
            name="北京协和医院 — Translational Oncology",
            assay_type="nci60",
            target_axes=("metal", "ligand_class", "nci60_cell", "patient_cohort"),
            cycle_weeks=12,  # full NCI-60 GI50 panel + IRB
            irb_required=True,
            loi_path="TODO/completed/97b_wetlab_prework_docs.md#xiehe-loi",
            sop_path="TODO/completed/97b_wetlab_prework_docs.md#xiehe-sop",
            safety_constraints="bsl-2 + IRB + EHS review",
        ),
    }


# ---------------------------------------------------------------------------
# 2. Gaussian-process surrogate (CPU-only minimal Matern-5/2)
# ---------------------------------------------------------------------------


class GaussianProcessSurrogate:
    """Minimal Matern 5/2 Gaussian-process surrogate for in-silico ↔ wet-lab mapping.

    This is a **CPU-only scaffold** — for real bench we'd plug in scikit-learn /
    GPyTorch / BoTorch / Ax.  The implementation here stays within NumPy so
    the unit tests can run on bare CPU hosts (no pytorch, no sklearn).

    Following Williams & Rasmussen 2006 §4.2 (Matern 5/2 kernel):

        k(d) = sigma_f² · (1 + sqrt(5) d / l + 5 d² / (3 l²)) · exp(-sqrt(5) d / l)

    where ``d = ||x - x'||`` is Euclidean over the 4-D ARD axes.

    ARD note: each axis carries its own length-scale (Williams + Rasmussen 2006 §5.5.1)
    but our scaffold uses isotropic for clarity (4-D Euclidean, single l).
    """

    def __init__(
        self,
        cfg: WetLabConfig = WetLabConfig(),
        length_scale: float = 1.0,
        signal_var: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        x_np = np.asarray(x, dtype=float).reshape(-1, cfg_ard_dim(self.cfg))
        y_np = np.asarray(y, dtype=float).reshape(-1)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                f"GP fit: x/y length mismatch ({x_np.shape[0]} vs {y_np.shape[0]})"
            )
        if x_np.shape[0] == 0:
            raise ValueError("GP fit: empty training data")
        self.x_train = x_np
        self.y_train = y_np
        return self

    def kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matern 5/2 kernel (Williams + Rasmussen 2006 §4.2.2)."""
        # Pairwise Euclidean distance matrix.
        diff = a[:, None, :] - b[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        sqrt5 = math.sqrt(5.0)
        scaled = sqrt5 * d / self.length_scale
        k = self.signal_var * (1.0 + scaled + (5.0 / 3.0) * d * d / (self.length_scale * self.length_scale)) * np.exp(-scaled)
        return k

    def predict(
        self, x_test: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.x_train is None:
            raise RuntimeError("GP predict: fit() must be called first")
        x_t = np.asarray(x_test, dtype=float).reshape(-1, cfg_ard_dim(self.cfg))
        k_tr_te = self.kernel(self.x_train, x_t)
        k_tr_tr = self.kernel(self.x_train, self.x_train)
        k_te_te = self.kernel(x_t, x_t)
        # Noise on diagonal (Snoek et al. 2012 §2.2).
        n = self.x_train.shape[0]
        k_tr_tr = k_tr_tr + self.cfg.noise_var * np.eye(n)
        try:
            alpha = np.linalg.solve(k_tr_tr, self.y_train)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(k_tr_tr, self.y_train, rcond=None)[0]
        mean = k_tr_te.T @ alpha
        if not return_std:
            return mean, None
        try:
            v = np.linalg.solve(k_tr_tr, k_tr_te)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(k_tr_tr, k_tr_te, rcond=None)[0]
        var = np.diag(k_te_te) - np.sum(k_tr_te * v, axis=0)
        var = np.clip(var, 1e-12, None)
        std = np.sqrt(var)
        return mean, std


def cfg_ard_dim(cfg: WetLabConfig) -> int:
    return cfg.ard_lengthscales


# ---------------------------------------------------------------------------
# 3. q-EI batch acquisition (Hernández-Lobato 2017)
# ---------------------------------------------------------------------------


@dataclass
class qExpectedImprovement:
    """q-EI batch acquisition (deterministic baseline).

    Per Hernández-Lobato et al. 2017:

        q-EI(x) ≈ E[max(0, y - y* + epsilon)]  over batch
                       = (µ - y* + phi · σ) · Φ((µ - y*)/σ) + σ · φ((µ - y*)/σ)

    where ``µ(x)``, ``σ(x)`` are GP posterior moments, ``y*`` is the
    current best observation, ``phi`` / ``Phi`` are the standard
    normal density / CDF.  Deterministic scaffolding: we rank by the
    closed-form q-EI per candidate and pick the top-``q``.
    """

    gp: GaussianProcessSurrogate
    best_y: float
    epsilon: float = 1e-9

    def score(self, x_candidate: np.ndarray) -> np.ndarray:
        mu, sigma = self.gp.predict(x_candidate, return_std=True)
        if sigma is None:
            return np.maximum(mu - self.best_y + self.epsilon, 0.0)
        # Φ, φ on z = (µ - y* + ε) / σ.
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (mu - self.best_y + self.epsilon) / sigma_safe
        phi_z = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        # Φ via numpy-friendly approximation (Abramowitz & Stegun 7.1.26).
        # Use scipy.special.erf when available; fall back to vectorised erf.
        try:
            from scipy.special import erf as _erf
            Phi_z = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
        except ImportError:
            # Vectorised erf approximation (numpy's math.erf is scalar-only);
            # use a tanh-based identity: erf(z) ≈ tanh(...) is NOT exact; so we
            # use a std-normal CDF via the closed-form 0.5*(1+tanh approximation)
            # ... but to keep precision we instead use scipy via numpy.
            # If scipy is missing entirely, fall back to a coarse Φ(z):
            Phi_z = np.where(
                z > 0.0,
                1.0 - 0.5 * np.exp(-z * z / 2.0) / (np.abs(z) * math.sqrt(2 * math.pi) + 1e-12),
                0.5 * np.exp(-z * z / 2.0) / (np.abs(z) * math.sqrt(2 * math.pi) + 1e-12),
            )
        return (mu - self.best_y + self.epsilon) * Phi_z + sigma_safe * phi_z

    def select(self, x_pool: np.ndarray, q: int = 5) -> List[int]:
        scores = self.score(x_pool)
        order = np.argsort(-scores)
        return order[:q].tolist()


# ---------------------------------------------------------------------------
# 4. Synthetic mocks — in-silico top-20 + wet-lab measurements
# ---------------------------------------------------------------------------


METALDOM_DRUG_AXES: Tuple[Tuple[str, str, str, str], ...] = (
    # (metal, ligand_class, cell_line, cohort)
    ("Pt", "Cl-NH3", "A2780", "ovarian"),
    ("Pt", "Cl-NH3", "MCF7", "breast"),
    ("Pt", "Cl-NH3", "A2780cis", "ovarian_cis_resistant"),
    ("Pt", "Cl-NH3", "A549", "lung"),
    ("Pt", "CBDCA", "A2780", "ovarian"),
    ("Pt", "CBDCA", "MCF7", "breast"),
    ("Au", "S-CH2-CONH", "A2780", "ovarian"),
    ("Au", "S-CH2-CONH", "MCF7", "breast"),
    ("Au", "S-CH2-CONH", "A549", "lung"),
    ("Ru", "arene-Cl", "A2780", "ovarian"),
    ("Ru", "arene-Cl", "MCF7", "breast"),
    ("Ru", "arene-Cl", "A549", "lung"),
    ("Ir", "Cp_star-C5", "A2780", "ovarian"),
    ("Ir", "Cp_star-C5", "MCF7", "breast"),
    ("Ir", "Cp_star-C5", "A549", "lung"),
    ("Pd", "PPh3-Cl", "A2780", "ovarian"),
    ("Pd", "PPh3-Cl", "MCF7", "breast"),
    ("Pd", "PPh3-Cl", "A549", "lung"),
    ("Os", "arene-Cl", "A2780", "ovarian"),
    ("Os", "arene-Cl", "MCF7", "breast"),
)


def _encode_axis(axes: Sequence[Tuple[str, str, str, str]]) -> np.ndarray:
    """Encode (metal × ligand × cell × cohort) as a 4-D unit-cube vector.

    Encoding scheme: each axis maps to a distinct component on a fixed scale:

      - metal        → 0..1 (alphabetic order)
      - ligand_class → 0..1 (alphabetic order)
      - cell_line    → 0..1 (alphabetic order)
      - cohort       → 0..1 (alphabetic order)

    This keeps the encoding deterministic and pre-emptive of the GP's
    isotropic length-scale; real wet-lab data would use one-hot + ARD.
    """
    metals = sorted({a[0] for a in axes})
    ligands = sorted({a[1] for a in axes})
    cells = sorted({a[2] for a in axes})
    cohorts = sorted({a[3] for a in axes})
    out = np.zeros((len(axes), 4), dtype=float)
    for i, (m, l, c, h) in enumerate(axes):
        out[i, 0] = metals.index(m) / max(len(metals) - 1, 1)
        out[i, 1] = ligands.index(l) / max(len(ligands) - 1, 1)
        out[i, 2] = cells.index(c) / max(len(cells) - 1, 1)
        out[i, 3] = cohorts.index(h) / max(len(cohorts) - 1, 1)
    return out


def build_synthetic_in_silico_top20(
    seed: int = 0,
) -> Tuple[List[Dict[str, str]], np.ndarray]:
    """Build a deterministic 20-row mock D-MPNN top-20 + 4-D feature vector.

    Each row has columns: smiles, metal, ligand_class, cell_line, cohort.
    Feature vector X is (20, 4) per the encoding in :func:`_encode_axis`.
    """
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, str]] = []
    for metal, lig, cell, cohort in METALDOM_DRUG_AXES:
        rows.append({
            "smiles": f"X_{metal}_{lig}_{cell}",
            "metal": metal,
            "ligand_class": lig,
            "cell_line": cell,
            "cohort": cohort,
        })
    X = _encode_axis(METALDOM_DRUG_AXES)
    return rows, X


def build_synthetic_wetlab_measurements(
    n_cycles: int = 5,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Build a deterministic mock wet-lab IC50 measurement set.

    Cycles 1..n_cycles × 5 candidates per cycle → 25 mock data points.

    Synthetic pIC50 noise model: pIC50 = clip(metal_bias + sin + jitter, 4, 9)
    per Snoek et al. 2012 §2.2 noise model.
    """
    rng = np.random.RandomState(seed + 1000)
    measurements: List[Dict[str, float]] = []
    metal_bias = {
        "Pt": 7.5, "Au": 6.8, "Ru": 6.2, "Ir": 6.0, "Pd": 5.5,
        "Os": 6.0, "Rh": 5.5,
    }
    for cycle in range(n_cycles):
        for _, lig, cell, cohort in METALDOM_DRUG_AXES[:5]:
            metal = METALDOM_DRUG_AXES[cycle][0]
            base = metal_bias.get(metal, 5.0)
            sin_signal = math.sin(rng.uniform(0, math.pi))
            jitter = rng.normal(0.0, 0.3)
            value = float(np.clip(base + sin_signal + jitter - 1.0, 4.0, 9.0))
            measurements.append({
                "cycle": cycle,
                "metal": metal,
                "ligand_class": lig,
                "cell_line": cell,
                "cohort": cohort,
                "pic50": round(value, 4),
            })
    return measurements


# ---------------------------------------------------------------------------
# 5. In-silico vs wet-lab agreement (Pearson r)
# ---------------------------------------------------------------------------


def compute_silico_wetlab_agreement(
    predicted: Sequence[float],
    measured: Sequence[float],
) -> Dict[str, float]:
    """Compute Pearson r + RMSE between in-silico predictions and wet-lab measurements.

    Returns a dict with ``pearson_r``, ``rmse``, ``n``.  Honest framing:
    real bench shows r in 0.5..0.8 on PlatinAI 5K (Wf-58 honest-negative;
    r ≈ 0.55 on A2780cis).
    """
    p = np.asarray(predicted, dtype=float).reshape(-1)
    m = np.asarray(measured, dtype=float).reshape(-1)
    n = min(p.size, m.size)
    if n < 2:
        return {"pearson_r": 0.0, "rmse": 0.0, "n": int(n)}
    p = p[:n]
    m = m[:n]
    pm = p - p.mean()
    mm = m - m.mean()
    denom = math.sqrt(float((pm * pm).sum() * (mm * mm).sum()))
    pearson = float((pm * mm).sum() / denom) if denom > 0 else 0.0
    rmse = float(math.sqrt(((p - m) ** 2).mean()))
    return {
        "pearson_r": round(pearson, 6),
        "rmse": round(rmse, 6),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# 6. Iterative BO loop (CPU-only scaffolding)
# ---------------------------------------------------------------------------


def run_iterative_loop(
    top20_rows: List[Dict[str, str]],
    X_top20: np.ndarray,
    cfg: WetLabConfig = WetLabConfig(),
    seed: int = 0,
) -> Dict[str, list]:
    """Run the q-EI batch BO loop over ``cycles`` cycles.

    Each cycle:
      1. Fit GP on the running (cycle) measurement set.
      2. Score all 20 candidates via q-EI.
      3. Pick top-``q`` candidates for the next wet-lab cycle.

    Returns: cycles ledger + q-EI selection trace + GP mean / std trace.
    """
    rng = np.random.RandomState(seed + 2000)

    # Initial cycle uses 5 mock measurements (cycles[0]).
    measurements = build_synthetic_wetlab_measurements(n_cycles=1, seed=seed)
    cycles_log: List[dict] = []
    selection_trace: List[List[int]] = []

    # Encode measurements to ARD vectors by joining with TOP20 axes.
    for cycle_idx in range(cfg.cycles):
        # Build training set for GP.
        train_x_list: List[np.ndarray] = []
        train_y_list: List[float] = []
        for meas in measurements:
            row = next(
                (
                    (i, r)
                    for i, r in enumerate(top20_rows)
                    if r["metal"] == meas["metal"]
                    and r["ligand_class"] == meas["ligand_class"]
                    and r["cell_line"] == meas["cell_line"]
                ),
                None,
            )
            if row is None:
                continue
            train_x_list.append(X_top20[row[0]])
            train_y_list.append(meas["pic50"])
        if not train_y_list:
            break
        x_train = np.asarray(train_x_list, dtype=float)
        y_train = np.asarray(train_y_list, dtype=float)

        gp = GaussianProcessSurrogate(cfg=cfg, length_scale=1.0, signal_var=2.0)
        gp.fit(x_train, y_train)
        best_y = float(y_train.max())

        qei = qExpectedImprovement(gp=gp, best_y=best_y)
        sel = qei.select(X_top20, q=cfg.q)
        selection_trace.append(sel)

        # Simulate wet-lab measurement of the q selected.  Even if the row's
        # metal/ligand/cell tuple has no prior measurement template (q-EI
        # deliberately picks out-of-cohort candidates), we synthesise a fresh
        # measurement using the row's identity + the metal_bias lookup, then
        # add a small per-observation jitter (Snoek et al. 2012 §2.2).
        metal_bias = {
            "Pt": 7.5, "Au": 6.8, "Ru": 6.2, "Ir": 6.0, "Pd": 5.5,
            "Os": 6.0, "Rh": 5.5,
        }
        for s_idx in sel:
            row = top20_rows[s_idx]
            base = metal_bias.get(row["metal"], 5.0)
            jitter = rng.normal(0.0, 0.2)
            measurements.append({
                "cycle": cycle_idx + 1,
                "metal": row["metal"],
                "ligand_class": row["ligand_class"],
                "cell_line": row["cell_line"],
                "cohort": row.get("cohort", "default"),
                "pic50": round(float(np.clip(base + jitter, 4.0, 9.0)), 4),
            })
        cycles_log.append({
            "cycle": cycle_idx,
            "n_measurements": len(measurements),
            "best_y": best_y,
            "selected": sel,
        })

    return {
        "cycles_log": cycles_log,
        "selection_trace": selection_trace,
        "final_measurements": measurements,
    }


# ---------------------------------------------------------------------------
# 7. Top-level orchestrator
# ---------------------------------------------------------------------------


def synthesize_wetlab_loop(
    seed: int = 0,
    cfg: Optional[WetLabConfig] = None,
) -> dict:
    """Top-level orchestrator: run synthetic in-silico→wet-lab BO + per-institution report.

    Returns a JSON-serialisable dict with:
      * ``institution_table``     — 3-row ORNL/PKU/XIEHE protocol summary
      * ``n_cycles``              — number of wet-lab cycles run
      * ``q_per_cycle``           — batch size q
      * ``selection_trace``       — per-cycle q-EI selections (top-20 index lists)
      * ``final_measurements``    — running wet-lab ledger
      * ``silico_wetlab_agreement`` — Pearson r / RMSE on final cycle
    """
    cfg = cfg or WetLabConfig()
    top20_rows, X_top20 = build_synthetic_in_silico_top20(seed=seed)
    institutions = build_institution_table()
    loop_result = run_iterative_loop(top20_rows, X_top20, cfg=cfg, seed=seed)

    # Build synthetic in-silico predictions vs wet-lab measurements for the
    # Pearson / RMSE report.  In real bench, predictions = D-MPNN model output;
    # here we mock predictions by adding a measurable signal-noise offset.
    final = loop_result["final_measurements"]
    rng = np.random.RandomState(seed + 3000)
    predicted = []
    measured = []
    for m in final:
        # Naïve mock: D-MPNN = wet-lab + small bias, signal-noise 0.5.
        predicted.append(m["pic50"] + rng.normal(0.0, 0.5))
        measured.append(m["pic50"])
    agreement = compute_silico_wetlab_agreement(predicted, measured)

    return {
        "institution_table": {k: v.as_dict() for k, v in institutions.items()},
        "n_cycles": cfg.cycles,
        "q_per_cycle": cfg.q,
        "kernel": cfg.kernel,
        "acquisition": cfg.acquisition,
        "selection_trace": loop_result["selection_trace"],
        "cycles_log": loop_result["cycles_log"],
        "n_measurements_final": len(loop_result["final_measurements"]),
        "silico_wetlab_agreement": agreement,
    }


# ---------------------------------------------------------------------------
# 8. Smoke entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CPU-only smoke runner — run synthetic BO loop + report KPIs."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    molmetal_pkg = os.path.abspath(os.path.join(here, "..", ".."))
    for candidate in (repo_root, molmetal_pkg):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    result = synthesize_wetlab_loop(seed=0)
    print(
        f"wetlab_bayesian smoke: cycles={result['n_cycles']} "
        f"q={result['q_per_cycle']} "
        f"inst={len(result['institution_table'])} "
        f"r={result['silico_wetlab_agreement']['pearson_r']:.4f} "
        f"rmse={result['silico_wetlab_agreement']['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
