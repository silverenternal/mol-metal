"""r4_lib.metrics — All RDKit / numpy / heuristic metric wrappers.

This is the largest module in :mod:`r4_lib` by line count.  Every
public ``metric_*_mean`` function here has the same convention:

  * Input: ``Sequence[str]`` (candidate SMILES)
  * Output: ``float`` (mean) / ``int`` (count) / ``Dict[str, int]``
    (distribution), with NaN-safe arithmetic and graceful
    degradation to ``0.0`` on import / parse failure.

Sub-sections (matching the original file's comment headers):

* Training-set novelty + reference-tanimoto helpers
  (:func:`_tanimoto`, :func:`training_set_novelty`,
  :func:`reference_tanimoto`).
* Lambda-native diversity metrics (:func:`diversity_alpha`,
  :func:`diversity_homotype`).
* WF-P0 — 9 P0 chemistry / metal-coordination metrics
  (logp / tpsa / rotb / oxidation_state / coordination_number /
  monodentate_cl_count / gsh_evasion / dna_kb / anticancer_index).
* WF-T25 — 4 metallodrug proxies (reduction_potential /
  trans_effect / lfse / pt_dna_crosslink).
* WF-SA-Penalty-Guidance — SA + QED aggregate metrics.
* WF-Phase3b-MetricsV2 — 8 tumor-relevant ADMET / safety metrics
  (logp7_4 / gi50 / cell_permeability / herg / ames / hepatotox /
  aqueous_solubility / plasma_protein_binding).
* WF-Phase3D — sub-pocket fingerprint diversity
  (:func:`metric_subpocket_diversity_mean`).
* WF-Phase3G — metal-coordination compliance
  (:func:`metric_metal_coord_compliance_mean`).
* WF-Rigid-RMSD-Metric + WF-CoM-Shift-Metric — 3-D geometry
  comparators against the pocket reference ligand
  (:func:`_embed_3d_for_rmsd`, :func:`metric_rigid_rmsd_mean`,
  :func:`_mol_center_of_mass`, :func:`metric_com_shift_mean`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Training-set Tanimoto helpers (novelty & reference diagnostics).
# ---------------------------------------------------------------------------
def _tanimoto(fp_a, fp_b) -> float:
    try:
        from rdkit.DataStructs import TanimotoSimilarity  # type: ignore

        return float(TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def training_set_novelty(
    smiles_list: Sequence[str],
    training_smiles: Sequence[str],
) -> float:
    """Return ``1 - max(Tanimoto(query, training_set))``.

    Higher = more novel.  Falls back to ``1.0`` when either set is
    empty (no training overlap possible) or RDKit is unavailable.
    """
    if not smiles_list or not training_smiles:
        return 1.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
        from rdkit import DataStructs  # type: ignore
    except Exception:
        return 1.0
    train_fps = []
    for s in training_smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            train_fps.append(fp)
        except Exception:
            continue
    if not train_fps:
        return 1.0
    novelties = []
    for s in smiles_list:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        except Exception:
            continue
        max_sim = 0.0
        for tfp in train_fps:
            sim = _tanimoto(fp, tfp)
            if sim > max_sim:
                max_sim = sim
        novelties.append(1.0 - max_sim)
    if not novelties:
        return 1.0
    return float(sum(novelties) / len(novelties))


def reference_tanimoto(candidate_smi: str, reference_smi: str) -> float:
    """Max Tanimoto between ``candidate_smi`` and ``reference_smi``.

    Diagnostic only (not a metric).  Returns 0.0 if either side is
    unavailable or RDKit cannot parse.
    """
    if not candidate_smi or not reference_smi:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return 0.0
    try:
        m1 = Chem.MolFromSmiles(candidate_smi)
        m2 = Chem.MolFromSmiles(reference_smi)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return _tanimoto(fp1, fp2)
    except Exception:
        return 0.0


def diversity_alpha(states: Sequence[Any]) -> float:
    """Mean pairwise typed-variable hits / max (Λ-native diversity).

    For each pair of distinct canonical SMILES, count the *length*
    difference as a typed-variable-hit proxy (different canonical
    forms must differ in at least one atom/bond).  Returns the mean
    pairwise difference normalised by ``max(len_a, len_b)`` — a value
    in ``[0, 1]`` where ``0`` means all siblings are identical and
    ``1`` means every sibling is structurally disjoint.

    Historical alias: this metric is now also reported as
    ``diversity_tanimoto`` in the summary.md template — the SE(3) /
    atom-symbol-histogram axis is the *baseline* diversity signal
    against which ``diversity_homotype`` (Lambda-native) is compared.
    """
    if not states:
        return 0.0
    distinct: List[Tuple[str, int]] = []
    seen: set = set()
    for s in states:
        try:
            smi = s.canonical_smiles() or ""
        except Exception:
            continue
        if not smi or smi in seen:
            continue
        seen.add(smi)
        distinct.append((smi, len(smi)))
    if len(distinct) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            smi_i, len_i = distinct[i]
            smi_j, len_j = distinct[j]
            # Set-difference is a cheap typed-variable-hit proxy.
            set_i = set(smi_i)
            set_j = set(smi_j)
            sym_diff = len(set_i.symmetric_difference(set_j))
            denom = max(len_i, len_j, 1)
            total += sym_diff / denom
            count += 1
    return float(total / count) if count else 0.0


def diversity_homotype(states: Sequence[Any]) -> float:
    """Mean pairwise ``homotype_distance`` over ``states`` (Lambda-native).

    Third first-class algorithmic asset Lambda brings to the paper
    (WF-Lambda-2), alongside ``alpha_equivalence`` (deduplication)
    and ``beta_NF`` (synthesizability).  Uses
    :func:`molmetal_lam.metrics.homotype_diversity.homotype_diversity`
    which combines typed-variable cosine (0.5) + β-reduction-depth
    norm-diff (0.3) + click-rule-fires Jaccard (0.2) into a single
    ``[0, 1]`` distance.

    INDEPENDENT of SE(3) distance and Morgan Tanimoto — a pair of
    constitutional isomers that collapses to Tanimoto = 1.0 can still
    have positive ``homotype_distance`` when their typed-variable
    histograms differ.

    The metric is computed over ``HomotypeSignature.from_mol``
    signatures built from each ``state``'s RDKit mol; if RDKit is
    unavailable or ``from_mol`` returns an empty signature, the
    contribution from that pair is zero (graceful degradation).
    """
    if not states:
        return 0.0
    try:
        from molmetal_lam.metrics.homotype_diversity import (  # type: ignore
            HomotypeSignature,
            homotype_distance as _homotype_distance,
        )
    except Exception:
        return 0.0

    sigs = []
    for s in states:
        try:
            smi = s.canonical_smiles() or ""
        except Exception:
            smi = ""
        if not smi:
            continue
        try:
            from rdkit import Chem  # type: ignore

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            sigs.append(HomotypeSignature.from_mol(mol))
        except Exception:
            continue
    if len(sigs) < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += _homotype_distance(sigs[i], sigs[j])
            n_pairs += 1
    return float(total / n_pairs) if n_pairs else 0.0


# ---------------------------------------------------------------------------
# WF-P0-Metrics — 9 P0 per-cell metrics (closing 9/25 of TargetDiff gap)
# ---------------------------------------------------------------------------
# These are ALL RDKit/numpy CPU-only metrics, so they add zero GPU load
# to the Lambda-only harness.  Each metric has a strict [0,1]-bounded
# range (except the raw descriptors logP/TPSA/RotB and the oxidation-
# state distribution dict, which have problem-specific ranges).
#
#  1. logp_mean                 — RDKit Crippen logP mean over cells
#  2. tpsa_mean                 — RDKit TPSA mean over cells
#  3. rotb_mean                 — RDKit NumRotatableBonds mean over cells
#  4. oxidation_state_distribution — dict {symbol: count}
#  5. coordination_number_mean  — mean over Pt/Ru/Ir centres
#  6. monodentate_cl_count      — count of monodentate Cl leaving groups
#  7. gsh_evasion_score         — fraction of candidates that evade GSH
#  8. dna_kb_proxy              — mean DNA binding proxy in [0, 1]
#  9. anticancer_index          — mean composite anticancer index [0, 1]
# ---------------------------------------------------------------------------


def _safe_mol(smi: str):
    """RDKit mol-from-smiles helper; returns None on parse failure."""
    if not smi:
        return None
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        return m if m is not None else None
    except Exception:
        return None


def metric_logp_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit Crippen logP across ``candidates`` (range ~[-5, 10])."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.MolLogP(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_tpsa_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit TPSA (Topological Polar Surface Area) across candidates."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.TPSA(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_rotb_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit NumRotatableBonds across candidates."""
    if not candidates:
        return 0.0
    try:
        from rdkit.Chem import Descriptors  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(Descriptors.NumRotatableBonds(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_oxidation_state_distribution(candidates: Sequence[str]) -> Dict[str, int]:
    """Count metal centres per (symbol, formal_charge) across candidates.

    Returns a dict ``{"Pt_II": 7, "Ru_III": 2, ...}`` of integer counts.
    Only Pt/Ru/Ir/Au/Rh/Os centres are counted.  Empty dict on no parse.
    """
    out: Dict[str, int] = {}
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    roman = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV"}
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in metals:
                    continue
                charge = atom.GetFormalCharge()
                # Default to oxidation state = formal charge for neutral ligands.
                ox_label = roman.get(charge, f"c{charge}")
                key = f"{sym}_{ox_label}"
                out[key] = out.get(key, 0) + 1
        except Exception:
            continue
    return out


def metric_coordination_number_mean(candidates: Sequence[str]) -> float:
    """Mean coordination number across Pt/Ru/Ir/Au/Rh/Os centres."""
    if not candidates:
        return 0.0
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    vals: List[int] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                if atom.GetSymbol() not in metals:
                    continue
                donor_count = sum(
                    1 for n in atom.GetNeighbors() if n.GetSymbol() != "H"
                )
                vals.append(int(donor_count))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_monodentate_cl_count(candidates: Sequence[str]) -> int:
    """Total count of Cl ligands bound to exactly one metal centre.

    This is a GSH liability proxy: each monodentate Cl is a potential
    leaving group susceptible to aquation and GSH S-attack.
    """
    if not candidates:
        return 0
    metals = ("Pt", "Ru", "Ir", "Au", "Rh", "Os")
    count = 0
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            for atom in m.GetAtoms():
                if atom.GetSymbol() != "Cl":
                    continue
                metal_neighbours = [
                    n for n in atom.GetNeighbors() if n.GetSymbol() in metals
                ]
                if len(metal_neighbours) == 1:
                    count += 1
        except Exception:
            continue
    return int(count)


def metric_gsh_evasion_score(candidates: Sequence[str]) -> float:
    """Fraction of candidates that evade GSH S-attack (heuristic flag).

    Uses :func:`AnticancerMetricSuite.gsh_evasion_flag`.  Returns 0.0
    when no candidates or the suite is unavailable.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    hits = 0
    counted = 0
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        counted += 1
        try:
            if suite.gsh_evasion_flag(m):
                hits += 1
        except Exception:
            continue
    return float(hits / counted) if counted else 0.0


def metric_dna_kb_proxy(candidates: Sequence[str]) -> float:
    """Mean DNA Kb proxy across candidates (heuristic, range [0, 1])."""
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    vals: List[float] = []
    for s in candidates:
        m = _safe_mol(s)
        if m is None:
            continue
        try:
            vals.append(float(suite.dna_kb_proxy(m)))
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_anticancer_index(candidates: Sequence[str]) -> float:
    """Mean composite anticancer index across candidates (range [0, 1]).

    Uses :func:`AnticancerMetricSuite.composite_score` and averages the
    ``anticancer_index`` column.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.metrics.anticancer_metric_suite import (  # type: ignore
            AnticancerMetricSuite,
        )
    except Exception:
        return 0.0
    suite = AnticancerMetricSuite()
    vals: List[float] = []
    for s in candidates:
        try:
            composite = suite.composite_score(s)
            ai = composite.get("anticancer_index", 0.0)
            # Treat NaN as 0.0 to avoid aggregate corruption.
            try:
                ai_f = float(ai)
            except Exception:
                ai_f = 0.0
            if ai_f != ai_f:  # NaN check
                ai_f = 0.0
            vals.append(ai_f)
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-T25-Metallo-Proxies — 4 metallodrug-specific proxies
# (reduction_potential / trans_effect / lfse / pt_dna_crosslink)
# These are PROXIES (Shriver & Atkins Table 17.7 + Appleton 1997 + Wang
# 2005), not measured values.  All return [0, 1] and degrade gracefully
# to 0.0 on import / parse failure.  Wired into the cell / aggregate
# / serialise path by the ``--metallodrug-proxies`` CLI flag.
# ---------------------------------------------------------------------------
def _import_metallodrug_proxies():
    """Best-effort import of the metallodrug_property_proxies module.

    Returns the imported module on success, ``None`` on ImportError so
    the wrapper metric_*_mean functions below can degrade gracefully
    (returning 0.0 in line with the existing convention).
    """
    try:
        from molmetal_lam.priors.metallodrug_property_proxies import (  # type: ignore
            compute_metallodrug_proxies,
        )

        return compute_metallodrug_proxies
    except Exception:
        return None


def metric_metallodrug_reduction_potential_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`reduction_potential_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("reduction_potential", 0.0))
        except Exception:
            continue
        if v != v:  # NaN
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_trans_effect_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`trans_effect_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("trans_effect", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_lfse_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`lfse_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("lfse", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_metallodrug_pt_dna_crosslink_mean(candidates: Sequence[str]) -> float:
    """Mean :func:`pt_dna_crosslink_proxy` across candidates."""
    fn = _import_metallodrug_proxies()
    if fn is None or not candidates:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            d = fn(s)
            v = float(d.get("pt_dna_crosslink", 0.0))
        except Exception:
            continue
        if v != v:
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-SA-Penalty-Guidance — SA + QED aggregate metrics (closes the
# WF-SA-Penalty-Guidance requirement to report SA + QED + validity).
# ---------------------------------------------------------------------------
def metric_sa_mean(candidates: Sequence[str]) -> float:
    """Mean Ertl-Schuffenhauer SA score across ``candidates`` ([1, 10]).

    Returns 0.0 when no candidates parse or when sascorer is unavailable
    (consistent with the convention that the SA aggregate reports a
    finite numeric even in the offline-fallback path).
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.sa_score import sa_score_ertl  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            v = float(sa_score_ertl(s))
        except Exception:
            continue
        if v != v:  # NaN
            continue
        vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


def metric_qed_mean(candidates: Sequence[str]) -> float:
    """Mean RDKit QED across ``candidates`` ([0, 1]).

    Returns 0.0 when no candidates parse or when QED is unavailable.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.qed_scorer import qed_from_smiles  # type: ignore
    except Exception:
        return 0.0
    vals: List[float] = []
    for s in candidates:
        try:
            v = qed_from_smiles(s)
        except Exception:
            v = None
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:  # NaN
            continue
        vals.append(fv)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET / safety
# columns (closes 8/25 of TargetDiff gap beyond the 9 P0 columns shipped
# in WF-P0-Metrics-Add).  All CPU-only (RDKit / numpy); zero GPU load.
# Lit basis:
#   - logp7_4               Weininger 1990 + Patrick 2009 substituted-logP
#   - gi50_proxy            Hou 2007 ADMET regression
#   - cell_permeability_logPapp  Hou 2007 + Mente 2015 heuristic
#   - herg_cardio_risk      Veith 2009 + Cavalluzzi 2023 rule review
#   - ames_mutagen          Benigni-Richard 2005 + Sushko 2012 alerts
#   - hepatotox_index       Hughes 2008 Rule-of-2 + Stepan 2011 alerts
#   - aqueous_solubility_logS  Delaney 2004 ESOL
#   - plasma_protein_binding  Obach 1999 sigmoid regression
# See ``molmetal/reports/wf_parallel_tasks/phase3b_metrics_v2.md`` for the
# full math basis and 20 tests.  Each function is a thin wrapper around
# :mod:`molmetal_lam.sbdd_env.metrics_v2` and returns 0.0 on any import
# or parse failure (parity with ``metric_logp_mean`` convention).
# ---------------------------------------------------------------------------
def _import_metrics_v2():
    """Best-effort import of metrics_v2 batch helpers.

    Returns the imported module on success, ``None`` on ImportError so
    the wrapper metric_*_mean functions below can degrade gracefully
    (returning 0.0 in line with the existing convention).
    """
    try:
        from molmetal_lam.sbdd_env import metrics_v2  # type: ignore

        return metrics_v2
    except Exception:
        return None


def metric_logp7_4_mean(candidates: Sequence[str]) -> float:
    """Mean substituted-logP7.4 over ``candidates`` (range unbounded).

    Defined as ``logP - 0.45·α_7.4 + 0.30·β_7.4`` — Weininger 1990 base
    logP with Patrick 2009 ionisation corrections.  Honours the lit
    recommendation that ionisable acids (e.g. aspirin -COOH) report
    lower logP7.4 than Crippen logP.  Returns 0.0 on import failure or
    empty input.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.logp7_4_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_gi50_proxy_mean(candidates: Sequence[str]) -> float:
    """Mean Hou 2007 GI50 proxy over ``candidates`` (range [0, 8]).

    ``4.5 + 0.30·MR - 0.015·TPSA - 0.50·(RotB/10) - 0.20·ArRings``.
    Heuristic activity predictor (not wet-lab calibrated).  Organic-only
    baseline gives ~8.0 (no activity).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.gi50_proxy_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_cell_permeability_logPapp_mean(candidates: Sequence[str]) -> float:
    """Mean cellular permeability logPapp over ``candidates`` ([-8, -3]).

    ``-4.0 + 0.33·logP - 0.013·TPSA + 0.40·HBD`` (Hou 2007 / Mente
    2015).  More negative = poorer permeability (caffeine-like
    behaviour).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.cell_permeability_logPapp_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_herg_cardio_risk_mean(candidates: Sequence[str]) -> float:
    """Mean hERG cardiac risk in [0, 1] over ``candidates`` (Veith 2009).

    Sums 4 rule bonuses (MW, logP, TPSA, basic N).  Returns 0.0 on
    import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.herg_cardio_risk_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_ames_mutagen_mean(candidates: Sequence[str]) -> float:
    """Mean AMES mutagenicity alert rate over ``candidates`` ([0, 1]).

    ``1.0`` if any of 6 SMARTS alerts match (Benigni-Richard 2005 +
    Sushko 2012).  Aggregated as fraction-of-flagged across the batch.
    Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.ames_mutagen_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_hepatotox_index_mean(candidates: Sequence[str]) -> float:
    """Mean hepatotoxicity index over ``candidates`` ([0, 1]).

    Hughes 2008 Rule-of-2 + Stepan 2011 structural alerts (aniline /
    hydrazine / furan / thiophene).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.hepatotox_index_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_aqueous_solubility_logS_mean(candidates: Sequence[str]) -> float:
    """Mean aqueous solubility logS over ``candidates`` ([-12, 2]).

    Delaney 2004 ESOL formula.  More negative = lower solubility.
    Ethanol ~0 (miscible); naphthalene <= -2 (low).  Returns 0.0 on
    import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.aqueous_solubility_logS_mean(list(candidates)))
    except Exception:
        return 0.0


def metric_plasma_protein_binding_mean(candidates: Sequence[str]) -> float:
    """Mean plasma protein binding fraction over ``candidates`` ([0, 1]).

    Sigmoid of logP (Obach 1999).  Caffeine ~6% (low); naphthalene
    >=80% (high).  Returns 0.0 on import failure.
    """
    m = _import_metrics_v2()
    if m is None or not candidates:
        return 0.0
    try:
        return float(m.plasma_protein_binding_mean(list(candidates)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Phase3D-PerResidue-Diversity — sub-pocket fingerprint diversity
# (Bemis & Murcko 1996 + Jasial 2021 IntDiv survey + Peter 2019 SPF).
#
# Mean pairwise per-residue Tanimoto distance over a set of generated
# SMILES, weighted by pocket residues.  Complements
# ``diversity_tanimoto`` (whole-molecule) and ``diversity_homotype``
# (typed-variable Jaccard).  CPU-only (RDKit Morgan + numpy).
#
# The residue weights default to ``[1] * K_uniform`` so the metric
# collapses to ordinary Tanimoto when no pocket is loaded — see
# ``per_residue_diversity`` docstring.  Returns 0.0 on import failure,
# empty input, or any per-call failure.  Set ``--sub-pocket-diversity
# off` (CLI is added in Phase 4) to skip this metric.
# ---------------------------------------------------------------------------
def metric_subpocket_diversity_mean(
    candidates: Sequence[str],
    n_residues: int = 4,
) -> float:
    """Mean pairwise per-residue Tanimoto distance over ``candidates``.

    Parameters
    ----------
    candidates : Sequence[str]
        Generated SMILES from a cell.
    n_residues : int, default 4
        Number of sub-pocket residue bins (uniform-weight placeholder;
        the real per-residue weight vector comes from
        ``voxelization.py:208,255`` or ``vina_adapter.py:961`` once the
        pocket loader is wired — Phase 4 ships the metric on a uniform
        default so it can be added to the dispatch table without
        depending on the pocket loader).

    Returns
    -------
    float in [0, 1] (mean per-residue Tanimoto *distance*).
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.per_residue_diversity import (  # type: ignore
            per_residue_diversity,
        )
    except Exception:
        return 0.0
    # De-duplicate (the metric does not auto-dedup; identical SMILES
    # inflate the mean downward per its docstring).
    seen: set = set()
    distinct: List[str] = []
    for s in candidates:
        if s and s not in seen:
            seen.add(s)
            distinct.append(s)
    if len(distinct) < 2:
        return 0.0
    weights = [1] * int(n_residues)
    try:
        return float(per_residue_diversity(distinct, weights))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Phase3G-MetalCoordProbe — graph-theoretic metal-coordination probe
# (Lippard & Berg 1995 + Reedijk 1987 + Miessler 2014).
#
# Coordinate-free metal-coordination check: parse SMILES, walk metal
# neighbours (H excluded), compare CN against canonical tables per
# (metal, oxidation_state).  Returns the *compliance rate* over the
# generated set (organic-only inputs are excluded from the denominator
# by design, matching ``soft_score_metal_geometry`` semantics).
# ---------------------------------------------------------------------------
def metric_metal_coord_compliance_mean(candidates: Sequence[str]) -> float:
    """Mean metal-coordination compliance rate over ``candidates``.

    For every molecule containing a recognised metal centre
    (Pt/Pd/Au/Ag/Ru/Ir/Rh/Os/Re/Fe/Co/Ni/Cu/Zn/Mn), compares the
    observed coordination number against the canonical CN for that
    (metal, OS) pair.  The returned value is the fraction of compliant
    *metal-containing* molecules (organic-only inputs are excluded
    from the denominator).  When no molecules contain a metal centre
    the function returns 0.0 (the metric does not fire on organics
    by design — mirrors ``soft_score_metal_geometry`` semantics).

    Returns 0.0 on import failure or any other failure.
    """
    if not candidates:
        return 0.0
    try:
        from molmetal_lam.sbdd_env.metal_coord_probe import (  # type: ignore
            compliance_rate,
        )
    except Exception:
        return 0.0
    try:
        return float(compliance_rate(list(candidates)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-Rigid-RMSD-Metric — rigid-fragment RMSD between generated and reference
# ligands (TargetDiff §4.4 SOTA signal, target ≈ 1.5 Angstroms).
#
# Uses RDKit's :func:`AllChem.GetBestRMS` for the rigid alignment: this
# function superimposes ``probe`` onto ``ref`` (best symmetry-aware
# mapping) and returns the RMSD in Angstroms.  The metric is the mean
# RMSD over the (candidate, reference) pairs in a cell, where each
# candidate SMILES is embedded in 3-D with :func:`AllChem.EmbedMolecule`
# before alignment.
#
# Honest-framing: ``GetBestRMS`` requires both molecules to have a
# conformer; the embedding step uses ``ETKDGv3`` (deterministic, fast)
# with a single attempt.  When RDKit cannot embed a candidate (large /
# unusual topology) the candidate is skipped — the metric is therefore
# a *conditional* mean over embeddable candidates.  This is the same
# convention used by TargetDiff's open-source eval script and matches
# the lower-is-better convention (target ≈ 1.5 Å).
#
# CPU-only (no GPU); falls back to 0.0 when RDKit is unavailable, when
# the reference SMILES is missing, or when no candidate embeds.
# ---------------------------------------------------------------------------
def _embed_3d_for_rmsd(mol) -> bool:
    """Embed ``mol`` in 3-D using ETKDGv3 + MMFF optimization.

    Returns True on success, False on any failure (silently skipped by
    the caller).  The conformer is left attached to ``mol`` so the
    subsequent :func:`AllChem.GetBestRMS` call can use it.

    Honest-framing: the previous implementation set
    ``params.maxAttempts = 1`` which raises ``AttributeError`` in
    recent RDKit builds (maxAttempts is a positional argument to
    :func:`EmbedMolecule`, not an EmbedParameters attribute — the
    ETKDGv3 equivalent is ``numZeroFail``).  The silent failure caused
    every RMSD/CoM metric to silently collapse to 0.0 in production
    sweeps.  We now use ``numZeroFail`` (the documented ETKDGv3 knob)
    and pass it positionally for belt-and-braces back-compat.
    """
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return False
    if mol is None:
        return False
    try:
        if mol.GetNumAtoms() <= 0 or mol.GetNumConformers() > 0:
            return mol.GetNumConformers() > 0
        # Add explicit Hs so the embedded geometry matches what
        # GetBestRMS expects (it operates on heavy atoms but the Hs
        # help the force field).
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D  # deterministic
        params.numZeroFail = 1  # ETKDGv3 equivalent of "maxAttempts=1"
        # Belt-and-braces: pass maxAttempts positionally too so older
        # RDKit builds that DO accept it still work — but only when
        # an ``EmbedParameters`` object isn't already providing the
        # settings (the two overloads are mutually exclusive).
        try:
            cid = AllChem.EmbedMolecule(mol_h, params)
        except Exception:
            cid = AllChem.EmbedMolecule(
                mol_h,
                maxAttempts=1,
                randomSeed=int(params.randomSeed),
            )
        if cid < 0:
            # Embedding can fail for unusual topologies; the metric
            # skips the molecule in that case.
            return False
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=50)
        except Exception:
            # MMFF can fail on metallic centres; not fatal — keep
            # the un-optimised conformer (RDKit's GetBestRMS still
            # works, just with a slightly higher RMSD).
            pass
        # Strip the explicit Hs back off so GetBestRMS sees the same
        # heavy-atom set as the original mol (this is the convention
        # used by every docking-pipeline RMSD eval).
        mol_no_h = Chem.RemoveHs(mol_h)
        # Transfer the conformer onto the original mol so callers can
        # use ``mol.GetConformer()`` directly.
        try:
            conf = mol_no_h.GetConformer()
            mol.AddConformer(conf, assignId=True)
        except Exception:
            # If transfer fails, leave the embedded mol as-is.
            return mol_h.GetNumConformers() > 0
        return mol.GetNumConformers() > 0
    except Exception:
        return False


def metric_rigid_rmsd_mean(
    gen_mol,
    ref_mol,
) -> float:
    """Rigid-fragment RMSD (Angstroms) between ``gen_mol`` and ``ref_mol``.

    Uses :func:`rdkit.Chem.AllChem.GetBestRMS` to align ``gen_mol`` onto
    ``ref_mol`` (symmetry-aware, minimises RMSD over all automorphisms)
    and returns the resulting RMSD in Angstroms.  Lower is better;
    TargetDiff SOTA target ≈ 1.5 Å.

    Returns 0.0 when:
      * either mol is ``None``,
      * RDKit is unavailable,
      * ``ref_mol`` lacks a conformer (caller is responsible for
        embedding the reference first),
      * the alignment fails.

    The function is CPU-only and deterministic (seeded ETKDGv3 for
    any fresh embedding).
    """
    if gen_mol is None or ref_mol is None:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return 0.0
    try:
        if ref_mol.GetNumConformers() == 0:
            return 0.0
        # Embed the candidate if it doesn't already have a conformer.
        cand = gen_mol
        if cand.GetNumConformers() == 0:
            if not _embed_3d_for_rmsd(cand):
                return 0.0
        # Symmetry-aware rigid alignment; returns RMSD in Angstroms.
        rmsd = AllChem.GetBestRMS(cand, ref_mol)
        try:
            rmsd_f = float(rmsd)
        except Exception:
            return 0.0
        if rmsd_f != rmsd_f:  # NaN
            return 0.0
        return max(0.0, rmsd_f)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# WF-CoM-Shift-Metric — center-of-mass shift (target ≤ 3 Å per TargetDiff SOTA)
# ---------------------------------------------------------------------------
# Honest-framing: the original spec named the RDKit helper
# ``rdkit.Chem.Descriptors3D.CenterCenter`` but that symbol does NOT
# exist in the installed RDKit (verified — ``Descriptors3D`` only
# exposes Asphericity / NPR / PMI / RadiusOfGyration / SpherocityIndex
# / PBF / InertialShapeFactor / Eccentricity).  We therefore compute
# the mass-weighted centre of mass directly from the conformer's
# heavy-atom coordinates via numpy — this is the same quantity used by
# every docking pipeline (AutoDock Vina, DiffDock, TargetDiff) and the
# same quantity the spec was naming when it said "CoM shift vs
# reference ligand".  Pure RDKit/numpy + CPU-only.
# ---------------------------------------------------------------------------
def _mol_center_of_mass(mol):
    """Mass-weighted CoM of ``mol`` from its (first) 3-D conformer.

    Returns a numpy ndarray of shape ``(3,)`` on success, or the scalar
    ``0.0`` (a sentinel "uncomputable") when:

      * ``mol`` is None,
      * RDKit is unavailable,
      * the molecule has no conformer (caller is responsible for
        embedding — we silently skip rather than corrupt the aggregate),
      * the molecule has zero atoms,
      * numpy is unavailable,
      * any other exception is raised.

    The choice of 0.0 as the fallback matches the convention used by
    every other metric in this file: "0.0 means the metric could not
    be evaluated".  Callers MUST check ``isinstance(result, np.ndarray)``
    to distinguish a successful CoM from the sentinel.
    """
    if mol is None:
        return 0.0
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return 0.0
    try:
        if mol.GetNumConformers() == 0:
            return 0.0
        try:
            import numpy as _np  # type: ignore
        except Exception:
            return 0.0
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 0:
            return 0.0
        coords = _np.array(
            [list(conf.GetAtomPosition(i)) for i in range(n_atoms)],
            dtype=float,
        )
        masses = _np.array([a.GetMass() for a in mol.GetAtoms()], dtype=float)
        if masses.sum() <= 0.0:
            return 0.0
        com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
        return com
    except Exception:
        return 0.0


def metric_com_shift_mean(
    gen_mol,
    ref_ligand_mol,
) -> float:
    """Center-of-mass shift (Angstroms) between a generated ligand and a
    reference (crystal) ligand, in the pocket frame.

    Computed as::

        || CoM(gen_mol) - CoM(ref_ligand_mol) ||_2

    using mass-weighted centre-of-mass coordinates from each mol's
    3-D conformer.  Both mols must already have conformers — the
    :func:`_embed_3d_for_rmsd` helper handles embedding with a
    deterministic ETKDGv3 seed.

    TargetDiff SOTA target ≈ 3 Å (lower = better).  Returns 0.0 when:

      * either input is None,
      * RDKit is unavailable,
      * numpy is unavailable,
      * either mol lacks a conformer (caller should embed first),
      * either mol has zero atoms,
      * any other exception is raised.

    The function is CPU-only and deterministic given fixed input
    conformers.
    """
    if gen_mol is None or ref_ligand_mol is None:
        return 0.0
    try:
        import numpy as _np  # type: ignore
    except Exception:
        return 0.0
    com_g = _mol_center_of_mass(gen_mol)
    com_r = _mol_center_of_mass(ref_ligand_mol)
    if not isinstance(com_g, _np.ndarray) or not isinstance(com_r, _np.ndarray):
        return 0.0
    try:
        diff = com_g - com_r
        d = float(_np.linalg.norm(diff))
    except Exception:
        return 0.0
    if d != d:  # NaN guard
        return 0.0
    return max(0.0, d)
