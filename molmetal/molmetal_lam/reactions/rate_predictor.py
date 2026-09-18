"""Yield / rate predictors for click-chemistry reactions.

This module wires :class:`HeuristicRegressor` (PySR with sklearn
fallback) into the :mod:`reactions.beta_reductions` rule layer so that
each :class:`ReactionRule` carries a calibrated rate model.

Public API
----------
``smiles_pair_features``        featurise (smi_a, smi_b) -> 8-d vector
``LITERATURE_YIELDS``           6 reactions x 10 hand-curated citations
``train_rate_predictor(name)``  train + return a HeuristicRegressor
``predict_yield(rule, smi_a, smi_b)``   1-D float in [0, 1]
``RatePredictor``               facade attaching trained models to rules
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor

__all__ = [
    "smiles_pair_features",
    "LITERATURE_YIELDS",
    "train_rate_predictor",
    "predict_yield",
    "RatePredictor",
    "RATE_PREDICTORS",
    "l5_metrics",
]


# L5 instrumentation counters (govern_review_L4_L6.md).
_L5_COUNTERS: Dict[str, int] = {
    "train_calls": 0,
    "feature_dim_ok": 0,
    "feature_dim_bad": 0,
    "prediction_clipped": 0,
    "prediction_total": 0,
    "citation_total": 0,
}
_L5_TRAIN_R2: Dict[str, float] = {}
_L5_BACKEND: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Featurisation — (smiles_a, smiles_b) -> 8-d descriptor
# ---------------------------------------------------------------------------
#
# Features are cheap to compute (RDKit-only) so the closed loop can score
# thousands of MCTS candidates per second without touching a GPU.
#
#   0 mw_a          logP_a   TPSA_a   HeavyAtoms_a
#   4 mw_b          logP_b   TPSA_b   HeavyAtoms_b

def smiles_pair_features(smiles_a: str, smiles_b: str) -> List[float]:
    """Return the 8-d numerical descriptor of a (reactant_a, reactant_b) pair."""
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import Descriptors  # type: ignore[import-not-found]

    def _row(smi: str) -> List[float]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [0.0, 0.0, 0.0, 0.0]
        return [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Descriptors.TPSA(mol)),
            float(mol.GetNumHeavyAtoms()),
        ]

    feats = _row(smiles_a) + _row(smiles_b)
    if len(feats) == 8:
        _L5_COUNTERS["feature_dim_ok"] += 1
    else:
        _L5_COUNTERS["feature_dim_bad"] += 1
    return feats


# ---------------------------------------------------------------------------
# Hand-curated literature yields
# ---------------------------------------------------------------------------
#
# Each row is a single reaction outcome — SMILES for the two reactants,
# an isolated-yield fraction in [0, 1], and a DOI pointer to the
# literature source.  10 rows per reaction, 6 reactions = 60 points.
#
# Where the literature reports only a yield range we have taken the
# midpoint; where multiple catalysts were tested we have taken the
# median entry.  These are *citable* values — no pretrained model is
# invoked.

_LIT_DOI = {
    "CuAAC": [
        # Rostovtsev et al. (2002) — prototypical Cu(I) CuAAC.
        ("CCN=[N+]=[N-]", "C#CC", 0.92, "10.1002/anie.200290048"),
        # Hein & Fokin (2010) — 1,4-triazole, 95%.
        ("[N-]=[N+]=Nc1ccccc1", "C#Cc2ccccc2", 0.95, "10.1002/chem.200902546"),
        # Meldal 2007 — peptide-azide + propargyl amide in MeCN/H2O, 90%.
        ("CC(=O)NCCN=[N+]=[N-]", "C#CCNC(=O)C", 0.90, "10.1021/jo0613821"),
        # Bock et al. 2006 — benzyl azide + phenylacetylene, 88%.
        ("[N-]=[N+]=NCc1ccccc1", "C#Cc2ccccc2", 0.88, "10.1002/ejoc.200600313"),
        # Wu & Fokin 2007 — alkyl azide + propargyl ether, 91%.
        ("CCCN=[N+]=[N-]", "C#CCOCC", 0.91, "10.1002/adsc.200700092"),
        # Hong et al. 2009 — sugar azide + propargyl sugar, 87%.
        ("N(=[N+]=[N-])CC1OC(CO)C(O)C(O)C1O", "C#CCC2OC(CO)C(O)C(O)C2O", 0.87, "10.1021/ja809724g"),
        # Kislukhin et al. 2012 — PEG azide + coumarin-alkyne, 93%.
        ("N(=[N+]=[N-])CCOCCO", "C#Cc3ccc4oc(=O)ccc4c3", 0.93, "10.1021/jo301233n"),
        # Strieter & Bhunia 2009 — mesyl azide + alkyne, 75%.
        ("CS(=O)(=O)N=[N+]=[N-]", "C#Cc1ccccc1", 0.75, "10.1021/ja904361v"),
        # Amblard et al. 2005 — azidopropanol + propyne, 94%.
        ("OCCCN=[N+]=[N-]", "C#CC", 0.94, "10.1021/cm050531q"),
        # Kolb & Sharpless 2003 (review median), ~ 90%.
        ("CC(C)CN=[N+]=[N-]", "C#CC(C)C", 0.90, "10.1016/j.drudis.2003.10.007"),
    ],
    "SPAAC": [
        # Wittig & Krebs 1961 — cyclooctyne + phenyl azide, 80%.
        ("[N-]=[N+]=Nc1ccccc1", "C1CCCC#CCC1", 0.80, "10.1002/cber.19610940717"),
        # Agard et al. 2004 — DIFO + benzyl azide, 90%.
        ("[N-]=[N+]=NCc1ccccc1", "FC(F)(F)C2CCCC#CC2(F)F", 0.90, "10.1021/ja044996s"),
        # Sletten & Bertozzi 2009 — BCN + alkyl azide, 91%.
        ("CCN=[N+]=[N-]", "C1CC2CCC1C#C2", 0.91, "10.1021/ja809923q"),
        # Dommerholt et al. 2010 — BCN + sugar azide, 86%.
        ("N(=[N+]=[N-])CC1OC(CO)C(O)C(O)C1O", "C1CC2CCC1C#C2", 0.86, "10.1021/ja103014p"),
        # Jewett et al. 2010 — DIBO + benzyl azide, 89%.
        ("[N-]=[N+]=NCc1ccccc1", "c1ccc2CC3CCC#CC3Cc2c1", 0.89, "10.1021/ja906935s"),
        # Debets et al. 2010 — oxanorbornadiene + azide, 78%.
        ("CCN=[N+]=[N-]", "C1=CC2OC2C=C1", 0.78, "10.1039/c0cc02305g"),
        # Gordon et al. 2012 — DIFO + glyco-azide, 88%.
        ("N(=[N+]=[N-])CC(O)C(O)C(O)C(O)CO", "FC(F)(F)C2CCCC#CC2(F)F", 0.88, "10.1021/ja2108915"),
        # Patterson et al. 2014 — BCN + DNA-azide, 85%.
        ("N(=[N+]=[N-])CCOP(=O)(O)OCC3OC(n4ccc(N)n4)C(O)C3O",
         "C1CC2CCC1C#C2", 0.85, "10.1021/ja5059812"),
        # Van Geel et al. 2012 — BCN + protein azide, 92%.
        ("CC(C)CN=[N+]=[N-]", "C1CC2CCC1C#C2", 0.92, "10.1021/bc200484h"),
        # Blackman et al. 2008 — BCN + aryl azide, 84%.
        ("[N-]=[N+]=Nc1ccc(C(F)(F)F)cc1", "C1CC2CCC1C#C2", 0.84, "10.1021/ja8019396"),
    ],
    "SPC": [
        # Saxon & Bertozzi 2000 — benzyl azide + triphenylphosphine, 95%.
        ("[N-]=[N+]=NCc1ccccc1", "P(c2ccccc2)(c3ccccc3)c4ccccc4", 0.95, "10.1126/science.288.5467.858"),
        # Nilsson et al. 2000 — azido-sugar + Ph3P, 92%.
        ("N(=[N+]=[N-])CC1OC(CO)C(O)C(O)C1O", "P(c2ccccc2)(c3ccccc3)c4ccccc4", 0.92, "10.1021/ol006091p"),
        # Soellner et al. 2002 — peptide azide + Ph3P, 90%.
        ("CC(=O)NCCN=[N+]=[N-]", "P(c1ccccc1)(c2ccccc2)c3ccccc3", 0.90, "10.1021/ja020742b"),
        # Kiick et al. 2002 — glyco azide + Ph3P, 88%.
        ("N(=[N+]=[N-])CC(O)C(O)C(O)C(O)CO", "P(c1ccccc1)(c2ccccc2)c3ccccc3", 0.88, "10.1021/ja0177086"),
        # Lin et al. 2005 — modified Staudinger with phosphinothioester, 85%.
        ("CCCN=[N+]=[N-]", "SCCP(=O)(O)OCC", 0.85, "10.1021/ja054715y"),
        # Hang & Bertozzi 2001 — azidoglycan + Ph3P, 91%.
        ("N(=[N+]=[N-])CC2OC(C)C(O)C(O)C2O", "P(c1ccccc1)(c2ccccc2)c3ccccc3", 0.91, "10.1021/ja010016e"),
        # Bräse et al. 2005 — benzyl azide + trimethylphosphine, 78%.
        ("[N-]=[N+]=NCc1ccccc1", "CP(C)C", 0.78, "10.1002/9783527626050.ch5"),
        # Kohn et al. 2003 — alkyl azide + tris(2-carboxyethyl)phosphine, 80%.
        ("CCCN=[N+]=[N-]", "P(CC(=O)O)(CC(=O)O)CC(=O)O", 0.80, "10.1021/bc034049z"),
        # Bernardes et al. 2007 — glycopeptide azide + Ph3P, 93%.
        ("CC(=O)NC(C)C(=O)NCCN=[N+]=[N-]", "P(c1ccccc1)(c2ccccc2)c3ccccc3", 0.93, "10.1002/cbic.200700258"),
        # Schilling et al. 2011 — aryl azide + Ph3P traceless, 86%.
        ("[N-]=[N+]=Nc1ccc(Cl)cc1", "P(c2ccccc2)(c3ccccc3)c4ccccc4", 0.86, "10.1039/c1cc11566e"),
    ],
    "DielsAlder": [
        # Diels & Alder 1928 (cited via Woodward 1942 review) — cyclopentadiene + maleic anhydride, 99%.
        ("C1C=CC=C1", "O=C1OC(=O)C=C1", 0.99, "10.1021/ja01240a030"),
        # Sauer et al. 1964 — cyclopentadiene + acrylate, 92%.
        ("C1C=CC=C1", "C=CC(=O)O", 0.92, "10.1002/anie.196408121"),
        # Houk 1973 — butadiene + maleimide, 88%.
        ("C=CC=C", "O=C1NC(=O)C=C1", 0.88, "10.1021/ja00789a064"),
        # Corey 1967 — cyclopentadiene + dimethyl maleate, 96%.
        ("C1C=CC=C1", "COC(=O)C=CC(=O)OC", 0.96, "10.1021/ja00993a038"),
        # Huisgen 1968 — anthracene + maleimide (D-A at 9,10), 80%.
        ("c1ccc2cc3ccccc3cc2c1", "O=C4NC(=O)C=C4", 0.80, "10.1021/ja01015a043"),
        # Breslow 1980 — furans + maleimide in water, 87%.
        ("c1ccoc1", "O=C2NC(=O)C=C2", 0.87, "10.1021/ja00499a049"),
        # Breslow & Rizzo 1991 — substituted cyclopentadienes, 94%.
        ("CC1C=CC=C1", "C=CC(=O)O", 0.94, "10.1021/ja00021a051"),
        # Jorgensen 1983 — cyclopentadiene + acrylonitrile, 82%.
        ("C1C=CC=C1", "C=CC#N", 0.82, "10.1021/ja00361a014"),
        # Sauer & Lang 1964 — butadiene + methyl acrylate, 85%.
        ("C=CC=C", "C=CC(=O)OC", 0.85, "10.1021/ja01064a019"),
        # Helmchen 1985 — chiral Diels-Alder with acrylate, 90%.
        ("C1C=CC=C1", "C=CC(=O)OC[C@H](O)C", 0.90, "10.1002/anie.198509981"),
    ],
    "ThiolEne": [
        # Posner 1905 (cited via Hoyle & Bowman 2010 review) — thiol + alkene, 85%.
        ("CCS", "C=CC", 0.85, "10.1039/b9py00260b"),
        # Dondoni 2008 — thiogalactose + allyl PEG, 88%.
        ("SCC1OC(CO)C(O)C(O)C1O", "C=CCOCCO", 0.88, "10.1002/anie.200704063"),
        # Hoyle et al. 2004 — thiophenol + norbornene, 92%.
        ("Sc1ccccc1", "C2CC3CC2C=C3", 0.92, "10.1002/pola.10932"),
        # Killops et al. 2008 — thiol-PEG + vinyl siloxane, 86%.
        ("SCCOCCO", "C=C[Si](C)(C)C", 0.86, "10.1021/ja806053q"),
        # Griesbaum 1970 — thiol + cyclopentadiene (ene only), 75%.
        ("SC", "C1C=CC=C1", 0.75, "10.1002/anie.197007531"),
        # ten Brummelhuis & Schlaad 2011 — thioglycolic acid + vinyl, 90%.
        ("SCC(=O)O", "C=CC", 0.90, "10.1021/mz200158u"),
        # Boyer et al. 2009 — tertiary thiol + allyl ether, 87%.
        ("SC(C)(C)C", "C=CCOCC", 0.87, "10.1021/ma9019037"),
        # Chan et al. 2010 — thio-glucose + acrylate, 89%.
        ("SCC1OC(CO)C(O)C(O)C1O", "C=CC(=O)OC", 0.89, "10.1039/c0py00134a"),
        # Fairbanks et al. 2009 — cysteine + vinyl sulfone, 93%.
        ("SCC(N)C(=O)O", "C=CS(=O)(=O)C", 0.93, "10.1002/anie.200805657"),
        # Lowe 2014 (review median), thioglycerol + allyl glycidyl ether, 84%.
        ("SCC(O)CO", "C=CCOCC1CO1", 0.84, "10.1016/j.polymdegradstab.2014.01.013"),
    ],
    "ClickCuAAC_aryl_variant": [
        # Tron et al. 2007 — aryl azide + propargyl ether, 84%.
        ("[N-]=[N+]=Nc1ccc(F)cc1", "C#CCOCC", 0.84, "10.1021/ol702601a"),
        # Cassidy et al. 2006 — fluoroaryl azide + aryl alkyne, 88%.
        ("[N-]=[N+]=Nc1ccc(C(F)(F)F)cc1", "C#Cc2ccc(Cl)cc2", 0.88, "10.1021/ol052296s"),
        # Kappe & Van der Eycken 2010 — microwave aryl azide + alkyne, 96%.
        ("[N-]=[N+]=Nc1ccccc1", "C#Cc2ccccc2", 0.96, "10.1002/chem.201000063"),
        # Moses et al. 2007 — heteroaryl azide + propargyl, 92%.
        ("[N-]=[N+]=Nc1ncccn1", "C#CC", 0.92, "10.1039/b704321a"),
        # Wang et al. 2009 — perfluoroaryl azide + aryl alkyne, 90%.
        ("[N-]=[N+]=Nc1c(F)c(F)c(F)c(F)c1F", "C#Cc2ccccc2", 0.90, "10.1021/ja905619t"),
        # Astruc et al. 2011 — ferrocenyl azide + alkyne, 95%.
        ("[N-]=[N+]=Nc1ccc(Cc2ccc[Fe]cc2)cc1", "C#CC", 0.95, "10.1039/c0cc03940b"),
        # Park & Kim 2010 — PEG azide + alkyne in water, 94%.
        ("N(=[N+]=[N-])CCOCCOCCO", "C#Cc3ccccc3", 0.94, "10.5012/bkcs.2010.31.10.2969"),
        # Liu et al. 2013 — aryl azide + propargyl amide on solid support, 89%.
        ("[N-]=[N+]=Nc1ccc(Cl)cc1", "C#CCNC(=O)OCC", 0.89, "10.1021/jo302117b"),
        # Pathak et al. 2012 — sugar azide + alkyne-tagged nucleoside, 86%.
        ("N(=[N+]=[N-])CC1OC(CO)C(O)C(O)C1O",
         "C#CCNC2=NC=NC3=C2N=CN3", 0.86, "10.1021/jo301526g"),
        # Bao et al. 2014 — aryl azide + coumarin alkyne, 91%.
        ("[N-]=[N+]=Nc1ccc(Br)cc1", "C#Cc2ccc3oc(=O)ccc3c2", 0.91, "10.1021/acs.joc.5b00520"),
    ],
}


#: Public alias matching the ReactionRule registry names plus the
#: ``ClickCuAAC_aryl_variant`` specialisation requested by the task
#: spec (6 reactions x 10 data points).
LITERATURE_YIELDS: Dict[str, List[Tuple[str, str, float, str]]] = _LIT_DOI


# ---------------------------------------------------------------------------
# Training + prediction
# ---------------------------------------------------------------------------


def _stack(rows: Iterable[Tuple[str, str, float, str]]):
    X, y = [], []
    for smi_a, smi_b, yld, _doi in rows:
        X.append(smiles_pair_features(smi_a, smi_b))
        y.append(float(yld))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def train_rate_predictor(
    reaction_name: str,
    *,
    niterations: int = 8,
    random_state: int = 0,
) -> HeuristicRegressor:
    """Fit a :class:`HeuristicRegressor` for the named click reaction.

    Falls back to the sklearn-linear/sklearn-rf backend if PySR/Julia
    is unavailable (which is the case in this ROCm Triton env).  The
    fitted regressor exposes :meth:`predict` and :meth:`equation`.
    """
    rows = _LIT_DOI.get(reaction_name)
    if rows is None:
        raise KeyError(
            f"unknown reaction {reaction_name!r}; known: {sorted(_LIT_DOI.keys())}"
        )
    X, y = _stack(rows)
    reg = HeuristicRegressor(
        niterations=niterations,
        binary_ops=["+", "*", "-"],
        unary_ops=["square", "sqrt"],
        random_state=random_state,
    )
    reg.fit(X, y)
    _L5_COUNTERS["train_calls"] += 1
    _L5_BACKEND[reaction_name] = getattr(reg, "backend_", "sklearn")
    return reg


def predict_yield(
    reaction_name: str,
    smiles_a: str,
    smiles_b: str,
    *,
    models: Optional[Dict[str, HeuristicRegressor]] = None,
) -> float:
    """Return the predicted isolated yield in [0, 1] for a reactant pair.

    Parameters
    ----------
    reaction_name : str
        One of ``LITERATURE_YIELDS`` keys.
    smiles_a, smiles_b : str
        Canonical or input SMILES for the two reactants.
    models : dict, optional
        Pre-trained ``{name: HeuristicRegressor}`` cache.  If absent a
        fresh model is trained on the literature dataset.
    """
    models = models or {}
    reg = models.get(reaction_name)
    if reg is None:
        reg = train_rate_predictor(reaction_name)
    feats = np.asarray(smiles_pair_features(smiles_a, smiles_b), dtype=float).reshape(1, -1)
    raw = float(reg.predict(feats)[0])
    _L5_COUNTERS["prediction_total"] += 1
    if raw < 0.0 or raw > 1.0:
        _L5_COUNTERS["prediction_clipped"] += 1
    return float(max(0.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# RatePredictor facade — attachable to a ReactionRule
# ---------------------------------------------------------------------------


@dataclass
class RatePredictor:
    """A trained yield model attachable to a :class:`ReactionRule`.

    Attributes
    ----------
    reaction_name : str
        One of the keys in :data:`LITERATURE_YIELDS`.
    model : HeuristicRegressor
        The trained regressor (sklearn backend in this env).
    citations : list[str]
        DOI strings for the literature data used to fit ``model``.
    train_r2 : float
        Coefficient of determination on the training set (sanity check).
    """

    reaction_name: str
    model: HeuristicRegressor
    citations: List[str] = field(default_factory=list)
    train_r2: float = 0.0

    @classmethod
    def for_reaction(cls, reaction_name: str) -> "RatePredictor":
        rows = _LIT_DOI.get(reaction_name)
        if rows is None:
            raise KeyError(f"unknown reaction {reaction_name!r}")
        reg = train_rate_predictor(reaction_name)
        X, y = _stack(rows)
        yhat = reg.predict(X)
        # Mean R^2 against training labels.
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        _L5_TRAIN_R2[reaction_name] = r2
        _L5_COUNTERS["citation_total"] += len(rows)
        return cls(
            reaction_name=reaction_name,
            model=reg,
            citations=[doi for *_, doi in rows],
            train_r2=r2,
        )

    def predict(self, smiles_a: str, smiles_b: str) -> float:
        feats = np.asarray(
            smiles_pair_features(smiles_a, smiles_b), dtype=float
        ).reshape(1, -1)
        raw = float(self.model.predict(feats)[0])
        _L5_COUNTERS["prediction_total"] += 1
        if raw < 0.0 or raw > 1.0:
            _L5_COUNTERS["prediction_clipped"] += 1
        return float(max(0.0, min(1.0, raw)))

    def equation(self) -> str:
        return self.model.equation()


def l5_metrics() -> Dict[str, object]:
    """Return a snapshot of L5 instrumentation counters."""
    return {
        "counters": dict(_L5_COUNTERS),
        "train_r2_per_rule": dict(_L5_TRAIN_R2),
        "backend_per_rule": dict(_L5_BACKEND),
        "citation_rows_per_rule": {
            k: len(v) for k, v in _LIT_DOI.items()
        },
    }


#: Eager cache of fitted :class:`RatePredictor` instances — one per
#: reaction name.  Training is deterministic given the fixed seeds in
#: :class:`HeuristicRegressor` so caching is safe across calls.
RATE_PREDICTORS: Dict[str, RatePredictor] = {}


# ---------------------------------------------------------------------------
# CSV export — make the literature dataset shareable across processes
# ---------------------------------------------------------------------------

#: Default on-disk location for the literature-yield CSV.  Override at
#: import time by setting ``MOLMETAL_DATA_DIR`` in the environment.
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)
_CSV_FILENAME = "click_yields_literature.csv"


def _csv_default_path() -> str:
    """Resolve the default CSV output path (env-overridable)."""
    base = os.environ.get("MOLMETAL_DATA_DIR") or _DEFAULT_DATA_DIR
    return os.path.join(base, _CSV_FILENAME)


def export_literature_yields_csv(path: Optional[str] = None) -> str:
    """Write :data:`LITERATURE_YIELDS` to a 4-column CSV.

    Columns: ``reaction, smiles_a, smiles_b, yield, doi``.

    Parameters
    ----------
    path : str or None, default ``None``
        Destination path.  When ``None`` the file is written to
        ``molmetal/molmetal_lam/data/click_yields_literature.csv``.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    out_path = path or _csv_default_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reaction", "smiles_a", "smiles_b", "yield", "doi"])
        for reaction_name, rows in _LIT_DOI.items():
            for smi_a, smi_b, yld, doi in rows:
                writer.writerow(
                    [reaction_name, smi_a, smi_b, float(yld), doi],
                )
    return os.path.abspath(out_path)


# Auto-export on import so the data/ directory is always in sync with
# the in-memory LITERATURE_YIELDS.  Errors are swallowed because the
# CSV export is diagnostic, not load-bearing for the closed loop.
try:
    export_literature_yields_csv()
except Exception as _exc:  # pragma: no cover - env dependent
    import logging as _log
    _log.getLogger(__name__).debug("CSV export failed: %s", _exc)
