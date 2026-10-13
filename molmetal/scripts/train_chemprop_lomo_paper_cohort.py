"""PA3 — chemprop v2 (official D-MPNN) cross-metal LOMO on the Paper-A cohort.

WHY THIS SCRIPT EXISTS (BLK-PA3)
--------------------------------
Paper A Claim 1 reports a large within-metal -> cross-metal (LOMO) AUC collapse
measured with a Morgan-FP + XGBoost probe.  The supporting argument that "the
collapse is a property of the metal axis, not a broken backbone" currently rests
on MoleculeNet only, where this repo's D-MPNN loses to the chemprop reference
value on 4/4 tasks (Table 4 of the draft).  A reviewer can therefore argue the
cross-metal 0.44--0.62 column is *underfitting* rather than a metal-axis effect.

The fix (BLK-PA3 Step 1) is to re-run the cross-metal LOMO with the **official
chemprop v2 message-passing backbone** (``chemprop.nn.AtomMessagePassing``,
production ``d_h=300 / depth=3``) as the scorer, on the SAME cohort and the SAME
split the paper's Table 1 uses.  If the collapse reproduces under chemprop, it
cannot be an artefact of a weak backbone.

Same cohort / same split — the part that is easy to get wrong
------------------------------------------------------------
The paper's Table 1 numbers do NOT come from ``load_metal_cytotox()`` (that is
the full 26 801-row MetalCytoToxDB release).  They come from
``metal_activity_transfer_probe.py``, whose cohort is built as:

    MetalCytotoxDataset.from_csv(
        filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01,
                              metal_whitelist=[m]))
    -> collapse_duplicates(...)          # one row per unique ligand, majority
                                          # label, ties -> ACTIVE

and whose LOMO split drops from the training pool any ligand that appears in the
held-out metal's test set.  This script reuses that exact cohort builder
(:func:`collapse_duplicates` imported from the probe) and that exact leakage
rule, so the chemprop numbers are apples-to-apples with Table 1.

Scorer contract — single pooled binary head, no metal features
--------------------------------------------------------------
The XGB probe is a single pooled binary classifier with no metal input: it is
trained on the ligands of four metals and asked to rank the held-out metal's
ligands.  To be comparable, this script uses the chemprop encoder + ONE
``BinaryClassificationFFN`` binary head, and passes no metal embedding.  (The
existing ``train_chemprop_v2_lomo.py`` / ``wf146_chemprop_lomo.py`` use a
*per-metal-head* design in which the held-out head only ever sees zero targets
during LOMO training, which caps the fold AUC and is not the probe's contract.)

Honest framing (AGENTS.md §5.2)
-------------------------------
Every number this script emits is MEASURED on the held-out metal's deduplicated
ligands with ``sklearn.metrics.roc_auc_score`` on the sigmoid of the pooled
head's logit.  No fabricated / aspirational values.  If the chemprop collapse is
smaller than the XGB probe's, that is the finding and it is reported as such —
the honest-negative fallback in BLK-PA3 is to downgrade the model class and let
the XGB probe carry the main claim.

Launch (AGENTS.md §5.11 — YAML only)
------------------------------------
    scripts/launch_pa3_chemprop_lomo.sh [--dry-run]

Every knob lives in ``configs/pa3_chemprop_lomo_v1.yaml``; the wrapper passes
``--config`` (and ``--out-dir``) and nothing else.

Modes
-----
``--mode cross`` (default) reports the leave-one-metal-out column — the paper's
Table-1 *cross* column.  ``--mode within`` reports stratified 5-fold CV inside
each metal (the same ``StratifiedKFold(n_splits, shuffle=True,
random_state=seed)`` protocol the XGB probe uses, reporting the mean of the
per-fold AUCs) — the paper's Table-1 *within* column.  ``--mode both`` runs
each, which is what puts the chemprop within-vs-cross penalty on the same
footing as the probe's pair.

``--n-permutations N`` additionally attaches a label-permutation null to each
cross fold, computed on the ``--perm-seed`` run's fixed ranking via the probe's
own ``permutation_null`` (so the chemprop small-fold verdicts are judged against
the same exchangeability reference as ``metal_activity_transfer_permutation_null
.json``).  The per-row labels/probabilities for that seed are persisted under
``per_fold[*].perm_seed_scores``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_default() -> None:
    """Point ``MOLMETAL_DATA_DIR`` at the in-repo copy before cytotox imports.

    ``molmetal.data.cytotox`` freezes ``DEFAULT_CSV`` at import time from
    ``MOLMETAL_DATA_DIR`` (falling back to a /mnt/storage mount that does not
    exist on this host — MEASURED 2026-10-10 in the probe).  This module's own
    import chain pulls cytotox in, so the env var must be set at the top of the
    file, not inside :func:`build_probe_cohort`.
    """
    import os

    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


_load_env_default()

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from chemprop.data import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from sklearn.metrics import average_precision_score, roc_auc_score

from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2Encoder,
    _make_metal_head,
)

# ---------------------------------------------------------------------------
# Defaults (overridable from the YAML; AGENTS.md §5.11)
# ---------------------------------------------------------------------------
METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44, 49, 123)
DEFAULT_EPOCHS: int = 50
DEFAULT_HIDDEN: int = 300  # chemprop v2 production default
DEFAULT_DEPTH: int = 3  # chemprop v2 production default
DEFAULT_FFN_HIDDEN: int = 300
DEFAULT_FFN_LAYERS: int = 1  # chemprop BinaryClassificationFFN default
DEFAULT_DROPOUT: float = 0.0
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_TIME_THRESHOLD: float = 24.0
DEFAULT_IC50_MIN: float = 0.01
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "pa3_chemprop_lomo"

# --- PA15 metal conditioning -------------------------------------------------
# ``none``  = the PA3 metal-agnostic scorer (unchanged default).
# ``token`` = a learnable per-metal embedding concatenated to the graph readout.
# ``desc``  = a FIXED physicochemical descriptor vector (projected by a shared
#             learned linear) concatenated to the graph readout.
# In leave-one-metal-out the ``token`` arm's held-out row is never trained, so it
# is a random vector at test time — that is the point of the comparison.
DEFAULT_METAL_MODE: str = "none"
DEFAULT_METAL_EMBED_DIM: int = 16

#: Canonical coordination-geometry order for the one-hot block of the descriptor.
_COORD_ORDER: Tuple[str, ...] = ("octahedral", "square_planar", "linear", "tetrahedral")

#: Descriptor column names (order matches :func:`metal_descriptor_table`).
METAL_DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "atomic_number",
    "group",
    "period",
    "oxidation_state",
    "d_electron_count",
    *tuple(f"geom_{g}" for g in _COORD_ORDER),
)

#: Static, traceable source of the per-metal constants.  Every number below is
#: read from this file (period is parsed from the ``row`` column, e.g. "4d" -> 5)
#: plus the formal-charge identity ``d_count = group - oxidation_state``.  No
#: value is typed by hand — AGENTS.md §5.2 zero-fabrication.
METAL_TABLE_CSV = PROJECT_ROOT / "molmetal" / "data" / "transition_metals_30.csv"


def metal_descriptor_table(
    metals: Tuple[str, ...] = METALS,
) -> Tuple[np.ndarray, List[str]]:
    """Return ``(D, n_metals)`` standardised descriptors + column names.

    Built from ``molmetal/data/transition_metals_30.csv`` (``atomic_number``,
    ``group``, ``oxidation_state``, ``coordination_geometry``, and a period
    parsed from ``row``).  Standardised column-wise over the requested metals.
    """
    import csv as _csv

    rows: Dict[str, Dict[str, str]] = {}
    with METAL_TABLE_CSV.open() as fh:
        for rec in _csv.DictReader(fh):
            rows[rec["metal_symbol"]] = rec
    missing = [m for m in metals if m not in rows]
    if missing:
        raise KeyError(f"metals absent from {METAL_TABLE_CSV}: {missing}")

    feats: List[List[float]] = []
    for m in metals:
        rec = rows[m]
        group = int(rec["group"])
        ox = int(rec["oxidation_state"])
        period = int(str(rec["row"]).strip()[0])  # "4d" -> 4, "5d" -> 5
        geom = str(rec["coordination_geometry"]).strip().lower()
        onehot = [1.0 if geom == g else 0.0 for g in _COORD_ORDER]
        feats.append(
            [
                float(rec["atomic_number"]),
                float(group),
                float(period),
                float(ox),
                float(group - ox),  # formal-charge d-count approximation
                *onehot,
            ]
        )
    arr = np.asarray(feats, dtype=np.float32)  # (n_metals, D)
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    std = (arr - mu) / sd
    return std.astype(np.float32), list(METAL_DESCRIPTOR_NAMES)

logger = logging.getLogger("pa3_chemprop_lomo")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Cohort — byte-for-byte the probe's cohort construction.
# ---------------------------------------------------------------------------
def _draw_support(
    te_y: np.ndarray,
    scaffolds: Optional[List[str]],
    k: int,
    seed: int,
    sampler: str,
) -> np.ndarray:
    """Draw the K-shot support indices from the held-out metal's test rows.

    ``sampler`` (todo.json C3 / W3, added 2026-10-13):

    * ``stratified``       — label-stratified random (the historical arm);
    * ``random``           — uniform random, no label stratification;
    * ``scaffold_diverse`` — greedy round-robin over Bemis--Murcko scaffolds,
      so the support set covers as many distinct scaffolds as possible;
    * ``activity_scaffold``— label-stratified quotas, then round-robin over
      scaffolds *within* each label (balanced AND diverse).

    Every arm returns a sorted index array of size ~K; the query set is
    ``setdiff1d(all, support)`` and is IDENTICAL across arms for a given
    (metal, seed), so the comparison is paired.
    """
    rng = np.random.default_rng(int(seed))
    n = len(te_y)
    if sampler == "random":
        return np.sort(rng.permutation(n)[:k]).astype(int)

    def _quota(pool_size: int) -> int:
        return max(1, int(round(k * pool_size / n)))

    def _scaffold_round_robin(indices: np.ndarray, want: int) -> List[int]:
        """Greedy round-robin over Bemis-Murcko scaffolds of ``indices``."""
        groups: Dict[str, List[int]] = {}
        for i in indices:
            groups.setdefault(str(scaffolds[i]), []).append(int(i))
        keys = list(groups.keys())
        rng.shuffle(keys)
        picked: List[int] = []
        cursor = 0
        while len(picked) < want:
            progressed = False
            for key in keys:
                if len(picked) >= want:
                    break
                g = groups[key]
                if cursor < len(g):
                    picked.append(g[cursor])
                    progressed = True
            cursor += 1
            if not progressed:
                break
        return picked

    # 2026-10-13 FIX — `scaffold_diverse` and `activity_scaffold` previously
    # shared one code path (both fell into the label-stratified branch), so the
    # two arms were byte-identical and the C3 ablation had only 3 distinct arms.
    # Now: scaffold_diverse = pure scaffold coverage (NO label quota);
    # activity_scaffold = label-stratified quotas, scaffold round-robin inside.
    if sampler == "scaffold_diverse":
        if scaffolds is None:
            raise ValueError("sampler='scaffold_diverse' requires scaffolds")
        return np.asarray(
            sorted(set(_scaffold_round_robin(np.arange(n), k))), dtype=int
        )

    sup: List[int] = []
    for lab in (0, 1):
        li = np.where(te_y == lab)[0].copy()
        if li.size == 0:
            continue
        rng.shuffle(li)
        k_lab = _quota(li.size)
        if sampler == "activity_scaffold":
            if scaffolds is None:
                raise ValueError("sampler='activity_scaffold' requires scaffolds")
            sup.extend(_scaffold_round_robin(li, k_lab))
        else:  # stratified (historical arm)
            sup.extend(li[:k_lab].tolist())
    return np.asarray(sorted(set(sup)), dtype=int)


def _concat_pool(
    tr_mgs: List[object],
    tr_y: np.ndarray,
    tr_metal: Optional[np.ndarray],
    sup_mgs: List[object],
    sup_y: np.ndarray,
    sup_metal: Optional[np.ndarray],
):
    """Source train pool ++ K support rows, for the concat-retrain control."""
    pool_mgs = list(tr_mgs) + list(sup_mgs)
    pool_y = np.concatenate([np.asarray(tr_y, dtype=np.int64),
                             np.asarray(sup_y, dtype=np.int64)])
    pool_m = None
    if tr_metal is not None and sup_metal is not None:
        pool_m = np.concatenate([np.asarray(tr_metal), np.asarray(sup_metal)])
    return pool_mgs, pool_y, pool_m


def _auc_at(probs_all: np.ndarray, y_all: np.ndarray, idx: np.ndarray) -> float:
    yy = np.asarray(y_all)[idx]
    if len(set(yy.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(yy, np.asarray(probs_all)[idx]))


def build_probe_cohort(
    metals: List[str],
    time_threshold: float,
    ic50_min: float,
    cohort_spec: str = "24h",
) -> Tuple[Dict[str, Tuple[List[str], np.ndarray]], Dict[str, Dict[str, int]]]:
    """Per-metal deduplicated cohort, identical to the paper's Table-1 probe.

    ``cohort_spec="paper_funnel"`` (2026-10-13, W1) instead uses
    :func:`molmetal.data.cytotox.build_paper_cohort` — the Krasnov-style funnel
    (IC50-present, metal, measurement-level dedup, non-cancer exclusion, NO
    time window) that the cross-cohort comparison already uses.  MEASURED, the
    24 h window is what shrinks Os/Rh/Re to 36/51/74 unique ligands (below
    ``MIN_UNIQUE_FOR_GATE``); the paper funnel gives 333/334/250, so the same
    scorer resolves them.  Switching the spec changes the cohort, not the
    scorer, and must be applied consistently across every runner.
    """
    _load_env_default()
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    # Reuse the probe's dedup so the cohort is not a look-alike but the same one.
    from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates

    per_metal: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dup_stats: Dict[str, Dict[str, int]] = {}

    if cohort_spec == "paper_funnel":
        import os as _os

        import pandas as pd

        from molmetal.data.cytotox import build_paper_cohort

        data_dir = Path(
            _os.environ.get("MOLMETAL_DATA_DIR", str(PROJECT_ROOT / ".data" / "molmetal"))
        )
        raw = pd.read_csv(data_dir / "MetalCytoToxDB.csv", low_memory=False)
        for m in metals:
            sub = build_paper_cohort(raw, metal=m)
            ic50 = sub["IC50_Dark_value"].astype(float)
            # loader definition: active := IC50_Dark_value < 10 uM
            y_raw = (ic50 < 10.0).to_numpy(dtype=int)
            s_u, y_u, n_raw = collapse_duplicates(
                sub["SMILES_Ligands"].astype(str).to_numpy(), y_raw
            )
            per_metal[m] = (list(s_u), np.asarray(y_u, dtype=np.int64))
            dup_stats[m] = {
                "n_rows_raw": int(n_raw),
                "n_unique": int(len(s_u)),
                "pos_rate_dedup": float(np.mean(y_u)) if len(y_u) else 0.0,
            }
        return per_metal, dup_stats

    if cohort_spec != "24h":
        raise ValueError(f"unknown cohort_spec {cohort_spec!r} (expected 24h|paper_funnel)")

    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(
                time_threshold=float(time_threshold),
                ic50_min=float(ic50_min),
                metal_whitelist=[m],
            )
        )
        s_u, y_u, n_raw = collapse_duplicates(ds.smiles, ds.active)
        per_metal[m] = (list(s_u), np.asarray(y_u, dtype=np.int64))
        dup_stats[m] = {
            "n_rows_raw": int(n_raw),
            "n_unique": int(len(s_u)),
            "pos_rate_dedup": float(np.mean(y_u)) if len(y_u) else 0.0,
        }
    return per_metal, dup_stats


def lomo_split(
    per_metal: Dict[str, Tuple[List[str], np.ndarray]],
    held_out: str,
    metal_vocab: Tuple[str, ...] = METALS,
) -> Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    int,
]:
    """LOMO split with the probe's leakage rule.

    A ligand appearing in the held-out metal's test set is dropped from the
    training pool, so no ligand can straddle the split.

    Returns ``(tr_s, tr_y, tr_metal_idx, te_s, te_y, te_metal_idx, n_leaked)``
    where the ``*_metal_idx`` arrays index ``metal_vocab`` — the per-row metal
    id the PA15 conditioning arms consume.
    """
    te_s, te_y = per_metal[held_out]
    blocked = set(te_s)
    tr_s: List[str] = []
    tr_y: List[int] = []
    tr_m: List[int] = []
    n_leaked = 0
    for m, (s_list, y_arr) in per_metal.items():
        if m == held_out:
            continue
        mid = metal_vocab.index(m)
        for s, yv in zip(s_list, y_arr.tolist()):
            if s in blocked:
                n_leaked += 1
                continue
            tr_s.append(s)
            tr_y.append(int(yv))
            tr_m.append(mid)
    te_m = [metal_vocab.index(held_out)] * len(te_s)
    return (
        tr_s,
        np.asarray(tr_y, dtype=np.int64),
        np.asarray(tr_m, dtype=np.int64),
        list(te_s),
        np.asarray(te_y, dtype=np.int64),
        np.asarray(te_m, dtype=np.int64),
        n_leaked,
    )


# ---------------------------------------------------------------------------
# Featurisation cache (chemprop v2 SimpleMoleculeMolGraphFeaturizer).
# ---------------------------------------------------------------------------
def featurize_cache(
    smiles_iter,
) -> Tuple[Dict[str, MolGraph], int, int]:
    """SMILES -> MolGraph, computed once per unique SMILES.

    Invalid SMILES get a 1-atom / 0-bond dummy graph so ``BatchMolGraph`` keeps
    a consistent shape (mirrors ``train_chemprop_v2_lomo.prefeaturize_all_smiles``).
    """
    ftr = SimpleMoleculeMolGraphFeaturizer()
    cache: Dict[str, MolGraph] = {}
    atom_fdim = -1
    bond_fdim = -1
    for smi in smiles_iter:
        if smi in cache:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("RDKit parse failed")
            mg = ftr(mol)
            if atom_fdim < 0:
                atom_fdim = int(mg.V.shape[1])
                bond_fdim = int(mg.E.shape[1])
            cache[smi] = mg
        except Exception:
            if atom_fdim < 0:
                tmp = ftr(Chem.MolFromSmiles("C"))
                atom_fdim = int(tmp.V.shape[1])
                bond_fdim = int(tmp.E.shape[1])
            cache[smi] = MolGraph(
                np.zeros((1, atom_fdim), dtype=np.single),
                np.zeros((0, bond_fdim), dtype=np.single),
                np.zeros((2, 0), dtype=int),
                np.zeros((0,), dtype=int),
            )
    return cache, atom_fdim, bond_fdim


# ---------------------------------------------------------------------------
# Model — chemprop encoder + single pooled binary head (no metal embedding).
# ---------------------------------------------------------------------------
class ChempropBinaryLOMO(nn.Module):
    """chemprop encoder + one pooled binary head, optionally metal-conditioned.

    ``metal_mode``:
      * ``none``  — head sees the graph readout only (the PA3 scorer).
      * ``token`` — head sees ``[h ; E(metal_id)]`` with a learnable embedding.
      * ``desc``  — head sees ``[h ; W·d(metal_id)]`` where ``d`` is a FIXED
        descriptor vector (buffer) and ``W`` a shared learned projection.  In
        LOMO the held-out metal is unseen as a *token* but its descriptor is
        known, so this is the only arm that can transfer metal identity.
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        ffn_hidden: int,
        ffn_layers: int,
        dropout: float,
        metal_mode: str = DEFAULT_METAL_MODE,
        n_metals: int = len(METALS),
        metal_embed_dim: int = DEFAULT_METAL_EMBED_DIM,
        metal_desc: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.metal_mode = str(metal_mode)
        self.encoder = ChempropV2Encoder(hidden_dim=hidden_dim, depth=depth)
        extra = 0
        if self.metal_mode == "token":
            self.metal_embed = nn.Embedding(int(n_metals), int(metal_embed_dim))
            extra = int(metal_embed_dim)
        elif self.metal_mode == "desc":
            if metal_desc is None:
                raise ValueError("metal_mode='desc' requires metal_desc")
            desc = torch.as_tensor(np.asarray(metal_desc, dtype=np.float32))
            self.register_buffer("metal_desc", desc, persistent=False)
            self.metal_proj = nn.Linear(desc.shape[1], int(metal_embed_dim))
            extra = int(metal_embed_dim)
        elif self.metal_mode == "film":
            # L3: FiLM — the metal descriptor produces a per-channel scale and
            # shift applied to the pooled graph readout, rather than being
            # concatenated.  Unlike the concat arms this modulates the encoder
            # output multiplicatively, so the metal can gate which ligand
            # features reach the head.
            if metal_desc is None:
                raise ValueError("metal_mode='film' requires metal_desc")
            desc = torch.as_tensor(np.asarray(metal_desc, dtype=np.float32))
            self.register_buffer("metal_desc", desc, persistent=False)
            self.film = nn.Sequential(
                nn.Linear(desc.shape[1], int(metal_embed_dim)),
                nn.ReLU(),
                nn.Linear(int(metal_embed_dim), 2 * int(hidden_dim)),
            )
            extra = 0
        elif self.metal_mode != "none":
            raise ValueError(f"unknown metal_mode={self.metal_mode!r}")
        self.head = _make_metal_head(
            input_dim=hidden_dim + extra,
            hidden_dim=ffn_hidden,
            n_layers=ffn_layers,
            dropout=dropout,
        )

    def _metal_ctx(self, metal_ids: Optional[torch.Tensor], n: int, ref: torch.Tensor):
        """Return the metal context block, or ``None`` for metal_mode='none'."""
        if self.metal_mode == "none":
            return None
        if metal_ids is None:
            metal_ids = torch.zeros(n, dtype=torch.long, device=ref.device)
        metal_ids = metal_ids.to(ref.device).long()
        if self.metal_mode == "token":
            return self.metal_embed(metal_ids)
        return self.metal_proj(self.metal_desc[metal_ids])

    def logits(
        self,
        bmg: BatchMolGraph,
        metal_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_v = self.encoder.message_passing(bmg)  # (sum_N, hidden_dim)
        h = self.encoder.agg(h_v, bmg.batch)  # (B, hidden_dim)
        if self.metal_mode == "film":
            if metal_ids is None:
                metal_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
            metal_ids = metal_ids.to(h.device).long()
            gamma, beta = self.film(self.metal_desc[metal_ids]).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        else:
            ctx = self._metal_ctx(metal_ids, h.shape[0], h)
            if ctx is not None:
                h = torch.cat([h, ctx], dim=-1)
        return self.head.train_step(h).squeeze(-1)  # (B,)


def _batched(mgs: List[MolGraph], idx: np.ndarray, device: torch.device):
    bmg = BatchMolGraph([mgs[i] for i in idx])
    bmg.to(device)  # in-place; returns None
    return bmg


# ---------------------------------------------------------------------------
# One (fold, seed) run.
# ---------------------------------------------------------------------------
def _set_adapt_scope(model, scope: str) -> List[str]:
    """C4: restrict K-shot fine-tuning to a subset of parameters.

    ``full`` = every parameter (the historical arm).  ``head`` = only the FFN
    prediction head.  ``head_metal`` = head plus the metal-conditioning modules
    (``metal_embed`` / ``metal_proj`` / ``film``) when present.  Returns the
    names of the modules left trainable, for the report.
    """
    scope = str(scope)
    if scope == "full":
        for p in model.parameters():
            p.requires_grad = True
        return ["<all>"]

    for p in model.parameters():
        p.requires_grad = False

    trainable: List[str] = []
    if scope in ("head", "head_metal"):
        trainable.append("head")
        if scope == "head_metal":
            for name in ("metal_embed", "metal_proj", "film"):
                if hasattr(model, name):
                    trainable.append(name)
    else:
        raise ValueError(f"unknown adapt_scope={scope!r}")

    for name in trainable:
        for p in getattr(model, name).parameters():
            p.requires_grad = True
    return trainable


def run_one(
    *,
    model_factory,
    tr_mgs: List[MolGraph],
    tr_y: np.ndarray,
    te_mgs: List[MolGraph],
    te_y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    use_amp: bool,
    atom_fdim: int,
    bond_fdim: int,
    tr_metal: Optional[np.ndarray] = None,
    te_metal: Optional[np.ndarray] = None,
    return_scores: bool = False,
    te_support_idx: Optional[np.ndarray] = None,
    te_query_idx: Optional[np.ndarray] = None,
    ft_epochs: int = 0,
    ft_lr: float = 1.0e-3,
    adapt_scope: str = "full",
    max_steps: int = 0,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model_factory()
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n_tr = len(tr_mgs)
    idx = np.arange(n_tr)
    t0 = time.time()
    last_loss = float("nan")
    model.train()
    steps_done = 0
    for _epoch in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        seen = 0
        for start in range(0, n_tr, batch_size):
            bidx = idx[start : start + batch_size]
            bmg = _batched(tr_mgs, bidx, device)
            mid = (
                torch.from_numpy(tr_metal[bidx]).to(device)
                if tr_metal is not None
                else None
            )
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logit = model.logits(bmg, mid)
            tgt = torch.from_numpy(tr_y[bidx]).float().to(device)
            loss = loss_fn(logit.float(), tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.item()) * len(bidx)
            seen += len(bidx)
            steps_done += 1
            # C2 budget matching: cap the control arms at the SAME number of
            # optimizer steps the K-shot fine-tune performs, so the contrast is
            # "same compute, different procedure" rather than "different compute".
            if max_steps and steps_done >= int(max_steps):
                break
        last_loss = total / max(seen, 1)
        if max_steps and steps_done >= int(max_steps):
            break

    # Held-out scoring.
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(te_mgs), batch_size):
            bidx = np.arange(start, min(start + batch_size, len(te_mgs)))
            bmg = _batched(te_mgs, bidx, device)
            mid = (
                torch.from_numpy(te_metal[bidx]).to(device)
                if te_metal is not None
                else None
            )
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logit = model.logits(bmg, mid)
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)

    auc = float("nan")
    pr_auc = float("nan")
    if len(p) and len(set(te_y.tolist())) >= 2:
        auc = float(roc_auc_score(te_y, p))
        pr_auc = float(average_precision_score(te_y, p))

    out: Dict[str, object] = {
        "seed": int(seed),
        "n_train": int(n_tr),
        "n_test": int(len(te_mgs)),
        "test_auc": auc,
        "test_pr_auc": pr_auc,
        "final_train_loss": float(last_loss),
        "elapsed_s": float(time.time() - t0),
        "optimizer_steps": int(steps_done),
    }

    # ------------------------------------------------------------------
    # L3 K-shot adaptation: after the zero-shot model is frozen, fine-tune on
    # a small labelled SUPPORT set drawn from the held-out metal itself, then
    # re-score the disjoint QUERY set.  Zero-shot-on-query and k-shot-on-query
    # share the identical query rows, so the comparison is paired.
    # ------------------------------------------------------------------
    if (
        ft_epochs > 0
        and te_support_idx is not None
        and te_query_idx is not None
        and len(te_support_idx) > 0
        and len(te_query_idx) > 0
    ):
        sup_idx = np.asarray(te_support_idx, dtype=int)
        q_idx = np.asarray(te_query_idx, dtype=int)

        def _auc_on(probs_all: np.ndarray, idx: np.ndarray) -> float:
            yy = te_y[idx]
            if len(set(yy.tolist())) < 2:
                return float("nan")
            return float(roc_auc_score(yy, probs_all[idx]))

        zs_query_auc = _auc_on(p, q_idx)
        qy = te_y[q_idx]

        sup_mgs = [te_mgs[i] for i in sup_idx]
        sup_y = te_y[sup_idx]
        sup_m = te_metal[sup_idx] if te_metal is not None else None
        trainable_modules = _set_adapt_scope(model, adapt_scope)
        optim_ft = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=ft_lr
        )
        s_idx = np.arange(len(sup_mgs))
        model.train()
        for _e in range(ft_epochs):
            np.random.shuffle(s_idx)
            for start in range(0, len(sup_mgs), batch_size):
                bidx = s_idx[start : start + batch_size]
                bmg = _batched(sup_mgs, bidx, device)
                mid = (
                    torch.from_numpy(sup_m[bidx]).to(device)
                    if sup_m is not None
                    else None
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logit = model.logits(bmg, mid)
                tgt = torch.from_numpy(sup_y[bidx]).float().to(device)
                loss = loss_fn(logit.float(), tgt)
                optim_ft.zero_grad()
                loss.backward()
                optim_ft.step()

        model.eval()
        pq: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(q_idx), batch_size):
                bidx = q_idx[start : start + batch_size]
                bmg = _batched(te_mgs, bidx, device)
                mid = (
                    torch.from_numpy(te_metal[bidx]).to(device)
                    if te_metal is not None
                    else None
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logit = model.logits(bmg, mid)
                pq.append(torch.sigmoid(logit.float()).cpu().numpy())
        pq_all = np.concatenate(pq) if pq else np.zeros((0,), dtype=np.float32)
        ks_auc = (
            float(roc_auc_score(qy, pq_all))
            if len(qy) and len(set(qy.tolist())) >= 2
            else float("nan")
        )
        out["kshot"] = {
            "k": int(len(sup_idx)),
            "query_n": int(len(q_idx)),
            "ft_epochs": int(ft_epochs),
            "ft_steps": int(ft_epochs) * max(1, int(np.ceil(len(sup_idx) / float(batch_size)))),
            "adapt_scope": str(adapt_scope),
            "trainable_modules": list(trainable_modules),
            "zero_shot_query_auc": zs_query_auc,
            "kshot_query_auc": ks_auc,
            "delta": (ks_auc - zs_query_auc)
            if not (np.isnan(ks_auc) or np.isnan(zs_query_auc))
            else float("nan"),
        }

    if return_scores:
        out["test_labels"] = [int(v) for v in te_y.tolist()]
        out["test_probs"] = [float(v) for v in p.tolist()]
    return out


# ---------------------------------------------------------------------------
# Within-metal 5-fold CV — the field's own setting, and the paper's Table-1
# "Within" column.  Split protocol mirrors the probe exactly:
# StratifiedKFold(n_splits, shuffle=True, random_state=seed), and the reported
# number is the MEAN of the per-fold AUCs (not the pooled out-of-fold AUC).
# ---------------------------------------------------------------------------
def within_metal_chemprop(
    *,
    model_factory,
    s_u: List[str],
    y_u: np.ndarray,
    cache: Dict[str, MolGraph],
    seeds: List[int],
    n_splits: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    use_amp: bool,
    atom_fdim: int,
    bond_fdim: int,
    metal_idx: Optional[int] = None,
) -> Dict[str, object]:
    from sklearn.model_selection import StratifiedKFold

    if len(set(y_u.tolist())) < 2 or len(y_u) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y_u)} too small or single-class"}

    mgs = [cache[s] for s in s_u]
    # Within one metal the id is constant; the conditioning arms therefore see a
    # constant context block (absorbed into the head bias) — the within column is
    # a control that conditioning does not change the in-distribution setting.
    m_idx = np.full(len(y_u), 0 if metal_idx is None else int(metal_idx), dtype=np.int64)
    per_seed: Dict[str, Dict[str, object]] = {}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        fold_aucs: List[float] = []
        fold_pr: List[float] = []
        for tr_idx, te_idx in skf.split(np.zeros(len(y_u)), y_u):
            rec = run_one(
                model_factory=model_factory,
                tr_mgs=[mgs[i] for i in tr_idx],
                tr_y=y_u[tr_idx],
                te_mgs=[mgs[i] for i in te_idx],
                te_y=y_u[te_idx],
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=int(seed),
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                tr_metal=m_idx[tr_idx],
                te_metal=m_idx[te_idx],
            )
            fold_aucs.append(rec["test_auc"])
            fold_pr.append(rec["test_pr_auc"])
        per_seed[str(seed)] = {
            "fold_aucs": fold_aucs,
            "mean_fold_auc": float(np.mean(fold_aucs)),
            "fold_std": float(np.std(fold_aucs)),
            "mean_fold_pr_auc": float(np.mean(fold_pr)),
        }
    means = [v["mean_fold_auc"] for v in per_seed.values()]
    return {
        "status": "MEASURED",
        "n_unique": int(len(y_u)),
        "n_splits": int(n_splits),
        "pos_rate": float(y_u.mean()),
        "mean_auc": float(np.mean(means)),
        "std": float(np.std(means)),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# CLI (§5.11: --config + run-selecting flags only)
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=str, default=None, help="§5.11 YAML config path")
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--ffn-hidden", type=int, default=DEFAULT_FFN_HIDDEN)
    p.add_argument("--ffn-layers", type=int, default=DEFAULT_FFN_LAYERS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--metal-mode",
        choices=("none", "token", "desc", "film"),
        default=DEFAULT_METAL_MODE,
        help=(
            "PA15/L3 metal conditioning: none = metal-agnostic PA3 scorer; "
            "token = learnable per-metal embedding concatenated to the graph "
            "readout (untrained for the held-out metal in LOMO); desc = fixed "
            "physicochemical descriptor vector (from transition_metals_30.csv) "
            "through a shared learned projection — the only arm whose held-out "
            "representation is known at test time; film = the same descriptor "
            "drives a per-channel scale+shift on the pooled readout (L3)."
        ),
    )
    p.add_argument("--metal-embed-dim", type=int, default=DEFAULT_METAL_EMBED_DIM)
    p.add_argument(
        "--k-shot",
        type=int,
        default=0,
        help=(
            "L3 adaptation: hold out K labelled ligands from the held-out "
            "metal as a support set, fine-tune on them after the zero-shot "
            "model is frozen, and re-score the disjoint query set.  0 = off "
            "(the zero-shot LOMO protocol)."
        ),
    )
    p.add_argument(
        "--k-shot-ft-epochs",
        type=int,
        default=20,
        help="Fine-tuning epochs for --k-shot (default: 20).",
    )
    p.add_argument(
        "--k-shot-ft-lr",
        type=float,
        default=1.0e-3,
        help="Fine-tuning learning rate for --k-shot (default: 1e-3).",
    )
    p.add_argument(
        "--k-shot-adapt-scope",
        choices=("full", "head", "head_metal"),
        default="full",
        help=(
            "C4 adaptation depth: which parameters the K-shot fine-tune may "
            "update.  full = every parameter (historical arm); head = only the "
            "FFN prediction head (a linear probe that cannot destroy the "
            "pretrained encoder); head_metal = head plus the metal-conditioning "
            "modules when present.  Small-K full fine-tuning can overfit the "
            "support set and lower query AUC; the frozen-encoder arms test that."
        ),
    )
    p.add_argument(
        "--support-sampler",
        choices=("stratified", "random", "scaffold_diverse", "activity_scaffold"),
        default="stratified",
        help=(
            "How the K-shot support set is drawn from the held-out metal "
            "(C3/W3).  stratified = historical label-stratified random; "
            "scaffold_diverse / activity_scaffold maximise Bemis-Murcko "
            "coverage (the latter with label balance)."
        ),
    )
    p.add_argument(
        "--kshot-control-arms",
        nargs="*",
        default=[],
        choices=("from_scratch", "concat_retrain"),
        help=(
            "C2 controls: 'from_scratch' trains a fresh model on ONLY the K "
            "support ligands; 'concat_retrain' trains a fresh model on the "
            "source pool ++ the K support rows.  Both are scored on the "
            "identical query set, so they separate 'the adaptation' from "
            "'the extra K labels'."
        ),
    )
    p.add_argument(
        "--time-threshold", type=float, default=DEFAULT_TIME_THRESHOLD)
    p.add_argument("--ic50-min", type=float, default=DEFAULT_IC50_MIN)
    p.add_argument(
        "--cohort-spec",
        choices=("24h", "paper_funnel"),
        default="24h",
        help=(
            "24h = historical probe cohort (CytotoxFilter time<=24h); "
            "paper_funnel = Krasnov-style build_paper_cohort (no time window), "
            "which resolves Os/Rh/Re.  See W1 / blk_w1_cohort_expansion_probe.py."
        ),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--mode",
        choices=("cross", "within", "both"),
        default="cross",
        help=(
            "cross = leave-one-metal-out (the paper's Table-1 cross column); "
            "within = stratified 5-fold CV inside each metal (Table-1 within "
            "column); both = run each.  Default: cross."
        ),
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="StratifiedKFold splits for --mode within (default: 5, matches the probe).",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help=(
            "label-permutation null shuffles per cross fold (0 = skip).  "
            "Run on --perm-seed's fold predictions, matching "
            "metal_activity_transfer_permutation_null.json."
        ),
    )
    p.add_argument(
        "--perm-seed",
        type=int,
        default=42,
        help="Which seed's cross-fold predictions the permutation null uses.",
    )
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BF16 autocast on the chemprop encoder forward (default: True).",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument(
        "--metric-name",
        type=str,
        default="pa3_chemprop_lomo_probe_cohort",
        help=(
            "value written to the report's 'metric' field — a run descriptor, "
            "not a hyperparameter (lets the PA15 arms name their own artefact "
            "without renaming PA3's canonical file)."
        ),
    )

    args = p.parse_args(argv)
    if getattr(args, "config", None):
        from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else out_dir / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    use_amp = bool(args.bf16) and device.type == "cuda"

    logger.info(
        "device=%s bf16=%s metals=%s seeds=%s epochs=%d hidden=%d depth=%d "
        "ffn=%dx%d dropout=%.2f batch=%d lr=%.1e cohort=(%s,ic50>=%.2f) "
        "metal_mode=%s metal_embed_dim=%d",
        device,
        args.bf16,
        args.metals,
        args.seeds,
        args.epochs,
        args.hidden,
        args.depth,
        args.ffn_layers,
        args.ffn_hidden,
        args.dropout,
        args.batch_size,
        args.lr,
        args.cohort_spec,
        args.ic50_min,
        args.metal_mode,
        args.metal_embed_dim,
    )

    # PA15 — stable metal vocabulary (indices must not move with --metals) and
    # the fixed descriptor table.  The descriptors are public periodic-table
    # constants read from transition_metals_30.csv, never labels.
    metal_vocab: Tuple[str, ...] = METALS
    metal_desc, desc_names = metal_descriptor_table(metal_vocab)
    if args.metal_mode == "desc":
        logger.info(
            "metal descriptors (%d cols): %s",
            metal_desc.shape[1],
            desc_names,
        )

    t_cohort = time.time()
    per_metal, dup_stats = build_probe_cohort(
        list(args.metals), args.time_threshold, args.ic50_min,
        cohort_spec=args.cohort_spec,
    )
    logger.info(
        "probe cohort built in %.1fs: %s",
        time.time() - t_cohort,
        {m: dup_stats[m]["n_unique"] for m in per_metal},
    )

    # Featurise every unique SMILES once.
    all_smiles = set()
    for s_list, _ in per_metal.values():
        all_smiles.update(s_list)
    t_feat = time.time()
    cache, atom_fdim, bond_fdim = featurize_cache(sorted(all_smiles))
    logger.info(
        "featurised %d unique SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
        len(cache),
        time.time() - t_feat,
        atom_fdim,
        bond_fdim,
    )

    def model_factory():
        return ChempropBinaryLOMO(
            hidden_dim=args.hidden,
            depth=args.depth,
            ffn_hidden=args.ffn_hidden,
            ffn_layers=args.ffn_layers,
            dropout=args.dropout,
            metal_mode=args.metal_mode,
            n_metals=len(metal_vocab),
            metal_embed_dim=args.metal_embed_dim,
            metal_desc=metal_desc,
        )

    folds: List[Dict[str, object]] = []
    runs: List[Dict[str, object]] = []
    sweep_t0 = time.time()
    for held_out in args.metals if args.mode in ("cross", "both") else []:
        if held_out not in per_metal:
            logger.warning("held_out=%s missing from cohort; skip", held_out)
            continue
        tr_s, tr_y, tr_m, te_s, te_y, te_m, n_leaked = lomo_split(
            per_metal, held_out, metal_vocab
        )
        tr_mgs = [cache[s] for s in tr_s]
        te_mgs = [cache[s] for s in te_s]
        logger.info(
            "fold=%s n_train=%d n_test=%d n_leaked=%d pos_rate_test=%.3f",
            held_out,
            len(tr_s),
            len(te_s),
            n_leaked,
            float(np.mean(te_y)) if len(te_y) else 0.0,
        )
        per_seed: Dict[str, Dict[str, object]] = {}
        for seed in args.seeds:
            want_scores = int(seed) == int(args.perm_seed) and args.n_permutations > 0
            sup_idx = q_idx = None
            if args.k_shot > 0 and len(te_y) > args.k_shot + 10:
                # L3: K-shot support / disjoint query, seeded so the support set
                # is reproducible.  The query rows are identical across the
                # zero-shot and k-shot evaluations (paired), and identical
                # across support-sampler arms for a given (metal, seed).
                scaffolds = None
                if args.support_sampler in ("scaffold_diverse", "activity_scaffold"):
                    from molmetal.scripts.metal_activity_transfer_probe import (
                        _parent_murcko,
                    )

                    scaffolds = [_parent_murcko(str(s)) for s in te_s]
                sup_idx = _draw_support(
                    te_y, scaffolds, int(args.k_shot), int(seed), args.support_sampler
                )
                q_idx = np.setdiff1d(np.arange(len(te_y)), sup_idx)
            rec = run_one(
                model_factory=model_factory,
                tr_mgs=tr_mgs,
                tr_y=tr_y,
                te_mgs=te_mgs,
                te_y=te_y,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=int(seed),
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                tr_metal=tr_m,
                te_metal=te_m,
                return_scores=want_scores,
                te_support_idx=sup_idx,
                te_query_idx=q_idx,
                ft_epochs=int(args.k_shot_ft_epochs) if args.k_shot > 0 else 0,
                ft_lr=float(args.k_shot_ft_lr),
                adapt_scope=str(args.k_shot_adapt_scope),
            )
            # ---- C2 controls: is it the adaptation, or the extra K labels? --
            if (
                sup_idx is not None
                and q_idx is not None
                and len(args.kshot_control_arms) > 0
                and "kshot" in rec
            ):
                ctrl: Dict[str, float] = {}
                # C2 budget match: the K-shot fine-tune performs
                # ft_epochs * ceil(K/batch) optimizer steps.  Cap each control
                # at exactly that many, so the contrast is procedure, not compute.
                kshot_steps = int(args.k_shot_ft_epochs) * max(
                    1, int(np.ceil(len(sup_idx) / float(args.batch_size)))
                )
                for arm in args.kshot_control_arms:
                    if arm == "from_scratch":
                        a_mgs, a_y, a_m = (
                            [te_mgs[i] for i in sup_idx],
                            te_y[sup_idx],
                            te_m[sup_idx] if te_m is not None else None,
                        )
                    else:  # concat_retrain
                        a_mgs, a_y, a_m = _concat_pool(
                            tr_mgs, tr_y, tr_m,
                            [te_mgs[i] for i in sup_idx],
                            te_y[sup_idx],
                            te_m[sup_idx] if te_m is not None else None,
                        )
                    r = run_one(
                        model_factory=model_factory,
                        tr_mgs=a_mgs,
                        tr_y=a_y,
                        te_mgs=te_mgs,
                        te_y=te_y,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=device,
                        seed=int(seed),
                        use_amp=use_amp,
                        atom_fdim=atom_fdim,
                        bond_fdim=bond_fdim,
                        tr_metal=a_m,
                        te_metal=te_m,
                        return_scores=True,
                        te_support_idx=None,
                        te_query_idx=None,
                        ft_epochs=0,
                        ft_lr=float(args.k_shot_ft_lr),
                        max_steps=kshot_steps,
                    )
                    ctrl[f"{arm}_query_auc"] = _auc_at(
                        np.asarray(r["test_probs"]), te_y, q_idx
                    )
                    ctrl[f"{arm}_n_train"] = int(r["n_train"])
                    ctrl[f"{arm}_steps"] = int(r["optimizer_steps"])
                rec["kshot"].update(ctrl)
            per_seed[str(seed)] = rec
            runs.append({"metal": held_out, **rec})
            if args.k_shot > 0 and "kshot" in rec:
                ks = rec["kshot"]
                logger.info(
                    "fold=%s seed=%d k=%d zero-shot-query=%.4f k-shot-query=%.4f delta=%+.4f",
                    held_out,
                    seed,
                    ks["k"],
                    ks["zero_shot_query_auc"],
                    ks["kshot_query_auc"],
                    ks["delta"],
                )
            logger.info(
                "fold=%s seed=%d test_auc=%.4f pr_auc=%.4f loss=%.4f %.1fs",
                held_out,
                seed,
                rec["test_auc"],
                rec["test_pr_auc"],
                rec["final_train_loss"],
                rec["elapsed_s"],
            )
        aucs = [
            r["test_auc"]
            for r in per_seed.values()
            if isinstance(r["test_auc"], float) and not np.isnan(r["test_auc"])
        ]
        fold_rec: Dict[str, object] = {
            "metal": held_out,
            "status": "MEASURED" if aucs else "SKIPPED",
            "n_train": int(len(tr_s)),
            "n_test": int(len(te_s)),
            "n_train_rows_dropped_as_leaked": int(n_leaked),
            "pos_rate_test": float(np.mean(te_y)) if len(te_y) else 0.0,
            "per_seed_auc": {k: v["test_auc"] for k, v in per_seed.items()},
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "std": float(np.std(aucs)) if aucs else 0.0,
            "n_seeds_ok": len(aucs),
            "per_seed": per_seed,
        }
        if args.k_shot > 0:
            zs = [
                r["kshot"]["zero_shot_query_auc"]
                for r in per_seed.values()
                if "kshot" in r and not np.isnan(r["kshot"]["zero_shot_query_auc"])
            ]
            ks = [
                r["kshot"]["kshot_query_auc"]
                for r in per_seed.values()
                if "kshot" in r and not np.isnan(r["kshot"]["kshot_query_auc"])
            ]
            fold_rec["kshot_summary"] = {
                "k": int(args.k_shot),
                "ft_epochs": int(args.k_shot_ft_epochs),
                "zero_shot_query_mean": float(np.mean(zs)) if zs else float("nan"),
                "kshot_query_mean": float(np.mean(ks)) if ks else float("nan"),
                "delta_mean": float(np.mean(np.asarray(ks) - np.asarray(zs)))
                if (len(zs) and len(ks) and len(zs) == len(ks))
                else float("nan"),
                "n_seeds": len(ks),
            }
        # Label-permutation null on the --perm-seed run's fixed ranking.
        perm_rec = per_seed.get(str(args.perm_seed))
        if args.n_permutations > 0 and perm_rec and "test_probs" in perm_rec:
            from molmetal.scripts.metal_activity_transfer_probe import permutation_null

            fold_rec["permutation_null"] = permutation_null(
                perm_rec["test_labels"],
                perm_rec["test_probs"],
                int(args.n_permutations),
                int(args.perm_seed),
            )
            # Keep the raw scores in the report but out of the fold summary.
            fold_rec["perm_seed_scores"] = {
                "seed": int(args.perm_seed),
                "test_labels": perm_rec["test_labels"],
                "test_probs": perm_rec["test_probs"],
            }
            pn = fold_rec["permutation_null"]
            logger.info(
                "fold=%s permutation-null: obs=%.4f null_sd=%.4f "
                "p_above=%.4f p_below=%.4f",
                held_out,
                pn.get("observed_auc", float("nan")),
                pn.get("null_sd", float("nan")),
                pn.get("p_above_chance", float("nan")),
                pn.get("p_below_chance", float("nan")),
            )
        folds.append(fold_rec)

    mean_across = (
        float(
            np.nanmean(
                [
                    f["mean_auc"]
                    for f in folds
                    if not np.isnan(f["mean_auc"])
                ]
            )
        )
        if folds
        else float("nan")
    )
    total_min = (time.time() - sweep_t0) / 60.0

    # ------------------------------------------------------------------
    # Within-metal 5-fold CV (the paper's Table-1 "Within" column).
    # ------------------------------------------------------------------
    within: Dict[str, object] = {}
    if args.mode in ("within", "both"):
        for held_metal in args.metals:
            if held_metal not in per_metal:
                continue
            s_u, y_u = per_metal[held_metal]
            within[held_metal] = within_metal_chemprop(
                model_factory=model_factory,
                s_u=s_u,
                y_u=y_u,
                cache=cache,
                seeds=[int(s) for s in args.seeds],
                n_splits=int(args.n_splits),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                use_amp=use_amp,
                atom_fdim=atom_fdim,
                bond_fdim=bond_fdim,
                metal_idx=metal_vocab.index(held_metal),
            )
            w = within[held_metal]
            logger.info(
                "within %s: n=%s mean_auc=%.4f std=%.4f (%d splits)",
                held_metal,
                w.get("n_unique"),
                w.get("mean_auc", float("nan")),
                w.get("std", float("nan")),
                int(args.n_splits),
            )
    within_mean = (
        float(
            np.nanmean(
                [v["mean_auc"] for v in within.values() if v.get("status") == "MEASURED"]
            )
        )
        if within
        else float("nan")
    )

    note = (
        f"PA3 chemprop v2 (AtomMessagePassing d_h={args.hidden}, depth={args.depth}) "
        f"+ single pooled BinaryClassificationFFN ({args.ffn_layers}x{args.ffn_hidden}, "
        f"dropout={args.dropout}), metal_mode={args.metal_mode}, on the Paper-A "
        f"probe cohort "
        f"(MetalCytotoxDataset + CytotoxFilter(time<={args.time_threshold}h, "
        f"ic50>={args.ic50_min}uM) + collapse_duplicates, leak-dropped train pool). "
        f"mode={args.mode}; {len(runs)} cross (fold, seed) runs in "
        f"{total_min:.2f} min wall.  AUCs are MEASURED with sklearn roc_auc_score "
        f"on the sigmoid of the pooled head's logit.  Same cohort as "
        f"metrics/by_metric/metal_activity_transfer.json (the XGB probe); the "
        f"comparison isolates the scorer, not the cohort."
    )
    if args.metal_mode == "token":
        note += (
            "  NOTE (token arm): in LOMO the held-out metal's embedding row is "
            "never trained, so it stays at its random initialisation at test "
            "time — this arm measures whether an untrained token can help, not "
            "whether a learned one could."
        )
    elif args.metal_mode == "desc":
        note += (
            "  NOTE (desc arm): the held-out metal's descriptor vector is a known "
            "constant, so this is the only conditioning arm that can carry metal "
            "identity across the LOMO boundary."
        )

    report = {
        "metric": str(args.metric_name),
        "mode": str(args.mode),
        "metal_mode": str(args.metal_mode),
        "metal_embed_dim": int(args.metal_embed_dim),
        "description": (
            "ROC-AUC of the official chemprop v2 message-passing backbone on the "
            "Paper-A probe cohort — the same cohort as the Morgan-FP + XGBoost "
            "probe in metrics/by_metric/metal_activity_transfer.json.  "
            "--mode cross gives the leave-one-metal-out column (the paper's "
            "Table-1 cross), --mode within gives the stratified 5-fold CV inside "
            "each metal (Table-1 within).  Answers BLK-PA3: is the cross-metal "
            "collapse an artefact of a weak backbone?  With --metal-mode != none "
            "it also answers BLK-PA15: does explicit metal conditioning recover "
            "any of the cross-metal transfer?"
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/train_chemprop_lomo_paper_cohort.py",
        "model": (
            f"chemprop v2 AtomMessagePassing (d_h={args.hidden}, depth={args.depth}) "
            f"+ BinaryClassificationFFN (single pooled head); metal_mode="
            f"{args.metal_mode}"
            + (
                " (no metal features)"
                if args.metal_mode == "none"
                else (
                    f" (learnable metal embedding, dim={args.metal_embed_dim})"
                    if args.metal_mode == "token"
                    else (
                        f" (fixed descriptor vector [{', '.join(desc_names)}] "
                        f"-> shared linear projection to dim={args.metal_embed_dim})"
                    )
                )
            )
        ),
        "data_source": (
            "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838) — "
            "probe cohort"
        ),
        "cohort_spec": str(args.cohort_spec),
        "k_shot": int(args.k_shot),
        "k_shot_ft_epochs": int(args.k_shot_ft_epochs) if args.k_shot > 0 else 0,
        "k_shot_ft_lr": float(args.k_shot_ft_lr),
        "k_shot_adapt_scope": str(args.k_shot_adapt_scope),
        "support_sampler": str(args.support_sampler),
        "kshot_control_arms": list(args.kshot_control_arms),
        "cohort_rule": (
            (
                "per metal: build_paper_cohort (Krasnov-style: IC50-present, "
                "metal, measurement-level dedup, non-cancer exclusion, NO time "
                "window) then collapse_duplicates (one row per unique ligand, "
                "label = majority, ties -> ACTIVE)"
            )
            if args.cohort_spec == "paper_funnel"
            else (
                "per metal: MetalCytotoxDataset.from_csv(CytotoxFilter("
                f"time_threshold={args.time_threshold}, ic50_min={args.ic50_min}, "
                "metal_whitelist=[m])) then collapse_duplicates (one row per unique "
                "ligand, label = majority, ties -> ACTIVE)"
            )
        ),
        "leakage_rule": (
            "a ligand appearing in the held-out metal's test set is dropped from "
            "the LOMO training pool"
        ),
        "split_protocol": (
            "LOMO (leave-one-metal-out), ligand-disjoint"
            if args.mode == "cross"
            else (
                "within-metal stratified 5-fold CV"
                if args.mode == "within"
                else "LOMO (cross) + within-metal stratified 5-fold CV"
            )
        ),
        "comparison_baseline": "metrics/by_metric/metal_activity_transfer.json",
        "seeds": [int(s) for s in args.seeds],
        "duplication": dup_stats,
        "metal_conditioning": {
            "metal_mode": str(args.metal_mode),
            "metal_vocab": list(metal_vocab),
            "metal_embed_dim": int(args.metal_embed_dim),
            "descriptor_names": list(desc_names),
            "descriptor_source": str(METAL_TABLE_CSV.relative_to(PROJECT_ROOT)),
            "descriptor_table_standardised": {
                m: [float(v) for v in metal_desc[i]]
                for i, m in enumerate(metal_vocab)
            },
            "note": (
                "descriptors are public periodic-table constants read from "
                "transition_metals_30.csv (atomic_number, group, period parsed "
                "from the row column, oxidation_state, d_count = group - "
                "oxidation_state, and a coordination-geometry one-hot), "
                "standardised column-wise over the metal_vocab.  No labels enter "
                "the conditioning."
            ),
        },
        "config": {
            "metals": list(args.metals),
            "seeds": [int(s) for s in args.seeds],
            "mode": str(args.mode),
            "metal_mode": str(args.metal_mode),
            "metal_embed_dim": int(args.metal_embed_dim),
            "n_splits": int(args.n_splits),
            "n_permutations": int(args.n_permutations),
            "perm_seed": int(args.perm_seed),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "ffn_hidden": int(args.ffn_hidden),
            "ffn_layers": int(args.ffn_layers),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "time_threshold": float(args.time_threshold),
            "ic50_min": float(args.ic50_min),
            "device": str(device),
            "bf16_autocast": bool(use_amp),
        },
        "per_fold": folds,
        "mean_auc_across_metals": mean_across,
        "within_metal": within,
        "within_metal_mean_auc": within_mean,
        "total_min": float(total_min),
        "honest_framing_note": note,
        "runs": runs,
    }

    with out_json.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("wrote %s", out_json)
    if folds:
        logger.info("mean cross-metal AUC = %.4f", mean_across)
    if within:
        logger.info("mean within-metal AUC = %.4f", within_mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
