#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2026 Mol-Metal authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Wf-A142 (2026-10-31) — the Wf-A128 one-sided bond-length UPPER BOUND,
MEASURED on a CPU-only decode.

WHAT THIS IS
------------
Row 2 of ``/tmp/wfa140/frontier-lit.md``: TABASCO (Vonessen et al.,
arXiv:2507.00899) §3.5 "physically-constrained guidance" — pre-tabulated
element-pair distance BOUNDS, penalised ONE-SIDEDLY, parameter-free.  The
repo already IMPLEMENTS the "above the bound" half of that, behind three
DEFAULT-OFF kwargs on ``_BondLengthGuidance``
(``molmetal/adapters/flow_matching_lipman/__init__.py``):

  * ``upper_bound``            (default ``False``)  -- the switch
  * ``upper_bound_reach``      (default ``3.5`` A)  -- the bond head's own
                                                     candidate cutoff
  * ``upper_bound_strength``   (default ``None`` -> the spring constant)

This probe turns the switch ON and MEASURES what it does to a small CPU
decode.  It answers three questions with numbers that trace to a file on
disk:

  (a) does the one-sided bound change the decoded chemistry at all?
  (b) does it change the GEOMETRY — specifically, does it pull pairs out of
      the 2.5-3.5 A dead zone the spring's inner cutoff leaves open?
  (c) what does it cost (wall time)?

THE THREE ARMS (a decomposition, not three "settings")
--------------------------------------------------------
The bound lives INSIDE ``_BondLengthGuidance``, and the trainer only
constructs that object when ``bond_length_guidance_weight > 0``
(``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` :9200-9202,
``_with_bond_length_guidance``).  So a two-arm design ("bound off" vs
"bound on") CONFOUNDS the bound with the Wf-A75 spring unless the spring
weight is held fixed.  This probe therefore runs THREE arms:

  ``prod``    weight = 0.0        bound OFF   -- the PRODUCTION setting:
              every production CFM config in the repo leaves the guidance
              weight at its argparse default 0.0, so the guidance object is
              never even constructed (MEASURED, Wf-A128:
              ``molmetal/tests/test_wfa128_bond_length_reach.py``
              ``test_guidance_is_off_in_the_production_configs``).
  ``spring``  weight = W          bound OFF   -- the Wf-A75 MolDiff spring
              alone (the "bound OFF" arm proper).
  ``bound``   weight = W          bound ON    -- spring + the Wf-A128
              one-sided bound (the "bound ON" arm proper).

The bound's MARGINAL effect is ``bound`` minus ``spring``; the TOTAL effect
against production is ``bound`` minus ``prod``.  All three arms share the
same checkpoint, the same templates, and the same noise seed, so they are
paired at the NOISE level (identical x0); they diverge from the first ODE
step onward, which is the point.

CPU-ONLY
--------
This script is CPU-only by construction:

  * ``CUDA_VISIBLE_DEVICES`` is forced to ``""`` at MODULE IMPORT time,
    BEFORE ``torch`` is imported (setting it after the import has no
    effect);
  * the CUDA availability is asserted False at run time, and the run ABORTS
    if a CUDA device is visible;
  * every tensor is created on ``torch.device("cpu")``.

This OVERRIDES the project's standing GPU grant (AGENTS.md §5.1) for the
duration of this probe.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wfa142_upper_bound_probe.py \
        --ckpt results/wfa108j_budget36/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt \
        --n-clouds 16 --n-steps 50 --seed 0 \
        --out molmetal/reports/wfa142_upper_bound/probe.json

The output artifact is written through
``molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`` (POSIX
``os.replace``), with ``safe_overwrite_check`` + ``detect_truncation``
applied first (AGENTS.md §5.6).
"""

from __future__ import annotations

# CPU-ONLY, enforced BEFORE torch is imported.  Setting this after the
# torch import has no effect on an already-initialised CUDA runtime.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402  (import MUST follow the CUDA_VISIBLE_DEVICES set)

# --------------------------------------------------------------------------
# The dead zone this probe is about.  Both numbers are the repo's own.
# --------------------------------------------------------------------------
#: the Wf-A75 spring's inner cutoff — ``_INNER_BOND_CUTOFF`` in
#: ``molmetal/adapters/flow_matching_lipman/__init__.py``.
INNER_CUTOFF_A = 2.5
#: the YuelBond head's own candidate cutoff — ``flow.edge_cutoff`` in the
#: CFM configs and ``YuelBondDecoderHead.cutoff`` in
#: ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py``.
HEAD_CUTOFF_A = 3.5

#: the Wf-A128 repo config's own values —
#: ``configs/wfa128_bond_reach_upper_bound_2026_10_31.yaml`` lines 132/136/
#: 140/145 (``bond_length_guidance_weight`` / ``_upper_bound`` /
#: ``_upper_bound_reach`` / ``_upper_bound_strength``).
DEFAULT_GUIDANCE_WEIGHT = 0.1
DEFAULT_REACH_A = 3.5
DEFAULT_STRENGTH = 25.0


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------
def load_checkpoint_any(path: Path) -> Any:
    """Load a CFM checkpoint in EITHER serialisation the trainer emits.

    MEASURED (Wf-A81, 2026-10-18): since Wf-A30-1 the trainer writes
    checkpoints as a RAW PICKLE through
    ``wf386a3_atomic_write.atomic_write_file``, not through ``torch.save``;
    ``torch.load`` on such a file raises ``RuntimeError: Invalid magic
    number; corrupt file?``.  Order matters: ``torch.load`` first (zip
    magic), pickle only on the magic-number error, so a genuinely
    truncated file still raises instead of producing a confusing
    UnpicklingError.  This mirrors
    ``molmetal/scripts/cfm_property_panel.py::_load_checkpoint_any``.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        if "magic number" not in str(exc):
            raise
        with open(path, "rb") as fh:
            return pickle.load(fh)


def build_model(ckpt: Any):
    """Rebuild the CFMJointModel from the checkpoint's own serialised args.

    Uses the trainer's own factory (``build_joint_model``) with
    ``strict=True`` on the state dict, so a shape mismatch is reported
    rather than smoothed over.  Mirrors
    ``molmetal/scripts/cfm_property_panel.py::_rebuild_v2_model``.
    """
    import argparse

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_joint_model

    args_dict = ckpt.get("args") if isinstance(ckpt, dict) else None
    if not isinstance(args_dict, dict):
        raise SystemExit(
            "checkpoint carries no serialised 'args' dict; the architecture "
            "cannot be rebuilt honestly (a state_dict does not determine "
            "bond_head_kind / bond_z_table / the pair-feature flags)."
        )
    args = argparse.Namespace(**args_dict)
    model = build_joint_model(args)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(torch.device("cpu"))
    return model, args


def build_decoder(model: Any, args: Any):
    """Construct the YuelBond decoder with the TRAINER's own wiring.

    Every kwarg below is copied from the trainer's construction site
    (``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` :17296-17351)
    so the decode in this probe is the production decode, not a
    re-invention of it.
    """
    from molmetal.eval.bond_decision import bond_decision_settings
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    def _as_on_flag(value: Any, default: bool = True) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"on", "true", "1", "yes"}

    _decision, _threshold, _bias = bond_decision_settings(vars(args))
    return YuelBondBondDecoder(
        head=model.head,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        decision=_decision,
        decision_threshold=_threshold,
        connectivity_prior=_as_on_flag(
            getattr(args, "connectivity_prior", "on"), default=True
        ),
        connectivity_prior_min_logit=float(
            getattr(args, "connectivity_prior_min_logit", 0.0)
        ),
        class_logit_bias=_bias,
        dative_max_distance=float(
            getattr(args, "dative_distance_cutoff", 2.6)
        ),
        ring_prior=bool(getattr(args, "ring_prior", False)),
        ring_prior_min_ring_size=int(getattr(args, "ring_prior_min_ring_size", 4)),
        ring_prior_max_ring_size=(
            None
            if not getattr(args, "ring_prior_max_ring_size", None)
            or int(args.ring_prior_max_ring_size) <= 0
            else int(args.ring_prior_max_ring_size)
        ),
    )


def load_templates(args: Any, n_clouds: int) -> List[Any]:
    """Real val compositions from the checkpoint's OWN corpus.

    Only the COMPOSITION matters for the sampler: ``generate_clouds_from_noise``
    draws fresh noise and uses the template's ``atomic_numbers``; the
    template's geometry is not used.  The metal placement / dative cutoff
    are read from the checkpoint's own args so the cohort is built the way
    the checkpoint's trainer built it.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_csv_cohort

    csv_rel = getattr(args, "data_csv", None)
    if not csv_rel:
        raise SystemExit("checkpoint carries no 'data_csv'; no template cohort")
    csv_path = Path(str(csv_rel))
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.is_file():
        raise SystemExit(f"checkpoint's corpus is not on disk: {csv_path}")
    clouds = build_csv_cohort(
        str(csv_path),
        split=str(getattr(args, "val_split", "val")),
        n_clouds=int(n_clouds),
        seed=0,
        metal_placement=str(getattr(args, "metal_placement", "legacy")),
        dative_distance_cutoff=float(getattr(args, "dative_distance_cutoff", 2.6)),
    )
    if not clouds:
        raise SystemExit(f"no embeddable templates from {csv_path}")
    return list(clouds)[: int(n_clouds)]


# --------------------------------------------------------------------------
# geometry diagnostics on the sampled clouds
# --------------------------------------------------------------------------
def geometry_band_stats(clouds: Sequence[Any]) -> Dict[str, Any]:
    """Distance statistics of the SAMPLED clouds.

    The quantity of interest is the 2.5-3.5 A band — the DEAD ZONE: a pair
    there is inside the YuelBond head's own candidate set (so the head could
    bond it) but outside the spring's inner cutoff (so the guidance never
    touched it).  We report, per arm:

      * ``atoms_with_nn_in_band`` / ``frac_atoms_nn_in_band`` — atoms whose
        NEAREST NEIGHBOUR sits in the band;
      * ``mean_pairs_in_band`` — unordered pairs in the band per cloud;
      * ``mean_nn_distance`` — mean nearest-neighbour distance (the
        spring's reach is 2.5 A, so this says how many atoms the spring
        could act on at all);
      * ``frac_atoms_nn_below_1p2`` — atoms whose nearest neighbour is
        closer than 1.2 A, i.e. a STERIC CLASH.  The shortest realistic
        heavy-atom separation is ~1.2 A (a C-C triple bond); the frontier
        report names over-pulling into a clash as this technique's one
        real risk, so the clash fraction is measured, not assumed.

    Heavy atoms only: hydrogens are not in the bond head's candidate set
    the way heavy atoms are, and including them would swamp the count.

    MEASURED GOTCHA (this probe, 2026-10-31): ``torch.cdist(p, p)`` does
    NOT have an exact zero diagonal in float32 — the matmul path cancels
    catastrophically and leaves ~1e-4 A on the diagonal.  A naive
    ``d.min(dim=1).values`` therefore returns the SELF-distance for every
    atom and reports a mean nearest-neighbour of ~1e-4 A for a perfectly
    healthy geometry.  We compute the distances in float64 and fill the
    diagonal with +inf before taking the minimum.
    """
    atoms_total = 0
    atoms_nn_in_band = 0
    atoms_nn_clash = 0
    pairs_in_band_total = 0
    pairs_total = 0
    n_clouds = 0
    nn_dists: List[float] = []
    per_cloud: List[Dict[str, Any]] = []
    #: nearest-neighbour distance histogram edges (A).  The last bucket is
    #: open-ended.  Chosen to straddle the physically meaningful ranges:
    #: below 1.2 A = clash, 1.2-1.5 = short, 1.5-2.0 = normal bond,
    #: 2.0-2.5 = long / dative, 2.5-3.5 = the dead zone.
    NN_BINS = (0.0, 1.2, 1.5, 2.0, 2.5, 3.5, float("inf"))
    nn_hist = [0] * (len(NN_BINS) - 1)
    lo = float(INNER_CUTOFF_A)
    hi = float(HEAD_CUTOFF_A)
    for cloud in clouds:
        z = cloud.atomic_numbers.detach().cpu().long()
        keep = z > 1  # heavy atoms only
        pos = cloud.positions.detach().cpu().double()[keep]
        n = int(pos.shape[0])
        if n < 2:
            continue
        # float64 + +inf diagonal: see the MEASURED GOTCHA in the docstring.
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        off = ~torch.eye(n, dtype=torch.bool)
        dvals = d[off]
        nn = d.min(dim=1).values
        in_band = (nn >= lo) & (nn < hi)
        clash = nn < 1.2
        pairs_in_band = int(((dvals >= lo) & (dvals < hi)).sum().item())
        atoms_total += n
        atoms_nn_in_band += int(in_band.sum().item())
        atoms_nn_clash += int(clash.sum().item())
        pairs_in_band_total += pairs_in_band
        pairs_total += int(dvals.numel())
        nn_dists.extend(float(v) for v in nn.tolist())
        for b in range(len(NN_BINS) - 1):
            nn_hist[b] += int(
                ((nn >= NN_BINS[b]) & (nn < NN_BINS[b + 1])).sum().item()
            )
        per_cloud.append(
            {
                "n_heavy": n,
                "atoms_with_nn_in_band": int(in_band.sum().item()),
                "atoms_with_nn_below_1p2": int(clash.sum().item()),
                "pairs_in_band": pairs_in_band,
                "mean_nn_distance": float(nn.mean().item()),
            }
        )
        n_clouds += 1
    if not n_clouds:
        return {"n_clouds": 0}
    mean_nn = sum(nn_dists) / len(nn_dists) if nn_dists else None
    return {
        "n_clouds": n_clouds,
        "atoms_total": atoms_total,
        "atoms_with_nn_in_band": atoms_nn_in_band,
        "frac_atoms_nn_in_band": (
            atoms_nn_in_band / atoms_total if atoms_total else None
        ),
        "atoms_with_nn_below_1p2": atoms_nn_clash,
        "frac_atoms_nn_below_1p2": (
            atoms_nn_clash / atoms_total if atoms_total else None
        ),
        "nn_distance_histogram": {
            f"[{NN_BINS[b]:g},{NN_BINS[b + 1]:g})": nn_hist[b]
            for b in range(len(NN_BINS) - 1)
        },
        "mean_pairs_in_band_per_cloud": pairs_in_band_total / n_clouds,
        "frac_pairs_in_band": (
            pairs_in_band_total / pairs_total if pairs_total else None
        ),
        "mean_nn_distance": mean_nn,
        "band_definition": {
            "lo_A": lo,
            "hi_A": hi,
            "lo_source": "adapter _INNER_BOND_CUTOFF (Wf-A75 spring inner cutoff)",
            "hi_source": "YuelBond head candidate cutoff (flow.edge_cutoff)",
            "pair_counting": (
                "DIRECTED: each unordered pair is counted twice, matching "
                "radius_edge_index (which also builds a directed graph, and "
                "which is why a pair receives 2x the naive force)."
            ),
            "clash_threshold_A": 1.2,
        },
        "per_cloud": per_cloud,
    }


# --------------------------------------------------------------------------
# chemistry metrics on the decoded SMILES
# --------------------------------------------------------------------------
def chemistry_metrics(
    smiles: Sequence[str], chem_rows: Sequence[Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Corpus-relative chemistry metrics over the decoded molecules.

    Definitions (stated here because the repo has more than one convention
    and the difference matters):

      * ``connected_rate`` — fraction of decoded molecules with exactly ONE
        fragment.  NOTE: our corpus writes the metal as its own component by
        convention, so the corpus reference is 0.0007, NOT 1.0
        (``metrics/by_metric/c7_fullgen_pool_quality.json`` ->
        ``reference.connected_rate``).  Read this as "fully connected
        single-component", not "one molecule".
      * ``mean_n_frags`` — mean component count from the DECODE (the 3D
        cloud's own components).  This one counts the explicit hydrogens the
        cloud carries, so it is NOT comparable with the corpus reference
        (3.34), which counts SMILES components.
      * ``mean_n_frags_heavy`` — mean component count from the decoded
        SMILES (hydrogens are implicit in SMILES, so these are heavy-atom
        components).  THIS is the corpus-comparable number.
      * ``singletons_per_mol`` — mean count of components holding exactly
        ONE heavy atom.  The reference corpus has
        ``second_frag_is_single_atom`` = 0.0146.
      * ``heavy_atom_retention_pool`` — pool-level heavy atoms kept by
        largest-fragment selection / pool-level heavy atoms.  This is the
        mass-deletion number the frontier report cares about: the c7 pool
        loses 22.4732 % of its heavy atoms to largest-fragment selection.
      * ``non_singleton_heavy_frac`` — heavy atoms in components holding
        >= 2 heavy atoms / all heavy atoms.  This is the NO-MASS-DELETION
        connectivity view: it counts atoms that are actually in a
        multi-atom fragment.

    SURVIVORSHIP (read this before quoting any mean): every mean here is
    over the molecules that DECODED, i.e. over ``n_chem_rows``.  A cloud
    whose decode raised is a ``None`` row and drops out of the denominator.
    If one arm fails more decodes than another, its means are computed on a
    DIFFERENT (surviving) population — so the decode failure rate is
    reported alongside and must be read together with these means.
    """
    n = len(smiles)
    if n == 0:
        return {"n": 0}
    n_frags: List[int] = []
    n_frags_heavy: List[int] = []
    singletons: List[int] = []
    second_sizes: List[int] = []
    kept_heavy = 0
    total_heavy = 0
    nonsingleton_heavy = 0
    ring_hist: Dict[str, int] = {}
    n_rings = 0
    for row in chem_rows:
        if not row:
            continue
        nf = int(row.get("n_fragments", 0) or 0)
        n_frags.append(nf)
        sizes = row.get("fragment_sizes")
        if sizes:
            n_frags_heavy.append(len(sizes))
            singles = sum(1 for s in sizes if int(s) == 1)
            singletons.append(singles)
            ordered = sorted((int(s) for s in sizes), reverse=True)
            second_sizes.append(ordered[1] if len(ordered) > 1 else 0)
            total_heavy += sum(int(s) for s in sizes)
            kept_heavy += ordered[0]
            nonsingleton_heavy += sum(int(s) for s in sizes if int(s) >= 2)
        else:
            singletons.append(0)
            second_sizes.append(0)
        for rs in row.get("ring_sizes") or []:
            ring_hist[str(int(rs))] = ring_hist.get(str(int(rs)), 0) + 1
            n_rings += 1
    n_rows = len(n_frags)
    if n_rows == 0:
        return {"n": n, "n_chem_rows": 0}
    connected = sum(1 for v in n_frags if v == 1)
    two_to_six = sum(1 for v in n_frags if 2 <= v <= 6)
    ge21 = sum(1 for v in n_frags if v >= 21)
    three = ring_hist.get("3", 0)
    six = ring_hist.get("6", 0)
    n_heavy_rows = len(n_frags_heavy)
    return {
        "n": n,
        "n_chem_rows": n_rows,
        "connected_rate": connected / n_rows,
        "mean_n_frags": sum(n_frags) / n_rows,
        "mean_n_frags_heavy": (
            sum(n_frags_heavy) / n_heavy_rows if n_heavy_rows else None
        ),
        "connected_rate_heavy": (
            sum(1 for v in n_frags_heavy if v == 1) / n_heavy_rows
            if n_heavy_rows
            else None
        ),
        "singletons_per_mol": sum(singletons) / n_rows,
        "mean_second_frag_size": (
            sum(second_sizes) / n_rows if second_sizes else None
        ),
        "share_n_frags_2_to_6": two_to_six / n_rows,
        "share_n_frags_ge21": ge21 / n_rows,
        "heavy_atom_retention_pool": (
            kept_heavy / total_heavy if total_heavy else None
        ),
        "non_singleton_heavy_frac": (
            nonsingleton_heavy / total_heavy if total_heavy else None
        ),
        "n_rings": n_rings,
        "ring_size_distribution": dict(
            sorted(ring_hist.items(), key=lambda kv: int(kv[0]))
        ),
        "three_ring_share": three / n_rings if n_rings else None,
        "six_ring_share": six / n_rings if n_rings else None,
    }


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------
def run_arm(
    model: Any,
    decoder: Any,
    templates: Sequence[Any],
    device: torch.device,
    *,
    name: str,
    weight: float,
    upper_bound: bool,
    reach: float,
    strength: Optional[float],
    n_steps: int,
    seed: int,
    method: str,
    noise_sigma: float,
) -> Dict[str, Any]:
    """Run ONE arm: sample -> decode -> classify.  Returns the arm record."""
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        decode_clouds_and_classify,
        generate_clouds_from_noise,
    )

    t_sample0 = time.perf_counter()
    clouds = generate_clouds_from_noise(
        model,
        templates,
        device,
        method=method,
        n_steps=int(n_steps),
        seed=int(seed),
        noise_sigma=float(noise_sigma),
        bond_length_guidance_weight=float(weight),
        bond_length_guidance_upper_bound=bool(upper_bound),
        bond_length_guidance_upper_bound_reach=float(reach),
        bond_length_guidance_upper_bound_strength=strength,
    )
    t_sample = time.perf_counter() - t_sample0

    # ---- pass 1: an INDEX-ALIGNED SMILES vector ------------------------
    # ``decode_clouds_and_classify`` appends to ``smiles_out`` only for
    # SUCCESSFUL decodes, so that list is NOT aligned with the clouds.
    # ``chem_rows`` IS aligned (one slot per cloud, None on failure), so we
    # collect our own per-cloud SMILES here to compute exact fragment
    # sizes, and merge them into the chem rows by index.
    from rdkit import Chem

    cloud_smiles: List[str] = []
    for cloud in clouds:
        try:
            res = decoder.decode(cloud)
            cloud_smiles.append(
                str(getattr(getattr(res, "decoded", None), "smiles", None) or "")
            )
        except Exception:  # noqa: BLE001
            cloud_smiles.append("")

    smiles: List[str] = []
    rejected: List[Tuple[str, str]] = []
    per_cloud: List[bool] = []
    mol_valid: List[bool] = []
    chem_rows: List[Optional[Dict[str, Any]]] = []
    reason_counts: Dict[str, int] = {}
    flags: Dict[str, int] = {}
    t_dec0 = time.perf_counter()
    decode_clouds_and_classify(
        decoder,
        clouds,
        smiles,
        rejected,
        per_cloud,
        chem_rows,
        None,  # sa_target
        reason_counts,
        decode_flags_out=flags,
        molecule_valid_out=mol_valid,
    )
    t_decode = time.perf_counter() - t_dec0

    # Fragment sizes per molecule (the repo's chem row carries only counts),
    # recomputed from the index-aligned SMILES so the retention numbers are
    # exact.
    for i, row in enumerate(chem_rows):
        if not row or i >= len(cloud_smiles):
            continue
        smi = cloud_smiles[i]
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:  # noqa: BLE001
            mol = None
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        row["fragment_sizes"] = [len(f) for f in frags]
        row["smiles"] = smi

    geom = geometry_band_stats(clouds)
    chem = chemistry_metrics(smiles, chem_rows)
    n_ok = int(reason_counts.get("ok", 0)) + sum(
        int(v) for k, v in reason_counts.items() if k.startswith("ok_repaired:")
    )
    n_raised = int(reason_counts.get("decode_raised", 0))
    n_clouds = len(clouds)
    n_rejected = sum(
        int(v)
        for k, v in reason_counts.items()
        if k not in ("ok", "decode_raised", "decode_error_example")
        and not k.startswith("ok_repaired:")
    )
    return {
        "arm": name,
        "config": {
            "bond_length_guidance_weight": float(weight),
            "bond_length_guidance_upper_bound": bool(upper_bound),
            "bond_length_guidance_upper_bound_reach": float(reach),
            "bond_length_guidance_upper_bound_strength": strength,
            "n_steps": int(n_steps),
            "seed": int(seed),
            "method": method,
            "noise_sigma": float(noise_sigma),
        },
        "n_clouds": n_clouds,
        "n_decode_ok": n_ok,
        "n_decode_raised": n_raised,
        "n_decode_rejected_rdkit": n_rejected,
        "decode_success_rate": (n_ok / n_clouds) if n_clouds else None,
        "sample_wall_s": t_sample,
        "decode_wall_s": t_decode,
        "total_wall_s": t_sample + t_decode,
        "rdkit_parse_rate": (sum(1 for v in per_cloud if v) / len(per_cloud))
        if per_cloud
        else None,
        "molecule_validity": (sum(1 for v in mol_valid if v) / len(mol_valid))
        if mol_valid
        else None,
        "n_ok_rdkit_strict": n_ok,
        "failure_reasons": dict(reason_counts),
        "decode_flags": dict(flags),
        "chemistry": chem,
        "geometry": geom,
        "_smiles": list(cloud_smiles),
        "_chem_rows": chem_rows,
    }


def _smi_for(row: Dict[str, Any], smiles: Sequence[str]) -> str:
    """Unused helper kept out of the hot path.

    The chem rows are index-aligned with the CLOUDS, not with ``smiles``
    (``smiles`` only receives SUCCESSFUL decodes), which is why
    :func:`run_arm` keeps its own index-aligned SMILES vector.
    """
    return ""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wfa142_upper_bound_probe.py",
        description=(
            "Wf-A142: MEASURE the Wf-A128 one-sided bond-length upper bound "
            "on a CPU-only decode (no GPU, no retrain)."
        ),
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT
        / "results/wfa108j_budget36/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt",
        help="CFM checkpoint to decode from (CPU-loadable).",
    )
    p.add_argument("--n-clouds", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", type=str, default="euler")
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        help=(
            "x0 scale.  DEFAULT: the checkpoint's own `fm_noise_sigma` "
            "(the trainer's gen-eval path uses the same value — "
            "train_cfm_yuelbond_wasserstein.py :18874).  Sampling with a "
            "different sigma than training is a scale mismatch the repo "
            "measures (fm_prior_scale_check)."
        ),
    )
    p.add_argument(
        "--guidance-weight",
        type=float,
        default=DEFAULT_GUIDANCE_WEIGHT,
        help="Wf-A75 spring weight (must be > 0 for the bound to exist).",
    )
    p.add_argument("--reach", type=float, default=DEFAULT_REACH_A)
    p.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help="one-sided penalty stiffness (the Wf-A128 config's 25.0).",
    )
    p.add_argument(
        "--arms",
        type=str,
        default="prod,spring,bound",
        help="comma-separated arm names to run.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "molmetal/reports/wfa142_upper_bound/probe.json",
    )
    p.add_argument(
        "--smiles-csv",
        type=Path,
        default=None,
        help="optional per-arm decoded-SMILES CSV (defaults to <out>.csv)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if torch.cuda.is_available():
        raise SystemExit(
            "ABORT: a CUDA device is visible.  This probe is CPU-only "
            "(Wf-A142 hard constraint); run it with CUDA_VISIBLE_DEVICES=\"\"."
        )
    device = torch.device("cpu")

    ckpt_path = args.ckpt
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    t_load0 = time.perf_counter()
    ckpt = load_checkpoint_any(ckpt_path)
    model, ckpt_args = build_model(ckpt)
    decoder = build_decoder(model, ckpt_args)
    templates = load_templates(ckpt_args, args.n_clouds)
    t_load = time.perf_counter() - t_load0

    arm_defs = {
        "prod": dict(weight=0.0, upper_bound=False),
        "spring": dict(weight=args.guidance_weight, upper_bound=False),
        "bound": dict(weight=args.guidance_weight, upper_bound=True),
    }
    wanted = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    for a in wanted:
        if a not in arm_defs:
            raise SystemExit(f"unknown arm {a!r}; known: {sorted(arm_defs)}")

    arms: Dict[str, Any] = {}
    noise_sigma = (
        float(args.noise_sigma)
        if args.noise_sigma is not None
        else float(getattr(ckpt_args, "fm_noise_sigma", 1.0) or 1.0)
    )
    print(
        f"[wfa142] noise_sigma={noise_sigma} "
        f"(from checkpoint's fm_noise_sigma={getattr(ckpt_args, 'fm_noise_sigma', None)})",
        flush=True,
    )
    for a in wanted:
        d = arm_defs[a]
        print(f"[wfa142] arm={a} {d}", flush=True)
        arms[a] = run_arm(
            model,
            decoder,
            templates,
            device,
            name=a,
            weight=d["weight"],
            upper_bound=d["upper_bound"],
            reach=args.reach,
            strength=args.strength,
            n_steps=args.n_steps,
            seed=args.seed,
            method=args.method,
            noise_sigma=noise_sigma,
        )
        print(
            f"[wfa142] arm={a} wall={arms[a]['total_wall_s']:.2f}s "
            f"connected={arms[a]['chemistry'].get('connected_rate')} "
            f"mean_n_frags={arms[a]['chemistry'].get('mean_n_frags')}",
            flush=True,
        )

    # ---- deltas: the bound's MARGINAL effect (bound - spring) -----------
    deltas: Dict[str, Any] = {}
    if "bound" in arms and "spring" in arms:
        deltas["bound_minus_spring"] = _delta(arms["bound"], arms["spring"])
    if "bound" in arms and "prod" in arms:
        deltas["bound_minus_prod"] = _delta(arms["bound"], arms["prod"])

    payload = {
        "artifact": "wfa142_upper_bound",
        "wave": "Wf-A142",
        "date": "2026-10-31",
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "checkpoint": {
            "path": str(ckpt_path),
            "load_method": (
                "torch.load(map_location='cpu', weights_only=False); on "
                "'Invalid magic number' fall back to raw pickle.load — the "
                "trainer writes raw pickles via wf386a3_atomic_write "
                "(Wf-A81)."
            ),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "args_snapshot": {
                k: ckpt_args.__dict__.get(k)
                for k in (
                    "config",
                    "epochs",
                    "n_train",
                    "n_val",
                    "hidden_dim",
                    "n_layers",
                    "edge_cutoff",
                    "yuelbond_vocab",
                    "bond_z_table",
                    "bond_z_metal_set",
                    "atom_vocab_size",
                    "data_csv",
                    "device",
                    "seed",
                    "method",
                    "n_steps",
                )
            },
        },
        "templates": {
            "n": len(templates),
            "source": getattr(ckpt_args, "data_csv", None),
            "split": str(getattr(ckpt_args, "val_split", "val")),
            "note": (
                "Only the COMPOSITION is used by the sampler; the template "
                "geometry is not. All arms share these templates."
            ),
        },
        "load_wall_s": t_load,
        "arms": arms,
        "deltas": deltas,
        "corpus_reference": _corpus_reference(),
        "honest_framing": {
            "measured": [
                "The three arms are paired at the NOISE level: same "
                "checkpoint, same templates, same seed.",
                "Every number in 'arms' and 'deltas' was produced by this "
                "run and is reproducible with the CLI invocation in the "
                "docstring.",
            ],
            "not_measured": [
                "Any production-scale fragmentation lift. This is a "
                "small-N CPU probe on ONE smoke checkpoint.",
                "Any GPU / multi-epoch measurement.",
                "The head's own logits inside the ODE loop — the head runs "
                "at decode time only (adapter __init__.py :1245-1248), so a "
                "confidence-scaled guidance term is NOT measurable today.",
            ],
        },
    }

    out_path = args.out
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    _atomic_write(out_path, text)

    csv_path = args.smiles_csv or out_path.with_suffix(".csv")
    _write_smiles_csv(csv_path, arms)

    print(f"[wfa142] wrote {out_path}", flush=True)
    print(f"[wfa142] wrote {csv_path}", flush=True)
    return 0


def _delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """The paired difference between two arms (a - b)."""
    out: Dict[str, Any] = {}
    for section, keys in (
        ("chemistry", ("connected_rate", "connected_rate_heavy",
                       "mean_n_frags", "mean_n_frags_heavy",
                       "singletons_per_mol",
                       "heavy_atom_retention_pool", "non_singleton_heavy_frac",
                       "three_ring_share", "six_ring_share",
                       "share_n_frags_2_to_6")),
    ):
        for k in keys:
            va = (a.get(section) or {}).get(k)
            vb = (b.get(section) or {}).get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                out[f"{section}.{k}"] = va - vb
    ga = a.get("geometry") or {}
    gb = b.get("geometry") or {}
    for k in ("frac_atoms_nn_in_band", "mean_pairs_in_band_per_cloud",
              "frac_pairs_in_band", "mean_nn_distance",
              "frac_atoms_nn_below_1p2"):
        va, vb = ga.get(k), gb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[f"geometry.{k}"] = va - vb
    out["decode_success_rate"] = (
        (a.get("decode_success_rate") or 0.0) - (b.get("decode_success_rate") or 0.0)
    )
    out["total_wall_s"] = a.get("total_wall_s", 0.0) - b.get("total_wall_s", 0.0)
    return out


def _corpus_reference() -> Dict[str, Any]:
    """The corpus-relative reference numbers, read from the repo metric file."""
    path = REPO_ROOT / "metrics/by_metric/c7_fullgen_pool_quality.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"source": str(path), "error": repr(exc)}
    ref = d.get("reference") or {}
    ov = d.get("overall") or {}
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "reference_corpus": {
            "n": ref.get("n"),
            "mean_n_frags": ref.get("mean_n_frags"),
            "connected_rate": ref.get("connected_rate"),
            "second_frag_is_single_atom": ref.get("second_frag_is_single_atom"),
            "mean_heavy_atoms": ref.get("mean_heavy_atoms"),
            "three_ring_share": ref.get("three_ring_share"),
            "six_ring_share": ref.get("six_ring_share"),
            "ring_size_distribution": ref.get("ring_size_distribution"),
        },
        "c7_generated_pool": {
            "n_rows": ov.get("n_rows"),
            "mean_n_frags": ov.get("mean_n_frags"),
            "connected_rate": ov.get("connected_rate"),
            "mean_largest_frag_frac": ov.get("mean_largest_frag_frac"),
            "mean_heavy_atoms": ov.get("mean_heavy_atoms"),
            "three_ring_share": ov.get("three_ring_share"),
            "six_ring_share": ov.get("six_ring_share"),
        },
    }


def _atomic_write(path: Path, text: str) -> None:
    """Write through the repo's atomic-write protocol (AGENTS.md §5.6)."""
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        detect_truncation,
        safe_overwrite_check,
    )

    reason = detect_truncation(text, expected_min_length=200)
    if reason:
        raise SystemExit(f"refusing to write truncated output: {reason}")
    safe_overwrite_check(path, text, raise_on_risk=False)
    atomic_write_file(path, text)


def _write_smiles_csv(path: Path, arms: Dict[str, Any]) -> None:
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    lines = ["arm,index,smiles"]
    for name, rec in arms.items():
        for i, smi in enumerate(rec.get("_smiles") or []):
            lines.append(f"{name},{i},{smi}")
    atomic_write_file(path, "\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
