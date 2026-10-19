"""BLK-C-L3 / BLK-C-L5 -- can SAMPLING-TIME guidance help, on the existing checkpoint?

NO RETRAINING.  One forward ODE integration per arm, on an existing checkpoint.

QUESTION.  cfm_todo.json::BLK-C-L3 recommends the guidance family (EnFlow
arXiv:2512.22597 injects grad-E during ODE integration; PCFM ICLR 2026 applies
hard constraints at inference).  BLK-C-L5 asks why 18.8% of generated molecules
cannot be embedded in 3D by RDKit ETKDG.

Both are coordinate-space interventions, so both need the SAME premise to be
true before they are worth building:

    the model's SAMPLED COORDINATES are the defective object.

If the coordinates are already chemically sane and the failure lives one stage
downstream (in the bond head that turns coordinates into a graph, or in RDKit's
own metal-unaware embedder), then a coordinate-space force -- which is what
every cited method injects -- cannot fix it, however well it is tuned.

WHAT THIS DOES.  Two measurements, both on existing checkpoints, no training:

  1. COORDINATE HEALTH.  Integrate the ODE, then compare the sampled geometry
     against real tmQM on metrics that do not need RDKit: nearest-neighbour
     distance distribution, metal-ligand coordination distance, radius of
     gyration, and the fraction of atom pairs closer than 0.5 * (rcov_i+rcov_j)
     (a steric clash).  This is the premise test.

  2. GUIDANCE DELTA.  Inject the cheapest possible coordinate guidance -- a
     repulsive core force at the steric-clash scale, the gradient of a smooth
     inverse-power potential -- into the velocity field during integration, and
     measure what moves.  Guided vs unguided, SAME seed, SAME clouds, so the
     only variable is the guidance term.

The guidance potential here is deliberately the SIMPLEST one in the family (a
repulsion core). It is not EnFlow's learned energy model. If even a repulsion
core cannot move the decode outcome, a learned energy model -- which is strictly
more expensive and strictly more assumptions -- is not the next step.

WHY A REPULSION CORE IS THE RIGHT MINIMUM VIABLE PROBE.  Every method in the
cited family ultimately pushes atoms apart when they are too close. That is the
one force component all of them share. If that shared component produces no
measurable change, the family has no leverage here regardless of what else it
adds.

DEFAULT OFF. This script writes a JSON file and touches no shipped code path.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]

# Covalent radii (Angstrom), Cordero et al. 2008, for the elements tmQM contains
# plus the metals. Used ONLY as a steric reference scale, not as chemistry.
RCOV: Dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 33: 1.21, 35: 1.20, 53: 1.39, 78: 1.36, 44: 1.25,
    77: 1.35, 45: 1.35, 76: 1.35, 75: 1.35,
}
METAL_Z = {78, 44, 77, 45, 76, 75}


def _sys_path() -> None:
    for p in (str(REPO), str(REPO / "molmetal" / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_model(ckpt: Path, device: torch.device):
    _sys_path()
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        CFMJointModel,
        load_cfm_checkpoint,
        migrate_legacy_vel_state_dict,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    # Wf-A96 — the trainer has written raw pickles through atomic_write_file
    # since Wf-A30-1, and torch.load raises "Invalid magic number" on those.
    # load_cfm_checkpoint is the loader that handles both serialisations
    # (the same fix Wf-A90 applied to --resume-from).  Calling torch.load
    # here meant this probe could only ever read pre-Wf-A30-1 checkpoints.
    state = load_cfm_checkpoint(str(ckpt), map_location="cpu")
    a = state["args"]
    _sd = state["model_state_dict"]
    # Wf-A96 — two opt-in axes were added AFTER this checkpoint was written
    # and its args dict does not record them, but the parameters are in the
    # state_dict.  Infer them from the tensors so the model is built to the
    # shape the checkpoint actually has, rather than to a default that
    # cannot hold it.
    if not a.get("t_embedding_dim") and "vel_field.time_embed_proj.weight" in _sd:
        a["t_embedding_dim"] = int(_sd["vel_field.time_embed_proj.weight"].shape[1])
    if not a.get("pocket_inject_per_layer") and any(
        k.startswith("vel_field.pocket_per_layer_injector.") for k in _sd
    ):
        a["pocket_inject_per_layer"] = True
    # Wf-A96 (2026-10-19) — construct the SAME model the checkpoint was
    # trained as.  The loader used to pass only the Wf-A70-era kwargs, so any
    # checkpoint carrying an opt-in module (t_embedding_dim>0 ->
    # vel_field.time_embed_proj.*, pocket_inject_per_layer=True ->
    # vel_field.pocket_per_layer_injector.*) failed with a state_dict
    # mismatch and the probe could not run at all.  MEASURED on
    # cfm_fullcorpus_2026_10_10: unexpected keys were exactly those two
    # modules.  Every opt-in axis is read from the checkpoint's own args, so
    # a future axis added there does not silently break this script again.
    model = CFMJointModel(
        hidden_dim=int(a["hidden_dim"]),
        n_layers=int(a["n_layers"]),
        cutoff=float(a["edge_cutoff"]),
        fm_objective=bool(a["use_fm_objective"]),
        z_table=get_z_table(str(a.get("yuelbond_vocab", "pt"))),
        use_egnn_velocity=bool(a["use_egnn_velocity"]),
        max_atomic_number=int(a["egnn_max_atomic_number"]),
        context_dropout=float(a["egnn_context_dropout"]),
        tmqm_init_path=None,
        use_tmqm_init=False,
        q_head_enabled=bool(a.get("q_head_enabled", False)),
        t_embedding_dim=int(a.get("t_embedding_dim", 0) or 0),
        pocket_per_layer=bool(a.get("pocket_per_layer", False)),
        pocket_inject_per_layer=bool(a.get("pocket_inject_per_layer", False)),
        bond_predictor=str(a.get("bond_predictor", "yuelbond")),
        n_edge_layers=int(a.get("n_edge_layers", 4) or 4),
        bond_aux_temperature=float(a.get("bond_aux_temperature", 1.0) or 1.0),
        use_bond_head_skip=bool(a.get("use_bond_head_skip", False)),
        use_metal_pair_features=bool(a.get("use_metal_pair_features", False)),
        bond_z_metal_set=str(a.get("bond_z_metal_set", "all7")),
        bond_z_table=str(a.get("bond_z_table", "pt")),
        bond_head_kind=str(a.get("bond_head_kind", "single")),
        t_eval_grid=tuple(float(x) for x in (a.get("t_eval_grid") or (1.0,))),
        clash_penalty_weight=float(a.get("clash_penalty_weight", 0.0) or 0.0),
        bond_pattern_mask_in_train=bool(a.get("bond_pattern_mask_in_train", False)),
        use_pocket_encoder=bool(a.get("use_pocket_encoder", True)),
    )
    _sd, _mig = migrate_legacy_vel_state_dict(model, state["model_state_dict"])
    if _mig:
        state["_legacy_key_migration"] = _mig
    missing, unexpected = model.load_state_dict(_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: {missing[:5]} / {unexpected[:5]}")
    model.to(device).eval()
    return model, a, state


def load_clouds(a: dict, n_clouds: int):
    _sys_path()
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    root = str(a["tmqm_dir"])
    if not Path(root).is_dir():
        root = str(REPO / root)
    return load_tmqm_clouds(
        root,
        metals=[s.strip() for s in str(a["tmqm_metals"]).split(",") if s.strip()],
        coord_number=None,
        n_clouds=n_clouds,
        seed=int(a["seed"]),
        split="val",
        val_frac=float(a.get("tmqm_val_frac", 0.1)),
    )


# --------------------------------------------------------------------- geometry


def coord_stats(pos: torch.Tensor, zs: List[int]) -> Dict[str, Any]:
    """Steric / geometric health of ONE cloud, from coordinates alone."""
    n = int(pos.shape[0])
    if n < 2:
        return {"n_atoms": n}
    d = torch.cdist(pos.double(), pos.double())
    iu = torch.triu_indices(n, n, offset=1)
    pair_d = d[iu[0], iu[1]]
    zt = torch.tensor(zs)
    rc = torch.tensor([RCOV.get(int(z), 1.3) for z in zt], dtype=torch.float64)
    pair_rc = rc[iu[0]] + rc[iu[1]]

    out: Dict[str, Any] = {
        "n_atoms": n,
        "min_pair_distance": float(pair_d.min()),
        "p01_pair_distance": float(pair_d.quantile(0.01)),
        "median_pair_distance": float(pair_d.median()),
        "n_pairs": int(pair_d.numel()),
        # A pair closer than 0.7x the sum of covalent radii cannot be a bond and
        # is not a plausible non-bonded contact in tmQM.
        "frac_pairs_hard_clash": float((pair_d < 0.7 * pair_rc).double().mean()),
        "n_hard_clash_pairs": int((pair_d < 0.7 * pair_rc).sum()),
        # Substrate-compression signal (cfm_todo.json::next_actions[2], 2026-10-12
        # L3 decode-enabled A/B): the aromatic Wiberg BO 1.3-1.7 band corresponds
        # to C-C / C-N distances ~1.30-1.70 A. Generated clouds that COMPRESS
        # into this band are the ones the bond head has any chance of decoding
        # as aromatic; clouds whose 1.30-1.70 A band is empty are decoded as
        # saturated regardless of the head's class ranking. Round-6 evidence:
        # 1.06% on generated vs 4.01% on real tmQM.
        "frac_pairs_in_arom_band": float(
            ((pair_d >= 1.30) & (pair_d < 1.70)).double().mean()
        ),
        "n_pairs_in_arom_band": int(((pair_d >= 1.30) & (pair_d < 1.70)).sum()),
    }
    c = pos.double() - pos.double().mean(dim=0, keepdim=True)
    out["radius_of_gyration"] = float(torch.sqrt((c.pow(2).sum(dim=1)).mean()))

    # Metal-ligand coordination: distance from each metal to its nearest atoms.
    mind = []
    for k, z in enumerate(zs):
        if z not in METAL_Z:
            continue
        row = d[k].clone()
        row[k] = float("inf")
        for j in range(4):
            if int((row < float("inf")).sum()) == 0:
                break
            mind.append(float(row.min()))
            row[int(row.argmin())] = float("inf")
    if mind:
        out["metal_ligand_n"] = len(mind)
        out["metal_ligand_min"] = min(mind)
        out["metal_ligand_median"] = statistics.median(mind)
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "min_pair_distance", "p01_pair_distance", "median_pair_distance",
        "frac_pairs_hard_clash", "frac_pairs_in_arom_band",
        "radius_of_gyration", "metal_ligand_min",
        "metal_ligand_median",
    ]
    out: Dict[str, Any] = {"n_clouds": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            out[k] = {
                "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    out["n_clouds_with_any_hard_clash"] = sum(
        1 for r in rows if r.get("n_hard_clash_pairs", 0) > 0
    )
    out["total_hard_clash_pairs"] = sum(r.get("n_hard_clash_pairs", 0) for r in rows)
    # Aromatic-band coverage across the whole generation, complementary to the
    # per-cloud `frac_pairs_in_arom_band` mean above. Round-6 evidence: gen
    # 1.06% vs real 4.01% (cfm_todo.json::next_actions[2] priority 3, 2026-10-12).
    out["n_clouds_with_any_arom_band_pair"] = sum(
        1 for r in rows if r.get("n_pairs_in_arom_band", 0) > 0
    )
    out["total_pairs_in_arom_band"] = sum(
        r.get("n_pairs_in_arom_band", 0) for r in rows
    )
    return out


# --------------------------------------------------------------------- guidance


def guidance_force(
    x: torch.Tensor, z: torch.Tensor, rcov: torch.Tensor, strength: float
) -> torch.Tensor:
    """Gradient of a smooth repulsive core, in torch, differentiable-free.

    U(x) = strength * sum_{i<j} (d0_ij / d_ij)^3  with d0 = 0.85*(rcov_i+rcov_j)

    -dU/dx_i is the force. Computed in closed form so no autograd is needed:
    it keeps the ODESolver's internal no_grad context intact and costs one
    cdist per step, which is the same cost the radius graph already pays.

    This is the ONE component every method in the guidance family shares
    (EnFlow arXiv:2512.22597, PCFM ICLR 2026, FlowBack-Adjoint
    arXiv:2508.03619 all push overlapping atoms apart).  It is the cheapest
    possible probe of whether the family has leverage here at all.
    """
    b, n, _ = x.shape
    d = torch.cdist(x, x)  # (B, N, N)
    eye = torch.eye(n, dtype=torch.bool, device=x.device)
    d = d.masked_fill(eye.unsqueeze(0), 1e9)  # diagonal excluded
    # rcov is (B, N); unsqueeze(2) and unsqueeze(1) broadcast to (B, N, N).
    d0 = 0.85 * (rcov.unsqueeze(2) + rcov.unsqueeze(1))
    d0 = d0.masked_fill(eye, 1.0)  # (B, N, N)

    # U = strength * sum_{i<j} (d0_ij / d_ij)^3
    # Each unordered pair is counted ONCE, so
    #   dU/dx_i = sum_{j != i} (dU/dd_ij) * (x_i - x_j) / d_ij
    # and the FORCE (the term added to the ODE velocity) is -dU/dx_i.
    #
    # SIGN, pinned by brute force on a 2-atom system at x=[0.0, 1.2],
    # rcov=[1.5, 1.5], strength=+0.05:
    #   U = +s*(d0/d)^3 is a repulsion (U falls as the pair separates);
    #   dU/dx[0] = +1.199463 by autograd, hence the force is -1.199463
    #   (atom 0 at x=0 is pushed toward -x, away from atom 1 at +x).
    # Returning +dU/dx here gives an ATTRACTION, which collapses the geometry
    # further and would read as "guidance does not work". Guarded by
    # molmetal/tests/test_wf_blkc_l3_guidance_probe.py.
    du_dd = -3.0 * strength * (d0 ** 3) / (d.clamp_min(1e-3) ** 4)
    du_dd = du_dd.masked_fill(eye.unsqueeze(0), 0.0).unsqueeze(-1)  # (B, N, N, 1)

    diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 3), diff[b,i,j] = x_i - x_j
    denom = d.clamp_min(1e-3).unsqueeze(-1)
    f = (du_dd / denom * diff).sum(dim=2)  # sum over j -> (B, N, 3)
    assert f.shape == x.shape, f"guidance force shape {tuple(f.shape)} != {tuple(x.shape)}"
    return -f


def integrate(
    model,
    templates,
    device: torch.device,
    *,
    method: str,
    n_steps: int,
    seed: int,
    noise_sigma: float,
    guidance_strength: float,
) -> List[Dict[str, Any]]:
    """Integrate the ODE, optionally with guidance injected into the velocity.

    Mirrors generate_clouds_from_noise()'s grouped path (the DEFAULT, pad_groups
    OFF) exactly, so the guided and unguided arms differ ONLY by the guidance
    term.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )
    from molmetal.adapters.flow_matching_lipman._reference.solver import ODESolver

    gen = torch.Generator(device=device).manual_seed(int(seed))
    # Wf-A98 (2026-10-19) — carry the INPUT INDEX through the size grouping.
    #
    # The ODE has to run in same-size batches, so this function walks
    # sorted(by_size) and used to emit in that order.  Every caller, however,
    # pairs the result with its template list by position:
    #
    #     for cloud, c in zip(clouds, integrate(model, clouds, ...))
    #
    # On a mixed-size cohort those two orders differ, so the template (bond
    # truth, original geometry) and the sampled geometry came from different
    # molecules.  MEASURED while writing wf_a98: template 0 had 16 atoms, the
    # paired output had 14, and the truth graph indexed atom 15 of a 14-atom
    # cloud.  That raised instead of lying, but only because the probe touched
    # the template; a probe that reads only the sampled entry (as wf_a97 did)
    # cannot notice.  Emitting in input order makes the positional contract
    # true.
    by_size: Dict[int, List[Tuple[int, Any]]] = {}
    for _i, tpl in enumerate(templates):
        by_size.setdefault(int(tpl.atomic_numbers.shape[0]), []).append((_i, tpl))

    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for n_atoms in sorted(by_size):
            group = by_size[n_atoms]
            b = len(group)
            z_b = torch.stack(
                [g.atomic_numbers.long().to(device) for _i, g in group], dim=0
            )
            mask = torch.ones(b, n_atoms, dtype=torch.bool, device=device)
            rcov = torch.tensor(
                [RCOV.get(int(z), 1.3) for z in range(0, 0)] or [], dtype=torch.float32
            )
            rc = torch.tensor(
                [[RCOV.get(int(z_b[i, j]), 1.3) for j in range(n_atoms)] for i in range(b)],
                dtype=torch.float32, device=device,
            )
            x0 = torch.randn(b, n_atoms, 3, device=device, generator=gen) * float(
                noise_sigma
            )

            def _vf(x, t, _z=z_b, _m=mask, _rc=rc):
                edge_index, edge_mask = _block_diag_radius_edges(x, _m, model.cutoff)
                t_b = (
                    t.reshape(1).expand(x.shape[0])
                    if t.dim() == 0
                    else t.reshape(-1)[:1].expand(x.shape[0])
                )
                v = model.vel_field(x, _z, edge_index, t_b, edge_mask=edge_mask)["vel"]
                if guidance_strength > 0.0:
                    v = v + guidance_force(x, _z, _rc, guidance_strength)
                return v

            solver = ODESolver(velocity_model=_vf, method=method, n_steps=n_steps)
            x1 = solver.sample(x0, method=method, n_steps=n_steps)
            for i in range(b):
                _src, _tpl = group[i]
                out.append(
                    {
                        "positions": x1[i].detach().cpu(),
                        "zs": _tpl.atomic_numbers.cpu().tolist(),
                        "template_index": int(_src),
                    }
                )
    # Input order, so zip(templates, out) is correct by construction.
    out.sort(key=lambda r: r["template_index"])
    return out


def _count_aromatic_atoms(smiles: str) -> Optional[int]:
    """Aromatic atom count of a decoded SMILES, or None if RDKit refuses it.

    Wf-A96 (2026-10-19).  This is a REPORTING helper only: it never feeds
    back into the decode, and a None means "the SMILES the decoder emitted
    does not round-trip through RDKit", which is itself information.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return int(sum(1 for at in mol.GetAtoms() if at.GetIsAromatic()))
    except Exception:  # noqa: BLE001
        return None


def _aromatic_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Share of surviving molecules that contain aromaticity.

    Reported twice on purpose.  Wf-A84's last-resort repair demotes EVERY
    aromatic bond to single, so a repaired survivor is aromatic-free BY
    CONSTRUCTION.  Mixing it into the headline share would make the repair
    look like a model that stopped predicting aromaticity.  The clean-only
    number answers "what does the head predict"; the all-survivors number
    answers "what does the pipeline emit".
    """
    def _share(subset: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        vals = [r.get("n_aromatic_atoms") for r in subset]
        vals = [v for v in vals if isinstance(v, int)]
        if not vals:
            return None
        return {
            "n": len(vals),
            "frac_any_aromatic": sum(1 for v in vals if v > 0) / len(vals),
            "mean_aromatic_atoms": statistics.fmean(vals),
        }

    clean = [r for r in rows if not r.get("repair")]
    return {
        "all_survivors": _share(rows),
        "clean_decodes_only": _share(clean),
    }


def _count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        v = r.get(key)
        k = str(v) if v else "<none>"
        out[k] = out.get(k, 0) + 1
    return out


def decode(model, clouds: List[Dict[str, Any]], args_dict: dict, state=None):
    """Run the shipped bond head + decode_chemistry on sampled clouds.

    2026-10-12 FIX. This function had never executed: it imported
    ``YuelBondBondDecoder`` from ``lam_chem.yuelbond_decoder``, where it does
    NOT exist (the class lives in ``lam_chem.cfm_decoder_yuelbond``), it
    passed a ``device=`` kwarg the class does not accept, and it called a
    ``to_smiles`` method that does not exist either -- the real entry point is
    ``decode(cloud)`` returning a ``YuelBondDecodeResult`` whose
    ``decoded.smiles`` is the SMILES. The first ImportError surfaced only when
    the sweep was run WITHOUT ``--skip-decode``, which is why the original
    24-cloud measurement passed: it skipped decode, so this code was never
    reached. Same class as the six other failures today -- the harness
    reported success because the broken path was not taken.

    The three APIs used below are the ones
    ``molmetal/scripts/wf_cfm_decode_dump_smiles.py`` uses, which is the
    canonical decode call site in this repo.
    """
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _aromatic_bias_vector,
    )

    dec = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=int(args_dict.get("hidden_dim", 128)),
        n_layers=int(args_dict.get("n_layers", 3)),
        decision=str(args_dict.get("bond_decision", "threshold")),
        decision_threshold=float(args_dict.get("bond_decision_threshold", 0.5)),
        class_logit_bias=_aromatic_bias_vector(args_dict),
        connectivity_prior=str(args_dict.get("connectivity_prior", "on")) == "on",
        connectivity_prior_min_logit=float(
            args_dict.get("connectivity_prior_min_logit", 0.0)
        ),
        # Wf-A96 (2026-10-19) — the probe was constructing a decoder that the
        # trainer's gen_eval would NOT construct: no dative cutoff from the
        # checkpoint's own yaml, and no feature-schema restore.  Both decide
        # which pairs the head calls dative, so the L3 decode rate measured
        # here could differ from the shipped decode for reasons that have
        # nothing to do with guidance.  Mirror the trainer construction.
        dative_max_distance=float(
            args_dict.get("dative_distance_cutoff", 2.6)
        ),
    )
    if state is not None:
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            restore_feature_schema,
        )
        restore_feature_schema(model, dec, state)
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import decode_chemistry

    rows: List[Dict[str, Any]] = []
    for c in clouds:
        try:
            ac = AtomCloud(
                positions=c["positions"],
                atomic_numbers=torch.tensor(c["zs"], dtype=torch.long),
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            res = dec.decode(ac)
            smi = getattr(res.decoded, "smiles", "") or ""
        except Exception as exc:  # noqa: BLE001
            rows.append({"error": f"{type(exc).__name__}: {exc}"})
            continue
        if not smi:
            # 2026-10-12. The REJECTION REASON was being dropped on the floor.
            # `DecodedMol.error` is set at three paths in cfm_decoder_yuelbond.py
            # (:727 RWMol.GetMol failed, :779 empty_or_failed, and the
            # SanitizeMol branch at :733 where it is NOT set and an empty
            # smiles is the only signal) -- and this helper read only .smiles
            # and .sanitized, never .error.
            #
            # That made the aromatic-offset sweep UNINTERPRETABLE. Decode rate
            # fell 22 -> 13 -> 3 -> 0 of 64 as the bias rose, but with no
            # reason recorded there is no way to tell the BOND HEAD rejecting
            # its own output from RDKit sanitisation refusing a valence. Those
            # are opposite diagnoses: the first is a head-ranking problem, the
            # second is a bond-count problem. Without this field the sweep
            # cannot choose between them.
            _err = str(getattr(res.decoded, "error", "") or "")
            rows.append(
                {
                    "smiles": None,
                    "sanitized": False,
                    "decode_error": _err or None,
                    # Distinguishes "the decoder produced nothing and said why"
                    # from "it produced nothing silently", which is the case
                    # the SanitizeMol branch lands in.
                    "error_stage": (
                        "decoder" if _err else "unknown_no_error_field"
                    ),
                }
            )
            continue
        info = decode_chemistry(smi) or {}
        rows.append(
            {
                "smiles": smi,
                "sanitized": bool(getattr(res.decoded, "sanitized", False)),
                "decode_error": str(getattr(res.decoded, "error", "") or "") or None,
                # Wf-A96 — Wf-A84's last-resort repair returns a VALID but
                # DE-AROMATISED molecule.  Counting it as a clean decode
                # would report the aromatic fraction of a set that was
                # partly rebuilt without aromaticity, which is precisely the
                # number this re-measure exists to report.
                "repair": str(getattr(res.decoded, "repair", "") or "") or None,
                "bonds_accepted": getattr(res, "bonds_accepted", None),
                "bonds_rejected_overvalent": getattr(
                    res, "bonds_rejected_overvalent", None
                ),
                "n_fragments": info.get("n_fragments"),
                "largest_fragment_frac": info.get("largest_fragment_frac"),
                "has_metal": info.get("has_metal"),
                "n_aromatic_atoms": _count_aromatic_atoms(smi),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    """Build the parser and parse. Every knob defaults to None so that
    ``_merge_config`` can tell 'not passed' from 'passed the default value' --
    the same conflation that made apply_yaml_overrides silently override an
    explicit choice (AGENTS.md 5.11)."""
    ap = argparse.ArgumentParser(
        description="Sampling-time guidance probe (no training).",
        epilog=(
            "AGENTS.md 5.11: hyperparameters come from YAML, not the command "
            "line. --config supplies every knob except which run to do. A CLI "
            "flag beats the config, so a sweep cannot silently disagree with "
            "the file it claims to come from."
        ),
    )
    ap.add_argument("--ckpt", required=False, default=None)
    ap.add_argument("--out", required=False, default=None)
    ap.add_argument("--config", type=Path, default=None,
                    help="YAML supplying the defaults below")
    ap.add_argument("--n-clouds", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--guidance-strength", type=float, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument(
        "--bond-aromatic-logit-bias",
        dest="bond_aromatic_logit_bias",
        type=float,
        default=None,
        help=(
            "2026-10-12 (BLK-C-M4): additive offset on the BOND_AROMATIC "
            "logit. 0.0 is a no-op. Operating point only -- it cannot teach a "
            "head that never ranks aromatic first."
        ),
    )
    ap.add_argument("--skip-decode", action="store_true", default=None)
    ap.add_argument("--compare-real", action="store_true", default=None)
    # argparse needs a real default for the merge below; these are the values
    # used when no --config is given.
    ns = ap.parse_args()
    return ns


def _merge_config(ns):
    """Apply ``--config`` YAML under every flag the caller did not pass.

    Precedence is explicit CLI > YAML > argparse default, matching
    wfA2_config_loader.py (AGENTS.md 5.11). Keys are read BY THE argparse
    DEST NAME, so a typo in the YAML is an unknown key and is reported
    rather than silently ignored -- the silent-ignore class that
    apply_yaml_overrides was rewritten to eliminate.

    ``skip_decode`` and ``compare_real`` are store_true, so a YAML ``true``
    is honoured and an absent key leaves the flag off. They are read as
    bools, not as strings, because a store_true flag handed the string
    "false" is truthy.
    """

    # ckpt and out are here so that entry.command_template can be reduced to
    # `--config {yaml_path}` alone, which is the only shape AGENTS.md 5.11
    # permits in a command_template ("may use {placeholder} values... a
    # literal value or bare switch there is a violation"). The binding gate
    # fails any template carrying a trainer flag such as --guidance-strength,
    # so the strength grid and the checkpoint path both belong in the YAML.
    _DEFAULTS = {
        "ckpt": None, "out": None,
        "n_clouds": 48, "n_steps": 50, "method": "midpoint", "seed": 1234,
        "guidance_strength": 0.0, "arm": "probe",
        "skip_decode": False, "compare_real": False,
        # 2026-10-12 (BLK-C-M4) — aromatic logit offset. 0.0 means "no
        # offset", and `_aromatic_bias_vector` turns that into None so the
        # decode is byte-identical to the pre-2026-10-12 path.
        "bond_aromatic_logit_bias": 0.0,
    }

    out = ns
    if ns.config is None:
        for k, v in _DEFAULTS.items():
            if getattr(out, k, None) is None:
                setattr(out, k, v)
        return out

    import yaml  # type: ignore[import-not-found]

    data = yaml.safe_load(ns.config.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{ns.config}: expected a mapping at the top level")

    flat = {}
    for section, body in data.items():
        if isinstance(body, dict):
            for k, v in body.items():
                # A leaf is addressed BY ITS DEST NAME, not by its YAML path,
                # so the section is only a grouping convenience here. Record
                # the qualified name too, but ONLY when the bare name is not a
                # dest -- otherwise the unknown-key check would reject the
                # config it just read, which is what it did on the first run.
                if k in _DEFAULTS:
                    flat[k] = v
                else:
                    flat[f"{section}.{k}"] = v
        elif section in _DEFAULTS:
            flat[section] = body
        else:
            flat[section] = body

    unknown = sorted(k for k in flat if k not in _DEFAULTS and k != "ckpt")
    # `sweep:` is owned by scripts/launch_cfm_l3_guidance_sweep.sh, which reads
    # the strength grid from the same file so the two cannot disagree. It is
    # deliberately NOT accepted here: this script runs one arm, and silently
    # ignoring a grid it was handed would be exactly the silent-ignore class
    # AGENTS.md 5.11 was written to eliminate.
    #
    # `entry:` is the same story and is REQUIRED by
    # scripts/_config_binding_gate.py, which fails any configs/*.yaml that does
    # not name the script it launches. It is consumed by scripts/start.sh and
    # by nothing here.
    unknown = [
        k for k in unknown
        if not k.startswith("sweep.") and not k.startswith("entry.")
    ]
    if unknown:
        raise SystemExit(
            f"{ns.config}: keys resolve to no argparse dest: {unknown}. "
            "Every key must name a flag of this script."
        )

    for k, v in flat.items():
        if k not in _DEFAULTS:
            # Launcher-owned section (see the `sweep.` note above). Validated
            # as well-formed, deliberately not applied: this script runs one
            # arm, and applying a grid it cannot honour would be the silent lie
            # in a new place.
            continue
        # An explicit CLI flag wins; a store_true is only "passed" if True.
        if k in ("skip_decode", "compare_real"):
            if getattr(ns, k):
                continue
            setattr(out, k, bool(v))
            continue
        if getattr(ns, k, None) is not None:
            continue
        if _DEFAULTS[k] is not None:
            # A None default means "no declared type to check against" --
            # ckpt and out are paths whose default is genuinely absent. Check
            # they are strings rather than comparing against NoneType.
            _ok = isinstance(v, _DEFAULTS[k].__class__) or (
                isinstance(v, (int, float))
                and isinstance(_DEFAULTS[k], (int, float))
                and not isinstance(_DEFAULTS[k], bool)
            )
            if not _ok:
                raise SystemExit(
                    f"{ns.config}: {k}={v!r} is {type(v).__name__}, expected "
                    f"{_DEFAULTS[k].__class__.__name__}"
                )
        elif not isinstance(v, str):
            raise SystemExit(
                f"{ns.config}: {k}={v!r} is {type(v).__name__}, expected a "
                "path string"
            )
        setattr(out, k, v)

    # Every dest must end up with a value. A config that supplies some keys
    # but not others used to leave the rest as None, which surfaced much later
    # as `TypeError: '>' not supported between instances of 'NoneType' and
    # 'float'` deep in the guidance branch -- a crash in the compute path
    # caused by a config gap, which is the hardest kind of trace to follow
    # back. Fall back to the documented default for anything the file did not
    # set, and say which keys those are.
    missing = sorted(
        k for k, dflt in _DEFAULTS.items()
        if getattr(out, k, None) is None and dflt is not None
    )
    for k in missing:
        setattr(out, k, _DEFAULTS[k])
    if missing:
        print(
            f"note: {len(missing)} key(s) not set by {ns.config.name}, using the "
            f"built-in defaults: {', '.join(missing)}",
            file=sys.stderr,
        )

    return out


def main() -> int:
    cli = _merge_config(_parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, a, state = load_model(Path(cli.ckpt), device)
    clouds = load_clouds(a, cli.n_clouds)
    noise_sigma = float(a.get("fm_noise_sigma", 1.0))

    # Wf-A96 (2026-10-19) — record WHICH FILE was measured, not just its
    # path.  The 2026-10-12 arm and this one name the same path but are
    # different files (the checkpoint was overwritten on 2026-10-13), which
    # made a 0.109 -> 1.000 change look like a repair effect.  The date was
    # also hardcoded to 2026-10-12, so every later arm claimed the first
    # run's date.
    import datetime
    import hashlib

    _ckpt_path = Path(cli.ckpt)
    _h = hashlib.sha256()
    with _ckpt_path.open("rb") as _f:
        for _chunk in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_chunk)
    report: Dict[str, Any] = {
        "metric": f"cfm_{cli.arm}_guidance_probe",
        "tag": "MEASURED",
        "date": datetime.date.today().isoformat(),
        "arm": cli.arm,
        "checkpoint": str(_ckpt_path),
        "checkpoint_identity": {
            "sha256": _h.hexdigest(),
            "mtime_iso": datetime.datetime.fromtimestamp(
                _ckpt_path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "legacy_key_migration": state.get("_legacy_key_migration"),
        },
        "cohort": {
            "source": "tmQM real, split=val (composition templates)",
            "n_clouds": len(clouds),
            "method": cli.method,
            "n_steps": cli.n_steps,
            "seed": cli.seed,
            "noise_sigma": noise_sigma,
            "device": str(device),
            "mode": "eval() + forward ODE integration only -- NO TRAINING",
            "guidance_strength": cli.guidance_strength,
            # 2026-10-12 (BLK-C-M4) — the decode-time aromatic offset this
            # arm ran with. Recorded because the aggregator reads the
            # operating point from here rather than from the filename: an arm
            # file whose cohort block omits it would aggregate as "no offset"
            # no matter what the launcher actually passed.
            "bond_aromatic_logit_bias": float(
                cli.bond_aromatic_logit_bias or 0.0
            ),
        },
    }

    sampled = integrate(
        model,
        clouds,
        device,
        method=cli.method,
        n_steps=cli.n_steps,
        seed=cli.seed,
        noise_sigma=noise_sigma,
        guidance_strength=cli.guidance_strength,
    )
    gen_rows = [coord_stats(c["positions"], c["zs"]) for c in sampled]
    report["generated_geometry"] = aggregate(gen_rows)

    if cli.compare_real:
        real_rows = [coord_stats(c.positions, c.atomic_numbers.tolist()) for c in clouds]
        report["real_tmqm_geometry"] = aggregate(real_rows)

    if not cli.skip_decode:
        # 2026-10-12 FIX. `a` is the CHECKPOINT's training-time namespace
        # (`state["args"]`), not the run-time one. Passing it straight through
        # meant every RUN-TIME flag was invisible to the decoder — most
        # importantly `--bond-aromatic-logit-bias`, so an offset sweep
        # measured the same decode four times and produced byte-identical
        # SMILES on every arm.
        #
        # The two namespaces answer different questions and must not be
        # conflated: `a` says how the checkpoint was TRAINED (architecture,
        # vocab table, cutoff), `cli` says how it is being RUN (offset,
        # guidance, arm label). Architecture comes from `a`; the operating
        # point comes from `cli`. Run-time wins on anything both define, since
        # the whole point of a flag is to override what the checkpoint did.
        run_args = dict(a)
        for _k in (
            "bond_aromatic_logit_bias",
            "bond_decision",
            "bond_decision_threshold",
            "connectivity_prior",
            "connectivity_prior_min_logit",
        ):
            _v = getattr(cli, _k, None)
            if _v is not None:
                run_args[_k] = _v
        dec_rows = decode(model, sampled, run_args, state=state)
        got = [r for r in dec_rows if r.get("smiles")]
        report["decode"] = {
            "n_attempted": len(dec_rows),
            "n_with_smiles": len(got),
            "decode_rate": (len(got) / len(dec_rows)) if dec_rows else 0.0,
            # Wf-A96 — the aromatic fraction of the SURVIVORS, requested by
            # cfm_todo.json::L3_A_B_Phase_3_redo.  A repaired molecule
            # (Wf-A84 repair tag) is valid but has had its aromaticity
            # stripped, so the two shares are reported separately:
            # any_aromatic over every survivor, and the same quantity
            # restricted to clean (repair-free) decodes.
            "aromatic": _aromatic_summary(got),
            "n_repaired": sum(1 for r in got if r.get("repair")),
            "repair_counts": _count_by(got, "repair"),
            "single_fragment_ratio": (
                sum(1 for r in got if r.get("n_fragments") == 1) / len(got)
            )
            if got
            else None,
            "n_errors": sum(1 for r in dec_rows if r.get("error")),
            # 2026-10-12. A decode in which EVERY cloud raised is a broken
            # harness, not a model that generated nothing, and the two print
            # identically as "DECODE: 0 / N". Four broken imports in this very
            # function produced exactly that reading and it was nearly filed
            # as a result. Distinct the two here so no reader has to guess.
            "all_clouds_errored": bool(dec_rows) and all(
                r.get("error") for r in dec_rows
            ),
            "rows": dec_rows[:200],
        }

    tmp = Path(str(cli.out) + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, cli.out)  # atomic, per AGENTS.md 5.6
    print(json.dumps({k: v for k, v in report.items() if k != "decode"}, indent=1)[:2500])
    if "decode" in report:
        d = report["decode"]
        print("DECODE:", d["n_with_smiles"], "/", d["n_attempted"], "single_frag",
              d["single_fragment_ratio"])
        if d.get("all_clouds_errored"):
            # Loud, and non-zero exit: this is a harness failure and must not
            # be mistaken for a model that generated nothing.
            first = next((r["error"] for r in d["rows"] if r.get("error")), "?")
            print("DECODE HARNESS FAILURE: every cloud raised. First error:",
                  first, file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())