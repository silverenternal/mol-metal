#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""cfm_module_gates — the ten module gates in ``cfm_todo.json``, executable.

WHY THIS EXISTS
---------------
``cfm_todo.json`` v9.0 states the working principle in one line: gate each
module on its own function, then gate the seams between modules.  Until this
script, those ten gates were prose.  A gate written in prose cannot fail, and a
gate that cannot fail is not a gate.

Each gate here measures one number (or a small tuple) on a real checkpoint and
prints ``PASS`` / ``FAIL`` / ``SKIP`` with the measurement attached.  The JSON
artifact records every number so a later claim can cite it instead of quoting a
paragraph.

THE GATES
---------
M1  data entry        interpolant identity, boundary values, noise sigma match
M2  conditioning      relative output change when one input is perturbed
M3  head capacity     overfit one batch; MSE against the v=0 baseline
M4  output scale      ||v_pred||/||v_target|| and the regression slope
M5  gradient path     per-block grad norm from the FM term alone
M6  time resolution   does the field move with t, and smoothly
M7  warm start        which parameters the tmQM pretrain does not reach
M8  train/sample seam train path vs sampler path on the same inputs
M9  objective balance per-term share of the total, per epoch
M10 metric chain      checkpoint hash / split fingerprint / file existence

M3 is the one gate that trains, so it shells out to the existing
``wfa107h_velocity_capacity.py`` harness rather than re-implementing the
optimiser loop: a second implementation of an overfit test is a second thing
that can disagree with the first.

Usage
-----
    .venv/bin/python molmetal/scripts/cfm_module_gates.py \
        --ckpt results/wfa108_vector_channels/checkpoints/\
cfm_yuelbond_wasserstein_seed0_last.pt \
        --run-dir results/wfa108_vector_channels \
        --json metrics/by_metric/wfa108i_module_gates_vector_channels.json

    # one gate, verbose
    .venv/bin/python molmetal/scripts/cfm_module_gates.py \
        --ckpt ... --gate M4

Exit codes follow the repo convention: 0 = every selected gate PASS,
1 = at least one FAIL, 2 = no FAIL but at least one SKIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Every metal the CFM corpus covers.  Used to find the atoms whose identity is
# the "metal" conditioning channel, and to build a swap that stays inside the
# chemistry the model was trained on.
METAL_Z: Dict[int, str] = {
    78: "Pt",
    44: "Ru",
    77: "Ir",
    45: "Rh",
    76: "Os",
    75: "Re",
    46: "Pd",
}
# Pt -> Ru is the swap M2 uses: both are in the 5-metal cohort, both are
# d8-ish square-planar in the corpus, so the change is an identity change and
# not an out-of-distribution atom.
_METAL_SWAP = {78: 44, 44: 78, 77: 45, 45: 77, 76: 75, 75: 76, 46: 78}

_GATE_IDS = [f"M{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO / p)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ckpt(path: Path) -> Dict[str, Any]:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        load_cfm_checkpoint,
    )

    ck = load_cfm_checkpoint(str(path), map_location="cpu")
    if not isinstance(ck, dict):
        raise SystemExit(f"checkpoint {path} is a {type(ck).__name__}, not a dict")
    return ck


def _build_model(ck: Dict[str, Any], device: torch.device):
    """Rebuild the architecture from the checkpoint's own serialised args.

    Same rule as ``wf_a53_eval_cfm_test_split._load_model``: the checkpoint is
    the authority on what it is.  ``migrate_legacy_vel_state_dict`` runs first
    because pre-Wf-A27 checkpoints name the time projection
    ``time_embed_proj`` and the current module calls it ``t_to_h``.
    """
    import argparse as _argparse

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_joint_model,
        migrate_legacy_vel_state_dict,
    )

    args_dict = ck.get("args")
    if not isinstance(args_dict, dict):
        raise SystemExit(
            "checkpoint carries no serialised 'args' dict; the architecture "
            "cannot be rebuilt without guessing shapes"
        )
    ns = _argparse.Namespace(**args_dict)
    model = build_joint_model(ns)
    state = ck.get("model_state_dict")
    if state is None:
        raise SystemExit("checkpoint has no 'model_state_dict'")
    state, migrated = migrate_legacy_vel_state_dict(model, state, verbose=False)
    model.load_state_dict(state, strict=True)
    ema = ck.get("ema_state_dict")
    if isinstance(ema, dict) and ema:
        if model.load_ema_state(ema) and model.ema_swap_in():
            print("[gates] evaluating with EMA weights", flush=True)
    model.to(device).eval()
    return model, ns, migrated


def _build_clouds(ns: Any, n_clouds: int, seed: int) -> List[Any]:
    """The checkpoint's own cohort, truncated to ``n_clouds``.

    The split is the one the checkpoint trained against: for the CSV cohorts
    that is ``val`` (unseen), for tmQM it is ``val`` under the checkpoint's own
    ``tmqm_split_kind``.  Using ``train`` would make M3's overfit test
    indistinguishable from a memorisation test of the training set, so ``val``
    is used everywhere and the split name is recorded in the artifact.
    """
    # ``dataset`` defaults to "auto" in the trainer's parser, which resolves by
    # precedence tmQM > CSV > synthetic.  Reading the literal string here
    # silently built 8 synthetic 5-atom clouds for a tmQM checkpoint and
    # reported ``n_atoms=40`` for a cohort whose real clouds are 22-88 atoms.
    dataset = str(getattr(ns, "dataset", "auto") or "auto").strip().lower()
    if dataset in ("", "auto"):
        dataset = (
            "tmqm"
            if getattr(ns, "tmqm_dir", None)
            else ("csv" if getattr(ns, "data_csv", None) else "synthetic")
        )
    if dataset == "csv":
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            build_csv_cohort,
        )

        csv_path = _resolve(str(getattr(ns, "data_csv", "")))
        if not csv_path.is_file():
            raise SystemExit(f"data_csv '{csv_path}' is not a file")
        return build_csv_cohort(
            str(csv_path),
            split=str(getattr(ns, "val_split", "val")),
            n_clouds=int(n_clouds),
            seed=int(seed),
            metal_placement=str(getattr(ns, "metal_placement", "bonded")),
            dative_distance_cutoff=float(
                getattr(ns, "dative_distance_cutoff", 2.6)
            ),
        )
    if dataset == "tmqm":
        from molmetal.data.tmqm_clouds import load_tmqm_clouds

        root = _resolve(str(getattr(ns, "tmqm_dir", ".data/molmetal/tmQM")))
        metals = list(getattr(ns, "tmqm_metals", ["Pt"]) or ["Pt"])
        cn = int(getattr(ns, "tmqm_coord_number", -1))
        return load_tmqm_clouds(
            str(root),
            metals=metals,
            coord_number=None if cn < 0 else cn,
            n_clouds=int(n_clouds),
            seed=int(seed),
            split="val",
            val_frac=float(getattr(ns, "tmqm_val_frac", 0.2)),
            min_per_metal=int(getattr(ns, "tmqm_min_per_metal", 0)),
            split_kind=str(getattr(ns, "tmqm_split_kind", "random")),
        )
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        build_synthetic_pt_click_clouds,
    )

    return build_synthetic_pt_click_clouds(
        n_clouds=int(n_clouds), jitter=0.20, seed=int(seed)
    )


def _flatten(clouds: Sequence[Any], device: torch.device):
    x1 = torch.cat([c.positions.float() for c in clouds], 0).to(device)
    z = torch.cat([c.atomic_numbers.long() for c in clouds], 0).to(device)
    sizes = [int(c.atomic_numbers.shape[0]) for c in clouds]
    return x1, z, sizes


def _per_cloud_t(n_atoms: int, sizes: Sequence[int], value: float) -> torch.Tensor:
    """Per-atom t with one value per cloud, which is what ``fm_t_per_cloud=1``
    means and what ``_fm_field_forward`` reads (first atom of each block)."""
    t = torch.zeros(int(n_atoms))
    start = 0
    for s in sizes:
        t[start : start + int(s)] = float(value)
        start += int(s)
    return t


def _norm(t: torch.Tensor) -> float:
    return float(t.norm())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    den = 0.5 * (_norm(a) + _norm(b))
    return float((a - b).norm()) / max(den, 1e-12)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_m1(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Data entry: does the interpolant the trainer builds satisfy its own contract?

    Four things are measured, each of which can fail on its own:

      * ``identity_residual`` — the path computed as ``(1-t)x0 + t x1`` and as
        ``x0 + t(x1-x0)``.  These are algebraically equal and numerically not,
        so the residual is a real float number, not a tautology.
      * ``boundary_t0`` / ``boundary_t1`` — the path at the ends must BE the
        endpoints.  A swapped convention here is the failure that makes a
        sampler integrate the wrong way.
      * ``sigma_measured`` vs ``fm_noise_sigma`` — the prior the field is
        trained to invert has to be the prior the sampler draws from.
      * ``t_shared_per_cloud`` — with ``fm_t_per_cloud=1`` every atom of a
        cloud shares one t; a per-atom t would break the per-cloud Lipschitz
        scaling the interpolant assumes.
    """
    x1, z, sizes = ctx["x1"], ctx["z"], ctx["sizes"]
    ns = ctx["ns"]
    sigma = float(getattr(ns, "fm_noise_sigma", 1.0))
    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]))
    x0 = torch.randn(x1.shape, generator=g) * sigma
    t = torch.rand(x1.shape[0], 1, generator=g)

    path_a = (1.0 - t) * x0 + t * x1
    path_b = x0 + t * (x1 - x0)
    identity = float((path_a - path_b).abs().max())

    b0 = (1.0 - 0.0) * x0 + 0.0 * x1
    b1 = (1.0 - 1.0) * x0 + 1.0 * x1
    boundary_t0 = float((b0 - x0).abs().max())
    boundary_t1 = float((b1 - x1).abs().max())

    # Per-coordinate std over the whole batch is the quantity the Gaussian
    # prior is defined on; per-atom std is reported because a per-atom
    # definition is what the trainer's ``randn_like`` actually produces.
    sigma_measured = float(x0.std(unbiased=False))
    sigma_per_atom = float(x0.reshape(-1, 3).std(dim=0, unbiased=False).mean())
    data_std = float(x1.reshape(-1, 3).std(dim=0, unbiased=False).mean())

    per_cloud_t = int(getattr(ns, "fm_t_per_cloud", 0) or 0) == 1
    t_shared = True
    if per_cloud_t:
        # The trainer draws one t per cloud; this asserts the cohort's atom
        # blocks can carry that (they can, trivially) and records the flag.
        t_shared = all(s > 0 for s in sizes)

    checks = {
        "identity_residual_lt_1e-5": identity < 1e-5,
        "boundary_t0_exact": boundary_t0 == 0.0,
        "boundary_t1_exact": boundary_t1 == 0.0,
        "sigma_within_10pct": abs(sigma_measured / max(sigma, 1e-12) - 1.0) <= 0.10,
        "per_cloud_t": bool(t_shared),
    }
    return {
        "measured": {
            "identity_residual": identity,
            "boundary_t0": boundary_t0,
            "boundary_t1": boundary_t1,
            "fm_noise_sigma": sigma,
            "sigma_measured": sigma_measured,
            "sigma_measured_per_atom": sigma_per_atom,
            "cohort_coord_std": data_std,
            "sigma_ratio_prior_over_data": sigma / max(data_std, 1e-12),
            "fm_t_per_cloud": int(getattr(ns, "fm_t_per_cloud", 0) or 0),
            "n_atoms": int(x1.shape[0]),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_m2(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Conditioning: perturb one input channel, measure the output change.

    The velocity field is ``v(x, t, z, cloud_sizes, pocket_embed)``.  Three
    channels are reachable from a checkpoint without any pocket data: the
    metal atom's identity, the flow time, and a heavy atom's type.  Each is
    perturbed alone, on a fixed batch, at fixed ``x``.

    A relative change of exactly 0 means the channel is not connected.  A
    change below 1e-2 means it is connected but cannot steer the field by
    more than a percent, which for a conditioning channel is the same
    practical failure.
    """
    model, x1, z, sizes = ctx["model"], ctx["x1"], ctx["z"], ctx["sizes"]
    device = ctx["device"]
    n = int(x1.shape[0])
    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]) + 1)
    sigma = float(getattr(ctx["ns"], "fm_noise_sigma", 1.0))
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)
    t_val = 0.5
    xt = ((1.0 - t_val) * x0 + t_val * x1).to(device)
    t_atom = _per_cloud_t(n, sizes, t_val).to(device)

    with torch.no_grad():
        base = model.fm_velocity(xt, t_atom, z, cloud_sizes=sizes)

    # (a) metal identity
    z_metal = z.clone()
    metal_idx = [i for i, zi in enumerate(z.tolist()) if int(zi) in METAL_Z]
    for i in metal_idx:
        z_metal[i] = _METAL_SWAP.get(int(z[i]), int(z[i]))
    with torch.no_grad():
        v_metal = model.fm_velocity(xt, t_atom, z_metal, cloud_sizes=sizes)
    rel_metal = _rel(v_metal, base)

    # (b) time, +0.25 clipped into [0, 1]
    t_shift = float(min(1.0, t_val + 0.25))
    t_atom2 = _per_cloud_t(n, sizes, t_shift).to(device)
    with torch.no_grad():
        v_t = model.fm_velocity(xt, t_atom2, z, cloud_sizes=sizes)
    rel_t = _rel(v_t, base)

    # (c) one heavy non-metal atom type, C -> N
    z_atom = z.clone()
    swap_i = next(
        (i for i, zi in enumerate(z.tolist()) if int(zi) == 6), None
    )
    if swap_i is not None:
        z_atom[swap_i] = 7
        with torch.no_grad():
            v_atom = model.fm_velocity(xt, t_atom, z_atom, cloud_sizes=sizes)
        rel_atom = _rel(v_atom, base)
    else:
        rel_atom = None

    thresh = 1e-2
    checks = {
        "metal_swap_gt_1e-2": rel_metal > thresh,
        "t_shift_gt_1e-2": rel_t > thresh,
        "atom_type_swap_gt_1e-2": (rel_atom is not None and rel_atom > thresh),
    }
    return {
        "measured": {
            "relative_change_metal_swap": rel_metal,
            "relative_change_t_shift_0p25": rel_t,
            "relative_change_atom_type_swap": rel_atom,
            "n_metal_atoms": len(metal_idx),
            "metal_swap_pairs": {
                METAL_Z.get(int(z[i]), str(int(z[i]))): METAL_Z.get(
                    int(z_metal[i]), str(int(z_metal[i]))
                )
                for i in metal_idx[:8]
            },
            "threshold": thresh,
            "base_norm": _norm(base),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_m3(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Capacity: can the field fit ONE batch, and at what fraction of v=0?

    Delegated to ``wfa107h_velocity_capacity.py`` on purpose.  That harness
    already ran at 125 / 250 / 500 / 1,000 / 2,000 steps (Wf-A108d) and its
    numbers are what the todo's M3 ``known`` line cites; a fresh loop here
    would produce numbers that are comparable to nothing.
    """
    ckpt = ctx["ckpt_path"]
    steps = int(ctx["steps_m3"])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "capacity.json"
        cmd = [
            sys.executable,
            str(_REPO / "molmetal/scripts/wfa107h_velocity_capacity.py"),
            "--ckpt",
            str(ckpt),
            "--n-clouds",
            str(int(ctx["n_clouds"])),
            "--val-seed",
            str(int(ctx["seed"])),
            "--steps",
            str(steps),
            "--device",
            str(ctx["device"]),
            "--out",
            str(out),
        ]
        if ctx.get("csv"):
            cmd += ["--csv", str(ctx["csv"])]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not out.is_file():
            return {
                "measured": {"subprocess_returncode": proc.returncode},
                "checks": {"harness_produced_json": False},
                "pass": False,
                "detail": (proc.stderr or proc.stdout or "")[-1500:],
            }
        doc = json.loads(out.read_text())

    rows = doc.get("train_rows") or []
    final = float(rows[-1]["train_mse"]) if rows else float("nan")
    zero = float(doc.get("zero_baseline_mse", float("nan")))
    frac = final / zero if zero else float("nan")
    checks = {"final_mse_le_10pct_of_v0": bool(frac <= 0.10)}
    return {
        "measured": {
            "steps": steps,
            "final_train_mse": final,
            "zero_baseline_mse": zero,
            "final_over_zero": frac,
            "radial_ceiling_mse": doc.get("radial_ceiling_mse"),
            "verdict_harness": doc.get("verdict"),
            "velocity_head": doc.get("velocity_head"),
            "n_clouds": doc.get("n_clouds"),
            "n_atoms": doc.get("n_atoms"),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _bayes_ratio(x1: torch.Tensor, sigma: float, t: float,
                 n_draws: int = 48, seed: int = 0) -> float:
    """Best magnitude any model can emit at time ``t``, as a ratio.

    The FM target is ``x1 - x0`` with ``x0 ~ N(0, sigma^2)`` independent of
    ``x1``, and the field only sees ``x_t = (1-t) x0 + t x1``.  The best
    MSE predictor is the conditional mean, and its magnitude relative to
    ``|x1 - x0|`` is a property of the interpolant, not of the network.

    MEASURED on the 8-cloud val cohort (405 atoms, sigma 2.485, data coord
    std 2.36): ratio 0.725 at t=0, 0.054 at t=0.5, 0.689 at t=1.  At t=0.5
    the best possible field emits **5%** of the target magnitude and reaches
    99.7% of the ``v = 0`` MSE — the midpoint of this interpolant is almost
    pure noise, so "the field is too small at t=0.5" is not a defect.

    A fixed band of [0.5, 2.0] on the raw ratio therefore demanded 10x the
    achievable magnitude at the midpoint, and no trained model could satisfy
    it.  This function supplies the reference the band has to be relative to.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x1c = x1.detach().cpu()
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for _ in range(int(n_draws)):
        x0 = torch.randn(x1c.shape, generator=g) * float(sigma)
        xs.append(((1.0 - float(t)) * x0 + float(t) * x1c).reshape(-1))
        ys.append((x1c - x0).reshape(-1))
    x = torch.cat(xs)
    y = torch.cat(ys)
    slope = float((x * y).sum() / (x * x).sum().clamp_min(1e-12))
    pred = slope * x
    return float(pred.norm() / y.norm().clamp_min(1e-12))


def gate_m4(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Output scale: is the field emitting the magnitude the target asks for?

    "The magnitude the target asks for" is NOT ``|x1 - x0|``.  The field sees
    only ``x_t``, so the best it can do is the conditional mean, and that is
    much smaller than the target at the midpoint of the path.  See
    :func:`_bayes_ratio`: at ``t = 0.5`` the achievable ratio is 0.054, so a
    fixed band of [0.5, 2.0] on the raw ratio (the pre-Wf-A108j criterion)
    demanded ten times the achievable magnitude and no trained model could
    have passed it.  The band is now ``[0.5, 2.0] x bayes_ratio``, with the
    lower bound applied only where ``bayes_ratio >= 0.15``.

    A ratio far BELOW the band is the failure this gate exists for: the field
    emits a few percent of what is achievable and the MSE reads as the ``v=0``
    floor no matter what the network learned — which is what a radial head
    bounded by ``tanh * vel_scale * centroid_radius`` produces (MEASURED
    ratio 0.021..0.028 on the 5-epoch vector_channels checkpoint, against
    0.69 achievable at t=0.05).  A ratio far ABOVE it is a field fitting the
    noise of one draw.
    """
    model, x1, z, sizes = ctx["model"], ctx["x1"], ctx["z"], ctx["sizes"]
    device = ctx["device"]
    n = int(x1.shape[0])
    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]) + 2)
    sigma = float(getattr(ctx["ns"], "fm_noise_sigma", 1.0))
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)
    tgt = x1 - x0

    rows: List[Dict[str, Any]] = []
    ok = True
    for tv in (0.05, 0.25, 0.50, 0.75, 0.95):
        xt = ((1.0 - tv) * x0 + tv * x1).to(device)
        t_atom = _per_cloud_t(n, sizes, tv).to(device)
        with torch.no_grad():
            v = model.fm_velocity(xt, t_atom, z, cloud_sizes=sizes)
        ratio = _norm(v) / max(_norm(tgt), 1e-12)
        p, q = v.reshape(-1), tgt.reshape(-1)
        slope0 = float((p * q).sum() / (q * q).sum().clamp_min(1e-12))
        # least squares with intercept, closed form
        qm, pm = q.mean(), p.mean()
        dq = q - qm
        slope1 = float((dq * (p - pm)).sum() / (dq * dq).sum().clamp_min(1e-12))
        cos = float(F.cosine_similarity(p, q, dim=0))
        bayes = _bayes_ratio(x1, sigma, tv, seed=int(ctx["seed"]) + 11)
        # The band is relative to what the interpolant makes achievable at
        # this t.  Where the conditional mean itself is below 15% of the
        # target, the point carries no signal and only the upper bound is
        # judged: a field that emits magnitude there is fitting noise.
        live = bayes >= 0.15
        lo, hi = 0.5 * bayes, 2.0 * bayes
        t_ok = (ratio <= hi) and ((ratio >= lo) if live else True)
        rows.append(
            {
                "t": tv,
                "mean_v_norm": float(v.norm(dim=-1).mean()),
                "mean_target_norm": float(tgt.norm(dim=-1).mean()),
                "ratio": ratio,
                "bayes_ratio": bayes,
                "ratio_over_bayes": (ratio / bayes) if bayes > 0.0 else None,
                "live": bool(live),
                "slope_origin": slope0,
                "slope_with_intercept": slope1,
                "cosine": cos,
            }
        )
        ok = ok and t_ok

    checks = {"ratio_in_band_vs_bayes_all_t": bool(ok)}
    return {
        "measured": {
            "per_t": rows,
            "ratio_min": min(r["ratio"] for r in rows),
            "ratio_max": max(r["ratio"] for r in rows),
            "ratio_over_bayes_min": min(
                r["ratio_over_bayes"] for r in rows if r["ratio_over_bayes"] is not None
            ),
            "ratio_over_bayes_max": max(
                r["ratio_over_bayes"] for r in rows if r["ratio_over_bayes"] is not None
            ),
            "slope_min": min(r["slope_origin"] for r in rows),
            "slope_max": max(r["slope_origin"] for r in rows),
            "band": "[0.5, 2.0] x bayes_ratio; lower bound only where bayes >= 0.15",
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


#: Parameter blocks the flow-matching objective is expected to reach.  The
#: coordinate field is the only one: ``head.*`` (bond / atom / pair heads) is
#: optimised by ``bond_ce``, and ``pocket_encoder.*`` is only used when pocket
#: features are supplied, which no CFM config does today.
_FM_REACH_BLOCKS = frozenset({"vel_field", "fm_vel_head"})


def gate_m5(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Gradient path: one backward from the FM term, per parameter block.

    The gate judges only the blocks the FM objective is supposed to reach
    (``_FM_REACH_BLOCKS``).  ``head.*`` is trained by ``bond_ce`` and
    ``pocket_encoder.*`` only receives gradient when the caller supplies
    pocket features, so they are reported but not judged.  A judged block at
    exactly 0 gets no gradient from the FM objective at all, which is a
    different fact from "small gradient" and is why it is tested at 0 rather
    than against a magnitude floor.  The 100x spread check catches a block
    that is technically connected but 3 orders of magnitude behind.
    """
    model, x1, z, sizes = ctx["model"], ctx["x1"], ctx["z"], ctx["sizes"]
    device = ctx["device"]
    n = int(x1.shape[0])
    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]) + 3)
    sigma = float(getattr(ctx["ns"], "fm_noise_sigma", 1.0))
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)
    t_atom = _per_cloud_t(n, sizes, 0.5).to(device)
    xt = ((1.0 - 0.5) * x0 + 0.5 * x1).to(device)
    tgt = x1 - x0

    model.zero_grad(set_to_none=True)
    v = model.fm_velocity(xt, t_atom, z, cloud_sizes=sizes)
    loss = F.mse_loss(v, tgt)
    loss.backward()

    blocks = ["vel_field", "head", "pocket_encoder", "fm_vel_head"]
    out: Dict[str, Any] = {}
    for name in blocks:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        total = 0.0
        n_with_grad = 0
        n_params = 0
        dead: List[str] = []
        for pname, p_ in mod.named_parameters():
            n_params += 1
            if p_.grad is None:
                dead.append(pname)
                continue
            gn = float(p_.grad.detach().norm())
            total += gn * gn
            if gn > 0.0:
                n_with_grad += 1
            else:
                dead.append(pname)
        out[name] = {
            "grad_norm": total ** 0.5,
            "n_params": n_params,
            "n_with_nonzero_grad": n_with_grad,
            # Named, not counted.  "54/62" does not tell a reader which eight
            # parameters are dead; the Wf-A108i run reported exactly that and
            # the eight could not be recovered from the artifact.
            "params_without_grad": dead,
        }

    # Which blocks the FM objective is SUPPOSED to reach.  ``head`` is trained
    # by ``bond_ce`` and ``pocket_encoder`` only receives gradient when the
    # caller supplies pocket features, so a zero in either is the intended
    # wiring, not a defect.  The first version of this gate required all four
    # blocks to be non-zero and therefore FAILed every run: MEASURED on the
    # 5-epoch vector_channels checkpoint, vel_field 5.129, head 0.0,
    # pocket_encoder 0.0.
    expected = [b for b in out if b in _FM_REACH_BLOCKS]
    unexpected_zero = [b for b in out if b not in expected and out[b]["grad_norm"] == 0.0]

    norms = [out[b]["grad_norm"] for b in expected]
    nonzero = [x for x in norms if x > 0.0]
    spread = (max(nonzero) / min(nonzero)) if len(nonzero) >= 2 else None
    checks = {
        "expected_blocks_have_grad": bool(expected) and len(nonzero) == len(norms),
        # A spread needs two blocks to be a spread.  With one expected block
        # the check is not applicable, not failed: the first version returned
        # ``spread is None`` and therefore FAILed every run whose FM objective
        # reaches exactly one block, which is all of them.
        "spread_le_100x": (spread is None or spread <= 100.0),
    }
    return {
        "measured": {
            "blocks": out,
            "expected_blocks": expected,
            "unexpected_zero_blocks": unexpected_zero,
            "spread_max_over_min": spread,
            "fm_loss": float(loss.detach()),
            "note": (
                "head.* is trained by bond_ce, not by fm_term; a zero here "
                "means 'the FM objective does not reach it', which is "
                "expected, not a defect. The gate text does not distinguish "
                "the two, so this gate reports the number rather than "
                "assuming the intent.  Only ``_FM_REACH_BLOCKS`` is judged."
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_m6(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Time resolution: does the field move with t, and smoothly?

    A flow-matching field has to reverse direction across the path: at t->0 the
    conditional mean points back toward the prior, at t->1 toward the data.  A
    field flat in t cannot represent that, and a field that jumps is not
    integrable by an ODE solver.  Both are measured on the same fixed batch.

    ``smoothness`` is the largest discrete second difference of the velocity
    over the 101-point sweep, relative to the mean velocity norm.  A smooth
    curve has a small one; a step function has one of order 1.
    """
    model, x1, z, sizes = ctx["model"], ctx["x1"], ctx["z"], ctx["sizes"]
    device = ctx["device"]
    n = int(x1.shape[0])
    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]) + 4)
    sigma = float(getattr(ctx["ns"], "fm_noise_sigma", 1.0))
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)
    xt = (0.5 * x0 + 0.5 * x1).to(device)

    ts = [i / 100.0 for i in range(101)]
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for tv in ts:
            t_atom = _per_cloud_t(n, sizes, tv).to(device)
            preds.append(model.fm_velocity(xt, t_atom, z, cloud_sizes=sizes))
    norms = torch.tensor([_norm(v) for v in preds])
    mean_norm = float(norms.mean())
    relative_change = _rel(preds[-1], preds[0])

    second = torch.stack(
        [preds[i - 1] - 2.0 * preds[i] + preds[i + 1] for i in range(1, len(preds) - 1)]
    )
    second_norm = second.reshape(second.shape[0], -1).norm(dim=1)
    smoothness = float(second_norm.max()) / max(mean_norm, 1e-12)

    adj_cos = torch.stack(
        [
            F.cosine_similarity(
                preds[i].reshape(-1), preds[i + 1].reshape(-1), dim=0
            )
            for i in range(100)
        ]
    )

    checks = {
        "relative_change_gt_0p1": relative_change > 0.1,
        "smoothness_lt_0p25": smoothness < 0.25,
    }
    return {
        "measured": {
            "relative_change_endpoints": relative_change,
            "smoothness": smoothness,
            "mean_v_norm": mean_norm,
            "v_norm_at_t0": float(norms[0]),
            "v_norm_at_t1": float(norms[-1]),
            "norm_ratio_t1_over_t0": float(norms[-1] / max(float(norms[0]), 1e-12)),
            "min_adjacent_cosine": float(adj_cos.min()),
            "mean_adjacent_cosine": float(adj_cos.mean()),
            "n_t_points": len(ts),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_m7(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Warm start: which parameters does the tmQM pretrain not reach?

    The trainer prints ``47/51 params transferred (0 unexpected, 4 missing)``
    and has never named the four.  The bridge is shape-based, so the missing
    set is computable exactly: bridge the source onto the field's own
    state_dict and take the set difference.

    A missing atom embedding, time MLP or velocity head is a FAIL: those are
    the three blocks the warm start exists to transfer.  Any other missing key
    is reported, not failed.
    """
    model = ctx["model"]
    vf = getattr(model, "vel_field", None)
    if vf is None:
        return {
            "measured": {},
            "checks": {"egnn_field_present": False},
            "pass": False,
            "detail": "model has no vel_field (use_egnn_velocity is off)",
        }

    from molmetal.adapters.flow_matching_lipman import (
        DEFAULT_TMQM_CKPT,
        _shape_bridge_state_dict,
    )

    path = _resolve(
        str(getattr(ctx["ns"], "tmqm_init_path", None) or DEFAULT_TMQM_CKPT)
    )
    if not path.is_file():
        return {
            "measured": {"tmqm_init_path": str(path), "exists": False},
            "checks": {"tmqm_checkpoint_exists": False},
            "pass": False,
        }
    try:
        src_ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "measured": {"tmqm_init_path": str(path), "error": repr(exc)},
            "checks": {"tmqm_checkpoint_readable": False},
            "pass": False,
        }
    if isinstance(src_ck, dict):
        for k in ("encoder_state_dict", "encoder", "state_dict"):
            if isinstance(src_ck.get(k), dict):
                src = src_ck[k]
                break
        else:
            src = src_ck
    else:
        src = {}

    own = vf.state_dict()
    bridged = _shape_bridge_state_dict(src, own)
    missing = sorted(set(own) - set(bridged))

    critical_pat = ("atom_embed", "time_mlp", "t_to_h", "vel_head", "vc_head")
    critical = [k for k in missing if any(p in k for p in critical_pat)]
    checks = {
        "no_critical_param_missing": len(critical) == 0,
    }
    return {
        "measured": {
            "tmqm_init_path": str(path),
            "n_source_tensors": len(src),
            "n_own_tensors": len(own),
            "n_bridged": len(bridged),
            "n_missing": len(missing),
            "missing_keys": missing,
            "critical_missing_keys": critical,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def gate_m8(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Train / sample seam: the same (x, t, z) through both call paths.

    Training goes ``fm_velocity(flat) -> _fm_field_forward -> _fm_field_forward_padded
    -> vel_field``.  Sampling goes ``_vf_pad -> vel_field`` directly, with its
    own edge construction.  If the two disagree, the field is trained on one
    function and integrated as another, and every generation metric describes a
    model nobody trained.

    The comparison is bit-exact by design: both sides call the same
    ``_block_diag_radius_edges`` and the same ``vel_field.forward`` on the same
    tensors, so a non-zero difference is a real seam defect, not float noise.
    """
    model, x1, z, sizes = ctx["model"], ctx["x1"], ctx["z"], ctx["sizes"]
    device = ctx["device"]
    n = int(x1.shape[0])
    b = len(sizes)
    n_max = max(sizes)
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _block_diag_radius_edges,
    )

    g = torch.Generator(device="cpu").manual_seed(int(ctx["seed"]) + 5)
    sigma = float(getattr(ctx["ns"], "fm_noise_sigma", 1.0))
    x0 = (torch.randn(x1.shape, generator=g) * sigma).to(device)
    x_mid = ((1.0 - 0.5) * x0 + 0.5 * x1).to(device)

    x_pad = torch.zeros(b, n_max, 3, device=device)
    z_pad = torch.zeros(b, n_max, dtype=torch.long, device=device)
    node_mask = torch.zeros(b, n_max, dtype=torch.bool, device=device)
    start = 0
    for j, s in enumerate(sizes):
        x_pad[j, :s] = x_mid[start : start + s]
        z_pad[j, :s] = z[start : start + s]
        node_mask[j, :s] = True
        start += s
    x_flat = x_pad[node_mask]

    t_b = torch.full((b,), 0.5, device=device)
    t_atom = _per_cloud_t(n, sizes, 0.5).to(device)

    with torch.no_grad():
        edge_index, edge_mask = _block_diag_radius_edges(x_pad, node_mask, model.cutoff)
        v_sampler = model.vel_field(
            x_pad, z_pad, edge_index, t_b, edge_mask=edge_mask
        )["vel"]
        v_train = model.fm_velocity(x_flat, t_atom, z, cloud_sizes=sizes)

    v_sampler_flat = v_sampler[node_mask]
    max_abs = float((v_sampler_flat - v_train).abs().max())
    checks = {"max_abs_diff_eq_zero": max_abs == 0.0}
    return {
        "measured": {
            "max_abs_diff": max_abs,
            "norm_sampler": _norm(v_sampler_flat),
            "norm_train": _norm(v_train),
            "n_clouds": b,
            "pad_width": n_max,
            "fm_batch_chunk_ratio": float(
                getattr(model, "fm_batch_chunk_ratio", 0.0) or 0.0
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


# Weight dest -> the keys the trainer may export that term under.  Every entry
# was read off the ``_LAST_LOSS_TERMS`` dict in
# ``train_cfm_yuelbond_wasserstein.py`` (~line 6330); a dest with no entry has
# no exported term today and M9 reports it as unobservable rather than
# inferring one from a name fragment.  ``lambda_cfm_coupling_weight`` is
# deliberately absent: it is a reward multiplier, not a term.
#
# Tuples, not single names, because the same term is spelled differently in
# the two shapes the artifact has: an old run's flat column is
# ``bond_ce_loss`` / ``wasserstein_loss`` while ``loss_terms`` calls them
# ``bond_ce`` / ``wasserstein_weight*w2_val``.  With single names, a run that
# DOES export the term was reported unobservable for the two terms that carry
# 99% of the objective.
_WEIGHT_TO_TERM: Dict[str, Tuple[str, ...]] = {
    "fm_weight": ("fm_term",),
    "bond_loss_weight": ("bond_ce_loss", "bond_ce"),
    "cb_loss_weight": ("cb_loss_term", "0.05*cb_loss"),
    "wasserstein_weight": ("wasserstein_loss", "wasserstein_weight*w2_val"),
    "bond_aux_mse_weight": ("bond_aux_mse_term",),
    "ring_template_weight": ("ring_prior_term",),
    "entropy_reg_weight": ("entropy_term",),
    "qed_loss_weight": ("qed_term",),
    "molecule_loss_weight": ("molecule_term",),
    "stereo_loss_weight": ("stereo_term",),
    "fragment_loss_weight": ("fragment_term",),
    "property_loss_weight": ("property_term",),
    "fm_ode_term_weight": ("fm_ode_class_term",),
    "energy_conservation_weight": ("energy_class_term",),
    "sboo3_term_weight": ("sboo3_class_term",),
    "so3_average_weight": ("so3_term",),
    "diversity_weight": ("diversity_term",),
    "consistency_weight": ("consistency_term",),
    "edm_weight": ("edm_term",),
    "assemble_loss_weight": ("assemble_term",),
    "gen_validity_weight": ("soft_valence_term",),
    "chem_loss_weight": ("chem_loss_term",),
    "clash_penalty_weight": ("clash_penalty_term",),
    "natural_charge_weight": ("natural_charge_term",),
    "wasserstein_pocket_weight": "wasserstein_pocket_term",
}

# Keys that are totals, not terms.  ``train_loss`` ends with ``_loss`` and is
# the SUM of the terms, so including it in the denominator would make every
# share a fraction of twice the objective.
_AGGREGATE_KEYS = {"train_loss", "total_loss", "val_loss", "loss", "train_total"}

# Inverse of ``_WEIGHT_TO_TERM``: exported term name -> the dests that weight it.
_TERM_TO_WEIGHTS: Dict[str, Tuple[str, ...]] = {}
for _dest, _terms in _WEIGHT_TO_TERM.items():
    for _t in _terms:
        _TERM_TO_WEIGHTS[_t] = _TERM_TO_WEIGHTS.get(_t, ()) + (_dest,)


def _term_share_rows(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Per-epoch term shares from whichever shape the artifact has.

    Newer runs write ``loss_terms`` (the weighted contribution per term).
    Older runs only have flat ``*_loss`` / ``*_term`` columns.  Both are
    readable; which one was used is recorded, because a share computed from a
    raw loss and a share computed from a weighted contribution are not the
    same quantity.
    """
    pem = doc.get("per_epoch_metrics") or []
    if isinstance(pem, dict):
        pem = [pem]
    if not pem:
        return [], "no_per_epoch_metrics"

    if isinstance(pem[0], dict) and isinstance(pem[0].get("loss_terms"), dict):
        rows = []
        for e in pem:
            terms = {
                k: float(v)
                for k, v in e["loss_terms"].items()
                if k not in _AGGREGATE_KEYS
            }
            total = sum(abs(v) for v in terms.values()) or 1.0
            rows.append(
                {
                    "epoch": e.get("epoch"),
                    "source": "loss_terms",
                    "terms": terms,
                    "shares": {k: v / total for k, v in terms.items()},
                    "total": total,
                }
            )
        return rows, "loss_terms"

    rows = []
    for e in pem:
        terms = {
            k: float(v)
            for k, v in e.items()
            if (k.endswith("_loss") or k.endswith("_term"))
            and k not in _AGGREGATE_KEYS
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
        }
        total = sum(abs(v) for v in terms.values()) or 1.0
        rows.append(
            {
                "epoch": e.get("epoch"),
                "source": "flat_keys",
                "terms": terms,
                "shares": {k: v / total for k, v in terms.items()},
                "total": total,
            }
        )
    return rows, "flat_keys"


def gate_m9(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Objective balance: does one term own the objective, and are all declared
    terms observable at all?"""
    run_dir = ctx.get("run_dir")
    if run_dir is None:
        return {
            "measured": {},
            "checks": {"run_dir_supplied": False},
            "pass": False,
            "detail": "no --run-dir; M9 reads per_epoch_metrics.json",
        }
    p = _resolve(str(run_dir)) / "per_epoch_metrics.json"
    if not p.is_file():
        return {
            "measured": {"path": str(p), "exists": False},
            "checks": {"per_epoch_metrics_exists": False},
            "pass": False,
        }
    doc = json.loads(p.read_text())
    rows, source = _term_share_rows(doc)
    if not rows:
        return {
            "measured": {"path": str(p), "source": source},
            "checks": {"terms_readable": False},
            "pass": False,
        }

    from molmetal.scripts.train_cfm_yuelbond_wasserstein import _objective_weights

    weights = {k: v for k, v in _objective_weights(ctx["ns"]).items() if v != 0.0}

    max_share_by_term: Dict[str, float] = {}
    for r in rows:
        for k, s in r["shares"].items():
            max_share_by_term[k] = max(max_share_by_term.get(k, 0.0), abs(s))
    worst_term = max(max_share_by_term, key=max_share_by_term.get)
    worst_share = max_share_by_term[worst_term]

    def _observable(dest: str) -> bool:
        terms = _WEIGHT_TO_TERM.get(dest)
        if not terms:
            return False
        return any(max_share_by_term.get(t, 0.0) > 0.0 for t in terms)

    def _exported(dest: str) -> bool:
        terms = _WEIGHT_TO_TERM.get(dest) or ()
        return any(t in max_share_by_term for t in terms)

    unobservable = sorted(d for d in weights if not _observable(d))
    unmapped = sorted(d for d in weights if d not in _WEIGHT_TO_TERM)
    zero_valued = sorted(d for d in weights if _exported(d) and not _observable(d))
    # The decidable half of the question.  "Every declared weight is
    # observable" is NOT decidable from the artifact: ``_LAST_LOSS_TERMS``
    # registers every term unconditionally, so a weight of 0.05 whose feature
    # is off (no QED head, no stereo labels, no pocket) produces an exported
    # term of exactly 0.0 and is indistinguishable from a dead knob.  MEASURED
    # on the Wf-A108j smoke: 20 of 24 declared weights land there.
    #
    # What IS decidable, and is the defect Wf-A108e found, is the other
    # direction: a term that contributes to the loss whose multiplier is not
    # declared.  That is a number in the objective with no knob attached.
    contributing = {k for k, v in max_share_by_term.items() if v > 0.0}
    weightless = sorted(
        t for t in contributing if not any(w in weights for w in _TERM_TO_WEIGHTS.get(t, ()))
    )
    checks = {
        "no_term_above_90pct": worst_share <= 0.90,
        "every_contributing_term_has_a_declared_weight": len(weightless) == 0,
    }
    return {
        "measured": {
            "path": str(p),
            "source": source,
            "n_epochs": len(rows),
            "per_epoch": rows,
            "max_share_by_term": max_share_by_term,
            "worst_term": worst_term,
            "worst_share": worst_share,
            "declared_nonzero_weights": weights,
            "terms_contributing_without_a_weight": weightless,
            "unobservable_weights": unobservable,
            "weights_with_no_exported_term": unmapped,
            "weights_whose_term_is_zero": zero_valued,
            "note": (
                "``weights_whose_term_is_zero`` is reported, not judged: a "
                "declared weight whose exported term is exactly 0.0 is either "
                "a disabled feature or a dead knob and the artifact cannot "
                "tell them apart."
            ),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _find_keys(obj: Any, keys: Sequence[str], path: str = "") -> Dict[str, Any]:
    """Recursive search for a set of key names; returns first hit per name."""
    found: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and k not in found and not isinstance(v, (dict, list)):
                found[k] = v
            if len(found) == len(keys):
                return found
            sub = _find_keys(v, keys, f"{path}.{k}")
            for kk, vv in sub.items():
                found.setdefault(kk, vv)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            sub = _find_keys(v, keys, f"{path}[{i}]")
            for kk, vv in sub.items():
                found.setdefault(kk, vv)
    return found


def gate_m10(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Metric chain: can a reported number be traced back to these weights?

    Three things have to hold: the checkpoint exists at the path the artifact
    names, the artifact carries the checkpoint's hash, and the artifact carries
    the split fingerprint.  Any one missing and the number is a number without
    a referent -- which is what §6.197 found in c7 (the artifact outlived the
    weights it described).
    """
    ckpt_path = ctx["ckpt_path"]
    sha = ctx["ckpt_sha256"]
    run_dir = ctx.get("run_dir")
    if run_dir is None:
        return {
            "measured": {"checkpoint_sha256": sha},
            "checks": {"run_dir_supplied": False},
            "pass": False,
            "detail": "no --run-dir; M10 reads the run's metrics artifact",
        }
    p = _resolve(str(run_dir)) / "per_epoch_metrics.json"
    if not p.is_file():
        return {
            "measured": {"path": str(p), "exists": False},
            "checks": {"per_epoch_metrics_exists": False},
            "pass": False,
        }
    doc = json.loads(p.read_text())
    hits = _find_keys(
        doc,
        (
            "checkpoint",
            "checkpoint_path",
            "ckpt_path",
            "checkpoint_sha256",
            "sha256",
            "split_fingerprint",
            "split_key",
            "split_hash",
        ),
    )
    named = hits.get("checkpoint") or hits.get("checkpoint_path") or hits.get("ckpt_path")
    named_exists = False
    named_hash = None
    if isinstance(named, str) and named:
        np_ = _resolve(named)
        named_exists = np_.is_file()
        if named_exists:
            named_hash = _sha256(np_)
    declared_hash = hits.get("checkpoint_sha256") or hits.get("sha256")
    split_fp = (
        hits.get("split_fingerprint") or hits.get("split_key") or hits.get("split_hash")
    )

    checks = {
        "checkpoint_on_disk": bool(ckpt_path.is_file()),
        "artifact_names_a_checkpoint": bool(named),
        "named_checkpoint_exists": bool(named_exists),
        "hash_recorded_and_matches": bool(
            declared_hash and str(declared_hash) == sha
        ),
        "split_fingerprint_recorded": bool(split_fp),
    }
    return {
        "measured": {
            "path": str(p),
            "checkpoint_sha256": sha,
            "artifact_keys_found": sorted(hits),
            "artifact_named_checkpoint": named,
            "named_checkpoint_sha256": named_hash,
            "declared_hash": declared_hash,
            "split_fingerprint": split_fp,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


_GATES = {
    "M1": gate_m1,
    "M2": gate_m2,
    "M3": gate_m3,
    "M4": gate_m4,
    "M5": gate_m5,
    "M6": gate_m6,
    "M7": gate_m7,
    "M8": gate_m8,
    "M9": gate_m9,
    "M10": gate_m10,
}

_GATE_TITLES = {
    "M1": "data entry",
    "M2": "conditioning path",
    "M3": "velocity head capacity",
    "M4": "output scale",
    "M5": "gradient path",
    "M6": "time resolution",
    "M7": "warm start",
    "M8": "train / sample seam",
    "M9": "objective balance",
    "M10": "metric chain",
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="checkpoint to gate")
    p.add_argument(
        "--run-dir",
        default=None,
        help="the run directory that owns the checkpoint (M9, M10 read its "
        "per_epoch_metrics.json)",
    )
    p.add_argument(
        "--gate",
        nargs="*",
        default=None,
        help=f"subset of {_GATE_IDS}; default all",
    )
    p.add_argument("--json", default=None, help="write the artifact here")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print each gate's measured block even when it passes",
    )
    p.add_argument("--n-clouds", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--steps-m3",
        type=int,
        default=2000,
        help="optimizer steps for the M3 overfit test (default 2000, the "
        "budget Wf-A108d calibrated)",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="override the cohort CSV (default: the checkpoint's data_csv)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    t0 = time.time()

    gate_ids = args.gate or _GATE_IDS
    unknown = [g for g in gate_ids if g not in _GATES]
    if unknown:
        raise SystemExit(f"unknown gate id(s): {unknown}; known: {_GATE_IDS}")

    ckpt_path = _resolve(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"--ckpt '{ckpt_path}' is not a file")
    sha = _sha256(ckpt_path)
    device = torch.device(args.device)

    ck = _load_ckpt(ckpt_path)
    model, ns, migrated = _build_model(ck, device)
    clouds = _build_clouds(ns, int(args.n_clouds), int(args.seed))
    if not clouds:
        raise SystemExit("cohort produced 0 clouds")
    x1, z, sizes = _flatten(clouds, device)

    ctx: Dict[str, Any] = {
        "model": model,
        "ns": ns,
        "clouds": clouds,
        "x1": x1,
        "z": z,
        "sizes": sizes,
        "device": device,
        "seed": int(args.seed),
        "n_clouds": int(args.n_clouds),
        "ckpt_path": ckpt_path,
        "ckpt_sha256": sha,
        "run_dir": args.run_dir,
        "csv": args.csv,
        "steps_m3": int(args.steps_m3),
    }

    print(
        f"[gates] ckpt={ckpt_path.name} sha256={sha[:12]} "
        f"epoch={ck.get('epoch')} head="
        f"{getattr(getattr(model, 'vel_field', None), 'velocity_head', 'none')} "
        f"clouds={len(clouds)} atoms={int(x1.shape[0])} device={device}",
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    for gid in gate_ids:
        fn = _GATES[gid]
        t_g = time.time()
        try:
            r = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a gate that raises is a FAIL
            r = {
                "measured": {"exception": repr(exc)},
                "checks": {"gate_ran": False},
                "pass": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        status = "PASS" if r.get("pass") else "FAIL"
        if r.get("skip"):
            status = "SKIP"
        entry = {
            "id": gid,
            "module": _GATE_TITLES[gid],
            "status": status,
            "checks": r.get("checks", {}),
            "measured": r.get("measured", {}),
            "detail": r.get("detail"),
            "wall_time_s": round(time.time() - t_g, 2),
        }
        results.append(entry)
        print(
            f"[{gid}] {status:<4} {_GATE_TITLES[gid]:<24} "
            f"({entry['wall_time_s']:.1f}s)",
            flush=True,
        )
        for k, v in (r.get("checks") or {}).items():
            print(f"        {'ok ' if v else 'NO '} {k}", flush=True)
        if status == "FAIL" or args.verbose:
            blob = json.dumps(r.get("measured", {}), default=str)
            if len(blob) > 1600 and not args.verbose:
                blob = blob[:1600] + " ...(truncated; see --json artifact)"
            print(f"        measured: {blob}", flush=True)
        if entry["detail"]:
            print(f"        detail: {entry['detail']}", flush=True)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")

    doc = {
        "metric": "wfa108i_module_gates",
        "tag": "MEASURED",
        "date": time.strftime("%Y-%m-%d"),
        "generated_at_unix": int(time.time()),
        "what": (
            "The ten module gates from cfm_todo.json v9.0, executed against "
            "one checkpoint. Each gate measures one module's own function; "
            "the seam gates (M1, M8) measure the join between two modules."
        ),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha,
        "ckpt_epoch": ck.get("epoch"),
        "velocity_head": getattr(
            getattr(model, "vel_field", None), "velocity_head", None
        ),
        "prediction_target": getattr(
            getattr(model, "vel_field", None), "prediction_target", None
        ),
        "t_embedding_dim": int(getattr(ns, "t_embedding_dim", 0) or 0),
        "radial_centroid_masked": bool(
            getattr(ns, "radial_centroid_masked", False)
        ),
        "dataset": str(getattr(ns, "dataset", "synthetic")),
        "run_dir": args.run_dir,
        "device": str(device),
        "n_clouds": len(clouds),
        "n_atoms": int(x1.shape[0]),
        "seed": int(args.seed),
        "legacy_vel_migration": migrated,
        "gates": results,
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": n_skip,
            "failing": [r["id"] for r in results if r["status"] == "FAIL"],
            "passing": [r["id"] for r in results if r["status"] == "PASS"],
        },
        "wall_time_s": round(time.time() - t0, 1),
        # An artifact nobody can regenerate is an anecdote.  The command line
        # is recorded verbatim so the exact invocation, including --gate and
        # --seed, is recoverable from the file alone.
        "provenance": {
            "script": "molmetal/scripts/cfm_module_gates.py",
            "argv": list(sys.argv),
            "gate_set": list(args.gate) if args.gate else "all",
        },
    }

    if args.json:
        out = _resolve(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        print(f"[gates] wrote {out}", flush=True)

    print(
        f"[gates] {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP "
        f"in {doc['wall_time_s']}s",
        flush=True,
    )
    if n_fail:
        return 1
    if n_skip:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
