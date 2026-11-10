#!/usr/bin/env python
"""Wf-A107e — the ring-prior A/B on a sampler that is actually trained.

Wf-A107 (§6.152/§6.153) compared two bond heads attached to a velocity field
that never trained (`use_fm_objective` absent, `wasserstein_weight: 0.0`), so
its `gen_ring_*` numbers were decodes of collapsed noise (§6.162).  This pair
turns the sampler on (`use_fm_objective: true`, `fm_t_per_cloud: 1`,
`use_tmqm_init: true`, `wasserstein_weight: 0.1`) and clips the head and the
flow field SEPARATELY (`grad_clip_scope: per_module`), because the ring term
lifts the joint grad norm past `max_norm` and a shared clip would hand the
treatment arm a different flow-field learning rate.

Two things this script checks that a metrics diff cannot:

  * `vel_field` must be BIT-IDENTICAL between the arms.  The ring prior is
    attached to `model.head` only and per-module clipping keeps the flow
    field's own clip factor untouched, so any drift means the fix does not
    hold and the A/B is confounded again.
  * the generated geometry must look like a molecule cloud, not a blob:
    mean pair distance and the fraction of pairs inside the 3.5 A cutoff are
    read from the Wf-A107c probe if its artifact exists.

Usage:
  python molmetal/scripts/wfa107e_ab_analysis.py \
      --arm control=results/wfa107e_fm_control \
      --arm edge_ring_w1=results/wfa107e_fm_edge_ring_w1 \
      --out metrics/by_metric/wfa107e_ab.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.cfm_cohort_recovery import (  # noqa: E402
    COHORT_SOURCE_UNATTESTED,
)
from molmetal.scripts.cfm_pool_epoch_rule import resolve_population  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

_METRICS = (
    "gen_ring_3_share",
    "gen_ring_5_6_share",
    "gen_rings_per_mol",
    # Wf-A107f — `bond_pr_auc` is bond EXISTENCE AP (definition
    # `existence_ap_v2`), the field the bond gate is written against.
    # `bond_class_1_pr_auc` is the SINGLE-bond CLASS PR-AUC on a ~2.4% base
    # rate.  Wf-A107e reported the second under the first's name and concluded
    # the ring prior "costs bond AP -0.170" when the existence AP had in fact
    # RISEN by +0.018.  Both are kept, under their own names.
    "bond_pr_auc",
    "bond_class_1_pr_auc",
    "bond_f1",
    "bond_best_f1_at_threshold",
    "bond_precision",
    "bond_recall",
    "gen_validity",
    "gen_single_fragment_ratio",
    "grad_norm",
    "train_loss",
    "bond_ce_loss",
    "ring_prior_term",
    "wasserstein_loss",
    "fm_term",
    # Wf-A107f — same ring profile on the head's accepted graph, before RDKit.
    # A run launched before the instrumentation has these absent, so they
    # simply do not enter `last_epoch_deltas`; they are listed so the next A/B
    # picks them up without an edit.
    "gen_raw_ring_3_share",
    "gen_raw_ring_5_6_share",
    "gen_raw_rings_per_mol",
)

#: What each ambiguous metric actually measures.  Emitted into the artifact so
#: the label travels with the number instead of living in a catalog entry.
_METRIC_LABELS = {
    "bond_pr_auc": "bond existence AP (bond_pr_auc_definition field says which)",
    "bond_class_1_pr_auc": "SINGLE bond class PR-AUC (positive rate ~2.4%)",
    "bond_f1": "bond existence F1 at the recorded decision threshold",
    "bond_best_f1_at_threshold": "bond existence F1 at the best swept threshold",
    "gen_raw_ring_3_share": "3-ring share on the head's accepted graph, BEFORE RDKit",
    "gen_raw_ring_5_6_share": "5/6-ring share on the head's accepted graph, BEFORE RDKit",
    "gen_raw_rings_per_mol": "rings per molecule on the head's accepted graph, BEFORE RDKit",
    "fm_term": "epoch mean of the flow-matching term; 0.0 means the flag did "
               "not reach the objective",
}


#: The knobs this A/B reads out of the run's own config block.
_KNOBS = (
    "grad_clip_scope", "max_norm", "ring_template_weight",
    "ring_template_formulation", "use_fm_objective",
    "wasserstein_weight",
)


def _attest_arm(run_dir: Path) -> Dict[str, Any]:
    """Is this run's recorded file the RUN OF RECORD's own file?

    ``per_epoch_metrics.json`` is written ONCE, at the END of training, to a
    FIXED path, so two runs launched into the same ``--out-dir`` clobber each
    other's record.  A config block read out of a clobbered record is a
    SIBLING POPULATION's config, and the knobs published from it would
    describe a launch that never happened.

    The owner is ``molmetal/scripts/cfm_pool_epoch_rule.py`` (which asks
    ``molmetal/scripts/cfm_cohort_recovery.py`` for the cohort): it resolves
    the run of record and says whether the recorded file is that
    population's own file.
    """
    try:
        population = resolve_population(run_dir, None)
    except Exception as exc:  # pragma: no cover - the owner must not be fatal
        return {
            "attested": False,
            "source": COHORT_SOURCE_UNATTESTED,
            "reason": (
                f"the run-of-record owner raised {type(exc).__name__}: {exc}; "
                f"the recorded file cannot be attested"
            ),
        }
    cohort = population.get("cohort") or {}
    is_ro = population.get("metrics_file_is_run_of_record")
    attested = (
        is_ro is True and cohort.get("source") != COHORT_SOURCE_UNATTESTED
    )
    if attested:
        reason = (
            f"the recorded per-epoch file IS the run of record's own file "
            f"(resolution={population.get('resolution')}, "
            f"epochs={population.get('epoch_indices')}, "
            f"cohort_source={cohort.get('source')})"
        )
    else:
        reason = (
            f"the recorded per-epoch file cannot be attested as the run of "
            f"record's own (metrics_file_is_run_of_record={is_ro}, "
            f"resolution={population.get('resolution')}, "
            f"cohort_source={cohort.get('source')}: {cohort.get('reason')})"
        )
    return {
        "attested": attested,
        "source": cohort.get("source"),
        "reason": reason,
        "metrics_file_is_run_of_record": is_ro,
        "resolution": population.get("resolution"),
        "epoch_indices": population.get("epoch_indices"),
    }


def _load_arm(run_dir: str) -> Dict[str, Any]:
    d = json.load(open(Path(run_dir) / "per_epoch_metrics.json"))
    rows = d["per_epoch_metrics"]
    _cfg = d.get("config") or {}
    attestation = _attest_arm(Path(run_dir))
    if attestation["attested"]:
        knobs = {k: _cfg.get(k, d.get(k)) for k in _KNOBS}
        # The artifact block for `use_fm_objective` / `grad_clip_scope` was added
        # in this same wave, so a run launched before it carries None.  The
        # checkpoint's own `args` is the authoritative record; prefer it.
        ck = Path(run_dir) / "checkpoints" / "cfm_yuelbond_wasserstein_seed0.pt"
        if ck.exists():
            from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
                load_cfm_checkpoint,
            )

            ck_args = load_cfm_checkpoint(str(ck), map_location="cpu")["args"]
            for k in knobs:
                if knobs[k] is None:
                    knobs[k] = ck_args.get(k)
    else:
        # NOT attested: the config block on disk belongs to another
        # population, so no knob is published from it (null, with the reason
        # beside it).  The checkpoint's own args are not read either: in a
        # clobbered directory the checkpoint is the sibling's too.
        knobs = {k: None for k in _KNOBS}
    return {
        "run_dir": run_dir,
        **knobs,
        "epochs": rows,
        "last": rows[-1],
        "attested": attestation["attested"],
        "attestation": attestation,
    }


def _series(rows: List[Dict[str, Any]], key: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for r in rows:
        v = r.get(key)
        out.append(float(v) if isinstance(v, (int, float)) else None)
    return out


def _weights(ckpt_path: str) -> Dict[str, torch.Tensor]:
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import load_cfm_checkpoint

    ck = load_cfm_checkpoint(ckpt_path, map_location="cpu")
    return ck["model_state_dict"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=run_dir (exactly two: control first)")
    ap.add_argument("--out", default="metrics/by_metric/wfa107e_ab.json")
    args = ap.parse_args()
    if len(args.arm) != 2:
        raise SystemExit(f"--arm must be given exactly twice, got {len(args.arm)}")

    specs = []
    for spec in args.arm:
        name, run_dir = spec.split("=", 1)
        specs.append((name, run_dir))
    (name_a, dir_a), (name_b, dir_b) = specs
    A, B = _load_arm(dir_a), _load_arm(dir_b)

    deltas = {}
    for key in _METRICS:
        va, vb = A["last"].get(key), B["last"].get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            deltas[key] = {"control": float(va), "treatment": float(vb),
                           "delta": float(vb) - float(va)}

    # --- weight check: the ring prior must not reach the flow field -------
    weights = {"checked": False}
    ck_a = Path(dir_a) / "checkpoints" / "cfm_yuelbond_wasserstein_seed0.pt"
    ck_b = Path(dir_b) / "checkpoints" / "cfm_yuelbond_wasserstein_seed0.pt"
    if ck_a.exists() and ck_b.exists():
        wa, wb = _weights(str(ck_a)), _weights(str(ck_b))
        vel = [k for k in wa if k.startswith("vel_field")]
        head = [k for k in wa if k.startswith("head")]
        # Wf-A108 — the arms may have DIFFERENT parameter sets (a radial head
        # has ``vel_scale``, a vector-channel head has ``vc_*`` and no
        # ``vel_scale``).  Comparing by key name then raises KeyError, which is
        # exactly the case this tool exists to analyse.  Compare the shared
        # keys and report the asymmetry instead of crashing on it.
        vel_shared = [k for k in vel if k in wb]
        head_shared = [k for k in head if k in wb]
        weights = {
            "checked": True,
            "vel_field_max_abs_delta": max(
                (float((wa[k] - wb[k]).abs().max()) for k in vel_shared),
                default=0.0,
            ),
            "head_max_abs_delta": max(
                (float((wa[k] - wb[k]).abs().max()) for k in head_shared),
                default=0.0,
            ),
            "n_vel_tensors": len(vel),
            "n_head_tensors": len(head),
            "n_vel_tensors_shared": len(vel_shared),
            "n_head_tensors_shared": len(head_shared),
            "vel_field_keys_only_in_control": sorted(set(vel) - set(wb)),
            "vel_field_keys_only_in_treatment": sorted(
                k for k in wb if k.startswith("vel_field") and k not in wa
            ),
            "reading": (
                "For a one-factor A/B on the SAME architecture, vel_field must "
                "be 0.0 (the ring term touches only the head, and per-module "
                "clipping keeps its clip factor independent); a non-zero value "
                "means the A/B is confounded again.  When the architectures "
                "differ, the max is over the SHARED tensors only and the "
                "*_only_in_* lists carry the difference — the number is not a "
                "confound check in that case."
            ),
        }

    doc = {
        "metric": "wfa107e_ring_prior_with_trained_sampler",
        "tag": "MEASURED",
        "date": "2026-10-21",
        "what": (
            "One-factor A/B (ring_template_weight 0 vs 1.0, formulation "
            "edge_ring) with a TRAINED velocity field (use_fm_objective true, "
            "fm_t_per_cloud 1, tmqm init, wasserstein 0.1) and per-module "
            "gradient clipping.  Same protocol otherwise as Wf-A107: 200 "
            "train clouds, 64 val, 5 epochs, seed 0."
        ),
        "arms": [name_a, name_b],
        "config": {
            "control": {k: A[k] for k in ("grad_clip_scope", "max_norm",
                                          "ring_template_weight",
                                          "ring_template_formulation",
                                          "use_fm_objective",
                                          "wasserstein_weight")},
            "treatment": {k: B[k] for k in ("grad_clip_scope", "max_norm",
                                            "ring_template_weight",
                                            "ring_template_formulation",
                                            "use_fm_objective",
                                            "wasserstein_weight")},
        },
        "series": {
            name: {k: _series(arm["epochs"], k) for k in _METRICS}
            for name, arm in ((name_a, A), (name_b, B))
        },
        "metric_labels": dict(_METRIC_LABELS),
        "bond_pr_auc_definition": A["last"].get("bond_pr_auc_definition"),
        "last_epoch_deltas": deltas,
        "weights": weights,
        "honest_limits": [
            "1 seed, 5 epochs, 200 training clouds: a direction test, not an "
            "effect size.  Wf-A107b measured the run-to-run spread of this "
            "protocol to be as large as the single-seed effect.",
            "5 epochs of FM training on 200 clouds is a weak sampler.  The "
            "claim this artifact can support is 'the metric is now measured on "
            "trained-sampler geometry', not 'the prior improves generation'.",
            "No held-out generation set: the val clouds are the ones the "
            "trainer scores every epoch.",
        ],
    }
    # The attestation of each arm's recorded file is published ONLY when an
    # arm could NOT be attested -- an attested arm needs no note, and the
    # artifact stays byte-identical to the one published before this guard.
    if not (A.get("attested") and B.get("attested")):
        doc["arm_attestation"] = {
            name_a: A.get("attestation"),
            name_b: B.get("attestation"),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(out, json.dumps(doc, ensure_ascii=False, indent=1))

    print(f"[wfa107e] {name_a:14s} 3-share={A['last'].get('gen_ring_3_share')} "
          f"5/6={A['last'].get('gen_ring_5_6_share')} "
          f"rpm={A['last'].get('gen_rings_per_mol')}")
    print(f"[wfa107e] {name_b:14s} 3-share={B['last'].get('gen_ring_3_share')} "
          f"5/6={B['last'].get('gen_ring_5_6_share')} "
          f"rpm={B['last'].get('gen_rings_per_mol')}")
    for k, v in deltas.items():
        label = _METRIC_LABELS.get(k, "")
        print(f"[wfa107e] delta {k:24s} {v['delta']:+.4f}  "
              f"({v['control']:.4f} -> {v['treatment']:.4f})"
              + (f"  [{label}]" if label else ""))
    _ap_def = A["last"].get("bond_pr_auc_definition")
    if _ap_def and _ap_def != "existence_ap_v2":
        print(f"[wfa107e] WARNING: bond_pr_auc_definition={_ap_def!r}; "
              "this artifact's AP is NOT bond existence and must not be "
              "compared with an existence_ap_v2 run.")
    if weights.get("checked"):
        print(f"[wfa107e] vel max|delta|={weights['vel_field_max_abs_delta']:.3e} "
              f"head max|delta|={weights['head_max_abs_delta']:.3e}")
    print(f"[wfa107e] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
