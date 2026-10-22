#!/usr/bin/env python3
"""UP-18 — is the paired-evidence graph a graph, and which edges carry level information?

=====================================================================
Why this exists
=====================================================================
UP-10 counted paired units per metal pair and cut the edge list at
``n_groups >= 20``. That rule marked Ru-Rh (15 units, mean Delta 1.077)
UNSUPPORTED and Ir-Rh (170 units, mean Delta -0.046) OK. Group count is
not precision: Ru-Rh has the largest effect in the corpus and Ir-Rh has
an effect indistinguishable from zero.

This script replaces the count rule with two measurements:

  * the standard error of each edge's mean Delta, computed three ways
    (assay noise only / empirical between-unit / ligand-clustered), and
  * the graph those edges form, which decides what can be identified at
    all under leave-one-metal-out.

=====================================================================
The three standard errors
=====================================================================
    se_noise   = sqrt(sigma2_a + sigma2_b) / sqrt(n)
                 Assumes every unit is an independent draw whose only
                 variance is assay noise. This is the definition behind
                 the t values quoted in the round-three plan
                 (Ru-Os 11.14, Ru-Rh 9.54, Ir-Rh 1.50).

    se_emp     = sd(Delta) / sqrt(n)
                 Uses the observed spread of Delta across units, which
                 includes the ligand x metal interaction. This is the
                 sampling variance of the mean, and it is the one that
                 matters for a level estimate.

    se_cluster = sd(Delta) / sqrt(n) * sqrt(deff),
                 deff = 1 + (mean_units_per_ligand - 1) * ICC(1)
                 Units are clustered by ligand: one ligand measured in
                 five cell lines contributes five correlated Deltas.
                 Also reported as a 2000-draw ligand-clustered bootstrap,
                 which makes no exchangeability assumption.

The point estimates are unaffected by the choice; only the SEs are.

=====================================================================
What a tree implies
=====================================================================
On a connected graph with V nodes and E = V - 1 edges, the V-1 edge
equations plus the constraint sum(u) = 0 form a square system. The level
solution is exact, the residual degrees of freedom are zero, and there
is no cycle, so node-splitting (the standard inconsistency test in
network meta-analysis) has nothing to split. Precision then affects only
the standard errors of the levels, never the levels themselves.

A held-out node is isolated by construction -- all of its edges are
removed with it -- so metal_out can never identify the held-out level.
What varies across the five scenarios is how much of the *training*
level system remains identifiable.

=====================================================================
Gate (todo.json, frozen 2026-10-22T12:42+08:00)
=====================================================================
Diagnostic package, no numeric threshold for pass/fail of the science.
Four assertions are evaluated and reported individually; the kill line is
"cluster-corrected t(Ru-Rh) < 2 retracts the two-effective-edges claim".

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_metal_graph.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import (  # noqa: E402
    ENDPOINTS,
    METALS,
    _git_commit,
    _sha256,
)
from molmetal.scripts.msrs_support_audit import (  # noqa: E402
    MIN_GROUPS,
    load_long,
    metal_noise,
    unit_table,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 20261022
GATE_FROZEN_AT = "2026-10-22T12:42+08:00"


def _icc1(values: np.ndarray, groups: np.ndarray) -> Tuple[float, float, float]:
    """One-way random-effects ICC(1), the mean cluster size, and the F ratio."""
    d = pd.DataFrame({"v": values, "g": groups})
    n_tot = len(d)
    k = d["g"].nunique()
    if k < 2 or n_tot <= k:
        return float("nan"), float("nan"), float("nan")
    gm = d.groupby("g")["v"]
    ni = gm.size()
    mi = gm.mean()
    mbar = float(ni.mean())
    ssb = float((ni * (mi - d["v"].mean()) ** 2).sum())
    ssw = float(((d["v"] - gm.transform("mean")) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n_tot - k)
    if msw <= 0 or msb + (mbar - 1) * msw <= 0:
        return 0.0, mbar, float("nan")
    return float((msb - msw) / (msb + (mbar - 1) * msw)), mbar, float(msb / msw)


def _cluster_bootstrap_mean(
    delta: np.ndarray, ligands: np.ndarray, n_boot: int, seed: int
) -> Dict[str, object]:
    """Resample ligands with replacement; recompute the mean Delta."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ligands)
    idx = {u: np.flatnonzero(ligands == u) for u in uniq}
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws[b] = delta[np.concatenate([idx[u] for u in pick])].mean()
    return {
        "n_boot": n_boot,
        "seed": seed,
        "se": float(draws.std(ddof=1)),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def edge_table(
    units: pd.DataFrame, noise: Dict[str, Dict[str, Optional[float]]]
) -> List[Dict[str, object]]:
    """Every metal pair with at least one cell-matched unit."""
    rows: List[Dict[str, object]] = []
    for m1, m2 in itertools.combinations(METALS, 2):
        a = units[units["metal"] == m1].rename(columns={"mean": "y1", "size": "n1"})
        b = units[units["metal"] == m2].rename(columns={"mean": "y2", "size": "n2"})
        j = a.merge(b, on=["ligand_canonical", "cell_line_key"], how="inner")
        if j.empty:
            continue
        delta = (j["y1"] - j["y2"]).to_numpy(dtype=float)
        n = int(len(delta))
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1)) if n > 1 else float("nan")

        s2 = float((noise.get(m1, {}).get("sigma2") or 0.0)
                   + (noise.get(m2, {}).get("sigma2") or 0.0))
        se_noise = float(np.sqrt(s2) / np.sqrt(n)) if s2 > 0 else float("nan")
        se_emp = float(sd / np.sqrt(n)) if np.isfinite(sd) else float("nan")

        rho, mbar, f_ratio = _icc1(delta, j["ligand_canonical"].to_numpy())
        deff = float(1.0 + (mbar - 1.0) * rho) if np.isfinite(rho) else 1.0
        se_deff = float(se_emp * np.sqrt(max(deff, 1.0))) if np.isfinite(se_emp) else float("nan")

        boot = _cluster_bootstrap_mean(delta, j["ligand_canonical"].to_numpy(), N_BOOT, BOOT_SEED)
        se_cluster = float(boot["se"])
        lo, hi = boot["ci95"]

        def _t(se):
            return round(abs(mean) / se, 3) if np.isfinite(se) and se > 0 else None

        rows.append({
            "pair": f"{m1}-{m2}",
            "n_groups": n,
            "n_ligands": int(j["ligand_canonical"].nunique()),
            "n_cells": int(j["cell_line_key"].nunique()),
            "mean_delta": round(mean, 4),
            "sd_delta": round(sd, 4) if np.isfinite(sd) else None,
            "noise_sd_delta": round(float(np.sqrt(s2)), 4) if s2 > 0 else None,
            "se_noise": round(se_noise, 6) if np.isfinite(se_noise) else None,
            "t_noise": _t(se_noise),
            "se_empirical": round(se_emp, 6) if np.isfinite(se_emp) else None,
            "t_empirical": _t(se_emp),
            "icc_ligand": round(rho, 4) if np.isfinite(rho) else None,
            "icc_f_ratio": round(f_ratio, 3) if np.isfinite(f_ratio) else None,
            "mean_units_per_ligand": round(mbar, 3) if np.isfinite(mbar) else None,
            "design_effect": round(deff, 4),
            "se_cluster_deff": round(se_deff, 6) if np.isfinite(se_deff) else None,
            "t_cluster_deff": _t(se_deff),
            "se_cluster_bootstrap": round(se_cluster, 6),
            "t_cluster": _t(se_cluster),
            "mean_delta_ci95_cluster": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "status_up10": "OK" if n >= MIN_GROUPS else "UNSUPPORTED",
        })

    total_c = sum(r["se_cluster_bootstrap"] ** -2 for r in rows)
    total_n = sum(r["se_noise"] ** -2 for r in rows if r["se_noise"])
    for r in rows:
        pc = r["se_cluster_bootstrap"] ** -2
        pn = r["se_noise"] ** -2 if r["se_noise"] else 0.0
        r["precision_cluster"] = round(pc, 3)
        r["precision_share_cluster"] = round(pc / total_c, 4)
        r["precision_noise"] = round(pn, 3)
        r["precision_share_noise"] = round(pn / total_n, 4)
    return rows


def _components(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> List[List[str]]:
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set = set()
    out: List[List[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(sorted(adj[x] - seen))
        out.append(sorted(comp))
    return out


def _has_cycle(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> bool:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


def graph_diagnostics(edges: List[Dict[str, object]]) -> Dict[str, object]:
    pairs = [tuple(sorted(r["pair"].split("-"))) for r in edges]
    nodes_all = list(METALS)
    nodes_obs = sorted({m for e in pairs for m in e})
    isolated = [m for m in nodes_all if m not in nodes_obs]

    deg = {m: 0 for m in nodes_all}
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1

    comps_all = _components(nodes_all, pairs)
    comps_obs = _components(nodes_obs, pairs)

    cut_vertices = []
    for v in nodes_obs:
        before = len(_components(nodes_obs, pairs))
        rest_nodes = [n for n in nodes_obs if n != v]
        rest_edges = [e for e in pairs if v not in e]
        after = len(_components(rest_nodes, rest_edges))
        if after > before:
            cut_vertices.append(v)

    per_out = {}
    for held in nodes_all:
        train = [m for m in nodes_all if m != held]
        train_edges = [e for e in pairs if held not in e]
        comps = _components(train, train_edges)
        # V - #components is the dimension of the identifiable level subspace.
        per_out[held] = {
            "train_metals": train,
            "n_train_edges": len(train_edges),
            "train_edges": ["-".join(e) for e in train_edges],
            "n_train_components": len(comps),
            "train_components": comps,
            "identifiable_level_dim_train": len(train) - len(comps),
            "held_out_degree": deg[held],
            "held_out_connected_to_train": False,
        }

    return {
        "nodes_all_metals": nodes_all,
        "nodes_with_edges": nodes_obs,
        "isolated_metals": isolated,
        "n_nodes_all": len(nodes_all),
        "n_nodes_with_edges": len(nodes_obs),
        "n_edges": len(pairs),
        "edges": ["-".join(e) for e in pairs],
        "degrees": deg,
        "components_all_metals": comps_all,
        "components_observed_subgraph": comps_obs,
        "n_components_all_metals": len(comps_all),
        "is_forest_all_metals": len(comps_all) + len(pairs) == len(nodes_all),
        "is_tree_observed_subgraph": (
            len(comps_obs) == 1 and len(pairs) == len(nodes_obs) - 1
        ),
        "has_cycle": _has_cycle(nodes_all, pairs),
        "cut_vertices_observed_subgraph": cut_vertices,
        "identifiable_level_dim_full": len(nodes_obs) - len(comps_obs),
        "null_space_dim_full": len(nodes_obs) - (len(nodes_obs) - len(comps_obs)),
        "per_metal_out": per_out,
    }


def gls_levels(edges: List[Dict[str, object]], se_key: str) -> Dict[str, object]:
    """Weighted levels with sum(u) = 0. On a tree the weights do not move the point estimate."""
    nodes = sorted({m for r in edges for m in r["pair"].split("-")})
    idx = {m: i for i, m in enumerate(nodes)}
    A = np.zeros((len(edges), len(nodes)))
    d = np.zeros(len(edges))
    w = np.zeros(len(edges))
    for k, r in enumerate(edges):
        a, b = r["pair"].split("-")
        A[k, idx[a]] = 1.0
        A[k, idx[b]] = -1.0
        d[k] = float(r["mean_delta"])
        se = float(r[se_key])
        w[k] = 1.0 / se ** 2
    constraint = np.ones((1, len(nodes)))
    Aw = np.vstack([A * np.sqrt(w)[:, None], constraint])
    dw = np.concatenate([d * np.sqrt(w), np.zeros(1)])
    u, _, rank, _ = np.linalg.lstsq(Aw, dw, rcond=None)
    cov = np.linalg.pinv(Aw.T @ Aw)
    se_u = np.sqrt(np.diag(cov))
    return {
        "weights": se_key,
        "levels": {m: round(float(u[i]), 4) for i, m in enumerate(nodes)},
        "se": {m: round(float(se_u[i]), 4) for i, m in enumerate(nodes)},
        "t": {m: round(float(abs(u[i]) / se_u[i]), 3) if se_u[i] > 0 else None
              for i, m in enumerate(nodes)},
        "rank_edge_design": int(np.linalg.matrix_rank(A)),
        "rank_with_constraint": int(rank),
        "n_levels": len(nodes),
        "residual_df": int(len(edges) + 1 - rank),
        "identifiable_subspace_dim": int(np.linalg.matrix_rank(A)),
        "note": "On a tree the system is square, so the point estimates are identical "
                "under every weighting; only the SEs change.",
    }


def node_split(graph: Dict[str, object]) -> Dict[str, object]:
    if graph["has_cycle"]:
        return {"status": "DEFINED", "note": "A cycle exists; direct and indirect estimates can be compared."}
    return {
        "status": "NOT_APPLICABLE",
        "reason": "The observed-subgraph is a tree (E = V - 1, no cycle). Every level "
                  "difference is informed by exactly one edge, so there is no indirect "
                  "path to compare against a direct one. Inconsistency is not testable, "
                  "not absent.",
        "residual_df": 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet")
    ap.add_argument("--endpoint", default="potency", choices=list(ENDPOINTS))
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_graph.json")
    args = ap.parse_args(argv)

    if not args.long.exists():
        print(f"FAIL: missing {args.long}")
        return 1

    df = load_long(args.long)
    sub = df[df["endpoint"] == args.endpoint]
    noise = metal_noise(sub)
    units = unit_table(sub)

    edges = edge_table(units, noise)
    graph = graph_diagnostics(edges)
    gls_cluster = gls_levels(edges, "se_cluster_bootstrap")
    gls_noise = gls_levels(edges, "se_noise")
    split = node_split(graph)

    by_pair = {r["pair"]: r for r in edges}
    ruos = by_pair.get("Ru-Os")
    ruroh = by_pair.get("Ru-Rh")
    irrh = by_pair.get("Ir-Rh")

    assertions = {
        "assert_tree": {
            "pass": bool(graph["is_tree_observed_subgraph"]),
            "value": {
                "n_nodes_with_edges": graph["n_nodes_with_edges"],
                "n_edges": graph["n_edges"],
                "is_tree": graph["is_tree_observed_subgraph"],
                "has_cycle": graph["has_cycle"],
            },
            "definition": "Evaluated on the observed-node subgraph (metals with >=1 edge). "
                          "Including Re the graph is a forest with 2 components, E = V - 2; "
                          "Re has degree 0 in every scenario.",
        },
        "assert_edge_count": {
            "pass": bool(graph["n_edges"] == 3),
            "value": graph["n_edges"],
            "pairs": graph["edges"],
        },
        "assert_ruos_dominant": {
            "threshold": 0.80,
            "value_cluster": ruos["precision_share_cluster"] if ruos else None,
            "value_noise_only": ruos["precision_share_noise"] if ruos else None,
            "pass_cluster": bool(ruos and ruos["precision_share_cluster"] > 0.80),
            "pass_noise_only": bool(ruos and ruos["precision_share_noise"] > 0.80),
            "pass": bool(ruos and ruos["precision_share_cluster"] > 0.80),
            "definition": "Primary = ligand-clustered precision (1/se_cluster^2). The "
                          "noise-only share is reported because the frozen t values in the "
                          "plan were computed from it; it is not the robust one.",
        },
        "assert_ruroh_survives_cluster": {
            "threshold": 3.0,
            "value": ruroh["t_cluster"] if ruroh else None,
            "pass": bool(ruroh and ruroh["t_cluster"] is not None and ruroh["t_cluster"] >= 3.0),
        },
    }
    failing = [k for k, v in assertions.items() if not v["pass"]]

    kill = {
        "rule": "cluster-corrected t(Ru-Rh) < 2 retracts the two-effective-edges claim",
        "value": ruroh["t_cluster"] if ruroh else None,
        "triggered": bool(ruroh and ruroh["t_cluster"] is not None and ruroh["t_cluster"] < 2.0),
    }

    out = {
        "artifact": "msrs_metal_graph",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L0 graph diagnostics (UP-18)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long.relative_to(PROJECT_ROOT)),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "endpoint": args.endpoint,
            "unit": "(ligand_canonical, cell_line_key) measured under >=2 metals",
            "edge_effect": "mean over units of mean(y|a) - mean(y|b)",
            "se_noise": "sqrt(sigma2_a + sigma2_b) / sqrt(n); assay noise only",
            "se_empirical": "sd(Delta) / sqrt(n); includes ligand x metal interaction",
            "se_cluster": "ligand-clustered bootstrap of the mean, "
                          f"{N_BOOT} draws, seed {BOOT_SEED}",
            "design_effect": "1 + (mean_units_per_ligand - 1) * ICC(1) of Delta within ligand",
            "level_model": "weighted least squares on u_a - u_b = mean Delta, sum(u) = 0",
            "gate_frozen_at": GATE_FROZEN_AT,
        },
        "noise_floor_per_metal": noise,
        "edges": edges,
        "edges_by_pair": {r["pair"]: r for r in edges},
        "graph": graph,
        "levels": {
            "cluster_weighted": gls_cluster,
            "noise_weighted": gls_noise,
            "estimates_identical_across_weightings": (
                gls_cluster["levels"] == gls_noise["levels"]
            ),
        },
        "node_split": split,
        "gate": {
            "frozen_at": GATE_FROZEN_AT,
            "assertions": assertions,
            "failing_assertions": failing,
            "gate_pass": bool(not failing),
            "kill": kill,
        },
        "headline": {
            "edge_with_nonzero_effect": [
                r["pair"] for r in edges if r["ci_excludes_zero"]
            ],
            "edge_without_detectable_effect": [
                r["pair"] for r in edges if not r["ci_excludes_zero"]
            ],
            "up10_ok_but_flat": (
                irrh["pair"] if irrh and not irrh["ci_excludes_zero"] else None
            ),
            "up10_unsupported_but_large": (
                ruroh["pair"] if ruroh and ruroh["ci_excludes_zero"] else None
            ),
            "max_abs_level_t": max(
                (v for v in gls_cluster["t"].values() if v is not None), default=None
            ),
        },
        "caveats": [
            "A unit is cell-matched but not DOI-matched; part of Delta can be a paper "
            "effect, and the clustered SE does not remove it, it only widens the interval.",
            "ICC(1) assumes exchangeable units within a ligand. Cell lines differ in scale, "
            "so the deff estimate is approximate; the bootstrap SE is the primary one.",
            "A non-significant edge is not an uninformative edge. Ir-Rh's mean is "
            "indistinguishable from zero, which is itself a constraint (u_Ir ~ u_Rh). "
            "The GLS level solution uses all three edges.",
            "Levels are identifiable only up to the sum-zero constraint; the absolute "
            "offset of the whole metal block is not estimable from these data.",
            "Re has degree 0 in every scenario: it appears in no cell-matched pair, so no "
            "statement about its level is possible at all.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")

    print(f"[UP-18] endpoint={args.endpoint} nodes_with_edges={graph['n_nodes_with_edges']} "
          f"edges={graph['n_edges']} tree={graph['is_tree_observed_subgraph']} "
          f"isolated={graph['isolated_metals']}")
    for r in edges:
        print(f"[UP-18] {r['pair']:6s} n={r['n_groups']:4d} lig={r['n_ligands']:4d} "
              f"mean={r['mean_delta']:+.4f} sd={r['sd_delta']} icc={r['icc_ligand']} "
              f"t_noise={r['t_noise']} t_cluster={r['t_cluster']} "
              f"prec_share_cluster={r['precision_share_cluster']} "
              f"prec_share_noise={r['precision_share_noise']} "
              f"ci={r['mean_delta_ci95_cluster']}")
    print(f"[UP-18] levels(cluster)={gls_cluster['levels']} t={gls_cluster['t']}")
    print(f"[UP-18] gate_pass={not failing} failing={failing} kill={kill['triggered']}")
    for held, v in graph["per_metal_out"].items():
        print(f"[UP-18] metal_out {held:2s}: train_edges={v['train_edges']} "
              f"identifiable_level_dim={v['identifiable_level_dim_train']}")
    print(f"json: {args.out}")
    print("PASS" if not failing else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
