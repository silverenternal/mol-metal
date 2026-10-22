#!/usr/bin/env python3
"""UP-25 — one table, every number tagged with what it estimates and whether it is identifiable.

Three rounds produced 20+ artifacts, and their numbers do not share an estimand:
a pooled R² over all test rows, a within-metal centred R², a per-metal-pair balanced
accuracy, a conditional coverage, and an edge-level t statistic are five different
quantities.  Putting them in one table without labels is a §5.2 violation waiting to
happen — the reader cannot tell which numbers may be quoted for which claim.

So each row carries:

    ESTIMAND      pooled | centered | per_pair | conditional | edge_level
    IDENTIFIABLE  yes | no | partial

``pooled``      a single number over a mixed row population (includes rows that no
                model could place, so it can be negative while the model works).
``centered``    the same fit scored after removing each metal's test mean — the
                oracle-offset upper bound, not a deployable score.
``per_pair``    a within-metal-pair or within-group statistic.
``conditional`` a statement about a region of the prediction (coverage, width,
                worst stratum), not about the average.
``edge_level``  a property of one metal-pair edge of the metal graph.

``identifiable`` is a statement about the *target*, not the estimate's quality:
``no`` means the quantity depends on a level that a held-out metal cannot carry
(the metal's own mean), ``partial`` means it is identified only on the subpopulation
that has paired evidence, ``yes`` means the target is a function of what the model
can see.

Values are read from the artifacts, never typed in.  The machine-check block in
``docs/PREDICTIVE_LINE_INDEX.md`` is then used as an independent cross-check: any id
that appears in both must agree.

Usage
-----
::

    .venv/bin/python molmetal/scripts/msrs_main_table.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

INDEX = PROJECT_ROOT / "docs" / "PREDICTIVE_LINE_INDEX.md"
MD_OUT = PROJECT_ROOT / "docs" / "PREDICTIVE_LINE_MAIN_TABLE.md"
FENCE = re.compile(
    r"^```json\s+predictive-line-index\s*$(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL
)

# (id, artifact, dot-path, estimand, identifiable, reason)
ROWS: List[tuple] = [
    ("G.ligand.ecfp4", "msrs_seed_ensemble", "protocols.ligand.ensemble.r2", "pooled", "no",
     "ligand-grouped split: 89.4% of test rows share a DOI with training (retracted 2026-10-19)"),
    ("G.doi.maccs", "msrs_feature_expansion_5seed", "protocols.doi.r2_seed_averaged", "pooled", "yes",
     "whole-paper hold-out; the headline the paper quotes"),
    ("G.doi.fused", "msrs_feature_fusion", "protocols.doi.fused_r2", "pooled", "yes",
     "same split, GNN fused in"),
    ("C1.metal_blind_delta", "msrs_metal_ablation", "protocols.ligand.arms.metal_blind.delta_r2_vs_full", "pooled", "partial",
     "ligand split; the sign of the metal-aware gain is identified, the level is not"),
    ("C3.majority_class", "msrs_delta_metric_audit", "baselines.majority_class_rate", "per_pair", "partial",
     "strict 218-pair set only; a 0.58 raw score is below this"),
    ("C3.diff_raw", "msrs_delta_metric_audit", "routes.level_differencing.raw_sign", "per_pair", "partial",
     "same strict pairs; raw sign accuracy, not balanced"),
    ("C5.coverage", "msrs_calibration", "protocols.doi.variants.global.coverage", "conditional", "yes",
     "marginal coverage under the DOI protocol"),
    ("C5.median_width", "msrs_calibration", "protocols.doi.variants.global.median_width", "conditional", "yes",
     "width at that coverage; the pair must always be quoted together"),
    ("C6.lomo_resolvable", "lomo_resolvability_audit", "aggregates.resolvable_only.value", "pooled", "partial",
     "Ru+Ir folds only; Rh/Os/Re are below MIN_UNIQUE_FOR_GATE"),
    ("UP2b.ligand_cell_balanced", "msrs_listwise_ranker", "pairs.ligand_cell.overall.balanced_accuracy", "per_pair", "partial",
     "360 cross-paper pairs; no single metal pair clears 0.5 on its own (UP-16)"),
    ("UP6.cohort_metal_worst_stratum", "msrs_calibration_shift", "variants.cohort_metal.worst_stratum.coverage", "conditional", "yes",
     "worst stratum after Mondrian; a repair of the tail, not of the width"),
    ("UP8.zero_shot_os", "msrs_meta_kshot", "results.potency.splits.metal_out.per_metal.Os.zero_shot_r2", "pooled", "partial",
     "Os is the one metal whose level transfers; not a general statement"),
    ("UP10.potency.ruos_separability", "msrs_support_audit", "endpoints.potency.pairs.2.separability", "edge_level", "yes",
     "Ru-Os: the only edge whose effect exceeds its own assay noise"),
    ("UP10.potency.irrh_separability", "msrs_support_audit", "endpoints.potency.pairs.4.separability", "edge_level", "yes",
     "Ir-Rh: a precise zero, i.e. it constrains u_Ir ~ u_Rh rather than reporting an effect"),
    ("UP11.complete_case_ligands", "msrs_pair_delta", "rank_structure.complete_case_ligands", "edge_level", "no",
     "zero ligands span 3 metals, so the rank-1 interaction model has no identifiable row"),
    ("UP12.additive_within", "msrs_two_stage", "within_metal.aggregate.additive.weighted_r2", "centered", "yes",
     "ligand-grouped within-metal; the reference the interaction arms are measured against"),
    ("UP12.free_within", "msrs_two_stage", "within_metal.aggregate.free.weighted_r2", "centered", "yes",
     "free interaction, same protocol: +0.0391 over additive"),
    ("UP14.doi_pair_in_train_share", "msrs_support_stratified", "protocols.doi.unit_train_evidence.has_pair_in_train.share_of_rows", "conditional", "yes",
     "0.57% of DOI-hold-out test rows have paired evidence in training"),
    ("UP14.metal_out_weighted_r2", "msrs_support_stratified", "protocols.metal_out.weighted_r2_per_metal", "pooled", "no",
     "the held-out metal's level is unidentifiable; 94.5% of these rows have no paired partner"),
    ("UP16.osru_balanced", "msrs_ranking_support", "strata.listwise_ligand_cell.by_metal_pair.Os-Ru.balanced_accuracy", "per_pair", "partial",
     "Wilson CI [0.4990, 0.6548] includes 0.5"),
    ("UP16.irrh_balanced", "msrs_ranking_support", "strata.listwise_ligand_cell.by_metal_pair.Ir-Rh.balanced_accuracy", "per_pair", "partial",
     "Wilson CI [0.4799, 0.6190] includes 0.5"),
    ("UP17.ligand_metal_forward", "msrs_shape_decomposition", "independent.ligand_x_metal.share_forward", "pooled", "partial",
     "order-dependent: 0.8015 structure-first vs 0.0 batch-first"),
    ("UP17.paper_metal_batch_first", "msrs_shape_decomposition", "independent.paper_x_metal.share_batch_first", "pooled", "partial",
     "the same block under the other ordering: 0.5484 vs 0.0"),
    ("UP18.ruos_t_cluster", "msrs_metal_graph", "edges_by_pair.Ru-Os.t_cluster", "edge_level", "yes",
     "ligand-clustered; 3.364 with a 95% CI [0.0618, 0.2164] on the mean"),
    ("UP18.ruroh_t_cluster", "msrs_metal_graph", "edges_by_pair.Ru-Rh.t_cluster", "edge_level", "partial",
     "15 units, sd 1.4003; t=2.577 survives but the interval spans a factor of five"),
    ("UP18.irrh_t_cluster", "msrs_metal_graph", "edges_by_pair.Ir-Rh.t_cluster", "edge_level", "yes",
     "0.958: the mean is indistinguishable from zero, which is itself the constraint"),
    ("UP18.ruroh_precision_share_cluster", "msrs_metal_graph", "edges_by_pair.Ru-Rh.precision_share_cluster", "edge_level", "yes",
     "largest effect, worst precision: the two edge rankings are different sets"),
    ("UP19.identifiable_share", "msrs_identifiability", "decomposition_summary.identifiable_row_share", "conditional", "yes",
     "5.47% of metal-out test rows have a paired partner in training"),
    ("UP19.pooled_all_r2_abs", "msrs_identifiability", "metal_out.pooled.descriptors.all_rows.r2_abs", "pooled", "no",
     "dominated by the 94.5% of rows whose metal level cannot be recovered"),
    ("UP19.pooled_identifiable_r2_abs", "msrs_identifiability", "metal_out.pooled.descriptors.identifiable_unit_rows.r2_abs", "pooled", "yes",
     "same model restricted to rows that carry paired evidence: 0.3299"),
    ("UP19.pooled_identifiable_r2_centred", "msrs_identifiability", "metal_out.pooled.descriptors.identifiable_unit_rows.r2_centred", "centered", "yes",
     "oracle-offset upper bound 0.3806; not a deployable score"),
    ("UP20.cqr_coverage", "msrs_cqr", "aggregate.cqr.coverage_mean", "conditional", "yes",
     "marginal coverage at the nominal 0.90"),
    ("UP20.cqr_width", "msrs_cqr", "aggregate.cqr.median_width_mean", "conditional", "yes",
     "width at that coverage; 1.9871, i.e. the width target is not met"),
    ("UP21.base_r2", "msrs_paper_meta_offset", "aggregate.base.r2_mean", "pooled", "yes",
     "DOI hold-out with no offset correction"),
    ("UP21.ridge_design_r2", "msrs_paper_meta_offset", "aggregate.ridge_design.r2_mean", "pooled", "yes",
     "metadata-only offset model: 0.3557, worse than doing nothing"),
    ("UP21.lopo_ridge_r2", "msrs_paper_meta_offset", "per_seed.20261022.offset_diagnostics_fold0.lopo_ridge_r2", "pooled", "no",
     "leave-one-paper-out R2 of the offset model: -0.0259, i.e. worse than the mean"),
    ("UP22.rh_np0_r2", "msrs_paired_kshot", "aggregate.Rh.ridge|0.mean_r2", "pooled", "partial",
     "row-level sampling with zero paired points: the gain is not a pairing effect"),
    ("UP22.re_np3_r2", "msrs_paired_kshot", "aggregate.Re.ridge|3.mean_r2", "pooled", "partial",
     "Re has 0 paired rows, so this number cannot come from pairing by construction"),
    ("UP23.n_pairs_kept", "msrs_ranking_hierarchical", "design.n_pairs_kept", "per_pair", "yes",
     "315 of 651 pairs survive the deadzone (|delta| >= the metal pair's noise SD)"),
    ("UP23.lambdarank_none_balanced", "msrs_ranking_hierarchical", "arms.lambdarank_none.all_pairs.overall.balanced_accuracy", "per_pair", "partial",
     "reproduces UP-2b's 0.5985 on all 360 pairs"),
    ("UP23.lambdarank_deadzone_balanced", "msrs_ranking_hierarchical", "arms.lambdarank_deadzone.all_pairs.overall.balanced_accuracy", "per_pair", "partial",
     "tie-aware labels: 0.5612, worse than leaving the noisy pairs alone"),
    ("UP24.bce_auc", "msrs_lomo_pairwise", "aggregate.bce.auc_overall_resolvable", "pooled", "partial",
     "linear ECFP4 baseline on Ru+Ir; not the D-MPNN 0.5974"),
    ("UP24.pairwise_edge_auc", "msrs_lomo_pairwise", "aggregate.pairwise_edge.auc_overall_resolvable", "pooled", "partial",
     "+0.0080 over the matched BCE arm: the loss is not the binding constraint"),
]

ESTIMANDS = ("pooled", "centered", "per_pair", "conditional", "edge_level")
IDENTIFIABLE = ("yes", "no", "partial")


def walk(doc, path: str):
    cur = doc
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit() and int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            raise KeyError(f"{path!r} does not resolve at {seg!r}")
    return cur


def load_index() -> Dict[str, float]:
    text = INDEX.read_text(encoding="utf-8")
    m = FENCE.search(text)
    if m is None:
        raise SystemExit("FAIL: PREDICTIVE_LINE_INDEX.md has no machine-check block")
    return {e["id"]: float(e["value"]) for e in json.loads(m.group("body"))}


def load_registry() -> set:
    doc = yaml.safe_load((PROJECT_ROOT / "registry" / "metrics.yaml").read_text())
    return set(doc.get("metrics", {}))


def collect() -> List[dict]:
    cache: Dict[str, dict] = {}
    out = []
    for rid, art, path, estimand, ident, reason in ROWS:
        if art not in cache:
            cache[art] = json.loads(
                (PROJECT_ROOT / "metrics" / "by_metric" / f"{art}.json").read_text()
            )
        value = walk(cache[art], path)
        out.append(
            {
                "id": rid,
                "artifact": f"metrics/by_metric/{art}.json",
                "artifact_name": art,
                "path": path,
                "value": value,
                "estimand": estimand,
                "identifiable": ident,
                "reason": reason,
            }
        )
    return out


def emit_markdown(rows: List[dict], checks: dict) -> str:
    lines = [
        "# PREDICTIVE_LINE_MAIN_TABLE — 按估计目标分层的数字主表",
        "",
        f"由 `molmetal/scripts/msrs_main_table.py` 生成（{datetime.now(timezone.utc).isoformat(timespec='seconds')}）。",
        "**每个数字都从 artifact 读出，不手抄**；与 `docs/PREDICTIVE_LINE_INDEX.md` 的机器校验块交叉核对。",
        "",
        "## §1 怎么读",
        "",
        "| ESTIMAND | 含义 |",
        "|---|---|",
        "| `pooled` | 混合行总体上的一个数（含没有任何模型能定位的行，所以可以是负数而模型仍在工作） |",
        "| `centered` | 每个金属先减测试均值后再评分 —— oracle-offset 上界，不是可部署分数 |",
        "| `per_pair` | 金属对内 / 组内的统计量 |",
        "| `conditional` | 关于预测某个区域（覆盖、宽度、最差层）的陈述，不是均值 |",
        "| `edge_level` | 金属图某一条边的性质 |",
        "",
        "| IDENTIFIABLE | 含义 |",
        "|---|---|",
        "| `yes` | 目标可由模型看得见的东西决定 |",
        "| `partial` | 只在「有配对证据」的子群上被识别 |",
        "| `no` | 依赖留出金属自己携带的水平，任何留出协议都拿不到 |",
        "",
    ]
    for tag, title in (
        ("yes", "§2 可识别（可以按原样引用）"),
        ("partial", "§3 部分可识别（必须同时引用条件）"),
        ("no", "§4 不可识别（不得作为能力陈述引用）"),
    ):
        subset = [r for r in rows if r["identifiable"] == tag]
        lines += [f"## {title}", "", "| 指标 | 值 | ESTIMAND | artifact:path | 为什么是这个标签 |",
                  "|---|---:|---|---|---|"]
        for r in sorted(subset, key=lambda x: (ESTIMANDS.index(x["estimand"]), x["id"])):
            v = r["value"]
            vs = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(
                f"| `{r['id']}` | {vs} | `{r['estimand']}` | "
                f"`{r['artifact'].split('/')[-1]}:{r['path']}` | {r['reason']} |"
            )
        lines.append("")
    lines += [
        "## §5 一致性检查",
        "",
        "| 断言 | 结果 |",
        "|---|---|",
        f"| 每行都有 ESTIMAND + IDENTIFIABLE + artifact 路径 | {'PASS' if checks['all_rows_tagged'] else 'FAIL'} |",
        f"| 主表里每个 artifact 都在 registry 中 | {'PASS' if checks['no_orphan'] else 'FAIL'} |",
        f"| 与 PREDICTIVE_LINE_INDEX 机器校验块一致（{checks['n_cross_checked']} 个 id 交叉核对） | "
        f"{'PASS' if checks['index_consistent'] else 'FAIL'} |",
        "",
    ]
    if checks["mismatches"]:
        lines += ["不一致明细：", ""]
        for m in checks["mismatches"]:
            lines.append(f"- `{m['id']}`: 主表 {m['table']} vs 索引 {m['index']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_main_table.json",
    )
    ap.add_argument("--md", type=Path, default=MD_OUT)
    args = ap.parse_args(argv)

    rows = collect()

    # --- assertions -------------------------------------------------------
    all_tagged = all(
        r["estimand"] in ESTIMANDS and r["identifiable"] in IDENTIFIABLE and r["artifact"]
        for r in rows
    )
    registry = load_registry()
    orphans = sorted({r["artifact_name"] for r in rows if r["artifact_name"] not in registry})
    index = load_index()
    mismatches = []
    n_cross = 0
    for r in rows:
        if r["id"] in index:
            n_cross += 1
            if abs(float(r["value"]) - index[r["id"]]) > 1e-4:
                mismatches.append(
                    {"id": r["id"], "table": r["value"], "index": index[r["id"]]}
                )

    checks = {
        "all_rows_tagged": all_tagged,
        "no_orphan": not orphans,
        "orphans": orphans,
        "index_consistent": not mismatches,
        "n_cross_checked": n_cross,
        "mismatches": mismatches,
        "gate_pass": bool(all_tagged and not orphans and not mismatches),
    }

    md = emit_markdown(rows, checks)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_file(args.md, md + "\n")

    by_estimand = {e: sum(1 for r in rows if r["estimand"] == e) for e in ESTIMANDS}
    by_ident = {i: sum(1 for r in rows if r["identifiable"] == i) for i in IDENTIFIABLE}
    out = {
        "artifact": "msrs_main_table",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L0 summary / UP-25",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_main_table.py",
            "index": "docs/PREDICTIVE_LINE_INDEX.md",
            "markdown": str(args.md.relative_to(PROJECT_ROOT)),
        },
        "design": {
            "n_rows": len(rows),
            "by_estimand": by_estimand,
            "by_identifiable": by_ident,
            "estimand_vocabulary": list(ESTIMANDS),
            "identifiable_vocabulary": list(IDENTIFIABLE),
            "value_source": "read from the artifact at the recorded path; never typed in",
        },
        "rows": rows,
        "checks": checks,
        "gate": {
            "no_threshold": True,
            "assert_all_rows_tagged": all_tagged,
            "assert_no_orphan": not orphans,
            "assert_index_consistent": not mismatches,
            "gate_pass": checks["gate_pass"],
        },
        "verdict": "PASS" if checks["gate_pass"] else "FAIL",
        "caveats": [
            "The ESTIMAND / IDENTIFIABLE tags are analyst judgements, not measurements. They are "
            "recorded here so a reader can disagree with a specific tag rather than with a table.",
            "Rows are a curated headline set, not every number in every artifact. The index's "
            "machine-check block is the complete list; this table selects the ones a paper would "
            "quote.",
            "Cross-checking against the index catches drift in the value, not in the tag.",
        ],
    }
    atomic_write_file(args.out, json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[UP-25] rows={len(rows)} by_estimand={by_estimand} by_identifiable={by_ident}")
    print(f"[UP-25] checks={checks['all_rows_tagged']}/{checks['no_orphan']}/"
          f"{checks['index_consistent']} orphans={orphans} mismatches={mismatches}")
    print(f"[UP-25] verdict={out['verdict']} md={args.md} json={args.out}")
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
