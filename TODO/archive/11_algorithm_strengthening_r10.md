# Round-10 — Algorithm strengthening (no experiments)

**Status:** local core implementation complete; full ablation evidence pending
**Priority:** high
**Effort:** 5 days (1 ultracode round)
**Owner:** (unset)
**Depends on:** Round-9 tmQM key-shape bridge (`load_tmQM_pretrained` + production-shape
EGNNVelocityField instantiation)
**Blockers:** none beyond Round-9 closure
**Created:** 2026-09-13

## Local progress update (2026-09-13)

The locally executable parts now have evidence: mini-batch OT uses real groups,
the CFG and OT ablation harnesses auto-select ROCm/CUDA with an explicit
`--device`, CFG handles the single-molecule smoke shape, and RMSNorm retains a
differentiable ROCm path. Remaining reaction-expansion and full ablation axes
are still pending.

## Goal

Before running the top-journal-aligned 100-pocket sweep, **strengthen every
Lambda-unique axis** and prove each contributes on a controlled micro-benchmark.
No sweep — pure algorithmic work + unit tests + small ablation harness.

## Scope (6 axes)

### A — 5 click reactions fully wired into MCTS expansion
Currently only CuAAC is in the MCTS rule expansion path; SPAAC, thiol-ene,
Suzuki, amide coupling live in `lam_chem/rules.py` but are not consumed by
`proof_search.py:_resolve_expand_tile_pool`. Wire all 5 reactions into the
expansion path; extend the 204-tile SMARTS-diverse pool to **220 tiles**
(200 ChEMBL/ZINC + 4 click-handles × 5 reactions).

### B — Mini-batch OT coupling (Tong 2023) ablation
`mini_batch_ot_coupling` is callable in `flow_matching.optimal_transport` but
the CFM loss still uses vanilla OT. Wire `ConditionalFlowMatchingLoss
(use_minibatch_ot=True)` and write a parity unit test (same optimizer step →
loss curve within 0.5% vs vanilla OT) + a tiny training ablation on
`r4c_pilot --pockets 1` (no sweep; 50 epochs smoke).

### C — Classifier-free guidance (CFG)
Conditional vs unconditional velocity field head; `cfg_scale=2.0` default;
ablation harness on small EGNN batch.

### D — Square-planar Pt(II) prior ablation
The `MetalGeometryPrior` is shipped; prove it actually lowers Vina on a Pt-bound
pocket subset. Micro-benchmark on PDB 1h36 (HEM Fe, but Pt analogues in the
queue): with prior off vs on, compare Vina delta.

### E — β-NF + AST formal semantics appendix
Currently `lam_chem/ast.py` and `bonds/application.py` are documented but
without a formal proof of confluence + strong normalization. Write a paper-
ready appendix (PDF + .tex) proving β-NF is confluent (Church-Rosser) +
strongly normalizing (standard reduction). Even if the proof is mechanical,
it closes the "interpretability" reviewer challenge.

### F — Per-component sanity metrics
Use the 9-layer `lambda_layer_metrics.md` metric catalogue (round-3) as the
basis for a `pytest --benchmark` harness that reports:
- ARITY_HIT_RATE ≥ 0.95
- METAL_GEOMETRY_OK = 1.0 (Pt_II=4, Ru_II=6, Zn_II=4, Ir_III=6)
- DATIVE_FRACTION = 1.0 for coordination complexes
- BOND_KIND_DISTRIBUTION sanity
- Click-rule coverage = 5/5
- Tile-pool size = 220

## Success criterion

- All 6 axes implemented + unit-tested
- Per-axis ablation result (with-prior > without-prior on 1h36) committed
- 158 + ~15 new tests pass in spot-check (`pytest -q --tb=short`)
- NO sweep beyond 1-pocket micro-bench (1h36 or 830c only, ≤5 min wall)

## Out of scope (deferred to round-11+)

- QVina install (round-11)
- CrossDocked2020 100-pocket staging (round-11)
- SOTA checkpoint re-runs (round-11)
- N=5+ pilot at top-journal protocol (round-12)
- 100-pocket full sweep (round-13)

## Recommended orchestration

Single ultracode workflow `w_round10_algorithm_strengthening`, 4 phases:

| Phase | Agents | Goal |
|---|---|---|
| 1 — Wire 5 click + mini-batch OT | 2 parallel | A + B implementations + tests |
| 2 — CFG + Pt prior ablation | 2 parallel | C + D implementations + ablation harness |
| 3 — β-NF proof + per-component metrics | 2 parallel | E + F (proof appendix + metric harness) |
| 4 — Verify | 1 | 158 + 15 new tests pass; consolidated report `molmetal/reports/round10_final.md` |

ENV constraints (every agent): "先别跑实验" — only 1-pocket micro-bench
allowed. ROCm 7.2 / triton-rocm 3.8.0 / gfx1101 / wave64. Use
`from triton_kernels import ...` convention.

## Related reports

- `molmetal/reports/lambda_layer_metrics.md` (per-layer metric catalogue)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (§3 P0/P1/P2 priority list)
- `TODO/completed/09_metal_prior_done.md` (round-8 Pt prior)
- `TODO/completed/10_minibatch_ot_done.md` (round-8 mini-batch OT stub)

## R10 Pt(II) prior end-to-end Vina micro-bench summary (2026-09-14, append-only)

| pocket | n_mols | prior=0.0 mean ± std n= | prior=0.1 mean ± std n= | Delta (on − off) ± pooled SE | verdict |
| --- | --- | --- | --- | --- | --- |
| 1h36 | 20 req / 4 docked | -2.197 ± 0.459 n=4 | -2.197 ± 0.459 n=4 | +0.000 ± 0.324 kcal/mol | INCONCLUSIVE (n_docked=4, noise floor dominates; identical score vectors ⇒ prior silent on this Pt-free synthetic-data run) |

Source: `molmetal/reports/round10_e2e_pt_cfg_vina.md` + `molmetal/reports/r10_pt_prior_e2e/{run.log, r10_pt_prior_ablation_1h36.{csv,json,md}}`. Harness wired, gates intact, success criteria inconclusive (not failed) at this n; Round-12 acceptance still deferred.

## 2026-09-14 — CFG seed-sweep append (run on hardcoded `(42,0,1234)` tuple because harness has no `--seed` flag)

| pocket | n_requested | n_finite (=n_raw) | n_decoded | n_docked | decode_status_counts |
| --- | ---: | ---: | ---: | ---: | --- |
| test_001 + test_002 | 96 | 96 | 0 | 0 | disconnected_distance_graph: 92; atom_outside_training_vocabulary: 4 |

## 2026-09-14 — WF-CFM-Retrain-Diagnose verdict (axis C/D evidence section)

**Verdict:** FAILURE provisional — GPU 侧 HSA init 阻塞, 5000-step 训练从未执行.
**Decode_ratio at 5000-step:** NOT MEASURED (baseline 2000-step = 0/384, MEASURED; 5000-step = 0/0 trivially, NOT MEASURED).
**Bond_loss slope at 5000-step:** NOT MEASURED. 2000-step 曲线 7.37→5.82 (下降 ~21%) 但 decode 仍 0; 2000-step 后 loss 仍在下降, 但 bridge 到成功 decode 是 unknown.
**Axis C (CFG) evidence:** CFG harness 工作 (seed-sweep 输出存在), 但 decode_ratio=0/96 — CFG 不补救 decoder 失败, 它只 control 条件 vs 无条件 velocity field head. 5000-step 诊断目的是测试 CFM 容量; CFG 不是瓶颈.
**Axis D (Pt prior) evidence:** Pt prior harness 工作 (1h36 n=4 n_docked), 评分相同 (prior silent on Pt-free run), INCONCLUSIVE not failed. 5000-step CFM 重训成功的话, Pt prior 才有几何场可以前置.
**Recommendation for Round-12:** default to λ-only column (path c, 0 GPU hours). Path (a) full retrain @ 10000-step + hidden-dim 64 = 6-12 h GPU; path (b) CFM architectural rework = 5-10 d effort. Both deferred until HSA init 恢复.
**Full retrain budget (PROJECTED, not measured today):** 6-12 h GPU 单卡 + 0.5 h diagnostic gate; calendar = 1 day after GPU returns.
**Source:** `molmetal/reports/wf_cfm_diagnose_verdict.md` (verdict report) + `molmetal/reports/wf_cfm_diagnose/{final.md,preflight.log,run.log}` (run artefacts).


Source: `molmetal/reports/r10_cfg_real_seed1/{run.log,report.json}` + per-cell `raw_*.json`. The failure is **chemical-graph-connectivity** (decoder produces fragmented graphs from sparse covalent-distance atom clouds), not sanitization. The harness cannot accept a fresh seed via CLI; the failure is bit-identical to the v2 baseline `molmetal/reports/r10_cfg_real_crossdocked_v2_train32_2000/`. **No seed sweep will resolve this** — fix is at the decoder/atom-head architecture level.
