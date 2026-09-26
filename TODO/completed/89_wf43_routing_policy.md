# TODO-89 — Wf-43 Phase 2 close-out (Routing Policy)

**Date**: 2026-09-26
**Workflows**: Wf-43 (metric → real-tool routing policy)
**Status**: ✅ **SHIP** — routing policy table + machine-readable JSON + 20 new tests
**Parent**: Wf-43 Phase 1 (`molmetal.scoring.score_any` runtime router)
**Companion**: `molmetal/scoring/REGISTRY.md` (Phase 1 runtime metric table)

---

## §1 TL;DR

Wf-43 Phase 1 shipped the **runtime router** (`score_any` + `REGISTRY`)
that forces every registered metric through a real, locally-deployed
scoring tool — never a mock. Wf-43 Phase 2 ships the **policy table**
that names the mandatory tool per metric, declares the fallback
chain, and audits the mock-keys in the repo.

## §2 Ships

| File | Lines | Purpose |
|---|---:|---|
| `molmetal/scoring/ROUTING_POLICY.md` | 280 | Human-readable routing policy (13 sub-sections, 27 routed metrics + 10 model metrics + 7 training diagnostics) |
| `molmetal/scoring/policy.json` | 738 | Machine-readable policy (parsed by `test_score_any_policy.py` for sync verification) |
| `molmetal/scoring/score_any.py` | +330 | New `PolicyEntry` dataclass + `ROUTING_POLICY` dict (27 entries) + `_policy_register_many()` helper |
| `molmetal/scoring/__init__.py` | +18 | Exposes `ROUTING_POLICY` + `PolicyEntry` at package level |
| `molmetal/tests/test_score_any_policy.py` | 285 | 20 acceptance tests (5 sections: shape, alignment, content, JSON sync, e2e) |

## §3 Routing policy inventory

| Bucket | n | Examples |
|---|---:|---|
| **MEASURED** (real tool shipped + tested) | 14 | vina_score, vina_high_affinity, vina_min_score, vina_reference, diffdock_pose, flowdock_pose, boltz_affinity, af2_confidence, pb_pass_rate, aizynth_synthesis, pl_likelihood, targetdiff, sa_score, qed, flowr |
| **RESERVED** (metric name reserved, tool not shipped or env-gated) | 9 | vina_qvina2, vina_gpu, equibind_pose, pocket2mol, decomp_diff, molcraft, imold_ood, drugood_ood, glide_score |
| **MOCK_DELETED** (was a mock/stub; module absent) | 4 | coord_geometry_proxy, phototherapy_activity, ptiv_reduction_potential, metalloprotein_affinity |

**Total routed metrics**: 27.

**Model-evaluation metrics** (NOT routed through score_any — produced
by training scripts directly): 10 — `metalcytotox_ru_auc`,
`metal_lomo_auc`, `multimetal_lomo`, `moleculenet_auc`,
`drugood_ood_auc`, `nci60_precious_9cell_auc`, `platinai_4cell_auc`,
`platinai_3cell_resistance`, `metric18_protein_metal{,_v2}`.

**Training-diagnostics** (NOT paper-grade, NOT routed): 7 —
`decode_ratio`, `atom_vocab_coverage`, `n_distinct`, `n_train_scaleup`,
`metallodrug_vertical_coverage`, `baseline_random_smiles`,
`external_scorers_summary`.

## §4 Mock-audit findings

1. **`molmetal/adapters/mock.py`** (MockGenerator / MockDocker /
   MockPredictor / MockScorer) — **KEPT** for unit-test use only. NOT
   reachable from `score_any`. The `test_no_mock_keyword_in_registry`
   gate enforces that no metric name in REGISTRY contains `"mock"`.

2. **`molmetal/scripts/dock_10pockets_se3sb.py:_stub_score`** —
   **KEPT** for the genuine "no vina/meeko on PATH" failure case. The
   `_stub_score` function only fires inside `run_sweep(..., use_real_vina=False)`
   or when VinaDockingAdapter import fails — both are explicit opt-in
   failures, never silent substitutions. When routed through
   `score_any()`, the import is mandatory and `AdapterUnavailable` is
   raised on import failure.

3. **`molmetal/scripts/wf30_dock_summarize.py:scorer="stub"`** —
   **INFORMATIONAL ONLY**. Reads `use_real_vina` flag from upstream
   and labels the per-pocket summary `scorer: "stub" / "vina"`. No
   scoring happens here.

4. **Phase-4A external scorers** (`coord_geometry_proxy`,
   `phototherapy_activity`, `ptiv_reduction_potential`,
   `metalloprotein_affinity`) — **MOCK_DELETED**. They were the worst
   class of mock — they used a **missing module**
   (`molmetal_lam.sbdd_env.external_scorers`) whose absence caused
   `_safe_external_score` to silently return `None`. Wf-43 Phase 2
   marks them `MOCK_DELETED` and redirects each to a real
   alternative (where one exists) or honest non-paper disposition.

   * `coord_geometry_proxy` → `metal_coord_compliance` (RDKit SMARTS, real)
   * `metalloprotein_affinity` → D-MPNN trained on
     `molmetal/data/protein_metal_binding.csv` (real)
   * `phototherapy_activity` → no real ship (removed from paper §4)
   * `ptiv_reduction_potential` → DFT surrogate (outside scope)

## §5 Acceptance gate

```bash
# Phase 2 verification command (the one in the task spec):
python -c "from molmetal.scoring.score_any import score_any, ROUTING_POLICY; print(len(ROUTING_POLICY))"
# Output: 27

# Phase 2 full test suite:
pytest molmetal/tests/test_score_any_policy.py -v
# 20 passed
```

## §6 Honest-negative caveats

* `policy.json` `policy.<metric>.module` field for RESERVED entries
  (`vina_qvina2` → `molmetal.scripts.q_vina2_swap`) does NOT mean the
  adapter is shipped — the module may not exist yet. RESERVED
  disposition means "metric name reserved, tool not yet shipped; the
  wrapper raises `AdapterUnavailable` until the adapter lands".
* The 4 MOCK_DELETED entries are NOT silently removed — their JSON
  measurements in `metrics/by_metric/` are still present (the
  upstream `_safe_external_score` returned `None` for them, producing
  the JSONs you see). Future paper §4 revisions must explicitly
  remove or redirect these.

## §7 Files touched in this commit

```
 molmetal/scoring/ROUTING_POLICY.md      |  276 +++++++
 molmetal/scoring/__init__.py            |   49 ++
 molmetal/scoring/policy.json            |  738 ++++++++++++++++++
 molmetal/scoring/score_any.py           | 1266 +++++++++++++++++++++++++++++++
 molmetal/tests/test_score_any_policy.py |  285 +++++++
 5 files changed, 2614 insertions(+)
```

Commit: `6970a0a Wf-43 Phase 2: routing policy table — every metric routed to a real tool`