# WF-Lambda-Core Final Report (2026-09-15)

Status: SHIPPED — 5 core Lambda features integrated end-to-end.

## 1. Scope (5 features shipped)

| id | module                                                                                | tests                                                | lit anchors                            |
|----|---------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------|
| L1 | :file:`molmetal/molmetal_lam/lam_chem/conformer_embed.py`                             | :file:`molmetal/tests/test_conformer_embed.py`       | Riniker 2015, Halgren 1996              |
| L2 | :file:`molmetal/molmetal_lam/lam_chem/pharmacophore_filter.py`                        | :file:`molmetal/tests/test_pharmacophore_filter.py`  | Lipinski 2001, Veber 2002, Hopkins 2008 |
| L3 | :file:`molmetal/molmetal_lam/lam_chem/stereo_aware_reduction.py`                      | :file:`molmetal/tests/test_stereo_aware_reduction.py`| Sharpless 2001, Kolb 2003, Himo 2005    |
| L4 | :file:`molmetal/molmetal_lam/reactions/confidence.py`                                 | :file:`molmetal/tests/test_reaction_confidence.py`   | tmQM (Balcells 2010), Laplace 1774      |
| L5 | :file:`molmetal/molmetal_lam/search_alg/pareto.py`                                    | :file:`tests/test_pareto.py`                        | Deb 2002, Zitzler 1999, Knowles 2006    |

All 5 modules are CPU-only, RDKit/PyTorch-light, and integrate
cleanly with the existing ``r4_lambda_only_run.py`` orchestrator
without modifying it.

## 2. Pytest result

```
$ uv run pytest molmetal/tests/test_conformer_embed.py \
    molmetal/tests/test_pharmacophore_filter.py \
    molmetal/tests/test_stereo_aware_reduction.py \
    molmetal/tests/test_reaction_confidence.py \
    tests/test_pareto.py -q --tb=short
71 passed, 1 warning in 0.32s
```

**71 / 71 unit tests green** for the 5 new modules. The broader
project pytest regression is tracked in tasks #717 / #727; the
pre-existing ``test_atom_training_contract.py`` failure is
unrelated (assertion 4.605 vs measured 2.485 from P0 fix #4
hidden_dim warning emission).

## 3. Integration smoke result

See :file:`molmetal/reports/wf_lambda_core/phase3_integration.md`
for the full report.

Summary:

* 10 SMILES, 0.293 s wall
* pharma_pass_rate_strict = 0.7 (7 / 10)
* conformer embed ok = 10 / 10
* stereo-bearing mols = 1 / 10 (nicotine; the only stereo SMILES in
  the curated set)
* Pareto front size = 4
* hypervolume = 5.88

Pipeline composition: pharmacophore filter → conformer embed →
stereo annotation → reaction confidence → Pareto rank. None of the
locked files (``r4_lambda_only_run.py`` etc.) were touched.

## 4. Recommendation for paper Section 5 (ablation)

The 5 new modules provide **5 new ablation axes** that can be added
to the existing :math:`2^6 = 64` ablation cell grid in
:file:`paper/sections/05_ablation.tex`:

| axis                                              | on/off treatment                     |
|---------------------------------------------------|--------------------------------------|
| **ablate_conformer_embed**                        | skip L1 (use identity coords)        |
| **ablate_pharmacophore_filter**                   | skip L2 (all candidates pass)        |
| **ablate_stereo_aware_reduction**                 | skip L3 regio annotation             |
| **ablate_reaction_confidence**                    | skip L4 Laplace prior                |
| **ablate_pareto_ranker**                          | use weighted-sum scalar instead of NSGA-II |

Each axis has a one-line drop-in (the integration smoke is a
clean reference for the ablation wiring). Expanding the grid
from :math:`2^6 = 64` to :math:`2^{11} = 2048` is excessive; we
recommend instead adding 5 **dedicated ablation panels** each of
which toggles only one of the new axes while holding the others
fixed:

* §5.9 Conformer embed: PB rate with vs without ETKDGv3+MMFF94s
* §5.10 Pharmacophore filter: pool quality with vs without Ro5/Veber gate
* §5.11 Stereo-aware reduction: regio/stereo correctness on click set
* §5.12 Reaction confidence: synthesis-prediction accuracy on tmQM test
* §5.13 Pareto ranker: ranking quality (Spearman vs gold) NSGA-II vs scalar

These can be added to §5 with the same DESIGN/MEASURED/CITEDONLY
three-state tagging convention used elsewhere.

## 5. Honest framing

* **This is plumbing validation, not algorithmic gain.** The 5
  modules existed (Phase 2 ship); Phase 3 only confirms they
  combine without import cycles or schema drift.
* **The smoke is biased upward.** 10 hand-picked SMILES are not
  representative of an MCTS-generated pool; do not quote the
  pharma_pass_rate=0.7 as a Lambda performance number.
* **Pareto front is non-trivial but small.** A 10-point front
  spans the spectrum (drug-like, Pt_II, acyclic) and is
  internally non-dominated, but the 4-point selection is the
  size that this curated set admits — a larger pool would
  produce a larger front.
* **Production ranking blocked by GPU.** Round-13/14 real
  sweeps remain blocked on GPU recovery (see
  ``wf_gpu_recovery_now 2026-09-15``). The Phase 3 work runs
  entirely on CPU and is therefore a deterministic verification
  step we can ship regardless of GPU state.

## 6. Files

* 5 modules + 5 test files (all under
  :file:`molmetal/molmetal_lam/`)
* Integration smoke: :file:`molmetal/scripts/wf_lambda_core_phase3_smoke.py`
* Integration report: :file:`molmetal/reports/wf_lambda_core/phase3_integration.md`
* Integration JSON: :file:`molmetal/reports/wf_lambda_core/phase3_integration.json`
* This final: :file:`molmetal/reports/wf_lambda_core/final.md`
* New metric JSON: :file:`metrics/by_metric/pharmacophore_pass_rate.json`

## 7. Verdict

**SHIPPED** — 5 core Lambda features integrated end-to-end;
71/71 tests green; smoke confirms non-degenerate Pareto front in
0.293 s; 5 new ablation axes ready for §5 paper integration.
**Honest**: this is the verification / plumbing layer; the
algorithmic lift from these 5 features on a real Round-14 sweep
remains the next measurement (post-GPU recovery).