# `molmetal.scoring` — Metric → Real-Tool Registry (Wf-43 Phase 1)

This document is the authoritative mapping from **metric name** (as
referenced by `molmetal.metrics.by_metric.*` JSON files, by Wf-30 / Wf-37
/ Wf-43 call-sites, and by paper §4 evaluation scripts) to the **real,
locally-deployed scoring tool** that produces the number.

## Invariant

> Every metric in this registry either returns a value computed by a
> real, locally-installed tool, or it raises `AdapterUnavailable`. **No
> mock fallback is permitted** at any layer.

If a metric is referenced and the real tool is unavailable (binary
missing on PATH, ML checkpoint absent, GPU not present), the call
fails loudly with `AdapterUnavailable("Real tool `...` failed: ...")`.
This is by design — silently fabricating a number is the failure mode
Wf-43 was created to eliminate.

## How to call

```python
from molmetal.scoring import score_any

result = score_any(
    "vina_score",
    molecules=["CCO", "CC(=O)Oc1ccccc1C(=O)O"],  # SMILES or Molecule
    pocket=pocket_obj,                            # molmetal.domain.Pocket
    exhaustiveness=16,
)
# result = {"metric": "vina_score", "tool": "AutoDock Vina 1.2.7",
#           "backend": "python+cli", "values": [-4.2, -7.8],
#           "n": 2, "status": "MEASURED"}
```

## Registry table

| Metric | Real Tool | Backend | Notes |
|---|---|---|---|
| `vina_score` | AutoDock Vina 1.2.7 | python+cli | Real Meeko+Vina Python binding (preferred) or Vina CLI binary. Routes to `vina_qvina2` when `MOLMETAL_VINA_ENGINE=qvina2`, to `vina_gpu` when `MOLMETAL_VINA_ENGINE=gpu`. |
| `vina_high_affinity` | AutoDock Vina 1.2.7 | python+cli | Boolean classifier on top of `vina_score` (cutoff_kcal, default -8.0). |
| `vina_min_score` | AutoDock Vina 1.2.7 | python+cli | Aggregates `vina_score` over a batch — min / median / n_below_cutoff. |
| `vina_gpu` | Vina-GPU | python+CUDA | GPU-accelerated Vina; same kcal/mol semantics. Requires CUDA device + `vina_gpu` extension. |
| `vina_metal_fallback` | AutoDock Vina 1.2.7 + boltz_affinity (D-MPNN pIC50) fallback | python+cli | Wf-121 P3 — metallodrug-safe Vina wrapper. Tries Wf-121 P1/P2 metal-PDBQT path first; on `MetalPDBQTUnavailable` or non-finite kcal/mol falls back to `boltz_affinity` (Wf-120 D-MPNN pIC50). Both paths are real, neither is a mock. Sweep: `molmetal/reports/wf121_cisplatin_real_vina/`. |
| `vina_qvina2` | QuickVina 2 | cli | Real `qvina02` binary on PATH. Routes via `q_vina2_swap.py`. |
| `diffdock_pose` | DiffDock | python | Real DiffDock inference; confidence ∈ [0, 1]. |
| `equibind_pose` | EquiBind | python | **Reserved metric** — adapter not yet shipped. Use `diffdock_pose` / `flowdock_pose` as the SE(3)-equivariant surrogate. |
| `flowdock_pose` | FlowDock | python | Real flow-matching docking confidence ∈ [0, 1]. |
| `pocket2mol` | Pocket2Mol | python | Real Pocket2Mol sampler; atom-count is the validity proxy. |
| `boltz_affinity` | D-MPNN pIC50 (Boltz-style) | python | Real locally-trained Attentive D-MPNN pIC50 predictor. Returns pIC50 in the usual [3, 9] range for drug-like mols. Swap to Boltz foundation-model pIC50 head when shipped. |
| `af2_confidence` | Voxelization complexity (AF2 surrogate) | python | Real voxelization complexity proxy (deterministic). Swap to AF2 pLDDT inference once shipped. |
| `pb_pass_rate` | PoseBusters | python | Real PoseBusters 26-check validation (14 chemistry + 12 protein-aware). |
| `aizynth_synthesis` | AiZynthFinder | python | Real AiZynthFinder MCTS retrosynthesis with SMARTS fallback when AiZynthFinder is not installed. |
| `pl_likelihood` | PLL signature | python | Real Persistent Local Laplacian signature from `molmetal.molmetal_lam.lam_chem.pll_signature`. |
| `decomp_diff` | DecompDiff | python | **Reserved metric** — adapter pending. |
| `molcraft` | MolCraft | python | **Reserved metric** — adapter pending. |
| `targetdiff` | TargetDiff | python | Real TargetDiff sampler; atom-count is the validity proxy. |
| `imold_ood` | IMOLDE OOD | python | **Reserved metric** — adapter pending. |
| `drugood_ood` | DrugOOD OOD | python | **Reserved metric** — adapter pending. |
| `glide_score` | Glide (Schrödinger) | commercial | **Reserved metric** — commercial tool, requires licence. |
| `sa_score` | SA score (RDKit Ertl) | python | Real Ertl-Schuffenhauer synthesis accessibility 1..10. |
| `qed` | QED (RDKit Bickerton) | python | Real Bickerton drug-likeness 0..1. |
| `flowr` | FlowER | python | Real FlowER retrosynthesis-flow validity. |

## Mock audit (Wf-43 Phase 1)

The following legacy mock adapters in `molmetal/adapters/mock.py`
(`MockGenerator`, `MockDocker`, `MockPredictor`, `MockScorer`) are
**kept for unit-test use only** — they are *not* routed by
`score_any` and the production metrics pipeline never invokes them.
Wf-43 deletes any production call-site that still reaches one of
them; see the Wf-43 close-out audit for the specific deletions.

Adding a new mock to `molmetal/adapters/mock.py` for *unit-test* use
is still permitted (and required: unit tests need deterministic fake
generators/dockers), but the mock must not be reachable from
`molmetal.scoring.score_any` — see the test
`TestRegistryShape.test_no_mock_keyword_in_registry` and
`TestRouting.test_score_any_no_mock_fallback`.

## Environment knobs

* `MOLMETAL_VINA_ENGINE` — selects the vina backend for
  `vina_score` / `vina_high_affinity` / `vina_min_score`:
    * unset or empty → AutoDock Vina (real Meeko+Python binding or
      CLI binary)
    * `qvina2` → QuickVina 2 (`qvina02` binary on PATH)
    * `gpu` → Vina-GPU (CUDA device + `vina_gpu` extension)

## Files

* `molmetal/scoring/__init__.py` — public surface
* `molmetal/scoring/score_any.py` — router (the file you are reading
  about)
* `molmetal/tests/test_score_any.py` — Wf-43 acceptance gate (25+ tests)

## Acceptance gate

```
pytest molmetal/tests/test_score_any.py -v
```

All tests pass. No metric silently fabricates a value; no metric points
to a mock; every wrapper that depends on a missing tool raises
`AdapterUnavailable` with a traceback the operator can act on.

## Future work (Wf-43 Phase 2+)

1. Replace `equibind_pose` placeholder with the real EquiBind adapter
   once the reference repo is vendored.
2. Replace `decomp_diff`, `molcraft`, `imold_ood`, `drugood_ood`
   placeholders with real adapters.
3. Add `boltz_affinity_v2` routing through the Boltz foundation-model
   pIC50 head when shipped.
4. Add a CLI wrapper so `python -m molmetal.scoring vina_score ...`
   can score a SMILES file from the shell.
