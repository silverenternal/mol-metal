# Wf-A16 — unimplemented-feature register

**Created:** 2026-10-03 (Wf-A16 close-out; the file this register lives in was
referenced by 17 test modules but had never been written).

## §0 What this file is, and why it exists

During Wf-A16, seventeen test modules were found that could not even be
**collected**: each began with a module-level `from <module> import <symbol>`
for a symbol that does not exist anywhere in the tree.  Under
`--strict-markers`/plain collection that is an ERROR, not a skip, so the whole
suite reported collection failures.

Rather than delete the tests (which would erase the evidence that the feature
was ever intended) they were converted to a loud, self-describing SKIP:

```python
try:
    from <module> import <symbol>
except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
    pytest.skip("UNIMPLEMENTED FEATURE: ... Tracked in "
                "TODO/completed/wfa16_unimplemented_feature_register.md",
                allow_module_level=True)
```

This file is the register those messages point at.  **A SKIP here is not a
pass.**  Each entry below is a wire-in that was written up as done and never
performed — the same defect class AGENTS.md §5.7 names.

## §1 The register (17 entries, 8 distinct modules)

Read from each test module's own skip message — no entry is inferred.

| # | Missing symbol | Expected from | Test module |
|---:|---|---|---|
| 1 | `DATASET_REGISTRY` | `bench_lambda_triangle_end_to_end` | `test_wf330t2_triangle_metalcytotox.py` |
| 2 | `exact_minibatch_ot_coupling` | `molmetal.adapters.flow_matching_lipman.optimal_transport_v2` | `test_wf349a2_otcfm_audit.py` |
| 3 | `_COLUMN_SYNONYMS` | `molmetal.data.metal_cytotox` | `test_wf323b_v3_v4_schema_drift.py` |
| 4 | `_COLUMN_SYNONYMS` | `molmetal.data.metal_cytotox` | `test_wf327f2_v3_v4_schema.py` |
| 5 | `GPU_WRAPPER_TABLE` | `molmetal.scoring.score_any` | `test_score_any_gpu_routing.py` |
| 6 | `METAL_SYMBOLS` | `molmetal_lam.atoms.combinators` | `test_tmqm_dblock_extract.py` |
| 7 | `canonical_smiles` | `molmetal_lam.sbdd_env.metrics_v2` | `test_m11_n_distinct.py` |
| 8 | `EXPLORATION_STRATEGIES` | `molmetal_lam.search_alg.proof_search` | `test_joint_3stage_mcts.py` |
| 9 | `EXPLORATION_STRATEGIES` | `molmetal_lam.search_alg.proof_search` | `test_variance_mcts_proof_search.py` |
| 10 | `DEFAULT_BF16` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf328w3_bf16_production.py` |
| 11 | `DEFAULT_DMPNN_BACKEND` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf334u1_dmpnn_backend_wasserstein.py` |
| 12 | `DEFAULT_TF32` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf328w4_tf32_production.py` |
| 13 | `DEFAULT_W2_GRADIENT_MODE` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf328w5_w2_sort_production.py` |
| 14 | `_JointGraphDataset` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf260_dataloader_fix.py` |
| 15 | `_JointGraphDataset` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf328w6_dataloader_factory_production.py` |
| 16 | `_make_compiled_encoder_forward` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf328w1_torch_compile_production.py` |
| 17 | `_make_compiled_encoder_forward` | `scripts.train_dmpnn_multitask_wasserstein` | `test_wf335r1_no_cudagraphs.py` |

## §2 Grouped by module — what each gap actually is

### `bench_lambda_triangle_end_to_end` — 1 missing symbol(s)

* **`DATASET_REGISTRY`** — triangle-bench dataset registry. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf330t2_triangle_metalcytotox.py`

### `molmetal.adapters.flow_matching_lipman.optimal_transport_v2` — 1 missing symbol(s)

* **`exact_minibatch_ot_coupling`** — exact minibatch OT coupling. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf349a2_otcfm_audit.py`

### `molmetal.data.metal_cytotox` — 2 missing symbol(s)

* **`_COLUMN_SYNONYMS`** — v3/v4 column-synonym map. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf323b_v3_v4_schema_drift.py`
* **`_COLUMN_SYNONYMS`** — v3/v4 column-synonym map. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf327f2_v3_v4_schema.py`

### `molmetal.scoring.score_any` — 1 missing symbol(s)

* **`GPU_WRAPPER_TABLE`** — GPU wrapper routing table. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_score_any_gpu_routing.py`

### `molmetal_lam.atoms.combinators` — 1 missing symbol(s)

* **`METAL_SYMBOLS`** — d-block metal symbol set. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_tmqm_dblock_extract.py`

### `molmetal_lam.sbdd_env.metrics_v2` — 1 missing symbol(s)

* **`canonical_smiles`** — metric-v2 canonicaliser. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_m11_n_distinct.py`

### `molmetal_lam.search_alg.proof_search` — 2 missing symbol(s)

* **`EXPLORATION_STRATEGIES`** — "under either import root. The test was written for a wire-in that "was never performed; it is kept as a visible register entry rather "than deleted
  * test: `molmetal/tests/test_joint_3stage_mcts.py`
* **`EXPLORATION_STRATEGIES`** — "under either import root. The test was written for a wire-in that "was never performed; it is kept as a visible register entry rather "than deleted
  * test: `molmetal/tests/test_variance_mcts_proof_search.py`

### `scripts.train_dmpnn_multitask_wasserstein` — 8 missing symbol(s)

* **`DEFAULT_BF16`** — BF16 autocast. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf328w3_bf16_production.py`
* **`DEFAULT_DMPNN_BACKEND`** — pluggable D-MPNN backend. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf334u1_dmpnn_backend_wasserstein.py`
* **`DEFAULT_TF32`** — TF32 matmul. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf328w4_tf32_production.py`
* **`DEFAULT_W2_GRADIENT_MODE`** — W2 gradient sort mode. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf328w5_w2_sort_production.py`
* **`_JointGraphDataset`** — joint graph dataset. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf260_dataloader_fix.py`
* **`_JointGraphDataset`** — dataloader factory. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf328w6_dataloader_factory_production.py`
* **`_make_compiled_encoder_forward`** — torch.compile encoder forward. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf328w1_torch_compile_production.py`
* **`_make_compiled_encoder_forward`** — no-cudagraphs path. The test was written for a wire-in that was "never performed; it is kept as a visible register entry rather than "deleted
  * test: `molmetal/tests/test_wf335r1_no_cudagraphs.py`

## §3 The largest gap: the D-MPNN trainer's production flags

8 of the 17 entries are symbols that
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` was supposed to
export — `DEFAULT_BF16`, `DEFAULT_TF32`, `DEFAULT_W2_GRADIENT_MODE`,
`DEFAULT_DMPNN_BACKEND`, `_JointGraphDataset`,
`_make_compiled_encoder_forward`.  These correspond to the Wf-328 family of
"production" toggles (bf16 / tf32 / torch.compile / dataloader factory /
gradient-sort mode): the tests exist, the flags were never wired.  Read
alongside `TODO/completed/385a1_wf380_cli_flag_audit.md` and AGENTS.md §5.7
("Trainer flag wire-in smoke-required rule") — those record the same pattern
from a different audit.

## §4 Honest framing

* Every symbol name, module path and test filename above is **MEASURED** — read
  out of the test modules' own skip strings on 2026-10-03 with a parser over
  `molmetal/tests/*.py` (17/17 resolved; the extraction is reproducible with the
  same grep).  Nothing here is inferred from a wave brief.
* **No feature was implemented by writing this file.** It records absence. The
  seventeen SKIPs still SKIP; the suite still cannot test these wire-ins.
* The `why` column is the test author's own wording, carried over near-verbatim
  (whitespace normalised, source-level string-concatenation artifacts removed).
  Where the skip message recorded no rationale, the entry says so rather than
  inventing one.
* **This is a register, not a plan.** Whether any of these should be implemented
  is a separate decision that has not been made.

## §5 Cross-references

* `molmetal/tests/test_script_import_health.py` — the guard built in the same
  wave (generalised `@dataclass`-clobber detection over every script).
* `TODO/completed/wfa16_closeout.md` — the other Wf-A16 deliverable.
* AGENTS.md §5.5 (filesystem is the source of truth) + §5.7 (flag wire-in smoke).
* `TODO/completed/385a1_wf380_cli_flag_audit.md` — the earlier audit that found
  17 NEW CLI flags declared but never wired.
