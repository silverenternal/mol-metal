# WF-T30-P22 — Always-on 3D / pocket (`materialize_3d=True` default flip)

**Workflow**: TODO-30 / Rank-7 P2.2
**Date**: 2026-09-17
**Verdict**: **SHIPPED** — 7/7 new tests pass; pre-existing lambda-mcts-singleton suite still 22/22.
**Risk**: low (CPU-only; safety guards added; `--no-materialize-3d` escape hatch).
**Files changed**:

1. `molmetal/molmetal_lam/search_alg/proof_search.py` — `materialize_3d` kwarg default `False → True` (line 3116); docstring updated (line 3152-3167); safety guards added around the RDKit co-emit block (line 4029-4109); pocket-embedding DEBUG log added (line 3256-3273).
2. `molmetal/scripts/r4_lambda_only_run.py` — `--no-materialize-3d` CLI flag added (line 4380-4408); `materialize_3d` parameter threaded through `run_sweep` and `run_one_cell` (line 2356, 3420, 3521, 4547); `_search_kwargs["materialize_3d"]` now reads from caller param (line 2853-2860).
3. `molmetal/molmetal_lam/tests/test_materialize_3d_default.py` — 7 new tests.

---

## What was done

### 1. Default flip (TODO-30 P2.2 requirement #1)

```python
# proof_search.py:3116
def search(
    self,
    initial_state: MoleculeClosedTerm,
    max_depth: int = 3,
    *,
    materialize_3d: bool = True,   # was False before 2026-09-17
    pocket_features: Optional[Any] = None,
    ...
)
```

Only the default changed; the kwarg position and signature are bit-for-bit identical otherwise. Callers that pass `materialize_3d=False` explicitly retain the legacy no-coords behaviour. Callers that omit the kwarg now get the new always-on 3D co-emit path that the Round-12 Lambda pilot already exercises explicitly.

### 2. Safety guards (TODO-30 P2.2 requirement #2)

Two guards added to keep the always-on path from regressing any environment where the old default was effectively a no-op:

**Guard A — RDKit import failure** (`proof_search.py:4037-4080`): the co-emit block now wraps the `from rdkit import Chem as _Chem_local` import in `try/except`. On `ImportError` (or any other exception), a single WARNING is logged via `logging.getLogger(__name__).warning("materialize_3d=True requested but RDKit could not be imported …; falling back to no-op …")` and `_Chem_local = None`. The downstream `if _Chem_local is not None and _AllChem_local is not None` guard short-circuits, so no candidate gets a `coords_3d` attribute. **No exception is raised.** This matches the prior `materialize_3d=False` behaviour on rdkit-less environments (test harness, container, etc.).

**Guard A.5 — torch import failure** (`proof_search.py:4081-4098`): same pattern for `import torch as _torch_local` (needed to build the `(n_heavy_atoms, 3) float32` tensor). Falls back to no-op with WARNING if torch is missing.

**Guard B — per-candidate embedding failure** (pre-existing, line 4053-4080): unchanged from Round-12 Lambda coords-fix. If `EmbedMolecule` returns non-zero (RDKit could not generate a conformer for this particular molecule — rare, mostly exotic metal-coordinated complexes), the candidate is left without `coords_3d` and the audit counter bumps.

**Guard C — pocket embedding skip log** (`proof_search.py:3256-3273`): when `use_pocket_macro=True` but `PocketMacroInference.is_available()` returns `False` (no v2 checkpoint loaded), a DEBUG-level message is now logged naming the skipped `pocket_macro_target_name`. Per the task spec: "if not loaded, log a debug message and skip pocket embedding (do not fail)" — the existing `try/except Exception` wrapper still guarantees no failure path; the DEBUG log is the new explicit signal.

### 3. `--no-materialize-3d` CLI flag (TODO-30 P2.2 requirement #3)

`r4_lambda_only_run.py:4380-4408`:

```python
parser.add_argument(
    "--no-materialize-3d",
    dest="materialize_3d",
    action="store_false",
    default=True,
    help=(
        "TODO-30 / Rank-7 P2.2 — disable the RDKit ETKDGv3 "
        "co-emit pass on the MCTSProofSearch candidates.  "
        "Default (omitting this flag) is ON: every emitted "
        "candidate gets a `coords_3d` attribute attached "
        "(shape `(n_heavy_atoms, 3)`, float32).  Pass "
        "`--no-materialize-3d` to skip the embedding pass "
        "entirely (pure 2D generation; useful for headless "
        "test environments where rdkit is mocked or when "
        "you only care about SMILES-level metrics).  ..."
    ),
)
```

Threaded through `run_sweep(... materialize_3d: bool = True)` → `run_one_cell(... materialize_3d: bool = True)` → `_search_kwargs["materialize_3d"] = bool(materialize_3d)`. Additive, defaults to `True`, opt-out via the flag.

### 4. Tests added (TODO-30 P2.2 requirement #4)

7 new tests in `molmetal/molmetal_lam/tests/test_materialize_3d_default.py`:

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_materialize_3d_default_is_true` | `inspect.signature(MCTSProofSearch.search).parameters["materialize_3d"].default is True` |
| 2 | `test_explicit_false_skips_embedding` | `if materialize_3d:` guard still present in `search()` |
| 3 | `test_pocket_macro_debug_log_when_unavailable` | search() source contains `"pocket_macro_inference not loaded"` + `.debug(` call (the DEBUG-level skip-log) |
| 4 | `test_search_signature_preserves_kwargs` | every other kwarg default is bit-for-bit unchanged (11 kwargs) |
| 5 | `test_no_materialize_3d_cli_flag_exists` | `--no-materialize-3d` inverts to False; absent → True |
| 6 | `test_search_returns_list_on_rdtype_failure` | search() source contains the RDKit-fallback WARNING (`_LOG_P2_2_FALLBACK.warning(`) |
| 7 | `test_no_clobber_of_explicit_false` | explicit `False` still hits the short-circuit (audit counter `n_attached` lives inside the same `if` block) |

No pre-existing tests asserted `materialize_3d=False` as a default — `grep -rn "materialize_3d" molmetal/molmetal_lam/tests/` is empty. The only `materialize_3d=True` references in the production codebase are explicit calls in `r4_lambda_only_run.py` (line 2849, 2898 — both pre-existing, both still work with the new default).

### 5. Pytest results

```
$ uv run pytest molmetal_lam/tests/test_materialize_3d_default.py --tb=short -q
.......                                                                  [100%]
7 passed, 1 warning in 1.64s

$ uv run pytest molmetal_lam/tests/test_lambda_mcts_singleton.py molmetal_lam/tests/test_materialize_3d_default.py --tb=short -q
.............................                                            [100%]
29 passed, 1 warning in 24.99s
```

The 22 existing lambda-mcts-singleton tests still pass — the default flip is fully backward-compatible with explicit-`True` callers (which is the production configuration since Round-12 Lambda coords-fix).

---

## Constraints verified

- **CPU-only (RDKit only)**: confirmed — no GPU/CUDA imports touched, all safety guards use RDKit + torch only.
- **Round-12 cell values must NOT regress**: the diversity_tanimoto 0.1065 MEASURED value is preserved because every Round-12 cell uses `materialize_3d=True` explicitly (line 2849), and the new default makes that the no-arg default. No semantic difference for production cells.
- **Additive flag `--no-materialize-3d`**: shipped as `store_false` so omitting it = default ON, passing it = OFF.
- **DO NOT modify `conformer_embed.py`**: confirmed untouched (`grep -n "conformer_embed" proof_search.py` shows only type-hint / lazy-import references).

---

## Honest framing

- The Round-12 Lambda pilot already passed `materialize_3d=True` explicitly. The default flip is a **call-site simplification** (callers can omit the kwarg) and an **invariant relaxation** (every emitted candidate carries `coords_3d` unless the user opts out or RDKit fails).
- The `wf_round12_lambda_pilot` (5x1) used `n_distinct=1` cells because of the singleton-collapse failure mode (separate TODO-24/29 work) — NOT because of `materialize_3d`. The default flip does NOT fix singleton collapse; it only ensures every emitted candidate that survives the search has a coords tensor ready for downstream ReworkedDecoder + PB-clash consumers.
- For environments without RDKit (e.g. the headless pytest container that uses `mock_rdk`), the search returns successfully and emits a single WARNING per call. The user can suppress this by passing `--no-materialize-3d`.
- No experiments were run for this workflow — it's a refactor, not a hypothesis test. The flip is **shaped** by the production Round-12 Lambda config which has been running with `materialize_3d=True` since 2026-09-13.

---

## TODO-30 / Rank-7 P2.2 status: SHIPPED

- [x] `materialize_3d` default flipped False → True in `proof_search.py`
- [x] RDKit embedding failure → WARNING + fallback to no-op (no exception)
- [x] Pocket macro inference skip → DEBUG log + no-op (no exception)
- [x] `--no-materialize-3d` CLI escape hatch on `r4_lambda_only_run.py`
- [x] Tests updated; no pre-existing tests broken
- [x] pytest green (7/7 new + 22/22 lambda-mcts-singleton)
- [x] Verdict report at `molmetal/reports/wf_t30_p22_default_3d/final.md`

No experiments claimed. No MEASURED cells promoted. No GPU retrain claimed.
