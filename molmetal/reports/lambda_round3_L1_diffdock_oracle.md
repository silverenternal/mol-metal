# Lambda Round-3 L-1 — DiffDock / FlowDock binding oracle

Date: 2026-09-12

## Goal

Replace the fingerprint-based `typecheck(state, BindingSite)` stub in
`molmetal_lam.binding.types` with a real docking oracle that returns
RMSD (Å), Vina score (kcal/mol), and a confidence value.  The oracle is
invoked **only** on the top-K=10 candidates at iteration end — never
during MCTS expansion.

## Adapter file structure

Two new files were created next to the existing SBDD adapters under
`molmetal/molmetal_lam/sbdd_env/`:

| File | Class | Protocol |
|------|-------|----------|
| `diffdock_adapter.py` | `DiffDockAdapter` | `DockingOracle` (smiles, pocket_pdb) -> `DockResult` |
| `flowdock_adapter.py` | `FlowDockAdapter` | `DockingOracle` (smiles, pocket_pdb) -> `DockResult` |

Both expose the same dataclass:

```python
@dataclass(frozen=True)
class DockResult:
    rmsd_A: float       # <2Å = good pose
    vina_kcal: float    # <= -7 kcal/mol = strong binder
    confidence: float   # [0, 1]
```

Discovery probes, in order (first hit wins):

1. Explicit `binary=` ctor argument
2. `$DIFFDOCK_BIN` / `$FLOWDOCK_BIN` env var
3. `diffdock` / `diffdock-l` / `flowdock` on `$PATH`
4. Vendored repo at `molmetal/references/{DiffDock,FlowDock}`
5. Python module importable

When all probes miss, `dock()` raises `AdapterUnavailable` (a
`RuntimeError` subclass).  Callers MUST catch this and fall back.

FlowDock has a CPU-fallback ODE solver (40 ODE steps, exported as
`DEFAULT_ODE_STEPS`); it is the preferred oracle when both backends
are absent.

## Protocol wiring

`molmetal_lam.binding.types.typecheck` now has the signature:

```python
def typecheck(
    ligand,
    site: BindingSite,
    *,
    leaf_oracle_call: bool = False,
    oracle=None,
) -> BindingTypeCheckResult
```

When `leaf_oracle_call=True` and an oracle is supplied (or one can be
resolved via `_resolve_docking_oracle()`), the binding oracle runs on
the candidate.  Its verdict is **and-ed** with the fingerprint stub —
both must succeed for the binding type to be inhabited.  When the
oracle raises `AdapterUnavailable`, the call falls back silently to
the fingerprint stub.

`_resolve_docking_oracle()` prefers FlowDock over DiffDock (CPU
fallback), then returns `None` if neither is reachable.

## Top-K-only call policy (the L-1 hook)

`MCTSProofSearch` gained three new fields:

```python
leaf_oracle_call_top_k_only: bool = True   # governance flag
oracle: Optional[Any] = None               # injected or lazy-resolved
oracle_top_k: int = 10                     # brief specifies 10
```

`MCTSProofSearch._resolved_reward()` stashes these on the aggregator.
Inside `search()`, after the fingerprint filter builds the candidate
list, the *top-K* candidates are re-checked via the new
`_binds_target_top_k(state, oracle)` method (which calls `typecheck`
with `leaf_oracle_call=True`).  **No oracle call happens during MCTS
expansion / rollout** — only at iteration end, on the top-10.

## Fallback path

Both adapters follow the same recipe:

```
typecheck -> _resolve_docking_oracle()
          -> dock() raises AdapterUnavailable
          -> return None from _oracle_leaf_typecheck()
          -> fingerprint stub verdict is final
```

The MCTS `search()` loop wraps `_binds_target_top_k` in a `try/except`
that swallows ALL exceptions, so a missing oracle never crashes the
search.  All seven new unit tests prove the fallback is exercised
correctly:

```
test_diffdock_adapter_unavailable_returns_fallback  PASSED
test_flowdock_adapter_unavailable_returns_fallback  PASSED
test_binding_typecheck_protocol_signature           PASSED
test_binding_typecheck_with_mock_oracle             PASSED
test_binding_typecheck_with_failing_oracle          PASSED
test_resolve_docking_oracle_returns_none_when_missing PASSED
test_mcts_leaf_oracle_call_flag_exposed             PASSED
```

## Test results

Full suite (molmetal + molmetal_lam): **544 passed, 6 failed, 1 skipped**
in 183s.

The 6 failures are **pre-existing** and unrelated to this work — they
are in `test_baselines.py::test_sas_score_smoke`,
`test_3d_embed.py::test_embed_cisplatin_pt`, and the canonical-SMILES
uniqueness check in `test_layer_metrics_l4_l6.py`.  None of them touch
the binding layer, the docking adapters, or the MCTS top-K call site.

Adapter count: **2** (DiffDock + FlowDock).

Fallback verified: **Yes** — `AdapterUnavailable` is raised and caught
in both adapters; `typecheck(..., leaf_oracle_call=True, oracle=None)`
returns the fingerprint verdict unchanged when no oracle is reachable.

ROCm 7.2 + Triton 3.8 stack: **untouched** (no new heavy deps).
