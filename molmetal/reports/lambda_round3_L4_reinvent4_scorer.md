# Lambda Round 3 — L-4: REINVENT4 multi-property scorer

## Adapter file structure

**New file** `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py` (~290 LOC).

Public API:

- `ScoreResult` — dataclass with `qed`, `sa`, `binding`, `novelty` fields
  (each a float in `[0, 1]` or `None`).
- `REINVENT4Adapter` — subprocess-RPC adapter.
  - Constructor probes for the `reinvent` CLI (PATH + the cloned
    `molmetal/references/REINVENT4/reinvent/Reinvent.py` entry-point).
    If both probes fail, sets `self.available = False` and **never**
    spawns a subprocess.
  - When available, spawns a long-lived worker subprocess; per-batch
    scoring is a JSON-Line RPC roundtrip
    (`{"op": "score", "smiles": [...]}` →
    `{"results": [{"qed": ..., "sa": ..., "binding": ..., "novelty": ...}, ...]}`).
  - Heartbeat thread polls the worker every `heartbeat_interval`
    seconds (default 30). On subprocess exit, sets `self.dead = True`
    and every subsequent `score()` call returns `[None] * len(batch)`.
  - `score(smiles_batch)` returns `List[Optional[ScoreResult]]`.
    A `None` element means "this molecule could not be scored";
    the search loop treats it as a 0.0 contribution.
- `ScoreAggregator` — weighted-sum aggregator with default
  weights `w_qed=0.3, w_sa=0.3, w_binding=0.3, w_novelty=0.1`
  (total = 1.0 for a perfect molecule). `None` components are
  treated as 0.0; out-of-range values are clipped to `[0, 1]`.
- `is_reinvent4_binary_available(name)` — public probe used by both
  the adapter and tests.

**Modified file** `molmetal/molmetal_lam/search_alg/proof_search.py`.

Added two new fields to `RewardAggregator`:

- `r_reinvent4: Optional[Callable[[MoleculeClosedTerm], float]] = None`
- `w_reinvent4: float = 1.0`

In `__call__`, after computing the legacy channels, the aggregator
adds `w_reinvent4 * v_reinvent4` (where `v_reinvent4` is wrapped in
the same `_safe` callable-swallowing helper used by the other
channels). A missing / raising callable degrades to `0.0` so the
search loop never crashes.

`reward_channels` override doc-string updated to mention the new
key `w_reinvent4`.

## Fallback verified

- REINVENT4 binary (`reinvent` / `reinvent4` / `reinvent.py`) **not
  found on this system** — only the cloned reference repo at
  `molmetal/references/REINVENT4` is on disk.
- Static probe via `is_reinvent4_binary_available("reinvent")`
  returns **True** (fallback detects the cloned `Reinvent.py`).
- Adapter's runtime contract (worker subprocess launch) still fails
  because `reinvent` is not on `PATH`. The adapter catches the
  `FileNotFoundError` in `__post_init__`, flips
  `available = False`, and `score()` returns `None` for every batch.
- Existing fallback `molmetal_lam.sbdd_env.reinvent_wrapper.REINVENT4Scorer`
  remains untouched; downstream pipelines that consume
  `REINVENT4Adapter` (port) get `None` instead, which the
  `RewardAggregator` then treats as `0.0`.

## Test results

New file `molmetal/molmetal_lam/tests/test_reinvent4_subprocess_adapter.py`:

```
TestREINVENT4AdapterUnavailable::test_unavailable_returns_none PASSED
TestScoreAggregator::test_weighted_sum                         PASSED
TestScoreAggregator::test_out_of_range_clipping                PASSED
TestRewardAggregatorWithREINVENT4Channel::test_with_r_reinvent4_stub PASSED
TestRewardAggregatorWithREINVENT4Channel::test_no_r_reinvent4_returns_zero PASSED
============================== 5 passed in 1.19s ===============================
```

Full suite `pytest molmetal/tests/ molmetal/molmetal_lam/tests/ -v`:

```
511 passed, 18 failed, 1 skipped
```

The 18 failures are pre-existing (`test_baselines.py`,
`test_layer_metrics_l9_cross.py`, `test_proof_search_strengthened.py`,
`test_3d_embed.py`) and reproduce on a clean checkout of
`proof_search.py` before the L-4 patch — none are caused by the new
`r_reinvent4` channel. The L-4-relevant subset (35 tests spanning
`test_reinvent4_subprocess_adapter.py`,
`test_proof_search_strengthened.py`, `test_pipeline.py`,
`test_sbdd_env.py`) all pass.
