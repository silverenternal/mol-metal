# WF-Extra-2 — REINVENT4 multiproperty bridge wired into RewardAggregator

Date: 2026-09-14
Author: WF-Extra-2 (WF-Extra-2.A: new adapter; WF-Extra-2.B: aggregator wiring)
Scope: configure task-specific learned multiproperty scoring worker for
REINVENT4 in the isolated ROCm project at
`/mnt/storage/env-projects/reinvent4-rocm/`.  The new
`REINVENT4MultipropertyAdapter` drives REINVENT4's native multi-property
objective (sum of weighted components — logP / ring count / QED / …) and
exposes it as a single float in `[0, 1]`.  The `r_reinvent4` channel of
`RewardAggregator` is now wired to that adapter instead of being silently
off.

Honest-framing key:
* **MEASURED** — produced by an actual run inside this session, on the
  test bench (uv-managed Python 3.12, ROCm 7.2, RX 7800 XT gfx1101).
* **PROJECTED** — extrapolated from the same code paths, but on inputs we
  did not run end-to-end in this session.

---

## 1. Files changed

| Path | Status | Purpose |
|------|--------|---------|
| `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py` | modified | Added `REINVENT4MultipropertyAdapter` (subprocess-RPC mode `multiproperty`) and `from_config` factory.  Pre-existing `REINVENT4Adapter` is the four-component proxy (`prior_nll`-style mode kept as-is). |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | modified | Added `RewardAggregator.register_reinvent4_multiproperty_channel()` that wires the adapter into the `r_reinvent4` channel. |
| `molmetal/configs/reinvent_multiproperty_amd.json` | **NEW** | Configuration consumed by `REINVENT4MultipropertyAdapter.from_config()`. |
| `molmetal/molmetal_lam/tests/test_reinvent4_subprocess_adapter.py` | modified | Added 10 new tests (4 unit + 4 aggregator + 1 from_config + 1 dataclass). |

---

## 2. Adapter diff — `reinvent4_subprocess_adapter.py`

The new `REINVENT4MultipropertyAdapter` is a `@dataclass` that:

* Spawns a long-lived worker subprocess (`worker_python` + `worker_script`).
* Drives the JSON-lines protocol on stdin/stdout.
* Translates the user-supplied `components` dict (QED / logP / ring
  count / …) into REINVENT4's weighted scoring TOML via the worker
  (`reinvent4_multiproperty_jsonl_worker.py`).
* Returns a `list[float]` of the same length as the input batch (one
  weighted-aggregate score per SMILES).
* Caches the failure reason in `last_error` so callers can introspect
  *why* a score is 0.0 without re-deriving it from log scraping.

Notable contract decisions:

1. **One SMILES per request.**  The worker (`handle()`) accepts a single
   string under the `smiles` key.  The adapter loops over the input
   batch and sends one RPC per element.  The result is always the same
   length as the input.
2. **`available` is a structural probe.**  It is set `True` iff
   `worker_python` and `worker_script` both exist on disk.  The actual
   *runtime* health of the subprocess is reported via `last_error`
   after each call.
3. **`from_config()` refuses wrong modes** — the constructor raises
   `ValueError` if the config does not declare
   `"mode": "multiproperty"`.  No silent protocol switching.
4. **Graceful degradation** — `score(smiles)` returns `[0.0]` (single
   element) or `[0.0] * len(batch)` (batch) when the adapter is
   unavailable or the worker returns `ok=false`.  No exceptions ever
   propagate.
5. **Default components** are `{"logp": 0.4, "ring_count": 0.2, "qed":
   0.4}` per the WF-Extra-2 task spec; zero/negative weights are
   silently dropped.

The full diff (relative to the prior four-component proxy file) is
summarised in the docstring at the top of
`reinvent4_subprocess_adapter.py` (the public-API block now lists
`REINVENT4MultipropertyAdapter`).

---

## 3. RewardAggregator diff — `proof_search.py`

Added one new method, `register_reinvent4_multiproperty_channel(adapter,
*, default_components=None)`, right after `register_anticancer_channels`.
The method:

* **No-ops when the adapter is unavailable** (`adapter.available is
  False`) — the existing `_safe(None)` wrapper in `RewardAggregator.__call__`
  already degrades a `None` channel to `0.0`, so the channel stays
  `None` and the search loop never crashes.
* **Installs a self-contained closure** when the adapter is healthy.
  The closure:
  - extracts a SMILES string from the state (accepts both
    `MoleculeClosedTerm`-like objects and plain `str` inputs);
  - calls `adapter.score([smiles])[0]`;
  - returns `0.0` on any exception (preserving the graceful-fail
    contract);
  - clips the result to `[0, 1]` defensively (handles NaN + out-of-range).
* **Does not mutate `w_reinvent4`** — callers stay in control of the
  per-channel weight (default `1.0`).

The historical `r_reinvent4: Optional[Callable]` field is unchanged —
callers that want to wire a custom channel can still set the attribute
directly.  The new method is the *default* / recommended path.

---

## 4. Config — `reinvent_multiproperty_amd.json`

```json
{
  "mode": "multiproperty",
  "worker_python": "/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/python",
  "worker_script": "molmetal/molmetal_lam/sbdd_env/reinvent4_multiproperty_jsonl_worker.py",
  "device": "cuda:0",
  "timeout": 60.0,
  "components": {
    "logp": 0.4,
    "ring_count": 0.2,
    "qed": 0.4
  }
}
```

* `mode` — must be `"multiproperty"`.  The adapter refuses anything else
  with a structured `ValueError` (test: `test_from_config_rejects_wrong_mode`).
* `worker_python` — Python interpreter from the isolated REINVENT4
  ROCm venv.
* `worker_script` — relative to the config directory; resolved
  automatically by `from_config()` so the config is portable across
  checkouts.
* `device` — currently informational only (the worker runs on CPU).
* `timeout` — per-call wall-clock budget; mirrored on every RPC.
* `components` — optional.  Missing/empty values fall back to
  `{"logp": 0.4, "ring_count": 0.2, "qed": 0.4}`.

---

## 5. Test results

Command:

```
uv run pytest -q molmetal/molmetal_lam/tests/ --tb=short -k reinvent4
```

**MEASURED: 27 / 27 reinvent4 tests pass in 11.78 s.**

| Test file | Tests | Pass | Notes |
|-----------|-------|------|-------|
| `test_reinvent4_subprocess_adapter.py` | 16 | 16 | Includes 10 new tests in `TestREINVENT4MultipropertyAdapter` and `TestRewardAggregatorMultipropertyChannel`. |
| `test_reinvent4_multiproperty_worker.py` | 11 | 11 | Worker-level RPC tests, unchanged. |

The 10 new tests:

* `TestREINVENT4MultipropertyAdapter`
  - `test_default_components` — adapter falls back to `logp+ring+qed`.
  - `test_invalid_mode_rejected` — `mode=prior_nll` → `config_invalid`.
  - `test_score_unavailable_returns_zeros` — graceful 0.0 on every
    element.
  - `test_score_empty_batch` — empty list returned for empty input.
  - `test_zero_weight_components_dropped` — `<=0` weights filtered out.
  - `test_from_config_rejects_wrong_mode` — `from_config()` raises
    `ValueError` for the wrong mode.
* `TestRewardAggregatorMultipropertyChannel`
  - `test_reward_aggregator_reinvent4_channel_nonzero` — stubbed adapter
    returns `0.73` → `RewardAggregator` emits `0.73` (no longer the
    historical 0.0 hardcode).
  - `test_reward_aggregator_reinvent4_channel_accepts_str` — plain
    string input is accepted.
  - `test_reward_aggregator_reinvent4_offline_graceful` — adapter with
    `available=False` → channel stays `None` → aggregator emits `0.0`.
  - `test_reward_aggregator_reinvent4_adapter_exception_graceful` —
    `adapter.score()` raising at runtime → aggregator emits `0.0` (no
    crash).

---

## 6. End-to-end smoke (MEASURED)

```
$ PATH=/mnt/storage/env-projects/reinvent4-rocm/.venv/bin:$PATH \
  uv run python -c '...'
available: True last_error: None
ethanol:  [0.5625158053759707] elapsed 4.2s
last_error after call: None
batch:    [0.5625158053759707, 0.5234852222583378] elapsed 7.7s
```

* **MEASURED**: `r_reinvent4(CCO) = 0.5625…` (~4.2 s/call, single-threaded)
  and `r_reinvent4(c1ccccc1) = 0.5235…` — the multiproperty adapter
  drives the *real* REINVENT4 binary in
  `/mnt/storage/env-projects/reinvent4-rocm/.venv/bin/reinvent` and
  returns a weighted aggregate of the configured components
  (`{logp: 0.4, ring_count: 0.2, qed: 0.4}`).
* The single-line stderr `(null): No such file or directory` is a
  non-fatal warning from the REINVENT4 CLI complaining about a missing
  CUDA device file when running on CPU; the score is still produced
  and returned.

---

## 7. Schema for the parent agent

```json
{
  "n_tests": 27,
  "n_passed": 27,
  "smoke_r_reinvent4_for_ethanol": 0.5625158053759707
}
```

* `n_tests` = 27 (16 adapter-level + 11 worker-level under `-k reinvent4`).
* `n_passed` = 27 (all pass, no skips, no xfails).
* `smoke_r_reinvent4_for_ethanol` = 0.5625158053759707 (MEASURED on the
  ROCm project venv, single SMILES, single-process, ~4.2 s wall-clock).

---

## 8. What is *not* in this drop (PROJECTED / DEFERRED)

* **PROJECTED**: a Round-12 N=10×3 sweep comparing `r_reinvent4` off
  vs on (the historical baseline vs the new multiproperty channel) is
  not run here — that belongs to WF-3.
* **PROJECTED**: the closed-loop pipeline
  (`molmetal/orchestration/closed_loop.py:RewardAggregatorScorer`) does
  not yet call `register_reinvent4_multiproperty_channel()` — the
  wiring hook is on the `RewardAggregator` instance, and a 5-line
  patch in `closed_loop.py` is the next step (separate workflow).
* **PROJECTED**: GPU device routing for REINVENT4 is still on CPU
  (REINVENT4's built-in RDKit-style components don't need a GPU); the
  optional `device` field is informational only for now.
* **DEFERRED**: the historical four-component proxy
  (`REINVENT4Adapter` in the same file) coexists with the new
  multiproperty adapter.  No callers in the lam dir have been migrated
  yet; the new path is opt-in via the explicit `from_config()` factory.
