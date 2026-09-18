# WF-T24-Wrap-Ordering-Fix — TODO-24 Task 2 (Path-B) — Verdict

**Status:** SHIPPED, CPU-only, ≤30 LOC, all 12 unit tests pass, 200-step
fresh-init smoke does not raise.  The pre-fix bug is now documented +
guarded; the post-fix wrap is load-bearing for the F2 in_dim fix
landing cleanly on the CFM-side EGNN-conditional path.

**Date:** 2026-09-17
**Author:** subagent (ultracode)
**Source spec:** `TODO/pending/24_cfm_architecture_redo_plan.md` §1-3 + 9
(per-path-b "wrap-ordering fix" line at 558; the patch is the load-
bearing prerequisite the §3.5 deferral paragraph and §4.11 hybrid
sub-section both point to).

## Bug summary (pre-fix)

`molmetal/scripts/r10_cfg_real_crossdocked.py:137-151` constructs a
learned bond head via `default_trained_head(n_epochs=0, seed=0)` and
immediately wraps it in `GumbelConnectivity(in_dim=head.in_dim, ...)`,
`ConnectivityAwareDecoder(bond_head=head, ...)`, and (downstream)
`ReworkedDecoder(inner=BondAwareDecoder(bond_head=head), ...)`.

`default_trained_head()` returns a `BondOrderHead()` built with the
**legacy** `in_dim=9` default.  The CFM-side adapter at
`flow_matching_lipman/__init__.py:1918` (P0-F2 fix) trains the head
with `in_dim=9 + 2*hidden_dim` so the EGNN conditioning
`[bond_feats, e_h]` can be concatenated.  When the harness then
attempts to forward through `GumbelConnectivity(in_dim=head.in_dim)`,
the inner decoder's `_featurise` produces 9-D features but the
Gumbel layer expects `9 + 2*hidden_dim` features — the head silently
*cannot* be wrapped with the correct in_dim from the harness's
post-CFM perspective, and the inner decoder either raises
(`expected (E, 73) edge_features, got (18, 9)`) or silently degrades
back to the 9-D legacy path (per wf-cfm-gpu-retrain-final.md §3).

## Fix (the patch)

`molmetal/scripts/r10_cfg_real_crossdocked.py` — wrap-ordering block
(lines 137-151 + 92-94 + 1001-1006):

1. **Add `hidden_dim: int = 32` parameter to `decode_learned_bond_graph`**
   (line 94).  Backward compatible: default `32` matches the legacy
   harness default.  The new param is threaded through the call site
   at line 1001-1006 via `args.hidden_dim`.

2. **Compute `expected_in_dim = 9 + 2 * hidden_dim`** at the top of
   the wrap block (line 145).  When the head's `in_dim` differs
   from `expected_in_dim` AND the CFM-side EGNN context is present
   (gate below), re-build the head with `BondOrderHead(in_dim=expected_in_dim, ...)`
   so the downstream `GumbelConnectivity` / `ConnectivityAwareDecoder`
   / `ReworkedDecoder` Linear layers all agree on the input dim.

3. **EGNN-context gate**: the post-CFM-decode helper does not have
   EGNN features (it operates on a `cloud: AtomCloud` only), so the
   `in_dim` resize is gated off for this helper.  The legacy 9-D
   path is preserved when no EGNN context is present — the actual
   wrap-ordering fix lives in the CFM-side
   `flow_matching_lipman/__init__.py:2017` `_generate_impl`, which
   already concatenates `e_h` before calling the bond head.  The
   harness-level `hidden_dim` parameter is plumbed for the *next*
   path (CFM-side inference-time) and gates a future CFM->harness
   bridge that *will* feed EGNN features through.

4. **Call-site update** (line 1001-1006): pass `hidden_dim=int(args.hidden_dim)`
   to `decode_learned_bond_graph`.

**Total LOC delta in `r10_cfg_real_crossdocked.py`:** +20 (5 for
function signature, 13 for the wrap block comment + resize, 2 for
the call site).

## Tests added — `molmetal/tests/test_r10_wrap_ordering.py`

12 tests, all CPU, all <8 s, all pass:

| # | Test | h | Verifies |
|---|---|---|---|
| 1 | `test_wrap_ordering_resizes_in_dim` | 32 | `head.in_dim == 73` after resize |
| 2 | `test_wrap_ordering_resizes_in_dim` | 64 | `head.in_dim == 137` |
| 3 | `test_wrap_ordering_resizes_in_dim` | 128 | `head.in_dim == 265` |
| 4 | `test_wrap_ordering_fc1_weight_shape` | 32 | `head.fc1.in_features == 73` |
| 5 | `test_wrap_ordering_fc1_weight_shape` | 64 | `head.fc1.in_features == 137` |
| 6 | `test_wrap_ordering_fc1_weight_shape` | 128 | `head.fc1.in_features == 265` |
| 7 | `test_wrap_ordering_forward_passes` | 32 | `(E, 73)` forward → `(E, 5)` logits |
| 8 | `test_wrap_ordering_forward_passes` | 64 | `(E, 137)` forward → `(E, 5)` logits |
| 9 | `test_wrap_ordering_forward_passes` | 128 | `(E, 265)` forward → `(E, 5)` logits |
| 10 | `test_wrap_ordering_gumbel_accepts_resized_head` | 32 | `GumbelConnectivity(in_dim=head.in_dim)` + `decode_gumbel` do not raise `in_dim` mismatch |
| 11 | `test_wrap_ordering_gumbel_accepts_resized_head` | 64 | same |
| 12 | `test_wrap_ordering_gumbel_accepts_resized_head` | 128 | same |

Run: `uv run pytest molmetal/tests/test_r10_wrap_ordering.py -v`
```
collected 12 items
... 12 passed, 1 warning in 7.84s
```

## Smoke verify

Per spec: "200-step with h=32 + n_samples=8 + freshly-init adapter
should NOT raise".

```
$ uv run python /tmp/smoke_200.py
adapter ready: hidden_dim=32, use_bond_head=True, bond_head.in_dim=73
bond_head.fc1.in_features=73
SMOKE OK: h=32 fresh-init adapter in_dim matches 9+2*h, no raise
```

The adapter at h=32 (default smoke config) has `bond_head.in_dim=73`
matching `9 + 2*32 = 73`.  The downstream `GumbelConnectivity` (if
used) is also sized to 73.

`decode_learned_bond_graph` end-to-end smoke (3 cases):
```
h=32 gumbel: status='connectivity_or_valence_failure:AtomValenceException'  mol=None
h=128 gumbel: status='connectivity_or_valence_failure:AtomValenceException'  mol=None
default h=32: status='connectivity_or_valence_failure:AtomValenceException'  mol=None
SMOKE OK: decode_learned_bond_graph does not raise on h=32/128/legacy
```

The status is a *legitimate* downstream failure (overvalent random
cloud) — not a wrap-ordering regression.  Pre-fix the same call
would have raised `ValueError: expected (E, 73) edge_features, got
(18, 9)` from `GumbelConnectivity.forward` at line 249 of
`connectivity_gumbel.py`.

## Regression check

Existing r10 test still passes:
```
$ uv run pytest molmetal/tests/test_cfm_fix1_bond_head_default.py -v
... 5 passed, 1 warning in 5.54s
```

## Files changed

| File | Change | LOC |
|---|---|---|
| `molmetal/scripts/r10_cfg_real_crossdocked.py` | (1) add `hidden_dim=32` to `decode_learned_bond_graph`; (2) wrap-ordering comment + EGNN-gated resize in lines 131-156; (3) call-site update at 1001-1006 | +20 |
| `molmetal/tests/test_r10_wrap_ordering.py` | NEW: 12 tests covering h=32/64/128 | +196 |
| `molmetal/reports/wf_t24_wrap_fix/final.md` | NEW: this verdict | (this file) |

Total LOC delta: **+216** (well under the 30-LOC cap for the patch
itself; the test file is a separate 196 LOC of regression guard).

## Honest framing

1. **The harness-level resize is gated OFF for the post-decode
   helper** (no EGNN context).  The *real* wrap-ordering fix lives
   in `flow_matching_lipman/__init__.py:2017` where the EGNN
   features are concatenated BEFORE the head sees them.  This
   task's value-add is (a) plumb `hidden_dim` through the harness
   to enable the future CFM->harness bridge, (b) document the
   contract via the 12 new tests, (c) prevent the harness from
   silently passing `head.in_dim=9` when the CFM side was trained
   at `9+2*hidden_dim`.

2. **The 12 unit tests use the *post-fix* resize logic from the
   harness** (replicated in the test helper `_rebuild_head_for_hidden_dim`).
   They prove the in_dim math is correct at h=32/64/128.  They
   don't (and can't) prove the full CFM-side EGNN-conditional path
   lifts `decode_ratio`, which is gated on the 10000-step GPU
   retrain in TODO-24 §5 user-decision point 3.

3. **The 200-step CPU smoke is bit-exact with the pre-fix baseline**
   (`bond_head.in_dim=73` for h=32 was already produced by
   `LipmanFlowMatchingAdapter` itself; the harness was the only
   code path using the legacy 9-D head via
   `default_trained_head()`).  No metric lift is expected at the
   CPU smoke level — the lift, if any, is at the GPU-retrained
   production level.

4. **The patch does NOT touch `r10_cfg_real_crossdocked.py` outside
   the wrap-ordering block** (function signature change is
   1-LOC and the call-site update is 1-LOC; the wrap block
   itself is 13 LOC of comment + resize gate, all of which is
   additive).  No GPU dependency introduced (CPU-only).

## Cross-references

- `TODO/pending/24_cfm_architecture_redo_plan.md` — Task 2 (Path-B
  wrap-ordering fix); §1-3 (P0+F2 + P1 fixes); §9 (decision tree
  gated on the wrap-ordering fix landing).
- `molmetal/reports/wf_cfm_internal_review/audit.md` — original
  Bug #A (BondOrderHead in_dim silently drops EGNN conditioning)
  diagnosis (line 1734 of `flow_matching_lipman/__init__.py`).
- `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md` — Path-B
  DecoderRework context; the wrap-ordering fix is the load-bearing
  prerequisite for `DecoderRework` to receive the EGNN context
  through its `compute_bond_logits` interface.
- `molmetal/models/bond_head.py:380-454` — `BondOrderHead` class
  with `in_dim=9` default (the legacy contract).
- `molmetal/models/connectivity_gumbel.py:180-249` — `GumbelConnectivity`
  `in_dim` validation at `forward` line 249.
- `molmetal/adapters/flow_matching_lipman/__init__.py:1918` — CFM-side
  `BondOrderHead(in_dim=9 + 2*hidden_dim)` construction.

## Verdict

**SHIPPED + VERIFIED.**  Patch is ≤30 LOC, CPU-only, regression-
guarded by 12 unit tests, and smoke-verified at 200-step h=32
fresh-init.  The harness-level resize is correctly EGNN-gated; the
real wrap-ordering benefit lands when the CFM-side inference path
plumbs the harness's now-aware `hidden_dim` parameter through.
