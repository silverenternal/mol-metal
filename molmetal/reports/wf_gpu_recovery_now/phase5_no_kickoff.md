# WF-GPU-Recovery-Now — Phase 5 verdict (no path-(a) kick-off)

**Spec rule** (from `WF-CFM-Retrain-Diagnose` decision tree, repeated in
`molmetal/reports/wf_cfm_diagnose_verdict.md` §1):

```
if decode_ratio > 0.5:
    kick off path (a) — WF-CFM-Retrain-Full at 10000-step + hidden-dim 64
elif decode_ratio ∈ (0, 0.5]:
    path (b) — CFM 架构 rework (5-10 d)
else:  # decode_ratio = 0
    path (c) — λ-only column (already shipped)
```

**5000-step probe MEASURED decode_ratio = 0 / 192 = 0.000.**

→ **`decode_ratio > 0.5` is FALSE.**

→ **Path (a) is NOT kicked off. Round-12 stays on path (c) λ-only.**

This is **not** an oversight or an out-of-budget skip — the spec explicitly
gates path (a) on `decode_ratio > 0.5`. Running path (a) now would:

1. burn an estimated 6–12 h of GPU time on a 10000-step + hidden-dim 64
   experiment that the spec author already judged "not worth running" if
   the 5000-step gate fails;
2. violate the user's standing decision (`TODO/pending/21_lambda_model_coupling.md`
   "重训是可以接受的, 但不是现在"), which still applies;
3. introduce a `decode_ratio=0` 10000-step result alongside the 5000-step
   `decode_ratio=0` result without adding decision-relevant evidence (the
   5000-step already shows a clear plateau in bond_loss by step 3500).

## Phase 5 close-out checklist

| task | status |
|---|---|
| Decide kick-off | **NO** (gate failed) |
| Create path-(a) workflow stub | **NOT DONE** (would imply a future run; not warranted) |
| Update `wf_cfm_diagnose_verdict.md` | **DEFERRED** — the verdict is now superseded by `wf_gpu_recovery_now/final.md` §6; a follow-up edit to that earlier report can cross-link. Out of scope for this workflow. |
| Update `wf_cfm_retrain_full/final.md` | **DEFERRED** — same reason; the file's "GPU_BLOCKED" framing is now stale. Out of scope. |
| Update memory bank | **DONE** (this Phase 5 record) |
| Update TODO-21 | **DONE** (Phase 4 — see TODO/pending/21_lambda_model_coupling.md) |

## What this leaves on the table

The 5000-step probe MEASURED a meaningful but discouraging result:

- `bond_loss` trajectory clearly plateaus by step 3500, so further training
  in the same regime is unlikely to break the decode_ratio=0 wall.
- The decode failures are 97.4% connectivity (`disconnected_distance_graph`),
  pointing at *coordinate quality* (the EGNN velocity field outputs
  geometrically disconnected atoms) rather than at the bond head (which
  has saturated to 0.000 bond loss).
- Path (b) (architectural rework: discrete diffusion on atom-types + bond
  tensors jointly, or a stronger EGNN prior) is the **only** remaining
  option that *might* lift decode_ratio off zero at this scope. It is
  5–10 d of effort and out of scope for the next 2 weeks.

## Future-work hooks (not started)

If a future workflow wants to take up path (a) anyway (against the
spec's gate), the minimal scaffolding is:

```
PYTHONPATH=. timeout 43200 uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
  --seeds 42 0 1234 --train-steps 10000 --n-train 32 --n-samples 16 \
  --hidden-dim 64 --n-layers 2 --lr 0.0001 --vocab-mask --bond-head learned \
  --output-dir molmetal/reports/wf_cfm_10000_h64/ \
  --gpu-binary scripts/_fake_vina.sh
```

with the same fix at `molmetal/adapters/flow_matching_lipman/__init__.py:1462-1463`
still required. **NOT executed.**

## Memory-bank note

Updated memory entries (will be applied on next `update-config` cycle):

- `WF-GPU-Recovery-Now verdict 2026-09-15` — 5000-step probe: `decode_ratio=0/192`,
  `bond_loss` plateau at ~step 3500, FAILURE per spec gate, stay on path (c).
- `FlowMatchingLipman device() fix 2026-09-15` — single-line `str(DEFAULT_DEVICE)`
  cast at `molmetal/adapters/flow_matching_lipman/__init__.py:1462-1463`;
  independent of GPU; was blocking the harness pre-flight on every run.
