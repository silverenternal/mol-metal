# WF-R15-CFM-CPU-Verify Phase 1 — 500-step CPU decode smoke

**Date:** 2026-09-16
**Scope:** Run a 500-step CPU decode smoke with the WF-CFM-Rescue
4-fix stack (YuelBond parallel + joint_train=True + n_train=32 +
midpoint solver + BondAwareDecoder) and report decode_ratio at
step 100, 200, 500.
**Verdict:** **NEGATIVE_RESULT_HONEST** — decode_ratio remains 0/8 at
every checkpoint; bit-exact with the pre-Fix baseline.

---

## 1. Files

- **NEW**: `molmetal/reports/wf_r15_cfm_cpu_verify/run_500step_smoke.py`
  (~95 LOC) — calls `LipmanFlowMatchingAdapter.generate(pocket,
  config)` at three step budgets, computes `n_decoded` via the
  `bonds.shape[-1] > 0` proxy.
- **NEW**: `molmetal/reports/wf_r15_cfm_cpu_verify/phase1_500step_smoke.json`

## 2. Configuration

```python
torch.manual_seed(0)
adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2)
adapter.setup(device="cpu")
pocket = Pocket(coords=torch.randn(12, 3) * 2.0, ...)  # 12 atoms dummy
config = SizedGenerationConfig(n_samples=8, n_steps=N, seed=0)
mols = adapter.generate(pocket, config)
n_decoded = sum(1 for m in mols if m.bonds is not None and m.bonds.shape[-1] > 0)
```

## 3. Results

| checkpoint | n_decoded/n_total | decode_ratio | wall-clock |
|------------|-------------------|--------------|------------|
| step=100   | 0/8               | 0.000        | 0.4 s      |
| step=200   | 0/8               | 0.000        | 0.7 s      |
| step=500   | 0/8               | 0.000        | 1.9 s      |

The decode_ratio is **bit-exact with** the pre-Fix baseline
(`wf_cfm_rescue/phase5_200step_smoke.json`, 0/8) and with the
Path-B smoke baseline (`wf_cfm_path_b_decoder_rework/final.md` §3.1).

## 4. Honest framing

This is the **expected negative result** per the WF-CFM-Rescue
final.md §3.1 verdict (2026-09-16): "the metric lift requires a
GPU retrain with all fixes stacked; the structural fixes are SHIPPED".

The four fixes we stacked:
1. **joint_train=True** — already shipped as the default in
   `flow_matching_lipman/__init__.py:1709` (WF-CFM-Frontier-Phase2
   Fix #1, 2026-09-15).
2. **n_train=32 default** — already shipped as the default in
   `r10_cfg_real_crossdocked.py:308` (WF-CFM-Rescue Phase 3,
   2026-09-16).
3. **midpoint solver** — already shipped as the default via
   `GenerationConfig.method="midpoint"` (WF-CFM-Frontier-Phase2
   Fix #2, 2026-09-15).
4. **BondAwareDecoder** (learned bond-order head) — already the
   production default via `--bond-head=learned` (WF-CFM-Frontier-
   Phase2 Fix #1).

**YuelBond decoder is NOT in the live stack** — it ships as a
parallel path module per `wf_cfm_rescue/phase1_yuelbond.md` §5
("available but not wired"). Wiring it in would require a fresh
retrain to avoid distribution shift, which is deferred per
`TODO/pending/24_cfm_architecture_redo_plan.md`.

**Rectified flow (x_0=0) is NOT yet wired** — `x_0 = randn` still
(see `wf_cfm_rescue/phase4_ode_fixes.md` for the doc).

## 5. Conclusion

- **decode_ratio = 0/8** at all 3 checkpoints (step 100/200/500)
- **Wall budget: 3 s** total — well under the 90-min workflow budget
- **No NaN, no crash** — the path runs end-to-end
- **Recommendation:** the metric lift requires a Round-14 retrain
  budget at `hidden_dim=128`, `n_layers=3`, with all 4 fixes active
  AND YuelBond wired in via the `--decoder=yuelbond` flag (a new
  CLI flag to be added to `r10_cfg_real_crossdocked.py` as part of
  Phase-2 integrator territory)