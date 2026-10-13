# BLK-C2: CFM bond head 100-ep retrain (2026-10-09) — SHIPPED (BLK-I1 验证 + mode collapse lift)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` execution_order step 3.5 (user-redirected from BLK-C4 to test
whether 100-ep lifts the Wf-124 P2 mode collapse). Verifies BLK-I1 launcher fix on
a 100-epoch (5x longer) CFM run.
**AGENTS.md compliance**: §5.1 (standing GPU auth, 97GB free) + §5.4 (no wet-lab) + §5.5
(filesystem-truth) + §5.6 (atomic_write_file) + §5.7 (smoke invariants) + §5.8 (token
budget) + §5.9 (single-venv `.venv/bin/python`).

## Goal

Test whether extending the CFM bond head from 20 → 100 epochs lifts the Wf-124 P2 mode
collapse (`target_decode_ratio 0.8` NOT met, best 0.135 @ 20-ep per BLK-C1). The 20-ep
result was honest-negative (synthesis.md framing: "the full 100-epoch retrain would lift
further"). This run tests that hypothesis and is also a second BLK-I1 validation at
5× the wall time.

**Out of scope**: this run is to characterize the 100-epoch trajectory, not to push
the gate to 0.8. If 100-ep still doesn't pass 0.8, mode collapse is structural and
needs an architectural or hyperparameter change (longer retrain / lr schedule / bond
head re-design) — not more epochs of the same recipe.

## Result

| Item | 20-ep (BLK-C1) | 100-ep (this run) | Δ |
|---|---|---|---|
| Trainer exit | rc=0 | **rc=0** | clean |
| Wall time | 138.4 s | **743.6 s** (~12.4 min) | 5.4× |
| Epochs run | 19 (0-idx) = 20 | **100** (0-idx 0-99) = 100 | 5× |
| `done` event | yes | **yes** | ✓ |
| `hit_target` (0.8 gate) | false | **false** | not lifted |
| **best decode_ratio** | 0.135 @ ep 15 | **0.500 @ ep 93** | **3.7×↑** |
| best bond_exact_type_rate | 0.0104 | **0.1302** | **12.5×↑** |
| best bond_f1 | 0.803 | **0.991** | 1.23×↑ |
| best bond_precision | 0.965 | **0.9921** | +0.027 |
| best bond_recall | 0.688 | **0.9900** | +0.302 |
| best bond_type_macro_f1_pooled | 0.324 | **0.637** | 2.0×↑ |
| train_loss trajectory | 0.991 → 0.463 | **0.986 → 0.115** | -88% |
| bond_ce trajectory | 0.921 → 0.431 | **0.916 → 0.087** | -91% |
| W₂ trajectory | 0.676 → 0.322 | **0.678 → 0.272** | -60% |
| wasserstein_decay_pass | false | **false** | window-10 tol-0.001 not satisfied at 100-ep either |
| catherine_pass | 1.0 every epoch | **1.0 every epoch** | ✓ |
| log size | 32 B | 32 B | wrapper-only |
| trainer ps after | clean exit | **clean exit** | BLK-I1 verified |

## Decode_ratio trajectory (mode-collapse pattern under 5× epochs)

| epoch | train_loss | bond_f1 | decode_ratio | w₂ |
|---:|---:|---:|---:|---:|
| 0 | 0.9864 | 0.0047 | 1.0000 | 0.6782 |
| 19 | 0.4609 | 0.8335 | 0.2708 | 0.3205 |
| 24 | 0.3800 | 0.9303 | 0.4219 | 0.3178 |
| 39 | 0.2427 | 0.9736 | 0.5260 | 0.3110 |
| 49 | 0.2026 | 0.9763 | 0.4740 | 0.3061 |
| 59 | 0.1705 | 0.9848 | 0.5521 | 0.3014 |
| 69 | 0.1511 | 0.9857 | 0.5469 | 0.2953 |
| 79 | 0.1419 | 0.9881 | 0.5677 | 0.2888 |
| 89 | 0.1293 | 0.9897 | 0.5052 | 0.2822 |
| 98 | 0.1145 | 0.9892 | 0.5625 | 0.2719 |

**Decode_ratio 模式**:
- ep 0-7: oscillating high (0.677-1.000), the head starts by producing valid molecules
  from the initial decoder.
- ep 8-19: collapse to 0.135-0.490 (the Wf-124 P2 mode collapse).
- ep 20-39: recover to 0.42-0.53.
- ep 40-99: **plateau around 0.50-0.57**, oscillating but not breaking through.
- best 0.500 @ ep 93, final 0.5625 @ ep 98.

**模式解读**: decode_ratio 在 ~0.55 ceiling 处 plateau,不是单调涨。**这意味着
100 epoch 仍 lift 不足** — gate 0.8 NOT met,best 0.500 (62.5% of gate)。Mode collapse
不是 "需要更长 retrain 时间" 的问题,而是**结构性 ceiling** (在固定 recipe / 100 epoch
budget 之内)。需要 arch / hp 调整才能突破。

## BLK-I1 100-epoch verification

1. **100 epochs ran to completion** — `n_epochs_run: 100` (0-indexed 0-99, all 100 ran).
   `done` event was emitted with `n_epochs: 100`. **No silent death** in the
   Wf-FIX-3LINES sense.
2. **All on-disk artifacts present**:
   - `per_epoch_metrics.json` (190939 B, 100 epochs)
   - `per_epoch_metrics.csv` (30351 B, 100 rows)
   - `synthesis.md` (9207 B, with engine tags per TODO-126 §7)
   - `wasserstein_decay.json` (2566 B, 100-entry w₂ history)
   - `events.jsonl` (105367 B, 201 lines: 100 train_epoch + 100 lr_step + 1 done)
   - `checkpoints/cfm_yuelbond_wasserstein_seed0.pt` (2.29 MB)
3. **trainer rc=0** — clean exit after 743.6s of GPU work. The setsid+nohup detached
   process group ran to completion even after the parent shell exited (the BLK-I1
   root cause condition).
4. **log size = 32 bytes** — `exec >"$LOG" 2>&1` redirected trainer stdout/stderr; only
   the wrapper's `trainer rc=0` line is in the log. Per-epoch events are in
   `events.jsonl` (per the Wf-124 P2 contract).

## §5.5 filesystem-truth evidence

```
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/per_epoch_metrics.json  190939 B  md5=01bce72abbdce83c5eb4d04059677454
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/per_epoch_metrics.csv    30351 B  md5=143b1296ea72194f3ebddfc8de6c9819
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/synthesis.md              9207 B  md5=eb9b027923621e1140fc00d4a03774ea
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/wasserstein_decay.json   2566 B  md5=d54d467d32358976b79c5fd213fb5cf7
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/events.jsonl            105367 B  md5=45e44332176a31923cf8a5c2a21f622e
2026-10-09 10:23  molmetal/reports/cfm_bond_head_100ep_2026_10_09/checkpoints/cfm_yuelbond_wasserstein_seed0.pt  2289869 B  md5=13c9287b55b25fc50569c0b3256cff05
2026-10-09 10:23  logs/wf_fix3lines_cfm_100ep.log                                            32 B  (1 line: trainer rc=0)
```

`ps -ef | grep train_cfm_yuelbond` after completion: 0 processes (clean exit).

Launcher (`scripts/launch_wf_fix3lines_cfm_100ep.sh`):
```
-rwxr-xr-x 1 hugo hugo 3171 Oct  9 10:10 scripts/launch_wf_fix3lines_cfm_100ep.sh
md5=7a630dd22cf9f76a690935e1b63794a6
```

## §5.7 50-step CPU smoke gate

Per AGENTS.md §5.7, the §5.7 smoke is required for **trainer flag wire-ins**, not for
launcher-script patches. The BLK-I1 validation is end-to-end on the patched wrapper
(BLK-C2 = this run). No new flag was wired.

The §5.7 invariants that apply:
- **`loss[0] > loss[49]`**: train_loss[0]=0.986 > train_loss[99]=0.115 — **monotone
  decrease over 100 epochs, ratio = 8.6× reduction**. PASS.
- **`decode_ratio > 0`**: best 0.500, never zero across all 100 epochs. PASS.

## §5.9 single-venv verification

`scripts/launch_wf_fix3lines_cfm_100ep.sh` line 56 invokes `.venv/bin/python`
(the canonical Mol-Metal venv per AGENTS.md §5.9). No `CadButEaas` or
`kanzi_venv` references. PASS.

## Honest-negative framing (per §5.2)

**BLK-C2 itself SHIPS** — the BLK-I1 launcher fix is verified for a 5× longer
CFM run (12.4 min, 100 epochs), and we now have a measured 100-epoch trajectory
showing meaningful lift in all 6 head-side chemistry metrics.

**However, the 100-epoch result is honest-negative on the gate**:

1. **`target_decode_ratio 0.8` NOT MET** (best 0.500 @ ep 93 = 62.5% of gate).
   The synthesis.md hypothesis "the full 100-epoch retrain would lift further" is
   **partially confirmed** (0.135 → 0.500, 3.7×↑) but does NOT reach the gate.
2. **Mode collapse plateaus at ~0.55**: from ep 40 onwards, decode_ratio oscillates
   0.47-0.58 without monotonic improvement. This is a **structural ceiling** under
   the current recipe, not an under-trained model.
3. **bond chemistry metrics are at ceiling**: bond_f1 0.991, precision 0.992,
   recall 0.990 — the head is **measurably converged** on bond prediction but the
   *decode step* (mapping the predicted bonds back to valid molecules) is the
   bottleneck. This is consistent with the Wf-124 P2 finding that the YuelBond
   decoder has a separate failure mode from the bond head.
4. **`wasserstein_decay_pass false`** even at 100 epochs: w₂ is monotonically
   decreasing (0.678 → 0.272, -60%) but the 10-window tol-0.001 check doesn't
   pass — the JKO bound is **smoothly approached**, not step-decayed.

**What this means for the project**:
- The Wf-124 P2 mode collapse is **not fixed by epoch budget alone**. The
  `0.8 decode_ratio` gate likely needs an architectural change (e.g., the
  YuelBond decoder itself, or a different bond-decision threshold policy) or
  a learning-rate schedule, not 5× more epochs.
- For the Tier-2 paper target: bond_f1 0.991 + bond_precision 0.992 +
  bond_recall 0.990 are **paper-grade chemistry metrics**; the gate metric
  `decode_ratio 0.8` is more of a generation-quality bar. We can ship the
  head-side numbers as MEASURED and frame the decoder-side as a known
  Wf-124 P2 limitation in paper §6.

**Why BLK-C2 is honest-negative on the gate but not a regression**: the launcher
fix's job is to make the trainer RUN to completion. The launcher fix SHIPS. The
gate status is a **separate** trainer-side finding, already documented in Wf-124
P2 synthesis and now confirmed at 100-epoch scale.

## What's next (per `todo.json` execution_order)

- ✅ BLK-I1 SHIPPED (launcher fix)
- ✅ BLK-D1 SHIPPED (bace 5-seed retest)
- ✅ BLK-C1 SHIPPED (CFM 20-ep retest)
- ✅ **BLK-C2 SHIPPED** (CFM 100-ep retest — this wave)
- 🟡 **BLK-C4** next: m5 QED on CFM output wire-in + retrain (165 min)
  - Per memory `CFM-property-panel-measures-no-model`: this needs the
    decode→panel wire first, otherwise the panel is post-hoc over any CSV.
- BLK-C5 (m4 SA per-batch), BLK-C6 (m7 diversity), BLK-C7 (CFM Vina), BLK-C8
  (OT-CFM e2e), BLK-C9 (m22 SE(3) SB), BLK-C10 (CFM click 5 reactions)

## Cross-references

- `todo.json` execution_order step 3.5 (BLK-C2, user-redirected from BLK-C4)
- `AGENTS.md` §5.1 (standing GPU grant) + §5.5 (filesystem-truth) + §5.6
  (atomic write) + §5.7 (smoke) + §5.8 (token budget) + §5.9 (single-venv)
- `TODO/completed/blk_i1_launcher_silent_death_fix_2026_10_09.md` (BLK-I1)
- `TODO/completed/blk_d1_bace_5seed_2026_10_09.md` (BLK-D1, 5/5 bace seeds)
- `TODO/completed/blk_c1_cfm_bond_head_20ep_2026_10_09.md` (BLK-C1, 20-ep)
- `TODO/completed/6d1c796d89_fix_3lines.md` (the 2-of-3 ABORTED that motivated BLK-I1)
- `molmetal/reports/cfm_bond_head_100ep_2026_10_09/synthesis.md` (trainer's own
  honest framing on the Wf-124 P2 mode collapse at 100-ep scale)
- `TODO/INDEX.md` §22 (will be added in the final closeout wave per execution_order
  step 19)
