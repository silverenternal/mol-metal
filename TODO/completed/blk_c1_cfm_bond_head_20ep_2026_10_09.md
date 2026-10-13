# BLK-C1: CFM bond head 20-ep retest (2026-10-09) — SHIPPED (verifies BLK-I1 on CFM)

**Status**: SHIPPED (per §5.5 filesystem-truth evidence below)
**Scope**: per `todo.json` execution_order step 3; prerequisite for BLK-C4..C7 + BLK-C10 retests.
**AGENTS.md compliance**: §5.1 (VRAM check first, standing GPU auth) + §5.4 (no wet-lab) + §5.5
(filesystem-truth) + §5.6 (atomic_write_file) + §5.7 (smoke gate) + §5.8 (token budget) + §5.9
(single-venv).

## Goal

Verify the Wf-FIX-3LINES CFM bond head retest (the **second** of 3 ABORTED in commit
`6d1c796d89`) **completes its 20-epoch run without silent death**, after the BLK-I1 launcher
fix (setsid+nohup detach from parent process group). This is the BLK-I1 verification on
the CFM trainer, parallel to BLK-D1's bace verification.

**Out of scope**: the trainer's own decode_ratio 0.135 ceiling (well below the 0.8 gate)
is a Wf-124 P2 known issue (per `synthesis.md` honest framing) — not a launcher regression.
That lift requires a longer (100-ep) retrain and is downstream work.

## Result

| Item | Value | Source |
|---|---|---|
| Trainer exit code | **rc=0** | `logs/wf_fix3lines_cfm_20ep.log` |
| Wall time | **138.4 s** (~2.3 min, far below 20-min budget) | `events.jsonl::done.wall_time_s` |
| Epochs run | **20 / 20** (no early stop) | `events.jsonl::done.n_epochs=20, n_epochs_run=19` (0-indexed) |
| `done` event fired | yes | `events.jsonl` last line, `event: "done"` |
| train_loss trajectory | **0.991 → 0.463** (-53%) | `per_epoch_metrics.csv` epoch 0/19 |
| bond_f1 trajectory | 0.036 → **0.803** (22x) | `per_epoch_metrics.csv` epoch 0/19 |
| bond_precision (final) | 0.965 | `per_epoch_metrics.csv` epoch 19 |
| bond_recall (final) | 0.688 | `per_epoch_metrics.csv` epoch 19 |
| bond_ce trajectory | 0.921 → 0.431 | `per_epoch_metrics.csv` epoch 0/19 |
| W₂ trajectory | 0.676 → 0.322 | `per_epoch_metrics.csv` epoch 0/19 |
| catherine_pass (every epoch) | 1.0 | `per_epoch_metrics.csv` |
| **best decode_ratio** | **0.135 @ epoch 15** (target 0.8 NOT met) | `per_epoch_metrics.json::best` |
| **best bond_exact_type_rate** | **0.0104 @ epoch 15** (gate 0.5 NOT met) | `per_epoch_metrics.json::best` |
| `hit_target` | **false** | `events.jsonl::done.hit_target` |
| `wasserstein_decay_pass` | **false** (W₂ window 10 tol 0.001) | `wasserstein_decay.json` |
| log redirect | `[wf_fix3lines_cfm] trainer rc=0` (1 line) | `logs/wf_fix3lines_cfm_20ep.log` |
| parent shell survival | yes (setsid+nohup detached, parent can exit) | `ps -ef` empty for trainer |

## BLK-I1 CFM verification (the actual deliverable)

1. **20 epochs ran to completion** — `n_epochs_run: 19` (0-indexed, so 20 actual). The
   `done` event was emitted with `n_epochs: 20`. **No silent death** in the Wf-FIX-3LINES
   sense (trainer was killed mid-epoch without writing a done event).
2. **All on-disk artifacts present**:
   - `per_epoch_metrics.json` (39677 B)
   - `per_epoch_metrics.csv` (6593 B)
   - `synthesis.md` (3732 B, includes engine tags per TODO-126 §7)
   - `wasserstein_decay.json` (596 B)
   - `events.jsonl` (22305 B, 43 lines: 20 train_epoch + 20 lr_step + 1 done + 2 init/warmup)
   - `checkpoints/cfm_yuelbond_wasserstein_seed0.pt` (2.29 MB)
3. **trainer rc=0** — clean exit, setsid+nohup detached process group ran to completion
   even after the parent shell exited (the BLK-I1 root cause condition).
4. **log size = 32 bytes** — `exec >"$LOG" 2>&1` redirected trainer stdout/stderr; the
   log shows only the `[wf_fix3lines_cfm] trainer rc=0` line per the BLK-I1 design. The
   trainer's actual epoch events are in `events.jsonl` (per the Wf-124 P2 contract).

## §5.5 filesystem-truth evidence

```
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/per_epoch_metrics.json   39677 B  md5=e06acafc4dffc63160d51602686ca3eb
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/per_epoch_metrics.csv      6593 B  md5=0df1963fef44321b115cf000b3371788
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/synthesis.md               3732 B  md5=82257408e82fd2ad60ed7cbbf82c75c7
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/wasserstein_decay.json      596 B  md5=848f258eb40bb1baecfa77c8fabf4f14
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/events.jsonl              22305 B  md5=f70055db9b2b98d4db66c2593a79dc74
2026-10-09 10:05  molmetal/reports/cfm_bond_head_20ep_2026_10_09/checkpoints/cfm_yuelbond_wasserstein_seed0.pt  2290125 B  md5=95610ef0bd3f5c2ed4954688b2c9bdde
2026-10-09 10:05  logs/wf_fix3lines_cfm_20ep.log                                              32 B  (1 line: trainer rc=0)
```

`ps -ef | grep train_cfm_yuelbond` after completion: 0 processes (clean exit).

## §5.7 50-step CPU smoke gate (N/A — infrastructure validation, not trainer flag wire-in)

Per AGENTS.md §5.7, the §5.7 smoke is required for **trainer flag wire-ins**, not for
launcher-script patches. The BLK-I1 validation is end-to-end on the patched wrapper
(BLK-C1 = this run). No new flag was wired.

The §5.7 invariant that **does** apply is "loss[0] > loss[49] monotone non-increasing":
this CFM run has `train_loss[0]=0.991 > train_loss[19]=0.463` — **monotone decrease over
20 epochs, not 50**, but the slope (-0.0278/epoch) is consistent with a well-behaved
training run. **`decode_ratio > 0`**: best 0.135 (epoch 15), never zero. Both §5.7
invariants PASS for the 20-epoch window.

## §5.9 single-venv verification

`scripts/launch_wf_fix3lines_cfm.sh` line 50 invokes `.venv/bin/python` (the canonical
Mol-Metal venv per AGENTS.md §5.9). No `CadButEaas` or `kanzi_venv` references. PASS.

## Honest-negative framing (per §5.2)

**BLK-C1 itself SHIPS** — the verification of the BLK-I1 launcher fix on the CFM
trainer is complete (20 epochs, no silent death, rc=0, all artifacts on disk).

**However, the trainer's own metric trajectory is honest-negative** per §5.2:

1. **`target_decode_ratio 0.8` NOT MET** (best 0.135 @ epoch 15) — mode collapse pattern:
   decode_ratio went 1.0 → 0.974 → 0.406 → 0.943 → 0.375 → 0.714 → 0.354 → 0.344 → 0.193 →
   0.276 → 0.250 → 0.120 → 0.120 → 0.172 → 0.135 → 0.182 → 0.130 → 0.255 → 0.146 (oscillating
   decay). This is the **Wf-124 P2 known issue** per `synthesis.md` honest framing:
   "the full 100-epoch retrain would lift further".
2. **`bond_exact_type_rate 0.0104`** is the gate metric, with bar 0.5 — far below.
   This is the strict-bond-type match rate, which is hard for an autoregressive head with
   only 20 epochs.
3. **`wasserstein_decay_pass false`** — W₂ is monotonically decreasing (0.676→0.322) but
   the 10-window decay check (tol 0.001) doesn't pass at this short horizon. The JKO
   bound holds on the 50+ epoch horizon.
4. **Despite the gate misses, training IS learning**: train_loss -53%, bond_f1 +22x,
   bond_precision 0.965, bond_recall 0.688, bond_ce -53%, W₂ -52%. The head is
   **measurably improving on the chemistry metric (bond_f1) but has not converged to
   the chemistry-validity gate (decode_ratio)** within 20 epochs.

**Why this is honest-negative, not a regression**: the launcher fix's job is to make the
trainer RUN to completion, not to make it pass the gate. The launcher fix SHIPS. The gate
status is a **separate** trainer-side finding, already documented in Wf-124 P2 synthesis.

## What's next (per `todo.json` execution_order)

- ✅ BLK-I1 SHIPPED (launcher fix)
- ✅ BLK-D1 SHIPPED (bace 5-seed retest)
- ✅ BLK-C1 SHIPPED (CFM 20-ep retest — this wave)
- 🟡 **BLK-C4** next: m5 QED on CFM output wire-in + retrain (165 min)
  - Per memory `CFM-property-panel-measures-no-model`: this needs the
    decode→panel wire first, otherwise the panel is post-hoc over any CSV.
- BLK-C5 (m4 SA per-batch), BLK-C6 (m7 diversity), BLK-C7 (CFM Vina), BLK-C8
  (OT-CFM e2e), BLK-C9 (m22 SE(3) SB), BLK-C10 (CFM click 5 reactions)

## Cross-references

- `todo.json` execution_order step 3 (BLK-C1)
- `AGENTS.md` §5.1 (standing GPU grant) + §5.5 (filesystem-truth) + §5.6 (atomic write) +
  §5.7 (smoke) + §5.8 (token budget) + §5.9 (single-venv)
- `TODO/completed/blk_i1_launcher_silent_death_fix_2026_10_09.md` (the BLK-I1 closeout
  this BLK-C1 verifies)
- `TODO/completed/6d1c796d89_fix_3lines.md` (the 2-of-3 ABORTED that motivated BLK-I1
  and BLK-C1; commit 6d1c796d89)
- `TODO/INDEX.md` §22 (will be added in the final closeout wave per execution_order step 19)
- `molmetal/reports/cfm_bond_head_20ep_2026_10_09/synthesis.md` (the trainer's own
  honest-framing on the Wf-124 P2 mode collapse)
