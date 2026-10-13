# BLK-C4 FINAL RESULTS — CFM bond head 30-ep OneCycle + AdamW (2026-10-09)

**Status**: ✅ **RUN COMPLETE** (30/30 epochs, wall=3994.7s = 66.6 min, GPU 0 cuda:0).
**Owner**: BLK-C4 wire-in (`TODO/completed/blk_c4_cfm_bond_head_30ep_onecycle_2026_10_09.md`)
+ this FINAL closeout capturing the MEASURED-on-disk numbers.

**AGENTS.md compliance**: §5.1 (standing GPU auth) + §5.2 (honest framing; this doc)
+ §5.4 (no wet-lab) + §5.5 (filesystem-truth; all numbers from
`molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/per_epoch_metrics.csv` +
`events.jsonl` + `synthesis.md` + `wasserstein_decay.json`) + §5.6 (atomic write via
`atomic_write_file`) + §5.7 (50-step CPU smoke gate, PASS) + §5.8 (wf token budget,
direct edits not wf) + §5.9 (single-venv `.venv/bin/python`).

## §1 Final MEASURED outcomes (filesystem-truth, per §5.5)

| Metric | Value | Source |
|---|---:|---|
| Wall total | **3994.7s (66.6 min)** | `synthesis.md::Total wall time` |
| Epochs run | 30/30 | `events.jsonl::event=done,n_epochs_run=30` |
| Best decode_ratio | **1.000 @ epoch 0** | `events.jsonl::best_decode_ratio=1.0,best_epoch=0` |
| Target decode_ratio | 0.80 | `synthesis.md::Target` |
| `hit_target` (per synthesis) | **YES** (epoch 0-3 = 1.000) | `events.jsonl::hit_target=true` |
| Final epoch (29) decode_ratio | 0.188 (36/192) | `events.jsonl::event=train_epoch,epoch=29` |
| Final epoch (29) train_loss | 0.8443 | `events.jsonl::event=train_epoch,epoch=29` |
| Final epoch (29) bond_ce | 0.8077 | same |
| Final epoch (29) w2 | 0.3611 | same |
| W₂ decay check (window=10) | **PASS** | `wasserstein_decay.json::decay_pass=true` |

## §2 4-band honest framing (per memory `no-self-made-tables`)

The `hit_target=true` headline is **misleading** when read on its own. The
4-band table below makes the mode-collapse pattern visible:

| Band | n | decode_ratio | gen_validity | bond_f1 | bond_pr_auc |
|---|---:|---:|---:|---:|---:|
| ep 0 (warmup) | 1 | **1.000** | **0.979** | 0.454 | 0.445 |
| ep 1-3 | 3 | **1.000** | **0.646** | 0.488 | 0.489 |
| ep 4-9 | 6 | 0.840 | **0.000** | 0.536 | 0.524 |
| ep 10-19 | 10 | 0.357 | **0.000** | 0.587 | 0.579 |
| ep 20-29 | 10 | 0.221 | **0.002** | 0.638 | 0.665 |

Reading: OneCycle brought the early-epoch (0-3) decode_ratio to 1.000, but
**starting at epoch 4 the generation head mode-collapses** (gen_validity =
0.000 over n=48 generated samples). The collapse persists through epoch 29
and never recovers. Meanwhile **bond_f1 and bond_pr_auc keep climbing**
(0.45 → 0.64 / 0.45 → 0.67), so the bond prediction head is still
learning — the network has split: bond classifier fits, generation
decoder collapses.

This is the **same mode-collapse class** the BLK-C2 100-ep constant-LR
run produced — OneCycle is ~5× faster at training (66.6 min vs ~10 hr
projected for 100 ep), but does NOT escape the collapse. Per
`docs/HONEST_NEGATIVE_CATALOG.md` §6.38a (W2 is anti-overfitting regulariser,
not AUC lever) the issue is upstream of the LR schedule: the YuelBond
decoder head is being pulled by the W₂ term into a degenerate
bond-only regime that can predict bond type but cannot sample a
valid molecule. AGENTS.md §5.2 mandates this honest framing.

## §3 W₂ decay check (passing but uninformative)

```
w2_history (30 epochs, OneCycle+AdamW): 0.7640 → 0.7114 → 0.6425 → 0.5458 →
0.4205 → 0.3876 → 0.3719 → 0.3723 → 0.3672 → 0.3670 → 0.3657 → 0.3648 → 0.3646 →
0.3637 → 0.3642 → 0.3631 → 0.3632 → 0.3624 → 0.3629 → 0.3622 → 0.3624 → 0.3624 →
0.3617 → 0.3616 → 0.3615 → 0.3616 → 0.3617 → 0.3616 → 0.3619 → 0.3611
decay_window=10, decay_tol=0.001, decay_pass=true
```

W₂ decays monotonically (with float noise) from 0.7640 → 0.3611 over
30 epochs. The decay check passes. But per memory
`w2-wasserstein-is-not-load-bearing.md`, W₂ is an **anti-overfitting
regulariser**, not an AUC lever — the W₂ decay passing is consistent
with what we measured, not a counter-signal.

## §4 Wall clock vs PROJECTED

| Stage | PROJECTED (closeout 1) | MEASURED (this doc) | delta |
|---|---:|---:|---:|
| C4 30-ep OneCycle | 2-3 hr | **66.6 min** | -1.5-2 hr (faster) |
| Per-epoch wall | ~5 min | ~62s (epoch 1+), 1693s (epoch 0 cohort build) | 5× faster |

Epoch 0's 1693s (28 min) is the **scaffold + chemistry cohort build
(Chemprop, RDKit, torch_geometric first-time imports + 192-cell
scaffold table build)** — this is a one-shot amortised cost; all later
epochs run at 60-70s/epoch. Per-epoch PROJECTED in the wire-in
closeout was ~5 min/epoch which over-estimated; the MEASURED is
~63s/epoch. The total wall PROJECTED was 2-3 hr which assumed 5
min/epoch for 30 epochs; the actual 66.6 min is dominated by
epoch 0 + 30 × 63s.

## §5 What the C4 wave SHIPPED (wire-in + run)

| Deliverable | State | Evidence |
|---|---|---|
| `configs/cfm_bond_head_30ep_onecycle.yaml` | ✅ SHIPPED | 7989 B, `md5=633b898aa1dbf4...` (closeout 1) |
| `scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh` | ✅ SHIPPED | 5588 B (closeout 1) |
| `molmetal/tests/test_wf_c4_30ep_onecycle.py` | ✅ SHIPPED | ~5.5 KB (closeout 1) |
| `train_cfm_yuelbond_wasserstein.py` L404 fix (`defaults[0]` → `defaults["lr"]`) | ✅ SHIPPED | caught the latent KeyError; smoke pre-screens for it |
| 30-epoch run on cuda:0 (wall 66.6 min) | ✅ SHIPPED | `events.jsonl::event=done` |
| `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/` | ✅ on disk | `master.log` 93790 B, `events.jsonl` 66127 B, `synthesis.md` 4584 B, `per_epoch_metrics.csv` 11792 B, `wasserstein_decay.json` 866 B, `checkpoints/cfm_yuelbond_wasserstein_seed0.pt` 7800757 B (7.8 MB) |
| mode-collapse diagnostic (per §2 above) | ✅ MEASURED-on-disk | `per_epoch_metrics.csv::gen_validity` 0.000 ep 4-29 |

## §6 What this means for the next wave (per AGENTS.md §5.2 honest framing)

- The **architectural wire-in** is intact: YuelBond decoder, W₂
  regulariser, Catherine closure-bounded generation all wired end-to-end
  (synthesis.md engine tags confirm). The 30-ep run on cuda:0 produces
  finite loss, finite W₂, finite gradients (grad_norm 5-10/epoch
  throughout).
- The **training regime** still produces mode-collapse at the
  generation head, regardless of LR schedule (constant-LR 100 ep from
  BLK-C2 and OneCycle 30 ep from this run both collapse the same way).
- The honest-negative reading: **mode-collapse is upstream of the LR
  schedule** — likely in the W₂-vs-decode-ratio coupling
  (per `docs/HONEST_NEGATIVE_CATALOG.md` §6.38a) or in the YuelBond
  head's bond-vs-sample trade-off. The fix-stack candidates are the
  6 diversity flags (BLK-C6), the SA penalty (BLK-C5), and the
  QED penalty (BLK-C4-target-m5), all of which are deferred per
  todo.json execution_order step 4-7.
- **Do NOT cite C4's `hit_target=true` as a generation-quality claim**
  in any paper section — the headline is misleading on its own; the
  4-band table is the honest frame.

## §7 What's next (per `todo.json` execution_order, post-C4)

C4 GPU is now free. Per the user's standing GPU grant (§5.1) and the
dmpnn-line question, the next-wave candidates are:

| Block | Wall | GPU? | Source of next-step metric |
|---|---:|---|---|
| BLK-D1 Step 4 (5-seed bace AttentiveFP retest) | 40 min | yes (cuda:0) | `molmetal/reports/blk_d1_attentivefp_5seed/metrics.json` |
| BLK-D2 Step 3 (4-metal m19 drop-Rh GPU) | 75 min | yes (cuda:0) | `molmetal/reports/blk_d2_drop_rh_4metal/metrics.json` |
| BLK-D2 closeout §4 (5-seed bace chemprop_v3 frozen head, CPU-only) | 30 min | no (CPU OK) | `molmetal/reports/blk_d2_chemprop_v3_frozen_head_5seed/metrics.json` |

The user's prior message asked "todo.json 里不冲突的 dmpnn 线的任务能不能
做起来". C4 was the previous blocker; C4 is now done so all three
above are "not conflicting with C4". User decision pending.

## §8 Cross-references

- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/master.log` —
  93.8 KB human-readable trainer stdout (per-epoch narrative)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/events.jsonl` —
  66.1 KB structured per-epoch events (canonical machine-readable)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/per_epoch_metrics.csv` —
  11.8 KB flat CSV (the source of §2 4-band table)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/synthesis.md` —
  4.6 KB human-readable closeout (auto-generated by trainer)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/wasserstein_decay.json` —
  866 B W₂ history + decay check
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/checkpoints/cfm_yuelbond_wasserstein_seed0.pt` —
  7.8 MB final state_dict
- `TODO/completed/blk_c4_cfm_bond_head_30ep_onecycle_2026_10_09.md` —
  wire-in closeout (pre-launch)
- `TODO/completed/blk_c1_cfm_bond_head_20ep_2026_10_09.md` —
  prior 20-ep baseline (also hit_target=true on the same misleading
  axis; same ep 4+ mode-collapse pattern, smaller window)
- `TODO/completed/blk_c2_cfm_bond_head_100ep_2026_10_09.md` —
  100-ep constant-LR baseline (the original mode-collapse evidence)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.38a (W₂ anti-overfitting, not AUC lever)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.38c (class-weight refuted on bace)
- `AGENTS.md` §5.1 + §5.2 + §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `todo.json` execution_order step 3.5 (BLK-C4 — the user-redirected restarts)
- Smith-Topin 2018 — Super-Convergence (OneCycle applicability
  was a PROJECTED lift; the actual MEASURED is "faster convergence
  to the same mode-collapse floor")
- Loshchilov-Hutter 2019 — AdamW origin
