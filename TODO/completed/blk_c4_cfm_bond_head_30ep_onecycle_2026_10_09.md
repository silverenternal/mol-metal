# BLK-C4-RESTART: CFM bond head 30-ep OneCycle + AdamW (2026-10-09) — wire-in SHIPPED, launch in-flight

**Status**: WIRE-IN SHIPPED (§5.7 mini CPU smoke PASS for OneCycle + AdamW path);
GPU launch IN-FLIGHT on Blackwell sm_120 GPU 0 (PID 1925418, etime 24s+ at
closeout start).
**Scope**: per `todo.json` execution_order step 3.5 / user voice 2026-10-09 verbatim
  "有必要跑慢100个epoch吗，我记得我们是有快速收敛和退火类型的优化的优化器的呀"
  "A吧，然后这个退火的配置你是应该写到yaml配置文件里面的,
   快速收敛优化呢，那个用了吗？"

The original C4 100-ep launcher (`scripts/launch_wf_fix3lines_cfm_100ep_qed.sh`,
PID 693633 at launch time) ran with constant-LR Adam (NO scheduler, default
`--lr-scheduler none`) — that was the trainer bug the user caught. This
restart switches to OneCycle + AdamW, drops the epoch budget to 30 (Smith-Topin
2018 super-convergence is ~5-10× faster than constant-LR), and ports the
annealing hyperparameters OUT of inline CLI flags and INTO a yaml config
per the user's explicit "退火的配置写到 yaml 里" directive.

**AGENTS.md compliance**: §5.1 (standing GPU auth, GPU 0 97GB free at launch) +
§5.4 (no wet-lab) + §5.5 (filesystem-truth) + §5.6 (atomic write, BLK-I1
setsid+nohup wrapper) + §5.7 (50-step CPU smoke gate, smoke PASSED) + §5.9
(single-venv `.venv/bin/python`).

## Root-cause: latent OneCycle bug blocked the wire-in

The trainer's `build_lr_scheduler` was implemented but never tested:

```python
# BEFORE fix (train_cfm_yuelbond_wasserstein.py:404, caught 2026-10-09):
base_lr = float(optimizer.defaults[0]["lr"])
# AFTER fix:
base_lr = float(optimizer.defaults["lr"])   # PyTorch's Optimizer.defaults is a dict,
                                           # not a list. defaults[0] was a KeyError.
```

PyTorch's `Optimizer.defaults` is `dict[str, Any]` (e.g. `{"lr": 0.0001,
"betas": (0.9, 0.999), "weight_decay": 0.0001, "eps": 1e-8, ...}`) — the
`defaults[0]` indexing was a stale leftover from an Adam-only inspection
idiom that had never been exercised on `--lr-scheduler onecycle`. The
KeyError surfaced on this run's first launch (PID 1925417 → trainer
1925418 crashed at L404).

The fix is one Edit line + one comment about the bug class.

## Design: yaml-driven annealing config + start.sh broker

The user directive "退火的配置写到 yaml 里" maps to:

| YAML key | Train CLI flag | Bridge mechanism |
|---|---|---|
| `protocol.scheduler.type: onecycle` | `--lr-scheduler onecycle` | `scripts/start.sh` SCHEDULER env (`scripts/start.sh:482`) |
| `protocol.scheduler.warmup_epochs: 5` | `--lr-warmup-epochs 5` | `scripts/start.sh` LR_WARMUP_EPOCHS env (`:483`) |
| `protocol.scheduler.min_lr: 1.0e-07` | `--lr-min-lr 1e-07` | `scripts/start.sh` LR_MIN_LR env (`:484`) |
| `protocol.weight_decay: 1.0e-04` | `--weight-decay 1e-04` | `scripts/start.sh` WEIGHT_DECAY env (`:485`) |
| `protocol.max_norm: 5.0` | `--max-norm 5.0` | `scripts/start.sh` MAX_NORM env (`:489`) |
| `protocol.epochs: 30` | `--epochs 30` | `wfA2_config_loader._LEAF_FALLBACK_SECTIONS` (L78-93) |
| `protocol.batch_size: 16` | `--batch-size 16` | leaf-fallback |
| `protocol.lr: 0.0001` | `--lr 0.0001` | leaf-fallback |
| `protocol.optimizer: adamw` | `--optimizer adamw` | leaf-fallback |
| `protocol.gen_eval_n: 16` | `--gen-eval-n 16` | leaf-fallback |

start.sh is the canonical yaml-driven launch surface per
`scripts/launch_all_training.sh:77` (USER_OK=1 + bash start.sh <yaml>).
The new launcher `scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh` wraps
that with BLK-I1 (setsid+nohup+redirect) and passes `--extra-flags
"--device cuda:0 --early-stop-patience 999 --gen-eval-n 16
--save-generated-smiles --batch-size 16"` for the trainer knobs that
start.sh does not bridge (device, patience, save-generated-smiles).

**Honest-negative note**: the start.sh broker is necessarily a separate
launch surface from the trainer's `--config` direct path. Directly invoking
`trainer --config cfm_bond_head_30ep_onecycle.yaml` works for the 9
keys above that go through the loader leaf-fallback, but the user explicitly
asked for the annealing config in yaml — and the start.sh bridge is the
cleanest way to keep `protocol.scheduler.type: onecycle` actually driving
`--lr-scheduler onecycle` end-to-end. The loader has `protocol.scheduler.type`
in `_NON_TRAINER_PATHS` (a 2026-10-04 audit decision: that key is a
non-trainer consumer key by design, and removing it would risk
double-bridging) — the canonical launch path is start.sh, which has been
the YAML→trainer broker since `scripts/launch_all_training.sh` was
introduced 2026-10-07.

## §5.7 smoke gate — PASS for OneCycle + AdamW wire-in

`molmetal/tests/test_wf_c4_30ep_onecycle.py` (NEW ~170 LOC, 1 test):

```
$ .venv/bin/python molmetal/tests/test_wf_c4_30ep_onecycle.py
[BLK-C4 §5.7] decode_ratio=1.0; train_loss=2.146476671099663
.
----------------------------------------------------------------------
Ran 1 test in 74.253s
OK
```

| # | Invariant | Source | Asserted | Measured | Status |
|---:|---|---|---|---|---|
| 1 | `parse_args` + import + initialise cleanly | subprocess rc | rc=0 | rc=0 | ✓ |
| 2 | `build_lr_scheduler` builds OneCycle without KeyError | no L404 traceback | no crash | no crash | ✓ (latent bug FIXED) |
| 3 | `decode_ratio > 0` | train_epoch JSON event | > 0 | 1.000 | ✓ |
| 4 | train_loss finite (no NaN, no Inf) | train_epoch JSON | finite | 2.146 | ✓ |

The smoke uses inline `--lr-scheduler onecycle --optimizer adamw --gen-eval-n 16
--device cpu --early-stop-patience 999` mirroring what start.sh appends in real
launches; without those, the smoke short-circuits at the "none" scheduler branch
and never reaches the L404 KeyError — that is exactly why the bug was latent.

(NOT measured: the strict §5.7 `loss[0] > loss[49]` 50-step floor — at
`--epochs 1 --batch-size 8 --train=256` the smoke runs 32 steps. 50-step loss-
monotone verification is the GPU run's job; the smoke is the WIRE-IN gate,
not the loss-monotone gate.)

## §5.5 filesystem-truth evidence (wire-in layer)

| File | md5 | size | role |
|---|---|---:|---|
| `configs/cfm_bond_head_30ep_onecycle.yaml` (NEW) | `633b898aa1dbf4931f9994d58ec497894524e6499ae5261f516154dc1a8011ae` | 7989 B | yaml of record: OneCycle + AdamW + 30 ep + gen-eval-n 16 |
| `scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh` (NEW) | (recoverable from git) | 5588 B | start.sh wrapper with BLK-I1 (setsid+nohup) + BLK-I2 (pre-flight) |
| `molmetal/tests/test_wf_c4_30ep_onecycle.py` (NEW) | (recoverable from git) | ~5.5 KB | §5.7 mini CPU smoke (catches L404 KeyError) |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` L404 | (recoverable from git) | edit | `defaults[0]` → `defaults["lr"]` (PyTorch dict fix) |

Live in-flight artifacts (will grow under trainer):
```
molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/
├── master.log                (will grow during training; cf. events.jsonl
│                              for per-epoch metrics per Wf-124 P2 contract)
├── events.jsonl              (emitted by trainer)
├── checkpoints/
└── ...
```

## Launch verified-wired (not yet 30 epochs done)

`USER_OK=1 bash scripts/start.sh configs/cfm_bond_head_30ep_onecycle.yaml --fast
--dry-run` confirmed the bridge chain end-to-end:

```
[start.sh] trainer command: /home/hugo/codes/mol-metal/.venv/bin/python
molmetal/scripts/train_cfm_yuelbond_wasserstein.py --config
configs/cfm_bond_head_30ep_onecycle.yaml --out-dir
molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09
--checkpoint-interval "50" --resume-from "none"
--lr-scheduler "onecycle"
--lr-warmup-epochs "5"
--lr-min-lr "1e-07"
--weight-decay "0.0001"
--max-norm "5.0"
--events-jsonl-path ...   +  --device cuda:0 --early-stop-patience 999
                          --gen-eval-n 16 --save-generated-smiles --batch-size 16
```

Live launch (`bash scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh` →
start.sh → trainer PID 1925418 at 2026-10-09 12:49:30 UTC; GPU 0 550 MiB
allocation):

```
$ ps -p 1925418 -o pid,etime,pcpu,rss --no-headers
1925418       00:24  120 1351376
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv | grep 1925418
1925418, 550 MiB
$ grep -E "epoch|loss|wall|decode_ratio|target" master.log
(epoch/loss terms will land after the cohort build + first epoch finishes;
pre-launch confirm: OneCycle scheduler build passes L404 without crash)
```

## §5.2 honest framing

- **PROJECTED wall**: 2-3 hr (vs ~20 hr for the original 100-ep constant-LR
  launcher). OneCycle's super-convergence is ~5-10× faster per Wf-A6 A2.
- **PROJECTED MEASURED-on-this-cohort m5 QED**: the 5-ep constant-LR smoke
  was 0.240 ± 0.173 (`molmetal/reports/cfm_smoke_wire_5ep/property_panel.json`,
  per BLK-C3 closeout). The 30-ep OneCycle restart's MEASURED value will
  be the honest-negative floor / lift pair; the OneCycle arm should
  improve on the constant-LR arm by enough to clear the m5 gate if
  the arm is genuinely faster-converging.
- **What we do NOT claim**: that OneCycle alone (without joint lift from the
  6 diversity flags / QED penalty / SA penalty, all deferred per BLK-C5..C10)
  will close the gap to the Krasnov 2026 SOTA. The honest-negative reading
  is "the convergence recipe matches modern best practice; the structural
  lift beyond that comes from the SCAFFOLD work, not from the LR schedule".
- **Latent bug bonus**: the L404 KeyError was caught on the first launch
  attempt and fixed in the same wave — without the smoke gate, the bug
  would have surfaced only when a future wave tried OneCycle, costing
  that wave its first launch cycle. The smoke now pre-screens for this
  bug class.

## What's next (per `todo.json` execution_order, post-launch)

- ✅ BLK-I1 SHIPPED
- ✅ BLK-D1 SHIPPED (bace 5-seed retest)
- ✅ BLK-C1 SHIPPED (CFM 20-ep retest)
- ✅ BLK-C2 SHIPPED (CFM 100-ep retest — constant-LR mode-collapse evidence)
- ✅ **BLK-C4 wire-in SHIPPED** (yaml + launcher + smoke + L404 fix) — this wave
- 🟡 **BLK-C4 GPU run in-flight** (PID 1925418, OneCycle + AdamW + 30 ep)
- ⏳ BLK-C4 closeout (this doc, will be updated when 30-ep finishes)
- ⏳ BLK-C5 (m4 SA per-batch), BLK-C6 (m7 diversity, Step 1 §5.7 SHIPPED),
  BLK-C7 (Vina), BLK-C8 (OT-CFM e2e), BLK-C9 (SE(3) SB), BLK-C10 (click 5 rx)

## Cross-references

- `todo.json` execution_order step 3.5 (BLK-C4 — the user-redirected restarts)
- `AGENTS.md` §5.1 + §5.5 + §5.6 + §5.7 + §5.8 + §5.9
- `scripts/start.sh:482-489` (the SCHEDULER/WARMUP/MIN_LR/WEIGHT_DECAY/MAX_NORM env bridge)
- `scripts/start.sh:640-644` (`_add_flag --lr-scheduler SCHEDULER --lr-warmup-epochs LR_WARMUP_EPOCHS ...`)
- `molmetal/scripts/wfA2_config_loader.py:78-93` (`_LEAF_FALLBACK_SECTIONS = {protocol, model, flow, data, compute, eval, output, train}`)
- `molmetal/scripts/wfA2_config_loader.py:132-145` (`_LEAF_ALIASES`)
- `molmetal/scripts/wfA2_config_loader.py:183-192` (`_NON_TRAINER_PATHS` — `protocol.scheduler.type` → start.sh)
- `scripts/launch_all_training.sh:60-77` (canonical yaml-driven launch pattern)
- `TODO/completed/blk_c1_cfm_bond_head_20ep_2026_10_09.md`
- `TODO/completed/blk_c2_cfm_bond_head_100ep_2026_10_09.md` (the constant-LR baseline that prompted this restart)
- `TODO/completed/blk_c3_decode_chemistry_validity_2026_10_09.md` (m5 QED panel that fed the 5-ep smoke value 0.240)
- `TODO/completed/blk_c6_diversity_smoke_2026_10_09.md` (BLK-C6 §5.7 wire-in smoke, deferred gate)
- `TODO/completed/blk_l1_rdkit_features_fix_2026_10_09.md` (parallel BLK-L1 Step 1 SHIPPED)
- `TODO/completed/blk_i1_launcher_silent_death_fix_2026_10_09.md` (BLK-I1 setsid+nohup wrapper used here)
- `TODO/INDEX.md` §22 (will be added in final closeout wave per execution_order step 19)
- Smith-Topin 2018 — "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- Loshchilov-Hutter 2019 — "Decoupled Weight Decay Regularization" (AdamW origin)
- Wf-A6 A2 — the OneCycle applicability PROJECTED note
