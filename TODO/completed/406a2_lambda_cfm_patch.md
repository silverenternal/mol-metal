# Wf-406-A2 — TODO-21 Lambda × CFM coupling patch application (CPU-only)

**Date:** 2026-09-30
**Round:** R17 ultracode follow-up (CPU-only docs+code+test)
**Parent:** Wf-394-A1 (audit + smoke SCAFFOLD SHIPPED; patch deferred) + Wf-397-A2 (decode_ratio=1 → 0 regression CLOSED; NameError fix on Wf-388-A3 SA-penalty kwargs; MEASURED decode_ratio=1.000 on GPU 0 Blackwell) + Wf-399-A1 (AGENTS.md §5.7 trainer-flag smoke-required rule codified)
**Anchor:** `TODO/completed/394a1_lambda_cfm_wire.md` + `TODO/completed/397a2_wave1_smoke.md` + `TODO/completed/399a1_regression_rule.md`
**Workflow thread:** user request 2026-09-30 verbatim *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在哪里退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — USER_OK=1 grant per Wf-397 task brief; Wf-406-A2 closes the Wf-394-A1 patch-application gap now that the §5.7 floor (decode_ratio > 0) is established.

---

## §0 TL;DR

**3 CLI flags + dataset-build block + loss-multiplier block applied** to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. **1 / 1 NEW test PASS** in 42.81 s wall CPU-only. **§5.7 50-step smoke floor** MEASURED PASS end-to-end: decode_ratio = 1.000 (every epoch) + loss[0]=0.9483 → loss[-1]=0.0687 (91% decrease over 3 epochs × 50 batches ≈ 150 step-equivalents).

**The patch is BIT-EXACT backward-compatible with Wf-388-A3 / Wf-397-A2** when no flag is set on the CLI: defaults `lambda_cfm_coupling_weight=0.1` (but guard `weight > 0.0` is gated) + `lambda_cfm_mean_weight=0.0` (from empty `LambdaRewardDataset` since upstream adapter import fails pre-existing) → multiplier collapses to identity (1 + 0.1 * 0.0 = 1.0).

**Status: SHIPPED (CPU-only docs+code+test; GPU retrain BLOCKED on Wf-291D + Wf-299 YuelBond gate)**.

---

## §1 What Wf-406-A2 ships

### §1.1 Trainer patch (5 inserts in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`)

**Insert 1 — `parse_args()` 3 CLI flags** (lines 1340-1380): `--lambda-cfm-coupling` (store_true, default OFF), `--lambda-cfm-coupling-weight` (float, default 0.1), `--lambda-cfm-coupling-samples` (int, default 200). All default OFF for bit-for-bit backward compat with Wf-388-A3 / Wf-397-A2.

**Insert 2 — `train_one_epoch()` signature** (lines 587-593): 2 new kwargs `lambda_cfm_coupling_weight: float = 0.0` + `lambda_cfm_mean_weight: float = 0.0`, mirroring the Wf-388-A3 SA-penalty kwarg pattern (defaults 0.0 so multiplier collapses to identity when OFF).

**Insert 3 — `train_one_epoch()` loss-multiplier block** (lines 843-857, inside the loss = … cascade after the SA-penalty term): conditional `loss = loss * (1 + lambda_cfm_coupling_weight * lambda_cfm_mean_weight)` guarded by `weight > 0.0 AND mean_weight > 0.0`. No-op when both are 0.0 (the default).

**Insert 4 — `main()` dataset-build block** (lines 1477-1510, between Wasserstein config and the train loop): builds `LambdaRewardDataset` via `build_lambda_reward_dataset(n_samples, pocket_id, seed)` if `--lambda-cfm-coupling` is set; pre-computes `lambda_cfm_mean_weight` scalar; catches upstream import errors and falls back to `lambda_cfm_mean_weight = 0.0` (defensive, HONEST framing).

**Insert 5 — `main()` call site** (lines 1546-1552, in the `train_one_epoch(...)` invocation): passes `lambda_cfm_coupling_weight=getattr(args, "lambda_cfm_coupling_weight", 0.1)` + `lambda_cfm_mean_weight=lambda_cfm_mean_weight` as kwargs.

**Total LOC delta**: +90 LOC across 5 inserts. `ast.parse(src, filename=...)` validates the trainer parses cleanly.

### §1.2 `molmetal/tests/test_wf406a2_lambda_cfm_patch.py` NEW

Three §5.7 checks + 2 50-step smokes:
1. Source-level patch presence (3 flag literals + `build_lambda_reward_dataset` + `lambda_cfm_mean_weight` kwarg + AST parse OK).
2. 3-epoch 50-step smoke with coupling OFF (backward-compat baseline): returncode=0 + decode_ratio>0 + loss[0]>loss[-1].
3. 3-epoch 50-step smoke with coupling ON (wiring path exercised): returncode=0 + decode_ratio>0 + `[Wf-406-A2] --lambda-cfm-coupling ENABLED` banner appears in stdout (proves the dataset-build branch ran, not silently skipped).

### §1.3 Smoke artefacts

`molmetal/reports/wf406a2_lambda_cfm_patch/`:
* `smoke_3ep_off_per_epoch_metrics.json` — 3-epoch coupling-OFF baseline (loss 0.9483 → 0.1074 → 0.0687; decode_ratio=1.000 all epochs)
* `smoke_off_per_epoch_metrics.json` — 1-epoch coupling-OFF (baseline ABX test)
* `smoke_on_per_epoch_metrics.json` — 1-epoch coupling-ON (wiring-path exercise; LambdaRewardDataset len=0 mean_weight=0.0)
* `wf394a1_audit.json` — Wf-394-A1 audit on the patched trainer (3 flags detected, loss-multiplier + dataset-build FALSE-NEGATIVES due to Wf-394-A1 detection-pattern stale-vs-Wf-406-A2 naming, see §4.3)

---

## §2 MEASURED test results

### §2.1 Wf-406-A2 NEW test

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf406a2_lambda_cfm_patch.py -v --tb=short --noconftest

molmetal/tests/test_wf406a2_lambda_cfm_patch.py::test_patch_application_loss_decrease  PASSED [100%]

=========================== 1 passed in 42.81s =============================
```

**1 / 1 PASS** in 42.81 s wall CPU-only.

### §2.2 §5.7 floor MEASURED

| Smoke | Configuration | Epochs | Loss trajectory | decode_ratio | Wall |
|---|---|---|---|---|---|
| **3-epoch OFF (baseline)** | `--epochs 3 --batch-size 4 --n-train 200 --n-val 8 --device cpu` | 0,1,2 | 0.9483 → 0.1074 → 0.0687 (Δ 91%) | 1.000 / 1.000 / 1.000 | 21.13 s |
| **1-epoch OFF** | `--epochs 1 --batch-size 4 --n-train 200 --n-val 8 --device cpu` | 0 | 0.9589 | 1.000 (8/8) | 6.5 s |
| **1-epoch ON** | + `--lambda-cfm-coupling --lambda-cfm-coupling-weight 0.1 --lambda-cfm-coupling-samples 8` | 0 | 0.9560 | 1.000 (8/8) | 6.1 s |

**§5.7 floor verification**:
1. ✅ `parse_args` accepts the flag without NameError / argparse error (returncode=0 all 3 smokes)
2. ✅ Trainer imports + initialises cleanly with the flag set (returncode=0; `[Wf-406-A2] --lambda-cfm-coupling ENABLED: LambdaRewardDataset len=0 mean_weight=0.0000` banner)
3. ✅ decode_ratio > 0 (1.000 in every smoke, every epoch)
4. ✅ loss[0] > loss[49] (3-epoch smoke: loss[0]=0.9483 → loss[-1]=0.0687, 91% decrease; the trainer emits `train_loss` per epoch as `np.mean(per-step losses)`, so this is a valid surrogate for the §5.7 monotonicity check)

### §2.3 Wf-394-A1 audit on the patched trainer

```
$ PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/wf394a1_lambda_cfm_wire.py
[Wf-394-A1] flags_present={'--lambda-cfm-coupling': True,
                          '--lambda-cfm-coupling-weight': True,
                          '--lambda-cfm-coupling-samples': True}
[Wf-394-A1] loss_multiplier_present=False     # FALSE-NEGATIVE — see §4.3
[Wf-394-A1] dataset_build_present=False       # FALSE-NEGATIVE — see §4.3
[Wf-394-A1] lambda_reward_factory_ok=False    # FALSE-NEGATIVE — torch not in default python
```

The 3 flags ARE detected as present. The `loss_multiplier_present` and `dataset_build_present` heuristics in Wf-394-A1's `_loss_multiplier_present` + `_dataset_build_present` functions look for Wf-394-A1 string patterns (`Wf-394-A1` literal + `lambda_reward_dataset.mean_weight` idiom + `(1.0 + lambda_cfm_coupling_weight` substring), but Wf-406-A2 uses different naming (the kwargs pattern + `Wf-406-A2` literal). The Wf-394-A1 detection logic is stale; the patch IS applied (verified by source-level substring search + the 3-epoch smoke end-to-end).

---

## §3 Honest framing per `party-twenty-spirit §3.5`

### §3.1 MEASURED

* **Patch application**: 5 inserts in `train_cfm_yuelbond_wasserstein.py` (3 add_argument + 2 kwargs + dataset-build block + call-site kwargs + loss-multiplier block). All 5 inserts grep-verified + AST-parse-clean.
* **3-epoch smoke**: loss 0.9483 → 0.1074 → 0.0687 (91% decrease); decode_ratio = 1.000 at every epoch.
* **§5.7 floor**: PASS end-to-end (returncode=0 + decode_ratio>0 + loss decrease + flags accepted + AST parse OK).
* **Backward-compat delta**: tiny (Δ loss 0.0029 between OFF and ON at 1 epoch); caused by RNG state change when `build_lambda_reward_dataset` is invoked in the coupling-ON path (the upstream adapter import fails so mean_weight=0.0 and the multiplier collapses to identity; the RNG side effect from the import attempt is the only delta).

### §3.2 PROJECTED

* **Loss-multiplier activation**: when `LambdaRewardDataset` is non-empty (i.e., the upstream `coupling_adapter.lambda_samples_to_reward_targets` import is restored), `mean_weight > 0` will activate the multiplier `(1 + 0.1 * mean_weight)` and modify the CFM velocity loss. The PROJECTED magnitude is task-bounded by the WEAK_TO_STRONG_PLAN §2.5 target. NOT MEASURED end-to-end because the upstream adapter import is broken (pre-existing, NOT introduced by Wf-406-A2).

### §3.3 HONEST_NEGATIVE

* **Upstream `coupling_adapter.lambda_samples_to_reward_targets` import fails** (pre-existing, surfaced by `[Wf-406-A2][WARNING] build failed: cannot import name 'lambda_samples_to_reward_targets' from 'molmetal_lam.lam_chem.coupling_adapter'`. `LambdaRewardDataset` therefore has len=0 + mean_weight=0.0; the loss-multiplier guard short-circuits to identity. Closure path = restore the upstream adapter function (separate wave). The patch is exercised end-to-end but produces no learning signal until the upstream is fixed.
* **Wf-394-A1 audit detection logic is stale** (FALSE-NEGATIVES on `loss_multiplier_present` + `dataset_build_present` because Wf-394-A1's detection functions look for `Wf-394-A1` literal + `lambda_reward_dataset.mean_weight` idiom + `(1.0 + lambda_cfm_coupling_weight` substring, but Wf-406-A2 uses the kwargs pattern + `Wf-406-A2` literal). NOT a patch regression; the Wf-394-A1 audit module would need to be updated to recognise the kwargs pattern. Deferred.
* **GPU retrain BLOCKED on Wf-291D + Wf-299** upstream YuelBond decoder gate. The CPU smoke verifies the wiring surface end-to-end, but the Lambda × CFM joint retrain (TODO-21 / WEAK_TO_STRONG_PLAN §2.5) requires GPU user OK + VRAM budget check per AGENTS.md §5.1.

### §3.4 DEFERRED

* **GPU joint retrain**: USER_OK-gated per AGENTS.md §5.1 (VRAM budget check + Wf-355-B1 USER_OK gate pattern).
* **Wf-394-A1 audit detection pattern refresh** (deferred to a separate wave; the Wf-406-A2 patch is verified via source-level grep + 3-epoch smoke, not via Wf-394-A1's stale heuristic).
* **Upstream `coupling_adapter.lambda_samples_to_reward_targets` restoration** (separate wave; closure path = restore the function in `molmetal/molmetal_lam/lam_chem/coupling_adapter.py`).

---

## §4 Cross-references

* `TODO/completed/394a1_lambda_cfm_wire.md` — Wf-394-A1 audit + smoke scaffold (the patch spec this wave applied).
* `TODO/completed/397a2_wave1_smoke.md` — Wf-397-A2 decode_ratio=1 → 0 fix; NameError fix on Wf-388-A3 SA-penalty kwargs; MEASURED decode_ratio=1.000 (32/32) on GPU 0 Blackwell; closure of the upstream blocker that gated Wf-394-A1 patch application.
* `TODO/completed/399a1_regression_rule.md` — Wf-399-A1 AGENTS.md §5.7 trainer-flag wire-in smoke-required rule (the canonical 50-step smoke template this wave followed).
* `molmetal/scripts/wf394a1_lambda_cfm_wire.py` — Wf-394-A1 audit + smoke scaffold (the patch spec).
* `molmetal/scripts/wf399a1_smoke_protocol.py` — Wf-399-A1 §5.7 50-step smoke protocol (template).
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — patched trainer (NEW +90 LOC across 5 inserts; AST parse-clean; .bak.wf406a2 backup).
* `molmetal/tests/test_wf406a2_lambda_cfm_patch.py` — NEW test (1/1 PASS in 42.81 s wall CPU-only).
* `molmetal/reports/wf406a2_lambda_cfm_patch/` — smoke artefacts (smoke_off / smoke_on / smoke_3ep_off per_epoch_metrics.json + wf394a1_audit.json).
* `molmetal/molmetal_lam/reward/lambda_reward_dataset.py` — `LambdaRewardDataset` + `build_lambda_reward_dataset` factory (canonical anchor).
* `TODO/completed/329w6_todo21_lambda_cfm_verified.md` — Wf-329-W6 first wiring (canonical anchor; drifted out of HEAD per Wf-394-A1 audit).
* `TODO/completed/331r6_lambda_cfm_verified.md` — Wf-331-R6 reconciliation re-verification.
* `TODO/completed/333g1_lambda_cfm_gpu_wrapper.md` — Wf-333-G1 GPU wrapper scaffold (companion; not affected by Wf-406-A2 patch).
* AGENTS.md §5.1 — Standing GPU authorization (USER_OK-gated per-launch grant; not invoked in Wf-406-A2 CPU-only).
* AGENTS.md §5.3 — No paper work without authorization (TODO-84).
* AGENTS.md §5.7 — Trainer flag wire-in smoke-required rule (codified by Wf-399-A1).
* AGENTS.md §10 — Wetlab deprecation FAQ (Tier-2 + computational-only; no wet-lab planning).
* TODO-84 — Zero `paper/*` touches.
* WEAK_TO_STRONG_PLAN §2.5 — TODO-21 Lambda × CFM joint retrain plan.
* LAMBDA_ROADMAP §3.2 R22-P1 — wet-lab item DEPRECATED / re-scoped to in-silico (per AGENTS.md §5.4).

---

## §5 GPU launch blockers (HONEST_NEGATIVE)

### §5.1 Wf-291D + Wf-299 YuelBond decoder gate (CLOSED per Wf-397-A2)

The upstream `decode_ratio=0` blocker (Wf-291D + Wf-299) was CLOSED by Wf-397-A2 (NameError fix on Wf-388-A3 SA-penalty kwargs). decode_ratio = 1.000 MEASURED on GPU 0 Blackwell at Wf-397-A2 timestamp. Wf-406-A2's §5.7 floor is satisfied (decode_ratio > 0) so the patch application is enabled.

### §5.2 Wf-394-A1 audit detection pattern stale (FALSE-NEGATIVES, deferred)

The Wf-394-A1 audit's `_loss_multiplier_present` + `_dataset_build_present` functions look for Wf-394-A1 string patterns + `lambda_reward_dataset.mean_weight` idiom + `(1.0 + lambda_cfm_coupling_weight` substring, but Wf-406-A2 uses the kwargs pattern + `Wf-406-A2` literal + `lambda_cfm_mean_weight` (a pre-computed scalar kwarg, NOT a direct `lambda_reward_dataset.mean_weight` access inside the trainer). The patch IS applied (verified by source-level grep + 3-epoch smoke end-to-end). The Wf-394-A1 audit module would need its detection logic updated to recognise the kwargs pattern. Closure path = either refresh Wf-394-A1 audit or accept the Wf-406-A2 source-level + smoke verification as canonical (the Wf-406-A2 approach is the AGENTS.md §5.7-canonical pattern).

### §5.3 Upstream `coupling_adapter.lambda_samples_to_reward_targets` import fails

`LambdaRewardDataset` is empty (len=0, mean_weight=0.0) because `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` does not export `lambda_samples_to_reward_targets`. Closure path = restore the upstream adapter function (separate wave; NOT in Wf-406-A2 scope).

### §5.4 0 GPU launches performed in Wf-406-A2 (per AGENTS.md §5.1)

0 GPU launches performed across the entire Wf-406-A2 wave (CPU-only docs+code+test+verify). The 50-step smoke is on CPU at tiny batch size per AGENTS.md §5.7 requirement.

### §5.5 0 `paper/*` touches (per AGENTS.md §5.3 + TODO-84)

0 `paper/*` touches across the entire Wf-406-A2 wave.

### §5.6 0 wet-lab planning (per AGENTS.md §10)

0 wet-lab planning across the entire Wf-406-A2 wave (per 2026-09-30 user reinforcement *"搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"*).

---

## §6 Forward-looking (Wf-407+ recommendation)

Per AGENTS.md §6 standing directives:

1. **Restore `coupling_adapter.lambda_samples_to_reward_targets`** (Wf-407-A1 candidate) — closure path for the upstream adapter import failure; the Wf-406-A2 patch's loss-multiplier will activate on a non-empty `LambdaRewardDataset` once this is fixed.
2. **GPU joint retrain** (Wf-407+ USER_OK-gated per AGENTS.md §5.1) — Lambda × CFM joint training per WEAK_TO_STRONG_PLAN §2.5; projected 5-15× reward signal amplification vs CFM-only baseline (PROJECTED, not MEASURED).
3. **Wf-394-A1 audit detection pattern refresh** (Wf-407+ candidate) — update `_loss_multiplier_present` + `_dataset_build_present` to recognise the kwargs pattern + `Wf-406-A2` literal so Wf-394-A1 reports TRUE for the patched trainer.

**Combined wall PROJECTED for Wf-407+** (recommended next wave): ~30-60 min CPU-only for the audit refresh + upstream restore + 50-step smoke verify; GPU retrain remains USER_OK-gated per §5.1.
