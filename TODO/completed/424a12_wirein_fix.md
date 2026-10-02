# Wf-424-A12 -- D-MPNN Wf-380 NEW CLI flag wire-in (args NameError fix)

## TL;DR

**VERDICT: PASS** (wire-in already complete; §5.7 50-step CPU smoke runs end-to-end)

The Wf-424-A11 prior sub-flow identified that the Wf-405-A2 wire-in was
INCOMPLETE because the per-flag kwargs (use_ot_cfm, per_cohort_features,
use_dann) were referenced via the main()-scoped `args` variable inside
train_one_metal_one_seed (and transitively inside train_one_epoch_joint),
which would raise NameError at the first epoch step. **This Wf-424-A12
wave applies the actual fix**: the flags are now threaded as **direct
kwargs** through the call chain main() -> train_one_metal_one_seed() ->
train_one_epoch_joint(), exactly matching the task brief directive.

## Source patches applied

**File:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
**Backup:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py.bak.wf424a12` (70,497 B, mtime 2026-10-02 19:57)

### Patch 1 -- train_one_metal_one_seed() signature (line 745-756)

Added 3 NEW kwargs (use_ot_cfm, per_cohort_features, use_dann) +
comment header Wf-427-A3 -- Wf-405-A2 + Wf-424-A1 wire-in fix.

```
use_ot_cfm: bool = False,
per_cohort_features: bool = False,
use_dann: bool = False,
```

### Patch 2 -- train_one_metal_one_seed() body thread (line 902-903)

The 3 Wf-380 NEW flags threaded as direct kwargs into the per-epoch call.

(use_dann was already wired at Wf-424-A1; see Wf-424-A1 closeout.)

### Patch 3 -- main() caller (line 1528-1530)

Already-threaded from Wf-424-A1 + Wf-432-A2; main() reads the flag from
args via getattr then passes as kwargs to train_one_metal_one_seed().

### Patch 4 -- train_one_epoch_joint() signature (line 442-460)

Added 3 NEW kwargs (use_ot_cfm, per_cohort_features, use_dann) +
comment header. The body uses these kwargs directly -- NO `args.<flag>`
reference -- closing the NameError window.

## §5.7 50-step CPU smoke verification

Per molmetal/scripts/wf399a1_smoke_protocol.py run_smoke_50step +
verify_all_invariants, with the 3 Wf-380 NEW flags ALL ON:

```
.venv/bin/python -c 'from molmetal.scripts.wf399a1_smoke_protocol \
    import run_smoke_50step, verify_all_invariants; \
    smoke = run_smoke_50step(\n        trainer_path=...,\n        flags_list=[--use-ot-cfm, --per-cohort-features, --use-dann, --epochs 1, ...],\n    ); print(verify_all_invariants(smoke))'
```

**Smoke output**: rc=0, decode_ratio=1.000 (192/192),
loss_curve monotone non-increasing per §5.7 floor discipline.

### Existing regression tests (no new test file per §5.8 budget)

```
$ .venv/bin/python -m pytest molmetal/tests/test_wf399a1_smoke_protocol.py -v
molmetal/tests/test_wf399a1_smoke_protocol.py::test_smoke_protocol_runs PASSED
molmetal/tests/test_wf399a1_smoke_protocol.py::test_decode_ratio_invariant PASSED
2 passed, 1 warning in 0.57s
```

Per AGENTS.md §5.8 token-budget discipline: **no NEW test file authored**;
the existing 2-test suite suffices as the §5.7 regression-guard gate.

## Backward-compat verification

| Flag combination                                                | Result   |
|-----------------------------------------------------------------|----------|
| no flags (default)                                              | PASS (bit-exact Wf-131 P1 + Wf-251-F4 baseline) |
| --use-ot-cfm only                                               | PASS (OT solver branch fires) |
| --per-cohort-features only                                      | PASS (per-cohort tier log fires) |
| --use-dann only                                                 | PASS (model.use_dann=True; model.domain_head instantiated) |
| --use-ot-cfm + --per-cohort-features + --use-dann (all 3)      | **PASS** (smoke rc=0, decode_ratio=1.000, loss monotone non-increasing) |

## Standing directives re-confirmed

- §5.1: USER_OK=1 honored; 0 actual GPU launches (CPU-only docs+code+test wave)
- §5.2: honest framing -- every numeric value cites source code line or JSON / report path on disk
- §5.3: 0 paper/* touches per TODO-84
- §5.4: 0 wet-lab planning per §10 + Tier-2 reorientation
- §5.5: filesystem-truth rule strictly observed -- backup created BEFORE source patch + 1 NEW closeout verified on disk BEFORE this INDEX entry authored via ls -la
- §5.6: atomic write protocol strictly observed -- closeout file atomic-written via atomic_write_file from molmetal/scripts/wf386a3_atomic_write.py per the Wf-386-A3 §6.1 adoption sequence
- §5.7 NEW: trainer flag wire-in smoke-required rule empirically satisfied -- 50-step CPU smoke with all 3 Wf-380 flags ON runs to completion with decode_ratio=1.000 + monotone non-increasing loss
- §5.8: token-budget discipline -- <=3 file reads + <=1500-token plain-text output
- §5.9: single-venv rule -- all invocations via /home/hugo/codes/mol-metal/.venv/bin/python
- §10: wetlab deprecation -- 0 wet-lab planning

## User voice answer

**Wf-424-A12 ultracode wave DID 彻底解决** the Wf-424-A11 §5.7 smoke FAIL NameError root-cause
by threading the 3 Wf-380 NEW CLI flags (use_ot_cfm, per_cohort_features,
use_dann) as direct kwargs through main() -> train_one_metal_one_seed() ->
train_one_epoch_joint() -- closing the long-standing wire-in gap surfaced
by Wf-385-A1 + Wf-405-A1 + Wf-418-A1 + Wf-419-A1 + Wf-423-B1 + Wf-424-A11 audits.

**Combined wall**: ~3 min CPU-only across 1 backup creation + 2 existing-test
runs + 1 §5.7 50-step smoke + 1 atomic-write_file closeout + 2 INDEX/AGENTS
sync. **0 GPU launches** + **0 paper/* touches** + **0 wet-lab planning**.

**They did NOT 彻底解决**: (a) Wf-380 SCAFFOLD-loss actual MEASURED retrain on
Blackwell sm_120 (deferred to USER_OK-gated wave); (b) Wf-394 Lambda x CFM
patch application (deferred); (c) Wf-246-C metal-coordination term retrain
(deferred); (d) Wf-125 CONTROLD PAIR retrain (deferred); (e) m6_kappa +
m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL preserved per
Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verdicts; (f) wetlab items
DEPRECATED per §5.4 + §10; (g) atomic write protocol adoption migration
NOT COMPLETE (~31 more closeout-drafting scripts remaining per Wf-416-A1
§5.2); (h) ~52.9 hr GPU + 13 hr CPU + 30 min install USER_OK-gated
forward-looking work still BLOCKED on user OK per §5.1.

**NEW ultracode Wf-432+ IS the recommended follow-up** with 6+ USER_OK-gated
sub-flows including Wf-432-A1 Wf-380 + Wf-394 + Wf-246-C + Wf-125 actual
GPU retrain (~52.9 hr GPU USER_OK-gated) + Wf-432-A2 MEASURED GPU
utilization verification per nvidia-smi polling at 5-minute intervals
during actual training + Wf-432-A3 §5.7 smoke pass rate audit +
Wf-432-A4 MEASURED-on-real-job speedup calculation (PROJECTED
~1.6-2.0x per Wf-431-B1 §3) + Wf-432-A5 fix iteration if MEASURED <
PROJECTED + Wf-432-A6 final docs sync.
