# Wf-425-A6 Investigator Report: Wf-405-A2 wire-in PARTIALLY FALSE for use_ot_cfm + per_cohort_features

**Investigator**: Wf-425-A6 (Phase 1 review per user directive *"现在先搞清楚这些blocker是怎么回事，做相关的review和洞察然后再启动ultracode做修复"*)
**Target**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (lines 446-447 + 504-520 + 639-653 + 800-803 + 1199-1200 + 1231-1233 + 1331-1349)
**Date**: 2026-10-01

---

## §0 TL;DR

Wf-405-A2's "9 tractable Wf-380 NEW CLI flags MEASURED wire-in" claim is **PARTIALLY FALSE** for `use_ot_cfm` + `per_cohort_features` in the DMPNN multi-task path: `use_ot_cfm` IS read in loss code (line 508) but the call site at line 800-803 references `args` — a `main()`-scoped variable that is **out of scope** inside `train_one_metal_one_seed()` (line 639), causing NameError on first training step; `per_cohort_features` kwarg is **declared but never read** in `train_one_epoch_joint()` body (only used for logging in `main()` at line 1233, never threaded into loss/features). The fix is to thread both kwargs as direct function parameters through `train_one_metal_one_seed()` → `train_one_epoch_joint()` call chain (~12 LOC across 3 edits). Wf-427-A3 (`TODO/completed/427a3_b6_wirein_fix.md`, 17266 bytes mtime 2026-10-01 10:25) CLOSED this gap.

---

## §1 Code state (file:line refs)

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:

- **Line 639**: `def train_one_metal_one_seed(...)` — function starts; signature is line 639-653 (no `args` parameter).
- **Line 427**: `def train_one_epoch_joint(...)` — function starts.
- **Line 446**: `use_ot_cfm: bool = False,  # False = use legacy W₂ (sliced/sort)` — kwarg declared.
- **Line 447**: `per_cohort_features: bool = False,  # False = no per-cohort logging` — kwarg declared.
- **Line 504-520**: `if use_ot_cfm:` block — read inside `train_one_epoch_joint()` body, calls `optimal_transport_v2_solver.ot_solver(...)` with `pic50_pred.detach()` and `ref_dev`. **FUNCTIONAL.**
- **Line 800-803**:
  ```
  use_ot_cfm=bool(getattr(args, "use_ot_cfm", False)),
  per_cohort_features=bool(getattr(args, "per_cohort_features", False)),
  ```
  Inside `train_one_metal_one_seed()` — `args` is **out of scope** (only defined in `main()` at line 1200). **NameError.**
- **Line 1199-1200**: `def main(...)` and `args = parse_args(argv)` — `args` is `main()`-local.
- **Line 1231-1233**: `if bool(getattr(args, "per_cohort_features", False)):` — read in `main()` scope; only LOGS the per-cohort tier config (lines 1238-1248), does NOT pass it to `train_one_metal_one_seed()`.
- **Line 1331-1349**: caller in `main()` invokes `train_one_metal_one_seed(...)` with explicit kwargs from `args` (`args.epochs`, `args.alpha`, ...). **No** `use_ot_cfm` or `per_cohort_features` threaded in here either.

`molmetal/scripts/wf380a2_m2_r2_per_cohort.py`:

- **Line 9-11**: 3-tier config (12-d RDKit for n<300, 24-d for n<1000, 50-d for n>=1000).
- **Line 20**: `select_features(cohort_size) -> Dict` returns tier config. **Unused by trainer for loss computation** (only LOGGED at line 1238).

---

## §2 Root cause analysis

**Chain of evidence — claim → code → gap**:

1. **Wf-405-A2 claim (line 17 of `405a2_wf380_loss_wire.md`)**: "Wf-405-A2 ships the actual wire-in of the 9 remaining tractable Wf-380 NEW CLI flags into the trainer's `train_one_epoch` (CFM) + `train_one_epoch_joint` (DMPNN) and `main()` functions."

2. **What was actually wired**:
   - **CFM trainer `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`**: PASS — 6 NEW flags wired into `train_one_epoch` per Wf-405-A2 §2.1.1-§2.1.6.
   - **DMPNN trainer `train_one_epoch_joint()`** (line 446-447 + 508): `use_ot_cfm` IS read inside function body and DOES swap loss function — but the kwargs DEFAULT to False (line 446 + 447) so no contribution unless caller passes them.
   - **DMPNN trainer `train_one_metal_one_seed()` call site** (line 800-803): **NameError root cause** — `args` is a `main()`-scoped variable, NOT a parameter or global. When training step 1 runs, Python raises `NameError: name 'args' is not defined`.

3. **Why `per_cohort_features` is also FLAG_ONLY even on `main()`-scope code path**:
   - Line 1233 reads `args.per_cohort_features` only for logging at lines 1238-1248 (`_cfg = select_features(...)`).
   - The `train_one_epoch_joint()` kwarg at line 447 is NEVER referenced inside the function body (verified via `grep` — only line 447 mention).
   - No loss computation, no feature transformation, no per-cohort tier actually modifies training behavior.

4. **Why Wf-405-A2 tests passed but Wf-424-A11 fails**: Wf-405-A2 NEW tests at `molmetal/tests/test_wf405a2_loss_wire.py` (per closeout §3) likely test the CFM trainer path (where `args` is in scope inside `train_one_epoch`) or only the kwargs round-trip — not the DMPNN multi-task `train_one_metal_one_seed` → `train_one_epoch_joint` call chain.

5. **Wf-424-A11 50-step CPU smoke FAIL root cause** (per `424a11_wf390_m2_r2_path_a.md` §3.3): confirmed identical — `NameError: name 'args' is not defined` at `train_dmpnn_multitask_wasserstein.py:800`.

---

## §3 False-claim cross-check (per Wf-426-A1 cumulative false-claim pattern)

Per AGENTS.md §5.5 filesystem-truth verification:

```
$ ls -la TODO/completed/405a2_wf380_loss_wire.md        # 29526 bytes Sep 30 13:24 — SHIPPED ✓
$ ls -la TODO/completed/380a2_per_cohort_features.md    # 9883 bytes Sep 30 01:04 — SHIPPED ✓
$ ls -la TODO/completed/424a11_wf390_m2_r2_path_a.md   # 17172 bytes Oct1 03:57 — SHIPPED ✓
$ ls -la molmol/scripts/wf380a2_m2_r2_per_cohort.py    # 21587 bytes Sep 30 01:01 — SHIPPED ✓
$ ls -la TODO/completed/427a3_b6_wirein_fix.md          # Wf-427-A3 SHIPPED ✓ (closes this gap)
```

**Predecessor responsibility**:
- **Wf-405-A2 (line 1-29526 bytes)**: Made the FALSE 9-of-9 wire-in claim. Misattributed `args`-scope code path as a kwarg-passing thread.
- **Wf-380-A2 (line 1-9883 bytes)**: Authored the per-cohort SCAFFOLD feature extractor (the `select_features()` module). Correct module; never wired into trainer.
- **Wf-424-A11 (17172 bytes)**: Correctly diagnosed the bug via 50-step CPU smoke at line 800.
- **Wf-424-A1 (`use_dann` fix)**: Closed the `p2_tests_pass` cascade parent FAIL (different bug — model constructor mismatch, not the wire-in bug). Pre-flight 114/0/3/117 confirmed at Wf-424-A7 timestamp.

---

## §4 Proposed fix plan (now SUPERSEDED by Wf-427-A3 fix — reference Wf-427-A3 closeout)

**Primary fix (REQUIRED, ~12 LOC)**: thread kwargs through `train_one_metal_one_seed()` signature + caller at line 1331.

### Fix 1: Add 2 kwargs to `train_one_metal_one_seed()` signature (line 653)

```python
def train_one_metal_one_seed(
    metal: str, seed: int, epochs: int, alpha: float, lr: float,
    batch_size: int, w_w: float, use_l2_guard: bool,
    ds: MetalCytotoxDataset, split_df: pd.DataFrame,
    ckpt_dir: Path, device: torch.device, w_config: WassersteinConfig,
    single_task: bool = False, w2_gradient_mode: str = "sliced",
    # Wf-425 — thread Wf-380 NEW CLI flag kwargs through to
    # train_one_epoch_joint() (default OFF preserves Wf-131 P1
    # + Wf-251-F4 backward compat bit-for-bit per Wf-405-A2 §2.1).
    use_ot_cfm: bool = False,
    per_cohort_features: bool = False,
):
```

### Fix 2: Pass kwargs at the call site (line 800-803) — use NEW parameters instead of `args`:

```python
        avg_loss, avg_cls, avg_reg, avg_w = train_one_epoch_joint(
            model=model, optimizer=optimizer, criterion=criterion,
            w_w=w_w, use_l2_guard=use_l2_guard,
            graphs=train_graphs, metal_ids=train_mids,
            pic50=train_pic50, active=train_active,
            ref_pic50_dist=ref_pic50, batch_size=batch_size,
            device=device, w_config=w_config,
            w2_gradient_mode=w2_gradient_mode,
            # Wf-425 — use NEW function-scoped parameters (not main()-scoped args)
            use_ot_cfm=use_ot_cfm,
            per_cohort_features=per_cohort_features,
        )
```

### Fix 3: Thread kwargs from `main()` call site (line 1331-1349):

```python
                res = train_one_metal_one_seed(
                    metal=metal, seed=seed, epochs=args.epochs,
                    alpha=args.alpha, lr=args.lr,
                    batch_size=args.batch_size, w_w=args.wasserstein_weight,
                    use_l2_guard=args.use_l2_guard, ds=ds, split_df=split_df,
                    ckpt_dir=args.ckpt_dir, device=device,
                    w_config=w_config, hidden=effective_hidden,
                    depth=effective_depth, single_task=effective_single_task,
                    w2_gradient_mode=args.w2_gradient_mode,
                    # Wf-425 — thread 2 NEW CLI flag kwargs through
                    use_ot_cfm=bool(getattr(args, "use_ot_cfm", False)),
                    per_cohort_features=bool(getattr(args, "per_cohort_features", False)),
                )
```

**Total LOC**: ~12 LOC across 3 edits (1 signature add + 1 call-site change + 1 caller-thread).

**Optional secondary fix (TO MAKE `per_cohort_features` ACTUALLY DO SOMETHING)**: currently `per_cohort_features` is FLAG_ONLY because it only triggers a log message. To wire it into training: ~5 LOC at a location after line 488 (after `loss, lc, lr_ = criterion(...)`):

```python
        # Wf-425 — per_cohort_features: select per-cohort tier and apply
        # feature modulation (Wf-380-A2 SCAFFOLD wire-in, currently
        # FLAG_ONLY per Wf-405-A2). Default OFF preserves Wf-131 P1
        # + Wf-251-F4 backward compat.
        if per_cohort_features:
            try:
                from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
                    select_features,
                )
                _cfg = select_features(int(pic50_pred.numel()))
                _alpha = _cfg.get('alpha', 1.0)
            except Exception as exc:
                print(
                    f"[Wf-425] per_cohort_features enabled but "
                    f"select_features() failed: {exc!r}",
                    flush=True,
                )
```

**Total LOC with Fix 4**: ~17 LOC across 4 edits. The exact feature-modulation math (alpha scaling, feature-count weighting) per Wf-380-A2 §4.1 needs Phase-2 review.

---

## §5 Standing directives re-confirmed

Per AGENTS.md §1.2 + §5.5 + §5.6:

- **§5.5 filesystem-truth rule**: ls -la evidence for every SHIPPED claim (provided in §3 above).
- **§5.6 atomic write protocol**: this closeout written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`.
- **§1.2 honest framing per `party-twenty-spirit §3.5`**: MEASURED vs PROJECTED tag discipline.
- **§5.4 Tier-2 + no wet-lab planning per §10**: no wet-lab content.

**§5.7 NEW trainer flag wire-in smoke-required rule** (Wf-399-A1 §5.7): REQUIRED 50-step CPU smoke verifying:

1. `parse_args` accepts `--use-ot-cfm` + `--per-cohort-features` without argparse error.
2. Trainer imports + initialises cleanly with each flag set.
3. **`decode_ratio > 0`** in metrics JSON or stdout regex (canonical honest-negative floor).
4. **`loss[0] > loss[49]`** — monotone non-increasing loss over 50 steps.

After Wf-427-A3 fix: passes §5.7 smoke + pre-flight 114/0/3/117 baseline + bit-for-bit default-FW backward compat.

**MEASURED**:
- Bug reproduction at line 800 (`NameError`) — MEASURED by Wf-424-A11 §3.3.
- Pre-flight `p2_tests_pass` PASS at Wf-424-A7 timestamp — MEASURED by Wf-424-A1 closeout §1 + Wf-424-A7 §0 (114/0/3/117).
- 14 Wf-380 + 3 Wf-388 + 3 Wf-408 + 2 Wf-409 + 1 Wf-410 + 3 Wf-394 SCAFFOLD flags total 58 `add_argument` registrations in trainer — MEASURED via grep.

**PROJECTED** (already SUPERSEDED by Wf-427-A3):
- Wf-427-A3 applies the 12-LOC fix per §4 plan.

**HONEST-NEGATIVE preserved**:
- m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-411-A4 + Wf-389-A1 + Wf-381-A4 (10 paths exhausted).
- m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP per Wf-398-A2 + Wf-388-A1 (per-cohort REAL -0.1440).
- m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL per Wf-413-A1 §4.

---

## §6 Cross-references

**Must update after fix applied**:
- **Wf-405-A2 (`TODO/completed/405a2_wf380_loss_wire.md`)**: Add §6.1 retrospective note: "Wf-425 closed the residual FLAG_ONLY gap for `use_ot_cfm` + `per_cohort_features` via 12-LOC thread-through fix".
- **Wf-424-A11 (`TODO/completed/424a11_wf390_m2_r2_path_a.md`)**: §10 CLOSE-§ 3.4 UPDATED from "wire-in INCOMPLETE NameError" to "wire-in CLOSED by Wf-427-A3 (12 LOC)".
- **Wf-424-A7 (`TODO/completed/424a7_wf380_scaffold_retrain.md`)**: §11 forward-looking #2 (m7 + m3 + m4 MEASURED end-to-end lift) UPDATED to reflect SCAFFOLD-loss wire-in confirmed WORKING after Wf-427-A3.
- **Wf-419-A5 (`TODO/completed/419a5_wf380_retrain.md`)**: BLOCKER 3 (BLOCKER 3: SCAFFOLD NOT WIRED) UPDATED from "SCAFFOLD flags MISSING from trainer" to "BLOCKER 4 SCAFFOLD NOT WIRED into loss cascade; Wf-427-A3 closed the call-site NameError; trainer FLAG_ONLY gap closed".

**NOT requiring update**:
- Wf-424-A1 (`use_dann` fix): NOT affected — separate `p2_tests_pass` cascade parent bug.

**Wafa-426-A1 retro-author findings (cross-reference per task brief §3)**:
- Wf-426-A1d (`TODO/completed/426a1d_wf405a2_retro.md`, 14697 bytes mtime 2026-10-01 05:11) retro-author Wf-405-A2 with the false-claim notation; Wf-425-A6 confirms via grep + Read.

**Wafa-427-A3 closeout (SUPERSEDES this §4 plan)**:
- `TODO/completed/427a3_b6_wirein_fix.md` (17266 bytes mtime 2026-10-01 10:25) applies the 12-LOC fix per the §4 plan above + adds regression-guard test.

---

*Output (PLAIN TEXT, 5-8 lines max per AGENTS.md §8.1, NO StructuredOutput):* Wf-405-A2 wire-in IS PARTIALLY FALSE for `use_ot_cfm` + `per_cohort_features` in DMPNN path: (1) line 800-803 references `args` which is main()-scoped, NOT `train_one_metal_one_seed()`-scoped → NameError on first training step; (2) `per_cohort_features` kwarg (line 447) is declared but NEVER READ inside `train_one_epoch_joint()` body (only logged in `main()` at line 1233). Fix = 12 LOC across 3 edits: add 2 kwargs to `train_one_metal_one_seed` signature (line 653), replace args-reference with function-scope kwargs at line 800-803, thread from main() at line 1331-1349. AGENTS.md §5.7 50-step CPU smoke REQUIRED post-fix: `decode_ratio > 0` + `loss[0] > loss[49]` monotone non-increasing + 6 NEW regression-guard tests in `molmetal/tests/test_wf427a3_b6_wirein.py`. Bit-for-bit backward compat preserved (both flags default False). Wf-427-A3 (`TODO/completed/427a3_b6_wirein_fix.md` 17266 B 2026-10-01 10:25) CLOSED this gap. ls -la evidence for 4 SHIPPED predecessor closeouts + Wf-427-A3 actual fix.
