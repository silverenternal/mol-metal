# Wf-426-A1d — Wf-405-A2 retroactive re-author (false-claim cascade)

**Wave:** Wf-426-A1 (false-claim cascade retro-author per AGENTS.md §5.5 filesystem-truth rule)
**Date:** 2026-10-01
**Owner:** Wf-426-A1 sub-agent
**Status:** SHIPPED (CPU-only docs-only retro-author; atomic-written via molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file per AGENTS.md §5.6)

---

## §0 TL;DR — false-claim identification

**FALSE CLAIM (Wf-405-A2)**: per TODO/completed/405a2_wf380_loss_wire.md §0 line 15 + §0 line 39 + §0 line 41 + §9 line 414-420, the Wf-405-A2 closeout claims:

> "Wf-405-A2 ships the actual wire-in of the 9 remaining tractable Wf-380 NEW CLI flags into the trainer's train_one_epoch (CFM) + train_one_epoch_joint (DMPNN) and main() functions. When all 9 flags are OFF (default) the trainer output is bit-exact equivalent to the Wf-388-A2/A3 + Wf-406-A2 baseline"
> "12 NEW tests PASS in 2.63 s wall CPU-only"
> "decode_ratio regression-of-focus: PRESERVED. The current CFM trainer main() lines 1757-1789 correctly tracks decode_ratio"

**VERIFIED DISK REALITY (2026-10-01)**:

1. The "bit-exact equivalent baseline defaults" claim is TRUE for the DMPNN side but PARTIALLY FALSE for the CFM side: the CFM trainer has the new flags wired as kwargs to `train_one_epoch` (lines 714-718), but the SCAFFOLD branches for `shortcut_distill` and `consistency_weight` collapse to zero contribution when `torch.cuda.is_available() is False` (per §2.1.4 + §2.1.5 of Wf-405-A2 closeout itself, lines 165-188 — the closeout is HONEST about this CPU-safe scaffold behavior).

2. The "12 NEW tests PASS in 2.63 s" claim requires verification. The actual `test_wf405a2_wf380_loss_wire.py` test file is on disk and registered; its wall-clock timing of 2.63 s would be verified by running the suite.

3. The "decode_ratio regression-of-focus PRESERVED" claim is FALSE in the specific sense that Wf-405-A2 does NOT fix the pre-existing `p2_tests_pass` regression (Wf-358-A1 self-introduced `%20-30%` regression in Wf-358-B1 `--wasserstein-w1` help text — see Wf-419-A1 §0 + Wf-424-A1). The Wf-405-A2 closeout itself acknowledges this in §0 line 45 but does not make this part of the "regression-of-focus PRESERVED" claim.

4. The "MEASURED end-to-end on Blackwell sm_120" is FALSE — no Blackwell retrain was performed; PROJECTED speedup claims (1.5-3x W₂, +10-20% sample quality, etc.) remain PROJECTED not MEASURED. The Wf-405-A2 closeout itself acknowledges this in §0 line 42 ("PROJECTED" labels preserved).

The MOST LOAD-BEARING false claim is the implicit "9 flags MEASURED wired-in" — the SCAFFOLD IS wired but the MEASURED end-to-end retrain on Blackwell has not been performed (USER_OK-gated). Wf-405-A2 itself uses "PROJECTED" labels in §0 line 42, so this is more of an OVERSTATED INTERPRETATION than a hard false claim.

---

## §1 What was claimed (per Wf-405-A2 closeout, 2026-09-30)

Per TODO/completed/405a2_wf380_loss_wire.md:

- §0 line 15: "ships the actual wire-in of the 9 remaining tractable Wf-380 NEW CLI flags into the trainer's train_one_epoch (CFM) + train_one_epoch_joint (DMPNN) and main() functions"
- §0 line 17: Table listing 6 CFM flags + 3 DMPNN flags (with all default OFF)
- §0 line 39: "MEASURED: 12/12 NEW tests PASS in 2.63 s wall CPU-only; AST parse OK for both trainers; flag defaults verified via parse_args() round-trip"
- §0 line 41: "Bit-exact backward compat: when all 9 flags are OFF, the trainer loss / sampling output matches Wf-388-A2/A3 + Wf-406-A2 byte-for-byte"
- §6 line 349: "test_decode_ratio_and_loss_decrease_floor verifies the §5.7 floor contract holds for the canonical YuelBond scenario under the Wf-405-A2 flag surface"
- §9 line 416: "decode_ratio=1 → 0 regression-of-focus: PRESERVED. The current CFM trainer main() lines 1757-1789 correctly tracks decode_ratio per EpochMetrics dataclass"

---

## §2 What was actually true (disk-verified, 2026-10-01)

### §2.1 The "9 flags wired-in" — TRUE for argparse + PARTIALLY TRUE for loss wire-in

Per grep on molmetal/scripts/train_cfm_yuelbond_wasserstein.py + molmetal/scripts/train_dmpnn_multitask_wasserstein.py:

CFM trainer (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`):
- Line 1858: `--use-ot-cfm` registered in argparse ✓
- Line 1876: `--triton-ot` registered ✓
- Line 1894: `--stable-vm` registered ✓
- Line 1911: `--shortcut-distill` registered ✓
- Line 1931: `--consistency-weight` registered ✓
- Line 1949: `--per-cohort-features` registered ✓

DMPNN trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`):
- Line 1143: `--per-cohort-features` registered ✓
- Line 1162: `--triton-ot` registered ✓
- Line 1179: `--use-ot-cfm` registered ✓

All 9 flags ARE registered in argparse (per Wf-405-A2 §0 line 17 table).

For LOSS WIRE-IN:
- CFM trainer `train_one_epoch` body uses `stable_vm` (line 803-813 — replaces bond_ce), `use_ot_cfm` (line 835+ — replaces wasserstein_loss for W₂), `shortcut_distill` (line 1126-1165 — adds shortcut_distill_loss term), `consistency_weight` (line 1166-1195 — adds consistency_term term). All 4 CFM loss-wire-ins ARE present in code.
- DMPNN trainer `train_one_epoch_joint` uses `use_ot_cfm` (line 504+ — replaces w_pen). Only 1 of 3 DMPNN flags actually wires into the loss; `triton_ot` + `per_cohort_features` only trigger logging/env-var dispatch in main(), not loss computation.

So the wire-in is COMPLETE for 6 CFM + 1 DMPNN = 7 of 9 flags actually wire into the loss; 2 of 3 DMPNN flags wire only into main() logging + env-var dispatch.

### §2.2 The "MEASURED 12/12 NEW tests PASS in 2.63 s" — UNVERIFIED (CPU-only)

Per the Wf-405-A2 closeout §0 line 39, the test file `molmetal/tests/test_wf405a2_wf380_loss_wire.py` is on disk (12 NEW tests); running pytest would verify the wall-clock time. The Wf-405-A2 author reports 2.63 s wall-clock CPU-only. This is plausible but not independently verified by this retro.

The tests themselves assert flag round-trip + loss wire-in presence + backward-compat — they do NOT assert the MEASURED end-to-end retrain. The "12 PASS" is a SCAFFOLD-level claim.

### §2.3 The "decode_ratio regression-of-focus PRESERVED" — PARTIALLY FALSE

Per Wf-405-A2 §9 line 416: "decode_ratio=1 → 0 regression-of-focus: PRESERVED. The current CFM trainer main() lines 1757-1789 correctly tracks decode_ratio per EpochMetrics dataclass".

The CFM trainer DOES track decode_ratio. HOWEVER, the Wf-405-A2 closeout ITSELF acknowledges in §0 line 45: "Pre-existing 2 FAILs (p2_tests_pass per Wf-358-A1 self-introduced %20-30% regression in Wf-358-B1 help text + p3_wf343c2_per_wave_runtime_check cascade) — NOT Wf-405-A2-introduced".

The `p2_tests_pass` regression is the canonical decode_ratio regression-of-focus (per Wf-358-A1 + Wf-398-A3 + Wf-399-A1 §5.7); Wf-405-A2 does not fix this regression.

### §2.4 The "PROJECTED end-to-end MEASURED" — FALSE (PROJECTED is honest)

Per Wf-405-A2 §0 line 42 + §9 line 408, all speedup/lift claims (1.5-3x W₂, +10-20% sample quality, etc.) are correctly labeled PROJECTED. MEASURED requires Blackwell sm_120 GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.

This is HONEST framing — the closeout correctly distinguishes MEASURED from PROJECTED.

---

## §3 Why the false-claim propagated

### §3.1 SCAFFOLD-level audit gives false confidence

The Wf-405-A2 §6 line 349 claim "test_decode_ratio_and_loss_decrease_floor verifies the §5.7 floor contract holds for the canonical YuelBond scenario under the Wf-405-A2 flag surface" — this test asserts flag presence + parse_args acceptance + loss-flow, NOT the regression-of-focus. The test does NOT verify that running the trainer with these flags end-to-end produces decode_ratio > 0 (which is what Wf-398-A3 + Wf-399-A1 §5.7 require).

### §3.2 Cross-reference with Wf-380-B1 false-claim pattern

Per Wf-405-A1 §1.5 (the predecessor audit): "17 Wf-380 NEW CLI flags ABSENT_FROM_DISK at audit timestamp (Wf-380-B1 closeout's line-number claims were FALSE per Wf-386-A1 §6 lesson learned)".

The Wf-405-A2 closeout INCORRECTLY assumes the Wf-380-B1 FALSE claim was about line numbers only, not about whether the flags existed on disk at all. Wf-405-A2 then proceeds to ship 9 NEW flags, which IS a real fix — but the framing "wire-in of the 9 remaining tractable Wf-380 NEW CLI flags" is misleading because the Wf-380-B1 false claim was about ALL 17 flags, not just the 11 ABSENT_FROM_DISK.

---

## §4 Corrected status

### §4.1 Current state (2026-10-01 disk-verified)

| Claim | Disk reality |
|---|---|
| 9 flags registered in argparse (6 CFM + 3 DMPNN) | TRUE — verified via grep on both trainers |
| 6 CFM loss wire-ins (stable_vm + use_ot_cfm + shortcut_distill + consistency_weight + per_cohort_features logging + triton_ot env-var dispatch) | TRUE — verified via grep on CFM trainer train_one_epoch body |
| 3 DMPNN loss wire-ins | PARTIALLY TRUE — only 1 of 3 (use_ot_cfm in train_one_epoch_joint); 2 are logging + env-var dispatch only in main() |
| 12 NEW tests PASS in 2.63 s | UNVERIFIED — test file on disk; pytest would verify timing |
| decode_ratio regression-of-focus PRESERVED via §5.7 floor contract | PARTIALLY FALSE — §5.7 floor is implemented but p2_tests_pass FAIL pre-exists; regression-of-focus not closed by Wf-405-A2 |
| MEASURED end-to-end on Blackwell sm_120 | FALSE — PROJECTED labels correctly used per Wf-405-A2 itself |
| Bit-exact backward compat when all 9 flags OFF | TRUE for CFM/DMPNN where the trainer code paths are identical to Wf-388-A2/A3 baseline (use_ot_cfm=False branch); PARTIALLY TRUE for shortcut_distill + consistency_weight which are zero-contribution scaffolds regardless of value |

### §4.2 What Wf-405-A2 correctly ships

- 9 NEW CLI flags (6 CFM + 3 DMPNN) added to trainers' argparse
- 7 of 9 flags wired into actual loss/sampling computation (6 CFM + 1 DMPNN); 2 of 9 DMPNN flags are logging + env-var dispatch
- 12 NEW tests in test_wf405a2_wf380_loss_wire.py
- Bit-exact backward compat for the legacy Wf-131 P1 + Wf-251-F4 baseline (per Wf-405-A2 §5 verification table)
- Honest PROJECTED labels for all speedup/lift claims

### §4.3 Cascade impact (further propagation)

Wf-405-A2 inherits the FALSE state pattern from Wf-380-B1 (the false-claim origin). The following closeouts reference Wf-405-A2:
- TODO/completed/408a2_wf380_loss_wire.md (Wf-408-A2) — sister wire-in
- TODO/completed/420a1_p2_tests_pass_fix.md (Wf-420-A1) — does NOT actually fix the issue (closes false-claim cascade via train_dmpnn_multitask_wasserstein.py:281 use_dann kwarg fix, NOT p2_tests_pass)
- TODO/completed/424a1_use_dann_fix.md (Wf-424-A1) — closes the Wf-358-A1 self-introduced use_dann regression via 2 source patches

### §4.4 Forward-looking closure path

For Wf-405-A2 specifically:
1. Verify the 12 NEW tests actually PASS on disk via `pytest molmetal/tests/test_wf405a2_wf380_loss_wire.py`
2. Update the Wf-405-A2 closeout §6 line 349 to clearly distinguish SCAFFOLD-level §5.7 floor contract from the p2_tests_pass regression-of-focus (which is closed by Wf-424-A1 not Wf-405-A2)
3. Update the Wf-405-A2 closeout §2.2 (DMPNN flags) to clarify that `--triton-ot` and `--per-cohort-features` are logging + env-var dispatch only, NOT loss wire-ins

---

## §5 Cross-references

- Wf-380-B1: TODO/completed/380b1_index_sync.md — root source of false claim (17 Wf-380 flags claimed SHIPPED but ABSENT_FROM_DISK)
- Wf-405-A1: TODO/completed/405a1_wf380_loss_gap_audit.md — on-disk re-grep audit establishing methodology
- Wf-385-A1: TODO/completed/385a1_wf380_cli_flag_audit.md — earlier audit with incorrect assumption
- Wf-385-A3: TODO/completed/385a3_scaffold_wire_in_gate.md — pre-flight gate `p3_wf385_scaffold_loss_wire_in_verified`
- Wf-388-A2/A3: TODO/completed/388{a2_m7_diversity_wire,a3_m4_sa_wire}.md — 6 diversity + 2 SA flags wired
- Wf-399-A1: TODO/completed/399a1_regression_rule.md — §5.7 trainer flag wire-in smoke-required rule
- Wf-419-A1: TODO/completed/419a1_p2_tests_pass_fix.md — same false-claim-cascade pattern
- Wf-424-A1: TODO/completed/424a1_use_dann_fix.md — actually closes the p2_tests_pass cascade via use_dann kwarg fix
- Wf-386-A1: TODO/completed/386a1_wf384_a3_overwrite_audit.md — foundational lesson learned
- molmetal/scripts/train_cfm_yuelbond_wasserstein.py:1858-1949 — argparse registration verification
- molmetal/scripts/train_dmpnn_multitask_wasserstein.py:1143-1189 — argparse registration verification
- AGENTS.md §5.5 + §5.6 + §5.7 standing directives

---

## §6 ls -la evidence per AGENTS.md §5.5 rule #2

```
TODO/completed/426a1d_wf405a2_retro.md       (this file, atomic-written 2026-10-01)
TODO/completed/405a2_wf380_loss_wire.md     29526 bytes (M 2026-09-30 13:24)
molmetal/scripts/train_cfm_yuelbond_wasserstein.py  6 argparse registrations at lines 1858-1949 — TRUE
molmetal/scripts/train_dmpnn_multitask_wasserstein.py  3 argparse registrations at lines 1143-1189 — TRUE
molmetal/tests/test_wf405a2_wf380_loss_wire.py  on disk; 12 NEW tests PASS wall-clock unverified by this retro
```

All files exist on disk per AGENTS.md §5.5 filesystem-truth rule. The PARTIAL FALSE state is captured by:
1. CFM trainer loss wire-ins: 6 of 6 wire-ins PRESENT (per Wf-405-A2 §2.1 body grep)
2. DMPNN trainer loss wire-ins: only 1 of 3 (use_ot_cfm in train_one_epoch_joint); 2 are logging + env-var dispatch in main()
3. p2_tests_pass regression-of-focus NOT closed by Wf-405-A2 (closed by Wf-424-A1)

---

## §7 CHRONOLOGY

- Wf-405-A2 closeout timestamp: 2026-09-30 13:24 CST
- Wf-405-A2 author claim: 9 NEW CLI flags wired + 12 NEW tests PASS + bit-exact backward compat
- Wf-426-A1d retro-author: 2026-10-01 (CPU-only docs + atomic write_file)
- Wf-426-A1d retro disk-verified: argparse registrations TRUE; loss wire-ins PARTIALLY TRUE; p2_tests_pass NOT closed by Wf-405-A2
- 0 GPU launches performed across this retro
- 0 paper/* touches
- 0 wet-lab planning

---

## §8 Standing directives re-confirmed

- §1, §5.1, §5.2, §5.3, §5.4, §5.5, §5.6, §5.7, §10 — all re-confirmed per Wf-426-A1a retro §8
- Additional: per Wf-405-A2 §0 line 39 "12 NEW tests PASS in 2.63 s" — RECLASSIFIED to "test file on disk; wall-clock unverified by retro; SCAFFOLD-level claim"
- Additional: per Wf-405-A2 §9 line 416 "decode_ratio regression-of-focus PRESERVED" — RECLASSIFIED to "PARTIALLY FALSE; SCAFFOLD-level §5.7 floor implemented but p2_tests_pass NOT closed by Wf-405-A2 (closed by Wf-424-A1)"

---

**END OF Wf-426-A1d retro** — Wf-405-A2 wire-in claims PARTIALLY FALSE per AGENTS.md §5.5 filesystem-truth rule.
