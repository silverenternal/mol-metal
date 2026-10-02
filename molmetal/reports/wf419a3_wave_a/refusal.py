"""Wf-419-A3 closeout authoring script — REFUSES launch per AGENTS.md filesystem-truth rule."""
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file, safe_overwrite_check, detect_truncation,
)

CONTENT = '''# Wf-419-A3 Wf-299 Wave A launch — HONEST_NEGATIVE_REFUSED per AGENTS.md §5.5 filesystem-truth rule + §1.4 pre-flight PASS requirement

**Last updated:** 2026-09-30T23:22:00+0000 UTC (Wf-419-A3 timestamp)
**Wall:** ~3 min CPU-only (refusal + closeout authoring)
**Author:** Wf-419-A3 (per AGENTS.md §5.5 filesystem-truth rule)
**Refs:** Wf-419-A1 cascade fix + Wf-419-A2 pre-flight verify + Wf-415-A1 wrapper fix + AGENTS.md §1.4 + §5.1 + §5.2 + §5.3 + §5.5 + §5.6 + Wf-345-B3 + Wf-364-A1 + Wf-367-A2 + Wf-407-A3 + Wf-413-A4 + Wf-398-A1 SCAFFOLD_NOT_FUNCTIONAL precedent.

## §0 TL;DR

**DECISION: REFUSE Wf-299 Wave A GPU launch per AGENTS.md §1.4 pre-flight PASS requirement + §5.1 standing GPU authorization (conditional on pre-flight PASS) + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive.**

The task brief claims "Wf-419-A2 pre-flight verify PASS" but the on-disk pre-flight JSON state at Wf-419-A2 timestamp 2026-09-30T23:21:00+0000 UTC is **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** — NOT 113/0/3/117 = fully PASS.

Per AGENTS.md §5.5 filesystem-truth rule, the on-disk state governs. Per §1.4 "Pre-flight v2 25/25 PASS gate" + §5.1 "Pre-flight PASS still required (`pre_flight_check.py`)". The standing GPU authorization is **conditional** on pre-flight PASS, not unconditional.

The 2 FAILs are PRE-EXISTING upstream test bugs that Wf-419-A1 did NOT close:

1. `p2_tests_pass` (P2, tests): pytest FAIL (rc=1) — `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` raises `TypeError: MultiTaskWassersteinModel.__init__() got an unexpected keyword argument 'use_dann'`. The test instantiates `MultiTaskWassersteinModel(n_metals=2, hidden=32, depth=2, dropout=0.1, use_dann=False)` but the model class at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` `__init__()` only accepts `n_metals, atom_dim, bond_dim, hidden, depth, dropout` — no `use_dann` kwarg.
2. `p3_wf343c2_per_wave_runtime_check` (P3, wf347a1): wf343c2_per_wave_check.py FAIL (rc=1) — upstream `preflight_no_fail_gates` detects `p2_tests_pass` FAIL (cascade child per Wf-364-A1).

Launching a 5-hour GPU job under these conditions would waste ~5 hr × 95.59 GB Blackwell VRAM + GPU compute = ~5 hr wall-clock of GPU cycles for a launch that is contractually obligated to FAIL post-flight checks.

## §1 GPU Env Probe (GPU 0 IDLE)

**Command**: `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv`

**Output**:
```
index, memory.used [MiB], memory.free [MiB], utilization.gpu [%]
0, 2 MiB, 97340 MiB, 0 %
1, 2 MiB, 32202 MiB, 0 %
```

GPU 0 Blackwell 97.34 GB free IDLE (2 MiB used); GPU 1 RTX 5090 32.20 GB free IDLE. VRAM budget check PASS — sufficient free VRAM for Wf-299 Wave A launch per Wf-260 G6 timing (~12 hr wall for full sweep, ~5 hr for Wave A subset). However, VRAM alone is not sufficient — pre-flight PASS is also required per §1.4.

## §2 Launch Attempt Refusal per Pre-flight FAIL

Per AGENTS.md §5.1 + §1.4 + `kill_then_adapt §0`:

1. **USER_OK=1 grant honored**: USER_OK=1 is set per the standing GPU authorization §5.1. The refusal is NOT based on lack of user authorization — it is based on pre-flight contract obligation.
2. **Launcher functional**: Wf-299 launcher `molmetal/scripts/launch_wf299_cfm_main.sh` (29,179 bytes, mtime 2026-09-30 17:39) is operational per Wf-419-A2 §2 dry-run verification (exit code 0; 4 ablation configs logged; Wf-415-A1 EXTRA_FLAGS accumulator verified). The launcher itself is NOT the cause of the refusal.
3. **Pre-flight cascade FAIL**: The 2 FAILs (`p2_tests_pass` + cascade) are the gating constraint. Per §1.4 + Wf-302A AnchorGuard + Wf-345-B3 dynamic gate count, ALL gates in the pre-flight chain must be PASS or BY-DESIGN SKIP. FAIL is a hard block.

**Closure path**: Wf-420-A1 (CPU-only, ~5 LOC): Add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` (or update test_kendall_dann_wiring.py signature). After Wf-420-A1 fix, pre-flight becomes 113/0/3/117 → Wf-420-A3 actual GPU launch becomes safely executable.

## §3 Wf-299 launcher state

**Launcher**: `molmetal/scripts/launch_wf299_cfm_main.sh` (29,179 bytes, mtime 2026-09-30 17:39, executable bit set)

**Wf-415-A1 wrapper fix verified per Wf-419-A2 §2**:
- `EXTRA_FLAGS` accumulator at lines 210-258 working as designed
- 16 NEW trainer flags passthrough without wrapper-side rejection
- Exit code 0 in DRY_RUN mode
- 4 ablation configs logged (full_stack + yuelbond_only + w2_only + catherine_only)

**Per-pair argv expansion**: `PAIR_ARGS=( --pair "${pair}" )` per Wf-414-A1 precedent

**WAVELENGTH CONFLICT NOTE**: Wf-408-A3 `--wave 1/2/3` alias mapping was attempted but `--wave 3` literal still rejected per Wf-410-A4 contract (e) FAIL. The Wf-415-A1 wrapper + Wf-299 launcher's case statement at lines 211-216 accepts `--wave {A,B,C,all}` only. To use `--wave 3` literal, must patch the case statement OR use `--wave A` alias for what was intended as Wave 3.

## §4 Why refuse (vs. launch + fail later)

Launching a 5-hour GPU job with a known pre-flight cascade FAIL is wasteful per `kill_then_adapt §0` + AGENTS.md §5.2 honest framing:

1. **Wall waste**: ~5 hr × Blackwell GPU + ~95 GB VRAM = wasted compute cycle
2. **Audit trail pollution**: A launch attempt that exits at post-flight check pollutes the Wf-419 series closeout trail with a "launched but failed at gate X" record that requires additional closeout authorship to explain
3. **Standing directive violation**: §1.4 + §5.1 explicitly require pre-flight PASS as a launch precondition
4. **Alternative path exists**: Wf-420-A1 5-LOC CPU-only fix unblocks the launch in <30 min wall, vs. 5 hr of failed-launch closeout authoring

The honest-negative framing is: REFUSE LAUNCH per pre-flight contract; document the failure mode; recommend the minimal-cost fix path (Wf-420-A1); defer actual launch to Wf-420-A3 after Wf-420-A1 closes the `use_dann` kwarg mismatch.

## §5 MEASURED Lifts: NOT APPLICABLE (no launch)

No MEASURED lift per `party-twenty-spirit` §3.5 since no launch took place. The PROJECTED lifts per Wf-380-A3 + Wf-388-A2 + Wf-405-A2 + Wf-408-A2 SCAFFOLD modules remain PROJECTED until Blackwell sm_120 retrain validates per AGENTS.md §5.1 + kill_then_adapt §0.

Honest-framing caveat: PROJECTED ≠ MEASURED. MEASURED requires actual GPU launch + post-launch metrics JSON update + decode_ratio trajectory + loss-decrease trajectory — none of which can be produced without launching. Refusal preserves the honest framing contract.

## §6 Cross-references

- **AGENTS.md** §1.4 (pre-flight PASS gate) + §5.1 (standing GPU auth + conditional pre-flight PASS) + §5.2 (honest framing) + §5.3 (no paper/* touches) + §5.5 (filesystem-truth rule — applied: 2 FAILs recorded honestly, not paraphrased) + §5.6 (atomic write protocol — applied via `atomic_write_file` for this closeout) + §5.7 NEW (trainer flag wire-in smoke-required rule)
- **Wf-345-B3** dynamic gate count auto-compute — preserved consistency: `actual=117 == expected=117`
- **Wf-364-A1** cascade self-reference cycle — `p3_wf343c2_per_wave_runtime_check` excludes self when counting upstream FAILs
- **Wf-367-A2** `--strict` removal from gate invocation (preserves `--skip-live` semantics; 3 SKIPs not FAILs)
- **Wf-415-A1** `EXTRA_FLAGS` accumulator wrapper fix — verified working at Wf-419-A2 timestamp
- **Wf-385-A3** SCAFFOLD-loss wire-in gate — PASS (11/11 modules export OK + 50-step smoke PASSED)
- **Wf-405-A3** pre-flight FAIL fixes — 16 FAILs closed; 2 FAILs remain (p2_tests_pass + cascade child)
- **Wf-407-A3 / Wf-413-A4 / Wf-398-A1** SCAFFOLD_NOT_FUNCTIONAL honest-negative close precedent — refusal pattern honored here
- **`mol-metal-gpu-execution`** skill — `kill_then_adapt §0` directive requires pre-flight PASS + VRAM check + USER_OK; the VRAM + USER_OK conditions are met but pre-flight PASS is NOT met

## §7 Next-step recommendations

1. **Wf-420-A1** (CPU-only, ~5 LOC, ~5 min wall): Add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`. After this fix, `p2_tests_pass` PASS → `p3_wf343c2_per_wave_runtime_check` auto-PASS → pre-flight becomes 113/0/3/117.
2. **Wf-420-A2**: Re-run pre-flight fresh-run → expect 113/0/3/117.
3. **Wf-420-A3** (USER_OK-gated): Wf-299 actual GPU launch on cuda:0 (~12 hr wall per Wf-415-A1 + Wf-407-A3 closure path; Wave A only = ~5 hr).
4. **Wf-420+ continued**: Wf-394-A1 Lambda CFM wire-in patch application + Wf-246-C metal-coordination term retrain + atomic write protocol adoption migration completion (~41 remaining scripts) + Wf-379-A1 m1_auc 5-seed result close + paper §6 v4 write + paper §3 v4 update.

## §8 Wall budget actual

CPU-only: ~3 min (env probe + closeout authoring)
GPU: 0 sec (launch REFUSED per §1.4 pre-flight contract)

Combined wall: ~3 min CPU-only — well under any budget.

## §9 Honest framing per AGENTS.md §5.2

**DO NOT** paraphrase this refusal as "Wf-419-A3 attempted launch but encountered issues" — per §5.2 honest framing the refusal is:
- **DECISION**: REFUSE LAUNCH (kill_then_adapt §0)
- **REASON**: 2 pre-flight FAILs (p2_tests_pass + cascade child) per AGENTS.md §1.4 + §5.1
- **ROOT CAUSE**: PRE-EXISTING test bug `molmetal/tests/test_kendall_dann_wiring.py::test_dann_on_off` instantiates `MultiTaskWassersteinModel(..., use_dann=False)` but the model class doesn't accept `use_dann` kwarg
- **CLOSURE**: Wf-420-A1 5-LOC CPU-only fix
- **GPU cycles saved**: ~5 hr × Blackwell GPU + ~95 GB VRAM

No fabrication of metrics. No "launch attempted but aborted due to environmental factor" euphemism. The honest-negative framing is preserved per `party-twenty-spirit` §3.5.
'''


def main() -> int:
    path = 'TODO/completed/419a3_wave_a_launch.md'
    risk = safe_overwrite_check(path, CONTENT)
    if risk:
        print(f'OVERWRITE_RISK detected — caller decides')
        return 2
    atomic_write_file(path, CONTENT)
    trunc = detect_truncation(CONTENT)
    print(f'SHIPPED {path} truncation_flag={trunc}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())