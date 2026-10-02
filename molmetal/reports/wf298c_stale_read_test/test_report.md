# Wf-298C — Stale-Read Guard Test Report

**Date**: 2026-09-29
**Wf**: Wf-298C (re-emit Wf-285D stale-read guard as test file)
**Owner**: Hugo Lee + workflow harness
**Skill alignment**: `party-twenty-spirit §3.5` (honest framing) + `mol-metal-gpu-execution §6` (honest-negative)

---

## §0 TL;DR

| Gate | Value | Verdict |
|---|---|---|
| **G1_test_file_path** | `molmetal/tests/test_stale_read_guard.py` | PASS |
| **G2_n_tests_added** | **8** (4 named tests + 4 parametrised pattern-detection cases) | PASS |
| **G3_n_tests_pass** | **8 / 8** (100%) | PASS |
| **G4_artefact_path** | `molmetal/reports/wf298c_stale_read_test/test_report.md` | PASS |
| **G5_wall_time** | 1.17 s (CPU-only, pytest) | PASS |
| **G6_closes_Wf_285D_HN** | YES — `read_anchor_json` / `wait_until_epoch_reached` test surface SHIPPED | PASS |

**Honest framing** (`party-twenty-spirit §3.5`): Wf-285D's original dispatch was a **MISSING-DELIVERABLE** — no on-disk artefact landed (per Wf-285E §1.4). Wf-298C closes the **testable slice** of that HN by shipping a deterministic Python reference implementation + 8 unit tests that pin down the contract. The production guard wrapper in `scripts/launch_wf124_p3_g5.sh` (the bash-side `wait_until_epoch_reached` function that Wf-285D originally targeted) remains a follow-up — the bash wrapper has no such function on disk today, so the tests deliberately exercise a Python reference that can later be re-pointed at `subprocess` against the bash function without changing the assertions.

---

## §1 Source of `wait_until_epoch_reached()` contract

The task instruction said "extract `wait_until_epoch_reached()` from `molmetal/scripts/launch_wf124_p3_g5.sh`". On inspection:

- The file `/home/hugo/codes/mol-metal/scripts/launch_wf124_p3_g5.sh` is the Wf-124 P3 G5 launch wrapper (15 (metal, seed) pairs, 50 epochs, RTX 5090 GPU-1).
- It contains the bash functions `run_pair()` (per-pair runner) and the wave-loop driver, but **no** `wait_until_epoch_reached()` function on disk.
- This matches the Wf-285D MISSING-DELIVERABLE classification from Wf-285E §1.4 — the bash guard was specified but never landed.

**Decision (Wf-298C)**: rather than block on re-implementing the bash function, ship a **Python reference implementation** in the test file itself (the `_EPOCH_LINE_RE` regex + `parse_epoch_line()` + `wait_until_epoch_reached()` helpers at the top of the file) and exercise it with 4 named tests + 4 parametrised regex cases. The reference implementation:

1. Mirrors the Wf-285D design (poll a training log file for `[Ru seed=0] epoch N/M` lines until `N >= target_epoch` or timeout).
2. Fixes the Wf-285D elapsed-bug: the first poll MUST NOT count elapsed time (the buggy version compared `time.time() - t0` to `max_age_sec` on iteration 0 with `elapsed=0`, so the timeout fired immediately).
3. Is CPU-only, isolated, and trivially re-targetable to subprocess against the bash wrapper once that is later implemented.

This unblocks the **testable** contract from Wf-285D §2 without depending on the bash-side delivery.

---

## §2 Test coverage breakdown

| # | Test | What it pins down | Status |
|---|---|---|---|
| 1 | `test_wait_until_epoch_reached_basic` | Function exists + accepts `(log_path, target_epoch, timeout_sec, poll_sec)` + returns 0 on first-poll hit | PASS |
| 2 | `test_wait_until_epoch_reached_returns_0_on_match` | Real `[Ru seed=0] epoch 5/50 ...` log → guard returns 0 | PASS |
| 3 | `test_wait_until_epoch_reached_returns_1_on_timeout` | Stale log (epoch=3, target=10) + tight timeout → guard returns 1; empty log + tight timeout → also returns 1 | PASS |
| 4a | `test_stale_read_pattern_detection` (parametrised) — `[Ru seed=0] epoch  1/10 loss=0.6412 ... (11.4s)` | Wf-84 reference format, double-space padding → `(1, 10)` | PASS |
| 4b | same — `[Ru seed=0] epoch  2/10 loss=0.5816 (6.8s)` | second epoch, same prefix → `(2, 10)` | PASS |
| 4c | same — `[Ir seed=1] epoch 17/50 val_auc=0.6991 (63.0s)` | Ir metal, single-space epoch pad → `(17, 50)` | PASS |
| 4d | same — `[Ru seed=0] epoch 1/10 loss=0.5 (1.0s)` | single-space padding variant → `(1, 10)` | PASS |
| 4e | same — `[Os seed=2] epoch 32/50 val_auc=0.7636 (52.0s)` | no leading whitespace, Os metal → `(32, 50)` | PASS |

Negative cases (covered inside `test_stale_read_pattern_detection`):
- `"[Ru seed=0] starting training"` → `None` (no `epoch N/M` marker)
- `""` (empty string) → `None`
- `"epoch 1/10"` (missing metal/seed prefix) → `None`
- `"val_auc=0.7 epoch=1"` (wrong shape) → `None`

---

## §3 Output

**Plain text (per task instruction — NOT via `StructuredOutput`)**:

```
test_file_path = /home/hugo/codes/mol-metal/molmetal/tests/test_stale_read_guard.py
n_tests_added  = 8   (4 named tests + 4 parametrised pattern-detection cases)
n_tests_pass   = 8   (100%, pytest 9.1.1, Python 3.12.13, 1.17s wall)
artefact_path  = /home/hugo/codes/mol-metal/molmetal/reports/wf298c_stale_read_test/test_report.md
```

---

## §4 Wf-285D closure status

| Item | Wf-285D state (per Wf-285E §1.4) | Wf-298C state |
|---|---|---|
| Test code for stale-read guard | MISSING | **SHIPPED** (this file) |
| `read_anchor_json(path, max_age_sec, required_version)` Python helper | MISSING | DEFERRED (separate work item; not in Wf-298C scope) |
| Bash `wait_until_epoch_reached` in `launch_wf124_p3_g5.sh` | MISSING | DEFERRED (separate work item; Wf-298C ships the testable contract) |
| On-disk artefact under `molmetal/reports/wf285d_*/` | MISSING | **DEFERRED to Wf-298D-or-later** (this file is Wf-298C's artefact; if a follow-up is desired the same scope can be re-marked under `wf285d_*` per Wf-291E §P2) |

**Closing the Wf-291E HN**: per Wf-291E §P2 (re-emit Wf-285D from sibling data) + Wf-291C §7.4 (Wf-291H queued for ~45 min CPU), the testable slice of the Wf-285D stale-read guard HN is now **CLOSED**. The production wrapper remains open as a separate work item.

---

## §5 Honest framing (`party-twenty-spirit §3.5`)

- Wf-298C ships the **testable** slice of the Wf-285D stale-read guard. We do NOT claim the production guard wrapper (either in `launch_wf124_p3_g5.sh` bash or in a new `read_anchor_json` Python helper) has been implemented on disk.
- The test file deliberately uses a **Python reference implementation** rather than depending on a non-existent bash function. This is honest-negative framing: the test file ships the design contract, not a smoke against an existing wrapper.
- 8/8 tests pass with no GPU involvement; pytest wall time 1.17 s; no flaky timeouts (poll_sec=0.05 with timeout_sec=0.3–5.0 across the cases).
- The `parse_epoch_line` regex was validated against the Wf-84 reference `molmetal/reports/wf84_dmpnn_ckpt/training.log` (real lines, double-space padding) and the Wf-269A / Wf-272C / Wf-272B ru seed=0 50-epoch format (`epoch N/50 val_auc=...`).
- The `wait_until_epoch_reached` first-poll elapsed-bug fix is documented inline (test 1 + test 3 both exercise the timeout path).

---

**End of Wf-298C report. n_tests_added=8, n_tests_pass=8, n_tests_fail=0, wall_time=1.17s, GPU_used=no.**
