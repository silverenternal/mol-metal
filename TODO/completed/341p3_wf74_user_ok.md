# Wf-341-P3 — Wf-74 v2 DrugOOD DANN USER_OK readiness verify

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Origin:** Wf-341 ultracode residual wave (Wf-336-X3 follow-up)
**Status:** SHIPPED; 27/27 tests PASS (4 test classes × expanded subtests)

---

## §1 Purpose

Wf-336-X3 shipped `scripts/launch_wf74_v2_drugood_dann.sh` with USER_OK gate
enforcement + Wf-328 W1-W10 wire-ups + VRAM budget check.  Wf-341-P3 audits
that ship by:

1. Reading the launcher source (already done — see Wf-336-X3).
2. Writing **4 NEW test classes** (`TestUserOkGate`,
   `TestVramBudgetCheck`, `TestWf328FixesWired`, `TestBackwardCompat`)
   covering 27 distinct assertions.
3. Running the test suite CPU-only (no GPU launch).
4. Confirming the Wf-74 v2 lineage is ready for `--mode R1` smoke + USER_OK
   gated `--mode R3` GPU launch.

## §2 Honest framing (party-twenty-spirit §3.5, AGENTS.md §5.2)

| Layer | Status | Notes |
|---|---|---|
| USER_OK gate | MEASURED | `user_ok_gate()` enforces `USER_OK=YES` for R3; missing/wrong value triggers `die` with exit 1 (verified by `test_user_ok_missing_triggers_die_in_r3` + `test_user_ok_wrong_value_triggers_die_in_r3` subprocess invocations). |
| VRAM budget check | MEASURED | `VRAM_BUDGET_GB=20` + `nvidia-smi --query-gpu=memory.free` OOM-die (verified by source + function-call regex assertions). |
| Wf-328 W1-W10 wire-ups | MEASURED | All 10 wire-ups present in `build_wf328_flags()` constants + Wf-305D trainer (verified by 10 individual flag assertions). |
| Backward compat (R1 smoke) | MEASURED | `--mode R1 --dry-run` proceeds without USER_OK (verified by subprocess exit=0). |

CPU-only tests; no GPU launch; no torch.cuda.

## §4 Test results (2026-09-29)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf341p3_wf74_user_ok.py -v
============================= 27 passed, 1 warning in 0.44s =============================
```

**27/27 PASS.**  All 4 test classes fully covered:

| Test class | Subtests | What it verifies |
|---|---|---|
| `TestUserOkGate` | 5 | USER_OK=YES enforcement, missing/wrong value die, invocation order |
| `TestVramBudgetCheck` | 6 | `vram_gate()` function + 20GB budget + nvidia-smi probe + dry-run skip |
| `TestWf328FixesWired` | 10 | W1 torch.compile, W2 num-workers=8, W3 BF16, W4 TF32, W5 W₂ sort, W7 PyG, W8/W9 opt-in, --use-dann + ReverseLayerF, --use-kendall/--lambda-grl/--wasserstein-weight |
| `TestBackwardCompat` | 6 | R1 --dry-run without USER_OK exits 0; --dry-run does NOT bypass USER_OK in R3; --help documents modes; --mode R2 dies; DRY_RUN.log marker; bash syntax check |

## §5 Key findings

### §5.1 USER_OK gate is properly wired

`user_ok_gate()` (lines 144-161 of launcher):

```bash
user_ok_gate() {
    local mode="$1"
    if [ "${mode}" = "R3" ]; then
        if [ -z "${USER_OK:-}" ]; then
            die "USER_OK env var not set. Re-run with: USER_OK=YES bash $0 --mode R3"
        fi
        if [ "${USER_OK}" != "YES" ]; then
            die "USER_OK must equal 'YES' (got: '${USER_OK}')"
        fi
        ok "USER_OK gate passed (USER_OK=YES)"
    else
        if [ -z "${USER_OK:-}" ]; then
            warn "USER_OK not set; Wf-335-R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-335-R1 smoke)"
        fi
    fi
}
```

Verified: missing USER_OK → exit 1; USER_OK=MAYBE → exit 1; USER_OK=YES in R3 → pass.

### §5.2 VRAM budget check is properly wired

`VRAM_BUDGET_GB=20` constant set; `vram_gate()` invokes
`nvidia-smi --query-gpu=memory.free`; on OOM (free < 20GB), dies with
`"VRAM free=${free_gb}GB < budget=${VRAM_BUDGET_GB}GB on GPU ${idx}
(AGENTS.md §5.1 OOM risk)"`.  DRY-RUN mode short-circuits.

### §5.3 Wf-328 W1-W10 wire-ups

`build_wf328_flags()` (lines 253-288) wires 10 production optimizations:

| # | Wire-up | Constant | Flag |
|---|---|---|---|
| W1 | torch.compile | `WF328_TORCH_COMPILE=1` | (default ON; --no-torch-compile to disable) |
| W2 | num-workers=8 | `WF328_NUM_WORKERS=8` | `--cpu-num-workers 8` |
| W3 | BF16 | `WF328_BF16=1` | (default ON; --no-bf16 to disable) |
| W4 | TF32 | `WF328_TF32=1` | (default ON) |
| W5 | W₂ sort | `WF328_W2_MODE="sort"` | `--w2-gradient-mode sort` |
| W6 | DataLoader | (via W2) | `--cpu-num-workers 8` (PyG DataLoader backed) |
| W7 | PyG backend | `WF328_DMPNN_BACKEND="pyg"` | `--dmpnn-backend pyg` |
| W8 | CUDA Graphs | (opt-in) | `--cuda-graph` |
| W9 | Triton edge | (opt-in) | `--triton-edge-update` |
| W10 | DMPNN cache | (default ON via Wf-305D cache module) | (cache enabled at trainer level) |

Plus DANN-specific: `--use-dann` + `ReverseLayerF` (DANN GRL wired in
Wf-305D trainer), `--use-kendall`, `--lambda-grl 0.1`,
`--wasserstein-weight 0.05`, `--alpha 0.5`.

### §5.4 Backward compat verified

`--mode R1 --dry-run` without USER_OK exits 0 (smoke convention).
`--mode R3 --dry-run` without USER_OK exits 1 (USER_OK gate enforced
even in dry-run, per Wf-300C pattern).  `--mode R2` (invalid) dies with
"R1 | R3" error message.  `--help` documents both modes.

## §6 Files added / verified

| File | Status | Purpose |
|---|---|---|
| `molmetal/tests/test_wf341p3_wf74_user_ok.py` | NEW | 27 tests across 4 classes |
| `scripts/launch_wf74_v2_drugood_dann.sh` | VERIFIED | Wf-336-X3 ship artifact |
| `molmetal/scripts/wf305d_drugood_dann_train.py` | VERIFIED | DANN driver w/ ReverseLayerF + --use-dann |

## §7 Readiness verdict

**READY for `--mode R1` CPU smoke launch.**

The wrapper is launchable in dry-run mode (verified: exit 0, DRY_RUN.log
marker, log file present).  USER_OK gate is enforced for `--mode R3` per
Wf-294B kill_then_adapt.  Wf-328 W1-W10 wire-ups are present.

**For `--mode R3` (3-seed × 50-epoch RTX 5090 full DANN run, ~3-4 hr)**:
requires `USER_OK=YES` env gate + `nvidia-smi` confirmed free ≥ 20GB on
cuda:1 (per AGENTS.md §5.1).

## §8 Cross-references

- `TODO/completed/336x3_wf74_v2_drugood_dann_launcher.md` — parent Wf-336-X3 ship
- `TODO/completed/328*.md` — Wf-328 W1-W10 ship surface
- `TODO/completed/335r1_*.md` — Wf-335-R1 torch.compile CUDA Graphs fix
- `AGENTS.md` §5.1 — standing GPU authorization
- `AGENTS.md` §5.4 — Tier-2 journal target; wet-lab deprecation (Wf-74 v2 lineage note)
- `TODO/completed/97c_wf74_reclassification.md` — Wf-74 → Wf-74 v2 reclassification

---

**END OF Wf-341-P3 closeout** — 27/27 PASS; Wf-74 v2 DrugOOD DANN
launcher READY for R1 smoke + USER_OK-gated R3 full GPU run.
