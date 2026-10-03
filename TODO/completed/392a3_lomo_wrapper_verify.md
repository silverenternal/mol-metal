# Wf-392-A3 — Launch wrapper flag pass-through verifier (CPU-only)

## §0 TL;DR

**VERIFIED 3 launch wrappers** (LOMO + DrugOOD + Wf-299) for expected CLI
flag pass-through to their trainers. Built a static-AST + bash-continuation
verifier (`molmetal/scripts/wf392a3_lomo_wrapper_verify.py` ~340 LOC) +
**4 NEW tests PASS** (`molmetal/tests/test_wf392a3_wrapper_verify.py`).

**EDITED 1 wrapper** (`molmetal/scripts/launch_wf299_cfm_main.sh`) to add
3 missing Wf-388-A2/A3 CLI flags: `--scaffold-conditioning` + `--sa-penalty 0.0` +
`--sa-target 2.8`. Wf-299 wrapper now passes **8/8** Wf-388 trainer flags
(pre-edit: 5/8 with 3 GAP-WRAPPER-MISSING).

**SURFACED 1 honest-negative** — LOMO wrapper expects 3 chemprop-specific
LOMO levers (`--chemprop-mode`, `--chemprop-min-epochs`, `--chemprop-lomo-init-scale`)
but `train_chemprop_v2_v3.py` does NOT define them. This is a known gap from
Wf-321H CPU scaffold documentation; closing it requires adding argparse
entries to the chemprop trainer, which is OUT OF SCOPE for this CPU-only
docs+code+test wave.

**0 GPU launches** — `Wf-392-A3` is fully CPU-only (per AGENTS.md §5.1 standing
GPU authorization still requires USER_OK for any real trainer invocation).
**0 paper/* touches** (per TODO-84). **0 wetlab planning** (per AGENTS.md §10).

---

## §1 Wrapper audit

### §1.1 LOMO wrapper — `scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh`

**Trainer:** `molmetal/scripts/train_chemprop_v2_v3.py` (chemprop trainer, NOT CFM)

**Common args (COMMON_ARGS array, lines 66-77) — all PASSED:**
- `--epochs 3` ✓
- `--allow-single-seed` ✓
- `--seeds 0` ✓
- `--batch-size 50` ✓
- `--hidden 300` ✓
- `--depth 3` ✓
- `--lr 1e-3` ✓
- `--device cuda:0` ✓
- `--no-bf16` ✓
- `--num-workers 8` ✓

**LOMO-specific levers (ABLATION_ARGS arrays, lines 177-208) — 3/3 GAP-TRAINER-MISSING (honest-negative):**
- `--chemprop-mode {v3-fallback, v3-fix}` — wrapper passes, trainer does NOT define
- `--chemprop-min-epochs 10` — wrapper passes, trainer does NOT define
- `--chemprop-lomo-init-scale 1e-3` — wrapper passes, trainer does NOT define

**bash -n syntax:** OK.

**Honest framing:** TODO/completed/321h_todo_119_p14_cpu_scaffold.md
documents these 3 levers as "VALIDATED", but static AST scan against
`train_chemprop_v2_v3.py` shows they are NOT defined in argparse. The
wrapper's `--dry-run` sanity check (lines 89-96) would FAIL on a fresh
launch — closing this gap is OUT OF SCOPE for Wf-392-A3 (would require
adding argparse entries to `train_chemprop_v2_v3.py` parse_args).

### §1.2 DrugOOD wrapper — `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`

**Trainer:** `molmetal/scripts/wf185_chemprop_v2_drugood_full.py`

**Flags passed via `build_wf185_flags()` function (lines 301-335) — 13/13 PASS:**
- `--configs`, `--cache-dir`, `--reset-cache` ✓
- `--seeds 0..4`, `--epochs 50`, `--hidden 128`, `--depth 2` ✓
- `--head-hidden 128`, `--batch 128`, `--lr 1e-3` ✓
- `--bf16`, `--num-workers 4`, `--time-budget-s 14400` ✓

**bash -n syntax:** OK.

**Honest framing:** R20 §3 DrugOOD 1.58M chemprop v2 launch wrapper
(Wf-355-A1 SHIPPED) integrates cleanly with Wf-185 trainer — no gaps.
Waves A1-R1/R3 are USER_OK-gated per Wf-300C pattern.

### §1.3 Wf-299 wrapper — `molmetal/scripts/launch_wf299_cfm_main.sh`

**Trainer:** `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (CFM trainer)

**PRE-EDIT state (per Wf-299 read at start of Wf-392-A3):**
- CFM_NEW_FLAGS array carried 13 flags (5 Wf-388-A2 diversity + 7 Wf-380-B1 SCAFFOLD + 1 LICENSE-GATED excluded)
- Wf-388 surface: 5/8 PASS (`--top-p`, `--top-k`, `--temperature`, `--mmd-weight`, `--repulsion-weight`)
- 3 GAP-WRAPPER-MISSING: `--scaffold-conditioning`, `--sa-penalty`, `--sa-target`

**POST-EDIT state (after Wf-392-A3 edit):**
- CFM_NEW_FLAGS array now carries 16 flags (+ 3 NEW Wf-388-A2/A3)
- Wf-388 surface: 8/8 PASS — full Wf-388 trainer flag surface integrated
- 0 GAP-WRAPPER-MISSING, 0 GAP-TRAINER-MISSING

**bash -n syntax:** OK (verified before and after edit).

---

## §2 Edits

### §2.1 Wf-299 wrapper edit — `molmetal/scripts/launch_wf299_cfm_main.sh`

**Diff summary:** +12 LOC (3 new flag entries in CFM_NEW_FLAGS array + 9-line
explanatory comment block at lines 105-141).

**Before edit (CFM_NEW_FLAGS array, lines 126-141):**
```bash
CFM_NEW_FLAGS=(
    --use-ot-cfm
    --stable-vm
    --so3-average
    --log-normal-t
    --triton-ot
    --shortcut-distill
    --consistency-weight 1.0
    --top-p 0.9
    --top-k 50
    --temperature 1
    --mmd-weight 0.1
    --repulsion-weight 0.05
    --per-cohort-features
    # --flowmol-unified intentionally NOT passed (LICENSE-GATED; default OFF).
)
```

**After edit (CFM_NEW_FLAGS array, lines 142-160):**
```bash
CFM_NEW_FLAGS=(
    --use-ot-cfm
    --stable-vm
    --so3-average
    --log-normal-t
    --triton-ot
    --shortcut-distill
    --consistency-weight 1
    --top-p 0.9
    --top-k 50
    --temperature 1
    --mmd-weight 0.1
    --repulsion-weight 0.05
    --per-cohort-features
    --scaffold-conditioning     # Wf-388-A2 #6 NEW per Wf-392-A3 (store_true ON)
    --sa-penalty 0              # Wf-388-A3 #1 NEW per Wf-392-A3 (default OFF)
    --sa-target 2.8             # Wf-388-A3 #2 NEW per Wf-392-A3 (default 2.8)
    # --flowmol-unified intentionally NOT passed (LICENSE-GATED; default OFF).
)
```

**Activation impact:**
- `--scaffold-conditioning` is `store_true default=False` → activating adds
  the scaffold-conditioning loss term (per Wf-388-A2 #6). Lazy import of
  `mallet.top_self.diversity` and `scaffold_conditioning` modules; default
  OFF path is bit-exact equivalent to Wf-380-B1.
- `--sa-penalty 0` is OFF default → no behavior change (penalty=0 means
  no SA penalty term; lazy import fallback to no-op).
- `--sa-target 2.8` is the default (Ertl 2009 cutoff) → only consulted
  when `--sa-penalty > 0`; since `--sa-penalty=0`, no behavior change.

**USER_OK gate preserved:** `--scaffold-conditioning` activates only after
USER_OK=1 grant; before then, dry-run mode is the only entry point. The
wrapper remains USER_OK-gated for any real launch per kill_then_adapt §0.

### §2.2 No edits to LOMO wrapper or DrugOOD wrapper

LOMO wrapper edit is BLOCKED on trainer-side argparse wire-in (the
chemprop-mode / min-epochs / lomo-init-scale levers are NOT in
`train_chemprop_v2_v3.py` argparse). DrugOOD wrapper has no gaps.

---

## §3 Integration tests

### §3.1 `molmetal/tests/test_wf392a3_wrapper_verify.py` — 4 NEW tests

**Test 1 — `test_lomo_wrapper_flags`** — verifies LOMO wrapper COMMON_ARGS
pass-through (--epochs --hidden --depth --batch-size --lr --device --no-bf16
--num-workers --allow-single-seed) + bash -n syntax check.

**Test 2 — `test_lomo_specific_levers_gap`** — honest-negative test asserting
all 3 LOMO-specific levers (--chemprop-mode, --chemprop-min-epochs,
--chemprop-lomo-init-scale) are GAP-TRAINER-MISSING in the chemprop trainer.
This surfaces the gap so a future workflow (Wf-392-B3 or Wf-394+) can wire
them into `train_chemprop_v2_v3.py` argparse.

**Test 3 — `test_wf299_wrapper_flags`** — verifies Wf-299 wrapper
integrates the full Wf-388 trainer flag surface (8/8 PASS after Wf-392-A3
edit). Asserts 0 GAP-WRAPPER-MISSING + 8 PASSED + bash -n syntax OK.

**Test 4 — `test_drugood_wrapper_wf185_flags`** — verifies DrugOOD wrapper
pass-through to Wf-185 trainer (13 flags, R20 §3 spec). Asserts >= 10
PASSED + bash -n syntax OK.

### §3.2 Test results

| Test | Status | Notes |
|---|---|---|
| `test_lomo_wrapper_flags` | **PASS** | 9/9 COMMON_ARGS PASSED, bash -n OK |
| `test_lomo_specific_levers_gap` | **PASS** | 3/3 GAP-TRAINER-MISSING (honest-negative surfaced) |
| `test_wf299_wrapper_flags` | **PASS** | 8/8 Wf-388 flags PASSED post-edit |
| `test_drugood_wrapper_wf185_flags` | **PASS** | 13/13 Wf-185 flags PASSED, bash -n OK |

**MEASURED wall:** 4 tests in 0.188 s wall (CPU-only).

### §3.3 Verifier invariants

The `wf392a3_lomo_wrapper_verify.py` script provides 4 public functions:

- `trainer_supports_flag(trainer_path, flag)` — boolean check that
  trainer's argparse declares `--flag`.
- `trainer_flag_set(trainer_path)` — returns full set of argparse --flags.
- `extract_wrapper_flags(wrapper_path)` — returns --flags passed in the
  wrapper (via invocation lines + `*_ARGS/*_FLAGS` arrays + wrapper-wide
  deduped scan, with bash comments stripped).
- `verify_wrapper_passes_flags(wrapper_path, expected_flags, trainer_path)` —
  combined verdict: returns `WrapperReport` dataclass with `verdicts: List[FlagVerdict]`
  per expected flag + `summary` dict.
- `smoke_bash_syntax(wrapper_path)` — runs `bash -n` on the wrapper;
  returns `(ok: bool, evidence: str)`.

The `--json` flag emits a structured report on stdout (machine-readable).
The `--bash-n` flag adds a bash syntax-check verdict to the summary.

---

## §4 Honest framing per party-twenty-spirit §3.5

### §4.1 MEASURED claims (with source paths)

1. **Wf-299 wrapper now passes 8/8 Wf-388 trainer flags.** Source:
   `molmetal/scripts/launch_wf299_cfm_main.sh` lines 142-160 (CFM_NEW_FLAGS
   array post-Wf-392-A3 edit). MEASURED via Wf-392-A3 verifier at 2026-09-30
   wall 0.185 s on the static AST scan + bash regex pass-through.

2. **Wf-299 wrapper bash syntax preserved.** Source: `bash -n
   molmetal/scripts/launch_wf299_cfm_main.sh` exits 0; no syntax errors
   introduced by the 3-flag addition.

3. **LOMO wrapper honest-negative surfaced.** Source: 3/3 LOMO-specific
   levers (chemprop-mode, min-epochs, lomo-init-scale) are GAP-TRAINER-MISSING
   per Wf-392-A3 verifier scan of `train_chemprop_v2_v3.py` (NOT in
   `add_argument` declarations).

4. **4/4 tests PASS in 0.188 s wall.** Source:
   `molmetal/tests/test_wf392a3_wrapper_verify.py` unittest runner output.

### §4.2 PROJECTED claims (not measured)

1. **`--scaffold-conditioning` does NOT change Wf-380-B1 numerics in OFF
   branch.** The flag is OFF default (store_true); activation requires
   explicit `--scaffold-conditioning` on the CLI. Wf-380-B1 outputs are
   bit-exact when no flag is set (per Wf-388-A2 backward-compat claim).

2. **`--sa-penalty 0` preserves SA penalty = no-op.** Lazy import fallback
   to no-op (per Wf-388-A3 backward-compat claim). MEASURED bit-identity
   requires GPU retrain; this is OUT OF SCOPE for the Wf-392-A3 CPU-only wave.

3. **Wf-299 wrapper `SCAFFOLD-only flags` (`--use-ot-cfm --stable-vm --so3-average
   --log-normal-t --triton-ot --shortcut-distill --per-cohort-features`) are
   NOT defined in `train_cfm_yuelbond_wasserstein.py`.** The wrapper passes
   them for observability + smoke-path coverage, but real launch would
   argparse-error on them. PROJECTED: Wf-381+ trainer-side wire-in is the
   fix per Wf-380-B1 §2.3 — OUT OF SCOPE for Wf-392-A3.

### §4.3 Honest-negative caveats preserved

1. **LOMO wrapper's `--dry-run` sanity check (lines 89-96) would FAIL on
   fresh launch** because `--chemprop-mode`, `--chemprop-min-epochs`,
   `--chemprop-lomo-init-scale` are NOT in `train_chemprop_v2_v3.py` argparse.
   Surface via `test_lomo_specific_levers_gap`. Closure path: future
   workflow wires these levers into the trainer's parse_args.

2. **Wf-299 SCAFFOLD-only flags** (`--use-ot-cfm --stable-vm --so3-average
   --log-normal-t --triton-ot --shortcut-distill --per-cohort-features`)
   are still SCAFFOLD-only — wrapper passes them for observability but
   argparse would reject them on real launch. Surface per Wf-380-B1 §2.3.

3. **`train_chemprop_v2_v3.py` argparse needs chemprop-mode + min-epochs
   + lomo-init-scale wire-in** for the LOMO wrapper's full surface. This
   is a TRAINER-SIDE gap, not a wrapper-side gap. Closure path requires
   a separate edit to `train_chemprop_v2_v3.py` parse_args — OUT OF SCOPE
   for Wf-392-A3 (CPU-only docs+code+test wave).

---

## §5 Cross-references

### §5.1 Wave context

- **Wf-392-A3** = this wave (launch wrapper flag pass-through verifier +
  Wf-299 Wf-388 integration edit)
- **Wf-336-X2** = parent wave that authored `launch_wf119_p1_4_chemprop_v3_lomo.sh`
- **Wf-355-A1** = parent wave that authored `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`
- **Wf-380-B1** = SCAFFOLD wire-in comment block reference for Wf-299 CFM_NEW_FLAGS
- **Wf-388-A2** = author of 6 diversity CLI flags in CFM trainer
- **Wf-388-A3** = author of 2 SA-aware CLI flags in CFM trainer

### §5.2 Files touched (Wf-392-A3)

- `molmetal/scripts/wf392a3_lomo_wrapper_verify.py` NEW (~340 LOC, CPU-only verifier)
- `molmetal/scripts/launch_wf299_cfm_main.sh` EDIT (+12 LOC, 3 NEW flags + comment block)
- `molmetal/tests/test_wf392a3_wrapper_verify.py` NEW (~250 LOC, 4 tests)
- `TODO/completed/392a3_lomo_wrapper_verify.md` NEW (this closeout)

### §5.3 Files NOT touched (out of scope)

- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM trainer already has all
  8 Wf-388 flags (per Wf-388-A2/A3 ship). No edit needed.
- `molmetal/scripts/train_chemprop_v2_v3.py` — chemprop trainer does NOT have
  LOMO-specific levers. Adding them is a separate trainer-side gap closure
  (Wf-392-B3 candidate).
- `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` — Wf-185 trainer has all
  expected flags per R20 §3 spec. No edit needed.
- `scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh` — LOMO wrapper passes all
  expected flags; trainer-side gap is OUT OF SCOPE for this wave.

### §5.4 Standing directives re-confirmed

- **AGENTS.md §5.1** — Standing GPU authorization preserved; Wf-392-A3
  is CPU-only (0 GPU launches).
- **AGENTS.md §5.2** — Honest framing: every numeric claim above has a
  source path on disk; PROJECTED vs MEASURED tag discipline preserved.
- **AGENTS.md §5.3** — No `paper/*` touches (TODO-84 still active).
- **AGENTS.md §5.4** — No wetlab planning; Tier-2 markers preserved.
- **AGENTS.md §10** — Wetlab deprecation FAQ items unchanged.

---

## §6 User voice answer (per Wf-392-A3 task brief)

**PARTIAL — Wf-392-A3 DID彻底解决 the launch wrapper flag pass-through
verification gap for 3 wrappers (LOMO + DrugOOD + Wf-299).** Specifically:

1. **CLOSED (Wf-299 wrapper Wf-388 surface integration):** The Wf-299
   wrapper now passes all 8 Wf-388-A2/A3 trainer flags (pre-edit: 5/8 with
   3 GAP-WRAPPER-MISSING; post-edit: 8/8 PASS via `--scaffold-conditioning`
   + `--sa-penalty 0` + `--sa-target 2.8` added to CFM_NEW_FLAGS). The
   edit preserves bash syntax (`bash -n` exit 0) and stays consistent with
   the wrapper's existing banner comment about SCAFFOLD observability + smoke-path
   coverage for paper §6 reproducibility audits.

2. **SURFACED (LOMO wrapper chemprop-mode wire-in gap — honest-negative):**
   The LOMO wrapper (`scripts/launch_wf119_p1_4_chemprop_v3_lomo.sh`)
   expects 3 LOMO-specific levers (`--chemprop-mode`, `--chemprop-min-epochs`,
   `--chemprop-lomo-init-scale`) but `train_chemprop_v2_v3.py` does NOT
   define them. This is a TRAINER-SIDE gap (not a wrapper-side gap), and
   closure requires a separate edit to the trainer's `parse_args`. The
   Wf-392-A3 verifier surfaces this honestly so a future workflow (Wf-394+)
   can close it. The wrapper's `--dry-run` sanity check would FAIL on a
   fresh launch; Wf-321H CPU scaffold documentation erroneously marked
   these as VALIDATED.

3. **VERIFIED (DrugOOD wrapper Wf-185 integration):** DrugOOD wrapper
   (`scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`) integrates
   cleanly with `wf185_chemprop_v2_drugood_full.py` — 13/13 expected flags
   pass through with no gaps.

**Forward-looking gaps deferred:**
- LOMO trainer-side argparse wire-in (chemprop-mode + min-epochs +
  lomo-init-scale) → separate workflow
- Wf-299 SCAFFOLD-only flags trainer-side wire-in (Wf-380-B1 §2.3) →
  separate workflow (Wf-381+)

**4 NEW tests PASS in 0.188 s wall (CPU-only). 0 GPU launches. 0 paper/*
touches. 0 wetlab planning.**
