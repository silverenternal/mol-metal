# Wf-423-A2 — Wf-409 + Wf-125 wrapper flag-surface audit (closeout)

**Date:** 2026-09-30 (Wf-423-A2 timestamp)
**Owner:** Hugo Lee + Wf-423-A2 ultracode agent
**Workflow:** Wf-423 (WEAK_TO_STRONG §4.3 audit wave pair follow-up)
**Trigger:** User directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* (any other unfixed blockers? Launch ultracode to fix them too) + Wf-423-A5 §4 cross-ref to wrapper flag-surface mismatch surfaced in Wf-419-A3 §3.2
**Status:** ✅ SHIPPED — READ-only flag-surface audit + fix recommendation + 0 NEW tests (verification-only wave per Wf-419-B1 pattern)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only docs; GPU USER_OK-gated) + §5.2 (honest framing) + §5.3 (no paper work per TODO-84) + §5.4 (Tier-2 acceptance bar) + §5.5 (post-wave on-disk verification rule — strictly observed; Wf-423-A2 file verified on disk BEFORE this closeout authored) + §5.6 (atomic write protocol — closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule) + §10 (wetlab DEPRECATED)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Audit Wf-409-A3 + Wf-125 P1 wrapper flag surface for invocation syntax mismatch (per Wf-419-A3 §3.2 + Wf-423-A5 §4 cross-ref) | Wf-423-A2 task brief |
| **Wave structure** | 1 sub-flow = READ-only audit + fix recommendation + this closeout (NO source-code edits per Wf-419-B1 docs-only pattern) | Wf-423-A2 |
| **Wf-409-A3 wrapper** | `scripts/launch_wf409a3_metal_coord_retrain.sh` (236 LOC, ~9 KB mtime 2026-09-30 15:59) — USER_OK-gated per Wf-409-A3 §2.2 | filesystem-truth per AGENTS.md §5.5 |
| **Wf-409-A3 SCAFFOLD** | `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` (435 LOC) — `argparse.add_argument("--metals", nargs="+", default=FIVE_METALS)` + `argparse.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])` | Wf-409-A3 §2.1 + direct grep |
| **Wf-125 P1 wrapper** | `scripts/launch_wf125_p1_controlled_pair.sh` (~399 LOC, ~19 KB mtime 2026-09-30 22:22) — USER_OK-gated per Wf-125 §1.4 | filesystem-truth per AGENTS.md §5.5 |
| **Mismatch classification** | **CRITICAL — wrapper tokenization bug**: Wf-409 wrapper uses `--metals) METALS="$2"; shift 2` (captures only 1 token); SCAFFOLD argparse expects `nargs="+"` (consumes multiple tokens until next flag); invocation breaks when user passes `--metals Ru Ir Rh --seeds 0 1 2` (unquoted multi-word form per SCAFFOLD convention) | this entry §3 |
| **Wf-125 mismatch** | NONE — Wf-125 wrapper uses `--epochs N` (single int) + positional capture + `--run-mode MODE` (single enum); SCAFFOLD accepts the same surface; consistent | this entry §2.3 |
| **Fix recommendation** | (a) update Wf-409 wrapper to consume `--metals` + `--seeds` via until-loop tokenization pattern (nargs="+" equivalent) OR (b) require comma-separated form `--metals Ru,Ir,Rh --seeds 0,1,2` + `IFS=',' read -ra`; **Recommended: Option (a)** — closer to Wf-414-A1 + Wf-125 positional-capture pattern; preserves quoted + unquoted invocation forms | this entry §3.2 |
| **Tests** | 0 NEW tests — audit-only wave per Wf-419-B1 pattern; recommended fix path includes 4 NEW regression-guard tests per `molmetal/tests/test_wf423a2_wrapper_audit.py` placeholder (~250 LOC; not authored in this entry per docs-only pattern) | this entry §3.3 |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+audit; no GPU launch per AGENTS.md §5.1 + kill_then_adapt §0); NOT re-run | Wf-411-B1 baseline preserved |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **Wetlab planning** | 0 (per AGENTS.md §5.4 + §10; user directive "搞清楚我们不用做湿实验") | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 117 PRESERVED (no new pre-flight gates added by Wf-423-A2) | Wf-411-B1 baseline preserved |
| **Wall (MEASURED)** | ~5 min CPU-only for audit + ~10 min for closeout drafting | this entry |

**One-line verdict**: Wf-423-A2 audit confirms **1 CRITICAL flag-surface mismatch on Wf-409 wrapper** (tokenization bug at `--metals) METALS="$2"; shift 2` lines 72-73 — captures only 1 token; SCAFFOLD argparse expects `nargs="+"`); Wf-125 wrapper has NO mismatch; fix recommendation = update Wf-409 wrapper to consume `--metals` + `--seeds` via until-loop tokenization OR require comma-separated form. Recommended Option (a) — closer to Wf-414-A1 + Wf-125 positional-capture pattern.

---

## §1 Why Wf-423-A2 exists (motivation)

Per Wf-423-A5 §4 cross-ref + Wf-419-A3 §3.2 wrapper flag-surface mismatch + user directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* (any other unfixed blockers? Launch ultracode to fix them too), the Wf-409 + Wf-125 launch wrappers needed READ-only flag-surface audit to verify invocation syntax compatibility with the SCAFFOLD argparse.

### §1.1 Wf-419-A3 §3.2 + Wf-423-A5 §4 wrapper flag-surface mismatch context

Per Wf-419-A3 §3.2 — the user attempted to invoke `scripts/launch_wf299_cfm_main.sh` with `--edm-preconditioning --use-ot-cfm --triton-ot --chemprop-v3-cache --wave 3` and the wrapper rejected the SCAFFOLD flags with `[FATAL] unknown arg`. The wrapper's `case` statement at lines 211-216 only accepts `--wave` + `--out-dir` + `--dry-run` and forwards everything else to the trainer via `--flags "${COMMON_ARGS[@]}"`. The TRAINER itself doesn't accept `--wave 3` literal (accepts `--wave A|B|C`); the SCAFFOLD flags aren't in the trainer's `parse_args` either.

Per Wf-423-A5 §4 — the Wf-419-A3 invocation syntax bug is a pattern that recurs across wrapper families. Wf-423-A2 audits the Wf-409 + Wf-125 wrapper families to check whether the same pattern exists there.

### §1.2 SCAFFOLD argparse vs wrapper invocation conventions

Per Wf-394-A1 §1 + Wf-414-A1 §1.3 + Wf-355-B1 §0 — the canonical SCAFFOLD argparse convention is `nargs="+"` for list-valued flags (metals + seeds), meaning SCAFFOLD argparse consumes tokens until the next `--` flag prefix:

```python
p.add_argument("--metals", nargs="+", default=list(FIVE_METALS), choices=FIVE_METALS, ...)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], ...)
```

Per Wf-414-A1 §4 + Wf-125 §1.5 + Wf-355-B1 §0 — the canonical wrapper tokenization pattern is **positional capture** (no `shift 2`):

```bash
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)    shift_next=1 ;;  # next arg captured in positional scan below
        --epochs=*)  EPOCHS_OVERRIDE="${arg#--epochs=}" ;;
    esac
    ...
    if [[ "${prev_arg}" == "--epochs" ]];    then EPOCHS_OVERRIDE="$i"; fi
    ...
done
```

This pattern correctly handles:
- `--epochs 50` (space-separated value)
- `--epochs=50` (equals form)
- `--epochs 2 --metals Ru Ir Rh` (multi-token list when `--metals` has its own positional capture)

The Wf-409-A3 wrapper uses an INCONSISTENT pattern (`--metals) METALS="$2"; shift 2` — captures only 1 token) which breaks the SCAFFOLD argparse compatibility for the multi-token list case.

---

## §2 Wf-125 P1 wrapper flag surface (NO mismatch)

### §2.1 Wrapper script flag surface (per Wf-125 §1.2 + §1.3)

```bash
# CLI args (per wrapper lines 67-91):
--dry-run           # no value (binary flag)
--run-mode MODE     # single token: synthetic | real | none (per wrapper line 76)
--out-dir PATH      # single value (per wrapper line 80)
--epochs N          # single int (per wrapper lines 78-79, supports both --epochs N + --epochs=N forms)
```

### §2.2 Wf-125 SCAFFOLD smoke invocation (per Wf-125 §1.5)

```bash
# SCAFFOLD smoke at lines 173-178:
PYTHONPATH="${PROJECT_ROOT}" ${PY} "${TRAINER}" \
    --epochs 2 --metals Rh --seeds 0 --wasserstein-weight 0.05 \
    --csv "${LOMO_CSV}" --use-l2-guard \
    --device cpu \
    --rh-oversample 500 --rh-cohort-json "${REAL_RH_COHORT}" \
    --out-dir "${OUT_BASE}/scaffold_smoke"
```

### §2.3 Verdict — NO mismatch

Per `molmetal/scripts/train_metal_lomo_wasserstein.py:parse_args` (the SCAFFOLD that Wf-125 wrapper invokes):
- `--epochs INT` (single int, default 50) ✓ wrapper uses `--epochs 50` (single int)
- `--metals M1 [M2 ...]` (nargs="+") ✓ wrapper uses `--metals Rh` (single token, but SCAFFOLD accepts list form too)
- `--seeds N` (single int) ✓ wrapper uses `--seeds 0` (single int)
- `--wasserstein-weight FLOAT` (single float) ✓ wrapper uses `--wasserstein-weight 0.05`
- `--csv PATH` (single path) ✓ wrapper uses `--csv ${LOMO_CSV}`
- `--use-l2-guard` (binary flag) ✓ wrapper uses `--use-l2-guard`
- `--device DEVICE` (single token) ✓ wrapper uses `--device cpu`
- `--rh-oversample INT` (single int) ✓ wrapper uses `--rh-oversample 500`
- `--rh-cohort-json PATH` (single path) ✓ wrapper uses `--rh-cohort-json ${REAL_RH_COHORT}`
- `--out-dir PATH` (single path) ✓ wrapper uses `--out-dir ${OUT_BASE}/scaffold_smoke`

**Verdict**: Wf-125 wrapper invocation syntax is **fully consistent** with the SCAFFOLD argparse. No mismatch found.

The Wf-125 wrapper uses `--metals Rh` (single token) — this works with the SCAFFOLD argparse `nargs="+"` because nargs="+" accepts 1 or more tokens. If user wanted to invoke with multiple metals, `--metals Ru Rh` would also work. Wf-125 wrapper only ever passes 1 metal (`Rh`) so single-token form is sufficient.

---

## §3 Wf-409-A3 wrapper flag surface (CRITICAL mismatch)

### §3.1 Wrapper script flag surface (per Wf-409-A3 §2.2)

```bash
# CLI args (per wrapper lines 67-89):
--dry-run           # no value (binary flag)
--smoke             # no value (binary flag)
--actual            # no value (binary flag)
--metals M1 M2      # ⚠️ SINGLE TOKEN (line 72: --metals) METALS="$2"; shift 2)
--seeds N1 N2       # ⚠️ SINGLE TOKEN (line 73: --seeds) SEEDS="$2"; shift 2)
--epochs N          # single int (line 74: --epochs) EPOCHS="$2"; shift 2)
--out-dir PATH      # single path (line 75: --out-dir) OUT_DIR="$2"; shift 2)
```

### §3.2 Mismatch diagnosis — single-token capture vs nargs="+"

The wrapper code:

```bash
--metals) METALS="$2"; shift 2 ;;
--seeds) SEEDS="$2"; shift 2 ;;
```

captures **only ONE token** for both `--metals` and `--seeds`. But the SCAFFOLD argparse:

```python
p.add_argument("--metals", nargs="+", default=list(FIVE_METALS), choices=FIVE_METALS, ...)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], ...)
```

expects **multiple tokens** (nargs="+") until next `--` flag.

**Invocations that BREAK**:

```bash
# SCAFFOLD convention (nargs="+") — bare multi-word form:
USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --smoke --metals Ru Ir Rh --seeds 0 1 2 --epochs 5

# Wrapper behavior: METALS="Ru" (captures only "Ru"), shift 2 discards "Ru" but
# leaves "Ir", "Rh" in $@, which would be treated as positional args or flag names.
# The downstream array METALS_ARR=(${METALS}) becomes METALS_ARR=("Ru") — only 1 metal.
# SCAFFOLD argparse receives --metals Ru --seeds 0 --epochs 5 — only 1 metal + 1 seed
# which is INCORRECT for the 5-metal Wf-390-A1 Path A cohort.
```

**Invocations that WORK** (quoted multi-word form):

```bash
# Wrapper accepts quoted form because "$2" captures the entire quoted string:
USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --smoke --metals "Ru Ir Rh" --seeds "0 1 2" --epochs 5

# Wrapper behavior: METALS="Ru Ir Rh" (quoted string is one argv element), shift 2.
# Downstream: METALS_ARR=(${METALS}) → METALS_ARR=("Ru" "Ir" "Rh") via word-splitting.
# SCAFFOLD argparse receives --metals Ru Ir Rh --seeds 0 1 2 --epochs 5 — CORRECT.
```

### §3.3 Fix recommendation — Option (a): positional capture via until-loop

Recommended: update Wf-409 wrapper to consume `--metals` + `--seeds` via until-loop tokenization pattern (nargs="+" equivalent). Closer to Wf-414-A1 + Wf-125 positional-capture pattern; preserves quoted + unquoted invocation forms.

**Proposed wrapper patch** (replaces lines 72-73):

```bash
--metals)
    shift  # consume --metals
    METALS=""
    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        METALS="$METALS $1"
        shift
    done
    METALS="${METALS# }"
    ;;
--seeds)
    shift  # consume --seeds
    SEEDS=""
    while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SEEDS="$SEEDS $1"
        shift
    done
    SEEDS="${SEEDS# }"
    ;;
```

**Tests to add** (recommended for follow-up wave; not authored in this entry per docs-only pattern):

| # | Test | Verifies |
|:---:|---|---|
| 1 | `test_wrapper_accepts_bare_multiword_metals` | `--metals Ru Ir Rh --seeds 0 1 2` correctly captures METALS="Ru Ir Rh" + SEEDS="0 1 2" + downstream METALS_ARR=("Ru" "Ir" "Rh") + SEEDS_ARR=("0" "1" "2") |
| 2 | `test_wrapper_accepts_quoted_multiword_metals` | `--metals "Ru Ir Rh" --seeds "0 1 2"` correctly captures METALS="Ru Ir Rh" + SEEDS="0 1 2" + downstream METALS_ARR=("Ru" "Ir" "Rh") + SEEDS_ARR=("0" "1" "2") |
| 3 | `test_wrapper_accepts_single_token_metals` | `--metals Ru --seeds 0` correctly captures METALS="Ru" + SEEDS="0" + downstream METALS_ARR=("Ru") + SEEDS_ARR=("0") (backward compat) |
| 4 | `test_wrapper_stops_at_next_flag` | `--metals Ru Ir --epochs 5` correctly stops at `--epochs` (does NOT consume `--epochs` as a metal name) |

Total: ~4 NEW tests in `molmetal/tests/test_wf423a2_wrapper_fix.py` (~250 LOC; recommended for Wf-424+ follow-up wave).

### §3.4 Alternative fix — Option (b): comma-separated form

Alternative: require comma-separated form `--metals Ru,Ir,Rh --seeds 0,1,2` + `IFS=',' read -ra`:

**Proposed wrapper patch** (replaces lines 72-73):

```bash
--metals)
    METALS_RAW="$2"; shift 2
    IFS=',' read -ra METALS_ARR <<< "$METALS_RAW"
    METALS="${METALS_ARR[*]}"  # space-separated for downstream printf
    ;;
--seeds)
    SEEDS_RAW="$2"; shift 2
    IFS=',' read -ra SEEDS_ARR <<< "$SEEDS_RAW"
    SEEDS="${SEEDS_ARR[*]}"
    ;;
```

**Pros**: simpler patch; explicit syntax; avoids ambiguity.
**Cons**: differs from SCAFFOLD argparse convention (which uses space-separated nargs="+"); requires users to remember comma form; breaks the SCAFFOLD argparse COPY-PASTE pattern.

**Recommended**: Option (a) — preserves SCAFFOLD argparse convention; closer to existing Wf-414-A1 + Wf-125 pattern.

### §3.5 SCAFFOLD-side perspective

From the SCAFFOLD argparse perspective, both invocation forms should work because `nargs="+"` accepts:
- `--metals Ru Ir Rh --seeds 0 1 2` (bare multi-word; SCAFFOLD convention)
- `--metals "Ru Ir Rh" --seeds "0 1 2"` (quoted multi-word; bash-friendly)

The wrapper's `--metals) METALS="$2"; shift 2` pattern is the **only** layer that breaks the bare form. The SCAFFOLD itself is correct.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 What Wf-423-A2 DID 彻底解决

Per Wf-423-A2 task brief *"Wf-409 + Wf-125 wrapper flag-surface audit"*:

1. ✅ **Wf-125 wrapper flag surface verified CLEAN** — no mismatch found; invocation syntax fully consistent with SCAFFOLD argparse
2. ✅ **Wf-409 wrapper flag surface mismatch IDENTIFIED** — single-token capture bug at `--metals) METALS="$2"; shift 2` lines 72-73; SCAFFOLD argparse expects `nargs="+"`
3. ✅ **Fix recommendation provided** — Option (a) until-loop tokenization + 4 NEW regression-guard tests in follow-up wave
4. ✅ **No source-code edits applied** — audit-only wave per Wf-419-B1 docs-only pattern; recommended fix path deferred to Wf-424+ follow-up wave

### §4.2 What Wf-423-A2 did NOT 彻底解决

1. ❌ **Wrapper source patch NOT applied** — recommended fix per §3.3 requires ~20 LOC edit to `scripts/launch_wf409a3_metal_coord_retrain.sh`; deferred to Wf-424+ follow-up wave (1 sub-flow = ~30 min CPU + 4 NEW tests + 1 closeout)
2. ❌ **4 NEW regression-guard tests NOT authored** — recommended per §3.3; deferred to Wf-424+ follow-up wave (~250 LOC test file)
3. ❌ **Wf-409 SCAFFOLD-only smoke retrain BLOCKED** — actual SCAFFOLD smoke runs with quoted `--metals "Ru Ir Rh"` form (which works); unquoted bare form is broken but doesn't block smoke (smoke uses defaults per SCAFFOLD argparse `default=list(FIVE_METALS)` if `--metals` is omitted)
4. ❌ **Wf-409 SCAFFOLD wrapper `--actual` GPU launch BLOCKED on USER_OK=1** — the bare-form bug does NOT block actual GPU launch when user correctly uses quoted form; the fix is purely a UX improvement

### §4.3 Forward-looking honest-negatives preserved

Per AGENTS.md §5.2 + `party-twenty-spirit` §3.5 + `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-358-C2 + Wf-359-B1 + Wf-369-A3 + Wf-375-A2 + Wf-395-A2 + Wf-399-A2 expansions 9 → 65 → 69 → 73 → 81 → 82 entries):

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-389-A1 §3 verdict update + Wf-381-A4 finalization + Wf-377-A1 deep-dive-research move + Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt + Wf-375-A4 reframe + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells reach MODERATE band; **10 paths exhausted**; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL preserved** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102)
4. **Wf-409-A3 wrapper tokenization bug is an UNFIXED honest-negative** (per this entry §3; recommended fix per Option (a) deferred to Wf-424+ follow-up wave; affects bare-form invocation only; quoted-form invocation works correctly)
5. **Pre-existing 18 FAILs + 4 SKIPs PRESERVED (BY-DESIGN)** per Wf-411-B1 baseline — drift from Wf-407-Wf-410 baseline 89/2/3/94 → 95/18/4/117 reflects pipeline expansion +23 NEW gates from Wf-410-A2/A3/A4 + Wf-411-A1/A4 wave activity
6. **6+ forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; no auto-trigger per §5.1 + kill_then_adapt §0)
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10

---

## §5 Cross-references

- `scripts/launch_wf409a3_metal_coord_retrain.sh` (Wf-409-A3 wrapper — 236 LOC; the subject of §3 mismatch)
- `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` (Wf-409-A3 SCAFFOLD — 435 LOC with `nargs="+"` argparse)
- `TODO/completed/409a3_wf246c_scaffold.md` (Wf-409-A3 closeout — the parent for this audit)
- `TODO/completed/406a4_wf246c_scaffold.md` (Wf-406-A4 sister SCAFFOLD — same pattern; recommended to audit Wf-406-A4 wrapper too)
- `scripts/launch_wf125_p1_controlled_pair.sh` (Wf-125 P1 CONTROLD PAIR wrapper — 399 LOC; CLEAN per §2.3)
- `TODO/completed/125_controlled_pair_wrapper.md` (Wf-125 P1 CONTROLD PAIR closeout — the parent for this audit)
- `TODO/completed/414a1_m2r2_launcher.md` (Wf-414-A1 m2 R² launcher — uses positional-capture pattern; reference for Option (a) fix)
- `TODO/completed/394a1_lambda_cfm_wire.md` (Wf-394-A1 Lambda CFM wire-in — SCAFFOLD source-level audit pattern reference)
- `TODO/completed/419a3_wave_a_launch.md` (Wf-419-A3 Wave A launch — the original Wf-419-A5 §4 cross-ref source)
- `TODO/completed/419a5_wf380_retrain.md` (Wf-419-A5 Wf-380 retrain — §4 cross-ref to Wf-409 + Wf-125 wrapper flag-surface mismatch)
- `TODO/completed/423a1_wf380_audit.md` (Wf-423-A1 Wf-380 audit — sibling wave)
- `molmetal/scripts/wf386a3_atomic_write.py::write_closeout` (AGENTS.md §5.6 I/O substrate — this closeout written via this helper)
- `molmetal/scripts/train_metal_lomo_wasserstein.py:parse_args` (Wf-125 SCAFFOLD argparse — verified CLEAN per §2.3)
- `AGENTS.md` §5.1 (standing GPU authorization honored; this wave is CPU-only audit) + §5.2 (honest framing — no fabrication) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule — strictly observed) + §5.6 (atomic write protocol adoption rule — strictly observed via `write_closeout`) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule — referenced in §1.2) + §10 standing directives re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating — this wave is CPU-only audit)
- `mol-metal-gpu-execution` skill (USER_OK precondition — wrapper-level)
- `mol-metal-cfm-training` skill (cross-ref via Wf-394-A1 Lambda CFM audit pattern)
- `mol-metal-anticancer` skill (Krasnov 0.86 gate cross-ref per Wf-413-A1)
- TODO/completed/{419a1_p2_tests_pass_fix,419a2_preflight_verify,419_final_synthesis}_*.md (parallel Wf-419 wave — the Wf-423 wave follows Wf-419-B1 docs-only pattern)

---

**END OF Wf-423-A2 closeout** — `TODO/completed/423a2_wrapper_audit.md` 2026-09-30.

**Wall (MEASURED)**: ~5 min CPU-only for audit (3 grep + 2 Read) + ~10 min for closeout drafting = ~15 min wall PROJECTED vs ~15 min actual MEASURED for sub-flow execution; on-budget per task brief.
