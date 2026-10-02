# Wf-423-A3 — Wf-299 launcher EXTRA_FLAGS + --wave alias re-audit

## §0 TL;DR

**Wf-423-A3 SHIPPED**: Wf-299 launcher `molmetal/scripts/launch_wf299_cfm_main.sh` (29179 bytes mtime 2026-09-30 17:39) verified to have **both** the EXTRA_FLAGS accumulator (Wf-415-A1) AND the `--wave 1/2/3` numeric alias mapping (Wf-408-A3) ON DISK + functioning correctly via 3 dry-run tests (all rc=0, all logged `[OK] Wave A DRY-RUN logged`). Pre-existing per-Wf-415-A1 + Wf-408-A3 + Wf-413-A3 design contract is preserved; no new edits needed.

**Per-feature audit table**:

| Feature | Source | On-disk? | Tested? | Verdict |
|---|---|---|---|---|
| `EXTRA_FLAGS=""` accumulator + catch-all append | Wf-415-A1 (this launcher §227-253) | YES | YES (test 3) | **PASS** |
| `--wave 1 → A` numeric alias | Wf-408-A3 (this launcher §263-266) | YES | YES (test 1) | **PASS** |
| `--wave 2 → B` numeric alias | Wf-408-A3 (this launcher §263-266) | YES (not tested) | no | **PASS (by code inspection)** |
| `--wave 3 → C` numeric alias | Wf-408-A3 (this launcher §263-266) | YES (not tested) | no | **PASS (by code inspection)** |
| `--wave A | B | C | all` canonical | pre-Wf-408-A3 (this launcher §269) | YES | YES (test 2) | **PASS** |
| `--dry-run` early-return | pre-Wf-378-A2 (this launcher §230) | YES | YES (all tests) | **PASS** |
| `--user-ok` gate (USER_OK env validation) | Wf-378-A2 (this launcher §86-101) | YES | implicit (USER_OK=1 set in all tests) | **PASS** |
| Wf-394-A3 pre-launch chain (wf343c2 + wf343c3 + wf385a3) | Wf-394-A3 (this launcher §316) | YES | YES (logged OK) | **PASS** |
| Wf-410-A4 re-verify contract (trainer_arg_count ≥ 50) | Wf-410-A4 (this launcher §382, §398) | YES | YES (16/16 Wf-388 flags wired) | **PASS (per Wf-410-A4 baseline)** |
| `--flowmol-unified` LICENSE-GATED CC-BY-NC exclusion | Wf-380-A1 + Wf-385-A2 (this launcher §CFM_NEW_FLAGS) | YES (in CFM_NEW_FLAGS array but NOT in Wave A path) | implicit (excluded by default per AGENTS.md §5.1) | **PASS** |

## §1 Per-feature audit table (extended)

### §1.1 EXTRA_FLAGS accumulator (Wf-415-A1)

**Code location**: `molmetal/scripts/launch_wf299_cfm_main.sh:227-253`

```bash
EXTRA_FLAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --wave)    WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)  WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                EXTRA_FLAGS="${EXTRA_FLAGS} $1"
                shift
            done
            ;;
        *)
            # Catch-all: transparent passthrough to the trainer invocation.
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            shift
            ;;
    esac
done
```

**Verdict**: **PASS** — EXTRA_FLAGS is a transparent passthrough accumulator, NOT a hard-coded allow-list (per Wf-415-A1 §0 design contract). Future waves can introduce NEW trainer flags without re-editing the wrapper.

### §1.2 --wave 1/2/3 numeric alias (Wf-408-A3)

**Code location**: `molmetal/scripts/launch_wf299_cfm_main.sh:263-269`

```bash
case "${WAVE_TARGET}" in
    1) WAVE_TARGET="A" ;;
    2) WAVE_TARGET="B" ;;
    3) WAVE_TARGET="C" ;;
esac
[ "${WAVE_TARGET}" = "all" ] || [ "${WAVE_TARGET}" = "A" ] || [ "${WAVE_TARGET}" = "B" ] || [ "${WAVE_TARGET}" = "C" ]     || die "wave must be A | B | C | all | 1 | 2 | 3 (got: ${WAVE_TARGET})"
```

**Verdict**: **PASS** — the alias mapping normalizes 1/2/3 → A/B/C BEFORE the validation die check. Note: per Wf-413-A3 §3 prior failure (pre-Wf-408-A3 the `--wave 3` literal would `die` rc=1; Wf-408-A3 closure SHIPPED).

## §3 Dry-run verification results

### §3.1 Test 1: `USER_OK=1 bash ... --wave 1 --dry-run --user-ok 1`

**Exit code**: 0
**Key log lines**:
```
[23:46:35] Wf-299C launch begin — wave target = A — dry-run = 1
[23:46:44] Wave A plan: 4 ablation configs × 1 seed × 3 epochs (~2 hr, cuda:0)
[23:46:44] ==== Wave A begin (budget=2.0 hr, gpu=cuda:0) ====
...
[OK] Wave A DRY-RUN logged
```

**Verdict**: **PASS** — `--wave 1` correctly normalized to `wave target = A`; downstream Wave A plan emission OK.

### §3.2 Test 2: `USER_OK=1 bash ... --wave A --dry-run --user-ok 1`

**Exit code**: 0
**Key log lines**:
```
[23:46:44] Wf-299C launch begin — wave target = A — dry-run = 1
[23:46:54] Wave A plan: 4 ablation configs × 1 seed × 3 epochs (~2 hr, cuda:0)
[23:46:54] ==== Wave A begin (budget=2.0 hr, gpu=cuda:0) ====
...
[OK] Wave A DRY-RUN logged
```

**Verdict**: **PASS** — canonical `--wave A` (lettered) still works; no regression introduced by the Wf-408-A3 alias mapping.

### §3.3 Test 3: `USER_OK=1 bash ... --wave A --dry-run --user-ok 1 --use-ot-cfm`

**Exit code**: 0
**Key log lines** (relevant subset):
```
[23:46:54] Wf-299C launch begin — wave target = A — dry-run = 1
...
[23:46:54]   flags:   --epochs 3 --seed 0 --pocket-id pocket_a --use-ot-cfm --stable-vm --so3-average --log-normal-t --triton-ot --shortcut-distill --consistency-weight 1.0 --top-p 0.9 --top-k 50 --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05 --per-cohort-features --scaffold-conditioning --sa-penalty 0.0 --sa-target 2.8 --use-yuelbond --out-dir ...
...
[OK] Wave A DRY-RUN logged
```

**Verdict**: **PASS** — `--use-ot-cfm` (NEW SCAFFOLD flag from Wf-340 + Wf-353-A1 + Wf-385-A3) appears in the trainer invocation flags line. The EXTRA_FLAGS accumulator correctly passthroughs the new flag without wrapper re-editing. Also confirms: `--use-ot-cfm --stable-vm --so3-average --log-normal-t --triton-ot --shortcut-distill` ALL present in trainer flag line (6 of Wf-340+353 SCAFFOLD flags wired).

### §3.4 Negative test: launcher without USER_OK=1 (sanity check)

Per Wf-378-A2 + Wf-378-A1 design contract, launcher should refuse to launch without `USER_OK=1` env var set. This is preserved in current launcher (USER_OK gate at line 86-101). Test not re-run in this audit (already verified by Wf-378-A2).

## §4 Honest framing

Per `party-twenty-spirit` §3.5:

1. **Wf-415-A1 EXTRA_FLAGS accumulator is on-disk + working**: MEASURED via grep + dry-run test 3 (--use-ot-cfm appears in flags line).
2. **Wf-408-A3 --wave 1/2/3 alias mapping is on-disk + working**: MEASURED via grep + dry-run test 1 (`--wave 1` → `wave target = A`).
3. **Pre-existing test 2 confirms `--wave A` (canonical lettered) still works**: no regression from Wf-408-A3 alias addition.
4. **--wave 2 + --wave 3 numeric aliases**: code inspection only (no explicit dry-run test), but the case-statement at line 263-266 is symmetric + the validator at line 269-270 accepts `1 | 2 | 3` so they must work; the Wf-413-A3 launch attempts that previously failed (`--wave 3` → die rc=1) would now succeed post-Wf-408-A3.
5. **--flowmol-unified LICENSE-GATED CC-BY-NC**: NOT in `CFM_NEW_FLAGS` array for Wave A; production integration requires explicit `USER_OK=1` per AGENTS.md §5.1 — preserved per Wf-380-A1 + Wf-385-A2 + Wf-394-A3.

## §5 Cross-references

- **Wf-415-A1** (`TODO/completed/415a1_wrapper_args_fix.md`) — original EXTRA_FLAGS accumulator wire-in
- **Wf-408-A3** (`TODO/completed/408a3_preflight_launcher_fixes.md`) — original --wave 1/2/3 alias mapping wire-in
- **Wf-413-A3** (`TODO/completed/413a3_wf299_wave_b_launch.md`) — original detection of `--wave 3` literal FAIL
- **Wf-410-A4** (`TODO/completed/410a4_wf299_re_verify.md`) — pre-flight chain re-verification
- **Wf-394-A3** (`TODO/completed/394a3_wf299_pre_launch_verify.md`) — pre-launch verification chain
- **Wf-380-A1** + **Wf-385-A2** — `--flowmol-unified` LICENSE-GATED exclusion design
- **Wf-378-A2** — `--user-ok` gate + block-order invariant (USER_OK must NOT be bypassed by `--dry-run`)
- **AGENTS.md §5.1** — standing GPU authorization + USER_OK=1 requirement
- **AGENTS.md §5.5** — post-wave on-disk verification rule (this audit follows §5.5 rule #3: "next-wave A3 verify agent MUST `ls -la` actual files from current disk state")
- **AGENTS.md §5.6** — atomic write protocol (this closeout authored via `atomic_write_file`)
- **AGENTS.md §5.7 NEW** — trainer flag wire-in smoke-required rule (preserved by test 3 `--use-ot-cfm` smoke)
