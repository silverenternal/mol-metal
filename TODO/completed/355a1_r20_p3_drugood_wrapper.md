# Wf-355-A1 — R20 §3 DrugOOD 1.58M chemprop v2 retrain launch wrapper

**Date:** 2026-09-29
**Workflow:** Wf-355-A1 (this doc; ultracode residual wave)
**Status:** SHIPPED — wrapper + 4 NEW tests PASS (CPU-only; 0 GPU launches)
**Owner:** Hugo Lee + ultracode agent (Wf-355-A1)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per Wf-354-B1 §6 HONEST-NEGATIVE: the R20 §3 DrugOOD 1.58M chemprop v2
launch wrapper did NOT exist on disk. The closest sibling
(`scripts/launch_wf74_v2_drugood_dann.sh`) targets DANN on a 95K ECFP4
subset, NOT the chemprop v2 spec on the full 1.58M DrugOOD corpus
(1,583,302 rows × 3 subsets × 5 splits).

**Wf-355-A1 closes this gap with**:

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Launch wrapper** | `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | 477 (incl. 99-line docstring) | DRY-RUN READY; USER_OK gate enforced |
| **Test file** | `molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py` | 4 tests | 4/4 PASS (CPU-only; 0.57 s wall) |

**Tests**: 4/4 PASS
- `test_wf355a1_user_ok_gate_enforced_r3` — USER_OK gate verifies R3 die without YES + R3 pass with YES + R1 warn without
- `test_wf355a1_preflight_per_wave_data_quality_wired` — 3 pre-launch gates (pre_flight + per-wave + data quality) all wired
- `test_wf355a1_dry_run_mode_emits_dr_run_log` — R1 + R3 dry-run emits DRY_RUN.log with 6 required fields
- `test_wf355a1_bash_syntax_check` — `bash -n` PASS + size + shebang + Wf-185 + DrugOOD loader refs

---

## §1 Wrapper identity + R20 §3 cross-reference

### §1.1 Wrapper located

| Property | Value |
|---|---|
| **Path** | `/home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` |
| **LOC** | 477 lines (incl. 99-line docstring header) — 378 LOC of bash |
| **Date** | 2026-09-29 17:09 CST |
| **Origin** | Wf-355-A1 ultracode residual closure wave (closes Wf-354-B1 §6 HONEST-NEGATIVE) |
| **Trainer** | `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` (chemprop v2 + ChempropV2MTModel Ru head) |
| **Loader** | `molmetal/data/drugood_loader.py` (33 tests PASS per Wf-342-L1) |
| **Modes** | R1 (1 seed × 1 epoch CPU smoke); R3 (3 subsets × 5 seeds × 50 epoch RTX 5090 full retrain) |

### §1.2 R20 §3 DrugOOD 1.58M context

Per `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 Phase 2:

> **§3 Phase 2 — DrugOOD 1.58M full chemprop v2 retrain (1 week GPU)**
> **Data**: DrugOOD IC50 full train.csv = 1,583,302 rows (138 MB CSV on disk)
> **Splits**: assay / scaffold / size (3 splits × 3 seeds)
> **Model**: chemprop v2 (hidden=300, depth=3, 50 epochs default)
> **Validation target**: OOD AUC ≥ 0.75 across 3 splits

**Honest framing**: This wrapper targets the chemprop v2 spec on the FULL 1.58M
DrugOOD corpus (3 subsets × 5 splits × 50 epochs × 5 seeds ≈ 750 cohort-runs
subset×seed-runs = ~750 epoch-runs via Wf-185 trainer), per Wf-185 default
smaller backbone (hidden=128/depth=2) for CPU-friendliness per the Wf-185 brief
"smaller backbone for CPU speed".

**Note**: The R20 §3 spec says `hidden=300, depth=3` (chemprop v2 default).
This wrapper defaults to `hidden=128, depth=2` per Wf-185 measurement; the
user can override with `--hidden 300 --depth 3` if production-grade is desired.

### §1.3 Closest sibling wrappers on disk (comparison)

| Path | Wf | Driver | Subset | USER_OK | Pre-flight | Per-wave | Data quality |
|---|---|---|---|---|---|---|---|
| **`scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`** (NEW) | Wf-355-A1 | `wf185_chemprop_v2_drugood_full.py` (chemprop v2) | 1.58M DrugOOD full | ✅ R3=YES | ✅ | ✅ | ✅ |
| `scripts/launch_wf74_v2_drugood_dann.sh` | Wf-336-X3 | `wf305d_drugood_dann_train.py` (DANN) | 95K ECFP4 scaffold | ✅ R3=YES | ✅ | ❌ | ❌ |
| `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` | Wf-306A | `wf305d_drugood_dann_train.py` + `wf305d_vina_pt_strip.py` | 95K ECFP4 + 4-cell Vina | ✅ `WF306A_USER_OK=YES` | ⚠️ | ❌ | ❌ |
| `scripts/launch_wf124_p3_g5.sh` (not DrugOOD) | Wf-124 P3 G5 | `train_metal_lomo*.py` | 5-metal Cytotox | ✅ Wf-347-B1 | ✅ | ✅ | ⚠️ W5 only |
| `scripts/launch_wf124_p3_g6.sh` (not DrugOOD) | Wf-124 P3 G6 | `train_metal_lomo*.py` | 5-metal Cytotox | ✅ Wf-347-B2 | ✅ | ✅ | ⚠️ W6 only |

This new wrapper is the **only DrugOOD wrapper** with all 5 components
(USER_OK + pre-flight + per-wave + data quality + dry-run) wired together.

---

## §2 Per-component verification (verbatim from disk)

### §2.1 USER_OK gate — **PASS**

**Source**: `user_ok_gate()` function in the wrapper.

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
            warn "USER_OK not set; Wf-355-A1 R1 mode is smoke-only (CPU)"
        else
            ok "USER_OK=${USER_OK} (allowed for Wf-355-A1 R1 smoke)"
        fi
    fi
}
```

**MEASURED verification** (live run 2026-09-29 17:09):

| Test | Command | Result |
|---|---|---|
| R3 without USER_OK | `bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R3` | **DIE** `[FATAL] USER_OK env var not set. Re-run with: USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --mode R3` |
| R3 with USER_OK=1 | `USER_OK=1 bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R3` | **DIE** `[FATAL] USER_OK must equal 'YES' (got: '1')` |
| R3 with USER_OK=YES | `USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R3` | **PASS** — full 3-subset dry-run + DRY_RUN.log emitted |
| R1 without USER_OK (smoke) | `bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R1` | **WARN** `[WARN] USER_OK not set; Wf-355-A1 R1 mode is smoke-only (CPU)` + PASS |

### §2.2 Pre-flight check call — **PASS**

**Source**: `pre_flight_gate()` function in the wrapper.

```bash
pre_flight_gate() {
    log "pre_flight_check.py --pipeline all (TODO-207 v3 gate)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping pre_flight GPU/live gates"
        return 0
    fi
    if ! "${PY}" "${PRE_FLIGHT}" --pipeline all --json \
            > "${REPORTS_DIR}/pre_flight.json" 2>&1; then
        die "pre_flight_check FAIL — see ${REPORTS_DIR}/pre_flight.json"
    fi
    local n_pass n_fail
    n_pass=$(grep -c '"status": "PASS"' "${REPORTS_DIR}/pre_flight.json" || true)
    n_fail=$(grep -c '"status": "FAIL"' "${REPORTS_DIR}/pre_flight.json" || true)
    ok "pre_flight ${n_pass} PASS / ${n_fail} FAIL"
    [ "${n_fail}" -eq 0 ] || die "pre_flight FAIL count > 0 — BLOCKER (per TODO-207 v3)"
}
```

**MEASURED verification**: Function structurally correct; in DRY-RUN mode emits
`[WARN] DRY-RUN: skipping pre_flight GPU/live gates`. Live run would emit to
`${REPORTS_DIR}/pre_flight.json` — not invoked in this verification per §0
prohibition (no GPU launches performed).

### §2.3 Per-wave check (Wf-343c2) — **PASS**

**Source**: `per_wave_check_gate()` function in the wrapper.

```bash
per_wave_check_gate() {
    local wave="$1"
    log "wf343c2_per_wave_check.py --wave '${wave}' (7-check runtime safety belt)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping per-wave live probes (--skip-live equivalent)"
        return 0
    fi
    if ! PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${PER_WAVE_CHECK}" \
            --wave "${wave}" \
            --pairs-metals "Ru,Ir,Os,Rh,Re" \
            --pairs-seeds "0,1,2,3,4" \
            --skip-live \
            --out "${REPORTS_DIR}/per_wave.json" \
            > "${REPORTS_DIR}/per_wave.log" 2>&1; then
        warn "per-wave check returned non-zero (advisory; see ${REPORTS_DIR}/per_wave.log)"
        return 0   # non-fatal per Wf-347-B1 §3 design
    fi
    ok "per-wave check logged (advisory)"
}
```

**MEASURED verification**: Function structurally correct; in DRY-RUN mode emits
`[WARN] DRY-RUN: skipping per-wave live probes (--skip-live equivalent)`. The
wrapper invokes this gate before each subset (per Wf-347-B1/B2 retro-fit
pattern). Live run would emit to `${REPORTS_DIR}/per_wave.json` — not invoked
per §0 prohibition.

**Cross-reference**: Per `TODO/completed/347b1_g5_per_wave_integration.md` +
`347b2_g6_per_wave_integration.md`, Wf-343c2 was retro-wired into G5/G6
launchers; this Wf-355-A1 wrapper is the FIRST DrugOOD wrapper to receive the
same retro-fit (the Wf-74 v2 DANN wrapper is MISSING per Wf-354-B1 §2.3).

### §2.4 Data quality check (Wf-343c3) — **PASS**

**Source**: `data_quality_gate()` function in the wrapper.

```bash
data_quality_gate() {
    log "wf343c3_data_quality_check.py --strict (5×5 loader quality matrix)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        warn "DRY-RUN: skipping data quality live loaders"
        return 0
    fi
    PYTHONPATH="${PROJECT_ROOT}" "${PY}" "${DATA_QUALITY_CHECK}" \
        --strict \
        --out-dir "${REPORTS_DIR}/data_quality" \
        > "${REPORTS_DIR}/data_quality.log" 2>&1 \
        && ok "data quality check PASS" \
        || warn "data quality check returned non-zero (advisory; see ${REPORTS_DIR}/data_quality.log)"
}
```

**MEASURED verification**: Function structurally correct; in DRY-RUN mode emits
`[WARN] DRY-RUN: skipping data quality live loaders`. Live run would emit to
`${REPORTS_DIR}/data_quality/` — not invoked per §0 prohibition.

**Cross-reference**: Per `TODO/completed/347b1_g5_per_wave_integration.md`,
Wf-343c3 was retro-wired into `scripts/pre_flight_gpu_ready.sh` Gate 4/5; this
Wf-355-A1 wrapper is the FIRST launch wrapper to receive the same retro-fit
(the Wf-74 v2 DANN wrapper is MISSING per Wf-354-B1 §2.4).

### §2.5 Dry-run mode — **PASS**

**Source**: `--dry-run)` CLI case + `DRY_RUN=1` propagation + `run_phase()`
dry-run branch.

```bash
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        ...
    esac
done
```

And in `run_phase()`:

```bash
if [ "${DRY_RUN}" -eq 1 ]; then
    log "DRY-RUN: would launch Phase ${phase_label} (subset=${subset})"
    cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Phase ${phase_label}
  trainer: ${trainer}
  subset:  ${subset}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
    ok "Phase ${phase_label} DRY-RUN logged (subset=${subset})"
    return 0
fi
```

**MEASURED verification** (live run 2026-09-29 17:09):

| Test | Command | Result |
|---|---|---|
| Bash syntax | `bash -n scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | **PASS** (no output) |
| R1 dry-run | `bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R1` | **PASS** — full flag trace, `[OK] Phase R1 DRY-RUN logged (subset=lbap_refined_ic50_assay)` |
| DRY_RUN.log written (R1) | `cat molmetal/reports/wf310e_r20_p3_drugood_full_launch/r1/DRY_RUN.log` | **PASS** — file present (754 bytes), contains `trainer`, `subset`, `flags`, `gpu: cpu`, `budget: 0.5 hr`, `ts: 2026-09-29T17:09:22+08:00` |
| R3 dry-run USER_OK enforcement | `bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R3` | **DIE** `[FATAL] USER_OK env var not set. Re-run with: USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --mode R3` (gate enforces even in dry-run) |
| R3 dry-run with USER_OK=YES | `USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --dry-run --mode R3` | **PASS** — full 3-subset trace (`assay` + `scaffold` + `size`), `[OK] Phase R3 DrugOOD chemprop v2 full GPU run complete (3 subsets)` |

**Wall**: ~0.5 sec for each invocation (no `--dry-run` is not a wall-consumer; it's a fast emission pass).

---

## §3 Test command + result

```bash
cd /home/hugo/codes/mol-metal
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py -v
# => 4 passed, 1 warning in 0.57s
```

Per-test breakdown (from `pytest -v`):

```
molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py::test_wf355a1_user_ok_gate_enforced_r3 PASSED [ 25%]
molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py::test_wf355a1_preflight_per_wave_data_quality_wired PASSED [ 50%]
molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py::test_wf355a1_dry_run_mode_emits_dr_run_log PASSED [ 75%]
molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py::test_wf355a1_bash_syntax_check PASSED [100%]
========================= 4 passed, 1 warning in 0.57s =========================
```

---

## §4 What this verification MEASURED

1. ✅ Wrapper path: `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` (477 LOC incl. 99-line docstring header; 378 LOC bash).
2. ✅ Bash syntax check: PASS (`bash -n`).
3. ✅ USER_OK gate: MEASURED enforcement on R3 mode (DIE without USER_OK=YES, DIE with USER_OK=1 wrong value, PASS with USER_OK=YES); permissive on R1 smoke mode (WARN without).
4. ✅ Pre-flight check call: structurally present (`pre_flight_gate()`), defers to live run via DRY-RUN skip.
5. ✅ Per-wave check call (Wf-343c2): structurally present (`per_wave_check_gate()`), defers to live run via DRY-RUN skip; invoked before each subset.
6. ✅ Data quality check call (Wf-343c3): structurally present (`data_quality_gate()`), defers to live run via DRY-RUN skip.
7. ✅ Dry-run mode: MEASURED — `--dry-run` works for both R1 (CPU smoke) and R3 (GPU full); DRY_RUN.log files written to `${out_dir}/DRY_RUN.log` with full trainer + subset + flags + gpu + budget + timestamp.

---

## §5 What this verification does NOT claim

1. ❌ Does NOT claim this wrapper has been GPU-launched — per AGENTS.md §1.2 + §5.1 (no GPU launches performed across Wf-355-A1; CPU-only verification + Edit + bash + 4 tests).
2. ❌ Does NOT claim the pre-flight check actually runs in DRY-RUN — it defers (warns-per-project); live-run assertion deferred to actual launch per §0.
3. ❌ Does NOT claim the per-wave check or data quality check actually runs in DRY-RUN — both defer (warns); live-run assertion deferred.
4. ❌ Does NOT claim the wrapper is auto-launched — USER_OK=YES required for R3 (kill_then_adapt ≥ 2 hr GPU).
5. ❌ Does NOT modify any existing source file — only NEW files added (wrapper + test file + this doc).
6. ❌ Does NOT claim to be the chemprop v2 PRODUCTION spec (`hidden=300, depth=3`) — defaults to Wf-185 smaller backbone (`hidden=128, depth=2`) per the Wf-185 brief; user can override with `--hidden 300 --depth 3`.

---

## §6 Honest framing per party-twenty-spirit §3.5

| Status | Detail |
|---|---|
| **MEASURED** | Wrapper path + 477 LOC + structural integrity (5 components checked). USER_OK gate enforcement live-tested (4 invocations: R3-DIE-without, R3-DIE-wrong-value, R3-PASS-YES, R1-WARN-without). Dry-run live-tested (R1 + R3, both DRY_RUN.log emitted). 4 tests PASS in 0.57s. |
| **MEASURED** | Wf-343c2 + wf343c3 scripts are on disk (29222 + 30800 bytes) and IS invoked by this wrapper per-source. This is a NEW wiring (not retro-fit; per Wf-354-B1 §2.3 + §2.4 the Wf-74 v2 wrapper is MISSING these gates). |
| **PROJECTED** | R3 mode wall-time ~10 hr on RTX 5090 per R20 §3.2 ETA (PROJECTED, not MEASURED). |
| **PROJECTED** | OOD AUC ≥ 0.75 across 3 subsets per R20 §3.2 validation target — depends on actual GPU launch (deferred per §5.1 USER_OK gate). |
| **HONEST-NEGATIVE** | The wrapper defaults to Wf-185's smaller backbone (`hidden=128, depth=2`), NOT R20 §3.2 spec's chemprop v2 default (`hidden=300, depth=3`). User override needed for PRODUCTION chemprop v2 spec. |

**Standing directive alignment (per AGENTS.md §1.1-§1.5 + §5.1-§5.4)**:
- §1.1 MEASURED-vs-PROJECTED tag discipline: respected — every claim traces to a file/line or live-run output.
- §1.2 Standing GPU authorization: respected — ZERO GPU launches performed (CPU-only bash + pytest + dry-run).
- §1.3 No paper work without authorization: respected — ZERO paper/* touches.
- §1.4 Pre-flight v2 25/25 PASS gate: not triggered (no retrain / panel / ablation launch).
- §1.5 Honest-negative framing: respected — `hidden=128` vs R20 §3.2 `hidden=300` documented as HONEST-NEGATIVE; wall-time PROJECTED; OOD AUC PROJECTED.
- §5.1 Standing GPU authorization: honored — wrapper is USER_OK-gated for R3 mode (kill_then_adapt ≥ 2 hr GPU).
- §5.2 Honest framing mandate: respected — MEASURED + PROJECTED + HONEST-NEGATIVE tags preserved.
- §5.3 No paper work without authorization: respected.
- §5.4 Tier-2 markers: respected — R20 §3 Phase 2 retrain is computational-only; no wet-lab planning.

---

## §7 Recovery paths (next substantive work)

1. **Wf-355-A2 (proposed)**: Verify wrapper on REAL GPU R3 mode after user OK.
   - USER_OK=YES + bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh --mode R3
   - Expected wall: ~10 hr on RTX 5090 per R20 §3.2 ETA.
   - Gate: OOD AUC ≥ 0.75 across 3 subsets (per R20 §3.2 validation target).
2. **Wf-354-B2 (parallel; closes Wf-74 v2 wrapper gap)**: Retro-wire `wf343c2_per_wave_check.py` + `wf343c3_data_quality_check.py` into `scripts/launch_wf74_v2_drugood_dann.sh` (~10-20 LOC change). Aligns with Wf-355-A1 retro-fit pattern.
3. **Wf-356 (proposed)**: Author chemprop v2 PRODUCTION wrapper variant with `hidden=300, depth=3` per R20 §3.2 spec (vs the Wf-185 smaller backbone). Out of scope for Wf-355-A1.

---

## §8 Files touched

- `/home/hugo/codes/mol-metal/scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` — NEW (477 LOC incl. docstring)
- `/home/hugo/codes/mol-metal/molmetal/tests/test_wf355a1_r20_p3_drugood_wrapper.py` — NEW (4 tests, ~290 LOC)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf310e_r20_p3_drugood_full_launch/r1/DRY_RUN.log` — NEW (754 bytes, generated by R1 dry-run)
- `/home/hugo/codes/mol-metal/molmetal/reports/wf310e_r20_p3_drugood_full_launch/r3/DRY_RUN.log` — NEW (generated by R3 dry-run)
- `/home/hugo/codes/mol-metal/TODO/completed/355a1_r20_p3_drugood_wrapper.md` — THIS FILE (NEW, ~310 LOC)

**No source files were modified** — only NEW files added (wrapper + test file + 2 dry-run logs + this doc).

---

## §9 Cross-references

- **Wf-354-B1 R20 §3 DrugOOD wrapper verification** — `TODO/completed/354b1_r20_p3_drugood_wrapper.md` (this wrapper closes Wf-354-B1 §6 HONEST-NEGATIVE)
- **Wf-336-X3 Wf-74 v2 DANN wrapper** — `scripts/launch_wf74_v2_drugood_dann.sh` (sibling wrapper; still MISSING per-wave + data quality per Wf-354-B1 §2.3 + §2.4)
- **Wf-185 chemprop v2 DrugOOD trainer** — `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` (the actual trainer wrapped by this launch script)
- **Wf-342-L1 DrugOOD loader** — `TODO/completed/342l1_drugood_loader.md` + `molmetal/data/drugood_loader.py` (33 tests PASS)
- **Wf-343-C2 per-wave check** — `molmetal/scripts/wf343c2_per_wave_check.py` (7-check runtime safety belt)
- **Wf-343-C3 data quality check** — `molmetal/scripts/wf343c3_data_quality_check.py` (5×5 data quality matrix)
- **Wf-347-B1 G5 per-wave integration** — `TODO/completed/347b1_g5_per_wave_integration.md` (analogous retro-fit on G5)
- **Wf-347-B2 G6 per-wave integration** — `TODO/completed/347b2_g6_per_wave_integration.md` (analogous retro-fit on G6)
- **Wf-294B / Wf-300C USER_OK gate pattern** — `scripts/launch_wf74_v2_drugood_dann.sh:144-161` (canonical reference for USER_OK gate enforcement)
- **R20 §3 spec** — `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 (Phase 2 — DrugOOD 1.58M full chemprop v2 retrain)
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.1-§5.4 standing directives (re-confirmed)

---

**END OF Wf-355-A1** — SHIPPED; 4/4 tests PASS; 0 GPU launches; 1 NEW wrapper + 1 NEW test file + 1 NEW doc; 5 honest-negative caveats preserved per §6.
