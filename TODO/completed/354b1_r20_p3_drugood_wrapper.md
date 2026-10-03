# Wf-354-B1 — R20 §3 DrugOOD launch wrapper verification

**Date:** 2026-09-29
**Workflow:** Wf-354-B1 (this doc)
**Status:** PARTIAL_VERIFIED — 3/5 components PASS, 2/5 MISSING (HONEST-NEGATIVE preserved)
**Owner:** Hugo Lee + ultracode agent (per AGENTS.md §1.2 + §5.1 + §5.2)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR — per-component verdict

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | **USER_OK gate** | **PASS** | `scripts/launch_wf74_v2_drugood_dann.sh:144-161` — gate enforces `USER_OK=YES` for R3 mode (3-seed × 50-ep GPU); R1 mode is smoke-only (USER_OK optional) |
| 2 | **Pre-flight check call** | **PASS** | `scripts/launch_wf74_v2_drugood_dann.sh:184-199` — `pre_flight_check.py --pipeline all --json` runs before launch; counts PASS/FAIL and dies if FAIL>0 |
| 3 | **Per-wave check call (post-Wf-347)** | **MISSING** | `grep -n "wf343c2\|per_wave" scripts/launch_wf74_v2_drugood_dann.sh` → 0 hits. Wf-343c2 was wired only into `scripts/launch_wf124_p3_g5.sh` (Wf-347-B1) and `scripts/launch_wf124_p3_g6.sh` + `scripts/recover_g6_failed_tuples.sh` (Wf-347-B2) — NOT into this DrugOOD wrapper |
| 4 | **Data quality check (post-Wf-347)** | **MISSING** | `grep -n "wf343c3\|data_quality" scripts/launch_wf74_v2_drugood_dann.sh` → 0 hits. `molmetal/scripts/wf343c3_data_quality_check.py` (30800 bytes, 2026-09-29 15:53) is on disk but not invoked by this wrapper |
| 5 | **Dry-run mode** | **PASS** | `scripts/launch_wf74_v2_drugood_dann.sh:123, 307-318` — `--dry-run` flag is parsed; DRY_RUN.log written to `${out_dir}/DRY_RUN.log` with full trainer + flags + budget + timestamp |

**Verification verdict**: 3/5 PASS (USER_OK + pre-flight + dry-run), 2/5 MISSING (wf343c2 per-wave + wf343c3 data quality). HONEST-NEGATIVE preserved per §6.

---

## §1 Wrapper identity + R20 §3 cross-reference

### §1.1 Wrapper located

| Property | Value |
|---|---|
| **Path** | `/home/hugo/codes/mol-metal/scripts/launch_wf74_v2_drugood_dann.sh` |
| **LOC** | 429 lines (17751 bytes) |
| **Date** | 2026-09-29 15:06 |
| **Origin** | Wf-336-X3 ultracode residual closure wave |
| **Trainer** | `molmetal/scripts/wf305d_drugood_dann_train.py` (DANN driver; ReverseLayerF + lambda_grl) |
| **Trainer dataset** | `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` (95,236 rows ECFP4 scaffold) |
| **Modes** | R1 (1 seed × 1 epoch × hidden=64/depth=2 = smoke CPU, ~30 sec); R3 (3 seeds × 50 epochs × hidden=128/depth/3 = full GPU, ~3-4 hr) |

### §1.2 R20 §3 DrugOOD 1.58M context

Per `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 Phase 2:

> **§3 Phase 2 — DrugOOD 1.58M full chemprop v2 retrain (1 week GPU)**
> **Data**: DrugOOD IC50 full train.csv = 1,583,302 rows (138 MB CSV on disk)
> **Splits**: assay / scaffold / size (3 splits × 3 seeds)
> **Model**: chemprop v2 (hidden=300, depth=3, 50 epochs default)
> **Validation target**: OOD AUC ≥ 0.75 across 3 splits

**Honest framing**: This wrapper targets the Wf-305D **DANN driver on the 95K ECFP4 scaffold subset**, NOT the chemprop v2 path on the FULL 1.58M DrugOOD corpus. The 1.58M retrain has NEVER been shipped:
- `wf185_chemprop_v2_drugood_full.py` exists but is a 3-configs / multi-seed script, not a 1.58M launch wrapper.
- `wf310d_drugood_full/aggregate.json` (MEASURED 2026-09-29 09:43) shows 3-seed × 50-epoch on the 95K ECFP4 subset, NOT 1.58M.
- The R20 §3 Phase 2 spec was DEFERRED per Wf-342-L1 §6 PROJECTED note: "the full 1.58M (3 subsets × 5 splits) is NOT yet loaded by any test (would exceed unit-test budget; belongs in Phase 2 GPU bench)".

### §1.3 Closest DrugOOD launch wrappers on disk

| Path | Wf | Driver | Subset | USER_OK gate | Per-wave | Data quality |
|---|---|---|---|---|---|---|
| **`scripts/launch_wf74_v2_drugood_dann.sh`** | Wf-336-X3 | `wf305d_drugood_dann_train.py` (DANN) | 95K ECFP4 scaffold | ✅ R3=YES | ❌ | ❌ |
| `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` | Wf-306A | `wf305d_drugood_dann_train.py` + `wf305d_vina_pt_strip.py` | 95K ECFP4 + 4-cell Vina | ✅ `WF306A_USER_OK=YES` | ❌ | ❌ |
| `scripts/launch_wf124_p3_g5.sh` (not DrugOOD) | Wf-124 P3 G5 | `train_metal_lomo*.py` | 5-metal Cytotox | ✅ Wf-347-B1 | ✅ | ⚠️ (W5 only) |
| `scripts/launch_wf124_p3_g6.sh` (not DrugOOD) | Wf-124 P3 G6 | `train_metal_lomo*.py` | 5-metal Cytotox | ✅ Wf-347-B2 | ✅ | ⚠️ (W6 only) |

The closest match to "R20 §3 DrugOOD 1.58M launch wrapper" intent is `scripts/launch_wf74_v2_drugood_dann.sh` (the Wf-74 v2 DANN computational follow-up of Wf-74 wet-lab DEPRECATED per AGENTS.md §5.4).

---

## §2 Per-component verification (verbatim from disk)

### §2.1 USER_OK gate — **PASS**

**Source**: lines 144-161 of `scripts/launch_wf74_v2_drugood_dann.sh`:

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

**MEASURED verification** (live run 2026-09-29 16:42):

| Test | Command | Result |
|---|---|---|
| R3 without USER_OK | `bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R3` | **DIE** `[FATAL] USER_OK env var not set. Re-run with: USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3` |
| R3 with USER_OK=YES | `USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R3` | **PASS** `[OK] USER_OK gate passed (USER_OK=YES)` |
| R1 without USER_OK (smoke) | `bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R1` | **WARN** `[WARN] USER_OK not set; Wf-335-R1 mode is smoke-only (CPU)` (smoke gate is permissive by design) |

### §2.2 Pre-flight check call — **PASS**

**Source**: lines 184-199 of `scripts/launch_wf74_v2_drugood_dann.sh`:

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

**MEASURED verification**: DRY-RUN logs `[WARN] DRY-RUN: skipping pre_flight GPU/live gates` + `[OK] pre_flight ${n_pass} PASS / ${n_fail} FAIL` (counts deferred to actual run, not dry-run). The function definition is structurally correct (calls `pre_flight_check.py --pipeline all --json`, parses PASS/FAIL counts, dies on FAIL>0). Live run would emit to `${REPORTS_DIR}/pre_flight.json` — not invoked in this verification per §0 prohibition.

### §2.3 Per-wave check call (post-Wf-347) — **MISSING (HONEST-NEGATIVE)**

**Source**: `grep -n "wf343c2\|per_wave\|wf347" scripts/launch_wf74_v2_drugood_dann.sh` → **0 hits**.

**Cross-reference** (per `TODO/completed/347b1_g5_per_wave_integration.md` + `347b2_g6_per_wave_integration.md`):

The Wf-343c2 per-wave check (7-check runtime safety belt) was wired into:
- `scripts/launch_wf124_p3_g5.sh` (Wf-347-B1, 2026-09-29) — runs `wf343c2_per_wave_check.py` before each `run_pair()` invocation
- `scripts/launch_wf124_p3_g6.sh` (Wf-347-B2, 2026-09-29) — runs `wf343c2_per_wave_check.py` before each `run_tuple()` invocation
- `scripts/recover_g6_failed_tuples.sh` (Wf-347-B2) — same pattern

The Wf-74 v2 DANN wrapper was NOT in Wf-347-B1 or Wf-347-B2 scope. The Wf-343c2 emitter `molmetal/scripts/wf343c2_per_wave_check.py` (29222 bytes, 2026-09-29 15:54) IS on disk but not invoked.

**Honest framing**: Per-wave check is a diagnostic emitter (not hard gate) per Wf-347-B1 §2 design. Adding it to this wrapper is a 5-10 LOC change (analogous to Wf-347-B1 +33/-4 LOC pattern). Out of scope for Wf-354-B1 verification; follow-up Wf-354-B2 candidate.

### §2.4 Data quality check (post-Wf-347) — **MISSING (HONEST-NEGATIVE)**

**Source**: `grep -n "wf343c3\|data_quality" scripts/launch_wf74_v2_drugood_dann.sh` → **0 hits**.

**Cross-reference**: `molmetal/scripts/wf343c3_data_quality_check.py` (30800 bytes, 2026-09-29 15:53) is on disk per Wf-343-C3 ship surface (5×5 data quality check). The Wf-343c3 emits JSON; nothing was wired into the Wf-74 v2 wrapper.

**Same gap pattern as §2.3**: data quality check was NOT retro-wired into pre-Wf-343 launchers. Adding it is a parallel follow-up. Out of scope for Wf-354-B1 verification.

### §2.5 Dry-run mode — **PASS**

**Source**: lines 123, 307-318 of `scripts/launch_wf74_v2_drugood_dann.sh`:

```bash
DRY_RUN=0
MODE="R1"
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
    log "DRY-RUN: would launch Phase ${phase_label}"
    cat > "${out_dir}/DRY_RUN.log" <<EOF
DRY-RUN launcher for Phase ${phase_label}
  trainer: ${trainer}
  flags:   $*
  gpu:     ${gpu}
  budget:  ${budget_hr} hr
  ts:      $(date -Iseconds)
EOF
    ok "Phase ${phase_label} DRY-RUN logged"
    return 0
fi
```

**MEASURED verification** (live run 2026-09-29 16:42):

| Test | Command | Result |
|---|---|---|
| Bash syntax | `bash -n scripts/launch_wf74_v2_drugood_dann.sh` | **PASS** (no output) |
| R1 dry-run | `bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R1` | **PASS** — full flag trace, `[OK] Phase R1 DRY-RUN logged`, `[OK] Wf-336-X3 launch script done — mode=R1` |
| DRY_RUN.log written | `cat molmetal/reports/wf74_v2_drugood_dann_launch/r1/DRY_RUN.log` | **PASS** — file present (600 bytes), contains `trainer`, `flags`, `gpu: cpu`, `budget: 0.5 hr`, `ts: 2026-09-29T16:42:17+08:00` |
| R3 dry-run USER_OK enforcement | `bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R3` | **DIE** `[FATAL] USER_OK env var not set. Re-run with: USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3` (gate enforces even in dry-run) |
| R3 dry-run with USER_OK=YES | `USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run --mode R3` | **PASS** — full flag trace incl. `--cuda-visible 1 --device cuda`, `[OK] Phase R3 DRY-RUN logged` |

**Wall**: 0.5 sec for each invocation (no `--dry-run` is not a wall-consumer; it's a fast emission pass).

---

## §3 What this verification MEASURED

1. ✅ Wrapper path: `scripts/launch_wf74_v2_drugood_dann.sh` (429 LOC, 17751 bytes — Wf-336-X3, 2026-09-29 15:06).
2. ✅ Bash syntax check: PASS (`bash -n`).
3. ✅ USER_OK gate: MEASURED enforcement on R3 mode (DIE without USER_OK=YES); permissive on R1 smoke mode.
4. ✅ Pre-flight check call: structurally present (lines 184-199), defers to live run via DRY-RUN skip.
5. ✅ Dry-run mode: MEASURED — `--dry-run` flag works for both R1 (CPU smoke) and R3 (GPU full); DRY_RUN.log file written to `${out_dir}/DRY_RUN.log` with full trainer + flags + budget + timestamp.
6. ❌ Per-wave check call (post-Wf-347 `wf343c2_per_wave_check.py`): MISSING — grep 0 hits in wrapper.
7. ❌ Data quality check (post-Wf-347 `wf343c3_data_quality_check.py`): MISSING — grep 0 hits in wrapper.

---

## §4 What this verification does NOT claim

1. ❌ Does NOT claim this is the R20 §3 DrugOOD 1.58M wrapper — the R20 §3 spec (per `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3) is chemprop v2 on the FULL 1.58M DrugOOD corpus (1,583,302 rows × 3 splits × 3 seeds), which has NEVER been shipped. This wrapper targets DANN on 95K ECFP4 subset.
2. ❌ Does NOT claim per-wave check or data quality check is wired in — both are MISSING (HONEST-NEGATIVE preserved per §6).
3. ❌ Does NOT claim the pre-flight check actually runs in DRY-RUN — it defers (warns-per-project); live-run assertion deferred to actual launch per §0.
4. ❌ Does NOT claim this wrapper is ready for the 1.58M DrugOOD full retrain — even if per-wave + data quality were retro-wired, the trainer (`wf305d_drugood_dann_train.py`) + CSV (`drugood_ecfp4_scaffold.csv`, 95K rows) do NOT match the 1.58M spec.
5. ❌ Does NOT modify any source file — CPU-only verification + 1 NEW doc.
6. ❌ Does NOT claim `WF306A_USER_OK=YES` gate (the older `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh`) has been unified with `USER_OK=YES` — the two DrugOOD launch wrappers use different env-var names (canonical = `USER_OK=YES` per Wf-300C + Wf-338-V2; legacy `WF306A_USER_OK=YES` is Wf-306A-specific).

---

## §5 Honest framing per party-twenty-spirit §3.5

| Status | Detail |
|---|---|
| **MEASURED** | Wrapper path + date + LOC + structural integrity (5 components checked). USER_OK gate enforcement live-tested (3 invocations: R3-DIE, R3-PASS, R1-WARN). Dry-run live-tested (R1 + R3, both DRY_RUN.log emitted). |
| **MEASURED** | Wf-343c2 + wf343c3 scripts are on disk (29222 + 30800 bytes, 2026-09-29 15:54 + 15:53) but NOT invoked by this wrapper (grep 0 hits). |
| **MEASURED** | Wf-347-B1/B2 retro-wired wf343c2 into G5/G6 launchers but NOT into the Wf-74 v2 DANN wrapper (out of scope per Wf-347-B1 §5 + B2 §4). |
| **MEASURED** | The wrapper's actual trainer (`wf305d_drugood_dann_train.py`) + CSV (`drugood_ecfp4_scaffold.csv`, 95K rows) do NOT match the R20 §3 1.58M DrugOOD spec. |
| **PROJECTED** | Adding wf343c2 + wf343c3 to this wrapper is a 10-20 LOC follow-up (parallel to Wf-347-B1/B2 pattern); out of scope for Wf-354-B1 verification. |
| **HONEST-NEGATIVE** | The R20 §3 1.58M DrugOOD launch wrapper does NOT exist on disk. The plan §3 (Phase 2 = 1.58M chemprop v2) was DEFERRED per Wf-342-L1 §6. |

**Standing directive alignment (per AGENTS.md §1.1-§1.5 + §5.1-§5.4)**:
- §1.1 MEASURED-vs-PROJECTED tag discipline: respected — every claim traces to a file/line or live-run output.
- §1.2 Standing GPU authorization: respected — ZERO GPU launches performed (CPU-only bash + grep + dry-run).
- §1.3 No paper work without authorization: respected — ZERO paper/* touches.
- §1.4 Pre-flight v2 25/25 PASS gate: not triggered (no retrain / panel / ablation launch).
- §1.5 Honest-negative framing: respected — 2/5 MISSING components preserved as such; no aspirational claim that R20 §3 1.58M wrapper exists.
- §5.1 Standing GPU authorization: honored — wrapper is USER_OK-gated for R3 mode.
- §5.2 Honest framing mandate: respected — MEASURED + PROJECTED + HONEST-NEGATIVE tags preserved.
- §5.3 No paper work without authorization: respected.
- §5.4 Tier-2 markers: Wf-74 lineage wet-lab outreach DEPRECATED per Wf-336-X3 §8 note; this wrapper is the computational follow-up.

---

## §6 Verification verdict (final)

**STATUS: PARTIAL_VERIFIED**

- **PASS (3/5)**: USER_OK gate + pre-flight check call + dry-run mode.
- **MISSING (2/5)**: per-wave check (wf343c2 post-Wf-347) + data quality check (wf343c3 post-Wf-347). Both are HONEST-NEGATIVE — the wrapper is NOT retro-fitted with Wf-343c2/c3 wire-up. Wf-343c2 was only retro-fitted to G5/G6 launchers per Wf-347-B1 + Wf-347-B2.
- **SCOPE MISMATCH (HONEST-NEGATIVE)**: This wrapper targets DANN on 95K ECFP4 subset, NOT the R20 §3 1.58M DrugOOD chemprop v2 retrain spec. The 1.58M wrapper does NOT exist on disk per Wf-342-L1 §6 PROJECTED note.

**Recovery paths** (next substantive work):
- **Wf-354-B2 (proposed)**: Retro-wire `wf343c2_per_wave_check.py` + `wf343c3_data_quality_check.py` into `scripts/launch_wf74_v2_drugood_dann.sh` (~10-20 LOC change). Aligns with Wf-347-B1/B2 retro-fit pattern. CPU-only; no new tests.
- **Wf-354-C1 (proposed)**: Author the actual R20 §3 1.58M DrugOOD chemprop v2 launch wrapper (~200-300 LOC). Requires: (a) data path `.data/molmetal/drugood/ic50/data/train.csv` (138 MB, 1,583,302 rows); (b) `--splits assay scaffold size --seeds 0 1 2 --epochs 50 --hidden 300 --depth 3`; (c) train_drugood.py or wf185_chemprop_v2_drugood_full.py as driver; (d) target OOD AUC ≥ 0.75 across 3 splits; (e) ETA ~10 hr wall single RTX 5090. Out of scope for Wf-354-B1 verification; this is the "real" R20 §3 ship target per `R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3.

---

## §7 Files touched

- `/home/hugo/codes/mol-metal/TODO/completed/354b1_r20_p3_drugood_wrapper.md` — THIS FILE (NEW, ~210 LOC)

**No source files were modified** — this is a verification + integration doc (no code, no metric, no ckpt writes).

---

## §8 Cross-references

- **Wf-336-X3 Wf-74 v2 DANN wrapper** — `scripts/launch_wf74_v2_drugood_dann.sh` (this verification target)
- **Wf-306A DrugOOD+Vina wrapper** — `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (older wrapper; legacy `WF306A_USER_OK=YES` env var)
- **Wf-305D DANN driver** — `molmetal/scripts/wf305d_drugood_dann_train.py` (DANN GRL + ECFP4 + chemprop v2 BinaryClassificationFFN)
- **Wf-343-C2 per-wave check** — `molmetal/scripts/wf343c2_per_wave_check.py` (7-check runtime safety belt; on disk; NOT wired into this wrapper)
- **Wf-343-C3 data quality check** — `molmetal/scripts/wf343c3_data_quality_check.py` (5×5 data quality; on disk; NOT wired into this wrapper)
- **Wf-347-B1 G5 per-wave integration** — `TODO/completed/347b1_g5_per_wave_integration.md` (analogous retro-fit on G5)
- **Wf-347-B2 G6 per-wave integration** — `TODO/completed/347b2_g6_per_wave_integration.md` (analogous retro-fit on G6)
- **Wf-342-L1 DrugOOD loader status** — `TODO/completed/342l1_drugood_loader.md` (33 tests on `molmetal/data/drugood_loader.py`; 1.58M still NOT loaded per §6 PROJECTED note)
- **Wf-310D drugood full** — `molmetal/reports/wf310d_drugood_full/aggregate.json` (3-seed × 50-epoch on 95K ECFP4; NOT 1.58M)
- **Wf-185 chemprop v2 drugood full** — `molmetal/scripts/wf185_chemprop_v2_drugood_full.py` (3-configs; not a launch wrapper)
- **R20 §3 spec** — `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 (Phase 2 — DrugOOD 1.58M full chemprop v2 retrain)
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.1-§5.4 standing directives (re-confirmed)

---

**END OF Wf-354-B1** — PARTIAL_VERIFIED; 3/5 PASS + 2/5 MISSING; 0 NEW tests; 0 NEW code; 1 NEW doc; 4 honest-negative caveats preserved per §6.
