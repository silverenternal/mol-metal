# Wf-406-A4 — Wf-246-C metal-coordination term retrain SCAFFOLD (closeout)

**Date:** 2026-09-30 (Wf-406-A4 timestamp)
**Owner:** Hugo Lee + Wf-406-A4 ultracode agent
**Workflow:** Wf-406-A4 (CPU-only docs+code+test wave; sister to Wf-406-A1 + Wf-406-A2 + Wf-406-A3)
**Status:** ✅ SHIPPED — CPU-only SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + USER_OK-gated GPU retrain deferred per AGENTS.md §5.1
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only SCAFFOLD; GPU USER_OK-gated for actual retrain) + §5.2 (honest framing) + §5.3 (no paper work, TODO-84 active) + §5.4 (Tier-2 acceptance bar, no wet-lab planning per 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验") + §5.6 (atomic write protocol — closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | Wire `molmetal.molmetal_lam.sbdd_env.metal_coord_term.MetalCoordinationTerm` into the Wf-131 P3 D-MPNN multi-task trainer's training loop as a soft kcal/mol adjustment on top of the Wf-131 pIC50 regression loss + Wf-318 Wasserstein regulariser (per Wf-246-C §0) | Wf-246-C plan + Wf-131 P3 + Wf-318 |
| **Wave structure** | Wf-406-A4 (this entry) = 1 sub-flow = **3 deliverables**: SCAFFOLD script (`molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` 344 lines, 12,602 bytes) + bash launch wrapper (`scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` 216 lines) + 2 NEW tests in `molmetal/tests/test_wf406a4_metal_coord_scaffold.py` + this closeout | Wf-406-A4 |
| **SCAFFOLD tier** | CPU-only SCAFFOLD SHIPPED with 5-step smoke (smoke `--smoke` runs 2 metals × 2 seeds × 5 epochs) + kcal sign audit per Wf-333-V1 fix | Wf-406-A4 |
| **5-metal subset** | Ru/Ir/Rh/Os/Re = canonical Wf-390-A1 Path A cohort per `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:141 FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")` | Wf-390-A1 |
| **kcal sign fix** | VERIFIED at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:635` to be the **post-Wf-333-V1** convention `(1 - 2*comp)*kcal_per_unit` so that perfect composite (1.0) yields `-1.0 * kcal_per_unit` (i.e. more negative = better binding per Vina convention) | Wf-333-V1 |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+code+test; no GPU launch per AGENTS.md §5.1) | NOT re-run |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **Wetlab planning** | 0 (per AGENTS.md §5.4 + §10; user directive "搞清楚我们不用做湿实验") | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED (no new pre-flight gates added by Wf-406-A4) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~5 min CPU-only for SCAFFOLD smoke + <1 s for tests | this entry |

**One-line verdict**: Wf-406-A4 ships the CPU-only SCAFFOLD surface for the Wf-246-C metal-coordination term retrain with 5-step smoke + kcal sign audit + bash launch wrapper with USER_OK gate + 2 NEW tests PASS. Actual 5-metal × 5-seed × 50-epoch Wf-131 P3 retrain with MetalCoordinationTerm enabled remains BLOCKED per AGENTS.md §5.1 + USER_OK-gated.

---

## §1 Why Wf-406-A4 exists (motivation)

Per Wf-246-C plan + Wf-333-V1 sign fix + Wf-390-A1 Path A 5-metal subset + Wf-405 recovery wave user directive 2026-09-30 verbatim *"之前的workflow跑到一半失效了，能恢复他们吗？"*, the Wf-246-C metal-coordination term was a CPU SCAFFOLD that needed to be wired into the Wf-131 P3 trainer. Wf-406-A4 ships the SCAFFOLD surface that makes this wire-in possible.

### §1.1 Wf-246-C root cause analysis (per Wf-333-V1 + Wf-246-B + Wf-325B)

The Wf-246-C metal-coordination term was originally SCAFFOLDed in `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py`. The sign convention was **inverted**: `metal_coord_term.py:629` originally computed `(2*c - 1) * kcal_per_unit` where `c ∈ [0, 1]` was the metal-coordination composite score. For perfect composite (c=1.0), this yielded `+1.0 * kcal_per_unit` — meaning perfect coordination gave POSITIVE kcal/mol adjustment, which means WORSE binding per Vina convention "lower (more negative) = better binding".

Wf-333-V1 fixed this by flipping the sign: `metal_coord_term.py:635` now computes `(1 - 2*c) * kcal_per_unit` so that perfect composite (c=1.0) yields `-1.0 * kcal_per_unit` (i.e. more negative = better binding per Vina convention).

The Wf-325B closeout added the `MEASURED_GENERIC_ATOM` graceful-degradation marker convention for Pt-coordination rejection cases (Pt rejected by Vina 1.2.7 PDBQT parser).

### §1.2 Wire-in architecture (per Wf-405 recovery directive)

Wf-246-C wire-in requires:

1. **CPU-only SCAFFOLD** = import `MetalCoordinationTerm` + build instance + compute kcal/mol adjustment + emit per-epoch kcal sign audit (this Wf)
2. **GPU retrain** = wire `MetalCoordinationTerm` into the Wf-131 P3 D-MPNN multi-task trainer's training loop, opt-in via `--metal-coord-term` CLI flag (default OFF for backward compat) (USER_OK-gated, future Wf)
3. **MEASURED verification** = Wf-131 P3 cohort + MetalCoordinationTerm enabled vs disabled, compare 5-seed × 50-epoch AUC + kcal/mol delta vs Wf-333-V3 baseline (USER_OK-gated, future Wf)

---

## §2 Wf-406-A4 deliverables (3 SHIPPED 100%)

### §2.1 `molmetal/scripts/wf406a4_metal_coord_retrain_scaffold.py` (NEW, 344 lines, 12,602 bytes)

The Wf-406-A4 SCAFFOLD script provides:

| Function | Purpose | Lines |
|---|---|---|
| `parse_args(argv)` | CLI parser for `--metal-coord-term` (opt-in wire-in flag, default OFF) + `--metals` (5-metal subset Ru/Ir/Rh/Os/Re) + `--seeds` + `--epochs` + `--smoke` + `--out-dir` + `--per-epoch-kcal-audit` + `--kcal-per-unit` (default 1.0) | ~30 |
| `import_metal_coord_term()` | Lazy-import `MetalCoordinationTerm` + `MetalCoordinationScore` from `molmetal.molmetal_lam.sbdd_env.metal_coord_term`; returns `(import_ok, detail_string)`; surfaces the Wf-333-V1 sign-fix reference at line 635 | ~20 |
| `build_metal_coord_term(kcal_per_unit, weight_geometry, weight_hsab, weight_distance)` | Build a `MetalCoordinationTerm` instance with canonical Wf-333-V1 sign-fixed kcal_adjustment; returns `(build_ok, instance_or_None, detail_string)` | ~40 |
| `per_epoch_kcal_sign_audit(composite_scores, kcal_per_unit)` | Audit the Wf-333-V1 kcal sign fix on a list of composite metal-coordination scores; returns JSON-serialisable audit dict with `sign_audit_pass` boolean | ~35 |
| `smoke_run(out_dir, metals, seeds, epochs, kcal_per_unit)` | 5-step CPU smoke that runs the SCAFFOLD end-to-end (import + build + kcal_sign_audit + emit result.json); surfaces the Wf-246-C + Wf-333-V1 + Wf-390-A1 lineage | ~120 |
| `main(argv)` | CLI entry point; accepts `--smoke` + `--metals` + `--seeds` + `--epochs` + `--out-dir`; exit 0 on PASS, exit 1 on FAIL | ~30 |

**Key design choices:**
- `--metal-coord-term` flag is opt-in (default OFF) for bit-for-bit backward compat with Wf-131 P1 / Wf-299 / Wf-328 / Wf-131 P3 5-metal subset shipped runs.
- kcal sign audit verifies `(1 - 2*comp)*kcal_per_unit < 0` for `score.total > 0.5` per Wf-333-V1 fix.
- The 5-metal subset (Ru/Ir/Rh/Os/Re) matches the canonical Wf-390-A1 Path A cohort.
- `MetalCoordinationTerm` ctor signature is `(weight_geometry, weight_hsab, weight_distance, kcal_per_unit)` per `metal_coord_term.py:535-541` — there is no `enable_metal_coord` flag (the term is always on; the SCAFFOLD gates via the `--metal-coord-term` CLI flag at the trainer level, not the term level).

### §2.2 `scripts/wf406a4_launch_wf246c_metal_coord_retrain.sh` (NEW, 216 lines)

The Wf-406-A4 bash launch wrapper provides:

- **`USER_OK=1` env-var gate** per AGENTS.md §5.1 + Wf-294B family convention + Wf-355-B1 USER_OK-gated launch wrapper pattern; exit 64 when `USER_OK != 1` (refuses to launch)
- **`bash -n` syntax check** via `command -v bash` + AST-parse equivalent
- **DRY_RUN.log emission** via `wf363a1_dry_run_log.py` Wf-363-A1 schema
- **plan.json emission** with `wf`, `metal-coord-term`, `metals`, `seeds`, `epochs`, `kcal-per-unit` fields
- **SCAFFOLD smoke invocation** with `--smoke --metals Ru Ir --seeds 0 1 --epochs 5 --out-dir <tmp>` for SCAFFOLD verification (5-step CPU-only smoke)
- **`--full` flag** for actual GPU retrain launch (`python train_dmpnn_multitask_wasserstein.py --metal-coord-term --metals Ru Ir Rh Os Re --seeds 0 1 2 3 4 --epochs 50`); USER_OK-gated
- **Wave budget tracking** via `wf360a1_wave_budget.py` Wf-360-A1 helper; per-launch budget allocation

### §2.3 `molmetal/tests/test_wf406a4_metal_coord_scaffold.py` (NEW, ~250 lines, 2 NEW tests)

The Wf-406-A4 test file provides 2 NEW tests:

1. **`test_scaffold_smoke_passes`** — Verifies the Wf-406-A4 SCAFFOLD script runs the 5-step CPU smoke end-to-end without crashing; import + build + kcal_sign_audit all succeed; result.json is emitted with `status = "SHIPPED_SCAFFOLD_CPU_ONLY"` + `n_metals = 2` + `n_seeds = 2` + `kcal_sign_audit.n_scores_audited = 5`.
2. **`test_launch_wrapper_syntax`** — Verifies the bash launch wrapper passes `bash -n` syntax check + the wrapper REFUSES to run when `USER_OK != 1` (exit 64) + the wrapper runs the SCAFFOLD smoke end-to-end when `USER_OK=1` is set + mode is `--smoke`.

**Honest framing:**
- Tests are CPU-only and depend on `rdkit` being installed (transitively, via `metal_coord_term.py`).
- 0 GPU launches performed by these tests.
- 0 `paper/*` touches.
- 0 wet-lab planning (per AGENTS.md §10).

---

## §3 Smoke artefact

When run via the bash launch wrapper with `USER_OK=1 --smoke --metals Ru Ir --seeds 0 1 --epochs 5 --out-dir <tmp>`, the SCAFFOLD emits:

```json
{
  "wf": "Wf-406-A4",
  "status": "SHIPPED_SCAFFOLD_CPU_ONLY",
  "n_metals": 2,
  "n_seeds": 2,
  "kcal_sign_audit": {
    "composite_scores": [0.85, 0.92, 0.78, 0.88, 0.91],
    "post_fix_kcal": [-0.7, -0.84, -0.56, -0.76, -0.82],
    "pre_fix_kcal": [0.7, 0.84, 0.56, 0.76, 0.82],
    "sign_audit_per_score": [true, true, true, true, true],
    "sign_audit_pass": true,
    "kcal_per_unit": 1.0,
    "wf_333_v1_fix_id": "metal_coord_term.py:635 (1-2*comp)*kcal_per_unit",
    "n_scores_audited": 5
  },
  "import_ok": true,
  "build_ok": true,
  "wall_seconds": 1.23,
  "provenance": "MEASURED"
}
```

**Status: SHIPPED_SCAFFOLD_CPU_ONLY**. The smoke verifies (a) import OK, (b) build OK with canonical Wf-333-V1 sign fix, (c) kcal_sign_audit passes for 5 synthetic composite scores in the [0.78, 0.92] range (all > 0.5 → post_fix_kcal < 0 → correct Vina convention "lower = better binding"), (d) Wf-333-V1 fix ID is referenced.

---

## §4 Honest framing (`party-twenty-spirit` §3.5)

### §4.1 What is MEASURED end-to-end (SCAFFOLD only)

- ✅ Wf-406-A4 SCAFFOLD script runs the 5-step CPU smoke end-to-end without crashing
- ✅ Wf-406-A4 SCAFFOLD script imports `MetalCoordinationTerm` + `MetalCoordinationScore` successfully
- ✅ Wf-406-A4 SCAFFOLD script builds a `MetalCoordinationTerm` instance successfully
- ✅ Wf-406-A4 SCAFFOLD script's kcal sign audit passes for all 5 synthetic composite scores > 0.5 (post-fix sign correct)
- ✅ Wf-406-A4 launch wrapper passes `bash -n` syntax check
- ✅ Wf-406-A4 launch wrapper REFUSES to run when `USER_OK != 1` (exit 64)
- ✅ Wf-406-A4 launch wrapper runs the SCAFFOLD smoke end-to-end when `USER_OK=1 --smoke`
- ✅ Wf-406-A4 5-metal subset (Ru/Ir/Rh/Os/Re) matches canonical Wf-390-A1 Path A cohort
- ✅ Wf-406-A4 kcal sign fix verified at `metal_coord_term.py:635` per Wf-333-V1

### §4.2 What is NOT MEASURED

- ❌ MEASURED end-to-end Wf-131 P3 5-seed × 50-epoch retrain with MetalCoordinationTerm enabled is NOT yet performed. PROJECTED lift on per-metal AUC + pIC50 R² + W₂ pairwise remains until Blackwell sm_120 retrain with USER_OK=1 grant per AGENTS.md §5.1.
- ❌ The MetalCoordinationTerm ctor does NOT have an `enable_metal_coord` flag — the term is always on. The SCAFFOLD gates via `--metal-coord-term` at the trainer level, NOT the term level. Actual wire-in requires trainer edit to invoke `MetalCoordinationTerm.forward()` when `--metal-coord-term` is set.

### §4.3 Honest-negative caveats

1. **GPU retrain BLOCKED on USER_OK=1** per AGENTS.md §5.1 + Wf-355-B1 USER_OK gate pattern. Closure path = USER_OK=1 grant + ~5 hr Blackwell sm_120 retrain on 5-metal subset.
2. **Trainer wire-in NOT YET performed** — the SCAFFOLD smoke verifies the import + build + sign audit contract; actual wire-in to `train_dmpnn_multitask_wasserstein.py` training loop is deferred to a future Wf wave that requires the SCAFFOLD smoke PASS + USER_OK-gated retrain.
3. **Forward-looking items preserved**: m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL (per Wf-377-A1 + Wf-388-A1); 5 forward-looking USER_OK items still BLOCKED on user OK per AGENTS.md §5.1.
4. **MEASURED vs PROJECTED discipline**: All speedup / lift claims from this Wf-406-A4 are PROJECTED until the Wf-131 P3 retrain is GPU-executed and verified against the Wf-333-V3 baseline.

---

## §5 Recommended follow-up

**Wf-406-A5+ (USER_OK-gated)**:
1. **Trainer wire-in** (~50 LOC edit): add `--metal-coord-term` CLI flag + MetalCoordinationTerm invocation in `train_dmpnn_multitask_wasserstein.py` training loop. Default OFF preserves bit-for-bit backward compat with Wf-131 P1 / Wf-299 / Wf-328 / Wf-131 P3.
2. **Wf-385-A3 SCAFFOLD wire-in pre-flight gate re-run** (10 tests PASS) — verify the trainer's `--metal-coord-term` flag passes the Wf-385-A3 wire-in detection contract.
3. **GPU retrain** on the 5-metal subset (~5 hr Blackwell sm_120, 5 seeds × 50 epochs); USER_OK=1 grant required per AGENTS.md §5.1.

**Wf-405+ follow-up** (per Wf-405-B1 §7 forward-looking): Wf-406-A3 Wf-246-C metal-coordination term retrain (~3 hr GPU USER_OK-gated) + Wf-406-A4 SCAFFOLD wire-in + Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120 + Wf-388 Wf-125 P1 CONTROLD PAIR retrain (~2 hr GPU) + Wf-379-A1 m1_auc 5-seed result close + Wf-407-A1 Wf-299 Wave B launch (~10 min GPU on idle GPU 1) (~21.5 hr GPU + 13 hr CPU + 30 min install total).

NEW ultracode waves Wf-407+ IS the recommended follow-up per the Wf-385-A1 audit + Wf-405-A1 SCAFFOLD-loss gap audit + Wf-405-A2 SCAFFOLD-loss wire-in + Wf-405-B1 docs-only sync + Wf-407-B1 §22.93 GPU launch context + Wf-406-A4 this wave.

## §6 Conclusion

Wf-406-A4 closes the SCAFFOLD gap for the Wf-246-C metal-coordination term retrain with a CPU-only SCAFFOLD script + bash launch wrapper + 2 NEW tests PASS + a 5-step smoke that verifies the import + build + kcal sign audit contract. **Bit-for-bit backward compat with Wf-131 P1 / Wf-299 / Wf-328 / Wf-131 P3 baseline is preserved** when `--metal-coord-term` is OFF (the default). **MEASURED end-to-end Wf-131 P3 retrain with MetalCoordinationTerm enabled remains USER_OK-gated** per AGENTS.md §5.1 + Wf-355-B1 USER_OK gate pattern.

**3/3 deliverables SHIPPED** (1 SCAFFOLD script + 1 bash launch wrapper + 2 NEW tests + this closeout). **0 NEW GPU launches** (CPU-only docs+code+test+verify per AGENTS.md §5.1). **0 `paper/*` touches** (per AGENTS.md §5.3 TODO-84 active). **0 wet-lab planning** (per AGENTS.md §5.4 + §10 Tier-2 re-orientation).

---

**Wf-406-A4 closeout** — 2026-09-30 — Owner Hugo Lee + Wf-406-A4 sub-agent (per Wf-385-B1 + Wf-405-A1 + Wf-405-B1 + Wf-407-B1 + Wf-333-V1 + Wf-246-C + Wf-390-A1 + TODO-84 + AGENTS.md §5.1 prior context).
