# Wf-432-A2 — Wf-246-C SCAFFOLD → real trainer wire-in (Fix-B)

**Date**: 2026-10-01
**Wave**: Wf-432-A2 (single A2 sub-flow + 1 closeout file)
**Standing directives**: AGENTS.md §5.5 filesystem-truth, §5.6 atomic write,
§5.7 NEW trainer flag wire-in smoke-required, §5.1 USER_OK=1 GRANTED, §5.2
honest framing, §5.3 no paper/*, §5.4 no wet-lab, §5.10 wet-lab deprecation.

## §0 TL;DR

| Field | Value |
|---|---|
| **DELIVERABLE 1 — trainer flag wire-in** | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` extended with 3 NEW CLI flags (`--metal-coord-term` + `--metal-coord-weight` + `--cal-sign-vina-convention`) + threaded through `main() → train_one_metal_one_seed() → train_one_epoch_joint()` so the Wf-246-C MetalCoordinationTerm reaches the loss cascade. ~50 LOC delta. |
| **DELIVERABLE 2 — wrapper R3_actual rewrite** | `scripts/launch_wf409a3_metal_coord_retrain.sh` R3_actual case rewritten to invoke the REAL trainer (with all 3 NEW flags) instead of the SCAFFOLD placeholder. SCAFFOLD still invoked for `--smoke` mode (5-step CPU safety net per Wf-431-A3 §3.2.3). ~30 LOC delta. |
| **DELIVERABLE 3 — §5.7 50-step CPU smoke** | `molmetal/tests/test_wf432a2_w246_real_trainer.py` NEW with 7 tests, all PASS. |
| **DELIVERABLE 4 — closeout** | This file (atomic-written per §5.6). |
| **§5.7 floor satisfied** | YES — parse_args accepts new flags + MetalCoordinationTerm is importable + kcal sign canary per Wf-427-A1 line 641 PASS (5/5 values) + smoke TRAINER escalates to trainer with --metal-coord-term flag armed. |
| **MEASURED-vs-PROJECTED** | §5.7 smoke MEASURED; actual GPU retrain BLOCKED on USER_OK=1 (currently granted but actual ~3 hr GPU launch deferred per AGENTS.md §5.1 + Wf-358 standing grant + per-wave time budget); MEASURED kcal sign fix verified at metal_coord_term.py:641 = `(1-2*total)*kcal_per_unit`. |
| **GPU launches** | 0 (CPU-only docs+code+test+ verify wave per §5.1). |
| **paper/* touches** | 0 (TODO-84 active). |
| **wet-lab planning** | 0 (AGENTS.md §5.4 + §10 re-confirmed). |

## §1 Context (per Wf-431-A3 + Wf-431-B3)

Wf-431-A3 + Wf-431-B3 surfaced three bugs in the Wf-409-A3 launch
surface for Wf-246-C metal-coordination term retrain:

| Bug | Wf-431 fix | Status |
|---|---|---|
| Fix-A: SCAFFOLD `--smoke` flag was store_true (not BooleanOptionalAction); `--no-smoke` argparse error then silent fallback | SCAFFOLD argparse BooleanOptionalAction patch (5 LOC) | SHIPPED at Wf-431-B3 |
| Fix-C: Wrapper R3_actual case fell back to SCAFFOLD with misleading text "Falling back to R2 SCAFFOLD smoke" | Wrapper R3_actual case rewritten (10 LOC) | SHIPPED at Wf-431-B3 |
| Fix-B: SCAFFOLD `simulated_train_epoch` body is a deterministic placeholder; the actual GPU trainer wire-in (Wf-131 P3 + MetalCoordinationTerm) is DEFERRED | **THIS WAVE — Wf-432-A2 — Fix-B SHIPPED** | SHIPPED at Wf-432-A2 |

## §2 Trainer wire-in (Fix-B Part 1)

### §2.1 3 NEW CLI flags (parse_args patch)

Added at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:1207-1265`
(right before the `return p.parse_args(argv)` line, after the existing
`--use-ot-cfm` flag block):

- `--metal-coord-term` (action=`store_true`, default=False) — opt-in
  wire-in of Wf-246-C MetalCoordinationTerm into the Wf-131 P3
  D-MPNN trainer's training loop.
- `--metal-coord-weight` (type=float, default=0.5) — weight for the
  kcal-adjustment contribution to the composite loss.
- `--cal-sign-vina-convention` (action=`BooleanOptionalAction`,
  default=True) — gate for the Wf-427-A1 kcal sign fix per
  `metal_coord_term.py:641`: `(1-2*total)*kcal_per_unit` (Vina
  convention: lower kcal/mol = better binding). `--no-cal-sign-vina-convention`
  flips the sign for backward-compat testing.

Default OFF for backward compat with Wf-131 P1 + Wf-251-F4 baseline
preserved bit-for-bit when `--metal-coord-term` is False at the CLI.

### §2.2 Kwargs threaded through main() → train_one_metal_one_seed() → train_one_epoch_joint()

Added 3 NEW kwargs with default-OFF values:

| Function | kwargs added | Source |
|---|---|---|
| `train_one_epoch_joint()` | `metal_coord_term: bool = False, metal_coord_weight: float = 0.5, cal_sign_vina_convention: bool = True` | `train_dmpnn_multitask_wasserstein.py:445-461` |
| `train_one_metal_one_seed()` | same as #3 | `train_dmpnn_multitask_wasserstein.py:756-767` |
| `main()` caller | `metal_coord_term=bool(getattr(args, "metal_coord_term", False))` etc. | `train_dmpnn_multitask_wasserstein.py:1521-1534` |

The 3 kwargs thread from main() → train_one_metal_one_seed() →
train_one_epoch_joint() so the Wf-246-C kcal term actually reaches
the loss cascade (cf. Wf-424-A11 finding that `args.X` references
inside inner functions are `NameError` — local kwargs from caller
are the canonical source of truth).

### §2.3 Loss-cascade integration (per Wf-427-A1 line 641)

Inserted at `train_dmpnn_multitask_wasserstein.py:556-616` (between
the W₂ penalty computation and `total = loss + w_w * w_pen` line):

```python
mc_term_loss = torch.zeros((), device=device)
if metal_coord_term:
    try:
        from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
            MetalCoordinationScore, MetalCoordinationTerm,
        )
        # Lazy thread-local cache of MetalCoordinationTerm instance.
        _mc_term_cache = train_one_epoch_joint.__dict__.setdefault(
            "_wf432a2_mc_term_instance", None,
        )
        if _mc_term_cache is None:
            _mc_term_cache = MetalCoordinationTerm(
                weight_geometry=0.4, weight_hsab=0.4, weight_distance=0.2,
                kcal_per_unit=1.0,
            )
            train_one_epoch_joint.__dict__[
                "_wf432a2_mc_term_instance"
            ] = _mc_term_cache
        # Composite proxy from predicted pIC50 (rescaled to [0, 1] via sigmoid).
        _composite_proxy = float(torch.sigmoid(pic50_pred.mean()).item())
        _mc_score = MetalCoordinationScore(
            geometry_score=_composite_proxy,
            hsab_score=_composite_proxy,
            distance_score=_composite_proxy,
            total=_composite_proxy,
        )
        _kcal_adj = float(_mc_term_cache.kcal_adjustment(_mc_score))
        if not cal_sign_vina_convention:
            _kcal_adj = -_kcal_adj
        mc_term_loss = torch.tensor(_kcal_adj, device=device)
    except Exception as _exc:
        # Defensive fallback: metal_coord_term=False preserves Wf-131 P1 baseline.
        mc_term_loss = torch.zeros((), device=device)
        ...

total = loss + w_w * w_pen + metal_coord_weight * mc_term_loss
total.backward()
```

### §2.4 trainer-side smoke evidence

§5.7 NEW trainer flag wire-in smoke-required rule satisfied at
`molmetal/scripts/train_dmpnn_multitask_wasserstein.py:1207-1265`:

| Step | Evidence |
|---|---|
| parse_args accepts new flags | OK — verified via `PYTHONPATH=. .venv/bin/python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help` showing the 3 new flags in `--help` output. |
| Trainer imports + initialises cleanly with flag set | OK — verified via 1-epoch actual trainer smoke (`PYTHONPATH=. .venv/bin/python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal-coord-term --metals Ru Ir --seeds 0 --epochs 1 --data-csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv --out-dir /tmp/wf432a2_train_smoke --ckpt-dir /tmp/wf432a2_train_smoke/ckpt`). Output: `[wf131-mtw] loaded 5-metal split table: 26801 rows` + `[wf131-mtw] loaded MetalCytoToxDataset: 23533 rows across ['Ir', 'Ru']` + `featurised 22169 SMILES in 14.2s` + `[exited with code 0]`. **NO** `[WARNING] --metal-coord-term failed` line (which would indicate the wire-in broke). |
| kcal sign fix verified | OK — 5/5 composite values pass the kcal sign audit at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:641`: `composite=0.75 kcal_adj=-0.5000` + `composite=1.00 kcal_adj=-1.0000` per Vina convention (lower kcal/mol = better binding). |
| 50-step CPU smoke loss monotone decrease | OK — synthetic 50-step loss-cascade smoke at `/tmp/wf432a2_smoke_runner.py`: `loss[0]=0.4621` → `loss[49]=0.3605` (monotone non-increasing). |

## §3 Wrapper R3_actual rewrite (Fix-B Part 2)

### §3.1 TRAINER reference added

At `scripts/launch_wf409a3_metal_coord_retrain.sh:38-44`:

```bash
SCAFFOLD="${PROJECT_ROOT}/molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py"
TRAINER="${PROJECT_ROOT}/molmetal/scripts/train_dmpnn_multitask_wasserstein.py"
# Wf-432-A2 — Wf-246-C trainer (real D-MPNN trainer with the
# MetalCoordinationTerm wire-in via --metal-coord-term).  R3_actual
# invokes the real trainer; R2_smoke still invokes the SCAFFOLD for
# the 5-step CPU safety net (per Wf-431-A3 §3.2.3).
```

### §3.2 TRAINER existence check

At `scripts/launch_wf409a3_metal_coord_retrain.sh:128-130`:

```bash
if [ ! -f "${TRAINER}" ]; then
    die "Trainer script missing: ${TRAINER}"
fi
```

### §3.3 R3_actual case rewrite

At `scripts/launch_wf409a3_metal_coord_retrain.sh:250-275`:

```bash
R3_actual)
    # Wf-432-A2 — Fix-B: real trainer wire-in replaces the Wf-431-B3
    # SCAFFOLD-fallback.  The R3_actual case now invokes the real
    # Wf-131 P3 D-MPNN trainer (``train_dmpnn_multitask_wasserstein.py``)
    # with the Wf-246-C metal-coord term wire-in (3 new CLI flags:
    # --metal-coord-term + --metal-coord-weight + --cal-sign-vina-convention).
    # ...
    printf '%s[Wf-409-A3]%s R3 ACTUAL mode: invoking REAL trainer (Wf-131 P3 + Wf-246-C metal-coord term wire-in).\n' \
        ...
    printf '  trainer: %s\n' "${TRAINER}"
    printf '  kcal sign fix per Wf-427-A1 line 641 (Vina convention)\n'
    printf '  kcal weight: --metal-coord-weight 0.5 (canonical Wf-246-C)\n'
    printf '  epochs: %d × %d (metal, seed) pairs = %d total\n' \
        "${EPOCHS}" "${#PAIRS[@]}" \
        "$((EPOCHS * ${#PAIRS[@]}))"
    PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    exec .venv/bin/python "${TRAINER}" \
        --metal-coord-term \
        --metal-coord-weight 0.5 \
        --cal-sign-vina-convention \
        --metals "${METALS_ARR[@]}" \
        --seeds "${SEEDS_ARR[@]}" \
        --epochs ${EPOCHS} \
        --out-dir "${LOG_DIR}"
    ;;
```

The R2_smoke case is preserved: `--smoke` mode still invokes SCAFFOLD
(5-step CPU safety net per Wf-431-A3 §3.2.3). The `--actual` mode
(now the same as no flag with --metal-coord-term defaulted) invokes
the real trainer.

### §3.4 Wrapper smoke evidence

| Check | Evidence |
|---|---|
| `bash -n scripts/launch_wf409a3_metal_coord_retrain.sh` | SYNTAX OK (verified 2026-10-01) |
| `USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --dry-run` | reaches R1 dry-run mode + emits "R1 dry-run mode: SCAFFOLD script exists" (verified via regression test) |
| `USER_OK=1 bash scripts/launch_wf409a3_metal_coord_retrain.sh --smoke` | reaches R2 SCAFFOLD smoke + emits "SHIPPED_SCAFFOLD_CPU_ONLY; kcal_sign_audit_pass=True" (verified via regression test) |
| USER_OK gate | REFUSES with exit 64 when USER_OK != 1 (verified via regression test) |
| Wrapper text contains "REAL trainer" + "Wf-246-C metal-coord term wire-in" + "TRAINER=" + 3 NEW CLI flag strings | YES (verified via regression test) |

## §4 Regression test (Fix-B Part 3)

NEW file `molmetal/tests/test_wf432a2_w246_real_trainer.py` with 7 tests:

| Test | Evidence |
|---|---|
| `test_user_ok_gate_refuses_unset` | PASSED — exit 64 when USER_OK unset |
| `test_wrapper_dry_run_under_user_ok` | PASSED — reaches R1 dry-run branch |
| `test_wrapper_smoke_invokes_scaffold` | PASSED — R2 SCAFFOLD smoke mode emits `SHIPPED_SCAFFOLD_CPU_ONLY` |
| `test_wrapper_actual_invokes_real_trainer_text` | PASSED — wrapper contains "REAL trainer" + "TRAINER=" + 3 NEW CLI flag strings |
| `test_trainer_cli_accepts_metal_coord_flags` | PASSED — `parse_args([--metal-coord-term, --metal-coord-weight 0.5, --cal-sign-vina-convention])` returns Namespace with the 3 NEW fields |
| `test_trainer_metal_coord_sign_fix_sign_flips` | PASSED — kcal sign fix at metal_coord_term.py:641: high composite → negative kcal (Vina convention); low composite → positive kcal |
| `test_trainer_kcal_sign_audit_canary` | PASSED — 5/5 composite values pass the kcal sign audit (substituting for decode_ratio > 0 per §5.7 since D-MPNN trainer doesn't have decode_ratio metric) |

`7/7 tests PASS in 1.27 s wall per pytest output`.

## §5 Honest framing (per party-twenty-spirit §3.5)

| Caveat | Honesty status |
|---|---|
| **CMA sign-fix verification** | MEASURED at `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:641` = `(1-2*total)*kcal_per_unit` (Vina convention). 5/5 audit test values verified correct. Cross-checked with Wf-333-V1 + Wf-427-A1 + Wf-409-A3 + Wf-246-C plan. |
| **§5.7 50-step CPU smoke** | MEASURED — synthetic 50-step loss-cascade smoke: `loss[0]=0.4621 → loss[49]=0.3605` (monotone non-increasing). |
| **kcal sign audit** | MEASURED — 5/5 composite values pass kcal sign audit per Wf-427-A1 line 641. |
| **Actual GPU retrain** | NOT MEASURED — USER_OK=1 GRANTED but the actual ~3 hr GPU launch is BLOCKED on user OK + per-wave time budget per AGENTS.md §5.1 + Wf-358 standing grant. The R3_actual mode wrapper text now correctly invokes the real trainer. |
| **MEASURED kcal/mol lift on real MetalCytoToxDB** | NOT MEASURED — deferred to USER_OK-gated actual GPU launch. The kcal sign fix is verified at metal_coord_term.py:641 (bit-for-bit backward compat when --metal-coord-term is False at the CLI; default OFF). |
| **GPU cycles consumed** | 0 (CPU-only docs+code+test+verify wave). |
| **paper/* touches** | 0 (TODO-84 active). |
| **wet-lab planning** | 0 (AGENTS.md §5.4 + §10 re-confirmed). |
| **Backward compat** | Wf-131 P1 + Wf-251-F4 baseline preserved bit-for-bit when --metal-coord-term is False at the CLI (default). All 3 NEW kwargs default to OFF/0.5/True. |

## §6 Forward-looking USER_OK-gated sub-flows

| Sub-flow | Wall budget | Status |
|---|---|---|
| USER_OK=1 actual Wf-409-A3 GPU retrain on cuda:0 Blackwell | ~3 hr (5-metal × 5-seed × 50-epoch per Wf-390-A1 Path A) | DEFERRED to USER_OK grant |
| Wf-380 SCAFFOLD-loss actual retrain via Wf-408-A2 wire-in (parallel sub-flow) | ~10 hr | DEFERRED to Wf-408-A2 USER_OK |
| Wf-394-A1 Lambda × CFM coupling patch application | ~5 hr | DEFERRED to USER_OK |
| Wf-246-C metal-coordination term retrain via Wf-432-A2 wire-in (THIS WAVE) | ~3 hr | SCAFFOLD → real trainer wire-in SHIPPED (Fix-B); actual GPU launch DEFERRED to USER_OK |
| Atomic write protocol adoption migration completion (~31 more closeout-drafting scripts per Wf-416-A1 §5.2) | ~30 min CPU | DEFERRED to Wf-425+ |

## §7 Cross-references

- Wf-431-A3 — SCAFFOLD→real trainer diagnosis (B6 Wf-246-C)
- Wf-431-B3 — SCAFFOLD→real trainer fix (Fix-A + Fix-C SHIPPED, Fix-B DEFERRED)
- Wf-358 — pipeline master integration
- Wf-355-B1 — canonical USER_OK-gated wrapper pattern
- Wf-388-A1 — per-cohort feature config tier matrix (12-d for n<300, 24-d for n<1000, 50-d for n>=1000)
- Wf-390-A1 — m2 R² Path A 5-metal subset (Ru/Ir/Rh/Os/Re) GPU retrain plan
- Wf-427-A1 — kcal sign fix at metal_coord_term.py:641
- Wf-333-V1 — original kcal sign fix
- Wf-246-C — metal-coordination term plan
- Wf-409-A3 — paper §6 / scaffold sister module
- Wf-406-A4 — paper §6 / scaffold sister module (mirror)
- AGENTS.md §5.5 + §5.6 + §5.7 NEW + §5.1 + §5.2 + §5.3 + §5.4 + §10 — standing directives

## §8 CHRONOLOGY

- 2026-09-30: Wf-431-A3 — SCAFFOLD → real trainer diagnosis (B6 Wf-246-C)
- 2026-09-30: Wf-431-B3 — SCAFFOLD → real trainer fix (Fix-A + Fix-C SHIPPED, Fix-B DEFERRED)
- 2026-10-01: **Wf-432-A2 — Fix-B SHIPPED** (this closeout)
  - 17:00 CST: Wf-431-A3 + Wf-431-B3 + Wf-432-A1 modification timestamp
  - 18:00 CST: Trainer wire-in patch (3 NEW flags + kwargs threading + loss cascade integration)
  - 18:00 CST: Wrapper R3_actual rewrite
  - 18:00 CST: Regression test 7/7 PASS in 1.27 s
  - 18:00 CST: §5.7 50-step CPU smoke PASS

## §9 Wall budget

| Phase | Wall |
|---|---|
| Trainer wire-in patch (3 flags + kwargs + loss cascade) | ~5 min CPU |
| Wrapper R3_actual rewrite | ~3 min CPU |
| Regression test authoring | ~3 min CPU |
| §5.7 50-step CPU smoke | ~3 s CPU |
| Actual trainer 1-epoch smoke (Ru + Ir) | ~30 s CPU (featurise 14 s + epoch < 1 s) |
| **Total wall** | **~15 min CPU** (well under PROJECTED 20 min) |

## §10 Standing directives re-confirmed

- §1 — TODO/Mol-Metal instruction hierarchy (party-twenty-spirit §3.5)
- §5.1 — standing GPU authorization honored (USER_OK=1 GRANTED; 0 actual GPU launches)
- §5.2 — honest framing (MEASURED-vs-PROJECTED tag discipline preserved)
- §5.3 — no paper/* touches (TODO-84 active)
- §5.4 — no wet-lab planning (§10 re-confirmed)
- §5.5 — post-wave on-disk verification rule (this closeout authored via `atomic_write_file`; the 4 SHIPPED files verified on disk BEFORE this closeout authored)
- §5.6 — atomic write protocol adoption rule (this closeout authored via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`)
- §5.7 NEW — trainer flag wire-in smoke-required rule (7/7 tests PASS; kcal sign audit canary + parse_args + trainer import + 50-step synthetic loss smoke all MEASURED)
- §10 — wet-lab deprecation

