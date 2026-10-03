# Wf-376-A2 — m1_auc multi-seed evaluation scaffold (CPU-only; extends Wf-359-A2 + paper §6 framing)

**Date**:        2026-09-30
**Workflow**:    Wf-376-A2 (m1_auc CPU-only follow-up to Wf-371-A1 m6_kappa close-out)
**Status**:      SHIPPED — scaffold + 3 NEW tests PASS (CPU-only; 0 GPU launches; 0 paper/* touches)
**Owner**:       Hugo Lee + Wf-376-A2 ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per **Wf-371-A1 §6 forward-looking recommendation** ("after m6_kappa is
closed out, m1_auc is the next focus"), Wf-376-A2 ships a **CPU-only
multi-seed evaluation scaffold** for m1 MetalCytotoxDB Ru AUC. It
extends Wf-359-A2 (post-hoc per-seed aggregation) with full 5-fold CV
+ multi-seed averaging + 95% confidence interval + ENV-gated dispatch
+ paper §6 framing fields.

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Scaffold script** | `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` | ~340 LOC | SHIPPED; SCAFFOLD mode READY; LIVE mode USER_OK-gated |
| **Test file** | `molmetal/tests/test_wf376a2_m1_multiseed.py` | ~310 LOC (3 tests) | 3/3 PASS (0.13 s wall, CPU-only) |
| **Aggregate output** | `molmetal/reports/wf376a2_m1_auc_multiseed_scaffold/aggregate.json` | EMITTED on scaffold run | 5-metal × 5-seed × 5-fold schema |

**Tests**: 3/3 PASS
- `test_wf376a2_scaffold_structure` — public API + CLI flags + aggregate schema
- `test_wf376a2_env_dispatch_logic` — SCAFFOLD default + USER_OK=1 → LIVE; --mode override; `paper_section6_framing` populated
- `test_wf376a2_per_metal_auc_aggregation` — mean ± std + 95% CI math + honest-negative contract (MEASURED / HONEST_NEGATIVE_1SEED / PARTIAL_2SEED / NOT_MEASURED)

**Honest framing**: SCAFFOLD mode uses SYNTHETIC deterministic per-seed AUCs (not real measurements). LIVE mode requires USER_OK=1 + VRAM budget check per AGENTS.md §5.1 + Wf-355-B1 wrapper.

---

## §1 Scaffold design

### §1.1 Public API

| Symbol | Purpose | Source |
|---|---|---|
| `build_aggregate(metals, seeds, n_folds, mode)` | Build aggregate dict for metals × seeds × folds | Wf-376-A2 §1.4 |
| `aggregate_metal_multiseed(metal, seeds, per_seed_aucs)` | Per-metal mean ± std + 95% CI | Wf-376-A2 §1.3 |
| `resolve_mode()` | ENV-gated mode detection (SCAFFOLD vs LIVE) | Wf-376-A2 §1.5 |
| `synthetic_per_seed_auc(metal, seed, n_folds)` | Deterministic synthetic per-fold AUC generator | Wf-376-A2 §1.2 |
| `build_per_seed_aucs_scaffold(metals, seeds, n_folds)` | SCAFFOLD-mode per-seed builder | Wf-376-A2 §1.5 |
| `build_per_seed_aucs_live(metals, seeds, n_folds)` | LIVE-mode stub (USER_OK=1 → Wf-355-B1 wrapper) | Wf-376-A2 §1.5 |
| `main()` | CLI entry point | Wf-376-A2 §1.6 |

### §1.2 Synthetic per-seed AUC generator (SCAFFOLD mode)

Deterministic function: `synthetic_per_seed_auc(metal, seed, n_folds)`
returns `[fold0_auc, fold1_auc, ..., foldN_auc]` in `[0, 1]`. Each
value is derived from a hash of `(metal, seed)` with per-fold jitter
±0.05 + bias correction so the 5-fold mean lands near the
PROJECTED envelope `krasnov_baseline + PROJECTED_LIFT_W335_M2 =
0.7123 + 0.049 ≈ 0.7613`.

**Honest framing**: these AUCs are SYNTHETIC — they exercise the
aggregation math end-to-end without GPU work. They are NOT real
measurements. The PROJECTED envelope references the Wf-335-M2
CEILING_BREAK lift on metalcytotox_combined (R² +0.2227, AUC analogue
+0.049) per `party-twenty-spirit` §3.5.

### §1.3 Per-metal aggregation (mean AUC + std + 95% CI)

For each metal, the scaffold:
1. Flattens all `(seed, fold)` pairs into a single observation list.
2. Computes mean AUC = `sum(flat) / n_obs`.
3. Computes population std = `sqrt(sum((v - mean)²) / n_obs)`.
4. Computes 95% CI = `mean ± 1.96 * std / sqrt(n_obs)` (z-score normal approx).
5. Emits verdict per honest-negative contract (see §1.4 below).

### §1.4 Honest-negative contract (per Wf-357-B3 §4)

| n_seeds_used | verdict | semantics |
|:---:|---|---|
| 0 | `NOT_MEASURED` | No per-seed data on disk; emit empty block (mean=None) |
| 1 | `HONEST_NEGATIVE_1SEED` | Single-seed reading; emit with `n_seeds_used=1` tag; **no fake std** |
| 2 | `PARTIAL_2SEED` | 2-seed mean; population std on 2-sample pair |
| ≥3 | `MEASURED` | mean ± population std ± 95% CI; paper-grade |

This tag discipline preserves `party-twenty-spirit` §3.5 honest
framing — the scaffold NEVER silently promotes 1-seed data to a
5-seed mean.

### §1.5 ENV-gated dispatch

| Env | Mode | Behaviour |
|---|---|---|
| (none) | SCAFFOLD | CPU-only; synthetic per-seed AUCs; default per kill_then_adapt §0 |
| `USER_OK_SCAFFOLD=1` | SCAFFOLD | Explicit opt-in to scaffold (same as default) |
| `USER_OK=1` | LIVE | Would invoke Wf-355-B1 wrapper for real GPU retrain (USER_OK-gated) |
| `--mode live` (without USER_OK=1) | FAIL | Exit code 1 per kill_then_adapt §0 |

The `build_aggregate` function accepts an explicit `mode` arg; the
`--mode` CLI flag overrides ENV-gated detection. LIVE mode requires
`USER_OK=1` to actually launch (the script refuses with exit code 1
otherwise).

### §1.6 CLI entry point

```
PYTHONPATH=/home/hugo/codes/mol-metal python \\
    molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py \\
    [--metals Pt Ru Ir Au Pd] [--seeds 0 1 2 3 4] \\
    [--n-folds 5] [--mode {scaffold,live}] \\
    [--out /path/to/aggregate.json] [--print]
```

The stdout invariant per AGENTS.md §4 emits `PASS` / `FAIL` / `SKIP`.
The default `--metals` is the Wf-131 P3 5-metal subset (Pt / Ru /
Ir / Au / Pd); default `--seeds` is 0..4 (5-seed averaging).

---

## §2 ENV dispatch status (MEASURED)

| Test invocation | resolve_mode() | build_aggregate(mode=...) | main() exit |
|---|---|---|---|
| Default (no env vars) | SCAFFOLD | n/a | 0 (PASS) |
| `USER_OK_SCAFFOLD=1` | SCAFFOLD | SCAFFOLD | 0 (PASS) |
| `USER_OK=1` | LIVE | n/a | 0 (PASS, if `--mode live` also passed) |
| `--mode live` (no USER_OK=1) | n/a | n/a | **1 (FAIL — kill_then_adapt guard)** |

The `test_wf376a2_env_dispatch_logic` test verifies all four paths.

---

## §3 Paper §6 framing

### §3.1 `paper_section6_framing` field

The aggregate JSON carries a dedicated `paper_section6_framing` block
with 7 sub-fields designed for direct copy-paste into the Tier-2
paper §6 Limitations draft:

| Key | Purpose |
|---|---|
| `subsection` | Target paper section: "§6 Limitations / §6.2 Per-metric honest-negatives" |
| `m1_metric_id` | Canonical metric identifier: "m1 MetalCytotoxDB Ru AUC" |
| `wf_origin_short` | Provenance: "Wf-376-A2 multi-seed evaluation scaffold" |
| `MEASURED_today` | Honest accounting: CPU-only synthetic AUCs in SCAFFOLD mode |
| `PROJECTED_lift_envelope` | Expected lift when USER_OK=1 launches real training |
| `honest_negative_posture` | Per `party-twenty-spirit` §3.5: PROJECTED tag discipline |
| `kill_then_adapt_directive` | Per AGENTS.md §5.1: USER_OK=1 + VRAM budget check |

### §3.2 Honest framing for paper §6

Per **AGENTS.md §5.2** + `party-twenty-spirit` §3.5, the SCAFFOLD-mode
aggregate MUST be framed as PROJECTED, NOT MEASURED. The
`paper_section6_framing.MEASURED_today` field states:
> "CPU-only synthetic per-seed AUCs in SCAFFOLD mode; no real GPU
> measurements taken"

The `PROJECTED_lift_envelope` field states:
> "m1 Ru AUC +0.049 per Wf-335-M2 W₂+chemprop-v2 protocol when
> Wf-355-B1 USER_OK launch completes"

This dual-tag discipline preserves the §6 posture for Tier-2 paper
acceptance.

---

## §4 Test inventory (3/3 PASS; CPU-only)

### §4.1 test_wf376a2_scaffold_structure

| Aspect | Value |
|---|---|
| LOC | ~50 |
| What it tests | Public symbols exist (build_aggregate, aggregate_metal_multiseed, resolve_mode, synthetic_per_seed_auc, DEFAULT_METALS, DEFAULT_SEEDS, N_FOLDS, PROJECTED_LIFT_W335_M2, main); DEFAULT_METALS = (Pt, Ru, Ir, Au, Pd); DEFAULT_SEEDS = (0,1,2,3,4); N_FOLDS = 5; CLI flags include --metals/--seeds/--n-folds/--mode/--out/--print; aggregate JSON has all 15 top-level keys |
| Honest framing | Pure structural verification; no IO |

### §4.2 test_wf376a2_env_dispatch_logic

| Aspect | Value |
|---|---|
| LOC | ~80 |
| What it tests | Default → SCAFFOLD; USER_OK_SCAFFOLD=1 → SCAFFOLD; USER_OK=1 → LIVE (wins over USER_OK_SCAFFOLD); aggregate JSON carries `mode` field + env snapshot; paper_section6_framing block populated with honest-negative posture |
| Honest framing | Verifies SCAFFOLD is the safe default per kill_then_adapt §0 |

### §4.3 test_wf376a2_per_metal_auc_aggregation

| Aspect | Value |
|---|---|
| LOC | ~110 |
| What it tests | Synthetic 5-seed × 5-fold constant-AUC input → mean + std + 95% CI; honest-negative contract (MEASURED, HONEST_NEGATIVE_1SEED, PARTIAL_2SEED, NOT_MEASURED); m1 Ru-specific gate target delta; non-degenerate 95% CI math (alternating 0.6/0.8 input → mean=0.68, std=√0.0096 ≈ 0.09798, CI margin = 1.96 × std / 5) |
| Honest framing | Pure arithmetic verification; no IO; verifies the scaffold NEVER silently promotes 1-seed to 5-seed |

---

## §5 Cross-references

- **Wf-371-A1** — `TODO/completed/371a1_m6_kappa_degenerate.md` §6 (forward-looking: "m1_auc is the NEXT focus after m6_kappa is closed out")
- **Wf-359-A2** — `TODO/completed/359a2_m1_multi_seed_scaffold.md` (parent: per-seed JSON aggregation from disk)
- **Wf-357-B3** — `TODO/completed/357b3_tier2_honest_neg_metrics.md` §4 (m1 Ru AUC 0.7613 1-seed anchor; multi-seed gap)
- **Wf-355-B1** — `TODO/completed/355b1_wf131_p3_deferred.md` §1.1, §1.2, §2 (USER_OK-gated Wf-131 P3 7 DEFERRED-runs wrapper)
- **Wf-354-A1** — `TODO/completed/354a1_wf342_p2_lomo_integration.md` §1.2 + §5 (parent; 7 DEFERRED runs origin)
- **Wf-335-M2** — `TODO/completed/335m2_r2_w2_chemprop_full.md` (W₂+chemprop-v2 protocol reference for PROJECTED lift envelope)
- **Wf-131 P3 anchor** — `metrics/by_metric/metal_lomo_auc.json` (5-metal LOMO schema)
- **Wf-124 P3 anchor** — `metrics/by_metric/metalcytotox_ru_auc.json` `value=0.7613` (1-seed, MEASURED_PARTIAL_LIFT)
- **Wf-369-A2** — `TODO/completed/369a2_m2_r2_final_verdict.md` (FINAL verdict pattern for Tier-2 paper §6)
- **Wf-359-B2** — `TODO/completed/359b2_paper_section6.md` (paper §6 Limitations draft scaffold in `docs/`)
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.1 standing GPU authorization + §5.2 honest framing mandate + §5.3 no paper work + §5.4 Tier-2 markers UNCHANGED
- **`party-twenty-spirit`** §3.5 (MEASURED-vs-PROJECTED tag discipline + 0-tolerance honest framing)

---

## §6 What happens after USER_OK (per AGENTS.md §5.1)

When the user clears the m1_auc multi-seed gate and approves the Wf-355-B1 launch:

1. **Launch command**:
   ```bash
   USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh
   ```
2. **Wall**: ~5 hr GPU sequential (PROJECTED per Wf-260 G6 timing)
3. **Outputs**: per-(metal, seed) AUCs at `molmetal/reports/wf131_p3_deferred_runs/{metal}/per_seed_metrics_seed{N}.json`
4. **Aggregation** (this Wf-376-A2 scaffold in LIVE mode):
   ```bash
   USER_OK=1 PYTHONPATH=/home/hugo/codes/mol-metal python \
       molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py --mode live
   ```
5. **Result**: `molmetal/reports/wf376a2_m1_auc_multiseed_scaffold/aggregate.json`
   re-emitted with real per-metal mean AUC ± 95% CI from the 7
   DEFERRED runs (Rh 3+4 + Re 0-4).
6. **Operator review** (TODO-84 gated): the operator reviews the new
   aggregate against `metrics/by_metric/metalcytotox_ru_auc.json` and
   decides whether to promote any of the new per-metal readings into
   the canonical m1 metric JSON (per `party-twenty-spirit` §3.5 honest
   framing).

---

## §7 Honest framing per `party-twenty-spirit` §3.5

### §7.1 What this scaffold MEASURED today

1. ✅ Scaffold `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` SHIPPED (~340 LOC, `python -c "import ..."` clean)
2. ✅ 3 NEW tests PASS (CPU-only, 0.13 s wall)
3. ✅ Aggregate emitted to `molmetal/reports/wf376a2_m1_auc_multiseed_scaffold/aggregate.json` (5-metal × 5-seed × 5-fold schema)
4. ✅ ENV-gated dispatch verified: SCAFFOLD default + USER_OK=1 → LIVE
5. ✅ LIVE-mode guard verified: `python ... --mode live` without USER_OK=1 → exit 1 (FAIL per kill_then_adapt §0)
6. ✅ Paper §6 framing fields present + non-empty + carry MEASURED/PROJECTED honest-negative posture
7. ✅ Zero GPU launches performed (CPU-only verification)
8. ✅ No `paper/*` touches (TODO-84 honored)
9. ✅ Stdout invariant per AGENTS.md §4 (`PASS` / `FAIL` / `SKIP`) emitted

### §7.2 What this scaffold does NOT claim

1. ❌ Does NOT claim the SCAFFOLD-mode per-seed AUCs are real measurements — they are SYNTHETIC deterministic generator output, per `paper_section6_framing.MEASURED_today` field
2. ❌ Does NOT claim the Krasnov 2026 ≥ 0.86 gate is HIT — the synthetic envelope is ~0.7613 (matching Wf-124 P3 1-seed anchor by construction), which is BELOW the 0.86 gate
3. ❌ Does NOT claim the Wf-355-B1 wrapper has been launched — USER_OK gate remains HARD-REQUIRED per AGENTS.md §5.1
4. ❌ Does NOT claim the 7 DEFERRED runs are COMPLETE — they remain DEFERRED until USER_OK=1 launches the wrapper
5. ❌ Does NOT modify `metrics/by_metric/metalcytotox_ru_auc.json` — output is placed in `molmetal/reports/wf376a2_m1_auc_multiseed_scaffold/` for operator-promotion after honest-negative review
6. ❌ Does NOT claim the paper §6 entry has been written — `paper_section6_framing` is the SCAFFOLD for the §6 entry; the actual write to `paper/sections/06_limitations_v4.tex` requires USER_OK per TODO-84

### §7.3 Cross-metric honest framing

The SCAFFOLD-mode envelope `0.7613 = krasnov_baseline (0.7123) +
PROJECTED_LIFT_W335_M2 (0.049)` references the Wf-335-M2 CEILING_BREAK
protocol. The mean across (seed, fold) pairs is calibrated to this
envelope by construction, with ±0.05 per-fold jitter and seed-conditional
bias correction.

**Per the Wf-371-A1 §6 forward-looking recommendation**: this scaffold
is the NEXT step toward closing the m1_auc multi-seed gap. The full
closure requires:
1. USER_OK=1 launch of Wf-355-B1 wrapper (~5 hr GPU sequential)
2. Real per-seed AUCs populated into
   `molmetal/reports/wf131_p3_deferred_runs/{metal}/per_seed_metrics_seedN.json`
3. Re-run this scaffold in LIVE mode to populate the aggregate with
   real measurements
4. Operator review + promotion to canonical `metrics/by_metric/metalcytotox_ru_auc.json`

The m1_auc honesty gate (Krasnov ≥ 0.86) is NOT YET HIT — the Wf-124
P3 1-seed 0.7613 is the canonical anchor and requires the 5-seed ×
50-epoch full sweep to fully resolve.

---

## §8 Files touched

| Path | Action | LOC |
|---|---|---:|
| `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` | NEW | ~340 |
| `molmetal/tests/test_wf376a2_m1_multiseed.py` | NEW | ~310 (3 tests) |
| `molmetal/reports/wf376a2_m1_auc_multiseed_scaffold/aggregate.json` | EMITTED on scaffold run | ~86 lines |
| `TODO/completed/376a2_m1_multiseed_scaffold.md` | NEW (this doc) | ~250 |

**No source files were modified** beyond the NEW scaffold + tests + this doc + the emitted aggregate JSON. Zero `paper/*` touches (TODO-84 honored).

---

**END of Wf-376-A2** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; SHIP (CPU-only verification; 3 NEW tests; 0 GPU launches; 1 NEW scaffold; 1 NEW test file; 1 NEW aggregate JSON; 1 NEW doc; 6 honest-negative caveats preserved per §7.2).
