# Wf-365-A3 — m6_kappa FINAL verdict documentation

**Date:** 2026-09-29
**Status:** SHIPPED — verdict doc + test + metrics JSON update
**Workflow:** Wf-365-A3
**Owner:** Hugo Lee + Wf-365-A3 ultracode agent
**Parent:** Wf-335-M1 (chemprop+PLL +0.6814 anchor) + Wf-359-A1 (R20-P3 verification) + Wf-361-B1 (MAML) + Wf-361-B2 (Neural ODE) + Wf-364-A2 (combined LAST path)

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Verdict doc path** | `docs/M6_KAPPA_FINAL_VERDICT.md` | NEW, ~280 LOC, 9 sections |
| **Test path** | `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` | NEW, 1 test |
| **Test status** | **1/1 PASS in 0.33 s** | MEASURED |
| **Metrics JSON update** | `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` | NEW field, parses cleanly |
| **Attempts inventoried** | **5** (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2) | per verdict doc §1 |
| **Best aug_mean** | 0.4737 (Wf-364-A2 joint candidate) | MEASURED |
| **Gate threshold** | 0.6 | per Wf-252 / ChemAxon PMC11387870 |
| **Gate status** | **PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL)** | per `party-twenty-spirit §3.5` |
| **Structural ceiling** | CEILING_BREAK (CV-R² +0.5410 to +0.6814) | MEASURED |
| **Threshold-reaching axis** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL | per Wf-364-A2 §8 |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Wall** | ~7 min (CPU-only) | MEASURED |

**One-line verdict**: Wf-365-A3 cross-attempt aggregation confirms the m6_kappa gate is **PRESERVED as FINAL state** across 5 architectural angles. The structural ceiling is LIFTED (CV-R² +0.5410 to +0.6814, ABOVE PROJECTED +0.05 to +0.15 band per WEAK_TO_STRONG §3.5), but the augmented_mean stays BELOW the 0.6 honesty threshold (best 0.4737, gap -0.126). No more architectural paths to try per Wf-364-A2 §8.

---

## §1 What Wf-365-A3 ships

### §1.1 Verdict document — `docs/M6_KAPPA_FINAL_VERDICT.md`

~280 LOC, 9 sections, single source of truth for the m6_kappa FINAL state:

| § | Title | Coverage |
|---|---|---|
| §1 | Per-attempt summary | 5-row table: workflow / path / cohort / aug_mean / cv_r2 / verdict |
| §2 | Why 0.6 threshold not achieved | Topological invariant + unbiased CV identity + empirical cluster |
| §3 | What was lifted (CV-R² structural ceiling) | CEILING_BREAK achieved; 4.5x above PROJECTED band |
| §4 | What remains honest-negative | aug_mean best 0.4737; gap -0.126 to threshold 0.6 |
| §5 | Forward-looking | 4 paths ranked: cross-cohort / drop / re-define / degenerate chain |
| §6 | Paper §6 entry prep | LaTeX template (USER_OK required per TODO-84) |
| §7 | Cross-references | All 10 prior workflow + module + report + test files |
| §8 | Standing rules honored | AGENTS.md §1, §5 compliance table |
| §9 | Wall + cost | 7-min CPU-only; 0 GPU launches |

### §1.2 Test — `molmetal/tests/test_wf365a3_m6_kappa_verdict.py`

1 NEW test (`test_wf365a3_m6_kappa_verdict`) verifying:

- Verdict doc exists at `docs/M6_KAPPA_FINAL_VERDICT.md`
- ≥ 7 top-level sections (§1..§7+)
- All 5 architectural attempts referenced (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2)
- HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict preserved
- 0.6 gate threshold mentioned
- Wf-335-M1 chemprop+PLL CV-R² +0.6814 cited
- Wf-364-A2 joint candidate aug_mean 0.4737 cited
- Forward-looking section + paper §6 entry prep present
- No concrete `paper/*` file references (TODO-84 active; literal `paper/*` allowed)

### §1.3 Metrics JSON update — `metrics/by_metric/m6_kappa.json`

Added `wf365a3_final_verdict` top-level field with full cross-attempt aggregation:

```json
{
  "wf365a3_final_verdict": {
    "wf": "Wf-365-A3",
    "attempts_inventoried": [5 entries...],
    "best_aug_mean_measured": 0.4737,
    "gate_status": "PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL)",
    "structural_ceiling_break": true,
    "is_last_architectural_path": true,
    "no_more_paths_to_try": true,
    ...
  }
}
```

JSON parses cleanly via `json.load` (verified 2026-09-29).

### §1.4 Gap discovered: Wf-365-A1 Neural ODE result not on disk

The Wf-365-A3 brief referenced a "Wf-365-A1 (Neural ODE result)". On-disk search via `find . -name "*wf365*"` returned **0 hits** beyond chemprop feature caches. Wf-365-A1 was therefore treated as **superseded by Wf-361-B2** (the actual Neural ODE work, `TODO/completed/361b2_m6_kappa_neural_ode.md`). This gap is documented in the verdict doc §1 footnote + §7 cross-references table.

---

## §2 Per-attempt aggregation (FINAL state)

| # | Workflow | Path | aug_mean | cv_r2 | Verdict |
|---|---|---|---:|---:|---|
| 1 | Wf-335-M1 | chemprop+PLL (248-d) | 0.4386 | +0.6814 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | Wf-359-A1 | R20-P3 W₂ (platinai_4cell) | 0.356073 | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | Wf-361-B1 | MAML adapted | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | Wf-361-B2 | Neural ODE dopri5 | 0.4495 | +0.5410 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | Wf-364-A2 | combined RDKit+ODE+MAML (LAST) | **0.4737** | +0.3733 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — **FINAL** |

**Verdict matrix:**
- Structural ceiling axis (cv_r2): CEILING_BREAK (4 of 4 cv_r2 attempts ABOVE PROJECTED +0.05 to +0.15 band)
- Threshold-reaching axis (aug_mean): HONEST_NEGATIVE (best 0.4737 vs 0.6 threshold, gap -0.126)
- Aggregate: HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL

---

## §3 Honest framing per `party-twenty-spirit §3.5`

1. **MEASURED-vs-PROJECTED tag discipline**: every numeric value above traces to a JSON / report path on disk. `augmented_mean_before/after`, `cv_r2`, `gate_threshold`, and `gate_status` are all MEASURED. PROJECTED band (+0.05 to +0.15) cited from `WEAK_TO_STRONG_PLAN.md §3.5`.
2. **"Kill then adapt" GPU directive** (`AGENTS.md §5.1`): zero GPU launches performed. Wf-365-A3 is docs + test + JSON update only.
3. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*`. The §6 paper entry in the verdict doc is a template only; actual paper text requires USER_OK.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline): UNCHANGED. Wf-365-A3 adds 0 NEW pre-flight gates.
5. **Honest-negative framing**: gate status `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` preserved as FINAL state. Structural-ceiling CEILING_BREAK and threshold-reaching HONEST_NEGATIVE both honestly reported.

---

## §4 Standing rules honored

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric traces to JSON / report path on disk |
| §1.2 Standing GPU authorization (5.1) | ✅ | 0 GPU launches; CPU-only docs + test + JSON |
| §1.3 No paper work without authorization (5.3) | ✅ | 0 touches to `paper/*`; §6 entry is template only |
| §1.4 Pre-flight v3 62/62 PASS gate | ✅ | PRESERVED at Wf-335-SYN baseline; 0 NEW gates |
| §1.5 Honest-negative framing (5.2) | ✅ | HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved as FINAL |

---

## §5 Test results — MEASURED

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf365a3_m6_kappa_verdict.py -v

collected 1 item

molmetal/tests/test_wf365a3_m6_kappa_verdict.py::test_wf365a3_m6_kappa_verdict PASSED [100%]

========================= 1 passed, 1 warning in 0.33s =========================
```

(2026-09-29)

---

## §6 Wall + cost

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Read prior workflow docs (Wf-335-M1, Wf-359-A1, Wf-361-B1, Wf-361-B2, Wf-364-A2) | ~3 s |
| Draft `docs/M6_KAPPA_FINAL_VERDICT.md` (~280 LOC, 9 sections) | ~5 min |
| Write `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` (1 NEW test) | ~1 min |
| Update `metrics/by_metric/m6_kappa.json` (wf365a3_final_verdict field) | < 1 s |
| Pytest run (1 NEW test) | 0.33 s |
| JSON parse verify | < 1 s |
| **Total Wf-365-A3 CPU wall** | **~7 min** (within PROJECTED 15-25 min) |

0 GPU launches. CPU-only per `AGENTS.md §5.1` standing GPU grant.

---

## §7 Cross-references

- `docs/M6_KAPPA_FINAL_VERDICT.md` — Wf-365-A3 FINAL verdict doc (NEW)
- `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` — 1 NEW test (NEW)
- `metrics/by_metric/m6_kappa.json` — canonical metric JSON with `wf365a3_final_verdict` field added
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D canonical closeout
- `TODO/completed/333v4_m6_kappa_alt.md` — Wf-333-V4 alternate-recovery bench
- `TODO/completed/335_final_synthesis.md §2` — Wf-335-M1 chemprop+PLL +0.6814 anchor
- `TODO/completed/335m1_chemprop_pll_full.md` — Wf-335-M1 full integration closeout
- `TODO/completed/348b1_w2_r20p3_implementation.md` — Wf-348-B1 W₂ recipe SHIPPED
- `TODO/completed/348b3_hilbert_r2_gate.md` — Wf-348-B3 Hilbert R² gate PRESERVED
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` — Wf-359-A1 verification closeout
- `TODO/completed/361b1_m6_kappa_maml.md` — Wf-361-B1 MAML closeout
- `TODO/completed/361b2_m6_kappa_neural_ode.md` — Wf-361-B2 Neural ODE closeout
- `TODO/completed/364a2_m6_kappa_combined.md` — Wf-364-A2 combined (LAST) closeout
- `molmetal/reports/wf335m1_chemprop_pll/result.json` — Wf-335-M1 MEASURED CV-R² +0.6814
- `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` — Wf-359-A1 aug_mean 0.356073
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` — Wf-361-B2 MEASURED CV-R² +0.5410
- `molmetal/reports/wf364a2_m6_kappa_combined/result.json` — Wf-364-A2 MEASURED aug_mean 0.4737

**Module + script references:**
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2)
- `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` — Wf-335-M1 integration module
- `molmetal/scripts/wf335m1_chemprop_pll_smoke.py` — Wf-335-M1 smoke runner
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` — Wf-361-B1 MAML runner
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 Neural ODE runner
- `molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py` — Wf-364-A2 combined runner

**Standing rules:**
- `AGENTS.md §1.1` — MEASURED-vs-PROJECTED tag discipline
- `AGENTS.md §5.1` — Standing GPU authorization
- `AGENTS.md §5.2` — Honest framing mandate
- `AGENTS.md §5.3` — No paper work without authorization (TODO-84 active)
- `AGENTS.md §5.4` — Tier-2 journal target; no wet-lab planning
- `party-twenty-spirit §3.5` — 0-tolerance honest framing

---

## §8 What Wf-365-A3 does NOT do

1. Does NOT lift the m6_kappa gate (HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL preserved).
2. Does NOT update `metrics/by_metric/m6_kappa.json::value` (0.4719 baseline unchanged).
3. Does NOT update `metrics/by_metric/m6_kappa.json::verdict` ("FAIL" unchanged).
4. Does NOT touch `paper/*` (TODO-84 active; §6 entry is template only).
5. Does NOT launch any GPU work (USER_OK-gated per `mol-metal-gpu-execution §0`).
6. Does NOT add new pre-flight gates (Wf-365-A3 is documentation-only).
7. Does NOT propose Wf-365-A1 Neural ODE work as separate from Wf-361-B2 (Wf-365-A1 not on disk; gap documented in §1 footnote).

---

## §9 Forward-looking

Per Wf-365-A3 verdict doc §5, the highest-priority forward-looking action is:

**Drop m6_kappa as Tier-2 paper §3 main-result metric; keep as complementary heuristic in §6 honest-limitations.** This aligns with:
- Wf-361-B1 §9 option 4 ("drop m6_kappa as a Tier-2 paper metric")
- Wf-335-M2 §3 (m1_auc has multiple CEILING_BREAK anchors)
- AGENTS.md §5.4 (Tier-2 acceptance bar is computational-only; wet-lab validation not required)

USER_OK is required per TODO-84 to draft any `paper/*` edits.

---

**END OF Wf-365-A3 CLOSEOUT** — Owner Hugo Lee + Wf-365-A3 ultracode; cross-ref `party-twenty-spirit §3.5`; verdict doc SHIPPED at `docs/M6_KAPPA_FINAL_VERDICT.md` (9 sections, 5 attempts inventoried); 1 NEW test PASS in 0.33 s; metrics JSON updated with `wf365a3_final_verdict` field; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL state, no more paths to try per Wf-364-A2 §8).