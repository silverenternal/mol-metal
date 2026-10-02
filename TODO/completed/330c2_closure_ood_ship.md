# Wf-330-C2 — Cross-domain OOD closure (3 datasets, Tanimoto<0.4)

**Date**: 2026-09-29
**Workflow**: Wf-330-C2 (ultracode bundle)
**Status**: DRAFT (bench in flight at 21 min; ship on completion)
**Priority**: HIGH (L12 cross-domain OOD with 50 samples/metal sweep)
**Owner**: Hugo Lee + Wf-330-C2 ultracode agent
**Parent**: `TODO/completed/92_wf41_phase2_closure_ood.md` (Wf-41 P2 baseline) + `TODO/completed/330c3_wf44_p2_phase2.md` (Wf-44 P2 10-sample sweep)
**Skill alignment**: `mol-metal-closure-catherine` + `mol-metal-triangle-closure` (Wf-41 P2 + Wf-44 P2 ship surface)

---

## §0 TL;DR

Wf-330-C2 is the **dense 50 samples/metal** cross-domain OOD closure sweep
on PlatinAI 5K + MetalCytoToxDB 5-metal + tmQM d-block with the
Tanimoto<0.4 OOD protocol. It extends C3's 10-sample sweep (5x denser)
and ships **4 NEW pytest contracts** pinning the per-domain reachability
contract, the Tanimoto<0.4 OOD protocol, and the persisted output schema.

| Metric | Value |
|---|---|
| Wall (50 samples/metal × 3 domains, full sweep) | TBD (in flight) |
| Tests shipped | 4 NEW (test_wf330c2_closure_ood.py) + bonus invariants |
| Domains benchmarked | 3 (platinai_5k, metalcytotox_5metal, tmqm_dblock) |
| Tanimoto threshold | 0.40 (Morgan-ECFP4) |
| Method | `closure+ood_filter` (strict L12 OOD subset) |

---

## §1 MEASURED-vs-PROJECTED (per `party-twenty-spirit` §3.5)

### §1.1 MEASURED — Wf-41 P2 baseline (TODO-92)

| Domain | closure_reachability_rate | ood_rate | n_input |
|---|---:|---:|---:|
| platinai_5k | 0.5581 | ~1.0 (OOD-filtered subset) | 350 |
| metalcytotox_5metal | 0.0300 | ~1.0 (OOD-filtered subset) | 250 |
| tmqm_dblock | n/a | 0.0 (train pool) | n/a |
| Tanimoto<0.4 fraction_below_threshold | 0.9978 | n/a | n/a |

### §1.2 MEASURED — Wf-330-C3 10-sample sweep (TODO-330c3, prior)

| Domain | N input | N OOD | closure_ok | **reachability_rate** |
|---|---:|---:|---:|---:|
| platinai_5k | 70 | 44 | 27 | **0.6136** |
| metalcytotox_5metal | 50 | 50 | 0 | **0.0000** |
| tmqm_dblock | 293 | 0 | 0 | **n/a** (n_ood=0) |

### §1.3 MEASURED — Wf-330-C2 50-sample sweep (this run, IN FLIGHT)

**Run date**: 2026-09-29 13:41 UTC (estimated completion ~14:10 UTC)
**Tool**: `molmetal/scripts/bench_closure_ood.py`
**Method**: `closure+ood_filter`
**Tanimoto threshold**: 0.40
**Max depth**: 1
**Max products**: 64
**n_samples_per_metal**: 50

`UPDATE mn | floats` (placeholder, populated when bench completes)

| Domain | N input | N OOD | closure_ok | **reachability_rate** | mean_wall_s |
|---|---:|---:|---:|---:|---:|
| platinai_5k | TBD | TBD | TBD | TBD | TBD |
| metalcytotox_5metal | TBD | TBD | TBD | TBD | TBD |
| tmqm_dblock | TBD | TBD | TBD | TBD (n_ood=0) | TBD |

### §1.4 PROJECTED — per-domain reachability ≥ 0.85 (per skill §8.5.2)

Per `mol-metal-closure-catherine` skill §8.5.2, the projected per-domain
reachability target is ≥ 0.85. PROJECTED values are NOT measured and
must NOT be cited as main results. They are the design intent only.

* **platinai_5k** PROJECTED ≥ 0.85 (MEASURED 0.5581 baseline; +0.05 lift
  per C3 10-sample sweep → 0.6136). Gap to PROJECTED 0.85 is +0.24 pp
  on top of the C3 lift.
* **metalcytotox_5metal** PROJECTED ≥ 0.85 (MEASURED 0.0300 baseline;
  0.0000 in C3 sweep → **HUGE gap**, CPU-only may not reach per skill
  §6 honest framing).
* **tmqm_dblock** n/a (train pool — every sample has max_tanimoto ≥ 0.4
  to itself, so the OOD filter rejects all).

### §1.5 PROJECTED — Tanimoto<0.4 fraction_below_threshold ≥ 0.99

MEASURED baseline 0.9978. PROJECTED ≥ 0.99 — already met by baseline.
This is the easiest honesty gate.

---

## §2 Honest framing per `mol-metal-closure-catherine` skill §6

### §2.1 What we know

* `bench_closure_ood.py` exists (~1100 LOC, Wf-41 P2 ship) and runs
  end-to-end on all 3 documented domains.
* CLI exposes `--domains`, `--method {closure,closure+ood_filter}`,
  `--tanimoto-threshold`, `--output-dir`, `--n-samples`,
  `--max-depth`, `--max-products`, `--seeds`.
* Tanimoto<0.4 OOD protocol uses Morgan-ECFP4 fingerprints with the
  Pocket2Mol / TargetDiff convention.
* MEASURED PlatinAI OOD reachability ≈ 0.5581 (Wf-41 P2); MEASURED
  MetalCytoTox OOD ≈ 0.0300 (Wf-41 P2).
* C3 10-sample sweep: PlatinAI 0.6136 (BETTER than baseline),
  MetalCytoTox 0.0000 (WORSE — within sampling noise).

### §2.2 What is **NOT** known / honest negatives

* **metalcytotox_5metal reachability is the gap** — at 0.0300 baseline
  → 0.85 PROJECTED is +0.82 pp. Per skill §6, CPU-only may not reach.
* **tmqm_dblock is the train pool** — by construction, n_ood=0 always;
  cannot meaningfully report reachability on it.
* **R² on real pIC50** is NOT a closure-OOD metric — covered by
  `mol-metal-hilbert-astsr-r2` + `mol-metal-hilbert-astsr-e2e`
  (TODO-93), orthogonal axis.
* **GPU portion** of the per-domain reachability lift is deferred to
  TODO-93 / Wf-37 P3 / R20-P2 (Hilbert-ASTSR) per `party-twenty-spirit`
  §3.5 honest-negative close-out.

### §2.3 Recovery path for the MetalCytoTox gap

If the 50-sample sweep confirms MetalCytoTox ≤ 0.10 reachability:

1. **Honest-negative declaration** — `HONEST_NEGATIVE_METALCYTOX_OOD`
   in `TODO/INDEX.md` §22.
2. **GPU portion deferred** to TODO-93 (R20-P2 Hilbert-ASTSR R² lift).
3. **Paper §6 honest limitations** — cite MEASURED baseline 0.0300 +
   per-domain gap + L3 honest-neg gate per skill §6.
4. **No fabricated "achievable"** number — every PROJECTED value
   must carry a wall budget + a recovery path.

---

## §3 Tests shipped (4 NEW + bonus)

`molmetal/tests/test_wf330c2_closure_ood.py`:

| # | Test | Purpose |
|---|---|---|
| 1 | `test_bench_runs_on_three_domains` | Verifies `run_closure_ood_benchmark` runs on all 3 domains + returns `per_domain` entry for each. |
| 2 | `test_per_domain_reachability_contract` | Per-domain `closure_reachability_rate ∈ [0,1]`, `n_input` / `n_ood` audit tuple. Pins contract, NOT 0.85 PROJECTED value (per honest framing). |
| 3 | `test_tanimoto_lt_0_4_ood_protocol` | Method = `closure+ood_filter`, threshold = 0.4, OOD filter does not inflate n_input, Markdown references Tanimoto. |
| 4 | `test_output_schema` | JSON + per_domain.csv + Markdown carry documented schema (`schema_version=1`, `tool`, `pilot_id`, `method`, `aggregate` keys). |

Bonus invariants:

| # | Test | Purpose |
|---|---|---|
| 5 | `test_domain_specs_has_three_domains` | DOMAIN_SPECS has all 3 Wf-330-C2 datasets. |
| 6 | `test_three_domain_metals_are_dblock_subset` | Union of PLATINAI / METALCYTO / TMQM metals covers Pt, Ru, Au, Re, Fe. |
| 7 | `test_three_domain_smoke_completes_under_budget` | Smoke completes < 120 s wall. |

All tests are CPU-only, deterministic, and complete in < 180 s combined.

---

## §4 Commands

### §4.1 Run the bench (CPU-only)

```
PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/.venv-flowmol311/bin/python \\
    molmetal/scripts/bench_closure_ood.py \\
    --domains platinai_5k metalcytotox_5metal tmqm_dblock \\
    --method closure+ood_filter \\
    --tanimoto-threshold 0.4 \\
    --output-dir molmetal/reports/wf330_closure_ood/ \\
    --n-samples 50
```

### §4.2 Run the tests

```
PYTHONPATH=/home/hugo/codes/mol-metal \\
    /home/hugo/.venv-flowmol311/bin/python -m pytest -v \\
    molmetal/tests/test_wf330c2_closure_ood.py --tb=short
```

---

## §5 Outputs

* `molmetal/reports/wf330_closure_ood/ood_closure.json` — full per-domain
  + per-metal + per-sample payload (schema_version=1).
* `molmetal/reports/wf330_closure_ood/per_domain.csv` — one row per
  (domain, metal) with closure_rate + ood_rate + n_ood + mean_wall.
* `molmetal/reports/wf330_closure_ood/ood_closure.md` — human-readable
  Markdown summary with per-domain table.
* `molmetal/tests/test_wf330c2_closure_ood.py` — 4 NEW pytest contracts.

---

## §6 Ship status

* [x] Bench in flight (50 samples/metal × 3 domains)
* [x] 4 NEW pytest contracts drafted
* [ ] Bench outputs persisted to `molmetal/reports/wf330_closure_ood/`
* [ ] Per-domain reachability MEASURED values populated in §1.3
* [ ] Tests PASS (target 4 / 4 + 3 bonus invariants)
* [ ] MEASURED-vs-PROJECTED honest framing for the MetalCytoTox gap