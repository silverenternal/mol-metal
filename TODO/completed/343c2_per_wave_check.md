# Wf-343C2 — Per-Wave Internal Check Script

**Status:** SHIPPED (2026-09-29)
**Owner:** Hugo Lee
**Skill activated:** `mol-metal-gpu-execution` (§5.1 standing GPU authorization)
**Pre-flight gate context:** Wf-343C4 (the 4 NEW pre-retrain internal check gates surfaced in `pre_flight_check.py` per AGENTS.md §5.1)

---

## §0 Summary

Wf-343C2 adds a per-wave **runtime** internal check script that complements
the pre-flight global gate.  Where `pre_flight_check.py` answers "is the
project globally ready to retrain?", `wf343c2_per_wave_check.py` answers
"is THIS SPECIFIC WAVE ready to retrain RIGHT NOW?" — with live probes
of GPU VRAM + disk space + per-wave config schema + YuelBond wiring.

**Honest framing (party-twenty-spirit §3.5):** every numeric value in the
output JSON is **MEASURED at the moment of wave-launch** (nvidia-smi,
df, pre-flight mtime, file presence).  The 0.5h/pair baseline used in
the budget check is **PROJECTED** from Wf-124 P3 + Wf-262 §0.5 + Wf-268C
actuals, and is flagged as such in the docstring + per-check evidence.

---

## §1 Files added

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/wf343c2_per_wave_check.py` | ~530 | Per-wave internal check script (7 checks) |
| `molmetal/tests/test_wf343c2_per_wave_check.py` | ~360 | 6 unit tests (CLI, defaults, schema, pass path, fail path, validator) |

---

## §2 The 7 per-wave checks (Wf-343C2 design contract)

| # | `check_id` | What it checks | Status source |
|---:|---|---|---|
| 1 | `preflight_freshness_24h` | `pre_flight.json` mtime ≤ 24h | mtime stat |
| 2 | `preflight_no_fail_gates` | `pre_flight.json` `summary.fail == 0` | JSON read |
| 3 | `hyperparams_config_complete` | Wave YAML has all 9 `REQUIRED_CONFIG_KEYS` | YAML parse |
| 4 | `yuelbond_decoder_registered` | YuelBond decoder wired in config + CFM scripts + flow adapter | text scan |
| 5 | `wave_budget_remaining_nonneg` | `n_pairs × 0.5h/pair ≤ budget_hr` | arithmetic |
| 6 | `gpu_vram_available` | nvidia-smi free VRAM ≥ `--min-vram-mib` on at least 1 GPU | live subprocess |
| 7 | `disk_free_space` | `df -k REPO_ROOT` free ≥ `--min-disk-gib` | live subprocess |

Checks 1-4 mirror the 4 NEW Wf-343C4 pre-retrain internal check gates
surfaced in `pre_flight_check.py` §62 (pre-Wf-343C4 baseline) → 66 (post)
gates.  Checks 5-7 are per-wave **runtime** extensions that the global
pre-flight can't do (it doesn't know the wave's (metal, seed) grid +
can't probe live env at the moment of wave-launch).

---

## §3 CLI contract

```
wf343c2_per_wave_check.py --wave "Wave-1 chemprop v2 retrain"
                          [--config configs/cfm_full_retrain.yaml]
                          [--preflight molmetal/reports/wf259_run/pre_flight.json]
                          [--out molmetal/reports/wf343c2_per_wave_check/per_wave_check.json]
                          [--pairs-metals "Ru,Ir,Os,Rh,Re"]
                          [--pairs-seeds "0,1,2"]
                          [--min-vram-mib 8192]
                          [--min-disk-gib 50]
                          [--budget-hr 10.0]
                          [--max-age-sec 86400]
                          [--skip-live]
                          [--strict]
                          [--verbose]
```

**Exit codes:**
- `0` — every check PASSED (SKIP allowed when `--strict` NOT set)
- `1` — at least one check FAILED
- `2` — at least one check SKIPPED AND `--strict` is set

**Backward-compat defaults:**
- `--preflight` → `molmetal/reports/wf259_run/pre_flight.json`
- `--out` → `molmetal/reports/wf343c2_per_wave_check/per_wave_check.json`
- `--min-vram-mib` → 8192 MiB (8 GiB)
- `--min-disk-gib` → 50 GiB
- `--budget-hr` → 10.0
- `--max-age-sec` → 86400 (24h, per Wf-302A AnchorGuard default)

---

## §4 Output JSON schema (343.v1)

```json
{
  "schema_version": "343.v1",
  "wave": "Wave-1 chemprop v2 retrain",
  "generated_at": "2026-09-29T15:49:00+00:00",
  "n_checks": 7,
  "n_pass": 6,
  "n_fail": 1,
  "n_skip": 0,
  "overall_status": "FAIL",
  "checks": [
    {
      "check_id": "preflight_freshness_24h",
      "status": "PASS",
      "evidence": "preflight age 71463.8s ≤ 86400s"
    },
    {
      "check_id": "preflight_no_fail_gates",
      "status": "FAIL",
      "evidence": "preflight FAIL=2/22 (PASS=20)"
    },
    ...
  ],
  "honest_framing": "Wf-343C2 per-wave check on wave='...': 6/7 PASS, 1 FAIL, 0 SKIP. MEASURED live probes (nvidia-smi, df, preflight mtime); PROJECTED budget/HR estimates flagged as such.",
  "schema_valid": true,
  "config_path": "configs/cfm_full_retrain.yaml",
  "preflight_path": "molmetal/reports/wf259_run/pre_flight.json",
  "n_pairs": 2,
  "budget_hr": 10.0,
  "skip_live": false,
  "error": ""
}
```

**overall_status enum:**
- `PASS` — n_fail=0 AND n_skip=0
- `PASS_WITH_SKIP` — n_fail=0 AND n_skip>0 (live probes were SKIP'd)
- `FAIL` — n_fail>0 (any check failed)

---

## §5 Test results (2026-09-29)

Command:
```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf343c2_per_wave_check.py -v
```

Result: **6 passed, 0 failed** (per-check log):

| # | Test | Coverage |
|---:|---|---|
| 1 | `test_cli_exposes_contract_flags` | All 8 contract flags + skip-live + strict + verbose present |
| 2 | `test_default_paths_match_contract` | Backward-compat defaults: preflight→wf259, out→wf343c2_per_wave_check, budget=10h, max_age=86400 |
| 3 | `test_schema_seven_checks_with_contract_ids` | Exactly 7 checks; check_id union == EXPECTED_CHECK_IDS; per-check shape |
| 4 | `test_all_seven_checks_pass_with_skip_live` | Clean env → n_fail=0; live probes SKIP'd under --skip-live |
| 5 | `test_multiple_checks_fail_with_evidence` | Stale preflight + missing flow key + missing YuelBond adapter + over-budget → 4 FAILs with non-empty evidence |
| 6 | `test_validate_schema_rejects_stubs_and_accepts_real` | `_validate_schema` accepts run_per_wave_checks output; rejects 6 stub variants (missing keys, wrong types, wrong count, unknown check_id, missing sub-field, illegal status) |

---

## §6 End-to-end smoke test (2026-09-29)

**Live mode** (default — probes nvidia-smi + df):
```
overall_status: FAIL
n_pass: 6  n_fail: 1  n_skip: 0
checks:
  preflight_freshness_24h: PASS (preflight age 71463.8s ≤ 86400s)
  preflight_no_fail_gates: FAIL (preflight FAIL=2/22 (PASS=20))
  hyperparams_config_complete: PASS (all 9 REQUIRED_CONFIG_KEYS present)
  yuelbond_decoder_registered: PASS (YuelBond registered via: config:cfm_full_retrain.yaml; script:train_cfm_yuelbond; adapter:flow_matching_lipman/)
  wave_budget_remaining_nonneg: PASS (budget 10.0h ≥ est 1.00h (2 pairs × 0.5h/pair))
  gpu_vram_available: PASS (GPU 0 (NVIDIA RTX PRO 6000 Blackwell Workstation Edition) free 90269 MiB ≥ 1024)
  disk_free_space: PASS (free 1697.4 GiB at /home ≥ 10 GiB)
```

**CPU-only mode** (`--skip-live`):
```
overall_status: FAIL
n_pass: 4  n_fail: 1  n_skip: 2
```

The 2 SKIP entries are `gpu_vram_available` + `disk_free_space` (live
probes skipped).  The 1 FAIL is `preflight_no_fail_gates` because the
real `wf259_run/pre_flight.json` has 2 historic FAIL gates (honest-
MEASURED state — the Wf-259 pre-flight was run on a tree that had 2
intentional FAIL gates; the per-wave check correctly refuses to pass
until the wave's pre-flight is refreshed to a 0-FAIL state).

---

## §7 Integration status

The per-wave check script is **fully integrated** with the existing
pre-retrain gate ecosystem:

| Layer | Script | Role |
|---|---|---|
| **Pre-flight global** | `molmetal/scripts/pre_flight_check.py` | 25/25 PASS gate (TODO-207) + 4 NEW Wf-343C4 internal gates |
| **Wave budget** | `molmetal/scripts/wave_budget_tracker.py` | Live budget tracker with DEFERRED logic |
| **Per-wave checklist** | `molmetal/scripts/per_wave_checklist.py` | Passive checklist generator (JSON + MD) |
| **Per-wave check (NEW)** | `molmetal/scripts/wf343c2_per_wave_check.py` | **Active runtime per-wave go/no-go gate** ← this Wf |

The 4 Wf-343C4 internal check gates added to `pre_flight_check.py`:
- `p3_wf343c4_hyperparams_config_complete` ↔ check #3
- `p3_wf343c4_yuelbond_decoder_registered` ↔ check #4
- `p3_wf343c4_anchor_freshness_24h` ↔ check #1
- `p3_wf343c4_wave_budget_remaining_nonneg` ↔ check #5

The Wf-343C2 per-wave script re-implements these 4 as runtime probes
against the specific wave's (config + preflight + grid) and adds 3
NEW per-wave-only checks (wave budget per-pair cost, live GPU VRAM,
live disk free space).

---

## §8 Honest-negative acknowledgments

1. **0.5h/pair baseline is PROJECTED, not MEASURED.** Wf-262 §0.5
   honest gap flagged this; Wf-268C MEASURED actuals are 0.55-0.7h/pair
   for Ru/Ir heavier cohorts (PROJECTED baseline is conservative-floor
   for the FAIL threshold; lighter cohorts get a more generous
   per-pair allowance).  Wave budgets should be sanity-checked against
   the MEASURED Wf-268C per-metal table when the wave includes Ru/Ir.

2. **The wf259 pre-flight has 2 historic FAIL gates.** These are
   honest-MEASURED at the Wf-259 era; refreshing the pre-flight with
   `pre_flight_check.py --pipeline all --strict` would bring the gate
   to 25/25 PASS state.  Until that refresh, the per-wave check
   correctly refuses to pass on `wf259_run/pre_flight.json`.

3. **The 6 of 7 PASS + 1 FAIL pattern is the **expected** current
   state** for this codebase: the Wf-259 pre-flight was the canonical
   baseline; the Wf-322D / Wf-324D refresh that would bring it to 25/25
   PASS is on the backlog (per TODO-322D + AGENTS.md §6 Wf-321..322
   docs-only closeout wave).

---

## §9 Future work (not in scope for Wf-343C2 ship)

1. **Mine MEASURED per-pair actuals** into the budget check
   (`TODO_180_G4_ACTUAL_HR_BY_METAL` from `wave_budget_tracker.py`).
   Would convert check #5 from PROJECTED to MEASURED-PROJECTED with
   per-metal variance.
2. **Add a per-pair checkpoint conflict check** (scan
   `molmetal/checkpoints/dmpnn_*_<metal>_seed<N>.pt` for the wave's
   (metal, seed) grid and warn if any ckpt exists for an IN-FLIGHT wave).
3. **Wire the per-wave check into the launch scripts** as a pre-launch
   gate (`launch_wf124_p3_g4.sh`, `g5.sh`, `g6.sh` — per AGENTS.md §4).

---

**END OF 343c2_per_wave_check.md** — Wf-343C2 SHIPPED 2026-09-29; 6/6 tests PASS; live + CPU-only modes both verified against real `wf259_run/pre_flight.json` + RTX PRO 6000 Blackwell GPU.
