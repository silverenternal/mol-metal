# Wf-345-C2 — Wf-143 chemotype dock sweep stale anchor refresh

**Wf:** Wf-345 (wx63q8fwe ultracode residual)
**Date:** 2026-09-29
**Audit source:** `molmetal/reports/wf345a3_anchor_audit/audit.json`

---

## §1 Identification of the Wf-143 stale anchor

Per Wf-345-A3 audit (`audit.json`), 11 paper-grade anchors were inspected under glob
patterns `{m14_anchor,wf269c_report,m17_drugood_reeval,sweep_summary}.json`. Wf-143 P4
chemotype dock sweep summary is one of the **4 stale anchors** (age > 24h) flagged for
refresh:

| # | path | age_hr (before) |
|---|---|---:|
| 1 | `wf121_cisplatin_real_vina/sweep_summary.json` | 60.6599 |
| **2** | **`wf143_p4_chemotype_dock/sweep_summary.json`** | **57.731** |
| 3 | `wf170_permetal_singletask/sweep_summary.json` | 29.0894 |
| 4 | `wf67_p4_3_100pocket/sweep_summary.json` | 70.8641 |

This refresh covers anchor #2 — `molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json`.
It had been stale since **2026-09-27T06:16:45+08:00** (= 2026-09-26T22:16:45 UTC, **57.6 hr
old** at the time of audit), exceeding the `read_anchor_json.py` `max_age_sec = 86400` (24 hr)
freshness gate.

---

## §2 Source script

Located via `grep -rln "wf143" molmetal/scripts/`:

```
molmetal/scripts/bench_cisplatin_chemotype_vdw.py
```

This is the canonical emitter for `molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json`.
Per the script docstring, Wf-143 P4 runs the canonical cisplatin docking sweep on the 10
Wf-30 pockets × 3 seeds and records the real Vina 1.2.7 kcal/mol with **chemotype-tuned VDW**
(`Pt` atype PRIMARY; falls back to `Mg` if Vina rejects Pt). The probe
`_probe_vina_accepts_chemotype_atype` decides at runtime.

Default invocation per script docstring (line 58-62):

```bash
source env.sh
python molmetal/scripts/bench_cisplatin_chemotype_vdw.py \
    --exhaustiveness 8 --n-poses 5 --seeds 0 1 2 \
    --out-dir molmetal/reports/wf143_p4_chemotype_dock
```

The pre-flight gate that consumes this anchor is `p3_wf143_chemotype_dock_anchor` (one of the
62 pre-flight gates per Wf-330 ULTRACODE); see `molmetal/scripts/pre_flight_check.py:1354`.

---

## §3 Refresh run

Command executed:

```bash
cd /home/hugo/codes/mol-metal
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/bench_cisplatin_chemotype_vdw.py \
    --exhaustiveness 8 --n-poses 5 --seeds 0 1 2 \
    --out-dir molmetal/reports/wf143_p4_chemotype_dock
```

(Ran with the shared `/home/hugo/codes/CadButEaas/.venv-vlm` Python because `vina 1.2.7` +
`meeko 0.8.0` are installed there. `source env.sh` followed by `uv run` failed because
`uv 0.11.8` ignored `UV_PROJECT_ENVIRONMENT` under this project's `uv.lock`; using the
direct venv Python bypassed that.)

Wall: **~26 s** (CPU-only; 10 pockets × 3 seeds × 5 poses = 150 Vina runs).

Sweep log headline: `Sweep complete: 10/10 pockets docked; summary at
molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json` — all 10 pockets docked
without exception.

For every pocket the chemotype rewrite path was exercised. Vina 1.2.7 rejected `Pt` atype in
all 10 cases (`PDBQT parsing error: Atom type Pt is not a valid AutoDock type`) and the probe
rewrote `Pt → Mg` via `_replace_chemotype_atype_with_fallback`, falling back to the generic
VDW surrogate. **Result: 10/10 pockets report `chemotype_atype_accepted: False`** — identical
to the prior 2026-09-26 anchor (the chemotype rewrite contract is to transparently fall back
when Vina rejects Pt, and the fallback path is bit-deterministic for a fixed seed + receptor +
ligand structure).

Resulting `sweep_summary.json` metadata:

```
_generated_utc            = 2026-09-29T08:56:24.369325+00:00  (was 2026-09-26T22:16:45.413830+00:00)
_hardware                 = CPU (vina Python binding 1.2.7)
_exhaustiveness           = 8
_n_poses                  = 5
_seeds                    = [0, 1, 2]
_n_pockets_docked         = 10/10
```

---

## §4 Drift verification (old vs new)

Per the brief's honest-framing rule "drift > 1e-3 → HONEST-NEGATIVE":

| metric | OLD (2026-09-26) | NEW (2026-09-29) | drift (abs) |
|---|---:|---:|---:|
| `_aggregate_chemotype_mean_kcal_mol` | -3.1552 | -3.1552 | 0.0 |
| `_baseline_10pocket_mean_kcal_mol` | -3.1552 | -3.1552 | 0.0 |
| `_delta_vs_wf121_baseline_kcal_mol` | 0.0 | 0.0 | 0.0 |
| 1fmc chemotype_kcal_per_seed | [-3.131, -3.131, -3.138] | [-3.131, -3.131, -3.138] | 0.0 / 0.0 / 0.0 |
| 1g9v chemotype_kcal_per_seed | [-3.057, -3.039, -3.040] | [-3.057, -3.039, -3.040] | 0.0 / 0.0 / 0.0 |
| 1h36 chemotype_kcal_per_seed | [-3.597, -3.575, -3.589] | [-3.597, -3.575, -3.589] | 0.0 / 0.0 / 0.0 |
| 2hnx chemotype_kcal_per_seed | [-3.387, -3.376, -3.376] | [-3.387, -3.376, -3.376] | 0.0 / 0.0 / 0.0 |
| 3eml chemotype_kcal_per_seed | [-2.826, -2.827, -2.827] | [-2.826, -2.827, -2.827] | 0.0 / 0.0 / 0.0 |
| 3ny8 chemotype_kcal_per_seed | [-3.721, -3.716, -3.734] | [-3.721, -3.716, -3.734] | 0.0 / 0.0 / 0.0 |
| 4aaw chemotype_kcal_per_seed | [-3.287, -3.275, -3.283] | [-3.287, -3.275, -3.283] | 0.0 / 0.0 / 0.0 |
| 4rlu chemotype_kcal_per_seed | [-2.995, -2.981, -2.974] | [-2.995, -2.981, -2.974] | 0.0 / 0.0 / 0.0 |
| 4rn0 chemotype_kcal_per_seed | [-2.548, -2.547, -2.549] | [-2.548, -2.547, -2.549] | 0.0 / 0.0 / 0.0 |
| 830c chemotype_kcal_per_seed | [-3.033, -3.067, -3.030] | [-3.033, -3.067, -3.030] | 0.0 / 0.0 / 0.0 |

**max_drift = 0.0 kcal/mol** (bit-identical per-seed + per-pocket). drift_status =
**PASS** (well below 1e-3 threshold).

Interpretation: the sweep is **fully reproducible** — Vina 1.2.7 with fixed seed + receptor +
ligand structure + fixed chemotype_atype_accepted=False rewrite path produces bit-identical
kcal/mol values across re-runs. The 57.6h staleness did not reflect any drift; it reflected
purely that nobody re-emitted the sweep_summary.json within the 24 hr freshness window.

This also re-confirms the Wf-143 P4 HONEST-NEGATIVE finding (carried forward from §22.23 / Wf-333
+ §22.27 / Wf-335-SYN): **Vina 1.2.7 cannot natively accept `Pt` atype**, so the chemotype VDW
contract transparently degrades to the Mg-fallback path. The 10-pocket mean of -3.1552 kcal/mol
is **identical** to the Wf-121 P3 generic-Mg baseline, which is the honest finding — the
"chemotype VDW" mode has no effect under Vina 1.2.7 because the rewrite collapses to the same
surrogate. Vina acceptance of `Pt` is BLOCKED on Vina upstream (no per-receptor AD4 parameter
file is in scope).

---

## §5 Anchor freshness post-refresh

```
now_utc           = 2026-09-29T08:56:52Z
modify_utc        = 2026-09-29T08:56:24.369682Z
age_sec           = ~28 s
age_hr            = 0.0078 hr  (well under 24h gate)
max_age_sec gate  = 86400 (24h)
gate_status       = PASS (FRESH, NOT STALE)
```

Pre-flight gate `p3_wf143_chemotype_dock_anchor` will now PASS for the next 24h wall-clock.

---

## §6 Honest framing per party-twenty-spirit §3.5

| Item | Value | Tag |
|---|---|---|
| 57.6h stale | observed | **MEASURED** (audit.json timestamp diff) |
| Tests | 0 NEW (no code change) | **MEASURED** |
| Wall | 26 s (10 pockets × 3 seeds) | **MEASURED** |
| Drift (per-seed + per-pocket) | 0.0 kcal/mol | **MEASURED** (bit-identical) |
| Aggregate -3.1552 kcal/mol (10-pocket chemotype mean) | matches OLD | **MEASURED** |
| chemotype_atype_accepted across all 10 pockets | False (Mg fallback) | **MEASURED** |
| Vina 1.2.7 Pt acceptance | BLOCKED on Vina upstream | **HONEST_NEGATIVE** |
| Gate `p3_wf143_chemotype_dock_anchor` next 24h | FRESH | **PROJECTED** |

---

## §7 Deliverables + next steps

**Deliverables (this doc):**
- `molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json` — refreshed, age=28s.
- `molmetal/reports/wf143_p4_chemotype_dock/{1fmc,1g9v,1h36,2hnx,3eml,3ny8,4aaw,4rlu,4rn0,830c}/per_seed_metrics.json` — refreshed.
- `molmetal/reports/wf143_p4_chemotype_dock/synthesis.md` — refreshed.
- `TODO/completed/345c2_wf143_anchor_refresh.md` — this closeout.

**Next steps:**
- Anchor #1 (wf121_cisplatin_real_vina) refresh queued separately.
- Anchor #3 (wf170_permetal_singletask) refresh queued separately.
- Anchor #4 (wf67_p4_3_100pocket) already refreshed per `TODO/completed/345c4_4th_anchor_refresh.md`.

---

## §10 References

- Wf-143 P4 sweep script: `molmetal/scripts/bench_cisplatin_chemotype_vdw.py` (625 LOC).
- Wf-121 P3 baseline: `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json`.
- Anchor gate: `p3_wf143_chemotype_dock_anchor` in `molmetal/scripts/pre_flight_check.py:1354`.
- Wf-345-A3 audit: `molmetal/reports/wf345a3_anchor_audit/audit.json`.
- Honest-negative propagation: AGENTS.md §5.2 + §22.23.5 + §22.27.5.

**END OF Wf-345-C2**