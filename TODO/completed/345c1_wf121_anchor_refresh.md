# Wf-345-C1 — Wf-121 cisplatin Vina anchor refresh

**Generated:** 2026-09-29 16:56 UTC (wall 110.83+ s)
**Workflow:** Wf-345 ultracode residual closeout wave — anchor-refresh sub-flow C1
**Anchor file:** `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json`
**Sweep script:** `molmetal/scripts/bench_cisplatin_real_vina.py` (1,287 LOC)
**Predecessor anchor mtime:** 2026-09-26T19:21:01 UTC (61.6 hr old, exceeds 24h `max_age_sec`)

---

## §1 Motivation

Per Wf-345-A3 anchor audit (cf. `wx63q8fwe` ultracode review),
`molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` is **60.6 h** old —
exceeding the `max_age_sec=86400` (24 h) contract from Wf-302A AnchorGuard. The
file is consumed by:

1. `molmetal/scripts/bench_cisplatin_exhaustiveness_lift.py` (reads
   `sweep_summary.json` + per-pocket `per_seed_metrics.json` as baseline).
2. `molmetal/scripts/bench_cisplatin_chemotype_vdw.py` (reads
   `_delta_vs_wf121_baseline_kcal_mol`).
3. `molmetal/scripts/pre_flight_check.py` (per-pipeline precondition checks).

All three paths report STALE on the 24-h boundary. Wf-345-C1 refreshes the
anchor in-place so downstream consumers can resume from `PASS`.

## §2 Sweep invocation

```bash
cd /home/hugo/codes/mol-metal
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/bench_cisplatin_real_vina.py \
    --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 4rn0 1fmc 1g9v 2hnx \
    --seeds 0 1 2 --exhaustiveness 8 \
    --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf121_cisplatin_real_vina
```

Environment probe:

* `vina` Python binding 1.2.7 (imported successfully).
* `meeko` 0.8.0 (imported successfully).
* All 10 Wf-30 staged `receptor.pdbqt` files present at
  `molmetal/reports/wf30_pocket_prep/<pocket>/receptor.pdbqt`.
* CPU-only sweep (Vina Python binding + local RDKit); no GPU launch.

## §3 Results

### §3.1 Wall time

| Phase | Wall (s) |
|---|---:|
| Script start (2026-09-29T16:56:06.048Z) → first dock log | ~3 |
| 10 pockets × 3 seeds × ~1 dock/run (warm + dock) | ~7 |
| Script end (2026-09-29T16:56:16.929Z) | — |
| **Total sweep dock wall** | **10.88** |

`PROJECTED 10-20 min` from the Wf-345-C1 brief vastly over-estimated — real
sweep completed in **~11 s** because Vina's Python binding on 9-atom
cisplatin is trivial (per-pocket CPU seconds, not minutes).

### §3.2 New anchor freshness

```
new mtime: 2026-09-29T08:56:16Z
new age_hr: 0.0024  (PASS — well below 24h max_age_sec=86400)
```

### §3.3 Per-pocket drift (old vs new kcal/mol)

| pocket | seed0 Δ | seed1 Δ | seed2 Δ | mean Δ | old status | new status |
|---|---:|---:|---:|---:|---|---|
| 1fmc | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 1g9v | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 1h36 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 2hnx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 3eml | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 3ny8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 4aaw | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 4rlu | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 4rn0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |
| 830c | 0.000000 | 0.000000 | 0.000000 | 0.000000 | MEASURED | MEASURED_GENERIC_ATOM |

```
MAX drift: 0.000000 kcal/mol
AVG drift: 0.000000 kcal/mol
PASS drift < 1e-3: True (NO_DRIFT)
```

**Verdict:** All 30 (pocket, seed) kcal/mol values are **bit-identical** to
the predecessor anchor. Drift = 0.000000 < 1e-3 ⇒ NO HONEST-NEGATIVE
measurement-drift flag.

### §3.4 Why drift is zero (deterministic Vina)

Both the old (Wf-121 P3 ship) and new runs:

1. Use the same 3 seeds `0, 1, 2` (uint32 → int31 conversion is fixed).
2. Run the same ligand `[Pt](N)(N)(Cl)Cl` with the same meeko-emitted
   cisplatin.pdbqt.
3. Rewrite the `M` atype → `Mg` for the Pt atom (the Wf-121 P1 SMARTS rule
   that pre-dates Wf-143 P3).
4. Dock against the same Wf-30 staged `receptor.pdbqt` files.

Hence the per-seed kcal/mol is **fully reproducible** at the kcal/mol
truncation that Vina emits (3 decimal places). Status field changed from
`MEASURED` → `MEASURED_GENERIC_ATOM` because the new default
`--chemotype-vdw=True` triggers the Wf-143 P3 probe path that detects Vina's
strict rejection of the chemotype atype `Pt` and falls back to `Mg`. The
underlying atype-on-disk is the same (`Mg`) in both runs, so kcal/mol is
identical.

## §4 Anchor status after refresh

| Property | Pre-Wf-345-C1 | Post-Wf-345-C1 |
|---|---|---|
| age_hr | 60.6 | 0.0024 |
| max_age_sec | 86400 | 86400 |
| Stale flag | STALE | FRESH |
| AnchorGuard verdict | FAIL | PASS |
| n_pockets | 10/10 | 10/10 |
| n_seeds | 3 | 3 |
| kcal/mol reproducibility | — | bit-identical (0.0 drift) |
| Drift vs predecessor | n/a | NO_DRIFT |
| _generated_utc | 2026-09-26T19:21:01.409649+00:00 | 2026-09-29T08:56:16.929392+00:00 |

**anchor_status: FRESH_PASS** (consumers no longer report STALE on the 24-h
boundary).

## §5 Honest framing (party-twenty-spirit §3.5)

* 60.6h stale ⇒ MEASURED (per Wf-345-A3 audit) ⇒ resolved to 0.0024h fresh
  ⇒ MEASURED. Anchor is now within `max_age_sec` window.
* 0.0 kcal/mol drift ⇒ MEASURED (deterministic Vina reproducibility). The
  script's atype rewrite path is byte-deterministic; no source of randomness
  between the two runs.
* Sweep wall 10.88 s (vs PROJECTED 10-20 min) ⇒ PROJECTED vastly
  over-estimated because Vina on a 9-atom ligand is CPU-second-cheap.
  Recorded as MEASURED to avoid future PROJECTED-style optimistic briefs.
* Status field delta (`MEASURED` → `MEASURED_GENERIC_ATOM`) is a
  **cosmetic-only** change caused by the Wf-143 P3 default-flag refactor —
  not a measurement change. Underlying kcal/mol is identical because the
  atype-on-disk is `Mg` in both runs. This is documented but does not
  trigger an honest-negative flag.

## §6 Deliverables

* `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` — refreshed
  anchor (now 0.0024h old, FRESH_PASS).
* `molmetal/reports/wf121_cisplatin_real_vina/<pocket>/per_seed_metrics.json`
  — 10 per-pocket per-seed JSON files re-written with refreshed
  `_generated_utc` (identical kcal/mol values).
* `molmetal/reports/wf121_cisplatin_real_vina/synthesis.md` — re-written
  synthesis (kcal/mol table unchanged).
* `/tmp/wf121_sweep_summary_old.json` — backup of the 60.6h-old predecessor
  for drift comparison (this doc).

## §7 Tests

* **0 NEW tests** (anchor refresh only; no new functionality).
* **0 NEW scripts**.
* Drift is the verification: 30/30 (pocket, seed) kcal/mol values match
  bit-identical to the 60.6h-old predecessor.

## §8 References

* Wf-302A AnchorGuard contract (`max_age_sec=86400`,
  `required_version` check): `molmetal/scripts/read_anchor_json.py`.
* Wf-121 P3 sweep: `molmetal/scripts/bench_cisplatin_real_vina.py`.
* Wf-143 P3 chemotype VDW atype rewrite + probe fallback path: same script,
  `_rewrite_metal_atype_for_vina` + `_probe_vina_accepts_chemotype_atype`.
* Wf-345-A3 anchor audit (predecessor): `TODO/completed/345a3_*.md` (TODO if
  present; otherwise implicit via `wx63q8fwe` review).
* Wf-302A honest-negative drift threshold: `1e-3` per pocket.

## §9 Sign-off

* sweep_wall_s (MEASURED): **10.88**
* new_age_hr (MEASURED): **0.0024**
* drift_status (MEASURED): **NO_DRIFT** (0.0 kcal/mol across 30 pairs)
* anchor_status (MEASURED): **FRESH_PASS** (< 24h max_age_sec)
* script_path: `molmetal/scripts/bench_cisplatin_real_vina.py`
* forward consumers (`bench_cisplatin_exhaustiveness_lift`,
  `bench_cisplatin_chemotype_vdw`, `pre_flight_check`): PASS
* GPU launches performed: **0** (CPU-only sweep)
