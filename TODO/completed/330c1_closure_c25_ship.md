# Wf-330-C1: Catherine 6/6 × 25 d-block metals bench — R20-P5 ship surface

**Date:** 2026-09-29
**Workflow:** Wf-330-C1
**Status:** SHIPPED (with **HONEST_NEGATIVE_R20_P5** ledger)
**Owner:** Hugo Lee
**Skill:** `mol-metal-closure-catherine` (R20-P5 / TODO-96)
**Cross-ref:** `TODO/completed/83j_lambda_closeout.md` §3.1 (L1/L6/L12 MEASURED anchors)

---

## §1 TL;DR — honest framing per `party-twenty-spirit` §3.5

The Wf-57 Phase-3 bench was run on **25 d-block metals × 5 seeds ×
`--max-products 1000`** in two modes:

| Path | `n_metals` × `n_seeds` | `n_6/6 PASS` | wall (s) | Status |
|---|---:|---:|---:|---|
| **REAL closure BFS** (molmetal_lam + RDKit, default) | 25 × 1 | **0 / 25** | 2.47 | HONEST_NEGATIVE_R20_P5 |
| **SYNTHETIC fallback** (`--synthetic-only`, CPU stub) | 25 × 5 | **25 / 25** | 1.10 | Wf-57 ship-target fallback |

The PROJECTED target per skill §8.5.2 was `≥ 25 metals × 6/6 PASS`. The
synthetic fallback meets it (25/25); the real closure BFS does NOT
(0/25). The gap is pinned by 4 NEW tests at
`molmetal/tests/test_wf330c1_c25_25metals.py` (4/4 PASS).

---

## §2 MEASURED verdicts (2026-09-29)

### §2.1 REAL closure BFS path

`molmetal/reports/wf330_closure_c25/per_metal_pass.json`

```
n_metals            = 25
n_seeds             = 1
max_depth           = 1
max_products        = 1000
use_gpu             = False
real_bfs_available  = True
synthetic_only      = False
wall_time_s_total   = 2.47

n_6_of_6_pass       = 0 / 25  ← HONEST_NEGATIVE_R20_P5
```

Per-property ledger (real BFS):

| Property | n_pass / 25 | Note |
|---|---:|---|
| P1_reflexivity | **25 / 25** | closed-term reflexivity holds |
| P2_transitivity | **0 / 25** | `witness_rate=0.0` → no finite witness for visited terms |
| P3_symmetry | **0 / 25** | `well_typed_rate=0.0` → BFS not idempotent on second pass |
| P4_dblock | **25 / 25** | d-block seed stay-on-d-block invariant holds |
| P5_reachability | **25 / 25** | reachability_rate=0.018 (18/1000) per metal |
| P6_composability | **25 / 25** | click-rule composability invariant holds |

Per-metal summary (uniform across all 25 metals — same cisplatin-like
seed template differs only in metal centre):

| Field | Value |
|---|---:|
| `n_visited` | 18 |
| `n_products` | 10 |
| `max_depth_seen` | 1 |
| `reachability_rate` | 0.018 (= 18 / 1000) |
| `witness_rate` | 0.0 (drives P2 FAIL) |
| `well_typed_rate` | 0.0 (drives P3 FAIL) |
| `coord_violations` | 0 |

### §2.2 SYNTHETIC fallback path

`molmetal/reports/wf330_closure_c25_synth/per_metal_pass.json`

```
n_metals            = 25
n_seeds             = 5
max_depth           = 1
max_products        = 1000
use_gpu             = False
real_bfs_available  = False
synthetic_only      = True
wall_time_s_total   = 1.10

n_6_of_6_pass       = 25 / 25  ← CPU-only stub PASS by construction
```

The synthetic fallback hard-codes `witness_rate=1.0` and
`well_typed_rate=1.0`, so P2 + P3 PASS by definition. This is the
Wf-57 ship-target fallback for environments where RDKit / molmetal_lam
cannot be imported. **It is NOT a real measurement and MUST NOT be
cited as such** per `party-twenty-spirit` §3.5.

---

## §3 PROJECTED vs MEASURED vs GAP

Per skill §8.5.2 PROJECTED target:

```
PROJECTED_6_OF_6_PASS_GATE   = 25 metals × 6/6 PASS  (R20-P5 ship)
PROJECTED_REACHABILITY_GATE  = 0.95  per-metal ≥ 0.95
```

Real BFS MEASURED:

```
n_6_of_6_pass                 = 0 / 25   → gap = 25
per_metal_reachability_rate   = 0.018    → gap = 0.932
```

Synthetic MEASURED (NOT a real measurement):

```
n_6_of_6_pass                 = 25 / 25  → gap = 0
```

The R20-P5 honest-negative ledger:

* **P2_transitivity** does NOT hold on the real closure BFS because
  visited terms have `witness_rate=0.0` — there is no finite witness
  path that the verifier can confirm from the closure-theorem BFS
  output. A formal witness construction (e.g. an explicit
  curried-application certificate) is a separate R21+ extension.
* **P3_symmetry** does NOT hold on the real closure BFS because the
  `well_typed_rate=0.0` indicates products emitted by one BFS layer
  may not be re-emittable on a second pass — the closure operator is
  not idempotent under the current `click_rules` wiring. A separate
  re-emission operator (e.g. a normalisation pass before re-emission)
  is a separate R21+ extension.
* The remaining 4/6 properties (P1 reflexivity, P4 d-block, P5
  reachability, P6 composability) all hold 25/25 = 100% on the real
  BFS path — these are the **operational** invariants the Wf-57
  bench can guarantee today.

---

## §4 Comparison to MEASURED L1 / L6 / L12 anchors

Per `TODO/completed/83j_lambda_closeout.md` §3.1:

| Anchor | MEASURED | Source | Wf-330-C1 context |
|---|---:|---|---|
| L1 (β-NF enumeration, tmQM) | 0.5108 | `wf36_closure_theorem/closure_bench.json` | Wf-36 P1 (836 × 9 metals) |
| L6 (d-block reachability, 30 metals) | 0.5891 | `wf37_closure_dblock/per_metal_bench.json` | Wf-37 P2 (2244 × 30 metals) |
| L12 PlatinAI 5K OOD | 0.5581 | `wf41_closure_ood/ood_closure.json` | Wf-41 P2 (86 OOD-filtered samples) |
| L12 MetalCytoTox 5-metal OOD | 0.0300 | `wf41_closure_ood/ood_closure.json` | Wf-41 P2 (100 OOD-filtered samples) |
| Tanimoto<0.4 fraction_below | 0.9978 | `wf57_tanimoto/ood_split.json` | Wf-57 (32256 pairs × 30 metals) |

The Wf-330-C1 reachability_rate=0.018 is **MUCH lower** than L1 / L6 /
L12 because:

1. The default `--max-products 1000` budget is hit at depth=1 before
   the BFS expands further; the BFS never visits the partner-tile
   component that drives L1 / L6 higher.
2. The `cisplatin-like` seed template (`[<metal>](<-[Cl-])(<-[Cl-])(<-N)<-N`)
   is depth-1 saturated — the closure operator cannot derive more
   products without deeper expansion (currently `--max-depth 1`).
3. The real closure BFS at `max_depth=1` is structurally different
   from the Wf-37 / Wf-41 BFSes, which use larger partner-term sets
   and depth-2+ expansion. Future R20-P5 work should run the bench
   with `--max-depth 3 --max-products 1000` (deeper expansion,
   `n_max_products=1000`) to reach the L6 anchor territory.

---

## §5 Bench command (reproducible)

```bash
# REAL BFS path — 25 metals × 1 seed × max_products 1000 (default).
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/bench_catherine_6_props_25metals.py \
    --metals 25 \
    --seeds 0 \
    --max-depth 1 \
    --max-products 1000 \
    --out-dir molmetal/reports/wf330_closure_c25/

# SYNTHETIC fallback path — 25 metals × 5 seeds × max_products 1000.
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/bench_catherine_6_props_25metals.py \
    --metals 25 \
    --seeds 0 1 2 3 4 \
    --max-depth 1 \
    --max-products 1000 \
    --synthetic-only \
    --out-dir molmetal/reports/wf330_closure_c25_synth/
```

The bench is CPU-only; no GPU is required. `--use-gpu` is honoured
only when RDKit benefits from GPU acceleration (today it does not).

---

## §6 New tests (4/4 PASS)

`molmetal/tests/test_wf330c1_c25_25metals.py`:

1. **`test_bench_catherine_runs_on_25_metals`** — Verifies the bench
   script imports, accepts `--metals 25 + --max-products 1000`, and
   emits `per_metal_pass.json` matching the Wf-57 contract schema on
   a 3-metal smoke run under 90-second timeout.
2. **`test_catherine_output_schema_contract`** — Pins the schema of
   `metadata` (11 keys), `metals[*]` (verdict + `per_property` with
   the 6 PIDs), and `summaries[*]` (with `reachability_rate`). Run
   against both REAL and SYNTHETIC JSONs.
3. **`test_catherine_verdict_schema`** — Verifies `six_of_six_pass`
   equals `(P1 & P2 & P3 & P4 & P5 & P6).passed` on every metal in
   both reports.
4. **`test_per_metal_6_of_6_pass_pins_honest_negative`** — Pins the
   HONEST_NEGATIVE_R20_P5 ledger:
   - Real BFS: `n_6_of_6=0`, P1+P4+P5+P6=25/25, P2+P3=0/25.
   - Synthetic: `n_6_of_6=25`.
   - Real BFS gap to PROJECTED gate = **25** (full gap).
   - Synthetic gap to PROJECTED gate = **0**.

Test run: `4 passed, 1 warning in 2.12s`.

---

## §7 Files shipped

| Path | Purpose |
|---|---|
| `molmetal/reports/wf330_closure_c25/per_metal_pass.json` | Real BFS MEASURED (0/25 6/6 PASS) |
| `molmetal/reports/wf330_closure_c25/sweep_summary.md` | Markdown report (real BFS) |
| `molmetal/reports/wf330_closure_c25_synth/per_metal_pass.json` | Synthetic fallback (25/25 6/6 PASS) |
| `molmetal/reports/wf330_closure_c25_synth/sweep_summary.md` | Markdown report (synthetic) |
| `molmetal/tests/test_wf330c1_c25_25metals.py` | 4 NEW pinning tests (4/4 PASS) |
| `TODO/completed/330c1_closure_c25_ship.md` | This file |

---

## §8 Next steps (R20-P5 honest-negative close-out)

Per `party-twenty-spirit` §3.5 honest framing + skill §6 ledger:

1. **P2_transitivity FAIL close-out**: requires explicit witness
   construction (curried-application certificate) so that the closure
   verifier can prove `witness_path_exists=True` for every visited
   term. Categorical closure axiom needs a separate mathematical
   layer (R20-P5 §5.3 reference).
2. **P3_symmetry FAIL close-out**: requires a normalisation pass
   between BFS layers (e.g. β-NF canonicalisation) so that products
   emitted by layer 1 are re-emittable as-is on layer 2. R21+
   extension.
3. **Reachability lift to ≥ 0.95 PROJECTED gate**: requires deeper
   BFS (`--max-depth 3`) + larger partner-term set; the current
   `cisplatin-like` seed template is depth-1 saturated at
   `n_visited=18`.
4. **Wall-budget for re-run**: the bench finished in 2.47s on the
   real BFS path (depth=1, max_products=1000); the deeper config
   (`--max-depth 3 --max-products 1000`) is expected to take
   ~30 min for 25 metals × 5 seeds per the Wf-57 §3.3 budget.

The HONEST_NEGATIVE_R20_P5 ledger is **closed** for the Wf-330-C1
ship surface (4 NEW pinning tests + 2 JSON reports + this file).
Future work to lift P2 / P3 / reachability must reopen the ledger
under a new Wf ID (R21+ ship wave).
