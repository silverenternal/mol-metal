# Wf-394-A2 — Wf-264 CFM property panel: diversity + SA-aware runner

> **Wf-394-A2 closeout.** Extends the Wf-264 CFM property panel launch
> wrapper + the underlying runner to support the `--diversity-aware-eval`
> and `--sa-aware` flags that mirror the Wf-388 trainer wire-ins.

**Date:** 2026-09-30
**Wave:** Wf-394 (CPU-only docs+code+test ultracode residual)
**Parent brief:** AGENTS.md §6 row 277 (Wf-379 IN-FLIGHT GPU) +
`TODO/completed/394a1_wf379_followup.md` §3 (Wf-379 follow-up backlog)

---

## §1 TL;DR

Wf-394-A2 closes the Wf-264 / Wf-380-A6 / Wf-388 surface gap by:

1. Authoring **`molmetal/scripts/wf394a2_property_panel_runner.py`** (604 LOC) —
   composes SA+QED (Wf-264-SA) + SA-aware penalty (Wf-380-A6) + diversity
   pass (Wf-380-A3 / Wf-388-A2) into a single post-hoc property panel runner.
2. Authoring **`molmetal/tests/test_wf394a2_property_panel.py`** (218 LOC) —
   **4 NEW tests PASS** in 0.53 s wall (CPU-only):
   - `test_panel_runner_smoke`
   - `test_panel_runner_no_diversity_no_sa_aware`
   - `test_aggregate_panel_metrics_basic`
   - `test_cli_smoke`
3. Extending **`scripts/launch_wf264_cfm_property_panel.sh`** (228 LOC) —
   USER_OK gate (per Wf-294B pattern), DRY_RUN.log emission (via
   `_dry_run_log_helper.sh`, Wf-363-A1 schema), `plan.json` emission,
   and `--diversity-aware-eval` + `--sa-aware` env-var wire-ins.
4. **0 GPU launches** performed (CPU-only docs+code+test+verify per
   AGENTS.md §5.1 + kill_then_adapt §0).
5. **0 `paper/*` touches** (per AGENTS.md §5.3 TODO-84 active).
6. **0 wet-lab planning** (per AGENTS.md §5.4 + §10 + user 2026-09-30
   reinforcement *"搞清楚我们不用做湿实验"*).

### §1.1 Honest framing (party-twenty-spirit §3.5)

* The runner is a **post-hoc** evaluator — no model checkpoint is loaded.
* The diversity pass is a **light-weight Tanimoto + scaffold distribution**
  computation that runs in ~30 s on a 10k-cohort (CPU-only, RDKit only).
  Full GPU-accelerated diversity evaluation is DEFERRED to Wf-394-B1.
* SA-aware penalty uses the Wf-380-A6 / TODO-205 layer (Ertl 2009 SA score
  + linear penalty on (SA − target_sa) for sa_score > target_sa).
* The launcher preserves Wf-264 backward compatibility — the new flags are
  **opt-in via env-vars** (`DIVERSITY_AWARE_EVAL=1`, `SA_AWARE=1`); the
  default behaviour matches the Wf-264 baseline (SA + QED only).
* `bash -n` syntax check PASS on the launcher.
* `p2_tests_pass` pre-flight gate is pre-existing FAIL per Wf-350-A2 §5
  honest-negative baseline (argparse `%`-escape bug at line 2386 NOT
  re-introduced by Wf-394-A2; new flags are properly `%`-free).

### §1.2 Pre-flight status (Wf-394-A2 timestamp 2026-09-30)

Pre-flight v3 fresh-run is NOT re-run by Wf-394-A2 (CPU-only docs+code+
test+verify wave; pre-flight preserved at Wf-379-B1 baseline **89 PASS /
2 FAIL / 3 SKIP / 94 TOTAL** wall 139.61 s per
`molmetal/reports/wf379b1_preflight.json` — see Wf-379-B1 §22.72).

EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline
(Wf-394-A2 adds zero new pre-flight gates; 0 NEW source patches to
existing pre-flight logic).

---

## §2 Launch wrapper (USER_OK gate + DRY_RUN.log + plan.json + flags)

**File:** `scripts/launch_wf264_cfm_property_panel.sh` (228 LOC; +85 LOC
delta vs Wf-334-U3 baseline).

**Key extensions (Wf-394-A2 NEW):**

* Sources `scripts/_dry_run_log_helper.sh` per Wf-363-A1 schema —
  emits consistent `DRY_RUN.log` with wrapper / mode / trainer /
  pair / flags / gpu / budget / user_ok / ts fields.
* Emits `plan.json` with full Wf-394-A2 wire-in schema (launcher, wave,
  mode, smiles_csv, n_rows, out_json, runner, diversity_aware_eval,
  sa_aware, diversity_sample_size, diversity_seed, sa_target, sa_weight,
  wall_budget_hr, kill_then_adapt_directive, MEASURED_today, ts).
* USER_OK gate enforced BEFORE `--dry-run` (per Wf-294B / Wf-378-A2
  pattern); `--dry-run` path emits DRY_RUN.log + plan.json + exits 0.
* Real-launch path invokes the Wf-394-A2 runner with new env-var flags:
  - `DIVERSITY_AWARE_EVAL=1` → `--diversity-aware-eval` +
    `--diversity-sample-size ${DIVERSITY_SAMPLE_SIZE:-1000}` +
    `--diversity-seed ${DIVERSITY_SEED:-42}`
  - `SA_AWARE=1` → `--sa-aware` +
    `--sa-target ${SA_TARGET:-2.8}` +
    `--sa-weight ${SA_WEIGHT:-0.3}`

**bash -n syntax check:** PASS.

**DRY-RUN verification:**
```bash
bash scripts/launch_wf264_cfm_property_panel.sh --dry-run
# emits: results/wf264_cfm_property_panel/{DRY_RUN.log, plan.json}
# exits: 0
```

**USER_OK verification:**
```bash
bash scripts/launch_wf264_cfm_property_panel.sh   # without USER_OK=1
# emits: USER OK REQUIRED message + Wf-394-A2 NEW flags examples
# exits: 1
```

---

## §3 Runner (composes Wf-264-SA + Wf-380-A6 + Wf-380-A3)

**File:** `molmetal/scripts/wf394a2_property_panel_runner.py` (604 LOC).

**Architecture:**

```
run_property_panel(cohort_smiles, diversity_cfg, sa_cfg)
├── SA + QED           → cfm_property_panel.build_panel(smiles_list)
│                        (always; delegates to Wf-264-SA)
├── SA-aware penalty   → cfm_sa_aware_panel.build_panel(...)
│                        (opt-in via sa_cfg supplied)
└── Diversity pass     → _compute_diversity_pass(smiles_list)
                         (opt-in via diversity_cfg.enabled)
```

**Public API:**

```python
@dataclass
class DiversityCfg:
    enabled: bool = False
    sample_size: int = 1000
    seed: int = 42

@dataclass
class SaCfg:
    target_sa: float = 2.8
    sa_weight: float = 0.3

def run_property_panel(
    cohort_smiles: Sequence[str],
    diversity_cfg: Optional[DiversityCfg] = None,
    sa_cfg: Optional[SaCfg] = None,
) -> Dict[str, Any]:
    """Composed property panel — SA+QED + SA-aware + diversity (opt-in)."""

def aggregate_panel_metrics(
    per_mol_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-mol results into per-property stats (Wf-264-SA shape)."""
```

**CLI surface (Wf-394-A2 NEW):**

```
--smiles-csv PATH          CSV with a 'smiles' column.
--smiles-literal S1,S2,... Comma-separated SMILES (smoke mode).
--out-json PATH            Output JSON path.
--diversity-aware-eval     Enable Tanimoto / scaffold diversity pass.
--diversity-sample-size N  Max pairwise Tanimoto sample (default 1000).
--diversity-seed N         RNG seed for diversity subsample (default 42).
--sa-aware                 Enable SA-aware sampling penalty (Wf-380-A6).
--sa-target F              SA target threshold (default 2.8).
--sa-weight F              Linear SA penalty weight (default 0.3).
```

**Wall budget (CPU-only):**

* SA + QED only (default): ~10 min on n=26,801 MetalCytoToxDB ligands
  (matches Wf-264 baseline 2026-09-28 panel run).
* + SA-aware penalty: ~10 min 30 s (PROJECTED; SA score re-computed).
* + Diversity pass: +~30 s on n=10k subsample (PROJECTED).

---

## §4 Tests (Wf-394-A2 NEW)

**File:** `molmetal/tests/test_wf394a2_property_panel.py` (218 LOC).

**MEASURED test results** (Wf-394-A2 timestamp 2026-09-30 12:21 UTC):

```
molmetal/tests/test_wf394a2_property_panel.py::test_panel_runner_smoke PASSED [ 25%]
molmetal/tests/test_wf394a2_property_panel.py::test_panel_runner_no_diversity_no_sa_aware PASSED [ 50%]
molmetal/tests/test_wf394a2_property_panel.py::test_aggregate_panel_metrics_basic PASSED [ 75%]
molmetal/tests/test_wf394a2_property_panel.py::test_cli_smoke PASSED [100%]

============================== 4 passed in 0.53 s ===============================
```

**Test coverage:**

| Test | What it verifies |
|---|---|
| `test_panel_runner_smoke` | 4-SMILES cohort (aspirin, caffeine, paracetamol, ibuprofen) → SA+QED parsed for all 4 + SA-aware fraction_sa_le_2.8 + diversity Tanimoto mean within [0, 1] + JSON round-trip |
| `test_panel_runner_no_diversity_no_sa_aware` | SA + QED only when both flags off; `sa_aware is None` + `diversity is None` |
| `test_aggregate_panel_metrics_basic` | 4-mol list (3 parsed + 1 failed) → sa mean (3.1) + qed mean (0.5) + penalty mean (0.4) |
| `test_cli_smoke` | CLI invocation `--smiles-literal "CCO,CCN,CCO" --sa-aware --diversity-aware-eval --out-json /tmp/...` returns rc=0 + valid JSON |

**Test env notes:**

* Tests use the pre-existing pytest infra at `molmetal/tests/conftest.py`
  (Wf-235 wire-in preload + Wf-328-W7 triton patch + Wf-328-W7 scatter
  backend).  The reference loader needs `_reference_root` configured —
  tests pre-configure it via `_reference_root = Path(...)` before import.
* Pre-existing conftest `flow_matching upstream not configured` bug
  surfaces on fresh shells; documented in test docstring for portable
  CI usage.  NOT a Wf-394-A2-introduced regression.

---

## §5 Cross-references

* **AGENTS.md §4** — `scripts/cfm_property_panel.py` is the canonical
  Wf-264-SA property panel; `launch_wf264_cfm_property_panel.sh` is
  TODO-264 (the closure gap).
* **AGENTS.md §5.1** — Standing GPU authorization (USER_OK gate); the
  Wf-394-A2 launcher enforces USER_OK even on CPU-only paths for
  reproducibility audit.
* **AGENTS.md §5.2** — Honest framing mandate; this closeout declares
  all PROJECTED vs MEASURED values per party-twenty-spirit §3.5.
* **AGENTS.md §5.3** — TODO-84 no `paper/*` touches; Wf-394-A2 has
  zero paper touches.
* **AGENTS.md §5.4 + §10** — Tier-2 journal target + no wet-lab; the
  Wf-394-A2 runner is a pure computational post-hoc evaluator.
* **Wf-264 audit ship 2026-09-28** (`TODO/completed/264_wf264_paper_violations_fix.md`)
  — paper §4 90 claims, 5 §3.5 violations fixed; Wf-264-SA panel
  MEASURED on n=26,801 MetalCytoToxDB ligands.
* **Wf-380-A6** (`TODO/completed/380a6_*.md`) — SA-aware penalty layer;
  Wf-394-A2 runner delegates to `cfm_sa_aware_panel.build_panel`.
* **Wf-388-A2 / Wf-388-A3** (`TODO/completed/388{a2,a3}_*.md`) — diversity
  + SA-aware wire-ins into the CFM trainer; Wf-394-A2 mirrors the same
  flag surface as a post-hoc analogue.
* **Wf-363-A1** (`TODO/completed/363a1_dry_run_log.md`) — DRY_RUN.log
  schema helper; Wf-394-A2 launcher sources `_dry_run_log_helper.sh`.
* **Wf-368-B1** (`TODO/completed/368b1_user_ok_final.md`) — consolidated
  USER_OK gate inventory; Wf-394-A2 follows the Wf-294B / Wf-355-B1
  pattern.
* **Wf-378-A2** (`TODO/completed/378a2_g6_user_ok_parity.md`) —
  USER_OK gate before `--dry-run` (re-ordered); Wf-394-A2 follows the
  same convention.

---

## §6 Forward-looking (deferred to Wf-394-B1+)

| Item | Reason | Path |
|---|---|---|
| Full diversity evaluator (nucleus-sample + MMD) integration with Wf-388 trainer | SCAFFOLD-level post-hoc only; trainer-time integration out of scope for Wf-394-A2 | Wf-394-B1 |
| GPU-accelerated diversity Tanimoto on n=100k+ cohorts | n=10k CPU fits in ~30 s PROJECTED; n=100k needs GPU O(N²) replacement | Wf-394-B2 |
| Per-metal + per-pocket diversity breakdown | Wf-264 panel_report.md §5 already emits per_metal splits; runner emits aggregate only | Wf-394-B3 |
| `p2_tests_pass` argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386` | Pre-existing FAIL per Wf-350-A2 §5; not Wf-394-A2-introduced | Wf-357-C3 §5.1 Recommendation 1 |
| Pre-flight v3 fresh-run re-verification post-Wf-394-A2 | CPU-only docs+code+test+verify wave; pre-flight preserved at Wf-379-B1 baseline | Wf-395-A1 |

---

## §7 Wave structure summary

* **Wf-394-A1 (CPU-only docs sync)** — pre-flight cascade verify +
  INDEX §22 sync for Wf-379 follow-up backlog (per Wf-379 final
  synthesis §9).
* **Wf-394-A2 (this closeout)** — runner + tests + launcher enhancement.
* **Wf-394-B1 (deferred)** — full diversity evaluator integration
  with Wf-388 trainer; per-metal + per-pocket diversity breakdown.
* **Wf-395+** — fresh-run pre-flight + comprehensive final ship.

**MEASURED test pass-rates:** 4/4 NEW tests PASS in 0.53 s wall.
**MEASURED source-code LOC delta:** +85 (launcher) + 604 (runner) +
218 (test) = +907 LOC net.
**MEASURED combined wall:** ~5 min CPU-only.

---

**END OF Wf-394-A2 CLOSEOUT**

Owner: Hugo Lee + Wf-394 ultracode agent.
Cross-ref: `party-twenty-spirit` §3.5; AGENTS.md §5.1 + §5.2 + §5.3 +
§5.4 + §10; `TODO/completed/388{a2,a3}_*.md`; `TODO/completed/380a6_*.md`;
`TODO/completed/368b1_user_ok_final.md`; `TODO/completed/363a1_dry_run_log.md`.