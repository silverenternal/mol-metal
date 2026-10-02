# Wf-350-A3 — Hyperparameter Provenance Log Gate (NeurIPS §D.9)

**Date:** 2026-09-29
**Ship:** Wf-350-A3 closeout
**Priority:** P3 pre-retrain internal check
**Contract:** NeurIPS §D.9 reproducibility protocol + Wf-343A-R2 recommendation

---

## §1 Ship summary

Wf-350-A3 ships **1 NEW pre-retrain internal check gate** on the pre-flight
v2 checklist:

- **`p1_hp_provenance_log`** — records hyperparameter ``git-SHA provenance
  + recent commit history + uncommitted-change status`` for every config
  file in ``configs/`` and emits a JSONL artefact at
  ``molmetal/reports/wf350a3_hp_provenance.jsonl`` so downstream consumers
  (paper §3 Wf-350 attribution appendix + paper dashboard reproducibility
  table + F2(a) hyperparam rewind list) can trace every retrain back to the
  exact commit that produced its hyperparameters.

The gate is the **3rd** of the Wf-343A-R2 pre-retrain internal check trio:

1. **Wf-343C4** (4 NEW gates) — hyperparams config complete + YuelBond decoder
   env + anchor freshness 24h + wave budget non-negative
2. **Wf-347A1** (2 NEW gates) — per-wave runtime check + data quality check
   subprocess wires
3. **Wf-350-A3** (1 NEW gate) — hyperparameter provenance log + NeurIPS §D.9
   reproducibility contract

**EXPECTED_GATE_COUNTS["all"]** bumped from 68 → 70 in lockstep with the
audit doc (``TODO/completed/321d_preflight_gate_audit.md``) — the +1 from
Wf-350-A3 plus the sibling +1 from Wf-350-A1 (runtime + GPU energy probe
gate, NeurIPS §E.4 reproducibility protocol). Both Wf-350-A1 + Wf-350-A3
landed concurrently per the 2026-09-29 user ultracode directive.

---

## §2 Gate specification

**Function signature:**

```python
def gate_p1_hp_provenance_log(
    name: str,
    priority: str,
    pipeline: str,
    config_paths: Tuple[Path, ...],
    jsonl_path: Path,
    *,
    recent_commits_count: int = 20,
    timeout: float = 30.0,
) -> GateResult:
```

**Probes (3 git subprocess calls per config file, all CPU-only):**

| # | Probe | Purpose |
|---|---|---|
| 1 | `git rev-parse HEAD` | current commit SHA (the canonical attribution anchor) |
| 2 | `git log --pretty=format:"%H\|%ai\|%s" -n 20 -- <rel_cfg>` | up to 20 most-recent commits touching the config |
| 3 | `git status --short -- <rel_cfg>` | uncommitted-changes boolean (uncommitted YAMLs are honest, not paper-grade) |

**JSONL schema (one per config file, one per line):**

```json
{
    "config_path": "configs/cfm_full_retrain.yaml",
    "current_sha": "9b12e6f574dfc397b5b5e8c568c952266e82ef37",
    "recent_commits": [
        {"sha": "9b12e6f5...", "message": "Wf-243 to Wf-252: ...", "date": "2026-09-28 18:54:36 +0800"},
        {"sha": "4527e4f3...", "message": "Initial public release of Mol-Metal (R16 close-out)", "date": "2026-09-18 15:51:56 +0800"}
    ],
    "uncommitted_changes": false
}
```

**PASS criteria:**

- All config paths exist on disk (`is_file`).
- For each config: `git rev-parse HEAD` succeeds.
- One JSON record per config file is written (7 records for the canonical 7 configs).
- Each record carries the 4 mandatory keys: `config_path` (str) +
  `current_sha` (str, 40-hex git SHA) + `recent_commits` (list, possibly empty
  if the config is uncommitted) + `uncommitted_changes` (bool).
- The gate round-trips: re-reads the JSONL after write and verifies the
  schema via a second JSON parse (catches a future change that silently drops
  a key).

**FAIL criteria:**

- Missing config file, missing JSONL artefact, missing record, schema violation,
  or git subprocess failure (timeout / non-zero exit code).

**Wall budget (CPU-only):**

- Per config file: ~3 git subprocess calls × < 5 ms each = ~15 ms.
- Total 7 configs: ~100 ms (measured 0.11 s on the live host).
- Test suite: 3 tests in < 1 s.

---

## §3 Honest framing (party-twenty-spirit §3.5)

### §3.1 What this gate DOES

- **Records provenance.** Every config in ``configs/`` is fingerprinted
  with the current git SHA + recent commit history + uncommitted-changes
  boolean so any retrain (or panel / ablation) can be traced back to the
  exact commit that produced its hyperparameters.
- **Surfaces uncommitted configs honestly.** If a YAML has never been
  committed (e.g. a new Wf-124 P3 YAML), the JSONL carries
  `recent_commits: []` + `uncommitted_changes: true` so downstream
  consumers know the config is brand-new (and possibly experimental).
- **Prevents silent provenance drift.** The Wf-321D audit invariant
  (EXPECTED_GATE_COUNTS["all"] == actual_runner_count) is preserved by
  registering the gate in `build_gates("all")` and bumping
  `EXPECTED_GATE_COUNTS["all"]` from 68 → 70 in lockstep.

### §3.2 What this gate does NOT do

- **It does NOT validate that hyperparameters are reasonable.** That's the
  Wf-343C4 `p3_wf343c4_hyperparams_config_complete` gate's contract
  (verifies `train.batch_size` / `train.lr` / `model.hidden_dim` /
  `model.n_layers` exist in each YAML per AGENTS.md §5.1 standing GPU grant).
  Wf-350-A3 is the provenance / traceability axis; Wf-343C4 is the
  contract-compliance axis.
- **It does NOT block retrain launches.** A `recent_commits: []` or
  `uncommitted_changes: true` outcome is **PASS** — the gate's contract
  is to RECORD reality, not to BLOCK on reality. Any policy decision
  (e.g. "block retrain unless config has ≥1 commit") belongs to a
  separate downstream gate.
- **It does NOT verify GPU / disk / wave budget.** Those are Wf-343C4 +
  Wf-347A1 + Wf-350-A1 contracts.
- **It does NOT modify any pre-existing gate / runner / path.** The
  Wf-321D gate-count-consistency audit invariant is preserved by the
  bump from 68 → 70 in lockstep with the audit doc.

### §3.3 Known limitations

- **Subprocess-based git probes**: the gate invokes 3 `git` subprocesses
  per config file. On a fresh sandbox without git history, the gate will
  FAIL on `git log` (rc≠0) — this is the intentional honest behavior
  per the §2 FAIL criteria. Test #3 (`test_gate_p1_hp_provenance_log_registered_in_build_gates`)
  accepts either PASS or FAIL on the smoke call because the contract
  is "registered + callable + returns a valid GateResult", not
  "PASS on every host".
- **`recent_commits_count=20` is a default, not a guarantee.** A config
  with >20 commits in history will only show the 20 most recent.
  Hyperparam provenance paper §D.9 typically requires ~5-10 most-recent
  commits for the reviewer table, so 20 is a generous ceiling.
- **JSONL is one-line-per-config, not one-line-per-commit.** If a paper
  reviewer needs the per-commit diff for a config, that's a downstream
  `git log -p` call, not Wf-350-A3's contract.

---

## §4 Test surface (3 NEW tests, all MEASURED PASS on live host)

`molmetal/tests/test_wf350a3_hp_provenance.py` — 3 NEW tests:

| # | Test | Asserts |
|---|---|---|
| 1 | `test_gate_p1_hp_provenance_log_schema_validity` | Gate PASSes on live host; JSONL exists with exactly 7 records; every record carries the 4 mandatory keys with correct types (str/str/list/bool); every `current_sha` matches the 40-hex git SHA pattern. |
| 2 | `test_gate_p1_hp_provenance_log_per_config_records` | All 7 canonical configs appear in JSONL; every `config_path` is one of the canonical 7; every `current_sha` matches the live `git rev-parse HEAD`; the uncommitted-changes / recent-commits branches are both valid (recent_commits = [].implies uncommitted_changes = True); every non-empty recent_commits entry has sha (40-hex) + message + date. |
| 3 | `test_gate_p1_hp_provenance_log_registered_in_build_gates` | `p1_hp_provenance_log` is in `build_gates("all")`; `EXPECTED_GATE_COUNTS["all"]` equals `len(build_gates("all"))` (Wf-321D self-consistency invariant); runner returns a valid GateResult on smoke call. |

**Test results (live host, 2026-09-29):**

```
molmetal/tests/test_wf350a3_hp_provenance.py::test_gate_p1_hp_provenance_log_schema_validity PASSED
molmetal/tests/test_wf350a3_hp_provenance.py::test_gate_p1_hp_provenance_log_per_config_records PASSED
molmetal/tests/test_wf350a3_hp_provenance.py::test_gate_p1_hp_provenance_log_registered_in_build_gates PASSED

3 passed, 1 warning in 0.38s
```

---

## §5 Pre-flight integration

`gate_p1_hp_provenance_log` is registered in `build_gates("all")` after the
Wf-347A1 pair (last in the 'all' pipeline runner list) so the Wf-321D
gate-count-consistency audit (also last in 'all') sees the new count
without re-running the audit recursively.

**Pre-flight run (live host, 2026-09-29 16:32 UTC, wall 133.85 s):**

```
summary: {pass: 68, fail: 2, skip: 1, total: 71, total_wall_sec: 133.8455}
```

**Wf-350-A3 gate result:**

```
NAME: p1_hp_provenance_log
STATUS: PASS
PIPELINE: wf350a3
PRIORITY: P3
EVIDENCE: Wf-350-A3 hp provenance log PASS: 7/7 config records written to
  /home/hugo/codes/mol-metal/molmetal/reports/wf350a3_hp_provenance.jsonl
  (current_sha=9b12e6f574df..., recent_commits[0]=0 entries,
  uncommitted_changes=True)
WALL: 0.1127s
```

**Pre-existing FAILs (not related to Wf-350-A3):**

1. `p3_wf343c4_anchor_freshness_24h` — 4/11 paper-grade anchors STALE (> 24h)
   per Wf-302A / Wf-304A AnchorGuard freshness contract.
2. `p3_wf343c2_per_wave_runtime_check` — `wf343c2_per_wave_check.py` FAIL
   on the Wf-347A1 pre-retrain smoke wave.

Both are pre-existing honest-negative outcomes; Wf-350-A3 does not regress
either.

---

## §6 Files touched

| Path | Change |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | +1 NEW gate function (`gate_p1_hp_provenance_log`, ~170 LOC); `EXPECTED_GATE_COUNTS["all"]` 68 → 70; `EXPECTED_GATE_COUNTS` doc-block comment updated to reflect Wf-350-A3 ship; +1 NEW runner registration in `build_gates("all")` (with verbose §3.4 design-contract comment). |
| `molmetal/tests/test_wf350a3_hp_provenance.py` | NEW file, 3 tests, ~280 LOC. |
| `molmetal/reports/wf350a3_hp_provenance.jsonl` | NEW artefact (7 records, written by gate on each pre-flight run). |
| `TODO/completed/350a3_hp_provenance_log.md` | THIS FILE (closeout ship summary). |

---

## §7 Standing directives honoured

| Rule | Status |
|---|---|
| AGENTS.md §1 MEASURED-vs-PROJECTED discipline | All numeric values trace to a JSON / report path on disk. Wall (0.113 s), 71-total, 7-records-per-artefact are MEASURED on the live host. |
| AGENTS.md §5.1 Standing GPU authorization | Not invoked (CPU-only gate; no GPU launch). |
| AGENTS.md §5.2 Honest framing mandate | §3.1 + §3.2 + §3.3 explicitly enumerate what the gate does, what it does NOT do, and the 3 known limitations. |
| AGENTS.md §5.3 No paper work without authorization | Not invoked (paper §3 attribution downstream consumer is READ-only on this file). |
| Wf-321D gate-count-consistency invariant | Preserved: `EXPECTED_GATE_COUNTS["all"] = 70 = len(build_gates("all"))`. Test #3 explicitly asserts this self-consistency. |

---

## §8 Remaining work for Wf-350-B+ (none for Wf-350-A3)

Wf-350-A3 is closed:

- Gate function SHIPPED.
- Tests SHIPPED (3/3 PASS).
- Pre-flight wire-in SHIPPED (gate runs in 0.11 s, PASS on live host).
- JSONL artefact SHIPPED (`molmetal/reports/wf350a3_hp_provenance.jsonl`,
  7 records, refreshed on every pre-flight invocation).
- Closeout doc SHIPPED (this file).

**No GPU launches performed** (CPU-only by design).

**No paper §3 touches** (paper §3 attribution downstream consumer is
READ-only on this gate's output).

**Future extension (Wf-350-B+):** A separate policy gate could read
`wf350a3_hp_provenance.jsonl` and BLOCK retrain launches where
`uncommitted_changes: true` for an active config. This is a separate
workflow (Wf-350-B+) and not part of Wf-350-A3's contract.

---

**END OF Wf-350-A3 CLOSEOUT** — Owner Hugo Lee; cross-ref AGENTS.md §5.2;
last update 2026-09-29 16:35 UTC.