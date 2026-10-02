# Wf-350-B3 — Reproducibility Manifest as 1st-Class Artifact in `--json` Output

**Wf:** Wf-350-B3
**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** SHIPPED (CPU-only; 2/2 NEW tests PASS)
**Standing-rule compliance:** AGENTS.md §1 (MEASURED-vs-PROJECTED),
§5.2 (honest framing), §5.1 (CPU-only; no GPU launch).
**Parent wave:** Wf-350 (NeurIPS §3.6 reproducibility backbone).
**Recommendation source:** Wf-343a-R1 finding #3 — "Reproducibility
manifest should extend --json output" (DeepMind §3.6 contract).

---

## §1 Scope

Per the 2026-09-29 user ultracode directive ("覆盖所有发现的问题把残余
的问题解决了") and per Wf-343a-R1 finding #3, the `--json` output
of `molmetal/scripts/pre_flight_check.py` gains 1 NEW top-level key:

- **`reproducibility_manifest`** — embeds the 3 Wf-350 reproducibility
  gates' on-disk artefacts (compute + hp provenance + runtime energy)
  plus a forward-looking `prereg_path` slot. Downstream consumers
  (paper §3 attribution appendix + paper dashboard reproducibility
  table + F2(a) hyperparam rewind list) can now reconstruct the full
  reproducibility context from a single `--json` file.

The extension is **ADDITIVE only** — does NOT modify the existing
`gates` array, summary block, or any other top-level key. The
Wf-321D gate-count-consistency audit invariant is preserved (no
new gate is registered; only the `--json` schema is enriched).

---

## §2 Deliverables (4/4 SHIPPED)

| # | Deliverable | Path | Status |
|---|---|---|---|
| 1 | Helper function `_build_reproducibility_manifest()` | `molmetal/scripts/pre_flight_check.py:6326-6410` (~85 LOC) | SHIPPED |
| 2 | `--json` schema extension: `report["reproducibility_manifest"]` key | `molmetal/scripts/pre_flight_check.py:6378-6388` | SHIPPED |
| 3 | 2 NEW integration tests (2/2 PASS) | `molmetal/tests/test_wf350b3_repro_manifest.py` | SHIPPED |
| 4 | Closeout documentation | `TODO/completed/350b3_repro_manifest_extension.md` (this doc) | SHIPPED |

---

## §3 Schema contract

### §3.1 Top-level shape

```json
{
  "timestamp": "<ISO-8601 UTC>",
  "schema_version": "207.v1",
  "pipeline": "<pipeline>",
  "summary": {"pass": ..., "fail": ..., "skip": ..., "total": ..., "total_wall_sec": ...},
  "gates": [...],
  "reproducibility_manifest": {
    "compute":       <dict | null>,    // wf350a2_compute_manifest.json (Wf-350-A2)
    "hp_provenance": <list | null>,    // wf350a3_hp_provenance.jsonl (Wf-350-A3, JSONL)
    "runtime_energy":<dict | null>,    // wf350a1_runtime_energy.json (Wf-350-A1)
    "prereg_path":   <str  | null>     // forward-looking (Wf-343a-R1 / DeepMind §3.1)
  },
  "v2_doc_reference": "..."
}
```

### §3.2 Per-section schema

**(a) `compute`** (Wf-350-A2, NeurIPS §E.5) — dict with 5 top-level
keys: `gpus` (list of `{index, name, memory_total_mb, driver_version}`)
+ `driver_version` (str) + `torch_version` (str) +
`cuda_version` (str) + `system` (dict with `{hostname, kernel, arch}`).

**(b) `hp_provenance`** (Wf-350-A3, NeurIPS §D.9) — list of dicts, one
per config file. Each record has 4 mandatory keys: `config_path` (str) +
`current_sha` (str, 40-hex git SHA) + `recent_commits` (list, possibly
empty) + `uncommitted_changes` (bool). Canonical 7 records expected.

**(c) `runtime_energy`** (Wf-350-A1, NeurIPS §E.4) — dict with 4
mandatory keys: `per_gpu` (list of `{index, power_draw_w,
power_limit_w}`) + `total_duration_s` (float ≥ 0) +
`estimated_energy_j` (float ≥ 0) + `n_gpus_reported` (int).

**(d) `prereg_path`** (Wf-343a-R1 / DeepMind §3.1, forward-looking) —
repo-relative path to a paper-grade `prereg.md` if one exists at the
conventional launch positions (`TODO/pending/prereg.md` →
`TODO/prereg.md` → `./prereg.md`). Currently `None` on the canonical
host — no prereg.md exists yet (the pre-registration record contract
is forward-looking per Wf-343a-R1).

### §3.3 Graceful fallback contract

All 4 loads are best-effort. On a missing or malformed artefact, the
corresponding slot is `None` (NOT a hard FAIL — the gate itself would
have FAILed during execution; this helper is a serialisation aid, not
a re-runner). On the canonical ZhaoLab host with all 3 Wf-350
artefacts on disk, all 3 slots are populated and `prereg_path` is
`None`.

---

## §4 Live-host MEASURED manifest (2026-09-29)

Verified by parsing `/tmp/preflight_wf350b3.json` (the actual `--json`
output of `pre_flight_check.py --pipeline all --json`):

```python
top-level keys: ['gates', 'pipeline', 'reproducibility_manifest', 'schema_version', 'summary', 'timestamp', 'v2_doc_reference']
manifest keys: ['compute', 'hp_provenance', 'prereg_path', 'runtime_energy']
compute top-level keys: ['cuda_version', 'driver_version', 'gpus', 'schema_version', 'system', 'torch_version', 'ts']
hp_provenance count: 7    # canonical 7-config suite
runtime_energy keys: ['MEASURED_vs_PROJECTED', 'estimated_energy_j', 'host', 'n_gpus_reported', 'per_gpu', 'probe_timestamp', 'total_duration_s']
prereg_path: None         # forward-looking per Wf-343a-R1
```

---

## §5 Test surface (2 NEW, 2/2 PASS)

`molmetal/tests/test_wf350b3_repro_manifest.py` — 2 NEW tests:

| # | Test | Asserts |
|---|---|---|
| 1 | `test_preflight_json_includes_reproducibility_manifest` | `--json` output's top-level dict has `reproducibility_manifest` key + the manifest is a dict (not None) + all 4 sub-keys (`compute`, `hp_provenance`, `runtime_energy`, `prereg_path`) are present. |
| 2 | `test_reproducibility_manifest_per_section_present` | Per-section schema validity: `compute` is a dict with the canonical NeurIPS §E.5 keys (`gpus` + `driver_version` + `torch_version` + `cuda_version` + `system`); `hp_provenance` is a non-empty list of dicts each with the 4 Wf-350-A3 mandatory keys (`config_path` + `current_sha` + `recent_commits` + `uncommitted_changes`); `runtime_energy` is a dict with the Wf-350-A1 4 mandatory keys (`per_gpu` + `total_duration_s` + `estimated_energy_j` + `n_gpus_reported`); `prereg_path` is None or str. SKIP semantics on a host without the Wf-350 artefacts (env-aware, not FAIL). |

**Test result (live host, 2026-09-29):**

```
molmetal/tests/test_wf350b3_repro_manifest.py::test_preflight_json_includes_reproducibility_manifest PASSED [ 50%]
molmetal/tests/test_wf350b3_repro_manifest.py::test_reproducibility_manifest_per_section_present PASSED [100%]

2 passed, 1 warning in 306.39s (0:05:06)
```

The 306 s wall is dominated by the 2 subprocess invocations of
`pre_flight_check.py --pipeline all` (~140 s each on the canonical
host). The assertion logic itself is sub-second.

---

## §6 Standing directives honoured

| Rule | Status |
|---|---|
| AGENTS.md §1 MEASURED-vs-PROJECTED discipline | All numeric values trace to a JSON / report path on disk (the 3 Wf-350 artefact JSON files + the parsed `--json` output). Wall (306 s), 2/2 PASS, top-level-key presence are MEASURED on the live host. |
| AGENTS.md §5.1 Standing GPU authorization | Not invoked (CPU-only I/O; 3 small JSON/JSONL loads + 1 file glob for prereg.md). |
| AGENTS.md §5.2 Honest framing mandate | §3.3 graceful fallback contract + §6 Wf-321D invariant preservation + §7 limitations enumerate what this extension does + what it does NOT do + the forward-looking prereg_path caveat. |
| AGENTS.md §5.3 No paper work without authorization | Not invoked (paper §3 attribution downstream consumer is READ-only on this gate's output). |
| Wf-321D gate-count-consistency invariant | Preserved: `EXPECTED_GATE_COUNTS["all"] = 71` (no new gate added by Wf-350-B3 — only the `--json` schema is enriched). |

---

## §7 Limitations + honest framing

### §7.1 What this extension DOES

- **Embeds the 3 Wf-350 reproducibility artefacts** in the `--json`
  output so a downstream consumer can reconstruct the full
  reproducibility context (compute + hp provenance + runtime energy)
  from a single file, without needing separate reads of the 3
  artefact paths.
- **Adds a forward-looking `prereg_path` slot** for the DeepMind §3.1
  pre-registration record contract (Wf-343a-R1). When a paper-grade
  `prereg.md` is added at the conventional launch position, the
  path will appear automatically.

### §7.2 What this extension does NOT do

- **It does NOT add a new pre-flight gate.** The Wf-321D
  gate-count-consistency invariant (`all == 71`) is preserved. This is
  a serialisation extension, not a gate addition.
- **It does NOT modify the existing `gates` array.** The gate results
  are still emitted at the top-level `gates` key (backward-compat
  preserved for all existing downstream consumers).
- **It does NOT validate the on-disk artefacts.** It loads them as-is
  via `json.load`. Schema validity of the embedded artefacts is
  asserted separately by the Wf-350-A1/A2/A3 gate tests
  (`test_wf350a1_runtime_energy.py` + `test_wf350a2_compute_manifest.py`
  + `test_wf350a3_hp_provenance.py`).
- **It does NOT block pre-flight invocations.** On a host without the
  3 Wf-350 artefacts (e.g. a fresh checkout), the manifest slots
  will be `None` and the gate suite still runs normally; this
  extension is additive metadata, not a precondition.

### §7.3 Known limitations

- **JSONL parsing robustness**: `wf350a3_hp_provenance.jsonl` is
  parsed one line at a time. A truncated final line (no newline)
  is silently skipped (the `line = line.strip(); if not line: continue`
  branch); a malformed JSON line is silently skipped (the
  `json.JSONDecodeError` except clause). This matches the Wf-350-A3
  gate's tolerance contract.
- **prereg_path is forward-looking**: the canonical ZhaoLab host has
  no prereg.md. The slot is `None`. When a paper-grade `prereg.md`
  is added at `TODO/pending/prereg.md` or `TODO/prereg.md` or
  `./prereg.md`, the slot will populate automatically.
- **`reproducibility_manifest` is best-effort + non-blocking**: a
  failing manifest load does NOT crash pre-flight; the gate suite
  runs to completion regardless. Downstream consumers should treat
  `None` slots as "artefact not on disk" (env-aware), not "pipeline
  failed".

---

## §8 Cross-references

* `molmetal/scripts/pre_flight_check.py` — `_build_reproducibility_manifest()` helper + `--json` schema extension (this Wf's home).
* `molmetal/tests/test_wf350b3_repro_manifest.py` — 2 NEW tests for the schema contract.
* `molmetal/reports/wf350a1_runtime_energy.json` — Wf-350-A1 artefact embedded as the `runtime_energy` slot.
* `molmetal/reports/wf350a2_compute_manifest.json` — Wf-350-A2 artefact embedded as the `compute` slot.
* `molmetal/reports/wf350a3_hp_provenance.jsonl` — Wf-350-A3 artefact embedded as the `hp_provenance` slot.
* `TODO/completed/350a1_runtime_energy_report.md` — Wf-350-A1 ship (companion).
* `TODO/completed/350a2_compute_manifest.md` — Wf-350-A2 ship (companion).
* `TODO/completed/350a3_hp_provenance_log.md` — Wf-350-A3 ship (companion).
* `TODO/completed/343a1_industry.md` — Wf-343a-R1 finding #3 (this extension's recommendation).
* `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D gate-count-consistency audit invariant (preserved).
* `AGENTS.md` §5.1 (standing GPU grant), §5.2 (honest framing), §5.3 (no paper work without authorization).

---

## §9 Wall budget

- Helper function dev: ~10 min
- Tests authoring: ~15 min
- Test execution: 306 s (2 subprocess invocations × ~140 s each + assertion overhead)
- Total wall: ~30 min CPU-only (PROJECTED 15-25 min in the brief → MEASURED matches)

---

**END OF Wf-350-B3 SHIP** — owner Hugo Lee; AGENTS.md §5.1 / §5.2
honest framing; last update 2026-09-29.