# Wf-A205 — the mechanical citability gate (no third caveat-less headline)

**Date:** 2026-11-02 · **Repo:** `/home/hugo/codes/mol-metal` · **HEAD:** `1b9470a0`
**Shipped:** `molmetal/tests/test_wfa205_citability_gate.py` (the gate) + this directory.
**Scope of this wave:** the gate only. No registry, metric, doc, trainer or ledger file was written.

> ⚠️ **CORRECTION (2026-11-02, read-only repair wave):** **3 of the 8 code citations**
> in `payload_direction_proof.json` pointed at the **WRONG line** of
> `molmetal/scripts/registry.py` — `1361 → 1362`, `2061 → 2056`, `2077 → 2072`.
> The cited **statements were all true**; only the **locators** moved. The
> 2026-11-02 rendering is preserved **VERBATIM** (AGENTS.md §5.5 rule #6); the
> correction is **additive and dated** — see **§3.1** below and
> `payload_direction_proof.json` → `corrections` + `citation_line_check`.

---

## 1. The rule as shipped

> **Every registry row with `quotable: true` AND `headline_grade: true` AND
> `registry_role: paper_line` must carry a non-empty `headline_caveat` in the
> registry row AND in its `source_json` payload on disk.**

A non-empty caveat means: the value is a `str` whose `.strip()` is non-empty —
the same test `assess_citability` applies (`registry.py:1245-1247`).

Five clauses, each with its own test:

| Clause | Statement | Test |
|---|---|---|
| **R4** | The governed set is **derived from code**, not hand-listed: `LINE_METRICS` (`registry.py:103-136`) must equal the set of rows carrying `registry_role: paper_line`. A new paper line is covered automatically. | `test_governed_scope_matches_line_metrics` |
| **R2** | Every governed registry row carries a non-empty `headline_caveat`. | `test_registry_row_carries_a_non_empty_caveat` |
| **R3** | Every governed row's **payload on disk** carries a non-empty `headline_caveat` — the durable direction (§4). | `test_payload_carries_the_caveat` |
| **R5** | For **every** row that is `quotable` + `headline_grade` — governed **or not** — the registry row's caveat must equal what `assess_citability` derives from the payload. A registry-only caveat is a failure, not a decoration. | `test_headline_rows_registry_caveat_equals_payload_caveat`, `test_no_headline_row_carries_a_registry_only_caveat` |
| **R5-converse** | A row that is `quotable` but **not** `headline_grade` needs **no** caveat, and the gate must not fire on it. | `test_quotable_but_not_headline_rows_need_no_caveat`, `test_a_soft_row_without_a_caveat_does_not_fail_the_gate` |

### Why the scope is `paper_line` and not every `quotable + headline_grade` row

The task's literal rule — *every* `quotable: true` AND `headline_grade: true` row —
**fails on the current registry.** MEASURED 2026-11-02 on
`registry/metrics.yaml` (sha256 `9ddd971eae8e333ed7bf169660da28f95ab500528e53d6a4eb1c17df1480e684`),
corpus = all 582 rows, match rule = `quotable is True AND headline_grade is True`
(record: `summary.json`, `row_census.json`):

| population | n | ids |
|---|---|---|
| registry rows | 582 | — |
| `quotable` | 79 | — |
| `quotable` + `headline_grade` | 6 | `m19_lomo`, `m2_r2_per_cell`, `m21_nci60_9cell`, `m2_r2_per_cohort`, `nci60_precious_9cell_auc`, `platinai_3cell_resistance_real_strat` |
| … of those, carry a caveat | 4 | `m19_lomo`, `m2_r2_per_cell`, `m21_nci60_9cell`, `m2_r2_per_cohort` |
| … of those, carry **no** caveat | **2** | `nci60_precious_9cell_auc`, `platinai_3cell_resistance_real_strat` |
| … of those, `registry_role: paper_line` | 3 | `m19_lomo`, `m2_r2_per_cell`, `m21_nci60_9cell` — **all three carry a caveat** |

Both caveat-less rows are `registry_role: artifact`. Shipping the unscoped rule would have
blocked a correct registry, so the rule was narrowed to the population the citability rule
actually governs — and the narrowing is *itself* machine-checked:

* `registry.py cite` — the report `docs/METRIC_CITABILITY_RULE.md` points the reader at —
  prints only paper-line rows: `if r.get("registry_role") != "paper_line": continue`
  (`molmetal/scripts/registry.py:2276-2277`).
* `docs/METRIC_CITABILITY_RULE.md` §1 defines the three marks over "one line per paper-line
  metric"; §7 (line 396) counts 11 paper-line metrics.
* `registry.py check` counts headline-grade over `paper_rows`
  (`molmetal/scripts/registry.py:1708-1721`).
* `registry.py` assigns `registry_role` from two build passes: `"paper_line"` for a
  `LINE_METRICS` member (`registry.py:1397`) and `"artifact"` for the disk-walk rows
  (`registry.py:1487`).
* Wf-A203's own census used the same population ("of the 11 `registry_role: paper_line` rows,
  EXACTLY TWO were … and carried NO `headline_caveat`" — `molmetal/tests/test_wfa203_citability_caveat.py`).

**R4 keeps the narrowing honest.** The governed set is re-derived from `LINE_METRICS` on every
run and compared against the registry. If a paper line is added to code and the registry is not
regenerated, the two sets drift and the gate fails. So the gate cannot silently stop watching a
row — and the moment a caveat-less artifact row is **promoted** to the paper surface, the gate
fires (bite 4, §5).

### The two out-of-scope rows are reported, not dropped

| metric_id | role | status | value | source_json | registry caveat | payload caveat |
|---|---|---|---|---|---|---|
| `nci60_precious_9cell_auc` | artifact | `MEASURED_SMOKE` | 0.6883250000000001 | `metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json` | absent | absent |
| `platinai_3cell_resistance_real_strat` | artifact | `MEASURED` | 0.6125442587251778 | `metrics/by_metric/platinai_3cell_resistance_real_strat.json` | absent | absent |

These are a **coverage census**, not a defect count (AGENTS.md §5.5 rule #6 keeps superseded
renderings verbatim, so a census is never a defect tally). The bound is recorded in
`summary.json` under `census_bound`. Both rows are still covered by **R5**: their registry
caveat must equal their payload caveat (both absent today → they agree), so neither may carry
a hand-injected registry-only caveat.

---

## 2. The registry parse

The gate reads the registry with **the same loader `registry.py` uses**:

```python
# molmetal/scripts/registry.py:255-261
def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
```

Key names read — the names `assess_citability` writes into the row
(`molmetal/scripts/registry.py:1241-1247`): `quotable`, `headline_grade`, `headline_caveat`;
plus `registry_role` and `source_json`.

**Speed note (MEASURED 2026-11-02):** the gate reads the 682 KB registry with PyYAML's
`CSafeLoader` (the C implementation of `SafeLoader`): 0.10 s vs 0.76 s for `yaml.safe_load`.
`test_loader_equivalence` asserts the two produce **the same document** on the real registry, so
the fast path cannot change what the gate sees.

---

## 3. The payload-direction proof

**Claim:** a `headline_caveat` that exists **only** in the registry row is **erased** by the next
`registry.py generate`. The payload is the durable direction.

**Proof from the code** (full record: `payload_direction_proof.json`):

| file:line | code | meaning |
|---|---|---|
| `molmetal/scripts/registry.py:1361` | `out: Dict[str, Any] = {}` | `build_metrics` starts from an **empty** dict — the metric layer is rebuilt from the payloads. There is **no** preservation pass for metrics (the preservation helper `_preserve_underived_datasets` is DATASETS-only, `registry.py:2022-2024`). |
| `molmetal/scripts/registry.py:1239` | `caveat = payload.get("headline_caveat")` | the **only** source of a row's `headline_caveat` is the payload. |
| `molmetal/scripts/registry.py:1245-1247` | `"headline_caveat": (caveat.strip() if isinstance(caveat, str) and caveat.strip() else None)` | a blank or non-string caveat becomes `None` — the same rule the gate enforces. |
| `molmetal/scripts/registry.py:1426-1429` | `rec.update(assess_citability(payload, rec["value"], _rel(src_path), tracked))` | `build_metrics` writes the row from `assess_citability`'s return value. |
| `molmetal/scripts/registry.py:2061` | `metrics_doc = { … "metrics": metrics, … }` | `generate` dumps those rows into the metrics document. |
| `molmetal/scripts/registry.py:2077-2079` | `text = yaml.safe_dump(doc, …)` → `atomic_write_file(path, text)` | and writes it over `registry/metrics.yaml`. |

### 3.1 CORRECTION (2026-11-02) — 3 of the 8 code citations were mis-located

**MEASURED 2026-11-02** by resolving every citation against
`molmetal/scripts/registry.py` **on disk** (working-tree sha256
`18fae08ab25dfb866dd857dc1a92654e4647d63b9b75f37a36f33004a568a438`, 2346 lines):

| cited | what that line actually reads | the real line | verdict |
|---|---|---|---|
| 1361 | `    """` — the **closing** line of `build_metrics`' docstring | **1362** = `    out: Dict[str, Any] = {}` | **WRONG by 1** |
| 2061 | `        "lines": list(LINE_METRICS.keys()),` — a *member* of the literal | **2056** = `    metrics_doc = {` | **WRONG by 5** |
| 2077 | `        if path.is_file() and _strip_generated_at(...) == …` — the no-op write guard | **2072** = `        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)` | **WRONG by 5** |

The other five citations (1239, 1245-1247, 1426-1429, 2276-2277, 103) and the
`atomic_write_file` reference (2079) resolve **correctly**. **The substantive
claims are unchanged** — `build_metrics` still starts from an empty dict,
`assess_citability` still reads the caveat from the payload, and `generate`
still writes the document with `atomic_write_file`. **Only the locators moved.**

The table above is preserved **VERBATIM** (AGENTS.md §5.5 rule #6). The
corrected citations are in `payload_direction_proof.json` → `corrections`
(3 dated entries, each carrying the superseded rendering and the corrected
line side by side) and → `citation_line_check` (the live resolution of all 8).

**Two further range citations in this README and in the gate's docstring were
also loose** (MEASURED the same way, same run):

| cited | the real range | note |
|---|---|---|
| `registry.py:103-136` | `registry.py:103-135` | 136 is the blank line after the `LINE_METRICS` literal |
| `registry.py:1364-1434` (the `LINE_METRICS` pass) | `registry.py:1366-1429` | the pass is `for _line, mapping in LINE_METRICS.items():` → `out[metric_id] = rec` |
| `registry.py:1436-1500` (the disk walk) | `registry.py:1435-1529` | the walk is `for src_path in sorted(BY_METRIC_DIR.glob("*.json")):` → `return out, covered_files` (1530) |

**WHY NOTHING CAUGHT IT (R2) — the repo HAS a locator auditor, and it cannot
see this class of citation.** `molmetal/scripts/wfa200_crosswave_locator_audit.py`
is the guard for exactly this failure mode, but its `LOCATOR_RE`
(`wfa200_crosswave_locator_audit.py:225`) requires **path-then-colon-then-digits
ADJACENCY**, and a machine-readable record that stores the two in **separate
fields** — `{"file": "molmetal/scripts/registry.py", "line": 1361}` — has **no
adjacency**; and `METADATA_PROBE_RE` lists `line` among the **metadata** probes,
so a `"line"` field is never classified as a claim.

**MEASURED 2026-11-02 by re-running it read-only on this tree**
(`--paths molmetal/reports/wfa205_annotation_reproducibility --gitignored-under …`):
census **7** files, **7** scanned, **53** locator hits, **1** `live_drift`.
**0 of the 10 hits attributed to `payload_direction_proof.json` come from the 8
structured records** — every one comes from a **prose string** (the `meaning`
text at line 8, the `registry_parse.loader` string at line 55) or from the
correction block's own `superseded_rendering_verbatim` / `corrected_to`
strings. The class breakdown is `RESOLVES-NO-CLAIM` 42 · `TARGET-ABSENT` 5 ·
`DATED-SUPERSESSION` 4 · `RESOLVES-BUT-DIFFERS` 1 · `RESOLVES-AND-MATCHES` 1.
⇒ **the 8 records the file exists to publish are still invisible to the
auditor, and the 3 rotted locators are still not flagged.** The auditor's one
`live_drift` is the known **FALSE POSITIVE**: the §1 bullet that cites
`registry.py:1397` **and** `registry.py:1487` in the same sentence — the
auditor took the probe belonging to the **second** claim (`"artifact"`) and
compared it against the **first** claim's line, which reads
`"registry_role": "paper_line",`. **Both citations are CORRECT on disk**
(`registry.py:1397` = `"registry_role": "paper_line",` ·
`registry.py:1487` = `"registry_role": "artifact",`). The auditor's own report
is a published MEASURED artefact and is **not** rewritten here. (The 5
`TARGET-ABSENT` hits are `/tmp` shadow paths parsed out of `bite_results.json`
— noise, not drift.)

**WHAT NOW BITES (added 2026-11-02, same wave):**

1. `evidence.py` **resolves every citation from the file by content** (an
   anchor block that must appear verbatim at the cited line) and **refuses to
   write** when a citation drifts and no dated correction covers the move.
2. `molmetal/tests/test_wfa205_citability_gate.py` gained
   `test_every_code_citation_points_at_a_line_that_carries_its_text` and
   `test_the_correction_block_preserves_the_superseded_rendering` — the guard
   that was missing. MEASURED: the first one **fails** on the pre-correction
   proof file and passes on the corrected one (the demonstration is in the
   test's own docstring).

**Proof by execution (MEASURED 2026-11-02):** for each of the 3 governed rows,
`assess_citability(payload, row["value"], row["source_json"], None)` — the exact call
`build_metrics` makes — rebuilds a `headline_caveat` byte-identical to the registry row's
(`test_assess_citability_reproduces_the_caveat_from_the_payload`). The proof record carries the
sha256 of the payload, of the registry caveat and of the payload caveat per row
(`payload_direction_proof.json` → `rows[]`).

**Proof by mechanism (bite 7):** `assess_citability` on a payload with no `headline_caveat`
returns `headline_caveat: None`; adding a string returns it; a whitespace-only string returns
`None`. A payload that does not carry the caveat **cannot** yield a row that does.

---

## 4. The bites — the gate fails when it should

Shadows are copies of `registry/metrics.yaml` written under a temp dir (`/tmp/wfa205_bite_*`),
never the repo's file; every shadow path is asserted to be outside the repo. Full output:
`bite_results.json`.

| # | scenario | mutation | result |
|---|---|---|---|
| 1 | `bite1_blank_caveat` | `m21_nci60_9cell.headline_caveat = ""` | **FAIL** (gate fired), pytest rc=1 |
| 2 | `bite2_whitespace_caveat` | `m21_nci60_9cell.headline_caveat = "   \n\t  \r\n "` | **FAIL** (gate fired), pytest rc=1 |
| 3 | `bite3_removed_caveat_key` | `del m21_nci60_9cell["headline_caveat"]` | **FAIL** (gate fired), pytest rc=1 |
| 4 | `bite4_promoted_artifact_row` | `nci60_precious_9cell_auc.registry_role = "paper_line"` (the caveat-less row enters the paper surface) | **FAIL** (gate fired), pytest rc=1 |
| 5 | `bite5_registry_only_caveat` | the caveat lives only in the registry row, not in the payload | **FAIL** (gate fired), pytest rc=1 |
| 6 | `control_unmutated` | the unmutated copy | **PASS** (gate green), pytest rc=0 |

Exact failure output (the long caveat text is elided here; it is verbatim in `bite_results.json`):

```
============ bite1_blank_caveat ============
E  AssertionError: gate violations:
E    m21_nci60_9cell: registry_caveat_missing: quotable + headline_grade +
E    registry_role=paper_line, but the registry row's 'headline_caveat' is
E    absent/blank (raw='')
E    m21_nci60_9cell: caveat_not_durable: registry 'headline_caveat'=None !=
E    payload 'headline_caveat'="0.7294 is the 9-cell panel mean and it clears
E    this metric's own 0.7 gate. But DU145 (n_test=7) and SKOV3 (n_test=3, …)" …

============ bite2_whitespace_caveat ============
E  AssertionError: gate violations:
E    m21_nci60_9cell: registry_caveat_missing: … absent/blank (raw='   \n\t  \r\n ')

============ bite3_removed_caveat_key ============
E  AssertionError: gate violations:
E    m21_nci60_9cell: registry_caveat_missing: … absent/blank (raw=None)

============ bite4_promoted_artifact_row ============
E  AssertionError: gate violations:
E    nci60_precious_9cell_auc: registry_caveat_missing: quotable +
E    headline_grade + registry_role=paper_line, but the registry row's
E    'headline_caveat' is absent/blank (raw=None)
E    nci60_precious_9cell_auc: payload_caveat_missing: the payload
E    'metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json' carries no
E    'headline_caveat'; a regenerate would erase any registry-only caveat, so the
E    disclosure is not durable

============ bite5_registry_only_caveat ============
E  AssertionError: gate violations:
E    m21_nci60_9cell: caveat_not_durable: registry
E    'headline_caveat'='hand-injected registry-only caveat: not in the payload,
E    so a regenerate erases it' != payload 'headline_caveat'="0.7294 is the
E    9-cell panel mean and …"

============ control_unmutated ============
1 passed in 0.23s
```

Bite 5 is the payload-direction bite: the registry row *does* carry a non-empty string, so R2
passes — but R5 fires, because the caveat is not reproducible from the payload and a regenerate
would erase it.

---

## 5. The row list

### 5.1 The governed rows (3) — the ones the gate requires a caveat of

| metric_id | status | value | source_json | registry caveat | payload caveat |
|---|---|---|---|---|---|
| `m19_lomo` | `MEASURED_BELOW_TARGET_BASELINE` | 0.5575471269535148 | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | present | present |
| `m21_nci60_9cell` | `MEASURED_HONEST_NEGATIVE` | 0.7293802091886383 | `metrics/by_metric/nci60_precious_9cell_auc.json` | present | present |
| `m2_r2_per_cell` | `MEASURED` | −0.20452201768910916 | `metrics/by_metric/m2_r2_real_4cell.json` | present | present |

### 5.2 The converse — `quotable: true`, `headline_grade: false` (73 rows): **no caveat required**

The 5 paper-line rows in this set, showing that the gate does not fire on them:

| metric_id | caveat | why it is not headline-grade (the registry's own `not_headline` reason) |
|---|---|---|
| `m14_decode_ratio` | **absent** | `status 'MEASURED_BUT_NOT_A_QUALITY_METRIC': the file itself declines the claim` |
| `m7_diversity` | **absent** | `status 'MEASURED_NOT_A_CLAIM': the file itself declines the claim`; + `definition was changed (REDEFINED); values measured before the change are not comparable` |
| `m1_ru_auc` | present | `single run: quote as a single-run measurement, not a table mean` |
| `l6_closure` | **absent** | `single run: quote as a single-run measurement, not a table mean` |
| `m6_kappa` | **absent** | `single run: quote as a single-run measurement, not a table mean` |

Four of the five paper-line soft rows carry **no** caveat today and the gate is still green —
the rule bites a table-grade row only. The other 68 are `registry_role: artifact`. The full
73-row list is in `row_census.json` (every one of the 582 registry rows is recorded there with
the five columns the gate reads).

---

## 6. The test result

```
$ CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa205_citability_gate.py -q
29 passed, 1 warning in 4.96s
```

CPU-only; `torch` is never imported by the gate; the file writes nothing under `registry/` or
`metrics/`. Runtime ≈ 5 s.

---

## 7. Residuals — reported, NOT fixed

These are outside this wave's write scope (or are registry-side changes this wave may not make).
None of them is silently absorbed by the gate.

1. **`docs/METRIC_CITABILITY_RULE.md:288-292` still calls the caveat optional.**
   Line 288 is the heading `### Optional disclosure field`; line 290 reads
   *"`headline_caveat` (string) is printed under the row whatever the mark. Use it when the number
   is table-grade but a reader needs one sentence of context"*. The doc now **understates** the
   rule: for a paper-line, table-grade row the caveat is **mandatory**, and it must live in the
   payload. **Proposed wording** (not applied — `docs/**` is not in this wave's write scope):
   *"### Disclosure field — mandatory for a paper-line table-grade row. `headline_caveat`
   (string) is printed under the row whatever the mark. A row that is `quotable` **and**
   `headline_grade` **and** `registry_role: paper_line` **must** carry a non-empty
   `headline_caveat`, and it must live in the metric payload under `metrics/by_metric/`, not
   only in the registry row — `registry.py generate` rebuilds every metric row from the payload
   and would erase a registry-only caveat. The mechanical check is
   `molmetal/tests/test_wfa205_citability_gate.py`."*

2. **`registry.py` has no hook for this rule.** The gate lives in the test suite, so it runs under
   pytest but **not** under `registry.py check`. A registry-side hook (e.g. a
   `p3_headline_caveat_check` in `pre_flight_check.py`, or a new line in `check_metrics`) would
   put the rule in the pre-flight gate. **Not done here**: `molmetal/scripts/registry.py` and
   `molmetal/scripts/pre_flight_check.py` are outside this wave's write scope. Report only.

3. **`MEASURED_SMOKE` can be `headline_grade`.** `_NON_HEADLINE_STATUSES`
   (`registry.py:818-821`) contains only `MEASURED_NOT_A_CLAIM` and
   `MEASURED_BUT_NOT_A_QUALITY_METRIC`, so `nci60_precious_9cell_auc` — a **3-seed, 5-epoch
   smoke run** with value 0.6883250000000001, a family sibling of the paper line
   `m21_nci60_9cell` (0.7293802091886383) — is quotable **and** headline-grade. A reader who
   met the two numbers in the same family could mistake one for the other. This is a
   **registry-predicate** gap, not a citability gap: the fix belongs in
   `assess_citability`/`_NON_HEADLINE_STATUSES`, which this wave may not edit. Report only.
   (The gate still catches the moment such a row reaches the paper surface — bite 4.)

4. **`docs/METRIC_CITABILITY_RULE.md` §7 (line 396, "State (2026-10-21, MEASURED)") is a
   dated snapshot** and no longer matches the registry: it lists `[CITE] = 2` and puts
   `m2_r2_per_cell` in `[NO ]`, while the registry now has **3** paper-line headline-grade rows
   (`m19_lomo`, `m21_nci60_9cell`, `m2_r2_per_cell`). It is explicitly dated, so it is a
   historical statement rather than a live claim — **not** amended here (`docs/**` is out of
   scope).

---

## 8. Reproduction

```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest molmetal/tests/test_wfa205_citability_gate.py -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/reports/wfa205_annotation_reproducibility/gate/bite_demo.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/reports/wfa205_annotation_reproducibility/gate/evidence.py
```

---

## 9. Files in this directory

| file | what it is |
|---|---|
| `README.md` | this report |
| `bite_demo.py` | runs the 5 bites + the control against shadow copies and captures the exact pytest output |
| `bite_results.json` | the machine-readable bite record (verdict, shadow path, exact failure output) |
| `evidence.py` | emits the census + the payload-direction proof (atomic writes) |
| `row_census.json` | all 582 registry rows with the five columns the gate reads, plus the payload-side caveat flag |
| `payload_direction_proof.json` | the code citations + the per-row sha256 and `assess_citability` reproduction |
| `summary.json` | the counts + the census bound |
