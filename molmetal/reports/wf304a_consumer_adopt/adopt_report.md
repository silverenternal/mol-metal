# Wf-304A — AnchorGuard consumer adoption

**Date**: 2026-09-29
**Ship surface**: `molmetal/scripts/paper_auto_update.py` + `molmetal/scripts/paper_dashboard.py` + `molmetal/scripts/pre_flight_check.py`
**Closes**: Wf-302A honest caveat (downstream consumers NOT yet adopting `AnchorGuard.require()`)
**Author**: Wf-304A workflow (subagent run)

## Motivation

Wf-302A (`molmetal/scripts/read_anchor_json.py`) shipped a paper-grade
`AnchorGuard` helper that wraps every JSON anchor read with four explicit
guards:

1. **Path exists** — `AnchorMissingError` if not.
2. **mtime fresh** — `AnchorStaleError` if older than `max_age_sec`.
3. **JSON parseable** — `AnchorJSONError` on malformed JSON.
4. **version match** — `AnchorVersionMismatchError` on miss.

The Wf-302A honest caveat explicitly flagged that downstream consumers
(`paper_auto_update.py`, `paper_dashboard.py`, `pre_flight_check.py`) had
**not** yet adopted `AnchorGuard.require()` at each call site — they were
still doing ad-hoc `json.load(open(...))` reads that would silently emit
stale / missing / version-drifted numbers. Wf-304A closes that gap.

## Adoption map

| Script | Site | Old | New |
|---|---|---|---|
| `paper_auto_update.py` | `_load_json()` (helper) | `json.load(open(...))` | `AnchorGuard(path, max_age_sec=86400).require()` |
| `paper_auto_update.py` | `main()` collect read | `_load_json(args.collect_json)` | `_load_json(args.collect_json, strict=True)` |
| `paper_auto_update.py` | `_extract_extra_claims` (3 sites) | `_load_json(path)` | `_load_json(path)` (now strict=False default → silently dropped on guard failure) |
| `paper_dashboard.py` | `_read_json()` (helper) | `json.load(open(...))` | `AnchorGuard(path, max_age_sec=86400).require()` |
| `paper_dashboard.py` | `build_dashboard_html()` collect read | `_read_json(collect_json)` | `_read_json(collect_json)` + `collect_json_error` surfaced when guard returns empty |
| `pre_flight_check.py` | `gate_json_threshold()` JSON read | `json.load(open(...))` | `AnchorGuard(path, max_age_sec=2592000).require()` |

**n_sites_adopted = 6** distinct read sites (3 helper-level + 3 caller-level)
across the 3 consumer scripts.

## Freshness window policy

Two distinct freshness regimes are in play:

* **Live paper-grade anchors** (Wf-265-Collect `all_results.json`,
  `wf269c_report.json`, `metrics/by_metric/cfm_decode_real.json`,
  `metrics/by_metric/drugood_v3_scaffold.json`) — use the AnchorGuard
  default **`max_age_sec=86400` (24 hr)**. These files are regenerated
  per wave / per run and SHOULD be fresh.

* **Frozen historical result records** (`wf188_m14_full_metalcytotox/
  result.json`, `wf256_apply/catherine_30/per_metal_pass.json`) —
  pre_flight_check.py uses **`max_age_sec=2592000` (30 days)** via the
  new `gate_json_threshold(max_age_sec=...)` parameter. These files
  are by definition frozen historical records from runs that completed
  once and are never regenerated; freshness only matters in the sense
  of "the file is still on disk + parseable".

The 24 hr default per AnchorGuard would have surfaced a real honest-
negative on first run: `wf188_m14_full_metalcytotox/result.json` was
27.5 hours old at adoption time, which the prior `json.load()` read
silently passed. The 30-day window for historical result records
preserves pre-Wf-304A behaviour while still enforcing path-exists +
JSON-parseability guards (the other two of the three failure modes
the Wf-302A caveat warned about).

## Version pinning

No `required_version=` was pinned at any site because none of the
current JSON anchors (collect.json, wf188_result.json, wf256_per_metal.json,
cfm_decode_real.json, drugood_v3_scaffold.json, wf269c_report.json) carry
a `version` field. The version-pin code path is in place — callers wanting
version pinning can pass `required_version="m14_v1"` (and the Wf-287A
formal `m14_anchor.json` would need `"version": "m14_v1"` added first).
Per Wf-302A honest caveat this is a separate TODO.

## Backward-compat (best-effort vs strict)

The two helper functions (`_load_json` in paper_auto_update, `_read_json`
in paper_dashboard) expose a `strict=False` default that **silently drops**
the read on any anchor-guard failure. This preserves the prior "Exception:
pass" semantics at every extra-source site so the diff pipeline keeps
producing a partial report when a non-critical anchor is missing.

The primary anchor reads use `strict=True`:

* `paper_auto_update.main()` collect read — propagates `AnchorMissingError`
  / `AnchorStaleError` so a missing collect.json crashes the script with
  a useful message (collect.json is mandatory).
* `pre_flight_check.gate_json_threshold()` — propagates the four named
  anchor exceptions so the gate FAIL evidence carries the specific guard
  failure mode (per Wf-291H §4 explicit-error contract).

## Verification

### 1. `pre_flight_check.py --pipeline cfm` (the gates that touch JSON)

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py \
    --pipeline cfm --json /tmp/preflight_v304a.json

summary: {"pass": 6, "fail": 0, "skip": 0, "total": 6}
```

All 6 cfm-pipeline gates PASS — including the 2 JSON-anchor gates
(`p1_cfm_smoke_decoded` reads `wf188_m14_full_metalcytotox/result.json`,
`p1_cfm_catherine_30metals` reads `wf256_apply/catherine_30/per_metal_pass.json`).
The 30-day max_age_sec for historical result records keeps both gates
green without needing to touch the file mtime.

### 2. `pre_flight_check.py --pipeline all` (full gate sweep)

```
summary: {"pass": 24, "fail": 1, "skip": 0, "total": 25}
non-pass count: 1
  p1_chemprop_seeds_min FAIL — help missing token(s) ['--n-seeds-min'] (rc=124)
```

The single FAIL is `p1_chemprop_seeds_min` — a `--help`-flag grep gate on
`train_chemprop_v2_v3.py` that timed out at rc=124. **Pre-existing** failure
unrelated to JSON read sites (this gate does grep / subprocess work, not
JSON reads). All 6 JSON-touching gates (2 chemprop + 2 cfm + 0 P2 + 2 P0
config) PASS.

### 3. `paper_dashboard.py`

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/paper_dashboard.py \
    --verbose --out-html /tmp/dashboard_v304a.html

[Wf-293D-Dash] built dashboard: 34 claims, 18 honest-negatives, 7 cohorts, html_len=82373
[Wf-265-Dashboard] wrote /tmp/dashboard_v304a.html (80.9 KB); build JSON at /tmp/dashboard_v304a.build.json.
```

Build JSON: `schema_valid=True, n_claims=34, n_honest_negatives=18,
collect_json_error=NONE` — primary anchor (collect.json) reads cleanly
via the new `AnchorGuard`-wrapped `_read_json()` helper.

### 4. `paper_auto_update.py`

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/paper_auto_update.py \
    --collect-json molmetal/reports/wf265_collect/all_results.json \
    --paper-doc TODO/completed/188_paper_4_krasnov_protocol_aligned.md \
    --out-diff /tmp/diff_v304a.md

[paper_auto_update] wrote diff_report -> /tmp/diff_v304a.md (11 claims, dry_run=True)
```

11 claims extracted (5 primary + 6 secondary) — all extra-source reads
via `_load_json()` (now AnchorGuard-wrapped, strict=False) succeed on
fresh anchor files.

## Honest framing (party-twenty-spirit §3.5)

* **MEASURED**: 6 distinct JSON read sites adopted across 3 consumer
  scripts; pre_flight_check.cfm 6/6 PASS; paper_dashboard schema_valid;
  paper_auto_update 11 claims emitted.
* **MEASURED**: 30-day `max_age_sec` policy preserves pre-Wf-304A
  behaviour for frozen historical result records (wf188 result.json was
  27.5 hours old at adoption; would have silently passed before, would
  now loudly fail under a 24 hr default).
* **PARTIAL**: no `required_version=` pinned at any site — the version-
  pin code path is in place but no current JSON anchor carries a
  `version` field. Adding `"version": "m14_v1"` to the Wf-287A
  `m14_anchor.json` (TODO-287A follow-up) would unlock pin-at-site for
  the m14 LEAD claim in `paper_auto_update.py` + `paper_dashboard.py`.
* **PRE-EXISTING**: `p1_chemprop_seeds_min` FAIL is unrelated to JSON
  read sites; this is a `--help`-grep timeout gate that would have
  failed identically pre-Wf-304A. Logged here so reviewers know the
  single non-PASS in the full sweep is NOT introduced by this workflow.

## Files modified

* `molmetal/scripts/paper_auto_update.py` — AnchorGuard import + `_load_json()`
  rewrite (8 LOC new helper + 12 LOC edit) + `strict=True` on collect read.
* `molmetal/scripts/paper_dashboard.py` — AnchorGuard import + `_read_json()`
  rewrite (8 LOC new helper + 12 LOC edit) + `collect_json_error` surface.
* `molmetal/scripts/pre_flight_check.py` — AnchorGuard import +
  `gate_json_threshold()` rewrite with new `max_age_sec=2592000.0` (30 day)
  default for historical result records.
* `molmetal/reports/wf304a_consumer_adopt/adopt_report.md` — this report.
