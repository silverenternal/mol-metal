# Wf-305A — PlatinAI Path Switching (S7 stratified extension + S6 Rusanov outreach draft)

**Date:** 2026-09-29
**Workflow:** Wf-305A (TODO-119 P1.3 path-switching follow-up)
**Trigger:** Wf-304B BLOCK_CF + Wf-301C stratified bias (A549=100, others=0)
**Owner:** Hugo Lee + ultracode
**Status:** **PARTIAL — S7 BLOCK_RATE_LIMIT; S6 LOI drafted**

---

## §0 TL;DR

- **S7 (MB-Finder stratified)** — **BLOCK_RATE_LIMIT** on MB-Finder API. First Wf-305A run (2026-09-29 01:35 UTC) hit a soft rate-limit (`success=false, retry_after=1467s` ≈ 24 min) on the very first `get-count` probe. The MB-Finder team is throttling our IP for repeated Wf-295B / Wf-301C / Wf-305A bursts. **Per-cell-line counts: ALL ZERO** (no Pt compounds harvested). **n_total=0**.
- **S6 (Rusanov outreach)** — drafted NEW LOI at `TODO/pending/wetlab_loi_rusanov_2025_platinai.md`. Subject: "PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)". **0 sent** — awaiting user OK per TODO-84.
- **Honest framing**: per Wf-297B §6 + Wf-304B honest-negative + this Wf's BLOCK_RATE_LIMIT, **S7 coverage-only is NOT a substitute for S6 NUMERIC outreach**. Both paths are needed in parallel; S6 is the binding constraint.

---

## §1 S7 — MB-Finder stratified extension

### §1.1 Wf-301C stratified bias — root cause

Previous Wf-301C run (`n_target_per_cell=100, cell_lines=A549,MCF7,A2780,A2780cis, start_page=850`) returned:

```json
"coverage": {"mcf7": 0, "a549": 100, "a2780": 0, "a2780cis": 0}
```

This means: **MB-Finder pages 850+ in the Pt corpus are A549-monoculture**. Compounds in this page range have bio-data ONLY for A549 (none for MCF7/A2780/A2780cis). The stratified mode correctly stopped after A549 hit 100 because the early stopping condition `all(per_cell_count[cl] >= n_target_per_cell)` is satisfied by A549=100 (degenerate) — wait, actually the loop continues as long as NOT all targets are met. Let me re-check...

Re-reading the loader code (lines 224-265): the loop continues until ALL cell lines reach n_target OR `page > max_pages`. Since A549 reached 100 first but MCF7/A2780/A2780cis were 0, the loop should have continued. But the loop also has a per-page filter: only Pt compounds are kept (`imageId.startswith("MB-Pt-")`), and only those with `has_<line>=1` contribute to per-cell counts.

So the actual root cause: **pages 850+ in MB-Finder's Pt corpus contain very few compounds with MCF7/A2780/A2780cis bio-data**. The stratified fetch scanned 20 pages × 5 compounds/page = 100 Pt compounds, all of which happened to be A549-only.

### §1.2 Wf-305A fix — start from page 1

`--start-page 1` scans earlier MB-Finder pages where compound IDs (lower `mb_id`) may have different cell-line distributions. Adding `--max-pages 400` caps walltime at ~25 min.

### §1.3 Loader changes (this Wf)

Two improvements (no breaking changes):

1. **`--max-pages` CLI flag** (workflow tractability cap):

```diff
+    parser.add_argument("--max-pages", type=int, default=1547,
+                        help="Maximum pages to scan before stopping "
+                             "(stratified mode; default 1547 = full Pt corpus)")
```

2. **Soft rate-limit detection in `_post`** (NEW):

```diff
+    if r.status_code == 200:
+        try:
+            payload = r.json()
+        except ValueError:
+            # Non-JSON 200 (e.g. CF challenge HTML) — treat as soft block
+            print(f"      [warn] {endpoint}: non-JSON 200 response ...")
+            return None
+        if isinstance(payload, dict) and payload.get("success") is False:
+            msg = payload.get("message", "")
+            retry_after = payload.get("retry_after", "?")
+            print(f"      [BLOCK_RATE_LIMIT] {endpoint}: {msg} "
+                  f"(retry_after={retry_after}s)", flush=True)
+            return None
+        return payload
```

This prevents the loader from wasting 400 pages × 2.5s = 1000s on guaranteed-empty responses after MB-Finder applies a soft rate-limit.

### §1.4 Stratified fetch command

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/platinai_real_label_loader.py \
    --cell-lines A549,MCF7,A2780,A2780cis \
    --n-target-per-cell 100 \
    --start-page 1 --delay 2.5 \
    --max-pages 400 \
    --out .data/molmetal/platinai_4cell_mbfinder_full.csv
```

### §1.5 Results (2026-09-29 01:35-01:36 UTC)

**BLOCK_RATE_LIMIT** — first `get-count` probe returned:

```json
{
  "success": false,
  "message": "You've made too many requests in a short time. Please wait a moment and try again. If you have a legitimate need for higher-volume or bulk access, we're happy to help — please get in touch via our Contact page at https://mb-finder.org/contact.",
  "contact_url": "https://mb-finder.org/contact",
  "retry_after": 1467
}
```

`retry_after=1467s` ≈ **24 min throttle window**. Process killed after 1st page request to avoid wasting 400 pages of guaranteed-empty responses.

| Metric | Value |
|---|---|
| **per_cell_line_count** | `{A549: 0, MCF7: 0, A2780: 0, A2780cis: 0}` (all zero — no harvest) |
| **n_total** | **0** Pt compounds harvested |
| **stratified_all_targets_met** | `False` |
| **walltime** | ~60 s (1 probe + 1 page attempt + kill) |
| **API requests** | 2 (both soft-blocked) |

### §1.5.1 MB-Finder API contact path

The MB-Finder team's `success=false` message suggests contacting them via https://mb-finder.org/contact for higher-volume access. This is a different path from S6 (Rusanov outreach) — it asks the MB-Finder platform maintainers rather than the PlatinAI paper authors. Worth investigating in a future Wf if S7 retries continue to fail.

### §1.6 Honest framing

- Even if Wf-305A stratified extension succeeds (A549/MCF7/A2780/A2780cis ≥ 100 each), the data is STILL **coverage matrix** not NUMERIC IC50. Per-cell-line pIC50 model training requires NUMERIC values.
- S7 coverage-only is a **ranked-candidate selector** (which 200 compounds to ask Rusanov for), NOT a substitute for S6 outreach.
- Both paths ship in parallel; S6 is the binding constraint.

---

## §2 S6 — Rusanov outreach draft

### §2.1 New file

`TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (LOI-RUSANOV-2026-09, rev 1.0, 16 sections)

### §2.2 Subject line

> "PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)"

### §2.3 What we ask

- Per-cell-line NUMERIC IC50 for ~200 Pt compounds spanning MCF7/A549/A2780/A2780cis (selected from Wf-295B + Wf-305A coverage matrix).
- Direct author-to-author email transfer of Rusanov 2025 Table S4-S5 supplementary xlsx.

### §2.4 What we offer

- Mid-author position on Mol-Metal paper §5 + prominent Acknowledgement naming Rusanov 2025 / MB-Finder team.
- Co-corresponding authorship if NUMERIC labels drive a published D-MPNN retrain beating synthetic-label baseline by ≥ 0.1 log units on held-out MCF7 + A2780cis.
- CC-BY-4.0 release of the resulting real-label dataset under `molmetal/data/platinai_real_ic50.csv`.
- No funding exchange; no exclusivity clause.

### §2.5 Status

- **DRAFT — 0 LOI sent at this revision**.
- **Sending requires user OK per `TODO/pending/84_no_paper_task_without_authorization.md`** (outreach is NOT gated by TODO-84, but sending cold email is the user's responsibility).

### §2.6 60-day cadence

Day 0 (send) → Day +7 (light follow-up) → Day +21 (second follow-up) → Day +28 (intro Zoom if requested) → Day +60 (NUMERIC in hand, Phase 2 launches).

---

## §3 Path-Switching Decision Matrix

| Scenario | S7 stratified status | S6 outreach status | Action |
|---|---|---|---|
| Wf-305A ≥ 100 all 4 cell lines | OK | LOI drafted, 0 sent | Send LOI immediately with concrete compound list |
| Wf-305A ≥ 100 only A549+MCF7 | PARTIAL | LOI drafted, 0 sent | Send LOI with smaller compound list (~100), honest framing of partial coverage |
| Wf-305A < 100 ≥ 2 cell lines | INSUFFICIENT | LOI drafted, 0 sent | Send LOI anyway (NUMERIC is the binding constraint; coverage matrix is selector) |
| Wf-305A fails entirely (network / API change) | FAIL | LOI drafted, 0 sent | Send LOI as PRIMARY path; document S7 BLOCK_CF extension |
| User does NOT OK LOI send | n/a | BLOCKED | Fall back to Wf-295B coverage-only + synthetic pIC50 (Wf-278B) for R20-P2 |

---

## §4 Cross-references

- `molmetal/scripts/platinai_real_label_loader.py` — Wf-295B + Wf-301C + Wf-305A stratified loader (this Wf adds `--max-pages` CLI flag)
- `molmetal/reports/wf304b_selenium/selenium_report.md` — S1 BLOCK_CF honest-negative
- `molmetal/reports/wf295b_platinai_loader/loader_report.md` — S7 coverage-only ship
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` — §3 strategy matrix
- `molmetal/reports/wf301c_stratified_platinai/` — Wf-301C stratified bias root cause
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` — THIS Wf's S6 outreach LOI draft (NEW)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` — parallel ORNL LOI (binding validation)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` — parallel PKU LOI (cisplatin synthesis + IC50)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` — parallel Xiehe LOI (NCI-60 expansion)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` — R20-P6 wet-lab IC50 plan
- `TODO/pending/84_no_paper_task_without_authorization.md` — paper rule; outreach NOT gated

---

## §5 Revision history

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-29 | Wf-305A | S7 BLOCK_RATE_LIMIT (`retry_after=1467s`) + S6 Rusanov outreach LOI drafted (0 sent, awaiting user OK). Loader patched with `--max-pages` CLI flag + soft rate-limit detection in `_post`. |
