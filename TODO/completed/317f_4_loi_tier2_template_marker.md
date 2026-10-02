# Wf-317F — 4 LOI pending files Tier-2 marker (historical reference only)

**Date**: 2026-09-29
**Workflow**: Wf-317F (per user directive relayed through harness)
**Status**: ✅ SHIPPED — Tier-2 marker banner added to 4 LOI pending files
**Owner**: Hugo Lee (per user directive)
**Driver**: User-originated request: "我们发二区就可以了，不用规划湿实验"
**Parent directive**: `AGENTS.md` §5.4 (2026-09-29 snapshot)

---

## §0 Why this doc exists

Per `AGENTS.md` §5.4 (2026-09-29 user directive), the project re-orients from
Tier-1 (JCIM) to Tier-2 (JCAMD / J. Mol. Graph. Model. / Comput. Chem. WIREs /
Mol. Divers. / Comput. Biol. Chem.) journal target. Tier-2 acceptance bar is
**computational-only**; wet-lab validation is a 6–12 month external-dependency
that does NOT gate Tier-2 acceptance.

As a consequence, the 4 institution-tailored wet-lab **Letter of Intent (LOI)**
drafts (Rusanov 2025 PlatinAI / PKU cisplatin IC50 / 协和 NCI-60 / ORNL
protein-metal binding) are **DEPRECATED for project scope**. They remain on
disk as **historical reference** — not for sending.

This Wf-317F pass adds a clear **Tier-2 marker banner** to each LOI so that
future agents (and the human reviewer) immediately see that sending is
deferred, and the LOIs are reference-only.

---

## §2 Files updated

| # | Path | Banner added |
|---|---|---|
| 1 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_loi_rusanov_2025_platinai.md` | ✅ Tier-2 marker inserted after Owner line (line 9) |
| 2 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | ✅ Tier-2 marker inserted after Owner line (line 9) |
| 3 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_loi_xiehe_nci60.md` | ✅ Tier-2 marker inserted after Owner line (line 9) |
| 4 | `/home/hugo/codes/mol-metal/TODO/pending/wetlab_loi_ornl_protein_metal.md` | ✅ Tier-2 marker inserted after Owner line (line 9) |

**n_lois_updated = 4.**

Banner text (verbatim, identical on all 4 files):

```markdown
> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)**
>
> Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
> Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
> This LOI draft is preserved as historical reference but NOT for sending.
```

Placement: directly after the `**Owner:** Hugo Lee ...` metadata line, before
the first `---` horizontal rule and `## 1. Purpose` / `## 1. 目的` section.
This positions the banner as the last item of the document header block — most
visible at top of file while preserving the original title + ID + Version +
Status + Target PI + Authoring date + Owner metadata structure.

---

## §3 Cross-reference to AGENTS.md §5.4 + Wf-317G

This Wf-317F pass is the LOI counterpart to **Wf-317G** (closeout:
`TODO/completed/317g_wetlab_sop_tier2_marker.md`) which marked the 3 wet-lab
SOPs + IC50 pipeline template. Together they form a complete Tier-2 scope-cut
of all 7 wet-lab pending items (3 SOPs + IC50 pipeline + 4 LOIs).

Per `AGENTS.md` §5.4 (Tier-2 journal target + NO wet-lab planning, 2026-09-29
user directive):

> "Wf-74 v2 wet-lab outreach → DEPRECATED."

The 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) all fall under the Wf-74 v2
deprecation umbrella. The banner added by Wf-317F prevents accidental send
while preserving the institutional-tailored outreach artifacts for paper §6
honest-limitations or §7 future-work citation.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **No LOI sent.** All 4 files remain **DRAFT — not sent at this revision**
  state (per the original §0 Status line in each file). The Tier-2 banner
  reinforces this by explicitly marking the file as historical reference.
- **No wet-lab data collected.** All 4 LOIs describe upstream wet-lab validation
  work that is now OUT OF SCOPE.
- **No paper §5 claim.** The 4 LOI drafts are reference-only; paper §5
  cross-validation gate is OUT OF SCOPE for Tier-2.
- **TODO-84 paper rule still applies** for `paper/*` writes; the LOIs are in
  `TODO/pending/`, NOT in `paper/*`, so this Wf-317F pass is paper-rule-clean.
- **Tier-2 metric lift is the in-scope deliverable.** Per `AGENTS.md` §5.4,
  every `metrics/by_metric/*.json` and `TODO/completed/*.md` is Tier-2-shippable
  as is.

---

## §5 Pre-flight verification

**Pre-flight v2 status: 48/48 PASS, 0 FAIL, 0 SKIP** (wall_sec = 309.76s).
Run at `/tmp/wf317f_preflight.log` (full JSON output captured 2026-09-29
02:28:07 UTC). Per `AGENTS.md` §4, pre-flight v2 25/25 PASS gate covers:
GPU env probe, anchor JSON, chemprop v2/v3 indices fix, chemprop kendall
optimizer, CFM training, P2 25/25, all 113 metrics files, R/Wf ledger, etc.
All 48 gates PASS — including the `p3_loi_send_ready` check (PASS:
"4/4 LOI drafts ready-to-send + review helper present") which validates the
4 LOI draft structure still exists on disk in DRAFT state.

Per `AGENTS.md` §4 scope rule, pre-flight is required for **retrain / panel /
ablation launch**. Wf-317F is a doc-marker edit (no GPU launch, no model
retrain, no panel re-emission); pre-flight is recorded here as the
2026-09-29 snapshot per Wf-317G convention.

---

## §6 Revision history

| Rev | Date | Author | Notes |
|---|---|---|---|
| 1.0 | 2026-09-29 | Wf-317F | Initial Tier-2 marker banner pass on 4 LOI pending files (Rusanov / PKU / 协和 / ORNL). Per user directive relayed through harness + `AGENTS.md` §5.4. Pre-flight v2 48/48 PASS logged. |