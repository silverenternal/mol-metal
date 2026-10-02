# Wf-302C — Text-Only Task Brief Template Report

## What was produced

A text-only task brief template plus three concrete example briefs
(bug fix, audit, synthesis). All artifacts live under
`molmetal/reports/wf302c_text_briefs/` and are plain Markdown.

## Files

| Path | Lines | Purpose |
|---|---|---|
| `molmetal/reports/wf302c_text_briefs/template.md` | 122 | Main template (§0 TL;DR, §1 template, §2 worked example, §3 pitfalls) |
| `molmetal/reports/wf302c_text_briefs/example_bug_fix.md` | 31 | Bug-fix brief example (WF-303 m6_kappa restoration) |
| `molmetal/reports/wf302c_text_briefs/example_audit.md` | 33 | Audit brief example (WF-304 Wf-291C regression sweep) |
| `molmetal/reports/wf302c_text_briefs/example_synthesis.md` | 34 | Synthesis brief example (WF-305 lock-in to project standard) |
| `molmetal/reports/wf302c_text_briefs/template_report.md` | this | Report (plain text summary) |

## Lines per template (counting the three example briefs)

- example_bug_fix.md: 31
- example_audit.md: 33
- example_synthesis.md: 34

## Adoption checklist (for downstream workflows)

1. Replace every StructuredOutput spec in Wf-298E follow-ups with the
   `template.md` block.
2. Audit (Wf-304) the metric producers that Wf-291C touched and queue
   restorations under Wf-303+.
3. Synthesis (Wf-305) locks the insight into TODO/03_plans/LAMBDA_ROADMAP.md
   so future R-round kickoffs default to text briefs.

## Honest framing note

- MEASURED: counts above come from `wc -l` on the artifacts.
- PROJECTED: downstream Wf-303/304/305 are seeded by this report, not yet
  executed.
- BLOCKED: deeper adoption into TODO/INDEX.md awaits reviewer bandwidth
  for the LAMBDA_ROADMAP.md update.