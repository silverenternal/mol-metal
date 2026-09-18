# WF-Decisions Template — user reply form

**Date:** 2026-09-14
**Companion to:** `molmetal/reports/wf_decisions_summary.md`
**Purpose:** a single text block the user can fill in and reply with.
**How to use:** for each decision, keep the option letter that matches your call (or replace with `Other` + a one-line rationale). Paste the whole block back. Claude proceeds.

---

## Reply template

```
D6              = (a)            # REINVENT4 install path: (a) separate venv [recommended] / (b) shared molmetal venv / (c) Docker
D7              = (c)            # Vina→QVina swap: (a) keep Vina 1.2.7 default / (b) flip to QuickVina 2 default / (c) both engines in headline table [recommended]
test_005        = (a)            # 4RN0 ASP B101 cohort: (a) keep labelled "modeled atoms" [recommended] / (b) drop entirely
cite-only-SOTA  = (a)            # Cite-only SOTA column: (a) ship column with per-row footnote [recommended] / (b) measured rows only
MW-range        = (c)            # MW-range flag: (a) Lipinski MW only / (b) IV 300-700 Da only / (c) both side-by-side [recommended]
journal         = (a) defer      # Round-13 journal: (a) Digital Discovery [recommended default] / (b) JCIM / (c) Patterns / (d) Briefings in Bioinformatics
                                  # or write "journal = (b) defer to Round-12" / "journal = (c) defer to Round-12" / etc.
                                  # or write "journal = defer to Round-12; default (a) unless metal-anticancer benchmark dominates"
```

### Notes / overrides (optional)

```
# Optional free-form overrides (one line per decision if needed):
# D6 = Other: <reason>
# D7 = Other: <reason>
# test_005 = Other: <reason>
# cite-only-SOTA = Other: <reason>
# MW-range = Other: <reason>
# journal = Other: <reason>
```

---

## Cross-references

- `TODO/pending/19_user_decisions.md` — full per-decision evidence and rationale
- `TODO/pending/20_post_r10_r11_action_plan.md` — 4-action plan, D7 evidence (r=0.9983 / n=46 paired)
- `molmetal/reports/round11_engine_parity_n50.md` — N=50 parity experiment
- `molmetal/reports/round10_e2e_pt_cfg_vina.md` — Round-10 Pt(II) prior end-to-end
- `molmetal/reports/anticancer_vs_general_metrics_survey.md` — IV MW 300-700 + logP/TPSA/RotB recommendations
- `molmetal/reports/ultracode_audit/PROJECT_STATUS.md` §8 — decision status table
- `TODO/pending/decisions.md` — D1-D5 already approved; D6/D7 reproduced in `19_user_decisions.md`
