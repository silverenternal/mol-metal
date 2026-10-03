# Wf-317E — LAMBDA_ROADMAP Tier-2 Update (R22-P1 computational-only, R23/R24/R25 DEPRECATED)

**Date**: 2026-09-29
**Wf**: Wf-317E
**Status**: SHIPPED
**Owner**: Hugo Lee + ultracode agent
**Pre-flight**: PASS (52/52, 0 fail, 0 skip per `molmetal/scripts/pre_flight_check.py` 2026-09-29)
**MEASURED-vs-PROJECTED tag discipline**: ✅ ALL edits are scope-re-scoping documentation only; no new MEASURED numbers introduced.

---

## §0. Trigger

User directive 2026-09-29 (relayed via Wf-317 harness, verbatim):

> "哦对了，我们发二区就可以了，不用规划湿实验"
> ("Oh right, we just need to submit to a Tier-2 journal, no need to plan wet-lab experiments.")

**Interpretation**:
1. **Submission target**: Tier-2 journal (Digital Discovery / J. Cheminform., IF 5-8) — NOT tier-1 (Nature MI / Nat. Comput. Sci., IF 14-25).
2. **No wet-lab planning**: All R23 Phase I + R24 Phase II + R25 Phase III clinical trial phases = OUT OF SCOPE going forward.

---

## §1. Edits to `TODO/03_plans/LAMBDA_ROADMAP.md`

### §1.1 §1.6 R20 honesty framing — add Tier-2 line

Added a 4th bullet to the existing §1.6 honesty-framing block:

> - **Tier-2 submission target, NO wet-lab planning (per user 2026-09-29 directive "我们发二区就可以了，不用规划湿实验")**: R20-R25 scope = computational-only (m3 Vina / m7 Diversity / m4 Success Rate / m16 Affinity / ASTSR R² / closure theorem / MCTS variance all stay computational); R22-P1 m18 multi-modal Protein-Metal binding re-scoped to computational-only (AlphaFold-Multimer + MetalSite in-silico, no wet-lab validation); R23 Phase I clinical trial + R24 Phase II + R25 Phase III + IND filing + GLP tox + DSMB + patient cohort + FDA/NMPA approval all **DEPRECATED** (no further planning); paper trajectory §4.3 R20 = Digital Discovery / J. Cheminform. tier-2 (IF 5-8) instead of Nat. Comput. Sci. tier-1.

### §1.2 §3.1 R22-R25 strategic 4-phase table — DEPRECATED markers

Added a top-of-section scope-update banner and DEPRECATED tags on R23, R24, R25 rows (strikethrough on the 4-column "key output" / "Lambda 端 ship" cells, ⛔ DEPRECATED prefix on the "phase" cell). R22 row updated to "Phase B/C tier-1 真实突破 (computational-only)".

### §1.3 §3.2 R22-P1 m18 — re-scoped to computational-only

- Header line: appended "(computational-only, no wet-lab per 2026-09-29 user directive)".
- 目标 paragraph: explicit "**all computational-only**, no wet-lab validation. m18 tier-1 lift uses AlphaFold-Multimer predicted structures + MetalSite computational binding-site prediction + in-silico docking + virtual screening cohort, not wet-lab IC50/MTT/SRB assay."
- R22-P1 row: marked with ⭐ + parenthetical "(**computational-only**: AlphaFold-Multimer + MetalSite in-silico + virtual m7 docking panel; **NO wet-lab MTT/SRB/IC50 validation per 2026-09-29 user directive**)".
- R22-P3 row: parenthetical "(computational TCGA + in-silico organoid proxy; no wet-lab)".
- R22-P6 row: paper target updated from "Paper v3 (Nature MI / Nat. Comput. Sci.)" → "Paper v3 (Digital Discovery / J. Cheminform. **tier-2**, IF 5-8)".

### §1.4 §3.3 R23 Phase I clinical — DEPRECATED

- Header line: "⛔ DEPRECATED — clinical trial 注册 + Phase I (no further planning per 2026-09-29 user directive ...)".
- 目标 paragraph: wrapped in > blockquote with ⚠️ DEPRECATED 2026-09-29 + "OUT OF SCOPE" + explicit list of do-NOT actions ("do not allocate Wf-57..Wf-62, do not recruit wet-lab / clinical PI, do not seek ORNL/PKU/sponsor / GMP facility / IRB approval").
- Phase table: 6 rows (R23-P1..P6) all struck through + ⛔ DEPRECATED suffix.
- 关键 output block: 4 bullets all struck through.

### §1.5 §3.4 R24 Phase II/III — DEPRECATED

- Header line: "⛔ DEPRECATED — Phase II/III + 算法 tier-1 长期演化 (no further planning per 2026-09-29 user directive ...)".
- 目标 paragraph: R24-P1/P2 fully clinical = OUT OF SCOPE; R24-P3/P4 computational items also marked DEPRECATED for routing clarity (re-route to R22 only if budget allows, but no new clinical resource).
- Phase table: 4 rows all struck through + ⛔ DEPRECATED suffix (with conditional re-route note on R24-P3/P4).
- 关键 output block: 4 bullets all struck through.

### §1.6 §3.5 R25 Phase III — DEPRECATED

- Header line: "⛔ DEPRECATED — post-trial 真正 paper + 算法收敛 (no further planning per 2026-09-29 user directive ...)".
- 目标 paragraph: R25-P1/P2 fully clinical = OUT OF SCOPE; R25-P3 Nature MI tier-1 paper explicitly OFF the table (user wants tier-2 only); R25-P4 capstone DEPRECATED because no Phase III + no tier-1 paper = no capstone; algorithm convergence report (R17-R25 9-year retrospective) DEPRECATED → only R17-R22 6-year computational retrospective is in scope if needed.
- Phase table: 4 rows all struck through + ⛔ DEPRECATED suffix.
- 关键 output block: 4 bullets all struck through.

---

## §2. n_phases_deprecated = 3

| Phase | Status | Rationale |
|---|---|---|
| R23 (Phase I clinical) | ⛔ DEPRECATED | User directive 2026-09-29: no wet-lab planning; tier-2 journal target |
| R24 (Phase II + Phase II/III + M44/DiffDock) | ⛔ DEPRECATED | User directive 2026-09-29: no wet-lab planning |
| R25 (Phase III + post-trial Nature MI + convergence) | ⛔ DEPRECATED | User directive 2026-09-29: tier-2 only; no Phase III |

**Count**: n_phases_deprecated = **3** (R23 + R24 + R25)

R22 remains ACTIVE with R22-P1 m18 re-scoped to computational-only.

---

## §3. Pre-flight status

`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py` 2026-09-29:

```
"summary": {
  "pass": 52,
  "fail": 0,
  "skip": 0,
  "total": 52,
  ...
}
EXIT_CODE = 0
```

**Pre-flight status**: **PASS (52/52, 0 fail, 0 skip)**

Per AGENTS.md §4 / TODO-207 v2 multi-workflow template (52 items, supersedes the original 25/25 v1): all P0 / P1 / P2 / P3 gates green. No stale flag (refreshed today).

---

## §4. Out-of-scope follow-ups (not in this Wf)

These items are **flagged for downstream Wf cycles**, not edited in Wf-317E:

1. **`TODO/03_plans/LAMBDA_ROADMAP.md` §4.3 paper trajectory table** — R23-R25 rows still cite Nature MI / Nat. Comput. Sci. tier-1; should be struck through or marked DEPRECATED in a follow-up Wf-317F if needed (the §4.3 R20 row already implicitly points to tier-2 via the §1.6 edit, but explicit §4.3 update would be cleaner).
2. **`TODO/03_plans/LAMBDA_ROADMAP.md` §5.3 + §5.5 R23-R25 resource tables** — clinical facility + Phase I/II/III trial + ORNL/PKU/sponsor + $271.5M funding budget still in §5.3 and §5.5; should be struck through / reduced in follow-up.
3. **`TODO/03_plans/LAMBDA_ROADMAP.md` §6.4 R19 skill ecosystem table** — `mol-metal-anticancer` row "R22 update (m20 tumor + Phase I)" + `mol-metal-wet-lab` (NEW R19) row "R22 update (AlphaFold-Multimer + patient organoid)" should be re-scoped in follow-up.
4. **`TODO/03_plans/LAMBDA_ROADMAP.md` §6.1 R19 → R20+ 接力 table** — "Real patient wet-lab trial → R23 (Phase I)" + "9-cell LOMO transfer AUC (m19) Tier-1 future metric → R22-P2" + "Multi-modal Protein-Metal binding AUC (m18) Tier-1 future metric → R22-P1" rows need updates.
5. **`TODO/03_plans/LAMBDA_ROADMAP.md` §3.6 R22-R25 long-horizon honesty framing** — "Phase I trial patient enrollment 慢 (COVID-style 风险)" + "Phase II/III 50/200 patient cohort efficacy" honest-negative risks should be struck through.
6. **`TODO/03_plans/LAMBDA_ROADMAP.md` §8.2 R22-R25 honest-negative 风险** — same items as §3.6 above.
7. **`TODO/03_plans/LAMBDA_ROADMAP.md` §9 final verdict** — "R22-R25 长期演化 + tier-1 metric + clinical trial" should be updated to "R22 computational-only tier-1 metric; R23-R25 DEPRECATED".

These follow-ups are not load-bearing for the user directive (the directive is operationalized by §1.6, §3.1, §3.2, §3.3, §3.4, §3.5 above); they are documentation polish for future Wf cycles.

---

## §5. Honest framing (party-twenty-spirit §3.5)

- **No new MEASURED numbers introduced** — all edits are documentation / scope-rescoping. The pre-flight 52/52 PASS status was already PASS before this Wf (the file edits are non-functional re-scoping).
- **DO claim**: "LAMBDA_ROADMAP §3.1 / §3.3 / §3.4 / §3.5 explicitly mark R23 / R24 / R25 as DEPRECATED with the user 2026-09-29 directive verbatim; R22-P1 m18 re-scoped to computational-only per the same directive."
- **DO NOT claim**: "All paper §4 lab-tier-1 metric trajectories have been updated to tier-2" — only §1.6 + §3.1 + §3.2 changed; §4.3 + §5.3 + §5.5 + §6.1 + §6.4 + §3.6 + §8.2 + §9 still carry tier-1 references (flagged in §4 above as follow-up scope).
- **DO NOT claim**: "Wet-lab Phase I/II/III are impossible" — they are DEPRECATED for THIS PROJECT's planning; a future project that wants tier-1 / clinical should re-scope, not copy this file.

---

## §6. Cross-references

- `TODO/03_plans/LAMBDA_ROADMAP.md` (edited file)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` (pre-flight v2 template, 52/52 PASS)
- `AGENTS.md` §1 rule #2 (kill-then-adapt) — not triggered (no GPU launch in this Wf)
- `AGENTS.md` §1 rule #3 (no paper work without authorization) — not triggered (paper section not touched)
- `AGENTS.md` §1 rule #4 (pre-flight v2 PASS) — ✅ verified, 52/52 PASS
- `party-twenty-spirit` §3.5 (honest framing) — §5 above this document

---

**Date**: 2026-09-29 (Wf-317E ship; pre-flight 52/52 PASS)
