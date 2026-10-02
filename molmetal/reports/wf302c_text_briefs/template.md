# Wf-302C — Text-Only Task Brief Template

## §0. TL;DR — Why Text-Only Briefs

**Root cause (Wf-291C):** when agents were handed a complex StructuredOutput schema, they conflated the schema-as-spec (instructions) with the schema-as-deliverable (output). They produced JSON-shaped fragments, satisfied the field list, and skipped the underlying synthesis the schema was meant to elicit. The regression class is broad: every Wf since Wf-291C that mixed JSON deliverable + prose interpretation under-delivered on the prose half.

**Fix:** keep the brief as plain text. Tell the agent which tools to use (`Bash`, `Read`, `Edit`), cap the reply at 5–8 lines, name the gates, pin the output paths, and require an honest-framing note (what was MEASURED vs PROJECTED, what was BLOCKED, what was an honest negative). No JSON envelope, no Pydantic shape, no `structured-output=` flag — just prose the agent can quote verbatim.

**Scope:** every sub-agent workflow spawned from Wf-302C onward that previously asked for StructuredOutput. Audit-style and synthesis-style briefs benefit most; bug-fix briefs benefit because the "expected" line is prose, not a typed object.

---

## §1. Template (copy-paste, fill in)

```
WF-NNN: <one-line title>

CONTEXT: <2-3 sentences. Why this brief exists. Cite the prior Wf it follows
from, and the insight that triggered the follow-up.>

TASK (plain text — no JSON / no StructuredOutput):
  1. <verb + concrete file or metric>. Output path: <abs path>.
  2. <verb + concrete file or metric>. Output path: <abs path>.
  3. <verb + concrete file or metric>. Output path: <abs path>.

TOOLS: USE Bash, Read, Edit. Do NOT call StructuredOutput.
       Do NOT write a JSON envelope. Do NOT return a typed object.

REPLY FORMAT (5–8 lines max, plain text):
  - status: OK | PARTIAL | BLOCKED
  - files_touched: <comma-separated absolute paths>
  - gates_passed: G1=<pass/fail reason>, G2=<...>, G3=<...>
  - honest_framing: <MEASURED vs PROJECTED, BLOCKED, honest-negative note>

GATES:
  G1. <verifiable gate — e.g. "pytest -k X exits 0">
  G2. <verifiable gate — e.g. "new file exists at <abs path> with N>0 lines">
  G3. <verifiable gate — e.g. "metric JSON field delta vs baseline ±X">

OUTPUT FILES (absolute paths):
  - <abs path A>
  - <abs path B>

HONEST-FRAMING NOTE (required, 1-2 sentences):
  <What was MEASURED on real data, what was PROJECTED from priors,
   what is BLOCKED, what was an honest-negative close.>
```

**How to use it.** Drop the block into a sub-agent invocation as plain text.
The agent's reply must satisfy the five reply-format lines; the gates
provide the success criterion; the honest-framing note blocks the
"everything passed" optimism that produced Wf-291C's regression.

---

## §2. Worked Example (Wf-302C-style bug fix)

```
WF-303: Fix m6_kappa.json regression after Wf-291C JSON refactor

CONTEXT: Wf-291C rewrote metrics/by_metric/m6_kappa.json via a structured
output that dropped the per-metal cohort breakdown. Wf-302C audit surfaced
the gap: m6_kappa no longer reports per-metal AUC, only pooled. Restore the
breakdown so the closure-theorem per-metal reachability claims (Wf-37) hold.

TASK (plain text — no JSON / no StructuredOutput):
  1. Read metrics/by_metric/m6_kappa.json. Confirm per-metal cohort field is missing.
  2. Edit the producer at molmetal/scripts/train_dmpnn_multitask_wasserstein.py
     so it emits per-metal AUC alongside the pooled AUC.
  3. Re-run the producer for 5 seeds × Pt/Ru/Ir/Au/Pd and write the result to
     metrics/by_metric/m6_kappa.json. Output path: /home/hugo/codes/mol-metal/metrics/by_metric/m6_kappa.json.

TOOLS: USE Bash, Read, Edit. Do NOT call StructuredOutput.

REPLY FORMAT (5–8 lines max, plain text):
  - status: OK | PARTIAL | BLOCKED
  - files_touched: <abs paths>
  - gates_passed: G1=<...>, G2=<...>, G3=<...>
  - honest_framing: <MEASURED vs PROJECTED>

GATES:
  G1. metrics/by_metric/m6_kappa.json contains a per_metal field with 5 entries.
  G2. Each per-metal entry has n_samples ≥ 200 and AUC in [0,1].
  G3. Pooled AUC delta vs prior m6_kappa.json is within ±0.02.

OUTPUT FILES (absolute paths):
  - /home/hugo/codes/mol-metal/metrics/by_metric/m6_kappa.json

HONEST-FRAMING NOTE (required, 1-2 sentences):
  <Which metals were MEASURED on real seeds, which were PROJECTED from
   the pooled run, which metal was BLOCKED by data scarcity.>
```

---

## §3. Common Pitfalls to Avoid

1. **Re-introducing StructuredOutput in the "REPLY FORMAT" lines.** If the
   reply format starts looking like a JSON schema (`{"status": "...", ...}`),
   the regression returns. Keep the five reply lines as bullet points with
   `key: value` pairs, not as JSON.
2. **Vague gates.** "It works" or "tests pass" is not a gate. Name the test
   selector, the file path, the metric delta. An agent should be able to
   copy the gate into a Bash one-liner.
3. **Forgetting the honest-framing note.** This is the single biggest
   safeguard against Wf-291C-style optimistic close. If the brief omits it,
   agents will skip it. Make it required, with a one-line template.
4. **Tool drift.** "Use StructuredOutput" or "return a typed object" anywhere
   in the brief re-opens the regression. Audit every line; the TOOLS line
   must say `Bash / Read / Edit only`.
5. **Path relativity.** Sub-agent shells reset cwd between calls. Every
   output path must be absolute. Relative paths silently drop files in
   `$HOME` or `/tmp`.
6. **Reply length creep.** The 5–8-line cap exists because long replies
   re-introduce the "schema-as-deliverable" failure mode: the agent pads
   prose to fill space instead of writing a tight summary. If a sub-agent
   returns 30 lines, the brief is wrong, not the agent.
7. **Confusing the brief with the deliverable.** The brief is input to the
   agent, not output to the user. Don't paste the template block into
   user-facing reports. The worked example in §2 is for agents only.
8. **Skipping the prior-Wf citation in CONTEXT.** Without the citation the
   agent has no way to know which prior insight triggered the follow-up,
   and the work becomes decontextualised re-implementation.