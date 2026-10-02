WF-305: Synthesise Wf-291C insight into the text-brief template standard

CONTEXT: Wf-291C exposed the schema-as-spec vs schema-as-deliverable
confusion. Wf-302C produced a text-brief template. This brief asks for
a synthesis doc that locks the insight into the project's brief-writing
standard so future workflows do not re-introduce StructuredOutput.

TASK (plain text — no JSON / no StructuredOutput):
  1. Read molmetal/reports/wf302c_text_briefs/template.md and the three
     example briefs (bug_fix, audit, synthesis).
  2. Draft a 1-page synthesis linking the Wf-291C root cause, the
     Wf-302C fix, and the per-workflow adoption checklist.
  3. Cross-reference TODO/INDEX.md and TODO/03_plans/LAMBDA_ROADMAP.md
     to identify which in-flight workflows should adopt text briefs first.

TOOLS: USE Bash, Read, Edit, Grep. Do NOT call StructuredOutput.

REPLY FORMAT (5–8 lines max, plain text):
  - status: OK | PARTIAL | BLOCKED
  - files_touched: <abs paths>
  - gates_passed: G1=<...>, G2=<...>, G3=<...>
  - honest_framing: <MEASURED vs PROJECTED>

GATES:
  G1. Synthesis doc links the Wf-291C root cause to the Wf-302C template.
  G2. Adoption checklist names >= 3 in-flight workflows by Wf-NNN.
  G3. Doc explicitly disallows StructuredOutput in adopted briefs.

OUTPUT FILES (absolute paths):
  - /home/hugo/codes/mol-metal/molmetal/reports/wf302c_text_briefs/synthesis.md

HONEST-FRAMING NOTE (required, 1-2 sentences):
  <Which insights are MEASURED from prior Wf logs, which are PROJECTED
   from the template behaviour, which adoption paths are BLOCKED on
   reviewer bandwidth.>