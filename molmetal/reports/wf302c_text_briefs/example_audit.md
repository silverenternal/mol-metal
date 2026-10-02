WF-304: Audit Wf-291C-style StructuredOutput regressions across metrics/

CONTEXT: Wf-291C rewrote multiple metric producers using StructuredOutput.
Wf-302C needs an audit pass to identify which metric files lost fields
or granularity in the rewrite. The audit feeds the Wf-303+ recovery
queue.

TASK (plain text — no JSON / no StructuredOutput):
  1. Walk metrics/by_metric/*.json and git-log each file back to its
     pre-Wf-291C state.
  2. Produce a regression list: file -> lost field -> downstream
     consumer that depends on it. Output path: molmetal/reports/wf302c_text_briefs/audit_regressions.md.
  3. Cross-reference each lost field against TODO/INDEX.md for the
     workflow that depends on it.

TOOLS: USE Bash, Read, Edit, Grep. Do NOT call StructuredOutput.

REPLY FORMAT (5–8 lines max, plain text):
  - status: OK | PARTIAL | BLOCKED
  - files_touched: <abs paths>
  - gates_passed: G1=<...>, G2=<...>, G3=<...>
  - honest_framing: <MEASURED vs PROJECTED>

GATES:
  G1. audit_regressions.md exists and lists >= 1 lost field per affected metric.
  G2. Each entry cites the downstream workflow by Wf-NNN reference.
  G3. No metric file is silently re-introduced; every entry is traceable via git blame.

OUTPUT FILES (absolute paths):
  - /home/hugo/codes/mol-metal/molmetal/reports/wf302c_text_briefs/audit_regressions.md

HONEST-FRAMING NOTE (required, 1-2 sentences):
  <Which regressions are MEASURED via git diff, which are PROJECTED from
   the consumer list, which metrics could not be audited (BLOCKED).>