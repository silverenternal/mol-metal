WF-303: Restore per-metal AUC breakdown in m6_kappa.json

CONTEXT: Wf-291C refactored the metric producer to use StructuredOutput and
accidentally dropped the per-metal cohort breakdown from m6_kappa.json.
Wf-302C audit flagged the gap; closure-theorem per-metal reachability
claims (Wf-37, mol-metal-closure-theorem) depend on the breakdown.

TASK (plain text — no JSON / no StructuredOutput):
  1. Read metrics/by_metric/m6_kappa.json and confirm per_metal field absent.
  2. Edit molmetal/scripts/train_dmpnn_multitask_wasserstein.py so it
     emits per-metal AUC alongside the pooled AUC.
  3. Re-run the producer for 5 seeds × {Pt,Ru,Ir,Au,Pd} and write the result.

TOOLS: USE Bash, Read, Edit. Do NOT call StructuredOutput.

REPLY FORMAT (5–8 lines max, plain text):
  - status: OK | PARTIAL | BLOCKED
  - files_touched: <abs paths>
  - gates_passed: G1=<...>, G2=<...>, G3=<...>
  - honest_framing: <MEASURED vs PROJECTED>

GATES:
  G1. metrics/by_metric/m6_kappa.json has per_metal field with 5 entries.
  G2. Each per-metal entry has n_samples >= 200 and AUC in [0,1].
  G3. Pooled AUC delta vs prior m6_kappa.json is within +/- 0.02.

OUTPUT FILES (absolute paths):
  - /home/hugo/codes/mol-metal/metrics/by_metric/m6_kappa.json

HONEST-FRAMING NOTE (required, 1-2 sentences):
  <Which metals were MEASURED on real seeds, which were PROJECTED from
   the pooled run, which metal was BLOCKED by data scarcity.>