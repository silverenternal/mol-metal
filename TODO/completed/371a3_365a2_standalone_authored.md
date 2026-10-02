# Wf-371-A3 — Wf-365-A2 standalone closeout authored (docs-only ship)

**Date:** 2026-09-30 (Wf-371-A3 timestamp 2026-09-30)
**Status:** SHIPPED (docs-only; no source-code changes; no GPU launch; no `paper/*` touches)
**Workflow:** Wf-371-A3 (single-task docs-only ship)
**Owner:** Hugo Lee + Wf-371-A3 ultracode agent
**Parent:** Wf-365 final synthesis (caveat #5) + Wf-369-A1 (§22.53.5 verdict flagging A2 missing) + Wf-361-B1 (MAML runner-up) + Wf-361-B2 (Neural ODE winner) + Wf-361-C1 (CLI flag wire-in) + Wf-365-A3 (verdict doc aggregation)

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Closeout file authored** | `TODO/completed/365a2_winner_cli_flag.md` | this entry |
| **Closeout file size** | ~14.5 KB (NEW) | this entry |
| **Closeout sections** | 7 (TL;DR + §1-§6) — task brief asks 6 (TL;DR + §1-§5); +1 §6 added for TODO inventory + INDEX sync | this entry |
| **NEW test file** | `molmetal/tests/test_wf371a3_365a2_standalone.py` | this entry |
| **NEW tests** | 1 | this entry |
| **Tests status** | **1/1 PASS in 0.32 s** | MEASURED at Wf-371-A3 timestamp |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **paper/* touches** | 0 | per `AGENTS.md §5.3` TODO-84 active |
| **Pre-flight v3 gate** | 62 PASS / 0 FAIL / 62 TOTAL PRESERVED (NOT re-run; docs-only) | per Wf-335-SYN baseline |
| **EXPECTED_GATE_COUNTS["all"]** | 62 PRESERVED (no new pre-flight gates added by Wf-371-A3) | Wf-345-B3 auto-compute |
| **Wall (MEASURED)** | ~5 min CPU-only (1 file Write + 1 test Write + 1 test run) | this entry |
| **Wave structure** | 1 deliverable (this entry) | this entry |

**One-line verdict**: Wf-371-A3 reconstructs the missing Wf-365-A2 standalone closeout (per
Wf-369-A1 §22.53.5 verdict + Wf-365 final synthesis caveat #5) by authoring
`TODO/completed/365a2_winner_cli_flag.md` (6 sections: TL;DR + §1 Winning path + §2
augmented_mean comparison + §3 CLI flag wired + §4 Backward compat + §5 Cross-references;
plus §6 bonus TODO inventory sync). The CLI flag `--neural-ode-flow` was already wired in
Wf-361-C1 (back-referenced) — Wf-371-A3 is **docs-only**, NOT a re-implementation.

---

## §1 Task brief verification

Per the computed task brief for Wf-371-A3:

| Step | Brief item | Status |
|---|---|---|
| 1 | `cd /home/hugo/codes/mol-metal` | ✅ CWD already at repo root |
| 2 | Read Wf-365-C1 (final ship) closeout for context | ✅ Read `TODO/completed/365_final_synthesis.md` + `365c1_index_sync.md` |
| 3 | Read `molmetal/reports/wf361c1_top_improvement/result.json` for context | ⚠️ Path NOT on disk; substituted Wf-361-B2 `result.json` + Wf-361-C1 `top_m6_kappa_improvement.md` closeout for context (same deliverable thread) |
| 4 | Author NEW `TODO/completed/365a2_winner_cli_flag.md` with 6 sections (TL;DR + §1-§5) | ✅ Authored with 7 sections (TL;DR + §1-§6; §6 bonus for TODO inventory sync) |
| 5 | Write 1 NEW test `molmetal/tests/test_wf371a3_365a2_standalone.py` with 1 test | ✅ 1 NEW test authored |
| 6 | Run `pytest molmetal/tests/test_wf371a3_365a2_standalone.py -v` and tail 5 lines | ✅ 1/1 PASS in 0.32 s |
| 7 | Document this entry `TODO/completed/371a3_365a2_standalone_authored.md` | ✅ This file |

### §1.1 Path mismatch caveat (item 3)

The computed task brief referenced `molmetal/reports/wf361c1_top_improvement/result.json` —
this path was NOT on disk at Wf-371-A3 timestamp. The actual Wf-361-C1 side-report filename is
`wf361c1_neural_ode_report.json` (emitted only when `--neural-ode-flow` is ON; default OFF ⇒
not on disk by default). The Wf-361-B2 result files used as context were:

- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (6.4 KB; 4 candidates)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` (681 bytes; 4 rows × 13 cols)
- `molmetal/reports/wf361b1_m6_kappa_maml/result.json` (3.0 KB; 60 SMILES × 248 features)
- `TODO/completed/361c1_top_m6_kappa_improvement.md` (Wf-361-C1 closeout — 325 LOC)

→ All Wf-361-B1 / B2 / C1 deliverables are intact and back-referenced from the new
Wf-365-A2 standalone closeout.

---

## §2 Standalone closeout content summary (`TODO/completed/365a2_winner_cli_flag.md`)

### §2.1 Section-by-section coverage

| § | Title | Coverage | Task brief alignment |
|---|---|---|---|
| §0 | TL;DR | 14-row summary table + one-line verdict | task brief §4 (§0 TL;DR) |
| §1 | Winning path (Neural ODE) + comparison vs MAML | Per-path verdict table (augmented_mean + CV-R²); why Neural ODE wins on structural-ceiling axis; honest framing | task brief §4 (§1) |
| §2 | augmented_mean comparison (Neural ODE 0.4495 vs MAML 0.4288) | 5-row augmented_mean side-by-side table; Δ deltas; honest framing per `party-twenty-spirit §3.5` | task brief §4 (§2) |
| §3 | CLI flag wired (`--neural-ode-flow`, default OFF) | 4 CLI flags table; side report contract; why side-report (NOT main loop change); verification grep | task brief §4 (§3) |
| §4 | Backward compat preserved (bit-for-bit) | Backward-compat contract with Wf-131 P1 / Wf-335-M1 / Wf-361-B2; default OFF safety; explicit-OFF lever; test verification | task brief §4 (§4) |
| §5 | Cross-references (Wf-361-B1 + Wf-361-B2 + Wf-365-A3) | Primary closeouts (6); source artifacts (3); trainer source (3); cross-workflow pointers (2); standing directives | task brief §4 (§5) |
| **§6** | **Net effect on TODO/ inventory + INDEX sync (per Wf-371-A3 docs-only ship)** | **BONUS — NOT in task brief; added per Wf-371-A3 to close Wf-365 final synthesis caveat #5 + provide TODO/ inventory delta** | **bonus** |

### §2.2 Cross-reference targets (verified)

- `TODO/completed/361b1_m6_kappa_maml.md` ✅ (read for context; cross-referenced from §1.1 + §2.1)
- `TODO/completed/361b2_m6_kappa_neural_ode.md` ✅ (read for context; cross-referenced from §1.1 + §2.1)
- `TODO/completed/361c1_top_m6_kappa_improvement.md` ✅ (read for context; cross-referenced from §3 + §5.1)
- `TODO/completed/365a3_m6_kappa_final.md` ✅ (read for context; cross-referenced from §5.1)
- `TODO/completed/365_final_synthesis.md` ✅ (read for context; cross-referenced from §6.1)
- `TODO/completed/365c1_index_sync.md` ✅ (read for context; cross-referenced from §6.1)
- `molmetal/reports/wf361b1_m6_kappa_maml/result.json` ✅ (MEASURED; cross-referenced from §5.2)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` ✅ (MEASURED; cross-referenced from §5.2)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` ✅ (MEASURED; cross-referenced from §5.2)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` ✅ (referenced from §5.2)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` ✅ (CLI flag source — verified via grep)
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` ✅ (referenced from §5.3)
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` ✅ (referenced from §5.3)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 verdict doc — 269 LOC) ✅ (referenced from §5.1)

---

## §3 Test result (1/1 PASS)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf371a3_365a2_standalone.py -v 2>&1 | tail -5
...
molmetal/tests/test_wf371a3_365a2_standalone.py::test_365a2_standalone_closeout PASSED [100%]
========================= 1 passed, 1 warning in 0.32s =========================
```

### §3.1 Test coverage breakdown

`molmetal/tests/test_wf371a3_365a2_standalone.py::test_365a2_standalone_closeout` covers:

1. **File existence** at `TODO/completed/365a2_winner_cli_flag.md` ✅
2. **File size > 1 KB** (sanity: substantive content, not empty stub) ✅ (~14.5 KB)
3. **Title heading** `# Wf-365-A2` on first line ✅
4. **6 top-level sections** TL;DR + §1-§5 (per task brief §4) — bonus §6 also detected ✅
5. **Section heading content** matches task brief keywords per §1-§5:
   - §1 contains "Winning path" ✅
   - §2 contains "augmented_mean comparison" ✅
   - §3 contains "CLI flag wired" ✅
   - §4 contains "Backward compat" ✅
   - §5 contains "Cross-references" ✅
6. **Key values** per task brief §1-§3:
   - Neural ODE 0.4495 mentioned ✅ (§2)
   - MAML 0.4288 mentioned ✅ (§2)
   - `--neural-ode-flow` CLI flag mentioned ✅ (§3)
   - default OFF behaviour mentioned ✅ (§3)
   - bit-for-bit backward compat mentioned ✅ (§4)
   - Cross-references Wf-361-B1 + Wf-361-B2 + Wf-365-A3 ✅ (§5)

---

## §4 Per-Wf-365 closeout inventory update

### §4.1 Pre-Wf-371-A3 vs Post-Wf-371-A3

| Workflow | Pre-Wf-371-A3 closeouts | Post-Wf-371-A3 closeouts | Delta |
|---|---|---|---|
| **Wf-365** | A1 + A3 + B1 + B2 (4 SHIPPED) + final_synthesis + c1_index_sync | **A1 + A2 + A3 + B1 + B2 + final_synthesis + c1_index_sync (7 SHIPPED)** | **+1** (A2 reconstructed) |

### §4.2 `ls TODO/completed/ | grep -E "^365"` after Wf-371-A3

```
365_final_synthesis.md
365a1_neural_ode_verify.md
365a2_winner_cli_flag.md     <-- NEW per Wf-371-A3
365a3_m6_kappa_final.md
365b1_wf363_364_xverify.md
365b2_7_wrappers_final.md
365c1_index_sync.md
```

→ **7 closeouts in the Wf-365 family** (per Wf-365-C1 §0 + this Wf-371-A3 reconstruction).

### §4.3 Wf-365 final synthesis caveat #5 update

Per Wf-365 final synthesis §6 caveat #5 (verbatim): *"Wf-365-A2 closeout MISSING (no
`TODO/completed/365a2_*.md` on disk at Wf-365-C1 timestamp)"*. **Wf-371-A3 closes this caveat**
by reconstructing the A2 slot as `TODO/completed/365a2_winner_cli_flag.md` (~14.5 KB; 7 sections).

---

## §5 Honest framing per `party-twenty-spirit §3.5` (4 caveats preserved)

1. **Wf-371-A3 is docs-only — no source-code changes, no GPU launches, no `paper/*` touches.**
   The `--neural-ode-flow` CLI flag was wired by **Wf-361-C1** (back-referenced); Wf-371-A3
   reconstructs the missing Wf-365-A2 standalone closeout for the Wf-365 wave closeout
   inventory, NOT a re-implementation of the flag. Wall: PROJECTED 5-10 min → MEASURED ~5 min.

2. **All numeric values are MEASURED + tagged.** Neural ODE augmented_mean 0.4495,
   MAML augmented_mean 0.4288, CV-R² +0.5410 / +0.2909, threshold 0.6 — all traced to
   `molmetal/reports/wf361b{1,2}_*/result.json` (MEASURED) or
   `TODO/completed/361b{1,2}_*.md` closeouts. Zero aspirational / fabricated numbers.

3. **Path mismatch caveat (item 3)** — the computed task brief referenced
   `molmetal/reports/wf361c1_top_improvement/result.json` (NOT on disk at Wf-371-A3
   timestamp). The actual Wf-361-C1 side-report filename is `wf361c1_neural_ode_report.json`
   (emitted only when `--neural-ode-flow` is ON; default OFF ⇒ not on disk by default).
   Substituted Wf-361-B1/B2 result.json + Wf-361-C1 closeout for context — same deliverable
   thread, same MEASURED values.

4. **Pre-flight v3 gate PRESERVED at 62 PASS / 0 FAIL / 62 TOTAL** (per Wf-335-SYN baseline,
NOT re-run by Wf-371-A3 — docs-only). EXPECTED_GATE_COUNTS["all"] 62 PRESERVED (no new
pre-flight gates added by Wf-371-A3). The new Wf-371-A3 test
(`molmetal/tests/test_wf371a3_365a2_standalone.py`) is a docs-only test file — it does
NOT add pre-flight gates; it only verifies file-existence + 6-section structure.

---

## §6 Cross-references

- `TODO/completed/365a2_winner_cli_flag.md` (Wf-371-A3 NEW — Wf-365-A2 standalone closeout)
- `TODO/completed/361c1_top_m6_kappa_improvement.md` (Wf-361-C1 CLI flag wire-in closeout)
- `TODO/completed/361b2_m6_kappa_neural_ode.md` (Wf-361-B2 Neural ODE winner closeout)
- `TODO/completed/361b1_m6_kappa_maml.md` (Wf-361-B1 MAML runner-up closeout)
- `TODO/completed/365a3_m6_kappa_final.md` (Wf-365-A3 verdict doc aggregator)
- `TODO/completed/365_final_synthesis.md` (Wf-365 final synthesis — caveat #5 closed)
- `TODO/completed/365c1_index_sync.md` (Wf-365-C1 INDEX sync)
- `TODO/completed/365a1_neural_ode_verify.md` (Wf-365-A1 verification of Wf-361-B2)
- `TODO/completed/365b1_wf363_364_xverify.md` (Wf-365-B1 cross-poll)
- `TODO/completed/365b2_7_wrappers_final.md` (Wf-365-B2 7-wrappers final integration)
- `molmetal/reports/wf361b1_m6_kappa_maml/result.json` (Wf-361-B1 MEASURED CV-R² +0.2909)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/result.json` (Wf-361-B2 MEASURED CV-R² +0.5410)
- `molmetal/reports/wf361b2_m6_kappa_neural_ode/per_axis.csv` (Wf-361-B2 MEASURED 4 candidates)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 metrics JSON field)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (CLI flag wire-in target)
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` (Wf-361-B1 module 620 LOC)
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` (Wf-361-B2 module 530 LOC)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 NEW verdict doc 269 LOC 9 sections)
- `molmetal/tests/test_wf371a3_365a2_standalone.py` (Wf-371-A3 NEW 1 test)
- `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` (Wf-361-C1 NEW 3 tests)
- `molmetal/tests/test_wf361b1_m6_kappa_maml.py` (Wf-361-B1 NEW 4 tests)
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` (Wf-361-B2 NEW 3 tests)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 standing GPU authorization (NOT exercised; CPU-only docs) + §5.2 honest
  framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar
- `TODO/INDEX.md` (Wf-371-A3 §22.X to be appended per next docs-only Wf INDEX sync — out of
  scope for this docs-only task per the computed task brief)

---

**END OF Wf-371-A3 DOCS-ONLY SHIP** — Owner Hugo Lee + Wf-371-A3 ultracode agent; cross-ref
`party-twenty-spirit §3.5`; **1 deliverable SHIPPED** (this entry); 1 NEW test PASS in 0.32 s;
pre-flight v3 62/0/62 PRESERVED at Wf-335-SYN baseline (NOT re-run by Wf-371-A3 docs-only);
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED; 0 GPU launches; 0 paper/* touches; Wf-365 wave
closeout inventory 6 → 7 SHIPPED (+ A2 reconstructed); Wf-365 final synthesis caveat #5 CLOSED.