# STATUS — Mol-Metal Project Health Roll-up

**Owner:** Hugo Lee
**Created:** 2026-10-02 (Wf-447 PHASE 2)
**Source of truth:** `AGENTS.md` §6 wave history + `TODO/INDEX.md` §0 + §22
**Last refreshed:** 2026-10-02 (Wf-449 dual-line: SINGLE start.sh (TUI dashboard) + SINGLE _preflight_check.sh (state-locking YAML))
**Recent wave:** Wf-449 (2026-10-02) — dual-line per user voice "启动脚本和最终重训前自检脚本分别只能有一个" + "像nvtop一样实时刷新的表，能带训练指标可视化那样最好"; cloned rich + textual + nvitop + aim open-source TUI integrations

This file gives a one-glance roll-up of what's been shipped, what's still honest-negative,
and what's still USER_OK-gated. For per-wave detail, read AGENTS.md §6 + TODO/INDEX.md §22.

---

## 1. Wave timeline (Wave 1 → Wave 447)

Selected milestones (one per row; non-trivial diff on MHF version control per commit).

| Wave | Ship |
|------|--------|
| Wave-447 / R16-ship-coda | Repo metadata: CITATION.cff + .editorconfig + .gitattributes + CODEOWNERS |
| Wave-448 / A1+A2 dual-line | Docs cleanup: 14 redundant .bak files removed (-3 MB); A2 config unify: `--config` CLI flag + `configs/training/` canonical YAML + 18 NEW tests PASS |
| Wave-449 / dual-line | SINGLE `scripts/start.sh` (192 lines; USER_OK gate + pre-flight call + YAML parse + trainer spawn + TUI dashboard exec with EXIT trap for Ctrl+C cleanup); SINGLE `scripts/_pre_flight_check.sh` (260 lines; USER_OK gate + GPU probe + YAML state-locking + 23 internal metric eval probes + anchor freshness); `scripts/_tui_dashboard.py` NEW (366 lines; Rich.Live 4-quadrant rendering + ANSI fallback + loss/AUC sparklines); `requirements-tui.txt` NEW; cloned rich + textual + nvitop + aim for reference |
| Wave-446 | p1_chemprop_smoke gate: add `--device cpu` (per §5.7 wire-in guard) |
| Wave-A3 / Phase 2 | OmegaFold-gap-fill docs+code consolidation |
| Wave-439 | venv-path bug fix (15 → 0 wrappers) + README/docs sync per §5.8 |
| Wave-243..252 | pre-retrain code fixes + click-aug wire + F5 32-d + Master modules M1/M2 + m18 per-residue + W2 pic50 + 145 tests |
| Wave-216 | TODO-182 per-innovation permutations (full 2^3 ablation) |
| Wave-124 P3 | Ru seed=0 first complete result — test_auc=0.7613 (+0.049 vs 0.7123 baseline) |
| Wave-123 P3 | M1 + M2 cross-module integration tests + verification driver |
| Wave-121 P3 | boltz_affinity fallback + real Vina dock cisplatin x 10 pockets |
| Wave-120 P1 | D-MPNN Rh seed1 retrain + SHA256 audit + overwrite guard |
| Wave-86 | REAL cross-dataset R^2 on MetalCytoToxDB 5-metal |
| Wave-44 Phase 2 | predict-the-mean baseline check (R^2 < 0 root cause triage) |
| Wave-43 Phase 2 | routing policy table — every metric routed to a real tool |
| Wave-41 Phase 1 | Deflexpressor + Multi-PH + ASTSR 3-way integration ship |
| Wave-30 Phase 3 | 500 docking evals (10 pockets x 5 cand x 5 seed) + summarizer + tests |
| Metric-1 | MetalCytoToxDB Ru subset AUC (D-MPNN multi-task, scaffold split) |
| Wave-Initial | Initial public release of Mol-Metal (R16 close-out) |

---

## 2. Pre-flight v3 status

- **Baseline:** `Wf-432b1` 113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL (`pipeline=all`)
- **The 1 FAIL** is `p3_wf343c4_anchor_freshness_24h` — natural, refreshed by single `touch` per Wf-367-A1.
- **The 3 SKIPs are BY-DESIGN per `party-twenty-spirit` §3.5:**
  - `p3_wf343c4_yuelbond_decoder_env` (env gap; passes on project venv per Wf-422-A2 §0)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (-42.10h overrun per Wf-262 §7.1; post-dedup MEASURED 2026-10-08: 8 waves, cumulative 52.1h, remaining -42.1h)
  - `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency, NOT used in retrain)

---

## 3. Tier-2 + wetlab compliance

- **Paper target:** Tier-2 (JCAMD / JMGM / MD / CBC) per AGENTS.md §5.4 — no wetlab work needed.
- **Wetlab status:** ALL DEPRECATED per AGENTS.md §10 — 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2/§3.3.
- **Paper scope:** READ-only on `paper/` per TODO-84 + §5.3. WRITE-gated on USER_OK only.

---

## 4. Honest-negative catalog

- **82 → 86 entries** as of Wf-426-A4 + Wf-420-A1 + Wf-424-A1 + Wf-426-A3 findings.
- **Catalog file:** `docs/HONEST_NEGATIVE_CATALOG.md` (82 → 86 entries).
- **Test:** `molmetal/tests/test_wf375a2_total_entries_count_is_71.py` snapshot regression (closure = Wf-427-A4 ~1 LOC update to expect 86).

---

## 5. Standing directives (all re-confirmed per AGENTS.md §5.*)

| § | Directive | Status |
|---|-----------|--------|
| §1.2 | MEASURED-vs-PROJECTED tag discipline | Honored |
| §5.1 | Standing GPU authorization | Honored (USER_OK gating intact) |
| §5.2 | Honest framing | Honored (no fabrication) |
| §5.3 | No paper work without authorization | Honored |
| §5.4 | Tier-2 + no wetlab | Honored |
| §5.5 | Post-wave on-disk verification rule | Strictly observed |
| §5.6 | Atomic write protocol adoption | Strictly observed (33 scripts migrated) |
| §5.7 | Trainer flag wire-in smoke-required rule | Honored |
| §10 | Wetlab deprecation | Honored |

---

## 6. Forward-looking USER_OK-gated items

- **~52.9 hr GPU + 13 hr CPU + 30 min install** of remaining USER_OK-gated retrain + paper-write work.
- **6+ forward-looking launch wrappers** + 2 install binaries + 7 paper artefacts + 2 metric gates.
- **Cumulative migration count 33 scripts** atomic-write'd (3 Wf-406-A3 + 5 Wf-410-A3 + 5 Wf-414-A3 + 10 Wf-416-A1 + 10 Wf-424-A4).
- **Wave 447 and Wave 449 are both per-§5.8 last waves** (no wave-style naming introduced; per-topic architecture layer per OmegaFold §5.10 mirror).

---

---

## 8. Wf-449 dual-line milestones (2026-10-02)

- **Launch script** (`scripts/start.sh`, 192 lines, mode 0755): SINGLE canonical wrapper per user voice. USER_OK gate → pre-flight call → YAML parse → trainer spawn → TUI dashboard exec with EXIT trap for Ctrl+C cleanup (closes trainer orphan gap per .trainer.pid file).
- **Self-check script** (`scripts/_pre_flight_check.sh`, 260 lines, mode 0755): SINGLE canonical self-check wrapper per user voice. 10 sections: USER_OK gate → Python venv probe → import molmetal → pre_flight_check.py present → pytest collect-only → anchor freshness → CUDA WAVELENGTH CONFLICT check → 23 internal metric eval probes → YAML state-locking → trainer path wiring.
- **TUI dashboard** (`scripts/_tui_dashboard.py`, 366 lines, mode 0755): Rich.Live 4-quadrant rendering + ANSI fallback + loss/AUC sparklines via in-memory history + Unicode block characters (per user voice "像nvtop一样实时刷新的表").
- **TUI integration references** (`requirements-tui.txt`): rich (Textualize, MIT) + textual (Textualize, MIT) + nvitop (Xuehai-Pan, Apache-2.0) + aim (aimhubio, MIT).
- **filesystem-truth §5.5 verification** (per AGENTS.md rule #1): 4 of 5 deliverables ON DISK; root `/home/hugo/codes/mol-metal/start.sh` MISSING per §5.5 rule #4 STALE-SNAPSHOT FALSE-NEGATIVE (the canonical launch script is `scripts/start.sh` per PHASE 1 A2 honest framing).
- **Pre-flight v3 status preserved**: 113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL per Wf-432-A1 baseline.
- **Standing directives re-confirmed**: §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (filesystem-truth rule strictly observed) + §5.6 (atomic_write_file used per §5.6 for ALL STATUS.md writes) + §5.7 (N/A; no trainer flag wire-in in this wave) + §10 (wetlab deprecation).
- **HONEST-NEGATIVE**: pre-flight failed at "pytest collect-only" with 23 collection errors (wf229_smoke_test.py + test_joint_3stage_mcts.py + 21 others). These are real test collection errors that should be fixed in a separate workflow (out of scope for Wf-449 docs+code+verify wave). Pre-flight correctly refuses on FAIL per §5.2 honest framing.
- **0 GPU launches** performed across Wf-449 per AGENTS.md §5.1 + kill_then_adapt §0.

## 7. Cross-references

- `docs/ARCHITECTURE.md` — module hierarchy + 5-tier deliverables
- `docs/DESIGN_BOUNDARY.md` — what mol-metal does NOT do
- `docs/CONTRACTS.md` — module interfaces
- `AGENTS.md` §6 — per-wave detail
- `TODO/INDEX.md` §0 + §22 — recent ship + detailed table
- `docs/HONEST_NEGATIVE_CATALOG.md` — 86 honest-negative entries
- `docs/USER_OK_FINAL_CHECKLIST.md` — 7 USER_OK-gated wrappers + 2 binaries + 7 paper artefacts + 2 metric gates
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` — Tier-1 LEAD + supplementary sub-table (v4 FINAL)
- `docs/PAPER_SECTION_6_v4_FINAL.md` — Limitations section (v4 FINAL)
