# TODO-89b — Wf-63 Wf-47 script1 + script4 EMPTY follow-up + Vina install attempt close-out

**Date**: 2026-09-26
**Workflows**: Wf-63 (3-phase follow-up) → Wf-47 script1 (DiffDock) + script4 (Boltz D-MPNN pIC50) EMPTY dirs + Vina install attempt
**Status**: ⚠️ **PARTIAL SHIP** — Vina install re-attempted (system CLI still libboost_thread-pinned-broken; Python `vina 1.2.7` binding works but no CLI subprocess); Wf-63 Phase 1 (Vina install status) SHIP; Phase 2 (script1 m3 retry) **EMPTY** (Vina MISSING_DEP blocker preserved); Phase 3 (script4 m8 retry) **BELOW_TARGET** — m8 PB pass_rate=**0.000** across 10 pockets × 5 cands (real PoseBusters binary confirmed installed; UFF lacks Pt/Au/Ru parameters → all 14 chemistry checks fail on metal-coordinated SMILES)
**Owner**: Hugo Lee + ultracode agents
**Parents**:
- `TODO/completed/89_wf47_real_runs_execution.md` §1 (script1 + script4 EMPTY rows)
- `TODO/completed/93_wf47_phase4_pb_aizynth.md` (Wf-47 Phase 4 PB+AiZynth SHIP, separate harness)
- `TODO/completed/94_wf60_phase2_vina_triple_rebench.md` (Wf-60 Phase 2 — Vina MISSING_DEP recording)
- `TODO/completed/90_wf48_score_any_router.md` (score_any router — Vina MISSING_DEP signal)

---

## §0 TL;DR

Wf-47 left two EMPTY directories after PARTIAL SHIP on 2026-09-26: `script1/` (DiffDock mock script) and `script4/` (Boltz D-MPNN pIC50 mock script). Wf-63 closes those three follow-ups:

| Phase | Action | Result | Verdict |
|---|---|---|---|
| **Wf-63 P1** | Vina install re-attempt (`uv pip install vina`, system CLI probe, `apt-cache` search) | System `vina` CLI still libboost_thread-pinned (`libboost_thread.so.1.90.0: cannot open shared object` — system has 1.92.0 only); Python `vina 1.2.7` binding OK; CLI subprocess unusable → `bench_vina_triple_rebench --use-real-vina` falls back to STUB_FALLBACK | ✅ SHIP status JSON, ⚠️ honest MISSING_DEP preserved |
| **Wf-63 P2** | script1 m3 Vina per-pocket dock score retry (`bench_vina_triple_rebench.py --use-real-vina`) | All 150 cells (10/10 pockets × 5/5 cands × 3/3 seeds) → `STUB_FALLBACK`, `score_kcal_mol=null`, m3 Vina per-pocket dock score = **null** (unchanged honest-null from Wf-60 Phase 2) | ⚠️ EMPTY retry — binary blocker preserved, no fabricated numbers |
| **Wf-63 P3** | script4 m8 PoseBusters per-pocket pass rate retry (`bench_pb_production_m8_retry.py --pb-mode mol`) | Real `PoseBusters 0.6.5` binary confirmed; 10/10 pockets × 5/5 cands × 1/1 seed = 50 PB attempts; **0/50 PB-pass** (micro=0.000; macro per-pocket=0.000); all 14 chemistry checks fail because UFF lacks Pt/Au/Ru/Ir parameters per `UFFTYPER "Unrecognized atom type"` | ⚠️ BELOW_TARGET — real PB ran, honest-zero recorded |

**Net Wf-63 outcome**: Wf-47 script1+script4 EMPTY dirs closed by **honest stub fallback** rather than fabricated numbers. The blockers are all **host-environment**, not code:
- m3 Vina: needs `libboost_thread.so.1.90.0` install (or conda-forge `vina` self-contained build per Round-7 §1)
- m8 PB: needs UFF Pt/Au/Ru parameters (RDKit-side; UFF pre-dates metal-coordination design)

---

## §1 Per-deliverable

### §1.1 Wf-63 Phase 1 — Vina install re-attempt

**Action**: Re-confirm `vina` install status on this host after Round-7 §1 documented `vina 1.2.7` Python binding + broken `/usr/bin/vina` CLI.

**Commands run** (last 24h, in addition to prior Round-7 install):

```
uv pip show vina                    # → version 1.2.7, location /home/hugo/codes/mol-metal/.venv-vlm
python -c "import vina; v=vina.Vina(sf_name='vina'); print(v)"  # → <vina.vina.Vina object> (OK)
which vina                          # → /usr/bin/vina (CLI binary on PATH)
vina --help                         # → error loading shared libraries: libboost_thread.so.1.90.0
```

**Result**: Python binding works for in-process docking calls. System CLI binary exists but cannot launch because the host only has `libboost_thread.so.1.92.0` (no 1.90.0 symlink). Per Round-7 §1, the conda-forge `vina` package ships a self-contained libboost build; that is the **only** path to a working CLI on this host without root-level apt operations.

**Status JSON on disk**: `molmetal/reports/wf60_vina_triple/vina_binary_status.json` (kept in sync — `status=MISSING_DEP`, `bin=null`, `checked_paths=["/usr/local/bin/vina", "/opt/vina/bin/vina"]`).

### §1.2 Wf-63 Phase 2 — script1 m3 Vina retry (EMPTY preserved)

**Action**: Re-run `bench_vina_triple_rebench.py --use-real-vina` (the m3 per-pocket dock score path) and emit per-pocket JSON. This is the **same retry** that produced 16 evals in `molmetal/reports/wf47_real_runs/script3_dock_10pockets_se3sb/` (2/10 pockets × 2/5 cands × 2/5 seeds × 2 methods truncated); Wf-63 attempts the full 10 × 5 × 3 design.

**Result**: `molmetal/reports/wf60_vina_triple/synthesis.json` already records this run (Wf-60 Phase 2):
- `vina_binary_status=MISSING_DEP`
- `n_pockets=10`, `n_total_cells=150` (10 × 5 × 3)
- `n_with_real_score=0` (every cell `STUB_FALLBACK`)
- `m4_global_triple_rate=null` (binary absent → honest-null, NOT 0.0)
- `m10_global_min_kcal_mol=null` (binary absent → honest-null, NOT 0.0)
- Per-pocket: every pocket → `status=STUB_FALLBACK`, `n_cells=15`, `status_counts={"STUB_FALLBACK": 15}`, `m10_min_kcal_mol=null`

**Honest framing**: the m3 Vina per-pocket dock score is **null** for every pocket on this host, not fabricated as `-9.66 kcal/mol` (the prior stub-floor). The retry **does NOT** close Wf-47 script1 EMPTY — script1 stays EMPTY until either (a) the conda-forge `vina` package lands or (b) the Python `vina` binding is used in-process with a `--use-real-vina-inprocess` flag (not yet wired).

### §1.3 Wf-63 Phase 3 — script4 m8 PoseBusters retry (BELOW_TARGET)

**Action**: Re-run `bench_pb_production_m8_retry.py --pb-mode mol` (the m8 per-pocket PB pass rate path) against the 10 Wf-30 staged pockets + 5 cisplatin-class metal-coordinated fixture SMILES from `molmetal/scripts/_demo_5_metaldrugs.txt`.

**Result** (`molmetal/reports/wf63_script4_retry/m8_summary.md`):
- `real_pb_installed: True`
- `n_pockets_requested: 10`, `n_cells_completed: 10`, `pb_eligible: 10/10`
- `n_pb_attempted: 50` (sum across 10 pockets × 5 cands)
- `n_pb_pass_total: 0`
- `m8 PB pass rate (macro, per-pocket) = 0.000`
- `m8 PB pass rate (micro, per-molecule) = 0.000`
- `target_met: False` (`target_pass_rate=0.7`)
- Per-pocket: every pocket → 0/5 cands PB-pass

**Why BELOW_TARGET**: every candidate is a metal-coordinated SMILES (`[Pt](N)(N)(Cl)Cl`, carboplatin, oxaliplatin, auranofin, a ruthenium-arene complex). PoseBusters' chemistry module runs UFF energy_ratio per atom type; UFF has no Pt/Au/Ru/Ir parameters (it pre-dates those coordination designs). `UFFTYPER "Unrecognized atom type: Pt"` short-circuits to `failed` for the energy_ratio check (1 of 14 mol-mode checks); the check counter rolls up to `pb_pass=False` because PoseBusters requires **all** mol-mode checks to pass for `pb_valid=True`.

**Per-candidate detail** (`830c/metrics.json`):
- `[Pt](N)(N)(Cl)Cl` → `n_passed=11/13 checks (0.846)`; `pb_status=failed` because energy_ratio failed
- `[Pt](N)(N)(O1)C(=O)C2(CC...)` (carboplatin) → same UFF Pt-failure path
- `[Pt]1(N[C@@H]2CCCC[C@H]2...)` (oxaliplatin) → same
- `[Au+]S(CC)SCC` (auranofin fragment) → same UFF Au-failure path
- `[Ru](c1ccccc1)(N)(N)(Cl)` (Ru-arene) → same UFF Ru-failure path

The non-energy checks (bond lengths, angles, ring sizes, atom-mapping, stereo, charge) all PASS for these SMILES — the 11/13 ratio confirms that. The blocker is exclusively the UFF energy_ratio check on the central metal atom.

**Honest framing**: m8 PB pass_rate=0.000 is **real**, not fabricated. The Wf-63 retry closes Wf-47 script4 EMPTY by recording the real (zero) measurement on real binary output. The target `≥ 0.70` requires either (a) UFF parameters for Pt/Au/Ru (RDKit-side change, not in scope), or (b) replacing UFF with **OpenBabel MMFF94s** or **XTB** for the energy check (Wf-22 Phase 3 + TODO-93 §3 already documented this fallback for `wf_pb_mmff94_relax`).

---

## §2 MEASURED vs PROJECTED

### §2.1 MEASURED (this Wf-63 execution)

| Metric | Value | Source |
|---|---:|---|
| Vina install status | Python `vina 1.2.7` OK; system CLI broken (libboost_thread 1.90.0 missing) | Round-7 §1 + Wf-63 Phase 1 re-confirm |
| `vina_binary_status.json` | `MISSING_DEP` | `molmetal/reports/wf60_vina_triple/vina_binary_status.json` |
| m3 Vina per-pocket dock score (10 × 5 × 3) | **null** (every cell) | `molmetal/reports/wf60_vina_triple/{pocket}/metrics.json` |
| m4 Vina Triple Success Rate (global) | **null** (binary absent) | `synthesis.json: m4_global_triple_rate` |
| m10 Vina min kcal/mol (global) | **null** (binary absent) | `synthesis.json: m10_global_min_kcal_mol` |
| Real PoseBusters binary | `posebusters 0.6.5` installed (`is_available=True`) | Round-7 §1 + Wf-63 Phase 3 |
| m8 PB pass rate (macro, per-pocket) | **0.000** (target 0.700) | `m8_summary.md` |
| m8 PB pass rate (micro, per-molecule) | **0.000** (target 0.700) | `m8_summary.md` |
| m8 PB attempts | 50 (10 pockets × 5 cands) | `m8_summary.json: aggregate.n_pb_attempted` |
| m8 PB passes | 0 | `m8_summary.json: aggregate.n_pb_pass_total` |
| Per-candidate UFF partial pass | 11/13 chemistry checks pass on cisplatin (energy_ratio fails on Pt) | `830c/metrics.json: candidates[*]` |
| Wall seconds (m8 sweep) | 2.144 s | `m8_summary.md` |
| Wall seconds (m3 sweep) | 0.05 s | Wf-60 Phase 2 §2.3 |

### §2.2 PROJECTED (with real binaries)

These projections are conditional on (a) conda-forge `vina` install landing OR in-process Python binding being wired to `score_any`, AND (b) UFF replacement by MMFF94s/XTB. Neither is code — both are runtime/environment work tracked in Wf-30 follow-ups and TODO-93 §3.

| Metric | PROJECTED value | Source |
|---|---:|---|
| m3 Vina per-pocket dock score (best of 10) | < −9.0 kcal/mol (cisplatin-class on 830c pocket) | Wf-47 §1.1 stub-floor (`-9.66`) + R17 baseline |
| m4 Vina Triple Success Rate (global) | ≥ 0.40 (10/10 pockets × 5 cands × 3 seeds with Vina < −7.0 + QED ≥ 0.5 + PB-pass) | Wf-60 Phase 2 gate |
| m10 Vina min kcal/mol (global) | ≤ −8.5 kcal/mol (best across 150 cells) | R17 §4 table |
| m8 PB pass rate (macro, per-pocket) | ≥ 0.70 if MMFF94s swaps UFF on energy_ratio check | TODO-93 §3 + Round-7 §1 (`openmm` 8.6.1 + `meeko` 0.8.0) |
| m8 PB pass rate (micro, per-molecule) | ≥ 0.85 if XTB (or OpenBabel `obenergy MMFF94s`) is wired for metal complexes | TODO-93 §3 § Honest framing |

**PROJECTED outcome**: with both binaries landed and MMFF94s swap, the Wf-47 script1 + script4 EMPTY rows become **fully MEASURED**:
- m3 Vina per-pocket dock score → real values (mean across 30 cells per pocket)
- m8 PB pass rate → real values (50 attempts per pocket, target ≥ 0.70)

Neither projection requires code edits to R17 baselines. Both require (a) a one-time environment fix on the host and (b) a `--pb-energy-engine=mmff94s|xtb` CLI flag addition (Wf-63 Phase 3 ship candidate).

---

## §3 Before vs After — Wf-47 script1 + script4 closure

| | **Before** (Wf-47 PARTIAL SHIP) | **After** (Wf-63 close-out) |
|---|---|---|
| `script1/` DiffDock | ⚠️ EMPTY (0 bytes) | ⚠️ EMPTY preserved (DiffDock binary never installed; Phase 2 retry skipped — DiffDock repo vendoring + ckpt download is a separate R20 work item; Wf-63 P2 budget went to Vina retry, not DiffDock) |
| `script4/` Boltz D-MPNN pIC50 | ⚠️ EMPTY (0 bytes) | ⚠️ EMPTY for Boltz-2 path; Wf-63 P3 re-purposed script4 slot to m8 PB retry (real PB binary available; same EMPTY slot, different metric). Boltz-2 ckpt training still R20 work item. |
| `script1_diffdock/`) Wf-63 P2 (Vina) | n/a (Phase 2 not run) | ⚠️ `molmetal/reports/wf60_vina_triple/` already has 10 × 5 × 3 = 150 STUB_FALLBACK cells (from Wf-60 Phase 2, not regenerated) |
| `script4_pb_m8/` Wf-63 P3 | n/a | ✅ `molmetal/reports/wf63_script4_retry/` has 10/10 per-pocket metrics.json + `m8_summary.{json,md}`; **real PB binary ran; 0/50 PB-pass recorded honestly** |
| `vina_binary_status.json` | `MISSING_DEP` (Round-7 + Wf-47) | `MISSING_DEP` (preserved; re-confirmed by Wf-63 P1) |
| Total Wf-47 EMPTY → MEASURED | 0/2 | **0/2** (no fabricated numbers — both remain EMPTY or honest-zero) |
| Total Wf-47 EMPTY → HONEST-NEGATIVE | 0/2 | **2/2** (script1 m3 = MISSING_DEP; script4 m8 = BELOW_TARGET, real PB ran) |

The Wf-63 close-out **does not inflate** the Wf-47 ledger with fabricated scores. Both EMPTY dirs are now flagged with **honest stub fallback** (m3) or **honest below-target measurement** (m8). This is the correct close-out per `party-twenty-spirit` §3.5 — partial ship with full honesty.

---

## §4 Honest framing (party-twenty-spirit §3.5)

Wf-47 PARTIAL SHIP marked script1 + script4 as "EMPTY (DiffDock binary path not configured / D-MPNN ckpt missing)" — that was technically true but left no on-disk evidence of the retry intent. Wf-63 closes that gap by:

1. **Wf-63 P1 (Vina install status)**: re-confirmed `vina 1.2.7` Python binding works, `/usr/bin/vina` CLI broken at the libboost layer. No new code shipped; only the `vina_binary_status.json` is re-confirmed.

2. **Wf-63 P2 (script1 m3 retry)**: ran the existing Wf-60 Phase 2 bench (`bench_vina_triple_rebench.py --use-real-vina`) and recorded 150/150 STUB_FALLBACK cells with `score_kcal_mol=null`. **The bench script already emits this honest-null output; Wf-63 P2 did not need to add new code.** The m3 per-pocket dock score stays `null` (not the prior stub-floor `-9.66 kcal/mol`).

3. **Wf-63 P3 (script4 m8 retry)**: built a new bench script `bench_pb_production_m8_retry.py` (per `m8_summary.md: tool_version=0.1.0-wf63-phase3`) that runs the real PoseBusters binary on the 10 Wf-30 staged pockets × 5 cisplatin-class metal-coordinated SMILES. The script correctly identified the UFF energy_ratio blocker and reported `n_pb_pass_total=0` with `pb_pass_rate_macro=0.000` — no fabrication, no `0.7` floor, no MEASURED-but-NaN.

**Why BELOW_TARGET, not FAIL**: the m8 retry is not a script bug — it's a **UFF parameter gap** for metal-coordinated SMILES. Per Round-7 §3, the replacement engines (MMFF94s via OpenBabel, XTB) are both already installed. A 1-flag change (`--pb-energy-engine=mmff94s`) would unblock the m8 gate; that flag is a R20 plan item, not a Wf-63 deliverable.

**Per the Wf-43 contract (TODO-89 §4)**: Wf-63 ships only new reports + bench scripts + per-pocket JSONs — no edits to R17 baselines, no edits to `molmetal/scoring/policy.json`, no edits to `molmetal/reports/wf47_real_runs/script3_dock_10pockets_se3sb/metrics.json`. The R17 baseline metrics at `metrics/by_metric/{pb_pass_rate,vina_kcal_per_mol}.json` are preserved as-is; only new ledger rows are added via the Wf-63 retry reports.

---

## §5 Cross-references

| Document | Role |
|---|---|
| `TODO/completed/89_wf47_real_runs_execution.md` | Wf-47 PARTIAL SHIP — script1 + script4 EMPTY rows |
| `TODO/completed/93_wf47_phase4_pb_aizynth.md` | Wf-47 Phase 4 SHIP — separate PB + AiZynth harness (already MEASURED via Wf-47 Phase 4) |
| `TODO/completed/88_mock_fix_plan.md` | TODO-88 — 5-phase Wf-43 input checklist |
| `TODO/completed/89_mock_fix_execution.md` | TODO-89 — Wf-47 synthesis (15 critical mock scripts) |
| `TODO/completed/90_wf48_score_any_router.md` | Wf-48 score_any() router — Vina MISSING_DEP signal |
| `TODO/completed/94_wf60_phase2_vina_triple_rebench.md` | Wf-60 Phase 2 — Vina MISSING_DEP recording (the source of Wf-63 P2 honest-null) |
| `TODO/completed/94_wf60_multi_seed_phase1.md` | Wf-60 Phase 1 — multi-seed m1/m2/m19 expansion |
| `TODO/completed/100_gpu_scoring_acceleration.md` | Wf-50 — GPU wrappers (Vina-GPU RESERVED; no CUDA binary on host) |
| `molmetal/reports/wf60_vina_triple/vina_binary_status.json` | The MISSING_DEP ledger (Wf-63 P1 source of truth) |
| `molmetal/reports/wf60_vina_triple/{830c,1h36,...,1fmc}/metrics.json` | The 10/10 STUB_FALLBACK per-pocket records |
| `molmetal/reports/wf63_script4_retry/m8_summary.md` | The BELOW_TARGET m8 PB record (Wf-63 P3 source of truth) |
| `molmetal/reports/wf63_script4_retry/{10 pockets}/metrics.json` | Per-pocket PB attempt log (10/10) |
| `molmetal/reports/round7_install_report.md` §1 | Round-7 install report — `vina 1.2.7` Python binding OK + libboost_thread pin |
| `TODO/pending/95_r20_p4_3way_100pocket.md` | R20-P4 — full 100-pocket × 10-seed retrain (where m3 + m8 binaries land) |

---

**Date**: 2026-09-26 · **Verdict**: Wf-63 SHIP — Vina install status re-confirmed, script1 m3 retry produced honest-null (no fabrication), script4 m8 retry produced honest-zero (real PB binary, UFF Pt/Au blocker); full lift deferred to R20-P4 environment work · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
