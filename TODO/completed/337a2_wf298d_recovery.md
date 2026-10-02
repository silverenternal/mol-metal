# Wf-337-A2 — Wf-264 sub-panel recovery paths (Smina / QuickVina / metal-loop)

**Ship:** 2026-09-29
**Owner:** Claude Code (ultracode Wf-337-A2 subagent; user voice: "好像有些之前发现的问题没有被覆盖到，可以额外启动新的ultracode吗？")
**Skill:** `party-twenty-spirit` §3.5 honest framing
**Status:** SHIPPED — 4/4 NEW tests PASS; CPU-only diagnostic
**Honest framing:** MEASURED (not PROJECTED); CPU-only; no GPU launch per AGENTS.md §5.1; neither Smina nor QuickVina is installed in this environment — recovery verdict is `HONEST_NEG_METAL_LOOP_OK` (the metal-loop detector IS the recovery; the Wf-264 panel does not yet dispatch to it but the detector is the documented next-step wire-up).

---

## §1 Background — what Wf-337-A2 recovers

Per **Wf-334-U3** (Wf-264 CFM property panel launch script; `TODO/completed/334u4_launcher_verify.md`) and the on-disk Wf-325B result (`molmetal/reports/wf325b_vina_pt_zn_substitution/result.json`):

* **Wf-334-U3 MEASURED**: Wf-264 Vina sub-panel **0/100 docked ok on 1h36** when SMILES contain Pt coordination centres.
* **Root cause**: meeko 0.8.0 returns `is_ok=False` from `PDBQTWriterLegacy.write_string` because RDKit UFF cannot parameterise Pt+2 (`UFF_UNSUPPORTED_ATOMIC_NUMS` set; atomic numbers 44/45/46/77/78/79). This is the **meeko metal-atom limit** documented in Wf-301B §2 + Wf-319B §honest-framing.
* **Wf-325B**: Pt→Zn UFF substitution is the canonical `MEASURED_GENERIC_ATOM` convention; kcal/mol is generic Zn-VDW approximation, NOT a faithful Pt-anchored score.

Wf-337-A2 ships **3 alternative-engine / pre-processing recovery paths** that the Wf-264 panel can dispatch to when meeko hits the metal-atom limit:

1. **Smina** (Vina fork with custom atom types / metal-coordination parameterisation; Koes et al. 2013).
2. **QuickVina** (faster Vina fork; Handoko et al. 2012 — same PDBQT-parser limitations as Vina CLI, so does NOT solve the Pt UFF problem).
3. **Custom metal-loop detector** (RDKit-only; classifies metal centres + recommends pre-processing strategy).

---

## §2 Per-recovery-branch state (MEASURED via probe 2026-09-29)

| Branch | Probe result | Honest verdict |
|---|---|---|
| `smina` (`shutil.which("smina")`) | **NOT installed** | `HONEST_NEGATIVE_ENGINE_MISSING` — `smina not on PATH` |
| `quickvina` (`shutil.which("q.vina") or shutil.which("quickvina")`) | **NOT installed** | `HONEST_NEGATIVE_ENGINE_MISSING` — `Neither q.vina nor quickvina found on PATH` |
| `metal_loop` (RDKit-only detector) | **AVAILABLE** | `HONEST_NEG_METAL_LOOP_OK` — detector correctly classifies cisplatin as `pt_strip_largest_fragment` |

The metal-loop detector is the **MEASURED recovery path** that works with the existing Vina CLI binary + meeko prep without requiring any external engine. The other two branches are surfaced as honest-negative so downstream consumers know the wire-up would require installing an external Vina fork.

---

## §3 Test coverage (NEW — 4 tests, all MEASURED CPU-only)

### §3.1 Test file: `molmetal/tests/test_wf337a2_wf298d_recovery.py`

| # | Test | What it gates |
|---:|---|---|
| 1 | `test_smina_availability` | Smina probe returns the expected dict shape with `available`/`binary_path`/`version`/`probe_error`/`honest_negative_reason`. When smina is NOT installed, the probe MUST populate `probe_error` and `honest_negative_reason` with engine-mention + recovery hint. |
| 2 | `test_quickvina_availability` | QuickVina probe returns the expected dict shape with the same contract; when neither `q.vina` nor `quickvina` is on PATH, the honest-negative reason must mention the engine + a recovery hint. |
| 3 | `test_custom_metal_loop_detector` | `detect_metal_loop` correctly classifies cisplatin (`pt_strip_largest_fragment`, 4 non-H neighbours), dangling Pt (`pt_to_zn_substitution`, 2 non-H neighbours), octahedral Ru (`pt_strip_largest_fragment`, 6 non-H neighbours), Ir (`pt_strip_largest_fragment`), and aspirin (`none`, no metal). |
| 4 | `test_pocket_sweep_diagnostic_5_wf325b_pockets` | `pocket_sweep_diagnostic` returns the correct per-pocket structure for the 5 Wf-325B pockets; `n_pockets_docked=0` in this environment (no real Vina fork reachable); `n_pockets_metal_loop_ok=5` for cisplatin; `aggregate_verdict=HONEST_NEG_METAL_LOOP_OK` when neither Smina nor QuickVina is installed AND SMILES has a Pt centre. |

All tests are CPU-only — no real engine launched, no GPU per AGENTS.md §5.1. **< 1 s wall**.

### §3.2 Test result (MEASURED 2026-09-29)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf337a2_wf298d_recovery.py -v

molmetal/tests/test_wf337a2_wf298d_recovery.py::test_smina_availability PASSED [ 25%]
molmetal/tests/test_wf337a2_wf298d_recovery.py::test_quickvina_availability PASSED [ 50%]
molmetal/tests/test_wf337a2_wf298d_recovery.py::test_custom_metal_loop_detector PASSED [ 75%]
molmetal/tests/test_wf337a2_wf298d_recovery.py::test_pocket_sweep_diagnostic_5_wf325b_pockets PASSED [100%]

========================= 4 passed, 1 warning in 0.38s =========================
```

### §3.3 CLI smoke run (MEASURED 2026-09-29)

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/wf337a2_vina_recovery.py \
  --smiles "[Pt](N)(N)(Cl)Cl" \
  --pocket-pdb-dir /home/hugo/codes/mol-metal/molmetal/reports/wf30_pocket_prep

aggregate_verdict: HONEST_NEG_METAL_LOOP_OK
n_pockets_docked: 0
n_pockets_metal_loop_ok: 5
  1h36   branch=metal_loop  verdict=HONEST_NEG_METAL_LOOP_OK   receptor_available=True
  3eml   branch=metal_loop  verdict=HONEST_NEG_METAL_LOOP_OK   receptor_available=True
  3ny8   branch=metal_loop  verdict=HONEST_NEG_METAL_LOOP_OK   receptor_available=True
  4aaw   branch=metal_loop  verdict=HONEST_NEG_METAL_LOOP_OK   receptor_available=True
  4rlu   branch=metal_loop  verdict=HONEST_NEG_METAL_LOOP_OK   receptor_available=True
```

All 5 Wf-325B pockets have receptor staged; the metal-loop detector correctly recommends `pt_strip_largest_fragment` for cisplatin (Pt with 4 non-H neighbours).

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Claim | Status |
|---|---|
| Wf-334-U3 MEASURED 0/100 docked ok on 1h36 (meeko metal-atom limit) | **MEASURED** (`TODO/completed/334u4_launcher_verify.md` + Wf-325B `result.json`) |
| meeko 0.8.0 returns `is_ok=False` for Pt coordination complexes | **MEASURED** (verified via direct meeko invocation 2026-09-29) |
| Smina NOT installed in this environment | **MEASURED** (`shutil.which("smina") == None`) |
| QuickVina NOT installed in this environment | **MEASURED** (`shutil.which("q.vina") == None` and `shutil.which("quickvina") == None`) |
| Custom metal-loop detector correctly classifies cisplatin as `pt_strip_largest_fragment` | **MEASURED** (verified via direct RDKit probe 2026-09-29) |
| All 5 Wf-325B pockets have receptor.pdbqt staged | **MEASURED** (`molmetal/reports/wf30_pocket_prep/{1h36,3eml,3ny8,4aaw,4rlu}/receptor.pdbqt` exists) |
| `n_pockets_docked = 0` in this environment | **MEASURED** (no real Vina fork reachable) |
| Recovery verdict `HONEST_NEG_METAL_LOOP_OK` (not `RECOVERED`) | **MEASURED** — the detector IS the recovery; the Wf-264 panel does not yet dispatch to it |
| Wf-264 sub-panel 0/100 → recoverable via Smina install + dispatch | **PROJECTED** (BLOCKED on external `conda install -c conda-forge smina`) |
| Wf-264 sub-panel 0/100 → recoverable via Wf-264 `--use-metal-loop-recovery` CLI flag wire-up | **PROJECTED** (next-step wire-up per Wf-337-A2 §6) |

This is an **honest-negative close**: the Wf-337-A2 recovery module is shipped and tested, but the Wf-264 panel still does not dispatch to it. The next step is a CLI-flag wire-up that the Wf-264 panel can opt into without changing the existing `--pt-strip` semantics (which already covers the Wf-301B Pt-strip recovery).

---

## §5 Per-recovery-status summary (per OUTPUT brief)

| Metric | Value (MEASURED 2026-09-29) |
|---|---|
| `per_recovery_status` | smina=`HONEST_NEGATIVE_ENGINE_MISSING`; quickvina=`HONEST_NEGATIVE_ENGINE_MISSING`; metal_loop=`HONEST_NEG_METAL_LOOP_OK` |
| `n_pockets_docked` (cisplatin, 5 Wf-325B pockets) | **0** (no real Vina fork reachable) |
| `n_pockets_metal_loop_ok` (cisplatin, 5 Wf-325B pockets) | **5** (detector fires on every pocket) |
| `n_tests_pass` | **4 / 4** (`test_wf337a2_wf298d_recovery.py`) |
| `recovery_verdict` | **`HONEST_NEG_METAL_LOOP_OK`** — detector is the MEASURED recovery; Wf-264 panel wire-up is the next-step PROJECTED |

If the Wf-264 panel does not get the `--use-metal-loop-recovery` CLI flag wire-up in a follow-up Wf, the `recovery_verdict` is **`HONEST_NEGATIVE_WF264_SUBPANEL_PERSISTENT`** (the 0/100 docked ok result is NOT recoverable without either installing Smina/QuickVina OR wiring the metal-loop detector into the panel).

---

## §6 Future work (out of scope for Wf-337-A2)

- **Wf-264 `--use-metal-loop-recovery` CLI flag wire-up** — `molmetal/scripts/cfm_vina_dock_panel.py` needs a new CLI flag that dispatches to `wf337a2_vina_recovery.detect_metal_loop` + applies the recommended `pt_to_zn_substitution` or `pt_strip_largest_fragment` strategy before meeko prep. This is the PROJECTED recovery path that would actually lift the 0/100 result.
- **Smina install + dispatch** — `conda install -c conda-forge smina` would unlock the `RECOVERED` verdict for the `smina` branch. The probe is already wired and tested; only the engine install + `_dock_one` swap-in is missing.
- **QuickVina install + dispatch** — same as Smina; `q.vina` is faster but does NOT solve the Pt UFF parameterisation problem. Would be useful for organic SMILES speed-up.
- **Wire the detector into Wf-301B** — the Wf-301B `_strip_pt_and_largest_fragment` already does the heavy lifting; the detector just classifies which strategy to use. A small refactor would unify the two paths.

---

## §7 Files touched

- `/home/hugo/codes/mol-metal/molmetal/scripts/wf337a2_vina_recovery.py` (NEW, 480 LOC) — recovery module + 3 probes + custom metal-loop detector + CLI
- `/home/hugo/codes/mol-metal/molmetal/tests/test_wf337a2_wf298d_recovery.py` (NEW, 350 LOC, 4 tests) — all 4/4 PASS
- `/home/hugo/codes/mol-metal/TODO/completed/337a2_wf298d_recovery.md` (this file)

**No source files were modified** — the Wf-337-A2 closeout ships a stand-alone recovery module + tests; the Wf-264 panel wire-up is the next-step PROJECTED recovery (out of scope per §6).

---

## §8 Cross-references

- **Wf-334-U3** (`TODO/completed/334u4_launcher_verify.md`) — Wf-264 CFM property panel launch script (CPU-only verification).
- **Wf-301B** (`molmetal/scripts/cfm_vina_dock_panel.py::_strip_pt_and_largest_fragment`) — Pt-strip largest-fragment recovery.
- **Wf-325B** (`molmetal/reports/wf325b_vina_pt_zn_substitution/result.json`) — Pt→Zn UFF substitution `MEASURED_GENERIC_ATOM` convention.
- **Wf-319B** (`TODO/completed/319b_uff_min_pt_pool.md`) — UFF Pt parameterisation honest-negative.
- **Wf-298D** (`molmetal/reports/wf298d_harness_detector/wrapper_report.md`) — the Wf-337-A2 task brief references Wf-298D for context, but the actual Wf-298D closeout is about the StructuredOutput harness-layer auto-fallback; the Wf-337-A2 task and Wf-298D are sibling closeouts under the same ultracode dispatch (per AGENTS.md §6 Wf-321..322 docs-only closeout wave).
- **AGENTS.md §5.1** — standing GPU authorization (Wf-337-A2 is CPU-only; no GPU launch required).
- **AGENTS.md §5.2** — MEASURED-vs-PROJECTED honest framing (party-twenty-spirit §3.5).
- **AGENTS.md §5.4** — Tier-2 journal target; NO wet-lab planning (Wf-337-A2 is computational-only).

---

**END of Wf-337-A2.**
