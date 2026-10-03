# 246 — Wf-246-C metal-coordination term retrain SCAFFOLD

**Document ID:** TODO-246 / Wf-246-C-METAL-COORD-RETRAIN-SCAFFOLD
**Version:** 1.0 (initial SCAFFOLD brief)
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per Wf-246-C plan + Wf-333-V1 kcal sign fix + Wf-390-A1 Path A
5-metal subset + Wf-409-A3 ultracode wave
**Cross-refs:**
- `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py` (Wf-333-V1 sign-fixed kcal_adjustment at line 635)
- `TODO/completed/333v3_vina_p2p3.md` (Wf-333-V3 Vina P2/P3)
- `TODO/completed/406a4_wf246c_scaffold.md` (Wf-406-A4 sister SCAFFOLD)
- `TODO/completed/390a3_se3_sb_wire.md` (Wf-390-A3 SE(3) SB wire-in audit)

---

## §1 Status

**Status:** SCAFFOLD SHIPPED via Wf-409-A3 (this entry — moved to
`TODO/completed/246_wf246c_metal_coord_retrain_scaffold.md` per Wf-409-A3
close). Actual GPU retrain BLOCKED on `USER_OK=1` per AGENTS.md §5.1 +
Wf-355-B1 USER_OK gate pattern.

---

## §2 Goal

Wire `molmetal.molmetal_lam.sbdd_env.metal_coord_term.MetalCoordinationTerm`
into the Wf-131 P3 D-MPNN multi-task trainer's training loop as a soft
kcal/mol adjustment on top of the Wf-131 pIC50 regression loss +
Wf-318 Wasserstein regulariser.

The wire-in is **opt-in** via the `--metal-coord-term` CLI flag
(default OFF for bit-for-bit backward compat with Wf-131 P1 / Wf-299 /
Wf-328 / Wf-131 P3 5-metal subset shipped runs).

---

## §3 Wf-409-A3 deliverables

- SCAFFOLD script: `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py`
  435 LOC (5-step CPU smoke + kcal sign audit + per-epoch audit)
- Launch wrapper: `scripts/launch_wf409a3_metal_coord_retrain.sh` 236 LOC
  (USER_OK gate + DRY_RUN.log + plan.json + R1/R2/R3 mode dispatch)
- Tests: `molmetal/tests/test_wf409a3_metal_coord_scaffold.py` 2 NEW tests
  PASS in ~3 s wall CPU-only (pre-flight conftest bug pre-existing,
  verified via `--noconftest`)

---

**END OF TODO-246 brief** — see `TODO/completed/409a3_wf246c_scaffold.md`
for full Wf-409-A3 closeout.
