# Wf-326H — Wf-326 series synthesis closeout (FORMAL CLOSE)

**Date:** 2026-09-29
**Wf:** Wf-326H (slot allocated in Wf-326 ultracode 2026-09-29)
**Owner:** Hugo Lee
**Status:** **FORMALLY CLOSED** — synthesis already shipped via Wf-327 +
Wf-330-E + §332_final_synthesis; gap closed via Wf-331-X1 closeout (this
doc).
**Re-scope verdict:** The Wf-326H "synthesis" target was **absorbed by
adjacent ship surfaces**; a standalone Wf-326H doc would duplicate
existing ledger rows.

---

## §0 Why this closeout exists

Per Wf-327-T1 audit + Wf-331-X1 task brief:

> Wf-326H: synthesis. Was supposed to aggregate Wf-326 findings.
> Either re-execute (CPU-only synthesis) or formally close as
> out-of-scope.

This doc is the 1-paragraph re-scope note.

---

## §1 Re-scope decision: FORMAL CLOSE (already-shipped synthesis)

**Verdict:** Formally closed as **ALREADY-SHIPPED via downstream
synthesis surfaces** — re-running would produce a duplicate ledger.
**Evidence:**

1. **Wf-327 synthesis** (`TODO/completed/327_synthesis.md`) — the
   authoritative Wf-326B follow-up ledger; 7/9 items marked SHIPPED
   via Wf-328 wave (W1-W10); 2 items marked PARTIAL with HONEST-NEGATIVE
   disclosure (chemprop MoleculeDataLoader + CFM EGNN loader).
2. **Wf-327-F8 closeout** (`TODO/completed/327f8_dmpnn_cache_verified.md`)
   — synthesises the Wf-326F cache status: VERIFIED, 4/4 tests PASS,
   cache shipped as Wf-328-W10. **Wf-326H "synthesis" supersedes this.**
3. **Wf-330-E closeout** (`TODO/completed/330e_wf327_wf328_closeouts.md`)
   — synthesises the Wf-327 + Wf-328 wave into a single 10-row ledger.
4. **§332_final_synthesis** (`TODO/completed/332_final_synthesis.md` §22
   row 4) — the project-wide ship ledger; Wf-326 row lists all 8 sub-flows
   (326A + 326B + 326C + 326D + 326E + 326F + 326G + 326H) with
   MEASURED-vs-PROJECTED + honest-negative framing per `party-twenty-spirit
   §3.5`. **This is the authoritative Wf-326 synthesis.**

**Action taken:** Wf-331-X1 ships a 2-test suite
(`molmetal/tests/test_wf331x1_326f_326h_closeout.py`) that locks in:
- This closeout doc exists at the canonical path.
- The Wf-326H slot is **no longer referenced as TODO/pending** in
  `TODO/pending/` (or any other open-status surface).

---

## §2 Authoritative Wf-326 ledger (per §332_final_synthesis §22 row 4)

| Sub-flow | Topic | Status |
|---|---|---|
| **326A** | GPU utilization profile (RTX 5090 + RTX PRO 6000) | SHIPPED — `TODO/completed/326a_gpu_utilization_profile.md`; G6 SM 4-7% MEASURED |
| **326B** | DataLoader optimization | SHIPPED — `molmetal/scripts/wf326b_dataloader_optimized.py` + 8/8 tests; Wf-328-W6 production wire |
| **326C** | torch.compile + CUDA Graphs | SHIPPED — `TODO/completed/326c_torch_compile_optimization.md`; MEASURED 2.72× encoder + 2.41× W₂ |
| **326D** | AMP/BF16 batch optimization | SHIPPED — `TODO/completed/326d_amp_batch_optimization.md`; BF16 + batch=256 PROJECTED; 3 SKIP tests HONEST-NEGATIVE |
| **326E** | Triton kernel coverage audit | SHIPPED — `TODO/completed/326e_triton_audit.md`; CPU-only audit; harness + scopes locked |
| **326F** | Memory ceiling / D-MPNN featurization cache | SHIPPED (via Wf-328-W10) + FORMALLY CLOSED (this Wf-331-X1 closeout) |
| **326G** | Missed optimizations plan | SHIPPED — `TODO/completed/326g_missed_optimizations_plan.md`; 8/8 audit tests PASS + 11-row priority table |
| **326H** | Synthesis closeout | SHIPPED (via Wf-327 + Wf-330-E + §332) + FORMALLY CLOSED (this Wf-331-X1 closeout) |

**Net Wf-326 verdict (per §332 row 4 column 6):**
HONEST-NEGATIVE on GPU SM occupancy (CONFIRMED);
**CONFIRMED bottleneck = Python per-mol loop NOT kernel-side**.
PROJECTED 8-20× wall-time reduction on the P0 entries catalogued in
Wf-326G §5 (BatchMolGraph 5-15× + scatter_reduce 1.2-1.5× + atom-feature
caching 2-3× + edge-update batching 2-4× + torch.compile + TF32 + BF16).

---

## §3 Cross-references

- **AGENTS.md §5.2** — honest framing mandate (party-twenty-spirit §3.5)
- **Wf-327 synthesis** — `TODO/completed/327_synthesis.md`
- **Wf-327-F8 cache verification** — `TODO/completed/327f8_dmpnn_cache_verified.md`
- **Wf-330-E Wf-327+Wf-328 closeouts** — `TODO/completed/330e_wf327_wf328_closeouts.md`
- **§332_final_synthesis** — `TODO/completed/332_final_synthesis.md` §22 row 4

---

**End of Wf-326H closeout** — 2026-09-29 — FORMALLY CLOSED, gap closed by Wf-331-X1.
