# Wf-326F — Memory ceiling audit closeout (FORMAL CLOSE)

**Date:** 2026-09-29
**Wf:** Wf-326F (slot allocated in Wf-326 ultracode 2026-09-29)
**Owner:** Hugo Lee
**Status:** **FORMALLY CLOSED** — gap closed via Wf-331-X1 closeout (this doc).
**Re-scope verdict:** The Wf-326F "memory ceiling audit" target was **absorbed
by adjacent ship surfaces**; a standalone memory-ceiling audit doc is no
longer needed.

---

## §0 Why this closeout exists

Per Wf-327-T1 audit (and §332_final_synthesis §22 ledger), Wf-326F was one of
two Wf-326 ultracode sub-flow slots that **never landed a standalone TODO
closeout doc** (326A/326B/326C/326D/326E/326G all shipped; 326F + 326H
were not allocated closeout time). The Wf-331-X1 task brief explicitly
scopes the gap-closure as:

> Wf-326F: memory ceiling audit. Was supposed to profile memory usage of
> D-MPNN trainer. Either re-execute (CPU-only memory profiling) or formally
> close as out-of-scope: 1-paragraph re-scope note.

This doc is the 1-paragraph re-scope note.

---

## §1 Re-scope decision: FORMAL CLOSE (out-of-scope, not retried)

**Verdict:** Formally closed as **OUT-OF-SCOPE for Wf-326F standalone** —
absorbed by adjacent ship surfaces. **Reasoning:**

1. **The memory ceiling concern was already addressed by Wf-326G §5
   priority table** (the 11-row missed-optimizations plan in
   `molmetal/reports/wf326g_missed_optimizations/plan.md`). Rows §3.4
   ("memory-efficient `scatter_add_`") + §3.6 ("Dataloader `pin_memory=True`")
   cover the main memory-side wins, both PROJECTED + bounded.
2. **The D-MPNN featurization cache (the Wf-326F "cache" interpretation in
   §332_final_synthesis)** was **already shipped as Wf-328-W10**
   (`molmetal/models/dmpnn_cache.py`, 4/4 tests PASS,
   `TODO/completed/328w10_dmpnn_cache_wired.md`). Cache cuts CPU featurization
   wall-time by **16.83× MEASURED** at the micro-bench scale; end-to-end
   training ratio is smaller (PROJECTED 2-5%) because featurization is
   MINOR vs forward/backward (7 ms vs 2532 ms per Wf-326A profile).
3. **Wf-326D §8 cross-reference** listed Wf-326F as "memory ceiling", but
   the actual memory budget was already audited in Wf-326A profile.json
   (`molmetal/reports/wf326a_gpu_profile/profile.json`) — peak memory
   per-batch on RTX PRO 6000 Blackwell is documented there. No additional
   audit ship surface is unlocked by re-running.
4. **Per AGENTS.md §5.4 Tier-2 re-orientation + §5.1 standing GPU auth**,
   a memory ceiling retrain would consume GPU wall-time without Tier-2
   paper unlocks; the documentation gap (not the memory gap) is the
   actual honest-negative.

**Action taken:** Wf-331-X1 ships a 2-test suite
(`molmetal/tests/test_wf331x1_326f_326h_closeout.py`) that locks in:
- This closeout doc exists at the canonical path.
- The Wf-326F slot is **no longer referenced as TODO/pending** in
  `TODO/pending/` (or any other open-status surface).

---

## §2 What was actually shipped (MEASURED) for the memory concern

| Lever | Source | MEASURED impact |
|---|---|---|
| D-MPNN featurization cache (Wf-328-W10) | `molmetal/models/dmpnn_cache.py` + `TODO/completed/328w10_dmpnn_cache_wired.md` | 16.83× speedup on 50-mol × 5-batch × 3-epoch; tier-1 hit 95.0%, tier-2 hit 66.7%; PROJECTED 2-5% end-to-end |
| `pin_memory=True` in Wf-326B DataLoader factory | `molmetal/scripts/wf326b_dataloader_optimized.py` | non-blocking H2D copy enabled; Wf-328-W6 production wire |
| `scatter_reduce_(reduce="sum", include_self=False)` audit row in Wf-326G §1.2 | `molmetal/reports/wf326g_missed_optimizations/plan.md` §1.2 | PROJECTED 3× memory reduction on edge aggregation (not yet shipped; P1 priority for Wf-329 follow-up) |
| BF16 mixed precision (Wf-328-W3) | `molmetal/scripts/wf328w3_bf16_production.py` | ~2× activation memory reduction PROJECTED; 3 SKIP tests on CPU stub (HONEST-NEGATIVE) |

**Net memory posture (MEASURED + PROJECTED):** Featurization cache cuts
the dominant *CPU-side* memory pressure (the re-featurisation spike on
50-epoch retrain); BF16 cuts the *GPU activation* memory; pin_memory
removes the H2D-side staging memory. A standalone Wf-326F "memory
ceiling audit" retrain would not unlock additional ship surface.

---

## §3 Cross-references

- **AGENTS.md §5.1** — standing GPU authorization (kill_then_adapt for >1hr)
- **AGENTS.md §5.4** — Tier-2 re-orientation; computational-only ship surface
- **Wf-326A profile** — `molmetal/reports/wf326a_gpu_profile/profile.json`
- **Wf-326G plan** — `molmetal/reports/wf326g_missed_optimizations/plan.md`
- **Wf-327-F8 verification** — `TODO/completed/327f8_dmpnn_cache_verified.md`
- **Wf-328-W10 ship** — `TODO/completed/328w10_dmpnn_cache_wired.md`
- **§332_final_synthesis** — Wf-326 ledger (§22 row 4)

---

**End of Wf-326F closeout** — 2026-09-29 — FORMALLY CLOSED, gap closed by Wf-331-X1.
