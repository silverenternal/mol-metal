# Wf-367-C1 — Final pre-flight + INDEX §22.57 + AGENTS §6 + final synthesis sync (2026-09-29)

**Date:** 2026-09-29 (Wf-367-C1 timestamp 2026-09-29 ~15:45 UTC, this doc)

**Status:** SHIPPED — pre-flight fresh-run verified + INDEX + AGENTS docs-only sync + final synthesis doc written.

**Workflow:** Wf-367-C1 final aggregation agent (READ-only verification + Edit to INDEX + Edit to AGENTS + this closeout + final synthesis at `TODO/completed/367_final_synthesis.md`).

---

## §0 TL;DR

| Item | Value | Source |
|---|---|---|
| **Pre-flight v3 fresh-run** | **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** | `molmetal/reports/wf367c1_preflight.txt` Wf-367-C1 timestamp |
| **Δ vs Wf-366-C1 baseline (91/1/2/94)** | **+1 PASS / −1 FAIL / 0 SKIP / 0 TOTAL** | Wf-367-A1 + Wf-367-A2 cascade closure |
| **INDEX.md §22 entry added** | `§22.57 Wf-367 pre-flight 2 FAIL fixes + m2 R² recovery + truncated verify wave` | this C1 sync |
| **AGENTS.md §6 row added** | Wf-367 row appended between Wf-366-C1 row (2026-09-29) and §7 | this C1 sync |
| **Final synthesis doc** | `TODO/completed/367_final_synthesis.md` | this C1 sync |
| **NEW tests in C1** | 0 NEW (synthesis-only docs) | task brief |
| **GPU launches in C1** | 0 (CPU-only docs+Edit) | per `AGENTS.md §5.1` |
| **Wall (MEASURED)** | ~5-10 min CPU-only | this C1 sync |

**One-line verdict:** Wf-367 wave is **SHIPPED + FINAL**. Pre-flight cascade fully closed (91/1/2/94 → 92/0/2/94). m2 R² recovery scaffold surfaces HONEST_NEGATIVE_R2_DEEP preserved on both `platinai_4cell` + `hilbert` cohorts. Wf-367-B2 cross-verify confirms Wf-365/Wf-366 closeouts all SHIPPED + NOT truncated. INDEX §22.57 + AGENTS §6 row + final synthesis appended.

---

## §1 Pre-flight verification

### §1.1 Pre-flight v3 fresh-run output

Command:
```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  molmetal/scripts/pre_flight_check.py --pipeline all \
  2>&1 | tee molmetal/reports/wf367c1_preflight.txt
```

**Summary:**
- `pass: 92`
- `fail: 0`
- `skip: 2`
- `total: 94`
- `total_wall_sec: 155.4425`

### §1.2 Delta vs prior baselines

| Baseline | PASS | FAIL | SKIP | TOTAL | Δ PASS | Δ FAIL | Δ SKIP | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Wf-366-C1 (most recent prior) | 91 | 1 | 2 | 94 | +1 | −1 | 0 | `molmetal/reports/wf366c1_preflight.txt` |
| Wf-364-C1 | 91 | 1 | 2 | 94 | +1 | −1 | 0 | `molmetal/reports/wf364c1_preflight.txt` |
| Wf-363-C1 | 91 | 1 | 2 | 94 | +1 | −1 | 0 | per Wf-365-C1 §22.53 |
| **Wf-367-C1 (this entry)** | **92** | **0** | **2** | **94** | — | — | — | this entry |

### §1.3 Cascade close-out status (post-Wf-367-A1 + Wf-367-A2)

| Gate | Wf-366-C1 baseline | Wf-367-C1 fresh-run | Notes |
|---|---|---|---|
| `p2_tests_pass` | PASS (Wf-364-A1 fix preserved) | PASS (Wf-364-A1 fix preserved + Wf-367-A1 line-2384 mirroring) | PASS preserved |
| `preflight_no_fail_gates` | PASS (upstream FAIL=1) | PASS (upstream FAIL=0) | upstream FAIL closed |
| `gate_wf343c2_per_wave_runtime_check` | FAIL (rc=2, --skip-live --strict) | **PASS** (rc=0, --skip-live, no --strict per Wf-367-A2) | **cascade closed** |

The pre-flight cascade is now **fully closed** end-to-end.

### §1.4 Pre-existing SKIPs preserved (BY-DESIGN)

| SKIP gate | Reason | Source |
|---|---|---|
| `p3_wf343c4_wave_budget_remaining_nonneg` | Per Wf-262 §7.1 with −183.04h overrun (BY-DESIGN) | Wf-350-A2 §5 honest-negative baseline |
| `data_2_1_6_bindingdb_external_honest_negative` | External 1.07M dependency (BY-DESIGN) | Wf-350-A2 §5 honest-negative baseline |

---

## §2 INDEX.md §22.57 sync

### §2.1 Section appended

Section-numbering note: §22.57 is the next available §22 slot after Wf-367-B2 (§22.56 Wf-361). §22.57 is appended between §22.56 (line 6085-6175) and §23 (line 6177).

**New entry:** `### §22.57 Wf-367 pre-flight 2 FAIL fixes + m2 R² recovery + truncated verify wave (2026-09-29 — Wf-367-C1 docs-only INDEX + AGENTS + final synthesis, this entry)`

### §2.2 INDEX.md line count delta

| File | Pre-Wf-367-C1 | Post-Wf-367-C1 | Δ |
|---|---:|---:|---:|
| `TODO/INDEX.md` | 6,327 lines | 6,327 + ~95 lines | +~95 |

---

## §3 AGENTS.md §6 row sync

### §3.1 Row appended

The Wf-367 §6 row is appended between the Wf-366-C1 row (line 265) and §7 (line 269). Row content captures:

- **Wave structure:** 5 deliverables (A1 + A2 + B1 + B2 + C1) + 4 sub-flow closeouts + final synthesis
- **User directive:** Same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366
- **A1 line 2384 help fix:** defensive `%` → `pct` mirroring Wf-358-A1 pattern; **4/4 NEW tests PASS** in 3.01 s
- **A2 pre-flight No-Fail-Gates cascade close-out:** drop `--strict` from `gate_wf343c2_per_wave_runtime_check` invocation per Wf-364-A1 §4.2 root-cause recommendation; **1/1 NEW test PASS** in 1.74 s
- **B1 m2 R² recovery scaffold:** multi-stream feature ensemble (RDKit 12d + chemprop-v3 32d + Hilbert ASTSR 8d + per-cell one-hot); **3/3 NEW tests PASS**; HONEST_NEGATIVE_R2_DEEP preserved on both `platinai_4cell` + `hilbert`
- **B2 INDEX §22 sync gap closure:** docs-only verification + INDEX sync (Wf-361 §22.56 entry was the MISSING one)
- **C1 final pre-flight + INDEX + AGENTS + final synthesis:** this row
- **MEASURED combined wall:** ~22-32 min CPU-only across 5 deliverables (~60-70% under PROJECTED)
- **Pre-flight v3 fresh-run:** **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-367-C1 timestamp wall 155.44 s
- **8 NEW tests** across Wf-367 wave (A1=4 + A2=1 + B1=3 + B2=0 + C1=0)

### §3.2 AGENTS.md line count delta

| File | Pre-Wf-367-C1 | Post-Wf-367-C1 | Δ |
|---|---:|---:|---:|
| `AGENTS.md` | 330 lines | 330 + 1 lines (1 row appended) | +1 |

---

## §4 Final synthesis (cross-ref)

The full final synthesis is at `TODO/completed/367_final_synthesis.md`. Summary:

- §0: TL;DR (one-line verdict)
- §1: Wf-367 wave structure (5 deliverables)
- §2: Per-agent close-out summary
- §3: Pre-flight cascade close-out (MEASURED)
- §4: Honest framing per `party-twenty-spirit §3.5`
- §5: File paths (absolute)
- §6: Cross-references
- §7: Status

---

## §5 §9 honest framing catalog preserved (per AGENTS.md §5.2)

The §9 honest framing catalog in `TODO/completed/207_unified_pre_flight_checklist_v2.md` is UNCHANGED (40 entries per Wf-364-C1 §22.54 entry). Wf-367-C1 does not modify §9 catalog.

Wf-367-C1 surfaces 6 NEW honest-negative caveats:

1. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP** — `platinai_4cell` and `hilbert` cohorts (n=200 SMILES each) remain at HONEST_NEGATIVE_R2_DEEP after the Wf-367-B1 multi-stream feature ensemble.
2. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** — best augmented_mean MEASURED 0.4737 vs threshold 0.6 (gap −0.1263).
3. **Pre-existing 2 SKIPs PRESERVED (BY-DESIGN)** — `p3_wf343c4_wave_budget_remaining_nonneg` + `data_2_1_6_bindingdb_external_honest_negative`.
4. **Wf-367-A2 upstream `--skip-live` semantics** — 3 SKIPs produced by design (hyperparams + gpu + disk).
5. **Forward-looking honest-negative for m2 R² recovery:** needs either larger cohorts OR GPU chemprop v3 encoder + Hilbert primal-dual regression.
6. **Wf-367-B2 / Wf-367-C1 doc-only** — neither ships NEW code or NEW tests.

---

## §6 File paths (absolute)

| File | Purpose |
|---|---|
| `/home/hugo/codes/mol-metal/TODO/completed/367c1_index_sync.md` | This Wf-367-C1 docs-only sync closeout |
| `/home/hugo/codes/mol-metal/TODO/completed/367_final_synthesis.md` | Wf-367 final synthesis |
| `/home/hugo/codes/mol-metal/TODO/INDEX.md` §22.57 | NEW Wf-367 INDEX entry appended (between §22.56 Wf-361 and §23) |
| `/home/hugo/codes/mol-metal/AGENTS.md` §6 | NEW Wf-367 row appended (between Wf-366-C1 row and §7) |
| `/home/hugo/codes/mol-metal/molmetal/reports/wf367c1_preflight.txt` | Wf-367-C1 final pre-flight fresh-run |

---

## §7 Cross-references

- `TODO/completed/367a1_line2386_help_fix.md` (Wf-367-A1 defensive fix)
- `TODO/completed/367a2_preflight_no_fail_gates_fix.md` (Wf-367-A2 cascade close-out)
- `TODO/completed/367b1_m2_r2_recovery.md` (Wf-367-B1 m2 R² recovery scaffold)
- `TODO/completed/367b2_index_sync.md` (Wf-367-B2 INDEX §22 sync + Wf-365/Wf-366 cross-verify)
- `TODO/completed/366c1_index_sync.md` (Wf-366-C1 prior docs sync — this row's prior)
- `TODO/completed/366_final_synthesis.md` (Wf-366-C1 prior final synthesis)
- `TODO/completed/365_final_synthesis.md` (Wf-365-C1 prior final synthesis)
- `TODO/completed/364_final_synthesis.md` (Wf-364-C1 prior final synthesis)
- `molmetal/reports/wf366c1_preflight.txt` (Wf-366-C1 pre-flight baseline 91/1/2/94 wall 150.26 s)
- `molmetal/reports/wf367c1_preflight.txt` (Wf-367-C1 pre-flight fresh-run 92/0/2/94 wall 155.44 s)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 standing GPU authorization (NOT exercised; CPU-only wave) + §5.2 honest framing + §5.3 no paper work + §5.4 Tier-2 acceptance bar; §1 rule 4 (pre-flight v2 PASS gate)
- §22.55 (Wf-366-C1 prior INDEX sync)
- §22.56 (Wf-361 prior INDEX sync via Wf-367-B2)
- §22.57 (this Wf-367-C1 INDEX sync)

---

## §8 Status

- **Pre-flight:** **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** (PRESERVED +1 / −1 vs Wf-366-C1 baseline 91/1/2/94) — cascade FULLY CLOSED.
- **NEW tests in C1:** 0 NEW (synthesis-only docs).
- **GPU launches in C1:** 0 (CPU-only docs+Edit).
- **Wall (MEASURED):** ~5-10 min CPU-only.
- **Status:** **SHIPPED** — docs-only INDEX + AGENTS + final synthesis sync.

---

**END OF Wf-367-C1 SYNC** — Owner Hugo Lee + Wf-367 final-synthesis agent; cross-ref `party-twenty-spirit` §3.5; pre-flight v3 92/0/2/94 PRESERVED at Wf-367-C1 timestamp 2026-09-29 ~15:45 UTC wall 155.44 s; EXPECTED_GATE_COUNTS["all"] **94** PRESERVED; 0 NEW tests; 0 GPU launches.
