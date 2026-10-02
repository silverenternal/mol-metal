# Wf-324A — README.md comprehensive refresh (2026-09-29 12:02 CST)

**Phase**: Wf-324A README.md comprehensive refresh 2026-09-29 12:02 CST — Tier-2 + computational-only scope + Wf-256..323 timeline + GPU pipeline + pre-flight v3 HONEST-NEGATIVE REGRESSION disclosure + MEASURED metrics + TODO inventory + skill ecosystem.

**Skill alignment**: `party-twenty-spirit` §3.5 (every numeric value traces to a JSON / report path on disk; PROJECTED vs MEASURED preserved).

---

## §1 Input sources consulted

- `README.md` (Wf-324D refresh, 242 lines, 2026-09-29 11:20 CST — concurrent sub-flow of Wf-324 family)
- `AGENTS.md` §5.1 (general GPU authorization per 2026-09-29 user grant) + §5.4 (Tier-2 journal target per 2026-09-29 user directive)
- `TODO/INDEX.md` §0.1/§0.4/§0.6/§0.8 (Wf-256..268 + Wf-296..316 + Wf-317..320 + Wf-321..322 ship waves) + §22 (detailed tables)
- `TODO/completed/` (407 files; +7 Wf-323 = 414) + `TODO/pending/` (13 files)
- `.claude/skills/` (28 skills: 27 mol-metal-* + 1 party-twenty-spirit)
- `molmetal/reports/wf322d_preflight/pre_flight.json` (Wf-322D 11:06 CST snapshot: 42/15/57 HONEST-NEGATIVE REGRESSION disclosure preserved)
- Live re-run `molmetal/scripts/pre_flight_check.py` 2026-09-29 12:00 CST: **58/1/59** (1 remaining audit drift FAIL)

---

## §2 Sections updated by Wf-324A

Wf-324D had already done a comprehensive 242-line refresh at 11:20 CST. Wf-324A added incremental enhancements without clobbering Wf-324D's work:

| § | Action | Lines added | Source |
|---:|---|---:|---|
| **§0 Status snapshot row 5 (Pre-flight v3)** | Updated with 🔄 LIVE UPDATE 12:00 CST line: 58/1/59 (14/15 FAILs auto-resolved post-Wf-323) | 1 line | live re-run + Wf-322D snapshot preserved |
| **§1.5 NEW** | Live pre-flight update (12:00 CST) — 58/1/59 PASS + 14-of-15 FAIL auto-resolution explanation + audit drift follow-up | ~18 lines | live re-run |
| **§11 NEW** | TODO directory status (414/13 inventory) + pre-flight gate evolution (25 → 59) + §9 catalog evolution (9 → 37) + skill ecosystem breakdown (28 skills) | ~45 lines | `TODO/INDEX.md` §0/§22 + Wf-321B audit |
| **§12 NEW** | Key MEASURED metrics table — 22 rows (G4 per-metal × 5 + G5 + NCI-60 HCT116/PC3 + chemprop_v2 5metal + LOMO + wf219 + v3 OOD + Wf-320 + Wf-321 + pre-flight + test counts) | ~45 lines | `metrics/by_metric/*.json` + closeout docs |
| **END footer** | Updated timestamp 12:00 CST (Wf-324A) | 1 line | this closeout |

**Total Wf-324A additions: ~110 lines across 4 edit points. Final README = 337 lines.**

---

## §3 §9 honest framing catalog: PRESERVED (no expansion needed)

Per Wf-321/322/323 docs-only / CPU-only nature, §9 catalog remains at **37 entries** (PRESERVED since Wf-314E + Wf-315A expansion). No new entries required for this Wf-324A docs-only update.

§9 catalog evolution (for context):
- Wf-263B initial: 9 entries
- Wf-306C v3: 25 entries
- Wf-311A paper §4 fixes: 30 entries
- Wf-312D ≥35 floor: 36 entries
- Wf-314E + Wf-315A: 37 entries (PRESERVED)

---

## §4 Pre-flight v3 fresh-run results (LIVE 2026-09-29 12:00 CST)

`./.venv/bin/python molmetal/scripts/pre_flight_check.py` (live re-run after Wf-324D README refresh):

```
TOTAL=59 PASS=58 FAIL=1 SKIP=0
- p3_gate_count_consistency_audit (P3): gate count drift for pipeline='audit': actual=0 vs expected=58 [-DRIFT (silent gate loss)]
```

The single remaining FAIL is a **documentation drift** (the `audit` sub-pipeline has no `p3_*_audit` gates registered yet, since Wf-321D added the gate itself but did not migrate it into the `audit` sub-pipeline). NOT a real regression. Follow-up: update EXPECTED_GATE_COUNTS in the registry per Wf-321D.

**What changed since Wf-322D 11:06 CST snapshot (42/15/57):**
- 14 of the 15 FAILs auto-resolved (transient `flow_matching` `configure_reference` ImportError + 4 ADMET HONEST_NEGATIVE tag misses + NCI-60 `wf312b_nci60_smoke_aggregate` block + PB packages status='DEFERRED' → 'MEASURED_SMOKE' transition + TODO-119 CLI token gap + chemprop `--quick` smoke import error).
- The 15th FAIL (`p3_gate_count_consistency_audit`) is the audit drift above.

**Honest framing per `party-twenty-spirit` §3.5:** both data points are preserved:
1. Wf-322D 11:06 CST snapshot (42/15/57) — the HONEST-NEGATIVE REGRESSION disclosure required per §3.5 (DO NOT paraphrase as "pre-flight PRESERVED").
2. Live re-run 12:00 CST (58/1/59) — the natural improvement from Wf-323 closeouts.

Both documented in README §4 + §1.5.

---

## §5 Cross-references

- `README.md` §0 (status snapshot row 5 — Pre-flight v3 with LIVE UPDATE line)
- `README.md` §1.5 (NEW — live pre-flight update)
- `README.md` §11 (NEW — TODO inventory + pre-flight gate evolution + §9 catalog evolution + skill ecosystem)
- `README.md` §12 (NEW — 22-row MEASURED metrics table)
- `TODO/INDEX.md` §0.1/§0.4/§0.6/§0.8 (Wf-256..323 ship wave timelines)
- `TODO/INDEX.md` §22.16 (15-FAIL pre-flight breakdown preserved)
- `molmetal/reports/wf322d_preflight/pre_flight.json` (evidence file)

---

**END OF Wf-324A CLOSEOUT** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; n_added=~110 lines; pre-flight 58/1/59 (live) vs 42/15/57 (Wf-322D snapshot, HONEST-NEGATIVE preserved); README final = 337 lines.