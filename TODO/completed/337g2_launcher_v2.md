# Wf-337-G2 — G6 / G5 / G4 launcher v2 with consolidated Wf-328 + Wf-335 fixes

**Date:** 2026-09-29
**Wf:** Wf-337-G2 (post-Wf-335 + Wf-328 ultracode consolidation; per
Wf-328-SYN §10 "Wf-328-W7 candidate" follow-up)
**Status:** SHIPPED — 4/4 NEW tests PASS, 27/27 regression tests PASS,
3 launchers wired with all 4 Wf-328 + Wf-335 fixes + the Wf-296C
per-config ablation levers.

---

## §1 TL;DR

Per Wf-328-SYN §10 (2026-09-29): the 6 Wf-328-W2 launchers had been
updated with `--cpu-num-workers 8` (Wf-328-W2) + `--no-cuda-graphs`
(Wf-335-R3) but **3 of those fixes were not yet propagated** to the
G6 / G5 / G4 launchers per Wf-328-W5 (--w2-gradient-mode sort),
Wf-328-W6 (--num-workers 8), and Wf-335-R1 (torch.compile mode default
verified).  Wf-337-G2 consolidates all 4 fixes into the 3 launchers
and writes 4 NEW regression tests + this doc.

| Item | Status |
|---|---|
| G6 launcher (`launch_wf124_p3_g6.sh`) updated | MEASURED — see §2.1 |
| G5 launcher (`launch_wf124_p3_g5.sh`) updated | MEASURED — see §2.2 |
| G4 launcher (`launch_wf124_p3_g4.sh`) updated | MEASURED — see §2.3 |
| `--w2-gradient-mode sort` wired (Wf-328-W5) | MEASURED — replaces legacy `sliced` override |
| `--num-workers 8` wired (Wf-328-W6) | MEASURED — factory override; prefetch=8 |
| `--cpu-num-workers 8` parity preserved (Wf-328-W2) | MEASURED — unchanged |
| `--no-cuda-graphs` safety lever preserved (Wf-335-R3) | MEASURED — unchanged |
| `--no-kendall` / `--no-dann` per-config preserved (Wf-296C) | MEASURED — G6 CONFIRMED_FLAGS only |
| `molmetal/tests/test_wf337g2_launcher_v2.py` NEW (4 tests) | MEASURED — 4/4 PASS |
| Regression: Wf-335-R3 + Wf-328-W2/W5/W6 tests | MEASURED — 27/27 PASS |

---

## §2 Per-launcher edits

### §2.1 G6 launcher (`scripts/launch_wf124_p3_g6.sh`)

| Line | Change | Source |
|---:|---|---|
| 9-19 | Header comment block: added 4 paragraphs documenting Wf-328-W5 (`sort`), Wf-328-W6 (`--num-workers`), Wf-335-R1+R3 (`--no-cuda-graphs`), Wf-296C (`--no-kendall`/`--no-dann`), Wf-337-G2 v2 reference | this Wf |
| 47 | `--w2-gradient-mode sliced` → `--w2-gradient-mode sort` | Wf-328-W5 |
| 53 | NEW `--num-workers 8` (after `--cpu-num-workers 8`) | Wf-328-W6 |
| 60-66 | `--no-cuda-graphs` safety lever comment block (preserved verbatim from Wf-335-R3) | Wf-335-R3 |
| 78-82 | `CONFIRMED_FLAGS` (per-config ablation) preserves `--no-kendall` / `--no-dann` per Wf-296C | Wf-296C |

### §2.2 G5 launcher (`scripts/launch_wf124_p3_g5.sh`)

| Line | Change | Source |
|---:|---|---|
| 8-22 | Header comment block: added 4 paragraphs (parity with G6) | this Wf |
| 47 | `--w2-gradient-mode sliced` → `--w2-gradient-mode sort` | Wf-328-W5 |
| 51 | NEW `--num-workers 8` (after `--cpu-num-workers 8`) | Wf-328-W6 |
| 53-58 | `--no-cuda-graphs` safety lever comment block (preserved verbatim) | Wf-335-R3 |

G5 does NOT need `--no-kendall` / `--no-dann` (it runs the `mtl_baseline`
only; per-config levers are G6 ablation only per Wf-296C).

### §2.3 G4 launcher (`scripts/launch_wf124_p3_g4.sh`)

| Line | Change | Source |
|---:|---|---|
| 14-26 | Header comment block: added 4 paragraphs (parity with G6) | this Wf |
| 37 | `--w2-gradient-mode sliced` → `--w2-gradient-mode sort` | Wf-328-W5 |
| 43 | NEW `--num-workers 8` (after `--cpu-num-workers 8`) | Wf-328-W6 |
| 45-50 | `--no-cuda-graphs` safety lever comment block (preserved verbatim) | Wf-335-R3 |

G4 does NOT need `--no-kendall` / `--no-dann` (same as G5).

---

## §3 Test coverage (NEW)

### §3.1 Test file: `molmetal/tests/test_wf337g2_launcher_v2.py`

**4 distinct test functions:**

| # | Test | What it gates |
|---|---|---|
| 1 | `test_g6_launcher_has_all_fixes` | G6 launcher (the v2 reference) explicitly wires all 4 Wf-328 + Wf-335 fixes + the 2 Wf-296C per-config ablation levers. Also asserts `--w2-gradient-mode sliced` is REMOVED (anti-regression on the Wf-328-W5 lift). |
| 2 | `test_g5_launcher_parity` | G5 launcher mirrors the G6 v2 COMMON_ARGS config (4 fix flags; no --no-kendall/--no-dann since G5 is mtl_baseline-only). |
| 3 | `test_g4_launcher_parity` | G4 launcher mirrors the G6 v2 COMMON_ARGS config (4 fix flags). |
| 4 | `test_backward_compat` | Trainer argparse still exposes all the legacy CLI flags (`--w2-gradient-mode sliced`/`sort` choices, `--cpu-num-workers`, `--num-workers`, `--no-cuda-graphs`, `--no-kendall`, `--no-dann`). Plus parse_args() round-trip on the full v2 config. |

All tests are CPU-only — no torch.cuda, no real launches. < 5 s wall.

### §3.2 Test result

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf337g2_launcher_v2.py -v

molmetal/tests/test_wf337g2_launcher_v2.py::test_g6_launcher_has_all_fixes PASSED
molmetal/tests/test_wf337g2_launcher_v2.py::test_g5_launcher_parity PASSED
molmetal/tests/test_wf337g2_launcher_v2.py::test_g4_launcher_parity PASSED
molmetal/tests/test_wf337g2_launcher_v2.py::test_backward_compat PASSED

========================= 4 passed, 1 warning in 1.28s =========================
```

**4/4 NEW tests PASS in 1.28s.**

### §3.3 Regression check — Wf-335-R3 + Wf-328-W2/W5/W6 tests still pass

```bash
$ PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf335r3_g6_cuda_graphs.py \
           molmetal/tests/test_wf328w2_launcher_num_workers.py \
           molmetal/tests/test_wf328w5_w2_sort_production.py \
           molmetal/tests/test_wf328w6_dataloader_factory_production.py

molmetal/tests/test_wf335r3_g6_cuda_graphs.py .........                  [ 33%]
molmetal/tests/test_wf328w2_launcher_num_workers.py ..........           [ 70%]
molmetal/tests/test_wf328w5_w2_sort_production.py ....                   [ 85%]
molmetal/tests/test_wf328w6_dataloader_factory_production.py ....        [100%]

======================== 27 passed, 1 warning in 8.56s ========================
```

**27/27 regression tests PASS — no regression.** Wf-337-G2 only ADDS
new launcher parity + backward-compat coverage; it does not modify any
trainer behaviour that the Wf-328-W1/W2/W5/W6 + Wf-335-R3 tests gate.

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Claim | Status |
|---|---|
| G6 launcher wired with all 4 fixes + 2 per-config levers | **MEASURED** (`test_g6_launcher_has_all_fixes`) |
| G5 launcher wired with all 4 fixes (no per-config levers needed) | **MEASURED** (`test_g5_launcher_parity`) |
| G4 launcher wired with all 4 fixes | **MEASURED** (`test_g4_launcher_parity`) |
| Trainer argparse still exposes all 6 legacy CLI flags | **MEASURED** (`test_backward_compat`) |
| `parse_args()` round-trip on full v2 config | **MEASURED** (`test_backward_compat` Case (f)) |
| `--w2-gradient-mode sliced` is REMOVED from G4/G5/G6 COMMON_ARGS | **MEASURED** (anti-regression asserts in tests 1/2/3) |
| End-to-end GPU verification of the consolidated config | **PROJECTED** (BLOCKED on AGENTS.md §5.1 kill_then_adapt; needs > 1 hr GPU run to confirm the Wf-326A MEASURED 2-3× W₂ speedup materializes end-to-end with all 4 fixes stacked) |

This is a **structural parity + backward-compat** close: the fix is correct
and necessary (preventing the Wf-328-W5 + Wf-328-W6 lifts from being
silently skipped on future G6 / G5 / G4 launches), and end-to-end GPU
verification is BLOCKED on AGENTS.md §5.1 standing GPU authorization
(per standing grant, no per-launch OK needed; just need a clean GPU
window on RTX 5090 / RTX PRO 6000 to run a 50-epoch G6 tuple).

---

## §5 Cross-references

- Wf-328-SYN §10 (the Wf-328-W7 candidate follow-up that motivated this Wf):
  [`TODO/completed/328_synthesis.md`](328_synthesis.md)
- Wf-335-R1 trainer default flip:
  [`TODO/completed/335r1_no_cudagraphs_fix.md`](335r1_no_cudagraphs_fix.md)
- Wf-335-R3 G6 launch path safety lever:
  [`TODO/completed/335r3_g6_no_cudagraphs.md`](335r3_g6_no_cudagraphs.md)
- Wf-328-W2 launch-script `--cpu-num-workers 0→8`:
  [`TODO/completed/328w2_launcher_num_workers_fix.md`](328w2_launcher_num_workers_fix.md)
- Wf-328-W5 W₂ sort default flip (with launcher override convention):
  [`TODO/completed/328w5_w2_sort_wired.md`](328w5_w2_sort_wired.md)
- Wf-328-W6 DataLoader factory wire-up (with --num-workers CLI flag):
  [`TODO/completed/328w6_dataloader_factory_wired.md`](328w6_dataloader_factory_wired.md)
- Wf-296C (TODO-182 per-config `--no-kendall` / `--no-dann` lever):
  [`TODO/completed/296c_wf124_p3_g6_wireup.md`](../296c_wf124_p3_g6_wireup.md) (cross-ref)
- Wf-330-B4 MEASURED CUDAGraphs regression (the original G6 RuntimeError):
  [`TODO/completed/330b4_regression_smoke.md`](330b4_regression_smoke.md)
- AGENTS.md §5.1: standing GPU grant + kill_then_adapt directive
- AGENTS.md §5.2: MEASURED-vs-PROJECTED honest framing (party-twenty-spirit §3.5)
- AGENTS.md §5.4: Tier-2 paper target + NO wet-lab planning

---

**END OF Wf-337-G2** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; honest-negative: end-to-end GPU verification BLOCKED on AGENTS.md
§5.1 kill_then_adapt; structure-level fix verified at 3 launcher layers
+ 1 trainer-source backward-compat layer (4/4 layers MEASURED; full GPU
re-run is PROJECTED).
