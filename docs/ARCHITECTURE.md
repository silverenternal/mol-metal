# Architecture — 21 Mechanisms M1-M21

Decision-version of `TODO/DESIGN_BOUNDARY.md` mapping the 21-mechanism phase diagram. Source: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (Wf-313H + Wf-320G final, 107/107 tests PASS).

## Phase diagram

```
INPUT (SMILES / 3D pose / λ-ast / SMARTS)
  │
  ├── M1  chemprop v2 backbone         [STABLE] ────────┐
  ├── M2  D-MPNN multitask backbone   [STABLE] ──┐      │
  └── M3  chemprop v3 enhanced head    [STABLE-PROPOSED]
                                                   │
COORDINATION ──────────────────────────────────────┤
  ├── M4  Click-aug wire (CuAAC/SPAAC/Suzuki)     │
  │     [STABLE] per Wf-249 + Wf-313H            │
  └── M5  Metal-ligand exchange (Krasnov 2026)    │
        [STABLE] per Wf-329-W6                   │
                                                   │
BACKBONE ──────────────────────────────────────────┤
  ├── M6  CFM Flow Matching (Lipman 2023)         │
  │     [STABLE] per Wf-340 + Wf-353              │
  ├── M7  YuelBond decoder                          │
  │     [FROZEN] pre-flight gate per Wf-345-A2     │
  └── M8  DMPNN cache featurization               │
        [STABLE] per Wf-326 + Wf-328 + Wf-382-A2    │
                                                   │
SEARCH ─────────────────────────────────────────────┤
  ├── M9  VariancePUCT MCTS (Burch-Bardsley 2022) │
  │     [STABLE] per Wf-329-W6 + Wf-353-A1       │
  ├── M10 Score-Repellent MC (Hu et al. 2026 ICML)│
  │     [PROPOSED] scaffold                       │
  └── M11 Replay learner                           │
        [PROPOSED] per Wf-327-F10 + Wf-353-A2     │
                                                   │
OPTIMIZATION ──────────────────────────────────────┤
  ├── M12 Hilbert primal-dual (Mahmood 2024)      │
  │     [PROPOSED] HONEST_NEGATIVE FINAL          │
  │     (Wf-377-A1 + Wf-389-A1)                   │
  ├── M13 RDKit F5_RDKIT_2D 32-d features          │
  │     [STABLE] per Wf-249 + Wf-339              │
  ├── M14 Multi-PH + PLL Fourier 32-bin + 128-d   │
  │     [PROPOSED] per Wf-382-A1 (R20-P3)         │
  └── M15 Wasserstein gradient flow                 │
        [STABLE] per Wf-320 + Wf-339-B2            │
                                                   │
PROOF ──────────────────────────────────────────────┤
  ├── M16 Catherine 6-property verifier            │
  │     [FROZEN] 6/6 PASS ≥ 25 metals × 1k samples │
  ├── M17 Tanimoto<0.4 OOD protocol                │
  │     [FROZEN] per Wf-41-P2                     │
  ├── M18 Per-residue metal coordination            │
  │     [STABLE] per Wf-249 + Wf-339              │
  └── M19 Click-compatible metal-complex SMARTS    │
        [STABLE] per Wf-329-W6                     │
                                                   │
OUTPUT ──────────────────────────────────────────────┤
  ├── M20 Tier-2 computational paper                │
  │     [OUT-OF-SCOPE wetlab] per AGENTS.md §5.4   │
  └── M21 Honest framing catalog                    │
        [FROZEN] 81 entries per Wf-426-A4           │
                                                   │
METRICS OUTPUT ─────────────────────────────────────┘
  ↓
m1_auc + m2 R² + m3 Vina + m4 SA + m6_kappa + m7 Diversity + m14 decode_ratio + m15 RMSD + m18 + m21 + ...
  (every metric → MEASURED-vs-PROJECTED honest-neg gate per `party-twenty-spirit` §3.5)
```

## Mechanism composition rules

1. **STABLE mechanisms compose freely** — backwards-compat by default.
2. **PROPOSED mechanisms compose via SCAFFOLD stage only** — opt-in flag surface; trainer wiring deferred to MEASURED-end-to-end validation on real labelled pIC50 / OOD pIC50.
3. **FROZEN mechanisms compose only with §5.5 filesystem-truth verification** — git commit + honest-neg disclosure + paper §6 Limitations entry.
4. **OUT-OF-SCOPE mechanisms REFUSE** per `party-twenty-spirit` §5.

## Cross-refs

- Per-mechanism ↔ metric mapping: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (107/107 tests PASS)
- Per-Wf ship surface: `TODO/INDEX.md` §22.NNN entries (canonical historical record; wave-versioning policy deprecation per AGENTS.md §6 + 2026-10-01 user directive)
- Per-metric honesty ledger: `docs/HONEST_NEGATIVE_CATALOG.md` (81 entries per Wf-426-A4)
- Design boundary FROZEN/STABLE/PROPOSED/OUT-OF-SCOPE: `docs/DESIGN_BOUNDARY.md`
