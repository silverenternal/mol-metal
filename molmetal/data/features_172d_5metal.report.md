# Wf-128 Phase 1 — 172-d feature build report

- **n_rows**: 1000
- **feature_dim**: 172 (12 RDKit + 32 Multi-PH + 128 PLL)
- **n_all_fail**: 0
- **fail_per_stream**: {"multiph": 1, "pll": 1}
- **wall_s**: 153.02
- **RDKit_DIM**: 12
- **MULTI_PH_DIM**: 32
- **PLL_DIM**: 128
- **METALS_5**: ['Ru', 'Ir', 'Os', 'Rh', 'Re']

Honest framing: per-stream failures are **isolated** — a row
with one failed stream is column-mean imputed; a row with
all three failed is preserved as an all-zero vector.
