# WF-Pivot-A Phase 1 — 5 ADD bibitems (de novo §2 rewrite)

**Date**: 2026-09-16
**Phase**: Pivot-A Phase 1 — bibliography expansion
**Agent**: WF-Pivot-A-Bib-Add (task #884)
**Status**: COMPLETED

## Summary

Added 5 bibliographic entries to `/home/hugo/codes/try_triton_on_rocm/paper/refs.bib` to support the Pivot-A §2 Related Work rewrite (de novo / generative-model framing). All entries were web-verified against canonical bibliographic records before insertion. Two venue/arXiv-ID errors in the user-supplied templates were caught and corrected; the audit trail is preserved in-line in `refs.bib` (see the comment block immediately above the new entries).

## Entries added (alphabetical by key)

| Cite key | Authors | Year | Venue | arXiv / DOI | Insertion point |
|---|---|---|---|---|---|
| `olivecrona2017molecular` | Olivecrona, Blaschke, Engkvist, Chen (REINVENT) | 2017 | J. Cheminform. 9:48 | doi:10.1186/s13321-017-0235-x | line 821 (after `openmm`) |
| `graphaf_shi2020` | Shi, Xu, Zhu, Zhang, Zhang, Tang (GraphAF) | 2020 | **ICLR** (not ICML as drafted) | arXiv:2001.09382 | line 965 (after `gat2022pacbayes`) |
| `jtvae_jin2018` | Jin, Barzilay, Jaakkola (JTVAE) | 2018 | ICML | arXiv:1802.04364 | line 985 (after `karczewski2024egnn`) |
| `equivariant_fm_klein2023` | Klein, Krämer, Noé | 2023 | NeurIPS | arXiv:2306.15030 | line 998 (after `jtvae_jin2018`) |
| `moldqn_zhou2019` | Zhou, Kearnes, Li, Zare, Riley | 2019 | **Sci. Rep. 9:10752** (not ICML as drafted) | doi:10.1038/s41598-019-47148-x; arXiv:1810.08678 | line 1162 (after `yu2020pcgrad`) |

## Honest corrections (caught during verification)

- **GraphAF** (Shi 2020): the user template marked it as ICML 2020. Web verification against the arXiv-issued DOI (10.48550/arXiv.2001.09382) and the published proceedings (ICLR 2020, code at github.com/DeepGraphLearning/GraphAF) shows it was an **ICLR 2020** paper. Author list corrected to full 6-author form (Shi, Xu, Zhu, Zhang, Zhang, Tang).
- **MolDQN** (Zhou 2019): the user template marked it as ICML 2019 with arXiv:1904.00312. Web verification (PubMed PMID 31341196, doi 10.1038/s41598-019-47148-x, PMC6656766) shows it was published as **Scientific Reports 9:10752** with **arXiv:1810.08678**. The 1904.00312 preprint number does not match this paper.
- **Equivariant FM** (Hoffman 2022 template): no such paper was locatable on web search. The closest verifiable "Equivariant Flow Matching" reference is **Klein, Krämer, Noé (NeurIPS 2023, arXiv:2306.15030)**, which is the canonical Boltzmann-generator equivariant-flow paper. The key was renamed to `equivariant_fm_klein2023` to reflect the correct authors.

## File-level verification

- `paper/refs.bib`: 1098 → 1177 lines (+79)
- Total `@` entries: 107 → 112 (+5)
- Brace balance: `{` count = `}` count = 953 (perfectly balanced)
- All five new cite keys confirmed present by `grep` on lines 821, 965, 985, 998, 1162
- 0 existing entries removed (Agent 1D's 6-REMOVE step is a separate Phase-2 operation)

## Files touched

- `/home/hugo/codes/try_triton_on_rocm/paper/refs.bib` (+5 entries, 1 audit comment block)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pivot_a/final_phase1_bib.md` (this file)

## Next phase (Phase 2, owned by other agents)

- Agent 1D: REMOVE 6 SBDD-only entries from `refs.bib` (separate workflow).
- Agent 1E: rewrite §1 + §2 with these 5 cite keys anchored.
- Agent 1F: recompile + CROSS_REFS + master report.