# arXiv Category Decision — Mol-Metal

**Paper**: *Mol-Metal: a hybrid molecular lambda calculus + flow-matching
framework for precious-metal drug design*
**Submission window**: W49 = 2026-12-09 (Wed freeze)
**Decision owner**: Hugo Lee (corresponding author)
**Companion documents**: `paper/SUBMISSION_CHECKLIST.md` (mechanics),
`paper/main.tex` (title + abstract), `paper/refs.bib` (cited literature).
**Status**: pre-flight; this document is the contract that the
2026-12-07 category-selection task references.

---

## 1. Target categories (primary + cross-list)

Mol-Metal is a hybrid manuscript that touches three distinct arXiv
communities at once:

| Slot | arXiv ID | Full name | Role |
|---|---|---|---|
| **Primary** | `q-bio.QM` | Quantitative Biology — Quantitative Methods | Subject-matter home (precious-metal anticancer drug design) |
| **Cross-list 1** | `cs.LG` | Computer Science — Machine Learning | Method home (SE(3)-equivariant CFM, MCTS with virtual loss, 7-channel RewardAggregator) |
| **Cross-list 2** | `physics.chem-ph` | Physics — Chemical Physics | Methodology home (metal-aware geometric priors, coordination-geometry closure, click-chemistry reaction registry) |

No other categories are submitted at this time. The remaining candidate
slots (`q-bio.BM`, `stat.ML`, `cs.AI`) are held in reserve — see §5.

---

## 2. Decision tree

The primary category is chosen by the deepest research question, not the
shiniest method. Walk top-down:

1. **Is the subject matter a quantitative biology / chemistry problem
   with a measurable biological or chemical endpoint?**
   - *Yes* — the headline task is designing precious-metal (Pt(II),
     Ru(II), Ir(III)) anticancer candidates with measured docking
     affinity, ADMET properties and (prospectively) cytotoxicity.
     → leads to `q-bio.*`.
2. **Within `q-bio.*`, is the contribution a quantitative method
   (computational pipeline, generative model, formal calculus) rather
   than a new biomolecule / new binding assay / new clinical result?**
   - *Yes* — Mol-Metal contributes a typed Molecular Lambda Calculus
     and a CFM decoder; no new wet-lab assays, no new clinical data.
     → leads to `q-bio.QM` (Quantitative Methods) rather than
     `q-bio.BM` (Biomolecules, which expects new structures / sequences).
3. **Does the paper introduce a general ML method (architecture,
   training objective, search algorithm) reusable beyond the named
   application?**
   - *Yes* — the SE(3)-equivariant conditional flow matching
     component, the MCTS-with-virtual-loss driver and the
     RewardAggregator design are ML contributions that generalise
     beyond precious-metal chemistry. → cross-list `cs.LG`.
4. **Does the paper engage with chemical-physics methodology (PDEs,
   reaction energetics, coordination chemistry, quantum effects)?**
   - *Yes* — the MetalGeometryPrior encodes coordination-geometry
     constraints (square-planar Pt(II), octahedral Ru/Ir(III)) and the
     five click reactions are registered as typed β-reductions with
     metal-aware catalyst annotations. → cross-list `physics.chem-ph`.

If the answer to (3) were *No*, we would drop `cs.LG`. If the answer to
(4) were *No*, we would drop `physics.chem-ph`. We retain both.

---

## 3. Recommended submission

| Field | Value |
|---|---|
| **Primary category** | `q-bio.QM` |
| **Cross-list 1** | `cs.LG` |
| **Cross-list 2** | `physics.chem-ph` |
| **Number of cross-lists** | 2 (within the arXiv limit of 4) |
| **Submission type** | New (Round-17 close-out manuscript) |
| **License** | CC BY 4.0 (consistent with `paper/SUBMISSION_CHECKLIST.md` §6.7) |

Submission sequence in the arXiv UI:

1. Log in as the submitter account (Hugo Lee ORCID-linked).
2. Start new submission → upload `paper/main.pdf` + source bundle.
3. **Title**, **authors**, **abstract** auto-filled from the TeX
   metadata (verify byte-for-byte against `paper/main.tex:120` and
   `paper/main.tex:131–168`).
4. **Primary** → `q-bio.QM`. **Cross-list** → add `cs.LG` then
   `physics.chem-ph`. No further cross-lists.
5. Confirm MSC / ACM / PACS codes (none requested; leave blank).
6. Submit; record the arXiv ID in
   `TODO/co_author_signoffs/notification.txt` per checklist §6.4.

---

## 4. Rationale

**Why `q-bio.QM` is the right primary.** The majority of recent
precious-metal anticancer design papers — including the platinum(II)
complex design literature, the ruthenium / iridium coordination-chemistry
work, and the crossdocked-suite evaluations used by our comparators —
land in `q-bio.QM` when their contribution is computational / generative
rather than wet-lab. arXiv moderators route pure-chemistry manuscripts
to `physics.chem-ph` and pure-biology manuscripts to `q-bio.BM` /
`q-bio.NC`. Mol-Metal is a *generative pipeline applied to a biological
target class (metal-binding anticancer pockets)*, which `q-bio.QM`
explicitly covers as "computational methods applied to biological
problems". This is also the slot used by the MolCRAFT / Pocket2Mol /
TargetDiff family of structure-based drug design papers that we cite
in §2 (Related Work).

**Why cross-list `cs.LG`.** A non-trivial fraction of the paper is a
reusable ML contribution — the typed MLC as a search-restriction
language, the SE(3)-equivariant CFM with a metal-aware prior, the
synthesizability-aware MCTS driver, the 7-channel RewardAggregator
that fuses RDKit sanitisation, QuickVina 2 docking, PoseBusters,
AdmetAI and REINVENT4. The ML community (`cs.LG`) is the natural
secondary audience for these methods, even though the headline
application is biology.

**Why cross-list `physics.chem-ph`.** The metal-aware geometric prior
and the typed β-reduction rule set engage directly with chemical-
physics methodology: coordination-geometry constraints, click-reaction
energetics, catalyst stoichiometry. Cross-listing to `physics.chem-ph`
gives the manuscript visibility to the chem-ph community that works
on reaction enumeration and metal-coordination geometry.

**Why only two cross-lists.** arXiv allows up to four cross-list
categories, but adding more dilutes discoverability and risks moderator
pushback for "over-broad" submission. Two cross-lists is the natural
breadth of the hybrid ML × chem-ph × q-bio manuscript.

**Why NOT `q-bio.BM` as primary.** `q-bio.BM` (Biomolecules) expects
manuscripts whose central contribution is a *new biomolecule* — a new
protein structure, a new RNA fold, a new sequence design — rather than
a generative *pipeline*. Mol-Metal's contribution is the pipeline; the
metallodrug candidates are illustrative outputs. `q-bio.QM` is the
correct slot.

**Why NOT `cs.LG` as primary.** The headline research question is
biological (designing better metal anticancer candidates), not
algorithmic. A `cs.LG`-primary submission would be rejected by the q-bio
community that should be the primary readers, and the moderators would
likely demote it to a `cs.LG` cross-list automatically.

---

## 5. Backup categories

If arXiv moderators reject the primary category (rare, but possible
on a hybrid manuscript) the fallback chain is:

| Rank | Primary | Cross-lists | Rationale |
|---|---|---|---|
| **A (recommended)** | `q-bio.QM` | `cs.LG`, `physics.chem-ph` | §4 — best fit for hybrid biology + ML + chem-ph |
| **B (moderator demoted)** | `cs.LG` | `q-bio.QM`, `physics.chem-ph` | If moderators rule the contribution is methodological, demote to ML primary; retain both biological + chemical cross-lists |
| **C (chemistry-led reframing)** | `physics.chem-ph` | `q-bio.QM`, `cs.LG` | If the manuscript is reframed for a chemical-physics venue that emphasises the coordination-geometry prior, swap primary and cross-lists |
| **D (last resort)** | `stat.ML` | `cs.LG`, `q-bio.QM` | If `cs.LG` is over-saturated and the manuscript is reframed as a statistical-learning methods paper; retain biology cross-list, drop `physics.chem-ph` |
| **E (broadest)** | `cs.AI` | `q-bio.QM`, `cs.LG` | If the MCTS + proof-search component is emphasised as a general AI planning contribution; drops `physics.chem-ph` |

Reserve categories `q-bio.BM` and `cs.CE` (Computational Engineering)
were considered and **dropped**: `q-bio.BM` is too narrow for a
generative pipeline, and `cs.CE` is the wrong audience for ML × biology
hybrid work. They are not listed in the fallback chain.

---

## 6. Acronym list

| Code | Full name | Scope |
|---|---|---|
| `q-bio.QM` | Quantitative Biology — Quantitative Methods | Primary |
| `q-bio.BM` | Quantitative Biology — Biomolecules | Reserve (dropped) |
| `cs.LG` | Computer Science — Machine Learning | Cross-list 1 |
| `cs.AI` | Computer Science — Artificial Intelligence | Backup D / E |
| `stat.ML` | Statistics — Machine Learning | Backup D |
| `physics.chem-ph` | Physics — Chemical Physics | Cross-list 2 |

Other arXiv IDs that appear in this document but are **not** part of
the submission:

- `q-bio.NC` (Neurons and Cognition) — not relevant; no neuroscience claim.
- `cond-mat.soft` (Soft Condensed Matter) — not relevant; no material claim.
- `math.OC` (Optimization and Control) — not relevant; no control-theory claim.

---

**Document version**: 1.0 (2026-09-25, R17 close-out).
**Maintainer**: Hugo Lee.
**Next review**: at the 2026-12-07 category-selection checkpoint
(`paper/SUBMISSION_CHECKLIST.md` §6.1) and again at the 2026-12-08
pre-submission walk-through (§7).