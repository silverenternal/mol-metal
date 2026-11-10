# Wf-A168 refutation review — branch "HLTF primary-source axis" (wfa168_hltf_axis)

**VERDICT: NOT REFUTED** — every headline claim of the branch was independently re-derived and holds.

| check | result |
|---|---|
| C1 re-derive the numbers from the primary source | PASS |
| C2 comparison not overstated | PASS |
| C3 regression test bites | PASS_WITH_GAP (no pytest file for this branch; the pin machinery is the guard, 4/4 negative controls bite) |
| C4 no number from a search summary | PASS |
| C5 published tree unmodified | PASS |
| C6 no GPU | PASS |

Independent retrieval: `https://arxiv.org/e-print/2603.27113` -> HTTP 200, final URL
`https://arxiv.org/src/2603.27113`, 504186 B, sha256 `87c1788e9e9826cfbdf4e0788feed1a796c8a50218eecc6905df771f10c9c329`
(identical to the branch's record); `https://arxiv.org/abs/2603.27113` -> HTTP 200, 41764 B, sha256
`374ece252cad92826dcc1f36a321ca971a650a7897acb6eec57581502d714804`. The branch's own pin machinery, run against
the reviewer's own download: 15/15 numbers pinned, 13/13 context pins located, 0 site mismatches. Our side re-derived
from `metrics/by_metric/cfm_fragmentation_postproc_arm.json` and from the raw pool CSV: 1078/12668 = 0.08509630565203663,
delta 91.4903694348 pts, ratio 13.655279, specks 21.0215 %, mass discarded 0.22474803194442472, 12669/12669 rows carry a
cohort metal — all exact matches.

Surviving concerns: S1 no CI guard against a future arXiv v2 (medium); S2 the quotable "13.66x" headline;
S3 our delta is the FIRST PP step only (UFF inapplicable to metals); S4 N is not at the per-row LaTeX site;
S5 the paper dual-cites the PP (Rappe 1992 for UFF, Buttenschoen arXiv:2505.00518 for "PP"); S6 the corrected house
format is a proposal with no enforcement; S7 `--out-dir` outside the repo crashes (informational).

Full evidence: `molmetal/reports/wfa168_hltf_axis_refutation/refutation.json`.
