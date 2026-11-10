# Wf-A156 -- EF-TALFM covalent-radius bond threshold applied to our CFM decode pool

*Generated 2026-11-01T03:40:49.697652+00:00 | CPU-only (no torch, no GPU) | tool `wfa156_bond_threshold_frontier`*

## 0. COMPARABILITY CAVEAT (read before any number below)

```
COMPARABILITY CAVEAT -- applies to EVERY number in this report
================================================================
The frontier band this branch was told about -- EF-TALFM (arXiv 2609.08333)
"Connected 92.98 %, Sanitized 98.25 %, Unique+sanitized 98.21 %,
Novel+unique+sanitized 94.33 %" -- is NOT our band.  NOTHING in this report may
be read as a comparison against it, and no number here is a reproduction of it.

FOUR INDEPENDENT BREAKS (each against the LIVE fetch of arXiv 2609.08333v1 and
its LaTeX source, fetched 2026-11-01):

(1) DENOMINATOR.  Their rates are computed over 10,000 generated molecules per
    method (App. G.3: "We sample 10,000 molecules"; Table 5 caption: "All
    rates are computed over 10,000 generated molecules").  Ours are over the
    12,669-row raw decode pool
    molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv -- ONE
    decode of ONE checkpoint.
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment", computed AFTER retaining the atom-count
    largest connected component and after RDKit sanitization.  Ours is an RDKit
    fragment count over RAW rows.
(3) POST-PROCESSING.  Their pipeline keeps the largest component, applies
    overlong-multiple-bond correction, adds explicit hydrogens and runs a
    restrained force-field relaxation before the numbers are counted.  Our pool
    rows are RAW: no relaxation, no hydrogen layer, no largest-component
    truncation in the counted population.
(4) HYDROGENS + CHEMISTRY, AND -- WORST -- A DIFFERENT BRANCH.  Their headline
    PCQM4Mv2 run is the ALL-ATOM branch ("we train EF-TALFM on PCQM4Mv2 using an
    all-atom molecular representation"), whose connectivity is recovered by
    OPEN BABEL perception, not by the tau_ij rule.  The tau_ij rule lives in the
    HEAVY-ATOM branch (Appendix D, Algorithm 4).  So THE FRONTIER'S HEADLINE
    NUMBERS WERE NOT PRODUCED BY THE MECHANISM THIS REPORT MEASURES.  On top of
    that: 12668 of our 12669 pool rows carry a metal
    (metal_ratio 0.999921) and every paper cohort is all-organic.

CONSEQUENCE: adopting tau_ij changes WHICH pairs the decoder MAY bond.  It does
NOT make our numbers comparable to theirs, and it does NOT make any fragment
rate in this repository comparable to the 92.7-94 % frontier band.

A FIFTH, SEPARATE LIMIT OF THIS REPORT, WHICH IS OURS AND NOT THEIRS:
the decoder's own interatomic distances are NOT RECOVERABLE from the pool
artefacts (the CSV header is epoch,smiles,n_frags_raw,largest_frag_frac -- no
coordinates; the run directory has no geometry dump).  So the literal
"would the emitted bond have survived tau_ij in the decoder's OWN geometry" is
NOT MEASURED here.  The report gives (a) exact geometry-free bounds, (b) an
exact rule-vs-rule comparison on REAL tmQM DFT geometry, and (c) a
length-TRANSFERRED estimate for the pool, each labelled as such.

AND THE THRESHOLD ITSELF IS CONTESTED, BY THE LIVE FETCH:
the paper's rule is tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j)), where
lambda_POLICY is the tolerance factor of the reconstruction policy pi chosen by
ChoosePolicy (src/appendix.tex:1039 and :1043).  The subscript pi is NOT the
constant pi.  The branch brief's "lambda * pi * (...)" -- and the repo's own
Wf-A155 radii_scaled_factor() -- are the misreading, and they differ from the
paper by a factor of 3.14159.  Both conventions are reported below under the
labels "paper" and "repo_pi".  The paper's numeric lambda_POLICY is NOT
recoverable: the whole LaTeX source has no numeric value for it.
```

### 0b. SUPERSEDED RENDERING + Wf-A180 CORRECTION (2026-11-01)

The rendering this tool emitted BEFORE the Wf-A180 correction is
preserved VERBATIM below (AGENTS.md §5.5 rule #6 -- append-only, dated,
never a silent edit):

```
COMPARABILITY CAVEAT -- applies to EVERY number in this report
================================================================
The frontier band this branch was told about -- EF-TALFM (arXiv 2609.08333)
"Connected 92.98 %, Sanitized 98.25 %, Unique+sanitized 98.21 %,
Novel+unique+sanitized 94.33 %" -- is NOT our band.  NOTHING in this report may
be read as a comparison against it, and no number here is a reproduction of it.

FOUR INDEPENDENT BREAKS (each against the LIVE fetch of arXiv 2609.08333v1 and
its LaTeX source, fetched 2026-11-01):

(1) DENOMINATOR.  Their rates are computed over 10,000 generated molecules per
    method (App. G.3: "We sample 10,000 molecules"; Table 5 caption: "All
    rates are computed over 10,000 generated molecules").  Ours are over the
    12,669-row raw decode pool
    molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv -- ONE
    decode of ONE checkpoint.
(2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
    component, i.e., one fragment", computed AFTER retaining the atom-count
    largest connected component and after RDKit sanitization.  Ours is an RDKit
    fragment count over RAW rows.
(3) POST-PROCESSING.  Their pipeline keeps the largest component, applies
    overlong-multiple-bond correction, adds explicit hydrogens and runs a
    restrained force-field relaxation before the numbers are counted.  Our pool
    rows are RAW: no relaxation, no hydrogen layer, no largest-component
    truncation in the counted population.
(4) HYDROGENS + CHEMISTRY, AND -- WORST -- A DIFFERENT BRANCH.  Their headline
    PCQM4Mv2 run is the ALL-ATOM branch ("we train EF-TALFM on PCQM4Mv2 using an
    all-atom molecular representation"), whose connectivity is recovered by
    OPEN BABEL perception, not by the tau_ij rule.  The tau_ij rule lives in the
    HEAVY-ATOM branch (Appendix D, Algorithm 4).  So THE FRONTIER'S HEADLINE
    NUMBERS WERE NOT PRODUCED BY THE MECHANISM THIS REPORT MEASURES.  On top of
    that: 12668 of our 12669 pool rows carry a metal
    (metal_ratio 0.999921) and every paper cohort is all-organic.

CONSEQUENCE: adopting tau_ij changes WHICH pairs the decoder MAY bond.  It does
NOT make our numbers comparable to theirs, and it does NOT make any fragment
rate in this repository comparable to the 92.7-94 % frontier band.

A FIFTH, SEPARATE LIMIT OF THIS REPORT, WHICH IS OURS AND NOT THEIRS:
the decoder's own interatomic distances are NOT RECOVERABLE from the pool
artefacts (the CSV header is epoch,smiles,n_frags_raw,largest_frag_frac -- no
coordinates; the run directory has no geometry dump).  So the literal
"would the emitted bond have survived tau_ij in the decoder's OWN geometry" is
NOT MEASURED here.  The report gives (a) exact geometry-free bounds, (b) an
exact rule-vs-rule comparison on REAL tmQM DFT geometry, and (c) a
length-TRANSFERRED estimate for the pool, each labelled as such.

AND THE THRESHOLD ITSELF IS CONTESTED, BY THE LIVE FETCH:
the paper's rule is tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j)), where
lambda_POLICY is the tolerance factor of the reconstruction policy pi chosen by
ChoosePolicy (src/appendix.tex:1039 and :1043).  The subscript pi is NOT the
constant pi.  The branch brief's "lambda * pi * (...)" -- and the repo's own
Wf-A155 radii_scaled_factor() -- are the misreading, and they differ from the
paper by a factor of 3.14159.  Both conventions are reported below under the
labels "paper" and "repo_pi".  The paper's numeric lambda_POLICY is NOT
recoverable: the whole LaTeX source has no numeric value for it.
```

**SUPERSEDED (2026-11-01, Wf-A180).**  The following sentence was
emitted as a settled fact and is REFUTED ON DISK:

> (2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
>     component, i.e., one fragment", computed AFTER retaining the atom-count
>     largest connected component and after RDKit sanitization.  Ours is an RDKit
>     fragment count over RAW rows.

**CORRECTION (2026-11-01, Wf-A180).**

> (2) METRIC.  Their "Connectivity" is "% of molecules with exactly one connected
>     component, i.e., one fragment".  THAT CELL IS RAW: it is a single-component
>     rate counted BEFORE the atom-count largest-component retention.  The paper's
>     CAPTION sentence is AMBIGUOUS as to the order -- it grammatically attaches
>     the largest-component clause to connectivity too -- but the NUMBER is RAW:
>     a post-retention connectivity count would read 100 % by construction, and
>     the paper reports 92.98 %.  Ours is an RDKit fragment count over RAW rows.

> the published band's provenance is NOT settled on disk: the frontier
> post-processing matrix record (`frontier_matrix.json`) holds TWO competing
> readings of which frontier cells the band names -- the first under
> `band_provenance` (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94 %, both
> `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 %
> as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity
> `C. = 92.98 %`, counted BEFORE the largest-component retention, with the
> post-processed `N. & U. & S. = 94.33 %` companion cell).  They do not even
> agree on the numerals (92.7/94 vs 92.98/94.33), so the band's source cells are
> NOT settled on disk.  Under BOTH readings the band is EXTERNAL, is NOT one
> homogeneous quantity, and is NOT on our scale -- OUR number IS a
> connected-component count on the RAW emitted SMILES, and nothing else.  (In
> THIS pool: 1078 of 12668 parsed rows have exactly
> one connected component.)

**ON-DISK ADJUDICATIONS (read these before trusting either rendering).**

- molmetal/reports/wfa168_eftalfm_axis/report.md -- "connectivity is RAW; the caption's sentence is ambiguous, not the number."
- molmetal/reports/wfa171_frontier_postproc/report.md -- "92.98 % is a RAW single-component rate", counted BEFORE the largest-component retention.

**LOCATORS (JSON PATHS, not line numbers).**  molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance` and `sources`.`EF-TALFM` (JSON PATHS, not line numbers)

### 0c. Wf-A196 CORRECTION (2026-11-02) -- the NUMBER of competing
readings is itself NOT settled on disk

**DEFECT.**  The block titled `CORRECTION (2026-11-01, Wf-A180)` above
carries a SETTLED COUNT of the competing readings.  That count is NOT
settled on disk: the repo's primary-source verification records a
FURTHER reading that likewise does not reproduce the quoted numerals,
so two on-disk records disagree about the SIZE of the contest.  The
count claim in the 2026-11-01 block is therefore SUPERSEDED by this
dated section; that block is preserved VERBATIM above (AGENTS.md §5.5
rule #6 -- append-only, dated, never a silent edit).

**CORRECTION (2026-11-02, Wf-A196).**

> the published band's provenance is NOT settled on disk: the frontier
> post-processing matrix record (`frontier_matrix.json`) holds COMPETING readings
> of which frontier cells the band names -- the first under `band_provenance`
> (GeoLDM Validity & Uniq 92.7 % and HLTF PB-Valid 94 %, both
> `post_processed: false`, plus YuelBond's per-bond-type classification F1 92.7 %
> as a third candidate for the same numeral) and the second under
> `sources`.`EF-TALFM` (EF-TALFM arXiv:2609.08333v4 Table 3(a): RAW connectivity
> `C. = 92.98 %`, counted BEFORE the largest-component retention, with the
> post-processed `N. & U. & S. = 94.33 %` companion cell).  The NUMBER of
> competing readings is itself NOT settled on disk: this record enumerates two,
> and the repo's primary-source verification
> (`molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json`
> -> `invariants`.`competing_readings`) records a further reading that likewise
> does not reproduce the quoted numerals.  They do not even agree on the
> numerals (92.7/94 vs 92.98/94.33), so the band's source cells are NOT settled
> on disk.  Under EVERY reading the band is EXTERNAL, is NOT one homogeneous
> quantity, and is NOT on our scale -- OUR number IS a connected-component count
> on the RAW emitted SMILES, and nothing else.  (In THIS pool:
> 1078 of 12668 parsed rows have exactly one
> connected component.)

**THE FOUR INVARIANTS ARE UNCHANGED.**  Under EVERY reading the band is
EXTERNAL, is NOT one homogeneous quantity, and is NOT on our scale --
OUR number IS a connected-component count on the RAW emitted SMILES,
and nothing else.  The band's source cells are NOT settled on disk.

**LOCATORS (JSON PATHS, not line numbers).**  molmetal/reports/wfa158_frontier_postproc_matrix/frontier_matrix.json -> `band_provenance` and `sources`.`EF-TALFM` (JSON PATHS, not line numbers); molmetal/reports/wfa186_frontier_primary_source/primary_source_verification.json -> `invariants`.`competing_readings`

## 1. The frontier rule, as the live fetch states it

Fetched live on 2026-11-01 from the arXiv LaTeX source of EF-TALFM
(`https://arxiv.org/e-print/2609.08333`), `src/appendix.tex`:

```latex
\STATE Select reconstruction policy $\pi \gets \texttt{ChoosePolicy}(\hat{\mathbf z}, \hat{\mathbf R}, \hat Q)$
\STATE Compute bond threshold $\tau_{ij} \gets \lambda_\pi \big(r^{\mathrm{cov}}(\hat z_i) + r^{\mathrm{cov}}(\hat z_j)\big)$
```

So the paper's rule is **tau_ij = lambda_POLICY * (r_cov(z_i) + r_cov(z_j))**.
`lambda_POLICY` is the tolerance factor OF THE SELECTED POLICY `pi`; the `pi`
is a SUBSCRIPT, not the constant pi.  The branch brief's `lambda * pi * (...)`
-- and this repo's `yuelbond_decoder.py::radii_scaled_factor` -- read it as a
product with 3.14159.  Both are reported below (`paper` / `repo_pi`).

The paper's numeric `lambda_POLICY` is **NOT recoverable**: the whole LaTeX
source contains the word `policy` exactly twice (both in Algorithm 4) and no
numeric value for `lambda_pi` anywhere.  The lambdas below are a SWEEP.

## 2. Live fetch log

| URL | tool | resolved | what came back |
|---|---|---|---|
| `http://export.arxiv.org/abs/2609.08333` | WebFetch | YES | arXiv abstract page. Returned title 'Fixed-Dimensional Latent Flow for Generating Variable-Size 3D Molecules', authors Weichi Yao, Cameron Gruich, Bryan R. Goldsmith, Yixin Wang. Abstract-only: NO appendix, NO bond-threshold formula, NO 'Connected'/'Sanitized' percentages. |
| `https://arxiv.org/abs/2609.08333v1` | WebFetch | YES | Abstract/listing page only -- no appendix text, no tau_ij formula. The summariser additionally emitted a '89.4% vs 75.6% UAE-3D / 69.8% FlowMol' passage that does NOT appear in this paper's HTML and is therefore treated as SUMMARISER NOISE, not as a quote. Discarded; every number in this report comes from the raw HTML/LaTeX fetches below, not from this page. |
| `https://arxiv.org/html/2609.08333v1` | curl (HTTP 200, 1,234,045 bytes, saved to /tmp/wfa156_arxiv_html_2609.08333v1.html) | YES | Full LaTeX-rendered HTML. Confirmed IN TEXT: 'We propose candidate heavy-heavy bonds between heavy atom pairs whose Euclidean distance falls below a scaled sum of covalent radii for the pair.' Confirmed: 'Connectivity: % of molecules with exactly one connected component, i.e., one fragment.' Confirmed: 'All rates are computed over 10,000 generated molecules.' Confirmed: 'For fragmented outputs, SMILES-based and PoseBusters metrics are computed on the atom-count largest connected component after RDKit sanitization.' Confirmed: 'Relative to FlowMol, EF-TALFM nearly matches the sanitization rate (98.25% versus 98.72%) ... while its connectivity is moderately lower (92.98% versus 98.72%).' Confirmed: the PCQM4Mv2 run uses the ALL-ATOM representation, whose connectivity comes from Open Babel perception, while the heavy-atom branch (Algorithm 4) is where the tau_ij rule lives. |
| `https://arxiv.org/e-print/2609.08333` | curl (HTTP 200, 8,508,562 bytes gzip, extracted to /tmp/wfa156_src/) | YES | LaTeX source. THE AUTHORITATIVE QUOTE: src/appendix.tex:1043 -- \STATE Compute bond threshold $\tau_{ij} \gets \lambda_\pi \big(r^{\mathrm{cov}}(\hat z_i) + r^{\mathrm{cov}}(\hat z_j)\big)$ ; src/appendix.tex:1039 -- \STATE Select reconstruction policy $\pi \gets \texttt{ChoosePolicy}(...). The word 'policy' occurs EXACTLY TWICE in the whole source (both in Algorithm 4) and NO numeric value for \lambda_\pi appears anywhere. Confirmed Table 5: C. 92.98, S. 98.25 for EF-TALFM over 10,000 molecules. |
| `https://arxiv.org/abs/2508.12629` | curl (HTTP 200, 42,489 bytes) | YES | FlowMol3: 'FlowMol3: Flow Matching for 3D DeNovo Small-Molecule Generation', Dunn, Ian; Koes, David R., dated 2025/08/18. Abstract states 'nearly 100% molecular validity for drug-like molecules with explicit hydrogens'. NO journal_ref and NO 'Digital Discovery 2026 5(5):2052-2066' metadata on the arXiv record, and the '% Valid 99.9+-0.1 / % PB-Valid 91.9+-0.7' figures were NOT verified from this page. |
| `https://arxiv.org/abs/2411.05676` | curl (HTTP 200, 42,606 bytes) | YES | GGFlow ORIGINAL: 'Improving Molecular Graph Generation with Flow Matching and Optimal Transport', Hou, Zhu, Ren, Bu, Gao, Zhang, Sun, dated 2024/11/08. A DISCRETE 2D molecular-GRAPH flow model -- no 3D geometry, hence no covalent-radius bond threshold of any kind. The 'retitled TMLR 2026 / OpenReview K8RlXtMgzo' identity in the branch brief was NOT verified from this page. |
| `https://arxiv.org/abs/2509.01486` | curl (HTTP 200, 41,794 bytes) | YES | PAFlow: 'Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number', Zhou, Qian, Tu, Xu, dated 2025/09/01. SBDD model on CrossDocked2020, Avg. Vina Score up to -8.31. NOT the paper the brief calls 'arXiv 2509.01486 NeurIPS 2025' in a way that changes any number here -- it contributes NO bond-threshold rule. |

## 3. The decoder's own rule (NO `master.log` next to this pool -- CODE DEFAULTS only)

```
(NO master.log next to this pool: molmetal/reports/cfm_c7_fullgen_2026_10_20/master.log)
```

* geometric **candidacy** gate: flat `d_ij <= 3.5 A`
* learned **acceptance**: `p(bonded) >= 0.5` on the YuelBond head
* dative bonds additionally capped at `2.6 A`
* the D2 connectivity prior may ADD bridging bonds, but only between pairs
  already inside the same 3.5 A candidate gate (never invents a long bond)

**Status of the 3.5 A premise.**  It is the CODE DEFAULT, not an observed
value for this run: `YuelBondDecoderHead.cutoff` defaults to 3.5 A
(`yuelbond_decoder.py:799`, applied as `mask = d_raw < self.cutoff` at :1244
and :1934) and `CFMYuelBondDecoder.distance_cutoff` defaults to 3.5 A
(`cfm_decoder_yuelbond.py:1048`, assigned at :1172).
NO master.log was found next to this pool, so NOTHING about THIS run's
decoder configuration was observed: every value above is a CODE DEFAULT
and nothing more.
**Every T1/T2 number below is therefore CONDITIONAL on this
run having used the 3.5 A default.**  If a different cutoff was used, the
T1 comparison point moves with it.

**The decoder's geometry is NOT on disk.**  The pool CSV header is
`epoch,smiles,n_frags_raw,largest_frag_frac` (writer:
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py:19250`) and the run
directory holds no coordinate dump -- only checkpoints, events.jsonl, generated_smiles.csv, per_epoch_metrics.csv, per_epoch_metrics.json, synthesis.md, wasserstein_decay.json.
So the literal "would the emitted
bond survive tau_ij in the decoder's OWN geometry" is **not measurable**
here.  Three tiers are reported instead, each labelled.

## 4. The pool (MEASURED)

* rows read: **12669**; parsed by RDKit: **12668**; unparsed: **1**
* emitted bonds recovered from the SMILES: **502164**
* of which metal-incident: **40077**
* bonds with a tmQM-DFT empirical reference length: **501664** (missing: 500)
* fragment-count histogram (RDKit, over parsed rows): `{'1': 1078, '2': 2421, '3': 2655, '4': 1979, '5': 1060, '6': 515, '7': 195, '8': 73, '9': 21, '10': 4, '11': 3, '13': 1, '21': 9, '22': 86, '24': 170, '25': 184, '26': 83, '27': 86, '28': 29, '29': 171, '30': 34, '31': 6, '32': 123, '33': 220, '34': 62, '35': 87, '36': 60, '37': 167, '38': 129, '39': 88, '40': 95, '41': 84, '42': 40, '43': 50, '44': 18, '45': 45, '46': 77, '47': 86, '48': 9, '49': 70, '50': 22, '51': 49, '52': 15, '53': 21, '54': 26, '55': 32, '56': 22, '57': 30, '59': 15, '60': 3, '61': 8, '62': 6, '64': 6, '65': 9, '67': 6, '69': 6, '73': 6, '74': 2, '79': 9, '89': 2}`
* bond-order histogram: `{'1': 501338, '2': 685, '4': 141}`

**Population cross-check** -- computed by THIS tool from the same parse, so a
reader can confirm it sees the population this run actually measured:

* rows carrying a metal: **12668/12669** (= 0.9999 metal_ratio)
* metal atoms sharing a fragment with a non-metal: **9612**
* metal atoms ALONE in their fragment: **3056**
* rows containing `->`: **0**
* rows with exactly one fragment (RDKit): **1078**

### 4.1 T1 -- EXACT, geometry-free: is tau_ij stricter or looser than the
### decoder's own 3.5 A gate, per emitted bond?

| convention | lambda | frac of emitted bonds where tau < 3.5 A (frontier STRICTER) | n stricter | n looser |
|---|---|---|---|---|
| paper | 0.8 | 1.0000 | 502164 | 0 |
| paper | 1.0 | 1.0000 | 502164 | 0 |
| paper | 1.2 | 1.0000 | 502164 | 0 |
| paper | 1.4 | 0.9974 | 500847 | 1317 |
| paper | 1.6 | 0.9401 | 472078 | 30086 |
| repo_pi | 0.8 | 0.0102 | 5128 | 497036 |
| repo_pi | 1.0 | 0.0000 | 0 | 502164 |
| repo_pi | 1.2 | 0.0000 | 0 | 502164 |
| repo_pi | 1.4 | 0.0000 | 0 | 502164 |
| repo_pi | 1.6 | 0.0000 | 0 | 502164 |

### 4.2 T2 -- EXACT, geometry-free bounds

| convention | lambda | frac guaranteed SURVIVE | frac guaranteed FAIL | frac undecidable |
|---|---|---|---|---|
| paper | 0.8 | 0.0000 | 0.9926 | 0.0074 |
| paper | 1.0 | 0.0000 | 0.0000 | 1.0000 |
| paper | 1.2 | 0.0000 | 0.0000 | 1.0000 |
| paper | 1.4 | 0.0026 | 0.0000 | 0.9974 |
| paper | 1.6 | 0.0599 | 0.0000 | 0.9401 |
| repo_pi | 0.8 | 0.9898 | 0.0000 | 0.0102 |
| repo_pi | 1.0 | 1.0000 | 0.0000 | 0.0000 |
| repo_pi | 1.2 | 1.0000 | 0.0000 | 0.0000 |
| repo_pi | 1.4 | 1.0000 | 0.0000 | 0.0000 |
| repo_pi | 1.6 | 1.0000 | 0.0000 | 0.0000 |

### 4.3 T3 -- LENGTH-TRANSFERRED ESTIMATE (pool bonds scored at the
### tmQM-DFT empirical MEAN length for their element pair)

**This is an ESTIMATE for the pool**, not a measurement of the decoder's
geometry.  It assumes each emitted bond has the same length as the average
real tmQM bond of its element pair and order.

| convention | lambda | frac surviving (transferred) | n survived | n failed | denominator |
|---|---|---|---|---|---|
| paper | 0.8 | 0.0000 | 4 | 501660 | 501664 |
| paper | 1.0 | 0.9400 | 471581 | 30083 | 501664 |
| paper | 1.2 | 1.0000 | 501664 | 0 | 501664 |
| paper | 1.4 | 1.0000 | 501664 | 0 | 501664 |
| paper | 1.6 | 1.0000 | 501664 | 0 | 501664 |
| repo_pi | 0.8 | 1.0000 | 501664 | 0 | 501664 |
| repo_pi | 1.0 | 1.0000 | 501664 | 0 | 501664 |
| repo_pi | 1.2 | 1.0000 | 501664 | 0 | 501664 |
| repo_pi | 1.4 | 1.0000 | 501664 | 0 | 501664 |
| repo_pi | 1.6 | 1.0000 | 501664 | 0 | 501664 |

## 5. BASELINE -- the tmQM ground truth, on REAL DFT geometry

**NOT MEASURED** -- the tmQM baseline could not be built in this run.
## 6. What this report does NOT establish

* It does NOT measure the decoder's own geometry: the pool has no
  coordinates, so the emitted bonds' real distances are unknown.
* It does NOT reproduce, approximate or claim the frontier's Connected
  92.98 % / Sanitized 98.25 %; the four comparability breaks in section 0
  are unrepaired by any number above.
* It does NOT establish that adopting tau_ij would reduce fragmentation:
  no decode was run (0-GPU branch).  Every 'would' here is arithmetic on
  a threshold, not an outcome.
* It does NOT recover the paper's `lambda_POLICY`; the value is absent from
  the paper's own source, so the sweep is a sweep.
* It does NOT resolve the `lambda * pi` misreading in the repo's Wf-A155
  `radii_scaled_factor`; that file is owned by another wave and was READ
  ONLY here.  Both conventions are reported so the reader can see the
  3.14159x gap rather than inheriting one of them silently.
* It does NOT provide a ROW-MATCHED ground truth for the pool: the pool
  CSV has no source-complex column, so the baseline is the tmQM
  GROUND-TRUTH POPULATION (same 6 metals), compared by chemistry only.
* It does NOT independently confirm the 3.5 A candidate gate FOR THIS RUN:
  that is the CODE DEFAULT, and the run's log prints a cutoff only for
  the EGN velocity field.  The T1/T2 numbers are conditional on it.
* It does NOT treat the baseline's Wiberg BO > 0.1 bond definition as a
  chemical bond census: it includes long weak contacts, which is why some
  real bonds fail every tau in the sweep.

## 7. Reproduction

```bash
CUDA_VISIBLE_DEVICES="" /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/wfa156_bond_threshold_frontier.py
```

Outputs in this directory:

* `report.json` -- the full machine-readable payload (git-ignored by
  `.gitignore`'s `molmetal/reports/**/*.json` rule)
* `report.md` -- this file (tracked)
* `per_row.csv` -- the compact, TRACKED per-row survival table: one line
  per pool row with its fragment structure and its own survival fractions
* `per_row.json` -- the FULL per-row record including every emitted bond's
  element pair, reference length and tau at every (convention, lambda)

