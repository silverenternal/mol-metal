# Wf-A160 — Independent verification of the EF-TALFM bond-threshold rule

**Date:** 2026-11-01
**Owner:** main thread (not a wave) — the file under test
(`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`) is owned by Wf-A155, so
this wave **READ ONLY** it and did not touch it.
**Machine-generated artefact:** `molmetal/reports/wfa160_bond_threshold_independent_verification/verification.json`
**Re-runnable script:** `molmetal/scripts/wfa160_bond_threshold_verify.py`
**CPU-only:** the script downloads a LaTeX tarball and greps text. No torch, no
GPU, no decode, no training.

---

## 1. The claim under test

`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::radii_scaled_factor`
returns `lambda_value * math.pi`, and the surrounding docstring asserts:

> "The multiplicative factor applied to the radius sum is exactly
> ``lambda * pi``, as quoted; ``radii_scaled_factor`` is the single place
> that product is formed."

and cites "EF-TALFM (arXiv 2609.08333 App. C.2.1, Algorithm 3)".

Wf-A156's frontier branch reported that the paper's rule is
`tau_ij <- lambda_POLICY * (r_cov(z_i) + r_cov(z_j))` — the `pi` being a
**subscript** on lambda (the reconstruction policy), not a multiplicative
constant. A claim that shifts a bond threshold by a factor of pi must not rest
on one agent's report, so it was re-derived here from the paper's own source.

## 2. Provenance (MEASURED)

| Item | Value |
|---|---|
| Source | `https://arxiv.org/e-print/2609.08333` (arXiv e-print) |
| Bytes | 8,508,562 |
| **sha256** | `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` |
| Fetched | 2026-11-01, independently of Wf-A156 (same byte count, same hash) |
| Script verdict | `THE-PI-IS-A-SUBSCRIPT` (exit 0) |

The hash is recorded in the script as `EXPECTED_EPRINT_SHA256`. If a future
fetch does not match it, the script returns `DRIFT` and **does not** re-derive
the verdict — the paper may have been revised.

## 3. What the paper actually says

`src/appendix.tex:1043` (the rule):

```
\STATE Compute bond threshold $\tau_{ij} \gets \lambda_\pi \big(r^{\mathrm{cov}}(\hat z_i) + r^{\mathrm{cov}}(\hat z_j)\big)$
```

`src/appendix.tex:1039` (what pi is):

```
\STATE Select reconstruction policy $\pi \gets \texttt{ChoosePolicy}(\hat{\mathbf z}, \hat{\mathbf R}, \hat Q)$
```

`src/appendix.tex:1003` (the prose the algorithm implements):

> "We propose candidate heavy-heavy bonds between heavy atom pairs whose
> Euclidean distance falls below **a scaled sum of covalent radii for the
> pair**."

**Three independent confirmations that the pi is a subscript, not a factor:**

1. `\lambda_\pi` — TeX puts the subscript between the two braces, so the pi is
   attached to the lambda.
2. The preceding algorithmic step *defines* `\pi` as the **reconstruction
   policy** selected by `ChoosePolicy`. A policy cannot multiply a threshold.
3. A whole-source scan for any `\lambda ... \pi` **product** (with the
   subscripted form excluded) returns **0 hits**.

The paper never states a numeric value for `\lambda_\pi` — the string
`\lambda_\pi` occurs exactly once in the entire source. So the repo's
"lambda is not recoverable, therefore default to the neutral identity
reading" reasoning is sound; only the extra `* math.pi` is wrong.

## 4. Where the rule lives — and a correction to Wf-A156's report

The rule is inside the **heavy-atom** branch. The source-level derivation
(appendix top-level sections in document order; the paper has one `\appendix`
and one appendix file):

| Appendix letter | Title | Line |
|---|---|---|
| A | Unconditional generation: protocols and supporting results | 1 |
| B | Property-conditioned generation: protocols and supporting results | 291 |
| **C** | **Methods** | **884** |
| D | Training configurations | 1267 |

⇒ C.2 = "Molecular graph reconstruction and geometry refinement" (963),
C.2.1 = "Molecular graph reconstruction" (987). The algorithm float is
physically placed *after* the C.2.2 "Geometry refinement" heading, but the
paper **introduces it from C.2.1** via `\Cref{algo:reconstruction}`.

| Reference | Derived value |
|---|---|
| **Appendix** | **C.2.1** |
| **Algorithm** | **3** (of 4 `\begin{algorithm}` in the whole source; the main text has none) |

**⇒ The repo's own citation "App. C.2.1, Algorithm 3" is CORRECT.**

**⇒ Wf-A156's report says "Appendix D, Algorithm 4" — that is WRONG on both
counts.** Appendix D is "Training configurations"; Algorithm 4 is the fourth
algorithm float (line 1118), not the reconstruction algorithm. Recorded here
so the wave's report is not propagated.

Also confirmed from the same source: the **all-atom** branch does not use this
rule at all — `src/appendix.tex:1016` says it uses "the rule-based
bond-perception routines in Open Babel". The headline PCQM4Mv2 numbers come
from that branch, so the radii-scaled rule is not the rule behind the
frontier's headline connectivity numbers (a fifth comparability break
against our metric — see §5).

## 5. Severity: LATENT, not active

| Question | Answer (MEASURED on disk) |
|---|---|
| Does the pi-factor affect any published metric? | **No.** `radii_scaled_candidacy` defaults to `False` at all three constructor sites (`yuelbond_decoder.py:1008`, `:1744`, `:2174`); the default branch is `d_raw < self.cutoff`, bit-identical to pre-Wf-A155. |
| Is the flag ever switched on outside a unit test? | **No.** Repo-wide grep for `radii_scaled_candidacy=True` hits only the docstring and `molmetal/tests/test_wfa155_decoder_bond_candidacy.py` (synthetic). No script, config, launcher, or report enables it. |
| What is the impact if someone flips it? | The candidate-bond threshold becomes **pi = 3.1416x more permissive** than the paper's rule, so the decoder would admit bonds out to `3.14 * (r_cov_i + r_cov_j)` instead of `1.0 * (r_cov_i + r_cov_j)` — and since a permissive threshold ADDS candidate bonds, it would **understate** fragmentation in the very chain whose headline is "the CFM emits fragments". |

**Honest framing:** this is a *latent* defect plus a *fabricated attribution*.
No published number changes. What is wrong today is (a) the semantics of an
opt-in flag that has never been enabled, and (b) a docstring that claims the
paper says something it does not say — the same citation-integrity family as
catalog §6.263.

## 6. Repair spec (to be applied when Wf-A155 releases the file)

1. `radii_scaled_factor(lambda_value)` returns `float(lambda_value)` — the
   identity. The name stays; the pi is gone.
2. The docstring quotes the rule **verbatim** as
   `tau_ij <- lambda_pi * (r_cov(z_i) + r_cov(z_j))`, states that the pi is a
   **subscript** denoting the reconstruction policy, and states that
   `\lambda_\pi` has **no recoverable numeric value** in the paper (one
   occurrence in the whole source) so `DEFAULT_RADII_SCALED_LAMBDA = 1.0` is
   the neutral identity reading and is **not** the paper's value.
3. The citation becomes "EF-TALFM (arXiv 2609.08333, Appendix C.2.1,
   Algorithm 3)" — the derivation in §4.
4. The wording "as quoted" is removed wherever it attaches to `lambda * pi`.
5. `molmetal/tests/test_wfa155_decoder_bond_candidacy.py`'s
   `test_the_factor_is_lambda_times_pi` and the five arithmetic assertions
   around it are re-pointed at the identity.
6. MEASURED/PROJECTED tagging is preserved. The claim "the radii-scaled rule
   improves fragmentation" stays **PROJECTED** — no re-decode was run, and
   under the 0-GPU mandate none may be.

## 7. Reproduce

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa160_bond_threshold_verify.py \
    --out molmetal/reports/wfa160_bond_threshold_independent_verification/verification.json
```

Exit 0 = the verdict `THE-PI-IS-A-SUBSCRIPT` was re-derived from the source;
exit 1 = DRIFT / NOT-FOUND / INCONCLUSIVE (and the verdict must NOT be quoted).
