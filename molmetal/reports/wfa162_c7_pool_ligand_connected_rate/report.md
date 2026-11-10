# Wf-A162 — the ligand-only arm on the c7 pool (the 8.51 % headline's OWN population)

**Date:** 2026-11-01
**Owner:** main thread (not a wave)
**GPU:** 0 — CPU-only, RDKit-only, no checkpoint, no decode, no sampling
**Machine artefact:** `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json`
**Producer script:** `molmetal/scripts/wf_a155_ligand_connected_rate.py` (Wf-A155 branch LIG),
re-run by Wf-A162 with `--pool-csv` pointed at the **c7** pool instead of its default c8 pool
**Cross-checked against:** `metrics/by_metric/cfm_fragmentation_postproc_arm.json`,
`metrics/by_metric/cfm_fragmentation_audit.json`

---

## 1. The gap this closes

The published fragmentation story had **two populations and one definition**:

| number | population | what it measured |
|---|---|---|
| **8.51 %** | the c7 pool (`cfm_c7_fullgen_2026_10_20`), 12,668 parsed rows | metal-inclusive single-fragment (i.e. *connected*) rate, RAW |
| **25.05 %** | the c8 pool (`cfm_c8_fullgen_2026_10_24`), 2,112 rows | the same definition, different pool |

The **ligand-only arm existed only on c8** (2.84 %), so "is the metal hiding the ligand's
fragmentation?" could be answered for c8 but **not for the pool the headline actually comes from**.
This run measures it on c7.

## 2. Result (MEASURED)

| quantity | c7 pool (12,668 rows) | c8 pool (2,112 rows, Wf-A155) |
|---|---|---|
| metal-inclusive single-fragment (= connected) rate | **0.08509630565203663** (1078) | 0.25047348484848486 (529) |
| ligand-only single-fragment rate | **0.02083991158825387** (264) | 0.028409090909090908 (60) |
| mean n_frags, metal-inclusive | 10.244 | 2.474 |
| mean n_frags, ligand-only | **11.603** | 4.757 |
| largest_frag_frac median, metal-inclusive | 0.9268 | 0.9655 |
| largest_frag_frac median, ligand-only | 0.7632 | 0.7143 |
| rows where the metal sits alone in its own fragment | 3056 (24.12 %) | 30 |
| rows with a free halide counterion | 1903 | 60 |

### 2.1 Paired contingency (c7)

| cell | n | meaning |
|---|---|---|
| a | 220 | connected with the metal AND connected without it |
| b | **44** | "the metal fell off", ligand intact |
| c | **858** | connected ONLY because the metal bridges otherwise-disconnected ligand pieces |
| d | 11,546 | disconnected either way |

McNemar exact two-sided **p = 8.635e-197** (log10 −196.06; discordant 902; exact rational
numerator/denominator stored, **not** underflowed to zero).

### 2.2 The answer to "why are the molecules fragments?"

**On the headline's own pool, the metal is a connectivity HUB and it is CONCEALING the ligand's
fragmentation, not causing it.**

- Only **44 / 12,668** rows (0.35 %) are "the metal fell off and the ligand stayed intact".
  The "metal falls off" hypothesis is **REFUTED** as the dominant explanation.
- **858 of the 1,078 "connected" molecules (79.6 %)** are connected *only* because the metal
  bridges ligand pieces that are otherwise disconnected. Strip the metal and they shatter.
- Ligand-only connectivity is **2.08 %** vs 8.51 % metal-inclusive — a **4.1×** gap on the same
  rows, and the mean fragment count rises 10.24 → 11.60 when the metal is deleted.
- ⇒ **The repair belongs on the ligand skeleton, not on keeping the metal attached.** A
  metal-coordination fix would move the 44-row cell, which is 0.35 % of the pool.

### 2.3 Cross-validation — the headline is reproduced bit-for-bit

This run was produced by a **different script** from the one that published the headline
(`wf_a155_ligand_connected_rate.py` vs the Wf-A133 post-processing arm). They agree to the
last bit:

| published (`cfm_fragmentation_postproc_arm.json`) | this run | equal? |
|---|---|---|
| `raw_axis/single_fragment_rate_over_parsed = 0.08509630565203663` | `metal_inclusive/single_fragment_rate = 0.08509630565203663` | ✅ identical |
| `raw_axis/n_single = 1078` | `n_single_fragment = 1078` | ✅ identical |
| `per_epoch/0/raw_single_rate = 0.028409090909090908` | `0.028409090909090908` | ✅ identical |
| `per_epoch/1/raw_single_rate = 0.022727272727272728` | `0.022727272727272728` | ✅ identical |
| `per_epoch/2/raw_single_rate = 0.04403409090909091` | `0.04403409090909091` | ✅ identical |

Plus the script's own `csv_cross_check`: the metal-inclusive fragment count was **recomputed from
scratch** and compared against the c7 pool's own `n_frags_raw` column →
**`n_frags_raw_mismatches = 0`** over all 12,668 rows, `largest_frag_frac_max_abs_delta = 5e-05`
(rounding). And `metal_set_invariance.identical = true` — the answer is unchanged under the broad
d-block metal predicate and under the narrow trainer cohort {Ru, Rh, Re, Os, Ir, Pt}, so it is not
pool-fitted.

⇒ **Three independent code paths agree on the c7 headline.** It is not a measurement artefact.

### 2.4 Two different quantities both print as "8.51 %"

| | value | quantity |
|---|---|---|
| A | `0.08506944444444445` | **median over 9 runs** of `single_molecule_rate_end_to_end` — "one connected parseable molecule per DRAWN sample" (`cfm_fragmentation_audit.json`) |
| B | `0.08509630565203663` | **single-pool raw rate** over 12,668 parsed rows (`cfm_fragmentation_postproc_arm.json`) |

They differ by 2.69e-05 absolute (0.032 % relative) and **both render as "8.51 %"**. This run
reproduces **B**. Any document that writes "8.51 %" without naming which of the two it means is
ambiguous — same class as catalog §6.263 (a number that renders correctly but says the wrong thing).

## 3. NEW DEFECT #1 — the measurement tool hard-codes its own headline

**`molmetal/scripts/wf_a155_ligand_connected_rate.py` embeds seven pool-specific claims as string
literals.** For its default pool (c8) every one of them is true. For any other pool the tool emits
**false statements inside a `tag: MEASURED` artifact** — which AGENTS.md §5.2 forbids outright.

| # | site | the literal | truth on c7 |
|---|---|---|---|
| 1 | `purpose` | "never a restatement of the 8.51 % headline" | **FALSE** — it reproduces it to 17 digits |
| 2 | `not_comparable_to.8_51_percent_headline` | "over ONE pool of **2112** rows" | **FALSE** — 12,668 rows |
| 3 | `not_comparable_to.25_05_percent_single_pool_fragment_count` | "the metal-inclusive arm of THIS tool reproduces **that** number" (the 25.05 %) | **FALSE** — it reproduces the 8.51 % number |
| 4 | `comparability.breaks[0].ours` | "ONE pool of **2112** rows from ONE post-hoc decode (**cfm_c8_fullgen_2026_10_24/generated_smiles.csv**)" | **FALSE** — wrong row count **and the wrong pool path** |
| 5 | `caveats[2]` | "ONE pool, ONE post-hoc decode, **2112** rows" | **FALSE** — 12,668 rows |
| 6 | `caveats[4]` | "The **60** rows with a free halide counterion" | **FALSE** — 1,903 rows |
| 7 | `comparability.statement` | "Four independent breaks" | OK (4 breaks are listed) |

Site 4 is the worst: **the artifact names the wrong source pool in its provenance-bearing text.**
A reader following `source_pool` would be sent to a different population than the one measured.

The defect is **LATENT for the published metric** — `metrics/by_metric/wfa155_ligand_connected_rate.json`
is the c8 run, so sites 1–6 happen to be true there (site 1 is true for c8; site 3 is true for c8).
The **published** artifact is not wrong. The **tool** is.

**Repair spec (read-only-review-drivable, no GPU):**

1. Compute every one of those numbers from the run: `n_analyzed`, `n_with_free_halide_counterion`,
   the pool's basename, the pool's own row count.
2. Replace the "reproduces the 8.51 % / 25.05 %" claims with a **computed** cross-check: compare
   this run's metal-inclusive rate against the published value **for this pool** (read the pool's
   name out of `cfm_fragmentation_postproc_arm.json` / `cfm_fragmentation_audit.json`) and state
   the result, instead of asserting which number it is.
3. `comparability.breaks[0].ours` must quote `source_pool` and the measured row count.
4. Add a **self-consistency assertion**: every integer appearing in a narrative string must also
   appear in a machine field of the same artifact (or the run must fail). This is the guard that
   stops the class, not just the instance.
5. Keep the MEASURED/PROJECTED tagging; keep the ligand-only arm's "diagnostic, not a target"
   framing (a coordination complex legitimately needs the metal in its connectivity).

## 4. NEW DEFECT #2 — the pre-flight allocates GPU memory to answer "is there a GPU?"

**MEASURED.** The GPU watchdog (`bpzsdj6aj`, polling `nvidia-smi` every 5 s since 2026-10-31
05:49) recorded **one** mol-metal GPU allocation in its whole life:

```
GPU-VIOLATION mol-metal pid=700241, /home/hugo/codes/mol-metal/.venv/bin/python, 550 MiB
```

- 24 consecutive lines ⇒ the process held the allocation for ≈ 120 s.
- The watchdog file's **last write is 2026-10-31 17:52:41** ⇒ the event is
  **2026-10-31 ≈ 17:50–17:52**. **Zero** mol-metal GPU allocations in the ~31 h since.
- PID 700241 is gone; the surrounding window is a pytest run (Wf-A142 / Wf-A133 test files were
  being written at 17:45–18:00).
- **No `.pt` file was written anywhere in the repo on 2026-10-31** (verified with a pruned
  `find -name '*.pt' -newermt "2026-10-31 00:00"`), and the chemprop smoke's checkpoint
  (`molmetal/checkpoints/wf200_chemprop_v2_v3/`) still carries its 2026-10-09 mtimes.
  ⇒ **No GPU training, sampling or decode ran.** What allocated the memory was a **CUDA context**
  created by a device-visibility query, not a workload.

**Mechanism (read from the installed torch, not executed):**
`molmetal/scripts/pre_flight_check.py::_smoke_device()` (:13056) spawns

```
[VENV_PY, "-c", "import torch;print(torch.cuda.is_available())"]
```

with **no `env=`**, so the child inherits the ambient environment. In the installed torch
(`.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:217-229`), `is_available()` takes the
`else` branch unless `PYTORCH_NVML_BASED_CUDA_CHECK=1` — and that branch's own comment says it
*"uses the CUDA Runtime API `cudaGetDeviceCount` which in turn **initializes the CUDA Driver API
via `cuInit`**"*. A CUDA context is ≈ 500 MiB. **The observed 550 MiB is that context.**

Two aggravating facts:

1. The pre-flight **already knows** about the repo's CPU lock — `pre_flight_check.py:6902` reads
   `_session_cuda_locked = os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""` and SKIPs the
   GPU-REQUIRED gate with the honest framing *"the repo's standing CPU lock"*. `_smoke_device()`
   **does not consult it.** Under a locked session the probe answers "cpu" for free; the probe
   only allocates when the lock is absent.
2. When the lock is absent, `_smoke_device()` returns `"cuda:0"` — and the two gates that consume
   it (`pre_flight_check.py:13293` Wf-446 H4 D-MPNN smoke, `:16861` Wf-446 H2 chemprop smoke) then
   run an **actual 2-epoch training on the GPU**. The code says so itself:
   *"device comes from `_smoke_device()` — cuda:0 when this host has a GPU, cpu only as the
   fallback."* ⇒ **the repo's own gate tool can silently start GPU training.**

⇒ **Under the standing 禁止使用gpu mandate, running the pre-flight without
`CUDA_VISIBLE_DEVICES=""` allocates GPU memory and, on this host, would run GPU training.**
That is a live hazard, not a historical one.

**Repair spec (no GPU):**

1. `_smoke_device()` must short-circuit on `_session_cuda_locked` and return `"cpu"`
   **without spawning torch at all** — behaviour-identical (the probe would have answered
   "cpu" anyway) and it removes the allocation entirely.
2. Belt-and-braces: give the probe `env={**os.environ, "PYTORCH_NVML_BASED_CUDA_CHECK": "1"}`
   so that even in a GPU session the answer comes from NVML **without `cuInit`**. (This must be
   checked against `CUDA_VISIBLE_DEVICES` masking semantics before it lands — NVML-based
   `device_count()` may not honour the mask, so the short-circuit in (1) is the load-bearing
   part and (2) is the belt.)
3. Add a regression test: with `CUDA_VISIBLE_DEVICES=""`, `_smoke_device()` returns `"cpu"` and
   spawns **no** subprocess.
4. Make the Wf-446 H4 / H2 smoke gates assert that the device they are about to use is the one the
   session authorised, and FAIL loudly (not silently train) if the session forbids GPU.

## 5. What is NOT measured here

- **Nothing was re-decoded.** Every number above is a post-hoc analysis of an existing CSV.
- The ligand-only arm is a **counterfactual** (the metal deleted), not a real molecule.
- Both arms are **RAW**: no LargestFragmentChooser, no explicit H, no UFF, no post-processing.
  The published *post-processed* 100 % connected rate is a different quantity and is not
  restated here.
- One pool, one post-hoc decode — not a training-set-distribution sampling, so this is not a rate
  over the generator's output distribution.
- The pools carry implicit hydrogens; the frontier datasets carry explicit ones.
- **No GPU** was used, and none may be used to confirm any of this under the current mandate.

## 6. Reproduce

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wf_a155_ligand_connected_rate.py \
    --pool-csv molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv \
    --out molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json
```

Runtime ≈ 4 min wall, CPU-only, RDKit-only.


---

## 7. REPAIR LOG (appended 2026-11-01, Wf-A162 → Wf-A163)

Append-only. Everything above is the state as MEASURED before the repairs; nothing above was
edited. The superseded renderings live in `docs/HONEST_NEGATIVE_CATALOG.md` §6.265 (AGENTS.md §5.5
rule #6).

### 7.1 DEFECT #1 repaired — the `lambda * pi` misreading (Wf-A160)

`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::radii_scaled_factor` returned
`lambda * math.pi` and its docstring claimed the factor was *"exactly `lambda * pi`, as quoted"*
from EF-TALFM (arXiv 2609.08333). The e-print was re-fetched and its sha256 verified
(`e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224`): **π is a SUBSCRIPT**
(`tau_ij <- lambda_pi * (r_cov(z_i) + r_cov(z_j))`, Appendix C.2.1 Algorithm 3), π denotes the
*reconstruction policy* selected by `ChoosePolicy(ẑ, R̂, Q̂)`, and the source contains **zero**
`lambda*pi` products. The function is now the **identity** on the supplied `lambda_pi`, and every
docstring in the module quotes the rule verbatim.

**MEASURED, and it flips the expected effect sign:** the largest pair-sum in the on-disk Cordero
table and in `_YUELBOND_Z_TABLE` is **2.78 Å** (I–I, z=53), so at `lambda_pi = 1` the paper's rule
is **STRICTLY STRICTER** than the flat 3.5 Å cutoff for every emittable pair (the minimum
`lambda_pi` that would reach 3.5 Å for the worst pair is **1.2590**). The buggy π gave
`lambda_eff = 3.1416` and `tau_max = 8.73 Å` — "3.14x the flat cutoff in disguise". Severity
**LATENT**: `radii_scaled_candidacy` defaults to `False` at all three constructor sites and is
enabled nowhere except a synthetic unit test, so no published metric changes — but the defect would
have **understated** fragmentation in the very chain whose headline is "the CFM emits fragments".

### 7.2 DEFECT #2 repaired — the pre-flight's CUDA-context allocation (Wf-A162)

`molmetal/scripts/pre_flight_check.py`:

- `_smoke_device()` now short-circuits to `"cpu"` when the session blanks `CUDA_VISIBLE_DEVICES`,
  **without spawning torch at all** (the probe would have answered "cpu" anyway — behaviour
  identical, allocation removed);
- when the probe does run it carries `PYTORCH_NVML_BASED_CUDA_CHECK=1`, so the answer comes from
  NVML rather than `cuInit` (NVML does not honour the device mask, which is why the short-circuit
  is the load-bearing half);
- a new `_device_authorisation_error(device)` refuses any device the session did not authorise —
  including a `-1` mask, which hides every device without blanking the variable — and **both**
  consuming gates (Wf-446 H4 D-MPNN smoke, H2 chemprop smoke) now **FAIL loudly, smoke NOT run**,
  instead of silently starting a 2-epoch GPU training.

Regression surface: `molmetal/tests/test_wfa162_smoke_device_cpu_lock.py` — **10 tests**, including
two **AST-based** structural guards (every `_smoke_device()` call site must live in a function that
also calls `_device_authorisation_error(`, and no probe may run without an explicit `env=`). AST,
not text, because the module's own prose mentions both symbols — a text scan flagged the
documentation, not the code (the second time this session a naive text scan produced a false
finding).

### 7.3 DEFECT #3 repaired — the hard-coded narrative (Wf-A162)

`molmetal/scripts/wf_a155_ligand_connected_rate.py`:

1. every number describing OUR population is **computed** from the run (`build_comparability_breaks(pool_name, n_analyzed)`);
2. "this run reproduces published number X" is now a **computed cross-check**
   (`_published_cross_check`): a published artifact is a candidate only when its own `source_pool`
   names THIS pool, and the delta is measured — never asserted;
3. the frontier band and its denominator became **machine fields** (`comparability.frontier_band`,
   `comparability.frontier_denominator`), so the numbers quoted in prose are backed;
4. a **self-consistency guard** (`_narrative_self_consistency`) requires every number rendered in a
   narrative string to be backed by a machine field of the same artifact. An unbacked number is a
   DEFECT: the tool raises `NarrativeSelfConsistencyError`, writes the artifact with
   `"status": "FAIL"`, and exits 1.

**MEASURED, both pools, guard green:**

| pool | rows | guard | prose leaves | prose numbers | unbacked | cross-check |
|---|---|---|---|---|---|---|
| c7 `cfm_c7_fullgen_2026_10_20` | 12,668 | PASS | 31 | 24 | 0 | reproduces `cfm_fragmentation_postproc_arm` **bit-for-bit** (Δ = 0.0, 1078 = 1078) |
| c8 `cfm_c8_fullgen_2026_10_24` | 2,112 | PASS | 31 | 24 | 0 | reproduces `wfa155_ligand_connected_rate` **bit-for-bit** (Δ = 0.0, 529 = 529) |

**The guard's own hole, found by a negative control.** The first implementation resolved a narrative
path with a `path + "."` prefix. A narrative path that designates a **LIST** (`caveats`,
`comparability.breaks`) therefore matched nothing: its elements were classified as MACHINE fields,
every number inside them went unchecked, and the guard reported PASS on an artifact it had never
really examined — 13 prose numbers checked instead of 24. The negative control (inject the literal
`2112 rows` into a synthetic artifact and require the guard to reject it) exposed it; the fix
matches `path + "."` **and** `path + "["`. A guard is never shown a failing case is not known to
work, and a guard's own census is not a coverage measurement.

Regression surface: `molmetal/tests/test_wfa162_narrative_self_consistency.py` — **14 tests**,
including the negative controls, a **vacuity guard** (every `NARRATIVE_PATHS` entry must resolve to
at least one leaf — a path that resolves to nothing is silently unchecked), and the cross-check
tests. Together with §7.1/§7.2 the three repair suites are **85 tests, all green, CPU-only**.

**Files re-published** (the superseded c8 artifact is preserved VERBATIM at
`molmetal/reports/wfa162_c7_pool_ligand_connected_rate/wfa155_ligand_connected_rate.c8.SUPERSEDED_2026_11_01.json`,
sha256 `f2367e046cfe07a605f9bac1a7deb53f957e39e063b56278a0635b87bf1267ce`):

- `metrics/by_metric/wfa155_ligand_connected_rate.json` (c8, 14,167 B)
- `molmetal/reports/wfa162_c7_pool_ligand_connected_rate/ligand_connected_rate_c7.json` (c7, 19,216 B)

---

## 8. APPENDED 2026-11-01 (Wf-A162 close-out + Wf-A163 BRANCH V)

### 8.1 The pre-flight's contended measurement — and the FIRST EMPIRICAL CONFIRMATION of §7.2

The repaired `pre_flight_check.py` was run end-to-end **under the standing CPU lock**
(`CUDA_VISIBLE_DEVICES=""`) **while the Wf-A163 ultracode wave was executing concurrently** — the
CONTENDED measurement, not a quiet-host one.

| | value |
|---|---|
| PASS | **118** |
| FAIL | **0** |
| SKIP | **11** |
| WARN | **0** |
| TIMEOUT | **0** |
| TOTAL | **129** |
| wall | **81.603 s** |

**The counts are IDENTICAL to Wf-A157's quiet measurement (118/0/11/0/0 at 280.78 s).** The wall
time fell to 81.603 s because the CPU lock short-circuits the GPU probe (§7.2) — a *lock* effect,
not a *gate* effect.

**The confirmation that matters:** the mol-metal GPU watchdog's output file — whose last write is
**2026-10-31 17:52**, the original 550 MiB event — was **NOT written during this run**, and
`nvidia-smi --query-compute-apps` lists **no** mol-metal process. ⇒ **129 gates ran with ZERO GPU
allocation.** The 2026-10-31 run of the same tool allocated 550 MiB. **The §7.2 repair is
measured, not asserted.**

**The 11 SKIPs are each HONEST-NEGATIVE BY-DESIGN** (`p1_gpu_env_probe` is SKIP *because* the lock
is on; the other 10 are absent artefacts / external dependencies) — **0 FAILs**.

Published at `molmetal/reports/wfa162_preflight_freshrun_2026_11_01.txt` (2,415 B). The
Wf-415-A3 report (`molmetal/reports/wf415a3_preflight_freshrun.txt`, 161 B, mtime 2026-09-30
18:38, 112/2/3/117) is **left byte-identical** — it is what the four `TODO/completed/415*.md`
documents cite, and those citations remain true (AGENTS.md §5.5 rule #6).

### 8.2 The census republish — one narrative changed, no headline number moved

The rebuilt census artefact (sha256 `40ef9c51ae93eeac35b16afd2301ed41d67f7226ac0cb984270ca90dafe49e5a`)
was published over `metrics/by_metric/cfm_run_census.json` (previous sha256
`0774296dcaca85abc49ba717d1b7936ed397f5a5a9a24e46cd768b1e05e8dd11`).

**MEASURED diff (BEFORE vs NEW, both read from disk):** `n_runs` 59 → 59; all **four** census
medians **UNCHANGED** (CLEAN last-epoch 0.0625 n=27 / ALL-runs last-epoch 0.06380208333333333 n=30
/ CLEAN best-epoch 0.35546875 n=29 / ALL-runs best-epoch 0.34440104166666663 n=32).

What changed is a **meaning**: `fragmentation_summary.collapsed_note` now says the last-epoch
end-to-end rate of a collapsed run is **PUBLISHED AS null, NOT as 0.0** — the generation eval
DREW samples (`gen_validity_n > 0`) and SCORED none of them (`gen_validity_val == 0`), so the
denominator is EMPTY and the rate is undefined (the Wf-A127 null-vs-0.0 convention). The old
wording said the rate *"reads 0.0 and understates the run by its full range"* — i.e. it called an
undefined quantity a number. **The four medians quoted in `docs/HONEST_NEGATIVE_CATALOG.md`
§6.264 remain quotable.**

**Rule 6:** the superseded artefact is preserved byte-identical at
`molmetal/reports/wfa162_census_republish/cfm_run_census.SUPERSEDED_2026_11_01.json` (557,099 B,
sha256 `0774296d…`), and the registry row `cfm_run_census` was re-pointed
(`registry/metrics.yaml:3203`). **`registry.py check` PASSes: 579 current, 0 stale, 0 missing.**
Acceptance gates re-run green: `test_wfa131_repo_path_reference_guard.py` +
`test_wfa152_census_guard_coverage.py` + `test_wfa134_census_requote_guard.py` — **52 passed**;
`git status --porcelain metrics/` stayed at the **38-line** baseline.

### 8.3 BRANCH V — an INDEPENDENT re-derivation of the c7 headline, with NO shared code

`molmetal/scripts/wfa163_c7_rederive.py` re-derives §2 from scratch: **no import of**
`wf_a155_ligand_connected_rate`; connected components = its **own BFS** over a hand-built bond
adjacency (no `Chem.GetMolFrags`, no `RWMol.RemoveAtom` — asserted on the parsed AST); the metal
predicate **extracted at run time with `ast`** from `platinai_dataset` / `tmqm_dataset` /
`pt_click_compat` (never copied in); McNemar = its **own exact rational** (`Fraction` + `math.comb`),
with `scipy` used only as a cross-check.

**VERDICT: 13/13 MATCH, 0 MISMATCH, delta = 0.0 exactly (tol 0.0).**

| quantity | independent re-derivation | published | Δ |
|---|---|---|---|
| metal-inclusive rate | 0.08509630565203663 (1078/12668) | 0.08509630565203663 | 0.0 |
| ligand-only rate | 0.02083991158825387 (264/12668) | 0.02083991158825387 | 0.0 |
| contingency | a=220 b=44 c=858 d=11546 | a=220 b=44 c=858 d=11546 | 0.0 |
| McNemar exact p | 8.635435522167073e-197 | 8.635435522167073e-197 | **bit-identical** |
| per-epoch raw rates | 18/18 MATCH | 18/18 | 0.0 |
| own BFS vs pool's `n_frags_raw` | 0 mismatches over 12,668 rows | — | 0.0 |

`scipy.stats.binomtest` gives 8.635435522166859e-197 (rel Δ 2.5e-14) — the exact-rational
agreement being bit-identical shows the publisher used an exact rational computation, not
`scipy`.

**Probes that bound the claim (all MEASURED):**

- **DENOMINATOR.** 12,668, not 12,669 — independently confirmed: the CSV has 12,669 rows, 0
  empty, and **exactly 1 unparseable** at `generated_smiles.csv:9848` (epoch 13) — RDKit:
  *"Maximum BFS search size exceeded … highly symmetric fused ring system"*. Using 12,669 would
  give 0.08508958875996527 (Δ −6.7e-06).
- **ROW ORDER.** `shuffle(seed=0)` → rates and contingency identical.
- **EPOCH COLUMN.** A cyclic epoch mislabelling moves 17/18 per-epoch rates but moves the pooled
  rates and the contingency **not at all** ⇒ the epoch column carries no weight for the headline.
- **METAL PREDICATE.** The ligand-only rate is **0.02083991158825387 under all four** metal
  vocabularies tried — `dblock_platinai` (30), `dblock_tmqm` (30), `mle82` (82), and the
  pool-derived set (6: Ir Os Pt Re Rh Ru). ⇒ **the number is invariant across METAL vocabularies.**
  A **strawman control** (d-block **plus** the pool's metalloids B Si As Se) gives
  **0.018866435112093464** (n=239) ⇒ the invariance holds across *metal* vocabularies but
  **breaks** if the predicate is widened to *metalloids*. **Bounded, not a defect.**
- **CONNECTED DEF.** "1 component AND ≥1 heavy" vs "1 component only" → identical for both arms
  (hidden work = `False`); 0 rows have <2 heavy atoms, so "≥2 heavy atoms" also gives
  0.08509630565203663.
- **FREE HALIDES.** 1,903 rows (every line number listed in the artefact). Excluding them changes
  the denominators and leaves the **numerators UNCHANGED** (1078 / 264) ⇒ metal-inclusive
  0.10013934045517882, ligand-only 0.024523920111472364 (**+17.7 % relative**). ⇒ **the free
  halides DILUTE the ligand-only arm; they do not inflate it.**

**The one caveat surfaced by the re-derivation, and it is a scoping rule for §2 and §6:** the
ligand-only arm is a **METAL-rule-dependent quantity**. It may be quoted **only under a stated
metal predicate**, never under a metalloid-inclusive one. Every other probe found **no offender**:
the c7 headline survives all of them.

**Files:** `molmetal/scripts/wfa163_c7_rederive.py`,
`molmetal/tests/test_wfa163_c7_rederive.py` (**6/6 PASS**; SKIP-not-FAIL when the pool is
absent, verified), `molmetal/reports/wfa163_c7_rederive/rederive.json` + `report.md`.
