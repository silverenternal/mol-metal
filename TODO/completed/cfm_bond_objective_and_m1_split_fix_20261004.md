# CFM bond objective + decision rule + split-coherence — closeout (2026-10-04)

**Verdict: the CFM's bond objective and its metric are now correct and
measured; the m1 paper-protocol CV driver had three defects that made a
10-fold run report nothing. Both are fixed and guarded. The production
20-epoch retrain has NOT been run — every CFM number below is PROBE SCALE.**

---

## §1 What was actually wrong

### 1.1 The CFM bond head was training on a target with no chemistry in it

Chased across 7 layers in the preceding session; the two fatal ones:

| # | Layer | Measured effect |
|---|---|---|
| 1 | `_bond_target_for_clean_cloud` synthesised the target from a hardcoded `pt_z = 78` | 1331/1331 targets BOND_NO_BOND on a Ru complex |
| 2 | The OUTER `train_clouds_d` device-move rebuilt each cloud WITHOUT `bonds` | fixing layer 1 alone changed nothing — cost a retrain to discover |

Guards: `test_cfm_bond_target_truth.py` (5 tests).

### 1.2 `decode_ratio` is anti-correlated with correctness

MEASURED: 0.977 while the bond head predicted BOND_NO_BOND for nearly every
pair (loose atoms parse trivially as `C.C.C.C`), 0.219 once the head learned
real bond classes. It counts PARSES, not molecules, and it was m14's headline
metric and the checkpoint gate.

Guards: `test_cfm_chemistry_blind_spot.py` (13 tests), including a test that
asserts `C.C.C.C` and `c1ccccc1CCN` are BOTH RDKit-valid — the separation only
the chemistry report can make.

### 1.3 `--stable-vm` silently DESTROYED the bond loss

It replaced `bond_ce` with `stable_vm_loss(all_logits, zeros)` — a velocity
MSE between the bond class logits and an all-zero tensor — and discarded the
real cross-entropy in the same assignment. A run with the flag on reports a
falling `bond_ce` while the head learns to emit zeros. Now raises.

---

## §2 MEASURED — the class-weight sweep

One cohort, one code revision, 4 epochs each, tmQM real geometry, all six
metals, 192 held-out clouds carrying 8,634 true bonds (positive rate 0.299).

| weight power | weights | prec | rec | **F1** | **PR-AUC** | gap to achievable |
|---|---|---|---|---|---|---|
| 0.0 (unweighted) | all 1.0 | 0.613 | 0.042 | 0.079 | 0.564 | 0.491 |
| **0.5 (new default)** | 0.691 / 1.615 / 5.258 / 20.386 / 1.434 | **0.628** | **0.390** | **0.481** | **0.593** | **0.089** |
| 1.0 (textbook) | 0.287 / 1.567 / 16.602 / 249.629 / 1.235 | 0.313 | 0.987 | 0.475 | 0.467 | 0.063 |

Three findings, all MEASURED:

1. **power 0.5 wins on both axes** — highest argmax F1 AND highest
   threshold-free PR-AUC. It is the default because it measured best, not by
   taste.
2. **The textbook power 1.0 is WORSE than no weighting at ranking** (PR-AUC
   0.467 vs 0.564). It gives BOND_TRIPLE — 0.08% of pairs, 28 examples — a
   250x weight, which saturates recall at 0.987 with precision 0.313.
3. **sqrt weighting nearly closes the calibration gap**: argmax-vs-achievable
   goes 0.491 → 0.089, so at power 0.5 the argmax operating point is close to
   optimal and threshold tuning buys little.

Weight renormalisation is on `sum_c(p_c * w_c) = 1`, which is EXACTLY 1 for the
raw inverse frequency (so it is a no-op at power 1.0) and keeps `bond_ce`'s
scale in `total = loss + ...` at every exponent. Normalising on `w.mean()==1`
instead divides by 53.9 and shrinks `bond_ce` 54x, silently deleting the term.

---

## §3 MEASURED — the m1 protocol-CV driver was reporting nothing

Fold 0 trained 833 s and recorded `test_auc: null`. Three separate defects:

1. **No `test` label was ever written.** The held-out fold was labelled `val`,
   so `build_metal_split_indices` returned an empty test index and the trainer
   printed `n_test=0 test_auc=nan`. The metric extractor's regex could not
   match `nan` either, so the miss became `null` rather than a failure.
2. **No early-stopping set**, so a best epoch would have been selected on the
   rows it was reported on.
3. **`doi_split_cv.py` never passed `smiles=`**, so the outer folds were
   DOI-coherent but NOT complex-coherent — a complex in two articles had its
   articles in different folds. MEASURED with raw-DOI grouping downstream:
   **36 DOI groups on both sides**. After the fix: 1–3 per fold.

`run_m1_paper_cv.py` now writes a three-way split (test = held-out fold,
val = a 10%-of-groups slice of the training rows), asserts SMILES
disjointness (the trainer's actual contract — it never reads DOI), reports
DOI overlap as a diagnostic, and raises on a non-finite AUC instead of
recording `null`.

The `doi_group_ids` union-find was extracted from `assign_doi_folds` so
"which rows must travel together" has ONE definition.

---

## §4 Deliverables — filesystem evidence (AGENTS.md §5.5)

```
$ ls -la metrics/by_metric/cfm_bond_accuracy.json
-rw-r--r-- 1 hugo hugo 4634 Oct  4 14:21 metrics/by_metric/cfm_bond_accuracy.json
$ ls -la metrics/by_metric/decode_ratio.json
-rw-r--r-- 1 hugo hugo 18100 Oct  4 14:11 metrics/by_metric/decode_ratio.json
$ ls -la molmetal/tests/test_cfm_chemistry_blind_spot.py
-rw-r--r-- 1 hugo hugo 9285 Oct  4 14:11 molmetal/tests/test_cfm_chemistry_blind_spot.py
$ ls -la molmetal/tests/test_cfm_bond_f1_gate.py
-rw-r--r-- 1 hugo hugo 5197 Oct  4 14:15 molmetal/tests/test_cfm_bond_f1_gate.py
$ ls -la molmetal/tests/test_cfm_bond_target_truth.py
-rw-r--r-- 1 hugo hugo 8776 Oct  4 14:12 molmetal/tests/test_cfm_bond_target_truth.py
$ ls -la molmetal/tests/test_paper_cohort_doi_cv.py
-rw-r--r-- 1 hugo hugo 10307 Oct  4 14:23 molmetal/tests/test_paper_cohort_doi_cv.py
$ ls -la molmetal/tests/test_wf386a3_atomic_write.py
-rw-r--r-- 1 hugo hugo 14675 Oct  4 14:22 molmetal/tests/test_wf386a3_atomic_write.py
$ ls -la molmetal/scripts/run_m1_paper_cv.py
-rw-r--r-- 1 hugo hugo 17346 Oct  4 14:18 molmetal/scripts/run_m1_paper_cv.py
$ ls -la molmetal/scripts/doi_split_cv.py
-rw-r--r-- 1 hugo hugo 6388 Oct  4 14:17 molmetal/scripts/doi_split_cv.py
$ ls -la molmetal/scripts/wf386a3_atomic_write.py
-rw-r--r-- 1 hugo hugo 15241 Oct  4 14:22 molmetal/scripts/wf386a3_atomic_write.py
$ ls -la molmetal/scripts/pre_flight_check.py
-rw-r--r-- 1 hugo hugo 526672 Oct  4 14:08 molmetal/scripts/pre_flight_check.py
$ ls -la molmetal/scripts/train_cfm_yuelbond_wasserstein.py
-rw-r--r-- 1 hugo hugo 306288 Oct  4 14:14 molmetal/scripts/train_cfm_yuelbond_wasserstein.py
$ ls -la molmetal/data/cytotox.py
-rw-r--r-- 1 hugo hugo 27774 Oct  4 14:16 molmetal/data/cytotox.py
$ ls -la configs/cfm_tmqm_all6.yaml
-rw-r--r-- 1 hugo hugo 6884 Oct  4 14:14 configs/cfm_tmqm_all6.yaml
$ ls -la configs/cfm_full_retrain.yaml
-rw-r--r-- 1 hugo hugo 13775 Oct  4 14:14 configs/cfm_full_retrain.yaml
$ ls -la configs/cfm_per_innovation_ablation_p1.4.yaml
-rw-r--r-- 1 hugo hugo 12961 Oct  4 14:14 configs/cfm_per_innovation_ablation_p1.4.yaml
$ ls -la configs/cfm_real_decode_lift_p1.3.yaml
-rw-r--r-- 1 hugo hugo 10637 Oct  4 14:14 configs/cfm_real_decode_lift_p1.3.yaml```

**Tests — 87 passed, 1 skipped (40.4 s, CPU)** across the eleven touched-area
test modules:

```
test_cfm_bond_f1_gate.py            5   (2 pre-existing + 3 NEW decision-rule)
test_cfm_bond_target_truth.py       5   (bonds-carried-through-device-move)
test_cfm_chemistry_blind_spot.py   13   NEW — the C.C.C.C separation
test_cfm_todo119_decode_ratio.py    —
test_tmqm_wireup.py                 6
test_radius_graph_wiring.py         6
test_wf397a1_decode_ratio_fix.py    4
test_cfm_loss_gradient_liveness.py  —
test_de_novo_loop_wiring.py         3
test_paper_cohort_doi_cv.py         8 + 1 skipped  (3 NEW cohesion guards)
test_wf386a3_atomic_write.py       27   (was 1-2 failing nondeterministically)
```

Two failures were fixed that were **not** caused by this work, per the standing
"fix every failure regardless of origin" directive:

* `test_concurrent_writers_different_paths` asserted thread COMPLETION ORDER
  (`done == [1, 2]`), which is nondeterministic — MEASURED failing ~2 runs in 5
  as `[2, 1]`. It also declared a `threading.Barrier` and never used it, so the
  "different paths do not block" property was never actually exercised. Now
  uses the barrier and asserts `sorted(done) == [1, 2]`.
* `detect_truncation` returned `ends_with_bad_token(':')`, which no test
  grepping for the failure mode could match. Reasons are now named
  (`ends_with_colon(':')`).

**Two more test bugs, same class (filename replaced, assertions not):**

* `test_csv_hash_passes_on_real_metalcytotox` pointed at
  `metalcytotox_5metal_multitask_v4.csv` while asserting the 26,800 floor —
  but that floor belongs to the UNVERSIONED file (MEASURED 26,802 lines).  v4
  is the Wf-261 leak FIX and legitimately holds 26,632 rows, so a correct
  gate FAILS it there.  The test was asserting a failure and calling it a
  pass.  **The production gate is unaffected** — it reads
  `metalcytotox_5metal_multitask.csv` and PASSes at 26,802 >= 26,800.
* `test_scaffold_zero_leak_fails_on_v1` pointed its `v1` variable at the v4
  path, so it demanded FAIL from the file that is the fix (0 leaks, gate
  correctly PASSes).  MEASURED, real v1 leaks
  `[('train','val',29), ('test','train',48), ('test','val',32)]`.

Both now point at the right files; `test_pre_flight_p0_gates.py` is 12/12
stable across 3 runs.

**One defect found in the §5.6-mandated helper itself**: `atomic_write_file`
used `tempfile.mkstemp`, which creates files 0600, and `os.replace` carried
that mode onto the target — so every artifact written through the mandated
helper silently became owner-only in a shared repo (MEASURED: a new
`metrics/*.json` at `-rw-------` beside siblings at `-rw-r--r--`). Now
preserves an existing target's mode and otherwise applies what a normal create
would (`0o666 & ~umask`).

---

## §5 Configuration (yaml-driven — no CLI control surface)

Four new keys, all `flow.*`, added to all four active CFM configs:

```yaml
flow:
  bond_class_weights: true          # inverse-frequency bond CE
  bond_class_weight_power: 0.5      # sqrt softening — MEASURED best
  bond_decision: threshold          # P(bonded) cut, not 5-way argmax
  bond_decision_threshold: 0.5      # a PRIOR; the tuned value was ~0.30,
                                    # selected on the scored pairs, so it was
                                    # NOT adopted — that would bake in a
                                    # selection bias
```

All four resolve to argparse dests; the unresolved-key count is unchanged
(9 / 18 / 23 / 61, all pre-existing metadata sections).

Both decision rules are scored and printed every epoch
(`[cfm-bond] ... [rule=...; alt_rule_F1=...]`), so which rule to use stays a
measurement rather than an assumption.

---

## §6 What is NOT done — honest limitations

1. **The production CFM retrain has not been run.** Every CFM number here is
   4 epochs on a 946-cloud cohort against a 20-epoch production config.
   `bond_exact_match` is **0.000 at every setting** — no complex's full bond
   graph is reproduced — and PR-AUC 0.593 against a 0.299 base rate is a
   modest lift. These numbers establish the DIRECTION of each lever, not an
   achievable score.
2. **The decision threshold is not tuned.** 0.5 is a prior. The probe's
   F1-optimal cut was ~0.30 but was selected on the same pairs it scored, so
   adopting it would bake in a selection bias. Choosing a production value
   needs a held-out split.
3. **m1 CV: fold 0 MEASURED, 9 folds in flight.** The corrected driver
   produced a real held-out number on fold 0:

   | fold | n_train | n_val | n_test | test_auc | wall |
   |---|---|---|---|---|---|
   | 0 | 9,267 (cohort) | 1,098 | 1,152 | **0.8414** | 683 s |

   The run cold-started (verified: `resume_from=auto but no checkpoint found
   ... cold-starting`), so the weights never saw the held-out fold.

   **This is ONE fold and must not be quoted as beating the paper's
   0.810 ± 0.024.** Single-fold spread on this cohort is unknown; the
   protocol's own std is ±0.024. The 10-fold run is in flight.

   Two things the trajectory shows and that matter for interpreting it:
   `best_epoch=10` — the val AUC was still climbing at the LAST epoch
   (0.7232 -> 0.7437), so the 10-epoch budget is under-trained and the number
   is more likely a floor than a ceiling; and val 0.7437 sits well BELOW test
   0.8414, a 0.098 gap between two disjoint evaluation sets, which is larger
   than the paper's whole std and is not yet explained.
4. **The paper's 23 cell lines are not published**, so the top-23-by-row-volume
   reconstruction here is OUR choice. It gives 11,517 rows / 4,502 complexes /
   23 lines against the paper's 11,294 / 4,399 / 23 (within 2%).
5. **DOI test/val overlap of 1–3 DOIs per fold remains** (reported, not
   asserted). It comes from Ru measurements outside the top-23 cohort sharing
   a complex with a held-out cohort row. SMILES disjointness — the trainer's
   actual contract — is hard-asserted and holds.

---

## §7 Cross-references

| Doc | Relationship |
|---|---|
| `metrics/by_metric/decode_ratio.json` | `blind_spot` + `status: MEASURED_BUT_NOT_A_QUALITY_METRIC` + corrected `description` (the old one described one population while the file pools two) |
| `metrics/by_metric/cfm_bond_accuracy.json` | NEW — the bond-quality metric that supersedes it as the gate |
| `molmetal/scripts/pre_flight_check.py` | m14 gate evidence now carries the decode_ratio caveat: it asserts the value is present and non-degenerate, NOT that it indicates bond quality |
| `docs/HONEST_NEGATIVE_CATALOG.md` | eligible for a new entry on the decode_ratio over-claim |
| AGENTS.md §5.2 / §5.5 / §5.6 / §5.7 | honest framing; `ls -la` evidence above; atomic write used and repaired; the `--stable-vm` case is a §5.7-class wire-in |

---

## §8 ADDENDUM — the shipped CFM checkpoint was destroyed by a unit test

Found while taking an inventory of what remains open on the CFM.

### What happened

`--ckpt-dir` defaulted to the repo-global `molmetal/checkpoints` regardless of
`--out-dir`.  `test_cfm_loss_gradient_liveness.py` executes the trainer for ONE
32-sample epoch with `--out-dir <pytest tmp>` and — because the author fixed
only the READ side (`--resume-from none`) — no `--ckpt-dir`.  It therefore
wrote a 1-epoch/32-sample checkpoint over the shipped
`cfm_yuelbond_wasserstein.pt` and `..._seed0.pt`, on every pytest run.

MEASURED, both files on disk carried a pytest path in their saved args:

```
molmetal/checkpoints/cfm_yuelbond_wasserstein.pt
   epoch=0 best_epoch=None decode_ratio=1.0
   n_train=32 n_val=16 epochs=1
   out_dir=/tmp/pytest-of-hugo/pytest-4958/test_no_undeclared_loss_terms0/run
molmetal/checkpoints/cfm_yuelbond_wasserstein_seed0.pt
   out_dir=/tmp/pytest-of-hugo/pytest-5024/test_no_undeclared_loss_terms0/run
```

### The weights are LOST

Neither file is tracked by git.  Every `/tmp` copy of the name is a smoke
(MEASURED: the largest real-cohort copy is `n_train=512, epochs=8`); none is
the Wf-124 P2 50-epoch ship that `metrics/by_metric/cfm_decode_real.json`
cites as `cfm_yuelbond_wasserstein.pt (617 KB, 50-epoch trained)`.

**This is an honest-negative on the m14 provenance**: the artifact that the
m14 = 0.998 figure was measured with no longer exists on disk, and cannot be
restored from anything in the repo or in /tmp.  It also explains the earlier
"49,608-param smoke artifact" reading — that reading was CORRECT about the
file, and wrong only in being taken as a statement about the model design.

### The second half — why a test-only fix would not have held

`--resume-from` defaults to `auto`, which scans `ckpt_dir` and resumes.  So a
smoke writing into the shared directory poisons the NEXT real launch, which
silently continues from the smoke's weights.  That is exactly the
`train_seconds=0.0035` failure the D-MPNN trainer recorded in Wf-A10 — the
CFM side simply never received that fix.

### Fix (mirrors Wf-A10)

1. `--ckpt-dir` default → `None`; unset resolves to `<out_dir>/checkpoints` in
   the trainer, so a run can only ever resume from itself.
2. The three CFM configs that pinned `output.ckpt_dir: molmetal/checkpoints`
   now point at their own run directory (start.sh forwards this value
   verbatim, so a yaml value would otherwise override the new default).
3. `test_cfm_loss_gradient_liveness.py` passes `--ckpt-dir` explicitly —
   belt and braces, so a future default change cannot reintroduce it.
4. NEW `molmetal/tests/test_cfm_checkpoint_isolation.py` (4 guards):
   the default is None, the resolution derives from `--out-dir`, no CFM config
   pins the shared directory, and the liveness probe passes its own.

**Verified end-to-end**: a 32-sample CPU smoke with no `--ckpt-dir` left all
51 shared checkpoints byte-identical (`md5sum -c` all OK) and wrote to
`<out_dir>/checkpoints/`.

### Still open on this thread

* Nothing on disk records that the Wf-124 P2 weights are gone.  Four scripts
  (`bench_cfm_decode_real.py`, `wf188_m14_full_metalcytotox.py`,
  `wf181_m14_real_cfm_metalcytotox.py`, `wf333g1_lambda_cfm_gpu_wrapper.py`)
  and `de_novo_loop.py`'s `--cfm-ckpt` default still point at
  `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` and would silently load a
  pytest artifact.  They need either a real retrained checkpoint or an
  explicit refusal.
* Regenerating a replacement requires the 20-epoch production run, which has
  not been done.

---

## §9 The bond-TYPE axis — does the learned head earn its place?

The open question from §8 was that the CFM had NO bond-type metric, and type is
the only axis on which it could beat classical perception: OpenBabel already
recovers bond EXISTENCE almost perfectly, and its documented weakness is the
metal-donor ORDERS.

### What was built

1. **`bench_tmqm_bond_perception.py` can now score the CFM as a third method**
   (`--baselines cfm --cfm-ckpt ...`), through the SAME `score_bonds` as the
   classical two.  Provenance (epoch / n_train / weight power / decision rule)
   is recorded in the output so a probe cannot be mistaken for a trained model.
2. **`--heavy-atom-scope`** restricts EVERY method to the same non-hydrogen
   pair set.  This is what makes the rows comparable: the CFM trains on
   hydrogen-free clouds, so scoring it over all atoms would measure it out of
   distribution while the baselines kept the hydrogens.
3. **The trainer now reports type metrics every epoch**
   (`[cfm-bond-type] macro_f1_type(pooled)=... exact_type_rate=...`) and carries
   them into `EpochMetrics`.
4. **`molmetal/tests/test_cfm_bond_type_metric.py`** (5 tests) pins the
   trainer's formula to the bench's, and re-guards the bench's own historical
   bug (F1(exist) and F1(type) computed from the same expression).

### The measurement — 900 tmQM complexes, one non-hydrogen scope

| method | success | **F1(exist)** | **F1(type)** | exact_type |
|---|---|---|---|---|
| **CFM** (4-epoch probe) | 1.000 | **0.612** | **0.280** | 0.001 |
| rdkit_stripped | 0.303 | 1.000 | 0.281 | 0.022 |
| **OpenBabel** | 1.000 | **0.977** | **0.505** | 0.030 |

**MEASURED: the CFM beats neither. It loses to OpenBabel on both axes.**
OpenBabel is free, deterministic and instant.

`rdkit_stripped`'s F1(exist) of 1.000 is on the 30.3% of complexes it can
address at all, so that row is NOT comparable to the other two — which is
exactly why `success_rate` is printed beside every row.

### Honest framing

This is the most useful negative result of the pass, and it should not be
softened:

* The gap is EXPECTED (4 epochs of gradient descent vs a mature toolkit) and is
  **not** evidence the head is worthless.
* It IS evidence that **nothing currently shows it is worth anything**. The
  burden of proof is on the learned head, and the number that would discharge
  it is the TYPE column at production scale.
* The trainer's own numbers use a POOLED aggregation while the bench averages
  per complex; the two agree on a single complex (pinned by test) but are not
  interchangeable across a cohort.
* At epoch 1 of a short run the trainer reads type F1 0.267 and climbing
  (0.147 at epoch 0), so the trajectory is upward — but from far below 0.505.

**Correction to the framing above (added after §10).** The comparison is
correct and the harness is reusable, but the CFM row describes a model trained
on 256 clouds — under 1% of the corpus — and, as §10 shows, drawn from a pool
that was itself only a fifth of the data. The conclusion "the learned head
loses to OpenBabel" should be read as "an under-trained head, on an objective
introduced the same day, loses to OpenBabel". The burden of proof is still on
the head, but the current evidence does not discharge it in either direction.

---

## §10 THE ACTUAL BLOCKER — the corpus was silently capped at 20% of itself

Found by taking the user's challenge seriously ("肯定有什么东西搞错了 ... 只要同
协议对齐在大数据集上面训练一遍就行了"). The user was right, and the reason
training on the big dataset had never happened was not a config choice.

### The bug

`molmetal/data/tmqm_clouds.py::iter_tmqm_blocks` is a generator over the
`.xyz.gz` shards. Its per-file read loop ended with a bare `return`:

```python
for path in xyz_gz_paths:
    with gzip.open(str(p), "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                return        # <-- exits the WHOLE generator, not this file
```

In a generator `return` terminates everything, so **the loop never advanced
past `tmQM_X1.xyz.gz`**. `break` is what was meant.

### MEASURED impact

| | before | after |
|---|---|---|
| blocks yielded | **34,970** (X1 only) | **108,541** (all three) |
| six-metal complexes reachable | **6,872** | **33,429** |
| train split (`val_frac 0.2`) | ~5,500 | **26,715** |

73,571 complexes — **67.8% of the corpus** — were invisible to every tmQM
consumer. A caller asking for more than ~6,872 got 6,872 **with no warning**:
`load_tmqm_clouds(n_clouds=20000)` returned 6,872 and printed a line that read
like normal operation.

### Why this masqueraded as "it just needs a training run"

* `DEFAULT_TRAIN_CLOUDS = 256` / `DEFAULT_VAL_CLOUDS = 192`, and **no CFM
  config set `n_train`** — so every run took the default.
* 256 clouds is 0.96% of the trainable corpus.  It was right for the old
  synthetic Pt-click corpus (a few hundred clouds total) and became a silent
  cap when the corpus switched to tmQM.
* **No CFM checkpoint on disk has `n_train > 256`.**  The largest is 256
  (`results/cfm_per_innovation_ablation_p1.4/*`, `epochs=3`).  The recorded
  `decode_ratio` 0.998 cites `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`,
  which by §8 holds a 32-sample pytest artifact.
* So the project has never trained the CFM on the large corpus — and could
  not have, because the data path capped delivery at a fifth of it.

### Fixes

1. `return` → `break` in `iter_tmqm_blocks`.
2. `configs/cfm_tmqm_all6.yaml` now sets `n_train: 26715` / `n_val: 6714` (what
   the loader actually serves for that split) with the reasoning inline, plus
   `checkpoint_interval: 5`.
3. NEW `molmetal/tests/test_tmqm_shard_iteration.py` (3 guards).  **Verified to
   fail 3/3 against the unfixed file** and pass 3/3 with the fix.
4. An AST sweep of `molmetal/` for the same pattern (bare `return` inside a
   per-file loop of a generator) found no other instance.

### Estimated cost of the run that was never possible

MEASURED, 256 clouds ≈ 6–8 s/epoch after warm-up, so 26,715 clouds ≈ 11 min per
epoch; 20 epochs ≈ 3.7 h, call it 4–5 h with evaluation.  `batch_size` and `lr`
are unchanged — scaling the data is the minimal change that tests the
hypothesis, and tuning the optimiser before seeing one large-cohort run would
confound the two.
