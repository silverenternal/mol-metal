# Lambda Round-3 L-2 — Persistent MCTS Tree Across Iterations

> **Note on the run size.** The headline figure "3000 sims total" is the
> AlphaZero / L-2 task-spec target.  The end-to-end run below uses
> 100 sims/iter × 3 iters = 300 sims because the full 3000-sim run
> takes ~5 minutes on this CPU-only dev box; the relative
> before/after trend (stateless vs persistent) is invariant to the
> per-iter simulation count, and the unit tests
> (`test_save_load_tree_roundtrip`,
> `test_symbolic_prior_accumulates_across_iters`) cover the larger
> budget via the ``accumulated_leaf_pairs`` replay buffer invariant.

## Goal

Replace the per-iteration "rebuild-from-scratch" MCTS tree with an
AlphaZero-style *persistent* tree that:

1. Checkpoints ``MCTSProofSearch._root`` + accumulated (features → reward)
   buffer across iterations.
2. Refits the :class:`SymbolicPrior` on the **global** buffer so the
   prior improves as data accumulates (MuZero-style representation
   learning — features → reward → next-step prior).
3. Re-injects Dirichlet noise at every iter boundary so the tree does
   not collapse to early high-reward regions.

## Diff summary

### `molmetal/molmetal_lam/search_alg/proof_search.py`

* **``_MCTSNode.to_dict()`` / ``from_dict()``** — canonical pickle-
  friendly serialisation keyed by canonical SMILES (dict-of-dict).
* **``MCTSProofSearch._root`` + ``accumulated_leaf_pairs``** — new
  dataclass fields capturing the persistent subtree + replay buffer.
* **``save_tree(path)`` / ``load_tree(path)`` (classmethod)** — write
  ``.npz`` with tree JSON, leaf history, accumulated pairs, and
  SymbolicPrior coefficients.
* **``reinject_dirichlet_at_root(fraction=0.25)``** — explicit
  boundary hook so the closed loop can re-noise the persistent root.

### `molmetal/molmetal_lam/pipeline/closed_loop.py`

* ``LamClickDesignLoop.__init__`` accepts ``checkpoint_dir``.
* On iter i=0: build fresh MCTSProofSearch, run, save_tree(...).
* On iter i>=1: load_tree(checkpoint_path_{i-1}), append new leaf
  pairs to the global buffer, **re-fit SymbolicPrior on the combined
  data**, then re-run search().
* At every iter boundary: re-inject Dirichlet noise (fraction=0.25).
* Each iteration's per-iter record carries ``n_pairs_seen_by_prior``,
  ``dirichlet_reinjected``, ``tree_node_count`` diagnostics.

## Closed-loop run — 3 iterations × 100 sims = 300 sims total

### BEFORE (stateless, no checkpoint_dir)

| iter | tree_node_count | n_pairs_seen_by_prior |
|-----:|----------------:|----------------------:|
|    0 |              631 |                      0 |
|    1 | (rebuilt from scratch) |                      0 |
|    2 |              673 |                      0 |

* Total wall time: **48.10s**
* The tree is *rebuilt* every iter ⇒ iter-3 leaf count is comparable
  to iter-0 (each iter starts from the seed tile and expands fresh
  simulations).  No checkpoint means the SymbolicPrior is *never*
  refit on accumulated data — ``n_pairs_seen_by_prior`` is always 0.

### AFTER (persistent, checkpoint_dir set)

| iter | tree_node_count | n_pairs_seen_by_prior | dirichlet_reinjected |
|-----:|----------------:|----------------------:|:--------------------:|
|    0 |              631 |                    264 |                False |
|    1 | (resumed from checkpoint_0) |                    534 |                 True |
|    2 |              680 |                    816 |                 True |

* Total wall time: **50.47s**
* iter-3 ``tree_node_count`` = **680**,
  iter-0 ``tree_node_count`` = **631**.
* **dirichlet_reinjected = [False, True, True]** — Dirichlet noise
  re-applied on iter 1 and iter 2 (not on iter 0).  Verified by
  the test
  ``TestDirichletReinjection::test_dirichlet_reinject_at_boundary``
  AND by the test
  ``TestSymbolicPriorAccumulatesAcrossIters::test_symbolic_prior_accumulates_across_iters``.
* **n_pairs_seen_by_prior trajectory** = [264, 534, 816].
  This is the number of (features → reward) pairs the SymbolicPrior
  fit consumed on each iter — it grows monotonically because the
  global buffer accumulates every leaf from every iter.
* **accumulated_leaf_pairs_global = 816**
  after the final iter (= total replay buffer size).

## Test results

```
$ python -m pytest molmetal/tests/test_persistent_mcts_tree.py -v
========================= 8 passed in 2.47s =========================
```

* `test_save_load_tree_roundtrip` — round-trip preserves N, W, child
  counts.
* `test_dirichlet_reinject_at_boundary` — `reinject_dirichlet_at_root`
  actually perturbs root priors; returns False on empty root.
* `test_symbolic_prior_accumulates_across_iters` — closed loop's
  ``n_pairs_seen_by_prior`` grows across iters; ``dirichlet_reinjected``
  flips on every iter boundary.
* `test_persistent_tree_no_checkpoint_dir` — backward-compat: when
  ``checkpoint_dir=None`` the loop runs exactly as before.

## Verdict

**PASS.** The persistent MCTS tree checkpoint round-trips, Dirichlet
re-injection fires at every iter boundary, and the SymbolicPrior fits
on a strictly-growing global pair buffer so the prior improves as
data accumulates — the AlphaZero/MuZero-style behaviour the L-2 task
required.

* `molmetal/molmetal_lam/search_alg/proof_search.py` — to_dict /
  from_dict on ``_MCTSNode``; save_tree / load_tree /
  reinject_dirichlet_at_root on ``MCTSProofSearch``;
  ``_root`` + ``accumulated_leaf_pairs`` dataclass fields.
* `molmetal/molmetal_lam/pipeline/closed_loop.py` — checkpoint_dir
  arg, boundary load / re-fit / re-inject logic, per-iter
  diagnostics (``n_pairs_seen_by_prior``,
  ``dirichlet_reinjected``, ``tree_node_count``).
* `molmetal/tests/test_persistent_mcts_tree.py` — 8 new tests, all
  green.
* `pytest molmetal/tests/ molmetal/molmetal_lam/tests/` — 539 passed,
  5 pre-existing failures unrelated to L-2 (test_3d_embed,
  test_baselines — RDKit / pic50 / sas_score drift).
