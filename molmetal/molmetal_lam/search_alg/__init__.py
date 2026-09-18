"""search_alg — MCTS-based proof search over the Molecular Lambda Calculus.

This sub-package implements **Layer 8** of the Molecular Lambda Calculus
(MLC), formalized in ``TODO/13_lambda_clickchem/molecular_lambda_calculus.md``
§8 (Drug-Design-as-Proof-Search).

The central class is :class:`MCTSProofSearch`, which performs
**constructive proof search in λ-term space**: starting from an
*initial* closed term, it explores the β-reduction tree (each action =
one β-reduction step = one reaction rule applied to two tiles) and
collects molecules that

    1. inhabit every :class:`TypePredicate` in the target type
       signature (well-typed under the ADMET predicates), and
    2. type-check against the :class:`BindingSite` (inhabit the
       higher-order binding type).

The implementation is deliberately **dependency-light** at import time:
RDKit and torch are imported lazily, so the module is importable in any
environment.  Only the heavy helpers (``_expand``, ``_rollout``) actually
pay the RDKit / torch cost.
"""