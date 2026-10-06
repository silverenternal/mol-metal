"""Training-loop helpers extracted from `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`.

The split (Wf-??? monolith audit): small, pure helpers (no module-global
state, no `self.*` mutation) live here; stateful orchestrators (model,
train_one_*, main) stay in the trainer module.

This package is imported by the trainer as a backwards-compat surface:
existing callers that read `from molmetal.scripts.train_dmpnn_multitask_wasserstein
import _safe_classification_metrics, collate_multimetal` continue to work
because the trainer re-exports them.
"""

from __future__ import annotations
