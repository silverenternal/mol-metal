"""Gradient Reversal Layer (GRL) for Domain-Adversarial Neural Networks (DANN).

This module owns a verbatim port of the ``ReverseLayerF`` ``torch.autograd.Function``
from the public-domain reference implementation ``DANN_py3`` (fungtion, https://github.com/fungtion/DANN_py3),
adapted from Ganin et al. (2016), "Domain-Adversarial Training of Neural Networks",
Journal of Machine Learning Research 17(59):1-35.

The GRL has no learnable parameters and no effect on the forward pass; during the
backward pass it multiplies the incoming gradient by ``-alpha``.  This is the core
mechanism that allows DANN to learn domain-invariant features: the task head trains
on ``L_task`` while the domain classifier (which sits on top of the GRL) is trained
on ``L_domain`` whose gradient is reversed and pushed back into the shared backbone.

Reference paper
---------------
Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F.,
Marchand, M., & Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks.
Journal of Machine Learning Research, 17(59), 1-35.
http://jmlr.org/papers/v17/15-239.html

Source code provenance
----------------------
Verbatim port of https://github.com/fungtion/DANN_py3/blob/master/functions.py
(MIT-licensed fork of the original fungtion/DANN Py2 reference).  We "own the code"
inside the Mol-Metal repository per TODO-143 §2.2 so the import surface and
licence header live in one place; if the upstream ever disappears or changes
semantics we are insulated.

Adaptation to Mol-Metal D-MPNN
------------------------------
The D-MPNN backbone produces a (B, hidden_dim) pooled embedding.  We pass that
embedding through ``ReverseLayerF.apply(features, alpha)`` *before* the domain
classifier head, so the gradient that flows back into the backbone is reversed.
The alpha schedule is typically:

    alpha = 2.0 / (1.0 + exp(-10.0 * p)) - 1.0   # ramp 0 -> 1 over training

In Mol-Metal we default to a much smaller schedule (constant 0.1 per TODO-143 §4
Day 5-6) because the D-MPNN embedding space is already quite stable; large alpha
destabilises the task head.
"""

from __future__ import annotations

from torch.autograd import Function


class ReverseLayerF(Function):
    """Gradient Reversal Layer (Ganin et al., JMLR 2016).

    Forward pass is the identity.  Backward pass multiplies the incoming
    gradient by ``-alpha``.  This is implemented as a ``torch.autograd.Function``
    (not ``torch.autograd.Function`` + module wrapper) so callers invoke it
    via ``ReverseLayerF.apply(features, alpha)`` exactly as in the original
    fungtion/DANN_py3 reference.

    Args:
        x:     tensor whose forward pass should be identity
        alpha: scalar (Python float or 0-d tensor) controlling the magnitude
               of gradient reversal at backward time.  Set to 0 to disable
               adversarial pressure entirely; set to > 0 to invert the
               gradient on the domain head.  See TODO-143 §2.2 for the
               Mol-Metal default ``alpha=0.1``.

    Returns:
        Same tensor as ``x`` (forward identity).
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Negate the gradient and scale by alpha so the upstream parameters
        # (shared backbone) are pushed *away* from a domain-discriminative
        # direction.  Returning ``None`` for alpha tells autograd the second
        # input has no gradient (alpha is treated as a constant at backward).
        output = grad_output.neg() * ctx.alpha
        return output, None


__all__ = ["ReverseLayerF"]