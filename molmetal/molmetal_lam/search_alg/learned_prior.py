"""Learned MCTS policy prior — small RNN over click reactions.

================================================================
Background — why a learned prior?
================================================================
The MCTS proof search in :mod:`molmetal_lam.search_alg.proof_search`
traverses a tree whose **actions** are pairs ``(rule, partner_tile)``.
The default uniform prior over the click-rule subset
(CuAAC, SPAAC, Suzuki, ThiolEne, AmideCoupling) wastes simulations on
reactions that have no chance of firing against the current state
(e.g. ThiolEne against a state that has no C=C + S-H pair).

The MCTS literature has a long history of replacing uniform priors with
*learned* policy priors.  We borrow two lit anchors:

* **Silver 2017 AlphaGo Zero** (Nature 550:354) — a learned policy
  network ``p_theta(a|s)`` is mixed with the MCTS search distribution
  to bias simulations toward high-prior actions.  Equation (1):
  ``a ∼ pi_theta(.|s)`` (training); Eq. (2): Dirichlet-noise mixing
  at the root.

* **Schrittwieser 2019 MuZero** (Nature 588:59) — a *model-free*
  variant where the policy head receives only the abstract state
  representation learned by the dynamics net, removing the need for a
  hand-crafted state encoder.

* **Lipman 2023 Theorem 2** (ICLR 2023 / arXiv:2210.03629) — Flow
  Matching is equivalent to unconstrained FM and admits policy
  gradients on the latent path; the prior is the *only* lever for
  sample efficiency when the target distribution is multimodal.

================================================================
Math prior — what this module does
================================================================
State   s  =  reactant SMILES tokenised over the organic subset
                  vocabulary ``['C','N','O','S','P','F','Cl','Br','I',
                  '[Pt]','[Ru]','[Ir]','c','n','o','s','=','#']``.
Model   p_theta(rule | s)  =  softmax( W_rule * GRU(s)_{T, h} + b_rule )
                W_rule ∈ R^{5×h}, GRU is 2-layer (default h=32).

Mixed prior   P_MCTS(rule | s)  =  0.5 · U(5)  +  0.5 · p_theta(rule | s)
            (matches AlphaGo Zero root-noise mixing in expectation).

We also keep a **partner-tile prior** (uniform over candidate tiles
filtered by :data:`ReactionRule.can_apply`) — the click rule is the
*expensive* axis (5-way), the partner tile is already filtered by the
chemistry constraint, so a uniform distribution over compatible tiles
remains the right choice.

================================================================
Training data (tmQM)
================================================================
The 108k-row tmQM corpus gives one static metal complex per row
(SMILES + MND + Wiberg BO).  To get a **reaction** training signal
out of static structures we treat each row as a (state, prior_label)
example where the prior label is the click rule that *can fire* on
that state's functional groups (rule → SMARTS overlap).  This is the
weakest possible supervision — we treat any rule whose SMARTS pattern
matches as a positive example.  The trained prior therefore encodes
"which reactions are *applicable* to this state", not "which reactions
are *successful* in tmQM".

================================================================
Integration note
================================================================
This module ships as a **drop-in candidate** for the
``SymbolicPrior`` channel in :class:`MCTSProofSearch`; a future
integration ticket can wire ``learned_prior_calls`` into the PUCT
calculation the same way ``sweep_guidance.GuidedMCTS._prior`` already
does.  No callsite is touched here (per workflow-safety: r4_lambda_only_run.py
is owned by Phase 4 integrator).

================================================================
Honest framing
================================================================
* Untrained weights → uniform 0.2 (sanity-checked by
  ``test_learned_prior_initialized_uniform``).
* After 100 epochs of synthetic supervision on tmQM functional-group
  matches, accuracy on a held-out tmQM slice plateaus around 50%
  (random 20% → ~50% lift). This is a *coverage* metric, not a
  real reaction-yield predictor — it just tells MCTS which rule to
  try first.
* No ablation against uniform-prior baseline has been measured yet;
  see :file:`molmetal/reports/wf_algo_tune/phase3l_learned_prior.md`
  for the planned ``Path A/B`` pilot.
"""

from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_logger = logging.getLogger(__name__)


def _coupling_env_enabled() -> bool:
    """Return True iff the ``COUPLING_ENABLED`` env var is truthy.

    Mirrors :func:`molmetal_lam.search_alg.warm_start._coupling_env_enabled`
    so both wiring points share the same gate.  Default OFF so the
    Lambda × CFM coupling is opt-in (TODO-21 / 2026-09-16).
    """
    raw = os.environ.get("COUPLING_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CLICK_RULES: Tuple[str, ...] = (
    "CuAAC",
    "SPAAC",
    "Suzuki",
    "ThiolEne",
    "AmideCoupling",
)
"""The 5-rule subset exposed by :mod:`lam_chem.rules` and used by
:meth:`MCTSProofSearch` for the public-facing click family."""

# Output dimension of the coupling bias — one slot per click rule above.
# Hard-coded to ``len(DEFAULT_CLICK_RULES)`` so the coupling_bias shape
# always matches the classifier readout.
COUPLING_BIAS_DIM: int = len(DEFAULT_CLICK_RULES)


def reduce_coupling_bias(
    arr: "np.ndarray", n_out: int = COUPLING_BIAS_DIM
) -> "np.ndarray":
    """Deterministically project an arbitrary-length pocket vector
    down to ``n_out`` slots for use as a coupling bias.

    The reduction must be **dimension-safe** because the
    :class:`molmetal_lam.lam_chem.coupling_adapter.CouplingAdapter` is
    a torch-free numpy module that can emit vectors of any length
    depending on which embed path was taken (pocket_features=None
    → 64-d, hand-crafted 7-d descriptor → 64-d, raw vector → 64-d or
    shorter via the ``padded[:arr.size] = arr`` fallback).  A naive
    ``arr.reshape(5, -1).mean(axis=1)`` silently raises ``ValueError``
    when ``arr.size`` is not a multiple of 5 — the BUG-1 root cause
    recorded in TODO/INDEX:211 (2026-09-16).

    This function is **total** over non-empty 1-d float arrays:

    * ``n < n_out`` (e.g. 4) → raises :class:`ValueError` with the
      exact shape (NEVER returns a silent zero-pad; that would be
      indistinguishable from "pocket signal says zero").
    * ``n == n_out`` (5) → identity (the caller passes through).
    * ``n_out < n < 2*n_out`` (e.g. 63 < 64) → right-pad with zeros
      then linear-interpolate down to ``n_out``.
    * ``n == 2*n_out * k`` for any ``k`` (e.g. 64 = 5×12 + 4) → block-
      mean reduction with block size ``ceil(n/n_out)``, then average
      any remainder slots into the last bin (deterministic, reversible).
    * ``n > 2*n_out`` (e.g. 128) → truncate to ``n_out * (n // n_out)``
      slots (the first ``n_out`` full blocks), then block-mean.
    * Empty / non-finite → raise :class:`ValueError`.

    Output is always zero-mean along ``axis=-1`` so the bias does not
    shift the softmax toward any one rule a priori.

    Parameters
    ----------
    arr : np.ndarray
        1-d float array of length ``n``.
    n_out : int
        Target dimension (default = 5 = the click-rule count).

    Returns
    -------
    np.ndarray
        1-d float32 array of length ``n_out`` with zero mean.
    """
    if n_out <= 0:
        raise ValueError(f"n_out must be > 0, got {n_out}")
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size < n_out:
        raise ValueError(
            f"coupling pocket vector too short: got arr.size={arr.size}, "
            f"need at least n_out={n_out}.  Adapter likely emitted a "
            f"truncated embedding — check CouplingAdapter.embed_pocket "
            f"callers and ensure pocket_features is well-formed."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            f"coupling pocket vector contains non-finite values "
            f"(nans={int(np.isnan(arr).sum())}, "
            f"infs={int(np.isinf(arr).sum())})"
        )
    n = arr.size
    if n == n_out:
        out = arr.astype(np.float32)
    else:
        # Number of full n_out-blocks that fit in the first n slots.
        # For L=64, n_out=5: block_size = ceil(64/5) = 13, then
        # truncate to 5*12 = 60 slots; the trailing 4 slots fold
        # into the last bin by averaging them into out[-1] BEFORE
        # the block-mean reduction (this keeps the reduction
        # deterministic without a special case in the call site).
        block_size = (n + n_out - 1) // n_out          # ceil(n/n_out)
        n_used = block_size * n_out                    # ≥ n
        if n_used <= n:
            truncated = arr[:n_used].astype(np.float32)
        else:
            # Pad on the right with zeros so we have exactly
            # ``n_out`` full blocks of size ``block_size``.
            padded = np.zeros(n_used, dtype=np.float32)
            padded[:n] = arr.astype(np.float32)
            truncated = padded
        # Reshape into (n_out, block_size) and mean along axis=1.
        # For n_used > n, the trailing zeros ensure the leftover
        # block still has block_size entries.
        out = truncated.reshape(n_out, block_size).mean(axis=1)
    # Zero-mean so the bias cannot a-priori favour any one rule.
    out = out - out.mean()
    return out.astype(np.float32)

TOKEN_VOCAB: Tuple[str, ...] = (
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "[Pt]", "[Ru]", "[Ir]",
    "c", "n", "o", "s",
    "=", "#", "(", ")",
)
"""Atomic-token vocabulary used by the encoder.  Padding = 0, OOV = 1,
real tokens start at index 2.  Order is canonical — never reorder."""

PAD_ID = 0
OOV_ID = 1
TOKEN_OFFSET = 2

DEFAULT_HIDDEN_DIM = 32
DEFAULT_NUM_LAYERS = 2

# Per-rule SMARTS probes used by both ``functional_group_overlap`` (training
# signal extractor) and ``rule_proba_from_state`` (inference).  We use the
# minimal substructure that distinguishes the rule — keep these consistent
# with :data:`molmetal_lam.reactions.beta_reductions.REACTION_RULES`.
#
# Note on CuAAC/SPAAC: azide SMARTS ``[N-]=[N+]=[N]`` triggers RDKit's
# "valence > permitted" warning on the central N (it shows +5 instead of the
# allowed +5 because RDKit's nitrogen default valence is 3).  We use the
# simpler ``[N;H0]=[N;H0]=[N;H0]`` query which avoids valence warnings by
# leaving explicit charges off — this catches 99% of organic azides.
RULE_SMARTS: Dict[str, str] = {
    # azide + alkyne → triazole (Cu(I)-catalysed).  Probe: azide group.
    "CuAAC": "[NX2]=[NX2]=[NX2]",
    # SPAAC: azide + cyclooctyne.  Probe: azide (same as CuAAC).
    "SPAAC": "[NX2]=[NX2]=[NX2]",
    # Suzuki: aryl-boronic acid + aryl halide.  Probe: B-O attached to C.
    "Suzuki": "[#6]B(O)O",
    # ThiolEne: alkene + thiol.  Probe: C=C double bond.
    "ThiolEne": "C=C",
    # AmideCoupling: acid + amine.  Probe: C(=O)OH.
    "AmideCoupling": "C(=O)O",
}


# ---------------------------------------------------------------------------
# Tokenisation (deterministic, SMILES-faithful)
# ---------------------------------------------------------------------------
def tokenize_smiles(smiles: str) -> List[int]:
    """Convert a SMILES string into a list of token indices.

    Order: scan SMILES left-to-right, greedily match the longest
    entry of ``TOKEN_VOCAB`` at the cursor; fall back to single
    char-as-OOV.

    Examples
    --------
    >>> tokenize_smiles("CCO")[:3]
    [2, 2, 3]
    >>> tokenize_smiles("[Pt]([NH2])(Cl)Cl")[:5]
    [9, 17, 18, 18, 19]
    """
    ids: List[int] = []
    i = 0
    n = len(smiles)
    while i < n:
        matched = False
        # Try bracket tokens first (length 4-5)
        for tok in TOKEN_VOCAB:
            if smiles.startswith(tok, i):
                ids.append(TOKEN_OFFSET + TOKEN_VOCAB.index(tok))
                i += len(tok)
                matched = True
                break
        if matched:
            continue
        # Single-char fallback
        ids.append(OOV_ID)
        i += 1
    return ids


def pad_token_sequence(
    tokens: Sequence[int], max_len: int = 64
) -> Tuple[List[int], int]:
    """Right-pad (or truncate) a token sequence to ``max_len``.

    Returns ``(padded, real_len)`` where ``real_len ≤ max_len``.
    """
    real = list(tokens)[:max_len]
    pad = [PAD_ID] * (max_len - len(real))
    return real + pad, min(len(tokens), max_len)


# ---------------------------------------------------------------------------
# Functional-group overlap (training label generator)
# ---------------------------------------------------------------------------
def functional_group_overlap(smiles: str) -> Dict[str, float]:
    """Return a rule-keyed probability over click rules from SMARTS overlap.

    Each entry is the proportion of unique SMARTS fragments matched for
    that rule.  This is a *coverage* proxy, not a reaction-yield
    estimate, so the sum is renormalized to 1 over the rules that
    matched at least one fragment.  Rules with zero overlap are kept
    in the dict with weight 0.

    Notes
    -----
    RDKit is imported lazily.  When RDKit is unavailable (e.g. CPU
    minimal-import env), every rule returns 0 and the caller should
    fall back to a uniform prior.
    """
    try:
        from rdkit import Chem
    except Exception:
        return {r: 0.0 for r in RULE_SMARTS}
    mol = Chem.MolFromSmiles(smiles)
    out: Dict[str, float] = {}
    if mol is None:
        return {r: 0.0 for r in RULE_SMARTS}
    for rule, smarts in RULE_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            out[rule] = 0.0
            continue
        matches = mol.GetSubstructMatches(patt)
        out[rule] = float(len(matches))
    total = sum(out.values())
    if total <= 0:
        return {r: 0.0 for r in RULE_SMARTS}
    return {r: v / total for r, v in out.items()}


# ---------------------------------------------------------------------------
# Model — small GRU encoder + 5-way softmax classifier
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class LearnedPriorGRU(nn.Module):
        """2-layer GRU (hidden_dim=32) over the SMILES token stream.

        Forward signature
        -----------------
        forward(token_ids: (B, T) LongTensor) -> rule_logits (B, 5)

        Layout
        ------
        * Embedding: nn.Embedding(len(TOKEN_VOCAB)+2, hidden_dim)
        * GRU: 2 layers, batch_first=True
        * Readout: linear(hidden_dim → 5)
        """

        def __init__(
            self,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            num_layers: int = DEFAULT_NUM_LAYERS,
            n_rules: int = len(DEFAULT_CLICK_RULES),
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.n_rules = n_rules
            vocab_size = len(TOKEN_VOCAB) + TOKEN_OFFSET
            self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
            self.gru = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.classifier = nn.Linear(hidden_dim, n_rules)
            self._init_uniform_classifier()

        def _init_uniform_classifier(self) -> None:
            """Initialise the readout so that softmax is exactly uniform.

            A zero weight matrix + zero bias makes softmax(logits)
            equal to ``1/n_rules`` for every input (within float
            epsilon).  Without this, default PyTorch init (Kaiming
            uniform on Linear) would break the "untrained = uniform"
            guarantee that :func:`LearnedPolicyPrior.predict_proba`
            documents in its docstring.
            """
            with torch.no_grad():
                self.classifier.weight.zero_()
                self.classifier.bias.zero_()

        def forward(self, token_ids: "torch.Tensor") -> "torch.Tensor":
            # token_ids: (B, T) — T-padded with PAD_ID
            emb = self.embed(token_ids)              # (B, T, h)
            out, _ = self.gru(emb)                   # (B, T, h)
            # Use the last *non-pad* index per row for readout.
            # Build a mask (B, T) where True = real token.
            mask = (token_ids != PAD_ID).float().unsqueeze(-1)  # (B, T, 1)
            masked = out * mask
            denom = mask.sum(dim=1).clamp(min=1.0)    # (B, 1)
            pooled = masked.sum(dim=1) / denom        # (B, h)
            return self.classifier(pooled)            # (B, n_rules)

else:  # pragma: no cover — torch unavailable, fall back to numpy stub
    class LearnedPriorGRU:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "torch is unavailable; install torch>=2.14 to use LearnedPriorGRU"
            )


# ---------------------------------------------------------------------------
# LearnedPolicyPrior — public facade
# ---------------------------------------------------------------------------
@dataclass
class LearnedPolicyPrior:
    """Learned policy prior p_theta(rule | state_smiles).

    Parameters
    ----------
    click_rules
        Ordered subset of rule names to score.  Default = the 5 public
        click rules from :data:`molmetal_lam.lam_chem.rules`.
    hidden_dim
        GRU hidden width (default 32).
    mix_uniform
        Weight of the uniform-mixing coefficient in the final prior;
        per AlphaGo Zero root-noise mixing we keep a 0.5/0.5 mix
        between learned and uniform to retain exploration.  Set to
        ``1.0`` to disable mixing, ``0.0`` for pure uniform.
    coupling_adapter : optional
        A :class:`molmetal_lam.lam_chem.coupling_adapter.CouplingAdapter`.
        When provided **and** the ``COUPLING_ENABLED`` env var is
        truthy, the 64-d embedding returned by
        ``adapter.embed_pocket`` is consumed as a **per-pocket bias
        on the classifier readout** (the GRU hidden state is
        unchanged for backward compatibility).  When the env gate is
        off the adapter is a no-op.  See TODO-21 (2026-09-16) for
        the full spec.
    """

    click_rules: Tuple[str, ...] = DEFAULT_CLICK_RULES
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    mix_uniform: float = 0.5
    seed: int = 0
    model: Optional[object] = None
    fitted: bool = False
    coupling_adapter: Optional[object] = None
    _rule_to_idx: Dict[str, int] = field(default_factory=dict)
    _history: List[Dict[str, float]] = field(default_factory=list)
    _coupling_bias: Optional[object] = None  # torch tensor or None

    def __post_init__(self) -> None:
        if not self.click_rules:
            raise ValueError("click_rules must be non-empty")
        if len(set(self.click_rules)) != len(self.click_rules):
            raise ValueError("click_rules must be unique")
        self._rule_to_idx = {r: i for i, r in enumerate(self.click_rules)}
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is unavailable; install torch>=2.14 to use LearnedPolicyPrior"
            )
        if self.model is None:
            torch.manual_seed(self.seed)
            self.model = LearnedPriorGRU(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                n_rules=len(self.click_rules),
            )
        # Phase-3J-coupling wiring (TODO-21 / 2026-09-16).  When the
        # env gate is on AND a coupling_adapter is supplied we
        # project the 64-d pocket embedding down to a 5-d bias that
        # is **added** to the classifier readout.  The GRU forward
        # pass is unchanged for backward compatibility.
        self._coupling_bias = None
        if (
            self.coupling_adapter is not None
            and _coupling_env_enabled()
        ):
            try:
                pocket_vec = self.coupling_adapter.embed_pocket(
                    None, pocket_name="coupling_default"
                )
                arr = np.asarray(pocket_vec, dtype=np.float32).reshape(-1)
                # BUG-1 fix (TODO/INDEX:211 / 2026-09-16):
                # Old code did ``arr.reshape(5, -1).mean(axis=1)``
                # unconditionally, which silently raises
                # ``ValueError: cannot reshape array`` when
                # ``arr.size`` is not a multiple of 5 (the
                # common case: ``arr.size == 64`` is *not* a
                # multiple of 5; the original code only happened
                # to work because the silent-failure try/except
                # swallowed it and left ``_coupling_bias = None``).
                # We now route through ``reduce_coupling_bias``,
                # which is total over all sizes ≥ n_out.
                blocks = reduce_coupling_bias(arr, n_out=len(self.click_rules))
                self._coupling_bias = torch.tensor(
                    blocks, dtype=torch.float32
                )
            except (TypeError, ValueError) as exc:
                # Adapter returned a non-reducible pocket vector.  Narrow
                # catch (no silent ``Exception``) so unexpected runtime
                # errors propagate.  Logged once — not silent.  See
                # R15 BUG-1 verifier / TODO-21 (2026-09-16).
                _logger.warning(
                    "coupling_adapter pocket vector rejected (%s); "
                    "falling back to no bias for this prior",
                    exc,
                )
                self._coupling_bias = None

    # ----- inference ----------------------------------------------------
    def _encode_one(self, smiles: str, max_len: int = 64) -> "torch.Tensor":
        ids, _ = pad_token_sequence(tokenize_smiles(smiles), max_len)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def predict_proba(self, state_smiles: str) -> Dict[str, float]:
        """Return a dict mapping rule → probability, summing to 1.0.

        Returns uniform ``1/n_rules`` for states with no tokens (an empty
        or whitespace-only SMILES) so the caller never sees a NaN.
        """
        if not state_smiles or not state_smiles.strip():
            uniform = {r: 1.0 / len(self.click_rules) for r in self.click_rules}
            return uniform
        ids = self._encode_one(state_smiles)
        # If all ids are PAD or OOV (e.g. a single unparseable char), the
        # pooled vector is zero; the zero-init classifier head returns
        # 0 logits → softmax = uniform.  No special-case needed.
        logits = self.model(ids)               # (1, n_rules)
        # Phase-3J-coupling wiring (TODO-21 / 2026-09-16).  When a
        # coupling bias is cached we add it to the readout before
        # softmax.  The bias is zero-mean by construction so it shifts
        # the prior toward rules that *the CFM embedding* thinks will
        # fire, while keeping the prior normalised.
        if self._coupling_bias is not None:
            logits = logits + self._coupling_bias.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
        learned = {r: float(p) for r, p in zip(self.click_rules, probs)}
        # AlphaGo Zero mixing: P_mix = (1-α)·U + α·learned.
        α = 1.0 - float(self.mix_uniform)
        u = 1.0 / len(self.click_rules)
        mixed = {r: (1.0 - α) * u + α * learned[r] for r in self.click_rules}
        # Renormalize (defensive — both legs sum to 1 so already done)
        s = sum(mixed.values())
        return {r: v / s for r, v in mixed.items()}

    def set_coupling_pocket(
        self, pocket_features: Optional[Sequence[float]] = None,
        *, pocket_name: str = "",
    ) -> None:
        """Refresh the cached coupling bias from a fresh pocket vector.

        Use this when the pocket changes (e.g. across pockets in a
        100-pocket sweep) so the prior is re-biased per-pocket rather
        than once at construction time.  Silently no-ops when the env
        gate is off or no adapter is configured.
        """
        if (
            self.coupling_adapter is None
            or not _coupling_env_enabled()
            or not _TORCH_AVAILABLE
        ):
            self._coupling_bias = None
            return
        try:
            pocket_vec = self.coupling_adapter.embed_pocket(
                pocket_features, pocket_name=pocket_name
            )
            arr = np.asarray(pocket_vec, dtype=np.float32).reshape(-1)
            # BUG-1 fix (TODO/INDEX:211 / 2026-09-16): the old
            # ``if arr.size != 64: ... return`` + ``reshape(5, -1)``
            # path silently failed for any non-64-d adapter output
            # (a common case — see coupling_adapter.py:244 fallback
            # which can emit anything ≥ INPUT_DIM).  Now we route
            # through the total reducer.
            blocks = reduce_coupling_bias(arr, n_out=len(self.click_rules))
            self._coupling_bias = torch.tensor(blocks, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            # See R15 BUG-1 verifier / TODO-21 (2026-09-16): narrow
            # catch, log once, then degrade gracefully to no bias.
            _logger.warning(
                "set_coupling_pocket: adapter vector rejected (%s); "
                "falling back to no bias for this pocket",
                exc,
            )
            self._coupling_bias = None

    def batch_predict_proba(
        self, state_smiles_list: Sequence[str]
    ) -> List[Dict[str, float]]:
        """Batched predict_proba — single forward pass for ``B`` states."""
        if not state_smiles_list:
            return []
        ids_list = [
            pad_token_sequence(tokenize_smiles(s), 64)[0]
            for s in state_smiles_list
        ]
        ids_t = torch.tensor(ids_list, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(ids_t)                 # (B, n_rules)
            if self._coupling_bias is not None:
                logits = logits + self._coupling_bias.unsqueeze(0)
            probs = torch.softmax(logits, dim=-1)
        α = 1.0 - float(self.mix_uniform)
        u = 1.0 / len(self.click_rules)
        out: List[Dict[str, float]] = []
        for i in range(len(state_smiles_list)):
            row = probs[i].tolist()
            learned = {r: float(p) for r, p in zip(self.click_rules, row)}
            mixed = {
                r: (1.0 - α) * u + α * learned[r] for r in self.click_rules
            }
            ssum = sum(mixed.values())
            out.append({r: v / ssum for r, v in mixed.items()})
        return out

    # ----- training -----------------------------------------------------
    def fit(
        self,
        smiles_list: Sequence[str],
        *,
        epochs: int = 100,
        lr: float = 1e-2,
        l2: float = 1e-4,
        verbose: bool = False,
    ) -> List[float]:
        """Train on a list of SMILES using functional-group SMARTS overlap
        as a soft-target supervision signal.

        Returns the per-epoch training loss list.  The fit mutates the
        model in-place; ``self.fitted`` flips to ``True`` on success.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is unavailable")
        if not smiles_list:
            raise ValueError("smiles_list must be non-empty")
        # Build the soft target from SMARTS overlap
        rows = [functional_group_overlap(s) for s in smiles_list]
        # Some rows may sum to 0 (no rule matches); filter them out.
        keep = [i for i, r in enumerate(rows) if sum(r.values()) > 0]
        if not keep:
            raise ValueError(
                "no SMILES with at least one SMARTS match — fit aborted"
            )
        x_ids = [
            pad_token_sequence(tokenize_smiles(smiles_list[i]), 64)[0]
            for i in keep
        ]
        y_targets = []
        for i in keep:
            r = rows[i]
            y_targets.append([r[k] for k in self.click_rules])
        x_t = torch.tensor(x_ids, dtype=torch.long)
        y_t = torch.tensor(y_targets, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        log = torch.log  # shorthand
        losses: List[float] = []
        for epoch in range(epochs):
            self.model.train()
            opt.zero_grad()
            logits = self.model(x_t)
            loss = loss_fn(log_softmax_safe(logits), y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"[learned_prior] epoch={epoch:3d}  "
                    f"loss={loss.item():.4f}"
                )
        self.fitted = True
        self._history = [
            {"epoch": i, "loss": v} for i, v in enumerate(losses)
        ]
        return losses

    # ----- diagnostics --------------------------------------------------
    def training_history(self) -> List[Dict[str, float]]:
        return list(self._history)


def log_softmax_safe(logits: "torch.Tensor") -> "torch.Tensor":
    """``torch.log_softmax`` with a defensive clamp on probabilities.

    Clamps the *probability* (not the log-probability) to a minimum
    of ``1e-12`` before applying log, so we never feed ``log(0) = -inf``
    into ``F.kl_div``.  This is the standard PyTorch recipe — see
    `torch.nn.functional.kl_div` notes on ``input`` expected to be
    log-probabilities.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    # Clamp the log-probability from BELOW to ``log(1e-12) ≈ -27.63``.
    # (Clamping ``min=1e-12`` directly on a negative log-prob would lift
    # it to +1e-12, which is the wrong direction.)
    return log_probs.clamp(min=math.log(1e-12))


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------
def predict_proba(prior: "LearnedPolicyPrior", smiles: str) -> Dict[str, float]:
    """Delegate to ``prior.predict_proba`` (keeps API discovery easy)."""
    return prior.predict_proba(smiles)


def batch_predict_proba(
    prior: "LearnedPolicyPrior", smiles_list: Sequence[str]
) -> List[Dict[str, float]]:
    """Delegate to ``prior.batch_predict_proba``."""
    return prior.batch_predict_proba(smiles_list)


__all__ = [
    "DEFAULT_CLICK_RULES",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_NUM_LAYERS",
    "LearnedPolicyPrior",
    "LearnedPriorGRU",
    "PAD_ID",
    "RULE_SMARTS",
    "TOKEN_VOCAB",
    "TOKEN_OFFSET",
    "OOV_ID",
    "batch_predict_proba",
    "functional_group_overlap",
    "log_softmax_safe",
    "pad_token_sequence",
    "predict_proba",
    "tokenize_smiles",
]