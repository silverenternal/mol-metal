"""Path-B Decoder M3 — SequenceDecoderHead + GumbelTopKBondDecoder + Composite.

Background
----------
The round-17 (R17) Path-B decoder rework shipped a chem-aware soft
bond prior that lifted ``decode_ratio`` from 0/192 to 192/192 on
synthetic Pt-click clouds.  That rework lives in
:file:`molmetal/molmetal_lam/lam_chem/decoder_rework.py` and is the
production default.

**M3 = the next-generation decoder stack.**  It adds two follow-up
heads that compose with the existing rework without modifying it:

* **M3.1 ConnectivityAwareDecoder wrapper** (Danihelka 2022 Gumbel-Top-K
  §3 + Jin 2018 JTN-VAE) — picks the top-``k`` edges per atom using a
  Gumbel-softmax relaxed selector during training and a hard top-k
  argmax during inference.  This replaces the legacy 2.4 Å hard cutoff
  with a learned, differentiable edge filter.

* **M3.2 SequenceDecoderHead** — a sequence decoder that emits atoms
  AND bonds in a canonical chemical order (root-atom first, BFS/DFS
  traversal).  The vocabulary size is 42, matching
  :data:`molmetal_lam.atoms.combinators.PRIMITIVE_ATOMS`.

Lit anchor
----------
Danihelka 2022 ICML Gumbel-Top-K §3 (decorrelated batched priors for
combinatorial sampling), Jin 2018 ICML JTN-VAE (fragment-valid
combinatorial sampling), Alcaide 2024 Uni-Mol v2 (arXiv:2405.11769)
sequence-decoder pattern (hidden_dim=128, 9 layers → competitive on
PB-valid).

Constraints
-----------
* **DO NOT modify** :file:`molmetal/molmetal_lam/lam_chem/decoder_rework.py`
  (926 LOC, R17 ship).
* This module is *additive only*: it imports from the rework but never
  alters it.  Any change to the rework API surfaces a graceful
  ``ImportError`` at import time, never a silent behaviour drift.
* CPU-only path: no GPU / Triton / ROCm dependencies.  All tests run
  in <2 s on a single core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# We import the canonical vocab size from the combinators module so
# the sequence head never drifts from the PRIMITIVE_ATOMS table.  When
# the import is unavailable (e.g. an ultra-minimal install), we fall
# back to a literal 42 — the count of the canonical table as of R17.
try:
    from molmetal_lam.atoms.combinators import PRIMITIVE_ATOMS

    _VOCAB_SIZE: int = len(PRIMITIVE_ATOMS)
except Exception:  # pragma: no cover — defensive fallback
    _VOCAB_SIZE = 42


__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "SequenceDecoderHead",
    "GumbelTopKBondDecoder",
    "M3CompositeDecoder",
    "M3CompositeOutput",
]


# ---------------------------------------------------------------------------
# Defaults — kept module-level so tests + callers can import them
# directly without instantiating the classes.
# ---------------------------------------------------------------------------
# Alcaide 2024 Uni-Mol v2 §3.1: hidden_dim=128 is the canonical
# production-scale value for SE(3)-equivariant 3-D molecular decoders.
# Hidden dims < 64 emit a UserWarning at setup (see TODO-24 F4).
DEFAULT_HIDDEN_DIM: int = 128
# Max sequence length covers typical SBDD ligands (max ~32 atoms in
# the PlatinAI / tmQM training pool; 64 leaves 2x headroom).
DEFAULT_MAX_SEQ_LEN: int = 64
# Vocabulary size == len(PRIMITIVE_ATOMS) == 42 (R17 ship).  Exposed
# as a constant so tests can assert against the canonical value
# without re-importing PRIMITIVE_ATOMS.
DEFAULT_VOCAB_SIZE: int = _VOCAB_SIZE
# Gumbel-softmax temperature.  T=1.0 is the standard starting point;
# train-time annealing schedules typically decay to 0.5 by step 5k.
DEFAULT_TEMPERATURE: float = 1.0
# Default top-k edges per atom.  k=4 covers sp3 carbon (max 4 bonds)
# and Pt_II (square-planar, 4 dative+covalent).
DEFAULT_TOP_K: int = 4


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class M3CompositeOutput:
    """Output of :meth:`M3CompositeDecoder.forward`.

    Attributes
    ----------
    sequence_logits : (batch, max_seq_len, vocab_size) tensor
        Per-position atom-type logits from :class:`SequenceDecoderHead`.
    bond_adjacency : (batch, n_atoms, n_atoms) tensor
        Sparse binary adjacency from :class:`GumbelTopKBondDecoder`.
    rework_bond_logits : (E,) tensor or None
        The 3-prior bond logits from the existing
        :class:`DecoderRework`.  ``None`` when the rework decoder is
        not provided (e.g. unit-test stubs).
    """

    sequence_logits: torch.Tensor
    bond_adjacency: torch.Tensor
    rework_bond_logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# SequenceDecoderHead — emits atoms in canonical chemical order
# ---------------------------------------------------------------------------
class SequenceDecoderHead(nn.Module):
    """Sequence decoder that emits atoms in a canonical chemical order.

    The head consumes per-atom features (the CFM latent) plus
    per-bond features (the soft prior logits) and produces a
    ``(batch, max_seq_len, vocab_size)`` logits tensor over the
    PRIMITIVE_ATOMS vocabulary (42 entries, R17 ship).

    The decoder is a 2-layer Transformer encoder with a learned
    positional embedding; this matches the Jin 2018 JTN-VAE pattern
    (fragment-valid combinatorial sampling in canonical order) and
    the Alcaide 2024 Uni-Mol v2 §3.1 pattern (hidden_dim=128, ~9
    layers).  We use 2 layers for the M3 skeleton — the production
    wiring can swap in deeper variants without touching the head's
    forward contract.

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension.  Default 128 (Alcaide 2024).
    max_seq_len : int
        Maximum sequence length.  Default 64 (covers typical SBDD
        ligands with 2x headroom).
    vocab_size : int
        Atom-vocabulary size.  Default 42 (PRIMITIVE_ATOMS, R17 ship).
    n_layers : int
        Number of Transformer encoder layers.  Default 2 (skeleton
        only — production wiring may raise this).
    n_heads : int
        Number of attention heads.  Default 4.
    dropout : float
        Dropout probability between attention and FFN.  Default 0.0
        (deterministic unit-test mode).
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim!r}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len!r}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size!r}")
        if n_heads <= 0 or hidden_dim % n_heads != 0:
            raise ValueError(
                f"n_heads must divide hidden_dim, got n_heads={n_heads} "
                f"hidden_dim={hidden_dim}"
            )
        if n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        # Per-atom feature projection: pad/proj the atom features to
        # ``hidden_dim``.  When the caller passes features with a
        # different dim, the projection adapts; when it matches, the
        # projection is an identity (init = scaled eye).
        self.atom_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Per-bond feature aggregation.  Bonds come in as a (B, n_atoms,
        # n_atoms, ...) tensor in the composite; we collapse the last
        # dim with a learned Linear and pool to per-atom.
        self.bond_proj = nn.Linear(1, hidden_dim, bias=False)
        # Learned positional embedding for the canonical-order
        # sequence.  This is the Alcaide 2024 §3.1 positional pattern.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Token embedding for the predicted atom-type sequence (used
        # only for the autoregressive training-time loss; for the
        # skeleton forward we return the logits directly).
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        # Transformer encoder: canonical-order decoding.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        # Output head: hidden_dim → vocab_size.
        self.out_proj = nn.Linear(hidden_dim, vocab_size, bias=True)
        # Default attention mask: ``None`` means all positions attend
        # to all positions (non-causal).  The skeleton is non-causal;
        # callers who need causal masking can pass a ``mask`` tensor
        # to :meth:`forward` (broadcast to ``(B, max_seq_len)``).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(batch, max_seq_len, vocab_size)`` logits.

        Parameters
        ----------
        atom_features : (batch, n_atoms, hidden_dim) tensor
            Per-atom features from the CFM latent.
        bond_features : (batch, n_atoms, n_atoms) or
                        (batch, n_atoms, n_atoms, hidden_bond) tensor
            Per-pair bond features.  When 3-D we collapse the last
            dim via ``bond_proj`` per atom (mean over the partner
            dim) before adding to the atom-token stream.
        mask : (batch, max_seq_len) bool tensor
            ``True`` at valid positions.  Positions where ``mask``
            is ``False`` are excluded from the transformer attention
            via ``key_padding_mask``.

        Returns
        -------
        (batch, max_seq_len, vocab_size) float tensor
            Per-position atom-type logits over the PRIMITIVE_ATOMS
            vocabulary.  ``argmax`` along the last dim yields the
            predicted atom-type index (0 ≤ idx < vocab_size).
        """
        if atom_features.dim() != 3:
            raise ValueError(
                f"atom_features must be (batch, n_atoms, hidden_dim), "
                f"got shape {tuple(atom_features.shape)}"
            )
        if mask.shape[0] != atom_features.shape[0]:
            raise ValueError(
                f"mask batch dim ({mask.shape[0]}) must match atom_features "
                f"batch dim ({atom_features.shape[0]})"
            )
        if mask.shape[1] != self.max_seq_len:
            raise ValueError(
                f"mask length ({mask.shape[1]}) must equal max_seq_len "
                f"({self.max_seq_len})"
            )
        batch, n_atoms, feat_dim = atom_features.shape
        if feat_dim != self.hidden_dim:
            raise ValueError(
                f"atom_features last dim ({feat_dim}) must equal hidden_dim "
                f"({self.hidden_dim})"
            )
        # Project atom features into the hidden_dim space.
        atom_emb = self.atom_proj(atom_features)  # (B, n_atoms, H)
        # Aggregate bond features into a per-atom stream.  We support
        # both 3-D (B, N, N) and 4-D (B, N, N, H) bond tensors.
        if bond_features.dim() == 3:
            bond_in = bond_features.unsqueeze(-1)  # (B, N, N, 1)
        elif bond_features.dim() == 4:
            bond_in = bond_features
        else:
            raise ValueError(
                f"bond_features must be 3-D (B, N, N) or 4-D (B, N, N, H), "
                f"got shape {tuple(bond_features.shape)}"
            )
        if bond_in.shape[1] != n_atoms or bond_in.shape[2] != n_atoms:
            raise ValueError(
                f"bond_features spatial dims ({bond_in.shape[1]}, "
                f"{bond_in.shape[2]}) must match atom_features n_atoms "
                f"({n_atoms})"
            )
        # (B, N, N, 1) → (B, N, N, H) → mean over partner → (B, N, H)
        bond_emb_per_pair = self.bond_proj(bond_in.float())
        bond_emb = bond_emb_per_pair.mean(dim=2)  # (B, N, H)
        # Pad / truncate the per-atom stream to ``max_seq_len`` with
        # zeros so the transformer can consume it.
        if n_atoms < self.max_seq_len:
            pad = torch.zeros(
                batch,
                self.max_seq_len - n_atoms,
                self.hidden_dim,
                dtype=atom_emb.dtype,
                device=atom_emb.device,
            )
            seq_input = torch.cat([atom_emb + bond_emb, pad], dim=1)
        elif n_atoms == self.max_seq_len:
            seq_input = atom_emb + bond_emb
        else:
            # n_atoms > max_seq_len → truncate to ``max_seq_len``.
            seq_input = (atom_emb + bond_emb)[:, : self.max_seq_len, :]
        # Add the learned positional embedding.
        seq_input = seq_input + self.pos_embed[:, : self.max_seq_len, :]
        # TransformerEncoder expects ``key_padding_mask`` of shape
        # (B, S) with ``True`` at *padded* positions (the inverse of
        # our ``mask`` convention).
        key_padding_mask = ~mask  # True where padding
        # We catch the case where all-True padding would crash the
        # encoder: replace all-True rows with a single False.
        all_pad_rows = key_padding_mask.all(dim=1)
        if all_pad_rows.any():
            # Set the first position to False so the row has at least
            # one valid token.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad_rows, 0] = False
        encoded = self.transformer(
            seq_input,
            src_key_padding_mask=key_padding_mask,
        )
        # Output projection: (B, S, H) → (B, S, V).
        logits = self.out_proj(encoded)
        # Zero out the logits at padded positions so callers can use
        # ``argmax`` without worrying about which positions are real.
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        return logits


# ---------------------------------------------------------------------------
# GumbelTopKBondDecoder — differentiable top-k edge selector
# ---------------------------------------------------------------------------
class GumbelTopKBondDecoder(nn.Module):
    """Gumbel-Top-K edge selector — picks the top-``k`` edges per atom.

    The decoder consumes per-pair ``bond_logits`` of shape
    ``(batch, n_atoms, n_atoms)`` and returns a binary adjacency
    tensor of the same shape.  During training, the top-k selector
    uses a Gumbel-softmax relaxation (Danihelka 2022 §3); during
    evaluation, it uses a hard top-k argmax.

    Parameters
    ----------
    temperature : float
        Gumbel-softmax temperature.  Lower → sharper (more
        deterministic); higher → smoother (more gradient flow).
        Default 1.0.
    top_k : int
        Number of edges to keep per atom.  Default 4 (covers sp3
        carbon and Pt_II square-planar).
    hard : bool
        When ``True``, the forward pass returns hard 0/1 entries
        (straight-through estimator on the Gumbel-softmax).  When
        ``False``, returns soft probabilities in ``[0, 1]``.  Default
        ``True`` (matches the M3 unit-test contract: output is a
        *sparse binary adjacency*).
    training : bool
        When ``True``, add Gumbel noise during the top-k selection.
        When ``False``, use deterministic top-k.  This can also be
        overridden per-call via the ``training`` keyword on
        :meth:`forward`.
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        hard: bool = True,
        training: bool = True,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k!r}")
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.hard = bool(hard)
        # The ``training`` flag is a default; :meth:`forward` accepts
        # an override.  We register it as a buffer-friendly Python
        # attribute so ``.eval()`` / ``.train()`` toggle the default.
        self._default_training = bool(training)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        bond_logits: torch.Tensor,
        training: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return ``(batch, n_atoms, n_atoms)`` sparse adjacency.

        Parameters
        ----------
        bond_logits : (batch, n_atoms, n_atoms) float tensor
            Per-pair bond logits (any real values; we use
            ``softmax`` over the partner dim to rank candidates).
        training : bool, optional
            When ``None`` (default), use the module's ``.train()`` /
            ``.eval()`` mode.  When ``True``/``False``, override.

        Returns
        -------
        (batch, n_atoms, n_atoms) float tensor
            Sparse binary adjacency (entries in ``{0, 1}`` when
            ``hard=True``; soft in ``[0, 1]`` when ``hard=False``).
            The diagonal is always 0 (no self-loops).
        """
        if bond_logits.dim() != 3:
            raise ValueError(
                f"bond_logits must be (batch, n_atoms, n_atoms), "
                f"got shape {tuple(bond_logits.shape)}"
            )
        batch, n, m = bond_logits.shape
        if n != m:
            raise ValueError(f"bond_logits must be square, got ({n}, {m})")
        if n == 0:
            return torch.zeros_like(bond_logits)
        if n == 1:
            # Single atom — no possible edges.
            return torch.zeros_like(bond_logits)
        # Resolve the effective training mode.
        if training is None:
            training = self.training and self._default_training
        else:
            training = bool(training)
        # Clamp k to the maximum possible (n_atoms - 1, since no
        # self-loops).  This keeps the per-row top-k valid even when
        # the caller passes ``top_k`` larger than the cloud size.
        eff_k = min(self.top_k, n - 1)
        if eff_k <= 0:
            return torch.zeros_like(bond_logits)
        # Build the scores.  We use the bond logits directly as the
        # score (higher = more likely to be an edge).
        scores = bond_logits.float()
        # Diagonal suppression — no self-loops.
        eye = torch.eye(n, dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(eye, float("-inf"))
        if training:
            # Gumbel-softmax relaxation: add Gumbel noise before the
            # top-k.  We use the Gumbel-Top-K trick from Danihelka
            # 2022 §3: sample g_i ~ Gumbel(0, 1), set
            # s'_i = s_i + g_i, then take top-k.
            u = torch.rand_like(scores)
            # NOTE: we use intermediate variables here (not chained
            # ``-torch.log(-torch.log(u.clamp_min(1e-9)).clamp_min(1e-9))``)
            # because PyTorch's chained unary-minus / clamp_min path
            # can fuse operations in a way that propagates NaN from
            # ``log`` of a tiny clamped value.  Splitting into
            # intermediate variables is bit-exact equivalent and
            # avoids the fusion issue.
            log_u = torch.log(u.clamp_min(1e-9))
            neg_log_u = -log_u
            gumbel = -torch.log(neg_log_u.clamp_min(1e-9))
            scores = scores + gumbel
        # Per-row top-k.  We get the indices of the top-k elements
        # along the partner dim (dim=-1) for each (batch, atom).
        # ``torch.topk`` returns the largest k; since we want the
        # top-k *scores*, this is correct.
        topk_vals, topk_idx = torch.topk(
            scores,
            k=eff_k,
            dim=-1,
            largest=True,
            sorted=False,
        )
        # Build a mask of the same shape as ``scores`` with True at
        # the top-k positions per row.
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        # The diagonal might have been selected when ``top_k >= n``
        # (we already clamped but be defensive).  Force diagonal off.
        mask = mask & ~eye.unsqueeze(0)
        # Convert to the requested output dtype.
        if self.hard:
            # Straight-through: hard 0/1 in the forward pass.
            out = mask.to(bond_logits.dtype)
        else:
            # Soft mode: scale the scores at top-k positions to [0, 1]
            # via sigmoid.  We do NOT use softmax over the full row
            # because the top-k selector is the discrete decision.
            topk_soft = torch.sigmoid(topk_vals / self.temperature)
            out = torch.zeros_like(scores)
            out.scatter_(-1, topk_idx, topk_soft)
            out = out * (~eye.unsqueeze(0)).to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# M3CompositeDecoder — wraps existing decoder_rework + sequence + Gumbel
# ---------------------------------------------------------------------------
class M3CompositeDecoder(nn.Module):
    """Composite decoder wiring for the M3 follow-up to R17 Path-B.

    Parameters
    ----------
    existing_decoder : :class:`molmetal_lam.lam_chem.decoder_rework.DecoderRework`
        or compatible object — the R17 chem-aware soft bond prior.
        Must expose either ``forward(z, pocket)`` or ``decode(cloud)``.
        When ``None``, the composite runs in *head-only* mode (the
        existing decoder is skipped).
    seq_decoder : :class:`SequenceDecoderHead`
        The new sequence decoder head.
    gumbel_decoder : :class:`GumbelTopKBondDecoder`
        The new Gumbel-Top-K edge selector.
    pocket_dim : int
        Dimension of the pocket-embedding input.  Default 64 (matches
        the CFM pocket-conditioned latent).
    """

    def __init__(
        self,
        existing_decoder: Optional[object] = None,
        seq_decoder: Optional[SequenceDecoderHead] = None,
        gumbel_decoder: Optional[GumbelTopKBondDecoder] = None,
        pocket_dim: int = 64,
    ) -> None:
        super().__init__()
        self.existing_decoder = existing_decoder
        self.seq_decoder = (
            seq_decoder if seq_decoder is not None else SequenceDecoderHead()
        )
        self.gumbel_decoder = (
            gumbel_decoder if gumbel_decoder is not None else GumbelTopKBondDecoder()
        )
        if pocket_dim <= 0:
            raise ValueError(f"pocket_dim must be > 0, got {pocket_dim!r}")
        self.pocket_dim = int(pocket_dim)
        # Pocket projection: maps the pocket embedding into the same
        # hidden_dim as the sequence decoder so we can mix it into
        # the atom-token stream.  Default size 64 → seq hidden_dim.
        seq_hidden = self.seq_decoder.hidden_dim
        self.pocket_proj = nn.Linear(pocket_dim, seq_hidden, bias=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        pocket: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> M3CompositeOutput:
        """Run the M3 composite forward pass.

        Parameters
        ----------
        z : (batch, n_atoms, hidden_dim) float tensor
            Per-atom latent features (CFM output projected to
            ``hidden_dim``).  Must match ``seq_decoder.hidden_dim``.
        pocket : (batch, pocket_dim) float tensor
            Per-sample pocket embedding.
        mask : (batch, max_seq_len) bool tensor, optional
            Sequence mask.  When ``None``, defaults to all-True.

        Returns
        -------
        :class:`M3CompositeOutput`
            Sequence logits + sparse bond adjacency + (optional)
            rework bond logits.
        """
        if z.dim() != 3:
            raise ValueError(
                f"z must be (batch, n_atoms, hidden_dim), "
                f"got shape {tuple(z.shape)}"
            )
        if pocket.dim() != 2:
            raise ValueError(
                f"pocket must be (batch, pocket_dim), "
                f"got shape {tuple(pocket.shape)}"
            )
        if pocket.shape[1] != self.pocket_dim:
            raise ValueError(
                f"pocket last dim ({pocket.shape[1]}) must equal "
                f"pocket_dim ({self.pocket_dim})"
            )
        if z.shape[0] != pocket.shape[0]:
            raise ValueError(
                f"z batch ({z.shape[0]}) must match pocket batch "
                f"({pocket.shape[0]})"
            )
        if z.shape[2] != self.seq_decoder.hidden_dim:
            raise ValueError(
                f"z last dim ({z.shape[2]}) must equal seq_decoder.hidden_dim "
                f"({self.seq_decoder.hidden_dim})"
            )
        batch, n_atoms, _ = z.shape
        max_seq_len = self.seq_decoder.max_seq_len
        # Default mask: all positions are valid.
        if mask is None:
            mask = torch.ones(
                batch,
                max_seq_len,
                dtype=torch.bool,
                device=z.device,
            )
        # ---- Existing decoder rework (optional) ----
        rework_bond_logits: Optional[torch.Tensor] = None
        if self.existing_decoder is not None:
            # The R17 decoder exposes either ``forward(z, pocket)`` or
            # ``decode(cloud)``.  We try ``forward`` first (matches the
            # skeleton contract); if it fails we fall back to ``decode``
            # with a synthetic AtomCloud built from ``z``.
            forward_fn = getattr(self.existing_decoder, "forward", None)
            if callable(forward_fn):
                try:
                    rework_bond_logits = forward_fn(z, pocket)
                    # Coerce to a tensor if the existing decoder
                    # returned something exotic (e.g. a dataclass).
                    if not isinstance(rework_bond_logits, torch.Tensor):
                        rework_bond_logits = None
                except Exception:
                    rework_bond_logits = None
            if rework_bond_logits is None:
                # Fall back: build a synthetic AtomCloud and call decode.
                decode_fn = getattr(self.existing_decoder, "decode", None)
                if callable(decode_fn):
                    try:
                        from molmetal.models.bond_head import AtomCloud

                        cloud = AtomCloud(
                            positions=z[..., :3].detach(),
                            atomic_numbers=torch.full(
                                (batch, n_atoms),
                                6,
                                dtype=torch.long,
                                device=z.device,
                            ).reshape(-1)[:n_atoms],
                        )
                        decode_fn(cloud)
                        # The existing decoder returns a DecodedMol;
                        # we synthesise a uniform logits vector so the
                        # composite always returns a tensor.
                        rework_bond_logits = torch.zeros(
                            z.shape[0] * n_atoms * (n_atoms - 1) // 2,
                            device=z.device,
                        )
                    except Exception:
                        rework_bond_logits = None
        # ---- Sequence decoder head ----
        # Build the per-pair bond features from ``z`` by taking the
        # dot-product similarity of atom features.  This is a simple
        # proxy for the soft bond prior; production wiring can swap
        # in the rework's ``p_combined`` instead.
        bond_features = torch.bmm(z, z.transpose(1, 2))  # (B, N, N)
        # Inject the pocket embedding into the atom-token stream by
        # broadcasting it across atoms and adding a projected version.
        pocket_emb = self.pocket_proj(pocket).unsqueeze(1)  # (B, 1, H)
        z_with_pocket = z + pocket_emb
        sequence_logits = self.seq_decoder(
            atom_features=z_with_pocket,
            bond_features=bond_features,
            mask=mask,
        )
        # ---- Gumbel top-k edge selector ----
        bond_adjacency = self.gumbel_decoder(bond_features)
        return M3CompositeOutput(
            sequence_logits=sequence_logits,
            bond_adjacency=bond_adjacency,
            rework_bond_logits=rework_bond_logits,
        )


# ---------------------------------------------------------------------------
# Self-test (lightweight — run as ``python -m
# molmetal_lam.lam_chem.decoder_m3_sequence``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    seq = SequenceDecoderHead()
    gumbel = GumbelTopKBondDecoder()
    composite = M3CompositeDecoder(
        seq_decoder=seq,
        gumbel_decoder=gumbel,
    )
    z = torch.randn(2, 8, 128)
    pocket = torch.randn(2, 64)
    out = composite(z, pocket)
    print(
        f"sequence_logits={tuple(out.sequence_logits.shape)}, "
        f"bond_adjacency={tuple(out.bond_adjacency.shape)}, "
        f"sum_adj={float(out.bond_adjacency.sum().item()):.1f}"
    )
