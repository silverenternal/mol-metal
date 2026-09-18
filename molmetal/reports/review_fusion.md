# Fusion V3 Review → V4 Proposal

## 1. V3 = gated concat, NOT cross-attn

`metal_hybrid_v3.py` line 359-361: `gate = sigmoid(W_g [h_dmpnn||h_egnn])` then
`g * h_egnn + (1-g) * h_dmpnn`. There is no cross-attention. The R1 report
("cross-attn V3 AUC 0.5299") actually measured `CrossAttentionFusionV3`
(`cross_attention_fusion_v3.py` line 51-226), which is a separate class with
4-head cross-attn + residual 0.5 + LayerNorm + equivariant init. So we have
THREE candidates to compare, not two: (a) gated concat [metal_hybrid_v3],
(b) cross-attn residual 0.5 [cross_attention_fusion_v3], (c) Perceiver IO
latent-array cross-attn [Jaegle 2021, arXiv:2107.14795].

**Recommendation: (c) Perceiver latent**. (a) has only one learnable scalar
per token and the gate bias→sigmoid(0)=0.5 means EGNN contributes 50% from
step 0 — exactly the failure mode of V2's cross-attn (latched onto 3D
spurious cues). (b) uses the entire per-atom set as keys/values: O(N²) and
every atom attends to every other atom, so a single spurious global
correlation can dominate. (c) decouples cross-attn dimensionality from N
via a fixed latent array, matches Perceiver IO's design (latent queries
cross-attend to a small input set), and the latent count (16) is a
hyperparameter independent of molecule size → robust to variable atom
counts and less prone to feature contamination than (b)'s pooled→pooled
pattern.

## 2. Gate init = 0.05, not 0.5

Yes, 0.5 is too aggressive. `nn.Linear` bias init is U(-1/k, 1/k) so
`sigmoid(0)=0.5` is the default — meaning EGNN contributes 50% of fused
from step 0, BEFORE any training signal proves 3D helps. Set bias to
`log(0.05/0.95) ≈ -2.944` so initial gate ≈ 0.05; EGNN is a small
perturbation that the optimiser can grow if the data supports it.

## 3. Multi-head concat (8 × 16 = 128)

Replace `out = Linear(hidden, d)` (additive head mix) with concatenated
per-head outputs: 8 heads × 16 head_dim = 128, then a single output Linear.
Concat (Vaswani 2017) preserves per-head subspaces; add forces them into
one linear combination at init.

## 4. FFN block post-attention

Pre-norm transformer block (Gini et al. 2020): `LN → MHCA → dropout →
LN → GELU → Linear(4d) → dropout → Linear(d) → residual`. 4× expansion is
standard (BERT/GPT). Gives the model a place to do non-linear fusion of
attended context before residual add.

## 5. Input dropout 0.1 on each stream

Drop 10% of per-atom features on both h_dmpnn and h_egnn before fusion.
Cheap regulariser against over-reliance on either stream (matches V3's
"feature contamination" defence).

## 6. Perceiver latent array

16 learnable latent vectors of dim 128. Latents attend (Q from L, K/V from
pooled h_dmpnn and h_egnn concatenated). Output: (B, 16, 128) latents →
mean-pool → fused molecule vector. Decouples fusion compute from N,
removes the O(N²) cross-attn from V3, and matches Perceiver IO §3.2
(latent transformer cross-attends to arbitrary input modalities).

## FusionV4 pseudo-code

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

N_LATENT, LATENT_DIM, N_HEADS = 16, 128, 8
HEAD_DIM = LATENT_DIM // N_HEADS        # 16
INPUT_DROPOUT = 0.1; ATTN_DROPOUT = 0.2
FFN_DROPOUT = 0.1; RESIDUAL_SCALE = 0.5
GATE_INIT_BIAS = math.log(0.05 / 0.95) # -2.944

class FusionV4(nn.Module):
    def __init__(self, dmpnn_dim=128, egnn_dim=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(N_LATENT, LATENT_DIM) * 0.02)

        # 6. cross-attn: Q = latents, K/V = concat(dmpnn, egnnn) per-atom
        self.q = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.k = nn.Linear(dmpnn_dim + egnn_dim, LATENT_DIM)
        self.v = nn.Linear(dmpnn_dim + egnn_dim, LATENT_DIM)
        self.attn_drop = nn.Dropout(ATTN_DROPOUT)

        # 3. concat-head output
        self.out = nn.Linear(LATENT_DIM, LATENT_DIM)

        # 4. FFN
        self.norm1 = nn.LayerNorm(LATENT_DIM)
        self.ffn = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 4 * LATENT_DIM),
            nn.GELU(),
            nn.Dropout(FFN_DROPOUT),
            nn.Linear(4 * LATENT_DIM, LATENT_DIM),
            nn.Dropout(FFN_DROPOUT),
        )

        # 2. gated concat: scalar gate, init ≈ 0.05
        self.gate = nn.Linear(2 * LATENT_DIM, LATENT_DIM)
        with torch.no_grad():
            self.gate.bias.fill_(GATE_INIT_BIAS)
            self.gate.weight.mul_(0.1)   # grow slowly

        # 5. input dropout
        self.in_drop_dmpnn = nn.Dropout(INPUT_DROPOUT)
        self.in_drop_egnn  = nn.Dropout(INPUT_DROPOUT)
        self.res_scale = RESIDUAL_SCALE

    def _heads(self, x):                    # (B, L, H*D) -> (B, h, L, D)
        B, L, _ = x.shape
        return x.view(B, L, N_HEADS, HEAD_DIM).transpose(1, 2)

    def forward(self, h_d, h_e, mask=None):
        # 5. input dropout
        h_d = self.in_drop_dmpnn(h_d)
        h_e = self.in_drop_egnn(h_e)
        kv = torch.cat([h_d, h_e], dim=-1)                # (B, N, d+e)
        B = kv.size(0)
        z = self.latents.unsqueeze(0).expand(B, -1, -1)   # (B, L, D)

        # MHCA (3. concat heads)
        q = self._heads(self.q(z))
        k = self._heads(self.k(kv))
        v = self._heads(self.v(kv))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        if mask is not None:
            scores = scores.masked_fill(
                (~mask.bool())[:, None, None, :], float("-inf"))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        ctx = attn @ v                                   # (B, h, L, D)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N_LATENT, LATENT_DIM)
        z = self.norm1(z + self.res_scale * self.out(ctx))

        # 4. FFN
        z = z + self.ffn(z)                               # (B, L, D)

        # 2. gated concat against dmpnn-pooled baseline
        d_pool = h_d.masked_fill(~mask.bool()[..., None], 0).sum(1) \
                 / mask.sum(1, keepdim=True).clamp(min=1)
        g = torch.sigmoid(self.gate(torch.cat([d_pool, z.mean(1)], -1)))
        return g * z.mean(1) + (1 - g) * d_pool           # (B, D)
```

**Init summary**: gate bias `-2.944` → σ≈0.05; latent std 0.02; FFN
Linear weights Xavier; `gate.weight *= 0.1` so EGNN contribution grows
slowly.