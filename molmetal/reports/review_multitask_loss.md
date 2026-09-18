# Multi-task Loss Review — Hybrid V3 vs DMPNN-MT (R2)

## Why hybrid v3 doesn't reuse R2's `NormalizedLoss`

`metal_hybrid_v3.py` loss path is **stitched by hand** in the training
script: `α·BCE + (1-α)·L1 + γ·coord_refine` with α=0.5, γ=0.05 (config
`coord_loss_weight`). `MultiTaskLoss`/`NormalizedLoss`/`UncertWeightedLoss`
all live in `dmpnn_multitask.py`, take only `(pic50_pred, active_logits, ...)`
and use BCE + MSE — no coord term, no LS hook, no γ-annealing. To share
the wrapper the training loop would need (i) a uniform 4-arg API, (ii)
MMFF94 Δx as a separate tensor, (iii) learnable σ for Kendall. None of
that exists yet → hybrid silently re-implements the vanilla α-weighted sum.

## Recommendations (ordered by expected ROI)

| # | Change | Why |
|---|--------|-----|
| 1 | α-sweep {0.1, 0.3, 0.5} on Ru temporal | R2 already showed 0.5907@α=0.1 vs 0.5277@0.5 — small α wins |
| 2 | Adopt `UncertWeightedLoss` (Kendall 2018) | R2 OOD AUC 0.5460, removes α entirely |
| 3 | GradNorm (Chen & Jacob 2021) | Auto-tunes per-task weight w.r.t. shared-trunk grad norm |
| 4 | γ-anneal 0.1→0.0 cosine | Force 3-D signal early, relax late (kills EGNN drift) |
| 5 | Label-smoothing 0.05 on BCE | Cheap regulariser on noisy active label |
| 6 | Aux heads: MW / logP / TPSA / n_atoms | Cheap RDKit targets regularise the trunk |

## LossV4 wrapper — pseudo code

```python
class LossV4(nn.Module):
    def __init__(self, alpha=0.3, gamma_start=0.1, gamma_end=0.0,
                 ls=0.05, warmup=64, use_kendall=True,
                 use_gradnorm=True, aux_tasks=("mw","logp","tpsa","n")):
        super().__init__()
        self.alpha, self.gamma_start, self.gamma_end = alpha, gamma_start, gamma_end
        self.ls = ls
        self.use_kendall, self.use_gradnorm = use_kendall, use_gradnorm
        # Kendall learnable log-sigma
        self.log_s_cls = nn.Parameter(torch.tensor(0.0))
        self.log_s_reg = nn.Parameter(torch.tensor(0.0))
        self.log_s_crd = nn.Parameter(torch.tensor(0.0))
        # Aux heads (RDKit scalars → 1-unit MLP)
        self.aux_heads = nn.ModuleDict({t: nn.Linear(hidden, 1) for t in aux_tasks})
        # Normalisation buffers (R2 trick)
        self.register_buffer("cls_init", torch.tensor(0.0))
        self.register_buffer("reg_init", torch.tensor(0.0))
        self.register_buffer("crd_init", torch.tensor(0.0))
        self._n = 0; self.warmup = warmup
        # BCE with label smoothing
        self.bce = nn.CrossEntropyLoss(label_smoothing=ls)
        self.mse, self.l1 = nn.MSELoss(), nn.L1Loss()
        # GradNorm restore weights (per task)
        self.w_restore = {t: torch.tensor(1.0, requires_grad=True) for t in
                          ("cls","reg","coord","mw","logp","tpsa","n")}
        self.last_losses = {}

    def gamma_at(self, epoch, T):
        # cosine anneal γ: 0.1 → 0.0 over T epochs
        cos = 0.5 * (1 + math.cos(math.pi * epoch / max(T-1, 1)))
        return self.gamma_end + (self.gamma_start - self.gamma_end) * cos

    def _per_task(self, out, batch, epoch, T):
        active_lg, pic50_p = out["active_logits"], out["pic50_pred"]
        y_a, y_p = batch["active"], batch["pic50"]
        target_delta = batch["target_delta"]          # MMFF94 Δx
        lc = self.bce(active_lg, y_a)
        lr = self.l1(pic50_p, y_p)
        lcrd = ((out["delta_pred"] - target_delta) ** 2).mean()
        # auxiliaries
        laux = {t: self.mse(self.aux_heads[t](out["trunk"]), batch[f"y_{t}"]).reshape(())
                for t in self.aux_heads}
        # R2 normalisation (warmup)
        raw = {"cls": lc, "reg": lr, "coord": lcrd, **laux}
        if self._n < self.warmup:
            with torch.no_grad():
                self.cls_init += lc.detach(); self.reg_init += lr.detach()
                self.crd_init += lcrd.detach(); self._n += 1
                if self._n == self.warmup:
                    self.cls_init /= self.warmup; self.reg_init /= self.warmup
                    self.crd_init /= self.warmup
        else:
            raw = {k: v / (getattr(self, f"{k[:3] if k!='coord' else 'crd'}_init") + 1e-8)
                   for k, v in raw.items()}
        self.last_losses = {k: v.detach() for k, v in raw.items()}
        return raw

    def forward(self, out, batch, epoch=0, T=30, shared_params=None):
        raw = self._per_task(out, batch, epoch, T)
        if self.use_kendall:
            w = {k: torch.exp(-2.0 * getattr(self, f"log_s_{k[:3] if k!='coord' else 'crd'}").clamp(-3,3))
                 if k in ("cls","reg","coord") else torch.exp(-2.0 * self.log_s_aux[k]).clamp(-3,3)
                 for k in raw}
            reg = {k: w[k] * raw[k] + getattr(self, f"log_s_{k[:3] if k!='coord' else 'crd' if k!='cls'...}" )
                   for k in raw}   # + log_σ regulariser per task
        else:
            g = self.gamma_at(epoch, T)
            w = {"cls": self.alpha, "reg": 1.0 - self.alpha, "coord": g,
                 **{t: 0.05 for t in self.aux_heads}}
            reg = {k: w[k] * raw[k] for k in raw}
        # GradNorm: re-balance w.r.t. ||∇_shared L_k||
        if self.use_gradnorm and shared_params is not None:
            norms = {k: torch.autograd.grad(reg[k], shared_params, retain_graph=True)
                          .norm() for k in reg}
            target = norms["reg"].detach() * (reg[k].detach() / reg["reg"].detach()) ** 0.5
            for k in reg:
                self.w_restore[k] = self.w_restore[k] - 1e-3 * (norms[k] - target[k])
                reg[k] = self.w_restore[k].clamp(0, 5) * raw[k]
        total = sum(reg.values())
        return total, self.last_losses
```

### Wiring in `metal_hybrid_v3.py`

- Expose `trunk = fused_per_atom.mean(1)` and pass to `LossV4`.
- Compute MMFF94 Δx in collate once (cache per epoch).
- Aux targets `mw/logp/tpsa/n` from RDKit `Descriptors` + `mol.GetNumAtoms()`
  computed in `__getitem__` (zero-overhead if cached).
- Replace `coord_loss_weight` static γ with `LossV4.gamma_at(epoch, T)`.
- Test order: `LossV4` smoke → α-sweep 0.1/0.3/0.5 on 1 fold → gradnorm
  ablation on Ru temporal.