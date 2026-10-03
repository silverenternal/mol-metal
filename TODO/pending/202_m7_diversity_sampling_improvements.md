# 202 — m7 Diversity Sampling Improvements (top-p + top-k + MMD + repulsion)

**Document ID:** TODO-202 / Wf-382-M7-DIVERSITY
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + agents
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §3.4 — m7 Diversity 0.792482 close to SOTA 0.81, lift via sampling + diversity-aware loss

---

## §0 TL;DR

**问题**: m7 Diversity = 0.792482 (SOTA 0.81, gap -0.018). Current sampling uses greedy argmax, no diversity incentive.

**Solution**: 4 diversity improvements (CPU-only):
1. **Top-p + top-k + temperature sampling** (per nucleus sampling paper)
2. **MMD (Maximum Mean Discrepancy) loss** (penalize similar molecules)
3. **Scaffold-aware conditional generation** (scaffold conditioning)
4. **Repulsion loss** (explicit repulsion between generated samples)

**Wall**: 1-2 days CPU-only

---

## §1 Background

### Current state

- m7 Diversity = 0.792482 (per Wf-357-B1)
- SOTA = 0.81 (per Krasnov benchmark)
- Gap = -0.018
- Generation sampling: greedy argmax (no diversity incentive)

### Method

#### Top-p + top-k + temperature sampling (Holtzman 2020)

- **Top-p (nucleus)**: sample from smallest set of tokens whose cumulative probability ≥ p
- **Top-k**: sample from top k tokens
- **Temperature**: τ ∈ [0.5, 1.5] — higher = more diverse

```python
def nucleus_sample(logits, p=0.9, k=50, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return F.softmax(logits, dim=-1).multinomial(1)
```

#### MMD loss (Gretton 2012)

```python
def mmd_loss(samples_1, samples_2, kernel='rbf', sigma=1.0):
    K_xx = rbf_kernel(samples_1, samples_1, sigma)
    K_yy = rbf_kernel(samples_2, samples_2, sigma)
    K_xy = rbf_kernel(samples_1, samples_2, sigma)
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
```

#### Scaffold-aware conditional generation

- Condition generation on Murcko scaffold
- Per-scaffold diversity is higher than cross-scaffold diversity

#### Repulsion loss

```python
def repulsion_loss(samples, margin=1.0):
    # Penalize pairwise similarity > margin
    pairwise_sim = F.cosine_similarity(samples.unsqueeze(1), samples.unsqueeze(0), dim=-1)
    mask = torch.triu(torch.ones_like(pairwise_sim), diagonal=1)
    repulsion = F.relu(pairwise_sim - margin) * mask
    return repulsion.sum() / mask.sum()
```

---

## §2 File changes

### §2.1 New modules

- `molmetal/adapters/flow_matching_lipman/diversity_sampling.py`: NEW (top-p + top-k + temperature + MMD + repulsion, ~250 LOC)
- `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py`: NEW (scaffold-aware generation, ~100 LOC)

### §2.2 Wire to trainer

- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add CLI flags:
    - `--top-p` (default 0.9)
    - `--top-k` (default 50)
    - `--temperature` (default 1.0)
    - `--mmd-weight` (default 0.1)
    - `--repulsion-weight` (default 0.05)
    - `--scaffold-conditioning` (default OFF)

### §2.3 Wire to evaluator

- `molmetal/scripts/cfm_property_panel.py`: add `--diversity-aware-eval` flag (default ON)
- Per-scaffold diversity + cross-scaffold diversity

### §2.4 Metrics

- `metrics/by_metric/m7_diversity.json`: update with per-cohort diversity
- Per-sample diversity (within batch) + cross-batch diversity

---

## §3 Expected lift

### §3.1 m7 Diversity 0.792482 → 0.81+

- **Top-p + top-k + temperature**: +0.005-0.01
- **MMD loss**: +0.005-0.01
- **Scaffold conditioning**: +0.005-0.01
- **Repulsion loss**: +0.003-0.005
- **Total expected lift**: +0.02-0.035 (SOTA 0.81 in range)

### §3.2 Side effects

- m4 SA may improve (scaffold conditioning → SA-friendly scaffolds)
- m5 QED may slightly improve (scaffold conditioning → QED-friendly scaffolds)
- m14 decode may slightly degrade (sampling diversity vs determinism tradeoff)

---

## §4 Test plan

3 NEW tests:
1. **test_nucleus_sampling** — top-p sampling matches Holtzman 2020 reference
2. **test_mmd_loss** — MMD loss is non-negative + symmetric
3. **test_repulsion_loss** — repulsion loss penalizes similarity

---

## §5 Acceptance criteria

- 3 NEW tests PASS in <5 s
- `metrics/by_metric/m7_diversity.json` updated
- Top-p + top-k + temperature sampling integrated to `train_cfm_yuelbond_wasserstein.py`
- MMD + repulsion loss available as trainer flags

---

## §6 Cross-references

- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §3.4 — m7 deep-dive
- `molmetal/scripts/cfm_property_panel.py` — m7 evaluator
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — target trainer
- Wf-357-B1 — m7 measurement protocol
- AGENTS.md §5.2 — honest framing mandate

---

**END OF TODO-202** — 2026-09-30, 6 sections, 4 diversity methods, 3 NEW tests expected, ~1-2 days wall
