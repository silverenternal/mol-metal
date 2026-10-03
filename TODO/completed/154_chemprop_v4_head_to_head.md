# Chemprop v2 vs V4MultiTaskDMPNN — Side-by-Side Head-to-Head

> Wf-124 follow-up: MEASURED-only comparison of chemprop v2 (gold standard) vs V4MultiTaskDMPNN (our multi-task D-MPNN + Wasserstein pIC50 regression) on the 5-metal × 3-seed MetalCytoTox benchmark.

---

## §0 TL;DR

- **Verdict in 3 bullets:**
  1. **chemprop v2 wins decisively** on every metal tested: mean AUC 0.8863 vs V4's 0.6042, an overall delta of **+0.2821 AUC** in chemprop's favor across 14 paired runs.
  2. **V4 is 5.1x slower** per run (0.988 min vs 0.193 min) and produces **no AUC win on any of 5 metals** — chemprop dominates Ru / Ir / Os / Rh / Re uniformly.
  3. **Paired t-test is overwhelmingly significant** (t=6.754, **p=1.35e-5**, n=14), so the gap is not noise. V4 is **not competitive** with chemprop v2 at this scale.

---

## §1 Methodology

- **Models:**
  - **V4MultiTaskDMPNN**: Our multi-task D-MPNN + Wasserstein pIC50 regression head. Inputs: SMILES + metal identity (5-way task head). Trained 30 epochs, 3 seeds (0/1/2); Ru/Ir/Os/Rh have 3 seeds, Re has 2.
  - **chemprop v2**: Chemprop v2.0 (`chemprop` package) default D-MPNN classifier. Same 5-metal × 3-seed split, 50 epochs (chemprop default).
- **Data:** MetalCytoToxDB 5-metal subset (Ru/Ir/Os/Rh/Re) — same splits, same labels for both models. chemprop api_wiring_works = true (confirmed live Python invocation, not cached).
- **Metric:** Per-fold test ROC-AUC (held-out test set after split).
- **Hardware:** Same machine, same conda env (`.venv-vlm`). V4 ran on multi-task D-MPNN; chemprop ran on chemprop v2.0 D-MPNN. Both on CPU-class epochs (chemprop is dramatically faster — likely smaller per-metal training set due to chemprop's per-task training).
- **Statistical test:** Paired t-test across all metal × seed pairs (n=14 paired runs). Per-metal paired t-tests reported separately (each n=2 or n=3, low power).

---

## §2 Per-metal AUC table

Mean AUC over seeds per metal. chemprop wins on all 5 metals.

| Metal | V4 mean AUC | V4 std | chemprop mean AUC | chemprop std | Δ (chemprop − V4) | Winner |
|-------|-------------|--------|-------------------|--------------|-------------------|--------|
| Ru    | 0.5419      | 0.1021 | 0.9095            | 0.0022       | **+0.3675**       | chemprop |
| Ir    | 0.4803      | 0.0646 | 0.8846            | 0.0060       | **+0.4043**       | chemprop |
| Os    | 0.5788      | 0.2312 | 0.8890            | 0.0153       | **+0.3103**       | chemprop |
| Rh    | 0.7261      | 0.0293 | 0.8631            | 0.0208       | **+0.1369**       | chemprop |
| Re    | 0.6937      | 0.1626 | 0.8822            | 0.0283       | **+0.1885**       | chemprop |
| **Mean over metals** | **0.6042** | — | **0.8857** | — | **+0.2815** | chemprop |

**Key observation:** V4's best metal (Rh, 0.7261) is still 0.137 AUC below chemprop's worst metal (Rh, 0.8631). V4's worst (Ir, 0.4803) is below random. chemprop's worst std (Re, 0.0283) is still better than V4's best mean.

---

## §3 Per-metal wall-time table

Per-seed mean training time. chemprop is faster on every metal.

| Metal | V4 mean t (s) | chemprop mean t (s) | Speed ratio (V4 / chemprop) |
|-------|---------------|---------------------|------------------------------|
| Ru    | 54.1          | 24.2                | 2.2x slower                  |
| Ir    | 53.2          | 5.1                 | 10.4x slower                 |
| Os    | 50.1          | 1.6                 | 31.4x slower                 |
| Rh    | 71.3          | 1.7                 | 41.2x slower                 |
| Re    | 72.0          | 1.5                 | 46.7x slower                 |
| **Mean per run (min)** | **0.988** | **0.193** | **5.1x slower** |

**Key observation:** chemprop's per-metal training is dramatically faster for Ir/Os/Rh/Re (10-47x). The likely cause: chemprop trains each metal independently (5 small tasks), while V4 trains one multi-task network that pays the full D-MPNN cost even on small per-metal splits. Ru is comparable (2.2x) because Ru is the largest split and V4's multi-task overhead amortizes.

---

## §4 Verdict (V4 vs chemprop)

| Dimension | chemprop v2 | V4MultiTaskDMPNN | Δ |
|-----------|-------------|------------------|---|
| Mean AUC over 5 metals | **0.8857** | 0.6042 | **+0.2815 chemprop** |
| Wins on metals | **5 / 5** | 0 / 5 | chemprop sweep |
| Wall time / run | **0.193 min** | 0.988 min | 5.1x slower V4 |
| Paired t-test (chemprop − V4) | — | — | **t=6.754, p=1.35e-5 (highly sig.)** |
| Per-metal p-values | — | — | Ru p=0.025, Ir p=0.010, Os p=0.144, Rh p=0.027, Re p=0.421 |

**Statistical conclusion:** The AUC gap is overwhelmingly significant at the cross-metal paired level (p=1.35e-5). Even per-metal, Ru and Ir reach p<0.05; Os/Rh approach significance despite n=3 (low power); Re has n=2 so its t-test is underpowered.

**V4 is not competitive with chemprop v2 as a gold standard for per-metal cytotoxicity AUC on this benchmark.**

---

## §5 Honest framing per party-twenty-spirit §3.5

- **MEASURED-only claim scope:** This comparison is honest-negative for V4. The numbers are MEASURED (not projected), but the framing must be honest about what V4 was *designed for* vs what chemprop v2 was benchmarked on:
  - V4 is a **multi-task** model trained jointly on 5 metals with a Wasserstein pIC50 auxiliary regression head. It was designed to test *transfer / shared representation* across metals, not to win per-metal AUC against a per-metal specialist.
  - chemprop v2 here is trained **per-metal independently** (5 separate models), each on its own train fold. Per-metal specialization is the easier problem and chemprop exploits it cleanly.
  - **Apples-to-apples gap:** The fair comparison would be (a) chemprop multi-task on 5 metals with shared backbone, or (b) V4 per-metal independently without the Wasserstein auxiliary. Neither was run in this benchmark.
- **What V4 still proves:** Even in this unfair comparison, V4 reaches 0.7261 AUC on Rh (vs chemprop 0.8631) — i.e. V4 is not random, it does learn signal, and on its strongest metal the gap shrinks to 0.137 AUC. This is a useful **lower bound** on V4's per-metal capability.
- **What V4 does NOT prove:** That multi-task D-MPNN with Wasserstein pIC50 reg is a state-of-the-art model. It is not — chemprop v2 (a per-metal specialist) outperforms it handily.
- **Honest-negative discipline:** Do not promote V4 in paper §4 as "competitive with chemprop." It is not. If V4 is mentioned, it must be framed as (i) a multi-task baseline that learns shared metal representations and (ii) a lower-bound on what per-metal D-MPNN can achieve without specialist tuning.
- **Recommended action:** Drop V4 from the per-metal AUC main claim in paper §4. Lead with chemprop v2 as the AUC backbone. Optionally keep V4 as a §B appendix baseline if the multi-task transfer story is worth telling.

---

## §6 Paper §4 implications

1. **Drop V4MultiTaskDMPNN from the §4 main result.** chemprop v2 is the AUC backbone. Re-render Table 4.1 / Fig 4.2 with chemprop v2 numbers.
2. **Lead claim rewording:** "Our chemprop v2 backbone reaches 0.886 ± 0.020 mean AUC across 5 metals (Ru/Ir/Os/Rh/Re) on MetalCytoToxDB, with chemprop outperforming a multi-task D-MPNN+Wasserstein baseline (V4) by +0.282 AUC (paired t-test p=1.35e-5)."
3. **§4 limitations paragraph (new):** Add an honest-negative paragraph stating that the multi-task baseline V4 was tested but underperformed per-metal chemprop, and that the multi-task transfer story is relegated to §B appendix (or dropped entirely if §B space is tight).
4. **§6 limitations (paper):** Note that the per-metal AUC comparison is not apples-to-apples (chemprop per-metal specialist vs V4 multi-task); future work would need chemprop-multi-task vs V4-per-metal to isolate the effect of the Wasserstein pIC50 auxiliary.
5. **Speedup side-note:** chemprop is also ~5x faster, which is operationally relevant — flag in §4.3 / §5 deployment discussion.
6. **Do NOT cite V4 mean AUC 0.6042 as a positive result anywhere in the main text.** Cite only in §B appendix if at all, with the full paired-t-test p-value and the apples-to-apples caveat.

---

*Generated 2026-09-27 from Wf-124 P3 outputs. All numbers MEASURED. honest-negative framing per party-twenty-spirit §3.5.*
