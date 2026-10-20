# FIX-C6 — LOMO：0.5575 里有 0.04 是「不可解析的折」

**日期**：2026-10-20
**状态**：DONE / **KILL（但归因改变）**
**脚本**：`lomo_resolvability_audit.py`（审计）、`lomo_seed_ensemble.py`（集成）、
`train_metal_lomo_wasserstein.py` 新增 `--dump-test-scores-dir`（仅落盘）
**测试**：`test_lomo_seed_ensemble.py`（7 个，全过）
**产物**：`metrics/by_metric/lomo_resolvability_audit.json`、
`lomo_seed_ensemble.json`、`results/lomo_seed_ensemble/*.npz`（25 个）

## 问题

C6 的可写数字是「LOMO 0.5575 ± 0.0949 < 0.7」。但这个 0.5575 是**五金属等权
均值**，其中：

- Rh 51 个唯一配体、Os 36、Re 74 —— 全部低于项目自己的
  `MIN_UNIQUE_FOR_GATE = 200`（`metal_activity_transfer_probe.py:185`，
  注释原文：「below this many unique ligands a metal's AUC is not resolvable
  and must not be gated」）；
- 只有 Ru（1,311）与 Ir（538）可解析。

**这条规则早就存在，LOMO 的 headline 从来没有应用它。**

## MEASURED — 按规则拆开

| 折 | n | pos/neg | AUC | Hanley 95% CI | 可解析 |
|---|---:|---|---:|---|---|
| Ru | 1,311 | 366/945 | 0.5972 | [0.5622, 0.6322] | 是 |
| Ir | 538 | 249/289 | 0.5975 | [0.5495, 0.6456] | 是 |
| Os | 36 | 15/21 | 0.5831 | [0.3907, 0.7755] | 否 |
| Rh | 51 | 17/34 | 0.3812 | [0.2217, 0.5407] | 否 |
| Re | 74 | 15/59 | 0.6288 | [0.4628, 0.7947] | 否 |

- **all_metals = 0.5576**（已发布口径）
- **resolvable_only = 0.5974**（规则口径，Ru+Ir）
- 目标 0.6 → 差 **0.0026**，不是 0.0424。
- **Rh 的 CI 上界 0.5407 包含 0.5**：它在 n=51 上根本分不清是不是随机。五个 seed
  都落在 0.35–0.43，那是**模型方差**（同一测试集、不同初始化），不是采样误差；
  真正的采样误差是 Hanley SE = 0.081。

## MEASURED — seed 集成能不能补上这 0.003

已发布口径是「每个 seed 先算 AUC，再取平均」；部署时是「先平均分数，再算一次
AUC」。两者不同。给 trainer 加 `--dump-test-scores-dir`（只写它本来就算出的分数），
5 折 × 5 seed × 30 epoch **并发重跑 2 分 33 秒**，落 25 个 npz。

| 折 | seed-mean | 集成 | 增益 | 集成 CI |
|---|---:|---:|---:|---|
| Ru | 0.5885 | 0.5901 | +0.0016 | [0.5687, 0.6115] |
| Ir | 0.5987 | **0.6012** | +0.0025 | [0.5695, 0.6329] |
| Os | 0.5828 | 0.5636 | −0.0191 | [0.4412, 0.6861] |
| Rh | 0.3733 | 0.3814 | +0.0081 | [0.2718, 0.4910] |
| Re | 0.5438 | 0.5722 | +0.0284 | [0.4506, 0.6938] |

- **resolvable_only 集成 = 0.5957 < 0.6**（seed-mean 0.5936）。
- Ir 单折过了 0.6（0.6012），Ru 没有（0.5901）。
- 4/5 折有正增益，但幅度 +0.002 到 +0.028，**机制对，力度不够**。

## 复现检查失败（必须报）

预注册的 gate 要求「逐 seed AUC 复现已发布 0.5575，|diff| < 0.01」。实测
**0.5417，diff 0.0158 > 0.01**：Re 掉得最多（0.6288 → 0.5438，−0.085），
Os/Ru/Rh 也有 0.001–0.008 的漂移。这与项目已知的 D-MPNN CUDA 非确定性一致
（每折 ±0.08），本次没有加 `--deterministic`。

**所以这次重跑是同一配置下的另一次抽样，不是对已发布面板的复现。** 集成增益
（内部比较）仍然有效；与已发布数字的差值不可比。

## 判定

**KILL**（两条 kill 线都触发）：

- `resolvable_only` 集成 0.5957 < 0.5974（kill 线）；
- 复现检查 diff 0.0158 > 0.01（kill 线）。

## 诚实读数

1. **C6 的「差 0.04」是聚合方式的产物，不是模型差 0.04。** 按项目自己的规则，
   差 0.0026；按规则外的等权均值，差 0.0424。两个数都要报，且必须说明哪个入闸。
2. **Rh 不是模型失败，是测量地板。** 17 个正例，CI [0.22, 0.54]。把 Rh 的 0.38
   写进一个「gate ≥0.6」的均值里，等于用一个分辨不出 0.5 的数去拉低整体。
3. **seed 集成是正确方向但不够。** +0.0025（Ir）/ +0.0016（Ru），离 0.6 还差
   0.004（Ru）。要补上这 0.004 需要换表征，不是换聚合。
4. **两个可解析折的 CI 都包含 0.6**（Ru [0.562, 0.632]、Ir [0.550, 0.646]）。
   所以「达不到 0.6」和「与 0.6 无差别」在统计上同时成立——只能说**没有证据
   超过 0.6**，不能说**证明低于 0.6**。
5. **0.7 的旧 gate 从来不是这个任务的目标。** artifact 里的 `gate_target` 是 0.7，
   但 `target_overall_auc` 是 0.6；0.7 是更早一版的门槛。两个数字混用会让
   「0.5575 < 0.7」看起来差 0.14，而实际对照的是 0.6。
6. **复现失败本身是负结果。** 同一脚本、同一配置、同一 seed，Re 折差 0.085。
   在 CUDA 非确定性下，「可复现」需要显式 `--deterministic`，而历史面板没有记录
   这个开关。

## 对台账的影响

C6 从「NOT SUPPORTED（0.5575 差 0.042）」改为：

> **NOT SUPPORTED**。可解析金属（Ru 1,311 / Ir 538）的 LOMO 为 0.5974
> （集成 0.5957），低于 0.6 目标 0.0026，且 CI 包含 0.6；Rh/Os/Re 低于
> `MIN_UNIQUE_FOR_GATE=200`，其 AUC 不可解析（Rh 的 CI 含 0.5），不得入闸。
> 因此不能写「跨金属迁移达 0.6」，也不能写「差得很远」。

## 复现

```bash
# 审计（秒级）
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/lomo_resolvability_audit.py

# 25 run 并发（约 2.5 分钟；GPU0）
for m in Ru Ir Os Rh Re; do
  OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/train_metal_lomo_wasserstein.py \
    --metals $m --seeds 0 1 2 3 4 --epochs 30 --wasserstein-weight 0.05 --device cuda:0 \
    --dump-test-scores-dir results/lomo_seed_ensemble \
    --out-dir /tmp/lomo_ens/$m --out-metric /tmp/lomo_ens/$m/metrics.json &
done; wait

OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/lomo_seed_ensemble.py
```
