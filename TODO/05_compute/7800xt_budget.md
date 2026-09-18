# 7800 XT 算力预算

## 硬件规格

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 7800 XT |
| 架构 | RDNA 3 (gfx1101) |
| VRAM | 16 GB GDDR6 |
| FP32 算力 | ~52 TFLOPs |
| FP16 算力 | ~104 TFLOPs |
| 内存带宽 | 576 GB/s |
| 软件 | ROCm 7.2.4 + torch 2.14 + triton-rocm 3.8 |

## 模型尺寸估计

### D-MPNN (2D-only)
| 组件 | 参数量 |
|---|---|
| Atom embed (100 × 128) | 12.8k |
| Bond embed (4 × 128) | 0.5k |
| D-MPNN layers × 3 (h=128, msg=256) | ~600k |
| Readout MLP | 33k |
| **小计** | ~650k |

### EGNN (3D-aware)
| 组件 | 参数量 |
|---|---|
| Atom embed (100 × 128) | 12.8k |
| RBF embed (32 × 128) | 4k |
| EGNN layers × 3 (h=128, edge=128) | ~700k |
| vel_head (128 → 3) | 0.4k |
| **小计** | ~720k |

### Fusion + readout
| 组件 | 参数量 |
|---|---|
| Cross-attn (4 heads, 128 dim) | ~250k |
| Concat-MLP fusion | ~100k |
| Per-atom → graph sum-pool | 0 |
| Graph-level MLP (256 → 128 → 1) | ~33k |
| **小计** | ~380k |

### 总参数量
- **小模型**:~1.75M params (D-MPNN + EGNN + fusion)
- **中等模型**:3-5M (hidden=256)
- **大模型**:8-12M (hidden=384)

## VRAM 占用

### Forward pass (batch=32, max_atoms=29)

| 张量 | 形状 | dtype | 大小 |
|---|---|---|---|
| atom features | (32, 29, 128) | fp32 | 0.47 MB |
| edge features | (32, 812, 128) | fp32 | 13.3 MB |
| 3D coords | (32, 29, 3) | fp32 | 0.01 MB |
| **EGNN 激活** × 3 层 | ~40 MB/layer | | 120 MB |
| **D-MPNN 消息** × 3 层 | ~30 MB/layer | | 90 MB |
| Optimizer state (Adam) | 2 × params | | ~14 MB (1.75M × 8) |
| Gradients | = params | | ~7 MB |
| **总激活** | | | ~230 MB |
| **总参数 + 优化器** | | | ~25 MB |

**总占用**:~255 MB / 16 GB = **~1.6%** — 非常宽松

### 大 batch 估算
- batch=64:×2 = ~510 MB
- batch=128:×4 = ~1 GB
- batch=256:×8 = ~2 GB
- 留 50% 余量给 cuDNN/RNG → **batch_size=128 安全**

## 训练时长估计

### MetalCytoToxDB Ru 子集
- 复合物:~3000
- 复合物 × 多 cell lines × 多 time points:训练样本 ~10k

| batch_size | epoch 步数 | 单步时间 (7800 XT) | epoch 时间 |
|---|---|---|---|
| 32 | ~313 | ~0.3s | ~95s |
| 64 | ~157 | ~0.5s | ~80s |
| 128 | ~78 | ~0.9s | ~70s |

**50 epochs 总训练时间**:
- batch=32: ~80 分钟
- batch=64: ~70 分钟
- batch=128: ~60 分钟

### MetalCytoToxDB 全量
- 复合物:9406
- IC50 样本:35,567
- 训练样本 ~30k

| batch_size | epoch 步数 | epoch 时间 |
|---|---|---|
| 32 | ~940 | ~5 min |
| 64 | ~470 | ~4 min |
| 128 | ~235 | ~3.5 min |

**50 epochs 全量**:~3 小时

## 单次前向 / 反向吞吐

### D-MPNN forward
- batch=32: ~2000 samples/sec
- batch=64: ~3000 samples/sec
- batch=128: ~4500 samples/sec

### EGNN forward (含 3D scatter)
- batch=32: ~1500 samples/sec
- batch=128: ~3000 samples/sec

### 3D conformer generation (offline)
- **关键瓶颈**:RDKit ETKDGv3 + MMFF 一次约 **0.5-2 秒/分子**
- 9000 复合物 × 多个构象 → **2-3 小时**(单核)
- **优化**:并行 8 worker → ~20 分钟
- 一次性计算,缓存到 disk

## 内存预算汇总

| 用途 | 占用 | 占比 |
|---|---|---|
| 模型 + 优化器 | 25 MB | 0.16% |
| 激活 (batch=128) | ~1 GB | 6.25% |
| 3D conformer 缓存 | ~500 MB | 3.1% |
| DataLoader prefetch | ~2 GB | 12.5% |
| ROCm runtime + PyTorch | ~3 GB | 18.75% |
| **已用** | **~6.5 GB** | **40%** |
| **余量** | ~9.5 GB | 60% |

## FP16 / BF16 训练

启用混合精度 (BF16) 后:
- 激活占用减半 (~1 GB → ~500 MB at batch=128)
- 训练速度 +30-50% (BF16 利用率高)
- ⚠️ 数值稳定性:BF16 训练 EGNN 可能不稳 — 用 **fp32 master weights + BF16 forward** 的 mixed precision 策略

## 关键决策

1. **Batch size**:默认 **64**(训练稳定 + 利用率)
2. **Mixed precision**:**开**(BF16 + fp32 master)
3. **DataLoader workers**: **4-8**(Python GIL + RDKit GIL 共存)
4. **3D conformer**: **预计算 + 缓存**(不要在 DataLoader 里跑 ETKDGv3)
5. **EMA decay**: 0.999 (跟 MolFlow-Triton 一致)

## 不现实的事

- ❌ **batch_size > 256** — 即使模型小,Python 端 RDKit + PyTorch dispatch 开销会成为瓶颈
- ❌ **fp32 全精度** batch=128 + 3D — 余量够但没意义
- ❌ **跨多 GPU** — 只有 1 张卡

## 监控指标

训练期间看:
- `train/loss` (回归 + 分类)
- `train/pic50_rmse`
- `val/auc` (主指标)
- `lr` (cosine schedule)
- `ema_w_norm`
- GPU util / VRAM (`rocm-smi` 或 `torch.cuda.memory_summary()`)

## 与 MolFlow-Triton 训练比较

| | MolFlow-Triton QM9 | Mol-Metal Ru |
|---|---|---|
| 训练样本 | 105k | ~10k |
| Batch | 32 | 64 |
| 单 epoch | ~200s (4 EGNN layers, 29 atoms) | ~80s (3 D-MPNN + 3 EGNN, ~25 atoms) |
| 模型大小 | 779k | ~1.75M |
| Epochs | 5 (loss 0.02) | 50 (target loss <0.1) |
| 总训练时间 | ~17 min | ~70 min |

MolFlow-Triton 是 pretraining benchmark;Mol-Metal 是 production training。
**7800 XT 足够**。

## 测试建议

在开始大规模训练前,跑一个**mini-test**:
- batch=4, 5 batches(20 个样本)
- 验证 forward + backward + metric 链路无 bug
- 跑一次 3D conformer 生成看 RDKit 出错率

估计 mini-test 时间:**5 分钟**。