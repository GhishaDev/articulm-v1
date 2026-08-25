# ArticuLM-V1 训练报告：pseudo_strength_v2 对照实验

> Run 目录：`runs/baseline_v2_strength`（GPU 2，`CUDA_VISIBLE_DEVICES=2`）
> 配置：`config/train_v1_50m_strength_v2.yaml` + `config/model_v1_50m.yaml` + `config/data_v1_strength_v2.yaml`
> 目的：与归档基线（`archive/baseline_v1_20260821_synthetic/`）**单变量对照**，
> 唯一差异是 strength 规则版本（v2 去除了中文隐式 ×0.92 系数，见 docs/17 §7）

## 1. 实验设计

| 项 | 基线（v1，已归档） | 本 run（v2） |
|---|---|---|
| 数据切分 | data/{train,validation,test}.jsonl | data/v2/{...}（**句级完全一致**，126,000/6,009,118 tokens） |
| strength 规则 | pseudo_strength_v1（中文全乘 0.92） | pseudo_strength_v2（重音乘子仅英文） |
| 模型/超参/seed | 相同（49,859,639 参数，fp16，AdamW 3e-4 cosine） | 相同 |
| strength 分布（test） | zh mean 64.84 / en mean 72.57 | zh mean **70.41** / en mean 72.57（英文逐 token 不变） |

**判读口径**：两个 run 的 viseme 指标应基本一致（viseme 标签相同）；差异集中在
strength 维度--重点对比 strength MAE、分类别 MAE（尤其中文类）、以及
strength 输出分布的 zh/en 可比性。

## 2. 门禁（CLAUDE.md）

- 模型/损失实现未改动（仅数据值变化）：全量测试 124 passed（GPU 2 上验证）
- 数据校验 `articulm.data.validate --config config/data_v1_strength_v2.yaml` exit 0
- tiny-overfit 门禁：架构级门禁已通过（runs/tiny_overfit_gate，10/10），
  本次无代码路径变化，不重复执行

## 3. 环境与 GPU

- GPU 2（训练）/ GPU 3（评估）；GPU 0/1 已坏（句柄异常 / ECC 错误）
- fp16 混合精度（V100 auto）
- GPU 监控：`scripts/gpu_monitor.sh runs/baseline_v2_strength 30`（30s 采样至
  `runs/baseline_v2_strength/logs/gpu_usage.csv`）

## 4. 训练进度（实时）

启动时间：2026-08-22 11:02

| 时间 | 事件 |
|---|---|
| 11:02 | 训练启动（GPU 2） |
| 11:06 | 数据加载完成：126,000 句 / 6,009,118 tokens（231s，与 v1 run 一致） |
| 11:07 | 词表构建完成（271s） |
| 11:28 | 首个训练步（step 50/100，指标与 v1 起步一致：acc 0.098/0.281） |
| 11:29 | **训练崩溃**：uncorrectable ECC error @step 100 |

## 4.0.5 UUID 钉卡解决映射紊乱（2026-08-22 16:21，训练进行中）

15:47 的重跑仍崩溃：内核日志显示 pid 8216（钉 `0000:82:00.0`）实际执行在
**PCI 03:00.0**（序列号 8649，昨天 6 个 ECC 的卡）。即 **PCI 地址钉卡也会被
死卡（02:00 掉总线）扰乱映射**，数字索引 + PCI 地址两种方式全部失灵。

**解法：CUDA_VISIBLE_DEVICES 使用 GPU UUID**。实测（16:20）：
- UUID 钉卡负载精确落在目标卡（序列号 7200 显示 1425MB/100% util，其余卡空闲）
- 持续矩阵乘 + 20s 存活测试通过

16:21 以 `CUDA_VISIBLE_DEVICES=GPU-a4dd272a-9b88-362a-ec49-0c4be9dbc12b`
重启训练（PID 9978，env 已核验）。

另：`scripts/gpu_recovery_at_boot.sh` + 用户 crontab `@reboot` 已部署--
机器重启后自动按序列号找卡、UUID 钉卡、负载+落卡核验、通过才拉训练
（含 GPU 0 拖垮整体 nvidia-smi 查询的容错），日志在 `logs/gpu_recovery/`。

## 4.0.4 事故复盘更正 + 钉卡重跑（2026-08-22 15:47，训练进行中）

**更正之前的部分结论**（串号对照内核日志 kern.log 与 nvidia-smi）：

| nvidia-smi idx | PCI | 序列号 | 实际状态 |
|---|---|---|---|
| 0 | 02:00.0 | ...8670 | Xid 79 GPU 掉出 PCIe 总线（13:42），硬件级故障 |
| 1 | 03:00.0 | ...8649 | 6 个 ECC（08-21），硬件故障 |
| 2 | 82:00.0 | ...7200 | **可能从未坏过**：v1 基线 3 小时训练就是它跑的；此前两次"GPU 2 崩溃"实为张冠李戴 |
| 3 | 83:00.0 | ...8691 | 两次崩溃真凶（Xid 48 x8 + Xid 13，pid 与训练进程吻合），硬件故障 |

**根因**：CUDA 设备枚举会排除已死 GPU（02:00），因此 `CUDA_VISIBLE_DEVICES=2`
（无论加不加 PCI_BUS_ID）实际选中枚举列表第 3 张 = 83:00。此前"PCI_BUS_ID 可防漂移"
的结论不完整。**正确钉卡方式：`CUDA_VISIBLE_DEVICES=0000:82:00.0`（完整 PCI 地址）**。

**驱动层结论**：驱动（535.309.01）本身无问题--同一驱动在 82:00.0 上完成了
3 小时 v1 训练；Xid 79/48/13 均为硬件级错误，重装驱动无法修复 0/1/3 三张坏卡。

15:47 用 PCI 地址钉卡重启训练（PID 8216，`CUDA_VISIBLE_DEVICES=0000:82:00.0`，
env 已核验）。评估也将使用同一张卡（训练结束后）。

## 4.0.3 GPU 2 冷重启后仍崩（2026-08-22 15:09，训练再次中止）

- 冷重启后 GPU 2 通过了短时实测（矩阵乘），但正式训练 **step 50（GPU 上卡
  不足 1 分钟）再次撞 uncorrectable ECC** 崩溃。结论：GPU 2 硬件坏，断电无效
- 当前状态：GPU 0 句柄坏 / GPU 1 实测可用（ECC 0）/ GPU 2 ECC 坏 / GPU 3
  无法初始化。**唯一可用：GPU 1**
- 已按预案停止监控与定时任务，等待用户决定（GPU 1 是最后一张卡，训练评估
  只能共用或评估走 CPU）

## 4.0.2 断电冷重启与重跑（2026-08-22 14:41）

- 用户断电冷重启机器（~11:55 完成）。核验：**GPU 0 仍句柄错误、GPU 3 CUDA 无法初始化**
  （比 ECC 更严重）、**GPU 1 / GPU 2 恢复可用**（ECC 计数器清零 + torch 实测
  2048x2048 矩阵乘通过）。当前可用：GPU 1、GPU 2
- 14:41 重启训练：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`，
  已通过 /proc/PID/environ 确认环境变量传入（PID 5292）
- 评估改用 GPU 1（原计划 GPU 3 已不可用）

## 4.0 事故记录（2026-08-22 11:29，训练中止）

1. **实际落卡错误**：启动命令 `CUDA_VISIBLE_DEVICES=2` 未按预期落在物理 GPU 2，
   实际落在 **GPU 3**（GPU 监控证实：GPU 2 全程空闲，GPU 3 在 11:28:54 达
   95%/11.4GB/176W）。原因：CUDA 默认 FASTEST_FIRST 枚举，坏卡导致索引位移；
   v1 那次 =2 恰好对齐物理 2，纯属侥幸。**必须加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`**
2. **GPU 3 撞 2 个 uncorrectable ECC**（计数器确认），训练于 step 100 崩溃
3. **事后核验四卡全部不可用**：
   - GPU 0：设备句柄错误（08-21 起）
   - GPU 1：6 个 ECC（08-21）
   - GPU 3：2 个 ECC（本次新增）
   - GPU 2：ECC 计数器为 0，但实测 `PCI_BUS_ID+VISIBLE=2` 下 torch 分配/计算
     直接报 uncorrectable ECC error
4. 处置：按预案不自动重启；GPU 监控与定时任务已停止，等待用户决定
   （建议：**断电冷重启**，非 reboot--ECC 状态需掉电清除；若冷重启后仍坏，
   一天之内 3 张 ECC + 1 张句柄异常，高度怀疑供电/主板/散热系统性问题，需报修）

## 4.0.1 已完成的不受影响部分

- pseudo_strength_v2 语料与切分（data/v2/）已完整构建并校验，不受本次事故影响
- v1 基线与归档不受影响
- 重启机器后可用以下命令重跑（注意 PCI_BUS_ID）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 nohup setsid \
    .venv/bin/python -m articulm.train \
    --config config/train_v1_50m_strength_v2.yaml \
    --run-dir runs/baseline_v2_strength --device cuda \
    > runs/baseline_v2_strength_train.log 2>&1 &
```

### 4.1 训练指标（定期更新）

| 更新时间 | step / epoch | loss | viseme_acc | strength_MAE | lr | grad_norm | tokens/s | GPU util (均/峰) | 显存峰 | 功率(均/峰) | 温峰 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| （待更新） | | | | | | | | | | | |

### 4.2 验证曲线（每 2000 步）

| step | val_viseme_acc | val_macro_F1 | val_weighted_F1 | val_strength_MAE | val_strength_RMSE |
|---|---|---|---|---|---|
| （待更新） | | | | | |

监控命令：

```bash
tail -f runs/baseline_v2_strength/logs/events.jsonl
grep '"train_step"' runs/baseline_v2_strength/logs/events.jsonl | tail -5
grep '"validation"' runs/baseline_v2_strength/logs/events.jsonl | tail -5
tail -20 runs/baseline_v2_strength/logs/gpu_usage.csv
```

### 4.3 最终训练记录（08-22 16:21 启动，18:34 完成）

- UUID 钉卡运行于 82:00.0（序列号 7200）全程稳定：利用率均值 96%/峰 100%，
  显存 11.9GB，功率 119-217W，温峰 82°C，有效训练 2.2 小时
- **早停于 step 14000，best@4000（val macro F1 0.99981）**，无 NaN
- 与 v1（best@8000，停于 18000）相比收敛更快（warmup 期间即达最优）

## 5. 最终评估与对照（已完成）

评估命令（GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m articulm.evaluate \
    --checkpoint runs/baseline_v2_strength/checkpoints/best.pt \
    --data data/v2/test.jsonl --label-set synthetic \
    --out-dir runs/baseline_v2_strength/reports/synthetic_test
```

对照表（与归档基线同口径）：

| 指标 | v1 基线 | v2 本 run | 差异 |
|---|---|---|---|
| viseme accuracy | 0.99988 | 0.99985 | -0.00003（噪声级） |
| viseme macro F1 | 0.99984 | 0.99981 | -0.00003（噪声级） |
| 误分类 token 数 | 40 / 333,295 | 50 / 333,295 | 同量级，viseme 标签相同，符合预期 |
| strength MAE（总体） | 0.340 | 0.356 | +0.016 |
| strength MAE（中文 253,345 tok） | 0.238 | 0.291 | +0.053 |
| strength MAE（英文 79,950 tok） | 0.663 | **0.561** | **-0.102（改善 15%）** |
| best_step / 停止步 | 8,000 / 18,000 | 4,000 / 14,000 | v2 收敛更快 |

### 5.1 结论

1. **viseme 侧完全符合预期**：两组 run 指标差异在噪声范围（viseme 标签完全相同）
2. **strength 侧的三个观察**：
   - 英文 MAE 显著改善（0.663 -> 0.561）：英文标签两版完全一致，改善来自
     zh/en 共享表征的迁移效应--v2 的中文标签分布（去掉 ×0.92 压缩）与英文
     更可比（70.4 vs 72.6），表征空间中两语言的强度尺度对齐了
   - 中文 MAE 略升（0.238 -> 0.291）：绝对值仍在 0.3 以内；相对误差
     （MAE/均值）0.37% -> 0.41%，两版模型都把各自标签学到了位
   - 总体 MAE 0.356 vs 0.340：中文 token 占 76%，中文项主导
3. **v2 规则本身的评价**：去隐式 ×0.92 后，zh/en 强度尺度可比性更好，且
   英文泛化反而受益。若下游口型动画对中文强度绝对量级敏感（v1 中文整体
   偏低 8%），v2 更合理；后续 Human Gold 微调建议以 v2 语义为基准
4. 评估明细：`runs/baseline_v2_strength/reports/synthetic_test/`


---

## 6. fast 配方重训（2026-08-24 14:38 启动）

基于 §5 之后的优化复盘（docs 与 config/train_v1_50m_gpu_fast.yaml 头注释），
用 fast 配方重训同一份 v2 数据：max_steps 15k（原 200k）、warmup 0.03（原 0.05）、
patience 3（原 5）、eval/checkpoint 每 1000 步（原 2000）、
**save_best_by val_composite**（原 val_viseme_macro_f1）、编码缓存已预热。

预期：启动 ~5 分钟（原 26 分钟缓存命中）+ 训练 ~1 小时（原 2.2 小时）。

| 时间 | 事件 |
|---|---|
| 14:38 | 训练启动（UUID 钉卡 82:00.0，PID 129969） |
| 14:42 | 数据加载（228s）+ 词表（268s），与历史一致 |
| 15:03 | 编码缓存 **miss -> 重建(saved)**：预热时未带 source_weights 导致键不匹配，本次仍付 ~21min 编码（缓存已按本配置键重建，后续同配置可命中） |
| 15:04 | 首个训练步（step 50，上卡正确：idx2/11.4GB） |
| 15:09 | step 800：loss 0.327 / acc 0.9996 / MAE 0.97，lr 已到峰值 3e-4（warmup 450 步） |
| 15:42 | step 5200；val_composite 单调爬升：best 更新到 step 5000（0.99659），strength MAE 0.319 持续改善（选型不再被 macro F1 噪声欺骗） |

### 6.1 训练指标

| 更新时间 | step / epoch | loss | viseme_acc | strength_MAE | val_composite | tokens/s | GPU util | 温峰 |
|---|---|---|---|---|---|---|---|---|
| 15:24 | 2700 / 14 | 0.326 | 0.9999 | 0.74 | 0.99414 (best@1000=0.99484) | 72k | 98%/100% | 82°C |
| 15:44 | 5400 / 28 | 0.325 | 0.9999 | 0.55 | 0.99659 (best@5000) | 72k | 97%/100% | 82°C |

val_composite 验证曲线：

| step | macro F1 | strength MAE | val_composite |
|---|---|---|---|
| 1000 | 0.99959 | 0.475 | 0.99484 |
| 2000 | 0.99963 | 0.549 | 0.99414 |
| 3000 | 0.99975 | 0.374 | 0.99601 |
| 4000 | 0.99980 | 0.329 | 0.99651 |
| 5000 | 0.99977 | 0.319 | 0.99659 |
| 6000 | 0.99974 | 0.307 | 0.99667 |
| 7000 | 0.99980 | 0.307 | 0.99673 |
| 8000 | 0.99982 | 0.307 | 0.99675 |
| 9000 | 0.99985 | 0.255 | 0.99730 |
| 10000 | 0.99982 | 0.233 | 0.99749 |
| 11000 | 0.99984 | 0.193 | 0.99790 |
| 12000 | 0.99986 | 0.190 | 0.99795 |
| 13000 | 0.99985 | **0.181** | **0.99803** |

### 6.2 三方对照（完成）

训练于 16:56 结束：max_steps 15000 跑满（未触发早停，composite 全程单调改善到末步），
best@15000（composite 0.99804）。

| 指标 | v1 基线 | v2 原 run | v2 fast |
|---|---|---|---|
| viseme acc / macro F1 | 0.99988 / 0.99984 | 0.99985 / 0.99981 | **0.99992 / 0.99990** |
| strength MAE 总体 | 0.340 | 0.356 | **0.180** |
| strength MAE 中文 | 0.238 | 0.291 | **0.142** |
| strength MAE 英文 | 0.663 | 0.561 | **0.303** |
| best_step（选型指标） | 8000（macro F1） | 4000（macro F1） | 15000（val_composite） |
| 总时长 | ~3.0h | ~2.2h | ~2.4h（含 21min 编码 miss；命中后 ~1.8h） |

### 6.2.1 30k 延长重训（2026-08-24 17:01 启动）

15k run 到末步 strength MAE 仍单调下降（未触达平台），故 max_steps 提到 30000
重跑（同 seed、同 fast 配方，编码缓存应命中）。

| 时间 | 事件 |
|---|---|
| 17:01 | 训练启动（UUID 钉卡 82:00.0，PID 137053，config train_v1_50m_strength_v2_30k.yaml） |
| 17:07 | **编码缓存命中**（hit @368s，启动 ~6min vs 原 26min） |
| 17:32 | step 3400；val 3000 = composite 0.99619 / MAE 0.349（注：与 15k run 同 step 不同值，因 cosine 按 30k 规划，lr 衰减更缓，符合预期） |

val_composite 曲线（30k 版）：

| step | macro F1 | strength MAE | val_composite |
|---|---|---|---|
| 1000 | 0.99950 | 0.811 | 0.99139 |
| 2000 | 0.99948 | 0.595 | 0.99353 |
| 3000 | 0.99968 | 0.349 | 0.99619 |
| 4000 | 0.99972 | 0.301 | 0.99671 |
| 5000 | 0.99975 | 0.328 | 0.99647 |
| 6000 | 0.99976 | 0.442 | 0.99534 |

30k 版结果（早停 @7000，best@4000，composite 0.99671）：test 集 acc 0.99983 /
macro F1 0.99977 / MAE 0.298（zh 0.212 / en 0.570）。

**结论：30k 加长版不如 15k 版**。

| 对比 | 15k 版（best@15000） | 30k 版（best@4000） |
|---|---|---|
| strength MAE 总体 | **0.180** | 0.298 |
| zh / en MAE | 0.142 / 0.303 | 0.212 / 0.570 |
| macro F1 | **0.99990** | 0.99977 |

原因：cosine 按 30k 规划后 lr 衰减变缓（warmup 450->900 步、整体曲线拉长），
在 4000-7000 步区间 lr 仍偏高，导致 MAE 震荡（0.30↔0.50），patience 3 在
step 7000 提前触发早停，lr 还没降到让 strength 单调收敛的低位就停了。
15k 版衰减更快，lr 在后续步数降到位，MAE 一路单调降到 0.181。

**教训：对本任务，“更多步数”必须配“足够快的衰减”。** 单纯提高 max_steps 会拉长
cosine、推迟有效衰减，反而更差。15k 配方即为当前最优；`runs/baseline_v2_strength_fast/checkpoints/best.pt`
（step 15000，MAE 0.180）保持为最强模型。

### 6.3 结论

1. **fast 配方三项优化全部达成**，且收益超出预期：
   - **val_composite 选型**是最大单项收益来源：strength MAE 从原 v2 的 0.356
     降到 **0.180（-49%）**，因为选型跟住了 strength 的持续收敛（原 macro F1
     选型在 MAE 0.359 处就停住了）
   - **短调度 + cosine 满 15k**：strength MAE 到末步仍单调下降（0.475->0.181），
     说明 lr 衰减到低位是关键，15k 步都还没到平台
   - **viseme 也小幅改善**（0.99984 -> 0.99990）：更充分的有效训练 + composite
     选型兼顾了两头
2. **编码缓存**首次启用时 miss（预热漏传 source_weights，键不匹配），已按本配置键
   重建；下一轮同配置 run 会命中（启动从 ~26min 降到 ~5min）
3. **后续建议**：strength MAE 在 15k 步仍下降，若追求极致可将 max_steps 提到
   20-25k（成本仍远低于原 200k 配方）；但已显著优于两个基线，HG 微调可基于
   `runs/baseline_v2_strength_fast/checkpoints/best.pt` 作为起点
