# Skill · FNO 实验（公开 NS64）

## 这份说明给谁看

给要把 **Fourier Neural Operator** 接到已验证的频谱卷积上、并在**官方 Navier-Stokes 数据**上交相对 L2 的人。也给科学机器学习方向的评审：看我们有没有换数据、有没有用同一套 SUPA 算子组满 ≥4 层。

**目的**：用必选算子组装 FNO-NS 前向，在未改动的公开集上把相对 L2 从刚接入时的 0.041835 做到 **0.035012**，并画出对得上成绩的流场图。  
**价值**：神经算子 / PDE 代理模型的人能拿走「公开集协议、残差头、频域损失、演示必须用官方测试样本」这一套，避免用自建数据把成绩讲漂亮。

操作总入口：[`../../skill.md`](../../skill.md)。评测公式旁注：[`../fno_eval_protocol.md`](../fno_eval_protocol.md)。

---

## 1. 技术背景

FNO 把卷积换成频谱卷积：每层先抬到宽通道，做 SpectralConv，再非线性。赛题进阶要求 Fourier Layer ≥ 4，任务是二维涡度：前 10 帧预测下一帧，分辨率 64×64。

**数据必须是官方文件** `navier_stokes_v1e-3_N1200_T20.pt`（公开 NS64）。划分训练 1000 / 测试 128，种子 `20260722`。包内不带这份约 376MB 的 `.pt`，复现需自备。自建 `ns_like_v2` 只做工程旁注，**不能写成公开成绩**。

网络：4 层 Fourier Layer，width=32，modes=16。推理调用 `spectral_conv2d_supa`，不是另写一套 `torch.fft` 交差。权重：`fno_ns/checkpoints/fno_ns_public_demo.pt`。

---

## 2. 优化思路（在官方 GPU 与官方数据上怎么压）

分两条线：**算子复用**（快）和 **同一份官方数据上的精度**（准）。

| 思路 | 具体做法 | 为什么 |
|------|----------|--------|
| 层内复用必选算子 | `FourierLayer` 调同一套 fused SpectralConv | 进阶分看的是算子有没有真正进模型 |
| 设备常驻链 | `forward_supa_chain`：`to_cpu=False` 贯穿四层，最后再回 CPU | 每层 D2H 一次，四层就是四次冤枉拷贝 |
| 预热计划与 BN 统计 | `prepare_supa_eval()`：suFFT plan warmup；InstanceNorm 的 `running_mean/var` 显式 `.to("supa")` | `nn.Module.to()` 不会搬 IN 的 running 统计 |
| 残差头 | 预测相对最后一帧输入的增量，再加回去 | 公开 NS 帧间变化大，直接回归整场更晃 |
| 不改官方 `.pt` | 只改网络、损失、增广 | 成绩必须和别人在同一文件上可比 |
| 周期平移增广 | 训练时对涡度场做周期 shift | 频谱方法天然周期，增广对齐归纳偏置 |
| 频域加权 + H⁻¹ | 后期主要更新 spectral 权重，用 Sobolev 型损失压高频误差 | 肉眼平滑不够，测试 L2 吃高频 |
| 演示用官方测试样本 | `render_official_demo.py` 在 128 个测试样本上前向，选 L2 最接近主报的典型样本 | 相对误差热图和「随便一张生成场」都会误导评委 |

精度结果：同一官方数据、同一划分，**0.041835 → 0.035012**（约 16.3%）。  
推理吞吐旁注（batch=16，公开 NS64）：约 1.60M grid_points/s；chain 一致性相对误差建议 ≤ 1e-4。训练长路径用 CPU / `use_supa=False` 与提交 checkpoint 一致，如实标注，不把 SUPA mul 塞进每个 epoch（Host↔Device 风暴）。

---

## 3. 技术与问题

**用到 / 学到的技术**

- 神经算子：频谱卷积当层，而不是 CNN 堆层。
- 公开集协议：文件名、划分、种子、10→1 步，写进 `results/data_disclosure.md`。
- 可视化：预测 vs 真值、最好 / 典型 / 最差；典型样本选 per-sample L2 最接近主报的（例如与 0.035012 对齐的那一枚），不要用误差最大的当封面。
- 中文图题需要宿主机有中文字体（如文泉驿）；没有会出方块字。

**遇到的问题**

| 问题 | 学到什么 |
|------|----------|
| 生成数据 L2 很好看 | 不能和公开 NS64 并列当成绩 |
| `test_forward.py` 默认可能走生成缓存 | 正式演示必须 `render_official_demo.py` + 官方 `.pt` |
| 相对误差热图一片红 | 真值接近 0 的像素会把相对误差炸开，不适合当主图 |
| 训练误走每层 SUPA mul | 极慢；长训回 CPU einsum，SUPA 留给推理和单测 |
| InstanceNorm 在 SUPA 链上错 | running 统计还在 CPU |
| 想用官方脚本直接当 FNO 性能 | 官方算子脚本测的是参考实现，不是我们的 FNO 链 |

**怎么验**

```bash
# 算子已编译且 accuracy 通过之后
cd fno_ns
# 自备官方数据到 fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt
python3 render_official_demo.py
python3 visualize.py
```

主报数字以 `results/summary.json` → `fno_ns.public_ns64` 为准。流场图在 `demo/media/` 与 `results/figures/`。

---

## 4. 能力边界

- Fourier Layer 不少于 4 层。
- 海选和提交以可复现前向 + 公开集 L2 为主；把训练全程绑在 SUPA mul 上不是提交策略。
- 禁止把 SOL、tune、自建集写成正式 FNO 得分。

若这份 FNO 实验说明对你有用，欢迎给仓库点 Star 收藏：https://github.com/junfennie162-sketch/birensupa-spectralconv
