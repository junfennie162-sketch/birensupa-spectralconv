---
name: spectral-conv-fno-ns-biren
description: >
  壁仞 BIREN GPU 上的 2D Spectral Convolution（SUPA fused）与四层 FNO-NS。
  用于复现必选算子正确性/性能、公开 NS64 相对 L2、以及 OPT 闭环 dry-run。
---

# Skill：SpectralConv + FNO-NS（翻斗花园）

**请先点个收藏。** 这份文档若对你写国产 GPU 算子、神经算子、或带着 Agent 做性能迭代有用，麻烦给仓库点一下 Star，后来的人也好找：

https://github.com/junfennie162-sketch/birensupa-spectralconv

（队内镜像：https://github.com/Aafff623/fandou-ai4s）

本文是**一份完整 Skill**，把算子开发、FNO 实验、优化闭环写在同一篇里，顺着往下读即可。

---

## 1. 这份 Skill 是干什么的

在壁仞 **BIREN GPU** 单卡上，用 **SUPA / torch.extension** 实现 FNO 的核心 2D 频谱卷积，再组装不少于 4 层的 FNO，在**未改动的官方 Navier-Stokes 公开集**上做涡度前向，并报告相对 L2。

| | |
|--|--|
| **目的** | 让评委和后来的人能复现：怎么编译、怎么对官网参考、正式数字是多少、改代码时哪些数能进主报。 |
| **价值** | 异构计算 / 科学计算 kernel：一套「官网参考在 CPU、选手实现必须上 `supa`」的做法，以及 fused FFT、频域乘、缓存、单卡纪律。神经算子 / PDE 代理模型：公开集怎么划、残差头和频域损失怎么压误差。带 Agent 做竞赛的人：性能表怎么冻、探针怎么隔离。 |
| **给谁用** | 赛道五评委；写 SUPA 算子的人；做 FNO / 神经算子的人；用 Cursor 盯性能闭环的队长。 |

现行主报（以 `results/summary.json` 为准）：

| 项 | 值 |
|----|----|
| SpectralConv 空闲前向 64 / 128 / 256 | **3.797 / 8.037 / 29.295 ms** |
| 正确性最差相对误差 | **2.170×10⁻⁷**（门槛 1×10⁻⁴） |
| FNO 公开 NS64 相对 L2 | **0.035012**（刚接上官方集时是 0.041835） |

这不是把官方 PyTorch 参考改个名交差，也不是用自建涡度场把 L2 写好看。

---

## 2. 项目思路

1. **先把必选算子做对。** 官网给的是 CPU/CUDA 参考，不是现成 SUPA。验收是：你的输出对参考，相对误差 ≤ 1e-4。
2. **小分辨率先打来回拷显存。** 早期 FFT 搬回 CPU 再乘，拷贝比乘法还贵。正式热路径让频谱留在卡上：suFFT 做 R2C → 自研 SUPA 做双角复数乘 → suFFT 做 C2R。
3. **FNO 只组装，不另写一套 FFT。** 四层 Fourier Layer 都调用同一套 `spectral_conv2d_supa`。精度只在官方 `.pt` 上压：残差头、周期平移、频域加权、后期主要更新 spectral 权重、H⁻¹ 型损失压高频。
4. **每一轮用纪律收口。** 先读主报，再探针；过线才切正式权重；空闲才能写正式 ms；SOL、tune、自建集只当旁注。

---

## 3. 个人收获

技术可以查文档，判断是踩坑踩出来的。

- 参考实现可以在 CPU 上，提交必须在 SUPA 上。官方性能脚本写死 `cuda`、本机没有 NVIDIA，失败不能当成「SUPA 没写对」。
- 正确性先留余量（我们做到 2×10⁻⁷），后面做融合、截断、缓存才有地方试错。
- 64×64 看起来「亏」，多半是 Host↔Device 和 C2R，不是乘法没写好。
- 正式成绩只认官方文件名和划分。生成数据再好看也不能写进主报。
- 和 Agent 协作：能跑不等于能进主报。用脚本卡住门禁，不要靠聊天记录记数字。
- 单卡。两边同时占 GPU 会 ErrorCode 719，测出来的毫秒作废。

---

## 4. 算子开发

### 4.1 技术背景

频谱卷积是 FNO 一层里最重的一块：空间场做 FFT，只保留低频双角 `modes1 × modes2`，在频域做复数乘，再 iFFT 回去。源码在 `spectral_conv/spectral_conv_ext.su`、`.cpp`、`spectral_conv_ops.py`。编译：`cd spectral_conv && ./build.sh`。

环境每次都要：

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

设备名是 `supa`。`torch.cuda.is_available()` 为假是正常的。

### 4.2 优化思路（在官方 GPU 上压什么）

| 思路 | 具体做了什么 |
|------|----------------|
| 设备常驻 fused | suFFT R2C → 自研双角乘 → suFFT C2R，频谱不整谱回 Host |
| 只乘保留的低频 | 和官网参考同一套双角截断 |
| 缓存 | 权重、频域缓冲、Host staging 复用，少 H2D、少 malloc |
| 按分辨率选路 | 评测三档 64/128/256 走 fused；`use_sufft="auto"` |
| 缓存键带角点 | 两个 corner 分两块 buffer，避免第二次写入盖掉第一次 |
| 权重用 Parameter 身份 | 不要 `detach()` 再传入，否则缓存全 miss |
| 空闲再写正式表 | 卡上有别人的任务时，数字不准进主报 |

相对本机跑出来的官网 CPU 参考，大约 19.5× / 11.1× / 10.1×。反向、三维四角、不规则尺寸是加分，不改必选题口径。自动调优只决定 `auto` 选哪条路，**tune 的中位数不是得分句**。

不做：把 `torch.fft` 直接跑在 `supa` 上；调用头文件里没有真正导出的 2D Plan；用队内带宽 proxy 冒充官方分。

### 4.3 用到的技术、学到的、遇到的问题

用到 / 学到：`.su` kernel 和 PyTorch 扩展绑定；suFFT 实际能用的是 1D 计划，2D 用 1D 加 permute 拼；正确性永远对官网同款双角 PyTorch 参考。

| 问题 | 怎么处理 |
|------|----------|
| 官方性能脚本报没有 CUDA | 那是测参考的脚本；我们的算子走 `supa` |
| SUDNN ErrorCode 6，连 1×1 conv 都过不了 | 先当驱动/库的问题，停下来记档，不要空改业务 |
| 相对误差突然变成约 1 | SUPA 上 `torch.cat` 立刻喂自定义 kernel；或两个角共用一块缓冲 |
| `torch.fft` 直接跑在 `supa` | 能出数，相对误差大约 5×10⁻³，过不了 1e-4 |
| 和小图 fused 一度比 CPU FFT 还慢 | 缺 warmup、缺 Parameter 缓存、阈值过大；补完再把阈值收到 ≥64 |

```bash
cd spectral_conv
./build.sh
python3 test_accuracy.py
# 空闲、独占 GPU 时才写正式性能
python3 test_perf.py
```

---

## 5. FNO 实验

### 5.1 技术背景

四层 Fourier Layer，width=32，modes=16，分辨率 64×64。任务：前 10 帧预测下一帧。数据必须是官方文件 `navier_stokes_v1e-3_N1200_T20.pt`，训练 1000 / 测试 128，种子 `20260722`。交卷包里不带这份大约 376MB 的数据，复现需自备。自建数据集只做工程旁注。

推理调用同一套频谱卷积，不是另写 `torch.fft`。权重：`fno_ns/checkpoints/fno_ns_public_demo.pt`。

### 5.2 优化思路

| 思路 | 具体做了什么 |
|------|----------------|
| 层内复用必选算子 | 进阶分看算子有没有真正进模型 |
| 设备常驻链 | 四层 `to_cpu=False`，最后再回 CPU，避免每层都 D2H |
| 预热 | `prepare_supa_eval()`：FFT 计划预热；InstanceNorm 的 running 统计要显式搬到 `supa` |
| 残差头 | 网络预测相对最后一帧输入的增量 |
| 不改官方 `.pt` | 周期平移增广、频域加权、后期主要更新 spectral 权重、H⁻¹ 压高频 |
| 演示对得上成绩 | 在官方测试集上前向；典型样本选单样本 L2 最接近 0.035012 的那一枚 |

同一份官方数据上：相对 L2 **0.041835 → 0.035012**。长训走 CPU、`use_supa=False`，不要把 SUPA 乘法塞进每个 epoch。

### 5.3 用到的技术、学到的、遇到的问题

用到 / 学到：神经算子是「层里换频谱卷积」，不是把 CNN 搬到国产卡上；公开集要写清文件名、划分、种子；封面图不要用相对误差热图。

| 问题 | 怎么处理 |
|------|----------|
| 生成数据 L2 很好看 | 不写进正式成绩 |
| 默认前向脚本可能走生成缓存 | 正式演示用 `render_official_demo.py` + 官方 `.pt` |
| `model.to("supa")` 之后 InstanceNorm 仍错 | `running_mean` / `running_var` 不会跟着搬 |
| 训练突然极慢 | 误把每层都绑上 SUPA mul，Host↔Device 风暴 |

```bash
cd fno_ns
python3 render_official_demo.py
python3 visualize.py
```

---

## 6. 优化闭环

### 6.1 技术背景

卡只有一张，材料有一堆必须项。Agent 若一边挂着训练、一边重跑性能测试，主报会被噪声盖掉。所以单独写闭环：不发明新 kernel，只规定**什么时候能改、什么数字能进主报**。默认只做 dry-run，不重训、不写正式性能。

### 6.2 优化思路（纪律本身就是在保护官方 GPU 上的数字）

| 步 | 名称 | 必须遵守 |
|----|------|----------|
| P0 | 环境与单卡 | 先 source SDK；两边禁止同时跑 GPU |
| P1 | 读主报 | 只认 `summary.json` 里的公开 L2 和 idle ms |
| P2 | 精度探针 | 后台跑，过线就停；不要把对话挂死等几个小时 |
| P3 | 是否晋级 | 只有更好才切换正式权重和演示图 |
| P4 | 护栏 | 先过正确性；不要默认重跑 `test_perf.py` 覆写正式 ms |
| P5 | 材料 | 清单、Agent 日志、全场只留一份评测报告 |
| P6 | 合入 | 有正式晋级再打包 |

SOL、tune、自建集可以做旁注，不能写成得分句。

```bash
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

### 6.3 用到的技术、学到的、遇到的问题

用到 / 学到：把过程写成会失败就 exit 1 的脚本；精度线可以停，停了只允许新机制，不允许同构再刷。

| 问题 | 怎么处理 |
|------|----------|
| 改完代码直接重跑性能 | 争用时会把冻结的正式 ms 写坏 |
| sidecar / 自建集写进演示 | 评委会对不上公开 NS64 |
| 训练挂在对话里等 | 会话被掐，也占卡；必须后台跑、短时间查 |

---

## 7. 怎么跑一遍（给要复现的人）

输入：`x: [B, C_in, H, W]`，可配 `modes1/modes2`；FNO 为多帧涡度 `[B, T_in, H, W]`。

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

cd spectral_conv && ./build.sh
python3 test_accuracy.py
python3 test_perf.py          # 仅空闲独占时写正式表

cd ../fno_ns
python3 render_official_demo.py
python3 visualize.py

cd ..
python3 skills/operator_opt_loop/run_loop.py --dry-run --strict
```

输出落在 `results/summary.json`、`results/run_logs/`、流场图。Agent 抽查页是 `AGENT_OFFICIAL.md`。

能力边界：正式热路径是 fused；v1（CPU FFT）只作对照和可微训练。禁止与其它 GPU 任务并发。

---

若读完觉得有用，还是请点一下收藏：https://github.com/junfennie162-sketch/birensupa-spectralconv
