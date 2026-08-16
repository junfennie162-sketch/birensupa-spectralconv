# Skill · 算子开发（SpectralConv on BIREN SUPA）

## 这份说明给谁看

给要在**壁仞 BIREN GPU** 上交频谱卷积、或第一次把 PyTorch 参考改成 SUPA / `torch.extension` 的人。也给评审：看我们核心计算是不是真的上了国产卡，而不是只交一份 `torch.fft`。

**目的**：把 2D Spectral Convolution 做成可编译、可对参考、可测 64/128/256 的正式算子。  
**价值**：异构计算和科学计算 kernel 的人，能直接拿走「参考在 CPU、实现必须在 SUPA」的拆法，以及 fused 路径、缓存、单卡纪律这些在官方 GPU 上压出来的经验。

操作总入口仍是提交根 [`../../skill.md`](../../skill.md)。本页把压缩包里的「算子开发 Skill」写成完整中文。

---

## 1. 技术背景

频谱卷积是 FNO 一层里最重的算子。空间场做 FFT，只保留低频 `modes1 × modes2` 的「双角」，在频域做复数乘，再 iFFT 回去。官网 §3.1 给的是**原生 PyTorch 参考**（CPU 或 CUDA），用来对答案；选手必须自己写 **SUPA 或 Extension**，跑在设备名 `supa` 上，相对参考误差 ≤ 1e-4。

本机环境：SDK `1.11.0.0.rc2`，先

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
```

`torch.cuda.is_available()` 为 False 是预期。官方 `benchmark_performance()` 若写死 `torch.device("cuda")`，在这台机器上失败，不能当成算子没写 SUPA。

源码：`spectral_conv/spectral_conv_ext.su`、`.cpp`、`spectral_conv_ops.py`。编译：`cd spectral_conv && ./build.sh`。

---

## 2. 优化思路（在官方 GPU 上怎么压）

不是把 FFT「再实现一遍」，而是让 **BIREN 上少搬数据、计划暖和、乘法连续**。

| 思路 | 具体做法 | 为什么 |
|------|----------|--------|
| 设备常驻 fused | suFFT R2C → 自研 `spectral_mul` 双角复乘 → suFFT C2R，频谱不整谱回 Host | 64×64 上 H2D/D2H 曾经比乘法更贵 |
| 只乘保留的低频 | 按官网双角截断，不乘整谱 | 计算量和访存都按 modes 走 |
| 权重与谱缓存 | 同一组 Parameter / shape 不每步重新 H2D；频域 `_OUT_FREQ_CACHE`、Host staging `_HOST_OUT_CACHE` 复用 | 少 malloc、少 peak memory |
| `use_sufft="auto"` | 评测三档 64/128/256 都走 fused；极小图才退回 CPU FFT 对照 | 阈值是 sweep 出来的，不是拍脑袋 |
| 钉住 suFFT scratch | `sufftSetWorkArea` + 按 shape 的 workspace | 计划可复用，避免每次乱分配 |
| 缓存 key 带角点 | 两个 corner 分两块 buffer | 共用一块会被第二次写入覆盖，rel→1 |
| 权重用 Parameter 身份 | 不要 `weights.detach()` 再传入 | detach 换了 `id`，缓存全 miss |
| 单卡、空闲再写正式 ms | 与搭档仓禁止同时占 GPU | 争用时数字不能进主表 |

现行前向（warmup=10，iters=100，CPU 入 CPU 出）：**0.764 / 1.827 / 6.504 ms** @64/128/256；相对本机官网 CPU 参考大约 97.0× / 48.7× / 45.5×。上一主表 0.762 / 1.981 / 7.324 ms。正确性最差相对误差（默认裁剪路径）**7.162×10⁻⁶**。

加分项已做完、不改变必选题口径：反向 `SpectralMulFunction`、3D 四角前向、不规则尺寸。Auto-tune（`spectral_conv/tune.py`）只决定 `auto` 选哪条路径，**tune 中位数不是得分句**。

**明确不做（ROI 不够或会破坏正确性）**：`torch.fft` 直接跑在 `supa`；header 里没有导出的 `BuildPlan2d`；cuFFT 式 callback；用 SOL proxy 冒充正式分。

---

## 3. 技术与问题

**用到 / 学到的技术**

- SUPA kernel（`.su`）+ PyTorch Extension 绑定；复数用 float2、连续访存。
- suFFT 1D 计划：R2C / C2C / C2R 拼 2D，而不是幻想 2D Plan 已经能用。
- 正确性永远对官网同款双角 PyTorch 参考，而不是对自己的慢实现。
- 性能：warmup、同步计时、peak memory；formal 表冻结后只护栏、不默认重跑 `test_perf.py`。

**遇到的问题**

| 问题 | 学到什么 |
|------|----------|
| 参考脚本走 CUDA | 验收是「你的 SUPA 输出 vs 参考」，参考本身可以是 CPU |
| SUDNN ErrorCode 6，平凡 1×1 conv 也挂 | 先排除环境，再改业务；环境坏时停下来记档 |
| SUPA 上 `torch.cat` 立刻调自定义 kernel | 内存布局无效，rel≈1；要 contiguous，不要赌 cat |
| `torch.fft(..., device=supa)` | 相对误差约 5e-3，过不了 1e-4 |
| 小分辨率 fused 一度比 v1 慢 | 冷启动、缺 Parameter 缓存、阈值过大；warmup + 缓存后再把阈值收到 ≥64 |
| 双角共享一块输出缓冲 | 第二角覆盖第一角 |
| 和 `ai4s-n` 同时跑 | ErrorCode 719；单卡串行是硬纪律 |

**怎么验**

```bash
cd spectral_conv
./build.sh
python3 test_accuracy.py    # 相对误差 ≤ 1e-4
# 空闲、独占 GPU 时才允许写正式性能：
python3 test_perf.py
```

日志进 `results/run_logs/`。失败时先缩小 B/C/H/W 和 modes、固定种子，和 CPU 参考逐层对。

---

## 4. 能力边界

- 正式热路径是 fused suFFT + SUPA mul；v1（CPU FFT）只作对照和可微训练。
- 反向、3D 是加分，不是必选替代。
- 禁止与其它 GPU 任务并发。

若这份算子开发说明对你有用，欢迎给仓库点 Star 收藏：https://github.com/junfennie162-sketch/birensupa-spectralconv
