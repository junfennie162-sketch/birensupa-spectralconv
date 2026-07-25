# 算子优化待办 · 2026-07-25

> 接手给搭档（ai4s-f）的完整任务说明
> 关联：HANDOVER.md / SUBMISSION_MANIFEST.md / skill.md

---

## 0. TL;DR（一分钟读完）

当前已提交版本是 SpectralConv + FNO-NS 的**功能/正确性/性能已达标**的版本（5/5 准确，bs=1 下 5.32/13.69/52.64 ms，bs=16 下 0.88M/3.30M/4.93M grid_points/s，FNO L2=0.009516）。

**未跑通的优化路线**有 3 条（P0 ~ P2），按 ROI 从高到低：
- **P0**：SUDNN `nn.Conv2d` 环境性崩溃的规避方案（影响 FNO chain 评测）
- **P1**：`rfft2_sufft` SUPA-input correctness bug 的 C++ 层修复
- **P2**：Path B kernel fusion（已论证 ROI 太低，仅作记录）

每条都给出：**目标 → 思路 → 复现命令 → 验收标准 → 风险**。

---

## 1. 当前已实现的优化路径回顾（避免重复劳动）

### 1.1 SpectralConv 双路径

**v1 路径**（CPU 路线，min(H,W)<64 时走）：
- `torch.fft.rfft2` 在 CPU 上做 → SUPA 上做 `spectral_mul_supa_device` 频域乘 → CPU 上 `torch.fft.irfft2`
- 优点：小图快、稳定性高
- 缺点：H2D/D2H 多次

**fused 路径**（全 SUPA，min(H,W)≥64 时走）：
- `rfft2_sufft`（SUPA rfft kernel）→ `spectral_mul_supa_device` → `irfft2_sufft`（SUPA irfft kernel）
- 优点：大图快（无 H2D/D2H）
- 缺点：依赖 SUPA suFFT SDK

**自适应切换**：`resolve_use_sufft(height, width, "auto")` 根据 min(H,W) 自动选路径（阈值 64），可被 `_AUTO_TUNE_TABLE` 覆盖。

### 1.2 已实施的优化点（不要重做）

| 优化 | 位置 | 效果 |
|------|------|------|
| R1 频域 buffer 缓存 | `_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE` / `_OUT_FREQ_CPU_CACHE` | 减少 D2H 次数 |
| R2 auto 阈值 | `resolve_use_sufft` | 小图走 v1，大图走 fused |
| R3 SUPA 链路 | `forward_supa_chain` | FNO chain SUPA 化 |
| R4 `.detach()` 参数缓存 | `_weights_to_supa_cached` | -67% chain ms |
| R5 dual_out | `spectral_mul_dual_out` | 单 pybind 启动两个角点 kernel（-0.007 ms/call）|
| Auto-tune | `_AUTO_TUNE_TABLE` + `tune.py` | Pareto-best 配置 |
| Irregular 形状 | 9-shape 全过 | worst rel 3.92e-7 |
| 3D 扩展 | `spectral_conv3d_supa` | worst rel 1.07e-7 |
| Backward | `spectral_mul_autograd` | worst grad rel 6.25e-8 |

---

## 2. P0 · 修 SUDNN `nn.Conv2d` 偶发崩溃

### 2.1 问题描述

跑 `fno_ns/profile_chain.py` 或 `fno_ns/test_supa_chain.py` 时，SUDA 1×1 conv 在 plan 初始化阶段崩溃：

```
ERROR (SUDNN): .../SudnnPlanBuilder.h:667
Failed to finalize engine config descriptor, ErrorCode: 6, Sudnn Error
```

或者 `ErrorCode: 719` Fatal allocator error（更严重，会污染设备状态）。

**影响**：FNO chain 的逐层 forward_supa_chain 跑不起来。但单算子 `spectral_conv_combo/` 不受影响（不走 SUDA conv）。

**触发条件**：所有 `nn.Conv2d` 在 SUPA 上的 forward（即 FNO layer 里的 1×1 skip + lift + project 都中招）。

**根因分析**：
- 与代码改动无关（错误栈在 `at::supa::Conv_2d → nn.Conv2d.forward`）
- 即使最简单 `F.conv2d(4×32×32×32, 32×32×1×1, ...)` 也炸
- 推测：SUDA driver / device 状态从某次 `ErrorCode 719` 没完全恢复，或 SUDNN 库本身不稳定

### 2.2 接手时第一步：环境 reset

```bash
# 重启 Python kernel 或重启 container
# 然后单独跑确认单算子没受影响
cd /workspace/ai4s/submission/spectral_conv_combo
python3 test_accuracy.py       # 期望 5/5 通过
python3 test_perf.py           # 期望 5.32/13.69/52.64 ms
```

### 2.3 如果 SUDNN 仍然炸：fallback 方案

把 `fno_ns/model.py` 里所有 `nn.Conv2d(..., kernel_size=1)` 用 `torch.einsum` 替换：

```python
# 原
self.conv = nn.Conv2d(width, width, 1)

# 替换为
class EinsumConv1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c) * (1.0 / (in_c * out_c) ** 0.5))
    def forward(self, x):
        # x: [B, C, H, W]
        return torch.einsum('oi,bihw->bohw', self.weight, x)
```

需要替换的位置（看 `fno_ns/model.py`）：
- `FourierLayer.conv`（skip 分支，1×1 conv）
- `FNO2d.lift`（输入提升，1×1 conv）
- `FNO2d.project`（输出投影，1×1 conv）

**复现命令**：
```bash
cd /workspace/ai4s/submission/fno_ns
python3 test_forward.py   # 验证 L2 不退化（应仍 0.009516 左右）
python3 test_supa_chain.py  # 验证 SUPA chain 跑得起来
```

**验收**：替换后 L2 不退化 + SUPA chain 跑通。

### 2.4 风险

- einsum 在 SUPA 上走自定义 GEMM，性能可能比 nn.Conv2d 略差（差距约 5-10%）
- 但能跑就行，FNO L2 保持 0.009516 是底线

---

## 3. P1 · 修 `rfft2_sufft` SUPA-input correctness bug

### 3.1 问题描述

当 `rfft2_sufft` 的输入**已是 SUPA 驻留 tensor** 时，返回 garbage（rel≈1.4）。FNO chain 的逐层 rfft `rfft2_sufft(h_supa, ...)` 都会出错（h_supa 是上层的输出）。

之前 `test_accuracy.py` 一直过，是因为测试输入是 CPU tensor（自动转 SUPA），触不到 bug。

**已存在的临时方案**（Python 层，**保留**，不要删）：
```python
# spectral_conv_ops.py::spectral_conv2d_fused 内
if x.is_cuda or (hasattr(x, 'device') and str(x.device).startswith('supa')):
    x = x.detach().cpu().to('supa', torch.float32).contiguous()
```
代价：每次多一次 D2H+H2D ≈ 0.5 ms/call × 4 layers ≈ 2 ms/FNO forward。

### 3.2 目标

在 C++ 层（`spectral_conv_ext.cpp`）修掉 bug，让 SUPA 输入直接吃。

### 3.3 思路

`rfft2_sufft` 是 host-side `sufftExecR2C` 调用，需要 host 指针。问题在 `data_ptr<float>()` 当输入是 SUPA tensor 时返回的是 host-mirror 指针，sufft 读这个指针读到的不是 SUPA 上的数据。

修复方案：在 `rfft2_sufft` / `irfft2_sufft` 入口显式 `.cpu()` 强制 host-mirror：

```cpp
// spectral_conv_ext.cpp::rfft2_sufft 开头
torch::Tensor x_cpu = x.device().is_cuda() ? x.cpu() : x;
const float* x_ptr = x_cpu.data_ptr<float>();
// ... sufftExecR2C(x_ptr, ...)
// 最后输出回到 SUPA：output = output.to(x.device());
```

⚠️ **编译坑**：之前我试过类似修改报 `expected primary-expression before 'float'`。原因是 `data_ptr<float>()` 的语法在该文件其他位置工作正常，但这一处上下文有问题。建议：
- 把 `data_ptr` 调用抽到临时变量（避免 inline 调用）：
  ```cpp
  void* raw_ptr = x_cpu.data_ptr();
  const float* x_ptr = static_cast<const float*>(raw_ptr);
  ```
- 或显式加 `at::TensorAccessor`：
  ```cpp
  auto acc = x_cpu.accessor<float, 4>();
  ```
- 每次改完 `bash build.sh` 编译看错误，必要时 `git diff spectral_conv_ext.cpp` 对照 baseline

### 3.4 复现命令

```bash
cd /workspace/ai4s/submission/spectral_conv_combo
# 当前 bug 复现
python3 -c "
import torch, spectral_conv_ops as ops
x = torch.randn(2, 32, 64, 64).to('supa')
w = torch.randn(32, 64, 16, 16, dtype=torch.cfloat)
y = ops.spectral_conv2d_supa(x, w, w, 16, 16)
# 当前会返回 garbage
"
# 修复后：和 CPU 输入走相同路径，结果应该 rel < 1e-4
```

### 3.5 验收

- ✅ `test_accuracy.py` 5/5 通过
- ✅ 新增 SUPA-input test（写在 `test_supa_input_accuracy.py`，用例：把 CPU 输入 `.to('supa')` 后直接传 `spectral_conv2d_supa`）
- ✅ FNO chain `test_supa_chain.py` 跑通且 L2 接近 0.009516
- ✅ FNO chain perf 改善 ≈ 2 ms/forward（去掉 CPU workaround 的 D2H+H2D）

### 3.6 风险

- 改动 C++ 后编译失败概率较高，建议先在 `spectral_conv_ext.cpp` 里复制 `rfft2_sufft` 起一个新函数 `rfft2_sufft_v2` 做实验，原函数保留
- 修改完后跑全套测试确认无回归

---

## 4. P2 · Path B kernel fusion（已论证 ROI 太低，仅记录）

### 4.1 调研结论

写在 `ai4s-n/测试/提升版/results/run_logs/sdk_kernel_probe_2026-07-24.md`，关键点：

- SUPA kernel 编程能力 = CUDA + 显式 TCI tensor core 抽象，**可写 fused kernel**
- **真实 rfft+mul+irfft 三合一 fused kernel**：做不到（SUPA 不开源 FFT，TCI 是 tile 抽象不是蝶形）
- **拼缝级（B4/B5 fused irfft+pad / mul+irfft）**：天花板共 ~3 ms 改进
- **结论**：ROI 太低，不建议做

### 4.2 如果搭档想试

SDK 路径：`/usr/local/birensupa/sdk/1.11.0.0.rc2/supa/examples/{Simple,TensorCore}/`
- 看 `Simple/` 学会 `__global__` / `block_idx` / `warp_idx` / `warp_size` 写法
- 看 `TensorCore/` 学会 `tci::` TCI intrinsics
- `brcc` 编译：`brcc -O3 kernel.su -o kernel.o`

拼缝级 fused kernel 候选（按 ROI 排序）：
1. **mul+irfft fused**（节省 1 次 SUPA kernel launch + 1 次中间 buffer 写回）≈ -1 ms
2. **rfft+pad fused**（节省 1 次 launch + 1 次 buffer 分配）≈ -0.5 ms
3. **spectral_mul + iFFT pre-pad**（节省 corner buffer 切换）≈ -1.5 ms

加起来天花板 ~3 ms，相对当前 5.32 ms（64×64）就是 56% 加速，**对单个算子值得试**（前提是接受开发 1 周）。

### 4.3 复现命令

```bash
# SDK kernel 编程示例
ls /usr/local/birensupa/sdk/1.11.0.0.rc2/supa/examples/Simple/
cat /usr/local/birensupa/sdk/1.11.0.0.rc2/supa/examples/Simple/matrixMul.su | head -50

# 当前 fused 路径 perf baseline
cd /workspace/ai4s/submission/spectral_conv_combo
python3 test_perf.py
python3 test_perf_grid_points.py
```

---

## 5. 验证流程（接手后第一件事）

按顺序跑：

```bash
# 1. 资产检查
cd /workspace/ai4s/submission
bash scripts/maintain_assets.sh status

# 2. SpectralConv 单算子（确认 baseline）
cd spectral_conv_combo
bash build.sh
python3 test_accuracy.py             # 期望 5/5 通过
python3 test_perf.py                 # 期望 5.32/13.69/52.64 ms
python3 test_irregular_shapes.py     # 期望 9/9 通过
python3 test_backward.py             # 期望通过
python3 test_3d_accuracy.py          # 期望通过
python3 test_perf_grid_points.py     # 期望 0.88M/3.30M/4.93M grid_points/s

# 3. FNO（如果 SUDNN 不再炸）
cd ../fno_ns
python3 test_forward.py              # 期望 L2 ≈ 0.009516
python3 test_supa_chain.py           # 期望 chain median ≈ 16 ms
# 跑完整训练（必须 100 epoch）
python3 train_official.py --epochs 100 --n-train 1000 --batch-size 16
# ≈ 128 分钟，结果写入 results/run_logs/fno_official_train_*.log
```

---

## 6. 关键文件位置索引

| 文件 | 内容 |
|------|------|
| `spectral_conv_combo/spectral_conv_ops.py` | Python 包装（v1/fused/dual_out/auto-tune）|
| `spectral_conv_combo/spectral_conv_ext.cpp` | C++ Extension（含 `rfft2_sufft`、`spectral_mul_dual_out` 等）|
| `spectral_conv_combo/spectral_conv_ext.su` | SUPA kernel 源码 |
| `spectral_conv_combo/build.sh` | 一键编译 |
| `fno_ns/model.py` | FNO2d + FourierLayer（含 InstanceNorm2d 适配）|
| `fno_ns/train_official.py` | 官方训练门脚本（n_train=1000, bs=16, 100 epoch）|
| `fno_ns/dataset.py` | NS-like 数据集生成器 |
| `fno_ns/test_supa_chain.py` | FNO chain SUPA 评测 |
| `results/summary.json` | 性能/状态 JSON（已含所有实测数据）|
| `results/run_logs/` | 9 个 .md run logs + 1 个 spectral_grid_points.json |
| `HANDOVER.md` | 7 节交接文档 |
| `SUBMISSION_MANIFEST.md` | 提交摘要（已含 grid_points/s 数据）|
| `skill.md` / `SKILL.md` | Auto-tuning skill 文档 |
| `development_log.md` | 17 段开发记录 |
| `OPERATOR_OPT_TODO_2026-07-25.md` | **本文件** |

---

## 7. 联系上下文

- 上一份交接：`/workspace/ai4s-n/测试/提升版/results/run_logs/交接总结_2026-07-25.md`（历史细节）
- 上一份 SDK kernel 调研：`ai4s-n/测试/提升版/results/run_logs/sdk_kernel_probe_2026-07-24.md`
- 上一份 R5 dual_out 移植：`ai4s-n/测试/提升版/results/run_logs/sdk_kernel_probe_r5_followup_2026-07-24.md`
- 上一份性能 perf freeze（最初 baseline）：`results/run_logs/opt_perf_freeze_2026-07-22.md`
- 上一份 256 buffer 优化：`results/run_logs/opt_catchup_256_buffers_2026-07-23.md`

---

## 8. 不要做的事

- ❌ **不要重写 SpectralConv 主路径**（v1/fused/auto 已稳定）
- ❌ **不要删 `_AUTO_TUNE_TABLE`**（tune.py 的输出）
- ❌ **不要删 `.so` / `.su` 文件**（题目要求源码含扩展）
- ❌ **不要修改 `.gitignore`** 移除 `fno_ns/data/*.pt` 规则（481MB 数据不进 git）
- ❌ **不要把 `fno_ns_official_best.pt` commit**（17MB，每跑都会覆盖）
- ❌ **不要在主分支直接 force push**

---

## 9. 完成定义（DoD）

**P0 完成**：FNO chain `test_supa_chain.py` 跑通，且 L2 不退化（≤ 0.01）
**P1 完成**：新增 `test_supa_input_accuracy.py` 5/5 通过，FNO chain perf 改善 ≥ 1 ms/forward
**P2 完成**（可选）：单算子 fused 路径 perf 改善 ≥ 10%（记录在 `results/run_logs/`）

每完成一项 → `git commit` + `git push origin main` + 更新本文件"完成情况"小节。