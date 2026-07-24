# Spectral Convolution（必选）· 混合路线

## 计算流程

```text
x: [B, C_in, H, W]
  → CPU rFFT2
  → 截断低频 modes（正频角 + 负频角）
  → SUPA spectral_mul × weights1 / weights2
  → CPU iFFT2
y: [B, C_out, H, W]
```

## 文件

| 文件 | 作用 |
|------|------|
| `official_baseline.py` | 官网 PyTorch reference 原文 |
| `reference_pytorch.py` | 双角 SpectralConv2d（验收标准） |
| `spectral_conv_ext.su/.cpp` | SUPA kernel + pybind（与 `my_task_direct` 同源） |
| `build.sh` | Extension 构建 |
| `spectral_conv_ops.py` | 全链路组装 + `SpectralConv2dSupa` |
| `test_accuracy.py` / `test_perf.py` | 正确性 / 64·128·256 性能 |

方式一 C++ 测程：`/workspace/ai4s-n/my_task_direct`

## 编译与测试

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd /workspace/ai4s-n/submission/spectral_conv
./build.sh
python3 test_accuracy.py
python3 test_perf.py
```
