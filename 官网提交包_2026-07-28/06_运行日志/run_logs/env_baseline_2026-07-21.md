# 环境基线冒烟日志（2026-07-21）

手册 Part A「快速开始」。验证对象：**服务器竞赛环境**，非 SpectralConv 作品。

## 命令摘要

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2

# sniff
python3 -c "import torch, torch_br; x=torch.zeros(2, device='supa'); print(x.device)"
brsmi

# 方式一
cd /workspace/ai4s/gemv && make clean && make build SUPA_BASE=$SUPA_BASE
make run-accuracy SUPA_BASE=$SUPA_BASE

# 方式二（本队正式路线）
cd /workspace/ai4s/gemv/torch_extension && ./build.sh && python3 test_gemv_ext.py
```

## 结果

- `torch 2.9.0+cu128`，`torch_br` OK，`supa:0` 可用；`cuda_available=False` 预期
- GPU：Biren106B（`brsmi`）
- 方式一：`accuracy_ok=true`，3/3
- 方式二：`ok=True`；`perf_4096x1024` ≈ 2.94 ms

## Agent 解读

| 已确认 | 后续用途 |
|--------|----------|
| SDK + env 脚本 | 编译链接完整依赖 |
| brcc | 编 SpectralConv kernel |
| 单卡 BIREN | 测精度/性能、交日志 |
| torch_br / supa | Extension + FNO |
| GEMV 两路线 | 模板可迁；下一任务是 SpectralConv |

完整表见上级 `results.md` 与根目录 `AGENTS.md`「已验证基线」。
