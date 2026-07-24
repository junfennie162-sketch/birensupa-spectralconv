# 开发记录（Agent 辅助）

> 官方强制材料。至少 **5 段**有效交互，覆盖至少 **3 类**场景（kernel 设计调试、模型/超参、性能或平台适配、数据、可视化等）。  
> 工具：Cursor Agent（SSH 连接壁仞竞赛环境）。

## Agent 交互记录 1

- 工具 / Agent：Cursor Agent
- 目标：对齐赛道五选题与工作区结构（SpectralConv + FNO）
- 关键提示词或交互摘要：深度分析 workspace 与选手手册（当时路径 `docs/temp`，现已迁至 `docs/commpetition_docs`）；确认赛道五与进阶搭配
- Agent 建议：进阶优先 FNO-NS，与必选算子强复用；提交根用 `submission/`
- 采纳的修改：确定选题；落地 MVP 目录与根文档（`AGENTS.md` / `CONTEXT.md` / `submission/*`）
- 验证结果：目录与手册最低提交项对齐；业务代码尚未开始
- 未采纳内容及原因：完整 project-init 治理树过重，仅保留 MVP

## Agent 交互记录 2

- 工具 / Agent：Cursor Agent
- 目标：官方手册迁至 `docs/commpetition_docs` 并在资产中引用；确认「下一步」是否为环境 + GEMV 基准
- 关键提示词或交互摘要：对照 `docs/commpetition_docs` 两篇手册做资产引用；队长提到检查配置依赖 + 跑基准测试
- Agent 建议：本队只以《算子与模型赛道选手手册》为准；先过手册「快速开始」两条 GEMV 路线，再写 SpectralConv
- 采纳的修改：CONTEXT / AGENTS / README / submission / results / `.cursor/rules` 引用新路径；新增 `docs/commpetition_docs/README.md`
- 验证结果：`brsmi` 可见 Biren106B；方式一 `accuracy_ok=true`（3/3）；方式二 extension `ok=True`
- 未采纳内容及原因：非本赛道材料不纳入提交包

## Agent 交互记录 3

- 工具 / Agent：Cursor Agent
- 目标：划分 `ai4s-f` / `ai4s-n` 并行工作区，并在 `ai4s-f` 落地 SpectralConv 正确性第一版
- 关键提示词或交互摘要：环境基线已过；按 phase `spectral_accuracy` 生成 reference + SUPA + test；目录放在 `ai4s` 下两套骨架
- Agent 建议：两边同治理骨架；GPU 串行；先写 reference/test 再逼实现；相对误差用 Frobenius 范数比，避免近零点炸点
- 采纳的修改：根目录建立 `/workspace/ai4s-f` 与 `/workspace/ai4s-n`；在 f 侧实现 `spectral_mul` Extension + `test_accuracy.py`
- 验证结果：三档分辨率相对误差均 ≪ `1e-4`（worst ≈ `1.24e-7`）；`summary.json` / run_log 已写入
- 未采纳内容及原因：v1 暂未接 suFFT（正确性优先；FFT 留后续）

## Agent 交互记录 4

- 工具 / Agent：Cursor Agent
- 目标：按规划进入 `spectral_perf`，跑必选题性能基准（官网 §3.2：64/128/256）
- 关键提示词或交互摘要：对照 `赛题文档/` 与 phase 规划，在 `ai4s-f` 开始必选题基准测试；勿动 `ai4s-n`
- Agent 建议：补 `test_perf.py`，测自研 SUPA Extension 前向；配置对齐官网 B/C/modes；回写 `summary.json` / `results.md`
- 采纳的修改：新增 `submission/spectral_conv/test_perf.py`；写入 `spectral_conv.perf` 与性能表
- 验证结果：64×64 ≈ 14.07 ms / 4.8 MB；128×128 ≈ 24.11 ms / 4.8 MB；256×256 ≈ 222.91 ms / 4.8 MB；日志 `spectral_perf_2026-07-21.md`
- 未采纳内容及原因：未把 FFT 迁 suFFT（本 phase 只交付性能数字；优化留后续）

## Agent 交互记录 5

- 工具 / Agent：Cursor Agent
- 目标：`mark-done spectral_perf` 后进入 `fno_forward`（进阶 C：FNO-NS）
- 关键提示词或交互摘要：用户确认继续；回复改为正式简报 + 简单版双板块
- Agent 建议：用必选 SUPA SpectralConv 搭 ≥4 层 FNO；短训在 CPU 可微路径，推理走 SUPA；本机外网不稳则先用合成 64×64
- 采纳的修改：`fno_ns/model.py` / `test_forward.py` / `visualize.py`；回写 summary / results / 图
- 验证结果：4 层；相对 L2（SUPA）≈ 0.0506；图 `fno_ns_pred_vs_gt_2026-07-21.png`；torch 与 SUPA L2 一致
- 未采纳内容及原因：未下载公开 NS HDF5（github.com 超时）；完整长训留加分项

## Agent 交互记录 6

- 工具 / Agent：Cursor Agent
- 目标：深度优化 P0–P3（剖析拷贝 → fused GPU 常驻 → 双角对齐 FNO → 性能表冻结）
- 关键提示词或交互摘要：按「深度优化迭代计划」从 P0 起连续迭代；GPU 串行；改完测完再 commit
- Agent 建议：瓶颈在 `spectral_mul` 前后 Host↔Device；正式路径锁定 suFFT+SUPA mul fused；v1 作对照与可微训练
- 采纳的修改：`spectral_conv2d_fused`；双角 reference/test；FNO `use_sufft=True` 共用 API；`opt_baseline` / `opt_perf_freeze` 日志
- 验证结果：fused 正确性 ≤1e-4；正式性能 64/128/256 ≈ **6.25 / 15.94 / 87.61 ms**（相对旧 suFFT 桥接大幅下降）
- 未采纳内容及原因：未再开低于 5% ROI 的微改（P3 停止条件）

## 手工调试与实现记录

### 尝试 1 · 性能瓶颈（Host↔Device）

- 目标：确认旧 suFFT 路径为何小图更慢
- 修改：`profile_segments.py` 分段计时 FFT / bridge_mul / iFFT
- 结果：64 上 bridge_mul ≈ 26 ms，远大于 rfft；整谱回 CPU 再 mul 是主因
- 问题：GPU 并发时 `torch.zeros`/`slice` 易挂起
- 处理：串行占卡；实现 fused，mul 输入输出留 SUPA

## Agent 交互记录 7

- 工具 / Agent：Cursor Agent
- 目标：P4 反向 + FNO 加长训压 L2；纠正训练误走 SUPA mul 导致极慢
- 关键提示词或交互摘要：训练卡住在 epoch 1；用户询问为何变慢
- Agent 建议：长训用 CPU einsum；SUPA backward 用单测展示；误把 `spectral_mul_autograd` 塞进每层训练是 Host↔Device 风暴
- 采纳的修改：`SpectralMulFunction` + `test_backward.py`；训练改回 torch；256×40 加长训
- 验证结果：backward worst_rel ≈ 6.3e-8；FNO 相对 L2 ≈ **0.0173**（低于 0.02 目标）
- 未采纳内容及原因：训练全程绑 SUPA mul（正确但不可用作长训）

## Agent 交互记录 8

- 工具 / Agent：Cursor Agent
- 目标：P5 材料刷新 + P6 SpectralConv3d 扩展前向
- 关键提示词或交互摘要：按优化计划完成剩余 phase 并打包
- Agent 建议：刷新 README/results/demo；3D 用 CPU rFFT3 + 现有 mul reshape 复用
- 采纳的修改：文档与 SCP 指标；`spectral_conv3d_supa` + `test_3d_accuracy.py`；`run_tests.sh` 增加 backward/3d
- 验证结果：见对应 run_log；演示 media 同步最新图与指标
- 未采纳内容及原因：未做完整 3D FNO（计划允许只做算子前向）

## 总结

- Agent 帮助定位或解决的问题：选题、SpectralConv fused/反向/3D、FNO 加长训、性能冻结与材料打包
- 性能 / 正确性改进：fused 6.25/15.94/87.61 ms；backward ~6e-8；3D ~1e-7；FNO L2 ≈ 0.0173
- 可复用 prompt、skill 或工作流：见 `skills/`；演示入口 `scripts/run_demo.sh`

## Agent 交互记录 9

- 工具 / Agent：Cursor Agent
- 目标：对照搭档 `ai4s-n` 追平双角热路径（性能材料）
- 关键提示词或交互摘要：按 catch-up 分阶段：v1 单 sync、Parameter 缓存、fused `to_cpu`、FNO 延迟 D2H、256 缓冲复用
- Agent 建议：保持 API `spectral_conv2d_supa(..., use_sufft="auto", weights2=...)`；小图 v1、大图 fused；单卡串行
- 采纳的修改：`spectral_conv_ops.py` 热路径；`fno_ns/model.py` 传 Parameter；双角复测日志
- 验证结果：双角 auto 约 12–16 / ~79 ms（256）；device≈41.5 ms；正确性仍 ≪1e-4
- 未采纳内容及原因：未合入 n 的 packed 路径（已放弃）

## Agent 交互记录 10

- 工具 / Agent：Cursor Agent
- 目标：按官网新权重继续优化 `ai4s-f`（FNO 精度 25% 优先于再抠 256）
- 关键提示词或交互摘要：吸收飞书/官网评测标准后继续优化；工作区 `ai4s-f`
- Agent 建议：正确性已过硬门槛后，进阶侧优先压相对 L2；性能表刷新即可；扩展项 f 已有 backward/3D
- 采纳的修改：`resume_train.py`（+50 epoch，`n_train=768`）；回写 `summary.json` / 图 / `results.md`；补 Agent 记录
- 验证结果：SUPA rel_L2 ≈ **0.00952**（自 ~0.0248）；图 `fno_ns_pred_vs_gt_2026-07-23.png`；双角 auto 12.1 / 16.0 / 79.0 ms
- 未采纳内容及原因：未再开仅换个位数收益的 256 微改；未动 `ai4s-n`；未自动合入合并主线

## Agent 交互记录 11

- 工具 / Agent：Cursor Agent
- 目标：补回被误删的 `spectral_conv/spectral_conv_ops.py` / `fno_ns/model.py` / `README.md` / `development_log.md` / `demo/scp_description.md` / `results.md`，并把 256 auto 从 ~85 ms 压到 ≤75 ms（按 Phase-256 计划）
- 关键提示词或交互摘要：用户要求按 phase 优化；先 plan、再 P0–P5 执行；不并行占 GPU
- Agent 建议：先 `git checkout HEAD -- ...` 复原；清 `__pycache__` 让 `.py` 真正编译；定位 fused 路径段占比后做缓冲复用
- 采纳的修改：`_OUT_FREQ_CACHE` / `_HOST_OUT_CACHE`（fused 设备缓冲 + 钉扎内存）、`_OUT_FREQ_CPU_CACHE`（v1 CPU 复数缓冲）；fused `out_freq.zero_()` 复用 + `host.copy_(y).clone()`；v1 同上
- 验证结果：256 auto **85.59 → ~75 ms**（5 次中位数 76，最低 74.47）；64/128 中位 15/19 ms 无退步；正确性 ~1e-7、backward ~6e-8、3D ~1e-7 全过
- 未采纳内容及原因：把 auto 阈值降到 128 让 128 走 fused，结果 128 fused 总 40 ms 比 v1 22 ms 慢，回退阈值

## Agent 交互记录 12

- 工具 / Agent：Cursor Agent
- 目标：第二轮优化（R2 round-2）——把 auto 阈值扩到 ≥64，让 fused 服务 64/128/256 全部官方档
- 关键提示词或交互摘要：用户不接受当前落后（pyc 时代被掩盖的 cold start 退步），要求再做一轮
- Agent 建议：阈值从 ≥256 直接改为 ≥64；rfft2_sufft 在 64/128 上 H2D 整图代价 < CPU rfft/irfft；让 fused 三档全覆盖
- 采纳的修改：`resolve_use_sufft` 阈值 ≥64
- 验证结果（4 次 iters=100）：
  - 64 auto：**min 12.06 / median 13.50** vs 搭档 11.998（追平）
  - 128 auto：**min 14.11 / median 15.95** vs 搭档 14.156（追平）
  - 256 auto：**min 73.42 / median 75.33** vs 搭档 75.358（反超 ~0.03 ms）
  - 正确性 ~1e-7（test_accuracy）；backward ~6e-8；3D ~1e-7 全过
- 未采纳内容及原因：R3「128 混合路径」已无必要——R2 后 128 已追平

## Agent 交互记录 13

- 工具 / Agent：Cursor Agent
- 目标：跨 partner rebench —— 同协议下重新测 f 与 n，确认两方已收敛到 ~同一档性能
- 关键提示词或交互摘要：用户问 f 和 n 各项指标，要求实测全部覆盖旧对照表
- Agent 建议：同协议（`use_sufft="auto"` + warmup=10 + iters=50）下两边跑；不重业务代码；harness 放 `/tmp` 不入仓
- 采纳的修改：临时 `bench_f_auto.py` / `bench_f_fno_chain.py` / `bench_f_fno_l2.py` / `bench_f_accuracy_5case.py`
- 验证结果：5-case 双方 worst rel ≈ 2.84e-7 vs 2.83e-7（几乎打平 n 微严 0.4%）；64/128/256 auto 各 5.325/13.699/52.632 ms vs 5.326/13.783/52.727 ms（差距 ≤0.6% 噪声级）；FNO L2 f **0.009516** / n 空缺；FNO chain f 47 ms / n 15 ms（f 仍输）
- 未采纳内容及原因：旧版"R2 后仍未领先"的反超传言已不成立——同协议下两者几乎打平

## Agent 交互记录 14

- 工具 / Agent：Cursor Agent
- 目标：f FNO chain 落后 n 3× 的根因定位 + 修补（`forward_supa_chain`）
- 关键提示词或交互摘要：用户规划 8 项激进计划（A1–A3 + B1/B2 + C1/C2 + D1），同意复刻 n 的 suFFT device-resident 路径
- Agent 建议：在 `fno_ns/model.py` 加 `FourierLayer.forward_supa` + `FNO2d.forward_supa_chain` + `FNO2d.prepare_supa_eval`；保留原 `forward` 不动；用 `spectral_conv2d_supa(..., use_sufft="auto", to_cpu=False)` 端到端
- 采纳的修改：`FourierLayer.forward_supa`（含 `self.conv.to(device)` + IN 移到 device）；`FNO2d.forward_supa_chain`（含 `prepare_supa_eval`、可选 `use_gn_substitute`、末尾 `.cpu()`）；`FNO2d.prepare_supa_eval`（手动搬 IN running stats）
- 验证结果：
  - 新增 `fno_ns/test_supa_chain.py`：device-resident 断言通过（7 个中间张量全 `supa`，最终 `.cpu()`）
  - `bench_f_fno_chain_v2.py`：`forward_supa_chain` 链路 ~49 ms（vs n 15 ms）
  - 段剖析：每层 spectral 占 11.78 ms（与手写 harness 一致）—— 与 n 段差来自 input+weights + plan cache cold-state
- 未采纳内容及原因：未能把 chain 砍到 n 15 ms 同档——per-layer suFFT latency 仍是 ~3× 高于 n；同时段算法本身正确（不走 CPU/D2H）后续可以叠加 plan 预热与 `_OUT_FREQ_CACHE` 共享

## Agent 交互记录 15

- 工具 / Agent：Cursor Agent
- 目标：扩展 f 的 SUPA 算子能力 + 鲁棒性测试 + 学习笔记
- 关键提示词或交互摘要：用户在 plan mode 同意全套 8 项；本记录完成 B1/B2/C1/C2/D1
- Agent 建议：
  - **B1**：`spectral_conv_ext.cpp` 新增 `spectral_mul_out(x, w, y)` pybind；`spectral_conv_ops.py` 新增 `_y_freq_buffer(..., corner_id)` 并接入 fused 路径；**corner_id 必须区分**否则第二次写入覆盖（rel = 1.0 教训）
  - **B2**：`sufftSetWorkArea` + `sufftGetSize1d` 通过新建 `get_workspace()` 拼接 per-shape scratch buffer 绑定到 plan；`sufftDestroy` 释放临时 plan；workspace 不可绑定时宽容回退
  - **B2 未能** 接线 `sufftSetStream`——`<torch_br/csrc/core/supa/SUPAStream.h>` 链入 `sutlass.h`（SDK 头未装）
  - **C1**：9-shape irregular 测试（40×64 / 72 / 96 / 100 / 160 / 192 / 256×64 / 48 / 28）全 pass worst 3.20e-7
  - **C2**：FNO 6-shape 鲁棒测试（不同 modes × 不同 H×W）—— 6/6 全 finite，无 NaN/Inf
  - **D1**：`skills/spectral_chain_optimization.md` 落地 6 条笔记（auto 阈值、FNO chain、SDK API 边界、`_y_freq_buffer` corner pitfall、IN running stats、3× chain path）
- 采纳的修改：
  - `spectral_conv_ext.cpp`：`spectral_mul_out` 函数 + pybind；`get_workspace` + `WorkspaceKey` + `WorkspaceKeyHash` + `g_workspace_cache` + `sufftSetWorkArea` 接线
  - `spectral_conv_ops.py`：`_y_freq_buffer(..., corner_id)` + fused 路径改用 `spectral_mul_out`
  - `spectral_conv/test_irregular_shapes.py`：9-shape 表
  - `fno_ns/test_irregular_FNO.py`：6-shape + `torch.no_grad()` + 按 modes 重置权值
  - `skills/spectral_chain_optimization.md`：新文件
- 验证结果：
  - 5-case accuracy：worst rel 2.84e-7（与改前一致）
  - 64/128/256 auto perf：5.323 / 13.728 / 52.596 ms（持平 B1+B2，无显著 perf 收益，无退化）
  - test_irregular_shapes：9/9
  - test_irregular_FNO：6/6 finite
  - test_supa_chain：device-resident 中间张量全 `supa`
- 未采纳内容及原因：
  - `sufftBuildPlan2d/3d/Many` 不可用（SDK .so 未导出，已被 n 在 `14b7205` 验证）
  - `sufftSetStream` 接线不可行（`sutlass.h` 装不上，需 SDK 升级）
  - fused rFFT 跳 `permute().contiguous()`（n T1.1 估 +3–6 ms @256）尚未实施——memory layout 推断跨 R2C 的 stride 风险，等下次专项优化

## Agent 交互记录 16

- 工具 / Agent：Cursor Agent
- 目标：R4 进一步优化 FNO chain perf——把 f 链时从 R3 的 49.12 ms 砍到与 n 同档（~15 ms）
- 关键提示词或交互摘要：用户在 plan mode 下确认 R4 5 项（R4-1 删 .detach()、R4-2 删 prepare_supa_eval hot call、R4-3a/3b fused 直调+去零、复测 + 文档/合入）
- Agent 建议：用 `Parameter.detach()` 是 R3 未识别的元凶——`nn.Parameter.detach()` 返回新的 `id()` tensor，绕过 `_weights_to_supa_cached` 的 O(1) id 缓存，每次都走 D2H + numpy + blake2b 路径；f 路径上 2 weights × 4 layers = 8 次 hash round-trip / forward；删 .detach() 即命中 O(1)
- 采纳的修改：
  - `fno_ns/model.py:121-122`：`FourierLayer.forward_supa` 把 `weights1.detach()/weights2.detach()` 改成 `weights1/weights2`（保留 nn.Parameter 类型）
  - `fno_ns/model.py:248`：`FNO2d.forward_supa_chain` 不再内部调 `self.prepare_supa_eval()`（caller 责任）
  - `spectral_conv/spectral_conv_ops.py:395-404`：`spectral_conv2d_fused` fused 路径去掉 try/except，直接 `spectral_conv_ext.spectral_mul_out` 直调
  - `spectral_conv/spectral_conv_ops.py:227-228`：`_y_freq_buffer` cache-hit 路径去掉 `buf.zero_()`（`spectral_mul_out` 写满所有元素）
- 验证结果：
  - 5-case accuracy：worst rel 2.84e-7（不变）
  - auto perf 64/128/256：5.330 / 13.692 / 52.753 ms（R3 R3 5.34/13.80/53.01 ms 持平）
  - FNO chain full：**16.092 ms median / 16.022 ms min**（R3 49.118 ms → −67%）
  - 单层中位 ~3.5 ms（vs n 3.39 ms）
  - test_supa_chain：7/7 中间张量 supa（`forward_supa_chain` 不再触发 prepare_supa_eval 也不再走 D2H）
  - FNO L2：0.009516（与 R3 一致）
  - n 同协议：auto perf 5.331/13.845/52.861；FNO chain full **15.452 ms**
- 未采纳内容及原因：
  - fused rFFT 跳 `permute()`（R4 没做），估 +3–6 ms @256，但跨 R2C 的 stride 安全需要 R5 专项设计
  - 共享 plan-cache 跨 layer（plan cache 现已是 `(nx, batch, type)` keyed，4 层相同 shape 命中同一 handle；R4 不需要再做）
- 关键学习：`Parameter.detach()` 是隐藏的 wrapper allocation；cache 设计应优先走 id 路径而非 content hash 路径，必须 detach-它们的快捷需求与 id-stable 期待冲突时需明确说明。已写进 `skills/spectral_chain_optimization.md` "R4 fix" 章节

## Agent 交互记录 17

- 工具 / Agent：Cursor Agent
- 目标：R5 — 在 R4 同档基础上继续找可挖点 + 修一个隐藏的语义错误
- 关键提示词或交互摘要：用户要求开始 R5；先 brainstorm 后再动手；建议先修 `use_gn_substitute` 的语义错误，再加 profiler，再做小融合
- Agent 建议：
  - **R5-0 P0**：原 `forward_supa_chain` 里 `use_gn_substitute=True` 路径在已含 IN 的 `forward_supa` 输出上**再叠一遍 GroupNorm+GELU**，是双 norm；L2 验证正确但语义错、隐藏了 perf 真值。删这个 kwarg。
  - **R5-1**：写正式 `bench_f_fno_chain_layer_profile.py` ——替代 `/tmp` 的临时 harness，固化分层时延，作为后续 R5 决策基础设施。
  - **R5-3**：在 `spectral_conv_ext.cpp` 加 `spectral_mul_dual_out(x1,w1,x2,w2,y1,y2)`，一次 pybind 启动双角 kernel；fused 路径改用它。估 -0.16 ms/chain @ L=4。
  - **R5-4 (失败后退)**：尝试把 dual_scatter 写成 scatter 模式（直接写 `out_freq[:, :, :modes1, ...]` + zero 整个 buffer），结果 fused 路径 rel=1.0。**Bug**：`narrow+contiguous` 返回新 SUPA tensor，对其 in-place 写不会回传 `out_freq` 自身。已回退，仅保留 R5-3 的 dual_out。
  - **R5-2 (终止)**：从 `fno_ns_demo.pt` 续训 30 epoch（lr=2e-4）；test L2 从 baseline 0.009516 → 0.012811 (+5 epochs) → 0.013528 (+10 epochs)，**regress 严重**。Killed 后未保存。结论：当前 ckpt 已收敛；继续用 default Adam 在 L2 上无收益，需要 lr scheduler reset + cosine 才能探。
  - **R5-5 (spike)**：1 分钟 ABI check，`nm -D libsufft.so.0.7.0` 导出 = `BuildPlan1d / ExecR2C / ExecC2C / ExecC2R / SetStream / SetWorkArea`，**没有 `BuildPlanMany`/`2d`**。R5-A1 rFFT 跳 `permute()` 的 stride trick 在 ABI 上不可行，stop。
- 采纳的修改：
  - `fno_ns/model.py`：删除 `use_gn_substitute` kwarg；`forward_supa_chain` 内不再二次 norm
  - `fno_ns/test_supa_chain.py`：跟上 model.py 的 kwarg 移除
  - `fno_ns/bench_f_fno_chain_layer_profile.py`：新文件，正式 layer profiler
  - `.cursor/rules/parameter-cache.mdc`：新规则，固化 R4 lesson（防 `.detach()` 再破坏 cache）
  - `spectral_conv_ext.cpp`：加 `spectral_mul_dual_out` (+ `dual_scatter_out` 备用)，pybind 注册
  - `spectral_conv_ops.py`：fused 路径换成 `spectral_mul_dual_out`
  - `spectral_conv_ext.su`：未改
  - **`resume_train.py`**：新文件（保留，下次续训用 cosine scheduler）
- 验证结果（R5 final）：
  - 5-case accuracy：worst 2.84e-7
  - test_supa_chain：7/7 中间张量 supa
  - FNO irregular 6-shape：6/6 finite
  - spectral 64/128/256：5.322 / 13.697 / 52.623 ms
  - FNO chain full：**16.078 ms** median / 15.988 ms min
  - FNO L2：0.009516（与 R4 一致）
  - n 同协议：auto 5.311/13.715/52.839；chain full 15.453
- 未采纳内容及原因：
  - scatter fused dispatch（R5-4 失败后退）— `narrow+contiguous` 不会回写到原 out_freq
  - resume 续训（R5-2 中止）— test L2 严重 regress
  - rFFT stride trick（R5-5 spike stop）— SDK ABI 不支持
