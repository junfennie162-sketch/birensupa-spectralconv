# FNO-NS 数据与训练披露

## 正式主报：公开 NS64（2026-08-06 · v9）

- 文件：`fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`（来源 HF `abelsr1710/navier-stokes-2d-fno`，离线缓存）
- 布局：`[1200,20,64,64]`（loader 已从 `[N,H,W,T]` permute）
- 划分：**n_train=1000 / n_test=128**，seed=`20260722`
- 任务：T_in=10 → T_out=1（输入前 10 步涡度 → 预测第 11 步）
- 模型：4 层 FNO，width=32，modes=16
- 训练：… → freeze_r9/r11 → long_push（qt 过采样 / 末层解冻 / 双视图）→ **dualview_r2**
- **公开集 relative L2：0.03511497611179948**（2026-08-06 `dualview_r2` promote · 评测报告 v9；上一正式 v8=`freeze_r9` 0.035302）
- checkpoint：`fno_ns/checkpoints/fno_ns_public_demo.pt`
- 说明：必选 SpectralConv 不使用该 NS 数据；本披露仅覆盖 FNO-NS 精度主报。
- Promote 规则：公开集 1000/128 test L2 须破声明 gate（baseline−1e-4）且人工确认后才替换 primary。

## 附录 · ROUND10 旁注（非主报）

- 手跑 `freeze_r10` 近失：**0.035287**（gate 0.035202）→ NO_SIGNAL，未编 v。
- 历史：autochain 弱 0.035252 曾误写入 → 2026-08-04 回滚 v8；其后过程 live 含 freeze_r11 0.035223（未单独编报告 v）。
- 现行：`fno_ns_public_demo.pt` = **dualview_r2**；backup=`fno_ns_public_demo_pre_dualview_r2.pt`。

## 附录 A · 工程对照：自建 `generated_ns_like_v2`（非公开分）

以下数字**不得**表述为公开 NS64 / 官方榜单成绩。

生成器位于 `fno_ns/dataset.py`，输出布局为 `[N,T,H,W]`：

- 样本数：1024；时间步：30；分辨率：64×64；粘度：`1e-3`；非线性强度：`0.035`
- 随机种子：`20260722`；缓存：`fno_ns/data/ns_like_v2_N1024_T30_64.pt`
- 历史划分旁注：768/128（`rel_l2_768_split`≈0.00249）或同权 1000/128 continue3≈**0.005144**
- checkpoint（旁注）：`fno_ns/checkpoints/fno_ns_demo.pt`
- 归档：`results/archives/fno_v2_continue3_pre_public_20260731.tar.gz`

生成过程使用频谱衰减、固定 forcing 和可复现的结构化非线性扰动，用于离线工程验证。

## 公开数据接口

`fno_ns/dataset.py` 会优先读取 `fno_ns/data/` 下文件名不以 `ns_like` 开头的 `.pt` 文件，支持 `[N,T,H,W]`，并兼容部分 `[N,H,W,T]` 布局。提交默认不联网下载，以保证评测容器离线可复现。
