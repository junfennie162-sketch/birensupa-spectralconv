# Navier-Stokes 2D 数据说明

## 正式数据（提交时使用）

推荐公开数据集（与官网一致）：

- **Navier-Stokes 2D 涡度**，空间分辨率 **64×64**，粘度数据集  
- 常见来源：FNO 论文配套数据（Zongyi Li et al.），例如  
  - Google Drive / HuggingFace 镜像上的 `NavierStokes_V1e-5_N1200_T20` 等 64×64 包  
  - 或 PDEBench / neuraloperator 官方数据链接（以组委会最终通知为准）

下载后请将文件放到本目录，并在训练/评估脚本中通过 `--data_path` 指定。

| 字段 | 说明 |
|------|------|
| 形状 | 常见 `[N, T, H, W]`，H=W=64 |
| 任务 | 输入前 `T_in` 帧涡度 → 预测后续帧 |
| 许可证 | 以数据发布方声明为准（须在提交 README 注明来源） |

## 本仓库当前默认

为保证 **BIREN 单卡前向可复现**、不依赖外网，`test_forward.py` / `visualize.py` 使用 **合成涡度场**（确定性随机种子）。  
换成真实 NS 数据后，只需替换 DataLoader，模型与 SpectralConv 算子不变。

## 划分建议

- train / val / test 按官方或常用 1000/200 划分（若使用公开集）  
- 报告相对 L2：`||pred - gt|| / ||gt||`
