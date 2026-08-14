# 比赛提交材料（下载这个目录）

GitHub 单文件上限 100MB，完整 2.9G 工作区包无法入库。本目录是**可直接下载的官网交卷包**（约 39MB）。

## 下载

仓库：https://github.com/junfennie162-sketch/birensupa-spectralconv

| 文件 | 说明 |
|------|------|
| [`fandougarden_赛道五_提交材料_v10_20260814.tar.gz`](fandougarden_赛道五_提交材料_v10_20260814.tar.gz) | 交卷材料压缩包 |
| [`fandougarden_赛道五_提交材料_v10_20260814.tar.gz.sha256`](fandougarden_赛道五_提交材料_v10_20260814.tar.gz.sha256) | 校验 |

浏览器打开上面 `.tar.gz` 链接 → Download。命令行：

```bash
curl -L -o fandougarden_赛道五_提交材料_v10_20260814.tar.gz \
  https://github.com/junfennie162-sketch/birensupa-spectralconv/raw/main/contest_submit/fandougarden_%E8%B5%9B%E9%81%93%E4%BA%94_%E6%8F%90%E4%BA%A4%E6%9D%90%E6%96%99_v10_20260814.tar.gz
tar -xzf fandougarden_赛道五_提交材料_v10_20260814.tar.gz
cd fandougarden_赛道五_提交材料_v10_20260814
```

也可整库 ZIP：仓库页 Code → Download ZIP（体积更大，含源码树）。

## 包内即交卷根

解压后对应手册 `my_submission/`：`README.md`、`skill.md`、`AGENT_OFFICIAL.md`、`development_log.md`、`spectral_conv/`、`fno_ns/`、`results.md`、可视化与正式 ckpt。

不含公开 NS 数据 `.pt`（约 376MB）。复评 FNO 时自行放到 `fno_ns/data/navier_stokes_v1e-3_N1200_T20.pt`。

主报：公开 NS64 L2 **0.035012**（v10）· Spectral idle **3.797 / 8.037 / 29.295 ms**。
