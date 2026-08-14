# 可直接粘贴给云端 Agent 的提示词模板

把下面整段作为任务提示；并上传本 ZIP 内全部 Markdown。

---

你是答辩 PPT 生成 Agent。请严格阅读本压缩包内全部 `.md`（按 `00_README_先读我.md` 的顺序），为竞赛队伍「翻斗花园」生成 **16:9、10–12 页、简体中文** 的可投屏答辩 PPT（优先 `.pptx`）。

硬约束：
1. 数字只能使用 `04_FROZEN_METRICS.md`：公开 NS64 L2=**0.035302**（v8/`freeze_r9`）；Spectral idle=**3.811/8.054/29.560** ms；worst rel≈**2.17e-7**（阈值 1e-4）。
2. 内容必须对齐 `02_OFFICIAL_REQUIREMENTS.md` 的官网评分轴（必选：正确性35/性能25/代码10/Agent15/扩展15；进阶C：精度25/搭建30/性能10/可视化20/Agent15）。
3. 页结构遵循 `05_PPT_CONTENT_SPEC.md`；视觉遵循 `06_PPT_DESIGN_SPEC.md`（推荐 pyramid + swiss-minimal 或 dark-tech）。
4. 遵守 `09_RED_LINES.md`：禁止 SOL 得分句、禁止 v9、禁止把 0.00249/PF近失当主报、禁止称完整 3D FNO、精度永久停。
5. 无原图时按 `08_ASSET_MANIFEST.md` 做占位框，勿伪造流场图。
6. 交付前按 `01_CLOUD_AGENT_MISSION.md` 与 `09` 自检清单逐条勾选，并在回复中贴出自检结果。

开始生成。

---

## 委托方可选补充（粘贴在提示词后）

- 视觉偏好：`swiss-minimal` / `dark-tech`（二选一）  
- 是否需要英文对照标题：是 / 否  
- 是否附 speaker notes：是 / 否  
- 若另附 PNG：列出文件名  
