# SCP blurb · FanDou Garden

**SpectralConv + FNO on BIREN** (Biren Flying Cup · Models & Operators)

在壁仞 GPU 上用 SUPA / PyTorch Extension 实现 FNO 核心频谱卷积，并搭 ≥4 层 FNO 做二维涡度预测。

## 结果

| 模块 | 实测（Biren106B） |
|------|-------------------|
| 必选频谱卷积 | 正确性最差相对误差 **7.162×10⁻⁶**；64/128/256 **0.764 / 1.827 / 6.504 ms** |
| 进阶 FNO | 官方公开 NS64（1000/128）相对 L2 **0.035012** |

图由 `fno_ns/render_official_demo.py` 生成。现行两张：典型样本三连图、最好/典型/最差对照。说明见 `demo/media/README.md`。

## 怎么跑

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd spectral_conv && ./build.sh
cd ../fno_ns && python3 render_official_demo.py
```

队伍：翻斗花园 · 中北大学 · 赛道五
