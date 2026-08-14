# Spectral 六轴 · 答辩口播单页（2026-08-04）

> 正式 idle **冻结**：**3.811 / 8.054 / 29.560 ms**。勿默认 `test_perf`。  
> 索引母卡：[`spectral_perf_narrative_index.md`](spectral_perf_narrative_index.md)。  
> **硬禁**：SOL/proxy 得分句；夸大「真 FFT⊗mul 融合」；暗示将解冻 ms。

---

## 60 秒口播（六轴收成三句 + 三数）

1. **mul 已噪声级**（分段 ≈0.23 ms）；墙在 **C2R / FFT**。  
2. formal 三档冻结；对外讲 **vs 官网 CPU ≈19.5× / 11.1× / 10.0×**（旁注加速比，**不是** SOL 得分）。  
3. 显存 footprint 225 / 253 / 353 MB；融合诚实=设备常驻 + 稀疏 scatter，**非** SDK 级 FFT⊗mul 真融合。

---

## 评委打断 Q（红线答）

| Q | 标准答 |
|---|--------|
| 还能再抠 ms 吗？ | 墙在 C2R；SDK 无 Plan2d/真融合；formal **冻结**，改讲加速比/显存/Agent 瓶颈分析 |
| 你们做了融合吗？ | 频谱常驻 + gather/scatter mul 合一路径；**不是**单 kernel FFT⊗mul；见 fusion honesty 卡 |
| SOL 多少分？ | 队内 proxy **仅旁注**；**不进**官方得分句 / SCP 主句 |
| 256 为什么更慢？ | 分辨率放大后 C2R/数据运动主导；见 fused_segments 旁注 |
| 3D 呢？ | 算子四角扩展正确性 PASS；**不是**完整 3D FNO |
| 和竞品 GPU 比？ | 本赛道只报 Biren + 官网 CPU 参考；不做跨厂牌刷表 |

---

## 六轴 → 证据（抽查时打开）

| 轴 | 一句 | 打开 |
|----|------|------|
| 分段墙 | C2R 主导 | `spectral_fused_segments_2026-08-01.md` |
| 多分辨率 | 三档 idle + CPU 加速比 | `spectral_multires_story_2026-08-02.md` |
| 显存 | 225/253/353 + cache/packed | `spectral_memory_story_2026-08-02.md` |
| 并行度 | dual_scatter / packed | `spectral_parallelism_card_2026-08-02.md` |
| 融合诚实 | 边界披露 | `spectral_fusion_honesty_card.md` |
| 扩展 | bwd/3d/irregular 命令 | `extension_showcase.md` · `SPECTRAL_BONUS_AUDIT_CARD.md` |

---

## 与评委入口的链

- 一页包：[`JUDGE_3MIN_PACK_2026-08-04.md`](JUDGE_3MIN_PACK_2026-08-04.md)  
- PPT 第 3–4 页：[`PPT答辩冻结稿_2026-08-04.md`](../PPT答辩冻结稿_2026-08-04.md)  
