# 答辩 90 秒口播分镜（不拍视频）

主报锚：公开 L2 **0.035302**（`freeze_r9` · v8）；Spectral idle **3.811 / 8.054 / 29.560 ms**。  
版本链一句：v7 0.035725 → v8 0.035302（只比上一版 **+1.18%**；自 v1 累计约 +15.6%）。  
一页包：[`JUDGE_3MIN_PACK_2026-08-04.md`](JUDGE_3MIN_PACK_2026-08-04.md) · PPT：[`PPT答辩冻结稿_2026-08-04.md`](../PPT答辩冻结稿_2026-08-04.md)

| 秒 | 画面/动作 | 口播一句 |
|----|-----------|----------|
| 0–15 | `brsmi` + 架构图 | Biren 上 SUPA SpectralConv，FNO 复用必选算子 |
| 15–30 | accuracy PASS 表 | 正确性 worst≈2e-7，远低于 1e-4 |
| 30–50 | 三档 ms + C2R 墙 | mul 噪声级；墙在 C2R；formal 冻结；vs 官网 CPU ≈19.5×/11.1×/10.0×（≠ SOL） |
| 50–70 | 流场图 **2026-08-02**（`demo/media` 现行） | 公开 NS64 L2=0.035302；sched→soft→freeze 抛光 |
| 70–85 | matrix KEEP/ABORT + AUDIT | 失败诚实：r10 回滚、A1 NO_SIGNAL；扩展三命令可抽查 |
| 85–90 | Agent 索引 | 精品：stop-on-gate / Loop / freeze_r9 / 多 Agent / **记录35 回滚门禁** |

命令入口：`SPECTRAL_BONUS_AUDIT_CARD.md` · `extension_showcase.md` · `JUDGE_3MIN_PACK_2026-08-04.md` · README「评委 3 分钟路径」
