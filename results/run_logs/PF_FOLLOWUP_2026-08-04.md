# PF 启动后 · 后续工作清单（2026-08-04）

> 探针：`pf_clean_r1` · clean-anchor Pushforward · freeze spectral · epochs≤4 · gate=0.035202  
> 日志：`/tmp/pf_clean_r1.log` · summary：`results/run_logs/fno_public_pf_clean_r1_summary.json`  
> 主报在裁决前保持：**0.035302 · freeze_r9 · v8**

## 探针结果（已结束）

| 项 | 状态 |
|----|------|
| PF 短探针 | **DONE · NO_SIGNAL** · best=**0.03521636** · gate=0.03520218 · 4/4 epoch 全程小幅下降 |
| 相对主报 | Δ≈**+8.6e−5**（约 +0.24%）；差 gate ≈**1.4e−5**（近失） |
| 自动 promote | **关** · `promote=false` |
| 分支 | **B · INCUBATE 旁注** · 主报继续 v8 · 精度 **永久停** |

秒查：

```bash
tail -n 40 /tmp/pf_clean_r1.log
cat results/run_logs/fno_public_pf_clean_r1_summary.json 2>/dev/null
```

## 探针结束后的分支（预注册）

### A · 破 gate（best < 0.035202）→ SIGNAL

1. **人工确认**后再 `promote_public_ckpt.py --src …pf_clean_r1_best --tag pf_clean_r1`（勿自动）  
2. 重跑可视化 strip；刷新 `demo/media` 主图  
3. 评测报告换戳编 **v9**（§1 对 v8；§9 追加）  
4. `development_log` 完整段；`maintain check` + `run_loop --strict`  
5. 稳定文件 sync `/workspace/ai4s/` 后视需要 pack  

### B · 近失 / 弱提升未破 gate → INCUBATE 或 NO_SIGNAL

1. **不编 v**；主报继续 v8  
2. 若配对 bootstrap 可信且多数样本同向 → 记 `INCUBATE`（像 r10）  
3. 精度线 **永久停**（本条 PF 已用尽 Autopsy 授权）  
4. 答辩补一句：「AR 相关成立，但 4-epoch PF 未破正式 gate」  

### C · 变差 / early_stop / rebound → ABORT

1. 删或保留 ckpt 均可，**禁止**覆盖 demo  
2. 写 matrix 一行 ABORT  
3. 转纯答辩（JUDGE + Autopsy 三图）  

## 无论 A/B/C 都要做的材料收口

| 优先级 | 动作 | 入口 |
|--------|------|------|
| P0 | 更新 CURRENT 精度姿态与 PF 结果一行 | `results/run_logs/CURRENT.md` |
| P0 | matrix + development_log 段 | `experiment_matrix.md` / `development_log.md` |
| P1 | JUDGE / PPT 若有新数字再改一句；无 promote 则不动 v | `JUDGE_3MIN_PACK_*` / `PPT答辩冻结稿_*` |
| P1 | Autopsy 三图继续用于「协议不可比 + AR 分解」叙事 | `demo/media/protocol_*` 等 |
| P2 | 答辩演练 3 分钟（JUDGE 一页包） | 人口头 |

## 明确不做

- STLW / modes20 / width48 / hard_reweight 再开  
- 解冻 Spectral formal / 默认 `test_perf`  
- 无人工确认的 promote  
- f/n 并发占卡  

## 纪律锚

- gate = baseline − 1e−4 = **0.035202**  
- Autopsy：`ERROR_AUTOPSY_VERDICT_2026-08-04.md`（ρ(e1,g)=0.798 → 才授权本条 PF）  
- OPT Loop：`skills/operator_opt_loop/LOOP_PROCESS.md`
