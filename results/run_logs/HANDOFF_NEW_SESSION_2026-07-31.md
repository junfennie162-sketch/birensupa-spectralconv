# Handoff · 新 Session 交接 Prompt（2026-07-31）

> **历史**：boost/squeeze 时代交接稿。  
> **现行交接**请用 [`HANDOFF_NEW_SESSION_2026-08-02.md`](HANDOFF_NEW_SESSION_2026-08-02.md) · 指针 [`CURRENT.md`](CURRENT.md)。

---

## 可复制 Prompt

```text
你是接管 ai4s-f（队长侧）的 Agent。工作区默认只写 /workspace/ai4s-f；稳定后再合入 /workspace/ai4s/submission；禁止改 ai4s-n。单卡 GPU，禁止与长训并发写 Spectral formal perf。

## 先读（门禁）
1. /workspace/AGENTS.md 与 /workspace/ai4s-f/AGENTS.md
2. 行动总方针（计划卡「双题优化总方针」或落盘副本）：
   /workspace/ai4s-f/submission/results/run_logs/OPT_MASTER_PLAN_2026-07-31.md
3. 进度快照：
   - results/summary.json（看 meta.notes / fno_ns.public_ns64）
   - results/run_logs/fno_public_boost_chain_state.json
   - fno_ns/checkpoints/fno_ns_public_ns64_meta.json
   - pgrep -af 'run_public_boost_chain|train_public_ns64_boost'

## 上一会话已完成
- P0 材料口径闸：公开 NS64 主报对齐（PPT/scp/results/disclosure/fno_eval_protocol/phase_notes/summary 等）
- P1 Spectral 冻结：主表 3.811/8.054/29.560 ms；封死 Plan2d/torch.fft@SUPA/strided pack/R12–R14 等
- OPT_MASTER_PLAN + opt_dual_track_plan 已落盘/回写
- P2a-A：boostA_hf_aug 已 PROMOTE
  - 公开 L2：0.03961178520694375（原 0.041835）
  - ckpt：fno_ns/checkpoints/fno_ns_public_demo.pt（与 public_ns64_best 一致）

## 交接时现场（以你启动时 pgrep/文件为准）
- 后台可能仍在跑：
  - python3 -u run_public_boost_chain.py
  - train_public_ns64_boost.py --tag boostB_residual --epochs 100 --residual ...
- chain_state 曾显示 stage=B_residual_scratch；B 从零 residual，前期 L2~0.05x 差于 A 属正常，只有 < 0.039612 才 promote
- 勿误杀 boost 链；不要再开第二条长训抢 CPU
- 上一会话问题：长 sleep 轮询易被中断；改用短轮询或等 chain_final.json 出现

## 请你继续执行（按序）
1. **监控/收尾 P2a**：等 boost 链跑完 B→C；读 fno_public_boost_chain_final.json；有提升才 promote，并回写 summary/results/disclosure/PPT/scp/metrics_snapshot
2. **P2b**：对当前 public best 做至少 1 轮 freeze 或低 lr continue（公开 1000/128）；记录 ΔL2 与停条件计数
3. **P2c**：默认跳过（已 ≤0.040 达标带）；仅当用户要求或主报回退时再开 width/modes/F-FNO（早停）
4. **P3**：
   - 用 public demo 重跑 visualize，对齐 summary.visualization
   - maintain_assets.sh check（相关）
   - 合入 /workspace/ai4s/submission/（本地目录同步，不依赖 push）
   - 旧口径扫描：禁止把 0.005144/0.008768/5.3/13.7/52 写成正式公开成绩
   - development_log 补本轮段；更新 OPT_MASTER 执行状态表

## 红线
- 主报唯一：fno_ns.public_ns64 + Spectral idle 三档
- Promote 仅公开 1000/128 test L2 严格更优
- 正确性 ≤1e-4 未过不进主表
- 不挖 Spectral ms；不重开 v2 同构 0.005144
- 回答用中文；ai4s-f 相关先简述后详细

## 验收口令
- 主报 L2 / Spectral ms / 材料无伪官方 三者一致
- boost 链结束或明确 skip；P2b 有日志；P3 合入完成
```

---

## 给人类的一句话

新开会话后：先贴上面 Prompt → 让 Agent 核对 `pgrep` 与 `public_ns64_meta` → 从 **等 boostB/C 结束** 接着干，不要重做 P0/P1。
