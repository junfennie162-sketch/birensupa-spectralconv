# PF+Δ hybrid 再冲计划（2026-08-06）

> 用户口头「再冲一下」· 新机制组合 + gate + `--stop-on-gate` · 禁自动 promote

## 两步

1. **Soup 快评**：`demo` + `pf_clean_r1` + `delta_match_r1` 均匀/首尾平均  
2. **Hybrid 训**：clean-anchor PF + Δ-match 同损 · tag `pf_delta_r1`

## 预注册

| 项 | 值 |
|----|-----|
| 主报 baseline | **0.03522327** · freeze_r11 demo |
| gate | **0.03512327**（−1e−4） |
| hybrid | λ_pf=0.75 · λ_δ=0.35 · hf=0.15 · lr=4e−6 · ep≤4 · freeze spectral |
| init | demo（可选二次：从 pf_clean 热启，仅当第一步无明显 SIGNAL） |
| promote | **关** |

禁：STLW / modes20 / width48 / hard_reweight / 同构单机制 deepen。
