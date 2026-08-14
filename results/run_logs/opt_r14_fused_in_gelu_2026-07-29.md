# R14 · fused add+IN+GELU SUPA kernel · 2026-07-29 · ROLLBACK

## 猜想
用自定义 SUPA kernel 融合 `y+skip` + InstanceNorm + GELU，减少 launch/读带宽；输出写 device buffer（不碰 host-seeded safe）。

## 结果
| 项 | 结论 |
|----|------|
| 正确性（单线程 apply + exp-tanh GELU） | 可对齐 `F.gelu(..., approximate='tanh')` |
| 多线程 apply（共享 mean 后并行写） | **数值错误**（Biren shared 可见性/同步 quirk） |
| 设备 `tanhf`/`erff` | **不可用**（输出垃圾）；需 `expf` 自实现 tanh |
| 性能 @ B16×C32×64 | fused **~2.37 ms** vs torch IN+GELU **~0.75 ms** → **更慢** |
| InstanceNorm2d 默认 | `affine=False`，无 weight/bias |

**判决：ROLLBACK**（更慢 + 多线程路径不稳）

## 教训
1. FNO 层内 IN+GELU 不是主瓶颈（~0.9 ms vs spectral ~10 ms）；融合核必须显著快于 SUDNN/torch 才有意义。
2. Biren：慎用 `tanhf`/`erff`；大数组 shared reduce + 多线程读 shared 标量需充分验证。
3. 串行 thread0 reduce（H*W=4096）本身就打不过现成 IN。
