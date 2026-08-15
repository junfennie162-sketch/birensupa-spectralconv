# FNO · Navier–Stokes (advanced)

Reuses `../spectral_conv`. Four Fourier layers. Vorticity **10 → 1** on **official public NS64**.

## Formal result

| Item | Value |
|------|--------|
| Data | `data/navier_stokes_v1e-3_N1200_T20.pt` (1000/128 · seed 20260722) |
| Protocol | 10 frames → frame 11 · residual head · relative L2 |
| Checkpoint | `checkpoints/fno_ns_public_demo.pt` |
| Public L2 | **0.035012** |
| Shape | 4 layers · width=32 · modes=16 · 64×64 |

## Demo figures

Not hand-drawn, not synthetic. `render_official_demo.py` reads the official `.pt` and the formal weights, forwards on the test split, writes `checkpoints/demo_batch.pt`, then `visualize.py` draws prediction / ground truth / error.

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
cd spectral_conv && ./build.sh
cd ../fno_ns
python3 render_official_demo.py
```

Bring your own `data/navier_stokes_v1e-3_N1200_T20.pt`. Do not use `test_forward.py` as the demo entry: it defaults to a generated cache and can overwrite `demo_batch.pt`.

## Main files

| File | Role |
|------|------|
| `model.py` | FNO (inference calls the mandatory operator) |
| `dataset.py` | Official `.pt` loader and 1000/128 split |
| `render_official_demo.py` | **Demo entry**: official data + formal weights |
| `visualize.py` | Draw from `demo_batch.pt` |
| `test_forward.py` | Engineering regression (not the public set by default) |
| `test_chain_cpu_supa_consistency.py` | CPU vs device-chain consistency |
| `test_supa_chain.py` / `test_irregular_FNO.py` | Device-resident / irregular sizes |
| `train_public_ns64.py` | Public-set training entry |
| `train_public_ns64_boost.py` | Residual / spectral loss (reused by continue-train) |
| `train_public_sched_sampling.py` | Gated continue-train (produced v10) |
| `benchmark_fno_batch16.py` | batch-16 throughput (side note; writes summary) |
