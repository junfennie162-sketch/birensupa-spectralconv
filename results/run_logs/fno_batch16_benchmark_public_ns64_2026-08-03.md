# FNO batch=16 benchmark

- time_utc: 2026-08-03T13:27:04Z
- device: BIREN single card (Biren106B / supa)
- data: file:navier_stokes_v1e-3_N1200_T20.pt
- disclosure: public NS64 navier_stokes_v1e-3_N1200_T20.pt; n_train=1000/n_test=128
- checkpoint: `fno_ns/checkpoints/fno_ns_public_demo.pt`
- config: B=16, H=W=64, warmup=10, iters=50
- chain_consistency_rel (gate B=4): 8.799542410997674e-05
- chain_consistency_rel (batch16 note): 9.546995715936646e-05

| scope | grid_points/s | samples/s | ms/sample | ms/batch | peak MB |
|---|---:|---:|---:|---:|---:|
| pure forward | 1600295.313 | 390.697 | 2.559528 | 40.952 | 202.2 |
| with DataLoader | 1439739.645 | 351.499 | 2.844959 | 45.519 | 202.2 |
