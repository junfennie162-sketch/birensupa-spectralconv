# SCP blurb · FanDou Garden

**SpectralConv + FNO on BIREN** (Biren Flying Cup · Models & Operators)

SUPA / PyTorch extension implements FNO's core spectral convolution on a Biren GPU, then four FNO layers predict 2D vorticity.

## Results

| Module | Measured on Biren106B |
|--------|------------------------|
| Spectral convolution | Worst rel **2.170×10⁻⁷**; formal idle 64/128/256 **3.797 / 8.037 / 29.295 ms** |
| FNO-NS | Official public NS64 (1000/128) relative L2 **0.035012** |

Figures from `fno_ns/render_official_demo.py`. Two covers: typical-sample triple; best / typical / worst. See `demo/media/README.md`.

## Run

```bash
source /usr/local/birensupa/sdk/1.11.0.0.rc2/scripts/brsw_set_env.sh
export SUPA_BASE=/usr/local/birensupa/sdk/1.11.0.0.rc2
bash scripts/validate.sh
```

Team: FanDou Garden · North University of China · Track 5
