# SpectralConv suFFT accuracy

- time_utc: 2026-07-25T12:17:01Z
- worst_rel: 2.1701555345060285e-07
- ok: True

```json
[
  {
    "case": "tiny_8x8",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin2_Cout3_8x8",
    "modes": "2x2",
    "max_abs": 1.4901161193847656e-08,
    "max_rel": 8.007249089634666e-08,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "small_32x32",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin4_Cout4_32x32",
    "modes": "8x8",
    "max_abs": 8.195638656616211e-08,
    "max_rel": 1.8928233107327004e-07,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "target_64x64",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin4_Cout4_64x64",
    "modes": "12x12",
    "max_abs": 8.195638656616211e-08,
    "max_rel": 2.1701555345060285e-07,
    "threshold": 0.0001,
    "ok": true
  }
]
```
