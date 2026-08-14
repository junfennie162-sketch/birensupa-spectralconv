# SpectralConv suFFT accuracy

- time_utc: 2026-07-22T12:56:51Z
- worst_rel: 2.1152186444815025e-07
- ok: True

```json
[
  {
    "case": "tiny_8x8",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin2_Cout3_8x8",
    "modes": "2x2",
    "max_abs": 9.313225746154785e-09,
    "max_rel": 9.272938108979527e-08,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "small_32x32",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin4_Cout4_32x32",
    "modes": "8x8",
    "max_abs": 5.960464477539063e-08,
    "max_rel": 1.88624219407616e-07,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "target_64x64",
    "path": "sufft_r2c_supa_mul_sufft_c2r",
    "shape": "B2_Cin4_Cout4_64x64",
    "modes": "12x12",
    "max_abs": 5.960464477539063e-08,
    "max_rel": 2.1152186444815025e-07,
    "threshold": 0.0001,
    "ok": true
  }
]
```
