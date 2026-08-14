# SpectralConv Extension accuracy (dual-corner)

- time_utc: 2026-07-22T15:34:19Z
- worst_rel: 2.1199665255932433e-07
- ok: True

```json
[
  {
    "case": "tiny_8x8",
    "shape": "B2_Cin2_Cout3_8x8",
    "modes": "2x2",
    "rel": 8.783144571714367e-08,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "small_32x32",
    "shape": "B2_Cin4_Cout4_32x32",
    "modes": "8x8",
    "rel": 1.2002523689593637e-07,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "target_64x64",
    "shape": "B2_Cin4_Cout4_64x64",
    "modes": "12x12",
    "rel": 1.3147479627342867e-07,
    "threshold": 0.0001,
    "ok": true
  },
  {
    "case": "official_128",
    "shape": "B4_Cin32_Cout64_128x128",
    "modes": "16x16",
    "rel": 2.1199665255932433e-07,
    "threshold": 0.0001,
    "ok": true
  }
]
```
