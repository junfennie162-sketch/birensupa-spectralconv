# P6 · irfft C2C col_out stage cache · 2026-07-29

## Tried
- Cache C2C `col_out` (fully overwritten) — OK
- Strided `copy_` without `.contiguous()` — **ROLLBACK** (Biren CopyD2D shape error)
- Return contiguous without stage — mixed formal noise

## Keep
C2C col_out stage cache + packed view+contiguous copy (safe).

## Formal
4.633 / 9.558 / 35.716 ms. Multi-call PASS.
