# Segment timing round-2 (delivery narrative)

- time_utc: 2026-07-23T14:22:54Z
- formal_path: auto (v1 if min(H,W)<256 else fused)

- {'path': 'v1', 'resolution': '64x64', 'fft_ms': 0.105, 'bridge_mul_ms': 1.153, 'ifft_ms': 0.971}
- {'path': 'fused', 'resolution': '64x64', 'rfft_ms': 1.354, 'mul_ms': 0.219, 'irfft_ms': 2.604}
- {'path': 'v1', 'resolution': '128x128', 'fft_ms': 0.19, 'bridge_mul_ms': 1.126, 'ifft_ms': 38.908}
- {'path': 'fused', 'resolution': '128x128', 'rfft_ms': 3.514, 'mul_ms': 0.228, 'irfft_ms': 7.1}
- {'path': 'v1', 'resolution': '256x256', 'fft_ms': 0.898, 'bridge_mul_ms': 1.49, 'ifft_ms': 109.135}
- {'path': 'fused', 'resolution': '256x256', 'rfft_ms': 12.832, 'mul_ms': 0.237, 'irfft_ms': 27.015}
