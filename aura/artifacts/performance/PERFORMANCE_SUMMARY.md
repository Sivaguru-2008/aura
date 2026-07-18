# Performance Benchmark

- **Device:** NVIDIA GeForce RTX 5050 Laptop GPU · torch 2.11.0+cu128
- **Platform:** Windows-11-10.0.26200-SP0

- **CPU latency (1 img):** 83.171 ms (p95 107.134 ms, 12.02 img/s)
- **GPU latency (1 img):** 29.102 ms (p95 42.86 ms, 34.36 img/s)
- **Mixed precision:** 0.823x (25.648 → 31.167 ms/batch of 8)
- **Peak GPU memory:** 693.45 MB

## Batch throughput

| batch | img/s | ms/batch |
|---|---|---|
| 1 | 26.45 | 37.804 |
| 8 | 300.62 | 26.612 |
| 16 | 524.28 | 30.518 |
| 32 | 618.13 | 51.769 |
| 64 | 571.43 | 112.0 |