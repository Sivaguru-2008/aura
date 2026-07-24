# Performance Benchmark

- **Device:** NVIDIA GeForce RTX 5050 Laptop GPU · torch 2.11.0+cu128
- **Platform:** Windows-11-10.0.26200-SP0

- **CPU latency (1 img):** 47.882 ms (p95 51.65 ms, 20.88 img/s)
- **GPU latency (1 img):** 19.937 ms (p95 23.908 ms, 50.16 img/s)
- **Mixed precision:** 0.81x (21.006 → 25.948 ms/batch of 8)
- **Peak GPU memory:** 693.98 MB

## Batch throughput

| batch | img/s | ms/batch |
|---|---|---|
| 1 | 45.3 | 22.073 |
| 8 | 408.86 | 19.567 |
| 16 | 550.69 | 29.054 |
| 32 | 593.31 | 53.935 |
| 64 | 558.07 | 114.68 |