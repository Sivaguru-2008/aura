# Performance Benchmark

- **Device:** NVIDIA GeForce RTX 5050 Laptop GPU · torch 2.11.0+cu128
- **Platform:** Windows-11-10.0.26200-SP0

- **CPU latency (1 img):** 53.999 ms (p95 64.21 ms, 18.52 img/s)
- **GPU latency (1 img):** 24.271 ms (p95 30.017 ms, 41.2 img/s)
- **Mixed precision:** 0.781x (25.314 → 32.426 ms/batch of 8)
- **Peak GPU memory:** 693.98 MB

## Batch throughput

| batch | img/s | ms/batch |
|---|---|---|
| 1 | 35.37 | 28.27 |
| 8 | 349.68 | 22.878 |
| 16 | 582.7 | 27.458 |
| 32 | 551.54 | 58.02 |
| 64 | 519.24 | 123.257 |