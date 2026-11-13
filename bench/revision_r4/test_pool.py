import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import concurrent.futures
import time

def work(x):
    import torch
    torch.set_num_threads(2)
    t0 = time.time()
    # do a matrix multiply and small circuit
    a = torch.randn(200, 200, dtype=torch.complex128)
    for _ in range(500):
        a = a @ a / 200.0
    return x, time.time() - t0

if __name__ == "__main__":
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(work, range(8)))
    print(f"Finished 8 tasks across 4 workers in {time.time() - t0:.2f}s: {res}")
