import time
import torch
import numpy as np
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation import matched_optimiser as mo

Xtr, ytr, Xcal, ycal = v6.dev_splits()

for threads in [1, 2, 4]:
    torch.set_num_threads(threads)
    mo.SEED = 7
    mo.WEIGHT_DECAY = 0.0
    mo.MAX_EPOCHS = 5
    build = v6._build_quantum_fast(8, 4, "ring", 0.5, 0.01)
    t0 = time.time()
    rec, _, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, 0.01)
    dt = time.time() - t0
    print(f"Threads={threads}: 5 epochs in {dt:.2f}s ({dt/5:.2f}s/epoch)")
