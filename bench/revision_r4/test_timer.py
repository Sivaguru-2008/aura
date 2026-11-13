import time
import torch
import numpy as np
from aura.ml.evaluation import v6_confirmatory as v6
from aura.ml.evaluation import matched_optimiser as mo

Xtr, ytr, Xcal, ycal = v6.dev_splits()
print(f"Xtr: {Xtr.shape}, Xcal: {Xcal.shape}")

# Test 1 seed of C2 VQC
t0 = time.time()
mo.SEED = 7
mo.WEIGHT_DECAY = 0.0
build = v6._build_quantum_fast(8, 4, "ring", 0.5, 0.01)
rec, params, _ = mo._train_under_protocol(build, Xtr, ytr, Xcal, ycal, 0.01)
t1 = time.time()
print(f"VQC train time: {t1-t0:.2f}s, epochs: {rec['epochs_run']}, cal_nll: {rec['best_calibration_nll']:.4f}")

# Test scoring on test set (56534)
Xte, yte = v6.test_set()
t2 = time.time()
Lcal = v6._logits(lambda X: params[1] @ v6.torch_circuit(8, 4, "ring")(0.5 * X[:, :8], params[0]).T + params[2].unsqueeze(1), Xcal)
# Or using the forward function directly
t3 = time.time()
print(f"Cal logits time: {t3-t2:.2f}s")
