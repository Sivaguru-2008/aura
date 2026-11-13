import time
import torch
import numpy as np
from aura.services.fusion.device import entangler_pairs

def profile():
    n = 8
    n_layers = 4
    pairs = entangler_pairs("ring", n)
    B = 50
    dev = "cuda"

    def apply1(psi, q, m):
        psi = psi.movedim(q + 1, -1)
        if m.dim() == 2:
            psi = psi @ m.T
        else:
            shp = psi.shape
            psi = torch.einsum("bij,b...j->b...i", m, psi.reshape(shp[0], -1, 2)).reshape(shp)
        return psi.movedim(-1, q + 1)

    # Let's test apply1 vs batched tensor contraction
    psi = torch.randn((B,) + (2,) * n, dtype=torch.complex128, device=dev)
    m2 = torch.randn(2, 2, dtype=torch.complex128, device=dev)
    mb = torch.randn(B, 2, 2, dtype=torch.complex128, device=dev)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        psi = apply1(psi, 3, m2)
    torch.cuda.synchronize()
    print(f"100 apply1 (m2): {time.time() - t0:.4f}s")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        psi = apply1(psi, 3, mb)
    torch.cuda.synchronize()
    print(f"100 apply1 (mb): {time.time() - t0:.4f}s")

if __name__ == "__main__":
    profile()
