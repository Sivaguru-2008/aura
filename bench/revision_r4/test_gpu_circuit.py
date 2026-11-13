import time
import torch
import numpy as np

print(f"Device name: {torch.cuda.get_device_name(0)}")

# Define GPU version of torch_circuit
from aura.services.fusion.device import entangler_pairs

def gpu_torch_circuit(n_qubits: int, n_layers: int, entangler: str):
    pairs = entangler_pairs(entangler, n_qubits)
    n = n_qubits

    def apply1(psi, q, m):
        psi = psi.movedim(q + 1, -1)
        if m.dim() == 2:
            psi = psi @ m.T
        else:
            shp = psi.shape
            psi = torch.einsum("bij,b...j->b...i", m, psi.reshape(shp[0], -1, 2)).reshape(shp)
        return psi.movedim(-1, q + 1)

    def cnot(psi, c, t):
        psi = psi.clone()
        idx1 = [slice(None)] * (n + 1)
        idx1[c + 1] = 1
        sub = psi[tuple(idx1)]
        tt = t if t < c else t - 1
        psi[tuple(idx1)] = sub.flip(tt + 1)
        return psi

    def ry(th):
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

    def rz(th):
        e = torch.exp(-0.5j * th.to(torch.complex128))
        return torch.stack([torch.stack([e, torch.zeros_like(e)], -1),
                            torch.stack([torch.zeros_like(e), e.conj()], -1)], -2)

    def f(X, theta):
        X = X.to(torch.float64)
        dev = X.device
        B = X.shape[0]
        psi = torch.zeros((B,) + (2,) * n, dtype=torch.complex128, device=dev)
        psi[(slice(None),) + (0,) * n] = 1.0
        for i in range(n):
            psi = apply1(psi, i, ry(np.pi * X[:, i]).to(torch.complex128))
        for layer in range(n_layers):
            for i in range(n):
                psi = apply1(psi, i, ry(theta[layer, i, 0]).to(torch.complex128))
                psi = apply1(psi, i, rz(theta[layer, i, 1]))
            for c, t in pairs:
                psi = cnot(psi, c, t)
        p = (psi.real ** 2 + psi.imag ** 2)
        out = []
        for i in range(n):
            m = p.movedim(i + 1, -1).reshape(B, -1, 2).sum(1)
            out.append(m[:, 0] - m[:, 1])
        return torch.stack(out, -1)

    return f

f_cuda = gpu_torch_circuit(8, 4, "ring")
X = torch.rand(50, 8, device="cuda", dtype=torch.float64)
theta = torch.randn(4, 8, 2, device="cuda", dtype=torch.float64, requires_grad=True)

# Warmup
out = f_cuda(X, theta)
loss = out.sum()
loss.backward()
torch.cuda.synchronize()

print("GPU Warmup succeeded!")
t0 = time.time()
for _ in range(100):
    theta.grad = None
    out = f_cuda(X, theta)
    loss = out.sum()
    loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"100 batches on RTX 5050 GPU completed in {dt:.3f}s ({dt/100*1000:.2f} ms/batch)!", flush=True)

# Verify exact match with CPU version
from aura.ml.evaluation.v6_confirmatory import torch_circuit
f_cpu = torch_circuit(8, 4, "ring")
ref_cpu = f_cpu(X.cpu(), theta.cpu())
print(f"Max abs diff between GPU and CPU: {(out.cpu() - ref_cpu).abs().max().item()}", flush=True)
