import time
import torch
import numpy as np
from aura.services.fusion.device import entangler_pairs

def build_fast_gpu_circuit(n_qubits: int, n_layers: int, entangler: str):
    pairs = entangler_pairs(entangler, n_qubits)
    n = n_qubits
    
    # Precompute CNOT permutations
    cnot_perms = []
    for c, t in pairs:
        # bit convention: qubit 0 is most significant or least?
        # In torch_circuit: idx1[c+1]=1, sub.flip(tt+1)
        # Let's verify exact behavior against torch_circuit!
        pass

if __name__ == "__main__":
    print("Testing...")
