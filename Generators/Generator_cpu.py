import numpy as np
from scipy.linalg import expm

from .Generator import Generator
from .GeneratorData.GeneratorData import GeneratorData

class Generator_cpu(Generator):
    """
    Generator class for CPU-based frame generation.
    """

    def __init__(self):
        pass

    def generate_frame(self, input: GeneratorData, t:float, Vj=None) -> np.ndarray:
        U = expm(np.complex128(1j) * input.HK2D * t)
        Wt = U @ input.W @ np.linalg.inv(U)
        Wt_conj_T = Wt.conj().T
        One = np.identity(2*input.n**2, dtype=np.complex64)
        if Vj is None:
            Vj = np.identity(2*input.n**2, dtype=np.complex64)
        def measure(j):          # j - site of measurement   # This gives us the actual scrambling results
            Vj[2*j:2*j+2, 2*j:2*j+2] = input.pert
            Fj = np.linalg.det(One + input.M @ (Wt_conj_T @ Vj.conj().T @ Wt @ Vj - One))
            Cj = 2*(1 - np.real(Fj))
            Vj[2*j:2*j+2,2*j:2*j+2] = np.identity(2, dtype=np.complex64)
            return Cj
        
        result = np.zeros((input.n, input.n), dtype=np.float32)
        for a in range(input.n):
            for b in range(input.n):
                result[a, b] = measure(a*input.n + b)
                percent_done = (a*input.n + b) / (input.n**2) * 100
                print(f'{percent_done:.2f}%', end="\r")
        print("100.00%")
        return result

    def generate_batch(self, input: GeneratorData, t_batch:list[float]) -> list[np.ndarray]:
        return super().generate_batch(input, t_batch)