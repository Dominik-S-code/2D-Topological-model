import torch
import numpy as np

from .GeneratorData.GeneratorData import GeneratorData
from .GeneratorData.GeneratorData_torch import GeneratorData_torch
from .Generator import Generator


class Generator_torch(Generator):
    """
    Generator class for CPU-based frame generation.
    """

    def __init__(self, device='cuda'):
        self.device = device

    def generate_frame(self, input: GeneratorData, t:float, Vj=None) -> np.ndarray:
        if isinstance(input, GeneratorData_torch):
            pass
        elif isinstance(input, GeneratorData):
            input = GeneratorData_torch.from_generator_data(input, device=self.device)
        else:
            raise ValueError("Input must be of type GeneratorData.")
        n = (input.n**2) * 2
        Vj_torch = Vj
        if Vj_torch is None or not isinstance(Vj_torch, torch.Tensor):
            Vj_torch = torch.eye(n, dtype=torch.complex64, device=self.device)
        U_torch = torch.matrix_exp(1j * input.HK2D * t)
        Wt_torch = U_torch @ input.W @ torch.linalg.inv(U_torch)
        Wt_conj_T_torch = Wt_torch.conj().T
        one_torch = torch.eye(n, dtype=torch.complex64, device=self.device)

        def measure(j):          # j - site of measurement   # This gives us the actual scrambling results
            Vj_torch[2*j:2*j+2, 2*j:2*j+2] = input.pert
            Fj_torch = torch.linalg.det(one_torch + input.M @ (Wt_conj_T_torch @ Vj_torch.conj().T @ Wt_torch @ Vj_torch - one_torch))
            Cj_torch = 2*(1 - torch.real(Fj_torch))
            Vj_torch[2*j:2*j+2,2*j:2*j+2] = torch.eye(2, dtype=torch.complex64, device=self.device)
            return Cj_torch.cpu()
        
        result = np.zeros((input.n, input.n), dtype=np.float32)
        for a in range(input.n):
            for b in range(input.n):
                result[a, b] = measure(a*input.n + b)
                percent_done = (a*input.n + b) / (input.n**2) * 100
                print(f'{percent_done:.2f}%', end="\r")
        print("100.00%")
        return result
    
    def generate_batch(self, input: GeneratorData, t_batch:list[float]) -> list[np.ndarray]:
        input = GeneratorData_torch.from_generator_data(input, device=self.device)
        Vj = torch.eye(2*input.n**2, dtype=torch.complex64, device=self.device)
        result = [None] * len(t_batch)
        for i, t in enumerate(t_batch):
            print(f"Generating frame {i+1}/{len(t_batch)}")
            result[i] = self.generate_frame(input, t, Vj)
            
        return result