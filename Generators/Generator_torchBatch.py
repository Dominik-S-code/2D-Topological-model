import torch
import numpy as np

from .GeneratorData.GeneratorData import GeneratorData
from .GeneratorData.GeneratorData_torch import GeneratorData_torch
from .Generator import Generator


class Generator_torchBatch(Generator):
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
            Vj_torch = self._init_Vj(input.n, input.pert)
        U_torch = torch.matrix_exp(1j * input.HK2D * t)
        Wt_torch = U_torch @ input.W @ torch.linalg.inv(U_torch)
        Wt_conj_T_torch = Wt_torch.conj().T
        one_torch = torch.eye(n, dtype=torch.complex64, device=self.device)

        accumulator = Vj_torch.conj().transpose(-2, -1)
        accumulator = torch.matmul(Wt_conj_T_torch.unsqueeze(0).unsqueeze(0), accumulator)
        accumulator = torch.matmul(accumulator, Wt_torch.unsqueeze(0).unsqueeze(0))
        accumulator = torch.matmul(accumulator, Vj_torch)
        accumulator = accumulator - one_torch.unsqueeze(0).unsqueeze(0)
        accumulator = torch.matmul(input.M.unsqueeze(0).unsqueeze(0), accumulator)
        accumulator = accumulator + one_torch.unsqueeze(0).unsqueeze(0)
        accumulator = accumulator.view(-1, n, n)
        Fj_torch = torch.linalg.det(accumulator)
        Cj_torch = 2*(1 - torch.real(Fj_torch))
        result = Cj_torch.view(input.n, input.n).cpu().numpy()
        return result
    
    def generate_batch(self, input: GeneratorData, t_batch:list[float]) -> list[np.ndarray]:
        input = GeneratorData_torch.from_generator_data(input, device=self.device)
        Vj = self._init_Vj(input.n, input.pert)
        result = [None] * len(t_batch)
        for i, t in enumerate(t_batch):
            print(f"Generating frame {i+1}/{len(t_batch)}")
            result[i] = self.generate_frame(input, t, Vj)
            
        return result
    
    def _init_Vj(self, n, pert):
        """
        Initialize Vj as an identity matrix of size 2*n^2 and copies it creating n*n matrix of identity matrices.
        """
        matrix_size = 2*n**2
        Vj = torch.eye(matrix_size, dtype=torch.complex64, device=self.device)
        Vj = Vj.repeat(n, n, 1, 1)
        for a in range(n):
            for b in range(n):
                j = a*n + b
                Vj[a, b, 2*j:2*j+2, 2*j:2*j+2] = pert
        return Vj