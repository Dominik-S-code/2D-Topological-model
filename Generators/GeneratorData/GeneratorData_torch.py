import torch

from .GeneratorData import GeneratorData

class GeneratorData_torch(GeneratorData):
    # def __init__(self, generator_data: GeneratorData, device='cpu'):
    #     """
    #     GeneratorData_torch class for GPU-based frame generation.
    #     This class inherits from GeneratorData and is designed to work with PyTorch tensors.
    #     """
    #     self.device = device
    #     self.n = generator_data.n
    #     self.HK2D = torch.tensor(generator_data.HK2D, device=self.device)
    #     self.W = torch.tensor(generator_data.W, device=self.device)
    #     self.M = torch.tensor(generator_data.M, device=self.device)
    #     self.pert = torch.tensor(generator_data.pert, device=self.device)
    #     self.device = device
    def __init__(self, n, HK2D, W, M, pert, device='cpu'):
        """
        GeneratorData_torch class for GPU-based frame generation.
        This class inherits from GeneratorData and is designed to work with PyTorch tensors.
        """
        super().__init__(n, HK2D, W, M, pert)
        self.device = device
        self.HK2D = HK2D.to(self.device)
        self.W = W.to(self.device)
        self.M = M.to(self.device)
        self.pert = pert.to(self.device)

    @classmethod
    def from_generator_data(cls, generator_data, device='cpu'):
        """
        Convert from CPU generator data to GPU generator data.
        """
        HK2D = torch.tensor(generator_data.HK2D)
        W = torch.tensor(generator_data.W)
        M = torch.tensor(generator_data.M)
        pert = torch.tensor(generator_data.pert)
        return cls(generator_data.n, HK2D, W, M, pert, device=device)