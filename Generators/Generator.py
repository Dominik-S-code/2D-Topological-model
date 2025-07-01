import numpy as np
from .GeneratorData.GeneratorData import GeneratorData

class Generator:
    """
    Interface for generating frames.
    """

    def generate_frame(self, input: GeneratorData, t:float, Vj=None) -> np.ndarray:
        """
        Generate frame based on the provided arguments.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def generate_batch(self, input: GeneratorData, t_batch:list[float]) -> list[np.ndarray]:
        Vj = np.identity(2*input.n**2, dtype=np.complex64)
        result = [None] * len(t_batch)
        for i, t in enumerate(t_batch):
            print(f"Generating frame {i+1}/{len(t_batch)}")
            result[i] = self.generate_frame(input, t, Vj)
            
        return result