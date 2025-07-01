import numpy as np

class Input:
    def __init__(self, n: np.int32):
        self.n = n                      # System size
        self.mu = np.float32(2)         # Chemical potential
        self.R = np.int32(1)            # Number of neighbour hoppings
        self.J = np.float32(1)          # Hopping strength
        self.Delta = np.float32(0.45)   # Pairing energy

        self.W0 = np.float32(5)         #| Parameters of the perturbation (don't really matter much)
        self.V0 = np.float32(5)         #|
        self.j00 = np.array((1, self.n//2), dtype=np.int32)     # Site of initial perturbation
