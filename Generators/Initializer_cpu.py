import numpy as np
from scipy.linalg import expm

import Input
from .GeneratorData.GeneratorData import GeneratorData
from .Initializer import Initializer


class Initializer_cpu(Initializer):
    """
    Initializer using CPU.
    """

    def __init__(self):
        super().__init__()

    def initialize(self, input: Input) -> GeneratorData:
        """
        Create GeneratorData object based on the Input.
        """
        self._basic_data_preparation(input.n, input.j00, input.V0)
        self._hamiltonian(input.n)
        pert = self._perturbation(input.V0)
        HK2D = self._tensor_products(input.n, input.R, input.Delta, input.J, input.mu)
        # HK2D = self._modify_Hamiltonian(HK2D, input)
        self._eigenvalues_eigenvectors(input.n, HK2D)
        W, M = self._M_W_initialize(input.n, input.W0)
        
        data = GeneratorData(input.n, HK2D, W, M, pert)
        return data
    
    def _basic_data_preparation(self, n, j00, V0):
        self.taux = np.array([[0,1],[1,0]], dtype=np.complex64)
        self.tauy = np.array([[0,-1j],[1j,0]], dtype=np.complex64)
        self.tauz = np.array([[1,0],[0,-1]], dtype=np.complex64)
        self.j0 = np.int32((j00[0]-1)*n + j00[1] - 1)  # Coordinate in the matrix

    def _perturbation(self, V0):
        return np.array(expm(1j*V0*self.tauz), dtype=np.complex64)

    def _hamiltonian(self, n):
        self.Hx = np.diag(np.ones(n-1),1) # x direction hopping
        self.Hy = np.diag(np.ones(n-1),1) # y direction hopping

    def _tensor_products(self, n, R, Delta, J, mu):
        for a in range(R):
            self.Hx = np.kron(np.identity(n), self.Hx)
            self.Hx = np.array(np.kron(self.Hx, Delta*1j*self.taux - J*self.tauz), dtype=np.complex64)  # x direction hopping
            self.Hy = np.kron(self.Hy,np.identity(n))
            self.Hy = np.array(np.kron(self.Hy, Delta*1j*self.tauy - J*self.tauz), dtype=np.complex64) # y direction hopping

        HK2D = self.Hx + self.Hy
        HK2D += HK2D.conj().T - mu*np.array(np.kron(np.identity(n**2),self.tauz), dtype=np.complex64)   # Final Hamiltonian
        return HK2D
    
    def _modify_Hamiltonian(self, HK2D, input: Input):
        """
        Modify the Hamiltonian by modifying the chemical potential term in a part of the system.
        """
        modified_mu = 4
        a = input.n**2 // 2  # set 'a' as needed
        transform_area = np.zeros(input.n**2, dtype=np.int32)
        # transform_area = np.array([2*np.random.rand()-1 for a in range(input.n**2)], dtype=np.float32) # Disorder in the whole system
        transform_area[a:] = 1        # Bottom half of the system is modified
        # transform_area[:a] = 1        # Top half of the system is modified
        # transform_area[(input.n//2-2)*input.n:(input.n//2-2)*input.n+input.n//2] = 1              #|
        # transform_area[(input.n//2-1)*input.n:(input.n//2-1)*input.n+input.n//2] = 1              #|
        # transform_area[(input.n//2)*input.n:(input.n//2)*input.n+input.n//2] = 1                  #| A small part of the system on the left is modified (a 5 by n/2 rectangle)
        # transform_area[(input.n//2+1)*input.n:(input.n//2+1)*input.n+input.n//2] = 1              #|
        # transform_area[(input.n//2+2)*input.n:(input.n//2+2)*input.n+input.n//2] = 1              #|
        # transform_area[(input.n//2-1)*input.n:(input.n//2-1)*input.n+input.n] = [0.4*(2*np.random.rand()-1) for i in range(input.n)]        #| Horizontal line in the middle of the system
        # transform_area[(input.n//2-2)*input.n:(input.n//2-2)*input.n+input.n] = [0.4*(2*np.random.rand()-1) for i in range(input.n)]        #|
        transform_area = np.diag(transform_area)
        HK2D += modified_mu*np.array(np.kron(transform_area, self.tauz), dtype=np.complex64)  # Final Hamiltonian
        return HK2D

    def _eigenvalues_eigenvectors(self, n, HK2D):
        EVal, EVec = np.linalg.eigh(HK2D)
        surface_size = 2*n**2
        self.ESys = [[None, None]]*surface_size

        for a in range(surface_size):
            self.ESys[a] = [np.round(EVal[a],8), -(EVec.conj().T)[surface_size-1-a]]

    def _M_W_initialize(self, n, W0):
        W = np.identity(2*n**2, dtype=np.complex64)
        W[2*self.j0:2*self.j0+2,2*self.j0:2*self.j0+2] = expm(1j*W0*self.tauz)

        M1 = np.round(np.array([self.ESys[a][1] for a in range(n**2)], dtype=np.complex64), 8)
        M = np.round(np.dot(M1.conj().T, M1), 8)     # Correlation matrix
        return W, M
