import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Input
from .GeneratorData import GeneratorData

class Initializer:
    """
    Interface for initializers.
    """

    def initialize(self, input: Input) -> GeneratorData:
        """
        Create GeneratorData object based on the Input.
        """
        raise NotImplementedError("Subclasses should implement this method.")