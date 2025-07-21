from abc import abstractmethod
import numpy as np

class Layer:
    W: np.ndarray
    dW: np.ndarray
    
    @abstractmethod
    def forward(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def backwards(self,dY) -> np.ndarray:
        pass