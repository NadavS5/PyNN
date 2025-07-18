from abc import ABC, abstractmethod, abstractproperty
import numpy as np

class Layer:
    W: np.matrix
    dW: np.matrix
    
    @abstractmethod
    def forward(self) -> np.matrix:
        pass
    
    @abstractmethod
    def backwards(self,dY):
        pass