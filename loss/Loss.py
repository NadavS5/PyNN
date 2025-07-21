import numpy as np

from abc import abstractmethod

class Loss():
    n_logits: int
    name: str
    
    @abstractmethod
    def forward(predictions: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def backwards():
        pass