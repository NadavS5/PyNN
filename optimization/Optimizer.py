from abc import abstractmethod
from model import Model

class Optimizer:
    lr: float
    model: Model
    
    @abstractmethod
    def step():
        pass
