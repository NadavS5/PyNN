# from .MAE import MAE
# from .MSE import MSE
# from .CrossEntropy import CrossEntropy

from abc import abstractmethod

class Loss():
    in_features: int
    name: str
    
    @abstractmethod
    def forward(LastLayer: str) -> float:
        pass

    @abstractmethod
    def backwards():
        pass