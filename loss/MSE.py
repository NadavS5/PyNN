from .Loss import Loss

class MSE(Loss):
    in_features: int
    def __init__(self, in_features):
        self.in_features = in_features
        self.name = "MSE"
    
    def forward():
        pass

    def backwards():
        pass