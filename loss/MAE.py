from .Loss import Loss

class MAE(Loss):
    in_features: int
    def __init__(self, in_features):
        self.in_features = in_features
        self.name = "MAE"
    
    def forward():
        pass

    def backwards():
        pass    