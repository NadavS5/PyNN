from .Loss import Loss

class CrossEntropy(Loss):
    
    def __init__(self, in_features):
        self.in_features = in_features
        self.name = "CrossEntropy"
    
    def forward(correct_label: int):        
        pass

    def backwards():
        pass