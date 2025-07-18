from ..classifiers import Classifier
from ..layers import Layer

import numpy as np

class Model:
    layers: list[Layer]
    def __init__(self, layers: list[Layer], loss = None):
        self.layers = layers
    
    def __call__(self, X):
        X = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            X = layer.forward(X)
        return X
    
    def backwards(self):
        dY = np.ones(self.layers[-1].out_features)
        for layer in self.layers[::-1]:
            dY = layer.backwards(dY)