# from classifiers import Classifier
from layers import Layer
from loss import Loss, MSE
import numpy as np

from typing import Type

class Model:
    layers: list[Layer]
    batch_size: int
    loss_fn: Loss
    def __init__(self, layers: list[Layer], loss_fn:Type[Loss] = MSE):
        self.layers = layers
        self.loss = loss_fn(layers[-1].out_features)
        
    def __call__(self, X: np.ndarray):
        self.batch_size = X.shape[0]
        X = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            X = layer.forward(X)
        return X
    
    def backwards(self):
        
        dY = np.ones([self.batch_size, self.layers[-1].out_features])
        for layer in self.layers[::-1]:
            dY = layer.backwards(dY)
        return dY