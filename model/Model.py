# from classifiers import Classifier
from layers import Layer
from loss import Loss, MSE
import numpy as np

from typing import Type

class Model:
    layers: list[Layer]
    batch_size: int
    loss: Loss
    last_loss: float
    y: np.ndarray
    
    def __init__(self, layers: list[Layer], loss_fn:Type[Loss] = MSE):
        self.layers = layers
        self.loss = loss_fn(layers[-1].out_features)
        self.last_loss = None
    def __call__(self, X: np.ndarray, y = None):
        self.batch_size = X.shape[0]
        X = self.layers[0].forward(X)
    
        for layer in self.layers[1:]:
            X = layer.forward(X)
        
        if y is not None:
            self.last_loss = self.loss.forward(X,y)
            return X, self.last_loss
        self.last_loss.y = None
        
        return X
    
    def backwards(self):
        #check if there is saved y
        #if not calc the grads w.r.t last layer. (and not able to call optimizer.step())
        dy = None
        if self.loss.y is not None:
            dY = self.loss.backwards()
        else:
            dY = np.ones([self.batch_size, self.layers[-1].out_features])
        for layer in self.layers[::-1]:
            dY = layer.backwards(dY)
        return dY