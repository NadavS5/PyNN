from .Optimizer import Optimizer
from model import Model

class SGD(Optimizer):
    
    def __init__(self, model: Model, lr: float):
        self.model = model
        self.lr = lr
    
    def step(self):
        if self.model.last_loss is None :
            raise Exception("you can't call step when you didnt pass the y in forwards") 
        for layer in self.model.layers:
            layer.W -= layer.dW * self.lr * self.model.last_loss