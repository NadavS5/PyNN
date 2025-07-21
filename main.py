import numpy as np
from model import Model
from layers import Linear
from loss import MSE

model = Model(
    layers = [
        Linear(2,5),
        Linear(5,2)
    ],
    loss_fn= MSE
)

X = np.array([[1,2],[1,6]])

Y = model(X)

print(model.loss.forward(Y, np.ones_like(Y)))