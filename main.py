import numpy as np
from model import Model
from layers import Linear

model = Model(
    layers = [
        Linear(2,5),
    ]
)

X = np.array([[1,2]])
print("X.shape", X.shape)
print("Input :", X.shape)
print("Linear:", model.layers[0].weights.shape)
Y = model(X)
print("Y shape:", Y.shape)
dLdX = model.backwards()
print("Linear layer 1 derivitives : ", model.layers[0].dW)
print(dLdX)