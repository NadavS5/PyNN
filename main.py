import numpy as np
from model import Model
from layers import Linear
from loss import MSE
from optimization import SGD

model = Model(
    layers = [
        Linear(2,5),
        Linear(5,6)
    ],
    loss_fn= MSE
)


X = np.array([[1,2]])

optim = SGD(model, 5e-5)

for _ in range(1009):
    Y,loss = model(X, [[1,2,3,4,5,6]])
    print("loss: ", loss, "prediceted: ", Y)
    
    model.backwards()
    optim.step()
    