from .Loss import Loss
import numpy as np

class MSE(Loss):
    n_logits: int
    predictions: np.ndarray
    y: np.ndarray
    
    def __init__(self, n_logits: int):
        self.n_logits = n_logits
        self.y = None
        self.predictions = None
        
    def forward(self, predictions: np.ndarray, y: np.ndarray = None):
        #Li = (1/2 * n_logits) * Sigma(j=1,n) (y[j]-predictions[j])^2 for one batch item
        #L = 1/n Sigma(i=1,n) (Li)
        if y is not None:
            self.predictions = predictions
            self.y = y
        
        # return (np.square((y-predictions)) / 2 * y.shape[0] ).mean()
        return np.mean(np.square((predictions - y))) / 2

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backwards(self):
        if self.predictions is not None and self.y is not None:
            return self.predictions - self.y
        else:
            raise Exception("You didn't pass any targets/labels. pass loss_fn(y,labels)")