from .Layer import Layer, np

class Linear(Layer):
    
    X: np.ndarray
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.dW = np.zeros_like(self.W)
        
    def forward(self, X: np.ndarray):
        """
        :param inputs 
        
        """
        
        self.X = X
        return X @ self.W.T 
 
    def backwards(self, dY: np.array) -> np.ndarray:
        """backwards function for Linear layer

        Args:
            dY (out_features): dL w.r.t dY

        Returns:
            np.matrix(in_features) : dL w.r.t dX
            dX becomes dY for the previous layer
        """ 
        
       
        #instead of outer, for batching
        self.dW += dY.T @ self.X

        # print(dY.shape)
        # print(self.W.shape)
        dX = dY @ self.W
        
        return dX