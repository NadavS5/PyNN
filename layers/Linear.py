from Layer import Layer, np

class Linear(Layer):
    W: np.matrix
    dW: np.matrix
    X: np.matrix
    in_features: int
    out_features: int
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = np.random.randn(out_features, in_features) * 0.01
        
    def forward(self, X: np.matrix):
        """
        :param inputs 
        
        """
        
        self.X = X
        
        return self.weights @ X
 
    def backwards(self, dY: np.array) -> np.matrix:
        """backwards function for Linear layer

        Args:
            dY (out_features): dL w.r.t dY

        Returns:
            np.matrix(in_features) : dL w.r.t dX
            dX becomes dY for the previous layer
        """ 
        
        self.dW = np.outer(dY,self.X)
        dX = self.weights.T @ dY
        
        return dX