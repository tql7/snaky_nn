import numpy as np 

class MSE: 
    def __init__(self): 
        pass

    def loss(self, y_pred, y): 
        return 0.5 * np.square(y_pred - y)
