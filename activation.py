import numpy as np

class ReLU: 
    def __init__(self): 
        pass 
        # self.mask = np.zeros()
    def forward(self, x): 
        return np.maximum(0, x) # returns 0 if x is negative, other wise, return x
    
    def __call__(self, x):
        return self.forward(x)