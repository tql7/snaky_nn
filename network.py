import numpy as np 
from activation import ReLU
'''
task: regression 
'''

class Layer: 
    def __init__(self, input_size, num_neurons_next, activation):
        pass

        self.input_size = input_size
        self.num_neurons_next = num_neurons_next
        # size k x n where k = num_neurons of next layer and n = dim of input 
        self.W = np.random.rand(num_neurons_next, input_size)
        self.b = np.random.rand(num_neurons_next)
        self.activation = activation
    
    def forward(self, x): 
        return self.activation(np.matmul(self.W, x) + self.b) 
    
class Net: 
    def __init__(self): 
        self.relu = ReLU() 
        self.layer1 = Layer(4, 5, self.relu) # 4 is the dimension of the input data 
        self.layer2 = Layer(5, 5, self.relu)
        self.head = Layer(5, 4, self.relu) # regression head 

    def forward(self, x): 
        z1 = self.layer1.forward(x)
        z2 = self.layer2.forward(z1)
        output = self.head.forward(z2)

        return output 

