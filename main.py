'''
Data -> model -> loss calculation -> gradient calculation -> gradient update 
'''
import numpy as np 
from network import Net 
from loss import MSE 

x = np.array([1,2,3,4])
y = np.array([2,4,6,8]) 

net = Net() 
mse = MSE() 

output = net.forward(x)
loss = mse.loss(output, y)

print('output:', output) 
print('loss:', loss)