# XOR is a function in boolean alzebra
# Easier to calculate actual values 
# Easy  to fit 
# Not Linearly seperable


#neural network structure 
# ip1  -> 0----⬇️
# ip2  -> 0 -> 0 --->0--> op
# ip3  -> 0---> 0---⬆️

# 2 inputs --> 3 neurons --> 1 output

import logging
from models.denselayer import Dense
from models.loss_functions import mse, mse_prime
from models.tanh import TanH
import numpy as np

X= np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))
logger = logging.Logger('XOR')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('xor.log')
logger.addHandler(fh)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


network =[
    Dense(2,3),
    TanH(),
    Dense(3,1),
    TanH()
]

epochs = 10000
learning_rate = 0.01
logger.warning("NEW SESSION START")
for epoch in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        # Forward Pass
        output =x 
        for layer in network:
            output = layer.forward(output)
        #Error Calc -- Just for our sake 
        error += mse(y,output)
        #Backward Pass
        error_gradient= mse_prime(y,output) 
        for layer in reversed(network):
            error_gradient = layer.backward(error_gradient,learning_rate)
    logger.warning("Epoch= "+str(epoch+1)+" Error= "+str(round(error/len(X),4)))

def predict(network,_input):
    output = _input
    for layer in network:
        output = layer.forward(output)
    return output


print(predict(network,[[0],[0]]))

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()