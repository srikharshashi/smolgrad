import numpy as np
from models.layer import Layer

class ActivationLayer(Layer):
    
    def __init__(self,activation,activation_prime):
        self.activation=activation   #f(x)
        self.activation_prime=activation_prime #f'(x)
        

    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient,learning_rate):
        return np.multiply(output_gradient ,self.activation_prime(self.input)) 

        


    

        