from models.layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights = self.xavier_init(output_size,input_size)
        self.bias = np.zeros(shape=(output_size,1))
        
    def xavier_init(self,input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
    
    def he_init(self,input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights,self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient,self.input.T)
        input_gradient = np.dot(self.weights.T,output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    
    