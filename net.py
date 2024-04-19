import numpy as np
import pickle as pkl

class Module():
    def __init__(self):
        self.__eval__ = False

    def __call__(self, X):
        raise Exception('Not implemented!')
    
    def eval(self):
        self.__eval__ = True

    def train(self):
        self.__eval__ = False
    
    def backward(loss, learning_rate):
        raise Exception('Not implemented!')
    
class NeuralNetwork(Module):
    def __init__(self, modules):
        self.modules = modules

    def __call__(self, X):
        '''
        Forward path

        Parameters
        ----------
        X : input to the neural network
        :return: output of the neural network
        '''
        Y = X
        for module in self.modules:
            Y = module(Y)

        return Y
    
    def backward(self, gradient:np.array, learning_rate:np.float64) -> np.ndarray:
        """
        Backward path

        :param gradient: gradient of the loss function that should be backpropageted threw the network layers
        :param learning_rate: a number between 0 and 1 to smooth the learning process
        :return: gradient of the backward path
        """
        grad = gradient
        for module in reversed(self.modules):
            grad = module.backward(grad, learning_rate)

        return grad
    
    def eval(self):
        self.__eval__ = True
        for module in self.modules:
            module.eval()

    def train(self):
        self.__eval__ = False
        for module in self.modules:
            module.train()
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)