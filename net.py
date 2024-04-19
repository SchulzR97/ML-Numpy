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
        Y = X
        for module in self.modules:
            Y = module(Y)

        return Y
    
    def eval(self):
        self.__eval__ = True
        for module in self.modules:
            module.eval()

    def train(self):
        self.__eval__ = False
        for module in self.modules:
            module.train()
    
    def backward(self, gradient, learning_rate):
        grad = gradient
        for module in reversed(self.modules):
            grad = module.backward(grad, learning_rate)

        return grad
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)