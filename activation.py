import numpy as np
import net

class ReLU(net.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, X):
        Y = X
        Y[Y < 0] = 0.
        return Y
    
    def backward(self, gradient, learning_rate):
        next_grad = gradient
        next_grad[next_grad < 0] = 0.
        #next_grad[next_grad >= 0] = 1.
        return next_grad
    
class LeakyReLU(net.Module):
    def __init__(self, grad = 0.05):
        self.grad = grad
        
    def __call__(self, X):
        X[X < 0] = X[X < 0] * self.grad
        return X
    
    def backward(self, gradient, learning_rate):
        next_grad = gradient
        next_grad[next_grad < 0] *= self.grad
        #next_grad[next_grad >= 0] = 1.
        return next_grad
    
class Sigmoid(net.Module):
    def __call__(self, X):
        return 1 / (1 + np.e**X)
    
    def backward(self, gradient, learning_rate):
        sig = self.__call__(gradient)
        return sig * (1. - sig) * gradient