import numpy as np
import net

class DenseLayer(net.Module):
    def __init__(self, n_in:int, n_out:int):
        self.w = -1. + 2 * np.random.random((n_out, n_in))
        #self.w = np.zeros((n_out, n_in))
        self.b = -1. + 2 * np.random.random((n_out, 1))
        #self.b = np.zeros((n_out, 1))

        self.w *= .5
        self.b *= .5

    def __call__(self, X):
        #if X.shape[0] >= 1 and X.shape[1] >= 1:
        Y = np.zeros((X.shape[0], self.w.shape[0]))
        self.input = X
        for i in range(X.shape[0]):
            x = X[i].reshape((X[i].shape[0], 1))
            y = np.dot(self.w, x) + self.b
            Y[i] = y.T
        return Y
        #else:
        #    Y = np.dot(self.w, X) + self.b
        #    self.grad = X
        #    return Y
    
    def backward(self, output_gradient, learning_rate):
        if output_gradient.shape[0] > 1 and output_gradient.shape[1] >= 1:
            Grads = np.zeros((output_gradient.shape[0], self.w.shape[1], 1))
            dW = np.zeros(self.w.shape)
            dB = np.zeros(self.b.shape)
            for i in range(output_gradient.shape[0]):
                output_gradient_i = output_gradient[i]
                output_gradient_i = output_gradient_i.reshape((output_gradient_i.shape[0], 1))
                input_i = self.input[i]
                input_i = input_i.reshape((input_i.shape[0], 1))
                dw = -learning_rate * np.dot(output_gradient_i, input_i.T)

                grad_i = output_gradient[i]
                grad_i = grad_i.reshape((self.w.shape[0], 1))
                db = -learning_rate * grad_i

                #self.w += dw
                #self.b += db
                dW += dw / output_gradient.shape[0]
                dB += db / output_gradient.shape[0]

                #grad = np.dot(np.ones(self.w.shape).T, gradient[i])
                self_grad_i = self.input[i].reshape((self.input[i].shape[0], 1))
                #grad = np.dot(self.w.T, grad_i.T)
                #grad = grad.sum(axis=1)

                #grad = np.dot(self.w.T, output_gradient[i])
                grad = np.dot(self.w.T, output_gradient_i)

                Grads[i] = grad
                
                #grad = np.dot(dw.T, gradient)
            self.w += dW
            self.b += dB
            return learning_rate * Grads
        else:
            dw = -np.outer(output_gradient.T, self.input)
            db = -output_gradient

            self.w += dw
            self.b += db

            grad = np.dot(np.ones(self.w.shape).T, output_gradient)
            
            #grad = np.dot(dw.T, gradient)
            return grad
        
class Dropout(net.Module):
    def __init__(self, prop = 0.3):
        self.prop = prop
        super().__init__()

    def __call__(self, X):
        if not self.__eval__:
            self.zero_i = np.random.randint(0, X.shape[1], size=int(self.prop*X.shape[1]))

            for i in range(X.shape[0]):
                X[i][self.zero_i] = 0.

        return X
    
    def backward(self, gradient, learning_rate):
        if not self.__eval__:
            for i in range(gradient.shape[0]):
                gradient[i][self.zero_i] = 0.

        return gradient