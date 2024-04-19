import numpy as np
import net

class MeanSquareError(net.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        super().__init__()

    def __call__(self, Y, T):
        return self.backward(Y, T)
    
    def value(self):
        return self.__value__

    def backward(self, Y, T):
        if self.reduction == 'mean':
            self.__value__ = ((Y-T)**2).mean()
        elif self.reduction == 'sum':
            self.__value__ = ((Y-T)**2).sum()
        else:
            raise Exception(f'Reduction {self.reduction} is not supported.')
        if Y.shape[0] > 1 and Y.shape[1] >= 1:
            self.__value__ /= Y.shape[0]
        loss = (Y - T)
        return loss