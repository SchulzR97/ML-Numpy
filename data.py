import numpy as np

class Dataset():
    def __init__(self, X, T):
        self.X = X
        self.T = T

    def sample(self, batch_size):
        indices = np.random.choice(range(self.X.shape[0]), batch_size)
        return self.X[indices], self.T[indices]
    
def split(X, T, train_prop = 0.7, shuffle = True):
    if shuffle:
        indices = np.random.choice(range(X.shape[0]), X.shape[0])
        X = X[indices]
        T = T[indices]

    train_cnt = int(train_prop * X.shape[0])
    X_train = X[:train_cnt]
    T_train = T[:train_cnt]
    X_val = X[train_cnt:]
    T_val = T[train_cnt:]

    return X_train, T_train, X_val, T_val