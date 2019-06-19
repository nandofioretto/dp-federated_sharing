import numpy as np
from utils import normalize
from sklearn import metrics
from sklearn.preprocessing import scale

class Regression:
    """
    Basic Regression Class
    """
    def __init__(self, normalize=False, seed=1234):
        self.seed = seed
        self.rnd = np.random.RandomState(seed)
        self.to_scale = normalize
        self.w = None
        self.b = None

    @property
    def get_w(self):
        return self.w

    @property
    def get_b(self):
        return self.b

    def normalize(self, X, y, copy=True):
        if self.to_scale:
            if copy:
                return X, normalize(y.copy(), -1, 1)
            else:
                return X, normalize(y, -1, 1)
        else:
            return X, y

    def eval(self, Xtest, ytest, metric='mse'):
        if self.w is None:
            return None
        X, y = self.normalize(Xtest, ytest)

        ypred = self._classification(X, self.w, self.b)
        if metric == 'mse':
            # Mean squared error
            return metrics.mean_squared_error(y, ypred)
        if metric == 'amse':
            # Average Mean squared error
            return metrics.mean_squared_error(y, ypred) / len(y)
        if metric == 'r2':
            # Explained variance score: 1 is perfect prediction
            return metrics.r2_score(y, ypred)
        if metric == 'acc':
            return metrics.accuracy_score(y, ypred)
        else:
            return None

    def _classification(self, Xtest, w, b):
        pass

    def predict(self, Xtest):
        return self._classification(Xtest, self.w, self.b)