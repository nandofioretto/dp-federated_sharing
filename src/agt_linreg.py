import numpy as np

from regression import Regression
from sklearn import linear_model


class LinearRegressionAgt(Regression):
    '''
       (w, b) = argmin sum((x*w+b-y)^2)
    '''
    def __init__(self, normalize=True, seed=1234):
        super(LinearRegressionAgt, self).__init__(normalize, seed)

    def _classification(self, Xtest, w, b):
        return np.matmul(Xtest, w.flatten()) + b

    def fit(self, X, y, hyperparams=None):
        _X, _y = self.normalize(X, y)
        reg = linear_model.LinearRegression()
        reg.fit(_X, _y)
        self.w, self.b = reg.coef_, reg.intercept_
        return self.w, self.b