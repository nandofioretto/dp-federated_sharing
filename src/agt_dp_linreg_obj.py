import numpy as np
from sklearn import linear_model
from agt_linreg import LinearRegressionAgt

class DPobjLinearRegressionAgt(LinearRegressionAgt):
    '''Linear Regression with objective perturbaation'''
    def __init__(self, seed=1234):
        super(DPobjLinearRegressionAgt, self).__init__(normalize=True, seed=seed)

    def _find_M(self, X):
        return np.max([np.linalg.norm(x, 2) for x in X.values])

    def _find_N(self, Y):
        return np.max([abs(y) for y in Y.values])

    def _find_P(self, w):
        return np.linalg.norm(w, 2)

    def fit(self, X, y, hyperparams):
        '''
        Given X, y and epsilon, returns w, b s.t.:
            (w, b) = argmin sum((x * w + b - y)^2)
        :param _X: The training data features
        :param _y: The training data output
        :param epsilon: the privacy budget
        :return: (w, b)
        '''
        epsilon = hyperparams['epsilon']

        _X, _y = self.normalize(X, y)

        reg = linear_model.LinearRegression()
        reg.fit(_X, _y)
        n, k = X.shape
        M = self._find_M(_X)
        N = self._find_N(_y)
        P = self._find_P(reg.coef_)
        Deltaf = 2 * M * np.sqrt(k) * (N + P * M) / n

        self.w = reg.coef_ + np.random.laplace(0, Deltaf / epsilon)
        self.b = reg.intercept_

        return self.w, self.b