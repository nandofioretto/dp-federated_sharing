'''
function [w, b] = Functional_Linear(Train, ep)

Differentially private linear regression using Funcational Mechanism.

Input parameters:
Training data (Train) with last column attribute to be predicted.
Train = [x1, x2, ..., xd, y]

NOTICE: The values of EACH attribute (column) should be converted from [min,
max] to [-1,1] in order to match the privacy design of Functional
Mechanism. Please make sure that ALL values in Train are located in
[-1,1]. If Train is rescaled to meet this requirement, Test should be
converted IN THE SAME WAY to get the correct answer.

Privacy budget (ep) is a parameter in differential privacy, which
indicates the strength of privacy protection.

Model is

  (w, b) = argmin sum((x*w+b-y)^2)

Outputs are regression coefficients w and b.

Copyright 2012 Jun Zhang
'''

import numpy as np
from numpy import linalg as LA
import scipy.optimize as op
from agt_linreg import LinearRegressionAgt

class DPfunLinearRegressionAgt(LinearRegressionAgt):
    '''Private Linear Regression with Function perturbation'''
    def __init__(self, seed=1234):
        super(DPfunLinearRegressionAgt, self).__init__(normalize=True, seed=seed)

    def _funmin(self, w, c1, c2):
        wT = np.transpose(w)
        c1T = np.transpose(c1)
        return np.matmul(np.matmul(wT, c2), w) + np.matmul(c1T, w)

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

        #_X, _y = X, y
        _X, _y = self.normalize(X, y)

        n, d = _X.shape
        d += 1  # include dimension of y

        X = np.ones(shape=(n, d))
        X[:, :-1] = _X
        y = _y

        R0 = np.matmul(np.transpose(X), X)
        R1 = -2 * np.matmul(np.transpose(X), y)
        deltaQ = 2 * d * d + 4 * d

        noise = self.rnd.laplace(0, deltaQ/epsilon, size=(d,d))
        coef2 = R0 + noise
        coef2 = 0.5 * (np.transpose(coef2) + coef2)

        # Regularization (you may want to remove this)
        coef2 = coef2 + 5 * np.sqrt(2) * (deltaQ / epsilon) * np.identity(d)

        noise = self.rnd.laplace(0, deltaQ/epsilon, d)
        coef1 = R1 + noise
        val, vec = LA.eig(coef2)
        val = np.diag(val)

        # _del = np.where(np.diag(val) < 1e-8)
        # val = np.delete(val, _del, 0)
        # val = np.delete(val, _del, 1)
        # vec = np.delete(vec, _del, 1)
        coef2 = val
        coef1 = np.matmul(np.transpose(vec), coef1)

        g0 = np.random.rand(d)# - (len(_del)-1))

        # (w, b) = argmin sum((x * w + b - y)^2)
        Result = op.minimize(fun=self._funmin, x0=g0, args=(coef1, coef2))
        best_w = np.matmul(vec, Result.x)
        self.w, self.b = best_w[:-1], best_w[-1]

        # print(self.w, self.b)
        return self.w, self.b