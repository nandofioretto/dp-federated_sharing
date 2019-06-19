import numpy as np

from regression import Regression
from sklearn.linear_model import LogisticRegression


class LogisticRegressionAgt(Regression):
    '''
       (w, b) = argmin sum(log(1+exp(x*w+b))-y(x*w+b))
    '''
    def __init__(self, normalize=False, seed=1234, classes=[0,1]):
        super(LogisticRegressionAgt, self).__init__(normalize, seed)
        self.classes = classes

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y >= 0] = self.classes[1]
        y[y < 0] = self.classes[0]
        return y

    def fit(self, X, y, hyperparams=None):
        reg = LogisticRegression(random_state=self.seed, solver='lbfgs', multi_class='ovr')
        reg.fit(X, y)
        self.w, self.b = reg.coef_, reg.intercept_
        return self.w, self.b