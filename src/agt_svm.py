import numpy as np
from regression import Regression
from sklearn import svm


class SVMAgt(Regression):
    '''
       (w, b) = argmin sum(log(1+exp(x*w+b))-y(x*w+b))
    '''
    def __init__(self, normalize=False, seed=1234, classes=[0,1]):
        super(SVMAgt, self).__init__(normalize, seed)
        self.classes = classes

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y >= 0.5] = self.classes[1]
        y[y < 0.5] = self.classes[0]
        return y

    def fit(self, X, y, hyperparams=None):
        clf = svm.SVC(kernel='linear')
        #clf = svm.SVC(gamma='scale')
        clf.fit(X, y)
        self.w, self.b = clf.coef_, clf.intercept_
        return self.w, self.b


import dp_stats as dps

class DPSVMAgt(Regression):
    '''
       (w, b) = argmin sum(log(1+exp(x*w+b))-y(x*w+b))
    '''
    def __init__(self, normalize=False, seed=1234, classes=[0,1]):
        super(DPSVMAgt, self).__init__(normalize, seed)
        self.classes=classes

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y >= 0.5] = self.classes[1]
        y[y < 0.5] = self.classes[0]
        return y

    def fit(self, X, y, hyperparams=None):
        self.w = dps.dp_svm(X.values, y.values, 'obj', epsilon=hyperparams['epsilon'], Lambda=0.01, h=0.5)
        self.b = 0
        return self.w, self.b