'''
    Model Aggregators:
    Take several models (from different agents) and aggregates them into a unified model.
'''
import numpy as np
from agt_linreg import LinearRegressionAgt
from agt_logreg import LogisticRegressionAgt

class ModelAggregator:
    def __init__(self, agents: list):
        self.agents = agents

class RegressionModelAggregator(ModelAggregator, LinearRegressionAgt):
    '''
    Aggregator for Linear Regression Agents
    '''

    def __init__(self, agents, seed=1234):
        ModelAggregator.__init__(self, agents)
        LinearRegressionAgt.__init__(self, seed)

    def aggregate(self):
        assert len(self.agents) > 0

        n = len(self.agents)

        self.w = np.sum(agt.w for agt in self.agents) / n
        self.b = np.sum(agt.b for agt in self.agents) / n

    def fit(self, X, y, epsilon=None):
        pass

class LogRegAggregator(ModelAggregator, LogisticRegressionAgt):
    '''
    Aggregator for Logistic Regression Agents
    '''

    def __init__(self, agents, seed=1234, classes=[0, 1]):
        ModelAggregator.__init__(self, agents)
        LogisticRegressionAgt.__init__(self, seed, classes)

    def aggregate(self):
        assert len(self.agents) > 0

        n = len(self.agents)

        self.w = np.sum(agt.w for agt in self.agents) / n
        self.b = np.sum(agt.b for agt in self.agents) / n

    def fit(self, X, y, epsilon=None):
        pass

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y > 0.5] = self.classes[1]
        y[y <= 0.5] = self.classes[0]
        return y

    def predict(self, Xtest):
        y = np.matmul(Xtest, self.w.flatten()) #+ self.b
        y[y > 0.5] = self.classes[1]
        y[y <= 0.5] = self.classes[0]
        return y



class LinRegAggregator(ModelAggregator, LinearRegressionAgt):
    '''
    Aggregator for Logistic Regression Agents
    '''

    def __init__(self, agents, seed=1234):
        ModelAggregator.__init__(self, agents)
        LinearRegressionAgt.__init__(self, seed)

    def aggregate(self):
        assert len(self.agents) > 0

        n = len(self.agents)

        self.w = np.sum(agt.w for agt in self.agents) / n
        self.b = np.sum(agt.b for agt in self.agents) / n

    def fit(self, X, y, epsilon=None):
        pass

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y > 1] = 1.0
        y[y < -1] = -1.0
        return y

    def predict(self, Xtest):
        y = np.matmul(Xtest, self.w.flatten()) + self.b
        y[y > 1] = 1.0
        y[y < -1] = -1.0
        return y


from agt_svm import SVMAgt

class SVMAggregator(ModelAggregator, SVMAgt):
    '''
    Aggregator for Logistic Regression Agents
    '''

    def __init__(self, agents, seed=1234):
        ModelAggregator.__init__(self, agents)
        SVMAgt.__init__(self, seed=seed, classes=agents[0].classes)

    def aggregate(self):
        assert len(self.agents) > 0

        n = len(self.agents)

        self.w = np.sum(agt.w for agt in self.agents) / n
        self.b = np.sum(agt.b for agt in self.agents) / n

    def fit(self, X, y, epsilon=None):
        pass

    def _classification(self, Xtest, w, b):
        y = np.matmul(Xtest, w.flatten()) + b
        y[y >= 0] = self.classes[1]
        y[y < 0] = self.classes[0]
        return y

    def predict(self, Xtest):
        y = np.matmul(Xtest, self.w.flatten()) + self.b
        y[y >= 0] = self.classes[1]
        y[y < 0] = self.classes[0]
        return y