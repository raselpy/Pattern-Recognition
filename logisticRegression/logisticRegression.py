import numpy as np
from sklearn.decomposition import non_negative_factorization


class LogisticRegression(object):
    def __init__(self,learning_rate=0.01,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self,X,y,weights,bias):
        """
           To minimize the error, we aim to maximize the log-likelihood. Since optimization
           algorithms typically minimize rather than maximize, we use the negative
           log-likelihood as the cost function
        """
        m = len(y)
        h = self.sigmoid(np.dot(X,weights)+bias)
        cost= (-1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))
        return cost




    def fit(self):
        pass

    def predict(self):
        pass

    def plot_decision_boundary(self):
        pass
