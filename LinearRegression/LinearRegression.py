import numpy as np


class LinearRegression(object):

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def compute_cost(self, X, y):
        """
        Compute the Mean Squared Error (MSE) cost function.
        """
        n_samples = len(y)
        y_pred = np.dot(X, self.weights) + self.bias
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        return cost

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass