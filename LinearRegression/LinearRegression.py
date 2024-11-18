import numpy as np


class LinearRegression(object):

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def compute_cost(self, X, y):
        """
        Compute the Mean Squared Error (MSE) cost function.
        """
        n_samples = len(y)
        y_pred = np.dot(X, self.weights) + self.bias
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        return cost

    def compute_gradient(self, X, y):
        """
        Compute the gradient of the cost function w.r.t. weights and bias.
        """
        n_samples = len(y)
        y_pred = np.dot(X, self.weights) + self.bias
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        return dw, db

    def gradient_descent(self, X, y):
        """
        Perform Gradient Descent to update weights and bias.
        """
        for i in range(self.n_iterations):
            dw, db = self.compute_gradient(X, y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Monitor cost for debugging
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}, Cost: {cost:.4f}, And Current Weights: {self.weights}, And Bias: {self.bias}")

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        """
        n_features = X.shape[1]
        self.initialize_parameters(n_features)
        self.gradient_descent(X, y)

    def predict(self, X):
        """
        Predict target values for the input data.
        """
        return np.dot(X, self.weights) + self.bias