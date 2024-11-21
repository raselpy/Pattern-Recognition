import numpy as np


class LogisticRegression(object):
    def __init__(self,learning_rate=0.01,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self,shape):
        self.weights = np.zeros(shape)
        self.bias = 0

    def compute_cost(self,X,y):
        """
           To minimize the error, we aim to maximize the log-likelihood. Since optimization
           algorithms typically minimize rather than maximize, we use the negative
           log-likelihood as the cost function
        """
        m = len(y)
        h = self.sigmoid(np.dot(X,self.weights)+self.bias)
        cost= (-1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))
        return cost

    def compute_gradient(self, X, y):
        """
        Compute the gradients of the cost function with respect to weights (dw) and bias (db).
        """
        m = len(y)
        h = self.sigmoid(np.dot(X, self.weights) + self.bias)
        error = h - y
        dw = (1 / m) * np.dot(X.T, error)  # Gradient with respect to weights
        db = (1 / m) * np.sum(error)  # Gradient with respect to bias
        return dw, db

    def gradient_descent(self, X, y):
        """
        Trains the logistic regression model using gradient descent.
        """
        m, n = X.shape

        for _ in range(self.n_iter):
            # Linear combination
            z = np.dot(X, self.weights) + self.bias
            # Prediction using sigmoid
            y_pred = self.sigmoid(z)

            # Gradients
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def fit(self):
        pass

    def predict(self):
        pass

