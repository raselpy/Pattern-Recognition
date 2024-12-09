import numpy as np
from cvxopt import matrix, solvers


class SVM(object):
    def __init__(self, C=1.0, kernel="linear", degree=3, gamma=None, coef0=1):
        """
        Initialize the SVM model.

        Parameters:
        - C: Regularization parameter (default: 1.0)
        - kernel: Kernel function ('linear', 'poly', 'rbf') (default: 'linear')
        - degree: Degree for polynomial kernel (default: 3)
        - gamma: Kernel coefficient for RBF/poly kernels (default: 1/n_features)
        - coef0: Independent term for polynomial kernel (default: 1)
        """
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.w = None  # Weight vector (for linear kernel only)
        self.b = None  # Bias term
        self.alphas = None  # Lagrange multipliers
        self.support_vectors = None  # Indices of support vectors
        self.X_support = None  # Support vectors
        self.y_support = None  # Labels of support vectors

    def _optimize(self):
        pass

    def fit(self, X, y):
        """
            Train the SVM model using kernelized Lagrange duality optimization.

            Parameters:
            - X: Training data (n_samples, n_features)
            - y: Labels (n_samples, 1), must be +1 or -1
            """
        n_samples, n_features = X.shape
        if self.gamma is None:
            self.gamma = 1 / n_features

        # Step 1: Optimize to compute alphas
        alphas = self._optimize(X, y)
        self.alphas = alphas

    def predict(self):
        pass

X = np.array([[2, 3], [3, 3], [2, 1], [3, 1]])
y = np.array([1, 1, -1, -1])
svm = SVM(C=1.0, kernel="rbf", gamma=0.5)
svm.fit(X, y)