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

    def _compute_kernel(self , X, Z=None):
        """
        Compute the kernel matrix between X and Z (or X and itself if Z=None).

        Parameters:
        - X: Training data (n_samples, n_features)
        - Z: Data to compute kernel with (n_samples, n_features) (default: None)

        Returns:
        - Kernel matrix (n_samples_X, n_samples_Z)
        """
        if Z is None:
            Z = X

        if self.kernel == "linear":
            return np.dot(X, Z.T)
        elif self.kernel == "poly":
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            return (gamma * np.dot(X, Z.T) + self.coef0) ** self.degree
        elif self.kernel == "rbf":
            gamma = self.gamma if self.gamma else 1.0 / X.shape[1]
            X_norm = np.sum(X**2, axis=-1).reshape(-1, 1)
            Z_norm = np.sum(Z**2, axis=-1).reshape(1, -1)
            return np.exp(-gamma * (X_norm + Z_norm - 2 * np.dot(X, Z.T)))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _optimize(self, X, y):
        """
        Solve the quadratic programming problem to compute Lagrange multipliers.

        Parameters:
        - X: Training data (n_samples, n_features)
        - y: Labels (n_samples, 1), must be +1 or -1

        Returns:
        - alphas: Lagrange multipliers (n_samples,)
        """
        n_samples, _ = X.shape

        # Compute the kernel matrix+
        K = self._compute_kernel(X)

        # Construct the quadratic programming problem
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1), (1, n_samples), 'd')
        b = matrix(0.0)

        # Solve the QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution["x"])
        return alphas

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