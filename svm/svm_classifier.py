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

    def _compute_bias(self, X, y, alphas, K):
        """
        Compute the bias term using the support vectors.

        Parameters:
        - X: Training data (n_samples, n_features)
        - y: Labels (n_samples, 1), must be +1 or -1
        - alphas: Lagrange multipliers (n_samples,)
        - K: Kernel matrix (n_samples, n_samples)

        Returns:
        - b: Bias term (float)
        """
        support_vector_indices = (alphas > 1e-5) & (alphas < self.C)
        b_values = y[support_vector_indices] - np.sum(
            alphas * y * K[support_vector_indices], axis=1
        )
        return np.mean(b_values)

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

        # Step 2: Identify support vectors
        self.support_vectors = np.where(alphas > 1e-5)[0]
        self.X_support = X[self.support_vectors]
        self.y_support = y[self.support_vectors]

        # Step 3: Compute bias
        K = self._compute_kernel(X)
        self.b = self._compute_bias(X, y, alphas, K)

        # Compute weights only if using a linear kernel
        if self.kernel == "linear":
            self.w = np.sum((alphas[:, None] * y[:, None]) * X, axis=0)

    def predict(self, X):
        """
        Make predictions using the trained SVM model.

        Parameters:
        - X: Test data (n_samples, n_features)

        Returns:
        - Predictions (+1 or -1) for each sample
        """
        if self.kernel == "linear":
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            K = self._compute_kernel(X, self.X_support)
            decision = np.sum(self.alphas[self.support_vectors] * self.y_support * K, axis=1) + self.b
            return np.sign(decision)