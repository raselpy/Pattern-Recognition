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

    def fit(self):
        pass

    def predict(self):
        pass
