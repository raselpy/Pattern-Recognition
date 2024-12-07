class SVM(object):
    def __init__(self, C=1.0):
        """
        Initialize the SVM model.

        Parameters:
        - C: Regularization parameter (default: 1.0)
        """
        self.C = C  # Regularization parameter
        self.w = None  # Weight vector
        self.b = None  # Bias term
        self.alphas = None  # Lagrange multipliers
        self.support_vectors = None  # Indices of support vectors

    def fit(self):
        pass

    def predict(self):
        pass
