from sklearn.decomposition import non_negative_factorization


class LogisticRegression(object):
    def __init__(self,learning_rate=0.01,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self):
        pass

    def compute_cost(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def plot_decision_boundary(self):
        pass
