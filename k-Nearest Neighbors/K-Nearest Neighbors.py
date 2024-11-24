class KNN(object):
    def __init__(self, n_neighbor=3):
        self.n_neighbor = n_neighbor

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self):
        pass