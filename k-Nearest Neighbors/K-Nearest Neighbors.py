class KNN(object):
    def __init__(self, n_neighbor=3):
        self.n_neighbor = n_neighbor

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _predict_one(self,test):
        pass

    def predict(self, X):
        return [(self._predict_one(i))for i in X]