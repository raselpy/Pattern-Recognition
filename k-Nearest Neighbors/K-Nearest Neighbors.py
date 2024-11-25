class KNN(object):
    def __init__(self, n_neighbor=3):
        self.n_neighbor = n_neighbor

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        d = sum(abs(data1 - data2))
        return d

    def _predict_one(self,test):
        distances = sorted([(self._distance(x, test), y) for x, y in zip(self.X, self.y)])
        return distances

    def predict(self, X):
        return [self._predict_one(i)for i in X]