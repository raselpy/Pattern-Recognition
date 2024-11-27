import numpy as np
from collections import defaultdict

class KNN(object):
    def __init__(self, n_neighbor=3, weights='uniform', distance_type=2):
        self.n_neighbor = n_neighbor
        self.weights = weights
        self.distance_type = distance_type

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _distance(self, data1, data2):
        """1: Manhattan, 2: Euclidean"""
        if self.distance_type == 1:
            return sum(abs(data1 - data2))
        elif self.distance_type == 2:
            return np.sqrt(sum((data1 - data2)**2))
        raise ValueError("p not recognized: should be 1 or 2")

    def _compute_weights(self, distances):
        if self.weights == 'uniform':
            return [(1, y) for d, y in distances]
        elif self.weights == 'distance':
            matches = [(1, y) for d, y in distances if d == 0]
            return matches if matches else [(1/d, y) for d, y in distances]
        raise ValueError("weights not recognized: should be 'uniform' or 'distance'")

    def _predict_one(self,test):
        distances = sorted([(self._distance(x, test), y) for x, y in zip(self.X, self.y)])
        # Get the closest n_neighbors
        neighbors = distances[:self.n_neighbor]
        weights = self._compute_weights(distances[:self.n_neighbor])
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)
        dc = max((sum(val), key) for key, val in weights_by_class.items())
        return dc[1]

    def predict(self, X):
        return [self._predict_one(i)for i in X]