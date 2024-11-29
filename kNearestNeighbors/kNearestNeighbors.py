from collections import defaultdict
class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def _distance(self, data1, data2):
        # Compute the Manhattan distance (L1 norm)
        return sum(abs(data1 - data2))

    def _compute_weights(self, distances):
        # Compute weights for the k nearest neighbors.
        # Currently assigns a fixed weight of 1 for each neighbor.
        # Returns a list of tuples (weight, label).
        return [(1, y) for d, y in distances]

    def _predict_one(self, test):
        # 1. Calculate distances from the test point to each training point, paired with labels.
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))

        # 2. Select the k nearest neighbors (defined by self.n_neighbors).
        # 3. Compute weights for the k nearest neighbors.
        weights = self._compute_weights(distances[:self.n_neighbors])
        # Group weights by class labels.
        weights_by_class = defaultdict(list)
        for d, c in weights:
            weights_by_class[c].append(d)

        # Find the class with the highest total weight and return it.
        return max((sum(val), key) for key, val in weights_by_class.items())[1]

    def predict(self, X_test):
        return [self._predict_one(i) for i in X]


import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,0,0,0])
neighbor = KNeighborsClassifier(n_neighbors=5)
neighbor.fit(X, y)
# print(neighbor.predict(np.array([[1, 0], [-2, -2]])))