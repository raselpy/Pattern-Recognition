class KNeighborsClassifier(object):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self):
        pass

    def _predict_one(self):
        pass

    def predict(self):
        pass


import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1,1,1,0,0,0])
neighbor = KNeighborsClassifier(n_neighbors=5)
neighbor.fit(X, y)
print(neighbor.predict(np.array([[1, 0], [-2, -2]])))