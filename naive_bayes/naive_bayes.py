from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

nb = NaiveBayes()
nb.fit(X_train, y_train)