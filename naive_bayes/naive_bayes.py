import numpy as np


class NaiveBayes(object):

    def fit(self, X, y):
        # Get the number of samples and features
        n_samples, n_features = X.shape
        # Identify unique classes in the target variable
        self._classes = np.unique(y)
        # Determine the number of unique classes
        n_classes = len(self._classes)

        # Initialize arrays to store mean, variance, and prior probabilities for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate mean, variance, and prior probability for each class
        for idx, c in enumerate(self._classes):
            # Select all samples belonging to the current class
            X_c = X[y == c]
            # Compute mean of features for the current class
            self._mean[idx, :] = X_c.mean(axis=0)
            # Compute variance of features for the current class
            self._var[idx, :] = X_c.var(axis=0)
            # Compute prior probability for the current class
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        # Predict the class for each sample in the dataset
        y_pred = [self._predict(x) for x in X]
        # Return predictions as a numpy array
        return np.array(y_pred)

    def _predict(self,x):
        # List to store posterior probabilities for each class
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            # Compute log of the prior probability
            prior = np.log(self._priors[idx])
            # Compute log of the likelihood for the given sample
            posterior = np.sum(np.log(self._pdf(idx, x)))
            # Add prior to likelihood to get the posterior
            posterior = posterior + prior
            # Append the posterior probability for the current class
            posteriors.append(posterior)


        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self,idx,x):
        # Get mean and variance for the given class
        mean = self._mean[idx]
        var = self._var[idx]
        # Compute the numerator of the Gaussian PDF
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        # Compute the denominator of the Gaussian PDF
        denominator = np.sqrt(2 * np.pi * var)
        # Return the probability density for the given sample
        return numerator / denominator