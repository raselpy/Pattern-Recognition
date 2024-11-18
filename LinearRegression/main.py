import numpy as np
from matplotlib import pyplot as plt

from LinearRegression import LinearRegression

np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Features
y = 4 + 3 * X.flatten() + np.random.randn(100)  # Target with noise

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Data Visualization
plt.scatter(X_test, y_test, marker='X', color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Linear Regression')
plt.xlabel('Features')
plt.ylabel('Targets')
plt.show()


# Evaluate the model
from sklearn.metrics import mean_squared_error
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
