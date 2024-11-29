import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kNearestNeighbors import KNeighborsClassifier

df = pd.read_csv('../data/Social_Network_Ads.csv')

# Preprocessing
y = df.iloc[:, -1].values
X = df.iloc[:,:-1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Matrix
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))