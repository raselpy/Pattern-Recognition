import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from naive_bayes import NaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('Social_Network_Ads.csv')

# Preprocessing
y = df['Purchased']
X = df.drop(['Purchased'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"]))