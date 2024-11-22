import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logisticRegression import LogisticRegression
# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Preprocessing
y = df['Purchased']
X = df.drop(['Purchased'], axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(y_pred)