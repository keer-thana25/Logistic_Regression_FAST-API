
# TRAIN LOGISTIC REGRESSION MODEL


import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model

model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

# Prediction

y_pred = model.predict(X_test_scaled)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy : {accuracy:.2f}")

# Save Model and Scaler

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully")
