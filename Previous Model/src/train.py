import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("../dataset/indoor_data.csv")

# -----------------------------
# Rename Columns
# -----------------------------
data.rename(columns={
    "field1": "CO",
    "field2": "MQ135",
    "field3": "Temperature",
    "field4": "Humidity",
    "field5": "eCO2",
    "field6": "TVOC",
    "field7": "Dust"
}, inplace=True)

# Keep only required columns
data = data[
    [
        "CO",
        "MQ135",
        "Temperature",
        "Humidity",
        "eCO2",
        "TVOC",
        "Dust"
    ]
]

# Remove missing values
data.dropna(inplace=True)

# -----------------------------
# Create Ventilation Label
# -----------------------------
def get_ventilation(row):

    score = 0

    # eCO2
    if row["eCO2"] > 1200:
        score += 2
    elif row["eCO2"] > 800:
        score += 1

    # Air Quality (MQ135)
    if row["MQ135"] > 150:
        score += 2
    elif row["MQ135"] > 100:
        score += 1

    # Dust
    if row["Dust"] > 80:
        score += 2
    elif row["Dust"] > 40:
        score += 1

    # Temperature
    if row["Temperature"] > 32:
        score += 1

    # Humidity
    if row["Humidity"] > 75:
        score += 1

    # Final Label
    if score <= 2:
        return "Normal"

    elif score <= 5:
        return "Moderate"

    else:
        return "High"


data["Ventilation"] = data.apply(get_ventilation, axis=1)

# -----------------------------
# Features
# -----------------------------
X = data.drop("Ventilation", axis=1)

# Target
y = data["Ventilation"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

print("="*50)
print("MODEL ACCURACY")
print("="*50)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "../models/ventilation_model.pkl")

print("\nModel saved successfully!")