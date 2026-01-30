import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ----------------------------
# LOAD PREPROCESSED DATA
# ----------------------------
from preprocess import preprocess_pipeline

X, y, df = preprocess_pipeline("data/generated_certificates.csv")

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT for imbalance
)

# ----------------------------
# MODEL (FOCUS ON FAKE = 0)
# ----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)



# ----------------------------
# TRAIN
# ----------------------------
model.fit(X_train, y_train)

# ----------------------------
# PREDICT WITH PROBABILITY
# ----------------------------
y_prob_fake = model.predict_proba(X_test)[:, 0]   # probability of FAKE
y_pred = (y_prob_fake > 0.35).astype(int)

# ----------------------------
# EVALUATION
# ----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, "random_forest_model.pkl")
print("\nModel saved as random_forest_model.pkl ✅")
