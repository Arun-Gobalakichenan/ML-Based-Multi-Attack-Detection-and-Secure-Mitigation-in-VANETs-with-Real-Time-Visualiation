from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "packet_rate",
    "avg_speed",
    "speed_std",
    "neighbor_count",
    "packet_drop_rate",
    "position_deviation",
    "message_consistency",
    "signal_strength",
]

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

X_train = train[FEATURES]
y_train = train["label"]
X_test = test[FEATURES]
y_test = test["label"]

# Imputation is included inside the pipeline to keep preprocessing
# fitted only on training data.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test, predictions, digits=4, zero_division=0
)

print("\n=== Random Forest Baseline ===")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {len(FEATURES)}")
print(f"Test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)

with open(RESULTS / "random_forest_report.txt", "w") as f:
    f.write("Random Forest Baseline — Synthetic Dataset\n")
    f.write("This evaluation uses synthetic data, not simulation observations.\n\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n\n")
    f.write(report)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=35,
    cmap="Blues",
    values_format="d",
)
plt.title("Random Forest — Synthetic Test Dataset")
plt.tight_layout()
plt.savefig(RESULTS / "random_forest_confusion_matrix.png", dpi=160)
plt.close()

classifier = model.named_steps["classifier"]
importance = pd.Series(
    classifier.feature_importances_,
    index=FEATURES,
).sort_values(ascending=True)

importance.to_csv(RESULTS / "feature_importance.csv", header=["importance"])

joblib.dump(
    {"model": model, "features": FEATURES},
    ROOT / "ml" / "random_forest_model.joblib",
)

print("\nFiles saved:")
print("  results/random_forest_report.txt")
print("  results/random_forest_confusion_matrix.png")
print("  results/feature_importance.csv")
print("  ml/random_forest_model.joblib")
print("\nReminder: synthetic-data performance is not real-world validation.")
