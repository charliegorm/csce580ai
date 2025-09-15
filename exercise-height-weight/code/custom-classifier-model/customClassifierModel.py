import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../../data/cleaned_data.csv")
df = pd.read_csv(data_path)

# ----------------
# Classification: Predict BMI Category
# ----------------
df["BMI"] = df["Weight"] / (df["Height"]/100)**2

def bmi_category(bmi):
    if bmi < 18.5:
        return "C1"  # Underweight
    elif bmi < 25:
        return "C2"  # Healthy
    elif bmi < 30:
        return "C3"  # Overweight
    else:
        return "C4"  # Obese

df["BMI_Category"] = df["BMI"].apply(bmi_category)

# Remove categories with <2 samples
category_counts = df["BMI_Category"].value_counts()
valid_categories = category_counts[category_counts >= 2].index
df_class = df[df["BMI_Category"].isin(valid_categories)]

X_clf = df_class[["Height", "Weight"]]
y_clf = df_class["BMI_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ----------------
# Logistic Regression
# ----------------
clf_lr = LogisticRegression(max_iter=1000, multi_class="ovr")
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)
y_score_lr = clf_lr.predict_proba(X_test)

# ----------------
# Random Forest Classifier
# ----------------
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
y_score_rf = clf_rf.predict_proba(X_test)

# ----------------
# Metrics
# ----------------
print("\n=== Classification Metrics ===")
print("LogReg Macro F1:", f1_score(y_test, y_pred_lr, average="macro"))
print("RF Macro F1:", f1_score(y_test, y_pred_rf, average="macro"))

# ----------------
# ROC + AUC
# ----------------
classes = np.unique(y_clf)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(8,6))
for model_name, y_score in [("LogReg", y_score_lr), ("RandomForest", y_score_rf)]:
    # One-vs-Rest ROC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right")
plt.show()

# ----------------
# Predictions at Fixed Heights
# ----------------
heights = np.array([50, 100, 150, 200, 250])
print("\n=== Predictions for Fixed Heights (cm) ===")
for h in heights:
    # To classify BMI, we need both height and some assumed weight
    # → use regression RF predictions as "estimated weight"
    est_weight = clf_rf.estimators_[0].tree_.value.mean()  # fallback
    try:
        est_weight = 0.0025 * h**2  # approximate BMI=25 baseline weight
    except:
        pass
    sample = pd.DataFrame([[h, est_weight]], columns=["Height", "Weight"])
    pred_lr = clf_lr.predict(sample)[0]
    pred_rf = clf_rf.predict(sample)[0]
    print(f"Height {h} cm → LogReg: {pred_lr} | RandomForest: {pred_rf}")

