import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_data.csv")

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

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

# ----------------
# Classification Metrics
# ----------------
f1 = f1_score(y_test, y_pred_clf, average='macro')
print("\n=== Classification Metrics ===")
print("Macro F1-score:", f1)

# AUC-ROC (one-vs-rest)
classes = y_clf.unique()
y_test_bin = label_binarize(y_test, classes=classes)
y_pred_bin = label_binarize(y_pred_clf, classes=classes)
roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
print("Macro AUC-ROC:", roc_auc)
