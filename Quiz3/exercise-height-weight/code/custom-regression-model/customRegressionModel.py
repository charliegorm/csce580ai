import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

# Load cleaned dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../../data/cleaned_data.csv")
df = pd.read_csv(data_path)

# ----------------
# Regression: Predict Weight from Height
# ----------------
X_reg = df[["Height"]]
y_reg = df["Weight"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# --- Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# --- Random Forest Regression ---
rf_reg = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=1, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# ----------------
# Evaluation
# ----------------
print("\n=== Regression Metrics ===")
print("Linear Regression R²:", r2_score(y_test, y_pred_lin))
print("Random Forest Regression R²:", r2_score(y_test, y_pred_rf))

# ----------------
# Out-of-sample Predictions
# ----------------
heights = [50, 100, 150, 200, 250]

print("\n=== Predictions for Fixed Heights (cm) ===")
for h in heights:
    sample = pd.DataFrame({"Height": [h]})   # DataFrame with correct feature name
    lin_pred = lin_reg.predict(sample)[0]
    rf_pred = rf_reg.predict(sample)[0]
    print(f"Height {h} cm → LinearReg: {lin_pred:.2f} kg | RandomForest: {rf_pred:.2f} kg")