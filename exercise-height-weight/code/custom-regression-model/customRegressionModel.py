import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_data.csv")

# ----------------
# Regression: Predict Weight from Height
# ----------------
X_reg = df[["Height"]]
y_reg = df["Weight"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

print("\n=== Regression Metrics ===")
print("R²:", r2_score(y_test, y_pred_reg))
