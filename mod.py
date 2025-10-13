# model_train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# -----------------------------
# 1. Load dataset
# -----------------------------
# You can replace this with your CSV dataset
# Example: dataset = pd.read_csv("house_prices.csv")
# For demonstration, let's create a sample dataset
data = {
    "size": [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    "bedrooms": [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    "age": [10, 15, 20, 5, 30, 8, 12, 7, 25, 18],
    "price": [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}
df = pd.DataFrame(data)

# -----------------------------
# 2. Prepare X and y
# -----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# -----------------------------
# 3. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train the model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate the model
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model trained. RMSE on test set: {rmse:.2f}")

# -----------------------------
# 6. Save the model to .pkl
# -----------------------------
with open("house_price_model1.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as house_price_model1.pkl")
