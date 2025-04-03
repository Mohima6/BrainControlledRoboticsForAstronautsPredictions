import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Simulating Astronaut Mobility Dataset
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    "speed": np.random.uniform(0.5, 2.0, data_size),  # Random movement speeds
    "stability": np.random.uniform(0.3, 1.0, data_size),  # Stability measures
    "adaptability": np.random.uniform(0.2, 0.9, data_size),  # Adaptability factors
    "optimal_adjustment": np.random.uniform(0.1, 1.5, data_size)  # Ideal robotic adjustments
})

# Splitting Data for Training
X = df[["speed", "stability",      "adaptability"]]
y = df["optimal_adjustment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training XGBoost Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)

# Making Predictions
y_pred = xgb_model.predict(X_test)

# Evaluating Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#  Visualizing Feature Importance
plt.figure(figsize=(10, 6))  #  Increase figure size for better adaptability visibility
xgb.plot_importance(xgb_model, importance_type="weight")  #  Adjust feature importance calculation
plt.title("Feature Importance in Exoskeleton Optimization")
plt.tight_layout()  #  Prevents overlapping labels
plt.show()


print(" AI-Powered Exoskeleton Optimization Completed Successfully!")
