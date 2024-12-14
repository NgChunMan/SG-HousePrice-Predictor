import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import load_data
from src.models.normal_equation import normal_equation
from src.models.gradient_descent import gradient_descent
from src.metrics.metrics import mean_squared_error, mean_absolute_error

# Load data
data_filepath = "data/housing_data.csv"
X, y = load_data(data_filepath)

# Add bias column to X
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# Train model using Normal Equation
weights_ne = normal_equation(X_with_bias, y)
y_pred_ne = X_with_bias @ weights_ne

# Train model using Gradient Descent
lr = 0.01
epochs = 1000
weights_gd, losses_gd = gradient_descent(X_with_bias, y, lr=lr, epochs=epochs)
y_pred_gd = X_with_bias @ weights_gd

# Evaluate models
mse_ne = mean_squared_error(y, y_pred_ne)
mae_ne = mean_absolute_error(y, y_pred_ne)
mse_gd = mean_squared_error(y, y_pred_gd)
mae_gd = mean_absolute_error(y, y_pred_gd)

# Display results
print("Normal Equation:")
print(f"  MSE: {mse_ne:.4f}, MAE: {mae_ne:.4f}")
print("Gradient Descent:")
print(f"  MSE: {mse_gd:.4f}, MAE: {mae_gd:.4f}")

# Plot losses for Gradient Descent
plt.plot(range(len(losses_gd)), losses_gd, label="Gradient Descent Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent Loss Over Epochs")
plt.legend()
plt.show()

# Plot predictions
plt.scatter(X[:, 0], y, label="True Data")
plt.plot(X[:, 0], y_pred_ne, color="r", label="Normal Equation")
plt.plot(X[:, 0], y_pred_gd, color="g", label="Gradient Descent")
plt.xlabel("Size (Square Meters)")
plt.ylabel("Price (SGD)")
plt.title("Housing Prices Prediction")
plt.legend()
plt.show()
