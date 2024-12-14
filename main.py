import numpy as np
import matplotlib.pyplot as plt
from src.data_utils import load_data
from src.models.normal_equation import get_prediction_linear_regression
from src.models.gradient_descent import gradient_descent_multi_variable
from src.metrics.metrics import mean_squared_error, mean_absolute_error

# Load data
data_filepath = "data/housing_data.csv"
X, y = load_data(data_filepath)

# Train model using Normal Equation
y_pred_ne = get_prediction_linear_regression(X, y)

# Train model using Gradient Descent
lr = 1e-5
epochs = 250
bias_gd, weights_gd, loss_gd = gradient_descent_multi_variable(X, y, lr=lr, epochs=epochs)
y_pred_gd = X @ weights_gd + bias_gd

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
plt.plot(range(len(loss_gd)), loss_gd, label="Gradient Descent Loss")
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
