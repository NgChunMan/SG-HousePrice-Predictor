from src.data_utils import load_data
from src.linear_regression import get_prediction_linear_regression
from src.visualization import plot_predictions

data_filepath = "data/housing_data.csv"
X, y = load_data(data_filepath)

# Predict
predictions = get_prediction_linear_regression(X[:, 0].reshape((-1, 1)), y)

# Visualize
plot_predictions(X[:, 0].reshape((-1, 1)), y, predictions)
