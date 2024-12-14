import numpy as np
from src.metrics import mean_squared_error, mean_absolute_error

def test_mean_squared_error():
    y_true = np.array([[3], [5]])
    y_pred = np.array([[12], [15]])
    assert mean_squared_error(y_true, y_pred) in [45.25, 90.5]

def test_mean_absolute_error():
    y_true = np.array([[3], [5]])
    y_pred = np.array([[12], [15]])
    assert mean_absolute_error(y_true, y_pred) == 9.5
