import numpy as np
from src.linear_regression import add_bias_column, get_bias_and_weight, get_prediction_linear_regression
from src.metrics import mean_squared_error

def test_add_bias_column():
    without_bias = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 1, 2], [1, 3, 4]])

    assert np.array_equal(add_bias_column(without_bias), expected)

def test_get_bias_and_weight():
    public_X, public_y = np.array([[1, 3], [2, 3], [3, 4]]), np.arange(4, 7).reshape((-1, 1))

    test_1 = (round(get_bias_and_weight(public_X, public_y)[0], 5) == 3)
    test_2 = np.array_equal(np.round(get_bias_and_weight(public_X, public_y)[1], 1), np.array([[1.0], [0.0]]))
    test_3 = np.array_equal(np.round(get_bias_and_weight(public_X, public_y, False)[1], 2), np.round(np.array([[0.49], [1.20]]), 2))

    assert test_1 and test_2 and test_3

def test_get_prediction_linear_regression():
    test_X, test_y = np.array([[1, 3], [2, 3], [3, 4]]), np.arange(4, 7).reshape((-1, 1))

    assert round(mean_squared_error(test_y, get_prediction_linear_regression(test_X, test_y)), 5) == 0
