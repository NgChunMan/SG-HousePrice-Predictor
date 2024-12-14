import numpy as np
from src.metrics import mean_squared_error

def gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250):
    '''
    Approximate bias and weight that gave the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (n, 1) numpy matrix that specifies the weight constants.
        loss (list):
            A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    loss = []
    
    m = X.shape[0]
    pred = X @ weights + bias
    for _ in range(number_of_epochs):
        difference_pred_true_value = pred - y
        transposed_feature_matrix = np.transpose(X)
        partial_derivative_of_weights = np.matmul(transposed_feature_matrix, difference_pred_true_value) / m
        partial_derivative_of_weights = lr * partial_derivative_of_weights
        weights = weights - partial_derivative_of_weights

        partial_derivative_of_bias = np.sum(difference_pred_true_value) / m
        partial_derivative_of_bias = lr * partial_derivative_of_bias
        bias = bias - partial_derivative_of_bias

        pred = X @ weights + bias

        mean_square_score = mean_squared_error(y, pred)
        loss.append(mean_square_score)
    
    return bias, weights, loss
