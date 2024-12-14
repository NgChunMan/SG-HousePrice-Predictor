import numpy as np

def mean_squared_error(y_true, y_pred):
    '''
    Calculate mean squared error between y_pred and y_true.
    '''
    num_of_samples = y_true.shape[0]

    difference = y_pred - y_true
    square_value = np.square(difference)
    sum_of_sq_values = np.sum(square_value)
    mean_sq_error = sum_of_sq_values / (2 * num_of_samples)

    return mean_sq_error

def mean_absolute_error(y_true, y_pred):
    '''
    Calculate mean absolute error between y_pred and y_true.
    '''
    num_of_samples = y_true.shape[0]
    difference = y_pred - y_true
    absolute_value = np.abs(difference)

    sum_of_absolute_values = np.sum(absolute_value)
    mean_absolute_error = sum_of_absolute_values / num_of_samples
    
    return mean_absolute_error
