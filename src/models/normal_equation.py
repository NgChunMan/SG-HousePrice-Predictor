import numpy as np

def add_bias_column(X):
    '''
    Create a bias column and combine it with X.

    Parameters
    ----------
    X : (m, n) numpy matrix representing a feature matrix
    
    Returns
    -------
        new_X (np.ndarray):
            A (m, n + 1) numpy matrix with the first column consisting of all 1s
    '''

    num_of_rows = X.shape[0]

    bias_column = np.array([[1]] * num_of_rows)
    result = np.hstack((bias_column, X))
    return result

def get_bias_and_weight(X, y, include_bias = True):
    '''
    Calculate bias and weights that give the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term
    
    Returns
    -------
        bias (float):
            If include_bias = True, return the bias constant. Else,
            return 0
        weights (np.ndarray):
            A (n, 1) numpy matrix representing the weight constant(s).
    '''
    final_result = []  # first element is bias, second element is the weight
    
    if include_bias:
        matrix_with_bias = add_bias_column(X)
        transpose_matrix = np.transpose(matrix_with_bias)

        matrix_multiply = np.matmul(transpose_matrix, matrix_with_bias)
        inverse_of_matrix_multiply = np.linalg.inv(matrix_multiply)

        result = np.matmul(inverse_of_matrix_multiply, transpose_matrix)
        result = np.matmul(result, y)
        weights_vectors = result[1:]
        final_result.append(result[0][0])
        final_result.append(weights_vectors)
    else:
        transpose_matrix = np.transpose(X)

        matrix_multiply = np.matmul(transpose_matrix, X)
        inverse_of_matrix_multiply = np.linalg.inv(matrix_multiply)

        result = np.matmul(inverse_of_matrix_multiply, transpose_matrix)
        weights_vectors = np.matmul(result, y)
        final_result.append(0)
        final_result.append(weights_vectors)

    return final_result

def get_prediction_linear_regression(X, y, include_bias = True):
    '''
    Calculate the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term

    Returns
    -------
        y_pred (np.ndarray):
            A (m, 1) numpy matrix representing prediction values.
    '''
    result = get_bias_and_weight(X, y, include_bias)

    if include_bias:
        feature_matrix = add_bias_column(X)
        bias_constant = result[0]
        bias = np.array([[bias_constant]])
        weight_vector = np.append(bias, result[1], axis=0)
        
        y_pred = np.matmul(feature_matrix, weight_vector)
    else:
        y_pred = np.matmul(X, weight_vector)
    return y_pred
