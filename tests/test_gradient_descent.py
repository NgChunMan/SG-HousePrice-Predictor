import os
from src.models.gradient_descent import gradient_descent_multi_variable
from src.data_utils import load_data

def test_gradient_descent_multi_variable():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_filepath = os.path.join(project_root, 'data', 'housing_data.csv')
    
    X, y = load_data(data_filepath)

    _, _, loss = gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250)
    loss_initial = loss[0]
    loss_final = loss[-1]

    assert loss_initial > loss_final
