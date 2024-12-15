# Housing Price Prediction

This project predicts housing prices in Singapore using **Linear Regression**. Two methods are implemented for training the model:
1. **Normal Equation**: Directly calculates the best-fitting weights using a closed-form solution.
2. **Gradient Descent**: Iteratively updates weights to minimize the mean squared error.

The project provides step-by-step implementation of linear regression algorithms, including data preprocessing, model training, and evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Dataset
The dataset housing_data.csv consists of 90 housing data points. Each data point includes:
- floor_area_sqm - The size of the house in square meters (used as the primary feature for prediction).
- bedrooms - Number of bedrooms in the house.
- schools - Number of primary schools within a 1 km radius.
- asking_price - The price of the housing unit (used as the target variable).

## Project Objectives

1. Predict housing prices using a linear regression model.
2. Compare two methods for solving linear regression:
- Normal Equation: Computes optimal weights directly using a mathematical formula.
- Gradient Descent: Iteratively updates weights to minimize prediction error.
3. Evaluate the model using MSE (Mean Squared Error) and MAE (Mean Absolute Error).
4. Visualize model predictions against the actual data to assess performance.

## Getting Started
1. Clone the repository:
```
git clone https://github.com/your-username/Housing-Prices-Prediction.git
cd SG-HousePrice-Predictor
```

2. Install dependencies using pip:
```
pip install -r requirements.txt
```

3. Run the main script:
```
python main.py
```

4. The output will display the model results.

## Testing
Unit tests are provided to validate the implementation of the gradient descent algorithm and other utilities. Run the tests using pytest:
```
pytest tests/
```

## Results
A sample plot using floor_area_sqm as the only feature comparing the actual housing prices with model predictions:

Blue points represent the actual data.
Red line represents predictions using the Normal Equation.
Green line represents predictions using Gradient Descent.
