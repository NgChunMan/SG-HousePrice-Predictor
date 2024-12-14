import matplotlib.pyplot as plt

def plot_predictions(X, y, predictions):
    '''
    Plot the true data points and the prediction line.
    '''
    plt.scatter(X, y, label="True Data")
    plt.plot(X, predictions, color="red", label="Prediction Line")
    plt.xlabel("Size in square meters")
    plt.ylabel("Price in SGD")
    plt.legend()
    plt.show()
