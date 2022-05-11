import matplotlib.pyplot as plt
import numpy as np

X = np.loadtxt('data/winequality-redTrainData.csv', skiprows=1, delimiter=';')
y = np.loadtxt('data/winequality-redTrainLabel.csv')

def vis(X, y, ncols):
    k = X.shape[1]
    nrows = int(k / ncols) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.tight_layout()
    for f in range(k):
        plt.subplot(nrows, ncols, f + 1)
        plt.title('X%d vs y' %(f + 1))
        plt.scatter(X[:, f], y)
    plt.show()

# Features: 1 -> 11
vis(X, y, 4)
