import matplotlib.pyplot as plt
import numpy as np

X = np.loadtxt('data/winequality-redTrainData.csv', skiprows=1, delimiter=';')
y = np.loadtxt('data/winequality-redTrainLabel.csv')

n = X.shape[1]
fig, axes = plt.subplots(n, n)
for i in range(n):
    axes[i][0].scatter(X[:, i], y)
    for j in range(i):
        axes[i][j + 1].scatter(X[:, i], X[:, j])
plt.show()
