import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.loadtxt('data/winequality-redTrainData.csv', skiprows=1, delimiter=';')
y = np.loadtxt('data/winequality-redTrainLabel.csv')
X_test = np.loadtxt('data/winequality-redTestData.csv', skiprows=1, delimiter=';')

X = X[:, (10, 9, 6, 3, 1, 4, 7, 8, 5)]
X_test = X_test[:, (10, 9, 6, 3, 1, 4, 7, 8, 5)]

model = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=0))
clf = model.fit(X, y)
pred = clf.predict(X_test)
np.savetxt('predictions/prediction-1.csv', pred, fmt='%d')
