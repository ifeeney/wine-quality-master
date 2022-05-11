import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def test(X, y):
    X = X[:, (10, 1, 9, 6, 0, 7)]
    kf = KFold(n_splits=10)
    acc = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       model = make_pipeline(StandardScaler(), PCA(), GaussianNB())
       clf = model.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       acc.append(accuracy_score(y_test, y_pred))

    return np.mean(acc)
