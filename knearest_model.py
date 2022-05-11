import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def test(X, y):
    X = X[:, (10, 6, 4, 0, 9, 2)]
    kf = KFold(n_splits=10)
    acc = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
       clf = model.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       acc.append(accuracy_score(y_test, y_pred))

    return np.mean(acc)
