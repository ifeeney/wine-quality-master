import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def test(X, y):
    X = X[:, (10, 9, 6, 3, 1, 4, 7, 8, 5)]
    kf = KFold(n_splits=10)
    acc = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]

       model = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=0))
       clf = model.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       acc.append(accuracy_score(y_test, y_pred))

    return np.mean(acc)
