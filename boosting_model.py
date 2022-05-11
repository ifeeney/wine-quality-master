import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

def test(X, y):
    X = X[:, (6, 10, 4, 9, 1, 3, 5)]
    kf = KFold(n_splits=10)
    acc = []
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       base_model = RandomForestClassifier(max_depth=25, n_estimators=100, random_state=0)
       model = AdaBoostClassifier(base_estimator=base_model, n_estimators=25, learning_rate=10, algorithm='SAMME.R', random_state=0)
       clf = model.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       acc.append(accuracy_score(y_test, y_pred))

    return np.mean(acc)
