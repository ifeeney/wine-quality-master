import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def test(X, y):
   kf = KFold(n_splits=5)
   acc = []
   for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      model = MLPClassifier(solver='adam', activation='relu', batch_size=25, hidden_layer_sizes=(10, 10, 10, 10, 10))
      clf = model.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      acc.append(accuracy_score(y_test, y_pred))

   return np.mean(acc)
