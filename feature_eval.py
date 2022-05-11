import numpy as np
import linear_model
import ann_model
import dtree_model
import svm_model
import pca_model
import knearest_model
import boosting_model

X = np.loadtxt('data/winequality-redTrainData.csv', skiprows=1, delimiter=';')
y = np.loadtxt('data/winequality-redTrainLabel.csv')

features = np.arange(X.shape[1])
f_curr = []
f_accs = []

while len(features) > 0:
    best_acc = 0
    best_f = -1
    for i in range(len(features)):
        print("%d/%d completed." %(i + 1, len(features)))
        f = features[i]
        f_test = f_curr + [f]
        X_sub = X[:, f_test]
        acc = boosting_model.test(X_sub, y)
        if acc > best_acc:
            best_acc = acc
            best_f = f
    f_curr.append(best_f)
    features = features[features != best_f]
    f_accs.append(best_acc)

best_acc = 0
index = -1
for i in range(len(f_accs)):
    acc = f_accs[i]
    if acc > best_acc:
        best_acc = acc
        index = i
best_f = np.array(f_curr[: index + 1])
print("Best subset of features: (%s) with accuracy of %.2f%%" %(', '.join(map(str, best_f)), best_acc * 100))
