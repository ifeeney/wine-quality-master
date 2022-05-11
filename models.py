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

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

print("Linear Model Accuracy: %.2f%%" %(linear_model.test(X, y) * 100))
print("ANN Model Accuracy: %.2f%%" %(ann_model.test(X, y) * 100))
print("D-Tree Accuracy: %.2f%%" %(dtree_model.test(X, y) * 100))
print("SVM Model Accuracy: %.2f%%" %(svm_model.test(X, y) * 100))
print("PCA Model Accuracy: %.2f%%" %(pca_model.test(X, y) * 100))
print("K-Nearest Model Accuracy: %.2f%%" %(knearest_model.test(X, y) * 100))
print("Boosting Model Accuracy: %.2f%%" %(boosting_model.test(X, y) * 100))
