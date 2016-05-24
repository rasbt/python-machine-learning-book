import numpy as np
from share import *

def xor_data(num_points):
    X_xor = np.random.randn(num_points, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    return X_xor, y_xor

def circle_data(num_points, radius):    
    X = np.random.randn(num_points, 2)
    y = X[:, 0]**2 + X[:, 1]**2 - radius**2
    y = np.where(y > 0, 1, -1)

    return X, y

# data
np.random.seed(0)

num_points = 200

# svm
from sklearn.svm import SVC

for option in ['xor', 'circle']:
    if option == 'xor':
        X, y = xor_data(num_points)
    else:
        X, y = circle_data(num_points, 1)

    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X, y)
    plot_decision_regions(X, y, classifier=svm, title=option)
