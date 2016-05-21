import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import *
from adaline_gd import *

def scatter_data(X, y):
    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    #plt.savefig('./images/02_06.png', dpi=300)
    plt.show()
    
def plot_error(data):
    
    plt.plot(range(1, len(data) + 1), data, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    
    plt.tight_layout()
    # plt.savefig('./perceptron_1.png', dpi=300)
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02, xlabel='', ylabel='', title=''):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()

def plot_adaline(ada):
    
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    # plt.savefig('./adaline_3.png', dpi=300)
    plt.show()

def plot_adalines(ada1, ada2, eta1, eta2):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate ' + str(eta1))
    
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate ' + str(eta2))
    
    plt.tight_layout()
    # plt.savefig('./adaline_1.png', dpi=300)
    plt.show()

# main

df = pd.read_csv('../datasets/iris/iris.data', header=None)
df.tail()
#print(df[-5:])

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
scatter_data(X, y)

# training
ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plot_error(ppn.errors_)

plot_decision_regions(X, y, classifier=ppn, xlabel = 'sepal length [cm]', ylabel = 'petal length [cm]')

# adaline gd
eta1 = 0.01
eta2 = 0.0001

ada1 = AdalineGD(n_iter=10, eta=eta1).fit(X, y)
ada2 = AdalineGD(n_iter=10, eta=eta2).fit(X, y)

plot_adalines(ada1, ada2, eta1, eta2)

# standardize features
X_std = np.copy(X)
for col in range(X_std.shape[1]):
    X_std[:, col] = (X[:, col] - X[:, col].mean()) / X[:, col].std()

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada, title='Adaline - Gradient Descent', xlabel='sepal length [standardized]', ylabel = 'petal length [standardized]')

plot_adaline(ada)
