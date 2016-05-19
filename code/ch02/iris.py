import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import *

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
