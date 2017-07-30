# Sebastian Raschka, 2015 (http://sebastianraschka.com)
# Python Machine Learning - Code Examples
#
# Chapter 4 - Building Good Training Sets – Data Pre-Processing
#
# S. Raschka. Python Machine Learning. Packt Publishing Ltd., 2015.
# GitHub Repo: https://github.com/rasbt/python-machine-learning-book
#
# License: MIT
# https://github.com/rasbt/python-machine-learning-book/blob/master/LICENSE.txt


import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

#############################################################################
print(50 * '=')
print('Section: Dealing with missing data')
print(50 * '-')

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
print(df)
print('\n\nExecuting df.isnull().sum():')
print(df.isnull().sum())


#############################################################################
print(50 * '=')
print('Section: Eliminating samples or features with missing values')
print(50 * '-')

print('\n\nExecuting df.dropna()')
print(df.dropna())

print('\n\nExecuting df.dropna(axis=1)')
print(df.dropna(axis=1))

print("\n\nExecuting df.dropna(thresh=4)")
print("(drop rows that have not at least 4 non-NaN values)")
print(df.dropna(thresh=4))

print("\n\nExecuting df.dropna(how='all')")
print("(only drop rows where all columns are NaN)")
print(df.dropna(how='all'))

print("\n\nExecuting df.dropna(subset=['C'])")
print("(only drop rows where NaN appear in specific columns (here: 'C'))")
print(df.dropna(subset=['C']))


#############################################################################
print(50 * '=')
print('Section: Imputing missing values')
print(50 * '-')

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

print('Input Array:\n', df.values)
print('Imputed Data:\n', imputed_data)


#############################################################################
print(50 * '=')
print('Section: Handling categorical data')
print(50 * '-')

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
print('Input Array:\n', df)


#############################################################################
print(50 * '=')
print('Section: Mapping ordinal features')
print(50 * '-')

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)
print('Mapping:\n', df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df_inv = df['size'].map(inv_size_mapping)
print('\nInverse mapping:\n', df_inv)


#############################################################################
print(50 * '=')
print('Section: Encoding class labels')
print(50 * '-')

class_mapping = {label: idx for idx, label
                 in enumerate(np.unique(df['classlabel']))}
print('\nClass mapping:\n', class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print('Mapping:\n', df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df_inv = df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print('\nInverse mapping:\n', df_inv)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print('Label encoder tansform:\n', y)

y_inv = class_le.inverse_transform(y)
print('Label encoder inverse tansform:\n', y_inv)


#############################################################################
print(50 * '=')
print('Section: Performing one hot encoding on nominal features')
print(50 * '-')

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("Input array:\n", X)

ohe = OneHotEncoder(categorical_features=[0])
X_onehot = ohe.fit_transform(X).toarray()
print("Encoded array:\n", X_onehot)

df_dummies = pd.get_dummies(df[['price', 'color', 'size']])
print("Pandas get_dummies alternative:\n", df_dummies)


#############################################################################
print(50 * '=')
print('Section: Partitioning a dataset in training and test sets')
print(50 * '-')

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))

print('\nWine data excerpt:\n\n', df_wine.head())


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)


#############################################################################
print(50 * '=')
print('Section: Bringing features onto the same scale')
print(50 * '-')

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

ex = pd.DataFrame([0, 1, 2, 3, 4, 5])
print('Scaling Example:\n')
print('\nInput array:\n', ex)
ex[1] = (ex[0] - ex[0].mean()) / ex[0].std(ddof=0)

# Please note that pandas uses ddof=1 (sample standard deviation)
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
ex[2] = (ex[0] - ex[0].min()) / (ex[0].max() - ex[0].min())
ex.columns = ['input', 'standardized', 'normalized']
print('\nOutput array after scaling:\n', ex)


#############################################################################
print(50 * '=')
print('Section: Sparse solutions with L1-regularization')
print(50 * '-')

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
print('Intercept:', lr.intercept_)
print('Model weights:', lr.coef_)

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
# plt.savefig('./figures/l1_path.png', dpi=300)
plt.show()


#############################################################################
print(50 * '=')
print('Section: Sequential feature selection algorithms')
print(50 * '-')


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


knn = KNeighborsClassifier(n_neighbors=2)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.tight_layout()
# plt.savefig('./sbs.png', dpi=300)
plt.show()


k5 = list(sbs.subsets_[8])
print('Selected top 5 features:\n', df_wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('\nPerformance using all features:\n')
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('\nPerformance using the top 5 features:\n')
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))

#############################################################################
print(50 * '=')
print('Section: Assessing Feature Importances with Random Forests')
print(50 * '-')

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.savefig('./random_forest.png', dpi=300)
plt.show()

if Version(sklearn_version) < '0.18':
    X_selected = forest.transform(X_train, threshold=0.15)
else:
    from sklearn.feature_selection import SelectFromModel
    sfm = SelectFromModel(forest, threshold=0.15, prefit=True)
    X_selected = sfm.transform(X_train)

X_selected.shape
