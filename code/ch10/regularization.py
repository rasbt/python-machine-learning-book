import pandas as pd
import numpy as np

# read data
data_src = '../datasets/housing/housing.data'

df = pd.read_csv(data_src,
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#print(df.head())

# split data
from sklearn.cross_validation import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y_std, test_size=0.3, random_state=0)

# train and test
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_errors = []
test_errors = []

for alpha in alphas:
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

print(train_errors)
print(test_errors)

    
