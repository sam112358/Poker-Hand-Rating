# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train1.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


pred = pd.read_csv('test1.csv')
pred = pred.iloc[:, 1:].values

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
pred = sc.fit_transform(pred)

#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier( learning_rate =0.2,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=0.2,
 gamma=1.5,
 subsample=0.9,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread=4,
 scale_pos_weight=1,
 seed=0)
classifier.fit(X, y)


# Predicting the Test set results
y_pred = classifier.predict(pred)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#
#total = 0
#for i in range(0,9):
#    total += cm[i,i]
#accuracy = float(total/5002)



