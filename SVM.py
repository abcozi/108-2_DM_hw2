from sklearn import datasets
from sklearn import preprocessing
import sklearn as sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

data = pd.read_csv("datasets/covid.csv")
X = pd.get_dummies(pd.DataFrame(data))

# data_header = X.columns
# print(data_header)

data_y = pd.read_csv("datasets/covid_y.csv")
Y = pd.DataFrame(data_y)

le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

### data split ###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

### train/test ###
from sklearn.svm import SVC

svm = SVC(kernel='linear', probability=True)
svm.fit(x_train, y_train)

svm.predict(x_test)

predicted = svm.predict_proba(x_test)
# print(predicted)

y_pred = []
for i in predicted:
    if i[0] > i[1]:
        y_pred.append(0)
    else:
        y_pred.append(1)
print(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
