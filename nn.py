#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv


# In[2]:


import numpy as np
import pandas as pd


# In[35]:


import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


# In[249]:


data_ = []
header_ = []


# In[250]:


with open("data2.csv") as csvfile:
    i = 0
    rows = csv.reader(csvfile)
    for row in rows:
        if i == 0:
            header_ = row[1:]
        else:
            if row[-1] == "yes":
                data_.append(row[1:-1]+[1])
            else:
                data_.append(row[1:-1]+[0])
        i += 1


# In[251]:


#data_[1]


# In[252]:


data = []
header = []
columnData = []


# In[253]:


for j in range(len(header_)-1):
    vals = []
    for i in range(len(data_)):
        if data_[i][j].lower() not in vals:
            vals.append(data_[i][j].lower())
    columnData.append(vals)


# In[254]:


#for i in range(len(columnData)):
#   print(header_[i]+" len: "+str(len(columnData[i])))


# In[255]:


data = pd.get_dummies(pd.DataFrame(data_))
data = pd.DataFrame(data)


# In[256]:


#data["travel_history_location"]


# In[257]:


target_column = [63] 
predictors = list(set(list(data.columns))-set(target_column))
data[predictors] = data[predictors]/data[predictors].max()
data.describe().transpose()


# In[258]:


#data.to_csv(r'out.csv', index = False)


# In[259]:


X = data[predictors].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
print(X_train.shape); print(X_test.shape)


# In[260]:


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


# In[261]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))


# In[262]:


print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))


# In[ ]:




