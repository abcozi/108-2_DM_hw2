#!/usr/bin/env python
# coding: utf-8

# In[249]:


from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np
import pandas as pd
import graphviz
from graphviz import Graph

data_names = []
data = []
with open('latestdata_v3.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data.append(row)
data.pop(0)

target = []
with open('target.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        target.append(row)


# In[269]:


data_oe = pd.get_dummies(pd.DataFrame(data))
data_top = data_oe.columns

labelencoder = LabelEncoder()
target_le=pd.DataFrame(target)
target_le[0] = labelencoder.fit_transform(target_le[0])


# In[270]:


data_oe


# In[266]:


X_train, X_test, y_train, y_test = train_test_split(data_oe, target_le, test_size=0.2, random_state=1)


# In[273]:


clf = tree.DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))


# In[292]:


forest_clf = RandomForestClassifier(criterion='entropy',n_estimators=100)
forest_clf = forest_clf.fit(X_train,y_train.values.ravel())
y_pred = forest_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))


# In[288]:


# tree.plot_tree(clf.fit(data_le, target_le)) 


# In[262]:


dot_data = tree.export_graphviz(clf, out_file=None)


# In[263]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=data_top,
                      class_names=['0','1'],
                      filled=True, rounded=True,  
                      special_characters=True) 


# In[264]:


graph = graphviz.Source(dot_data)


# In[265]:


# g = Graph(format='png')
graph.format = 'png'
graph.render() 


# In[ ]:




