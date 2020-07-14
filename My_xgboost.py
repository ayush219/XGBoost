# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:24:49 2019

@author: Ayush
"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Dataset
dataset= pd.read_csv('Churn_Modelling.csv')
X= dataset.iloc[:,3:-1].values
Y= dataset.iloc[:,-1].values

#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])
labelencoder_X2= LabelEncoder()
X[:,2]= labelencoder_X.fit_transform(X[:,2])

onehotencoder= OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

#Splitting data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

#XGboost
from xgboost import XGBClassifier 
classifier=XGBClassifier()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()