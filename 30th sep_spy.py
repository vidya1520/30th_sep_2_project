# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:19:21 2018

@author: VIDYA
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("train.csv")
correlation_values = df.select_dtypes(include=[np.number]).corr()
correlation_values[["SalePrice"]]
selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]
x= df[["OverallQual","TotalBsmtSF","GrLivArea","GarageArea","GarageCars"]]
y= df["SalePrice"]
x_train,x_test,y_train,y_test = tts(x,y ,test_size =0.3,random_state =42)   
reg = LinearRegression()
reg.fit(x_test,y_test)
y_pred = reg.predict(x_test)
reg.score(x_test,y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse