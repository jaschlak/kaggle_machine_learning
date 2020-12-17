# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:26:54 2020

@author: jschlak
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection  import train_test_split

# Path of the file to read
iowa_file_path = '../../input/train.csv'


# get data, pick target and feature columns
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# choose random forest model and split data
forest_model = RandomForestRegressor(random_state=1)
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)


forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))