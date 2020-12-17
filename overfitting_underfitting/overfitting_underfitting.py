


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection  import train_test_split

# Path of the file to read
iowa_file_path = '../input/train.csv'

# %%

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# %%
    


home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

lowest_mae = 99999
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in list(range(60,80,1)):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    if my_mae < lowest_mae:
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
        lowest_mae = my_mae

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=71, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)