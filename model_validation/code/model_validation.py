


# %%

# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# %%

# Import the train_test_split function and uncomment
from sklearn.model_selection  import train_test_split

# fill in and uncomment
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

# %%

# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# %%

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# glance at results
print(val_predictions[0:5])

print(val_y.head)

# %% Average Error

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(val_y, val_predictions))
